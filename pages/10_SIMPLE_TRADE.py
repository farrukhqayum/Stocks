import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Your exact parameters
TICKER = "COIN"
EMA_FAST = 8
EMA_SLOW = 21
RSI_LEN = 14
MIN_ADX = 12
INITIAL_STOP_LOSS = 0.06
TRAIL_MULT = 2.5
PARTIAL_TP = 0.35
PARTIAL_SIZE = 0.50
INITIAL_CAPITAL = 20000

@st.cache_data(ttl=300)
def get_data(ticker, period="2y"):
    df = yf.download(ticker, period=period, progress=False, threads=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    if 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    return df.dropna()

def RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs)).fillna(50)

def ADX(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()
    return adx, plus_di, minus_di

def run_real_backtest(df):
    capital = INITIAL_CAPITAL
    equity_curve = []
    trade_records = []
    
    in_trade = False
    entry_price = 0
    original_entry = 0
    initial_stop = 0
    trail_stop = 0
    position_size = 1.0
    took_partial = False
    highest_price = 0
    
    for i in range(len(df)):
        row = df.iloc[i]
        current_close = row['Close']
        
        current_equity = capital
        
        if not in_trade:
            if row['BULL']:
                entry_price = current_close
                original_entry = current_close
                highest_price = current_close
                in_trade = True
                took_partial = False
                position_size = 1.0
                
                initial_stop = entry_price * (1 - INITIAL_STOP_LOSS)
                trail_stop = initial_stop
                
        else:
            if current_close > highest_price:
                highest_price = current_close
            
            if not took_partial and current_close >= original_entry * (1 + PARTIAL_TP):
                partial_profit = PARTIAL_SIZE * (current_close - original_entry) / original_entry
                capital *= (1 + partial_profit)
                position_size *= (1 - PARTIAL_SIZE)
                took_partial = True
            
            new_trail = highest_price - TRAIL_MULT * row["ATR"]
            trail_stop = max(trail_stop, new_trail)
            
            exit_triggered = False
            if current_close <= trail_stop:
                exit_triggered = True
                exit_price = current_close
            elif row['EMA_FAST'] < row['EMA_SLOW'] and current_close < row['EMA_FAST']:
                exit_triggered = True
                exit_price = current_close
            elif row['RSI'] < 35 and current_close < row['EMA_FAST']:
                exit_triggered = True
                exit_price = current_close
            
            if exit_triggered:
                pnl = position_size * (exit_price - original_entry) / original_entry
                capital *= (1 + pnl)
                in_trade = False
            
            if in_trade:
                unrealized = position_size * (current_close - original_entry) / original_entry
                current_equity = capital * (1 + unrealized)
        
        equity_curve.append(current_equity)
    
    return pd.Series(equity_curve, index=df.index), capital

# Streamlit App
st.set_page_config(layout="wide", page_title="Early Entry Signals")
st.title("🚀 Early Entry Trading Signals - FULL BACKTEST")

col1, col2 = st.columns([1, 1])
with col1:
    ticker = st.text_input("Ticker", value=TICKER)
    capital = st.number_input("Capital ($)", value=INITIAL_CAPITAL)
    period = st.selectbox("Data Period", ["2y", "1y", "6mo"], index=0)

if st.button("🔄 Run Full Analysis", type="primary"):
    st.rerun()

df_raw = get_data(ticker, period)
if df_raw is None:
    st.error("❌ Failed to load data. Try AAPL or SPY.")
    st.stop()

close = df_raw['Close']
df = df_raw.copy()
df['EMA_FAST'] = close.ewm(span=EMA_FAST).mean()
df['EMA_SLOW'] = close.ewm(span=EMA_SLOW).mean()
df['RSI'] = RSI(close)

adx, plus_di, minus_di = ADX(df)
df['ADX'] = adx
df['PLUS_DI'] = plus_di
df['MINUS_DI'] = minus_di

tr1 = df['High'] - df['Low']
tr2 = abs(df['High'] - close.shift())
tr3 = abs(df['Low'] - close.shift())
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df['ATR'] = tr.rolling(14, min_periods=1).mean()
df['PRICE_CHANGE'] = close.pct_change() * 100

df = df.dropna()

df["BREAKOUT"] = (
    (df['EMA_FAST'] > df['EMA_SLOW']) & 
    (df['Close'] > df['EMA_FAST'].shift(1)) & 
    (df['Close'].shift(1) <= df['EMA_FAST'].shift(1)) & 
    (df['ADX'] > MIN_ADX)
)

df["PULLBACK"] = (
    (df['EMA_FAST'] > df['EMA_SLOW']) & 
    (df['Close'] < df['EMA_FAST']) & 
    (df['Close'] > df['EMA_FAST'] * 0.97) & 
    (df['RSI'] < 55) & (df['RSI'] > 40) & 
    (df['PRICE_CHANGE'] > -2) & (df['ADX'] > MIN_ADX)
)

df["MOMENTUM"] = (
    (df['EMA_FAST'] > df['EMA_SLOW']) & 
    (df['Close'] > df['EMA_FAST']) & 
    (df['RSI'] > 50) & (df['RSI'] < 70) & 
    (df['PRICE_CHANGE'] > 0.5) & (df['ADX'] > MIN_ADX)
)

df["BULL"] = df["BREAKOUT"] | df["PULLBACK"] | df["MOMENTUM"]

latest = df.iloc[-1]

signals = {
    "BREAKOUT": (latest['EMA_FAST'] > latest['EMA_SLOW'] and 
                latest['Close'] > latest['EMA_FAST'] and 
                df.iloc[-2]['Close'] <= df.iloc[-2]['EMA_FAST'] and 
                latest['ADX'] > MIN_ADX),
    "PULLBACK": (latest['EMA_FAST'] > latest['EMA_SLOW'] and 
                latest['Close'] < latest['EMA_FAST'] and 
                latest['Close'] > latest['EMA_FAST'] * 0.97 and 
                40 < latest['RSI'] < 55 and 
                latest['PRICE_CHANGE'] > -2 and latest['ADX'] > MIN_ADX),
    "MOMENTUM": (latest['EMA_FAST'] > latest['EMA_SLOW'] and 
                latest['Close'] > latest['EMA_FAST'] and 
                50 < latest['RSI'] < 70 and 
                latest['PRICE_CHANGE'] > 0.5 and latest['ADX'] > MIN_ADX)
}

BULL = any(signals.values())

# RUN REAL BACKTEST
equity_series, final_capital = run_real_backtest(df)

# YOUR EXACT 4-PANEL PLOT
fig, axes = plt.subplots(4, 1, figsize=(16, 12), height_ratios=[3, 1, 1, 1.5])
fig.patch.set_facecolor('white')

# 1. PRICE
ax1 = axes[0]
ax1.plot(df.index, df['Close'], color='#2C3E50', linewidth=1.5, label='Close', alpha=0.8)
ax1.plot(df.index, df['EMA_FAST'], color='#3498DB', linewidth=1.5, label=f'EMA {EMA_FAST}')
ax1.plot(df.index, df['EMA_SLOW'], color='#E74C3C', linewidth=1.5, label=f'EMA {EMA_SLOW}')

for strategy, color in [('BREAKOUT', '#FF6B6B'), ('PULLBACK', '#4ECDC4'), ('MOMENTUM', '#95E1D3')]:
    strategy_points = df[df[strategy] == True]
    if not strategy_points.empty:
        ax1.scatter(strategy_points.index, strategy_points['Close'], 
                   marker='^', color=color, s=120, alpha=0.9, 
                   label=f'{strategy}', zorder=5, edgecolors='white', linewidth=1)

if BULL:
    ax1.scatter(df.index[-1], latest['Close'], color='limegreen', s=200, 
               marker='^', edgecolors='white', linewidth=2, zorder=10, label='LIVE BUY')

ax1.set_title(f'{ticker} - Early Entry Multi-Strategy (Real Backtest)', fontsize=16, fontweight='bold', pad=15)
ax1.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', framealpha=0.9, fontsize=9, ncol=2)
ax1.grid(True, alpha=0.2, linestyle='--')
ax1.set_facecolor('#F8F9FA')

# 2. RSI
ax2 = axes[1]
ax2.plot(df.index, df['RSI'], color='#9B59B6', linewidth=1.5, label='RSI')
ax2.axhline(y=70, color='#E74C3C', linestyle='--', alpha=0.5)
ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
ax2.axhline(y=30, color='#27AE60', linestyle='--', alpha=0.5)
ax2.set_ylabel('RSI', fontsize=11, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.2)
ax2.set_ylim(0, 100)
ax2.set_facecolor('#F8F9FA')

# 3. ADX
ax3 = axes[2]
ax3.plot(df.index, df['ADX'], color='#E67E22', linewidth=1.5, label='ADX')
ax3.axhline(y=MIN_ADX, color='#E74C3C', linestyle='--', alpha=0.5, label=f'Min={MIN_ADX}')
ax3.set_ylabel('ADX', fontsize=11, fontweight='bold')
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.2)
ax3.set_facecolor('#F8F9FA')

# 4. REAL EQUITY CURVE
ax4 = axes[3]
ax4.plot(equity_series.index, equity_series, color='#27AE60', linewidth=2.5, label='Portfolio')
ax4.fill_between(equity_series.index, INITIAL_CAPITAL, equity_series, alpha=0.3, color='#27AE60')
ax4.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5)
ax4.set_ylabel('Equity ($)', fontsize=11, fontweight='bold')
ax4.set_xlabel('Date', fontsize=12, fontweight='bold')
ax4.legend(loc='upper left', fontsize=9)
ax4.grid(True, alpha=0.2)
ax4.set_facecolor('#F8F9FA')

return_pct = (final_capital/INITIAL_CAPITAL-1)*100
fig.text(0.5, 0.02, f'Final: ${final_capital:,.0f} | Return: {return_pct:+.1f}% | Trades: {len(df[df["BULL"]])}', 
         ha='center', fontsize=13, fontweight='bold', 
         color='#27AE60' if return_pct > 0 else '#E74C3C',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout(rect=[0, 0.03, 1, 1])
st.pyplot(fig)

# Live Signals
col_signal, col_metrics = st.columns(2)
with col_signal:
    st.metric("Current Price", f"${latest['Close']:.2f}")
    signal_color = "🟢 BUY" if BULL else "🔴 NO SIGNAL"
    st.metric("Signal", signal_color)

with col_metrics:
    st.metric("RSI", f"{latest['RSI']:.0f}")
    st.metric("ADX", f"{latest['ADX']:.0f}")
    st.metric("Trend", "BULLISH" if latest['EMA_FAST'] > latest['EMA_SLOW'] else "BEARISH")

if BULL:
    st.success(f"**🎯 {next(k for k,v in signals.items() if v)} SIGNAL**")
    st.balloons()
    
    shares = int((capital * 0.95) / latest['Close'])
    stop_price = latest['Close'] * (1 - INITIAL_STOP_LOSS)
    
    st.subheader("📊 TRADE PLAN")
    col_buy, col_stop = st.columns(2)
    with col_buy:
        st.success(f"**BUY {shares:,} shares @ ${latest['Close']:.2f}**")
    with col_stop:
        st.warning(f"**STOP LOSS: ${stop_price:.2f}** ({INITIAL_STOP_LOSS*100:.0f}%)")
    
    st.info("""
    **EXIT RULES:**
    - Partial: Sell 50% at +35%
    - Trail: 2.5x ATR below highest
    - Trend: EMA8 < EMA21 + price < EMA8
    - RSI: < 35 + price < EMA8
    """)

st.caption(f"✅ FULL BACKTEST + Live Signal | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
