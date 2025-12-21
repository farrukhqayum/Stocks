import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Streamlit App
st.set_page_config(layout="wide", page_title="Simple Signals")
st.title("📈 Simple Signals")

# USER INPUTS
col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.text_input("Ticker", value="COIN")
    capital = st.number_input("Capital ($)", value=20000, min_value=1000)
with col2:
    ema_fast = st.number_input("EMA Fast", value=8, min_value=5, max_value=20)
    ema_slow = st.number_input("EMA Slow", value=21, min_value=15, max_value=50)
with col3:
    rsi_len = st.number_input("RSI Length", value=14, min_value=10, max_value=21)
    rsi_ema_len = st.number_input("RSI EMA Length", value=20, min_value=10, max_value=30)

min_adx = st.number_input("Min ADX", value=12, min_value=10, max_value=20)
initial_stop_loss = st.number_input("Stop Loss %", value=6.0, min_value=3.0, max_value=10.0)/100
period = st.selectbox("Data Period", ["2y", "1y", "6mo"], index=0)

st.markdown("""
**🟢 LIVE TRADE PLAN appears when:**
- EMA Fast > EMA Slow **(Uptrend)**
- ADX > Min ADX **(Trend strength)**
- **RSI > RSI_EMA(20)** *(NEW)*
- **ANY** Breakout/Pullback/Momentum signal
""")

@st.cache_data(ttl=300)
def get_data(ticker, period="2y"):
    df = yf.download(ticker, period=period, progress=False, threads=False)
    if df.empty: return None
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
    high, low, close = df["High"], df["Low"], df["Close"]
    tr1, tr2, tr3 = high-low, abs(high-close.shift()), abs(low-close.shift())
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

if st.button("🔄 Analyze", type="primary"):
    st.rerun()

df_raw = get_data(ticker, period)
if df_raw is None:
    st.error("❌ Failed to load data. Try AAPL or SPY.")
    st.stop()

# Calculate indicators
close = df_raw['Close']
df = df_raw.copy()
df['EMA_FAST'] = close.ewm(span=ema_fast).mean()
df['EMA_SLOW'] = close.ewm(span=ema_slow).mean()
df['RSI'] = RSI(close, rsi_len)
df['RSI_EMA'] = df['RSI'].ewm(span=rsi_ema_len).mean()  # NEW RSI EMA

adx, plus_di, minus_di = ADX(df, rsi_len)
df['ADX'] = adx

tr = pd.concat([df['High']-df['Low'], abs(df['High']-close.shift()), 
                abs(df['Low']-close.shift())], axis=1).max(axis=1)
df['ATR'] = tr.rolling(rsi_len, min_periods=1).mean()
df['PRICE_CHANGE'] = close.pct_change() * 100
df = df.dropna()

# Entry signals (YOUR ORIGINAL 3 + RSI_EMA filter)
df["BREAKOUT"] = (
    (df['EMA_FAST'] > df['EMA_SLOW']) & 
    (df['RSI'] > df['RSI_EMA']) &  # NEW
    (df['Close'] > df['EMA_FAST'].shift(1)) & 
    (df['Close'].shift(1) <= df['EMA_FAST'].shift(1)) & 
    (df['ADX'] > min_adx)
)

df["PULLBACK"] = (
    (df['EMA_FAST'] > df['EMA_SLOW']) & 
    (df['RSI'] > df['RSI_EMA']) &  # NEW
    (df['Close'] < df['EMA_FAST']) & 
    (df['Close'] > df['EMA_FAST'] * 0.97) & 
    (df['RSI'] < 55) & (df['RSI'] > 40) & 
    (df['PRICE_CHANGE'] > -2) & (df['ADX'] > min_adx)
)

df["MOMENTUM"] = (
    (df['EMA_FAST'] > df['EMA_SLOW']) & 
    (df['RSI'] > df['RSI_EMA']) &  # NEW
    (df['Close'] > df['EMA_FAST']) & 
    (df['RSI'] > 50) & (df['RSI'] < 70) & 
    (df['PRICE_CHANGE'] > 0.5) & (df['ADX'] > min_adx)
)

df["BULL"] = df["BREAKOUT"] | df["PULLBACK"] | df["MOMENTUM"]

latest = df.iloc[-1]

# Live signals with RSI_EMA
signals = {
    "BREAKOUT": (latest['EMA_FAST'] > latest['EMA_SLOW'] and 
                latest['RSI'] > latest['RSI_EMA'] and 
                df.iloc[-2]['Close'] <= df.iloc[-2]['EMA_FAST'] and 
                latest['ADX'] > min_adx),
    "PULLBACK": (latest['EMA_FAST'] > latest['EMA_SLOW'] and 
                latest['RSI'] > latest['RSI_EMA'] and 
                latest['Close'] < latest['EMA_FAST'] and 
                latest['Close'] > latest['EMA_FAST'] * 0.97 and 
                40 < latest['RSI'] < 55 and 
                latest['PRICE_CHANGE'] > -2 and latest['ADX'] > min_adx),
    "MOMENTUM": (latest['EMA_FAST'] > latest['EMA_SLOW'] and 
                latest['RSI'] > latest['RSI_EMA'] and 
                latest['Close'] > latest['EMA_FAST'] and 
                50 < latest['RSI'] < 70 and 
                latest['PRICE_CHANGE'] > 0.5 and latest['ADX'] > min_adx)
}

BULL = any(signals.values())

# 4-PANEL CHART
fig, axes = plt.subplots(4, 1, figsize=(16, 12), height_ratios=[3, 1, 1, 1])
fig.patch.set_facecolor('white')

# Price + RSI_EMA on price subplot
ax1 = axes[0]
ax1.plot(df.index, df['Close'], color='#2C3E50', linewidth=1.5, label='Close', alpha=0.8)
ax1.plot(df.index, df['EMA_FAST'], color='#3498DB', linewidth=1.5, label=f'EMA{ema_fast}')
ax1.plot(df.index, df['EMA_SLOW'], color='#E74C3C', linewidth=1.5, label=f'EMA{ema_slow}')

# Strategy entries
for strategy, color in [('BREAKOUT', '#FF6B6B'), ('PULLBACK', '#4ECDC4'), ('MOMENTUM', '#95E1D3')]:
    strategy_points = df[df[strategy] == True]
    if not strategy_points.empty:
        ax1.scatter(strategy_points.index, strategy_points['Close'], 
                   marker='^', color=color, s=120, alpha=0.9, 
                   label=f'{strategy}', zorder=5, edgecolors='white', linewidth=1)

if BULL:
    ax1.scatter(df.index[-1], latest['Close'], color='limegreen', s=200, 
               marker='^', edgecolors='white', linewidth=2, zorder=10, label='LIVE BUY')

ax1.set_title(f'{ticker} - Simple Signals (RSI > RSI_EMA{rsi_ema_len})', fontsize=16, fontweight='bold')
ax1.legend(loc='upper left', framealpha=0.9, fontsize=9, ncol=2)
ax1.grid(True, alpha=0.2)
ax1.set_ylabel('Price ($)')

# RSI + RSI_EMA
ax2 = axes[1]
ax2.plot(df.index, df['RSI'], color='#9B59B6', linewidth=1.5, label=f'RSI({rsi_len})')
ax2.plot(df.index, df['RSI_EMA'], color='#F39C12', linewidth=2, label=f'RSI_EMA({rsi_ema_len})')
ax2.axhline(y=70, color='#E74C3C', linestyle='--', alpha=0.5)
ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
ax2.axhline(y=30, color='#27AE60', linestyle='--', alpha=0.5)
ax2.set_ylabel('RSI')
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.2)
ax2.set_ylim(0, 100)

# ADX
ax3 = axes[2]
ax3.plot(df.index, df['ADX'], color='#E67E22', linewidth=1.5, label='ADX')
ax3.axhline(y=min_adx, color='#E74C3C', linestyle='--', alpha=0.5, label=f'Min={min_adx}')
ax3.set_ylabel('ADX')
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.2)

# Simple equity mock
ax4 = axes[3]
mock_equity = capital * (1 + df['Close'].pct_change().cumsum() * 0.05 + 1)
ax4.plot(df.index, mock_equity, color='#27AE60', linewidth=2.5, label='Equity')
ax4.axhline(y=capital, color='gray', linestyle='--', alpha=0.5)
ax4.set_ylabel('Equity ($)')
ax4.set_xlabel('Date')
ax4.legend(loc='upper left', fontsize=9)
ax4.grid(True, alpha=0.2)

plt.tight_layout(rect=[0, 0.03, 1, 1])
st.pyplot(fig)

# SIGNAL DEBUGGER
st.subheader("🔍 Signal Debugger")
col1, col2, col3, col4 = st.columns(4)
with col1:
    trend_ok = latest['EMA_FAST'] > latest['EMA_SLOW']
    st.metric("Trend", "✅" if trend_ok else "❌", delta=f"EMA{ema_fast}={latest['EMA_FAST']:.1f}")
with col2:
    rsi_ema_ok = latest['RSI'] > latest['RSI_EMA']
    st.metric("RSI>RSI_EMA", "✅" if rsi_ema_ok else "❌", delta=f"RSI={latest['RSI']:.0f}")
with col3:
    adx_ok = latest['ADX'] > min_adx
    st.metric("ADX", "✅" if adx_ok else "❌", delta=f"{latest['ADX']:.1f}")
with col4:
    signal_status = "🟢 LIVE" if BULL else "🔴 WAIT"
    st.metric("Signal", signal_status)

# LIVE TRADE PLAN (ONLY when BULL=True)
if BULL:
    st.success(f"🎯 **LIVE {next(k for k,v in signals.items() if v)} SIGNAL**")
    st.balloons()
    
    entry_price = latest['Close']
    shares = int((capital * 0.95) / entry_price)
    stop_price = entry_price * (1 - initial_stop_loss)
    partial_price = entry_price * (1 + 0.35)
    
    st.subheader("📊 **LIVE TRADE PLAN**")
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"""
        **BUY**
        • **{shares:,} shares**
        • **${entry_price:.2f}**
        • **${shares*entry_price:,.0f}**
        """)
        st.warning(f"**STOP LOSS** • **${stop_price:.2f}** • **-{initial_stop_loss*100:.0f}%**")
    with col2:
        st.info(f"**PARTIAL** • **${partial_price:.2f}** • **+35%**")
        st.info(f"**Trail**: {2.5}x ATR=${latest['ATR']*2.5:.2f}")

# ALWAYS SHOW HYPOTHETICAL
st.subheader("💡 Hypothetical Plan (Ready when signal fires)")
st.info(f"• Entry: **${latest['Close']:.2f}** • Stop: **${latest['Close']*(1-initial_stop_loss):.2f}**")

with st.expander("📋 Backtest Trades"):
    trades = len(df[df['BULL']])
    st.metric("Total Signals", trades)
    st.write("Click chart entries to see historical performance")
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
                
                trade_records.append({
                    'entry_date': df.index[i],
                    'entry_price': entry_price,
                    'strategy': 'LIVE' if i == len(df)-1 else 'HISTORICAL'
                })
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
            exit_price = current_close
            exit_reason = ""
            
            if current_close <= trail_stop:
                exit_triggered = True
                exit_reason = "TRAIL_STOP"
            elif row['EMA_FAST'] < row['EMA_SLOW'] and current_close < row['EMA_FAST']:
                exit_triggered = True
                exit_reason = "TREND_BREAK"
            elif row['RSI'] < 35 and current_close < row['EMA_FAST']:
                exit_triggered = True
                exit_reason = "RSI_WEAK"
            
            if exit_triggered:
                pnl = position_size * (exit_price - original_entry) / original_entry
                capital *= (1 + pnl)
                if len(trade_records) > 0:
                    trade_records[-1].update({
                        'exit_date': df.index[i],
                        'exit_price': exit_price,
                        'pnl_pct': pnl * 100,
                        'exit_reason': exit_reason,
                        'highest_price': highest_price
                    })
                in_trade = False
            
            if in_trade:
                unrealized = position_size * (current_close - original_entry) / original_entry
                current_equity = capital * (1 + unrealized)
        
        equity_curve.append(current_equity)
    
    return pd.Series(equity_curve, index=df.index), capital, trade_records

# Streamlit App
st.set_page_config(layout="wide", page_title="Early Entry Signals")
st.title("🚀 Early Entry Trading Signals - COMPLETE TRADE PLAN")

col1, col2 = st.columns([1, 1])
with col1:
    ticker = st.text_input("Ticker", value=TICKER)
    capital = st.number_input("Capital ($)", value=INITIAL_CAPITAL)
    period = st.selectbox("Data Period", ["2y", "1y", "6mo"], index=0)

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

# RUN COMPLETE BACKTEST WITH TRADE RECORDS
equity_series, final_capital, trades = run_real_backtest(df)

# 4-PANEL CHART
fig, axes = plt.subplots(4, 1, figsize=(16, 12), height_ratios=[3, 1, 1, 1.5])
fig.patch.set_facecolor('white')

# Price chart with entries
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

ax1.set_title(f'{ticker} - Early Entry Strategy', fontsize=16, fontweight='bold', pad=15)
ax1.set_ylabel('Price ($)')
ax1.legend(loc='upper left', framealpha=0.9, fontsize=9, ncol=2)
ax1.grid(True, alpha=0.2)
ax1.set_facecolor('#F8F9FA')

# RSI, ADX (same as before)
ax2 = axes[1]
ax2.plot(df.index, df['RSI'], color='#9B59B6', linewidth=1.5, label='RSI')
ax2.axhline(y=70, color='#E74C3C', linestyle='--', alpha=0.5)
ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
ax2.axhline(y=30, color='#27AE60', linestyle='--', alpha=0.5)
ax2.set_ylabel('RSI')
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.2)
ax2.set_ylim(0, 100)
ax2.set_facecolor('#F8F9FA')

ax3 = axes[2]
ax3.plot(df.index, df['ADX'], color='#E67E22', linewidth=1.5, label='ADX')
ax3.axhline(y=MIN_ADX, color='#E74C3C', linestyle='--', alpha=0.5, label=f'Min={MIN_ADX}')
ax3.set_ylabel('ADX')
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.2)
ax3.set_facecolor('#F8F9FA')

# REAL Equity
ax4 = axes[3]
ax4.plot(equity_series.index, equity_series, color='#27AE60', linewidth=2.5, label='Portfolio')
ax4.fill_between(equity_series.index, INITIAL_CAPITAL, equity_series, alpha=0.3, color='#27AE60')
ax4.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5)
ax4.set_ylabel('Equity ($)')
ax4.set_xlabel('Date')
ax4.legend(loc='upper left', fontsize=9)
ax4.grid(True, alpha=0.2)
ax4.set_facecolor('#F8F9FA')

return_pct = (final_capital/INITIAL_CAPITAL-1)*100
fig.text(0.5, 0.02, f'Final: ${final_capital:,.0f} | Return: {return_pct:+.1f}% | Trades: {len(trades)}', 
         ha='center', fontsize=13, fontweight='bold', 
         color='#27AE60' if return_pct > 0 else '#E74C3C',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout(rect=[0, 0.03, 1, 1])
st.pyplot(fig)

# LIVE SIGNAL + DETAILED TRADE PLAN
col1, col2 = st.columns(2)
with col1:
    st.metric("Current Price", f"${latest['Close']:.2f}")
    signal_color = "🟢 BUY NOW" if BULL else "🔴 NO SIGNAL"
    st.metric("Signal", signal_color)

with col2:
    st.metric("RSI", f"{latest['RSI']:.0f}")
    st.metric("ADX", f"{latest['ADX']:.1f}")
    st.metric("Trend", "🟢 BULLISH" if latest['EMA_FAST'] > latest['EMA_SLOW'] else "🔴 BEARISH")

if BULL:
    st.success(f"🎯 **{next(k for k,v in signals.items() if v)} SIGNAL ACTIVE**")
    st.balloons()
    
    # COMPLETE TRADE PLAN WITH EXACT NUMBERS
    entry_price = latest['Close']
    shares = int((capital * 0.95) / entry_price)
    position_value = shares * entry_price
    stop_price = entry_price * (1 - INITIAL_STOP_LOSS)
    partial_price = entry_price * (1 + PARTIAL_TP)
    atr_trail = latest['ATR'] * TRAIL_MULT
    
    st.subheader("📊 **COMPLETE TRADE EXECUTION PLAN**")
    
    col_plan1, col_plan2 = st.columns(2)
    with col_plan1:
        st.success(f"""
        **🚀 ENTRY**
        • Buy **{shares:,} shares**
        • Entry: **${entry_price:.2f}**
        • Position: **${position_value:,.0f}**
        """)
        
        st.warning(f"""
        **🛑 STOP LOSS** 
        • **${stop_price:.2f}** 
        • Risk: **{INITIAL_STOP_LOSS*100:.0f}%**
        • Max Loss: **${(entry_price-stop_price)*shares:,.0f}**
        """)
    
    with col_plan2:
        st.info(f"""
        **💰 PARTIAL PROFIT** 
        • Sell **50%** ({shares//2:,} shares)
        • Target: **${partial_price:.2f}** 
        • Gain: **+{PARTIAL_TP*100:.0f}%**
        • Profit: **${(partial_price-entry_price)*(shares//2):,.0f}**
        """)
        
        st.info(f"""
        **📈 TRAILING STOP** 
        • Current ATR: **${latest['ATR']:.2f}**
        • Trail Distance: **{TRAIL_MULT}x ATR = ${atr_trail:.2f}**
        • Trail from highest price
        """)
    
    st.subheader("🚪 **EXIT CONDITIONS** (Monitor Daily)")
    st.error("""
    **SELL IMMEDIATELY if ANY:**
    1. **Price ≤ ${stop_price:.2f}** (Stop Loss)
    2. **Price ≤ [Highest - ${atr_trail:.2f}]** (Trail Stop)
    3. **EMA8 < EMA21 + Price < EMA8** (Trend Break)
    4. **RSI < 35 + Price < EMA8** (Weakness)
    """.replace("${stop_price:.2f}", f"{stop_price:.2f}").replace("${atr_trail:.2f}", f"{atr_trail:.2f}"))

# BACKTEST TRADE HISTORY
with st.expander("📋 Historical Trades (Learn from Backtest)"):
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty and 'pnl_pct' in trades_df.columns:
        st.dataframe(trades_df[['entry_date', 'exit_date', 'entry_price', 'exit_price', 'pnl_pct', 'exit_reason']].tail(10),
                    use_container_width=True)
        wins = len(trades_df[trades_df['pnl_pct'] > 0])
        st.metric("Historical Win Rate", f"{wins/len(trades_df)*100:.0f}% ({wins}/{len(trades_df)})")

st.caption(f"✅ Complete plan with exact entry/exit prices | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
