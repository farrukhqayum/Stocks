import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(layout="wide", page_title="Simple Signals")
st.title("📈 Simple Signals")

# USER INPUTS - FULLY CUSTOMIZABLE
col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Ticker", value="COIN")
    capital = st.number_input("Capital ($)", value=20000, min_value=1000)
with col2:
    period = st.selectbox("Backtest Period", ["2y", "1y", "6mo", "3mo"], index=0)

col3, col4, col5 = st.columns(3)
ema_fast = col3.number_input("EMA Fast", value=8, min_value=5, max_value=20)
ema_slow = col4.number_input("EMA Slow", value=21, min_value=15, max_value=50)
rsi_len = col5.number_input("RSI Length", value=14, min_value=10, max_value=21)

col6, col7, col8 = st.columns(3)
rsi_ema_len = col6.number_input("RSI EMA Length", value=20, min_value=10, max_value=30)
min_adx = col7.number_input("Min ADX", value=12, min_value=10, max_value=25)
stop_loss_pct = col8.number_input("Stop Loss %", value=6.0, min_value=3.0, max_value=12.0)/100

st.markdown("""
**🟢 LIVE TRADE PLAN appears when:**
- EMA Fast > EMA Slow **(Uptrend)**
- ADX > Min ADX **(Trend strength)**
- **RSI > RSI_EMA** *(Your filter)*
- **ANY** Breakout/Pullback/Momentum signal
**📊 Backtest shows profitability of your exact parameters below**
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

def run_backtest(df, capital, ema_fast, ema_slow, rsi_len, rsi_ema_len, min_adx, stop_loss_pct):
    equity = [capital]
    trades = []
    
    in_position = False
    entry_price = 0
    highest_price = 0
    position_size = 0
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        
        # Calculate indicators for this row
        close = df['Close'].iloc[:i+1]
        ema_f = close.ewm(span=ema_fast).mean().iloc[-1]
        ema_s = close.ewm(span=ema_slow).mean().iloc[-1]
        rsi = RSI(close, rsi_len).iloc[-1]
        rsi_ema = RSI(close, rsi_len).ewm(span=rsi_ema_len).mean().iloc[-1]
        adx_val, _, _ = ADX(df.iloc[:i+1], rsi_len)
        adx_val = adx_val.iloc[-1]
        atr_val = row['ATR'] if 'ATR' in df.columns else 0
        
        current_price = row['Close']
        
        # Entry signal
        signal = (
            (ema_f > ema_s) and 
            (rsi > rsi_ema) and 
            (adx_val > min_adx) and
            ((current_price > ema_f) or  # Momentum or breakout
             (current_price > ema_f * 0.97 and rsi < 55 and rsi > 40))  # Pullback
        )
        
        if not in_position and signal and position_size == 0:
            entry_price = current_price
            position_size = capital / entry_price
            highest_price = current_price
            in_position = True
            trades.append({'entry_date': df.index[i], 'entry_price': entry_price})
        
        elif in_position:
            # Update highest
            if current_price > highest_price:
                highest_price = current_price
            
            # Exit conditions
            stop_price = entry_price * (1 - stop_loss_pct)
            trail_stop = highest_price * (1 - 0.04)  # 4% trail
            
            if current_price <= stop_price or current_price <= trail_stop:
                exit_price = current_price
                pnl = (exit_price - entry_price) / entry_price * position_size * entry_price
                capital += pnl
                trades[-1].update({
                    'exit_date': df.index[i], 
                    'exit_price': exit_price, 
                    'pnl': pnl,
                    'pnl_pct': (exit_price - entry_price) / entry_price * 100
                })
                in_position = False
                position_size = 0
        
        # Current equity
        if in_position:
            unrealized = (current_price - entry_price) * position_size
            equity.append(capital + unrealized)
        else:
            equity.append(capital)
    
    equity_series = pd.Series(equity, index=df.index[:len(equity)])
    return equity_series, capital, trades

if st.button("🚀 RUN BACKTEST & SIGNALS", type="primary"):
    st.rerun()

# Load and process data
df_raw = get_data(ticker, period)
if df_raw is None:
    st.error("❌ Failed to load data.")
    st.stop()

# Pre-calculate indicators for full dataset
close = df_raw['Close']
df = df_raw.copy()
df['EMA_FAST'] = close.ewm(span=ema_fast).mean()
df['EMA_SLOW'] = close.ewm(span=ema_slow).mean()
df['RSI'] = RSI(close, rsi_len)
df['RSI_EMA'] = df['RSI'].ewm(span=rsi_ema_len).mean()

df['ADX'], df['PLUS_DI'], df['MINUS_DI'] = ADX(df, rsi_len)
tr1 = df['High'] - df['Low']
tr2 = abs(df['High'] - close.shift())
tr3 = abs(df['Low'] - close.shift())
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df['ATR'] = tr.rolling(rsi_len, min_periods=1).mean()
df['PRICE_CHANGE'] = close.pct_change() * 100
df = df.dropna()

latest = df.iloc[-1]

# Live signals
signals = {
    "BREAKOUT": (latest['EMA_FAST'] > latest['EMA_SLOW'] and 
                latest['RSI'] > latest['RSI_EMA'] and 
                latest['ADX'] > min_adx),
    "PULLBACK": (latest['EMA_FAST'] > latest['EMA_SLOW'] and 
                latest['RSI'] > latest['RSI_EMA'] and 
                latest['Close'] < latest['EMA_FAST'] * 1.03 and
                latest['RSI'] < 55 and latest['RSI'] > 40 and
                latest['ADX'] > min_adx),
    "MOMENTUM": (latest['EMA_FAST'] > latest['EMA_SLOW'] and 
                latest['RSI'] > latest['RSI_EMA'] and
                latest['Close'] > latest['EMA_FAST'] and
                latest['RSI'] > 50 and latest['PRICE_CHANGE'] > 0.5 and
                latest['ADX'] > min_adx)
}
BULL = any(signals.values())

# RUN FULL BACKTEST
equity_series, final_capital, trades = run_backtest(df, capital, ema_fast, ema_slow, rsi_len, rsi_ema_len, min_adx, stop_loss_pct)

# BACKTEST RESULTS TABLE
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("📊 **BACKTEST RESULTS**")
    st.metric("Total Return", f"{(final_capital/capital-1)*100:+.1f}%")
    
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        wins = len(trades_df[trades_df['pnl_pct'] > 0])
        st.metric("Win Rate", f"{wins/len(trades_df)*100:.0f}% ({wins}/{len(trades_df)})")
        st.metric("Profit Factor", f"{trades_df[trades_df['pnl_pct']>0]['pnl'].sum()/abs(trades_df[trades_df['pnl_pct']<0]['pnl'].sum()):.2f}")
    else:
        st.info("No trades triggered - adjust parameters")

with col2:
    st.metric("Final Capital", f"${final_capital:,.0f}")
    st.metric("Max Drawdown", "TBD")

# 4-PANEL CHART WITH REAL EQUITY
fig, axes = plt.subplots(4, 1, figsize=(16, 12), height_ratios=[3, 1, 1, 1.5])
fig.patch.set_facecolor('white')

# Price chart
ax1 = axes[0]
ax1.plot(df.index, df['Close'], color='#2C3E50', linewidth=1.5, label='Close', alpha=0.8)
ax1.plot(df.index, df['EMA_FAST'], color='#3498DB', linewidth=1.5, label=f'EMA{ema_fast}')
ax1.plot(df.index, df['EMA_SLOW'], color='#E74C3C', linewidth=1.5, label=f'EMA{ema_slow}')

# Entry markers
entry_points = df[df['BULL']]
if not entry_points.empty:
    ax1.scatter(entry_points.index, entry_points['Close'], color='limegreen', 
               marker='^', s=100, alpha=0.8, label='Entries', zorder=5)

if BULL:
    ax1.scatter(df.index[-1], latest['Close'], color='limegreen', s=200, 
               marker='^', edgecolors='black', linewidth=2, zorder=10, label='LIVE')

ax1.set_title(f'{ticker} - Backtest: {((final_capital/capital-1)*100):+.1f}%', fontsize=16, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.2)
ax1.set_ylabel('Price ($)')

# RSI + RSI_EMA
ax2 = axes[1]
ax2.plot(df.index, df['RSI'], color='#9B59B6', linewidth=1.5, label=f'RSI({rsi_len})')
ax2.plot(df.index, df['RSI_EMA'], color='#F39C12', linewidth=2, label=f'RSI_EMA({rsi_ema_len})')
ax2.axhline(y=70, color='#E74C3C', linestyle='--', alpha=0.5)
ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
ax2.set_ylabel('RSI')
ax2.legend()
ax2.grid(True, alpha=0.2)
ax2.set_ylim(0, 100)

# ADX
ax3 = axes[2]
ax3.plot(df.index, df['ADX'], color='#E67E22', linewidth=1.5, label='ADX')
ax3.axhline(y=min_adx, color='#E74C3C', linestyle='--', alpha=0.5, label=f'Min={min_adx}')
ax3.set_ylabel('ADX')
ax3.legend()
ax3.grid(True, alpha=0.2)

# REAL EQUITY CURVE
ax4 = axes[3]
ax4.plot(equity_series.index, equity_series, color='#27AE60', linewidth=3, label='Equity')
ax4.axhline(y=capital, color='gray', linestyle='--', alpha=0.5, label='Start')
ax4.fill_between(equity_series.index, capital, equity_series, alpha=0.2, color='#27AE60')
ax4.set_ylabel('Portfolio ($)')
ax4.set_xlabel('Date')
ax4.legend()
ax4.grid(True, alpha=0.2)

plt.tight_layout()
st.pyplot(fig)

# SIGNAL DEBUGGER
st.subheader("🔍 **LIVE SIGNAL STATUS**")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Trend", "✅" if latest['EMA_FAST'] > latest['EMA_SLOW'] else "❌")
with col2:
    st.metric("RSI>EMA", "✅" if latest['RSI'] > latest['RSI_EMA'] else "❌")
with col3:
    st.metric("ADX", f"{latest['ADX']:.1f}", f">{min_adx}" if latest['ADX'] > min_adx else "")
with col4:
    st.metric("Signal", "🟢 LIVE" if BULL else "🔴 WAIT")

# LIVE TRADE PLAN
if BULL:
    st.success(f"🎯 **LIVE SIGNAL ACTIVE** - TRADE NOW!")
    st.balloons()
    
    entry_price = latest['Close']
    shares = int((capital * 0.95) / entry_price)
    stop_price = entry_price * (1 - stop_loss_pct)
    
    st.subheader("📋 **EXECUTE THIS TRADE**")
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**BUY {shares:,} shares @ ${entry_price:.2f}**")
    with col2:
        st.warning(f"**STOP LOSS: ${stop_price:.2f}**")

# TRADE HISTORY
with st.expander("📋 Trade History", expanded=False):
    if trades:
        trades_df = pd.DataFrame(trades)
        st.dataframe(trades_df[['entry_date', 'exit_date', 'pnl_pct', 'pnl']].tail(10))
    else:
        st.info("No completed trades with current parameters")

st.caption(f"✅ Real backtest with your parameters | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
