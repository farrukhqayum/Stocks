import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
from matplotlib.dates import DateFormatter

# Your exact parameters and functions from backtest
TICKER = "COIN"
EMA_FAST, EMA_SLOW, RSI_LEN = 8, 21, 14
MIN_ADX, INITIAL_STOP_LOSS = 12, 0.06
INITIAL_CAPITAL = 20000  # Your capital [memory:14]

@st.cache_data(ttl=300)  # Refresh every 5min
def get_data(ticker):
    df = yf.download(ticker, period="60d", progress=False, threads=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    if 'Adj Close' in df.columns: df['Close'] = df['Adj Close']
    
    close = df['Close']
    df['EMA_FAST'] = close.ewm(span=EMA_FAST).mean()
    df['EMA_SLOW'] = close.ewm(span=EMA_SLOW).mean()
    df['RSI'] = RSI(close)
    df['ADX'], df['PLUS_DI'], df['MINUS_DI'] = ADX(df)
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-close.shift()), 
                   abs(df['Low']-close.shift())], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14, min_periods=1).mean()
    df['PRICE_CHANGE'] = close.pct_change() * 100
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

# Streamlit App
st.set_page_config(layout="wide", page_title="Early Entry Signals")
st.title("🚀 Early Entry Trading Signals")

col1, col2 = st.columns([1,1])
with col1:
    ticker = st.text_input("Ticker", value=TICKER)
    capital = st.number_input("Capital ($)", value=INITIAL_CAPITAL)
    if st.button("🔄 Generate Signal", type="primary"):
        st.rerun()

df = get_data(ticker)
if df is None:
    st.error("❌ Data failed to load. Try AAPL or SPY.")
    st.stop()

latest = df.iloc[-1]
signals = {
    "BREAKOUT": (latest['EMA_FAST'] > latest['EMA_SLOW'] and 
                latest['Close'] > latest['EMA_FAST'] and 
                df['Close'].iloc[-2] <= df['EMA_FAST'].iloc[-2] and latest['ADX'] > MIN_ADX),
    "PULLBACK": (latest['EMA_FAST'] > latest['EMA_SLOW'] and 
                latest['Close'] < latest['EMA_FAST'] and 
                latest['Close'] > latest['EMA_FAST'] * 0.97 and 
                40 < latest['RSI'] < 55 and latest['PRICE_CHANGE'] > -2 and latest['ADX'] > MIN_ADX),
    "MOMENTUM": (latest['EMA_FAST'] > latest['EMA_SLOW'] and 
                latest['Close'] > latest['EMA_FAST'] and 50 < latest['RSI'] < 70 and 
                latest['PRICE_CHANGE'] > 0.5 and latest['ADX'] > MIN_ADX)
}
BULL = any(signals.values())

# Main Dashboard
col_signal, col_metrics = st.columns(2)
with col_signal:
    st.metric("Current Price", f"${latest['Close']:.2f}")
    signal_color = "🟢 BUY" if BULL else "🔴 NO SIGNAL"
    st.metric("Signal", signal_color, delta=None)
    
    if BULL:
        st.success(f"**🎯 {next(k for k,v in signals.items() if v)} SIGNAL**")
        st.balloons()

with col_metrics:
    st.metric("RSI", f"{latest['RSI']:.0f}", delta=None)
    st.metric("ADX", f"{latest['ADX']:.0f}", delta=f">{MIN_ADX}" if latest['ADX'] > MIN_ADX else None)
    st.metric("Trend", "BULLISH" if latest['EMA_FAST'] > latest['EMA_SLOW'] else "BEARISH")

# Position Sizing & Instructions
if BULL:
    risk_amount = capital * 0.95  # 95% allocation
    position_value = risk_amount
    shares = int(position_value / latest['Close'])
    stop_price = latest['Close'] * (1 - INITIAL_STOP_LOSS)
    risk_per_share = latest['Close'] - stop_price
    dollars_at_risk = shares * risk_per_share
    
    st.subheader("📊 TRADE PLAN")
    col_buy, col_stop = st.columns(2)
    with col_buy:
        st.success(f"**BUY {shares:,} shares @ ${latest['Close']:.2f}**")
        st.info(f"Position: ${position_value:,.0f} ({shares} x ${latest['Close']:.2f})")
    with col_stop:
        st.warning(f"**STOP LOSS: ${stop_price:.2f}** ({INITIAL_STOP_LOSS*100:.0f}%)")
        st.info(f"Risk: ${dollars_at_risk:,.0f} ({dollars_at_risk/capital*100:.1f}% of capital)")
    
    st.subheader("🚪 EXIT RULES")
    st.info("""
    - **Partial Profit**: Sell 50% at +35% gain
    - **Trailing Stop**: 2.5x ATR below highest price
    - **Trend Break**: EMA8 < EMA21 + price < EMA8
    - **RSI Weakness**: RSI < 35 + price < EMA8
    """)

# Matplotlib Chart
st.subheader("📈 Price Chart with Signals")
fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
ax.plot(df.index, df['Close'], color='#2C3E50', linewidth=2, label='Close', alpha=0.8)
ax.plot(df.index, df['EMA_FAST'], color='#3498DB', linewidth=2, label=f'EMA {EMA_FAST}', alpha=0.8)
ax.plot(df.index, df['EMA_SLOW'], color='#E74C3C', linewidth=2, label=f'EMA {EMA_SLOW}', alpha=0.8)

if BULL:
    ax.scatter(df.index[-1], latest['Close'], color='green', s=200, 
              marker='^', edgecolors='white', linewidth=2, zorder=5, label='BUY SIGNAL')

ax.set_title(f'{ticker} - Early Entry Multi-Strategy', fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_facecolor('#F8F9FA')
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Auto-refreshes every 5min")
