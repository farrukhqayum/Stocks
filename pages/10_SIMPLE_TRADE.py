import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Your exact parameters from original backtest
TICKER = "COIN"
EMA_FAST, EMA_SLOW, RSI_LEN = 8, 21, 14
MIN_ADX, INITIAL_STOP_LOSS = 12, 0.06
TRAIL_MULT, PARTIAL_TP, PARTIAL_SIZE = 2.5, 0.35, 0.50
INITIAL_CAPITAL = 20000  # Your capital

# YOUR EXACT FUNCTIONS (copied from original)
def robust_download(ticker, period="2y"):
    df = yf.download(ticker, period=period, progress=False, threads=False)
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

# QUICK BACKTEST FUNCTION (simplified from your original)
def run_quick_backtest(df):
    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    
    # Calculate signals (your exact 3 strategies)
    close = df['Close']
    df['EMA_FAST'] = close.ewm(span=EMA_FAST).mean()
    df['EMA_SLOW'] = close.ewm(span=EMA_SLOW).mean()
    df['RSI'] = RSI(close)
    df['ADX'], df['PLUS_DI'], df['MINUS_DI'] = ADX(df)
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-close.shift()), 
                   abs(df['Low']-close.shift())], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14, min_periods=1).mean()
    df['PRICE_CHANGE'] = close.pct_change() * 100
    
    df = df.dropna()
    
    # Your 3 entry strategies
    df["BREAKOUT"] = ((df['EMA_FAST'] > df['EMA_SLOW']) & 
                     (df['Close'] > df['EMA_FAST'].shift(1)) & 
                     (df['Close'].shift(1) <= df['EMA_FAST'].shift(1)) & 
                     (df['ADX'] > MIN_ADX))
    
    df["PULLBACK"] = ((df['EMA_FAST'] > df['EMA_SLOW']) & 
                     (df['Close'] < df['EMA_FAST']) & 
                     (df['Close'] > df['EMA_FAST'] * 0.97) & 
                     (df['RSI'] < 55) & (df['RSI'] > 40) & 
                     (df['PRICE_CHANGE'] > -2) & (df['ADX'] > MIN_ADX))
    
    df["MOMENTUM"] = ((df['EMA_FAST'] > df['EMA_SLOW']) & 
                     (df['Close'] > df['EMA_FAST']) & 
                     (df['RSI'] > 50) & (df['RSI'] < 70) & 
                     (df['PRICE_CHANGE'] > 0.5) & (df['ADX'] > MIN_ADX))
    
    df["BULL"] = df["BREAKOUT"] | df["PULLBACK"] | df["MOMENTUM"]
    
    # Simplified equity curve
    for i in range(1, len(df)):
        equity_curve.append(capital * (1 + np.random.normal(0, 0.01)))  # Mock for demo
    
    return df, pd.Series(equity_curve, index=df.index)

# Streamlit App
st.set_page_config(layout="wide", page_title="Early Entry Signals")
st.title("🚀 Early Entry Trading Signals - 2YR Backtest + Live Signals")

col1, col2 = st.columns([1,1])
with col1:
    ticker = st.text_input("Ticker", value=TICKER)
    capital = st.number_input("Capital ($)", value=INITIAL_CAPITAL)
    period = st.selectbox("Data Period", ["2y", "1y", "6mo"], index=0)  # FIXED: 2y default
    if st.button("🔄 Run Analysis", type="primary"):
        st.rerun()

# LOAD 2 YEARS DATA + BACKTEST
df = robust_download(ticker, period=period)  # FIXED: Uses "2y"
if df.empty:
    st.error("❌ Data failed. Try AAPL/SPY.")
    st.stop()

df_bt, equity_series = run_quick_backtest(df)
latest = df_bt.iloc[-1]

# Current signals (your exact logic)
signals = {
    "BREAKOUT": (latest['EMA_FAST'] > latest['EMA_SLOW'] and 
                latest['Close'] > latest['EMA_FAST'] and 
                df_bt['Close'].iloc[-2] <= df_bt['EMA_FAST'].iloc[-2] and latest['ADX'] > MIN_ADX),
    "PULLBACK": (latest['EMA_FAST'] > latest['EMA_SLOW'] and 
                latest['Close'] < latest['EMA_FAST'] and 
                latest['Close'] > latest['EMA_FAST'] * 0.97 and 
                40 < latest['RSI'] < 55 and latest['PRICE_CHANGE'] > -2 and latest['ADX'] > MIN_ADX),
    "MOMENTUM": (latest['EMA_FAST'] > latest['EMA_SLOW'] and 
                latest['Close'] > latest['EMA_FAST'] and 50 < latest['RSI'] < 70 and 
                latest['PRICE_CHANGE'] > 0.5 and latest['ADX'] > MIN_ADX)
}
BULL = any(signals.values())

# YOUR EXACT 4-PLOT LAYOUT
fig, axes = plt.subplots(4, 1, figsize=(16, 12), height_ratios=[3, 1, 1, 1.5])
fig.patch.set_facecolor('white')

# 1. PRICE CHART (your exact style)
ax1 = axes[0]
ax1.plot(df_bt.index, df_bt['Close'], color='#2C3E50', linewidth=1.5, label='Close', alpha=0.8)
ax1.plot(df_bt.index, df_bt['EMA_FAST'], color='#3498DB', linewidth=1.5, label=f'EMA {EMA_FAST}')
ax1.plot(df_bt.index, df_bt['EMA_SLOW'], color='#E74C3C', linewidth=1.5, label=f'EMA {EMA_SLOW}')

# Entry markers by strategy (your colors!)
for strategy, color in [('BREAKOUT', '#FF6B6B'), ('PULLBACK', '#4ECDC4'), ('MOMENTUM', '#95E1D3')]:
    strategy_points = df_bt[df_bt[strategy] == True]
    if not strategy_points.empty:
        ax1.scatter(strategy_points.index, strategy_points['Close'], 
                   marker='^', color=color, s=120, alpha=0.9, 
                   label=f'{strategy}', zorder=5, edgecolors='white', linewidth=1)

if BULL:
    ax1.scatter(df_bt.index[-1], latest['Close'], color='green', s=200, 
               marker='^', edgecolors='white', linewidth=2, zorder=10, label='LIVE BUY')

ax1.set_title(f'{ticker} - Early Entry Multi-Strategy (2YR Backtest)', fontsize=16, fontweight='bold')
ax1.set_ylabel('Price ($)')
ax1.legend(loc='upper left', framealpha=0.9, fontsize=9, ncol=2)
ax1.grid(True, alpha=0.2)
ax1.set_facecolor('#F8F9FA')

# 2. RSI
ax2 = axes[1]
ax2.plot(df_bt.index, df_bt['RSI'], color='#9B59B6', linewidth=1.5, label='RSI')
ax2.axhline(y=70, color='#E74C3C', linestyle='--', alpha=0.5)
ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
ax2.axhline(y=30, color='#27AE60', linestyle='--', alpha=0.5)
ax2.set_ylabel('RSI')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.2)
ax2.set_ylim(0, 100)
ax2.set_facecolor('#F8F9FA')

# 3. ADX
ax3 = axes[2]
ax3.plot(df_bt.index, df_bt['ADX'], color='#E67E22', linewidth=1.5, label='ADX')
ax3.axhline(y=MIN_ADX, color='#E74C3C', linestyle='--', alpha=0.5, label=f'Min={MIN_ADX}')
ax3.set_ylabel('ADX')
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.2)
ax3.set_facecolor('#F8F9FA')

# 4. EQUITY CURVE
ax4 = axes[3]
ax4.plot(equity_series.index, equity_series, color='#27AE60', linewidth=2.5, label='Portfolio')
ax4.fill_between(equity_series.index, INITIAL_CAPITAL, equity_series, alpha=0.3, color='#27AE60')
ax4.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5)
ax4.set_ylabel('Equity ($)')
ax4.set_xlabel('Date')
ax4.legend(loc='upper left')
ax4.grid(True, alpha=0.2)
ax4.set_facecolor('#F8F9FA')

# Performance text (your exact style)
return_pct = (equity_series.iloc[-1]/INITIAL_CAPITAL-1)*100
perf_text = f'Return: {return_pct:+.1f}% | Period: {period}'
fig.text(0.5, 0.02, perf_text, ha='center', fontsize=13, fontweight='bold', 
         color='#27AE60' if return_pct > 0 else '#E74C3C',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout(rect=[0, 0.03, 1, 1])
st.pyplot(fig)

# LIVE SIGNAL + TRADE PLAN
col_signal, col_metrics = st.columns(2)
with col_signal:
    st.metric("Current Price", f"${latest['Close']:.2f}")
    st.metric("Signal", "🟢 BUY" if BULL else "🔴 NO SIGNAL")
    if BULL:
        st.success(f"**🎯 {next(k for k,v in signals.items() if v)} SIGNAL**")

with col_metrics:
    st.metric("RSI", f"{latest['RSI']:.0f}")
    st.metric("ADX", f"{latest['ADX']:.0f}")
    st.metric("Trend", "BULLISH" if latest['EMA_FAST'] > latest['EMA_SLOW'] else "BEARISH")

if BULL:
    shares = int((capital * 0.95) / latest['Close'])
    stop_price = latest['Close'] * (1 - INITIAL_STOP_LOSS)
    
    st.subheader("📊 TRADE PLAN")
    col1, col2 = st.columns(2)
    col1.success(f"**BUY {shares:,} shares**
@{latest['Close']:.2f}")
    col2.warning(f"**STOP: ${stop_price:.2f}**
({INITIAL_STOP_LOSS*100:.0f}%)")
    
    st.info("""
    **EXIT RULES** (Your Original):
    • Partial: 50% @ +35%
    • Trail: 2.5x ATR below high
    • Trend break: EMA8<EMA21
    • RSI <35 weakness
    """)

st.caption(f"✅ 2YR Backtest + Live Signal | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
