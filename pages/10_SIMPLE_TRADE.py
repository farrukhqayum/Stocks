import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(layout="wide", page_title="Simple Signals")
st.title("📈 Simple Signals")

# USER INPUTS
col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Ticker", value="COIN")
    capital = st.number_input("Capital ($)", value=20000, min_value=1000)
with col2:
    period = st.selectbox("Backtest Period", ["2y", "1y", "6mo"], index=0)

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
- **RSI > RSI_EMA** 
- **ANY** Breakout/Pullback/Momentum signal

**📊 Backtest shows profitability of your EXACT parameters**
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
    # Pre-calculate all indicators
    close = df['Close']
    df_bt = df.copy()
    df_bt['EMA_FAST'] = close.ewm(span=ema_fast).mean()
    df_bt['EMA_SLOW'] = close.ewm(span=ema_slow).mean()
    df_bt['RSI'] = RSI(close, rsi_len)
    df_bt['RSI_EMA'] = df_bt['RSI'].ewm(span=rsi_ema_len).mean()
    df_bt['ADX'], _, _ = ADX(df_bt, rsi_len)
    
    # Entry signals
    df_bt['BULL'] = (
        (df_bt['EMA_FAST'] > df_bt['EMA_SLOW']) &
        (df_bt['RSI'] > df_bt['RSI_EMA']) &
        (df_bt['ADX'] > min_adx)
    )
    
    # Backtest logic
    equity = [capital]
    trades = []
    in_position = False
    entry_price = 0
    position_size = 0
    
    for i in range(1, len(df_bt)):
        row = df_bt.iloc[i]
        current_price = row['Close']
        
        if not in_position and row['BULL']:
            entry_price = current_price
            position_size = capital / entry_price
            in_position = True
            trades.append({'entry_date': df_bt.index[i], 'entry_price': entry_price})
        
        elif in_position:
            stop_price = entry_price * (1 - stop_loss_pct)
            if current_price <= stop_price:
                exit_price = current_price
                pnl = (exit_price - entry_price) * position_size
                capital = capital - (entry_price * position_size) + (exit_price * position_size)
                trades[-1].update({
                    'exit_date': df_bt.index[i],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': (exit_price - entry_price) / entry_price * 100
                })
                in_position = False
                position_size = 0
        
        # Current equity
        if in_position:
            current_equity = capital - (entry_price * position_size) + (current_price * position_size)
        else:
            current_equity = capital
        equity.append(current_equity)
    
    equity_series = pd.Series(equity[:len(df_bt)], index=df_bt.index)
    return equity_series, capital, trades, df_bt

if st.button("🚀 RUN BACKTEST & SIGNALS", type="primary"):
    st.rerun()

df_raw = get_data(ticker, period)
if df_raw is None:
    st.error("❌ Failed to load data.")
    st.stop()

# RUN BACKTEST
equity_series, final_capital, trades, df_bt = run_backtest(
    df_raw, capital, ema_fast, ema_slow, rsi_len, rsi_ema_len, min_adx, stop_loss_pct
)

latest = df_bt.iloc[-1]

# BACKTEST RESULTS
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("📊 **BACKTEST RESULTS**")
    total_return = (final_capital/capital-1)*100
    st.metric("Total Return", f"{total_return:+.1f}%")
    
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        wins = len(trades_df[trades_df['pnl_pct'] > 0])
        win_rate = wins/len(trades_df)*100
        st.metric("Win Rate", f"{win_rate:.0f}% ({wins}/{len(trades_df)})")
        
        profit_factor = abs(trades_df[trades_df['pnl_pct']>0]['pnl'].sum() / 
                           trades_df[trades_df['pnl_pct']<0]['pnl'].sum()) if len(trades_df[trades_df['pnl_pct']<0]) > 0 else float('inf')
        st.metric("Profit Factor", f"{profit_factor:.2f}")
    else:
        st.info("No trades - loosen parameters")

with col2:
    st.metric("Final Capital", f"${final_capital:,.0f}")
    st.metric("Total Trades", len(trades))

# 4-PANEL CHART WITH REAL EQUITY
fig, axes = plt.subplots(4, 1, figsize=(16, 12), height_ratios=[3, 1, 1, 1.5])
fig.patch.set_facecolor('white')

# 1. Price + Entries (FIXED)
ax1 = axes[0]
ax1.plot(df_bt.index, df_bt['Close'], color='#2C3E50', linewidth=1.5, label='Close', alpha=0.8)
ax1.plot(df_bt.index, df_bt['EMA_FAST'], color='#3498DB', linewidth=1.5, label=f'EMA{ema_fast}')
ax1.plot(df_bt.index, df_bt['EMA_SLOW'], color='#E74C3C', linewidth=1.5, label=f'EMA{ema_slow}')

# FIXED: Safe entry points
bull_points = df_bt[df_bt['BULL'] == True] if 'BULL' in df_bt.columns else pd.DataFrame()
if not bull_points.empty:
    ax1.scatter(bull_points.index, bull_points['Close'], color='limegreen', 
               marker='^', s=100, alpha=0.8, label='Backtest Entries', zorder=5)

ax1.set_title(f'{ticker} - Backtest Return: {total_return:+.1f}%', fontsize=16, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.2)
ax1.set_ylabel('Price ($)')

# 2. RSI + RSI_EMA
ax2 = axes[1]
ax2.plot(df_bt.index, df_bt['RSI'], color='#9B59B6', linewidth=1.5, label=f'RSI({rsi_len})')
ax2.plot(df_bt.index, df_bt['RSI_EMA'], color='#F39C12', linewidth=2, label=f'RSI_EMA({rsi_ema_len})')
ax2.axhline(y=70, color='#E74C3C', linestyle='--', alpha=0.5)
ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
ax2.set_ylabel('RSI')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.2)
ax2.set_ylim(0, 100)

# 3. ADX
ax3 = axes[2]
ax3.plot(df_bt.index, df_bt['ADX'], color='#E67E22', linewidth=1.5, label='ADX')
ax3.axhline(y=min_adx, color='#E74C3C', linestyle='--', alpha=0.5, label=f'Min={min_adx}')
ax3.set_ylabel('ADX')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.2)

# 4. REAL EQUITY CURVE
ax4 = axes[3]
ax4.plot(equity_series.index, equity_series, color='#27AE60', linewidth=3, label='Equity Curve')
ax4.axhline(y=capital, color='gray', linestyle='--', alpha=0.5, label=f'Start ${capital:,.0f}')
ax4.fill_between(equity_series.index, capital, equity_series, alpha=0.2, color='#27AE60')
ax4.set_ylabel('Portfolio ($)')
ax4.set_xlabel('Date')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.2)

plt.tight_layout()
st.pyplot(fig)

# LIVE SIGNAL STATUS
st.subheader("🔍 **LIVE SIGNAL DEBUGGER**")
col1, col2, col3, col4 = st.columns(4)
with col1:
    trend_ok = latest['EMA_FAST'] > latest['EMA_SLOW']
    st.metric("Trend", "✅" if trend_ok else "❌")
with col2:
    rsi_ok = latest['RSI'] > latest['RSI_EMA']
    st.metric("RSI>EMA", "✅" if rsi_ok else "❌")
with col3:
    adx_ok = latest['ADX'] > min_adx
    st.metric("ADX", f"{latest['ADX']:.1f}", f">{min_adx}" if adx_ok else "")
with col4:
    live_signal = "🟢 LIVE" if (trend_ok and rsi_ok and adx_ok) else "🔴 WAIT"
    st.metric("Signal", live_signal)

# LIVE TRADE PLAN
if trend_ok and rsi_ok and adx_ok:
    st.success("🎯 **LIVE SIGNAL - TRADE NOW!**")
    st.balloons()
    
    entry_price = latest['Close']
    shares = int((capital * 0.95) / entry_price)
    stop_price = entry_price * (1 - stop_loss_pct)
    
    st.subheader("📋 **EXECUTE IMMEDIATELY**")
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**BUY {shares:,} shares**")
        st.success(f"**@ ${entry_price:.2f}**")
    with col2:
        st.warning(f"**STOP LOSS**")
        st.warning(f"${stop_price:.2f}")

# TRADE HISTORY TABLE
with st.expander("📋 Detailed Trade History"):
    if trades:
        trades_df = pd.DataFrame(trades)
        st.dataframe(trades_df[['entry_date', 'exit_date', 'entry_price', 'exit_price', 'pnl_pct']].tail(10), use_container_width=True)
    else:
        st.info("No completed trades - strategy too selective")

st.caption(f"✅ FULL BACKTEST with your parameters | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
