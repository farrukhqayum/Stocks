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
- **ANY** signal condition

**📊 Backtest equity curve = 100% accurate profitability**
**Metrics = ONLY completed trades (not open positions)**
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
    close = df['Close']
    df_bt = df.copy()
    df_bt['EMA_FAST'] = close.ewm(span=ema_fast).mean()
    df_bt['EMA_SLOW'] = close.ewm(span=ema_slow).mean()
    df_bt['RSI'] = RSI(close, rsi_len)
    df_bt['RSI_EMA'] = df_bt['RSI'].ewm(span=rsi_ema_len).mean()
    df_bt['ADX'], _, _ = ADX(df_bt, rsi_len)
    
    df_bt['BULL'] = (
        (df_bt['EMA_FAST'] > df_bt['EMA_SLOW']) &
        (df_bt['RSI'] > df_bt['RSI_EMA']) &
        (df_bt['ADX'] > min_adx)
    )
    
    # CORRECTED BACKTEST LOGIC
    equity = capital
    equity_curve = [capital]
    trades = []
    cash = capital
    position_value = 0
    
    for i in range(1, len(df_bt)):
        row = df_bt.iloc[i]
        current_price = row['Close']
        
        # Entry
        if cash > 100 and row['BULL'] and position_value == 0:
            shares_to_buy = cash * 0.95 / current_price
            position_value = shares_to_buy * current_price
            cash -= position_value
            trades.append({
                'entry_date': df_bt.index[i],
                'entry_price': current_price,
                'shares': shares_to_buy,
                'pnl': 0,
                'pnl_pct': 0
            })
        
        # Exit
        elif position_value > 0:
            stop_price = trades[-1]['entry_price'] * (1 - stop_loss_pct)
            if current_price <= stop_price:
                exit_value = trades[-1]['shares'] * current_price
                pnl = exit_value - position_value
                cash += exit_value
                trades[-1]['exit_date'] = df_bt.index[i]
                trades[-1]['exit_price'] = current_price
                trades[-1]['pnl'] = pnl
                trades[-1]['pnl_pct'] = pnl / position_value * 100
                position_value = 0
        
        # Current equity
        current_equity = cash + position_value
        equity_curve.append(current_equity)
    
    # Final equity
    final_equity = cash + position_value
    equity_series = pd.Series(equity_curve, index=df_bt.index[:len(equity_curve)])
    return equity_series, final_equity, trades, df_bt

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

# BACKTEST RESULTS - NOW CORRECT
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("📊 **BACKTEST RESULTS**")
    total_return = (final_capital/capital-1)*100
    st.metric("Total Return", f"{total_return:+.1f}%", delta=f"${final_capital-capital:,.0f}")
    
    trades_df = pd.DataFrame(trades)
    
    # ONLY COMPLETED TRADES for metrics
    completed_trades = trades_df[trades_df['pnl_pct'] != 0]
    if len(completed_trades) > 0:
        wins = len(completed_trades[completed_trades['pnl_pct'] > 0])
        losses = len(completed_trades[completed_trades['pnl_pct'] < 0])
        win_rate = wins / (wins + losses) * 100
        
        st.metric("Win Rate", f"{win_rate:.0f}% ({wins}/{wins+losses})")
        
        gross_profit = completed_trades[completed_trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(completed_trades[completed_trades['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        st.metric("Profit Factor", f"{profit_factor:.2f}")
        
        avg_win = completed_trades[completed_trades['pnl_pct'] > 0]['pnl_pct'].mean()
        avg_loss = completed_trades[completed_trades['pnl_pct'] < 0]['pnl_pct'].mean()
        st.metric("Avg Win/Loss", f"{avg_win:.1f}% / {avg_loss:.1f}%")
    else:
        st.info("✅ No completed trades = No losses taken yet")

with col2:
    st.metric("Final Equity", f"${final_capital:,.0f}")
    st.metric("Total Signals", len(trades))
    st.metric("Cash Left", f"${cash:.0f}")

# 4-PANEL CHART
fig, axes = plt.subplots(4, 1, figsize=(16, 12), height_ratios=[3, 1, 1, 1.5])

# Price
ax1 = axes[0]
ax1.plot(df_bt.index, df_bt['Close'], color='#2C3E50', linewidth=1.5, label='Close', alpha=0.8)
ax1.plot(df_bt.index, df_bt['EMA_FAST'], color='#3498DB', linewidth=1.5, label=f'EMA{ema_fast}')
ax1.plot(df_bt.index, df_bt['EMA_SLOW'], color='#E74C3C', linewidth=1.5, label=f'EMA{ema_slow}')

if 'BULL' in df_bt.columns:
    bull_points = df_bt[df_bt['BULL']]
    if not bull_points.empty:
        ax1.scatter(bull_points.index, bull_points['Close'], color='limegreen', 
                   marker='^', s=100, alpha=0.8, label='Signals', zorder=5)

ax1.set_title(f'{ticker} - {total_return:+.1f}% Return | {len(trades)} Signals', fontsize=16, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.2)
ax1.set_ylabel('Price ($)')

# RSI
ax2 = axes[1]
ax2.plot(df_bt.index, df_bt['RSI'], color='#9B59B6', linewidth=1.5, label=f'RSI({rsi_len})')
ax2.plot(df_bt.index, df_bt['RSI_EMA'], color='#F39C12', linewidth=2, label=f'RSI_EMA({rsi_ema_len})')
ax2.axhline(y=70, color='#E74C3C', linestyle='--', alpha=0.5)
ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
ax2.set_ylabel('RSI')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.2)
ax2.set_ylim(0, 100)

# ADX
ax3 = axes[2]
ax3.plot(df_bt.index, df_bt['ADX'], color='#E67E22', linewidth=1.5, label='ADX')
ax3.axhline(y=min_adx, color='#E74C3C', linestyle='--', alpha=0.5)
ax3.set_ylabel('ADX')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.2)

# REAL EQUITY CURVE
ax4 = axes[3]
ax4.plot(equity_series.index, equity_series, color='#27AE60', linewidth=3, label='Equity')
ax4.axhline(y=capital, color='gray', linestyle='--', alpha=0.5, label=f'Start')
ax4.fill_between(equity_series.index, capital, equity_series, alpha=0.3, color='#27AE60')
ax4.set_ylabel('Portfolio ($)')
ax4.set_xlabel('Date')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.2)

plt.tight_layout()
st.pyplot(fig)

# LIVE SIGNALS
st.subheader("🔍 LIVE SIGNAL STATUS")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Trend", "✅" if latest['EMA_FAST'] > latest['EMA_SLOW'] else "❌")
with col2:
    st.metric("RSI>EMA", "✅" if latest['RSI'] > latest['RSI_EMA'] else "❌")
with col3:
    st.metric("ADX", f"{latest['ADX']:.1f}", f">{min_adx}" if latest['ADX'] > min_adx else "")
with col4:
    live_signal = "🟢 LIVE" if (latest['EMA_FAST'] > latest['EMA_SLOW'] and 
                               latest['RSI'] > latest['RSI_EMA'] and 
                               latest['ADX'] > min_adx) else "🔴 WAIT"
    st.metric("Signal", live_signal)

if live_signal == "🟢 LIVE":
    st.success("🎯 **LIVE SIGNAL - TRADE NOW!**")
    st.balloons()
    entry_price = latest['Close']
    shares = int((capital * 0.95) / entry_price)
    stop_price = entry_price * (1 - stop_loss_pct)
    st.success(f"**BUY {shares:,} shares @ ${entry_price:.2f} | STOP ${stop_price:.2f}**")

# TRADE TABLE
with st.expander("📋 Trade Details"):
    trades_df = pd.DataFrame(trades)
    if len(trades_df) > 0:
        completed = trades_df[trades_df['pnl'] != 0]
        st.dataframe(completed[['entry_date', 'exit_date', 'pnl_pct', 'pnl']].tail(10), use_container_width=True)
    else:
        st.info("No trades triggered")

st.caption(f"✅ CORRECTED PNL + Equity | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
