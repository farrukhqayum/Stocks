
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
- **ALL** conditions met

**📊 Backtest shows profitability of your EXACT parameters**
""")

@st.cache_data(ttl=300)
def get_data(ticker, period="2y"):
    try:
        df = yf.download(ticker, period=period, progress=False, threads=False)
        if df.empty: 
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        return df.dropna()
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None

def RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs)).fillna(50)

def ADX(df, period=14):
    high, low, close = df["High"].squeeze(), df["Low"].squeeze(), df["Close"].squeeze()
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
    
    atr = tr.rolling(window=period, min_periods=1).mean()
    plus_di = 100 * (plus_dm.rolling(period, min_periods=1).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period, min_periods=1).mean() / atr)
    
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(period, min_periods=1).mean()
    return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)

def run_backtest(df, capital, ema_fast, ema_slow, rsi_len, rsi_ema_len, min_adx, stop_loss_pct):
    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    
    df_bt = df.copy()
    df_bt['EMA_FAST'] = close.ewm(span=ema_fast, adjust=False).mean()
    df_bt['EMA_SLOW'] = close.ewm(span=ema_slow, adjust=False).mean()
    df_bt['RSI'] = RSI(close, rsi_len)
    df_bt['RSI_EMA'] = df_bt['RSI'].ewm(span=rsi_ema_len, adjust=False).mean()
    df_bt['ADX'], _, _ = ADX(df_bt, 14)
    
    # ATR for trailing stop
    tr = pd.concat([high-low, abs(high-close.shift()), abs(low-close.shift())], axis=1).max(axis=1)
    df_bt['ATR'] = tr.rolling(14, min_periods=1).mean()
    
    df_bt = df_bt.dropna()
    
    # Entry signal
    df_bt['BULL'] = (
        (df_bt['EMA_FAST'] > df_bt['EMA_SLOW']) &
        (df_bt['RSI'] > df_bt['RSI_EMA']) &
        (df_bt['ADX'] > min_adx)
    )
    
    # Backtest
    current_capital = capital
    equity = []
    trades = []
    in_position = False
    entry_price = 0
    entry_shares = 0
    stop_price = 0
    
    for i in range(len(df_bt)):
        row = df_bt.iloc[i]
        current_price = row['Close']
        current_low = row['Low']
        
        if not in_position:
            if row['BULL']:
                entry_price = current_price
                entry_shares = int((current_capital * 0.95) / entry_price)
                if entry_shares > 0:
                    in_position = True
                    stop_price = entry_price * (1 - stop_loss_pct)
                    trades.append({
                        'entry_date': df_bt.index[i],
                        'entry_price': entry_price,
                        'shares': entry_shares
                    })
        
        elif in_position:
            # Check stop loss
            if current_low <= stop_price:
                exit_price = stop_price
                pnl = (exit_price - entry_price) * entry_shares
                current_capital += pnl
                
                trades[-1].update({
                    'exit_date': df_bt.index[i],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': ((exit_price - entry_price) / entry_price) * 100
                })
                in_position = False
                entry_shares = 0
        
        # Calculate equity
        if in_position:
            current_equity = current_capital + ((current_price - entry_price) * entry_shares)
        else:
            current_equity = current_capital
        
        equity.append(current_equity)
    
    # Close any open position
    if in_position:
        exit_price = df_bt['Close'].iloc[-1]
        pnl = (exit_price - entry_price) * entry_shares
        current_capital += pnl
        
        trades[-1].update({
            'exit_date': df_bt.index[-1],
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': ((exit_price - entry_price) / entry_price) * 100
        })
    
    equity_series = pd.Series(equity, index=df_bt.index)
    return equity_series, current_capital, trades, df_bt

# Load data
df_raw = get_data(ticker, period)
if df_raw is None or df_raw.empty:
    st.error("❌ Failed to load data. Check ticker symbol.")
    st.stop()

# Run backtest
try:
    equity_series, final_capital, trades, df_bt = run_backtest(
        df_raw, capital, ema_fast, ema_slow, rsi_len, rsi_ema_len, min_adx, stop_loss_pct
    )
except Exception as e:
    st.error(f"❌ Backtest error: {e}")
    st.stop()

if df_bt.empty:
    st.error("❌ Insufficient data after calculations")
    st.stop()

latest = df_bt.iloc[-1]

# BACKTEST RESULTS
st.subheader("📊 BACKTEST RESULTS")
col1, col2, col3, col4 = st.columns(4)

total_return = ((final_capital / capital) - 1) * 100
col1.metric("Total Return", f"{total_return:+.1f}%")
col2.metric("Final Capital", f"${final_capital:,.0f}")
col3.metric("Total Trades", len(trades))

if trades:
    trades_df = pd.DataFrame(trades)
    completed_trades = trades_df[trades_df['pnl_pct'].notna()]
    
    if len(completed_trades) > 0:
        wins = completed_trades[completed_trades['pnl_pct'] > 0]
        losses = completed_trades[completed_trades['pnl_pct'] <= 0]
        win_rate = (len(wins) / len(completed_trades)) * 100 if len(completed_trades) > 0 else 0
        
        col4.metric("Win Rate", f"{win_rate:.1f}%")
        
        if len(wins) > 0 and len(losses) > 0:
            profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum())
            st.metric("Profit Factor", f"{profit_factor:.2f}")

# 4-PANEL CHART
fig, axes = plt.subplots(4, 1, figsize=(16, 12), height_ratios=[3, 1, 1, 1.5])
fig.patch.set_facecolor('white')

# 1. Price Chart
ax1 = axes[0]
ax1.plot(df_bt.index, df_bt['Close'], color='#2C3E50', linewidth=1.5, label='Close', alpha=0.8)
ax1.plot(df_bt.index, df_bt['EMA_FAST'], color='#3498DB', linewidth=1.5, label=f'EMA{ema_fast}', alpha=0.7)
ax1.plot(df_bt.index, df_bt['EMA_SLOW'], color='#E74C3C', linewidth=1.5, label=f'EMA{ema_slow}', alpha=0.7)

# Plot entries
if trades:
    trades_df = pd.DataFrame(trades)
    ax1.scatter(trades_df['entry_date'], trades_df['entry_price'], 
                color='limegreen', marker='^', s=120, alpha=0.9, 
                label='Entry', zorder=5, edgecolors='white', linewidths=1)
    
    # Plot exits
    exits = trades_df[trades_df['exit_date'].notna()]
    if len(exits) > 0:
        winners = exits[exits['pnl_pct'] > 0]
        losers = exits[exits['pnl_pct'] <= 0]
        
        if len(winners) > 0:
            ax1.scatter(winners['exit_date'], winners['exit_price'],
                       color='#27AE60', marker='v', s=120, alpha=0.9,
                       label='Exit (Win)', zorder=5, edgecolors='white', linewidths=1)
        if len(losers) > 0:
            ax1.scatter(losers['exit_date'], losers['exit_price'],
                       color='red', marker='v', s=120, alpha=0.9,
                       label='Exit (Loss)', zorder=5, edgecolors='white', linewidths=1)

ax1.set_title(f'{ticker} - Return: {total_return:+.1f}%', fontsize=16, fontweight='bold', pad=15)
ax1.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.2, linestyle='--')
ax1.set_facecolor('#F8F9FA')

# 2. RSI
ax2 = axes[1]
ax2.plot(df_bt.index, df_bt['RSI'], color='#9B59B6', linewidth=1.5, label=f'RSI({rsi_len})')
ax2.plot(df_bt.index, df_bt['RSI_EMA'], color='#F39C12', linewidth=2, label=f'RSI_EMA({rsi_ema_len})')
ax2.axhline(y=70, color='#E74C3C', linestyle='--', alpha=0.5, linewidth=1)
ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.3, linewidth=1)
ax2.axhline(y=30, color='#27AE60', linestyle='--', alpha=0.5, linewidth=1)
ax2.set_ylabel('RSI', fontsize=11, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.2)
ax2.set_ylim(0, 100)
ax2.set_facecolor('#F8F9FA')

# 3. ADX
ax3 = axes[2]
ax3.plot(df_bt.index, df_bt['ADX'], color='#E67E22', linewidth=1.5, label='ADX')
ax3.axhline(y=min_adx, color='#E74C3C', linestyle='--', alpha=0.5, linewidth=1, label=f'Min={min_adx}')
ax3.set_ylabel('ADX', fontsize=11, fontweight='bold')
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.2)
ax3.set_facecolor('#F8F9FA')

# 4. Equity Curve
ax4 = axes[3]
ax4.plot(equity_series.index, equity_series, color='#27AE60', linewidth=2.5, label='Portfolio Value')
ax4.fill_between(equity_series.index, capital, equity_series, 
                 alpha=0.3, color='#27AE60' if final_capital > capital else '#E74C3C')
ax4.axhline(y=capital, color='gray', linestyle='--', alpha=0.5, linewidth=1, label=f'Start ${capital:,.0f}')
ax4.set_ylabel('Equity ($)', fontsize=11, fontweight='bold')
ax4.set_xlabel('Date', fontsize=12, fontweight='bold')
ax4.legend(loc='upper left', fontsize=9)
ax4.grid(True, alpha=0.2)
ax4.set_facecolor('#F8F9FA')

plt.tight_layout()
st.pyplot(fig)

# LIVE SIGNAL STATUS
st.subheader("🔍 LIVE SIGNAL DEBUGGER")
col1, col2, col3, col4 = st.columns(4)

trend_ok = latest['EMA_FAST'] > latest['EMA_SLOW']
rsi_ok = latest['RSI'] > latest['RSI_EMA']
adx_ok = latest['ADX'] > min_adx
live_signal = trend_ok and rsi_ok and adx_ok

with col1:
    st.metric("Trend", "✅ UP" if trend_ok else "❌ DOWN", 
              f"EMA{ema_fast}>{ema_slow}" if trend_ok else "")
with col2:
    st.metric("RSI vs EMA", "✅" if rsi_ok else "❌", 
              f"{latest['RSI']:.1f} > {latest['RSI_EMA']:.1f}" if rsi_ok else f"{latest['RSI']:.1f} < {latest['RSI_EMA']:.1f}")
with col3:
    st.metric("ADX", f"{latest['ADX']:.1f}", 
              "✅ Strong" if adx_ok else "❌ Weak")
with col4:
    st.metric("Signal", "🟢 LIVE" if live_signal else "🔴 WAIT")

# LIVE TRADE PLAN
if live_signal:
    st.success("🎯 **LIVE SIGNAL - TRADE NOW!**")
    st.balloons()
    
    entry_price = latest['Close']
    shares = int((capital * 0.95) / entry_price)
    stop_price = entry_price * (1 - stop_loss_pct)
    
    st.subheader("📋 EXECUTE IMMEDIATELY")
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**BUY {shares:,} shares @ ${entry_price:.2f}**")
        st.info(f"Total: ${shares * entry_price:,.0f}")
    with col2:
        st.warning(f"**STOP LOSS: ${stop_price:.2f}**")
        st.error(f"Risk: ${shares * (entry_price - stop_price):,.0f}")

# TRADE HISTORY
with st.expander("📋 Detailed Trade History"):
    if trades:
        trades_display = pd.DataFrame(trades)
        if 'pnl_pct' in trades_display.columns:
            trades_display['pnl_pct'] = trades_display['pnl_pct'].round(2)
        st.dataframe(trades_display, use_container_width=True)
    else:
        st.info("No trades yet - waiting for signal")

st.caption(f"✅ Live backtest | Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
