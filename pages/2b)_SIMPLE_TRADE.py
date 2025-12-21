import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Simple Trade")
st.title("🚀 Simple RSI + SMA Strategy")

col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Ticker", value="COIN")
    capital = st.number_input("Capital ($)", value=10000, min_value=1000)
with col2:
    period = st.selectbox("Backtest Period", ["2y", "1y", "6mo"], index=0)

col1, col2, col3 = st.columns(3)
sma_fast_len = col1.number_input("SMA Fast", value=12, min_value=5, max_value=30)
rsi_len = col2.number_input("RSI Length", value=14, min_value=10, max_value=21)
rsi_ema_len = col3.number_input("RSI EMA Length", value=20, min_value=5, max_value=30)

col4, col5 = st.columns(2)
stop_loss_pct = col4.number_input("Stop Loss %", value=2.0, min_value=1.0, max_value=10.0)/100

@st.cache_data(ttl=300)
def get_data(ticker, period="2y"):
    try:
        df = yf.download(ticker, period=period, progress=False, threads=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        return df.dropna()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs)).fillna(50)

def run_simple_backtest(df, initial_capital, sma_fast_len, rsi_len, rsi_ema_len, stop_loss_pct):
    df_bt = df.copy()
    df_bt['SMA_FAST'] = df_bt['Close'].rolling(sma_fast_len, min_periods=1).mean()
    df_bt['RSI'] = RSI(df_bt['Close'], rsi_len)
    df_bt['RSI_EMA'] = df_bt['RSI'].ewm(span=rsi_ema_len, adjust=False).mean()
    df_bt = df_bt.dropna()
    
    cash = float(initial_capital)
    position_shares = 0.0
    entry_price = 0.0
    stop_price = 0.0
    equity_curve = []
    trades = []
    can_enter = True
    
    for i in range(len(df_bt)):
        row = df_bt.iloc[i]
        curr_close = float(row['Close'])
        curr_low = float(row['Low'])
        
        entry_condition = (row['RSI'] > row['RSI_EMA']) and (curr_close > row['SMA_FAST'])
        
        if position_shares == 0:
            if can_enter and entry_condition:
                position_shares = (cash * 0.95) / curr_close
                entry_cost = position_shares * curr_close
                cash -= entry_cost
                entry_price = curr_close
                stop_price = entry_price * (1 - stop_loss_pct)
                can_enter = False
                
                trades.append({
                    'entry_date': df_bt.index[i], 'entry_price': entry_price,
                    'exit_date': None, 'exit_price': None, 'pnl': 0, 'pnl_pct': 0
                })
        
        elif position_shares > 0:
            exit_triggered = False
            
            if curr_low <= stop_price:
                exit_price = stop_price
                exit_triggered = True
            elif curr_close < row['SMA_FAST']:
                exit_price = curr_close
                exit_triggered = True
            
            if exit_triggered:
                exit_value = position_shares * exit_price
                pnl = exit_value - (position_shares * entry_price)
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                cash += exit_value
                
                trades[-1]['exit_date'] = df_bt.index[i]
                trades[-1]['exit_price'] = exit_price
                trades[-1]['pnl'] = pnl
                trades[-1]['pnl_pct'] = pnl_pct
                
                position_shares = 0.0
                can_enter = True
        
        pos_value = position_shares * curr_close
        equity_curve.append(cash + pos_value)
    
    if position_shares > 0:
        final_price = float(df_bt['Close'].iloc[-1])
        final_value = position_shares * final_price
        cash += final_value
        pnl_pct = ((final_price - entry_price) / entry_price) * 100
        trades[-1]['exit_date'] = df_bt.index[-1]
        trades[-1]['exit_price'] = final_price
        trades[-1]['pnl_pct'] = pnl_pct
    
    equity_series = pd.Series(equity_curve, index=df_bt.index)
    final_capital = cash
    return equity_series, final_capital, trades, df_bt

df_raw = get_data(ticker, period)
if df_raw is None or df_raw.empty:
    st.error("❌ Failed to load data")
    st.stop()

equity_series, final_capital, trades, df_bt = run_simple_backtest(
    df_raw, capital, sma_fast_len, rsi_len, rsi_ema_len, stop_loss_pct
)

latest = df_bt.iloc[-1]
total_return = ((final_capital / capital) - 1) * 100

st.subheader("📊 SIMPLE STRATEGY RESULTS")
col1, col2, col3 = st.columns(3)
col1.metric("Total Return", f"{total_return:+.1f}%")
col2.metric("Final Capital", f"${final_capital:,.0f}")
col3.metric("Total Trades", len(trades))

trades_df = pd.DataFrame(trades)
completed = trades_df[trades_df['exit_date'].notna()]
if len(completed) > 0:
    win_rate = (completed['pnl'] > 0).mean() * 100
    st.metric("Win Rate", f"{win_rate:.1f}%")

fig, axes = plt.subplots(3, 1, figsize=(16, 10), height_ratios=[3, 1, 1.5])
fig.patch.set_facecolor('white')

ax1 = axes[0]
ax1.plot(df_bt.index, df_bt['Close'], color='#2C3E50', linewidth=1.5, label='Close', alpha=0.8)
ax1.plot(df_bt.index, df_bt['SMA_FAST'], color='#3498DB', linewidth=2, label=f'SMA{sma_fast_len}')

entry_signals = df_bt[(df_bt['RSI'] > df_bt['RSI_EMA']) & (df_bt['Close'] > df_bt['SMA_FAST'])]
ax1.scatter(entry_signals.index, entry_signals['Close'], color='limegreen', marker='^', s=50, 
           label=f'Entries ({len(entry_signals)})', zorder=5)

if len(trades) > 0:
    trades_plot = pd.DataFrame(trades)
    ax1.scatter(trades_plot['entry_date'], trades_plot['entry_price'], 
               color='green', marker='o', s=100, label=f'Trades ({len(trades)})', zorder=6)
    
    exits = trades_plot[trades_plot['exit_date'].notna()]
    if len(exits) > 0:
        winners = exits[exits['pnl'] > 0]
        losers = exits[exits['pnl'] <= 0]
        if len(winners) > 0:
            ax1.scatter(winners['exit_date'], winners['exit_price'], color='green', marker='v', s=100, zorder=6)
        if len(losers) > 0:
            ax1.scatter(losers['exit_date'], losers['exit_price'], color='red', marker='v', s=100, zorder=6)

ax1.set_title(f'{ticker} - Simple RSI+SMA | Return: {total_return:+.1f}% | {len(trades)} Trades', 
              fontsize=16, fontweight='bold')
ax1.legend(); ax1.grid(True, alpha=0.2)

ax2 = axes[1]
ax2.plot(df_bt.index, df_bt['RSI'], color='#9B59B6', linewidth=1.5, label=f'RSI({rsi_len})')
ax2.plot(df_bt.index, df_bt['RSI_EMA'], color='#F39C12', linewidth=2, label=f'RSI_EMA({rsi_ema_len})')
ax2.axhline(50, color='gray', ls='--', alpha=0.5)
ax2.set_ylabel('RSI'); ax2.legend(); ax2.grid(True, alpha=0.2)
ax2.set_ylim(0, 100)

ax3 = axes[2]
ax3.plot(equity_series.index, equity_series, color='#27AE60', linewidth=2.5, label='Portfolio')
ax3.axhline(capital, color='gray', ls='--', alpha=0.5)
ax3.set_ylabel('Equity ($)'); ax3.legend(); ax3.grid(True, alpha=0.2)

plt.tight_layout()
st.pyplot(fig)

st.subheader("🔍 LIVE STATUS")
col1, col2, col3 = st.columns(3)
live_entry = (latest['RSI'] > latest['RSI_EMA']) and (latest['Close'] > latest['SMA_FAST'])
col1.metric("RSI > RSI_EMA", "✅" if latest['RSI'] > latest['RSI_EMA'] else "❌", f"{latest['RSI']:.1f}")
col2.metric("Close > SMA", "✅" if latest['Close'] > latest['SMA_FAST'] else "❌", f"{latest['SMA_FAST']:.1f}")
col3.metric("Signal", "🟢 ENTRY" if live_entry else "🔴 WAIT")

if live_entry:
    st.success("🎯 LIVE ENTRY SIGNAL!")
    st.balloons()

with st.expander("📋 Trade History"):
    if len(trades) > 0:
        st.dataframe(pd.DataFrame(trades), use_container_width=True)
