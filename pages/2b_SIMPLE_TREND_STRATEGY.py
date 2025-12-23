import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Simple Momentum Strategy")
st.title("🚀 Simple Momentum Strategy") 

with st.expander("📖 Real-Time Trade Logic", expanded=False):
    st.markdown("""
    ## 🎯 **ENTRY: User-Tunable Signals**
    
    **1. TREND STRENGTH (ADX > [user input])**
    **2. MOMENTUM (RSI > [user input], <80 overheat filter)**
    **3. STRUCTURE (Close > EMA[user input])**
    
    ## 🚪 **EXIT: Wide Protection**
    
    **STOP LOSS ([user input]x ATR)**
    **EMA break backup**
    
    **RSI Sweet Spot: 50-80 (avoids overheated tops!)**
    """)

# INPUTS
col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.text_input("Ticker", value="COIN")
with col2:
    period = st.selectbox("Backtest Period", ["5y", "3y", "2y", "1y", "6mo"], index=2)
with col3:
    capital = st.number_input("Capital ($)", value=1000, min_value=1000)   

col1, col2, col3 = st.columns(3)
sma_fast_len = col1.number_input("EMA Fast", value=20, min_value=5, max_value=30)
sma_slow_len = col2.number_input("EMA SLOW", value=50, min_value=5, max_value=50)
rsi_len = col3.number_input("RSI Length", value=14, min_value=10, max_value=21)

# SIGNAL PARAMETERS
st.subheader("🎛️ Signal Parameters")
col1, col2, col3, col4 = st.columns(4)
adx_threshold = col1.number_input("ADX Threshold", value=15.0, min_value=1.0, max_value=40.0, step=1.0)
rsi_threshold = col2.number_input("RSI Threshold", value=50.0, min_value=40.0, max_value=70.0, step=1.0)
atr_mult_sl = col3.number_input("ATR SL Mult", value=1.5, min_value=1.0, max_value=10.0, step=0.5)
rsi_overheat = col4.number_input("RSI Overheated", value=75.0, min_value=70.0, max_value=100.0, step=0.5)

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

def ADX(df, period=14):
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    close = df["Close"].squeeze()
    tr1, tr2, tr3 = high-low, abs(high-close.shift()), abs(low-close.shift())
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
    return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0), atr.fillna(0)

def calculate_3lb_close(df, line_count=3):
    lb_close = df['Close'].copy()
    for i in range(line_count, len(df)):
        recent_high = df['High'].iloc[i-line_count:i].max()
        recent_low = df['Low'].iloc[i-line_count:i].min()
        curr_close = df['Close'].iloc[i]
        if curr_close > recent_high:
            lb_close.iloc[i] = recent_high 
        elif curr_close < recent_low:
            lb_close.iloc[i] = recent_low
        else:
            lb_close.iloc[i] = lb_close.iloc[i-1]
    return lb_close.fillna(method='ffill')

def run_backtest(df, initial_capital, sma_fast_len, sma_slow_len, rsi_len, atr_mult_sl, adx_threshold, rsi_threshold, rsi_overheat):
    df_bt = df.copy()
    df_bt['EMA_FAST'] = df_bt['Close'].ewm(span=sma_fast_len, adjust=False).mean()
    df_bt['EMA_SLOW'] = df_bt['Close'].ewm(span=sma_slow_len, adjust=False).mean()
    df_bt['3LB_Close'] = calculate_3lb_close(df_bt, line_count=3)
    df_bt['RSI_raw'] = RSI(df_bt['3LB_Close'], rsi_len)
    df_bt['RSI'] = df_bt['RSI_raw'].ewm(span=7, adjust=False).mean()
    df_bt['RSI_EMA'] = df_bt['RSI'].ewm(span=14, adjust=False).mean()
    df_bt['ADX'], df_bt['DI+'], df_bt['DI-'], df_bt['ATR'] = ADX(df_bt, 14)
    df_bt['ADX_ROC'] = df_bt['ADX'].pct_change(periods=5).fillna(0)
    df_bt['FAST_EMA_ROC'] = df_bt['EMA_FAST'].pct_change(20).fillna(0)
    df_bt['RSI_EMA_ROC'] = df_bt['RSI'].pct_change(20).fillna(0)
    df_bt = df_bt.dropna()
    
    cash = float(initial_capital)
    position_shares = 0.0
    entry_price = 0.0
    stop_price = 0.0
    equity_curve = []
    trades = []
    
    for i in range(len(df_bt)):
        row = df_bt.iloc[i]
        curr_close = float(row['Close'])
        curr_low = float(row['Low'])
        
        # CONSISTENT ENTRY CONDITION (same everywhere)
        entry_condition = (
            (row['ADX'] >= adx_threshold) &
            (row['RSI'] >= rsi_threshold) &
            (row['RSI'] <= rsi_overheat) &
            (curr_close >= row['EMA_FAST'])
        )

        if position_shares == 0 and entry_condition:
            position_shares = (cash * 0.95) / curr_close
            entry_cost = position_shares * curr_close
            cash -= entry_cost
            entry_price = curr_close
            atr_value = float(row['ATR'])
            stop_price = entry_price - (atr_mult_sl * atr_value)
            
            trades.append({
                'entry_date': df_bt.index[i], 'entry_price': entry_price,
                'exit_date': None, 'exit_price': None, 'pnl': 0, 'pnl_pct': 0
            })
        
        elif position_shares > 0:
            exit_triggered = False
            
            if curr_low <= stop_price:
                exit_price = stop_price
                exit_triggered = True
            elif curr_close < row['EMA_FAST']:
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

# RUN BACKTEST
df_raw = get_data(ticker, period)
if df_raw is None or df_raw.empty:
    st.error("❌ Failed to load data")
    st.stop()

equity_series, final_capital, trades, df_bt = run_backtest(
    df_raw, capital, sma_fast_len, sma_slow_len, rsi_len, atr_mult_sl, adx_threshold, rsi_threshold, rsi_overheat
)

latest = df_bt.iloc[-1]
total_return = ((final_capital / capital) - 1) * 100

# SINGLE SOURCE OF TRUTH: entry_signals DataFrame
entry_signals = df_bt[
    (df_bt['ADX'] >= adx_threshold) &
    (df_bt['RSI'] >= rsi_threshold) &
    (df_bt['RSI'] <= rsi_overheat) &
    (df_bt['Close'] >= df_bt['EMA_FAST'])
].copy()

# RESULTS
st.subheader(f"📊 {ticker}: RESULTS (ADX>{adx_threshold} RSI>{rsi_threshold} <{rsi_overheat})")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Return", f"{total_return:+.1f}%")
col2.metric("Final Capital", f"${final_capital:,.0f}")
col3.metric("Total Trades", len(trades))
col4.metric("Entry Signals", len(entry_signals))

trades_df = pd.DataFrame(trades)
completed = trades_df[trades_df['exit_date'].notna()]

r21, r22, r23 = st.columns(3)
if len(completed) > 0:
    avg_gain = completed[completed['pnl_pct'] > 0]['pnl_pct'].mean()
    avg_loss = completed[completed['pnl_pct'] < 0]['pnl_pct'].mean()
    r21.metric("Avg Win/Loss%", f"{avg_gain:.1f} / {avg_loss:.1f}")

    win_rate = (completed['pnl'] > 0).mean() * 100
    r22.metric("Win Rate", f"{win_rate:.1f}%")
    
    gross_profit = completed[completed['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(completed[completed['pnl'] <= 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    r23.metric("Profit Factor", f"{profit_factor:.2f}")

# ALL TRADES TABLE
with st.expander("📋 All Trades", expanded=False):
    if len(trades) > 0:
        trades_df_local = pd.DataFrame(trades)
        trades_df_local['entry_date'] = trades_df_local['entry_date'].dt.strftime('%Y-%m-%d')
        trades_df_local['exit_date'] = trades_df_local['exit_date'].fillna('OPEN')
        trades_df_local['exit_date'] = trades_df_local['exit_date'].astype(str)
        trades_df_local['status'] = np.where(trades_df_local['exit_date'] == 'OPEN', 'OPEN', 'CLOSED')
        
        display_cols = ['entry_date', 'entry_price', 'exit_date', 'exit_price', 'pnl_pct', 'status']
        display_df = trades_df_local[display_cols].round(2)
        display_df.columns = ['Entry', 'Entry $', 'Exit', 'Exit $', 'PnL%', 'Status']
        
        st.dataframe(display_df, use_container_width=True, height=400, hide_index=True)
        st.caption(f"📈 Showing {len(display_df)} total trades")
    else:
        st.info("📊 No trades executed yet")

# PLOTTING
fig, axes = plt.subplots(4, 1, figsize=(16, 12), height_ratios=[3, 1, 1, 1.5])
fig.patch.set_facecolor('white')

ax1 = axes[0]
x = df_bt.index
y = df_bt['3LB_Close'].values

colors = []
for i in range(len(df_bt)):
    f = df_bt['FAST_EMA_ROC'].iloc[i]
    r = df_bt['RSI_EMA_ROC'].iloc[i]
    if f < 0 and r > 0:
        colors.append('green')
    elif f > 0 and r < 0:
        colors.append('red')
    else:
        colors.append('gray')

for i in range(len(x)-1):
    ax1.plot(x[i:i+2], y[i:i+2], color=colors[i], linewidth=2, alpha=0.5)

ax1.plot(df_bt.index, df_bt['Close'], color='gray', linewidth=1, label='Close', alpha=0.5)
ax1.plot(df_bt.index, df_bt['EMA_FAST'], color='orange', linewidth=2, label=f'EMA{sma_fast_len}', alpha=0.5)
ax1.plot(df_bt.index, df_bt['EMA_SLOW'], color='red', linewidth=2, label=f'EMA{sma_slow_len}', alpha=0.5)

# ENTRY_SIGNALS DataFrame plotting
if len(entry_signals) > 0:
    ax1.scatter(entry_signals.index, entry_signals['Close'], color='orange', marker='d', s=15, 
               label=f'Signals ({len(entry_signals)})', alpha=0.6, zorder=5, 
               edgecolors='black', linewidths=1.0)

if len(trades) > 0:
    trades_plot = pd.DataFrame(trades)
    ax1.scatter(trades_plot['entry_date'], trades_plot['entry_price'], 
               color='blue', marker='^', s=25, label=f'Entries ({len(trades)})', alpha=0.7, zorder=4, 
               edgecolors='darkblue', linewidths=1.2)
    
    exits = trades_plot[trades_plot['exit_date'].notna()]
    if len(exits) > 0:
        winners = exits[exits['pnl'] > 0]
        losers = exits[exits['pnl'] <= 0]
        
        if len(winners) > 0:
            ax1.scatter(winners['exit_date'], winners['exit_price'], color='limegreen', marker='o', s=25, 
                       label=f'Winners ({len(winners)})', alpha=0.8, zorder=3, edgecolors='black', linewidths=1.2)
        if len(losers) > 0:
            ax1.scatter(losers['exit_date'], losers['exit_price'], color='darkred', marker='x', s=25, 
                       label=f'Losses ({len(losers)})', alpha=0.8, zorder=3, edgecolors='black', linewidths=1.2)

ax1.set_title(f'{ticker} | Return: {total_return:+.1f}% | Signals: {len(entry_signals)} | Trades: {len(trades)}', 
              fontsize=16, fontweight='bold', pad=15)
ax1.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=8, ncol=2)
ax1.grid(True, alpha=0.2, linestyle='--')
ax1.set_facecolor('#F8F9FA')
ax1.yaxis.tick_right()     
ax1.yaxis.set_label_position("right")

# RSI
ax2 = axes[1]
ax2.plot(df_bt.index, df_bt['RSI'], color='#9B59B6', linewidth=1.1, label=f'RSI({rsi_len})', alpha=0.5)
ax2.plot(df_bt.index, df_bt['RSI_EMA'], color='red', linewidth=2, label='RSI_EMA(14)', alpha=0.5)
ax2.axhspan(rsi_threshold, rsi_overheat, alpha=0.2, color='green', label=f'Profit-taking {rsi_threshold}-{rsi_overheat}')
ax2.axhspan(0, 10, alpha=0.2, color='red', label=f'Extremely Bearish (0-10)')
ax2.axhline(y=rsi_threshold, color='orange', linestyle='--', alpha=0.7, linewidth=1)
ax2.axhline(y=rsi_overheat, color='#E74C3C', linestyle=':', alpha=0.7, linewidth=1.5, label='Overheat')
ax2.axhline(y=25, color='#27AE60', linestyle='--', alpha=0.3, linewidth=1)
ax2.set_ylabel('RSI', fontsize=11, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.2)
ax2.set_ylim(0, 100)
ax2.set_facecolor('#F8F9FA')
ax2.yaxis.tick_right()       
ax2.yaxis.set_label_position("right")

# ADX
ax3 = axes[2]
ax3.plot(df_bt.index, df_bt['DI+'], color='green', linewidth=1, label='DI+', alpha=0.5)
ax3.plot(df_bt.index, df_bt['DI-'], color='red', linewidth=1, label='DI-', alpha=0.5)
ax3.plot(df_bt.index, df_bt['ADX'], color='gray', linewidth=2, label='ADX', alpha=0.5)
ax3.axhline(y=adx_threshold, color='orange', linestyle='--', alpha=0.7, linewidth=1, label=f'Signal: {adx_threshold}')
ax3.axhline(y=25, color='gray', linestyle='--', alpha=0.3, linewidth=1)
ax3.set_ylabel('ADX/DI', fontsize=11, fontweight='bold')
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.2)
ax3.set_facecolor('#F8F9FA')
ax3.yaxis.tick_right()       
ax3.yaxis.set_label_position("right")

# EQUITY
ax4 = axes[3]
ax4.plot(equity_series.index, equity_series, color='#27AE60', linewidth=2.5, label='Portfolio Value', alpha=0.5)
ax4.fill_between(equity_series.index, capital, equity_series, 
                 alpha=0.2, color='#27AE60' if final_capital > capital else '#E74C3C')
ax4.axhline(y=capital, color='gray', linestyle='--', alpha=0.5, linewidth=1, label=f'Start ${capital:,.0f}')
ax4.set_ylabel('Equity ($)', fontsize=11, fontweight='bold')
ax4.set_xlabel('Date', fontsize=12, fontweight='bold')
ax4.legend(loc='upper left', fontsize=9)
ax4.grid(True, alpha=0.2)
ax4.set_facecolor('#F8F9FA')
ax4.yaxis.tick_right()    
ax4.yaxis.set_label_position("right")

plt.tight_layout()
st.pyplot(fig)

# LIVE STATUS - SAME CONDITIONS
st.subheader("🔍 LIVE STATUS")
col1, col2, col3, col4 = st.columns(4)
live_entry = (
    (latest['ADX'] >= adx_threshold) and 
    (latest['RSI'] >= rsi_threshold) and
    (latest['RSI'] <= rsi_overheat) and
    (latest['Close'] >= latest['EMA_FAST'])
)
col1.metric("ADX", "✅" if latest['ADX'] >= adx_threshold else "❌", f"{latest['ADX']:.1f}>{adx_threshold}")
col2.metric("RSI", f"{latest['RSI']:.1f}", f"✅ {rsi_threshold}-{rsi_overheat}" if rsi_threshold <= latest['RSI'] <= rsi_overheat else "❌")
col3.metric("Structure", "✅" if latest['Close'] >= latest['EMA_FAST'] else "❌", f"EMA{sma_fast_len}: {latest['EMA_FAST']:.1f}")
col4.metric("SIGNAL", "🟢 ENTRY" if live_entry else "🔴 WAIT")

if live_entry:
    st.success("🎯 ENTRY SIGNAL ACTIVE!")
    st.balloons()

st.caption(f"✅ ALL FIXED | RSI Sweet Spot {rsi_threshold}-{rsi_overheat} | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
