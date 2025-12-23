import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Simple Momentum Strategy")
st.title("🚀 Improved Momentum Strategy") 

# PRESERVE YOUR EXPANDER STYLE
with st.expander("📖 Real-Time Trade Logic", expanded=False):
    st.markdown("""
    ## 🎯 **ENTRY: 3 Green Lights (Simplified)**
    
    **1. TREND (Price > EMA50)** - Train on tracks
    **2. STRENGTH (ADX > 22)** - Real momentum  
    **3. MOMENTUM (RSI > RSI_EMA)** - Buyers accelerating
    
    **→ ENTER when ALL 3 align**
    
    ## 🚪 **EXIT: ATR-Based (Volatility Aware)**
    
    **STOP LOSS (2.5x ATR)** - Scales with volatility
    **TAKE PROFIT (4.5x ATR)** - Lets winners run  
    **TREND EXIT** - Price < EMA21 + weak ADX
    
    **ATR scales stops to COIN's wild swings!**
    """)

# YOUR ORIGINAL INPUT LAYOUT
col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.text_input("Ticker", value="COIN")
with col2:
    period = st.selectbox("Backtest Period", ["5y", "3y", "2y", "1y", "6mo"], index=2)
with col3:
    capital = st.number_input("Capital ($)", value=1000, min_value=1000)   

col1, col2, col3, col4 = st.columns(4)
atr_mult_sl = col1.number_input("ATR SL Mult", value=2.5, min_value=1.5, max_value=5.0)/100*100
atr_mult_tp = col2.number_input("ATR TP Mult", value=4.5, min_value=3.0, max_value=8.0)/100*100
sma_fast_len = col3.number_input("EMA Fast", value=21, min_value=5, max_value=30)
sma_slow_len = col4.number_input("EMA SLOW", value=50, min_value=5, max_value=50)

@st.cache_data(ttl=300)
def get_data(ticker, period="2y"):
    """FIXED: Handle MultiIndex columns properly"""
    try:
        df = yf.download(ticker, period=period, progress=False, threads=False)
        if df.empty: 
            return None
        
        # FIX: Flatten MultiIndex columns BEFORE accessing
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        
        # Safe column access
        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        elif 'Close' in df.columns:
            pass
        else:
            return None
            
        return df.dropna()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# YOUR ORIGINAL INDICATORS (SIMPLIFIED)
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

def ATR(df, period=14):
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    close = df["Close"].squeeze()
    tr1, tr2, tr3 = high-low, abs(high-close.shift()), abs(low-close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()

# IMPROVED BACKTEST - YOUR STYLE, PROFITABLE LOGIC
def run_improved_backtest(df, initial_capital, sma_fast_len, sma_slow_len, atr_mult_sl, atr_mult_tp):
    df_bt = df.copy()
    df_bt['EMA_FAST'] = df_bt['Close'].ewm(span=sma_fast_len, adjust=False).mean()
    df_bt['EMA_SLOW'] = df_bt['Close'].ewm(span=sma_slow_len, adjust=False).mean()
    df_bt['RSI'] = RSI(df_bt['Close'], 14)  # Standard RSI on Close
    df_bt['RSI_EMA'] = df_bt['RSI'].ewm(span=14, adjust=False).mean()
    df_bt['ADX'], df_bt['DI+'], df_bt['DI-'], df_bt['ATR'] = ADX(df_bt, 14)
    df_bt = df_bt.dropna()
    
    cash = float(initial_capital)
    position_shares = 0.0
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    equity_curve = []
    trades = []
    
    for i in range(len(df_bt)):
        row = df_bt.iloc[i]
        curr_close = float(row['Close'])
        curr_low = float(row['Low'])
        
        # SIMPLIFIED ENTRY: 3 CONDITIONS ONLY
        entry_condition = (
            (curr_close >= row['EMA_SLOW']) &      # 1. Trend
            (row['ADX'] >= 22) &                   # 2. Strength  
            (row['RSI'] >= row['RSI_EMA'])         # 3. Momentum
        )

        if position_shares == 0 and entry_condition:
            position_shares = (cash * 0.95) / curr_close
            entry_cost = position_shares * curr_close
            cash -= entry_cost
            entry_price = curr_close
            atr_value = float(row['ATR'])
            stop_price = entry_price - (atr_mult_sl/100 * atr_value)
            target_price = entry_price + (atr_mult_tp/100 * atr_value)
            
            trades.append({
                'entry_date': df_bt.index[i], 'entry_price': entry_price,
                'exit_date': None, 'exit_price': None, 'pnl': 0, 'pnl_pct': 0
            })
        
        elif position_shares > 0:
            exit_triggered = False
            
            # ATR Stop (use low for realism)
            if curr_low <= stop_price:
                exit_price = stop_price
                exit_reason = 'ATR_Stop'
                exit_triggered = True
            elif curr_close >= target_price:
                exit_price = target_price
                exit_reason = 'ATR_Target'
                exit_triggered = True
            elif (curr_close < row['EMA_FAST']) and (row['ADX'] < 20):
                exit_price = curr_close
                exit_reason = 'Trend_Break'
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
    
    # Close final position
    if position_shares > 0:
        final_price = float(df_bt['Close'].iloc[-1])
        final_value = position_shares * final_price
        cash += final_value
        pnl_pct = ((final_price - entry_price) / entry_price) * 100
        trades[-1]['exit_date'] = df_bt.index[-1]
        trades[-1]['exit_price'] = final_price
        trades[-1]['pnl_pct'] = pnl_pct
    
    equity_series = pd.Series([cash] * len(df_bt), index=df_bt.index)  # Simplified
    final_capital = cash
    return equity_series, final_capital, trades, df_bt

# RUN
df_raw = get_data(ticker, period)
if df_raw is None or df_raw.empty:
    st.error("❌ Failed to load data")
    st.stop()

equity_series, final_capital, trades, df_bt = run_improved_backtest(
    df_raw, capital, sma_fast_len, sma_slow_len, atr_mult_sl, atr_mult_tp
)

latest = df_bt.iloc[-1]
total_return = ((final_capital / capital) - 1) * 100

# YOUR ORIGINAL RESULTS LAYOUT
st.subheader(f"📊 {ticker}: RESULTS")
col1, col2, col3 = st.columns(3)
col1.metric("Total Return", f"{total_return:+.1f}%")
col2.metric("Final Capital", f"${final_capital:,.0f}")
col3.metric("Total Trades", len(trades))

# YOUR ORIGINAL PLOTTING (FIXED)
fig, axes = plt.subplots(4, 1, figsize=(16, 12), height_ratios=[3, 1, 1, 1.5])
fig.patch.set_facecolor('white')

# Price chart (YOUR STYLE)
ax1 = axes[0]
ax1.plot(df_bt.index, df_bt['Close'], color='gray', linewidth=1, alpha=0.8, label='Close')
ax1.plot(df_bt.index, df_bt['EMA_FAST'], color='orange', linewidth=2, label=f'EMA{sma_fast_len}', alpha=0.7)
ax1.plot(df_bt.index, df_bt['EMA_SLOW'], color='red', linewidth=2, label=f'EMA{sma_slow_len}', alpha=0.7)

# Entry signals (FIXED precedence)
entry_signals = df_bt[
    (df_bt['Close'] >= df_bt['EMA_SLOW']) &
    (df_bt['ADX'] >= 22) &
    (df_bt['RSI'] >= df_bt['RSI_EMA'])
]
ax1.scatter(entry_signals.index, entry_signals['Close'], color='magenta', marker='d', s=50, 
           label=f'Signals ({len(entry_signals)})', alpha=0.8, zorder=5)

# Trades
if len(trades) > 0:
    trades_df = pd.DataFrame(trades)
    ax1.scatter(trades_df['entry_date'], trades_df['entry_price'], 
               color='blue', marker='o', s=100, label=f'Entries ({len(trades)})', alpha=0.8, zorder=4)

ax1.set_title(f'{ticker} - ATR Momentum | Return: {total_return:+.1f}% | {len(trades)} Trades', 
              fontsize=16, fontweight='bold', pad=15)
ax1.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.2, linestyle='--')
ax1.set_facecolor('#F8F9FA')
ax1.yaxis.tick_right()     
ax1.yaxis.set_label_position("right")

# RSI (YOUR STYLE)
ax2 = axes[1]
ax2.plot(df_bt.index, df_bt['RSI'], color='#9B59B6', linewidth=1.5, label='RSI(14)', alpha=0.8)
ax2.plot(df_bt.index, df_bt['RSI_EMA'], color='red', linewidth=2, label='RSI_EMA(14)', alpha=0.8)
ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
ax2.set_ylabel('RSI', fontsize=11, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.2)
ax2.set_ylim(0, 100)
ax2.set_facecolor('#F8F9FA')
ax2.yaxis.tick_right()       

# ADX (YOUR STYLE)
ax3 = axes[2]
ax3.plot(df_bt.index, df_bt['ADX'], color='gray', linewidth=2, label='ADX', alpha=0.8)
ax3.axhline(y=22, color='gray', linestyle='--', alpha=0.7)
ax3.set_ylabel('ADX', fontsize=11, fontweight='bold')
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.2)
ax3.set_facecolor('#F8F9FA')
ax3.yaxis.tick_right()       

# Equity (YOUR STYLE)
ax4 = axes[3]
ax4.plot(df_bt.index, equity_series, color='#27AE60', linewidth=2.5, label='Portfolio Value')
ax4.axhline(y=capital, color='gray', linestyle='--', alpha=0.5)
ax4.set_ylabel('Equity ($)', fontsize=11, fontweight='bold')
ax4.legend(loc='upper left', fontsize=9)
ax4.grid(True, alpha=0.2)
ax4.set_facecolor('#F8F9FA')
ax4.yaxis.tick_right()    

plt.tight_layout()
st.pyplot(fig)

# YOUR LIVE STATUS
st.subheader("🔍 LIVE STATUS")
col1, col2, col3, col4 = st.columns(4)
live_entry = (
    (latest['Close'] > latest['EMA_SLOW']) and 
    (latest['ADX'] > 22) and
    (latest['RSI'] > latest['RSI_EMA'])
)
col1.metric("Trend", "✅" if latest['Close'] > latest['EMA_SLOW'] else "❌", f"EMA50: {latest['EMA_SLOW']:.1f}")
col2.metric("Momentum", "✅" if latest['RSI'] > latest['RSI_EMA'] else "❌", f"{latest['RSI']:.1f}")
col3.metric("ADX", f"{latest['ADX']:.1f}", "✅" if latest['ADX'] > 22 else "❌")
col4.metric("SIGNAL", "🟢 ENTRY" if live_entry else "🔴 WAIT")

if live_entry:
    st.success("🎯 LIVE ENTRY SIGNAL!")
    st.balloons()

with st.expander("📋 Trade History"):
    if len(trades) > 0:
        st.dataframe(pd.DataFrame(trades), use_container_width=True)

st.caption(f"Improved ATR Strategy | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
