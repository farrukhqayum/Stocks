import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(layout="wide", page_title="Improved Momentum Strategy")
st.title("🚀 Improved ATR Momentum Strategy")

# Parameters (better defaults)
col1, col2, col3 = st.columns(3)
ticker = col1.text_input("Ticker", value="COIN")
period = col2.selectbox("Period", ["5y", "3y", "2y", "1y"], index=1)
capital = col3.number_input("Capital ($)", value=10000)

col1, col2, col3, col4 = st.columns(4)
atr_mult_sl = col1.number_input("ATR SL Mult", value=2.5, min_value=1.5, max_value=5.0, step=0.5)
atr_mult_tp = col2.number_input("ATR TP Mult", value=4.5, min_value=3.0, max_value=8.0, step=0.5)
ema_fast = col3.number_input("Fast EMA", value=21, min_value=10, max_value=50)
ema_slow = col4.number_input("Slow EMA", value=50, min_value=30, max_value=100)

@st.cache_data(ttl=300)
def get_data(ticker, period):
    df = yf.download(ticker, period=period, progress=False)
    df['Close'] = df['Adj Close']
    return df.dropna()

def calculate_indicators(df, ema_fast, ema_slow):
    df = df.copy()
    
    # Simple EMAs (no 3LB smoothing - too laggy)
    df['EMA_fast'] = df['Close'].ewm(span=ema_fast).mean()
    df['EMA_slow'] = df['Close'].ewm(span=ema_slow).mean()
    
    # Standard RSI (14) on Close
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).ewm(span=14).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_EMA'] = df['RSI'].ewm(span=14).mean()
    
    # ADX (simplified)
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift()), abs(low - close.shift())))
    df['ATR'] = tr.ewm(span=14).mean()
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    
    plus_di = 100 * (plus_dm.ewm(span=14).mean() / df['ATR'])
    minus_di = 100 * (minus_dm.ewm(span=14).mean() / df['ATR'])
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['ADX'] = dx.ewm(span=14).mean()
    
    df = df.dropna()
    return df

def backtest_strategy(df, capital, atr_mult_sl, atr_mult_tp):
    df_bt = calculate_indicators(df, ema_fast, ema_slow)
    
    cash = capital
    shares = 0
    entry_price = 0
    stop_price = 0
    target_price = 0
    equity = []
    trades = []
    
    for i in range(1, len(df_bt)):
        row = df_bt.iloc[i]
        prev_row = df_bt.iloc[i-1]
        curr_close = row['Close']
        curr_low = row['Low']
        
        # SIMPLIFIED ENTRY: 3 green lights only
        entry = (
            (curr_close > row['EMA_slow']) &           # Trend direction
            (row['ADX'] > 22) &                        # Trend strength  
            (row['RSI'] > row['RSI_EMA'])              # Momentum confirmation
        )
        
        # No position: check entry
        if shares == 0 and entry:
            shares = (cash * 0.98) / curr_close
            entry_price = curr_close
            atr_value = row['ATR']
            stop_price = entry_price - (atr_mult_sl * atr_value)
            target_price = entry_price + (atr_mult_tp * atr_value)
            
            trades.append({
                'entry_date': df_bt.index[i], 
                'entry_price': entry_price,
                'stop': stop_price,
                'target': target_price
            })
        
        # In position: check exits
        elif shares > 0:
            exit_triggered = False
            exit_price = curr_close
            exit_reason = ""
            
            # Stop loss (use low to be realistic)
            if curr_low <= stop_price:
                exit_price = stop_price
                exit_reason = "Stop Loss"
                exit_triggered = True
            # Take profit
            elif curr_close >= target_price:
                exit_price = target_price
                exit_reason = "Take Profit" 
                exit_triggered = True
            # Trailing stop alternative: exit on EMA break + ADX weakening
            elif (curr_close < row['EMA_fast']) and (row['ADX'] < 20):
                exit_price = curr_close
                exit_reason = "Trend Break"
                exit_triggered = True
            
            if exit_triggered:
                exit_value = shares * exit_price
                cash += exit_value
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                
                trades[-1].update({
                    'exit_date': df_bt.index[i],
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'exit_reason': exit_reason
                })
                shares = 0
        
        # Track equity
        pos_value = shares * curr_close if shares > 0 else 0
        equity.append(cash + pos_value)
    
    # Close final position
    if shares > 0:
        final_price = df_bt['Close'].iloc[-1]
        cash += shares * final_price
        pnl_pct = (final_price - entry_price) / entry_price * 100
        trades[-1].update({
            'exit_date': df_bt.index[-1],
            'exit_price': final_price,
            'pnl_pct': pnl_pct,
            'exit_reason': 'End'
        })
    
    equity_series = pd.Series(equity, index=df_bt.index)
    return equity_series, cash, trades, df_bt

# RUN BACKTEST
df = get_data(ticker, period)
if df is None or df.empty:
    st.stop()

equity, final_capital, trades, df_bt = backtest_strategy(
    df, capital, atr_mult_sl, atr_mult_tp
)

total_return = ((final_capital / capital) - 1) * 100
trades_df = pd.DataFrame(trades)
completed_trades = trades_df[trades_df['exit_date'].notna()]

# RESULTS
st.subheader(f"📊 {ticker} Results")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Return", f"{total_return:+.1f}%")
col2.metric("Final Capital", f"${final_capital:,.0f}")
col3.metric("Trades", len(completed_trades))
col4.metric("Win Rate", f"{(completed_trades['pnl_pct'] > 0).mean()*100:.0f}%" if len(completed_trades)>0 else "0%")

# PLOT (simplified but clear)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
fig.patch.set_facecolor('white')

# Price + signals
ax1.plot(df_bt.index, df_bt['Close'], label='Close', linewidth=1.5, color='black')
ax1.plot(df_bt.index, df_bt['EMA_fast'], label=f'EMA{ema_fast}', color='orange', alpha=0.7)
ax1.plot(df_bt.index, df_bt['EMA_slow'], label=f'EMA{ema_slow}', color='red', alpha=0.7)

# Plot trades
if len(trades) > 0:
    entries = trades_df[trades_df['exit_date'].isna() == True]
    exits = completed_trades
    ax1.scatter(entries['entry_date'], entries['entry_price'], color='green', marker='^', s=100, label='Entry', zorder=5)
    winners = exits[exits['pnl_pct'] > 0]
    losers = exits[exits['pnl_pct'] <= 0]
    ax1.scatter(winners['exit_date'], winners['exit_price'], color='green', marker='o', s=80, label='Win', zorder=4)
    ax1.scatter(losers['exit_date'], losers['exit_price'], color='red', marker='x', s=80, label='Loss', zorder=4)

ax1.set_title(f'{ticker} | {total_return:+.1f}% | Win Rate: {(completed_trades["pnl_pct"]>0).mean()*100:.0f}%', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# RSI
ax2.plot(df_bt.index, df_bt['RSI'], label='RSI', color='#9B59B6')
ax2.plot(df_bt.index, df_bt['RSI_EMA'], label='RSI EMA', color='red')
ax2.axhline(50, color='gray', linestyle='--', alpha=0.5)
ax2.set_ylabel('RSI')
ax2.legend()
ax2.grid(alpha=0.3)

# Equity curve
ax3.plot(equity.index, equity, color='green' if total_return > 0 else 'red', linewidth=2.5)
ax3.axhline(capital, color='gray', linestyle='--', alpha=0.5, label=f'Start ${capital:,.0f}')
ax3.fill_between(equity.index, capital, equity, alpha=0.3)
ax3.set_ylabel('Equity ($)')
ax3.legend()
ax3.grid(alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# LIVE STATUS
latest = df_bt.iloc[-1]
live_signal = (
    latest['Close'] > latest['EMA_slow'] and
    latest['ADX'] > 22 and
    latest['RSI'] > latest['RSI_EMA']
)

st.subheader("🔴 LIVE STATUS")
col1, col2, col3 = st.columns(3)
col1.metric("Trend", "✅" if latest['Close'] > latest['EMA_slow'] else "❌", f"EMA{ema_slow}: {latest['EMA_slow']:.2f}")
col2.metric("ADX", f"{latest['ADX']:.1f}", "✅" if latest['ADX'] > 22 else "❌") 
col3.metric("RSI Mom", "✅" if latest['RSI'] > latest['RSI_EMA'] else "❌", f"{latest['RSI']:.1f}")

st.metric("SIGNAL", "🟢 ENTER LONG" if live_signal else "🔴 WAIT", delta=None)

# Trade table
with st.expander("📋 Trade Log"):
    if len(completed_trades) > 0:
        st.dataframe(completed_trades[['entry_date', 'exit_date', 'entry_price', 'exit_price', 'pnl_pct', 'exit_reason']], use_container_width=True)
