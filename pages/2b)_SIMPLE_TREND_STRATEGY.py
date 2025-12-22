import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Simple Trade")
st.title("🚀 Simple Momentum Strategy")

<details>
<summary>a) How does this simple trade work?</summary>

### 1. Setup and intention
- Choose an asset (stock, ETF, crypto, FX pair, etc.) you want to trade.[web:1][web:7]  
- Form a directional view:
  - Bullish → plan to **buy** first and sell later (long).
  - Bearish → plan to **sell** first (via short/derivative) and buy back later (short).[web:3][web:7]  

---

### 2. Opening the trade (entry)
- Place an order with a broker or exchange specifying:
  - Side: buy or sell.
  - Quantity: number of units/contracts.
  - Order type: market (fill now at best price) or limit (only at your chosen price or better).[web:3][web:7]  
- Once filled, you have an **open position**:
  - Long: you own the asset and benefit from price going up.
  - Short: you owe the asset and benefit from price going down.[web:3][web:7]  

**Example:**  
- You buy 10 shares at 100 per share → position size = 1,000 notional.[web:3][web:7]  

---

### 3. While the trade is open
- Price moves due to supply and demand, news, and broader market flows.[web:1][web:7]  
- Your **unrealized P&L** changes continuously:
  - Long: P&L ≈ (current price − entry price) × quantity.
  - Short: P&L ≈ (entry price − current price) × quantity.[web:3][web:7]  
- You typically manage risk with:
  - Stop-loss: predefined exit if price moves against you.
  - Take-profit/target: predefined exit to lock in gains.[web:1][web:3]  

---

### 4. Closing the trade (exit)
- To **close**:
  - Close long → sell the same quantity you bought.
  - Close short → buy back the same quantity you sold.[web:3][web:7]  
- The trade’s **realized P&L** is locked in at exit.

**P&L formula (excluding fees, interest, etc.):**  
- Long trade:  
  - Profit/Loss = (Exit price − Entry price) × Quantity.[web:3][web:7]  
- Short trade:  
  - Profit/Loss = (Entry price − Exit price) × Quantity.[web:3][web:7]  

---

### 5. Simple numeric example (long)
- Entry: Buy 10 units at 100 → notional 1,000.[web:3][web:7]  
- Exit: Sell 10 units at 105.  
- Result: (105 − 100) × 10 = 50 profit (before fees).[web:3][web:7]  

If instead price fell to 95 and you sold:  
- (95 − 100) × 10 = −50 → a loss of 50.[web:3][web:7]  

---

### 6. Simple numeric example (short)
- Entry: Short 10 units at 100.  
- Exit at 90: (100 − 90) × 10 = 100 profit.[web:3][web:7]  
- Exit at 110: (100 − 110) × 10 = −100 loss.[web:3][web:7]  

---

### 7. Real-world frictions
In reality, a “simple” trade also has:

- Transaction costs: commissions, spreads, exchange fees.[web:3][web:7]  
- Financing/margin: interest or funding when using leverage or shorting.[web:3][web:5]  
- Slippage: fills at slightly worse prices than expected, especially in fast markets.[web:3][web:5]  

These reduce net profit or increase net loss compared with the idealized calculation.[web:3][web:7]  

</details>


col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Ticker", value="COIN")
    capital = st.number_input("Capital ($)", value=10000, min_value=1000)
with col2:
    period = st.selectbox("Backtest Period", ["5y", "3y", "2y", "1y", "6mo"], index=2)

col1, col2, col3 = st.columns(3)
sma_fast_len = col1.number_input("SMA Fast", value=20, min_value=5, max_value=30)
sma_slow_len = col1.number_input("SMA SLOW", value=50, min_value=5, max_value=50)
rsi_len = col2.number_input("RSI Length", value=14, min_value=10, max_value=21)
rsi_ema_len = col3.number_input("RSI EMA Length", value=14, min_value=9, max_value=50)

col4, col5 = st.columns(2)
stop_loss_pct = col4.number_input("Stop Loss %", value=2.0, min_value=1.0, max_value=100.0)/100
take_profit_pct = col5.number_input("Take Profit %", value=4.25, min_value=3.0, max_value=15.0)/100

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
    return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)

def ATR(df, period=14):
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    close = df["Close"].squeeze()
    tr1, tr2, tr3 = high-low, abs(high-close.shift()), abs(low-close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()

def run_simple_backtest(df, initial_capital, sma_fast_len, rsi_len, rsi_ema_len, stop_loss_pct, take_profit_pct):
    df_bt = df.copy()
    df_bt['SMA_FAST'] = df_bt['Close'].rolling(sma_fast_len, min_periods=1).mean()
    df_bt['SMA_SLOW'] = df_bt['Close'].rolling(sma_slow_len, min_periods=1).mean()

    df_bt['RSI_raw'] = RSI(df_bt['Close'], rsi_len)
    df_bt['RSI'] = df_bt['RSI_raw'].ewm(span=7, adjust=False).mean()
    df_bt['RSI_EMA'] = df_bt['RSI'].ewm(span=rsi_ema_len, adjust=False).mean()
    df_bt['ADX'], df_bt['DI+'], df_bt['DI-'] = ADX(df_bt, 14)
    df_bt['ADX_ROC'] = df_bt['ADX'].pct_change(periods=5).fillna(0)
    df_bt['FAST_EMA_ROC'] = df_bt['SMA_FAST'].pct_change(20).fillna(0)
    df_bt['RSI_EMA_ROC'] = df_bt['RSI'].pct_change(20).fillna(0)
    df_bt['ATR'] = ATR(df_bt, 14)
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
        rsi_block = (row['RSI'] < 30) and (row['RSI'] < row['RSI_EMA'])
        
        entry_condition = (
            (row['ADX'] > 25) &
            (row['ADX_ROC'] > 0) &
            (row['RSI'] > row['RSI_EMA']) &
            (row['RSI'] > 30) &
            (curr_close > row['SMA_FAST']) &
            (curr_close > row['SMA_SLOW']) &
            (~rsi_block)
        )

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
                exit_reason = 'Stop_Loss'
                exit_triggered = True
            elif curr_close < row['SMA_FAST']:
                exit_price = curr_close
                exit_reason = 'SMA_Exit'
                exit_triggered = True
            elif (curr_close / entry_price - 1) >= take_profit_pct:
                exit_price = curr_close
                exit_reason = 'Take_Profit'
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
    df_raw, capital, sma_fast_len, rsi_len, rsi_ema_len, stop_loss_pct, take_profit_pct
)

latest = df_bt.iloc[-1]
total_return = ((final_capital / capital) - 1) * 100

st.subheader(f"📊 {ticker}: RESULTS")
col1, col2, col3 = st.columns(3)
col1.metric("Total Return", f"{total_return:+.1f}%")
col2.metric("Final Capital", f"${final_capital:,.0f}")
col3.metric("Total Trades", len(trades))

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


fig, axes = plt.subplots(4, 1, figsize=(16, 12), height_ratios=[3, 1, 1, 1.5])
fig.patch.set_facecolor('white')

ax1 = axes[0]

x = df_bt.index
y = df_bt['Close'].values

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

#ax1.plot(df_bt.index, df_bt['Close'], c='gray', linewidth=2, label='Close', alpha=0.5)
ax1.plot(df_bt.index, df_bt['SMA_FAST'], color='orange', linewidth=2, label=f'SMA{sma_fast_len}', alpha=0.5)
ax1.plot(df_bt.index, df_bt['SMA_SLOW'], color='red', linewidth=2, label=f'SMA{sma_slow_len}', alpha=0.5)

rsi_block = (df_bt['RSI'] < 30) & (df_bt['RSI'] < df_bt['RSI_EMA'])

entry_signals = df_bt[
    (df_bt['ADX'] > 25 & (df_bt['ADX_ROC'] > 0)) &
    (df_bt['RSI'] > df_bt['RSI_EMA']) &
    (df_bt['RSI'] > 30) &
    (df_bt['Close'] > df_bt['SMA_FAST']) &
    (df_bt['Close'] > df_bt['SMA_SLOW']) &
    (~rsi_block)
]

if len(entry_signals) > 0:
    ax1.scatter(entry_signals.index, entry_signals['Close'], color='magenta', marker='d', s=10, 
               label=f'Signals ({len(entry_signals)})', alpha = 0.2, zorder=5)

if len(trades) > 0:
    trades_plot = pd.DataFrame(trades)
    ax1.scatter(trades_plot['entry_date'], trades_plot['entry_price'], 
               color='blue', marker='o', s=15, label=f'Entries ({len(trades)})',  alpha = 0.2, zorder=2, edgecolors='darkblue', linewidths=0.7)
    
    exits = trades_plot[trades_plot['exit_date'].notna()]
    if len(exits) > 0:
        winners = exits[exits['pnl'] > 0]
        losers = exits[exits['pnl'] <= 0]
        if len(winners) > 0:
            ax1.scatter(winners['exit_date'], winners['exit_price'], color='green', marker='o', s=20, label = f'Winners ({len(winners)})',  alpha = 0.2, zorder=1, edgecolors='darkgreen', linewidths=0.7)
        if len(losers) > 0:
            ax1.scatter(losers['exit_date'], losers['exit_price'], color='red', marker='o', s=20,  label = f'Losses ({len(losers)})',  alpha = 0.2, zorder=1, edgecolors='black', linewidths=0.7)

ax1.set_title(f'{ticker} - Simple RSI+SMA | Return: {total_return:+.1f}% | {len(trades)} Trades', 
              fontsize=16, fontweight='bold', pad=15)
ax1.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=8, ncol=2)
ax1.grid(True, alpha=0.2, linestyle='--')
ax1.set_facecolor('#F8F9FA')

ax2 = axes[1]
ax2.plot(df_bt.index, df_bt['RSI'], color='#9B59B6', linewidth=1.1, label=f'RSI({rsi_len})', alpha = 0.7)
ax2.plot(df_bt.index, df_bt['RSI_EMA'], color='red', linewidth=2, label=f'RSI_EMA({rsi_ema_len})', alpha = 0.7)
ax2.axhline(y=80, color='#E74C3C', linestyle='--', alpha=0.5, linewidth=1)
ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.4, linewidth=1)
ax2.axhline(y=25, color='#27AE60', linestyle='--', alpha=0.5, linewidth=1)
ax2.set_ylabel('RSI', fontsize=11, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.2)
ax2.set_ylim(0, 100)
ax2.set_facecolor('#F8F9FA')

ax3 = axes[2]
ax3.plot(df_bt.index, df_bt['DI+'], color='green', linewidth=1, label='DI+', alpha=0.5)
ax3.plot(df_bt.index, df_bt['DI-'], color='red', linewidth=1, label='DI-', alpha=0.5)
ax3.plot(df_bt.index, df_bt['ADX'], color='gray', linewidth=2, label='ADX', alpha=0.7)
ax3.axhline(y=25, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax3.set_ylabel('ADX/DI', fontsize=11, fontweight='bold')
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.2)
ax3.set_facecolor('#F8F9FA')


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

st.subheader("🔍 LIVE STATUS")
col1, col2, col3, col4 = st.columns(4)
live_entry = (
    (latest['ADX'] > 25) and 
    (latest['ADX_ROC'] > 0) and
    (latest['RSI'] > latest['RSI_EMA']) and
    (latest['RSI'] > 30) and
    (latest['Close'] > latest['SMA_FAST']) and
    (latest['Close'] > latest['SMA_SLOW'])
)
#live_entry = (latest['RSI'] > latest['RSI_EMA']) and (latest['Close'] > latest['SMA_FAST'])
col1.metric("Momentum", "✅" if latest['RSI'] > latest['RSI_EMA'] else "❌", f"{latest['RSI']:.1f}")
col2.metric("Price above avg?", "✅" if latest['Close'] > latest['SMA_FAST'] else "❌", f"{latest['SMA_FAST']:.1f}")
col3.metric("Trend Strength", f"{latest['ADX']:.1f}", "✅" if latest['ADX'] > 25 else "❌")
col4.metric("Signal (Mom & Trend)", "🟢 ENTRY" if live_entry else "🔴 WAIT")

if live_entry:
    st.success("🎯 LIVE ENTRY SIGNAL!")
    st.balloons()

with st.expander("📋 Trade History"):
    if len(trades) > 0:
        st.dataframe(pd.DataFrame(trades), use_container_width=True)

st.caption(f"Simple Strategy | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
