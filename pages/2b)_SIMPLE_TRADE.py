import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Hybrid Signals")
st.title("🚀 Hybrid Signals - Simple")

col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Ticker", value="COIN")
    capital = st.number_input("Capital ($)", value=10000, min_value=1000)
with col2:
    period = st.selectbox("Backtest Period", ["2y", "1y", "6mo"], index=0)

col1, col2, col3 = st.columns(3)
ema_fast = col1.number_input("EMA Fast", value=12, min_value=5, max_value=30)
ema_slow = col2.number_input("EMA Slow", value=26, min_value=15, max_value=50)
rsi_len = col3.number_input("RSI Length", value=14, min_value=10, max_value=21)

col4, col5, col6 = st.columns(3)
rsi_ema_len = col4.number_input("RSI EMA Len", value=9, min_value=5, max_value=30)
conf_thresh = col5.number_input("Conf %", value=65, min_value=50, max_value=100)
stop_loss_pct = col6.number_input("Stop Loss %", value=2.0, min_value=1.0, max_value=99.0)/100

st.sidebar.header("🎯 Hybrid Settings")
trail_mult = st.sidebar.number_input("Trail % Mult", value=2.5, min_value=1.5, max_value=4.0, step=0.5)
partial_tp_pct = st.sidebar.number_input("Partial TP %", value=4.0, min_value=2.0, max_value=8.0)/100
max_hold_days = st.sidebar.number_input("Max Hold Days", value=180, min_value=10, max_value=365)

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

def run_hybrid_backtest(df, initial_capital, ema_fast, ema_slow, rsi_len, rsi_ema_len, 
                       conf_thresh, stop_loss_pct, trail_mult, partial_tp_pct, max_hold_days):
    
    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    open_price = df['Open'].squeeze()
    
    df_bt = df.copy()
    df_bt['EMA_FAST'] = close.ewm(span=ema_fast, adjust=False).mean()
    df_bt['EMA_SLOW'] = close.ewm(span=ema_slow, adjust=False).mean()
    df_bt['RSI'] = RSI(close, rsi_len)
    df_bt['RSI_EMA'] = df_bt['RSI'].ewm(span=rsi_ema_len, adjust=False).mean()
    df_bt['ADX'], _, _ = ADX(df_bt, 14)
    df_bt = df_bt.dropna()
    
    sma20 = close.rolling(20, min_periods=1).mean()
    trend_up = df_bt['EMA_FAST'] > df_bt['EMA_SLOW']
    rsi_bullish = (df_bt['RSI'] > df_bt['RSI_EMA']) & (df_bt['RSI'] > 50)
    adx_strong = df_bt['ADX'] > 25
    price_above_sma20 = df_bt['Close'] > sma20
    
    df_bt['CONFIDENCE'] = (trend_up.astype(int) + rsi_bullish.astype(int) + 
                          adx_strong.astype(int) + price_above_sma20.astype(int)) * 25
    df_bt['BULL_HIGH_CONF'] = df_bt['CONFIDENCE'] >= conf_thresh
    
    cash = float(initial_capital)
    position_shares = 0.0
    entry_price = 0.0
    trail_stop = 0.0
    partial_taken = False
    entry_date = None
    equity_curve = []
    trades = []
    in_position = False
    
    for i in range(len(df_bt)):
        row = df_bt.iloc[i]
        current_date = df_bt.index[i]
        curr_open = float(row['Open'])
        curr_high = float(row['High'])
        curr_low = float(row['Low'])
        curr_close = float(row['Close'])
        
        if (not in_position and row['BULL_HIGH_CONF'] and cash > 100):
            risk_amount = cash * 0.02
            initial_stop = curr_close * (1 - stop_loss_pct)
            shares_to_risk = risk_amount / (curr_close - initial_stop)
            
            position_shares = shares_to_risk
            entry_cost = position_shares * curr_close
            cash -= entry_cost
            entry_price = curr_close
            trail_stop = initial_stop
            partial_taken = False
            entry_date = current_date
            in_position = True
            
            trades.append({
                'entry_date': entry_date, 'entry_price': entry_price, 'shares': position_shares,
                'exit_date': None, 'exit_price': None, 'pnl': 0, 'pnl_pct': 0,
                'confidence': row['CONFIDENCE'], 'exit_reason': None
            })
        
        elif in_position:
            days_held = (current_date - entry_date).days if entry_date else 0
            
            if (not partial_taken and curr_close >= entry_price * (1 + partial_tp_pct)):
                partial_shares = position_shares * 0.5
                partial_value = partial_shares * curr_close
                cash += partial_value
                position_shares *= 0.5
                partial_taken = True
            
            new_trail = entry_price * (1 + (stop_loss_pct * trail_mult))
            trail_stop = max(trail_stop, new_trail)
            
            exit_triggered = False
            exit_price = 0
            exit_reason = None
            
            if curr_low <= trail_stop:
                exit_price = curr_low; exit_reason = 'Trail_Stop'; exit_triggered = True      # ✅ Actual low hit
            elif curr_open <= trail_stop:
                exit_price = curr_open; exit_reason = 'Gap_SL'; exit_triggered = True        # ✅ Actual open price
            elif days_held >= max_hold_days:
                exit_price = curr_close; exit_reason = 'Max_Hold'; exit_triggered = True     # ✅ Close price
            elif row['EMA_FAST'] < row['EMA_SLOW']:
                exit_price = curr_close; exit_reason = 'Trend_Rev'; exit_triggered = True    # ✅ Close price

            
            if exit_triggered:
                exit_value = position_shares * exit_price
                pnl = exit_value - (position_shares * entry_price)
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                cash += exit_value
                trades[-1].update({
                    'exit_date': current_date, 'exit_price': exit_price, 
                    'pnl': pnl, 'pnl_pct': pnl_pct, 'exit_reason': exit_reason
                })
                position_shares = 0.0
                in_position = False
        
        pos_value = position_shares * curr_close
        equity_curve.append(cash + pos_value)
    
    if in_position:
        final_price = float(df_bt['Close'].iloc[-1])
        final_value = position_shares * final_price
        cash += final_value
        trades[-1].update({
            'exit_date': df_bt.index[-1], 'exit_price': final_price,
            'pnl': final_value - (position_shares * entry_price),
            'pnl_pct': ((final_price - entry_price) / entry_price) * 100,
            'exit_reason': 'End_of_Data'
        })
    
    equity_series = pd.Series(equity_curve, index=df_bt.index)
    final_capital = cash
    return equity_series, final_capital, trades, df_bt

df_raw = get_data(ticker, period)
if df_raw is None or df_raw.empty:
    st.error("❌ Failed to load data")
    st.stop()

try:
    equity_series, final_capital, trades, df_bt = run_hybrid_backtest(
        df_raw, capital, ema_fast, ema_slow, rsi_len, rsi_ema_len, 
        conf_thresh, stop_loss_pct, trail_mult, partial_tp_pct, max_hold_days
    )
except Exception as e:
    st.error(f"❌ Backtest error: {e}")
    st.stop()

if df_bt.empty:
    st.error("❌ No data after calculations")
    st.stop()

latest = df_bt.iloc[-1]
sma20_val = df_raw['Close'].rolling(20, min_periods=1).mean().iloc[-1]
total_return = ((final_capital / capital) - 1) * 100

st.subheader("📊 HYBRID RESULTS")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Return", f"{total_return:+.1f}%", delta=f"${final_capital-capital:,.0f}")
col2.metric("Final Capital", f"${final_capital:,.0f}")
col3.metric("Total Trades", len(trades))
col4.metric("Avg Confidence", f"{df_bt['CONFIDENCE'].mean():.0f}%")

trades_df = pd.DataFrame(trades)
completed = trades_df[trades_df['exit_date'].notna()]
if len(completed) > 0:
    win_rate = (completed['pnl'] > 0).mean() * 100
    col5, col6 = st.columns(2)
    col5.metric("Win Rate", f"{win_rate:.1f}%")
    col6.metric("Profit Factor", f"{completed[completed['pnl']>0]['pnl_pct'].sum()/abs(completed[completed['pnl']<=0]['pnl_pct'].sum()):.2f}x")

fig, axes = plt.subplots(4, 1, figsize=(16, 12), height_ratios=[3, 1, 1, 1.5])
fig.patch.set_facecolor('white')

ax1 = axes[0]
ax1.plot(df_bt.index, df_bt['Close'], color='#2C3E50', linewidth=1.5, label='Close', alpha=0.5)
ax1.plot(df_bt.index, df_bt['Low'], color='#2C3E50', linewidth=1., label='Close', alpha=0.2)
ax1.plot(df_bt.index, df_bt['High'], color='#2C3E50', linewidth=1.1, label='Close', alpha=0.2)

ax1.plot(df_bt.index, df_bt['EMA_FAST'], color='#3498DB', linewidth=1.5, label='EMA12', alpha=0.6)
ax1.plot(df_bt.index, df_bt['EMA_SLOW'], color='#E74C3C', linewidth=1.5, label='EMA26', alpha=0.6)

bull_signals = df_bt[df_bt['BULL_HIGH_CONF'] == True]
if len(bull_signals) > 0:
    ax1.scatter(bull_signals.index, bull_signals['Close'],
                color='orange', marker='x', s=10, alpha=0.7,
                label=f'Signals ({len(bull_signals)})', edgecolors = 'black', linewidths = 1, zorder=3)

if len(trades) > 0:
    trades_plot = pd.DataFrame(trades)
    ax1.scatter(trades_plot['entry_date'], trades_plot['entry_price'],
                color='gray', marker='o', s=10, alpha=0.9,
                label=f'Entries ({len(trades)})', zorder=5, edgecolors='black', linewidths=0.5)

    exits = trades_plot[trades_plot['exit_date'].notna()]
   
    winners = exits[exits['pnl'] > 0]
    losers = exits[exits['pnl'] <= 0]

    if len(winners) > 0:
        ax1.scatter(winners['exit_date'], winners['exit_price'],  # ← Real trade price
                   color='#27AE60', marker='v', label='Winners', s=10, alpha=0.9, zorder=5)
    if len(losers) > 0:
        ax1.scatter(losers['exit_date'], losers['exit_price'],  # ← Real trade price
                   color='red', marker='v', label='Lossers', s=10, alpha=0.9, zorder=5)

ax1.set_title(f'{ticker} - Hybrid Strategy | Return: {total_return:+.1f}% | {len(trades)} Trades',
              fontsize=16, fontweight='bold', pad=15)
ax1.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.2)
ax1.set_facecolor('#F8F9FA')

ax2 = axes[1]
ax2.plot(df_bt.index, df_bt['RSI'].ewm(span=5, adjust=False).mean(), color='#9B59B6', linewidth=1.5, label=f'RSI({rsi_len})')
ax2.plot(df_bt.index, df_bt['RSI_EMA'], color='#F39C12', linewidth=2, label=f'RSI_EMA({rsi_ema_len})')
ax2.axhline(70, color='#E74C3C', ls='--', alpha=0.5); ax2.axhline(50, color='gray', ls='--', alpha=0.3)
ax2.axhline(30, color='#27AE60', ls='--', alpha=0.5)
ax2.set_ylabel('RSI'); ax2.legend(loc='upper left', fontsize=9); ax2.grid(True, alpha=0.2)
ax2.set_ylim(0, 100); ax2.set_facecolor('#F8F9FA')

ax3 = axes[2]
ax3.plot(df_bt.index, df_bt['ADX'], color='#E67E22', linewidth=1.5, label='ADX')
ax3.axhline(25, color='#E74C3C', ls='--', alpha=0.5, label='ADX≥25')
ax3_twin = ax3.twinx()
ax3_twin.plot(df_bt.index, df_bt['CONFIDENCE'], color='orange', alpha=0.7, linewidth=1, label='Conf')
ax3_twin.axhline(conf_thresh, color='limegreen', ls='--', alpha=0.7, label=f'Thresh {conf_thresh}%')
ax3.set_ylabel('ADX'); ax3.legend(loc='upper left', fontsize=9)
ax3_twin.set_ylabel('Conf %'); ax3_twin.legend(loc='upper right', fontsize=9)
ax3.grid(True, alpha=0.2); ax3.set_facecolor('#F8F9FA')

ax4 = axes[3]
ax4.plot(equity_series.index, equity_series, color='#27AE60', linewidth=2.5, label='Portfolio')
ax4.fill_between(equity_series.index, capital, equity_series, alpha=0.3, color='#27AE60')
ax4.axhline(capital, color='gray', ls='--', alpha=0.5, label=f'Start ${capital:,.0f}')
ax4.set_ylabel('Equity ($)'); ax4.set_xlabel('Date')
ax4.legend(loc='upper left', fontsize=9); ax4.grid(True, alpha=0.2)
ax4.set_facecolor('#F8F9FA')

plt.tight_layout()
st.pyplot(fig)

st.subheader("🔍 LIVE STATUS")
col1, col2, col3, col4, col5 = st.columns(5)
trend_ok = latest['EMA_FAST'] > latest['EMA_SLOW']
rsi_ok = latest['RSI'] > latest['RSI_EMA']
adx_ok = latest['ADX'] > 25
sma_ok = latest['Close'] > sma20_val
conf_live = latest['CONFIDENCE']

col1.metric("Trend", "✅" if trend_ok else "❌")
col2.metric("RSI", "✅" if rsi_ok else "❌")
col3.metric("ADX", f"{latest['ADX']:.0f}", "✅" if adx_ok else "❌")
col4.metric("SMA20", "✅" if sma_ok else "❌")
col5.metric("Conf", f"{conf_live:.0f}%", "🟢" if latest['BULL_HIGH_CONF'] else "🔴")

if latest['BULL_HIGH_CONF']:
    st.success("🎯 LIVE SIGNAL - ENTER NOW!")
    st.balloons()

with st.expander("📋 Trade History"):
    if len(trades) > 0:
        st.dataframe(pd.DataFrame(trades).tail(10), use_container_width=True)

st.caption(f"Hybrid Strategy | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
