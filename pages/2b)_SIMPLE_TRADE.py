import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Trend Filter Strategy")
st.title("🚀 Trend Filter RSI + SMA Strategy")

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

col4, col5, col6 = st.columns(3)
stop_loss_pct = col4.number_input("Stop Loss %", value=99.0, min_value=1.0, max_value=99.0)/100
take_profit_pct = col5.number_input("Take Profit %", value=7.0, min_value=3.0, max_value=15.0)/100
risky_entry_pct = col6.number_input("Risky Entry %", value=20.0, min_value=5.0, max_value=30.0)/100

col7, col8, col9 = st.columns(3)
risky_tp_pct = col7.number_input("Risky TP %", value=15.0, min_value=3.0, max_value=25.0)/100
risky_sl_pct = col8.number_input("Risky SL %", value=9.0, min_value=2.0, max_value=25.0)/100
no_entry_pct = col9.number_input("No Entry Above SMA %", value=20.0, min_value=5.0, max_value=25.0)/100

# Filters
col_red1, col_red2 = st.columns(2)
red_days_wait = col_red1.number_input("Wait X Red Days After Profit", value=2, min_value=0, max_value=5)

col_filter1, col_filter2 = st.columns(2)
below_exit_pct = col_filter1.number_input("Entry must be X% below last exit", value=5.0, min_value=1.0, max_value=15.0)/100

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

def run_trend_filter_backtest(df, initial_capital, sma_fast_len, rsi_len, rsi_ema_len, stop_loss_pct, take_profit_pct, risky_entry_pct, risky_tp_pct, risky_sl_pct, no_entry_pct, red_days_wait, below_exit_pct):
    df_bt = df.copy()
    df_bt['SMA_FAST'] = df_bt['Close'].rolling(sma_fast_len, min_periods=1).mean()
    df_bt['RSI'] = RSI(df_bt['Close'], rsi_len)
    df_bt['RSI_EMA'] = df_bt['RSI'].ewm(span=rsi_ema_len, adjust=False).mean()
    df_bt['ADX'], df_bt['DI+'], df_bt['DI-'] = ADX(df_bt, 14)
    df_bt['ATR'] = ATR(df_bt, 14)
    
    df_bt['Trend_Signal'] = ((df_bt['Close'] > df_bt['SMA_FAST']) & (df_bt['RSI'] > df_bt['RSI_EMA'])).astype(int)
    df_bt['Red_Day'] = df_bt['Close'] < df_bt['Open']
    df_bt = df_bt.dropna()
    
    cash = float(initial_capital)
    position_shares = 0.0
    entry_price = 0.0
    stop_price = 0.0
    equity_curve = []
    trades = []
    red_days_count = 0
    just_exited_profit = False
    risky_pending = False
    risky_entry_price = 0.0
    risky_stop_price = 0.0
    is_risky_trade = False
    last_exit_price = 0.0  # NEW: Track last exit price
    waiting_for_pullback = False  # NEW: Pullback requirement
    
    for i in range(len(df_bt)):
        row = df_bt.iloc[i]
        curr_close = float(row['Close'])
        curr_low = float(row['Low'])
        curr_open = float(row['Open'])
        price_vs_sma = curr_close / row['SMA_FAST']
        is_red_day = row['Red_Day']
        trend_signal = row['Trend_Signal']
        
        if not is_red_day:
            red_days_count = 0
            just_exited_profit = False
        
        if position_shares == 0:
            # 1. Block entry if waiting for red days after profit
            if just_exited_profit and red_days_count < red_days_wait:
                pass
            # 2. NEW: Block entry if waiting for pullback from last exit
            elif waiting_for_pullback and (curr_close > last_exit_price * (1 - below_exit_pct)):
                pass
            # 3. ORIGINAL: No entry if too far above SMA
            elif price_vs_sma >= (1 + no_entry_pct):
                pass
            # 4. ORIGINAL NORMAL ENTRY (Trend + not too high above SMA)
            elif trend_signal == 1 and price_vs_sma < (1 + no_entry_pct):
                position_shares = (cash * 0.95) / curr_close
                entry_cost = position_shares * curr_close
                cash -= entry_cost
                entry_price = curr_close
                stop_price = entry_price * (1 - stop_loss_pct)
                is_risky_trade = False
                waiting_for_pullback = False  # Reset pullback
                
                trades.append({
                    'entry_date': df_bt.index[i], 'entry_price': entry_price,
                    'exit_date': None, 'exit_price': None, 'pnl': 0, 'pnl_pct': 0, 'type': 'Normal'
                })
            # 5. ORIGINAL: Set risky pending
            elif not risky_pending and price_vs_sma <= (1 - risky_entry_pct):
                risky_pending = True
            # 6. ORIGINAL RISKY ENTRY (Risky pending + Trend signal + pullback OK)
            elif risky_pending and trend_signal == 1 and (not waiting_for_pullback or curr_close <= last_exit_price * (1 - below_exit_pct)):
                position_shares = (cash * 0.95) / curr_open
                entry_cost = position_shares * curr_open
                cash -= entry_cost
                risky_entry_price = curr_open
                risky_stop_price = risky_entry_price * (1 - risky_sl_pct)
                risky_pending = False
                is_risky_trade = True
                waiting_for_pullback = False  # Reset pullback
                
                trades.append({
                    'entry_date': df_bt.index[i], 'entry_price': risky_entry_price,
                    'exit_date': None, 'exit_price': None, 'pnl': 0, 'pnl_pct': 0, 'type': 'Risky'
                })
        
        elif position_shares > 0:
            exit_triggered = False
            exit_price = 0
            
            if not row['Trend_Signal']:
                exit_price = curr_close
                exit_triggered = True
            elif is_risky_trade:
                if curr_low <= risky_stop_price:
                    exit_price = curr_low
                    exit_triggered = True
                elif (curr_close / risky_entry_price - 1) >= risky_tp_pct:
                    exit_price = curr_close
                    exit_triggered = True
            else:
                if curr_low <= stop_price:
                    exit_price = curr_low
                    exit_triggered = True
                elif (curr_close / entry_price - 1) >= take_profit_pct:
                    exit_price = curr_close
                    exit_triggered = True
            
            if exit_triggered:
                exit_value = position_shares * exit_price
                pnl = exit_value - (position_shares * (risky_entry_price if is_risky_trade else entry_price))
                pnl_pct = ((exit_price - (risky_entry_price if is_risky_trade else entry_price)) / (risky_entry_price if is_risky_trade else entry_price)) * 100
                cash += exit_value
                
                trades[-1]['exit_date'] = df_bt.index[i]
                trades[-1]['exit_price'] = exit_price
                trades[-1]['pnl'] = pnl
                trades[-1]['pnl_pct'] = pnl_pct
                
                # NEW: Set pullback requirement after ANY exit
                last_exit_price = exit_price
                waiting_for_pullback = True
                
                if pnl > 0:
                    just_exited_profit = True
                    red_days_count = 0
                else:
                    just_exited_profit = False
                    red_days_count = 0
                
                position_shares = 0.0
                is_risky_trade = False
        
        if just_exited_profit and is_red_day:
            red_days_count += 1
        
        pos_value = position_shares * curr_close
        equity_curve.append(cash + pos_value)
    
    if position_shares > 0:
        final_price = float(df_bt['Close'].iloc[-1])
        final_value = position_shares * final_price
        cash += final_value
        final_entry_price = risky_entry_price if is_risky_trade else entry_price
        pnl_pct = ((final_price - final_entry_price) / final_entry_price) * 100
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

# LIVE STATUS
st.subheader("🔍 LIVE TREND FILTER STATUS")
df_live = df_raw.copy()
df_live['SMA_FAST'] = df_live['Close'].rolling(sma_fast_len, min_periods=1).mean()
df_live['RSI'] = RSI(df_live['Close'], rsi_len)
df_live['RSI_EMA'] = df_live['RSI'].ewm(span=rsi_ema_len, adjust=False).mean()
df_live['Trend_Signal'] = ((df_live['Close'] > df_live['SMA_FAST']) & (df_live['RSI'] > df_live['RSI_EMA'])).astype(int)

latest = df_live.iloc[-1]
live_signal = latest['Trend_Signal']

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Close > SMA", "✅" if latest['Close'] > latest['SMA_FAST'] else "❌", f"{latest['Close']:.2f}")
col2.metric("RSI > RSI_EMA", "✅" if latest['RSI'] > latest['RSI_EMA'] else "❌", f"{latest['RSI']:.1f}")
col3.metric("Trend Signal", "🟢 LONG" if live_signal else "🔴 EXIT")
col4.metric("Price vs SMA", f"{((latest['Close']/latest['SMA_FAST']-1)*100):+.1f}%")
col5.metric("Pullback OK", "✅", f"{below_exit_pct*100:.0f}%")

if live_signal:
    st.success("🎯 TREND SIGNAL: GO LONG!")
    st.balloons()

st.divider()

# BACKTEST
equity_series, final_capital, trades, df_bt = run_trend_filter_backtest(
    df_raw, capital, sma_fast_len, rsi_len, rsi_ema_len, stop_loss_pct, 
    take_profit_pct,
