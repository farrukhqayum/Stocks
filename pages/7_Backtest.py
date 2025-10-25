#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

st.set_page_config(page_title="Bull Run Entry Backtester", layout="wide")

# ======================
# PAGE HEADER
# ======================
st.title("📈 Bull Run Entry Backtester (SMA10, SMA50, RSI Recovery)")
st.markdown("""
This strategy identifies **trend continuation entries** in a bull run when:
- SMA10 > SMA50 (trend confirmed)  
- Price dips below SMA10 (pullback)  
- RSI recovers from <40 to >50 (momentum recovery)  
Then it backtests entries with a 2R take profit and adaptive ATR-based stop loss.
""")

# ======================
# USER INPUTS
# ======================
col1, col2, col3, col4 = st.columns(4)
with col1:
    ticker = st.text_input("Ticker", value="COIN")
with col2:
    period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=1)
with col3:
    TP_RR = st.number_input("Take Profit (R:R)", value=2.0, step=0.5)
with col4:
    ATR_MULT = st.number_input("ATR Multiplier for SL", value=1.5, step=0.5)

# ======================
# MAIN LOGIC
# ======================
if st.button("Run Backtest"):
    st.write("Fetching data and running analysis...")

    df = yf.download(ticker, period=period, interval="1d")
    if df.empty:
        st.error("No data returned from Yahoo Finance.")
        st.stop()

    # Indicators
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()

    # Entry Conditions
    df['trend_up'] = df['SMA10'] > df['SMA50']
    df['pullback'] = (df['Close'].shift(1) < df['SMA10'].shift(1)) & (df['RSI'].shift(1) < 40)
    df['recovery'] = (df['Close'] > df['SMA10']) & (df['RSI'] > 50)
    df['signal'] = df['trend_up'] & df['pullback'] & df['recovery']

    # Simulate Trades
    trades = []
    for i in range(1, len(df)):
        if df['signal'].iloc[i]:
            entry_date = df.index[i]
            entry_price = df['Close'].iloc[i]
            SL = entry_price - ATR_MULT * df['ATR'].iloc[i]
            TP = entry_price + TP_RR * (entry_price - SL)

            # Forward simulate
            for j in range(i + 1, len(df)):
                low = df['Low'].iloc[j]
                high = df['High'].iloc[j]
                date = df.index[j]
                if low <= SL:
                    trades.append((entry_date, date, entry_price, SL, TP, 'SL'))
                    break
                elif high >= TP:
                    trades.append((entry_date, date, entry_price, SL, TP, 'TP'))
                    break
            else:
                trades.append((entry_date, df.index[-1], entry_price, SL, TP, 'Open'))

    results = pd.DataFrame(trades, columns=['EntryDate', 'ExitDate', 'Entry', 'SL', 'TP', 'Outcome'])
    if len(results) == 0:
        st.warning("No valid signals found in this period.")
        st.stop()

    results['Return_%'] = np.where(results['Outcome'] == 'TP',
                                   (results['TP'] - results['Entry']) / results['Entry'] * 100,
                                   (results['SL'] - results['Entry']) / results['Entry'] * 100)

    # ======================
    # STATS
    # ======================
    st.subheader("📊 Backtest Summary")
    c1, c2, c3, c4 = st.columns(4)
    win_rate = 100 * results['Outcome'].eq('TP').sum() / len(results)
    avg_return = results['Return_%'].mean()
    c1.metric("Total Trades", len(results))
    c2.metric("Win Rate", f"{win_rate:.1f}%")
    c3.metric("Avg Return", f"{avg_return:.2f}%")
    c4.metric("Last Signal Date", str(results['EntryDate'].iloc[-1].date()))

    st.dataframe(results.tail(10))

    # ======================
    # PLOT
    # ======================
    st.subheader("📈 Chart with Entries / Exits")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    # Price & SMA plot
    ax1.plot(df.index, df['Close'], label='Close', linewidth=1.2)
    ax1.plot(df.index, df['SMA10'], label='SMA10', linestyle='--', alpha=0.8)
    ax1.plot(df.index, df['SMA50'], label='SMA50', linestyle='--', alpha=0.8)

    for _, row in results.iterrows():
        if row['Outcome'] == 'TP':
            color = 'green'
            marker = '^'
        elif row['Outcome'] == 'SL':
            color = 'red'
            marker = 'v'
        else:
            color = 'blue'
            marker = 'o'
        ax1.scatter(row['EntryDate'], row['Entry'], color=color, marker=marker, s=80, zorder=5)

    ax1.set_title(f"{ticker} Bull Run Entries ({period})")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # RSI subplot
    ax2.plot(df.index, df['RSI'], color='purple', label='RSI(14)')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax2.set_title("RSI(14)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
