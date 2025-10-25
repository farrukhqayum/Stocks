#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from imports import *   # uses your custom ta.calculate_rsi and ta.calculate_atr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

st.set_page_config(page_title="Bull Run 5% TP/SL Backtester", layout="wide")

# ======================
# PAGE HEADER
# ======================
st.title("📈 Bull Run 5% TP/SL Backtester")
st.markdown("""
This strategy identifies **trend continuation entries** when:
- SMA10 > SMA50 (uptrend confirmed)
- Price dips below SMA10 (pullback)
- RSI recovers from <40 to >50 (momentum recovery)

Then it backtests with:
- **Take Profit:** +5 %
- **Stop Loss:** −5 %
""")

# ======================
# USER INPUTS
# ======================
ticker = st.text_input("Ticker", value="COIN")

# ======================
# MAIN LOGIC
# ======================
if st.button("Run Backtest"):
    st.write("Downloading 5-year data and running simulation...")

    df = yf.download(ticker, period="5y", interval="1d")
    if df.empty:
        st.error("No data from Yahoo Finance.")
        st.stop()

    # --- Indicators ---
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['RSI'] = ta.calculate_rsi(df)
    df['ATR'] = ta.calculate_atr(df['High'], df['Low'], df['Close'])

    # --- Entry Logic ---
    df['trend_up'] = df['SMA10'] > df['SMA50']
    df['pullback'] = (df['Close'].shift(1) < df['SMA10'].shift(1)) & (df['RSI'].shift(1) < 40)
    df['recovery'] = (df['Close'] > df['SMA10']) & (df['RSI'] > 50)
    df['signal'] = df['trend_up'] & df['pullback'] & df['recovery']

    # --- Backtest Simulation ---
    trades = []
    for i in range(1, len(df)):
        if df['signal'].iloc[i]:
            entry_date = df.index[i]
            entry_price = df['Close'].iloc[i]
            TP = entry_price * 1.05
            SL = entry_price * 0.95

            # simulate forward candle-by-candle
            for j in range(i + 1, len(df)):
                low = df['Low'].iloc[j]
                high = df['High'].iloc[j]
                exit_date = df.index[j]

                if low <= SL:
                    trades.append((entry_date, exit_date, entry_price, SL, TP, 'SL'))
                    break
                elif high >= TP:
                    trades.append((entry_date, exit_date, entry_price, SL, TP, 'TP'))
                    break
            else:
                trades.append((entry_date, df.index[-1], entry_price, SL, TP, 'Open'))

    results = pd.DataFrame(trades, columns=['EntryDate', 'ExitDate', 'Entry', 'SL', 'TP', 'Outcome'])

    if results.empty:
        st.warning("No valid signals found.")
        st.stop()

    # --- Stats ---
    results['Return_%'] = np.where(results['Outcome'] == 'TP', 5, -5)
    win_rate = 100 * results['Outcome'].eq('TP').sum() / len(results)
    avg_return = results['Return_%'].mean()

    st.subheader("📊 Backtest Summary (5 Years)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Trades", len(results))
    c2.metric("Win Rate", f"{win_rate:.1f}%")
    c3.metric("Avg Return per Trade", f"{avg_return:.2f}%")
    st.dataframe(results.tail(10))

    # --- Chart ---
    st.subheader("📈 Chart with 5% TP/SL Entries")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(df.index, df['Close'], label='Close', linewidth=1.2)
    ax1.plot(df.index, df['SMA10'], label='SMA10', linestyle='--', alpha=0.8)
    ax1.plot(df.index, df['SMA50'], label='SMA50', linestyle='--', alpha=0.8)

    for _, row in results.iterrows():
        color = 'green' if row['Outcome'] == 'TP' else 'red'
        marker = '^' if row['Outcome'] == 'TP' else 'v'
        ax1.scatter(row['EntryDate'], row['Entry'], color=color, marker=marker, s=80, zorder=5)

    ax1.set_title(f"{ticker} — Bull Run Entries (5y, ±5%)")
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
