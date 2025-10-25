#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from imports import *   # expects ta.calculate_rsi(df) and ta.calculate_atr(...)
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="Bull Run Weekly Entry — 7% TP/SL", layout="wide")

st.title("📈 Weekly-Entry Backtester — 7% TP & 7% SL")
st.markdown("""
Strategy:
- Generate entry signals on **weekly candles** (SMA10 > SMA50, pullback, RSI recovery).
- **Enter at weekly close**.
- Monitor daily candles after the entry for **TP% / SL%** (default 7%).
- If next-day open or next-week open gaps through TP/SL, exit at that open (cap at TP/SL).
- Mid-size moves (e.g. 3–5%) do **not** force exit — they are monitored normally.
- No overlapping trades: wait for an open trade to close before taking next weekly signal.
- Shows cumulative gains chart (compounded).
""")

# -------------------------
# User inputs
# -------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    ticker = st.text_input("Ticker", value="COIN")
with col2:
    period = st.selectbox("History period", ["2y", "3y", "5y", "7y"], index=2)
with col3:
    TP_pct = st.number_input("TP (%)", value=7.0, step=0.5)
with col4:
    SL_pct = st.number_input("SL (%)", value=7.0, step=0.5)

# extra params (optional)
col5, col6 = st.columns(2)
with col5:
    min_move_pct = st.number_input("Min mid-move threshold (%) (3-5% area)", value=3.0, step=0.5)
with col6:
    gap_exit_pct = st.number_input("Gap exit threshold (%) (next day/week open >= entry*(1+gap))", value=5.0, step=0.5)

if st.button("Run Weekly Backtest"):
    st.write(f"Downloading {period} daily data for {ticker} and running weekly-entry simulation...")
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    if df.empty:
        st.error("No data returned from Yahoo Finance for that ticker/period.")
        st.stop()

    # --- sanitize columns (yfinance sometimes returns MultiIndex) ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    # ensure core columns are Series
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

    # create weekly candles (use weekly ending on Sunday by default; pandas 'W' uses SUN)
    df_daily = df.copy()
    df_w = df_daily.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
    df_w.dropna(inplace=True)

    # --- Indicators on weekly candles ---
    df_w['SMA10'] = df_w['Close'].rolling(10).mean()
    df_w['SMA50'] = df_w['Close'].rolling(50).mean()

    # Use your calculate_rsi function on weekly frame (it expects a DataFrame with 'Close')
    df_w['RSI'] = ta.calculate_rsi(df_w)  # expects series or df
    if isinstance(df_w['RSI'], pd.DataFrame):
        df_w['RSI'] = df_w['RSI'].iloc[:, 0]

    # safe-cast
    df_w['SMA10'] = df_w['SMA10'].astype(float)
    df_w['SMA50'] = df_w['SMA50'].astype(float)
    df_w['RSI'] = df_w['RSI'].astype(float)

    # --- define weekly signals (relaxed pullback logic) ---
    df_w['trend_up'] = df_w['SMA10'] > df_w['SMA50']
    df_w['pullback'] = (df_w['Close'].shift(1) < df_w['SMA10'].shift(1)) & (df_w['RSI'].shift(1) < 45)
    df_w['recovery'] = (df_w['Close'] > df_w['SMA10']) & (df_w['RSI'] > 50)
    df_w['signal'] = df_w['trend_up'] & df_w['pullback'] & df_w['recovery']

    # Prepare storage
    trades = []
    in_trade = False
    last_exit_idx = None

    # iterate weekly rows for potential entries
    weekly_indices = df_w.index.tolist()
    daily_index = df_daily.index

    for w_idx in weekly_indices:
        if not df_w.loc[w_idx, 'signal']:
            continue

        # if already in a trade, skip until closed (no overlapping)
        if in_trade:
            continue

        entry_date_week = w_idx
        # entry price is weekly close
        entry_price = float(df_w.loc[entry_date_week, 'Close'])
        TP_price = entry_price * (1 + TP_pct / 100.0)
        SL_price = entry_price * (1 - SL_pct / 100.0)

        # find first daily index >= entry_date_week (weekly close date might be a Sunday - align to next market day)
        # choose the last business day of the week for the entry if weekly close doesn't exist in daily index:
        # find nearest previous daily index <= weekly date, prefer exact match
        possible_days = df_daily.index[df_daily.index <= entry_date_week]
        if len(possible_days) == 0:
            # no earlier daily date (unlikely) -> skip
            continue
        # entry's daily row is last trading day in that week (the weekly close corresponds to that)
        entry_day = possible_days[-1]
        # double-check close on entry_day matches entry_price (small differences might exist due to timezone/market)
        entry_price = float(df_daily.loc[entry_day, 'Close'])
        TP_price = entry_price * (1 + TP_pct / 100.0)
        SL_price = entry_price * (1 - SL_pct / 100.0)

        # Start monitoring from the next trading day after entry_day
        monitoring_start_idx = np.where(df_daily.index > entry_day)[0]
        if monitoring_start_idx.size == 0:
            # no future data -> open trade to end
            trades.append({
                'EntryDate': entry_day, 'ExitDate': df_daily.index[-1],
                'EntryPrice': entry_price, 'ExitPrice': df_daily.loc[df_daily.index[-1], 'Close'],
                'Outcome': 'Open', 'Return_%': np.nan
            })
            continue

        in_trade = True
        exit_found = False
        exit_price = None
        exit_date = None
        outcome = None

        # iterate every trading day after entry until closed
        for idx in range(monitoring_start_idx[0], len(df_daily)):
            day = df_daily.index[idx]
            o = float(df_daily.loc[day, 'Open'])
            h = float(df_daily.loc[day, 'High'])
            l = float(df_daily.loc[day, 'Low'])
            c = float(df_daily.loc[day, 'Close'])

            # 1) If intraday low <= SL -> SL hit (conservative: we treat SL before TP if both in same day)
            if l <= SL_price:
                exit_found = True
                exit_date = day
                exit_price = SL_price  # assume we get SL exactly
                outcome = 'SL'
                break

            # 2) If intraday high >= TP -> TP hit
            if h >= TP_price:
                exit_found = True
                exit_date = day
                exit_price = TP_price
                outcome = 'TP'
                break

            # 3) Next-day open gap exit: if today's open >= TP (gap-up), exit at min(open, TP). 
            #    Similarly if open <= SL, exit at max(open, SL).
            #    (We check this at the start of each day; for the very first monitored day it's effectively "next day".)
            if o >= TP_price:
                exit_found = True
                exit_date = day
                exit_price = min(o, TP_price)
                outcome = 'GapUp_TP'
                break
            if o <= SL_price:
                exit_found = True
                exit_date = day
                exit_price = max(o, SL_price)
                outcome = 'GapDown_SL'
                break

            # 4) Weekly-gap exit: if this day is the first trading day of a new week (i.e., weekly index > entry week),
            #    and week's open >= TP or <= SL, we will catch it above via 'o' check on the first trading day of that week.
            #    Also, treat mid-size moves (3-5%): if a day moves by >= min_move_pct but < TP/SL, we keep monitoring (no forced exit).

            # Continue until a hit or until the next weekly signal/next week passes; loop will continue.

        # if no exit found until end of data
        if not exit_found:
            exit_date = df_daily.index[-1]
            exit_price = float(df_daily.loc[exit_date, 'Close'])
            outcome = 'Open'

        # compute realized percent (cap at TP/SL absolute)
        ret_pct = (exit_price / entry_price - 1) * 100.0
        # But enforce that TP/SL outcomes have ±TP_pct/SL_pct (because if we capped at min/max we already did)
        if outcome in ('TP', 'GapUp_TP'):
            realized_pct = TP_pct
            exit_price = TP_price if outcome == 'TP' else exit_price
        elif outcome in ('SL', 'GapDown_SL'):
            realized_pct = -SL_pct
            exit_price = SL_price if outcome == 'SL' else exit_price
        else:
            realized_pct = ret_pct  # open or other

        trades.append({
            'EntryDate': entry_day,
            'ExitDate': exit_date,
            'EntryPrice': entry_price,
            'ExitPrice': exit_price,
            'Outcome': outcome,
            'Return_%': realized_pct
        })

        # mark trade closed
        in_trade = False

    # --- results dataframe ---
    results = pd.DataFrame(trades)
    if results.empty:
        st.warning("No weekly-entry signals found or no trades executed.")
        st.stop()

    # compute cumulative equity series: start with 1.0 and compound
    initial_cap = 1.0
    results['Return_factor'] = 1 + results['Return_%'] / 100.0
    results['Cumulative'] = initial_cap * results['Return_factor'].cumprod()
    # for plotting over time, build a time series of cumulative equity at each exit date
    equity_ts = pd.Series(data=results['Cumulative'].values, index=pd.to_datetime(results['ExitDate']))

    # --- display summary metrics ---
    total_trades = len(results)
    wins = results['Return_%'] > 0
    win_rate = 100.0 * wins.sum() / total_trades
    avg_return = results['Return_%'].mean()
    net_return_pct = (results['Cumulative'].iloc[-1] - initial_cap) / initial_cap * 100.0

    st.subheader("📊 Backtest Summary (Weekly entries)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades", total_trades)
    c2.metric("Win Rate", f"{win_rate:.1f}%")
    c3.metric("Avg Return per Trade", f"{avg_return:.2f}%")
    c4.metric("Net Return (compounded)", f"{net_return_pct:.2f}%")

    st.subheader("Trades (most recent 20)")
    st.dataframe(results.sort_values('EntryDate', ascending=False).head(20))

    # --- Plots: price with entry/exit markers and cumulative gains ---
    st.subheader("Charts")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    # price chart (daily) and mark entry points
    ax1.plot(df_daily.index, df_daily['Close'], label='Close', color='gray',  linewidth=1.0, alpha = 0.5)
    ax1.plot(df_daily.index, df_daily['SMA1'], label='df_daily.SMA1', color='yellow', linewidth=1.0, alpha = 0.5)
    ax1.plot(df_daily.index, df_daily['SMA2'], label='df_daily.SMA2', color='red', linewidth=1.0, alpha = 0.5)
    
    # mark weekly-entry points
    for _, r in results.iterrows():
        ax1.scatter(r['EntryDate'], r['EntryPrice'], color='blue', marker='^', s=80, zorder=5)
        color = 'green' if r['Return_%'] > 0 else 'red'
        ax1.scatter(r['ExitDate'], r['ExitPrice'], color=color, marker='o', s=60, zorder=5)

    ax1.set_title(f"{ticker} — Daily Close with Weekly Entries/Exits")
    ax1.legend(['Close', 'Entry', 'Exit'])
    ax1.grid(alpha=0.3)

    # cumulative equity plot
    ax2.plot(equity_ts.index, equity_ts.values, marker='o', linewidth=2)
    ax2.set_title("Cumulative Gains (compounded)")
    ax2.set_ylabel("Equity (start=1.0)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    st.success("Backtest complete.")
