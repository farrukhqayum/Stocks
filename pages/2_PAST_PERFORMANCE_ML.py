#!/usr/bin/env python
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.set_page_config(layout="wide")
st.title("📈 Machine Learning Backtest (NO LEAKAGE)")

# ---------------------------
# SIDEBAR
# ---------------------------

ticker = st.sidebar.text_input("Ticker", "COIN")
start = st.sidebar.text_input("Start Date", "2020-01-01")
interval = st.sidebar.selectbox("Interval", ["1d", "1h"], index=0)

tp_pct = st.sidebar.slider("Take Profit %", 1, 15, 6) / 100
sl_pct = st.sidebar.slider("Stop Loss %", 1, 10, 3) / 100
max_hold = st.sidebar.slider("Max Holding Days", 5, 40, 15)

conf_threshold = st.sidebar.slider("Confidence Threshold", 50, 90, 65) / 100
split_ratio = st.sidebar.slider("Train Split %", 50, 90, 60) / 100

capital = st.sidebar.number_input("Starting Capital", value=10000)
risk_per_trade = st.sidebar.slider("Risk Per Trade %", 5, 50, 15) / 100

run = st.sidebar.button("▶ Run Backtest")

# ---------------------------
# DATA LOAD
# ---------------------------

@st.cache_data
def load_data(ticker, start, interval):
    df = yf.download(ticker, start=start, interval=interval, group_by="column", auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open','High','Low','Close','Volume']]
    df.dropna(inplace=True)
    return df

# ---------------------------
# INDICATORS
# ---------------------------

def add_features(df):
    df = df.copy()

    df['Return'] = df['Close'].pct_change()

    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    ema_fast = df['Close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    mid = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()

    df['BB_MID'] = mid
    df['BB_UP'] = mid + 2*std
    df['BB_LOW'] = mid - 2*std

    tr = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift()),
        abs(df['Low'] - df['Close'].shift())
    ], axis=1).max(axis=1)

    df['ATR'] = tr.rolling(14).mean()

    return df


# ---------------------------
# LABELS
# ---------------------------

def create_labels(df, tp, sl, max_hold):
    labels = []

    for i in range(len(df) - max_hold):
        entry = df['Close'].iloc[i]
        future = df.iloc[i+1:i+1+max_hold]

        tp_price = entry * (1 + tp)
        sl_price = entry * (1 - sl)

        outcome = 0

        for j in range(len(future)):
            if future['High'].iloc[j] >= tp_price:
                outcome = 1
                break
            if future['Low'].iloc[j] <= sl_price:
                outcome = 0
                break

        labels.append(outcome)

    return np.array(labels)


# ---------------------------
# BACKTEST
# ---------------------------

if run:

    df = load_data(ticker, start, interval)
    df = add_features(df)
    df.dropna(inplace=True)

    FEATURES = ['RSI','SMA_20','SMA_50','MACD','Signal','ATR']

    labels = create_labels(df, tp_pct, sl_pct, max_hold)
    df = df.iloc[:len(labels)]
    df['Label'] = labels

    split = int(len(df) * split_ratio)

    equity = capital
    equity_curve = []
    trades = []
    positions = []

    st.write("✅ Data points:", len(df))

    progress = st.progress(0)

    for i in range(split, len(df)-max_hold):

        progress.progress((i-split) / (len(df)-max_hold-split))

        train = df.iloc[:i]
        test = df.iloc[i:i+1]

        X_train = train[FEATURES]
        y_train = train['Label']
        y_return = train['Return']

        X_test = test[FEATURES]

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=150, max_depth=6))
        ])

        reg = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=120, max_depth=6))
        ])

        clf.fit(X_train, y_train)
        reg.fit(X_train, y_return)

        prob = clf.predict_proba(X_test)[0][1]
        exp_return = reg.predict(X_test)[0]

        entry = df['Close'].iloc[i]

        if prob > conf_threshold and exp_return > 0:

            risk_amt = equity * risk_per_trade
            shares = risk_amt / entry

            future = df.iloc[i+1:i+1+max_hold]

            tp_price = entry * (1 + tp_pct)
            sl_price = entry * (1 - sl_pct)

            exit_price = None
            result = 0

            for j in range(len(future)):

                if future['Low'].iloc[j] <= sl_price:
                    exit_price = sl_price
                    result = -1
                    break

                if future['High'].iloc[j] >= tp_price:
                    exit_price = tp_price
                    result = 1
                    break

            if exit_price is None:
                exit_price = future['Close'].iloc[-1]
                result = 1 if exit_price > entry else -1

            profit = (exit_price - entry) * shares
            equity += profit

            trades.append(profit)
            positions.append((df.index[i], entry, profit))

        equity_curve.append(equity)


    trades = np.array(trades)

    total_return = ((equity - capital) / capital) * 100
    winrate = (trades > 0).sum() / len(trades) * 100 if len(trades) else 0
    profit_factor = trades[trades > 0].sum() / abs(trades[trades < 0].sum()) if any(trades < 0) else 0
    max_dd = max(np.maximum.accumulate(equity_curve) - equity_curve)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Trades", len(trades))
    col2.metric("Win Rate", f"{round(winrate,2)} %")
    col3.metric("Profit Factor", round(profit_factor,2))
    col4.metric("Total Return", f"{round(total_return,2)} %")

    st.metric("Max Drawdown", round(max_dd,2))
    st.metric("Final Capital", round(equity,2))

    # ---------------------------
    # PLOTS
    # ---------------------------

    equity_fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(equity_curve)
    ax.set_title(f"Equity Curve - {ticker}")
    ax.grid()
    st.pyplot(equity_fig)

    price_fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(df['Close'], label="Price")

    for date, entry, profit in positions:
        if profit > 0:
            ax.scatter(date, entry, color="green", s=20)
        else:
            ax.scatter(date, entry, color="red", s=20)

    ax.legend()
    ax.grid()
    st.pyplot(price_fig)

    st.success("Backtest complete ✅")
