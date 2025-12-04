#!/usr/bin/env python
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -----------------------------
# SETTINGS
# -----------------------------

TICKER = "COIN"
START = "2020-01-01"
INTERVAL = "1d"

TP_PCT = 0.06
SL_PCT = 0.03
MAX_HOLD = 15

CONF_THRESHOLD = 0.65
SPLIT_RATIO = 0.6

CAPITAL = 10000
RISK_PER_TRADE = 0.15


# -----------------------------
# DATA
# -----------------------------

df = yf.download(TICKER, start=START, interval=INTERVAL, group_by="column", auto_adjust=True)
df.columns = df.columns.get_level_values(0)
df.dropna(inplace=True)


# -----------------------------
# INDICATORS
# -----------------------------

def add_features(df):
    df['Return'] = df['Close'].pct_change()

    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    ema_fast = df['Close'].ewm(span=12).mean()
    ema_slow = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema_fast - ema_slow
    df['Signal'] = df['MACD'].ewm(span=9).mean()

    df['BB_MID'] = df['Close'].rolling(20).mean()
    df['BB_UP'] = df['BB_MID'] + 2 * df['Close'].rolling(20).std()
    df['BB_LOW'] = df['BB_MID'] - 2 * df['Close'].rolling(20).std()

    tr = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift()),
        abs(df['Low'] - df['Close'].shift())
    ], axis=1).max(axis=1)

    df['ATR'] = tr.rolling(14).mean()

    return df

df = add_features(df)
df.dropna(inplace=True)

FEATURES = ['RSI','SMA_20','SMA_50','MACD','Signal','ATR']

# -----------------------------
# LABELING (TP BEFORE SL)
# -----------------------------

def create_labels(df, tp_pct, sl_pct, max_hold):
    labels = []

    for i in range(len(df) - max_hold):
        entry = df['Close'].iloc[i]
        future = df.iloc[i+1:i+1+max_hold]

        tp = entry * (1 + tp_pct)
        sl = entry * (1 - sl_pct)

        outcome = 0

        for j in range(len(future)):
            if future['High'].iloc[j] >= tp:
                outcome = 1
                break
            if future['Low'].iloc[j] <= sl:
                outcome = 0
                break

        labels.append(outcome)

    return np.array(labels)

labels = create_labels(df, TP_PCT, SL_PCT, MAX_HOLD)
df = df.iloc[:len(labels)]
df['Label'] = labels

# -----------------------------
# WALK FORWARD BACKTEST
# -----------------------------

split = int(len(df) * SPLIT_RATIO)

equity = CAPITAL
equity_curve = []
positions = []
trades = []

for i in range(split, len(df) - MAX_HOLD):

    train = df.iloc[:i]
    test = df.iloc[i:i+1]

    X_train = train[FEATURES]
    y_train = train['Label']

    X_test = test[FEATURES]

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=200, max_depth=6))
    ])

    reg = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=150, max_depth=6))
    ])

    y_return = train['Return']

    clf.fit(X_train, y_train)
    reg.fit(X_train, y_return)

    prob = clf.predict_proba(X_test)[0][1]
    expected_return = reg.predict(X_test)[0]

    entry_price = df['Close'].iloc[i]

    if prob > CONF_THRESHOLD and expected_return > 0:

        capital_used = equity * RISK_PER_TRADE
        shares = capital_used / entry_price

        future = df.iloc[i+1:i+1+MAX_HOLD]

        tp = entry_price * (1 + TP_PCT)
        sl = entry_price * (1 - SL_PCT)

        exit_price = None
        result = 0

        for j in range(len(future)):

            if future['Low'].iloc[j] <= sl:
                exit_price = sl
                result = -1
                break

            if future['High'].iloc[j] >= tp:
                exit_price = tp
                result = 1
                break

        if exit_price is None:
            exit_price = future['Close'].iloc[-1]
            result = 1 if exit_price > entry_price else -1

        profit = (exit_price - entry_price) * shares
        equity += profit

        trades.append(profit)
        positions.append((df.index[i], entry_price, exit_price, profit))

    equity_curve.append(equity)

# -----------------------------
# STATS
# -----------------------------

trades = np.array(trades)

total_return = ((equity - CAPITAL) / CAPITAL) * 100
winrate = (trades > 0).sum() / len(trades) * 100 if len(trades) else 0
profit_factor = trades[trades > 0].sum() / abs(trades[trades < 0].sum()) if any(trades < 0) else 0
max_dd = max(np.maximum.accumulate(equity_curve) - equity_curve)

print("\n========== RESULTS ==========")
print(f"Trades            : {len(trades)}")
print(f"Win Rate          : {round(winrate,2)} %")
print(f"Profit Factor     : {round(profit_factor,2)}")
print(f"Max Drawdown      : {round(max_dd,2)}")
print(f"Total Return      : {round(total_return,2)} %")
print(f"Final Capital     : {round(equity,2)}")
print("=============================\n")

# -----------------------------
# PLOTS
# -----------------------------

plt.figure(figsize=(14,7))
plt.plot(equity_curve, label="Equity Curve")
plt.title(f"Equity Curve - {TICKER}")
plt.legend()
plt.grid()
plt.show()


plt.figure(figsize=(14,7))
plt.plot(df['Close'], label="Price")

for d, entry, exitp, profit in positions:
    if profit > 0:
        plt.scatter(d, entry, color='green', s=30)
    else:
        plt.scatter(d, entry, color='red', s=30)

plt.title(f"Trades - {TICKER}")
plt.legend()
plt.grid()
plt.show()
