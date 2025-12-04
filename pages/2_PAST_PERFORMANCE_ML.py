#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="ML - Stock Past Performance", layout="wide")
st.title("📈 ML Strategy Backtest (Safe Features Only)")

# -------------------------
# User inputs
# -------------------------
ticker = st.text_input("Ticker", value="COIN")
years = st.number_input("Years of history", min_value=1, max_value=10, value=3)
TP_pct = st.number_input("TP (%)", value=7.0, step=0.5)
SL_pct = st.number_input("SL (%)", value=14.0, step=0.5)
ml_confidence_threshold = st.number_input("ML Confidence Threshold", min_value=0, max_value=100, value=63, step=5)
max_holding_days = st.number_input("Max Holding Days", min_value=3, max_value=60, value=15, step=2)

# -------------------------
# Data fetch
# -------------------------
end_date = datetime.now()
start_date = end_date - timedelta(days=365 * years)
df_daily = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1), progress=False)

if df_daily.empty:
    st.error("No data returned from Yahoo Finance.")
    st.stop()

# -------------------------
# Technical indicators (safe, past-only)
# -------------------------
def add_indicators(df):
    df = df.copy()
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['RSI'] = df['Close'].diff().apply(lambda x: max(x,0)).rolling(14).mean() / \
                df['Close'].diff().apply(lambda x: -min(x,0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + df['RSI']))
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    return df.dropna()

df_daily = add_indicators(df_daily)

# -------------------------
# Labeling (future-based, only for training targets)
# -------------------------
def label_targets(df, tp=0.05, sl=0.05, window=14):
    df = df.copy()
    labels = []
    for i in range(len(df)):
        current_price = df['Close'].iloc[i]
        tp_price = current_price * (1 + tp)
        sl_price = current_price * (1 - sl)
        future_prices = df['Close'].iloc[i+1:i+1+window]
        if len(future_prices) == 0:
            labels.append(0)  # None
            continue
        if (future_prices >= tp_price).any():
            labels.append(2)  # TP
        elif (future_prices <= sl_price).any():
            labels.append(1)  # SL
        else:
            labels.append(0)  # None
    df['Hit_Label'] = labels
    df['Expected_Return'] = (df['Close'].shift(-window) - df['Close']) / df['Close']
    df['Expected_Loss'] = (df['Close'].shift(-window).rolling(window).min() - df['Close']) / df['Close']
    return df.dropna()

df_daily = label_targets(df_daily)

# -------------------------
# Train models (safe features only)
# -------------------------
SAFE_FEATURES = ['SMA10','SMA50','RSI','ATR']

df_model = df_daily.dropna(subset=SAFE_FEATURES + ['Hit_Label','Expected_Return','Expected_Loss'])

X_cls = df_model[SAFE_FEATURES]
y_cls = df_model['Hit_Label'].astype(int)

X_return = df_model[SAFE_FEATURES]
y_return = df_model['Expected_Return']

X_loss = df_model[SAFE_FEATURES]
y_loss = df_model['Expected_Loss']

scaler_cls = StandardScaler().fit(X_cls)
scaler_return = StandardScaler().fit(X_return)
scaler_loss = StandardScaler().fit(X_loss)

model_class = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42).fit(scaler_cls.transform(X_cls), y_cls)
model_return = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42).fit(scaler_return.transform(X_return), y_return)
model_loss = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42).fit(scaler_loss.transform(X_loss), y_loss)

models = (model_class, model_return, model_loss, scaler_cls, scaler_return, scaler_loss)

# -------------------------
# Prediction function
# -------------------------
def get_ml_prediction(df, models):
    model_class, model_return, model_loss, scaler_cls, scaler_return, scaler_loss = models
    latest = df[SAFE_FEATURES].iloc[[-1]]
    if latest.isnull().values.any():
        return None
    # Classification
    latest_scaled_cls = scaler_cls.transform(latest)
    class_probs = model_class.predict_proba(latest_scaled_cls)[0]
    predicted_class = model_class.predict(latest_scaled_cls)[0]
    # Regression
    latest_scaled_return = scaler_return.transform(latest)
    latest_scaled_loss = scaler_loss.transform(latest)
    predicted_return = model_return.predict(latest_scaled_return)[0]
    predicted_loss = model_loss.predict(latest_scaled_loss)[0]
    return {
        'will_hit': predicted_class,
        'predicted_return': predicted_return,
        'predicted_loss': predicted_loss,
        'confidence_score': class_probs[predicted_class]
    }

# -------------------------
# Backtest loop
# -------------------------
trades = []
in_trade = False
current_trade = {}

for i, current_date in enumerate(df_daily.index):
    current_data = df_daily.loc[:current_date]
    if len(current_data) < 100:
        continue
    ml_prediction = get_ml_prediction(current_data, models)
    if ml_prediction is None:
        continue

    # Entry
    if (not in_trade and ml_prediction['will_hit'] in [2,3,0] and ml_prediction['confidence_score']*100 >= ml_confidence_threshold):
        entry_price = current_data['Close'].iloc[-1]
        TP_price = entry_price * (1 + TP_pct/100)
        SL_price = entry_price * (1 - SL_pct/100)
        current_trade = {
            'entry_date': current_date,
            'entry_price': entry_price,
            'tp_price': TP_price,
            'sl_price': SL_price,
            'ml_confidence': ml_prediction['confidence_score']*100,
            'ml_signal': ml_prediction['will_hit']
        }
        in_trade = True

    # Exit
    elif in_trade:
        current_close = current_data['Close'].iloc[-1]
        if current_close >= current_trade['tp_price']:
            exit_reason = 'TP'
            exit_price = current_trade['tp_price']
        elif current_close <= current_trade['sl_price']:
            exit_reason = 'SL'
            exit_price = current_trade['sl_price']
        elif (current_date - current_trade['entry_date']).days >= max_holding_days:
            exit_reason = 'Max_Hold'
            exit_price = current_close
        else:
            continue
        trades.append({
            'EntryDate': current_trade['entry_date'],
            'ExitDate': current_date,
            'EntryPrice': current_trade['entry_price'],
            'ExitPrice': exit_price,
            'Outcome': exit_reason,
            'Return_%': (exit_price/current_trade['entry_price'] - 1)*100,
            'HoldingDays': (current_date - current_trade['entry_date']).days,
            'ML_Confidence': current_trade['ml_confidence'],
            'ML_Signal': current_trade['ml_signal']
        })
        in_trade = False
        current_trade = {}

# -------------------------
# Results
# -------------------------
results = pd.DataFrame(trades)
if results.empty:
    st.warning("No trades executed.")
else:
    st.subheader("📊 Backtest Results")
    st.dataframe(results)
    st.metric("Total Trades", len(results))
    st.metric("Win Rate", f"{(results['Return_%']>0).mean()*100:.1f}%")
    st.metric("Net Return", f"{results['Return_%'].sum():.1f}%")
