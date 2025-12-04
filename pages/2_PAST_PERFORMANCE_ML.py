#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ML - Stock Past Performance", layout="wide")

# ===============================
# STRATEGY SETTINGS
# ===============================
YEARS_OF_DATA = 3
PROFIT_TARGET = 0.0375
STOP_LOSS = 0.0375
_DAYS = 22
windows = [5,10,14,20]

FEATURES = [
    'High','Low','RSI','RSI_SMA','ATR','+DI','-DI','ADX',
    'SMA10','SMA20','SMA50',
    'return1','return2','return3','Volatility',
    'sumBuyVol','sumSellVol',
    'Bull','Bear','Short','Hold','Neutral',
    'PP_Avg','R1_Avg','S1_Avg','R2_Avg','S2_Avg'
]

# ===============================
# USER INPUT
# ===============================
col1,col2,col3,col4 = st.columns(4)
with col1:
    ticker = st.text_input("Ticker", value="COIN")
with col2:
    period = st.selectbox("History Period", ["1y","2y","3y","5y","7y"], index=2)
with col3:
    TP_pct = st.number_input("TP %", value=7.0)
with col4:
    SL_pct = st.number_input("SL %", value=14.0)

col5, col6 = st.columns(2)
with col5:
    ml_confidence_threshold = st.number_input("ML Confidence Threshold", min_value=0,max_value=100,value=63)
with col6:
    max_holding_days = st.number_input("Max Holding Days", min_value=3,max_value=60,value=15)

# ===============================
# DOWNLOAD DATA
# ===============================
def get_stock_data(ticker, period):
    df = yf.download(ticker, period=period, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.dropna(inplace=True)
    return df

# ===============================
# INDICATORS
# ===============================
def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100/(1+rs))

def calculate_atr(df, period=14):
    hl = df['High'] - df['Low']
    hc = abs(df['High'] - df['Close'].shift())
    lc = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([hl,hc,lc],axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calculate_dmi_adx(df, period=14):
    up = df['High'].diff()
    down = -df['Low'].diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)

    tr = np.maximum(
        df['High']-df['Low'],
        abs(df['High']-df['Close'].shift()),
        abs(df['Low']-df['Close'].shift())
    )

    atr = pd.Series(tr).rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()

    return plus_di, minus_di, adx

# ===============================
# FEATURE ENGINEERING
# ===============================
def add_features(df):
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()

    df['RSI'] = calculate_rsi(df)
    df['RSI_SMA'] = df['RSI'].rolling(14).mean()
    df['ATR'] = calculate_atr(df)

    df['+DI'], df['-DI'], df['ADX'] = calculate_dmi_adx(df)

    df['buy_volume'] = (df['Close'] > df['Close'].shift(1))*df['Volume']
    df['sell_volume'] = (df['Close'] < df['Close'].shift(1))*df['Volume']
    df['sumBuyVol'] = df['buy_volume'].rolling(9).sum()
    df['sumSellVol'] = df['sell_volume'].rolling(9).sum()

    df['return1'] = df['Close'].pct_change(7)
    df['return2'] = df['Close'].pct_change(14)
    df['return3'] = df['Close'].pct_change(21)
    df['Volatility'] = df['Close'].rolling(14).std()

    conditions = [
        (df['SMA10'] > df['SMA50']) & (df['RSI'] > 55),
        (df['SMA10'] < df['SMA50']) & (df['RSI'] < 45),
    ]
    choices = ['Bull','Bear']
    df['TI'] = np.select(conditions, choices, default='Neutral')

    df['Bull'] = (df['TI']=='Bull').astype(int)
    df['Bear'] = (df['TI']=='Bear').astype(int)
    df['Hold'] = ((df['Close'] > df['SMA50']) & (df['RSI']>50)).astype(int)
    df['Short'] = ((df['Close'] < df['SMA50']) & (df['RSI']<50)).astype(int)
    df['Neutral'] = (df['TI']=='Neutral').astype(int)

    return df

# ===============================
# PIVOTS
# ===============================
def add_pivots(df):
    for w in windows:
        high = df['High'].rolling(w).max()
        low = df['Low'].rolling(w).min()
        close = df['Close']

        PP = (high + low + close) / 3
        R1 = 2*PP - low
        S1 = 2*PP - high
        R2 = PP + (high - low)
        S2 = PP - (high - low)

        df[f'PP_{w}']=PP
        df[f'R1_{w}']=R1
        df[f'S1_{w}']=S1
        df[f'R2_{w}']=R2
        df[f'S2_{w}']=S2

    df['PP_Avg'] = df[[f'PP_{w}' for w in windows]].mean(axis=1)
    df['R1_Avg'] = df[[f'R1_{w}' for w in windows]].mean(axis=1)
    df['S1_Avg'] = df[[f'S1_{w}' for w in windows]].mean(axis=1)
    df['R2_Avg'] = df[[f'R2_{w}' for w in windows]].mean(axis=1)
    df['S2_Avg'] = df[[f'S2_{w}' for w in windows]].mean(axis=1)

    return df

# ===============================
# LABELING
# ===============================
def label_data(df, tp=0.05, sl=0.05, window=14):
    labels=[]
    closes = df['Close'].values

    for i in range(len(df)):
        entry = closes[i]
        target = entry*(1+tp)
        stop = entry*(1-sl)

        future = closes[i+1:i+1+window]
        if len(future)==0:
            labels.append(0)
            continue

        tp_hit = np.any(future >= target)
        sl_hit = np.any(future <= stop)

        if tp_hit and not sl_hit:
            labels.append(2)
        elif sl_hit and not tp_hit:
            labels.append(1)
        elif df['Hold'].iloc[i]==1:
            labels.append(3)
        elif df['Short'].iloc[i]==1:
            labels.append(4)
        else:
            labels.append(0)

    df['Hit_Label']=labels
    return df

# ===============================
# TRAIN
# ===============================
def train_models(df):
    features=[f for f in FEATURES if f in df.columns]
    df2=df.dropna(subset=features+['Hit_Label'])

    X=df2[features]
    y=df2['Hit_Label']

    scaler=StandardScaler()
    Xs=scaler.fit_transform(X)

    model=RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42)

    model.fit(Xs,y)
    return model,scaler,features

# ===============================
# PREDICT
# ===============================
label_map = {0:'None',1:'SL',2:'TP',3:'Hold',4:'Short'}

def get_prediction(df,model,scaler,features):
    last=df[features].iloc[[-1]]
    if last.isnull().any().any():
        return None
    Xs=scaler.transform(last)
    probs=model.predict_proba(Xs)[0]
    pred=model.predict(Xs)[0]
    return label_map[pred],max(probs)*100


# ===============================
# RUN
# ===============================
if st.button("Run ML Backtest"):

    df = get_stock_data(ticker,period)
    df = add_features(df)
    df = add_pivots(df)
    df = label_data(df,TP_pct/100,SL_pct/100)

    model,scaler,features = train_models(df)
    pred,conf = get_prediction(df,model,scaler,features)

    st.subheader("Latest ML Prediction")
    if pred:
        st.metric("Prediction",pred)
        st.metric("Confidence %",round(conf,2))

    st.subheader("Chart")
    st.line_chart(df['Close'])

    st.subheader("Data Preview")
    st.dataframe(df.tail(30))
