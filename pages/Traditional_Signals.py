from imports import *
import streamlit as st
from curl_cffi import requests
import time
import re
import warnings
import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- GLOBAL PARAMETERS ---
_DAYS = 22 # Used for SMA and training


# --- APP TITLE ---
st.title('Traditional Signals')

# --- USER INPUT ---
ticker = st.text_input("Enter Ticker Symbol (e.g., TSLA, CRM, COIN):").upper()
years = st.slider('Number of years of data:', min_value=1, max_value=5, value=1)

# --- FUNCTION TO GET DATA ---
def get_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1), interval='1d', auto_adjust=False, progress=False)
    if df.empty:
        return None
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.dropna()
    return df

# --- TECHNICAL INDICATORS ---
def add_technical_indicators(df):
    df['SMA1'] = df['Close'].ewm(span=int(_DAYS * 0.5), adjust=False).mean()
    df['SMA2'] = df['Close'].ewm(span=_DAYS, adjust=False).mean()
        
    df['ATR'] = ta.calculate_atr(high=df.High, low=df.Low, close=df.Close)
    df['RSI']= ta.calculate_rsi(df)
    df['RSI_SMA'] = df['RSI'].rolling(14).mean()
    df[['+DI', '-DI', 'ADX']] = ta.calculate_dmi(df, n=14)

    # Signal assignments
    conditions = [
        (
            (
                (df['SMA1'] > df['SMA2']) & 
                (df['RSI'] >= df['RSI_SMA']) & 
                (df['RSI'] >= 52) & 
                (df['+DI'] > df['-DI'])
            ) | 
            (
                (df['Close'] > df['SMA1']) & 
                (df['RSI'] > df['RSI_SMA'])
            )
        ),
        (
            (df['SMA1'] <= df['SMA2']) &
            (df['RSI'] < df['RSI_SMA']) &
            (df['RSI'] <= 42) &
            (df['+DI'] < df['-DI'])
        )  | 
            (
                (df['Close'] < df['SMA1']) & 
                (df['RSI'] < df['RSI_SMA'])
            ),
        (
            (df['SMA1'] <= df['SMA2']) &
            (df['RSI'].between(40, 60)) &
            (df['-DI'] > df['+DI']) &
            (df['Close'] < df['SMA1'])
        ),
        (
            (df['Close'] > df['SMA2']) &
            (df['RSI'] < df['RSI_SMA']) &
            (df['RSI'] >= 50)
        )
    ]
    choices = ['Bull', 'Bear', 'Short', 'Hold']
    
    df['Signal'] = np.select(conditions, choices, default='Neutral')
    df['TI'] = np.select(conditions, choices, default='Neutral')
    
    df['TI'] = df['TI'].astype('category')
    df_encoded = pd.get_dummies(df['TI'], prefix='', prefix_sep='')
    df= pd.concat([df, df_encoded], axis=1)

    df['gapStrength'] = ta.compute_gapStrength(df)
    df = ta.add_exhaustion_indicator(df)
    
    return df

# --- PLOTTING FUNCTION ---
def plot_price_with_signals(df):
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(12, 6), dpi=150, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    ax.plot(df.index, df['Close'], label='Close Price', alpha=0.7, color='gray')
    ax.plot(df.index, df['SMA1'], label='SMA1', color='gold', alpha=0.6)
    ax.plot(df.index, df['SMA2'], label='SMA2', color='red', alpha=0.5, linestyle='--')

    # Bull
    bull_points = df[df['Signal'] == 'Bull']
    ax.scatter(bull_points.index, bull_points['Close'], color='green', marker='^', s=25, alpha=0.85, label='Bull')
    # Bear
    bear_points = df[df['Signal'] == 'Bear']
    ax.scatter(bear_points.index, bear_points['Close'], color='red', marker='v', s=25, alpha=0.85, label='Bear')
    # Short
    short_points = df[df['Signal'] == 'Short']
    ax.scatter(short_points.index, short_points['Close'], color='purple', marker='s', s=18, alpha=0.7, label='Short')
    # Hold
    hold_points = df[df['Signal'] == 'Hold']
    ax.scatter(hold_points.index, hold_points['Close'], color='orange', marker='o', s=18, alpha=0.7, label='Hold')
    # Neutral
    neutral_points = df[df['Signal'] == 'Neutral']
    ax.scatter(neutral_points.index, neutral_points['Close'], color='gray', marker='.', s=6, alpha=0.3, label='Neutral')

    ax.set_title('Traditional Signals')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(True, alpha=0.4)

    bx.plot(df.index, df['RSI'], label='RSI', color='blue', linewidth=1.2)
    bx.plot(df.index, df['RSI_SMA'], label='RSI SMA', color='gold', linewidth=1.2, linestyle='--')
    bx.axhline(52, color='gray', linewidth=1.0, linestyle='--', alpha=0.5)
    bx.axhline(40, color='brown', linewidth=1.0, linestyle=':', alpha=0.5)
    bx.set_ylim(0, 100)
    bx.set_ylabel('RSI')
    bx.legend(loc='upper left', fontsize='small')
    bx.grid(True, alpha=0.4)

    plt.tight_layout()
    return fig

# --- MAIN WORKFLOW ---
if ticker:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)

    df = get_stock_data(ticker, start_date, end_date)
    if df is not None and not df.empty:
        df = add_technical_indicators(df)
        st.dataframe(df.tail(15), use_container_width=True)
        fig = plot_price_with_signals(df)
        st.pyplot(fig)
    else:
        st.warning(f"No data found for '{ticker}' in the selected period.")
else:
    st.info("Enter a ticker symbol and choose years, then analysis runs automatically.")

