import streamlit as st
from imports import *  # Assuming 'ta' and others are in this import
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- GLOBAL PARAMETERS ---
_DAYS = 22

st.title('Traditional Signals')

# --- USER INPUT ---
ticker = st.text_input("Enter Ticker Symbol (e.g., TSLA, CRM, COIN):").upper()
years = st.slider('Number of years of data:', 1, 5, 1)

# --- DATA FETCH ---
def get_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1), interval='1d', auto_adjust=False, progress=False)
    if df.empty:
        return None
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.dropna()
    return df

# --- CALCULATE RSI ---
def calculate_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_series = pd.Series(gain, index=df.index)
    loss_series = pd.Series(loss, index=df.index)
    avg_gain = gain_series.rolling(window=window).mean()
    avg_loss = loss_series.rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- ADD TECHNICAL INDICATORS ---
def add_technical_indicators(df):
    df['SMA1'] = df['Close'].ewm(span=int(_DAYS * 0.5), adjust=False).mean()
    df['SMA2'] = df['Close'].ewm(span=_DAYS, adjust=False).mean()

    # RSI
    df['RSI'] = calculate_rsi(df)
    df['RSI_SMA'] = df['RSI'].rolling(14).mean()

    # Plus and Minus DI using 'ta' if it has: ta.calculate_dmi()
    # Assuming these are calculated here:
    df[['+DI', '-DI', 'ADX']] = ta.calculate_dmi(df, n=14)

    # Conditions for signals
    conditions = [
        # Bullish condition
        (
            (
                (df['SMA1'] > df['SMA2']) & 
                (df['RSI'] >= df['RSI_SMA']) & 
                (df['RSI'] >= 52) & 
                (df['+DI'] > df['-DI'])
            ) | (
                (df['Close'] > df['SMA1']) & 
                (df['RSI'] > df['RSI_SMA'])
            )
        ),
        # Bearish condition
        (
            (df['SMA1'] <= df['SMA2']) & 
            (df['RSI'] < df['RSI_SMA']) & 
            (df['RSI'] <= 42) & 
            (df['+DI'] < df['-DI'])
        ) | (
            (df['Close'] < df['SMA1']) & 
            (df['RSI'] < df['RSI_SMA'])
        ),
        # Neutral or sideways
        (
            (df['SMA1'] <= df['SMA2']) & 
            (df['RSI'].between(40, 60)) & 
            (df['-DI'] > df['+DI']) & 
            (df['Close'] < df['SMA1'])
        ),
        # Uptrend weak or bullish
        (
            (df['Close'] > df['SMA2']) & 
            (df['RSI'] < df['RSI_SMA']) & 
            (df['RSI'] >= 50)
        )
    ]
    choices = ['Bull', 'Bear', 'Short', 'Hold']
    df['Signal'] = np.select(conditions, choices, default='Neutral')
    return df

# --- PLOTTING FUNCTION ---
def plot_signals(df):
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(12, 6), dpi=150, sharex=True)
    # Price plot with signals
    ax.plot(df.index, df['Close'], label='Close', color='gray', alpha=0.7)
    ax.plot(df.index, df['SMA1'], label='SMA1', color='gold')
    ax.plot(df.index, df['SMA2'], label='SMA2', color='red', linestyle='--')

    # Plot signals
    ax.scatter(df[df['Signal']=='Bull'].index, df[df['Signal']=='Bull']['Close'], color='green', marker='^', s=50, label='Bull')
    ax.scatter(df[df['Signal']=='Bear'].index, df[df['Signal']=='Bear']['Close'], color='red', marker='v', s=50, label='Bear')
    ax.scatter(df[df['Signal']=='Short'].index, df[df['Signal']=='Short']['Close'], color='purple', marker='s', s=40, label='Short')
    ax.scatter(df[df['Signal']=='Hold'].index, df[df['Signal']=='Hold']['Close'], color='orange', marker='o', s=40, label='Hold')
    ax.set_title('Traditional Signals')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left', fontsize='small')
    ax.grid()

    # RSI Plot
    bx.plot(df.index, df['RSI'], label='RSI', color='blue')
    bx.plot(df.index, df['RSI_SMA'], label='RSI SMA', color='gold', linestyle='--')
    bx.axhline(52, color='gray', linestyle='--', alpha=0.5)
    bx.axhline(40, color='brown', linestyle=':', alpha=0.5)
    bx.set_ylim(0, 100)
    bx.set_ylabel('RSI')
    bx.legend()
    bx.grid()

    plt.tight_layout()
    return fig

# --- MAIN ---
if __name__ == '__main__':
    if ticker:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * min(5, years))
        df = get_stock_data(ticker, start_date, end_date)
        if df is not None and not df.empty:
            df = add_technical_indicators(df)
            st.dataframe(df.tail(15))
            fig = plot_signals(df)
            st.pyplot(fig)
        else:
            st.warning('No data fetched for this ticker and period.')
    else:
        st.info('Please enter a ticker symbol.')
