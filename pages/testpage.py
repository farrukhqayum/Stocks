from imports import *
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import math
warnings.filterwarnings('ignore')

# Global Parameters - Adjusted for different timeframes
YEARS_OF_DATA = {
    '4H': 1,    
    '1D': 2,    
    '1W': 8
}

MIN_TRAIN_ROWS = {
    '4H': 50,
    '1D': 30, 
    '1W': 10   # Weekly needs fewer data points
}

DEFAULT_TICKER = "TSLA" 
PROFIT_TARGET = 0.04
STOP_LOSS = 0.03755
_DAYS = 21
_Nr = 10  # Reduced minimum data requirement
windows = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21] # For calculating returns

# Simplified features for faster processing
FEATURES = [
    # Price High, Low
    'High', 'Low',
    
    # Technical Indicators
    'RSI', 'RSI_SMA', 'CCI', '+DI', '-DI', 'ADX', 'ATR', 'VI+', 'KCu', 'KCl', 'Kasym', 'Kcount', 'STu', 'STl',

    # Moving Averages & Bands
    'EMA1', 'EMA2', 'EMA3', 'EMA_Ratio', 'Upper_Band', 'Lower_Band', 'Volume_MA20', 'SMIIO', 'SMIIO_Signal', 'SMIIO_Osc', 'MACD', 'Signal_Line',

    # Returns & Volatility
    'return1', 'return2', 'return3', 'Volatility', 'Scaled_Volatility', 'DD',

    # Volume Features
    'sumBuyVol', 'sumSellVol', 'vSpike', 'VPT', 'OBV', 'MFI', 'VWMA', 'CMF',

    # Candlestick Patterns
    'Candlesticks', 'gapStrength',

    # Market Sentiment & Signals
    'Bull', 'Bear', 'Short', 'Hold', 'Neutral', 'StrongBull', 'StrongBear', 'Exhaustion',

    # PIVOTS
    'PP_Avg', 'R1_Avg', 'R2_Avg', 'S1_Avg', 'S2_Avg'
]

label2str = {0: 'None', 1: 'SL', 2: 'TP', 3: 'Hold', 4: 'Short'}
expected_classes = [0, 1, 2, 3, 4]

def validate_ticker(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if hist.empty:
            return {"valid": False, "reason": "No price history"}
        return {"valid": True, "reason": "Ticker found"}
    except Exception as e:
        return {"valid": False, "reason": str(e)}
        
@st.cache_data(ttl=1200)
def get_stock_data(ticker, start_date, end_date, interval='1d'):
    """Get stock data for given timeframe with proper date handling"""
    try:
        # Map interval names for yfinance
        interval_map = {
            '4H': '4H',
            '1D': '1d', 
            '1W': '1wk'
        }
        
        yf_interval = interval_map.get(interval, interval)
        
        df = yf.download(ticker, start=start_date, end=end_date,
                        interval=yf_interval, progress=False, auto_adjust=True)
        
        if df.empty:
            st.error(f"No data found for {ticker} with interval {interval}")
            return None
        
        # Reset index to get Date as column
        df = df.reset_index()
        
        # Handle different column names from yfinance
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        elif 'Datetime' in df.columns:
            df['Date'] = pd.to_datetime(df['Datetime'])
            df.set_index('Date', inplace=True)
            df = df.drop('Datetime', axis=1)
        
        # Ensure we have the required columns
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Missing required column: {col}")
                return None
        
        # Clean data
        df = df[required_cols].dropna()
        
        if df.empty:
            st.error(f"No valid data after cleaning for {ticker}")
            return None
            
        return df
        
    except Exception as e:
        st.error(f"Error downloading data for {ticker}: {str(e)}")
        return None

def add_technical_indicators(df, timeframe='1D'):
    """Add essential technical indicators to dataframe with timeframe-specific adjustments"""
    try:
        close = df.Close
        df['Close'] = df[['Open', 'High', 'Low', 'Close']].mean(axis=1).rolling(3).mean()
        
        # Adjust parameters based on timeframe
        if timeframe == '1W':
            # Longer periods for weekly data
            sma_multiplier = 2
            atr_period = 5
            rsi_period = 9
            windows = [3, 5, 7, 9, 11]
            df = df.ffill().bfill()
            
        elif timeframe == '4H':
            sma_multiplier = 5  # Longer SMAs for weekly
            atr_period = 50
            rsi_period = 50
            windows = [10, 13, 15, 17, 19, 21]
        else:
            # Default periods for hourly/daily
            sma_multiplier = 3
            atr_period = 14  
            rsi_period = 14
        df['EMA1'] = df['Close'].ewm(span=int(_DAYS * 0.5 * sma_multiplier), adjust=False).mean()
        df['EMA2'] = df['Close'].ewm(span=_DAYS * sma_multiplier, adjust=False).mean()
        df['EMA3'] = df['Close'].ewm(span=int(_DAYS * 2 * sma_multiplier), adjust=False).mean()

        df['EMA_Ratio'] = df['EMA1'] / df['EMA2']
        df['ATR'] = ta.calculate_atr(high=df.High, low=df.Low, close=df.Close)
        df = ta.scaled_volatility(df)
        df = ta.add_candlestickpatterns(df)
        df['RSI']= ta.calculate_rsi(df)
        df['RSI_SMA'] = df['RSI'].rolling(14).mean()
        
        # Adjust MACD periods for weekly
        if timeframe == '1W':
            ema_short = 9
            ema_long = 22
        else:
            ema_short = 12
            ema_long = 26
            
        ema12 = df['Close'].ewm(span=ema_short, adjust=False).mean()
        ema26 = df['Close'].ewm(span=ema_long, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        df['SMIIO'], df['SMIIO_Signal'], df['SMIIO_Osc'] = ta.calculate_smiio(df)
        df['Upper_Band'] = df['EMA1'] + (2 * df['Close'].rolling(20).std())
        df['Lower_Band'] = df['EMA1'] - (2 * df['Close'].rolling(20).std())
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        df['buy_volume'] = (df.Close > df.Close.shift(1)) * df['Volume']
        df['sell_volume'] = (df.Close < df.Close.shift(1)) * df['Volume']
        df['sumBuyVol'] = df['buy_volume'].rolling(window=9).sum()
        df['sumSellVol'] = df['sell_volume'].rolling(window=9).sum()
        df['vSpike'] = np.where(df['Volume'] > 2 * df['Volume_MA20'], np.where(df['Close'] > df['Open'], 1, -1), 0)
        df['VPT'] = df['Volume'].mul((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)).cumsum()
        df['MFI'] = ta.calculate_mfi(df)
        df['CMF'] = ta.chaikin_money_flow(df, window=20)
        df['CCI'] = ta.calculate_cci(df)
        df['OBV'] = ta.calculate_obv(df)
        df[['+DI', '-DI', 'ADX']] = ta.calculate_dmi(df, n=14).rolling(3).mean()
        df['VWMA'] = ta.calculate_vwma(df)
        df[['KCm', 'KCu', 'KCl', 'KCu_outer','KCl_outer', 'Kasym', 'Kcount']] = ta.calculate_keltner(df).rolling(3).mean()
        df[['VI+', 'VI-']] = ta.calculate_vortex(df)
        df[['STu', 'STl']] = ta.calculate_supertrend(df)
        df['DD'] = df['Close'].rolling(14).apply(lambda x: x[-1] - x.max())

        # Adjust return periods based on timeframe
        if timeframe == '1W':
            df['return1'] = df['Close'].pct_change(4).rolling(2).mean()   # 1 month
            df['return2'] = df['Close'].pct_change(13).rolling(2).mean()  # 3 months
            df['return3'] = df['Close'].pct_change(26).rolling(2).mean()  # 6 months
        else:
            df['return1'] = df['Close'].pct_change(7).rolling(3).mean()
            df['return2'] = df['Close'].pct_change(14).rolling(3).mean()
            df['return3'] = df['Close'].pct_change(21).rolling(3).mean()
            
        df['Volatility'] = df['Close'].rolling(14).std().rolling(3).mean()
        
        # fill nans
        cols = ['EMA1', 'EMA2', 'RSI', '-DI', 'Close']
        df[cols] = df[cols].fillna(method='ffill').fillna(method='bfill')
        
        # Adjust RSI thresholds for weekly if needed
        rsi_lower = 25 if timeframe == '1W' else 18
        rsi_upper = 60 if timeframe == '1W' else 55

        conditions = [
        # 1️⃣ HOLD FIRST (Extended Rally - HIGHEST priority)
            (
                (df['Close'] > df['EMA2']) &
                (df['EMA1'] > df['EMA2']) &
                (df['RSI'].between(50, 90)) &
                (df['ADX'] > 40) &
                (df['+DI'] > df['-DI']) &
                (df['Close'] > df['Close'].shift(5) * 1.02)  # ✅ Rally proof!
            ),
            
            # 2️⃣ BULL (Entry signals)
            (
                (
                    ((df['EMA1'] >= df['EMA2']) &
                      (df['RSI'] >= df['RSI_SMA']) &
                      (df['RSI'].between(52, 95)) &
                      ((df['ADX'] > 24) & (df['+DI'] > df['-DI'])))
                    |
                    (
                        ((df['RSI'] >= df['RSI_SMA']) & (df['RSI'] > 50)) & 
                        ((df['ADX'] > 18) & (df['+DI'] > df['-DI']))
                    )
                )
            ),
            
            # 3️⃣ SHORT (Aggressive shorts)
            (
                ((df['Close'] <= df['EMA1']) &
                 (df['EMA1'] < df['EMA2']) &
                 (df['RSI'].between(50, 85)) &
                 (df['ADX'] > 24) & 
                 (df['+DI'] < df['-DI']))
            ),
            
            # 4️⃣ BEAR (Bearish entries - LOWEST priority)
            (
                (
                    ((df['EMA1'] < df['EMA2']) &
                      (df['RSI'].between(18,60)) &
                      (df['RSI'] < df['RSI_SMA']) &
                      ((df['ADX'] > 18) & (df['+DI'] < df['-DI'])))
                    |
                    (
                        ((df['RSI'] < df['RSI_SMA']) & 
                         (df['RSI'].between(20, 60)) &
                         ((df['ADX'] > 18) & (df['+DI'] < df['-DI'])))
                    )
                    |
                    (
                        ((df['RSI'] > df['RSI_SMA']) & 
                         (df['RSI_SMA'] < 37))
                    )
                )
            )
        ]
        
        choices = ['Hold', 'Bull', 'Short', 'Bear']
        df['TI'] = np.select(conditions, choices, default='Neutral')
        df['TI'] = df['TI'].astype('category')
        df_encoded = pd.get_dummies(df['TI'], prefix='', prefix_sep='')
        expected_cols = ['Bull', 'Bear', 'Short', 'Hold', 'Neutral']
        for col in expected_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df= pd.concat([df, df_encoded], axis=1)
    
        strongbull_condition = ((df['RSI'] > 52) & (df['ADX'] > 22) & (df['+DI'] > df['-DI']) & (df['sumBuyVol'] > df['sumSellVol']))
        strongbear_condition = ((df['RSI'] < 40) & (df['ADX'] > 22) & (df['+DI'] < df['-DI']) & (df['sumBuyVol'] < df['sumSellVol']))
        df['StrongBull'] = strongbull_condition.astype(int)
        df['StrongBear'] = strongbear_condition.astype(int)
        df['sNeutral'] = ((df['StrongBull'] == 0) & (df['StrongBear'] == 0)).astype(int)
        df['gapStrength'] = ta.compute_gapStrength(df)
        df = ta.add_exhaustion_indicator(df)
        df.Close = close
        return df
        
    except Exception as e:
        st.error(f"Error adding technical indicators: {str(e)}")
        return None

df = get_stock_data("Coin", "01-01-2023", "01-01-2025", "1d")
