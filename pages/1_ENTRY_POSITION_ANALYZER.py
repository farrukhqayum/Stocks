#!/usr/bin/env python
# coding: utf-8
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


# Configuration
st.cache_data.clear()
st.cache_resource.clear()
st.set_page_config(page_title="Entry Position Analyzer", layout="wide")
st.caption("Data sourced via Yahoo Finance • Updated dynamically")

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
    'Bear', 'Bull', 'Short', 'Hold', 'Neutral', 'StrongBull', 'StrongBear', 'Exhaustion',

    # PIVOTS
    'PP_Avg', 'R1_Avg', 'R2_Avg', 'S1_Avg', 'S2_Avg'
]

label2str = {0: 'None', 1: 'SL', 2: 'TP', 3: 'Hold', 4: 'Short'}
expected_classes = [0, 1, 2, 3, 4]

desc = """
    ### Mindset
    - Not everyone can succeed.
    - Given you think like, 'I told you so', 'Yes, I was right', you aren't yet fit for this.
    - Warren Buffet, 'Learn how to hold -50% loss' and buy more. If not, then again you don't fit for stocks.
    - Manage positions, DCA 2 or three times.
    - Get stuck or you consumed all the capital, relax and stay away for months or years.
    - Return when time passes or you got more capital.
    - It is extremely difficult to make livings with this but you can buy financial freedom in 3-5 years.
    
    ### Entry Conditions
    - Trade only strong and volatile stocks. This isn't for futuristic ideas.   
    - Be patient, entry should be when price recovers above moving averages.
    - Split into two or three buys as no one can predict with precision if the entries are right.
    - Daily RSI is above its smoothed average (RSI_SMA), indicating a bullish momentum
    - The machine learning model predicts a strong bullish movement ('TP' or 'Hold', or 'None - directionless') but the confidence is above 60% with a decent risk/reward ratio predicted by ML.
    - Take Profit (TP) price should be dynamic to the greater of your target or ML predicted return. Accept 3-7% gains all the times.
    - Stop Loss (SL) price should be far away from entries such that you h it less SL. To circumvent stop-loss you need to preserve capital to buy more if it dips 10-15%.

    ### Exit Conditions
    - Exit an acceptable loss or if you entered when the market was bearish (Close  is below Moving Averages), much like over-trading or revenge trading.
    - TP when you see 3-7% gains or higher, don't wait too long that it will go higher.

    ### Additional Notes
    - This approach balances preset risk management with ML-model-driven adaptiveness
    - The system continuously evaluates trade signals, adjusting entries and exits based on market dynamics and model confidence
    - Risk-reward ratio and confidence scores help assess and validate each trade decision
    """

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
        df['DD'] = df['Close'].where(df['Close'] < df['Close'].shift(1)).std()
        
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
        # BULL
            (
                (
                    (df['EMA1'] > df['EMA2']) &
                    (df['RSI'] >= df['RSI_SMA']) &
                    (df['RSI'].between(52, 95))
                )
                |
                (
                    (df['RSI'] >= df['RSI_SMA']) & 
                    (df['RSI'] > 50)
                )
            ),
            # BEAR
            (
                (df['EMA1'] < df['EMA2']) &
                (df['RSI'].between(18,60)) &
                (df['RSI'] < df['RSI_SMA'])
                |
                (
                    (df['RSI'] < df['RSI_SMA']) & 
                    (df['RSI'].between(20, 60))
                )
            ),
            # SHORT
            (
                (df['Close'] <= df['EMA1']) &
                (df['EMA1'] < df['EMA2']) &
                (df['RSI'].between(50, 85))
            ),
            # HOLD
            (
                (df['Close'] > df['EMA2']) &
                (df['EMA1'] > df['EMA2']) &
                (df['RSI'].between(50, 90))
            )
        ]
        choices = ['Bull', 'Bear', 'Short', 'Hold']
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

def add_pivot_levels(df, window=_DAYS):
    # Compute rolling high/low/close over the window
    high = df['High'].rolling(window)
    low = df['Low'].rolling(window)
    close = df['Close'].rolling(window)
    # Classic floor trader pivots (you can adjust formulas as needed)
    PP = (high.max() + low.min() + close.apply(lambda x: x[-1])).div(3)
    R1 = 2 * PP - low.min()
    S1 = 2 * PP - high.max()
    R2 = PP + (high.max() - low.min())
    S2 = PP - (high.max() - low.min())
    # Assign to DataFrame
    df['PP'] = PP.fillna(method='bfill')
    df['R1'] = R1.fillna(method='bfill')
    df['S1'] = S1.fillna(method='bfill')
    df['R2'] = R2.fillna(method='bfill')
    df['S2'] = S2.fillna(method='bfill')
    return df

def add_pivots(df, win=windows):
    for w in win:
        roll_high = df['High'].rolling(w)
        roll_low = df['Low'].rolling(w)
        roll_close = df['Close'].rolling(w)
        # Calculate rolling pivots
        PP = (roll_high.max() + roll_low.min() + roll_close.apply(lambda x: x[-1])).div(3)
        R1 = 2 * PP - roll_low.min()
        S1 = 2 * PP - roll_high.max()
        R2 = PP + (roll_high.max() - roll_low.min())
        S2 = PP - (roll_high.max() - roll_low.min())
        # Store in DataFrame
        df[f'PP_{w}'] = PP
        df[f'R1_{w}'] = R1
        df[f'S1_{w}'] = S1
        df[f'R2_{w}'] = R2
        df[f'S2_{w}'] = S2
    return df

def average_pivots(df, windows=[5, 10, 14, 20]):
    for level in ['PP', 'R1', 'S1', 'R2', 'S2']:
        cols = [f'{level}_{w}' for w in windows]
        # Take row-wise mean, ignore NaN for early rows
        df[f'{level}_Avg'] = df[cols].mean(axis=1)
    return df
    
def compute_expected_return(df, forward_window=14, r_cols=['R1', 'R2']):
    """
    Compute expected returns with breakout confirmation to avoid false early TP hits.
    """
    df['Expected_Return'] = np.nan
    close_prices = df['Close'].values
    
    confirm_candles = 2  # require at least 2 candles above pivot before confirming hit
    
    pivot_arrays = []
    for col in r_cols:
        if col in df.columns:
            pivot_arrays.append(df[col].values)
        else:
            pivot_arrays.append(np.full(len(df), np.nan))
    
    for i in range(len(df) - forward_window):
        current_price = close_prices[i]
        pivots = [arr[i] for arr in pivot_arrays if not np.isnan(arr[i])]
        target_level = float(max(pivots)) if pivots else None
        future_window = close_prices[i+1:i+1+forward_window]
        
        if target_level is not None and len(future_window) >= confirm_candles:
            hit = False
            for j in range(len(future_window) - confirm_candles + 1):
                segment = future_window[j:j+confirm_candles]
                # confirm that price stays above pivot for confirm_candles in a row
                if np.all(segment >= target_level):
                    df.iloc[i, df.columns.get_loc('Expected_Return')] = (target_level - current_price) / current_price
                    hit = True
                    break
            if not hit:
                df.iloc[i, df.columns.get_loc('Expected_Return')] = (np.nanmax(future_window) - current_price) / current_price
        else:
            if future_window.size > 0:
                df.iloc[i, df.columns.get_loc('Expected_Return')] = (np.nanmax(future_window) - current_price) / current_price
            else:
                df.iloc[i, df.columns.get_loc('Expected_Return')] = np.nan
                
    return df

def compute_expected_loss(df, forward_window=14, s_cols=['S1', 'S2']):
    """
    Compute expected losses with sustained breakdown confirmation to avoid early SL triggers.
    """
    df['Expected_Loss'] = np.nan
    close_prices = df['Close'].values
    
    confirm_candles = 2  # require at least 2 candles below pivot before confirming hit
    
    pivot_arrays = []
    for col in s_cols:
        if col in df.columns:
            pivot_arrays.append(df[col].values)
        else:
            pivot_arrays.append(np.full(len(df), np.nan))
    
    for i in range(len(df) - forward_window):
        current_price = close_prices[i]
        pivots = [arr[i] for arr in pivot_arrays if not np.isnan(arr[i])]
        target_level = float(min(pivots)) if pivots else None
        future_window = close_prices[i+1:i+1+forward_window]
        
        if target_level is not None and len(future_window) >= confirm_candles:
            hit = False
            for j in range(len(future_window) - confirm_candles + 1):
                segment = future_window[j:j+confirm_candles]
                # confirm that price stays below pivot for confirm_candles in a row
                if np.all(segment <= target_level):
                    df.iloc[i, df.columns.get_loc('Expected_Loss')] = (target_level - current_price) / current_price
                    hit = True
                    break
            if not hit:
                df.iloc[i, df.columns.get_loc('Expected_Loss')] = (np.nanmin(future_window) - current_price) / current_price
        else:
            if future_window.size > 0:
                df.iloc[i, df.columns.get_loc('Expected_Loss')] = (np.nanmin(future_window) - current_price) / current_price
            else:
                df.iloc[i, df.columns.get_loc('Expected_Loss')] = np.nan
                
    return df

def label_hit_prob_past(
    df,
    window=14,
    profit_target=0.05,
    stop_loss=0.05,
    lookback=60,
    tp_thresh=0.35,
    sl_thresh=0.35
):
    import numpy as np
    
    close_prices = df['Close'].values
    
    bull = (df['TI'] == 'Bull')
    bear = (df['TI'] == 'Bear')
    hold = (df['TI'] == 'Hold')
    short = (df['TI'] == 'Short')
    neutral = (df['TI'] == 'Neutral')

    EMA1 = df['EMA1'].values
    atr = df['ATR'].values
    rsi = df['RSI'].values
    adx = df['ADX'].values

    N = len(close_prices)
    labels = []
    
    for i in range(N):
        current_price = close_prices[i]
        tp = current_price * (1 + profit_target)
        sl = current_price * (1 - stop_loss)
        future_prices = close_prices[i + 1 : min(i + 1 + window, N)]
        tp_hit_idx = next((j for j, price in enumerate(future_prices) if price >= tp), None)
        sl_hit_idx = next((j for j, price in enumerate(future_prices) if price <= sl), None)
        
        lookback_start = max(0, i - lookback)
        history_tp, history_sl = [], []
        for j in range(lookback_start, i):
            hist_price = close_prices[j]
            hist_tp = hist_price * (1 + profit_target)
            hist_sl = hist_price * (1 - stop_loss)
            hist_future = close_prices[j + 1: j + 1 + window]
            
            if bull[j]:
                hist_tp_hit_idx = next((k for k, p in enumerate(hist_future) if p >= hist_tp), None)
                hist_sl_hit_idx = next((k for k, p in enumerate(hist_future) if p <= hist_sl), None)
                hit = hist_tp_hit_idx is not None and (hist_sl_hit_idx is None or hist_tp_hit_idx < hist_sl_hit_idx)
                history_tp.append(int(hit))
                
            if bear[j]:
                hist_tp_hit_idx = next((k for k, p in enumerate(hist_future) if p >= hist_tp), None)
                hist_sl_hit_idx = next((k for k, p in enumerate(hist_future) if p <= hist_sl), None)
                hit = hist_sl_hit_idx is not None and (hist_tp_hit_idx is None or hist_sl_hit_idx < hist_tp_hit_idx)
                history_sl.append(int(hit))
        
        tp_prob = np.mean(history_tp) if len(history_tp) >= 3 else min(np.mean(history_tp) if history_tp else 0.5, tp_thresh)
        sl_prob = np.mean(history_sl) if len(history_sl) >= 3 else min(np.mean(history_sl) if history_sl else 0.5, sl_thresh)
        
        # Initial label assignment priority: TP > SL > Hold > Short > Neutral
        if tp_hit_idx is not None and (sl_hit_idx is None or tp_hit_idx < sl_hit_idx) and bull[i] and tp_prob >= tp_thresh:
            labels.append(2)  # TP (bull)
        elif sl_hit_idx is not None and (tp_hit_idx is None or sl_hit_idx < tp_hit_idx) and bear[i] and sl_prob >= sl_thresh:
            labels.append(1)  # SL (bear)
        elif hold[i]:
            # Upgrade Hold to TP if breakout early within window
            if any(p >= tp for p in future_prices):
                labels.append(2)
            else:
                labels.append(3)
        elif short[i]:
            labels.append(4)
        else:
            if i >= N - window:
                if bull[i]:
                    labels.append(2)
                elif bear[i]:
                    labels.append(1)
                else:
                    labels.append(0)
            else:
                labels.append(0)
                
    for i in range(N):
        if labels[i] in [2, 3]:  # TP or Hold bars
            current_close = close_prices[i]
            EMA1_now = EMA1[i]
            atr_now = atr[i]
            rsi_now = rsi[i]
            adx_now = adx[i]

            future_end = min(i + 1 + window, N)
            future_closes = close_prices[i + 1 : future_end]
            future_EMA1 = EMA1[i + 1 : future_end]

            current_dip = current_close < EMA1_now or current_close < (EMA1_now - 0.5 * atr_now)
            future_dips = any((p < s) or (p < s - 0.5 * atr_now) for p, s in zip(future_closes, future_EMA1))

            bearish_momentum = (rsi_now < 40) and (adx_now > 22)
            fading_bullish = (rsi_now < 50) or (adx_now < 20)
            hold_extreme = (labels[i] == 3) and (rsi_now < 45)

            if (current_dip or future_dips) and (bearish_momentum or fading_bullish or hold_extreme):
                            if not ((rsi_now > 52) and (current_close > df['EMA2'].iloc[i])):
                                labels[i] = 1  # Trigger SL immediately
    
    df['Hit_Label'] = labels
    return df
                               
def handle_missing_data(df, required_cols, timeframe):
    """Handle missing data strategically instead of dropping all rows with any NaN"""
    df_clean = df[required_cols].copy()
    
    # For weekly data, be more lenient with missing values
    if timeframe == '1W':
        # Calculate how many NaN values per row
        nan_counts = df_clean.isnull().sum(axis=1)
        
        # Keep rows that have at most 20% missing features
        max_allowed_nans = len(required_cols) * 0.2
        keep_mask = nan_counts <= max_allowed_nans
        
        df_clean = df_clean[keep_mask]
        
        # Fill remaining NaNs with column means for numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        
        # For categorical columns, fill with mode
        categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')
    
    else:
        # For other timeframes, use standard dropna but be more careful
        # Only drop rows where critical columns are missing
        critical_cols = ['Hit_Label', 'Expected_Return', 'Expected_Loss', 'Close', 'High', 'Low']
        critical_cols_present = [col for col in critical_cols if col in df_clean.columns]
        df_clean = df_clean.dropna(subset=critical_cols_present)
        
        # Fill remaining NaNs with forward fill
        df_clean = df_clean.ffill().bfill()
    
    return df_clean
    
def train_models(df, timeframe):
    """Train ML models for the given timeframe"""
    try:
        # Check for required columns
        required_cols = FEATURES + ['Hit_Label', 'Expected_Return', 'Expected_Loss']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.warning(f"Missing columns for {timeframe}: {missing_cols}")
            return None, None, None, None, None, None
        
        # Use strategic NaN handling instead of strict dropna
        df_model = handle_missing_data(df, required_cols, timeframe)
        
        required_min = MIN_TRAIN_ROWS.get(timeframe, _Nr)
        
        if len(df_model) < required_min:
            st.warning(f"Insufficient data for {timeframe} modeling: {len(df_model)} rows (need {required_min})")
            return None, None, None, None, None, None
        
        progress_bar = st.progress(0)
        
        # Prepare features and labels for classification
        X_cls = df_model[FEATURES]
        y_cls = df_model['Hit_Label'].astype(int)
        
        X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
            X_cls, y_cls, test_size=0.2, random_state=42
        )
        
        scaler_cls = StandardScaler()
        X_train_scaled_cls = scaler_cls.fit_transform(X_train_cls)
        X_test_scaled_cls = scaler_cls.transform(X_test_cls)
        progress_bar.progress(25)
        
        # Train classification model on training set only
        model_class = RandomForestClassifier(
            n_estimators=400, 
            max_depth=12, 
            min_samples_split=4,
            min_samples_leaf=3,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42
        )
        model_class.fit(X_train_scaled_cls, y_train_cls)
        progress_bar.progress(50)
        
        cls_probs = model_class.predict_proba(scaler_cls.transform(X_cls))
        prob_df = pd.DataFrame(0, index=df_model.index, 
                              columns=[f'Prob_Class_{c}' for c in expected_classes])
        
        for i, c in enumerate(model_class.classes_):
            if c in expected_classes:
                prob_df[f'Prob_Class_{c}'] = cls_probs[:, i]
        
        FEATURES_with_probs = FEATURES + [f'Prob_Class_{c}' for c in expected_classes]
        X_reg = pd.concat([df_model[FEATURES], prob_df], axis=1)
        
        # Prepare data for regression (Expected_Return)
        y_return = df_model['Expected_Return']
        X_train_ret, X_test_ret, y_train_ret, y_test_ret = train_test_split(
            X_reg[FEATURES_with_probs], y_return, test_size=0.2, random_state=42
        )
        
        scaler_return = StandardScaler()
        X_train_scaled_ret = scaler_return.fit_transform(X_train_ret)
        X_test_scaled_ret = scaler_return.transform(X_test_ret)
        
        model_return = RandomForestRegressor(
            n_estimators=400,
            max_depth=14,
            min_samples_leaf=3,
            max_features='sqrt',
            ccp_alpha=0.001,
            random_state=42,
            n_jobs=-1
        )
        model_return.fit(X_train_scaled_ret, y_train_ret)
        progress_bar.progress(75)
        
        # Prepare data for regression (Expected_Loss)
        y_loss = df_model['Expected_Loss']
        X_train_loss, X_test_loss, y_train_loss, y_test_loss = train_test_split(
            X_reg[FEATURES_with_probs], y_loss, test_size=0.2, random_state=42
        )
        
        scaler_loss = StandardScaler()
        X_train_scaled_loss = scaler_loss.fit_transform(X_train_loss)
        X_test_scaled_loss = scaler_loss.transform(X_test_loss)
        
        model_loss = RandomForestRegressor(
            n_estimators=400,
            max_depth=14,
            min_samples_leaf=3,
            max_features='sqrt',
            ccp_alpha=0.001,
            random_state=42,
            n_jobs=-1
        )
        model_loss.fit(X_train_scaled_loss, y_train_loss)
        progress_bar.progress(100)
        
        return model_class, model_return, model_loss, scaler_cls, scaler_return, scaler_loss
        
    except Exception as e:
        st.error(f"Error training {timeframe} models: {str(e)}")
        return None, None, None, None, None, None

def make_prediction(model_class, model_return, model_loss, scaler_cls, scaler_return, scaler_loss, latest_data):
    """Make prediction for latest data"""
    try:
        # Check for missing features
        if latest_data[FEATURES].isnull().values.any():
            missing_features = latest_data[FEATURES].columns[latest_data[FEATURES].isnull().any()].tolist()
            st.warning(f"Missing features: {missing_features}")
            return None
        
        # Class prediction
        latest_scaled_cls = scaler_cls.transform(latest_data[FEATURES])
        latest_probs_raw = model_class.predict_proba(latest_scaled_cls)[0]
        
        latest_prob_features = {}
        for c in expected_classes:
            if c in model_class.classes_:
                idx = model_class.classes_.tolist().index(c)
                latest_prob_features[f'Prob_Class_{c}'] = latest_probs_raw[idx]
            else:
                latest_prob_features[f'Prob_Class_{c}'] = 0.0
        
        probs_of_interest = [latest_prob_features[f'Prob_Class_{c}'] for c in expected_classes]
        max_prob_index = probs_of_interest.index(max(probs_of_interest))
        pred_class = expected_classes[max_prob_index]
        will_hit = label2str.get(pred_class, "None")
        hit_prob = latest_prob_features[f'Prob_Class_{pred_class}'] * 100
        
        # Prepare features with probabilities for regression
        latest_prob_df = pd.DataFrame([latest_prob_features])
        latest_features_with_probs = pd.concat([latest_data[FEATURES].reset_index(drop=True), latest_prob_df], axis=1)
        
        # Return/Loss prediction
        latest_scaled_return = scaler_return.transform(latest_features_with_probs[FEATURES + list(latest_prob_features.keys())])
        latest_scaled_loss = scaler_loss.transform(latest_features_with_probs[FEATURES + list(latest_prob_features.keys())])
        
        current_price = latest_data['Close'].values[0]
        predicted_return = model_return.predict(latest_scaled_return)[0]
        predicted_loss = model_loss.predict(latest_scaled_loss)[0]
        
        predicted_tp = current_price * (1 + predicted_return)
        predicted_sl = current_price * (1 + predicted_loss)
        
        # Calculate percentage gain/loss from current price
        tp_percentage = ((predicted_tp - current_price) / current_price) * 100
        sl_percentage = ((predicted_sl - current_price) / current_price) * 100

        # Confidence calculation
        p_none  = latest_prob_features.get('Prob_Class_0', 0)
        p_sl    = latest_prob_features.get('Prob_Class_1', 0)
        p_tp    = latest_prob_features.get('Prob_Class_2', 0)
        p_hold  = latest_prob_features.get('Prob_Class_3', 0)
        p_short = latest_prob_features.get('Prob_Class_4', 0)

        bullish_prob = p_tp + p_hold
        bearish_prob = p_sl + p_short
    	
        if predicted_loss != 0:
            rr_ratio = predicted_return / abs(predicted_loss)
        else:
            rr_ratio = 0

        max_ratio = 10  # cap at 10:1
        log_ratio = np.log1p(rr_ratio)  # log(1+ratio)
        max_log_ratio = np.log1p(max_ratio)
        normalized_rr = log_ratio / max_log_ratio
    
        total_prob = bullish_prob + bearish_prob
        if total_prob > 0:
            prob_confidence = bullish_prob / total_prob 
        else:
            prob_confidence = 0.5
    
        confidence_score = (0.5 * prob_confidence + 0.5 * normalized_rr) * 100
                
        return {
            'will_hit': will_hit,
            'hit_prob': hit_prob,
            'predicted_tp': predicted_tp,
            'predicted_sl': predicted_sl,
            'predicted_return': predicted_return * 100,
            'predicted_loss': predicted_loss * 100,
            'tp_percentage': tp_percentage,  # NEW: Percentage from current price
            'sl_percentage': sl_percentage,  # NEW: Percentage from current price
            'confidence': confidence_score,
            'current_price': current_price
        }
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def plot_analysis(ticker, df, entry_price, timeframe, assessment, prediction=None, ind = 'OBV'):
    """Create analysis plot with TP and SL points"""
    
    try:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), 
                                       gridspec_kw={'height_ratios': [3, 1, 1]}, 
                                       sharex=True)
        
        # Price plot
        price = df['Close'].rolling(2).mean()
        ax1.plot(df.index, price, label='Price', color='gray', alpha=0.5, linewidth=1)
        
        # SMAs if available
        if 'EMA1' in df.columns:
            ax1.plot(df.index, df['EMA1'], label=f'EMA{int(_DAYS*0.5)}', color='orange', alpha=0.4, linewidth=1)
        if 'EMA2' in df.columns:
            ax1.plot(df.index, df['EMA2'], label=f'EMA{int(_DAYS*2)}', color='red', alpha=0.4, linewidth=1)
        ax1.fill_between(df.index, df.EMA1, df.EMA2, where=(df.EMA1 > df.EMA2), facecolor='green', alpha=0.15)
        ax1.fill_between(df.index, df.EMA1, df.EMA2, where=(df.EMA1 < df.EMA2), facecolor='red', alpha=0.15)
        
        # Entry point
        last_date = df.index[-1]
        ax1.plot(last_date, entry_price, 'o', markersize=5, color='black', alpha=0.3,
                 label=f'Entry: ${entry_price:.2f}')
        
        # Add TP and SL points if prediction is available
        if prediction is not None:
            # Take Profit point
            future_date = last_date + timedelta(days=20)
            
            tp_price = prediction['predicted_tp']
            ax1.plot(future_date, tp_price, '^', markersize=4, color='blue')
            ax1.annotate(f'TP: ${tp_price:.2f}', xy=(future_date, tp_price), xytext=(5, 5),
                         textcoords='offset points', ha='left', va='center', color='blue')
            
            sl_price = prediction['predicted_sl']
            ax1.plot(future_date, sl_price, 'v', markersize=4, color='red')
            ax1.annotate(f'SL: ${sl_price:.2f}', xy=(future_date, sl_price), xytext=(5, -5),
                         textcoords='offset points', ha='left', va='center', color='red')

            # Add horizontal lines for TP and SL
            future_date = future_date.tz_localize(None) if future_date.tzinfo else future_date
            x_max_date_num = ax1.get_xlim()[1]
            x_max_date = pd.to_datetime(mdates.num2date(x_max_date_num)).tz_localize(None)
            
            x_end_raw = future_date + pd.Timedelta(days=20)
            x_end = min(x_end_raw, x_max_date)
            
            ax1.axhline(y=tp_price, color='blue', linestyle='--', alpha=0.3, linewidth=1.2)
            ax1.axhline(y=sl_price, color='red', linestyle='--', alpha=0.3, linewidth=1.2)
                    
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left', fontsize='x-small')
        ax1.grid(True, alpha=0.5)

        hint = AnchoredText(
        "Hint: Buy closer to predicted SL to reduce risk\nand increase the chance of success.",
        loc='lower left',
        frameon=True,
        borderpad=1.5,
        prop=dict(size=10, color='gray', weight='bold')
        )

        latest = df.iloc[-1]

        entry_text = "Recent outlook is "
        cl = "gray"
        
        if latest["Bull"] == 1:
            entry_text += "Bullishness."
            cl = "green"
        elif latest["Bear"] == 1:
            entry_text += "Bearishness."
            cl = "red"
        else:
            entry_text += "Neutral."
        
        entry_desc = AnchoredText(
            entry_text,
            loc="lower right",
            frameon=False,
            borderpad=1.5,
            prop=dict(size=10, weight="bold"),
        )
        
        ax1.add_artist(hint)
        ax1.add_artist(entry_desc)
        hint.set_clip_on(True)
        hint.set_in_layout(True)
        hint.set_zorder(100)
        hint.patch.set_facecolor('honeydew')
        hint.patch.set_edgecolor('darkgreen')
        hint.patch.set_alpha(0.8)
        entry_desc.patch.set_alpha(0.5)
        entry_desc.txt._text.set_color(cl)
        
        # Assessment annotation
        color_map = {'Valid': 'green', 'Risky': 'orange', 'Wait and See': 'red'}
        assessment_color = color_map.get(assessment, 'gray')
        
        ax1.annotate(
            f'Assessment: {assessment}', 
            xy=(0.5, 0.95), xycoords='axes fraction',
            ha='center',  # horizontal alignment center
            fontsize=12,
            weight='bold',
            bbox=dict(boxstyle='round', facecolor=assessment_color, alpha=0.4)
        )

        # Add ticker name in the middle
        tx = f'{ticker} ({timeframe})'
        ax1.text(0.5, 0.5, tx, transform=ax1.transAxes, 
                     fontsize=50, color='grey', alpha=0.2,
                     horizontalalignment='center', verticalalignment='center',
                     rotation=0, weight='bold', style='italic')    

        # RSI plot if available
        if 'RSI' in df.columns:
            rsi_ = df['RSI'].rolling(3).mean()
            rsi_sma = df['RSI'].rolling(20).mean()
            ax2.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.4)
            ax2.plot(df.index, rsi_, label='RSI', color='gray', linewidth=1.5, alpha=0.4)
            ax2.plot(df.index, rsi_sma, label='RSI SMA', color='red', linewidth=1.5, alpha=0.45)
            ax2.fill_between(df.index, rsi_, 52, where=(df['RSI'] > 52), facecolor='green', alpha=0.15)
            ax2.fill_between(df.index, rsi_, 40, where=(df['RSI'] < 40), facecolor='red', alpha=0.15)
            ax2.fill_between(df.index, rsi_, rsi_sma, where=((df['RSI'] < df['RSI_SMA']) & (df.EMA1 > df.EMA2)), facecolor='orange', alpha=0.14, label='Dip(?)')
            ax2.axhline(70, color='red', linestyle='--', alpha=0.4)
            ax2.axhline(30, color='green', linestyle='--', alpha=0.4)
            ax2.axhline(50, color='gray', linestyle='-', alpha=0.4)
            
            _s = 5
            ax2.scatter(df.index[df['Bull'] == 1], rsi_[df['Bull'] == 1], color='green', marker='^', s=_s, alpha=0.3, label='Bull', zorder=7)
            ax2.scatter(df.index[df['Bear'] == 1], rsi_[df['Bear'] == 1], color='red', marker='v', s=_s, alpha=0.3, label='Bear', zorder=8)
            ax2.scatter(df.index[df['Short'] == 1], rsi_[df['Short'] == 1], color='red', marker='x', s=_s*3, alpha=0.4, label='Short', zorder=9)
            hold_mask = df['Hold'] == 1
            colors = np.where(df['EMA1'] < df['EMA2'], 'red', 'orange')
            ax2.scatter(df.index[hold_mask], rsi_[hold_mask], color=colors[hold_mask], marker='o', s=_s, alpha=0.3, label='Hold', zorder=10)
            ax2.yaxis.set_label_position("right")
            ax2.yaxis.tick_right()
            ax2.set_ylabel('RSI')
            ax2.set_ylim(0, 100)
            ax2.legend(loc='lower left', fontsize='x-small')
        else:
            ax2.text(0.5, 0.5, 'RSI data not available', ha='center', va='center', transform=ax2.transAxes)

        # 3. Lower Most Plot
        
        if ind in df.columns:
            ax3.plot(df.index, df[ind], label= ind, color='gray', alpha=0.4, linewidth=1.2)
            ax3.yaxis.set_label_position("right")
            ax3.yaxis.tick_right()
            ax3.set_ylabel(ind)
            ax3.grid(True, alpha=0.3)
            ax3.axhline(0, color='black', linestyle='--', alpha=0.25)
            if ind == "CCI":
                ax3.axhline(250, color='green', linestyle='--',  linewidth=1., alpha=0.25)
                ax3.axhline(200, color='green', linestyle=':',  linewidth=1., alpha=0.25, label = 'OverBought')
                ax3.axhline(-200, color='red', linestyle=':',  linewidth=1., alpha=0.25, label = 'OverSold')
                ax3.axhline(-250, color='red', linestyle='--',  linewidth=1., alpha=0.25)
        else:
            ax3.text(0.5, 0.5, 'selected data are not available', ha='center', va='center', transform=ax2.transAxes)

        ax2.grid(True, alpha=0.3)
        ax3.legend(loc='lower left', fontsize='x-small')
        plt.title(f'{timeframe} Analysis - {assessment}')
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
        # Return empty figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Plot error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
        return fig

def assess_entry(prediction, user_gain, user_loss, entry_price, current_price):
    """Assess if entry is valid, risky, or not recommended"""
    if prediction is None:
        return "Not Recommended", "Insufficient data for prediction"
    
    will_hit = prediction['will_hit']
    hit_prob = prediction['hit_prob']
    confidence = prediction['confidence']
    pred_return = prediction['predicted_return']
    pred_loss = prediction['predicted_loss']
    
    # Price proximity check
    price_diff_pct = abs(entry_price - current_price) / current_price * 100
    
    reasons = []
    
    # Bullish signal check
    if will_hit == 'TP' and hit_prob > 40:
        reasons.append("Bullish signal detected")
    else:
        reasons.append(f"Signal: {will_hit} (Prob: {hit_prob:.1f}%)")
    
    # Confidence check
    if confidence >= 70:
        reasons.append(f"High confidence ({confidence:.0f}%)")
    elif confidence > 58 and confidence < 70:
        reasons.append(f"Moderate confidence ({confidence:.0f}%)")
    else:
        reasons.append(f"Low confidence ({confidence:.0f}%)")
    
    # Risk-Reward check
    user_rr = user_gain / abs(user_loss) if user_loss != 0 else 0
    pred_rr = pred_return / abs(pred_loss) if pred_loss != 0 else 0
    
    if (pred_rr > user_rr and pred_rr >= 1.25):
        reasons.append("Good risk-reward ratio")
    elif (pred_rr > user_rr and pred_rr >= 1):
        reasons.append("Moderate risk-reward ratio")
    else:
        reasons.append("Poor risk-reward ratio")
    
    # Price proximity
    if price_diff_pct > 7:
        reasons.append("Entry price far from current price")
    elif price_diff_pct > 4:
        reasons.append("Entry price moderately different")
    else:
        reasons.append("Entry price close to current")
    
    # Overall assessment
    bullish_conditions = (will_hit in ['TP', 'Hold'] and hit_prob > 40 and confidence > 60 and pred_rr > 1.4)
    risky_conditions = (will_hit in ['TP', 'Hold', 'None'] and confidence > 50 and pred_rr > 1.2)
    
    if bullish_conditions and price_diff_pct <= 10:
        assessment = "Valid"
    elif risky_conditions and price_diff_pct <= 10:
        assessment = "Risky"
    else:
        assessment = "Wait and See"
    
    return assessment, " | ".join(reasons)

def avg_bull_bear_lengths(df):
    bull = (df['EMA1'] > df['EMA2'])
    bear = (df['EMA1'] < df['EMA2'])

    periods = []
    current_trend = None
    length = 0
    for is_bull, is_bear in zip(bull, bear):
        if is_bull:
            if current_trend == 'bull':
                length += 1
            else:
                if current_trend is not None:
                    periods.append((current_trend, length))
                current_trend = 'bull'
                length = 1
        elif is_bear:
            if current_trend == 'bear':
                length += 1
            else:
                if current_trend is not None:
                    periods.append((current_trend, length))
                current_trend = 'bear'
                length = 1
        else:
            if current_trend is not None:
                periods.append((current_trend, length))
            current_trend = None
            length = 0
    if current_trend is not None:
        periods.append((current_trend, length))
    bull_lengths = [length for trend, length in periods if trend == 'bull']
    bear_lengths = [length for trend, length in periods if trend == 'bear']
    avg_bull = sum(bull_lengths) / len(bull_lengths) if bull_lengths else 0
    avg_bear = sum(bear_lengths) / len(bear_lengths) if bear_lengths else 0
    return avg_bull, avg_bear

def update_entry_price():
    st.session_state.entry_price = st.session_state.entry_price_input
      
def get_current_price(ticker: str):
    try:
        data = yf.Ticker(ticker).history(period="1d")
        if data.empty or "Close" not in data:
            return None
        return data["Close"].iloc[-1]
    except Exception:
        return None

def update_price_and_reset_entry():
    ticker = st.session_state.get("ticker", None)
    if not ticker:
        return

    current_price = get_current_price(ticker)
    if current_price is None:
        st.error(f"Could not fetch price for {ticker}. Invalid or no data.")
        return
    st.session_state.current_price = current_price
    st.session_state.entry_price = current_price
    st.session_state.entry_price_input = current_price
    st.session_state.previous_ticker = ticker
    st.session_state.initial_prices_set = True


def clear_page_session_state():
    """Clear only this page's session state on load"""
    keys_to_remove = []
    for key in st.session_state.keys():
        if key.startswith('entry_analyzer_'):
            keys_to_remove.append(key)
            
    # Force price fields to reset on every page load
    keys_to_remove.extend(["current_price", "entry_price", "initial_prices_set", "previous_ticker"])

    for key in keys_to_remove:
        # Use .pop() for safer deletion
        st.session_state.pop(key, None)
        
def initialize_session_state():
    if "ticker_input" not in st.session_state:
        st.session_state.ticker_input = DEFAULT_TICKER
    if "previous_ticker" not in st.session_state:
        st.session_state.previous_ticker = ""
    if "current_price" not in st.session_state:
        st.session_state.current_price = 0.0
    if "entry_price" not in st.session_state:
        st.session_state.entry_price = 0.0
    if "entry_price_input" not in st.session_state:
        st.session_state.entry_price_input = 0.0
    if "initial_prices_set" not in st.session_state:
        st.session_state.initial_prices_set = False

    # If ticker just initialized OR changed, fetch and set prices
    ticker = st.session_state.ticker_input.upper()
    if not st.session_state.initial_prices_set or ticker != st.session_state.previous_ticker:
        current_price = get_current_price(ticker)
        if current_price is not None:
            st.session_state.current_price = current_price
            st.session_state.entry_price = current_price
            st.session_state.entry_price_input = current_price
            st.session_state.initial_prices_set = True
            st.session_state.previous_ticker = ticker
            
def main():
    st.title("📊 Entry Position Analyzer")
    st.write("Analyze your entry position using ML models trained on 4H, 1D, and 1W timeframes. Type ticker: e.g. TSLA or BTC-USD. Or find ticker name on yahoo finance.")

    with st.expander("Disciplined Entry and Exit Strategy (Expand and learn)", expanded=False):
        st.write(desc)
        
    initialize_session_state()
    col1, col2, col3 = st.columns(3)

    with col1:
        # ticker input with callback to update prices
        ticker = st.text_input(
            "Ticker Symbol",
            value="TSLA",
            key="ticker"
        ).upper()
        
        for key in ["current_price", "entry_price", "entry_price_input", "previous_ticker"]:
            if key not in st.session_state:
                st.session_state[key] = 0 if "price" in key else ""
                
        result = validate_ticker(ticker)
        if result["valid"]:
            st.success(f"{ticker} is valid ✅ ({result['reason']})")
        else:
            st.error(f"{ticker} is invalid ❌ ({result['reason']})")
            st.stop()
                
    with col2:
        # Set initial prices only once when ticker is valid
        if result["valid"] and not st.session_state.initial_prices_set:
            current_price = get_current_price(ticker)
            st.session_state.current_price = current_price
            st.session_state.entry_price = current_price
            st.session_state.initial_prices_set = True
            st.session_state.previous_ticker = ticker

        # Reset prices only when ticker actually changes
        if result["valid"] and st.session_state.previous_ticker != ticker:
            current_price = get_current_price(ticker)
            st.session_state.current_price = current_price
            st.session_state.entry_price = update_entry_price()
            st.session_state.previous_ticker = ticker

        # Display current price
        st.metric("Current Price", f"${st.session_state.current_price:.2f}")

        # Entry price number input - this will maintain its value between reruns
        entry_price = st.number_input(
            "Entry Price ($)",
            min_value=0.0,
            value=float(st.session_state.entry_price_input),  # ensure float type
            step=0.1,
            key="entry_price_input",
            on_change=update_entry_price,
        )

    with col3:
        user_gain = st.number_input(
            "Expected Gain (%)",
            min_value=0.1,
            max_value=15.0,
            value=3.75,
            step=0.1,
            key="user_gain",
            help="Tip: Realistic training needs modest/realistic repeatable gains like 2-7%. 10-15% gains results in less data and unrealistic results." 
        )
        user_loss = st.number_input(
            "Expected Loss (%)",
            min_value=0.1,
            max_value=15.0,
            value=3.75,
            step=0.1,
            key="user_loss",
            help="Tip: Realistic training needs modest/realistic repeatable gains like 2-7%. 10-15% gains results in less data and unrealistic results."             
        )

    opt = ['OBV', 'CCI', 'CMF', 'MFI', 'ADX']
    default_option_index = 0 
    
    ind = st.selectbox(
        "Choose 3rd indicator:",
        opt,
        index=default_option_index
    )

    if st.button("Analyze Entry Position"):
        with st.spinner("Training models and analyzing..."):
            try:
                # Get current date/time
                end_date = datetime.now()

                results = {}

                timeframes = [
                    ("1W", "1W"),
                    ("1D", "1D"),
                    ("4H", "4H")
                ]

                for timeframe, interval in timeframes:
                    st.subheader(f"{timeframe} ML of {ticker}")

                    years = YEARS_OF_DATA[timeframe]
                    start_date = end_date - timedelta(days=365 * years)

                    # Fetch data for timeframe
                    with st.spinner(f"Fetching {timeframe} data..."):
                        df = get_stock_data(ticker, start_date, end_date, interval)

                    if df is None:
                        st.warning(f"No data available for {timeframe} timeframe")
                        continue

                    required_min_raw = MIN_TRAIN_ROWS.get(timeframe, _Nr)
                    
                    if len(df) < required_min_raw:
                        st.warning(f"Insufficient raw data for {timeframe}: {len(df)} rows (need {required_min_raw})")
                        continue

                    st.write(f"Data points: {len(df)}")

                    # Calculate technical indicators
                    with st.spinner("Calculating technical indicators..."):
                        df = add_technical_indicators(df, timeframe)
                        df = add_pivot_levels(df, window=14)
                        df = add_pivots(df, windows)
                        df = average_pivots(df, windows)

                    if df is None:
                        st.warning(f"Error calculating indicators for {timeframe}")
                        continue

                    # Compute expected returns and losses
                    with st.spinner("Computing expected returns..."):
                        df = compute_expected_return(df, forward_window=14, r_cols=['R1_Avg', 'R2_Avg'])
                        df = compute_expected_loss(df, forward_window=14, s_cols=['S1_Avg', 'S2_Avg'])

                    # Label hit probabilities
                    with st.spinner("Labeling hit probabilities..."):
                        df = label_hit_prob_past(df, profit_target=user_gain / 100, stop_loss=user_loss / 100)

                    # Train models
                    with st.spinner(f"Training {timeframe} ML models..."):
                        models = train_models(df, timeframe)

                    if models[0] is None:
                        st.warning(f"Could not train models for {timeframe} timeframe")
                        continue

                    model_class, model_return, model_loss, scaler_cls, scaler_return, scaler_loss = models

                    # Make prediction
                    latest_data = df.iloc[[-1]]
                    prediction = make_prediction(
                        model_class,
                        model_return,
                        model_loss,
                        scaler_cls,
                        scaler_return,
                        scaler_loss,
                        latest_data,
                    )

                    if prediction:
                        current_price = prediction['current_price']
                        assessment, reasons = assess_entry(prediction, user_gain, user_loss, entry_price, current_price)

                        results[timeframe] = {
                            "prediction": prediction,
                            "assessment": assessment,
                            "reasons": reasons,
                            "df": df,
                        }

                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Current Price", f"${current_price:.2f}")

                            tp_percentage = prediction['tp_percentage']
                            tp_delta = f"{tp_percentage:+.1f}%"
                            st.metric(
                                "Predicted TP",
                                f"${prediction['predicted_tp']:.2f}",
                                delta=tp_delta,
                                delta_color="normal" if tp_percentage > 0 else "off",
                            )

                            sl_percentage = prediction['sl_percentage']
                            sl_delta = f"{sl_percentage:+.1f}%"
                            st.metric(
                                "Predicted SL",
                                f"${prediction['predicted_sl']:.2f}",
                                delta=sl_delta,
                                delta_color="normal" if sl_percentage < 0 else "off",
                            )

                        with col2:
                            hit_value = prediction['will_hit']
                            hit_prob = prediction['hit_prob']
                            rrr = abs(tp_percentage / sl_percentage if sl_percentage != 0 else float('inf'))
                            st.metric(label="Hits", value=f"{hit_value} ({hit_prob:.1f}%)")
                            st.metric(label="Risk/Reward", value=f"{rrr:.1f}")
                            color = "green" if rrr > 1.5 else "red"
                            st.markdown(f"<p style='color:{color};'>R/R is {'GOOD' if rrr > 1.5 else 'POOR'}</p>", unsafe_allow_html=True)
                            st.metric("CONFIDENCE", f"{prediction['confidence']:.1f}%")

                        if assessment == "Valid":
                            st.success(f"**Assessment**: {assessment}")
                        elif assessment == "Risky":
                            st.warning(f"**Assessment**: {assessment}")
                        else:
                            st.error(f"**Assessment**: {assessment}")

                        st.write(f"**Reasons**: {reasons}")
                        avg_bull, avg_bear = avg_bull_bear_lengths(df)
                        st.write(f"Average Bull Market, {timeframe}: {avg_bull:.0f}, Average Bear Market {timeframe}: {avg_bear:.0f}")
                
                        fig = plot_analysis(ticker, df, entry_price, timeframe, assessment, prediction, ind=ind)
                        st.pyplot(fig)
                    else:
                        st.warning(f"Could not generate prediction for {timeframe}")

                    st.write("---")

                # Overall recommendation
                if results:
                    st.subheader("🎯 Overall Recommendation")

                    assessments = [results[tf]['assessment'] for tf in results.keys()]
                    valid_count = assessments.count("Valid")
                    risky_count = assessments.count("Risky")
                    total_timeframes = len(results)

                    rr_values = []
                    conf_values = []
                    for tf in results.keys():
                        pred = results[tf]['prediction']
                        if pred:
                            tp_pct = pred.get('tp_percentage', 0)
                            sl_pct = pred.get('sl_percentage', 1)  # Avoid zero division (fallback 1)
                            rr = abs(tp_pct / sl_pct) if sl_pct != 0 else float('inf')
                            conf = pred.get('confidence', 0)
                            rr_values.append(rr)
                            conf_values.append(conf)
                    avg_rr = np.mean(rr_values) if rr_values else 0
                    avg_conf = np.mean(conf_values) if conf_values else 0
                
                    annotation = f"(Avg R/R: {avg_rr:.2f}, Avg Conf: {avg_conf:.1f}%)"
                
                    if valid_count == total_timeframes:
                        st.success(f"**STRONG BUY** - All timeframes show valid entry {annotation}")
                    elif valid_count >= total_timeframes / 2:
                        st.success(f"**BUY** - Majority of timeframes show valid entry {annotation}")
                    elif valid_count >= 1 or risky_count >= total_timeframes / 2:
                        st.warning(f"**CAUTIOUS BUY** - Mixed or risky signals {annotation}")
                    else:
                        st.error(f"**AVOID** - Poor signals across timeframes {annotation}")

                    # BUILD A SUMMARY TABLE
                    emoji_map = {
                        "Valid": "🟢 Valid",
                        "Risky": "🟡 Risky",
                        "Avoid": "🔴 Avoid",
                    }
                    
                    summary_data = []
                    for tf, res in results.items():
                        pred = res['prediction']
                        rr = abs(pred['tp_percentage'] / pred['sl_percentage']) if pred['sl_percentage'] != 0 else float('inf')
                        assessment_text = res['assessment']
                        display_assessment = emoji_map.get(assessment_text, assessment_text)
                        summary_data.append({
                            "Timeframe": tf,
                            "Price ($)": round(pred['current_price'], 2),
                            "TP ($)": round(pred['predicted_tp'], 2),
                            "SL ($)": round(pred['predicted_sl'], 2),
                            "Conf (%)": round(pred['confidence'], 1),
                            "Hits": pred['will_hit'],
                            "R/R": round(rr, 2),
                            "Assessment": display_assessment
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    summary_df = summary_df[["Timeframe", "Price ($)", "TP ($)", "SL ($)", "Conf (%)", "Hits", "R/R", "Assessment"]]
                                               
                    st.write(f"**Timeframe Summary ({ticker}):**")
                    st.dataframe(summary_df)

                else:
                    st.error("No successful analyses completed. Try with a different ticker or time period.")

            except Exception as e:
                st.error(f"Error analyzing {ticker}: {str(e)}")
                st.info("Try with a different ticker or check if market is open")

    with st.expander("How to use this analyzer"):
        st.write(
            """
        1. **Enter Ticker Symbol**: Stock symbol (e.g., AAPL, TSLA, NVDA) or crypto (BTC-USD)
        2. **Set Entry Price**: Your intended entry price
        3. **Define Expectations**: Your target gain and maximum acceptable loss (Conservative 2-5%, aggressive 5-12%, unrealistic 20% or higher
        4. **Click Analyze**: The system will train ML models and evaluate your entry

        **Timeframe Data Requirements:**
        - **4H**: 1 year of historical data (~2000+ data points)
        - **1D**: 2 years of historical data (~500+ data points)
        - **1W**: 5 years of historical data (~250+ data points)

        **Assessment Colors:**
        - 🟢 **Valid**: Good entry with strong bullish signals
        - 🟡 **Risky**: Moderate signals, proceed with caution  
        - 🔴 **Wait and See**: Poor risk-reward or bearish signals

        **The analysis considers:**
        - ML predictions for TP/SL hits
        - Risk-reward ratios
        - Technical indicator alignment
        - Price proximity to current levels
        - Confidence scores from ensemble models

        **Note**: 4H data may not be available for all tickers outside market hours.
        1W data requires at least 5 years of history for sufficient data points.
        """
        )


if __name__ == "__main__":
    main()
