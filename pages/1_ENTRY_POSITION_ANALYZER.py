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
st.set_page_config(page_title="📊 Entry Position Analyzer", layout="wide")
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
    'Bull', 'Bear', 'Short', 'Hold', 'Neutral', 'StrongBull', 'StrongBear', 'Exhaustion',

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
                        interval=yf_interval, progress=False, auto_adjust=True, actions=False)
        
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
        X_all = df_model[FEATURES]
        y_all = df_model['Hit_Label'].astype(int)
        
        # Filter to ensure only expected_classes are used for all steps
        valid_mask = y_all.isin(expected_classes)
        X_filtered = X_all[valid_mask]
        y_filtered = y_all[valid_mask]
        
        if y_filtered.empty or len(y_filtered) < MIN_TRAIN_ROWS.get(timeframe, _Nr):
            st.warning(f"Insufficient valid data points ({len(y_filtered)}) for {timeframe} modeling after label filtering.")
            return None, None, None, None, None, None
        
        X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
            X_filtered, y_filtered, test_size=0.2, random_state=42
        )
        
        scaler_cls = StandardScaler()
        X_train_scaled_cls = scaler_cls.fit_transform(X_train_cls)
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
        
        # Prediction probabilities on the full filtered set and safe mapping
        X_filtered_scaled = scaler_cls.transform(X_filtered)
        cls_probs = model_class.predict_proba(X_filtered_scaled)
        
        prob_df = pd.DataFrame(0.0, index=X_filtered.index, 
                              columns=[f'Prob_Class_{c}' for c in expected_classes])
        
        for i, c in enumerate(model_class.classes_):
            if c in expected_classes:
                prob_df[f'Prob_Class_{c}'] = cls_probs[:, i] 

        FEATURES_with_probs = FEATURES + [f'Prob_Class_{c}' for c in expected_classes]
        X_reg = pd.concat([X_filtered[FEATURES], prob_df], axis=1) 
        
        # Prepare data for regression (Expected_Return) - using filtered data
        y_return = df_model['Expected_Return'][valid_mask]
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
        
        # Prepare data for regression (Expected_Loss) - using filtered data
        y_loss = df_model['Expected_Loss'][valid_mask]
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
            latest_data_filled = latest_data[FEATURES].fillna(0) # Simple fill for safety
        else:
            latest_data_filled = latest_data[FEATURES]

        # Class prediction
        latest_scaled_cls = scaler_cls.transform(latest_data_filled)
        latest_probs_raw = model_class.predict_proba(latest_scaled_cls)[0]
        
        # Robust probability feature extraction for prediction
        latest_prob_features = {}
        for c in expected_classes:
            if c in model_class.classes_:
                idx = model_class.classes_.tolist().index(c)
                latest_prob_features[f'Prob_Class_{c}'] = latest_probs_raw[idx]
            else:
                latest_prob_features[f'Prob_Class_{c}'] = 0.0 # Class not seen in training
        
        probs_of_interest = [latest_prob_features[f'Prob_Class_{c}'] for c in expected_classes]
        max_prob_index = probs_of_interest.index(max(probs_of_interest))
        pred_class = expected_classes[max_prob_index]
        will_hit = label2str.get(pred_class, "None")
        hit_prob = latest_prob_features[f'Prob_Class_{pred_class}'] * 100
        
        # Prepare features with probabilities for regression
        latest_prob_df = pd.DataFrame([latest_prob_features])
        latest_features_with_probs = pd.concat([latest_data_filled.reset_index(drop=True), latest_prob_df], axis=1)
        
        # Define the exact list of features the scaler was trained on
        FEATURES_FOR_REGRESSION = FEATURES + list(latest_prob_features.keys())

        # Regression scaling
        latest_scaled_return = scaler_return.transform(latest_features_with_probs[FEATURES_FOR_REGRESSION])
        latest_scaled_loss = scaler_loss.transform(latest_features_with_probs[FEATURES_FOR_REGRESSION])
        
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
            'tp_percentage': tp_percentage,
            'sl_percentage': sl_percentage,
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

        from matplotlib.offsetbox import AnchoredText
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
        color_map = {'Valid': 'green', 'Risky': 'orange', 'Bearish': 'red', 'Wait and See': 'red'}
        assessment_color = color_map.get(assessment, 'gray')
        
        ax1.annotate(
            f'Assessment: {assessment}', 
            xy=(0.5, 0.95), xycoords='axes fraction',
            ha='center',
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
    bullish_conditions = (will_hit in ['TP', 'Hold'] and confidence > 60 and pred_rr > 1.4)
    risky_conditions = (will_hit in ['None'] and confidence > 54 and pred_rr > 1.2)
    bearish = (will_hit in ['SL', 'None'] and confidence < 45)
    
    if bullish_conditions and price_diff_pct <= 10:
        assessment = "Valid"
    elif risky_conditions and price_diff_pct <= 10:
        assessment = "Risky"
    elif bearish:
        assessment = 'Bearish'
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
        st.session_state.current_price = None
    if "entry_price" not in st.session_state:
        st.session_state.entry_price = None
    if "entry_price_input" not in st.session_state:
        st.session_state.entry_price_input = None
    if "initial_prices_set" not in st.session_state:
        st.session_state.initial_prices_set = False
    if "patience_score" not in st.session_state:
        st.session_state.patience_score = 0
    if "entry_journal" not in st.session_state:
        st.session_state.entry_journal = []
    if "last_analysis_time" not in st.session_state:
        st.session_state.last_analysis_time = None

def calculate_entry_score(prediction, df, timeframe):
    """Calculate an overall entry score from 0-100"""
    
    score_components = {}
    
    # 1. ML Confidence (max 30 points)
    score_components['ML Confidence'] = min(prediction['confidence'] * 0.3, 30)
    
    # 2. Risk/Reward Ratio (max 25 points)
    rr = abs(prediction['tp_percentage'] / prediction['sl_percentage']) if prediction['sl_percentage'] != 0 else 0
    if rr >= 2.0:
        score_components['Risk/Reward'] = 25
    elif rr >= 1.5:
        score_components['Risk/Reward'] = 20
    elif rr >= 1.2:
        score_components['Risk/Reward'] = 15
    else:
        score_components['Risk/Reward'] = 5
    
    # 3. Technical alignment (max 20 points)
    latest = df.iloc[-1]
    tech_score = 0
    if latest['Bull'] == 1 or latest['Hold'] == 1:
        tech_score += 10
    if latest.get('RSI', 50) > 50:
        tech_score += 5
    if latest.get('EMA1', 0) > latest.get('EMA2', 1):
        tech_score += 5
    score_components['Technical Setup'] = tech_score
    
    # 4. Volume confirmation (max 10 points)
    if 'Volume_MA20' in df.columns and 'Volume' in df.columns:
        volume_ratio = df['Volume'].iloc[-1] / df['Volume_MA20'].iloc[-1]
        if volume_ratio > 1.2:
            score_components['Volume'] = 10
        elif volume_ratio > 1.0:
            score_components['Volume'] = 7
        else:
            score_components['Volume'] = 3
    else:
        score_components['Volume'] = 5  # Default if no volume data
    
    # 5. Trend alignment (max 15 points)
    if timeframe == '1W':
        # Weekly trend is most important
        if latest.get('EMA1', 0) > latest.get('EMA2', 1):
            score_components['Trend'] = 15
        else:
            score_components['Trend'] = 5
    else:
        score_components['Trend'] = 10  # Default for lower timeframes
    
    total_score = sum(score_components.values())
    
    return total_score, score_components

def display_entry_warnings(current_price, entry_price, prediction=None):
    """Display warnings about early/desperate entries"""
    
    warnings = []
    
    # Price chasing warning
    price_diff_pct = abs(current_price - entry_price) / current_price * 100
    if price_diff_pct > 5:
        warnings.append(f"⚠️ **Chasing Price**: Entry is {price_diff_pct:.1f}% away from current. Consider waiting for pullback.")
    
    # Add RSI warning if available
    if prediction and 'will_hit' in prediction:
        if prediction['will_hit'] == 'TP' and prediction['hit_prob'] < 40:
            warnings.append("📉 **Low Hit Probability**: TP probability below 40% - weak signal")
    
    # Display warnings
    if warnings:
        with st.container():
            st.markdown("### 🚨 **ENTRY WARNINGS**")
            for warning in warnings:
                st.markdown(f"- {warning}")
            st.markdown("---")
    
    return len(warnings)

def display_patience_meter():
    """Show a patience meter visualization"""
    
    st.markdown("### 🧘 **Patience Meter**")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        if st.button("⏸️ I'll wait 1 more candle", 
                    help="Good discipline!", key="patience_wait"):
            st.session_state.patience_score += 10
    
    with col_m2:
        if st.button("🔍 Check smaller timeframe", 
                    help="Instead of entering, analyze lower TF", key="patience_check"):
            st.session_state.patience_score += 5
    
    with col_m3:
        if st.button("🏃 Enter now (FOMO)", 
                    help="High risk - you're probably chasing", key="patience_fomo"):
            st.session_state.patience_score = max(0, st.session_state.patience_score - 20)
    
    # Display meter
    patience_level = min(100, st.session_state.patience_score)
    
    if patience_level < 30:
        st.error(f"**IMPULSIVE** ({patience_level}/100)")
        st.write("🐇 You're acting on emotion, not analysis")
    elif patience_level < 60:
        st.warning(f"**NEEDS WORK** ({patience_level}/100)")
        st.write("🚶 You're improving but still impulsive")
    elif patience_level < 80:
        st.info(f"**PATIENT** ({patience_level}/100)")
        st.write("🚶‍♂️ Good discipline - waiting for confirmations")
    else:
        st.success(f"**MASTER TRADER** ({patience_level}/100)")
        st.write("🐢 Excellent patience - you wait for perfect setups")
    
    st.progress(patience_level / 100)
    st.markdown("---")

def calculate_technical_confirmation(df, timeframe, entry_price):
    """Calculate automatic technical confirmation score based on current price"""
    
    if df is None or len(df) < 2:
        return 0, {}
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    confirmation_items = []
    scores = {}
    
    # 1. Price vs Moving Averages (25 points)
    price_ma_score = 0
    if 'EMA1' in latest and 'EMA2' in latest:
        above_ema1 = entry_price > latest['EMA1']
        above_ema2 = entry_price > latest['EMA2']
        ema_bullish = latest['EMA1'] > latest['EMA2']
        
        if above_ema1 and above_ema2 and ema_bullish:
            price_ma_score = 25
            confirmation_items.append("✅ Price above both EMAs & EMA1 > EMA2")
        elif above_ema1 and above_ema2:
            price_ma_score = 20
            confirmation_items.append("✅ Price above both EMAs")
        elif above_ema1:
            price_ma_score = 15
            confirmation_items.append("⚠️ Price above EMA1 only")
        else:
            price_ma_score = 5
            confirmation_items.append("❌ Price below EMAs - Bearish")
    scores['MA Alignment'] = price_ma_score
    
    # 2. RSI Conditions (20 points)
    rsi_score = 0
    if 'RSI' in latest and 'RSI_SMA' in latest:
        rsi = latest['RSI']
        rsi_sma = latest['RSI_SMA']
        
        if 50 < rsi < 70 and rsi > rsi_sma:
            rsi_score = 20
            confirmation_items.append("✅ RSI 50-70 & above SMA - Ideal")
        elif 40 < rsi < 80 and rsi > rsi_sma:
            rsi_score = 15
            confirmation_items.append("⚠️ RSI 40-80 & above SMA - Acceptable")
        elif rsi > 70:
            rsi_score = 5
            confirmation_items.append("❌ RSI > 70 - Overbought")
        elif rsi < 40:
            rsi_score = 0
            confirmation_items.append("❌ RSI < 40 - Oversold")
        else:
            rsi_score = 10
            confirmation_items.append("⚠️ RSI neutral")
    scores['RSI Condition'] = rsi_score
    
    # 3. Volume Confirmation (15 points)
    volume_score = 0
    if 'Volume' in latest and 'Volume_MA20' in latest:
        volume_ratio = latest['Volume'] / latest['Volume_MA20']
        if volume_ratio > 1.5:
            volume_score = 15
            confirmation_items.append("✅ Strong volume (>1.5x avg)")
        elif volume_ratio > 1.2:
            volume_score = 12
            confirmation_items.append("⚠️ Good volume (>1.2x avg)")
        elif volume_ratio > 1.0:
            volume_score = 8
            confirmation_items.append("⚠️ Average volume")
        else:
            volume_score = 3
            confirmation_items.append("❌ Low volume (< avg)")
    else:
        volume_score = 7
        confirmation_items.append("⚠️ Volume data not available")
    scores['Volume'] = volume_score
    
    # 4. Trend Strength (ADX) (15 points)
    adx_score = 0
    if 'ADX' in latest:
        adx = latest['ADX']
        if adx > 25:
            adx_score = 15
            confirmation_items.append(f"✅ Strong trend (ADX: {adx:.1f})")
        elif adx > 20:
            adx_score = 12
            confirmation_items.append(f"⚠️ Moderate trend (ADX: {adx:.1f})")
        else:
            adx_score = 5
            confirmation_items.append(f"❌ Weak/no trend (ADX: {adx:.1f})")
    scores['Trend Strength'] = adx_score
    
    # 5. Directional Indicators (15 points)
    di_score = 0
    if '+DI' in latest and '-DI' in latest:
        if latest['+DI'] > latest['-DI']:
            di_score = 15
            confirmation_items.append("✅ +DI > -DI (Bullish momentum)")
        else:
            di_score = 5
            confirmation_items.append("❌ -DI > +DI (Bearish momentum)")
    scores['Momentum'] = di_score
    
    # 6. Price Action (10 points)
    price_action_score = 0
    if 'Close' in latest and 'Open' in latest and 'High' in latest and 'Low' in latest:
        is_bullish_candle = latest['Close'] > latest['Open']
        body_size = abs(latest['Close'] - latest['Open'])
        candle_range = latest['High'] - latest['Low']
        body_ratio = body_size / candle_range if candle_range > 0 else 0
        
        if is_bullish_candle and body_ratio > 0.6:
            price_action_score = 10
            confirmation_items.append("✅ Strong bullish candle")
        elif is_bullish_candle:
            price_action_score = 7
            confirmation_items.append("⚠️ Bullish candle")
        elif body_ratio > 0.6:
            price_action_score = 3
            confirmation_items.append("❌ Strong bearish candle")
        else:
            price_action_score = 5
            confirmation_items.append("⚠️ Neutral/Doji candle")
    scores['Price Action'] = price_action_score
    
    total_score = sum(scores.values())
    
    return total_score, scores, confirmation_items

def display_technical_confirmation_metric(all_timeframes_data, entry_price, ticker):
    """Display the Technical Analysis Confirmation metric after all timeframes are plotted"""
    
    if not all_timeframes_data:
        return
    
    st.markdown("---")
    st.subheader("📊 Technical Analysis Confirmation")
    
    # Calculate confirmation for each timeframe
    timeframe_scores = {}
    all_confirmation_items = []
    
    for timeframe, data in all_timeframes_data.items():
        if 'df' in data:
            df = data['df']
            score, score_details, items = calculate_technical_confirmation(df, timeframe, entry_price)
            timeframe_scores[timeframe] = {
                'score': score,
                'details': score_details,
                'items': items
            }
            all_confirmation_items.extend([f"{timeframe}: {item}" for item in items])
    
    # Calculate weighted average (weekly gets more weight)
    weights = {'1W': 0.4, '1D': 0.35, '4H': 0.25}
    weighted_sum = 0
    weight_total = 0
    
    for tf, weight in weights.items():
        if tf in timeframe_scores:
            weighted_sum += timeframe_scores[tf]['score'] * weight
            weight_total += weight
    
    if weight_total > 0:
        final_ta_score = weighted_sum / weight_total
    else:
        final_ta_score = 0
    
    # Display the main metric
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Color code based on score
        if final_ta_score >= 70:
            color = "green"
            delta_color = "normal"
            label = "STRONG CONFIRMATION"
        elif final_ta_score >= 50:
            color = "orange"
            delta_color = "off"
            label = "MODERATE CONFIRMATION"
        else:
            color = "red"
            delta_color = "inverse"
            label = "WEAK CONFIRMATION"
        
        st.metric(
            label=label,
            value=f"{final_ta_score:.0f}/100",
            delta=f"{'Bullish' if final_ta_score >= 60 else 'Neutral' if final_ta_score >= 40 else 'Bearish'}",
            delta_color=delta_color
        )
    
    with col2:
        # Score breakdown by timeframe
        st.markdown("**Timeframe Scores:**")
        for tf in ['1W', '1D', '4H']:
            if tf in timeframe_scores:
                score = timeframe_scores[tf]['score']
                color = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
                st.write(f"{color} {tf}: {score:.0f}/100")
    
    with col3:
        # Quick summary
        bull_count = sum(1 for item in all_confirmation_items if "✅" in item)
        warning_count = sum(1 for item in all_confirmation_items if "⚠️" in item)
        bear_count = sum(1 for item in all_confirmation_items if "❌" in item)
        
        st.markdown("**Signal Summary:**")
        st.write(f"✅ Bullish: {bull_count}")
        st.write(f"⚠️ Warning: {warning_count}")
        st.write(f"❌ Bearish: {bear_count}")
    
    # Progress bar
    st.progress(final_ta_score / 100)
    
    # Detailed breakdown in expander
    with st.expander("📋 Detailed Technical Confirmation Breakdown", expanded=False):
        for timeframe in ['1W', '1D', '4H']:
            if timeframe in timeframe_scores:
                st.markdown(f"### **{timeframe} Timeframe**")
                st.write(f"**Overall Score: {timeframe_scores[timeframe]['score']:.0f}/100**")
                
                # Display score details
                for category, score in timeframe_scores[timeframe]['details'].items():
                    col_a, col_b = st.columns([3, 1])
                    col_a.write(category)
                    col_b.progress(score / 100)
                    col_b.write(f"{score:.0f}")
                
                # Display confirmation items
                st.markdown("**Confirmation Items:**")
                for item in timeframe_scores[timeframe]['items']:
                    st.write(f"- {item}")
                
                st.markdown("---")
    
    # Interpretation
    st.markdown("### 🎯 Interpretation")
    
    if final_ta_score >= 75:
        st.success("""
        **STRONG TECHNICAL CONFIRMATION** - The current price shows excellent alignment with technical indicators across all timeframes.
        - ✅ Multiple bullish confirmations
        - ✅ Strong trend alignment
        - ✅ Good risk/reward setup
        """)
    elif final_ta_score >= 60:
        st.warning("""
        **MODERATE TECHNICAL CONFIRMATION** - Technical alignment is acceptable but has some concerns.
        - ⚠️ Mixed signals across timeframes
        - ⚠️ Some indicators need improvement
        - ✅ Overall setup is workable
        """)
    elif final_ta_score >= 40:
        st.error("""
        **WEAK TECHNICAL CONFIRMATION** - Significant technical concerns exist.
        - ❌ Multiple bearish signals
        - ❌ Poor alignment with key indicators
        - ⚠️ Consider waiting for better setup
        """)
    else:
        st.error("""
        **POOR TECHNICAL CONFIRMATION** - Strong technical warnings.
        - ❌ Most indicators are bearish
        - ❌ Price action is weak
        - ❌ Avoid entry at current price
        """)

def increase_patience_15():
    st.session_state.patience_score = min(100, st.session_state.patience_score + 15)

def increase_patience_10():
    st.session_state.patience_score = min(100, st.session_state.patience_score + 10)

def main():
    st.title("📊 Entry Position Analyzer")
    st.write("Analyze your entry position using ML models trained on 4H, 1D, and 1W timeframes. Type ticker: e.g. TSLA or BTC-USD. Or find ticker name on yahoo finance.")

    with st.expander("Disciplined Entry and Exit Strategy (Expand and learn)", expanded=False):
        st.write(desc)
    
    # ========== AUTOMATIC TECHNICAL CHECKLIST ==========
    st.markdown("### ⚙️ **Automatic Technical Analysis Checklist**")
    
    with st.expander("✅ Technical conditions will be automatically evaluated after analysis", expanded=False):
        st.info("""
        **The system will automatically check:**
        1. **Price vs Moving Averages** - Is price above key EMAs?
        2. **RSI Conditions** - Is RSI in optimal range and above its SMA?
        3. **Volume Confirmation** - Is volume supporting the move?
        4. **Trend Strength (ADX)** - Is there a clear trend?
        5. **Directional Indicators** - Is +DI > -DI for bullish momentum?
        6. **Price Action** - Is the current candle bullish?
        
        **Scoring:**
        - 🟢 **70+**: Strong technical confirmation
        - 🟡 **50-70**: Moderate confirmation  
        - 🔴 **<50**: Weak technical setup
        """)
        
        st.caption("💡 **Note**: This is an AUTOMATIC analysis based on current price as entry price. Manual checklist removed for objectivity.")
    # ========== END AUTOMATIC CHECKLIST ==========
    
    initialize_session_state()
    
    # Main input columns
    col1, col2, col3 = st.columns(3)

    with col1:
        # ticker input with callback to update prices
        ticker = st.text_input(
            "Ticker Symbol",
            value="TSLA",
            key="ticker",
            on_change=update_price_and_reset_entry,
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
        if result["valid"] and not st.session_state.initial_prices_set:
            current_price = get_current_price(ticker)
            if current_price is not None:
                st.session_state.current_price = current_price
                st.session_state.entry_price = current_price
                st.session_state.entry_price_input = current_price
                st.session_state.initial_prices_set = True
                st.session_state.previous_ticker = ticker
        
        # Fallback for display
        current_display = st.session_state.current_price or 0.0
        st.metric("Current Price", f"${current_display:.2f}")
        
        # Auto-set entry price to current price
        default_entry = current_display
        
        entry_price = st.number_input(
            "Entry Price (Auto-set to current)",
            min_value=0.0,
            value=float(default_entry),
            step=0.1,
            key="entry_price_input",
            on_change=update_entry_price,
            help="Automatically set to current price for technical analysis"
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
        
        # Entry delay reminder
        st.markdown("---")
        st.markdown("**⏱️ Patience Reminder**")
        st.info("Wait for Technical Analysis Confirmation score before entering")

    opt = ['OBV', 'CCI', 'CMF', 'MFI', 'ADX']
    default_option_index = 0 
    
    ind = st.selectbox(
        "Choose 3rd indicator:",
        opt,
        index=default_option_index,
        key="indicator_select"
    )

    # Monte Carlo settings
    if "mc_days" not in st.session_state: 
        st.session_state.mc_days = 90 
    if "mc_sims" not in st.session_state: 
        st.session_state.mc_sims = 10000
    if "mc_method" not in st.session_state:
        st.session_state.mc_method = 1
    
    # Sliders with session state
    days = st.slider(
        "Forecast Days", 
        min_value=30, max_value=365, 
        value=st.session_state.mc_days,
        key="mc_days_slider"
    )
    
    num_sims = st.slider(
        "Monte Carlo Simulations", 
        min_value=1000, max_value=20000, 
        value=st.session_state.mc_sims,
        key="mc_sims_slider"
    )
    
    # Radio with session state
    mc_method = st.radio(
        "Monte Carlo Method",
        ["Random Statistical Simulation", "Historical Paths Simulation"],
        index=st.session_state.mc_method,
        key="mc_method_radio"
    )
    
    # Update session state
    st.session_state.mc_days = days
    st.session_state.mc_sims = num_sims
    st.session_state.mc_method = ["Random Statistical Simulation", "Historical Paths Simulation"].index(mc_method)
    
    # ========== SIMPLIFIED PATIENCE METER ==========
    st.markdown("---")
    st.markdown("### 🧘 **Trading Discipline**")
    
    if "patience_score" not in st.session_state:
        st.session_state.patience_score = 50  # Start at neutral
    
    patience_level = st.session_state.patience_score
    
    # Simple patience indicator
    if patience_level < 40:
        st.error(f"**Impulsive Tendency Detected** ({patience_level}/100)")
        st.write("Consider waiting for Technical Analysis Confirmation before entering")
    elif patience_level < 70:
        st.warning(f"**Moderate Discipline** ({patience_level}/100)")
        st.write("Good, but wait for all confirmations")
    else:
        st.success(f"**Patient Trader** ({patience_level}/100)")
        st.write("Excellent discipline - follow the Technical Analysis Confirmation")
    
    st.progress(patience_level / 100)
    
    # Quick discipline buttons
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.button(
            "⏸️ I'll wait for confirmation", 
            key="wait_btn",
            on_click=increase_patience_15,
            use_container_width=True
        )
    
    with col_d2:
        st.button(
            "📊 Analyze first, decide later", 
            key="analyze_first",
            on_click=increase_patience_10,
            use_container_width=True
        )
    
    # ========== ANALYSIS BUTTON ==========
    st.markdown("---")
    
    col_btn1, col_btn2 = st.columns([2, 1])
    
    with col_btn1:
        analyze_button = st.button("📊 Analyze Entry Position", 
                                 key="analyze_main",
                                 use_container_width=True)
    
    with col_btn2:
        reset_btn = st.button("🔄 Reset Analysis", 
                            help="Clear previous analysis to start fresh",
                            use_container_width=True,
                            key="reset_analysis")
    
    if reset_btn:
        st.session_state.last_analysis_time = None
        st.session_state.patience_score = 50
        st.rerun()
    
    if analyze_button:
        with st.spinner("Training models and analyzing..."):
            try:
                # Update last analysis time
                st.session_state.last_analysis_time = datetime.now()
                
                # Update patience score positively for analyzing
                st.session_state.patience_score = min(100, st.session_state.patience_score + 5)
                
                # Get current date/time
                end_date = datetime.now()

                results = {}
                all_timeframes_data = {}  # Store data for technical confirmation

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

                    if timeframe == "1D":
                        daily_df = df.copy()

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
                        
                        # Display warnings
                        warning_count = display_entry_warnings(current_price, entry_price, prediction)
                        
                        # Calculate entry score
                        entry_score, score_details = calculate_entry_score(prediction, df, timeframe)
                        
                        results[timeframe] = {
                            "prediction": prediction,
                            "assessment": assessment,
                            "reasons": reasons,
                            "df": df,
                            "score": entry_score,
                            "score_details": score_details
                        }
                        
                        # Store for technical confirmation
                        all_timeframes_data[timeframe] = {
                            "df": df,
                            "prediction": prediction,
                            "assessment": assessment
                        }

                        # Display ML results
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
                            st.metric("ML Confidence", f"{prediction['confidence']:.1f}%")

                        if assessment == "Valid":
                            st.success(f"**ML Assessment**: {assessment}")
                        elif assessment == "Risky":
                            st.warning(f"**ML Assessment**: {assessment}")
                        else:
                            st.error(f"**ML Assessment**: {assessment}")

                        st.write(f"**Reasons**: {reasons}")
                        avg_bull, avg_bear = avg_bull_bear_lengths(df)
                        st.write(f"Average Bull Market, {timeframe}: {avg_bull:.0f}, Average Bear Market {timeframe}: {avg_bear:.0f}")
                
                        fig = plot_analysis(ticker, df, entry_price, timeframe, assessment, prediction, ind=ind)
                        st.pyplot(fig)
                        
                    else:
                        st.warning(f"Could not generate prediction for {timeframe}")

                    st.write("---")

                # ========== DISPLAY TECHNICAL ANALYSIS CONFIRMATION METRIC ==========
                if all_timeframes_data:
                    display_technical_confirmation_metric(all_timeframes_data, entry_price, ticker)
                
                # Overall recommendation
                if results:
                    st.subheader("🎯 Overall ML Recommendation")

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
                
                    annotation = f"(Avg R/R: {avg_rr:.2f}, Avg ML Conf: {avg_conf:.1f}%)"
                
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
                        "Wait and See": "🔴 Wait and See",
                        "Not Recommended": "🔴 Not Recommended"
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
                            "ML Conf (%)": round(pred['confidence'], 1),
                            "Hits": pred['will_hit'],
                            "R/R": round(rr, 2),
                            "Score": round(res['score'], 0),
                            "Assessment": display_assessment
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    summary_df = summary_df[["Timeframe", "Price ($)", "TP ($)", "SL ($)", "ML Conf (%)", "Hits", "R/R", "Score", "Assessment"]]
                                               
                    st.write(f"**ML Timeframe Summary ({ticker}):**")
                    st.dataframe(summary_df)
                    
                    # ------------------------
                    # Monte Carlo Simulation
                    # ------------------------
                    st.header("📈 Monte Carlo Simulation of Entry")
                    
                    returns = daily_df["Close"].pct_change().dropna()
                    mu = returns.mean() * 252
                    sigma = returns.std() * np.sqrt(252)
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Annualized Return (Statistical)", f"{mu*100:.1f}%")
                    c2.metric("Annualized Volatility", f"{sigma*100:.1f}%")
                    
                    # Cache the path generation
                    @st.cache_data(show_spinner=False)
                    def mc_gbm_paths(current_price, mu, sigma, days, num_sims):
                        dt = 1 / 252
                        paths = np.zeros((days + 1, num_sims))
                        paths[0] = current_price
                        for t in range(1, days + 1):
                            rand = np.random.standard_normal(num_sims)
                            paths[t] = paths[t - 1] * np.exp(
                                (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rand
                            )
                        return paths

                    @st.cache_data(show_spinner=False)
                    def mc_bootstrap_paths(current_price, returns, days, num_sims):
                        paths = np.zeros((days + 1, num_sims))
                        paths[0] = current_price
                        for i in range(num_sims):
                            resampled_returns = np.random.choice(returns.values, size=days)
                            price_factors = 1 + resampled_returns
                            paths[1:, i] = current_price * price_factors.cumprod()
                        return paths
                    
                    current_day_price = daily_df["Close"].iloc[-1]
                    
                    if mc_method == "Random Statistical Simulation":
                        paths = mc_gbm_paths(
                            current_price=current_day_price,
                            mu=mu,
                            sigma=sigma,
                            days=st.session_state.mc_days,
                            num_sims=st.session_state.mc_sims
                        )
                    elif mc_method == "Historical Paths Simulation":
                        paths = mc_bootstrap_paths(
                            current_price=current_day_price,
                            returns=returns,
                            days=st.session_state.mc_days,
                            num_sims=st.session_state.mc_sims
                        )

                    # Path calculation
                    final_prices = paths[-1]
                    
                    # Plot paths and distribution
                    fig3, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 9), height_ratios=[3, 1])
                    
                    sample_paths = min(50, num_sims)
                    for i in range(sample_paths):
                        ax3.plot(range(days + 1), paths[:, i], color="gray", alpha=0.3, linewidth=0.5)
    
                    mean_path_all = paths.mean(axis=1) 
                    ax3.plot(range(days + 1), mean_path_all, color="red", linewidth=2, linestyle='--', label="Expected Path")
                        
                    percentiles = np.percentile(paths[-1], [5, 25, 50, 75, 95])
                    
                    ax3.axhline(percentiles[2], color="red", linestyle=":", linewidth=1, label=f"Final Median: ${percentiles[2]:.2f}")
                    ax3.axhline(entry_price, color="black", linestyle="-.", linewidth=2, label= f"Entry: ${entry_price:.2f}")
                    ax3.set_title(f"Monte Carlo Price Simulation ({mc_method})", fontsize=14, fontweight="bold")
                    ax3.set_xlabel("Days")
                    ax3.set_ylabel("Price ($)")
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
                    
                    ax4.hist(paths[-1], bins=50, alpha=0.7, color="skyblue", edgecolor="black", density=True)
                    ax4.axvline(percentiles[2], color="red", linestyle="--", linewidth=2, label=f"Median: ${percentiles[2]:.2f}")
                    ax4.axvline(entry_price, color="black", linestyle="-.", linewidth=2, label= f"Entry: ${entry_price:.2f}")
                    
                    ax4.set_title("Final Price Distribution (Density)")
                    ax4.set_xlabel("Price ($)")
                    ax4.legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig3)
                    plt.close(fig3)
                    
                    # Final probability calculations
                    final_prices = paths[-1]
                    prob_profit = np.mean(final_prices > entry_price) * 100
                    prob_loss = np.mean(final_prices < entry_price) * 100

                    st.subheader("Probability vs. Entry Price")
                    
                    col_p1, col_p2 = st.columns(2)
                    
                    col_p1.metric(
                        "Chance of Profit", 
                        f"{prob_profit:.1f}%",
                        delta_color="normal"
                    )
                    col_p2.metric(
                        "Chance of Loss", 
                        f"{prob_loss:.1f}%",
                        delta_color="inverse"
                    )

                else:
                    st.error("No successful analyses completed. Try with a different ticker or time period.")

            except Exception as e:
                st.error(f"Error analyzing {ticker}: {str(e)}")
                st.info("Try with a different ticker or check if market is open")

    with st.expander("How to use this analyzer"):
        st.write(
            """
        1. **Enter Ticker Symbol**: Stock symbol (e.g., AAPL, TSLA, NVDA) or crypto (BTC-USD)
        2. **Entry Price is Auto-set**: System automatically uses current price for technical analysis
        3. **Define Expectations**: Your target gain and maximum acceptable loss (Conservative 2-5%, aggressive 5-12%)
        4. **Click Analyze**: The system will train ML models and evaluate your entry
        5. **Check Technical Analysis Confirmation**: Wait for the automatic technical confirmation score after all timeframes are plotted
        6. **Follow the Signals**: 
           - 🟢 **70+ TA Score + 🟢 ML Valid** = Strong entry signal
           - 🟡 **50-70 TA Score + 🟡 ML Risky** = Cautious entry
           - 🔴 **<50 TA Score + 🔴 ML Wait** = Avoid entry

        **Technical Analysis Confirmation Checks:**
        - **Price vs Moving Averages**: Is price above key EMAs?
        - **RSI Conditions**: Is RSI in optimal range (50-70)?
        - **Volume Confirmation**: Is volume supporting the move?
        - **Trend Strength (ADX)**: Is there a clear trend (>25)?
        - **Directional Indicators**: Is +DI > -DI for bullish momentum?
        - **Price Action**: Is the current candle bullish?

        **Key Metrics to Watch:**
        - **ML Confidence**: Machine learning model's confidence in prediction
        - **Technical Analysis Confirmation**: Automated technical indicator alignment
        - **Risk/Reward Ratio**: Should be >1.5 for good trades
        - **Probability of Profit**: From Monte Carlo simulation

        **Golden Rule**: Only enter when **BOTH** ML Assessment is "Valid" **AND** Technical Analysis Confirmation is "Strong" (>70).
        """
        )


if __name__ == "__main__":
    main()
