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
warnings.filterwarnings('ignore')


# Configuration
st.set_page_config(page_title="Entry Position Analyzer", layout="wide")

# Global Parameters - Adjusted for different timeframes
YEARS_OF_DATA = {
    '1H': 1,    # 1 year for hourly
    '1D': 2,    # 2 years for daily  
    '1W': 5     # 5 years for weekly (minimum for sufficient data points)
}
PROFIT_TARGET = 0.0375
STOP_LOSS = 0.0375
_DAYS = 28
_Nr = 30  # Reduced minimum data requirement
windows = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29] # For calculating returns

# Simplified features for faster processing
FEATURES = [
    # Price High, Low
    'High', 'Low',
    
    # Technical Indicators
    'RSI', 'RSI_SMA', 'CCI', '+DI', '-DI', 'ADX', 'ATR', 'VI+', 'KCu', 'KCl', 'KCu_outer', 'KCl_outer', 'Kasym', 'Kcount', 'STu', 'STl',

    # Moving Averages & Bands
    'SMA1', 'SMA2', 'SMA3', 'SMA_Ratio', 'Upper_Band', 'Lower_Band', 'Volume_MA20', 'SMIIO', 'SMIIO_Signal', 'SMIIO_Osc', 'MACD', 'Signal_Line',

    # Returns & Volatility
    'return1', 'return2', 'return3', 'Volatility', 'Scaled_Volatility', 'DD',

    # Volume Features
    'sumBuyVol', 'sumSellVol', 'vSpike', 'VPT', 'OBV', 'MFI', 'VWMA', 'CMF',

    # Candlestick Patterns
    'Candlesticks', 'gapStrength',

    # Market Sentiment & Signals
    'Bear', 'Bull', 'Short', 'Hold', 'Neutral', 'StrongBull', 'StrongBear', 'Neutral', 'Exhaustion',

    # PIVOTS
    'PP_Avg', 'R1_Avg', 'R2_Avg', 'S1_Avg', 'S2_Avg'
]

label2str = {0: 'None', 1: 'SL', 2: 'TP', 3: 'Hold', 4: 'Short'}
expected_classes = [0, 1, 2, 3, 4]

def get_current_price(ticker):
    stock = yf.Ticker(ticker)
    # Get historical data for 1 day (latest available)
    data = stock.history(period='1d')
    # Return the closing price of the last available trading session
    return data['Close'][-1]

def get_stock_data(ticker, start_date, end_date, interval='1d'):
    """Get stock data for given timeframe with proper date handling"""
    try:
        # Map interval names for yfinance
        interval_map = {
            '1H': '1h',
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
            sma_multiplier = 2  # Longer SMAs for weekly
            atr_period = 14
            rsi_period = 14
        else:
            # Default periods for hourly/daily
            sma_multiplier = 2
            atr_period = 14  
            rsi_period = 14
            
        df['SMA1'] = df['Close'].ewm(span=int(_DAYS * 0.5 * sma_multiplier), adjust=False).mean()
        df['SMA2'] = df['Close'].ewm(span=_DAYS * sma_multiplier, adjust=False).mean()
        df['SMA3'] = df['Close'].ewm(span=int(_DAYS * 2 * sma_multiplier), adjust=False).mean()
        df['SMA_Ratio'] = df['SMA1'] / df['SMA2']
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
        df['Upper_Band'] = df['SMA1'] + (2 * df['Close'].rolling(20).std())
        df['Lower_Band'] = df['SMA1'] - (2 * df['Close'].rolling(20).std())
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
        cols = ['SMA1', 'SMA2', 'RSI', '-DI', 'Close']
        df[cols] = df[cols].fillna(method='ffill').fillna(method='bfill')
        
        # Adjust RSI thresholds for weekly if needed
        rsi_lower = 25 if timeframe == '1W' else 18
        rsi_upper = 60 if timeframe == '1W' else 55
        
        conditions = [
            # BULL
            (
                (
                    (df['SMA1'] > df['SMA2']) &
                    (df['RSI'] >= df['RSI_SMA']) &
                    (df['RSI'].between(52, 95)) &
                    (df['+DI'] > df['-DI']) &
                    (df['+DI'].between(18, 55))
                ) &
                (
                    (df['Close'] > df['SMA1']) &
                    (df['RSI'] > df['RSI_SMA'])
                )
            ),
            # BEAR
            (
                (
                    (df['SMA1'] < df['SMA2']) &
                    (df['RSI'].between(rsi_lower, 60)) &
                    (df['RSI'] < df['RSI_SMA']) &
                    (df['+DI'] < df['-DI']) &
                    (df['-DI'].between(18, 55))
                )           
            ),
            # SHORT
            (
                (df['SMA1'] < df['SMA2']) &
                (df['RSI'].between(25, 50)) &
                (df['-DI'].between(30, 55)) &
                (df['Close'] > df['SMA1'])
            ),
            # HOLD
            (
                (
                    (df['SMA1'] > df['SMA2']) &
                    (df['RSI'] >= 50)
                ) |
                (
                    (df['RSI'] < df['RSI_SMA']) &
                    (df['ADX'].between(40, 75))
                )
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
    df['Expected_Return'] = np.nan
    close_prices = df['Close'].values
    
    # Pre-extract pivot arrays
    pivot_arrays = []
    for col in r_cols:
        if col in df.columns:
            pivot_arrays.append(df[col].values)
        else:
            pivot_arrays.append(np.full(len(df), np.nan))
    
    for i in range(len(df) - forward_window):
        current_price = close_prices[i]
        
        # Gather valid pivot values for this row
        pivots = [arr[i] for arr in pivot_arrays if not np.isnan(arr[i])]
        target_level = max(pivots) if pivots else None
        
        future_window = close_prices[i+1:i+1+forward_window]
        
        if target_level is not None:
            # Check if future price hits the pivot level
            hit = False
            for future_price in future_window:
                if future_price >= target_level:
                    df.iloc[i, df.columns.get_loc('Expected_Return')] = (target_level - current_price) / current_price
                    hit = True
                    break
            if not hit and future_window.size > 0:
                df.iloc[i, df.columns.get_loc('Expected_Return')] = (np.nanmax(future_window) - current_price) / current_price
        else:
            if future_window.size > 0:
                df.iloc[i, df.columns.get_loc('Expected_Return')] = (np.nanmax(future_window) - current_price) / current_price
            else:
                df.iloc[i, df.columns.get_loc('Expected_Return')] = np.nan
    return df

def compute_expected_loss(df, forward_window=14, s_cols=['S1', 'S2']):
    df['Expected_Loss'] = np.nan
    close_prices = df['Close'].values
    
    pivot_arrays = []
    for col in s_cols:
        if col in df.columns:
            pivot_arrays.append(df[col].values)
        else:
            pivot_arrays.append(np.full(len(df), np.nan))
    
    for i in range(len(df) - forward_window):
        current_price = close_prices[i]
        
        pivots = [arr[i] for arr in pivot_arrays if not np.isnan(arr[i])]
        target_level = min(pivots) if pivots else None
        
        future_window = close_prices[i+1:i+1+forward_window]
        
        if target_level is not None:
            hit = False
            for future_price in future_window:
                if future_price <= target_level:
                    df.iloc[i, df.columns.get_loc('Expected_Loss')] = (target_level - current_price) / current_price
                    hit = True
                    break
            if not hit and future_window.size > 0:
                df.iloc[i, df.columns.get_loc('Expected_Loss')] = (np.nanmin(future_window) - current_price) / current_price
        else:
            if future_window.size > 0:
                df.iloc[i, df.columns.get_loc('Expected_Loss')] = (np.nanmin(future_window) - current_price) / current_price
            else:
                df.iloc[i, df.columns.get_loc('Expected_Loss')] = np.nan
    return df

def label_hit_prob_past(df, window=14, profit_target=0.05, stop_loss=0.05):
    """Label hit probabilities - simplified version"""
    try:
        close_prices = df['Close'].values
        N = len(close_prices)
        labels = []
        
        for i in range(N):
            if i >= N - window:
                labels.append(0)  # Not enough future data
                continue
                
            current_price = close_prices[i]
            tp_price = current_price * (1 + profit_target)
            sl_price = current_price * (1 - stop_loss)
            
            future_prices = close_prices[i+1:i+1+window]
            
            # Check which hits first
            tp_hit = False
            sl_hit = False
            
            for price in future_prices:
                if price >= tp_price:
                    tp_hit = True
                    break
                if price <= sl_price:
                    sl_hit = True
                    break
            
            if tp_hit and not sl_hit:
                labels.append(2)  # TP hit
            elif sl_hit and not tp_hit:
                labels.append(1)  # SL hit
            else:
                labels.append(0)  # Neither hit
        
        df['Hit_Label'] = labels
        return df
        
    except Exception as e:
        st.error(f"Error labeling hit probabilities: {str(e)}")
        return df

def train_models(df, timeframe):
    """Train ML models for the given timeframe"""
    try:
        # Check for required columns
        required_cols = FEATURES + ['Hit_Label', 'Expected_Return', 'Expected_Loss']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.warning(f"Missing columns for {timeframe}: {missing_cols}")
            return None, None, None, None, None, None
        
        df_model = df.dropna(subset=required_cols)
        
        # Adjust minimum data requirement based on timeframe
        min_data = {
            '1H': 100,
            '1D': 50, 
            '1W': 30   # Weekly needs fewer data points due to longer timeframe
        }
        
        required_min = min_data.get(timeframe, _Nr)
        
        if len(df_model) < required_min:
            st.warning(f"Insufficient data for {timeframe} modeling: {len(df_model)} rows (need {required_min})")
            return None, None, None, None, None, None
        
        # Progress indicator
        progress_text = f"Training {timeframe} models..."
        progress_bar = st.progress(0)
        
        # Classifier for Hit Label
        X_cls = df_model[FEATURES]
        y_cls = df_model['Hit_Label'].astype(int)
        
        scaler_cls = StandardScaler()
        X_scaled_cls = scaler_cls.fit_transform(X_cls)
        progress_bar.progress(25)
        
        model_class = RandomForestClassifier(
            n_estimators=50,  # Reduced for speed
            max_depth=8, 
            random_state=42,
            n_jobs=-1
        )
        model_class.fit(X_scaled_cls, y_cls)
        progress_bar.progress(50)
        
        # Get class probabilities
        cls_probs = model_class.predict_proba(X_scaled_cls)
        prob_df = pd.DataFrame(0, index=df_model.index, 
                              columns=[f'Prob_Class_{c}' for c in expected_classes])
        
        for i, c in enumerate(model_class.classes_):
            if c in expected_classes:
                prob_df[f'Prob_Class_{c}'] = cls_probs[:, i]
        
        # Prepare features with probabilities
        FEATURES_with_probs = FEATURES + [f'Prob_Class_{c}' for c in expected_classes]
        X_reg = pd.concat([df_model[FEATURES], prob_df], axis=1)
        
        # Return model
        y_return = df_model['Expected_Return']
        scaler_return = StandardScaler()
        X_scaled_return = scaler_return.fit_transform(X_reg[FEATURES_with_probs])
        
        model_return = RandomForestRegressor(
            n_estimators=50,  # Reduced for speed
            max_depth=8, 
            random_state=42,
            n_jobs=-1
        )
        model_return.fit(X_scaled_return, y_return)
        progress_bar.progress(75)
        
        # Loss model
        y_loss = df_model['Expected_Loss']
        scaler_loss = StandardScaler()
        X_scaled_loss = scaler_loss.fit_transform(X_reg[FEATURES_with_probs])
        
        model_loss = RandomForestRegressor(
            n_estimators=50,  # Reduced for speed
            max_depth=8, 
            random_state=42,
            n_jobs=-1
        )
        model_loss.fit(X_scaled_loss, y_loss)
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
        
        # Confidence score
        ratio = (predicted_return / abs(predicted_loss)) if (will_hit != 'None' and predicted_loss != 0) else 0
        ratio = max(ratio, 0)
        #confidence_score = max((hit_prob/100) * ratio, 0) * 100
        confidence_score = min(max((hit_prob/100) * ratio, 0) * 100, 100)
        
        return {
            'will_hit': will_hit,
            'hit_prob': hit_prob,
            'predicted_tp': predicted_tp,
            'predicted_sl': predicted_sl,
            'predicted_return': predicted_return * 100,
            'predicted_loss': predicted_loss * 100,
            'confidence': confidence_score,
            'current_price': current_price
        }
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def plot_analysis(ticker, df, entry_price, timeframe, assessment):
    """Create analysis plot"""

    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                       gridspec_kw={'height_ratios': [3, 1]}, 
                                       sharex=True)
        
        # Price plot
        price = df['Close'].rolling(2).mean()
        ax1.plot(df.index, price, label='Price', color='gray', alpha=0.5, linewidth=1)
        
        # SMAs if available
        if 'SMA1' in df.columns:
            ax1.plot(df.index, df['SMA1'], label=f'SMA{int(_DAYS*0.5)}', color='blue', alpha=0.3, linewidth=1)
        if 'SMA2' in df.columns:
            ax1.plot(df.index, df['SMA2'], label=f'SMA{int(_DAYS*2)}', color='red', alpha=0.3, linewidth=1)
        
        # Entry point
        last_date = df.index[-1]
        ax1.plot(last_date, entry_price, '^', markersize=10, color='green', 
                 label=f'Entry: ${entry_price:.2f}')
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        ax1.set_ylabel('Price')
        ax1.legend(loc='lower left', fontsize='x-small')
        ax1.grid(True, alpha=0.3)
        
        # Assessment annotation
        color_map = {'Valid': 'green', 'Risky': 'orange', 'Not Recommended': 'red'}
        assessment_color = color_map.get(assessment, 'gray')
        
        ax1.annotate(f'Assessment: {assessment}', 
                    xy=(0.02, 0.95), xycoords='axes fraction',
                    fontsize=12, weight='bold',
                    bbox=dict(boxstyle='round', facecolor=assessment_color, alpha=0.3))
        
        # Add ticker name in the middle
        ax1.text(0.5, 0.5, f'@{ticker}', transform=ax1.transAxes, 
                     fontsize=50, color='grey', alpha=0.2,
                     horizontalalignment='center', verticalalignment='center',
                     rotation=0, weight='bold', style='italic')    

        # RSI plot if available
        if 'RSI' in df.columns:
            rsi_ = df['RSI'].rolling(3).mean()
            rsi_sma = df['RSI'].rolling(20).mean()
            ax2.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.3)
            ax2.plot(df.index, rsi_, label='RSI', color='gray', linewidth=1.5, alpha=0.3)
            ax2.plot(df.index, rsi_sma, label='RSI SMA', color='red', linewidth=1.5, alpha=0.35)
            ax2.fill_between(df.index, rsi_, 52, where=(df['RSI'] > 52), facecolor='green', alpha=0.15)
            ax2.fill_between(df.index, rsi_, 40, where=(df['RSI'] < 40), facecolor='red', alpha=0.15)
            ax2.fill_between(df.index, rsi_, rsi_sma, where=((df['RSI'] < df['RSI_SMA']) & (df.SMA1 > df.SMA2)), facecolor='orange', alpha=0.3, label='Dip(?)')
            ax2.axhline(70, color='red', linestyle='--', alpha=0.3, label='Overbought')
            ax2.axhline(30, color='green', linestyle='--', alpha=0.3, label='Oversold')
            ax2.axhline(50, color='gray', linestyle='-', alpha=0.3)

            ax2.scatter(df.index[df['Bull'] == 1], rsi_[df['Bull'] == 1], color='green', marker='^', s=5, alpha=0.4, label='Bull', zorder=7)
            ax2.scatter(df.index[df['Bear'] == 1], rsi_[df['Bear'] == 1], color='red', marker='v', s=5, alpha=0.4, label='Bear', zorder=8)
            ax2.scatter(df.index[df['Short'] == 1], rsi_[df['Short'] == 1], color='red', marker='x', s=5, alpha=0.4, label='Short', zorder=10)
            ax2.scatter(df.index[df['Hold'] == 1], rsi_[df['Hold'] == 1], color='orange', marker='o', s=5, alpha=0.4, label='Hold', zorder=10)
            ax2.yaxis.set_label_position("right")
            ax2.yaxis.tick_right()
            ax2.set_ylabel('RSI')
            ax2.set_ylim(0, 100)
            ax2.legend(loc='lower left', fontsize='x-small')
        else:
            ax2.text(0.5, 0.5, 'RSI data not available', ha='center', va='center', transform=ax2.transAxes)
        
        ax2.grid(True, alpha=0.3)
        
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
    if confidence > 50:
        reasons.append("High confidence")
    elif confidence > 30:
        reasons.append("Moderate confidence")
    else:
        reasons.append("Low confidence")
    
    # Risk-Reward check
    user_rr = user_gain / abs(user_loss) if user_loss != 0 else 0
    pred_rr = pred_return / abs(pred_loss) if pred_loss != 0 else 0
    
    if pred_rr >= 2:
        reasons.append("Good risk-reward ratio")
    elif pred_rr >= 1:
        reasons.append("Moderate risk-reward ratio")
    else:
        reasons.append("Poor risk-reward ratio")
    
    # Price proximity
    if price_diff_pct > 7:
        reasons.append("Entry price far from current price")
    elif price_diff_pct > 5:
        reasons.append("Entry price moderately different")
    else:
        reasons.append("Entry price close to current")
    
    # Overall assessment
    bullish_conditions = (will_hit == 'TP' and hit_prob > 40 and confidence > 40 and pred_rr > 1.5)
    risky_conditions = (will_hit in ['TP', 'Hold'] and confidence > 30 and pred_rr > 1)
    
    if bullish_conditions and price_diff_pct <= 5:
        assessment = "Valid"
    elif risky_conditions and price_diff_pct <= 7:
        assessment = "Risky"
    else:
        assessment = "Not Recommended"
    
    return assessment, " | ".join(reasons)

# Streamlit App
def main():
    st.title("📊 Entry Position Analyzer")
    st.write("Analyze your entry position using ML models trained on 1H, 1D, and 1W timeframes. Type ticker: e.g. TSLA or BTC-USD. Or find ticker name on yahoo finance.")
    
    # User inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ticker = st.text_input("Ticker Symbol", "TSLA").upper()
    
    with col2:
        price = get_current_price(ticker)
        entry_price = st.number_input("Entry Price ($)", min_value=0.01, value=price, step=0.1)
    
    with col3:
        user_gain = st.number_input("Expected Gain (%)", min_value=0.1, value=3.75, step=0.1)
        user_loss = st.number_input("Expected Loss (%)", min_value=0.1, value=3.75, step=0.1)
    
    if st.button("Analyze Entry Position"):
        with st.spinner("Training models and analyzing..."):
            try:
                # Get current date
                end_date = datetime.now()
                
                results = {}
                
                # Analyze all timeframes
                timeframes = [
                    ("1H", "1H"),
                    ("1D", "1D"), 
                    ("1W", "1W")
                ]
                
                for timeframe, interval in timeframes:
                    st.subheader(f"{timeframe} Timeframe Analysis")
                    
                    # Get appropriate start date based on timeframe
                    years = YEARS_OF_DATA[timeframe]
                    start_date = end_date - timedelta(days=365 * years)
                    
                    # Get data
                    with st.spinner(f"Fetching {timeframe} data..."):
                        df = get_stock_data(ticker, start_date, end_date, interval)
                    
                    if df is None:
                        st.warning(f"No data available for {timeframe} timeframe")
                        continue
                        
                    if len(df) < 30:
                        st.warning(f"Insufficient {timeframe} data for {ticker}: {len(df)} rows")
                        continue
                    
                    st.write(f"Data points: {len(df)}")
                    
                    # Add technical indicators with timeframe-specific adjustments
                    with st.spinner("Calculating technical indicators..."):
                        df = add_technical_indicators(df, timeframe)
                        df = add_pivot_levels(df, window=14)
                        df = add_pivots(df, windows)
                        df = average_pivots(df, windows)
                    
                    if df is None:
                        st.warning(f"Error calculating indicators for {timeframe}")
                        continue
                        
                    # Compute expected returns/losses
                    with st.spinner("Computing expected returns..."):
                        df = compute_expected_return(df, forward_window=14, r_cols=['R1_Avg', 'R2_Avg'])
                        df = compute_expected_loss(df, forward_window=14, s_cols=['S1_Avg', 'S2_Avg'])
                    
                    # Label hit probabilities
                    with st.spinner("Labeling hit probabilities..."):
                        df = label_hit_prob_past(df, profit_target=user_gain/100, stop_loss=user_loss/100)
                    
                    # Train models
                    with st.spinner(f"Training {timeframe} ML models..."):
                        models = train_models(df, timeframe)
                    
                    if models[0] is None:
                        st.warning(f"Could not train models for {timeframe} timeframe")
                        continue
                    
                    model_class, model_return, model_loss, scaler_cls, scaler_return, scaler_loss = models
                    
                    # Make prediction
                    latest_data = df.iloc[[-1]]
                    prediction = make_prediction(model_class, model_return, model_loss, 
                                               scaler_cls, scaler_return, scaler_loss, latest_data)
                    
                    if prediction:
                        current_price = prediction['current_price']
                        assessment, reasons = assess_entry(prediction, user_gain, user_loss, entry_price, current_price)
                        
                        # Store results
                        results[timeframe] = {
                            'prediction': prediction,
                            'assessment': assessment,
                            'reasons': reasons,
                            'df': df
                        }
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Current Price", f"${current_price:.2f}")
                            st.metric("Predicted TP", f"${prediction['predicted_tp']:.2f}")
                            st.metric("Predicted SL", f"${prediction['predicted_sl']:.2f}")
                            
                        with col2:
                            st.metric("Will Hit", prediction['will_hit'])
                            st.metric("Hit Probability", f"{prediction['hit_prob']:.1f}%")
                            st.metric("Confidence", f"{prediction['confidence']:.1f}%")
                        
                        # Assessment with color
                        if assessment == "Valid":
                            st.success(f"**Assessment**: {assessment}")
                        elif assessment == "Risky":
                            st.warning(f"**Assessment**: {assessment}")
                        else:
                            st.error(f"**Assessment**: {assessment}")
                            
                        st.write(f"**Reasons**: {reasons}")
                        
                        # Plot
                        fig = plot_analysis(ticker, df, entry_price, timeframe, assessment)
                        st.pyplot(fig)
                    else:
                        st.warning(f"Could not generate prediction for {timeframe}")
                    
                    st.write("---")
                
                # Overall recommendation
                if results:
                    st.subheader("🎯 Overall Recommendation")
                    
                    assessments = [results[tf]['assessment'] for tf in results.keys()]
                    valid_count = assessments.count('Valid')
                    risky_count = assessments.count('Risky')
                    
                    total_timeframes = len(results)
                    
                    if valid_count == total_timeframes:
                        st.success("**STRONG BUY** - All timeframes show valid entry")
                    elif valid_count >= total_timeframes / 2:
                        st.success("**BUY** - Majority of timeframes show valid entry")
                    elif valid_count >= 1 or risky_count >= total_timeframes / 2:
                        st.warning("**CAUTIOUS BUY** - Mixed or risky signals")
                    else:
                        st.error("**AVOID** - Poor signals across timeframes")
                        
                    # Show timeframe summary
                    st.write(f"**Timeframe Summary ({ticker}):**")
                    for tf in results.keys():
                        assessment = results[tf]['assessment']
                        color = "🟢" if assessment == "Valid" else "🟡" if assessment == "Risky" else "🔴"
                        st.write(f"{color} {tf}: {assessment}")
                        
                else:
                    st.error("No successful analyses completed. Try with a different ticker or time period.")
                        
            except Exception as e:
                st.error(f"Error analyzing {ticker}: {str(e)}")
                st.info("Try with a different ticker or check if market is open")

    # Instructions
    with st.expander("How to use this analyzer"):
        st.write("""
        1. **Enter Ticker Symbol**: Stock symbol (e.g., AAPL, TSLA, NVDA) or crypto (BTC-USD)
        2. **Set Entry Price**: Your intended entry price
        3. **Define Expectations**: Your target gain and maximum acceptable loss
        4. **Click Analyze**: The system will train ML models and evaluate your entry
        
        **Timeframe Data Requirements:**
        - **1H**: 1 year of historical data (~2000+ data points)
        - **1D**: 2 years of historical data (~500+ data points)  
        - **1W**: 5 years of historical data (~250+ data points)
        
        **Assessment Colors:**
        - 🟢 **Valid**: Good entry with strong bullish signals
        - 🟡 **Risky**: Moderate signals, proceed with caution  
        - 🔴 **Not Recommended**: Poor risk-reward or bearish signals
        
        **The analysis considers:**
        - ML predictions for TP/SL hits
        - Risk-reward ratios
        - Technical indicator alignment
        - Price proximity to current levels
        - Confidence scores from ensemble models
        
        **Note**: 1H data may not be available for all tickers outside market hours.
        1W data requires at least 5 years of history for sufficient data points.
        """)

if __name__ == "__main__":
    main()
