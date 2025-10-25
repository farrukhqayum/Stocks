#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from imports import *
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ML Daily Entry with Weekly Trend Filter", layout="wide")

st.title("🤖 ML Daily Entry — Weekly Trend Filter with 7% TP/SL")
st.markdown("""
Strategy:
- **Weekly Trend Filter**: Only take trades when SMA10 > SMA50 on weekly timeframe
- **Daily ML Entries**: Use ML predictions for entry signals on daily timeframe  
- **Entry**: At daily close when ML predicts upward movement
- **Exit**: 7% TP or 7% SL based on daily price action
- **Risk Management**: Close trade if price jumps ±7% intraday or next day open
- **Non-overlapping**: Wait for current trade to close before taking next signal
""")

# -------------------------
# Strategy Parameters
# -------------------------
YEARS_OF_DATA = 3
PROFIT_TARGET = 0.07  # 7%
STOP_LOSS = 0.07      # 7%
_DAYS = 22
windows = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]

FEATURES = [
    'High', 'Low', 'RSI', 'RSI_SMA', 'CCI', '+DI', '-DI', 'ADX', 'ATR', 'VI+', 
    'KCu', 'KCl', 'KCu_outer', 'KCl_outer', 'Kasym', 'Kcount', 'STu', 'STl',
    'SMA1', 'SMA2', 'SMA3', 'SMA_Ratio', 'Upper_Band', 'Lower_Band', 'Volume_MA20', 
    'SMIIO', 'SMIIO_Signal', 'SMIIO_Osc', 'MACD', 'Signal_Line',
    'return1', 'return2', 'return3', 'Volatility', 'Scaled_Volatility', 'DD',
    'sumBuyVol', 'sumSellVol', 'vSpike', 'VPT', 'OBV', 'MFI', 'VWMA', 'CMF',
    'Candlesticks', 'gapStrength', 'Bear', 'Bull', 'Short', 'Hold', 'Neutral', 
    'StrongBull', 'StrongBear', 'Exhaustion', 'PP_Avg', 'R1_Avg', 'S1_Avg', 'R2_Avg', 'S2_Avg'
]

# -------------------------
# User inputs
# -------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    ticker = st.text_input("Ticker", value="COIN")
with col2:
    period = st.selectbox("History period", ["2y", "3y", "5y", "7y"], index=2)
with col3:
    TP_pct = st.number_input("TP (%)", value=7.0, step=0.5)
with col4:
    SL_pct = st.number_input("SL (%)", value=7.0, step=0.5)

# ML prediction settings
col5, col6 = st.columns(2)
with col5:
    ml_confidence_threshold = st.number_input("ML Confidence Threshold", value=0.6, step=0.1)
with col6:
    max_holding_days = st.number_input("Max Holding Days", value=30, step=5)

# -------------------------
# Your ML Functions
# -------------------------
def get_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1), 
                     interval='1d', auto_adjust=False, progress=False)
    if df.empty:
        return None
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df.dropna()
    if df.empty:
        return None
    return df

def add_technical_indicators(df):
    close = df.Close
    df['Close'] = df[['Open', 'High', 'Low', 'Close']].mean(axis=1).rolling(2).mean()
    df['SMA1'] = df['Close'].ewm(span=int(_DAYS * 0.5), adjust=False).mean()
    df['SMA2'] = df['Close'].ewm(span=_DAYS, adjust=False).mean()
    df['SMA3'] = df['Close'].ewm(span=int(_DAYS * 2), adjust=False).mean()
    df['SMA_Ratio'] = df['SMA1'] / df['SMA2']
        
    df['ATR'] = ta.calculate_atr(high=df.High, low=df.Low, close=df.Close)
    df = ta.scaled_volatility(df)
    df = ta.add_candlestickpatterns(df)

    df['RSI'] = ta.calculate_rsi(df)
    df['RSI_SMA'] = df['RSI'].rolling(14).mean()

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=24, adjust=False).mean()
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
    df['vSpike'] = np.where(df['Volume'] > 2 * df['Volume_MA20'],
                        np.where(df['Close'] > df['Open'], 1, -1), 0)
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

    df['return1'] = df['Close'].pct_change(7).rolling(3).mean()
    df['return2'] = df['Close'].pct_change(14).rolling(3).mean()
    df['return3'] = df['Close'].pct_change(21).rolling(3).mean()
    
    df['Volatility'] = df['Close'].rolling(14).std().rolling(3).mean()

    # Technical Indicator Conditions
    conditions = [
        ((df['SMA1'] > df['SMA2']) & (df['RSI'] >= df['RSI_SMA']) & 
         (df['RSI'].between(52, 95)) & (df['+DI'] > df['-DI']) & 
         (df['+DI'].between(18, 55)) & (df['Close'] > df['SMA1']) & 
         (df['RSI'] > df['RSI_SMA'])),
        
        ((df['SMA1'] < df['SMA2']) & (df['RSI'].between(18,60)) & 
         (df['RSI'] < df['RSI_SMA']) & (df['+DI'] < df['-DI']) & 
         (df['-DI'].between(18, 55))),
        
        ((df['SMA1'] < df['SMA2']) & (df['RSI'].between(25, 50)) & 
         (df['-DI'].between(30, 55)) & (df['Close'] > df['SMA1'])),
        
        (((df['SMA1'] > df['SMA2']) & (df['RSI'] >= 50)) | 
         ((df['RSI'] < df['RSI_SMA']) & (df['ADX'].between(40, 75))))
    ]

    choices = ['Bull', 'Bear', 'Short', 'Hold']
    df['TI'] = np.select(conditions, choices, default='Neutral')
    
    df['TI'] = df['TI'].astype('category')
    df_encoded = pd.get_dummies(df['TI'], prefix='', prefix_sep='')
    expected_cols = ['Bull', 'Bear', 'Short', 'Hold', 'Neutral']
    for col in expected_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df = pd.concat([df, df_encoded], axis=1)

    strongbull_condition = ((df['RSI'] > 52) & (df['ADX'] > 22) & 
                           (df['+DI'] > df['-DI']) & (df['sumBuyVol'] > df['sumSellVol']))
    strongbear_condition = ((df['RSI'] < 40) & (df['ADX'] > 22) & 
                           (df['+DI'] < df['-DI']) & (df['sumBuyVol'] < df['sumSellVol']))
    
    df['StrongBull'] = strongbull_condition.astype(int)
    df['StrongBear'] = strongbear_condition.astype(int)

    df['gapStrength'] = ta.compute_gapStrength(df)
    df = ta.add_exhaustion_indicator(df)

    df['Close'] = close
    return df

def add_pivots(df, win=windows):
    for w in win:
        roll_high = df['High'].rolling(w)
        roll_low = df['Low'].rolling(w)
        roll_close = df['Close'].rolling(w)
        PP = (roll_high.max() + roll_low.min() + roll_close.apply(lambda x: x[-1])).div(3)
        R1 = 2 * PP - roll_low.min()
        S1 = 2 * PP - roll_high.max()
        R2 = PP + (roll_high.max() - roll_low.min())
        S2 = PP - (roll_high.max() - roll_low.min())
        df[f'PP_{w}'] = PP
        df[f'R1_{w}'] = R1
        df[f'S1_{w}'] = S1
        df[f'R2_{w}'] = R2
        df[f'S2_{w}'] = S2
    return df

def average_pivots(df, windows=[5, 10, 14, 20]):
    for level in ['PP', 'R1', 'S1', 'R2', 'S2']:
        cols = [f'{level}_{w}' for w in windows]
        df[f'{level}_Avg'] = df[cols].mean(axis=1)
    return df

def compute_expected_return(df, forward_window=14, r_cols=['R1_Avg', 'R2_Avg']):
    df['Expected_Return'] = np.nan
    close_prices = df['Close'].values
    
    pivot_arrays = []
    for col in r_cols:
        if col in df.columns:
            pivot_arrays.append(df[col].values)
        else:
            pivot_arrays.append(np.full(len(df), np.nan))
    
    for i in range(len(df) - forward_window):
        current_price = close_prices[i]
        pivots = [arr[i] for arr in pivot_arrays if not np.isnan(arr[i])]
        target_level = max(pivots) if pivots else None
        future_window = close_prices[i+1:i+1+forward_window]
        
        if target_level is not None:
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
    return df

def compute_expected_loss(df, forward_window=14, s_cols=['S1_Avg', 'S2_Avg']):
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
    return df

def label_hit_prob_past(df, window=14, profit_target=0.05, stop_loss=0.05, lookback=60, tp_thresh=0.35, sl_thresh=0.4):
    close_prices = df['Close'].values
    bull = (df['TI'] == 'Bull')
    bear = (df['TI'] == 'Bear')
    hold = (df['TI'] == 'Hold')
    short = (df['TI'] == 'Short')
    neutral = (df['TI'] == 'Neutral')

    sma1 = df['SMA1'].values
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
        
        if tp_hit_idx is not None and (sl_hit_idx is None or tp_hit_idx < sl_hit_idx) and bull[i] and tp_prob >= tp_thresh:
            labels.append(2)  # TP (bull)
        elif sl_hit_idx is not None and (tp_hit_idx is None or sl_hit_idx < tp_hit_idx) and bear[i] and sl_prob >= sl_thresh:
            labels.append(1)  # SL (bear)
        elif hold[i]:
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
    
    # Post-process triggers
    for i in range(N):
        if labels[i] in [2, 3]:
            current_close = close_prices[i]
            sma1_now = sma1[i]
            atr_now = atr[i]
            rsi_now = rsi[i]
            adx_now = adx[i]

            future_end = min(i + 1 + window, N)
            future_closes = close_prices[i + 1 : future_end]
            future_sma1 = sma1[i + 1 : future_end]

            current_dip = current_close < sma1_now or current_close < (sma1_now - 0.5 * atr_now)
            future_dips = any((p < s) or (p < s - 0.5 * atr_now) for p, s in zip(future_closes, future_sma1))

            bearish_momentum = (rsi_now < 40) and (adx_now > 22)
            fading_bullish = (rsi_now < 50) or (adx_now < 20)
            hold_extreme = (labels[i] == 3) and (rsi_now < 45)

            if (current_dip or future_dips) and (bearish_momentum or fading_bullish or hold_extreme):
                labels[i] = 1
    
    df['Hit_Label'] = labels
    return df

def train_ml_models(df):
    """Train ML models using your existing pipeline"""
    df_model = df.dropna(subset=FEATURES + ['Hit_Label', 'Expected_Return', 'Expected_Loss'])
    
    if len(df_model) < 50:
        return None, None, None, None, None, None
    
    label2str = {0: 'None', 1: 'SL', 2: 'TP', 3: 'Hold', 4: 'Short'}
    expected_classes = [0, 1, 2, 3, 4]
    
    # Train classifier
    X_cls = df_model[FEATURES]
    y_cls = df_model['Hit_Label'].astype(int)
    scaler_cls = StandardScaler()
    X_scaled_cls = scaler_cls.fit_transform(X_cls)
    
    model_class = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42)
    model_class.fit(X_scaled_cls, y_cls)
    
    # Get class probabilities
    cls_probs = model_class.predict_proba(X_scaled_cls)
    prob_df = pd.DataFrame(0, index=np.arange(len(cls_probs)), columns=[f'Prob_Class_{c}' for c in expected_classes])
    for i, c in enumerate(model_class.classes_):
        if c in expected_classes:
            prob_df[f'Prob_Class_{c}'] = cls_probs[:, i]
    
    df_model = df_model.reset_index(drop=True)
    df_model = pd.concat([df_model, prob_df], axis=1)
    FEATURES_with_probs = FEATURES + [f'Prob_Class_{c}' for c in expected_classes]
    X_reg = df_model[FEATURES_with_probs]
    
    # Train return model
    y_return = df_model['Expected_Return']
    scaler_return = StandardScaler()
    X_scaled_return = scaler_return.fit_transform(X_reg)
    model_return = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42)
    model_return.fit(X_scaled_return, y_return)
    
    # Train loss model
    y_loss = df_model['Expected_Loss']
    scaler_loss = StandardScaler()
    X_scaled_loss = scaler_loss.fit_transform(X_reg)
    model_loss = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42)
    model_loss.fit(X_scaled_loss, y_loss)
    
    return model_class, model_return, model_loss, scaler_cls, scaler_return, scaler_loss

def get_ml_prediction(df, models):
    """Get ML prediction for the latest data point"""
    model_class, model_return, model_loss, scaler_cls, scaler_return, scaler_loss = models
    
    latest = df.iloc[[-1]]
    if latest[FEATURES].isnull().values.any():
        return None
    
    label2str = {0: 'None', 1: 'SL', 2: 'TP', 3: 'Hold', 4: 'Short'}
    expected_classes = [0, 1, 2, 3, 4]
    
    # Class prediction
    latest_scaled_cls = scaler_cls.transform(latest[FEATURES])
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
    hit_prob = latest_prob_features[f'Prob_Class_{pred_class}']
    
    # Regression predictions
    latest_prob_df = pd.DataFrame([latest_prob_features])
    latest_features_with_probs = pd.concat([latest[FEATURES].reset_index(drop=True), latest_prob_df], axis=1)
    latest_scaled_return = scaler_return.transform(latest_features_with_probs)
    latest_scaled_loss = scaler_loss.transform(latest_features_with_probs)
    
    current_price = latest['Close'].values[0]
    predicted_return = model_return.predict(latest_scaled_return)[0]
    predicted_loss = model_loss.predict(latest_scaled_loss)[0]
    
    # Confidence calculation
    ratio = (predicted_return / abs(predicted_loss)) if (will_hit != 'None' and predicted_loss != 0) else 0
    ratio = max(ratio, 0)
    confidence_score = max(hit_prob * ratio, 0)
    
    return {
        'will_hit': will_hit,
        'hit_prob': hit_prob,
        'predicted_return': predicted_return,
        'predicted_loss': predicted_loss,
        'confidence_score': confidence_score,
        'current_price': current_price
    }

# -------------------------
# Trading Strategy Backtest
# -------------------------
if st.button("Run ML Strategy Backtest"):
    st.write(f"Downloading data for {ticker} and running ML strategy...")
    
    # Download data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * YEARS_OF_DATA)
    
    df_daily = get_stock_data(ticker, start_date, end_date)
    df_weekly = yf.download(ticker, period=period, interval="1wk", progress=False)
    
    if df_daily.empty or df_weekly.empty:
        st.error("No data returned from Yahoo Finance.")
        st.stop()

    # Clean data
    for df in [df_daily, df_weekly]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]

    # Weekly trend analysis
    df_weekly['SMA10'] = df_weekly['Close'].rolling(10).mean()
    df_weekly['SMA50'] = df_weekly['Close'].rolling(50).mean()
    df_weekly['trend_up'] = df_weekly['SMA10'] > df_weekly['SMA50']

    # Prepare daily data with ML features
    st.write("Calculating technical indicators and training ML models...")
    df_daily = add_technical_indicators(df_daily)
    df_daily = add_pivots(df_daily, windows)
    df_daily = average_pivots(df_daily, [5, 10, 14, 20])
    df_daily = compute_expected_return(df_daily, forward_window=14, r_cols=['R1_Avg', 'R2_Avg'])
    df_daily = compute_expected_loss(df_daily, forward_window=14, s_cols=['S1_Avg', 'S2_Avg'])
    df_daily = label_hit_prob_past(df_daily, window=30, profit_target=PROFIT_TARGET, stop_loss=STOP_LOSS)

    # Train ML models
    models = train_ml_models(df_daily)
    if models[0] is None:
        st.error("Insufficient data for ML model training.")
        st.stop()

    # Backtest Logic
    trades = []
    in_trade = False
    current_trade = {}
    
    weekly_dates = df_weekly.index
    daily_dates = df_daily.index
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, current_date in enumerate(daily_dates):
        progress_bar.progress((i + 1) / len(daily_dates))
        status_text.text(f"Processing {current_date.strftime('%Y-%m-%d')}...")
        
        # Find corresponding weekly trend
        weekly_mask = weekly_dates <= current_date
        if not weekly_mask.any():
            continue
            
        latest_weekly_date = weekly_dates[weekly_mask][-1]
        weekly_trend_up = df_weekly.loc[latest_weekly_date, 'trend_up'] if pd.notna(df_weekly.loc[latest_weekly_date, 'trend_up']) else False
        
        # Skip if weekly trend is down
        if not weekly_trend_up:
            if in_trade:
                # Consider exiting due to trend change
                pass
            continue
        
        # Get ML prediction for current day
        current_data = df_daily.loc[:current_date].copy()
        if len(current_data) < 100:  # Need sufficient history for ML
            continue
            
        # Retrain models on current data (in practice, you might want to retrain less frequently)
        current_models = train_ml_models(current_data)
        if current_models[0] is None:
            continue
            
        ml_prediction = get_ml_prediction(current_data, current_models)
        if ml_prediction is None:
            continue
        
        current_ml_signal = ml_prediction['will_hit']
        current_ml_confidence = ml_prediction['confidence_score']
        
        # ENTRY LOGIC: Not in trade + ML buy signal + sufficient confidence + weekly trend up
        if (not in_trade and 
            current_ml_signal in ['TP', 'Hold'] and  # Bullish signals
            current_ml_confidence >= ml_confidence_threshold and
            weekly_trend_up):
            
            entry_price = float(df_daily.loc[current_date, 'Close'])
            TP_price = entry_price * (1 + TP_pct / 100.0)
            SL_price = entry_price * (1 - SL_pct / 100.0)
            
            current_trade = {
                'entry_date': current_date,
                'entry_price': entry_price,
                'tp_price': TP_price,
                'sl_price': SL_price,
                'entry_week': latest_weekly_date,
                'ml_confidence': current_ml_confidence,
                'ml_signal': current_ml_signal
            }
            in_trade = True
            
            st.write(f"📈 Entry on {current_date.strftime('%Y-%m-%d')} at {entry_price:.2f}")
        
        # EXIT LOGIC
        elif in_trade:
            entry_date = current_trade['entry_date']
            entry_price = current_trade['entry_price']
            TP_price = current_trade['tp_price']
            SL_price = current_trade['sl_price']
            
            days_in_trade = (current_date - entry_date).days
            
            current_open = float(df_daily.loc[current_date, 'Open'])
            current_high = float(df_daily.loc[current_date, 'High'])
            current_low = float(df_daily.loc[current_date, 'Low'])
            current_close = float(df_daily.loc[current_date, 'Close'])
            
            exit_reason = None
            exit_price = None
            
            # 1) Stop Loss hit
            if current_low <= SL_price:
                exit_reason = 'SL'
                exit_price = SL_price
            
            # 2) Take Profit hit
            elif current_high >= TP_price:
                exit_reason = 'TP'
                exit_price = TP_price
            
            # 3) Gap moves at open
            elif current_open <= SL_price:
                exit_reason = 'Gap_SL'
                exit_price = min(current_open, SL_price)
            elif current_open >= TP_price:
                exit_reason = 'Gap_TP'
                exit_price = max(current_open, TP_price)
            
            # 4) Max holding period
            elif days_in_trade >= max_holding_days:
                exit_reason = 'Max_Hold'
                exit_price = current_close
            
            # Exit trade if condition met
            if exit_reason:
                return_pct = (exit_price / entry_price - 1) * 100.0
                
                trades.append({
                    'EntryDate': entry_date,
                    'ExitDate': current_date,
                    'EntryPrice': entry_price,
                    'ExitPrice': exit_price,
                    'Outcome': exit_reason,
                    'Return_%': return_pct,
                    'HoldingDays': days_in_trade,
                    'ML_Confidence': current_trade['ml_confidence'],
                    'ML_Signal': current_trade['ml_signal']
                })
                
                st.write(f"📉 Exit on {current_date.strftime('%Y-%m-%d')} at {exit_price:.2f} ({exit_reason})")
                in_trade = False
                current_trade = {}

    # Handle open trade at end
    if in_trade:
        last_date = daily_dates[-1]
        exit_price = float(df_daily.loc[last_date, 'Close'])
        return_pct = (exit_price / current_trade['entry_price'] - 1) * 100.0
        
        trades.append({
            'EntryDate': current_trade['entry_date'],
            'ExitDate': last_date,
            'EntryPrice': current_trade['entry_price'],
            'ExitPrice': exit_price,
            'Outcome': 'Open',
            'Return_%': return_pct,
            'HoldingDays': (last_date - current_trade['entry_date']).days,
            'ML_Confidence': current_trade['ml_confidence'],
            'ML_Signal': current_trade['ml_signal']
        })

    progress_bar.empty()
    status_text.empty()

    # Results Analysis
    results = pd.DataFrame(trades)
    
    if results.empty:
        st.warning("No trades executed. Check ML predictions and weekly trend conditions.")
        st.stop()

    # Calculate performance metrics
    initial_cap = 1.0
    results['Return_factor'] = 1 + results['Return_%'] / 100.0
    results['Cumulative'] = initial_cap * results['Return_factor'].cumprod()
    equity_ts = pd.Series(data=results['Cumulative'].values, index=pd.to_datetime(results['ExitDate']))

    total_trades = len(results)
    wins = results['Return_%'] > 0
    win_rate = 100.0 * wins.sum() / total_trades if total_trades > 0 else 0
    avg_return = results['Return_%'].mean()
    net_return_pct = (results['Cumulative'].iloc[-1] - initial_cap) / initial_cap * 100.0
    
    # ML performance metrics
    successful_ml_predictions = len(results[results['Return_%'] > 0])
    ml_accuracy = 100.0 * successful_ml_predictions / total_trades if total_trades > 0 else 0

    st.subheader("📊 ML Strategy Performance Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", total_trades)
    col2.metric("Win Rate", f"{
