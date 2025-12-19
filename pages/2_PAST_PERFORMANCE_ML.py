#!/usr/bin/env python
# coding: utf-8
from imports import *
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings, math
warnings.filterwarnings('ignore')

# Set page config first
st.set_page_config(page_title="ML - Stock Past Performance", layout="wide")

st.title("🤖 Backtest: Study Prior to Real-world Trading")
with st.expander("Strategy Overview"):
    st.markdown("""
- **Daily Trend Filter:** 1D Time-frame data only to learn if the given stock is profitable.
- **ML Entry Signals:** Entries occur at daily close when the ML model predicts a bullish move ("TP", "Hold", or "None") with confidence above the threshold (Try 60%).
- **Trade Entry:** Opens a single position if not already holding a trade and entry signal conditions are met.
- **Trade Exit:** Closes a position when price reaches (TP) or (SL), or after maximum allowed holding days.
- **Sequential Trading:** The strategy waits for the current trade to close before opening a new one—no overlapping trades.
    """)

with st.expander("Backtest and ML Workflow"):
    st.markdown("""
- **Feature Engineering:** Daily OHLCV data is enriched with technical indicators and pivot points.
- **Labeling:** Each daily row is labeled as TP (take profit), SL (stop loss), or neutral based on future price moves over a lookahead window.
- **Model Training:** Random Forest classifiers and regressors predict the likelihood of TP/SL and expected returns/losses.
- **Prediction:** At each day, the ML model provides a prediction and confidence score that guide trade entries.
- **Performance:** The app tracks trade results, return distributions, and equity growth over the backtest period.
    """)

with st.expander("Example Entry Using Daily Data"):
    st.markdown("""
- Suppose the stock closes at $100.
- If the ML model predicts a bullish move with confidence above the threshold, and no open trade exists, entry happens at the daily close ($100).
- For TP: If the given TP% is higher than the ML predicted return, meaning ML expects the stock to gain less, it uses the ML target, else it retains given TP.
- Next trades only occur after closing the current trade.
    """)

with st.expander("Example Exit Conditions"):
    st.markdown("""
Trades are exited if:
- Intraday price reaches TP or SL levels.
- The gap open price exceeds TP or SL boundaries (gap exit).
- If the given SL is higher than the predicted ML loss, it assumes higher risks are taken and uses the higher given SL else it adopts predicted SL.
- The trade has reached maximum holding days.
Exits simulate realistic intraday stop-loss and take-profit triggers.
    """)

with st.expander("Intraday SL/TP Trigger Logic"):
    st.markdown("""
- Each day's high and low prices are checked to model intraday SL/TP triggers.
- Exit occurs immediately on a day when price touches SL or TP, even if closing price differs.
- Gap openings beyond SL or TP are detected and trigger immediate exit at gap price or SL/TP levels.
This prevents unrealistic trade closing at end-of-day prices only.
    """)

with st.expander("What are the biggest enemies?"):
    st.markdown("""
    - Avoiding garbage companies
    - Filtering with ML e.g. confidence and signals.
    - Stop-loss & Panic selling (Holding for 30-60 days)
    - Beating these enemies is more like:
        - 8–15 strong positions per year
        - 20–60% winners
        - Held for 1–3 months
        - Compounded
    """)

with st.expander("How often does the model retrain?"):
    st.markdown("""
    A common concern is whether the model skips data or uses less information when retraining less frequently.  
    **It does NOT.**
    
    Here's what actually happens:
    
    - The model is always trained using **all historical data available up to that date**
    - No candles are skipped
    - No rows are removed
    - No future data is used
    
    The only thing that changes is **how often the model is retrained**, not **how much data it sees**.
    
    For example:
    - Day 100 → model sees Days 1–100
    - Day 120 → model sees Days 1–120
    - Day 140 → model sees Days 1–140
    
    This mimics real trading behavior where strategies are updated periodically (weekly or monthly) instead of every single day, improving stability and performance while keeping full historical context.
    """)

with st.expander("What happens to the model after a trade is opened?"):
    st.markdown("""
    Once a trade is opened, the model becomes **locked (frozen)** until that trade is closed.
    
    - The model that generated the entry signal remains unchanged
    - No retraining occurs during the open trade
    - This prevents signal fluctuation and unrealistic hindsight bias
    
    Example:
    - Entry at Day 100 → model version at that date is locked
    - Trade remains open on Day 101, 102, 103…
    - Exit at Day 104 → model is then allowed to train again for the next entry
    
    This simulates real trading discipline:
    - One decision per trade
    - No signal-switching mid-trade
    - No emotional or hindsight adjustments
    - Clean, realistic performance tracking
    """)

with st.expander("Try ETH-USD or XRP-USD"):
    st.markdown("""
- 7% TP target
- Put SL far away like 99%, try your luck for free!
- Put ML Confidence to 60-70%
    """)

# -------------------------
# Strategy Parameters
# -------------------------
YEARS_OF_DATA = 3
PROFIT_TARGET = 0.0375
STOP_LOSS = 0.0375
_DAYS = 22
windows = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
confidence_data = [] 
will_hit_history = []

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

def validate_ticker(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if hist.empty:
            return {"valid": False, "reason": "No price history"}
        return {"valid": True, "reason": "Ticker found"}
    except Exception as e:
        return {"valid": False, "reason": str(e)}

# -------------------------
# User inputs (PAST PERFORMANCE TAB)
# -------------------------

col1, col2, col3, col4 = st.columns(4)

# --- Ticker ---
with col1:
    ticker = st.text_input("Ticker", value="COIN")
    result = validate_ticker(ticker)
    if result["valid"]:
        st.success(f"{ticker} is valid ✅ ({result['reason']})")
    else:
        st.error(f"{ticker} is invalid ❌ ({result['reason']})")
        st.stop()

# --- History period ---
with col2:
    period = st.selectbox("History period", ["1y", "2y", "3y", "5y", "7y"], index=2)

# --- Risk setup ---
with col3:
    risk_setup = st.selectbox(
        "Risk Setup",
        ["a) Extreme Risk (Beta <1.5)",
         "b) Moderate Risk (Beta >2)",
         "c) Low Risk (Less Rewards)"],
        index=2,
        key="risk_setup"
    )

# --- ML confidence ---
with col4:
    ml_confidence_threshold = st.number_input(
        "ML Confidence",
        min_value=0,
        max_value=100,
        value=63,
        step=5,
        key="ml_conf"
    )

# -------------------------
# Session State Initialization
# -------------------------

defaults = {
    "a) Extreme Risk (Beta <1.5)":  (15.0, 90.0, 180),
    "b) Moderate Risk (Beta >2)":   (15.0, 30.0, 90),
    "c) Low Risk (Less Rewards)":   (7.0, 14.0, 21),
}

# Initialize editable fields once
if "tp_input" not in st.session_state:
    st.session_state.tp_input = defaults[risk_setup][0]
if "sl_input" not in st.session_state:
    st.session_state.sl_input = defaults[risk_setup][1]
if "hold_input" not in st.session_state:
    st.session_state.hold_input = defaults[risk_setup][2]

# Track last selected risk setup
if "last_risk_setup" not in st.session_state:
    st.session_state.last_risk_setup = risk_setup

# -------------------------
# Update values ONLY when risk setup changes
# -------------------------

if st.session_state.last_risk_setup != risk_setup:
    preset_tp, preset_sl, preset_hold = defaults[risk_setup]

    st.session_state.tp_input = preset_tp
    st.session_state.sl_input = preset_sl
    st.session_state.hold_input = preset_hold

    st.session_state.last_risk_setup = risk_setup

# -------------------------
# Editable Inputs (bound to session_state)
# -------------------------

col_tp, col_sl, col_hold = st.columns(3)

TP_pct = col_tp.number_input(
    "TP %",
    min_value=0.0,
    max_value=100.0,
    step=0.5,
    key="tp_input"
)

SL_pct = col_sl.number_input(
    "SL %",
    min_value=0.0,
    max_value=100.0,
    step=0.5,
    key="sl_input"
)

max_holding_days = col_hold.number_input(
    "Hold Days",
    min_value=3,
    max_value=365,
    step=1,
    key="hold_input"
)

# -------------------------
# Display active settings
# -------------------------

st.info(f"📊 ACTIVE SETTINGS → TP={TP_pct}%, SL={SL_pct}%, Hold={max_holding_days} days")



# -------------------------
# Technical Analysis Functions (Simplified)
# -------------------------
def get_stock_data(ticker, start_date, end_date):
    """Get stock data from Yahoo Finance"""
    df = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1), 
                     progress=False)
    if df.empty:
        return None
    
    # Clean column names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    df.index = pd.to_datetime(df.index)
    df = df.dropna()
    return df

def calculate_rsi(df, period=14):
    """Calculate RSI"""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = np.maximum(np.maximum(high_low, high_close), low_close)
    atr = true_range.rolling(window=period).mean()
    return atr

def add_technical_indicators(df):
    """Add basic technical indicators"""
    df = df.copy()
    
    # Moving averages
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    
    # RSI
    df['RSI'] = calculate_rsi(df)
    df['RSI_SMA'] = df['RSI'].rolling(14).mean()
    
    # ATR
    df['ATR'] = calculate_atr(df)
    
    # Volume indicators
    df['Volume_MA20'] = df['Volume'].rolling(20).mean()
    df['buy_volume'] = (df['Close'] > df['Close'].shift(1)) * df['Volume']
    df['sell_volume'] = (df['Close'] < df['Close'].shift(1)) * df['Volume']
    df['sumBuyVol'] = df['buy_volume'].rolling(window=9).sum()
    df['sumSellVol'] = df['sell_volume'].rolling(window=9).sum()
    
    # Returns
    df['return1'] = df['Close'].pct_change(7).rolling(3).mean()
    df['return2'] = df['Close'].pct_change(14).rolling(3).mean()
    df['return3'] = df['Close'].pct_change(21).rolling(3).mean()
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(14).std().rolling(3).mean()
    df[['+DI', '-DI', 'ADX']] = ta.calculate_dmi(df, n=14).rolling(3).mean()
    conditions = [
    # BULL
        (
            (
                (df['SMA10'] > df['SMA50']) &
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
            (df['SMA10'] < df['SMA50']) &
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
            (df['Close'] <= df['SMA10']) &
            (df['SMA10'] < df['SMA50']) &
            (df['RSI'].between(50, 85))
        ),
        # HOLD
        (
            (df['Close'] > df['SMA50']) &
            (df['SMA10'] > df['SMA50']) &
            (df['RSI'].between(50, 90))
        )
    ]
    choices = ['Bull', 'Bear', 'Short', 'Hold']
    
    df['TI'] = np.select(conditions, choices, default='Neutral')
    df['Bull'] = (df['TI'] == 'Bull').astype(int)
    df['Bear'] = (df['TI'] == 'Bear').astype(int)
    df['Hold'] = (df['TI'] == 'Hold').astype(int)
    df['Short'] = (df['TI'] == 'Short').astype(int)
    df['Neutral'] = (df['TI'] == 'Neutral').astype(int)
    
    return df

def add_pivot_levels(df, window=_DAYS):
    high = df['High'].rolling(window)
    low = df['Low'].rolling(window)
    close = df['Close'].rolling(window)
    PP = (high.max() + low.min() + close.apply(lambda x: x[-1])).div(3)
    R1 = 2 * PP - low.min()
    S1 = 2 * PP - high.max()
    R2 = PP + (high.max() - low.min())
    S2 = PP - (high.max() - low.min())
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

def compute_expected_return(df, forward_window=14):
    """Compute expected return based on pivot levels"""
    df['Expected_Return'] = 0.0
    close_prices = df['Close'].values
    
    for i in range(len(df) - forward_window):
        current_price = close_prices[i]
        future_prices = close_prices[i+1:i+1+forward_window]
        
        if len(future_prices) > 0:
            max_future = np.nanmax(future_prices)
            df.iloc[i, df.columns.get_loc('Expected_Return')] = (max_future - current_price) / current_price
    
    return df

def compute_expected_loss(df, forward_window=14):
    """Compute expected loss based on pivot levels"""
    df['Expected_Loss'] = 0.0
    close_prices = df['Close'].values
    
    for i in range(len(df) - forward_window):
        current_price = close_prices[i]
        future_prices = close_prices[i+1:i+1+forward_window]
        
        if len(future_prices) > 0:
            min_future = np.nanmin(future_prices)
            df.iloc[i, df.columns.get_loc('Expected_Loss')] = (min_future - current_price) / current_price
    
    return df

def label_hit_prob_past(
    df,
    window=14,
    profit_target=0.05,
    stop_loss=0.05,
    lookback=60,
    tp_thresh=0.35,
    sl_thresh=0.4
):
    import numpy as np
    
    close_prices = df['Close'].values
    
    bull = (df['TI'] == 'Bull')
    bear = (df['TI'] == 'Bear')
    hold = (df['TI'] == 'Hold')
    short = (df['TI'] == 'Short')
    neutral = (df['TI'] == 'Neutral')

    sma1 = df['SMA10'].values
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
    
    # Post-process: Trigger SL immediately on price dip below SMA1 or SMA1-ATR buffer with momentum checks for Hold/TP
    for i in range(N):
        if labels[i] in [2, 3]:  # TP or Hold bars
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
                if not ((rsi_now > 52) and (df['Close'].iloc[i] > df['SMA20'].iloc[i])):
                    labels[i] = 1
    
    df['Hit_Label'] = labels
    return df

@st.cache_data
def prepare_features(df):
    df = add_technical_indicators(df)
    df = add_pivots(df, windows)
    df = average_pivots(df, windows)
    df = compute_expected_return(df)
    df = compute_expected_loss(df)
    df = label_hit_prob_past(df, profit_target=PROFIT_TARGET, stop_loss=STOP_LOSS)
    return df

# -------------------------
# ML Model Functions
# -------------------------
def train_ml_models(df):
    """Train ML models using the feature set"""
    # Select available features
    available_features = [f for f in FEATURES if f in df.columns]
    
    if len(available_features) < 10:
        st.warning(f"Only {len(available_features)} features available. Need more features for ML.")
        return None, None, None, None, None, None
    
    df_model = df.dropna(subset=available_features + ['Hit_Label', 'Expected_Return', 'Expected_Loss'])
    
    if len(df_model) < 50:
        st.warning("Insufficient data after cleaning for ML training.")
        return None, None, None, None, None, None
    
    # Train classifier
    X_cls = df_model[available_features]
    y_cls = df_model['Hit_Label'].astype(int)
    
    scaler_cls = StandardScaler()
    X_scaled_cls = scaler_cls.fit_transform(X_cls)
    
    model_class = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        min_samples_leaf=5, 
        random_state=42
    )
    model_class.fit(X_scaled_cls, y_cls)
    
    # Train return model
    y_return = df_model['Expected_Return']
    scaler_return = StandardScaler()
    X_scaled_return = scaler_return.fit_transform(X_cls)
    
    model_return = RandomForestRegressor(
        n_estimators=100, 
        max_depth=10, 
        min_samples_leaf=5, 
        random_state=42
    )
    model_return.fit(X_scaled_return, y_return)
    
    # Train loss model
    y_loss = df_model['Expected_Loss']
    scaler_loss = StandardScaler()
    X_scaled_loss = scaler_loss.fit_transform(X_cls)
    
    model_loss = RandomForestRegressor(
        n_estimators=100, 
        max_depth=10, 
        min_samples_leaf=5, 
        random_state=42
    )
    model_loss.fit(X_scaled_loss, y_loss)
    
    return model_class, model_return, model_loss, scaler_cls, scaler_return, scaler_loss

def get_ml_prediction(df, models):
    """Get ML prediction for the latest data point"""
    model_class, model_return, model_loss, scaler_cls, scaler_return, scaler_loss = models
    
    # Select available features
    available_features = [f for f in FEATURES if f in df.columns]
    latest = df[available_features].iloc[[-1]]
    
    if latest.isnull().values.any():
        return None
    
    # Class prediction
    latest_scaled_cls = scaler_cls.transform(latest)
    class_probs = model_class.predict_proba(latest_scaled_cls)[0]
    predicted_class = model_class.predict(latest_scaled_cls)[0]

    p_none  = class_probs[0] if len(class_probs) > 0 else 0
    p_sl    = class_probs[1] if len(class_probs) > 1 else 0
    p_tp    = class_probs[2] if len(class_probs) > 2 else 0
    p_hold  = class_probs[3] if len(class_probs) > 3 else 0
    p_short = class_probs[4] if len(class_probs) > 4 else 0

    
    bullish_prob = p_tp + p_hold
    bearish_prob = p_sl + p_short

    label_map = {0: 'None', 1: 'SL', 2: 'TP', 3: 'Hold', 4: 'Short'}
    will_hit = label_map.get(predicted_class, 'None')
    hit_prob = class_probs[predicted_class]
    
    # Regression predictions
    latest_scaled_return = scaler_return.transform(latest)
    latest_scaled_loss = scaler_loss.transform(latest)
    
    current_price = df['Close'].iloc[-1]
    predicted_return = model_return.predict(latest_scaled_return)[0]
    predicted_loss = model_loss.predict(latest_scaled_loss)[0]

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
        'predicted_return': predicted_return,
        'predicted_loss': predicted_loss,
        'confidence_score': confidence_score,
        'current_price': current_price
    }

# -------------------------
# Trading Strategy Backtest
# -------------------------
if st.button("Run ML Strategy Backtest"):
    st.write(f"Downloading daily data for {ticker} and running ML strategy...")

    end_date = datetime.now()
    #start_date = end_date - timedelta(days=365 * YEARS_OF_DATA)  # Full years as days lookback
    if period == "1y":
        start_date = end_date - timedelta(days=365*2)
    elif period == "2y":
        start_date = end_date - timedelta(days=365 * 3)
    elif period == "3y":
        start_date = end_date - timedelta(days=365 * 4)
    elif period == "5y":
        start_date = end_date - timedelta(days=365 * 6)
    elif period == "7y":
        start_date = end_date - timedelta(days=365 * 8)

    with st.spinner('Downloading daily market data...'):
        df_daily = get_stock_data(ticker, start_date, end_date)
    
    if df_daily is None or df_daily.empty:
        st.error("No daily data returned from Yahoo Finance.")
        st.stop()

    with st.spinner('Calculating technical indicators...'):
        df_daily = prepare_features(df_daily)

    st.write("Running backtest...")
    trades = []
    in_trade = False
    current_trade = {}
    daily_dates = df_daily.index
    progress_bar = st.progress(0)
    RETRAIN_EVERY = 5
    ml_tp_success_counter = 0

    for i, current_date in enumerate(daily_dates):
        if i % 50 == 0:
            progress_bar.progress(min((i + 1) / len(daily_dates), 1.0))
        
        # Use daily data only
        current_data = df_daily.iloc[:i+1]

        if len(current_data) < 100:
            continue
        
        if (not in_trade):
            if i % RETRAIN_EVERY == 0 or i < 120:
                models = train_ml_models(current_data)

            if models[0] is None:
                st.error("Insufficient data for ML model training.")
                st.stop()
    
            ml_prediction = get_ml_prediction(current_data, models)
            
        if ml_prediction is None or ml_prediction['confidence_score'] < ml_confidence_threshold:
            continue
    
        current_ml_signal = ml_prediction['will_hit']
        current_ml_confidence = ml_prediction['confidence_score']
        confidence_data.append({'Date': current_date, 'ML_Confidence': current_ml_confidence, 'ML_Signal': current_ml_signal})
            
        # ENTRY LOGIC
        if (not in_trade and
            current_ml_signal in ['TP', 'Hold', 'None'] and  
            current_ml_confidence >= ml_confidence_threshold):

            entry_price = float(df_daily.loc[current_date, 'Close'])

            tp_given = TP_pct / 100.0
            sl_given = -SL_pct / 100.0

            predicted_return = ml_prediction['predicted_return']
            predicted_loss = ml_prediction['predicted_loss']
            
            entry_price = float(df_daily.loc[current_date, 'Close'])
            
            if tp_given > predicted_return:
                TP_price = entry_price * (1 + predicted_return)
                used_ml_tp = True
            else:
                TP_price = entry_price * (1 + tp_given)
                used_ml_tp = False
            
            if np.abs(sl_given) > np.abs(predicted_loss):
                SL_price = entry_price * (1 + sl_given)
            else:
                SL_price = entry_price * (1 + predicted_loss)

            current_trade = {
                'entry_date': current_date,
                'entry_price': entry_price,
                'tp_price': TP_price,
                'sl_price': SL_price,
                'ml_confidence': current_ml_confidence,
                'ml_signal': current_ml_signal,
                'used_ml_tp': used_ml_tp
            }
            in_trade = True

        # EXIT LOGIC
        elif in_trade:
            exit_reason = None
            exit_price = None
  
            last_date = daily_dates[-1]
            entry_date = current_trade['entry_date']
            exit_days = (last_date - current_trade['entry_date']).days
            entry_price = current_trade['entry_price']
            TP_price = current_trade['tp_price']
            SL_price = current_trade['sl_price']
            days_in_trade = (current_date - entry_date).days
            
            current_open = float(df_daily.loc[current_date, 'Open'])
            current_high = float(df_daily.loc[current_date, 'High'])
            current_low = float(df_daily.loc[current_date, 'Low'])
            current_close = float(df_daily.loc[current_date, 'Close'])
            
            if current_low <= SL_price:
                exit_reason = 'SL'
                exit_price = SL_price
            elif current_high >= TP_price:
                exit_reason = 'TP'
                exit_price = TP_price
            elif current_open <= SL_price:
                exit_reason = 'Gap_SL'
                exit_price = min(current_open, SL_price)
            elif current_open >= TP_price:
                exit_reason = 'Gap_TP'
                exit_price = max(current_open, TP_price)
            elif days_in_trade >= max_holding_days:
                exit_reason = 'Max_Hold'
                exit_price = current_close
            
            if exit_reason:
                return_pct = (exit_price / entry_price - 1) * 100.0
                if (exit_reason in ['TP', 'Gap_TP'] and 
                    current_trade.get('used_ml_tp', False) and 
                    current_trade['ml_signal'] == 'TP'):
                    ml_tp_success_counter += 1
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
                in_trade = False
                current_trade = {}

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
    conf = pd.DataFrame(confidence_data)
    conf['Date'] = pd.to_datetime(conf['Date'])
    
    # Results Analysis
    results = pd.DataFrame(trades)

    if results.empty:
        st.warning("No trades executed. Check ML predictions and trend conditions.")
        st.stop()

    # Calculate performance metrics
    initial_cap = 1000.0
    results['Return_factor'] = 1 + results['Return_%'] / 100.0
    results['Cumulative'] = initial_cap * results['Return_factor'].cumprod()
    
    total_trades = len(results)
    wins = results['Return_%'] > 0
    win_rate = 100.0 * wins.sum() / total_trades if total_trades > 0 else 0
    avg_return = results['Return_%'].mean()
    net_return_pct = (results['Cumulative'].iloc[-1] - initial_cap) / initial_cap * 100.0
    avg_holding_days = results['HoldingDays'].mean()
    
    # Calculate no-trades days
    if len(results) > 1:
        results_sorted = results.sort_values('EntryDate').reset_index(drop=True)
        results_sorted['NextEntryDate'] = results_sorted['EntryDate'].shift(-1)
        results_sorted['NoTradeDays'] = (results_sorted['NextEntryDate'] - results_sorted['ExitDate']).dt.days
        avg_no_trade_days = results_sorted['NoTradeDays'].dropna().mean()
    else:
        avg_no_trade_days = 0
        
    # Display results
    st.subheader(f"📊 ML Strategy Summary ({ticker})")
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    col1.metric("Total Trades", total_trades)
    col2.metric("Win Rate", f"{win_rate:.1f}%")
    col3.metric("Avg. Return", f"{avg_return:.1f}%")
    col4.metric("ML TP Hits", ml_tp_success_counter)
    col5.metric("Avg. Holding Days", f"{avg_holding_days:.0f}")
    col6.metric("Avg. No-Trade Days", f"{avg_no_trade_days:.0f}")
    col7.metric("Net Return", f"{net_return_pct:.1f}%")

    # Trade outcomes breakdown
    st.subheader("Trade Outcomes")
    outcome_counts = results['Outcome'].value_counts()
    st.write(outcome_counts)

    # Display trades
    st.subheader("Trade History")
    st.dataframe(results.sort_values('EntryDate', ascending=False))

    # -------------------
    # Plot results
    # -------------------
    
    st.subheader("Backtest and Equity")
    
    fig, (ax, ax1, bx, cx) = plt.subplots(4, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1.5, 1, 1]})
    
    ax.plot(df_daily.index, df_daily['Close'], color='gray', linewidth=1.2, alpha=0.5, label = 'Price')
    ax.plot(df_daily.index, df_daily['SMA10'], color='orange', linewidth=1.0, alpha=0.7, label = 'SMA10')
    ax.plot(df_daily.index, df_daily['SMA50'], color='red', linewidth=1.0, alpha=0.5, label = 'SMA50')
    
    ax.fill_between(df_daily.index, df_daily['SMA10'], df_daily['SMA50'],
                    where=(df_daily['SMA10'] > df_daily['SMA50']),
                    color='green', alpha=0.15)
    
    ax.fill_between(df_daily.index, df_daily['SMA10'], df_daily['SMA50'],
                    where=(df_daily['SMA10'] < df_daily['SMA50']),
                    color='red', alpha=0.15)

    # RSI PLOT
    rsi_ = df_daily['RSI'].rolling(3).mean()
    ax1.plot(df_daily.index, rsi_, color='orange', linewidth=1.0, alpha=0.5, label = 'RSI')
    ax1.plot(df_daily.index, df_daily['RSI_SMA'], color='red', linewidth=1.0, alpha=0.5, label = 'RSI_SMA')
    ax1.axhline(y=70, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=30, color='red', linestyle='--', alpha=0.5, linewidth=1)

    tp_mask = df_daily['Hit_Label'] == 2
    sl_mask = df_daily['Hit_Label'] == 1
    hold_mask = df_daily['Hit_Label'] == 3
    short_mask = df_daily['Hit_Label'] == 4
    neutral_mask = df_daily['Hit_Label'] == 0
    _s = 5
    ax1.scatter(df_daily.index[tp_mask], rsi_[tp_mask], color='green', marker='^', s=_s, alpha=0.3, label='TP', zorder=6)
    ax1.scatter(df_daily.index[sl_mask], rsi_[sl_mask], color='red', marker='v', s=_s, alpha=0.3, label='SL', zorder=7)
    ax1.scatter(df_daily.index[hold_mask], rsi_[hold_mask], color='orange', marker='o', s=_s, alpha=0.3, label='Hold', zorder=8)
    ax1.scatter(df_daily.index[short_mask], rsi_[short_mask], color='purple', marker='x', s=_s, alpha=0.3, label='Short', zorder=9)
    ax1.scatter(df_daily.index[neutral_mask], rsi_[neutral_mask], color='gray', marker='.', s=_s, alpha=0.3, label='Neutral', zorder=10)

    ax1.fill_between(df_daily.index, rsi_, df_daily['RSI_SMA'],
                    where=(rsi_ > df_daily['RSI_SMA']),
                    color='green', alpha=0.15)
    
    ax1.fill_between(df_daily.index, rsi_, df_daily['RSI_SMA'],
                    where=(rsi_ < df_daily['RSI_SMA']),
                    color='red', alpha=0.15)
    ax1.set_ylabel('RSI', labelpad=10)
    ax1.yaxis.set_label_position('right')
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_coords(1.05, 0.5)

    bx.plot(results['ExitDate'], results['Cumulative'], color='gray', linewidth=1.0, alpha=0.5)
    max_cum = round(results['Cumulative'].max(), -1)
    mean_cum = round(max_cum/2, -1)
    tick_values = [0, mean_cum, max_cum]
    bx.set_ylim(0, max_cum *1.1)
    bx.set_yticks(tick_values)

    bx.axhline(1.0, color='red', linestyle='--', alpha=0.5)
    bx.set_ylabel('Equity ($)')
    bx.set_xlabel('Date')
    entry_label_shown = True
    tp_label_shown = True
    sl_label_shown = True
    other_label_shown = True
    entry_annotated = True
    nr = 1

    if len(results) > 50:
        nr = 1
    elif len(results) > 100:
        nr = 2
    elif len(results) > 150:
        nr = 3
    else:
        nr = 4
        
    for i in range(0, len(results), nr):
        outcome = results['Outcome'].iloc[i]
        color = 'green' if outcome == 'TP' else 'red' if outcome == 'SL' else 'gray'
        # Plot entry (all blue, no label)
        ax.scatter(results['EntryDate'].iloc[i], results['EntryPrice'].iloc[i], color='blue', s=5, zorder=5, alpha=0.5)
        # Plot exit by outcome
        label = None
        if outcome == 'TP' and 'TP' not in ax.get_legend_handles_labels()[1]:
            label = 'TP'
        elif outcome == 'SL' and 'SL' not in ax.get_legend_handles_labels()[1]:
            label = 'SL'
        elif outcome not in ['TP', 'SL'] and 'Other' not in ax.get_legend_handles_labels()[1]:
            label = 'Other'
        ax.scatter(results['ExitDate'].iloc[i], results['ExitPrice'].iloc[i], color=color, label=label, s=5, alpha = 0.5, zorder=5)
        # Annotate only the first blue entry if desired
        if not entry_annotated:
            ax.annotate('Entry', (results['EntryDate'].iloc[i], results['EntryPrice'].iloc[i]), xytext=(0, -12), textcoords='offset points', fontsize=8, color='blue')
            entry_annotated = True
    
    rd = pd.to_datetime(results['EntryDate'])
    cx.scatter(rd, results['ML_Confidence'], color='blue', alpha=0.5, s=7, label='ML Confidence (Entries)')
    cx.plot(conf['Date'], conf['ML_Confidence'], color='gray', alpha=0.5, linewidth=1.0, label='ML Confidence')
    cx.fill_between(df_daily.index, 0, ml_confidence_threshold, color='red', alpha=0.15)
    
    ax.set_title(f'{ticker} Price Chart')
    bx.set_title(f'Total Equity Over Time')
    cx.set_title(f'ML Confidence Over Time')

    ax.grid(alpha=0.3)
    bx.grid(alpha=0.3)
    cx.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize='x-small')
    ax1.legend(loc='upper left', fontsize='x-small')
    bx.legend(loc='lower left', fontsize='x-small')
    cx.legend()
    
    ax.set_ylabel('Price', labelpad=10)
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    ax.yaxis.set_label_coords(1.05, 0.5)
    
    bx.set_ylabel('Equity', labelpad=10)
    bx.yaxis.set_label_position('right')
    bx.yaxis.tick_right()
    bx.yaxis.set_label_coords(1.05, 0.5)

    cx.set_ylabel('ML Conf', labelpad=10)
    cx.yaxis.set_label_position('right')
    cx.yaxis.tick_right()
    cx.yaxis.set_label_coords(1.05, 0.5)
    cx.set_ylim(0, 100)

    fig.tight_layout()
    st.pyplot(fig)

    # ML Performance Analysis
    st.subheader("ML Signal Performance")
    if 'ML_Signal' in results.columns:
        signal_stats = results.groupby('Outcome').agg({
            'Return_%': ['count', 'mean', 'std', 'sum'],
            'ML_Confidence': 'mean',
            'HoldingDays': 'mean'
        }).round(2)
        signal_stats.columns = ['Trades', 'Avg Return %', 'Return Std %', 'Total Return %', 'Avg Conf', 'Avg Hold Days']
        st.dataframe(signal_stats)

    st.success("Backtest complete!")

    ############################## MULTIPLE % TEST #################################
    
    st.write(f"Running multiple TP/SL scenarios and building a performance table for {ticker}")
    
    TP_SL_list = [0.01, 0.03, 0.05, 0.07, 0.10]
    progress = st.progress(0)
    perf_rows = []
    
    for idx, pct in enumerate(TP_SL_list):
        trades = []
        in_trade = False
        current_trade = {}
        progress.progress((idx + 1) / len(TP_SL_list))
        
        for i, current_date in enumerate(daily_dates):
            current_data = df_daily.iloc[:i+1]

            if len(current_data) < 100:
                continue

            if (not in_trade):
                if i % RETRAIN_EVERY == 0 or i < 120:
                    models = train_ml_models(current_data)
                    ml_prediction = get_ml_prediction(current_data, models)
                    if ml_prediction is None or ml_prediction['confidence_score'] < ml_confidence_threshold:
                        continue

            current_ml_signal = ml_prediction['will_hit']
            current_ml_confidence = ml_prediction['confidence_score']
            
            # ENTRY LOGIC 
            if (not in_trade and
                current_ml_signal in ['TP', 'Hold', 'None'] and  
                current_ml_confidence >= ml_confidence_threshold):

                entry_price = float(df_daily.loc[current_date, 'Close'])

                # Symmetric TP & SL Scenario
                TP_price = entry_price * (1 + pct)
                SL_price = entry_price * (1 - pct)
                                
                current_trade = {
                    'entry_date': current_date,
                    'entry_price': entry_price,
                    'tp_price': TP_price,
                    'sl_price': SL_price,
                    'ml_confidence': current_ml_confidence,
                    'ml_signal': current_ml_signal
                }
                in_trade = True
            
            # EXIT LOGIC
            elif in_trade:
                last_date = daily_dates[-1]
                exit_days = (last_date - current_trade['entry_date']).days
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
                
                if current_low <= SL_price:
                    exit_reason = 'SL'
                    exit_price = SL_price
                elif current_high >= TP_price:
                    exit_reason = 'TP'
                    exit_price = TP_price
                elif current_open <= SL_price:
                    exit_reason = 'Gap_SL'
                    exit_price = min(current_open, SL_price)
                elif current_open >= TP_price:
                    exit_reason = 'Gap_TP'
                    exit_price = max(current_open, TP_price)
                elif days_in_trade >= max_holding_days:
                    exit_reason = 'Max_Hold'
                    exit_price = current_close
                
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
                    in_trade = False
                    current_trade = {}
        
        if in_trade:
            last_date = daily_dates[-1]
            exit_days = (last_date - current_trade['entry_date']).days
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
        
        result_df = pd.DataFrame(trades)
        wins = result_df['Return_%'] > 0
        n_win = wins.sum()
        n_loss = len(result_df) - n_win
        win_rate = 100. * n_win / len(result_df) if len(result_df) > 0 else 0
        total_return = result_df['Return_%'].cumsum().iloc[-1] if len(result_df) > 0 else 0
        if n_loss > 0:
            profit_factor = result_df.loc[wins, 'Return_%'].sum() / abs(result_df.loc[~wins, 'Return_%'].sum())
        else:
            profit_factor = np.nan
        
        perf_rows.append({
            'TP/SL %': f'{int(pct * 100)}%',
            'Wins': n_win,
            'Losses': n_loss,
            'Win Rate (%)': win_rate,
            'Total Return (%)': total_return,
            'Profit Factor': profit_factor
        })
    
    perf_table = pd.DataFrame(perf_rows)
    st.subheader(f'{ticker}: Symmetric TP/SL % (ML Confidence >= {ml_confidence_threshold}%)')
    with st.expander("Symmetric Tests"):
        st.markdown("""
        - **1-3%:** Notice if this is profitable.
        - **3-7%:** This needs patience and you will notice returns.
        - Symetric R/R is a perfect example of gambling, over time, results in losses.
        - Best approach is to follow high-confidence to buy-and-hold for 7-21 days and then decide if that is a winning on lossing trade.
        - Beat the trend by consistently booking profits.
        """)
    st.dataframe(perf_table)
