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
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set page config first
st.set_page_config(page_title="ML Daily Entry with Weekly Trend Filter", layout="wide")

st.title("🤖 BACKTEST - Combining Weekly & Daily")
with st.expander("Strategy Summary"):
    st.markdown("""
- **Weekly Trend Filter:** Only takes long positions when weekly SMA10 > SMA50 (uptrend detected).
- **Daily ML Entries:** Entry occurs on daily close when the ML model predicts a bullish move ("TP" or "Hold" signal) and confidence is above a set threshold.
- **Trade Entry:** One new position is opened only if not already in a trade, matching signal and weekly filter.
- **Trade Exit:** Position closes when price moves +7% (TP) or -7% (SL) intraday or at next open, or when maximum holding days are reached.
- **No Overlapping Trades:** Strategy waits for the current trade to close before taking the next entry—it never doubles up.
    """)
with st.expander("Backtest Machine Learning Workflow"):
    st.markdown("""
- **Feature Engineering:** Daily market data is enriched with technical indicators and pivot levels.
- **Labeling:** Each day is labeled as TP (target hit), SL (stop hit), or neutral, by examining next N days for price moves.
- **Model Training:** Random Forest models are trained—classification predicts label (TP, SL), regressors estimate expected returns and losses.
- **Prediction:** For each trade day, model confidence and predictions are computed, guiding entry decisions.
- **Performance Evaluation:** Trade entries/exits, returns, and equity growth are tracked. The app displays trade stats, signal breakdown, and growth curve.
    """)

with st.expander("Example: How a TSLA Trade is Entered"):
    st.markdown("""
- If TSLA (Tesla) closes at $433, and weekly SMA10 ($436) > SMA50 ($396), the weekly filter allows entry.
- If the ML model predicts a bullish move with strong confidence and no trade is open, the strategy enters at $433.72.
- TP = $464.08, SL = $403.36 set from entry. Next entry occurs only when current trade closes and all entry signals align again.
    """)

with st.expander("Example: How a TSLA Trade is Exited"):
    st.markdown("""
Trade exits when
- TSLA price reaches $464.08 (TP)
- TSLA price drops to $403.36 (SL)
- Or the trade holds for the maximum allowed days
Upon exit, waits for the next valid setup before re-entering.
    """)
    
with st.expander("SL/TP triggers inside each day?"):
    st.markdown('''
- After entry, *each day's high and low* are checked individually to simulate real-world price movement.
- If **Stop Loss (SL)** is hit by the day's **Low** (price <= SL), the trade is immediately closed on that date at SL price, even if price finishes higher later.
- If **Take Profit (TP)** is hit by the day's **High** (price >= TP), the trade is immediately closed on that date at TP price, even if price finishes lower later.
- If the next day's Open is outside either trigger, this gap condition is detected and the appropriate exit is done at open or the SL/TP boundary (whichever is reached first).
- This assures that the backtest does not "wait" for daily close, but instead closes as soon as the SL or TP threshold is touched, closely matching real trade slippage logic.
    ''')

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
    ml_confidence_threshold = st.number_input("ML Confidence Threshold", value=0.5, step=0.1)
with col6:
    max_holding_days = st.number_input("Max Holding Days", value=30, step=5)

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
    
    # Simple trend indicators
    df['Bull'] = ((df['SMA10'] > df['SMA50']) & (df['RSI'] > 50)).astype(int)
    df['Bear'] = ((df['SMA10'] < df['SMA50']) & (df['RSI'] < 50)).astype(int)
    df['Neutral'] = ((df['Bull'] == 0) & (df['Bear'] == 0)).astype(int)
    
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

def label_hit_prob_past(df, window=14, profit_target=0.05, stop_loss=0.05):
    """Label data based on hit probability"""
    df['Hit_Label'] = 0  # Default to neutral
    close_prices = df['Close'].values
    
    for i in range(len(df) - window):
        current_price = close_prices[i]
        tp = current_price * (1 + profit_target)
        sl = current_price * (1 - stop_loss)
        future_prices = close_prices[i+1:i+1+window]
        
        tp_hit = any(price >= tp for price in future_prices)
        sl_hit = any(price <= sl for price in future_prices)
        
        if tp_hit and not sl_hit:
            df.iloc[i, df.columns.get_loc('Hit_Label')] = 2  # TP
        elif sl_hit and not tp_hit:
            df.iloc[i, df.columns.get_loc('Hit_Label')] = 1  # SL
        elif tp_hit and sl_hit:
            # Which happened first?
            for j, price in enumerate(future_prices):
                if price >= tp:
                    df.iloc[i, df.columns.get_loc('Hit_Label')] = 2
                    break
                elif price <= sl:
                    df.iloc[i, df.columns.get_loc('Hit_Label')] = 1
                    break
    
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
    
    # Map class to label
    label_map = {0: 'None', 1: 'SL', 2: 'TP', 3: 'Hold', 4: 'Short'}
    will_hit = label_map.get(predicted_class, 'None')
    hit_prob = class_probs[predicted_class]
    
    # Regression predictions
    latest_scaled_return = scaler_return.transform(latest)
    latest_scaled_loss = scaler_loss.transform(latest)
    
    current_price = df['Close'].iloc[-1]
    predicted_return = model_return.predict(latest_scaled_return)[0]
    predicted_loss = model_loss.predict(latest_scaled_loss)[0]
    
    # Confidence calculation
    max_ratio = 10
    if predicted_loss != 0 and will_hit != 'None':
        ratio = predicted_return / abs(predicted_loss)
        log_ratio = np.log1p(ratio)
        max_log_ratio = np.log1p(max_ratio)
        normalized_confidence = log_ratio / max_log_ratio
        confidence_score = max(min(normalized_confidence, 1), 0) * 100
    else:
        confidence_score = hit_prob
    
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
    
    with st.spinner('Downloading market data...'):
        df_daily = get_stock_data(ticker, start_date, end_date)
        df_weekly = yf.download(ticker, period=period, interval="1wk", progress=False)
    
    if df_daily is None or df_daily.empty or df_weekly.empty:
        st.error("No data returned from Yahoo Finance.")
        st.stop()

    # Weekly trend analysis
    with st.spinner('Calculating weekly trends...'):
        df_weekly['SMA10'] = df_weekly['Close'].rolling(10).mean()
        df_weekly['SMA50'] = df_weekly['Close'].rolling(50).mean()
        df_weekly['trend_up'] = df_weekly['SMA10'] > df_weekly['SMA50']

    # Prepare daily data with ML features
    with st.spinner('Calculating technical indicators...'):
        df_daily = add_technical_indicators(df_daily)
        df_daily = add_pivots(df_daily, windows)
        df_daily = average_pivots(df_daily, windows)
        df_daily = compute_expected_return(df_daily)
        df_daily = compute_expected_loss(df_daily)
        df_daily = label_hit_prob_past(df_daily, profit_target=PROFIT_TARGET, stop_loss=STOP_LOSS)

    # Train ML models
    with st.spinner('Training ML models...'):
        models = train_ml_models(df_daily)
    
    if models[0] is None:
        st.error("Insufficient data for ML model training.")
        st.stop()

    # Backtest Logic
    st.write("Running backtest...")
    trades = []
    in_trade = False
    current_trade = {}
    
    weekly_dates = df_weekly.index
    daily_dates = df_daily.index
    
    progress_bar = st.progress(0)
    
    for i, current_date in enumerate(daily_dates):
        if i % 100 == 0:  # Update progress less frequently for performance
            progress_bar.progress(min((i + 1) / len(daily_dates), 1.0))
        
        # Find corresponding weekly trend
        weekly_mask = weekly_dates <= current_date
        if not weekly_mask.any():
            continue
            
        latest_weekly_date = weekly_dates[weekly_mask][-1]
        trend_up_value = df_weekly.loc[latest_weekly_date, 'trend_up']
        if isinstance(trend_up_value, pd.Series):
            trend_up_value = trend_up_value.iloc[0]  # take first, or handle as needed
        weekly_trend_up = bool(trend_up_value) if pd.notna(trend_up_value) else False

        # Skip if weekly trend is down (except for exiting trades)
        if not weekly_trend_up and not in_trade:
            continue
        
        # Get ML prediction (simplified - in practice, you'd retrain periodically)
        current_data = df_daily.loc[:current_date]
        if len(current_data) < 100:  # Need sufficient history
            continue
            
        ml_prediction = get_ml_prediction(current_data, models)
        if ml_prediction is None:
            continue
    
        confidence_history = []
        if ml_prediction is not None:
            confidence_history.append({'Date': current_date, 'Confidence': ml_prediction['confidence_score']})

        current_ml_signal = ml_prediction['will_hit']
        current_ml_confidence = ml_prediction['confidence_score']
        
        # ENTRY LOGIC
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
                'ml_confidence': current_ml_confidence,
                'ml_signal': current_ml_signal
            }
            in_trade = True
        
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
            
            # Check exit conditions in priority order
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

    # Results Analysis
    results = pd.DataFrame(trades)
    conf_df = pd.DataFrame(confidence_history)
    
    if results.empty:
        st.warning("No trades executed. Check ML predictions and weekly trend conditions.")
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
    
    # Display results
    st.subheader("📊 ML Strategy Performance Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", total_trades)
    col2.metric("Win Rate", f"{win_rate:.1f}%")
    col3.metric("Avg Return per Trade", f"{avg_return:.2f}%")
    col4.metric("Net Return", f"{net_return_pct:.2f}%")

    # Trade outcomes breakdown
    st.subheader("Trade Outcomes")
    outcome_counts = results['Outcome'].value_counts()
    st.write(outcome_counts)

    # Display trades
    st.subheader("Trade History")
    st.dataframe(results.sort_values('EntryDate', ascending=False))

    # Plot results
    st.subheader("Backtest and Equity")
    fig, (ax, bx, cx) = plt.subplots(3, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    
    ax.plot(df_daily.index, df_daily['Close'], color='gray', linewidth=1.2, alpha=0.5, label = 'Price')
    ax.plot(df_daily.index, df_daily['SMA10'], color='orange', linewidth=1.0, alpha=0.7, label = 'SMA10')
    ax.plot(df_daily.index, df_daily['SMA50'], color='red', linewidth=1.0, alpha=0.5, label = 'SMA50')
    
    ax.fill_between(df_daily.index, df_daily['SMA10'], df_daily['SMA50'],
                    where=(df_daily['SMA10'] > df_daily['SMA50']),
                    color='green', alpha=0.15)
    
    ax.fill_between(df_daily.index, df_daily['SMA10'], df_daily['SMA50'],
                    where=(df_daily['SMA10'] < df_daily['SMA50']),
                    color='red', alpha=0.15)
    
    bx.plot(results['ExitDate'], results['Cumulative'], color='gray', linewidth=1.0, alpha=0.5)
    
    bx.axhline(1.0, color='red', linestyle='--', alpha=0.5)
    bx.set_ylabel('Equity ($)')
    bx.set_xlabel('Date')
    entry_label_shown = False
    tp_label_shown = False
    sl_label_shown = False
    other_label_shown = False
    entry_annotated = False
    for i in range(0, len(results), 2):
        outcome = results['Outcome'].iloc[i]
        color = 'green' if outcome == 'TP' else 'red' if outcome == 'SL' else 'black'
        # Plot entry (all blue, no label)
        ax.scatter(results['EntryDate'].iloc[i], results['EntryPrice'].iloc[i], color='blue', s=7, zorder=5, alpha=0.5)
        # Plot exit by outcome
        label = None
        if outcome == 'TP' and 'TP' not in ax.get_legend_handles_labels()[1]:
            label = 'TP'
        elif outcome == 'SL' and 'SL' not in ax.get_legend_handles_labels()[1]:
            label = 'SL'
        elif outcome not in ['TP', 'SL'] and 'Other' not in ax.get_legend_handles_labels()[1]:
            label = 'Other'
        ax.scatter(results['ExitDate'].iloc[i], results['ExitPrice'].iloc[i], color=color, label=label, s=7, zorder=5, alpha=0.7)
        # Annotate only the first blue entry if desired
        if not entry_annotated:
            ax.annotate('Entry', (results['EntryDate'].iloc[i], results['EntryPrice'].iloc[i]), xytext=(0, -12), textcoords='offset points', fontsize=8, color='blue')
            entry_annotated = True
            
    rd = pd.to_datetime(results['EntryDate'])
    cx.scatter(rd, results['ML_Confidence'], color='violet', alpha=0.8, s=7, label='ML Confidence')
    #cx.set_ylim(0, 100)
    
    ax.set_title(f'{ticker} Price Chart')
    bx.set_title(f'Total Equity Over Time')
    cx.set_title(f'ML Confidence Over Time')

    ax.grid(alpha=0.3)
    bx.grid(alpha=0.3)
    cx.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize='x-small')
    bx.legend(loc='lower left', fontsize='x-small')
    cx.legend()
    
    ax.set_ylabel('Price', labelpad=20)
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    ax.yaxis.set_label_coords(1.08, 0.5)
    
    bx.set_ylabel('Equity', labelpad=20)
    bx.yaxis.set_label_position('right')
    bx.yaxis.tick_right()
    bx.yaxis.set_label_coords(1.08, 0.5)

    cx.set_ylabel('ML Conf', labelpad=20)
    cx.yaxis.set_label_position('right')
    cx.yaxis.tick_right()
    cx.yaxis.set_label_coords(1.08, 0.5)
    cx.set_ylimit(0, 100)

    fig.tight_layout()
    st.pyplot(fig)

    # ML Performance Analysis
    st.subheader("ML Signal Performance")
    if 'ML_Signal' in results.columns:
        signal_performance = results.groupby('ML_Signal').agg({
            'Return_%': ['count', 'mean', 'std'],
            'ML_Confidence': 'mean'
        }).round(2)
        st.dataframe(signal_performance)

    st.success("Backtest complete!")

    ################################################################################
    ### MULTIPLE % TEST #####
    
    st.write(f"Running Multiple scenarios of TP/SLs and building a table for {ticker}")
    TP_SL_list = [0.01, 0.03, 0.05, 0.07, 0.10]
    progress = st.progress(0)
    perf_rows = []
    for idx, pct in enumerate(TP_SL_list):
        trades = []
        in_trade = False
        current_trade = {}
        progress.progress((idx + 1) / len(TP_SL_list))
        for i, current_date in enumerate(daily_dates):
            # Weekly filter + ML prediction
            weekly_mask = weekly_dates <= current_date
            if not weekly_mask.any():
                continue
            latest_weekly_date = weekly_dates[weekly_mask][-1]
            trend_up_value = df_weekly.loc[latest_weekly_date, 'trend_up']
            if isinstance(trend_up_value, pd.Series):
                trend_up_value = trend_up_value.iloc[0]
            weekly_trend_up = bool(trend_up_value) if pd.notna(trend_up_value) else False
            if not weekly_trend_up and not in_trade:
                continue
            current_data = df_daily.loc[:current_date]
            if len(current_data) < 100:
                continue
            ml_prediction = get_ml_prediction(current_data, models)
            if ml_prediction is None or ml_prediction['confidence_score'] < 0.5:
                continue
            current_ml_signal = ml_prediction['will_hit']
            current_ml_confidence = ml_prediction['confidence_score']
            # ENTRY LOGIC
            if (not in_trade and current_ml_signal in ['TP', 'Hold'] and weekly_trend_up):
                entry_price = float(df_daily.loc[current_date, 'Close'])
                TP_price = entry_price * (1 + pct)
                SL_price = entry_price * (1 - pct)
                current_trade = {'entry_date': current_date, 'entry_price': entry_price, 'tp_price': TP_price, 'sl_price': SL_price, 'ml_confidence': current_ml_confidence, 'ml_signal': current_ml_signal}
                in_trade = True
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
        perf_rows.append({'TP/SL %': f'{int(pct * 100)}%', 'Wins': n_win, 'Losses': n_loss, 'Win Rate (%)': win_rate, 'Total Return (%)': total_return, 'Profit Factor': profit_factor})
    
    perf_table = pd.DataFrame(perf_rows)
    st.subheader(f'{ticker} Performance by TP/SL Percent (ML Confidence >= 50%)')
    st.dataframe(perf_table)

