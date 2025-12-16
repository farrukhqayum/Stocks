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
# NOTE: Removed cache clearing to help with performance, but if issues persist, you can uncomment these:
# st.cache_data.clear() 
# st.cache_resource.clear()
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
    'High_lag1', 'Low_lag1', 'Open_lag1', 
    
    # Simple Technicals (less noisy, better generalization)
    'SMA_10', 'EMA_10', 
    'RSI_14', 
    'MACD', 'MACD_Signal', 
    'Stoch_K', 'Stoch_D',
    
    # Volatility
    'ATR_14',
    
    # Volume
    'Volume_lag1'
]

# --- Helper Functions ---

# Function to clean and prep data
@st.cache_data(show_spinner=False)
def clean_data(df):
    df = df.copy()
    
    # Remove volume column and replace 0s with a small value
    df['Volume'] = df['Volume'].replace(0, 1)

    # Clean up column names
    df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df.index.name = 'Date'
    
    # Drop the Adjusted Close column
    df.drop(columns=['Adj Close'], inplace=True)
    
    # Drop the last row, as it's often incomplete data for the current period
    df.drop(df.tail(1).index, inplace=True)

    return df

# Function to calculate technical features
@st.cache_data(show_spinner=False)
def calculate_features(df):
    df = df.copy()
    
    # Price Lags
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[f'{col}_lag1'] = df[col].shift(1)
    
    # Simple Moving Average (SMA)
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()

    # Exponential Moving Average (EMA)
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    def calculate_rsi(df, window=14):
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    df['RSI_14'] = calculate_rsi(df)

    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Stochastic Oscillator
    def calculate_stoch(df, window=14, k_window=3, d_window=3):
        low_min = df['Low'].rolling(window=window).min()
        high_max = df['High'].rolling(window=window).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=d_window).mean()
        return df
    df = calculate_stoch(df)

    # Average True Range (ATR)
    def calculate_atr(df, window=14):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()
    df['ATR_14'] = calculate_atr(df)

    # Target Variable (Next period's return)
    df['Target_Return'] = df['Close'].pct_change().shift(-1)
    
    # Drop NaNs created by rolling windows/lags
    df.dropna(inplace=True)
    
    return df

# Function to train and predict using Random Forest
@st.cache_data(show_spinner=False)
def train_model(df, target_col):
    df = df.copy()
    
    if len(df) < MIN_TRAIN_ROWS.get(st.session_state.get('timeframe_select', '1D'), 30):
        # Handle case where not enough data is available (will be caught in main())
        return None, None, 0
    
    X = df[FEATURES]
    y = df[target_col]

    # Use the last 20% for testing (no leakage)
    test_size = int(len(df) * 0.20)
    train_size = len(df) - test_size
    
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # Scaling features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Classifier or Regressor based on target
    if target_col == 'Target_TP_SL':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5)
        model.fit(X_train_scaled, y_train)
        
        # Calculate confidence (for classification)
        y_pred_proba = model.predict_proba(X_test_scaled)
        confidence = np.max(y_pred_proba, axis=1).mean()
        
    else: # Regressor for 'Target_Return'
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5)
        model.fit(X_train_scaled, y_train)
        # Use R^2 as a rough 'confidence' metric for regression
        confidence = model.score(X_test_scaled, y_test)
        
    # Get the latest features for prediction (next step)
    latest_features = scaler.transform(X.tail(1))
    
    return model, latest_features, confidence

# Function to predict and summarize
def predict_and_summarize(df, model_cls, latest_features_cls, model_reg, latest_features_reg, entry_price, tp_level, sl_level):
    
    last_close = df['Close'].iloc[-1]
    
    # 1. Classification Prediction (Hit TP/SL/Hold)
    if model_cls is not None:
        cls_proba = model_cls.predict_proba(latest_features_cls)[0]
        prob_tp = cls_proba[2] # Assuming 2 is the 'TP Hit' class (depends on unique values)
        prob_sl = cls_proba[1] # Assuming 1 is the 'SL Hit' class
        
        # Re-check class mapping
        classes = model_cls.classes_
        try:
            tp_index = np.where(classes == 'TP Hit')[0][0]
            sl_index = np.where(classes == 'SL Hit')[0][0]
            hold_index = np.where(classes == 'Hold')[0][0]
            
            prob_tp = cls_proba[tp_index]
            prob_sl = cls_proba[sl_index]
            prob_hold = cls_proba[hold_index]
            
        except IndexError:
            # Fallback if class labels are unexpected
            prob_tp = cls_proba[0] # Defaulting to first class
            prob_sl = cls_proba[1] # Defaulting to second class
            prob_hold = 1 - prob_tp - prob_sl
            
        cls_confidence = model_cls.score(latest_features_cls, pd.Series(['Hold'])) # Dummy score
        
    else:
        prob_tp, prob_sl, prob_hold, cls_confidence = 0, 0, 0, 0

    # 2. Regression Prediction (Expected Return)
    if model_reg is not None:
        predicted_return = model_reg.predict(latest_features_reg)[0]
        predicted_price = last_close * (1 + predicted_return)
        reg_confidence = model_reg.score(latest_features_reg, pd.Series([0.0])) # Dummy score
    else:
        predicted_return, predicted_price, reg_confidence = 0, last_close, 0

    # 3. Risk/Reward Ratio Calculation
    risk_reward_ratio = (tp_level / sl_level)
    
    # 4. Proximity to Entry
    proximity_pct = (entry_price - last_close) / last_close * 100
    
    # 5. Technical Signal Alignment (Simplified for quick check)
    signal_alignment = []
    if df['SMA_10'].iloc[-1] > df['SMA_20'].iloc[-1]: signal_alignment.append("SMA_Bullish")
    if df['EMA_10'].iloc[-1] > df['EMA_20'].iloc[-1]: signal_alignment.append("EMA_Bullish")
    if df['RSI_14'].iloc[-1] > 50: signal_alignment.append("RSI_Bullish")
    if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]: signal_alignment.append("MACD_Bullish")
    
    signal_score = len(signal_alignment)
    
    # Summary dictionary
    results = {
        'last_close': last_close,
        'entry_price': entry_price,
        'tp_level': tp_level,
        'sl_level': sl_level,
        'prob_tp': prob_tp,
        'prob_sl': prob_sl,
        'prob_hold': prob_hold,
        'predicted_price': predicted_price,
        'predicted_return': predicted_return,
        'cls_confidence': cls_confidence,
        'reg_confidence': reg_confidence,
        'risk_reward_ratio': risk_reward_ratio,
        'proximity_pct': proximity_pct,
        'signal_score': signal_score
    }
    
    return results

# Function to format results into a summary table and recommendation
def format_results(results, profit_target, stop_loss):
    
    # Recommendation Logic
    recommendation = ""
    color = "black"
    
    # Thresholds (can be adjusted)
    min_prob_tp = 0.40
    min_rr_ratio = 1.2
    min_signal_score = 3
    
    # Check 1: Risk/Reward
    rr_ok = results['risk_reward_ratio'] >= min_rr_ratio
    
    # Check 2: ML Probability
    prob_ok = results['prob_tp'] > results['prob_sl'] and results['prob_tp'] >= min_prob_tp
    
    # Check 3: Technical Alignment
    signals_ok = results['signal_score'] >= min_signal_score
    
    # Check 4: Combined Confidence
    combined_confidence = (results['cls_confidence'] + results['reg_confidence']) / 2
    confidence_ok = combined_confidence > 0.45
    
    # Overall Assessment
    if rr_ok and prob_ok and signals_ok and confidence_ok:
        recommendation = "🟢 Valid Entry Position"
        color = "green"
    elif (rr_ok and prob_ok) or (rr_ok and signals_ok):
        recommendation = "🟡 Risky/Moderate Entry Position"
        color = "gold"
    else:
        recommendation = "🔴 Wait and See (Poor Risk/Reward or Bearish Signals)"
        color = "red"
        
    # Summary Table Construction
    summary_data = {
        "Metric": [
            "Current Price", "Intended Entry Price", "Target Price (TP)", "Stop Loss (SL)", 
            "Required Risk/Reward", 
            "ML: Probability of TP Hit", 
            "ML: Probability of SL Hit", 
            "ML: Predicted Next Return",
            "Technical Signal Score (out of 4)",
            "Model Confidence (Avg)"
        ],
        "Value": [
            f"${results['last_close']:.2f}", 
            f"${results['entry_price']:.2f}",
            f"${results['entry_price'] * (1 + profit_target):.2f}",
            f"${results['entry_price'] * (1 - stop_loss):.2f}",
            f"{results['risk_reward_ratio']:.2f} : 1",
            f"{results['prob_tp'] * 100:.2f}%",
            f"{results['prob_sl'] * 100:.2f}%",
            f"{results['predicted_return'] * 100:.2f}%",
            f"{results['signal_score']}",
            f"{combined_confidence * 100:.2f}%"
        ]
    }
    summary_table_df = pd.DataFrame(summary_data)
    
    return recommendation, color, summary_table_df


# --- Main Application Logic ---

def main():
    st.title("Intelligent Entry Position Analyzer 🤖")

    # --- Sidebar for Inputs ---
    with st.sidebar:
        st.header("1. Define Position")
        
        # Initialize session state for consistent inputs (if not already set)
        if 'ticker_input' not in st.session_state:
            st.session_state['ticker_input'] = DEFAULT_TICKER
        if 'entry_price' not in st.session_state:
            st.session_state['entry_price'] = 200.00
        if 'profit_target' not in st.session_state:
            st.session_state['profit_target'] = PROFIT_TARGET
        if 'stop_loss' not in st.session_state:
            st.session_state['stop_loss'] = STOP_LOSS
        if 'timeframe_select' not in st.session_state:
            st.session_state['timeframe_select'] = '1D'
            
        ticker = st.text_input("Ticker Symbol (e.g., AAPL, BTC-USD)", value=st.session_state['ticker_input'], key='ticker_input')
        entry_price = st.number_input("Intended Entry Price ($)", min_value=0.01, value=st.session_state['entry_price'], step=0.01, format="%.2f", key='entry_price')
        
        profit_target = st.slider("Profit Target (%)", min_value=1.0, max_value=20.0, value=st.session_state['profit_target'] * 100, step=0.1, format="%.1f") / 100.0
        stop_loss = st.slider("Max Acceptable Loss (%)", min_value=1.0, max_value=20.0, value=st.session_state['stop_loss'] * 100, step=0.1, format="%.1f") / 100.0

        st.header("2. Data & Model Settings")
        timeframe = st.selectbox(
            "Timeframe",
            ('4H', '1D', '1W'),
            index=1,
            key='timeframe_select'
        )
        
        # Calculate target TP/SL values
        tp_price = entry_price * (1 + profit_target)
        sl_price = entry_price * (1 - stop_loss)
        st.markdown(f"**Calculated TP/SL:**")
        st.info(f"TP: ${tp_price:.2f} | SL: ${sl_price:.2f}")

    
    # --- Analysis Trigger ---
    if st.button("Analyze Entry Position"):
        
        # Clear previous results and set status
        st.session_state['entry_analyzer_results'] = None
        
        # Input Validation
        if entry_price <= 0 or profit_target <= 0 or stop_loss <= 0:
            st.error("Entry Price, Profit Target, and Stop Loss must be positive.")
            return

        with st.spinner(f"Fetching data and training ML models for {ticker} ({timeframe})..."):
            
            try:
                # 1. Data Fetching
                end_date = datetime.now()
                start_date = end_date - timedelta(days=YEARS_OF_DATA[timeframe] * 365)
                
                interval = {'4H': '4h', '1D': '1d', '1W': '1wk'}.get(timeframe, '1d')
                
                data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)

                if data.empty:
                    st.error(f"Could not fetch data for {ticker}. Check the ticker or try a different timeframe.")
                    return

                # 2. Data Preparation
                daily_df_temp = clean_data(data)
                daily_df_temp = calculate_features(daily_df_temp)
                
                if daily_df_temp.empty:
                    st.error("Not enough historical data available after cleaning and feature calculation. Try a different timeframe or ticker.")
                    return
                
                # 3. Target Variable for Classification
                # Check if price hits TP (1+profit_target) or SL (1-stop_loss) in the next _Nr periods
                
                # Check for TP/SL hit over the next N periods
                def check_tp_sl_hit(df, N, tp, sl):
                    results = []
                    for i in range(len(df) - N):
                        future_prices = df['Close'].iloc[i+1:i+N+1]
                        current_close = df['Close'].iloc[i]
                        
                        current_tp = current_close * (1 + tp)
                        current_sl = current_close * (1 - sl)
                        
                        hit_tp = (future_prices >= current_tp).any()
                        hit_sl = (future_prices <= current_sl).any()
                        
                        if hit_tp and hit_sl:
                            # Prioritize TP if it happens first (simple version: who cares, we just check if it was hit)
                            # To simplify, we only care about which one was hit
                            if hit_tp: results.append('TP Hit')
                            else: results.append('SL Hit')
                        elif hit_tp:
                            results.append('TP Hit')
                        elif hit_sl:
                            results.append('SL Hit')
                        else:
                            results.append('Hold')
                            
                    # Pad the end with 'Hold' for the rows that don't have N future periods
                    results.extend(['Hold'] * N) 
                    return pd.Series(results, index=df.index)

                daily_df_temp['Target_TP_SL'] = check_tp_sl_hit(daily_df_temp, _Nr, profit_target, stop_loss)
                daily_df_temp.dropna(subset=['Target_TP_SL'], inplace=True)
                daily_df_temp.dropna(inplace=True)

                if len(daily_df_temp) < MIN_TRAIN_ROWS.get(timeframe, 30):
                    st.error(f"Not enough data points ({len(daily_df_temp)}) for training. Need at least {MIN_TRAIN_ROWS.get(timeframe, 30)} rows. Try a longer timeframe.")
                    return
                
                # 4. Model Training
                model_cls, latest_features_cls, cls_confidence = train_model(daily_df_temp, 'Target_TP_SL')
                model_reg, latest_features_reg, reg_confidence = train_model(daily_df_temp, 'Target_Return')
                
                if model_cls is None or model_reg is None:
                    st.error("Model training failed. Check data availability and try again.")
                    return

                # 5. Prediction and Summary
                results = predict_and_summarize(
                    daily_df_temp, 
                    model_cls, latest_features_cls, 
                    model_reg, latest_features_reg, 
                    entry_price, profit_target, stop_loss
                )
                
                recommendation_text, recommendation_color, summary_table_df = format_results(results, profit_target, stop_loss)
                
                
                # --- START OF FIX: STORE RESULTS IN SESSION STATE ---
                st.session_state['entry_analyzer_results'] = {
                    'results': results,
                    'daily_df': daily_df_temp,
                    'entry_price': entry_price,
                    'profit_target': profit_target, # Store for MC calculation
                    'stop_loss': stop_loss,       # Store for MC calculation
                    'ticker': ticker,
                    'timeframe': timeframe,
                    'recommendation_text': recommendation_text,
                    'recommendation_color': recommendation_color,
                    'summary_table_df': summary_table_df,
                }
                
                st.success(f"Analysis for **{ticker}** complete! Scroll down for results and interactive simulations.")
                # --- END OF FIX: STORE RESULTS IN SESSION STATE ---


            except Exception as e:
                st.error(f"An error occurred during analysis for {ticker}: {str(e)}")
                st.info("Try with a different ticker or check if market is open")


    # --- START OF FIX: DISPLAY LOGIC (MOVED OUTSIDE BUTTON BLOCK) ---
    if st.session_state.get('entry_analyzer_results') is not None:
        
        # Retrieve stored results
        stored_results = st.session_state['entry_analyzer_results']
        results = stored_results['results']
        daily_df_temp = stored_results['daily_df']
        entry_price = stored_results['entry_price']
        profit_target = stored_results['profit_target']
        stop_loss = stored_results['stop_loss']
        ticker = stored_results['ticker']
        timeframe = stored_results['timeframe']
        recommendation_text = stored_results['recommendation_text']
        recommendation_color = stored_results['recommendation_color']
        summary_table_df = stored_results['summary_table_df']

        st.markdown("---")
        st.subheader(f"📊 Analysis for **{ticker}** @ ${entry_price:.2f}")

        # Overall Recommendation
        st.subheader("🎯 Overall Recommendation")
        st.markdown(f"**<p style='color:{recommendation_color}; font-size: 20px;'>{recommendation_text}</p>**", unsafe_allow_html=True)
        
        # Summary Table
        st.subheader("📋 Analysis Summary")
        st.dataframe(summary_table_df, use_container_width=True, hide_index=True)

        # Monte Carlo Simulation
        with st.expander(f"🔮 Interactive Monte Carlo Simulation for {ticker}"):
            st.caption(f"Simulations use historical volatility and returns from the {timeframe} data. Interacting with these sliders will **NOT** remove your analysis results.")
            
            col_mc1, col_mc2 = st.columns([1, 1])

            with col_mc1:
                # Sliders are now outside the button and update session state
                days = st.slider("Forecast Horizon (Trading Days)", min_value=5, max_value=252, value=_DAYS, step=5, key='mc_days')
            with col_mc2:
                num_sims = st.slider("Number of Simulations", min_value=100, max_value=2000, value=500, step=100, key='mc_num_sims')
            
            mc_method = st.radio(
                "Monte Carlo Model",
                ("Geometric Brownian Motion (GBM)", "Historical Returns (Bootstrap)"),
                horizontal=True,
                index=0,
                key='mc_method'
            )

            # NOTE: The simulation uses the data stored in session state: daily_df_temp, entry_price, profit_target, stop_loss
            if st.button("Run Simulation", key='run_mc'): 
                
                # Check if data is valid
                if daily_df_temp is None or daily_df_temp.empty:
                    st.error("Simulation data is missing. Please run the initial analysis first.")
                    return
                
                st.info(f"Running {num_sims} simulations for the next {days} trading days starting from **${daily_df_temp['Close'].iloc[-1]:.2f}**.")
                
                # Calculate daily returns and volatility from stored data
                log_returns = np.log(daily_df_temp['Close'] / daily_df_temp['Close'].shift(1)).dropna()
                mu = log_returns.mean()
                sigma = log_returns.std()

                # Set up simulation
                S0 = daily_df_temp['Close'].iloc[-1]
                price_paths = []

                if mc_method == "Geometric Brownian Motion (GBM)":
                    for _ in range(num_sims):
                        # Generate random noise
                        z = np.random.normal(0, 1, days)
                        # Calculate daily returns
                        daily_returns = np.exp(mu - 0.5 * sigma**2 + sigma * z)
                        # Generate price path
                        path = S0 * daily_returns.cumprod()
                        price_paths.append(path)
                
                elif mc_method == "Historical Returns (Bootstrap)":
                    # Use bootstrapping from historical returns
                    for _ in range(num_sims):
                        # Resample 'days' number of returns
                        resampled_returns = np.random.choice(log_returns.values, size=days)
                        # Convert log returns back to price factors (exp(returns))
                        price_factors = np.exp(resampled_returns)
                        # Generate price path
                        path = S0 * price_factors.cumprod()
                        price_paths.append(path)


                # Analysis of Simulation Results
                final_prices = [path.iloc[-1] if isinstance(path, pd.Series) else path[-1] for path in price_paths]
                
                # Probabilities relative to the *Entry Price*
                prob_above_entry = np.mean(np.array(final_prices) > entry_price) * 100
                
                # Probabilities relative to TP/SL (if defined)
                tp = entry_price * (1 + profit_target)
                sl = entry_price * (1 - stop_loss)
                
                prob_hit_tp = np.mean(np.array(final_prices) >= tp) * 100
                prob_hit_sl = np.mean(np.array(final_prices) <= sl) * 100
                
                # Plotting
                fig, ax = plt.subplots(figsize=(12, 6))
                for path in price_paths:
                    ax.plot(range(1, days + 1), path, alpha=0.1, color='lightblue')
                
                # Highlight the last price of the real data
                ax.axhline(S0, color='grey', linestyle='--', label=f'Current Price: ${S0:.2f}', linewidth=1.5)
                # Highlight entry price
                ax.axhline(entry_price, color='purple', linestyle=':', label=f'Entry Price: ${entry_price:.2f}', linewidth=1.5)
                # Highlight TP/SL levels
                ax.axhline(tp, color='green', linestyle='--', label=f'Target Price (TP): ${tp:.2f}', linewidth=1.5)
                ax.axhline(sl, color='red', linestyle='--', label=f'Stop Loss (SL): ${sl:.2f}', linewidth=1.5)

                # Plot the average path
                average_path = np.mean(np.array(price_paths), axis=0)
                ax.plot(range(1, days + 1), average_path, color='blue', linewidth=2.5, label='Average Forecast')

                ax.set_title(f'{ticker} Monte Carlo Simulation ({mc_method})')
                ax.set_xlabel('Trading Day')
                ax.set_ylabel('Simulated Price ($)')
                ax.legend()
                st.pyplot(fig)
                
                st.subheader("Simulation Probabilities")
                
                col_mc_prob1, col_mc_prob2, col_mc_prob3 = st.columns(3)
                
                col_mc_prob1.metric("Probability Above Entry", f"{prob_above_entry:.2f}%")
                col_mc_prob2.metric("Probability of Hitting TP", f"{prob_hit_tp:.2f}%", delta=f"{tp:.2f} Target")
                col_mc_prob3.metric("Probability of Hitting SL", f"{prob_hit_sl:.2f}%", delta=f"-{sl:.2f} Loss")

                st.markdown("""
                    ---
                    *Note on Interpretation:*
                    - **Probability of Hitting TP/SL** assesses the likelihood of the price touching your target/loss level by the end of the forecast horizon.
                    - **Probability Above Entry** assesses the likelihood of the final simulated price being higher than your *intended entry price*.
                """)

            
    # --- END OF FIX: DISPLAY LOGIC (MOVED OUTSIDE BUTTON BLOCK) ---
    
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
        - **1W**: 8 years of historical data (~400+ data points)

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
        1W data requires at least 8 years of history for sufficient data points.
        """
        )


if __name__ == "__main__":
    main()
