#!/usr/bin/env python
# coding: utf-8

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

# Enhanced TA functions to replace the imports
class TechnicalAnalysis:
    @staticmethod
    def calculate_atr(high, low, close, window=14):
        """Calculate Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window).mean()
    
    @staticmethod 
    def calculate_rsi(series, window=14):
        """Calculate RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_ema(series, span):
        """Calculate Exponential Moving Average"""
        return series.ewm(span=span, adjust=False).mean()
    
    @staticmethod
    def calculate_sma(series, window):
        """Calculate Simple Moving Average"""
        return series.rolling(window).mean()

# Initialize TA
ta = TechnicalAnalysis()

# Configuration
st.set_page_config(page_title="Entry Position Analyzer", layout="wide")

# Global Parameters
YEARS_OF_DATA = 1  # Reduced for faster processing
PROFIT_TARGET = 0.0375
STOP_LOSS = 0.0375
_DAYS = 22
_Nr = 30  # Reduced minimum data requirement

# Simplified features for faster processing
FEATURES = [
    'High', 'Low', 'Close', 'Volume', 'RSI', 'RSI_SMA', 'SMA1', 'SMA2', 
    'SMA3', 'SMA_Ratio', 'ATR', 'MACD', 'Signal_Line', 'Volume_MA20',
    'sumBuyVol', 'sumSellVol', 'Volatility', 'Bull', 'Bear', 'Hold', 'Neutral'
]

label2str = {0: 'None', 1: 'SL', 2: 'TP', 3: 'Hold', 4: 'Short'}
expected_classes = [0, 1, 2, 3, 4]

def get_stock_data(ticker, start_date, end_date, interval='1d'):
    """Get stock data for given timeframe with proper date handling"""
    try:
        df = yf.download(ticker, start=start_date, end=end_date, 
                        interval=interval, progress=False, auto_adjust=True)
        
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

def add_technical_indicators(df):
    """Add essential technical indicators to dataframe"""
    try:
        # Store original close
        close_orig = df['Close'].copy()
        
        # Smooth close price for calculations
        df['Close_Smooth'] = df[['Open', 'High', 'Low', 'Close']].mean(axis=1).rolling(2).mean()
        
        # Moving averages
        df['SMA1'] = ta.calculate_ema(df['Close_Smooth'], span=int(_DAYS * 0.5))
        df['SMA2'] = ta.calculate_ema(df['Close_Smooth'], span=_DAYS)
        df['SMA3'] = ta.calculate_ema(df['Close_Smooth'], span=int(_DAYS * 2))
        df['SMA_Ratio'] = df['SMA1'] / df['SMA2']
        
        # RSI
        df['RSI'] = ta.calculate_rsi(df['Close_Smooth'])
        df['RSI_SMA'] = df['RSI'].rolling(14).mean()
        
        # ATR
        df['ATR'] = ta.calculate_atr(df['High'], df['Low'], df['Close_Smooth'])
        
        # MACD
        ema12 = ta.calculate_ema(df['Close_Smooth'], span=12)
        ema26 = ta.calculate_ema(df['Close_Smooth'], span=26)
        df['MACD'] = ema12 - ema26
        df['Signal_Line'] = ta.calculate_ema(df['MACD'], span=9)
        
        # Volume indicators
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        df['buy_volume'] = (df['Close_Smooth'] > df['Close_Smooth'].shift(1)) * df['Volume']
        df['sell_volume'] = (df['Close_Smooth'] < df['Close_Smooth'].shift(1)) * df['Volume']
        df['sumBuyVol'] = df['buy_volume'].rolling(window=9).sum()
        df['sumSellVol'] = df['sell_volume'].rolling(window=9).sum()
        
        # Volatility
        df['Volatility'] = df['Close_Smooth'].rolling(14).std()
        
        # Technical signals - simplified conditions
        conditions = [
            # Bull condition
            (df['SMA1'] > df['SMA2']) & (df['RSI'] > 50) & (df['Close_Smooth'] > df['SMA1']),
            
            # Bear condition  
            (df['SMA1'] < df['SMA2']) & (df['RSI'] < 50) & (df['Close_Smooth'] < df['SMA1']),
            
            # Hold condition
            (df['SMA1'] > df['SMA2']) & (df['RSI'] > 45) & (df['RSI'] < 70),
            
            # Short condition (simplified)
            (df['SMA1'] < df['SMA2']) & (df['RSI'] < 40)
        ]
        
        choices = ['Bull', 'Bear', 'Hold', 'Short']
        df['TI'] = np.select(conditions, choices, default='Neutral')
        
        # One-hot encode TI signals
        for signal in ['Bull', 'Bear', 'Hold', 'Short', 'Neutral']:
            df[signal] = (df['TI'] == signal).astype(int)
        
        # Restore original close price
        df['Close'] = close_orig
        df = df.drop('Close_Smooth', axis=1)
        
        return df
        
    except Exception as e:
        st.error(f"Error adding technical indicators: {str(e)}")
        return None

def compute_expected_return(df, forward_window=14):
    """Compute expected returns"""
    try:
        df['Expected_Return'] = np.nan
        close_prices = df['Close'].values
        
        for i in range(len(df) - forward_window):
            current_price = close_prices[i]
            future_prices = close_prices[i+1:i+1+forward_window]
            
            if len(future_prices) > 0:
                max_future = np.nanmax(future_prices)
                df.iloc[i, df.columns.get_loc('Expected_Return')] = (max_future - current_price) / current_price
        
        return df
    except Exception as e:
        st.error(f"Error computing expected returns: {str(e)}")
        return df

def compute_expected_loss(df, forward_window=14):
    """Compute expected losses"""
    try:
        df['Expected_Loss'] = np.nan
        close_prices = df['Close'].values
        
        for i in range(len(df) - forward_window):
            current_price = close_prices[i]
            future_prices = close_prices[i+1:i+1+forward_window]
            
            if len(future_prices) > 0:
                min_future = np.nanmin(future_prices)
                df.iloc[i, df.columns.get_loc('Expected_Loss')] = (min_future - current_price) / current_price
        
        return df
    except Exception as e:
        st.error(f"Error computing expected losses: {str(e)}")
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
        
        if len(df_model) < _Nr:
            st.warning(f"Insufficient data for {timeframe} modeling: {len(df_model)} rows")
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
        confidence_score = max((hit_prob/100) * ratio, 0) * 100
        
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

def plot_analysis(df, entry_price, timeframe, assessment):
    """Create analysis plot"""
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                       gridspec_kw={'height_ratios': [3, 1]}, 
                                       sharex=True)
        
        # Price plot
        ax1.plot(df.index, df['Close'], label='Price', color='black', alpha=0.7, linewidth=1)
        
        # SMAs if available
        if 'SMA1' in df.columns:
            ax1.plot(df.index, df['SMA1'], label=f'SMA{int(_DAYS*0.5)}', color='blue', alpha=0.7, linewidth=1)
        if 'SMA2' in df.columns:
            ax1.plot(df.index, df['SMA2'], label=f'SMA{int(_DAYS*2)}', color='red', alpha=0.7, linewidth=1)
        
        # Entry point
        last_date = df.index[-1]
        ax1.plot(last_date, entry_price, '^', markersize=10, color='green', 
                 label=f'Entry: ${entry_price:.2f}')
        
        # RSI-based background coloring if available
        if 'RSI' in df.columns:
            for i in range(len(df)-1):
                rsi_val = df['RSI'].iloc[i]
                if rsi_val > 70:
                    color = 'red'
                    alpha = 0.1
                elif rsi_val < 30:
                    color = 'green' 
                    alpha = 0.1
                else:
                    color = 'yellow'
                    alpha = 0.05
                
                ax1.axvspan(df.index[i], df.index[i+1], alpha=alpha, color=color)
        
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Assessment annotation
        color_map = {'Valid': 'green', 'Risky': 'orange', 'Not Recommended': 'red'}
        assessment_color = color_map.get(assessment, 'gray')
        
        ax1.annotate(f'Assessment: {assessment}', 
                    xy=(0.02, 0.95), xycoords='axes fraction',
                    fontsize=12, weight='bold',
                    bbox=dict(boxstyle='round', facecolor=assessment_color, alpha=0.3))
        
        # RSI plot if available
        if 'RSI' in df.columns:
            ax2.plot(df.index, df['RSI'], label='RSI', color='purple', linewidth=1)
            ax2.axhline(70, color='red', linestyle='--', alpha=0.5, label='Overbought')
            ax2.axhline(30, color='green', linestyle='--', alpha=0.5, label='Oversold')
            ax2.axhline(50, color='gray', linestyle='-', alpha=0.3)
            ax2.set_ylabel('RSI')
            ax2.set_ylim(0, 100)
            ax2.legend()
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
    if price_diff_pct > 5:
        reasons.append("Entry price far from current price")
    elif price_diff_pct > 2:
        reasons.append("Entry price moderately different")
    else:
        reasons.append("Entry price close to current")
    
    # Overall assessment
    bullish_conditions = (will_hit == 'TP' and hit_prob > 40 and confidence > 40 and pred_rr > 1.5)
    risky_conditions = (will_hit in ['TP', 'Hold'] and confidence > 30 and pred_rr > 1)
    
    if bullish_conditions and price_diff_pct <= 3:
        assessment = "Valid"
    elif risky_conditions and price_diff_pct <= 5:
        assessment = "Risky"
    else:
        assessment = "Not Recommended"
    
    return assessment, " | ".join(reasons)

# Streamlit App
def main():
    st.title("📊 Entry Position Analyzer")
    st.write("Analyze your entry position using ML models trained on 1H and 1D timeframes")
    
    # User inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ticker = st.text_input("Ticker Symbol", "TSLA").upper()
    
    with col2:
        entry_price = st.number_input("Entry Price ($)", min_value=0.01, value=250.0, step=0.1)
    
    with col3:
        user_gain = st.number_input("Expected Gain (%)", min_value=0.1, value=5.0, step=0.1)
        user_loss = st.number_input("Expected Loss (%)", min_value=0.1, value=3.0, step=0.1)
    
    if st.button("Analyze Entry Position"):
        with st.spinner("Training models and analyzing..."):
            try:
                # Get current date range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365 * YEARS_OF_DATA)
                
                results = {}
                
                # Analyze both timeframes
                for timeframe, interval in [("1H", "1h"), ("1D", "1d")]:
                    st.subheader(f"{timeframe} Timeframe Analysis")
                    
                    # Get data
                    with st.spinner(f"Fetching {timeframe} data..."):
                        df = get_stock_data(ticker, start_date, end_date, interval)
                    
                    if df is None or len(df) < 50:
                        st.warning(f"Insufficient {timeframe} data for {ticker}")
                        continue
                    
                    # Add technical indicators
                    with st.spinner("Calculating technical indicators..."):
                        df = add_technical_indicators(df)
                    
                    if df is None:
                        st.warning(f"Error calculating indicators for {timeframe}")
                        continue
                    
                    # Compute expected returns/losses
                    with st.spinner("Computing expected returns..."):
                        df = compute_expected_return(df)
                        df = compute_expected_loss(df)
                    
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
                        fig = plot_analysis(df, entry_price, timeframe, assessment)
                        st.pyplot(fig)
                    else:
                        st.warning(f"Could not generate prediction for {timeframe}")
                    
                    st.write("---")
                
                # Overall recommendation
                if len(results) == 2:
                    st.subheader("🎯 Overall Recommendation")
                    
                    assessments = [results[tf]['assessment'] for tf in ['1H', '1D']]
                    valid_count = assessments.count('Valid')
                    risky_count = assessments.count('Risky')
                    
                    if valid_count == 2:
                        st.success("**STRONG BUY** - Both timeframes show valid entry")
                    elif valid_count >= 1 or risky_count == 2:
                        st.warning("**CAUTIOUS BUY** - Mixed or risky signals")
                    else:
                        st.error("**AVOID** - Not recommended in both timeframes")
                        
                elif len(results) == 1:
                    timeframe = list(results.keys())[0]
                    assessment = results[timeframe]['assessment']
                    
                    if assessment == "Valid":
                        st.success(f"**CONSIDER BUY** - {timeframe} shows valid entry")
                    elif assessment == "Risky":
                        st.warning(f"**CAUTIOUS** - {timeframe} shows risky entry")
                    else:
                        st.error(f"**AVOID** - {timeframe} shows poor entry")
                        
            except Exception as e:
                st.error(f"Error analyzing {ticker}: {str(e)}")
                st.info("Try with a different ticker or check if market is open")

    # Instructions
    with st.expander("How to use this analyzer"):
        st.write("""
        1. **Enter Ticker Symbol**: Stock symbol (e.g., AAPL, TSLA, NVDA)
        2. **Set Entry Price**: Your intended entry price
        3. **Define Expectations**: Your target gain and maximum acceptable loss
        4. **Click Analyze**: The system will train ML models and evaluate your entry
        
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
        """)

if __name__ == "__main__":
    main()
