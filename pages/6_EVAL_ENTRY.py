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

# Import your custom TA module (make sure it's available)
try:
    import imports.ta as ta
except:
    # Fallback - define minimal TA functions if needed
    class TA:
        @staticmethod
        def calculate_atr(high, low, close, window=14):
            tr = np.maximum(high - low, 
                           np.maximum(abs(high - close.shift()), 
                                     abs(low - close.shift())))
            return tr.rolling(window).mean()
        
        @staticmethod 
        def calculate_rsi(df, window=14):
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
    
    ta = TA()

# Configuration
st.set_page_config(page_title="Entry Position Analyzer", layout="wide")

# Global Parameters
YEARS_OF_DATA = 2
PROFIT_TARGET = 0.0375
STOP_LOSS = 0.0375
_DAYS = 22
_Nr = 50

FEATURES = [
    'High', 'Low', 'RSI', 'RSI_SMA', 'CCI', '+DI', '-DI', 'ADX', 'ATR', 
    'VI+', 'KCu', 'KCl', 'KCu_outer', 'KCl_outer', 'Kasym', 'Kcount', 
    'STu', 'STl', 'SMA1', 'SMA2', 'SMA3', 'SMA_Ratio', 'Upper_Band', 
    'Lower_Band', 'Volume_MA20', 'SMIIO', 'SMIIO_Signal', 'SMIIO_Osc', 
    'MACD', 'Signal_Line', 'return1', 'return2', 'return3', 'Volatility', 
    'Scaled_Volatility', 'DD', 'sumBuyVol', 'sumSellVol', 'vSpike', 'VPT', 
    'OBV', 'MFI', 'VWMA', 'CMF', 'Candlesticks', 'gapStrength', 'Bear', 
    'Bull', 'Short', 'Hold', 'Neutral', 'StrongBull', 'StrongBear', 
    'Exhaustion', 'PP_Avg', 'R1_Avg', 'R2_Avg', 'S1_Avg', 'S2_Avg'
]

label2str = {0: 'None', 1: 'SL', 2: 'TP', 3: 'Hold', 4: 'Short'}
expected_classes = [0, 1, 2, 3, 4]

# Functions from your script (simplified)
def get_stock_data(ticker, start_date, end_date, interval='1d'):
    """Get stock data for given timeframe"""
    df = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1), 
                     interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        return None
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df.dropna()
    return df

def add_technical_indicators(df):
    """Add technical indicators to dataframe"""
    close = df.Close
    df['Close'] = df[['Open', 'High', 'Low', 'Close']].mean(axis=1).rolling(2).mean()
    
    # Moving averages
    df['SMA1'] = df['Close'].ewm(span=int(_DAYS * 0.5), adjust=False).mean()
    df['SMA2'] = df['Close'].ewm(span=_DAYS, adjust=False).mean()
    df['SMA3'] = df['Close'].ewm(span=int(_DAYS * 2), adjust=False).mean()
    df['SMA_Ratio'] = df['SMA1'] / df['SMA2']
    
    # Basic indicators
    df['ATR'] = ta.calculate_atr(high=df.High, low=df.Low, close=df.Close)
    df['RSI'] = ta.calculate_rsi(df)
    df['RSI_SMA'] = df['RSI'].rolling(14).mean()
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=24, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Volume
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    df['buy_volume'] = (df.Close > df.Close.shift(1)) * df['Volume']
    df['sell_volume'] = (df.Close < df.Close.shift(1)) * df['Volume']
    df['sumBuyVol'] = df['buy_volume'].rolling(window=9).sum()
    df['sumSellVol'] = df['sell_volume'].rolling(window=9).sum()
    
    # Technical signals
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
    
    # Encode TI signals
    df_encoded = pd.get_dummies(df['TI'], prefix='', prefix_sep='')
    expected_cols = ['Bull', 'Bear', 'Short', 'Hold', 'Neutral']
    for col in expected_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df = pd.concat([df, df_encoded], axis=1)
    
    df['Close'] = close
    return df

def add_pivot_levels(df, window=_DAYS):
    """Add pivot levels"""
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

def compute_expected_return(df, forward_window=14, r_cols=['R1', 'R2']):
    """Compute expected returns"""
    df['Expected_Return'] = np.nan
    close_prices = df['Close'].values
    
    for i in range(len(df) - forward_window):
        current_price = close_prices[i]
        future_window = close_prices[i+1:i+1+forward_window]
        
        if future_window.size > 0:
            df.iloc[i, df.columns.get_loc('Expected_Return')] = (
                np.nanmax(future_window) - current_price
            ) / current_price
    return df

def compute_expected_loss(df, forward_window=14, s_cols=['S1', 'S2']):
    """Compute expected losses"""
    df['Expected_Loss'] = np.nan
    close_prices = df['Close'].values
    
    for i in range(len(df) - forward_window):
        current_price = close_prices[i]
        future_window = close_prices[i+1:i+1+forward_window]
        
        if future_window.size > 0:
            df.iloc[i, df.columns.get_loc('Expected_Loss')] = (
                np.nanmin(future_window) - current_price
            ) / current_price
    return df

def label_hit_prob_past(df, window=14, profit_target=0.05, stop_loss=0.05, lookback=60, tp_thresh=0.35, sl_thresh=0.4):
    """Label hit probabilities"""
    close_prices = df['Close'].values
    N = len(close_prices)
    labels = []
    
    for i in range(N):
        current_price = close_prices[i]
        tp = current_price * (1 + profit_target)
        sl = current_price * (1 - stop_loss)
        future_prices = close_prices[i + 1 : min(i + 1 + window, N)]
        
        tp_hit_idx = next((j for j, price in enumerate(future_prices) if price >= tp), None)
        sl_hit_idx = next((j for j, price in enumerate(future_prices) if price <= sl), None)
        
        # Simplified labeling logic
        if tp_hit_idx is not None and (sl_hit_idx is None or tp_hit_idx < sl_hit_idx):
            labels.append(2)  # TP
        elif sl_hit_idx is not None and (tp_hit_idx is None or sl_hit_idx < tp_hit_idx):
            labels.append(1)  # SL
        else:
            labels.append(0)  # None
    
    df['Hit_Label'] = labels
    return df

def train_models(df, timeframe):
    """Train ML models for the given timeframe"""
    df_model = df.dropna(subset=FEATURES + ['Hit_Label', 'Expected_Return', 'Expected_Loss'])
    
    if len(df_model) < _Nr:
        return None, None, None, None, None, None
    
    # Classifier for Hit Label
    X_cls = df_model[FEATURES]
    y_cls = df_model['Hit_Label'].astype(int)
    
    scaler_cls = StandardScaler()
    X_scaled_cls = scaler_cls.fit_transform(X_cls)
    
    model_class = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    model_class.fit(X_scaled_cls, y_cls)
    
    # Return model
    cls_probs = model_class.predict_proba(X_scaled_cls)
    prob_df = pd.DataFrame(0, index=np.arange(len(cls_probs)), 
                          columns=[f'Prob_Class_{c}' for c in expected_classes])
    
    for i, c in enumerate(model_class.classes_):
        if c in expected_classes:
            prob_df[f'Prob_Class_{c}'] = cls_probs[:, i]
    
    df_model = df_model.reset_index(drop=True)
    df_model = pd.concat([df_model, prob_df], axis=1)
    FEATURES_with_probs = FEATURES + [f'Prob_Class_{c}' for c in expected_classes]
    
    X_reg = df_model[FEATURES_with_probs]
    y_return = df_model['Expected_Return']
    y_loss = df_model['Expected_Loss']
    
    scaler_return = StandardScaler()
    X_scaled_return = scaler_return.fit_transform(X_reg)
    model_return = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    model_return.fit(X_scaled_return, y_return)
    
    scaler_loss = StandardScaler()
    X_scaled_loss = scaler_loss.fit_transform(X_reg)
    model_loss = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    model_loss.fit(X_scaled_loss, y_loss)
    
    return model_class, model_return, model_loss, scaler_cls, scaler_return, scaler_loss

def make_prediction(model_class, model_return, model_loss, scaler_cls, scaler_return, scaler_loss, latest_data):
    """Make prediction for latest data"""
    if latest_data[FEATURES].isnull().values.any():
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
    hit_prob = latest_prob_features[f'Prob_Class_{pred_class}']
    
    # Return/Loss prediction
    latest_prob_df = pd.DataFrame([latest_prob_features])
    latest_features_with_probs = pd.concat([latest_data[FEATURES].reset_index(drop=True), latest_prob_df], axis=1)
    
    latest_scaled_return = scaler_return.transform(latest_features_with_probs)
    latest_scaled_loss = scaler_loss.transform(latest_features_with_probs)
    
    current_price = latest_data['Close'].values[0]
    predicted_return = model_return.predict(latest_scaled_return)[0]
    predicted_loss = model_loss.predict(latest_scaled_loss)[0]
    
    predicted_tp = current_price * (1 + predicted_return)
    predicted_sl = current_price * (1 + predicted_loss)
    
    # Confidence score
    ratio = (predicted_return / abs(predicted_loss)) if (will_hit != 'None' and predicted_loss != 0) else 0
    ratio = max(ratio, 0)
    confidence_score = max(hit_prob * ratio, 0) * 100
    
    return {
        'will_hit': will_hit,
        'hit_prob': hit_prob * 100,
        'predicted_tp': predicted_tp,
        'predicted_sl': predicted_sl,
        'predicted_return': predicted_return * 100,
        'predicted_loss': predicted_loss * 100,
        'confidence': confidence_score,
        'current_price': current_price
    }

def plot_analysis(df, entry_price, prediction, timeframe, assessment):
    """Create analysis plot"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                   gridspec_kw={'height_ratios': [3, 1]}, 
                                   sharex=True)
    
    # Price plot
    ax1.plot(df.index, df['Close'], label='Price', color='black', alpha=0.7, linewidth=1)
    ax1.plot(df.index, df['SMA1'], label=f'SMA{int(_DAYS*0.5)}', color='blue', alpha=0.7, linewidth=1)
    ax1.plot(df.index, df['SMA2'], label=f'SMA{int(_DAYS*2)}', color='red', alpha=0.7, linewidth=1)
    
    # Entry point
    last_date = df.index[-1]
    ax1.plot(last_date, entry_price, '^', markersize=10, color='green', 
             label=f'Entry: ${entry_price:.2f}')
    
    # Color background based on RSI
    rsi_colors = []
    for rsi_val in df['RSI']:
        if rsi_val > 70:
            rsi_colors.append('red')
        elif rsi_val < 30:
            rsi_colors.append('green')
        else:
            rsi_colors.append('yellow')
    
    for i in range(len(df)-1):
        ax1.axvspan(df.index[i], df.index[i+1], alpha=0.1, 
                   color=rsi_colors[i], label=None)
    
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Assessment annotation
    color_map = {'Valid': 'green', 'Risky': 'orange', 'Not Recommended': 'red'}
    ax1.annotate(f'Assessment: {assessment}', 
                xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=12, weight='bold',
                bbox=dict(boxstyle='round', facecolor=color_map.get(assessment, 'gray'), alpha=0.3))
    
    # RSI plot
    ax2.plot(df.index, df['RSI'], label='RSI', color='purple', linewidth=1)
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5, label='Overbought')
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5, label='Oversold')
    ax2.axhline(50, color='gray', linestyle='-', alpha=0.3)
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.title(f'{timeframe} Analysis - {assessment}')
    plt.tight_layout()
    
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
        ticker = st.text_input("Ticker Symbol", "AAPL").upper()
    
    with col2:
        entry_price = st.number_input("Entry Price ($)", min_value=0.01, value=150.0, step=0.1)
    
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
                    df = get_stock_data(ticker, start_date, end_date, interval)
                    if df is None or len(df) < 100:
                        st.warning(f"Insufficient {timeframe} data for {ticker}")
                        continue
                    
                    # Add technical indicators
                    df = add_technical_indicators(df)
                    df = add_pivot_levels(df)
                    df = compute_expected_return(df)
                    df = compute_expected_loss(df)
                    df = label_hit_prob_past(df, profit_target=user_gain/100, stop_loss=user_loss/100)
                    
                    # Train models
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
                        
                        st.info(f"**Assessment**: {assessment}")
                        st.write(f"**Reasons**: {reasons}")
                        
                        # Plot
                        fig = plot_analysis(df, entry_price, prediction, timeframe, assessment)
                        st.pyplot(fig)
                    
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
                        
            except Exception as e:
                st.error(f"Error analyzing {ticker}: {str(e)}")

    # Instructions
    with st.expander("How to use this analyzer"):
        st.write("""
        1. **Enter Ticker Symbol**: Stock symbol (e.g., AAPL, TSLA)
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
        """)

if __name__ == "__main__":
    main()
