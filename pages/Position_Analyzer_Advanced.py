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

# ============================================================
# ====== 1. Market Signal Classification (Final Ordered) =====
# ============================================================

def generate_market_signal(df, rsi_lower=30):
    conditions = [
        # 1. Bullish Reversal
        (df['SMA1'] > df['SMA2']) & (df['SMA1'].shift(1) <= df['SMA2'].shift(1)) &
        (df['RSI'] > 45) & (df['+DI'] > df['-DI']),

        # 2. Bearish Reversal
        (df['SMA1'] < df['SMA2']) & (df['SMA1'].shift(1) >= df['SMA2'].shift(1)) &
        (df['RSI'] < 55) & (df['-DI'] > df['+DI']),

        # 3. Overbought Bull
        (df['SMA1'] > df['SMA2']) & (df['RSI'] > 85) & (df['ADX'] > 65),

        # 4. Oversold Bear
        (df['SMA1'] < df['SMA2']) & (df['RSI'] < 30) & (df['ADX'] > 40),

        # 5. Bull
        (df['SMA1'] > df['SMA2']) & (df['RSI'] >= df['RSI_SMA']) &
        (df['RSI'].between(52, 95)) & (df['+DI'] > df['-DI']) &
        (df['+DI'].between(18, 55)) & (df['Close'] > df['SMA1']),

        # 6. Bear
        (df['SMA1'] < df['SMA2']) & (df['RSI'].between(rsi_lower, 60)) &
        (df['RSI'] < df['RSI_SMA']) & (df['+DI'] < df['-DI']) &
        (df['-DI'].between(18, 55)),

        # 7. Short
        (df['SMA1'] < df['SMA2']) & (df['RSI'].between(25, 50)) &
        (df['-DI'].between(30, 55)) & (df['Close'] > df['SMA1']),

        # 8. RangeBound
        (df['ADX'] < 20) & (df['RSI'].between(45, 55)) &
        (abs(df['SMA1'] - df['SMA2']) / df['SMA2'] < 0.01),

        # 9. Hold
        ((df['SMA1'] > df['SMA2']) & (df['RSI'] >= 50)) |
        ((df['RSI'] < df['RSI_SMA']) & (df['ADX'].between(40, 75)))
    ]

    choices = [
        'Bullish_Reversal', 'Bearish_Reversal', 'Overbought_Bull', 'Oversold_Bear',
        'Bull', 'Bear', 'Short', 'RangeBound', 'Hold'
    ]

    df['Signal'] = np.select(conditions, choices, default='Unclear')
    return df


# ============================================================
# ====== 2. Dynamic Exit Decision Logic ======================
# ============================================================

def dynamic_exit_signal(prediction, threshold_confidence=70, min_ratio=0.9):
    """
    Determines early exit or hold decision based on confidence and reward-risk ratio.
    """
    confidence = prediction.get('confidence', 50)
    reward = prediction.get('predicted_return', 0)
    risk = abs(prediction.get('predicted_loss', 1))
    ratio = reward / risk if risk != 0 else 0

    if confidence < threshold_confidence:
        return "Exit", f"Low confidence ({confidence:.1f}%)"

    if ratio < min_ratio:
        return "Exit", f"Unfavorable R/R ({ratio:.2f})"

    if confidence > 85 and ratio > 1.2:
        return "Hold", f"Strong signal (Conf {confidence:.1f}%, R/R {ratio:.2f})"

    return "Hold", f"Neutral (Conf {confidence:.1f}%, R/R {ratio:.2f})"


def optimize_stop_loss(prediction, volatility_factor=0.5):
    """
    Adjust stop-loss based on predicted loss and volatility.
    """
    predicted_loss = prediction.get('predicted_loss', 0.02)
    current_price = prediction.get('current_price', 1)
    sl_percentage = abs(predicted_loss)
    adjusted_sl = sl_percentage * (1 + volatility_factor)
    prediction['adjusted_sl'] = current_price * (1 - adjusted_sl)
    return prediction


# ============================================================
# ====== 3. Example Prediction (Placeholder ML Model) ========
# ============================================================

def make_prediction(df):
    """
    Example dummy prediction to demonstrate structure.
    Replace this with your trained ML model logic.
    """
    last_row = df.iloc[-1]
    return {
        "confidence": np.random.uniform(60, 95),
        "predicted_return": np.random.uniform(0.5, 3.0),
        "predicted_loss": np.random.uniform(0.3, 1.0),
        "current_price": last_row['Close']
    }


# ============================================================
# ====== 4. Streamlit Integration ============================
# ============================================================

def display_results(ticker, df):
    prediction = make_prediction(df)
    prediction = optimize_stop_loss(prediction)
    exit_decision, reason = dynamic_exit_signal(prediction)
    current_signal = df['Signal'].iloc[-1]

    st.subheader(f"📈 {ticker} Analysis Summary")
    st.markdown(f"**Market Signal:** {current_signal}")
    st.markdown(f"**Exit Decision:** {exit_decision} — {reason}")

    rr_ratio = prediction['predicted_return'] / abs(prediction['predicted_loss'])

    st.table(pd.DataFrame({
        "Metric": ["Confidence", "Predicted Return %", "Predicted Loss %", "R/R Ratio", "Adjusted SL", "Decision"],
        "Value": [
            f"{prediction['confidence']:.1f}%",
            f"{prediction['predicted_return']:.2f}%",
            f"{prediction['predicted_loss']:.2f}%",
            f"{rr_ratio:.2f}",
            f"${prediction['adjusted_sl']:.2f}",
            exit_decision
        ]
    }))

    # Optional: visualize signal phases
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df['Close'], label='Close', color='white')

    colors = {
        'Bull': 'green', 'Bear': 'red',
        'Bullish_Reversal': 'lime', 'Bearish_Reversal': 'maroon',
        'RangeBound': 'gray', 'Hold': 'orange',
        'Overbought_Bull': 'gold', 'Oversold_Bear': 'purple'
    }
    for signal, color in colors.items():
        ax.fill_between(df.index, df['Close'].min(), df['Close'].max(),
                        where=(df['Signal'] == signal), color=color, alpha=0.08)

    ax.set_title(f"{ticker} — Signal Map")
    ax.legend()
    st.pyplot(fig)


# ============================================================
# ====== 5. Streamlit App ===================================
# ============================================================

def main():
    st.title("Smart Market Signal & Prediction System")

    ticker = st.text_input("Enter a ticker (e.g. AAPL, COIN, TSLA):", "COIN")
    period = st.selectbox("Select period:", ["3mo", "6mo", "1y", "2y"], index=1)

    if st.button("Analyze"):
        df = yf.download(ticker, period=period, interval="1d")
        df['SMA1'] = df['Close'].rolling(20).mean()
        df['SMA2'] = df['Close'].rolling(50).mean()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['RSI_SMA'] = df['RSI'].rolling(5).mean()
        adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
        df['ADX'] = adx.adx()
        df['+DI'] = adx.adx_pos()
        df['-DI'] = adx.adx_neg()
        df.dropna(inplace=True)

        df = generate_market_signal(df)
        display_results(ticker, df)


if __name__ == "__main__":
    main()
