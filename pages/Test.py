import streamlit as st import pandas as pd import numpy as np import ta from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier from datetime import datetime, timedelta import yfinance as yf import matplotlib.pyplot as plt

=============================

Utility Functions

=============================

def load_data(ticker, period="1y", interval="1d"): data = yf.download(ticker, period=period, interval=interval) data.dropna(inplace=True) return data

def compute_indicators(df): df['RSI'] = ta.momentum.RSIIndicator(df['Close'], 14).rsi() df['MACD'] = ta.trend.MACD(df['Close']).macd() df['Signal'] = ta.trend.MACD(df['Close']).macd_signal() df['SMA1'] = df['Close'].rolling(20).mean() df['SMA2'] = df['Close'].rolling(50).mean() df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range() df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx() df['ATR%'] = df['ATR'] / df['Close'] * 100 df.dropna(inplace=True) return df

def compute_pivots(df): df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3 df['R1'] = 2 * df['Pivot'] - df['Low'] df['S1'] = 2 * df['Pivot'] - df['High'] return df

=============================

Model & Signal Logic

=============================

def train_models(df): features = ['RSI', 'MACD', 'SMA1', 'SMA2', 'ATR', 'ADX'] df = df.copy() df['Target'] = df['Close'].shift(-1) / df['Close'] - 1 df.dropna(inplace=True)

X = df[features]
y_reg = df['Target']
y_cls = (y_reg > 0.005).astype(int)  # TP-before-SL proxy

reg = RandomForestRegressor(n_estimators=100, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
reg.fit(X, y_reg)
clf.fit(X, y_cls)
return reg, clf

def make_recommendation(df, reg, clf): latest = df.iloc[-1:] features = ['RSI', 'MACD', 'SMA1', 'SMA2', 'ATR', 'ADX'] X_latest = latest[features]

predicted_return = reg.predict(X_latest)[0]
prob_TP = clf.predict_proba(X_latest)[0][1]

price = latest['Close'].values[0]
SMA1 = latest['SMA1'].values[0]
SMA2 = latest['SMA2'].values[0]
RSI = latest['RSI'].values[0]
ADX = latest['ADX'].values[0]
ATR_percent = latest['ATR%'].values[0]

uptrend = price > SMA1 > SMA2
downtrend = price < SMA1 < SMA2
trend_score = 1 if uptrend else 0
volatility_score = max(0, 1 - (ATR_percent / 10))

Confidence = (0.6 * prob_TP) + (0.2 * trend_score) + (0.2 * volatility_score)

predicted_tp = price * (1 + abs(predicted_return))
predicted_sl = price * (1 - abs(predicted_return) / 2)
rr_ratio = (predicted_tp - price) / (price - predicted_sl)

# Entry filters
if ADX < 25 or ATR_percent > 6:
    return '🟡 Sideways / Avoid', Confidence, predicted_tp, predicted_sl

if uptrend and 45 < RSI < 65 and prob_TP > 0.65 and rr_ratio >= 1.5:
    signal = '✅ Strong Buy'
elif downtrend and 35 < RSI < 55 and prob_TP < 0.35:
    signal = '🔻 Bearish'
elif 0.45 < prob_TP < 0.65:
    signal = '🟡 Neutral / Wait'
else:
    signal = '🟡 Wait for clarity'

if RSI > 70 or price > SMA1 * 1.02:
    signal = '🕓 Wait for Dip'

return signal, Confidence, predicted_tp, predicted_sl

=============================

Multi-Timeframe Consensus

=============================

def consensus_signals(signals): bullish = sum(s.startswith('✅') for s in signals) bearish = sum(s.startswith('🔻') for s in signals) if bullish >= 2: return '✅ Buy' elif bearish >= 2: return '🔻 Sell' else: return '🟡 Wait'

=============================

Streamlit App

=============================

st.set_page_config(page_title="Enhanced Entry Analyzer", layout="wide") st.title("📈 Enhanced Entry Analyzer — Smart ML Signals")

ticker = st.text_input("Enter ticker symbol (e.g. TSLA, COIN, AAPL):", "TSLA")

if ticker: timeframes = {"4H": ("60d", "4h"), "1D": ("1y", "1d"), "1W": ("2y", "1wk")} results = {}

for label, (period, interval) in timeframes.items():
    df = load_data(ticker, period=period, interval=interval)
    df = compute_indicators(compute_pivots(df))
    reg, clf = train_models(df)
    sig, conf, tp, sl = make_recommendation(df, reg, clf)
    results[label] = (sig, conf, tp, sl)

signals = [results[t][0] for t in results]
final = consensus_signals(signals)

st.subheader(f"Final Consensus Signal: {final}")

for tf, (sig, conf, tp, sl) in results.items():
    st.markdown(f"### {tf} — {sig}")
    st.write(f"Confidence: {conf:.2f} | Target: {tp:.2f} | Stop: {sl:.2f}")

# Plot last timeframe chart
df = load_data(ticker, period="1y", interval="1d")
plt.figure(figsize=(10,4))
plt.plot(df.index, df['Close'], label='Close', color='gray')
plt.title(f'{ticker} Price Trend')
plt.legend()
st.pyplot(plt)
