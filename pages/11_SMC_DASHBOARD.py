import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta

st.set_page_config(page_title="SMC Dashboard", layout="wide")
st.title("📈 SMART MONEY CONCEPTS (SMC) Dashboard")

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def compute_rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_lb_curve(df, lblen=10):
    close = df['close']
    high = df['high']
    low = df['low']
    lb = close.copy()
    for i in range(len(df)):
        if i == 0:
            lb.iloc[i] = close.iloc[i]
        else:
            start = max(0, i - lblen + 1)
            highest_lb_prev = lb.iloc[start:i].max()
            lowest_lb_prev = lb.iloc[start:i].min()
            if close.iloc[i] > highest_lb_prev:
                lb.iloc[i] = (high.iloc[i] + close.iloc[i]) / 2
            elif close.iloc[i] < lowest_lb_prev:
                lb.iloc[i] = (low.iloc[i] + close.iloc[i]) / 2
            else:
                lb.iloc[i] = lb.iloc[i-1]
    return ema(lb, lblen)

@st.cache_data
def load_weekly(ticker, start_date, end_date):
    df = yf.download(
        ticker,
        start=start_date,
        end=(pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1wk",
        auto_adjust=False,
        progress=False
    )
    if df is None or df.empty:
        return None
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df = df.dropna(subset=["open", "high", "low", "close"]).astype(float)
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df['ema20'] = ema(df.close, 20)
    df['ema50'] = ema(df.close, 50)
    df['ema200'] = ema(df.close, 200)
    df['rsi'] = compute_rsi(df.close)
    df['rsi_ema'] = ema(df['rsi'], 14)
    df['lb_crv'] = compute_lb_curve(df)
    return df

# ---------------------------
# PINE-SCRIPT CANDLE ENGINE
# ---------------------------

def detect_pattern(df):
    o = df.open.values
    h = df.high.values
    l = df.low.values
    c = df.close.values
    rsi = df.rsi.values
    rsi_ema = df.rsi_ema.values
    lb = df.lb_crv.values

    ema20 = df.ema20.values
    ema50 = df.ema50.values
    ema200 = df.ema200.values

    trend_up = (ema20 > ema50) | (c > lb)
    trend_down = (ema20 < ema50) | (c < lb)

    last_pattern = None
    pattern_bull = None
    pattern_idx = None

    for i in range(2, len(df)):
        body0 = abs(c[i] - o[i])
        body1 = abs(c[i-1] - o[i-1])
        body2 = abs(c[i-2] - o[i-2])
        rng0 = h[i] - l[i]
        wick_up = h[i] - max(o[i], c[i])
        wick_dn = min(o[i], c[i]) - l[i]

        bull_engulf = trend_down[i] and c[i-1] < o[i-1] and c[i] > o[i] and o[i] <= c[i-1] and c[i] >= o[i-1]
        bear_engulf = trend_up[i] and c[i-1] > o[i-1] and c[i] < o[i] and o[i] >= c[i-1] and c[i] <= o[i-1]

        bull_pierce = trend_down[i] and c[i] > (o[i-1] + c[i-1]) / 2
        bear_dark = trend_up[i] and c[i] < (o[i-1] + c[i-1]) / 2

        hammer = trend_down[i] and wick_dn > body0 * 2 and wick_up < body0 * 0.5
        star = trend_up[i] and wick_up > body0 * 2 and wick_dn < body0 * 0.5

        morning = trend_down[i] and c[i-2] < o[i-2] and body1 < body2 * 0.4 and c[i] > (o[i-2] + c[i-2]) / 2
        evening = trend_up[i] and c[i-2] > o[i-2] and body1 < body2 * 0.4 and c[i] < (o[i-2] + c[i-2]) / 2

        tweezer_bot = trend_down[i] and abs(l[i] - l[i-1]) < (rng0 * 0.1)
        tweezer_top = trend_up[i] and abs(h[i] - h[i-1]) < (rng0 * 0.1)

        if bull_engulf or bull_pierce or hammer or morning or tweezer_bot:
            last_pattern = "BULL"
            pattern_bull = True
            pattern_idx = i
        if bear_engulf or bear_dark or star or evening or tweezer_top:
            last_pattern = "BEAR"
            pattern_bull = False
            pattern_idx = i

    return last_pattern, pattern_bull, pattern_idx

def pine_signals(df):
    last_pattern, pattern_bull, idx = detect_pattern(df)
    if last_pattern is None:
        return None, None, None

    bars_ago = len(df) - 1 - idx
    if bars_ago > 20:
        return None, None, None

    pat_low = df.low.iloc[idx]
    pat_high = df.high.iloc[idx]
    close = df.close.iloc[-1]

    rejected = (close < pat_low) if pattern_bull else (close > pat_high)
    if rejected:
        return None, None, None

    rsi = df.rsi.iloc[-1]
    rsi_ema = df.rsi_ema.iloc[-1]
    lb = df.lb_crv.iloc[-1]

    bull_signal = pattern_bull and close > lb * 0.98 and rsi >= rsi_ema
    bear_signal = (not pattern_bull) and close <= lb and rsi <= rsi_ema

    if bull_signal:
        return "BUY", idx, "🟢 BUY THE DIP"
    if bear_signal:
        return "SELL", idx, "🔴 SELL THE RISE"

    return None, idx, None

# ---------------------------
# STREAMLIT UI
# ---------------------------

st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker", "ASML")
start_date = st.sidebar.date_input("Start Date", datetime(2022, 9, 1))
end_date = st.sidebar.date_input("End Date", datetime(2026, 3, 29))

df = load_weekly(ticker, start_date, end_date)
if df is None:
    st.error("No data found.")
    st.stop()

signal, idx, msg = pine_signals(df)

c1, c2 = st.columns(2)
with c1:
    if signal == "BUY":
        st.success(msg)
    else:
        st.info("—")

with c2:
    if signal == "SELL":
        st.error(msg)
    else:
        st.info("—")

st.write("Signal Pattern Index:", idx)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.close.values, label="Close")
ax.plot(df.lb_crv.values, label="LB", alpha=0.6)
ax.legend()
st.pyplot(fig)
