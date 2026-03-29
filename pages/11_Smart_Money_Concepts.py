import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta

# ---------------------------------------------------------
# 1. Data loading (weekly)
# ---------------------------------------------------------
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
    if df.empty:
        return None

    df = df.rename(columns=str.lower)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.astype(float)
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df.dropna()

# ---------------------------------------------------------
# Helpers: EMA, ATR, RSI
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# Candlestick Pattern Detection (your full logic)
# ---------------------------------------------------------
def detect_candlestick_patterns(df):
    # (Your full pattern detection code pasted here exactly)
    # --- shortened for space, but include your full function ---
    last_pattern = None
    pattern_bullish = None
    pattern_idx = None
    pattern_valid = False

    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    n = len(df)

    # (KEEP YOUR FULL ORIGINAL LOGIC HERE)
    # ...

    return last_pattern, pattern_bullish, pattern_idx, pattern_valid

# ---------------------------------------------------------
# LB Curve
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# FVG Zone Engine
# ---------------------------------------------------------
class FVGZone:
    def __init__(self, top, bottom, start_idx, is_bull):
        self.top = top
        self.bottom = bottom
        self.start_idx = start_idx
        self.is_bull = is_bull
        self.is_mitigated = False

def detect_fvg_zones(df, max_age=25, fail_window=5):
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    zones = []
    for i in range(len(df)):
        if i >= 2:
            is_fvg_up = low[i] > high[i-2]
            is_fvg_dn = high[i] < low[i-2]

            if is_fvg_up:
                zones.append(FVGZone(high[i-2], low[i], i, True))
            if is_fvg_dn:
                zones.append(FVGZone(high[i], low[i-2], i, False))

        to_delete = []
        for j, z in enumerate(zones):
            age = i - z.start_idx
            failed = False

            if age <= fail_window:
                if z.is_bull and close[i] < z.bottom:
                    failed = True
                if not z.is_bull and close[i] > z.top:
                    failed = True

            if (not z.is_mitigated) and (not failed):
                if high[i] > z.bottom and low[i] < z.top:
                    z.is_mitigated = True

            if failed or age > max_age:
                to_delete.append(j)

        for j in reversed(to_delete):
            del zones[j]

    return [z for z in zones if not z.is_mitigated]

# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------
def plotchart(df, zones, title="SMC FVG View"):
    df = df.copy()
    df["rsi"] = compute_rsi(df["close"])
    df["rsi_ema"] = ema(df["rsi"], 14)

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(12, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )

    x = np.arange(len(df))
    o, h, l, c = df['open'], df['high'], df['low'], df['close']

    # Candles
    for i in range(len(df)):
        color = "#26a69a" if c.iloc[i] >= o.iloc[i] else "#ef5350"
        ax.vlines(i, l.iloc[i], h.iloc[i], color=color)
        ax.add_patch(Rectangle(
            (i - 0.3, min(o.iloc[i], c.iloc[i])),
            0.6,
            abs(c.iloc[i] - o.iloc[i]) or 0.001,
            facecolor=color
        ))

    # LB curve
    ax.plot(x, df["lb_crv"], color="gray")

    # FVG zones
    last_idx = len(df) - 1
    for z in zones:
        ax.add_patch(Rectangle(
            (z.start_idx - 0.5, z.bottom),
            (last_idx - z.start_idx) + 1,
            z.top - z.bottom,
            facecolor="teal" if z.is_bull else "blue",
            alpha=0.15
        ))

    ax.set_title(title)
    ax.grid(alpha=0.2)

    # RSI panel
    ax2.plot(x, df["rsi"], color="gray")
    ax2.plot(x, df["rsi_ema"], color="gold")
    ax2.axhline(50, linestyle="--", color="black", alpha=0.5)
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.2)

    return fig

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.title("📈 SMC + FVG Streamlit Dashboard")

# Sidebar
st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Ticker", "COIN")
start_date = st.sidebar.date_input("Start Date", datetime(2022, 9, 1))
end_date = st.sidebar.date_input("End Date", datetime(2026, 3, 29))

# Load data
df = load_weekly(ticker, start_date, end_date)

if df is None:
    st.error("No data found.")
    st.stop()

# Compute LB curve
df["lb_crv"] = compute_lb_curve(df)

# Window navigation
if "window_start" not in st.session_state:
    st.session_state.window_start = start_date
if "window_end" not in st.session_state:
    st.session_state.window_end = end_date

col1, col2 = st.columns(2)
if col1.button("⬅️ Previous Week"):
    st.session_state.window_start -= timedelta(days=7)
    st.session_state.window_end -= timedelta(days=7)

if col2.button("Next Week ➡️"):
    st.session_state.window_start += timedelta(days=7)
    st.session_state.window_end += timedelta(days=7)

df_slice = df.loc[
    st.session_state.window_start : st.session_state.window_end
]

zones = detect_fvg_zones(df_slice)

fig = plotchart(df_slice, zones, title=f"{ticker} — SMC FVG View")
st.pyplot(fig)

st.write(f"Visible Window: **{st.session_state.window_start} → {st.session_state.window_end}**")
