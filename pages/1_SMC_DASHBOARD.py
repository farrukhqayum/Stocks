import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from datetime import datetime

st.set_page_config(layout="wide", page_title="SMC Dashboard (yfinance)")

# ============================================================
# === INDICATOR HELPERS ======================================
# ============================================================

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_rsi(series, length=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/length, adjust=False).mean()
    ma_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def compute_lb_curve(df, lb_len=10):
    lb = np.full(len(df), np.nan)
    lb[0] = df.close.iloc[0]

    for i in range(1, len(df)):
        window = df.close.iloc[max(0, i - lb_len):i]
        prev_high = window.max() if not window.empty else np.nan
        prev_low = window.min() if not window.empty else np.nan

        if df.close.iloc[i] > prev_high:
            lb[i] = (df.high.iloc[i] + df.close.iloc[i]) / 2
        elif df.close.iloc[i] < prev_low:
            lb[i] = (df.low.iloc[i] + df.close.iloc[i]) / 2
        else:
            lb[i] = lb[i - 1]

    return pd.Series(lb).ewm(span=lb_len, adjust=False).mean()

# ============================================================
# === YOUR WORKING LOADER (Option A) ==========================
# ============================================================

def load_data(ticker, start_date, interval):
    df = yf.download(
        ticker,
        start=start_date,
        end=datetime.today().strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if df is None or df.empty:
        return None

    # Flatten MultiIndex
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]

    # Ensure datetime
    if isinstance(df.index, pd.DatetimeIndex):
        df["Date"] = df.index
    elif "date" in df.columns:
        df["Date"] = pd.to_datetime(df["date"])
    elif "datetime" in df.columns:
        df["Date"] = pd.to_datetime(df["datetime"])
    else:
        raise ValueError("No valid datetime column found.")

    df.set_index("Date", inplace=True)

    # Clean numeric
    df = df.dropna(subset=["open", "high", "low", "close"]).astype(float)

    # Indicators
    df['ema20'] = ema(df.close, 20)
    df['ema50'] = ema(df.close, 50)
    df['ema200'] = ema(df.close, 200)
    df['rsi'] = compute_rsi(df.close)
    df['rsi_ema'] = ema(df['rsi'], 14)
    df['lb_crv'] = compute_lb_curve(df)

    df = df.bfill().ffill()
    return df

# ============================================================
# === STREAMLIT UI ===========================================
# ============================================================

st.title("SMC Dashboard (yfinance)")

with st.sidebar:
    ticker = st.text_input("Ticker", "AAPL").upper()
    start_date = st.date_input("Start Date", datetime(2022, 1, 1))
    timeframe = st.selectbox("Timeframe", ["1D", "1W", "1M"])
    interval_map = {"1D": "1d", "1W": "1wk", "1M": "1mo"}
    interval = interval_map[timeframe]

st.info(f"Downloading {ticker} from {start_date} ...")

df = load_data(ticker, start_date, interval)

if df is None or df.empty:
    st.error("No data returned. Try another ticker or timeframe.")
    st.stop()

# Convert to numpy arrays
df = df.copy()
df["x"] = mdates.date2num(df.index.to_pydatetime())
open_np = df.open.to_numpy()
high_np = df.high.to_numpy()
low_np = df.low.to_numpy()
close_np = df.close.to_numpy()
x_np = df.x.to_numpy()

n = len(df)

# ============================================================
# === ZONE DETECTION (FVG + OB) ==============================
# ============================================================

zones = []

for i in range(2, n):
    # FVG Bull
    if low_np[i] > high_np[i - 2]:
        zones.append({"start": i - 2, "end": i, "top": high_np[i - 2], "bot": low_np[i], "bull": True, "type": "FVG"})

    # FVG Bear
    if high_np[i] < low_np[i - 2]:
        zones.append({"start": i - 2, "end": i, "top": high_np[i], "bot": low_np[i - 2], "bull": False, "type": "FVG"})

    # OB Bull
    if close_np[i] > high_np[i - 1] and close_np[i] > open_np[i]:
        zones.append({"start": i - 1, "end": i, "top": high_np[i - 1], "bot": low_np[i - 1], "bull": True, "type": "OB"})

    # OB Bear
    if close_np[i] < low_np[i - 1] and close_np[i] < open_np[i]:
        zones.append({"start": i - 1, "end": i, "top": high_np[i - 1], "bot": low_np[i - 1], "bull": False, "type": "OB"})

# Mark active zones (not mitigated)
for z in zones:
    z["active"] = True
    for j in range(z["end"] + 1, n):
        if z["bot"] < close_np[j] < z["top"]:
            z["active"] = False
            break

active_zones = [z for z in zones if z["active"]]

# ============================================================
# === SIGNAL ENGINE (4 COLUMNS) ==============================
# ============================================================

latest = df.iloc[-1]

bull_signal = latest.close > latest.ema50 and latest.ema20 > latest.ema50
bear_signal = latest.close < latest.ema50 and latest.ema20 < latest.ema50

col1, col2, col3, col4 = st.columns(4)

col1.markdown(f"<div style='background:{'#16a34a' if bull_signal else '#94a3b8'};padding:18px;border-radius:8px;text-align:center;'><h3 style='color:white;margin:0;'>HOLD LONG</h3></div>", unsafe_allow_html=True)
col2.markdown(f"<div style='background:{'#dc2626' if bear_signal else '#94a3b8'};padding:18px;border-radius:8px;text-align:center;'><h3 style='color:white;margin:0;'>EXIT LONG</h3></div>", unsafe_allow_html=True)
col3.markdown(f"<div style='background:{'#dc2626' if bear_signal else '#94a3b8'};padding:18px;border-radius:8px;text-align:center;'><h3 style='color:white;margin:0;'>HOLD SHORT</h3></div>", unsafe_allow_html=True)
col4.markdown(f"<div style='background:{'#16a34a' if bull_signal else '#94a3b8'};padding:18px;border-radius:8px;text-align:center;'><h3 style='color:white;margin:0;'>EXIT SHORT</h3></div>", unsafe_allow_html=True)

# ============================================================
# === CHART ==================================================
# ============================================================

st.subheader(f"{ticker} Price Chart ({timeframe})")

fig, ax = plt.subplots(figsize=(14, 6))

# Candle width
if n > 1:
    candle_width = np.median(np.diff(x_np)) * 0.7
else:
    candle_width = 0.6

# Candles
for i in range(n):
    o, h, l, c = open_np[i], high_np[i], low_np[i], close_np[i]
    color = "#16a34a" if c >= o else "#dc2626"
    left = x_np[i] - candle_width / 2
    rect = Rectangle((left, min(o, c)), candle_width, abs(c - o), color=color)
    ax.add_patch(rect)
    ax.plot([x_np[i], x_np[i]], [l, h], color="black", linewidth=0.6)

# LB curve + EMAs
ax.plot(x_np, df.lb_crv, color="#1e90ff", linewidth=1.5, label="LB Curve")
ax.plot(x_np, df.ema20, color="#f59e0b", linewidth=1.0, label="EMA20")
ax.plot(x_np, df.ema50, color="#ef4444", linewidth=1.0, label="EMA50")
ax.plot(x_np, df.ema200, color="#7c3aed", linewidth=1.0, label="EMA200")

# Active zones
for z in active_zones:
    s, e = z["start"], z["end"]
    x0, x1 = x_np[s], x_np[e]
    width = x1 - x0
    height = z["top"] - z["bot"]
    color = (0, 0.6, 0, 0.18) if z["bull"] else (0.8, 0, 0, 0.18)
    rect = Rectangle((x0, z["bot"]), width, height, color=color)
    ax.add_patch(rect)

    # Marker
    mid = x0 + width / 2
    if z["type"] == "OB":
        ax.scatter(mid, z["top"] if z["bull"] else z["bot"], marker="^" if z["bull"] else "v", color="black", s=60)
    else:
        ax.scatter(mid, z["bot"] if z["bull"] else z["top"], marker="D", color="black", s=40)

# Formatting
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.grid(alpha=0.2)
ax.legend()

st.pyplot(fig)

# Tail
st.subheader("Data (tail)")
st.dataframe(df.tail(10))
