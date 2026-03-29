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

    # If empty → return None
    if df is None or df.empty:
        return None

    # Flatten multi-index columns
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]

    # Ensure required columns exist
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        return None

    # Clean data
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.astype(float)

    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    return df


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

def draw_smc_box(ax, df, zones):
    c = df['close'].values
    h = df['high'].values
    l = df['low'].values
    o = df['open'].values

    # --- Candlestick Pattern ---
    last_pattern, pattern_bullish, pattern_idx, pattern_valid = detect_candlestick_patterns(df)
    plot_pattern_label(ax, df, pattern_idx, last_pattern, pattern_bullish)

    if last_pattern is None:
        pattern_text = "None"
        pattern_color = "gray"
    else:
        age = len(df) - pattern_idx
        validity = "Active" if pattern_valid else "Rejected"
        pattern_text = f"{last_pattern} ({age} bars ago, {validity})"
        pattern_color = "green" if pattern_bullish else "red"

    # --- Sweep Detection ---
    bull_sweep = (l[-1] < l[-2]) and (c[-1] > l[-2])
    bear_sweep = (h[-1] > h[-2]) and (c[-1] < h[-2])

    if bull_sweep:
        sweep_text = "Sell-side Sweep (Bullish) ↑"
        sweep_color = "green"
    elif bear_sweep:
        sweep_text = "Buy-side Sweep (Bearish) ↓"
        sweep_color = "red"
    else:
        sweep_text = "None"
        sweep_color = "gray"

    # --- Trend (EMA 20/50/200) ---
    ema20 = ema(df['close'], 20)
    ema50 = ema(df['close'], 50)
    ema200 = ema(df['close'], 200)

    ema_bullish = (df['close'].iloc[-1] > ema20.iloc[-1]) and (ema20.iloc[-1] > ema50.iloc[-1])
    ema_bearish = (df['close'].iloc[-1] < ema20.iloc[-1]) and (ema20.iloc[-1] < ema50.iloc[-1])

    trend_text = "UP" if ema_bullish else "DOWN" if ema_bearish else "SIDEWAYS"
    trend_color = "green" if ema_bullish else "red" if ema_bearish else "gray"

    # --- Momentum (RSI + LB curve) ---
    rsi = df["rsi"].iloc[-1]
    rsi_ema = df["rsi_ema"].iloc[-1]
    close = df["close"].iloc[-1]
    lb = df["lb_crv"].iloc[-1]

    mom_bullish = (rsi > 52 and rsi > rsi_ema) or (close > lb)
    mom_bearish = (rsi < 52 and rsi < rsi_ema) or (close < lb)

    mom_text = "BULLISH" if mom_bullish else "BEARISH" if mom_bearish else "NEUTRAL"
    mom_color = "green" if mom_bullish else "red" if mom_bearish else "gray"

    # --- FVG Status ---
    last_close = c[-1]
    has_bull_fvg = any(z.is_bull for z in zones)
    has_bear_fvg = any(not z.is_bull for z in zones)
    inside_zone = any(min(z.bottom, z.top) < last_close < max(z.bottom, z.top) for z in zones)

    zone_text = f"FVG {'✓' if has_bull_fvg or has_bear_fvg else '✗'} | Inside {'✓' if inside_zone else '✗'}"
    zone_color = "green" if (has_bull_fvg or has_bear_fvg) else "gray"

    # --- Structure ---
    strong_bullish = ema_bullish and has_bull_fvg and close > lb
    strong_bearish = ema_bearish and has_bear_fvg and close < lb

    struct_text = "STRONG BULLISH" if strong_bullish else "STRONG BEARISH" if strong_bearish else "NEUTRAL"
    struct_color = "green" if strong_bullish else "red" if strong_bearish else "gray"

    # --- Entry Ready ---
    entry_ready = has_bull_fvg and mom_bullish and inside_zone
    entry_text = "🟢 ENTRY READY" if entry_ready else "—"
    entry_color = "green" if entry_ready else "gray"

    # --- Build Lines ---
    lines = [
        ("SMC & SIGNALS", "black"),
        (f"SWEEP: {sweep_text}", sweep_color),
        (f"PATTERN: {pattern_text}", pattern_color),
        (f"STRUCTURE: {struct_text}", struct_color),
        (f"TREND: {trend_text}", trend_color),
        (f"MOMENTUM: {mom_text}", mom_color),
        (f"ZONE: {zone_text}", zone_color),
        (f"{entry_text}", entry_color),
        (f"{------}")
    ]

    # --- Draw Background Box ---
    ax.add_patch(Rectangle(
        (0.01, 0.99 - 0.24),
        0.33,
        0.24,
        transform=ax.transAxes,
        facecolor=(0.95, 0.95, 0.95, 0.85),
        edgecolor="black",
        linewidth=0.5
    ))

    # --- Draw Text Lines ---
    y = 0.99 - 0.03
    for text, color in lines:
        ax.text(
            0.02, y, text,
            transform=ax.transAxes,
            fontsize=8,
            color=color,
            ha="left", va="top"
        )
        y -= 0.027

# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------
def plot_pattern_label(ax, df, pattern_idx, pattern_name, pattern_bullish):
    if pattern_idx is None or pattern_name is None:
        return

    high = df['high'].iloc[pattern_idx]
    low  = df['low'].iloc[pattern_idx]
    x = pattern_idx

    # Vertical offset
    offset = (high - low) * 0.15

    if pattern_bullish:
        y = low - offset
        va = "top"
        color = "green"
    else:
        y = high + offset
        va = "bottom"
        color = "red"

    ax.text(
        x, y,
        pattern_name,
        color=color,
        fontsize=6,
        ha="center",
        va=va,
        fontweight="bold",
        zorder=20
    )

def plotchart(df, zones, title="SMC FVG View"):
    df = df.copy()

    # --- Compute RSI + RSI EMA + LB curve already exists in df ---
    if "rsi" not in df.columns:
        df["rsi"] = compute_rsi(df["close"], 14)
    if "rsi_ema" not in df.columns:
        df["rsi_ema"] = ema(df["rsi"], 14)

    fig, (ax, ax2) = plt.subplots(
        2, 1,
        figsize=(12, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )

    x = np.arange(len(df))
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    width = 0.6
    up_color = "#26a69a"
    down_color = "#ef5350"

    # -----------------------------
    # PANEL 1 — PRICE + FVG + SMC
    # -----------------------------
    for i in range(len(df)):
        color = up_color if c.iloc[i] >= o.iloc[i] else down_color
        ax.vlines(i, l.iloc[i], h.iloc[i], color=color, linewidth=1)
        ax.add_patch(Rectangle(
            (i - width/2, min(o.iloc[i], c.iloc[i])),
            width,
            abs(c.iloc[i] - o.iloc[i]) or 0.001,
            facecolor=color,
            edgecolor=color
        ))

    # LB curve (already computed outside)
    ax.plot(x, df["lb_crv"], color="gray", linewidth=1.2)

    # FVG zones
    last_idx = len(df) - 1
    for z in zones:
        rect_x = z.start_idx - 0.5
        rect_width = (last_idx - z.start_idx) + 1
        color = "teal" if z.is_bull else "blue"
        ax.add_patch(Rectangle(
            (rect_x, z.bottom),
            rect_width,
            z.top - z.bottom,
            facecolor=color,
            alpha=0.15,
            edgecolor=color,
            linestyle="--"
        ))

    # 🔹 SMC dashboard box (your original logic)
    draw_smc_box(ax, df, zones)

    ax.set_title(title)
    ax.grid(alpha=0.2)

    # -----------------------------
    # PANEL 2 — RSI PANEL (with fills)
    # -----------------------------
    rsi = df["rsi"]
    rsi_ema = df["rsi_ema"]

    # Green fill when RSI > RSI EMA
    ax2.fill_between(
        x,
        rsi,
        rsi_ema,
        where=(rsi > rsi_ema),
        color="green",
        alpha=0.15
    )

    # Red fill when RSI < RSI EMA
    ax2.fill_between(
        x,
        rsi,
        rsi_ema,
        where=(rsi < rsi_ema),
        color="red",
        alpha=0.15
    )

    ax2.plot(x, rsi, color="gray", linewidth=1.2, label="RSI")
    ax2.plot(x, rsi_ema, color="gold", linewidth=1.2, label="RSI EMA")

    # Same reference lines as notebook
    for level in [25, 50, 78]:
        ax2.axhline(level, color="black", linestyle="--", linewidth=0.7, alpha=0.6)

    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI")
    ax2.grid(alpha=0.2)
    ax2.legend(loc="upper left")

    # X‑axis labels if index is datetime
    if isinstance(df.index, pd.DatetimeIndex):
        ax2.set_xticks(x[::max(1, len(x)//10)])
        ax2.set_xticklabels(
            df.index.strftime("%Y-%m-%d")[::max(1, len(x)//10)],
            rotation=45,
            fontsize=8
        )

    plt.tight_layout()
    return fig


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.title("📈 SMART MONEY CONCEPTS (SMC) Dashboard")

# Sidebar
st.sidebar.header("Settings")

with st.expander("ℹ️ Explanation of How This Dashboard Works"):
    st.markdown("""
### **1. Data Loading & Pre‑Processing**
- The app begins by downloading **weekly OHLC price data** using `yfinance`.  
- It ensures the dataset contains the required columns (`open`, `high`, `low`, `close`) and cleans missing values.  
- The index is converted to a proper datetime format so the chart can align candles correctly.  
- This weekly data becomes the foundation for all SMC, FVG, and momentum calculations.

### **2. Technical Indicators (EMA, RSI, LB Curve)**
- The script computes **Exponential Moving Averages (20/50/200)** to determine trend direction.  
- RSI and its EMA are calculated to measure **momentum shifts**.  
- A custom **Liquidity Breaker (LB) curve** is generated by tracking structural highs/lows and smoothing them with an EMA.  
- These indicators feed directly into the SMC dashboard logic.

### **3. Smart Money Concepts (SMC) Logic**
- The system detects **candlestick patterns**, **sweeps**, and **market structure bias**.  
- It evaluates trend, momentum, and pattern validity to classify the market as *bullish, bearish, or neutral*.  
- A compact SMC dashboard box is drawn on the chart summarizing:  
  - Sweep detection  
  - Candlestick pattern  
  - Trend bias  
  - Momentum bias  
  - FVG presence  
  - Structure strength  
  - Entry readiness  

### **4. Fair Value Gap (FVG) Engine**
- The script scans for **bullish and bearish FVGs** using a 3‑candle logic.  
- Each FVG is tracked for:  
  - Age  
  - Mitigation  
  - Failure conditions  
- Only **active, unmitigated zones** are displayed on the chart as shaded rectangles.  
- This allows traders to visually identify premium/discount zones.

### **5. Chart Rendering & Navigation**
- The chart is built manually using **matplotlib candlesticks**, giving full control over styling.  
- FVG zones, LB curve, and SMC dashboard are layered on top of the candles.  
- A second panel shows RSI with green/red fills to highlight momentum shifts.  
- The user can scroll through time using **Previous Week / Next Week** buttons, updating the visible window dynamically.
""")


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
df["rsi"] = compute_rsi(df["close"], 14)
df["rsi_ema"] = ema(df["rsi"], 14)

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
