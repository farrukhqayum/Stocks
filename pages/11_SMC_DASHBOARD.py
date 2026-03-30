import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta

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

    # Add basic indicators
    df['ema20']  = ema(df.close, 20)
    df['ema50']  = ema(df.close, 50)
    df['ema200'] = ema(df.close, 200)
    df['rsi']    = compute_rsi(df.close)
    df["rsi_ema"] = ema(df["rsi"], 14)
    df["lb_crv"] = compute_lb_curve(df)
    
    return df

# ---------------------------------------------------------
# Candlestick Pattern Detection
# ---------------------------------------------------------
def detect_candlestick_patterns(df):
    last_pattern = None
    pattern_bullish = None
    pattern_idx = None
    pattern_valid = False

    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    n = len(df)

    ema20 = df['ema20'].values
    ema50 = df['ema50'].values
    ema200 = df['ema200'].values
    
    trend_up   = df['close'].values > ema20
    trend_down = df['close'].values < ema20
    
    for i in range(1, n):
        o0, c0, h0, l0 = o[i], c[i], h[i], l[i]
        o1, c1, h1, l1 = o[i-1], c[i-1], h[i-1], l[i-1]
    
        body0 = abs(c0 - o0)
        body1 = abs(c1 - o1)
        range0 = h0 - l0

        is_uptrend   = trend_up[i]
        is_downtrend = trend_down[i]

        # -------------------------
        # 1. ENGULFING (correct logic)
        # -------------------------
 
        if (
            trend_down[i] and
            c1 < o1 and
            c0 > o0 and
            o0 <= c1 and
            c0 >= o1
        ):
            last_pattern = "Bull Engulf"
            pattern_bullish = True
            pattern_idx = i
            continue
        
        # Bearish Engulfing (strict + trend filtered)
        if (
            trend_up[i] and
            c1 > o1 and
            c0 < o0 and
            o0 >= c1 and
            c0 <= o1
        ):
            last_pattern = "Bear Engulf"
            pattern_bullish = False
            pattern_idx = i
            continue

        # -------------------------
        # 2. PIERCING / DARK CLOUD (correct logic)
        # -------------------------

        # Piercing Pattern (bullish)
        if (
            is_downtrend and
            c1 < o1 and              # previous bearish
            o0 < l1 and              # gap down
            c0 > (o1 + c1) / 2 and   # closes above midpoint of prev body
            c0 < o1                  # but not fully engulfing
        ):
            last_pattern = "Piercing"
            pattern_bullish = True
            pattern_idx = i
            continue

        # Dark Cloud Cover (bearish)
        if (
            is_uptrend and
            c1 > o1 and              # previous bullish
            o0 > h1 and              # gap up
            c0 < (o1 + c1) / 2 and   # closes below midpoint
            c0 > o1                  # but not fully engulfing
        ):
            last_pattern = "Dark Cloud"
            pattern_bullish = False
            pattern_idx = i
            continue

        # -------------------------
        # 3. HAMMER / SHOOTING STAR (correct logic)
        # -------------------------

        upper_wick = h0 - max(o0, c0)
        lower_wick = min(o0, c0) - l0

        # Hammer (bullish)
        if (
            is_downtrend and
            lower_wick >= body0 * 2 and
            upper_wick <= body0 * 0.3
        ):
            last_pattern = "Hammer"
            pattern_bullish = True
            pattern_idx = i
            continue

        # Shooting Star (bearish)
        if (
            is_uptrend and
            upper_wick >= body0 * 2 and
            lower_wick <= body0 * 0.3
        ):
            last_pattern = "Shooting Star"
            pattern_bullish = False
            pattern_idx = i
            continue

        # -------------------------
        # 4. DOJI (with trend validation)
        # -------------------------
        
        if body0 <= range0 * 0.1:
        
            # Gravestone Doji (bearish reversal) → requires uptrend
            if (
                upper_wick >= range0 * 0.6 and 
                lower_wick <= range0 * 0.1 and
                is_uptrend
            ):
                last_pattern = "Gravestone"
                pattern_bullish = False
                pattern_idx = i
                continue
        
            # Dragonfly Doji (bullish reversal) → requires downtrend
            if (
                lower_wick >= range0 * 0.6 and 
                upper_wick <= range0 * 0.1 and
                is_downtrend
            ):
                last_pattern = "Dragonfly"
                pattern_bullish = True
                pattern_idx = i
                continue
        
            # Neutral Doji → NO trend validation
            last_pattern = "Doji"
            pattern_bullish = None
            pattern_idx = i
            continue


        # -------------------------
        # 5. MORNING / EVENING STAR (correct logic)
        # -------------------------

        if i >= 2:
            o2, c2 = o[i-2], c[i-2]

            # Morning Star (bullish)
            if (
                is_downtrend and
                c2 < o2 and                # strong bearish candle
                body1 <= body0 * 0.5 and   # small middle candle
                c0 > (o2 + c2) / 2         # strong bullish close
            ):
                last_pattern = "Morning Star"
                pattern_bullish = True
                pattern_idx = i
                continue

            # Evening Star (bearish)
            if (
                is_uptrend and
                c2 > o2 and                # strong bullish candle
                body1 <= body0 * 0.5 and   # small middle candle
                c0 < (o2 + c2) / 2         # strong bearish close
            ):
                last_pattern = "Evening Star"
                pattern_bullish = False
                pattern_idx = i
                continue

        # -------------------------
        # 6. TWEEZER TOP / BOTTOM (correct logic)
        # -------------------------

        if is_downtrend and abs(l0 - l1) <= (range0 * 0.1):
            last_pattern = "Tweezer Bot"
            pattern_bullish = True
            pattern_idx = i
            continue

        if is_uptrend and abs(h0 - h1) <= (range0 * 0.1):
            last_pattern = "Tweezer Top"
            pattern_bullish = False
            pattern_idx = i
            continue

    # -------------------------
    # PATTERN VALIDATION
    # -------------------------
    if last_pattern is not None:
        pattern_low  = df['low'].iloc[pattern_idx]
        pattern_high = df['high'].iloc[pattern_idx]
    
        curr_low  = df['low'].iloc[-1]
        curr_high = df['high'].iloc[-1]
    
        if pattern_bullish:
            pattern_valid = curr_low >= pattern_low
        elif pattern_bullish is False:
            pattern_valid = curr_high <= pattern_high
        else:
            pattern_valid = False


    return last_pattern, pattern_bullish, pattern_idx, pattern_valid

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
        self.touched = False

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
                top = max(high[i-2], low[i])
                bottom = min(high[i-2], low[i])
                zones.append(FVGZone(top, bottom, i, True))
            
            if is_fvg_dn:
                top = max(high[i], low[i-2])
                bottom = min(high[i], low[i-2])
                zones.append(FVGZone(top, bottom, i, False))

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
        
            if (not z.touched) and (z.bottom < close[i] < z.top):
                z.touched = True

        for j in reversed(to_delete):
            del zones[j]

    return [z for z in zones if not z.is_mitigated]

# ---------------------------------------------------------
# ENTRY ZONES
# ---------------------------------------------------------
def bullish_entry(df, zones):
    last = df.iloc[-1]

    ema_bull = (last.close > last.ema20) and (last.ema20 > last.ema50)
    mom_bull = (last.rsi > last.rsi_ema)

    bull_zones = [z for z in zones if z.is_bull]

    inside_bull = any(z.bottom <= last.close <= z.top for z in bull_zones)
    first_touch = any(z.touched for z in bull_zones)

    # NEW: allow entry if price is ABOVE the FVG after mitigation
    near_zone = any(abs(last.close - z.bottom) < (last.close * 0.01) for z in bull_zones)

    if ema_bull and mom_bull and (inside_bull or first_touch or near_zone):
        return True
    return False

def bearish_entry(df, zones):
    last = df.iloc[-1]

    ema_bear = (last.close < last.ema20) and (last.ema20 < last.ema50)
    mom_bear = (last.rsi < last.rsi_ema)
    
    bear_zones = [z for z in zones if not z.is_bull]
    inside_bear = any(z.bottom < last.close < z.top for z in bear_zones)
    first_touch = any(z.touched and (len(df)-1 - z.start_idx) <= 2 for z in bear_zones)

    if ema_bear and mom_bear and (inside_bear or first_touch):
        return True
    return False

def bullish_exit(df, zones):
    last = df.iloc[-1]

    # Momentum flip
    if last.rsi < last.rsi_ema:
        return True

    # Trend flip
    if last.close < last.ema20:
        return True

    # FVG invalidation
    bull_zones = [z for z in zones if z.is_bull]
    for z in bull_zones:
        if last.close < z.bottom:
            return True

    return False

def bearish_exit(df, zones):
    last = df.iloc[-1]

    if last.rsi > last.rsi_ema:
        return True

    if last.close > last.ema20:
        return True

    bear_zones = [z for z in zones if not z.is_bull]
    for z in bear_zones:
        if last.close > z.top:
            return True

    return False


# ---------------------------------------------------------
# DRAWING
# ---------------------------------------------------------
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

    # --- FVG Status (Directional) ---
    last_close = c[-1]
    
    bull_zones = [z for z in zones if z.is_bull]
    bear_zones = [z for z in zones if not z.is_bull]
    
    has_bull_fvg = len(bull_zones) > 0
    has_bear_fvg = len(bear_zones) > 0
    
    inside_bull = any((z.bottom < last_close < z.top) for z in bull_zones)
    inside_bear = any((z.bottom < last_close < z.top) for z in bear_zones)
    
    first_touch_bull = any((z.touched and (len(df)-1 - z.start_idx) <= 2) for z in bull_zones)
    first_touch_bear = any((z.touched and (len(df)-1 - z.start_idx) <= 2) for z in bear_zones)

    zone_lines = []
    def yesno_color(flag):
        return "green" if flag else "red"

    zone_lines.append(("ZONE:", "gray"))
    zone_lines.append((f"  BULL FVG: {'✓' if has_bull_fvg else '✗'}",
                       yesno_color(has_bull_fvg)))
    zone_lines.append((f"  BEAR FVG: {'✓' if has_bear_fvg else '✗'}",
                       yesno_color(has_bear_fvg)))
    zone_lines.append((f"  Inside Bull: {'✓' if inside_bull else '✗'}",
                       yesno_color(inside_bull)))
    zone_lines.append((f"  Inside Bear: {'✓' if inside_bear else '✗'}",
                       yesno_color(inside_bear)))
    zone_lines.append((f"  1stTouch Bull: {'✓' if first_touch_bull else '✗'}",
                       yesno_color(first_touch_bull)))
    zone_lines.append((f"  1stTouch Bear: {'✓' if first_touch_bear else '✗'}",
                       yesno_color(first_touch_bear)))

    # --- Structure ---
    strong_bullish = ema_bullish and has_bull_fvg and close > lb
    strong_bearish = ema_bearish and has_bear_fvg and close < lb

    struct_text = "STRONG BULLISH" if strong_bullish else "STRONG BEARISH" if strong_bearish else "NEUTRAL"
    struct_color = "green" if strong_bullish else "red" if strong_bearish else "gray"

    # --- Build Lines ---
    lines = [
        ("SMC & SIGNALS", "black"),
        (f"SWEEP: {sweep_text}", sweep_color),
        (f"PATTERN: {pattern_text}", pattern_color),
        (f"STRUCTURE: {struct_text}", struct_color),
        (f"TREND: {trend_text}", trend_color),
        (f"MOMENTUM: {mom_text}", mom_color)
    ]
    lines.extend(zone_lines)
    
    # --- Dynamic Box Height ---
    line_height = 0.027
    padding = 0.02
    box_height = len(lines) * line_height + padding
        
    # --- Draw Text Lines ---
    y = 0.99 - 0.03
    for text, color in lines:
        ax.text(
            0.02, y, text,
            transform=ax.transAxes,
            fontsize=8,
            color=color,
            ha="left", va="top",
            wrap=False
        )
        y -= line_height


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
    ax.plot(x, df["lb_crv"], color="gray", alpha=0.75,  linewidth=1.2)
    ax.plot(x, df["ema20"], color="yellow", alpha=0.75, linewidth=1)
    ax.plot(x, df["ema50"], color="red", alpha=0.75, linewidth=1)

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
    debug = draw_smc_box(ax, df, zones)

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

# Data boundaries
data_start = df.index.min()
data_end   = df.index.max()

# Initialize session state AFTER loading data
if "window_start" not in st.session_state:
    st.session_state.window_start = data_start

if "window_end" not in st.session_state:
    st.session_state.window_end = data_end

# Navigation buttons
col1, col2 = st.columns(2)

if col1.button("⬅️ Previous Week"):
    st.session_state.window_start -= timedelta(days=7)
    st.session_state.window_end   -= timedelta(days=7)

if col2.button("Next Week ➡️"):
    st.session_state.window_start += timedelta(days=7)
    st.session_state.window_end   += timedelta(days=7)

# Convert to pandas timestamps for safe comparison
ws = pd.to_datetime(st.session_state.window_start)
we = pd.to_datetime(st.session_state.window_end)

# Clamp to data boundaries
if ws < data_start:
    ws = data_start
if we > data_end:
    we = data_end

# Save clamped values back
st.session_state.window_start = ws
st.session_state.window_end   = we

# Slice safely
df_slice = df.loc[ws:we]
zones = detect_fvg_zones(df_slice)
# ---- ENTRY / EXIT SIGNALS ----
long_signal  = bullish_entry(df_slice, zones)
short_signal = bearish_entry(df_slice, zones)

exit_long  = bullish_exit(df_slice, zones)
exit_short = bearish_exit(df_slice, zones)

# ---- DISPLAY SIGNALS IN 4 COLUMNS ----
c1, c2, c3, c4 = st.columns(4)

with c1:
    if long_signal:
        st.success("📈 LONG")
    else:
        st.info("—")

with c2:
    if short_signal:
        st.error("📉 SHORT")
    else:
        st.info("—")

with c3:
    if exit_long:
        st.warning("🔔 EXIT LONG")
    else:
        st.info("—")

with c4:
    if exit_short:
        st.warning("🔔 EXIT SHORT")
    else:
        st.info("—")

fig = plotchart(df_slice, zones, title=f"{ticker} — SMC FVG View")
st.pyplot(fig)

st.write(f"Visible Window: **{ws} → {we}**")

