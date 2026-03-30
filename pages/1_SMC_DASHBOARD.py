import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta

st.set_page_config(page_title="SMC Dashboard", layout="wide")
st.title("📈 SMART MONEY CONCEPTS (SMC)")

# ---------------------------------------------------------
# INDICATORS
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

@st.cache_data
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

    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]

    if isinstance(df.index, pd.DatetimeIndex):
        df["Date"] = df.index
    elif "date" in df.columns:
        df["Date"] = pd.to_datetime(df["date"])
    elif "datetime" in df.columns:
        df["Date"] = pd.to_datetime(df["datetime"])
    else:
        raise ValueError("No valid datetime column found in downloaded data.")

    df.set_index("Date", inplace=True)

    df = df.dropna(subset=["open", "high", "low", "close"]).astype(float)

    df['ema20'] = ema(df.close, 20)
    df['ema50'] = ema(df.close, 50)
    df['ema200'] = ema(df.close, 200)
    df['rsi'] = compute_rsi(df.close)
    df['rsi_ema'] = ema(df['rsi'], 14)
    df['lb_crv'] = compute_lb_curve(df)
    df = df.bfill()
    df = df.ffill()

    return df

# ---------------------------------------------------------
# FVG DETECTION
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
    open_ = df['open'].values
    close = df['close'].values

    zones = []

    for i in range(len(df)):
        if i >= 2:
            is_fvg_up = low[i] > high[i-2]   # bullish FVG
            is_fvg_dn = high[i] < low[i-2]   # bearish FVG

            if is_fvg_up:
                top = low[i]
                bottom = high[i-2]
                zones.append(FVGZone(top, bottom, i, True))

            if is_fvg_dn:
                top = high[i]
                bottom = low[i-2]
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

            body_high = max(open_[i], close[i])
            body_low = min(open_[i], close[i])

            if not z.is_mitigated and not failed:
                if body_high > z.bottom and body_low < z.top:
                    z.is_mitigated = True

            if (not z.touched) and (z.bottom < close[i] < z.top):
                z.touched = True

            if failed or age > max_age:
                to_delete.append(j)

        for j in reversed(to_delete):
            del zones[j]

    return [z for z in zones if not z.is_mitigated]

# ---------------------------------------------------------
# LAST BROKEN FVG (STRUCTURAL REFERENCES)
# ---------------------------------------------------------

def get_last_broken_fvg(df):
    """
    Returns:
      last_broken_bear: last bearish FVG that price broke ABOVE (bull reference)
      last_broken_bull: last bullish FVG that price broke BELOW (bear reference)
    """
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(df)

    last_broken_bear = None
    last_broken_bull = None

    for i in range(2, n):
        # Bearish FVG (for bulls)
        if h[i] < l[i-2]:
            top = h[i]
            bottom = l[i-2]
            for j in range(i, n):
                if c[j] > top:
                    last_broken_bear = {"low": bottom, "high": top, "break_idx": j}

        # Bullish FVG (for bears)
        if l[i] > h[i-2]:
            top = l[i]
            bottom = h[i-2]
            for j in range(i, n):
                if c[j] < bottom:
                    last_broken_bull = {"low": bottom, "high": top, "break_idx": j}

    return last_broken_bear, last_broken_bull

# ---------------------------------------------------------
# SIMPLE PATTERN BOX (OPTIONAL INFO)
# ---------------------------------------------------------

def draw_smc_box(ax, df, zones):
    last_close = df['close'].iloc[-1]
    ema20_last = df['ema20'].iloc[-1]
    ema50_last = df['ema50'].iloc[-1]

    ema_bullish = ema20_last > ema50_last
    ema_bearish = ema20_last < ema50_last

    if ema_bullish:
        trend_text = "TREND: UP (EMA20 > EMA50)"
        trend_color = "green"
    elif ema_bearish:
        trend_text = "TREND: DOWN (EMA20 < EMA50)"
        trend_color = "red"
    else:
        trend_text = "TREND: SIDEWAYS"
        trend_color = "gray"

    bull_zones = [z for z in zones if z.is_bull]
    bear_zones = [z for z in zones if not z.is_bull]

    has_bull_fvg = len(bull_zones) > 0
    has_bear_fvg = len(bear_zones) > 0

    def yn(flag): return "green" if flag else "red"

    zone_lines = [
        ("ZONE:", "gray"),
        (f"  BULL FVG: {'✓' if has_bull_fvg else '✗'}", yn(has_bull_fvg)),
        (f"  BEAR FVG: {'✓' if has_bear_fvg else '✗'}", yn(has_bear_fvg)),
    ]

    lines = [
        ("SMC STRUCTURE", "black"),
        (trend_text, trend_color),
    ] + zone_lines

    y = 0.96
    for text, color in lines:
        ax.text(
            0.02, y, text,
            transform=ax.transAxes,
            fontsize=8,
            color=color,
            ha="left", va="top"
        )
        y -= 0.035

# ---------------------------------------------------------
# CHART
# ---------------------------------------------------------

def plotchart(df, zones, title="SMC FVG View", exit_long=False, exit_short=False):
    df = df.copy()
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

    ax.plot(x, df["lb_crv"], color="gray", alpha=0.75, linewidth=1.2)
    ax.plot(x, df["ema20"], color="yellow", alpha=0.75, linewidth=1)
    ax.plot(x, df["ema50"], color="red", alpha=0.75, linewidth=1)

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

    draw_smc_box(ax, df, zones)

    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    rsi = df["rsi"]
    rsi_ema = df["rsi_ema"]

    ax2.fill_between(
        x,
        rsi,
        rsi_ema,
        where=(rsi > rsi_ema),
        color="green",
        alpha=0.15
    )

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

    for level in [25, 50, 78]:
        ax2.axhline(level, color="black", linestyle="--", linewidth=0.7, alpha=0.6)

    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI")
    ax2.grid(alpha=0.2)
    ax2.legend(loc="upper left")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    if isinstance(df.index, pd.DatetimeIndex):
        ax2.set_xticks(x[::max(1, len(x)//10)])
        ax2.set_xticklabels(
            df.index.strftime("%Y-%m-%d")[::max(1, len(x)//10)],
            rotation=45,
            fontsize=8
        )

    last_idx = len(df) - 1
    last_close = df["close"].iloc[-1]

    if exit_long:
        ax.scatter(last_idx, last_close, color="gold", marker="s", s=60, zorder=21)

    if exit_short:
        ax.text(
            last_idx, last_close, "❌",
            color="red", fontsize=16,
            ha="center", va="center",
            fontweight="bold", zorder=21
        )

    legend_text = (
        "■ EXIT LONG\n"
        "❌ EXIT SHORT"
    )

    ax.text(
        0.02, 0.02, legend_text,
        transform=ax.transAxes,
        fontsize=8, color="blue",
        ha="left", va="bottom",
        bbox=dict(
            facecolor="white",
            alpha=0.4,
            edgecolor="none",
            boxstyle="round,pad=0.3"
        )
    )

    plt.tight_layout()
    return fig

# ---------------------------------------------------------
# UI — TIMEFRAME, DATA LOADING, WINDOW MANAGEMENT
# ---------------------------------------------------------

st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Ticker", "ASML")

tf = st.sidebar.selectbox(
    "Timeframe",
    ["4H", "1D", "1W", "1M"],
    index=2
)

today = datetime.today()

if tf == "4H":
    start_date = today - timedelta(days=180)
    interval = "4h"
elif tf == "1D":
    start_date = today - timedelta(days=365)
    interval = "1d"
elif tf == "1W":
    start_date = today - timedelta(days=365*2)
    interval = "1wk"
elif tf == "1M":
    start_date = today - timedelta(days=365*5)
    interval = "1mo"

df = load_data(ticker, start_date, interval)

if df is None or df.empty:
    st.error("No data found.")
    st.stop()

if "window_start_idx" not in st.session_state:
    st.session_state.window_start_idx = 0

if "window_end_idx" not in st.session_state:
    st.session_state.window_end_idx = len(df) - 1

col1, col2, col3 = st.columns(3)

if col1.button("⬅️ Previous"):
    st.session_state.window_start_idx = max(0, st.session_state.window_start_idx - 1)
    st.session_state.window_end_idx = max(0, st.session_state.window_end_idx - 1)

if col2.button("Next ➡️"):
    st.session_state.window_start_idx = min(len(df) - 1, st.session_state.window_start_idx + 1)
    st.session_state.window_end_idx = min(len(df) - 1, st.session_state.window_end_idx + 1)

start_idx = st.session_state.window_start_idx
end_idx = st.session_state.window_end_idx

if start_idx > end_idx:
    start_idx, end_idx = end_idx, start_idx

df_slice = df.iloc[start_idx:end_idx + 1]

with col3:
    if len(df_slice) > 0:
        st.write(f"Data from **{df_slice.index[0].date()} → {df_slice.index[-1].date()}**")
    else:
        st.write("Visible Window: —")

st.session_state.window_start_idx = max(0, min(st.session_state.window_start_idx, len(df) - 1))
st.session_state.window_end_idx = max(0, min(st.session_state.window_end_idx, len(df) - 1))

start_idx = st.session_state.window_start_idx
end_idx = st.session_state.window_end_idx

if start_idx > end_idx:
    start_idx, end_idx = end_idx, start_idx

df_slice = df.iloc[start_idx:end_idx + 1]

zones = detect_fvg_zones(df_slice)

# ---------------------------------------------------------
# ADAPTIVE REGIME / STRUCTURAL ENGINE
# ---------------------------------------------------------

last_broken_bear, last_broken_bull = get_last_broken_fvg(df_slice)

close_last = df_slice["close"].iloc[-1]
open_last = df_slice["open"].iloc[-1]
ema20_last = df_slice["ema20"].iloc[-1]
ema50_last = df_slice["ema50"].iloc[-1]

bullish_candle = close_last > open_last
bearish_candle = close_last < open_last

bull_mask = (close_last > ema20_last) and (ema20_last > ema50_last)
bear_mask = (close_last < ema20_last) and (ema20_last < ema50_last)

if "in_long" not in st.session_state:
    st.session_state.in_long = False
if "in_short" not in st.session_state:
    st.session_state.in_short = False

long_entry = False
short_entry = False
exit_long = False
exit_short = False

# -----------------------------
# BULLISH REGIME (Buy the Dip / Breakout)
# -----------------------------
if last_broken_bear is not None:
    ref_bear_low  = last_broken_bear["low"]
    ref_bear_high = last_broken_bear["high"]

    # LONG ENTRY (can re-activate after exit)
    if not st.session_state.in_long and bull_mask:
        fvg_range = ref_bear_high - ref_bear_low

        # Breakout above bearish FVG
        if close_last > ref_bear_high and bullish_candle:
            long_entry = True

        # Buy the dip above FVG low
        elif bullish_candle and close_last > ref_bear_low:
            long_entry = True

        # 5% reclaim re-entry
        if bullish_candle and close_last > ref_bear_low + 0.05 * fvg_range:
            long_entry = True

    # LONG EXIT (state remains active until invalidated)
    if st.session_state.in_long:
        # 1) Bearish candle
        if bearish_candle:
            exit_long = True
        # 2) Break below FVG low (structural break)
        if close_last < ref_bear_low:
            exit_long = True
        # 3) Trend mask breaks
        if not bull_mask:
            exit_long = True

# -----------------------------
# BEARISH REGIME (Sell the Rise / Breakdown)
# -----------------------------
if last_broken_bull is not None:
    ref_bull_low  = last_broken_bull["low"]
    ref_bull_high = last_broken_bull["high"]

    # SHORT ENTRY (can re-activate after exit)
    if not st.session_state.in_short and bear_mask:
        fvg_range = ref_bull_high - ref_bull_low

        # Breakdown below bullish FVG
        if close_last < ref_bull_low and bearish_candle:
            short_entry = True

        # Sell the rise below FVG high
        elif bearish_candle and close_last < ref_bull_high:
            short_entry = True

        # 5% reclaim re-entry
        if bearish_candle and close_last < ref_bull_high - 0.05 * fvg_range:
            short_entry = True

    # SHORT EXIT (state remains active until invalidated)
    if st.session_state.in_short:
        # 1) Bullish candle
        if bullish_candle:
            exit_short = True
        # 2) Break above FVG high (structural break)
        if close_last > ref_bull_high:
            exit_short = True
        # 3) Trend mask breaks
        if not bear_mask:
            exit_short = True

# -----------------------------
# APPLY ENTRIES / EXITS + REGIME FLIP
# -----------------------------

# LONG ENTRY
if long_entry:
    st.session_state.in_long = True
    st.session_state.in_short = False

# SHORT ENTRY
if short_entry:
    st.session_state.in_short = True
    st.session_state.in_long = False

# LONG EXIT + possible flip to SHORT
if exit_long:
    st.session_state.in_long = False
    if last_broken_bull is not None:
        if close_last < last_broken_bull["low"] and bearish_candle:
            st.session_state.in_short = True

# SHORT EXIT + possible flip to LONG
if exit_short:
    st.session_state.in_short = False
    if last_broken_bear is not None:
        if close_last > last_broken_bear["high"] and bullish_candle:
            st.session_state.in_long = True

# ---------------------------------------------------------
# UI SIGNALS
# ---------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

with c1:
    if long_entry:
        st.success("📈 LONG ENTRY (FVG/Mask)")
    elif st.session_state.in_long:
        st.info("🟢 LONG ACTIVE")
    else:
        st.info("—")

with c2:
    if short_entry:
        st.error("📉 SHORT ENTRY (FVG/Mask)")
    elif st.session_state.in_short:
        st.info("🔴 SHORT ACTIVE")
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

# ---------------------------------------------------------
# DRAW CHART
# ---------------------------------------------------------

fig = plotchart(df_slice, zones, title=f"{ticker} — {tf} SMC FVG Regime View", exit_long=exit_long, exit_short=exit_short)
st.pyplot(fig)
