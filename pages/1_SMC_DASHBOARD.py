import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime, timedelta

# ---------------------------------------------------------
# INDICATORS & HELPERS
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

def compute_atr(df, length=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()

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
# DATA CLASSES
# ---------------------------------------------------------

@dataclass
class Zone:
    top: float
    bottom: float
    start_idx: int
    is_bull: bool
    is_ob: bool
    mitigated: bool = False
    taps: int = 0

@dataclass
class StructureState:
    last_hi: Optional[float] = None
    last_hi_idx: Optional[int] = None
    last_lo: Optional[float] = None
    last_lo_idx: Optional[int] = None
    is_uptrend: bool = False
    bos_events: List[Tuple[int, int, float, str]] = None

@dataclass
class PatternInfo:
    last_pattern: Optional[str]
    pattern_bull: Optional[bool]
    pattern_idx: Optional[int]
    rejected: bool
    expired: bool
    turning_point: bool
    turning_code: Optional[str]

@dataclass
class SignalState:
    bull_signal: bool
    bear_signal: bool
    in_long: bool
    in_short: bool
    exit_long: bool
    exit_short: bool

# ---------------------------------------------------------
# FVG + OB ENGINE
# ---------------------------------------------------------

class ZoneEngine:
    def __init__(self, max_age=100, fail_window=5, atr_gap_mult=0.1, max_taps=5):
        self.max_age = max_age
        self.fail_window = fail_window
        self.atr_gap_mult = atr_gap_mult
        self.max_taps = max_taps

    def _zones_overlap(self, z1, z2):
        return not (z1.bottom > z2.top or z2.bottom > z1.top)

    def _merge(self, zones):
        if len(zones) <= 1:
            return zones
        zones = sorted(zones, key=lambda z: z.start_idx)
        merged = [zones[0]]
        for z in zones[1:]:
            last = merged[-1]
            if self._zones_overlap(last, z):
                size1 = abs(last.top - last.bottom)
                size2 = abs(z.top - z.bottom)
                merged[-1] = last if size1 >= size2 else z
            else:
                merged.append(z)
        return merged

    def detect(self, df):
        high = df["high"].values
        low = df["low"].values
        open_ = df["open"].values
        close = df["close"].values
        atr = compute_atr(df).values
        n = len(df)

        zones: List[Zone] = []

        for i in range(2, n):
            gap_min = atr[i] * self.atr_gap_mult if not np.isnan(atr[i]) else 0.0

            # --- FVG (3-candle, gap-aware) ---
            fvg_up = low[i] > high[i-2] + gap_min
            fvg_dn = high[i] < low[i-2] - gap_min

            if fvg_up:
                zones.append(Zone(top=low[i], bottom=high[i-2], start_idx=i, is_bull=True, is_ob=False))
            if fvg_dn:
                zones.append(Zone(top=high[i], bottom=low[i-2], start_idx=i, is_bull=False, is_ob=False))

            # --- OB (displacement) ---
            disp_up = close[i] > high[i-1] and close[i] > open_[i]
            disp_dn = close[i] < low[i-1] and close[i] < open_[i]

            if disp_up and low[i-1] < low[i-2]:
                zones.append(Zone(top=high[i-1], bottom=low[i-1], start_idx=i-1, is_bull=True, is_ob=True))
            if disp_dn and high[i-1] > high[i-2]:
                zones.append(Zone(top=high[i-1], bottom=low[i-1], start_idx=i-1, is_bull=False, is_ob=True))

            # --- Gap OB ---
            if open_[i] > high[i-1] and close[i] > open_[i]:
                zones.append(Zone(top=open_[i], bottom=low[i-1], start_idx=i, is_bull=True, is_ob=True))
            if open_[i] < low[i-1] and close[i] < open_[i]:
                zones.append(Zone(top=high[i-1], bottom=open_[i], start_idx=i, is_bull=False, is_ob=True))

            # --- Update zones (age, taps, failure) ---
            to_delete = []
            for j, z in enumerate(zones):
                age = i - z.start_idx
                failed = False

                if age <= self.fail_window:
                    if z.is_bull and close[i] < z.bottom and close[i-1] < z.bottom:
                        failed = True
                    if not z.is_bull and close[i] > z.top and close[i-1] > z.top:
                        failed = True

                if not z.mitigated:
                    if high[i] > z.bottom and low[i] < z.top:
                        z.taps += 1
                    if z.taps > self.max_taps:
                        z.mitigated = True

                if age > self.max_age or failed:
                    to_delete.append(j)

            for j in reversed(to_delete):
                del zones[j]

        zones = [z for z in zones if not z.mitigated]
        return self._merge(zones)

# ---------------------------------------------------------
# MARKET STRUCTURE ENGINE (BOS / CHoCH)
# ---------------------------------------------------------

class StructureEngine:
    def __init__(self, swing_left=20, swing_right=5):
        self.swing_left = swing_left
        self.swing_right = swing_right

    def _pivots(self, df):
        high = df["high"].values
        low = df["low"].values
        n = len(df)
        sw_hi = np.full(n, np.nan)
        sw_lo = np.full(n, np.nan)

        for i in range(self.swing_left, n - self.swing_right):
            if high[i] == max(high[i-self.swing_left:i+self.swing_right+1]):
                sw_hi[i] = high[i]
            if low[i] == min(low[i-self.swing_left:i+self.swing_right+1]):
                sw_lo[i] = low[i]
        return sw_hi, sw_lo

    def compute(self, df):
        close = df["close"].values
        sw_hi, sw_lo = self._pivots(df)
        n = len(df)

        state = StructureState(bos_events=[])

        for i in range(n):
            if not np.isnan(sw_hi[i]):
                state.last_hi = sw_hi[i]
                state.last_hi_idx = i
            if not np.isnan(sw_lo[i]):
                state.last_lo = sw_lo[i]
                state.last_lo_idx = i

            bos_up = state.last_hi is not None and close[i] > state.last_hi
            bos_dn = state.last_lo is not None and close[i] < state.last_lo

            if bos_up:
                label = "BOS ↑" if state.is_uptrend else "CHoCH ↑"
                state.bos_events.append((state.last_hi_idx, i, state.last_hi, label))
                state.is_uptrend = True
                state.last_hi = None

            if bos_dn:
                label = "CHoCH ↓" if state.is_uptrend else "BOS ↓"
                state.bos_events.append((state.last_lo_idx, i, state.last_lo, label))
                state.is_uptrend = False
                state.last_lo = None

        return state

# ---------------------------------------------------------
# PATTERN ENGINE (Simplified but SMC-aware)
# ---------------------------------------------------------

class PatternEngine:
    def __init__(self, max_bars_valid=27):
        self.max_bars_valid = max_bars_valid

    def analyze(self, df):
        o = df["open"].values
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        n = len(df)

        if n < 3:
            return PatternInfo(None, None, None, False, True, False, None)

        last_pattern = None
        pattern_bull = None
        pattern_idx = None

        for i in range(2, n):
            body0 = abs(c[i] - o[i])
            body1 = abs(c[i-1] - o[i-1])
            body2 = abs(c[i-2] - o[i-2])

            # Morning Star
            if c[i-2] < o[i-2] and body1 < body2 * 0.4 and c[i] > (o[i-2] + c[i-2]) / 2:
                last_pattern = "Morning Star"; pattern_bull = True; pattern_idx = i-1

            # Evening Star
            if c[i-2] > o[i-2] and body1 < body2 * 0.4 and c[i] < (o[i-2] + c[i-2]) / 2:
                last_pattern = "Evening Star"; pattern_bull = False; pattern_idx = i-1

            # Bull Engulfing
            if c[i] > o[i] and c[i-1] < o[i-1] and c[i] >= h[i-1] and o[i] <= l[i-1]:
                last_pattern = "Bull Engulfing"; pattern_bull = True; pattern_idx = i

            # Bear Engulfing
            if c[i] < o[i] and c[i-1] > o[i-1] and c[i] <= l[i-1] and o[i] >= h[i-1]:
                last_pattern = "Bear Engulfing"; pattern_bull = False; pattern_idx = i

        if last_pattern is None:
            return PatternInfo(None, None, None, False, True, False, None)

        bars_ago = n - 1 - pattern_idx
        expired = bars_ago > self.max_bars_valid

        pat_low = l[pattern_idx]
        pat_high = h[pattern_idx]
        close_last = c[-1]

        rejected = close_last < pat_low if pattern_bull else close_last > pat_high

        turning_point = False
        turning_code = None

        if not expired and not rejected:
            if pattern_bull and close_last < o[-1]:
                turning_point = True; turning_code = "▼ Bearish Shift"
            if not pattern_bull and close_last > o[-1]:
                turning_point = True; turning_code = "▲ Bullish Shift"

        return PatternInfo(last_pattern, pattern_bull, pattern_idx, rejected, expired, turning_point, turning_code)

# ---------------------------------------------------------
# SIGNAL ENGINE
# ---------------------------------------------------------

class SignalEngine:
    def __init__(self):
        self.in_long = False
        self.in_short = False

    def compute(self, df, structure: StructureState, pattern: PatternInfo):
        close_last = df["close"].iloc[-1]
        open_last = df["open"].iloc[-1]
        ema20 = df["ema20"].iloc[-1]
        ema50 = df["ema50"].iloc[-1]
        lb = df["lb_crv"].iloc[-1]
        rsi = df["rsi"].iloc[-1]
        rsi_ema = df["rsi_ema"].iloc[-1]

        bullish = close_last > open_last
        bearish = close_last < open_last

        ema_bull = ema20 > ema50
        ema_bear = ema20 < ema50

        lb_up = close_last > lb * 1.02
        lb_down = close_last < lb * 0.98

        mom_bull = (rsi >= 50 and rsi > rsi_ema) or lb_up
        mom_bear = (rsi <= 44 and rsi < rsi_ema) or lb_down

        smc_bull = structure.is_uptrend
        smc_bear = not structure.is_uptrend

        bull_ok = smc_bull and ema_bull and mom_bull and lb_up
        bear_ok = smc_bear and ema_bear and mom_bear and lb_down

        bull_signal = False
        bear_signal = False
        exit_long = False
        exit_short = False

        if pattern.last_pattern and not pattern.expired and not pattern.rejected:
            if pattern.pattern_bull and bull_ok and bullish:
                bull_signal = True
            if not pattern.pattern_bull and bear_ok and bearish:
                bear_signal = True

        if bull_signal:
            self.in_long = True
            self.in_short = False

        if bear_signal:
            self.in_short = True
            self.in_long = False

        if self.in_long and (bearish or not bull_ok):
            exit_long = True
            self.in_long = False

        if self.in_short and (bullish or not bear_ok):
            exit_short = True
            self.in_short = False

        return SignalState(
            bull_signal=bull_signal,
            bear_signal=bear_signal,
            in_long=self.in_long,
            in_short=self.in_short,
            exit_long=exit_long,
            exit_short=exit_short
        )

# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

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
    else:
        df["Date"] = pd.to_datetime(df.index)
    df.set_index("Date", inplace=True)

    df = df.dropna(subset=["open", "high", "low", "close"]).astype(float)

    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["rsi"] = compute_rsi(df["close"])
    df["rsi_ema"] = ema(df["rsi"], 14)
    df["lb_crv"] = compute_lb_curve(df)

    return df

# ---------------------------------------------------------
# CHART
# ---------------------------------------------------------

def plot_pattern_label(ax, df, pattern: PatternInfo):
    if pattern.pattern_idx is None or pattern.last_pattern is None:
        return
    idx = pattern.pattern_idx
    high = df["high"].iloc[idx]
    low = df["low"].iloc[idx]
    offset = (high - low) * 0.15
    if pattern.pattern_bull:
        y = low - offset
        va = "top"
        color = "green"
    else:
        y = high + offset
        va = "bottom"
        color = "red"
    if pattern.rejected or pattern.expired:
        color = "gray"
    ax.text(
        idx, y,
        pattern.last_pattern,
        color=color,
        fontsize=6,
        ha="center",
        va=va,
        fontweight="bold",
        zorder=20
    )

def plotchart(df, zones, structure: StructureState, pattern: PatternInfo,
              title="SMC View", exit_long=False, exit_short=False):
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

    # ---------------------------------------------------------
    # SIGNAL MARKERS (TradingView-style)
    # ---------------------------------------------------------
    
    # Go Long
    long_df = df[df['long_signal'] == True]
    ax.scatter(long_df.index, long_df['Low'] * 0.995,
               marker='^', s=120, color='lime', label='Go Long')
    
    # Exit Long
    exit_long_df = df[df['exit_long'] == True]
    ax.scatter(exit_long_df.index, exit_long_df['High'] * 1.005,
               marker='o', s=90, color='yellow', label='Exit Long')
    
    # Go Short
    short_df = df[df['short_signal'] == True]
    ax.scatter(short_df.index, short_df['High'] * 1.005,
               marker='v', s=120, color='red', label='Go Short')
    
    # Exit Short
    exit_short_df = df[df['exit_short'] == True]
    ax.scatter(exit_short_df.index, exit_short_df['Low'] * 0.995,
               marker='o', s=90, color='orange', label='Exit Short')

    last_idx = len(df) - 1
    for z in zones:
        rect_x = z.start_idx - 0.5
        rect_width = (last_idx - z.start_idx) + 1
        color = "teal" if (z.is_bull and not z.is_ob) else \
                "blue" if (not z.is_bull and not z.is_ob) else \
                "green" if (z.is_bull and z.is_ob) else "brown"
        ax.add_patch(Rectangle(
            (rect_x, z.bottom),
            rect_width,
            z.top - z.bottom,
            facecolor=color,
            alpha=0.15,
            edgecolor=color,
            linestyle="--"
        ))

    # BOS / CHoCH lines
    if structure.bos_events:
        for start_i, end_i, price, label in structure.bos_events:
            ax.hlines(price, start_i, end_i, colors="orange", linestyles="--", linewidth=1)
            mid = (start_i + end_i) // 2
            ax.text(
                mid, price,
                label,
                color="orange",
                fontsize=7,
                ha="center",
                va="bottom",
                fontweight="bold"
            )

    plot_pattern_label(ax, df, pattern)

    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    rsi = df["rsi"]
    rsi_ema = df["rsi_ema"]

    ax2.fill_between(x, rsi, rsi_ema, where=(rsi > rsi_ema), color="green", alpha=0.15)
    ax2.fill_between(x, rsi, rsi_ema, where=(rsi < rsi_ema), color="red", alpha=0.15)

    ax2.plot(x, rsi, color="gray", linewidth=1.2)
    ax2.plot(x, rsi_ema, color="gold", linewidth=1.2)

    for level in [25, 50, 78]:
        ax2.axhline(level, color="black", linestyle="--", linewidth=0.7, alpha=0.6)

    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI")
    ax2.grid(alpha=0.2)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    if isinstance(df.index, pd.DatetimeIndex):
        ax2.set_xticks(x[::max(1, len(x)//10)])
        ax2.set_xticklabels(
            df.index.strftime("%Y-%m-%d")[::max(1, len(x)//10)],
            rotation=45,
            fontsize=8
        )

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

    legend_text = "■ EXIT LONG\n❌ EXIT SHORT"
    ax.text(
        0.02, 0.02, legend_text,
        transform=ax.transAxes,
        fontsize=8, color="blue",
        ha="left", va="bottom",
        bbox=dict(facecolor="white", alpha=0.4, edgecolor="none", boxstyle="round,pad=0.3")
    )

    if pattern.turning_point and pattern.turning_code:
        idx = len(df) - 1
        high_val = df["high"].iloc[idx]
        ax.text(
            idx, high_val * 1.01,
            pattern.turning_code,
            color="orange",
            fontsize=8,
            ha="center",
            va="bottom",
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="orange")
        )

    plt.tight_layout()
    return fig

# ---------------------------------------------------------
# UI — TIMEFRAME, DATA LOADING, WINDOW MANAGEMENT
# ---------------------------------------------------------

st.set_page_config(page_title="SMC Dashboard", layout="wide")
st.title("📈 SMART MONEY CONCEPTS (SMC)")

st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker", "ASML")
tf = st.sidebar.selectbox("Timeframe", ["4H", "1D", "1W", "1M"], index=2)

today = datetime.today()
if tf == "4H":
    start_date = today - timedelta(days=90)
    interval = "4h"
elif tf == "1D":
    start_date = today - timedelta(days=180)
    interval = "1d"
elif tf == "1W":
    start_date = today - timedelta(days=365)
    interval = "1wk"
else:
    start_date = today - timedelta(days=365*2)
    interval = "1mo"

df = load_data(ticker, start_date, interval)
if df is None or df.empty:
    st.error("No data found.")
    st.stop()

zone_engine = ZoneEngine()
structure_engine = StructureEngine()
pattern_engine = PatternEngine()
signal_engine = SignalEngine()

col1, col2, col3 = st.columns(3)

if "last_tf" not in st.session_state:
    st.session_state.last_tf = tf
if "window_start_idx" not in st.session_state:
    st.session_state.window_start_idx = 0
if "window_end_idx" not in st.session_state:
    st.session_state.window_end_idx = min(80, len(df)-1)

if st.session_state.last_tf != tf:
    st.session_state.window_start_idx = 0
    st.session_state.window_end_idx = len(df) - 1
    st.session_state.last_tf = tf

if col1.button("⬅️ Previous"):
    st.session_state.window_end_idx = max(
        st.session_state.window_start_idx + 20,
        st.session_state.window_end_idx - 1
    )

if col2.button("Next ➡️"):
    st.session_state.window_end_idx = min(
        len(df) - 1,
        st.session_state.window_end_idx + 1
    )

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

# ---------------------------------------------------------
# ENGINES: ZONES, STRUCTURE, PATTERN, SIGNALS
# ---------------------------------------------------------

zones = zone_engine.detect(df_slice)
structure_state = structure_engine.compute(df_slice)
pattern_info = pattern_engine.analyze(df_slice)
signal_state = signal_engine.compute(df_slice, structure_state, pattern_info)

# ---------------------------------------------------------
# UI SIGNALS
# ---------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

with c1:
    if signal_state.bull_signal:
        st.success("📈 LONG ENTRY")
    elif signal_state.in_long:
        st.info("🟢 LONG ACTIVE")
    else:
        st.info("—")

with c2:
    if signal_state.bear_signal:
        st.error("📉 SHORT ENTRY")
    elif signal_state.in_short:
        st.info("🔴 SHORT ACTIVE")
    else:
        st.info("—")

with c3:
    if signal_state.exit_long:
        st.warning("🔔 EXIT LONG")
    else:
        st.info("—")

with c4:
    if signal_state.exit_short:
        st.warning("🔔 EXIT SHORT")
    else:
        st.info("—")

# ---------------------------------------------------------
# DRAW CHART
# ---------------------------------------------------------

fig = plotchart(
    df_slice,
    zones,
    structure_state,
    pattern_info,
    title=f"{ticker} — {tf} SMC FVG/OB/Structure View",
    exit_long=signal_state.exit_long,
    exit_short=signal_state.exit_short
)

st.pyplot(fig)
