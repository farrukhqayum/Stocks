import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta

st.set_page_config(page_title="SMC Dashboard", layout="wide")
st.title("📈 SMART MONEY CONCEPTS (SMC)")

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

    # Normalize column names
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]

    # Handle datetime index properly
    if isinstance(df.index, pd.DatetimeIndex):
        df["Date"] = df.index
    elif "date" in df.columns:
        df["Date"] = pd.to_datetime(df["date"])
    elif "datetime" in df.columns:
        df["Date"] = pd.to_datetime(df["datetime"])
    else:
        raise ValueError("No valid datetime column found in downloaded data.")

    df.set_index("Date", inplace=True)

    # Clean OHLC
    df = df.dropna(subset=["open", "high", "low", "close"]).astype(float)

    # Indicators
    df['ema20'] = ema(df.close, 20)
    df['ema50'] = ema(df.close, 50)
    df['ema200'] = ema(df.close, 200)
    df['rsi'] = compute_rsi(df.close)
    df['rsi_ema'] = ema(df['rsi'], 14)
    df['lb_crv'] = compute_lb_curve(df)
    df = df.bfill()
    df = df.ffill()

    return df


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
            is_fvg_up = low[i] > high[i-2]
            is_fvg_dn = high[i] < low[i-2]

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

def pine_candle_engine(df):
    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    ema20 = df['ema20'].values
    ema50 = df['ema50'].values
    ema200 = df['ema200'].values
    rsi = df['rsi'].values
    rsi_ema = df['rsi_ema'].values
    lb = df['lb_crv'].values

    n = len(df)
    if n < 3:
        return {
            "last_pattern": None,
            "pattern_bull": None,
            "pattern_idx": None,
            "rejected": False,
            "expired": True,
            "bull_signal": False,
            "bear_signal": False,
            "bullSweep": False,
            "bearSweep": False,
            "ema_bullish": False,
            "ema_bearish": False,
            "mom_bullish": False,
            "mom_bearish": False,
            "strong_bullish": False,
            "strong_bearish": False
        }

    # --- Trend filters (match Pine) ---
    ema_up = (ema20 > ema50) & (ema50 > ema200)
    ema_down = (ema20 < ema50) & (ema50 < ema200)
    lb_up = c > lb
    lb_down = c < lb

    trend_up = (c > ema20) & (ema_up | lb_up)
    trend_down = (c < ema20) & (ema_down | lb_down)

    last_pattern = None
    pattern_bull = None
    pattern_idx = None  # index of the bar where pattern is anchored (like pattern_bar in Pine)

    # --- Scan bars, enforcing priority: 3-candle > 2-candle > 1-candle ---
    for i in range(2, n):
        body0 = abs(c[i] - o[i])
        body1 = abs(c[i-1] - o[i-1])
        body2 = abs(c[i-2] - o[i-2])
        crange0 = h[i] - l[i]
        wickHigh = h[i] - max(o[i], c[i])
        wickLow = min(o[i], c[i]) - l[i]

        # 3-candle patterns (highest priority)
        isMorning = (
            trend_down[i]
            and c[i-2] < o[i-2]
            and body1 < body2 * 0.4
            and c[i] > (o[i-2] + c[i-2]) / 2
        )
        isEvening = (
            trend_up[i]
            and c[i-2] > o[i-2]
            and body1 < body2 * 0.4
            and c[i] < (o[i-2] + c[i-2]) / 2
        )

        if isMorning:
            last_pattern = "Morning Star"
            pattern_bull = True
            pattern_idx = i - 1  # center bar, like Pine
            continue

        if isEvening:
            last_pattern = "Evening Star"
            pattern_bull = False
            pattern_idx = i - 1
            continue

        # 2-candle patterns
        bullEngulf = (
            trend_down[i]
            and c[i-1] < o[i-1]
            and c[i] > o[i]
            and o[i] <= c[i-1]
            and c[i] >= o[i-1]
        )
        bearEngulf = (
            trend_up[i]
            and c[i-1] > o[i-1]
            and c[i] < o[i]
            and o[i] >= c[i-1]
            and c[i] <= o[i-1]
        )

        bullPierce = (
            trend_down[i]
            and c[i-1] < o[i-1]
            and c[i] > (o[i-1] + c[i-1]) / 2
        )
        bearDark = (
            trend_up[i]
            and c[i-1] > o[i-1]
            and c[i] < (o[i-1] + c[i-1]) / 2
        )

        tweezerBot = (
            trend_down[i]
            and abs(l[i] - l[i-1]) < (crange0 * 0.1)
        )
        tweezerTop = (
            trend_up[i]
            and abs(h[i] - h[i-1]) < (crange0 * 0.1)
        )

        if bullEngulf:
            last_pattern = "Bull Engulfing"
            pattern_bull = True
            pattern_idx = i
            continue

        if bearEngulf:
            last_pattern = "Bear Engulfing"
            pattern_bull = False
            pattern_idx = i
            continue

        if bullPierce:
            last_pattern = "Piercing"
            pattern_bull = True
            pattern_idx = i
            continue

        if bearDark:
            last_pattern = "Dark Cloud"
            pattern_bull = False
            pattern_idx = i
            continue

        if tweezerBot:
            last_pattern = "Tweezer Bottom"
            pattern_bull = True
            pattern_idx = i - 1  # like Pine
            continue

        if tweezerTop:
            last_pattern = "Tweezer Top"
            pattern_bull = False
            pattern_idx = i - 1
            continue

        # 1-candle patterns (lowest priority)
        isHammer = (
            trend_down[i]
            and wickLow > body0 * 2
            and wickHigh < body0 * 0.5
        )
        isStar = (
            trend_up[i]
            and wickHigh > body0 * 2
            and wickLow < body0 * 0.5
        )

        if isHammer:
            last_pattern = "Hammer"
            pattern_bull = True
            pattern_idx = i
            continue

        if isStar:
            last_pattern = "Shooting Star"
            pattern_bull = False
            pattern_idx = i
            continue

    if last_pattern is None or pattern_idx is None:
        pattern_bull = None

    expired = True
    rejected = False
    bull_signal = False
    bear_signal = False

    if last_pattern is not None and pattern_idx is not None:
        # barsAgo: how many bars since pattern bar
        barsAgo = n - 1 - pattern_idx
        expired = barsAgo > 20

        patLow = l[pattern_idx]
        patHigh = h[pattern_idx]
        close_last = c[-1]

        if pattern_bull:
            rejected = close_last < patLow
        else:
            rejected = close_last > patHigh

        if not expired and not rejected:
            rsi_last = rsi[-1]
            rsi_ema_last = rsi_ema[-1]
            lb_last = lb[-1]
            if pattern_bull:
                bull_signal = (close_last > lb_last * 0.98) and (rsi_last >= rsi_ema_last)
            else:
                bear_signal = (close_last <= lb_last) and (rsi_last <= rsi_ema_last)

    # --- Trend / momentum / structure (match Pine semantics) ---
    ema_bullish = ema20[-1] > ema50[-1]
    ema_bearish = ema20[-1] < ema50[-1]

    rsi_last = rsi[-1]
    rsi_ema_last = rsi_ema[-1]
    lb_last = lb[-1]

    mom_bullish = (rsi_last > 51 and rsi_last > rsi_ema_last) or (c[-1] > lb_last * 1.02)
    mom_bearish = (rsi_last < 44 and rsi_last < rsi_ema_last) or (c[-1] < lb_last * 0.98)

    strong_bullish = ema_bullish and c[-1] > lb_last
    strong_bearish = ema_bearish and c[-1] < lb_last

    bullSweep = False
    bearSweep = False
    if n >= 2:
        bullSweep = (l[-1] < l[-2]) and (c[-1] > (h[-1] + l[-1]) / 2)
        bearSweep = (h[-1] > h[-2]) and (c[-1] < (h[-1] + l[-1]) / 2)

    return {
        "last_pattern": last_pattern,
        "pattern_bull": pattern_bull,
        "pattern_idx": pattern_idx,
        "rejected": rejected,
        "expired": expired,
        "bull_signal": bull_signal,
        "bear_signal": bear_signal,
        "bullSweep": bullSweep,
        "bearSweep": bearSweep,
        "ema_bullish": ema_bullish,
        "ema_bearish": ema_bearish,
        "mom_bullish": mom_bullish,
        "mom_bearish": mom_bearish,
        "strong_bullish": strong_bullish,
        "strong_bearish": strong_bearish
    }
    
def plot_pattern_label(ax, df, pattern_idx, pattern_name, pattern_bullish, rejected):
    if pattern_idx is None or pattern_name is None:
        return
    high = df['high'].iloc[pattern_idx]
    low = df['low'].iloc[pattern_idx]
    x = pattern_idx
    offset = (high - low) * 0.15
    if pattern_bullish:
        y = low - offset
        va = "top"
        color = "green"
    else:
        y = high + offset
        va = "bottom"
        color = "red"
    if rejected:
        color = "gray"
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

def draw_smc_box(ax, df, zones):
    info = pine_candle_engine(df)

    last_pattern = info["last_pattern"]
    pattern_bull = info["pattern_bull"]
    pattern_idx = info["pattern_idx"]
    rejected = info["rejected"]
    expired = info["expired"]
    bull_signal = info["bull_signal"]
    bear_signal = info["bear_signal"]
    bullSweep = info["bullSweep"]
    bearSweep = info["bearSweep"]
    ema_bullish = info["ema_bullish"]
    ema_bearish = info["ema_bearish"]
    mom_bullish = info["mom_bullish"]
    mom_bearish = info["mom_bearish"]
    strong_bullish = info["strong_bullish"]
    strong_bearish = info["strong_bearish"]

    plot_pattern_label(ax, df, pattern_idx, last_pattern, pattern_bull, rejected)

    if bullSweep:
        sweep_text = "SWEEP: Buy-side ✓"
        sweep_color = "green"
    elif bearSweep:
        sweep_text = "SWEEP: Sell-side ✓"
        sweep_color = "red"
    else:
        sweep_text = "SWEEP: None"
        sweep_color = "gray"

    if last_pattern is None:
        pattern_text = "PATTERN: None"
        pattern_color = "gray"
    else:
        age = len(df) - 1 - pattern_idx
        state = "Rejected" if rejected else "Expired" if expired else "Active"
        pattern_text = f"PATTERN: {last_pattern} ({age} bars, {state})"
        pattern_color = (
            "gray" if (rejected or expired)
            else ("green" if pattern_bull else "red")
        )

    if strong_bullish:
        struct_text = "STRUCTURE: STRONG BULLISH"
        struct_color = "green"
    elif strong_bearish:
        struct_text = "STRUCTURE: STRONG BEARISH"
        struct_color = "red"
    else:
        struct_text = "STRUCTURE: NEUTRAL"
        struct_color = "gray"

    if ema_bullish:
        trend_text = "TREND: UP"
        trend_color = "green"
    elif ema_bearish:
        trend_text = "TREND: DOWN"
        trend_color = "red"
    else:
        trend_text = "TREND: SIDEWAYS"
        trend_color = "gray"

    if mom_bullish:
        mom_text = "MOM: BULLISH"
        mom_color = "green"
    elif mom_bearish:
        mom_text = "MOM: BEARISH"
        mom_color = "red"
    else:
        mom_text = "MOM: NEUTRAL"
        mom_color = "gray"

    if bull_signal:
        sig_text = "SIGNAL: 🟢 BUY THE DIP"
        sig_color = "green"
    elif bear_signal:
        sig_text = "SIGNAL: 🔴 SELL THE RISE"
        sig_color = "red"
    else:
        sig_text = "SIGNAL: —"
        sig_color = "gray"

    last_close = df['close'].iloc[-1]
    bull_zones = [z for z in zones if z.is_bull]
    bear_zones = [z for z in zones if not z.is_bull]

    has_bull_fvg = len(bull_zones) > 0
    has_bear_fvg = len(bear_zones) > 0
    inside_bull = any(z.bottom < last_close < z.top for z in bull_zones)
    inside_bear = any(z.bottom < last_close < z.top for z in bear_zones)
    first_touch_bull = any(z.touched for z in bull_zones)
    first_touch_bear = any(z.touched for z in bear_zones)

    def yn(flag): return "green" if flag else "red"

    zone_lines = [
        ("ZONE:", "gray"),
        (f"  BULL FVG: {'✓' if has_bull_fvg else '✗'}", yn(has_bull_fvg)),
        (f"  BEAR FVG: {'✓' if has_bear_fvg else '✗'}", yn(has_bear_fvg)),
        (f"  Inside Bull: {'✓' if inside_bull else '✗'}", yn(inside_bull)),
        (f"  Inside Bear: {'✓' if inside_bear else '✗'}", yn(inside_bear)),
        (f"  1stTouch Bull: {'✓' if first_touch_bull else '✗'}", yn(first_touch_bull)),
        (f"  1stTouch Bear: {'✓' if first_touch_bear else '✗'}", yn(first_touch_bear)),
    ]

    lines = [
        ("SMC & SIGNALS", "black"),
        (sweep_text, sweep_color),
        (pattern_text, pattern_color),
        (struct_text, struct_color),
        (trend_text + " | " + mom_text, mom_color),
        (sig_text, sig_color),
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

def plotchart(df, zones, title="SMC FVG View"):
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


    # -----------------------------
    # SIGNAL ANNOTATIONS ON CHART
    # -----------------------------
    info = pine_candle_engine(df)
    
    bull_signal = info["bull_signal"]
    bear_signal = info["bear_signal"]

    last_idx = len(df) - 1
    last_close = df["close"].iloc[-1]
    last_high = df["high"].iloc[-1]
    last_low = df["low"].iloc[-1]

    if bull_signal:
        ax.scatter(last_idx, last_low * 0.995, color="green", marker="^", s=50, zorder=20)
    
    if bear_signal:
        ax.scatter(last_idx, last_high * 1.005, color="red", marker="v", s=50, zorder=20)
    
    if exit_long:
        ax.scatter(last_idx, last_close, color="gold", marker="s", s=50, zorder=20)
    
    if exit_short:
        ax.text( last_idx, last_close, "❌", color="red", fontsize=16,  ha="center", va="center",
        fontweight="bold", zorder=20 )

    legend_text = (
        "▲ BUY THE DIP\n"
        "▼ SELL THE RISE\n"
        "■ EXIT LONG\n"
        "❌ EXIT SHORT"
    )
    
    ax.text(0.02, 0.02, legend_text, transform=ax.transAxes,
        fontsize=8, color="blue", ha="left", va="bottom",
        bbox=dict(
            facecolor="white",
            alpha=0.4,
            edgecolor="none",
            boxstyle="round,pad=0.3"))

    plt.tight_layout()
    return fig
# ---------------------------------------------------------
# UI — TIMEFRAME, DATA LOADING, WINDOW MANAGEMENT, SCROLLING
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

# -----------------------------------------
# INITIALIZE WINDOW STATE
# -----------------------------------------

if "window_start_idx" not in st.session_state:
    st.session_state.window_start_idx = 0

if "window_end_idx" not in st.session_state:
    st.session_state.window_end_idx = len(df) - 1

# -----------------------------------------
# SCROLL BUTTONS — MOVE BY BARS
# -----------------------------------------

col1, col2, col3 = st.columns(3)

if col1.button("⬅️ Previous"):
    st.session_state.window_start_idx = max(0, st.session_state.window_start_idx - 1)
    st.session_state.window_end_idx = max(0, st.session_state.window_end_idx - 1)

if col2.button("Next ➡️"):
    st.session_state.window_start_idx = min(len(df) - 1, st.session_state.window_start_idx + 1)
    st.session_state.window_end_idx = min(len(df) - 1, st.session_state.window_end_idx + 1)

# Recompute slice AFTER button clicks
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

# -----------------------------------------
# CLAMP WINDOW
# -----------------------------------------

st.session_state.window_start_idx = max(0, min(st.session_state.window_start_idx, len(df) - 1))
st.session_state.window_end_idx = max(0, min(st.session_state.window_end_idx, len(df) - 1))

# -----------------------------------------
# SLICE DATAFRAME BY INDEX
# -----------------------------------------

start_idx = st.session_state.window_start_idx
end_idx = st.session_state.window_end_idx

if start_idx > end_idx:
    start_idx, end_idx = end_idx, start_idx

df_slice = df.iloc[start_idx:end_idx + 1]

zones = detect_fvg_zones(df_slice)

# -----------------------------------------
# SIGNAL ENGINE (Pine-style)
# -----------------------------------------

info_slice = pine_candle_engine(df_slice)

pattern_idx = info_slice["pattern_idx"]
pattern_bull = info_slice["pattern_bull"]

close_last = df_slice["close"].iloc[-1]
lb_last = df_slice["lb_crv"].iloc[-1]

ema_bullish = info_slice["ema_bullish"]
ema_bearish = info_slice["ema_bearish"]

# Pattern low/high for SL
patLow = None
patHigh = None
if pattern_idx is not None:
    patLow = df_slice["low"].iloc[pattern_idx]
    patHigh = df_slice["high"].iloc[pattern_idx]

bull_signal = info_slice["bull_signal"]
bear_signal = info_slice["bear_signal"]
mom_bearish = info_slice["mom_bearish"]
mom_bullish = info_slice["mom_bullish"]
strong_bullish = info_slice["strong_bullish"]
strong_bearish = info_slice["strong_bearish"]
rejected = info_slice["rejected"]
expired = info_slice["expired"]

# -----------------------------------------
# TRADE STATE
# -----------------------------------------

if "in_long" not in st.session_state:
    st.session_state.in_long = False
if "in_short" not in st.session_state:
    st.session_state.in_short = False

long_signal = bull_signal and not st.session_state.in_long
short_signal = bear_signal and not st.session_state.in_short

sl_long = (
    st.session_state.in_long
    and pattern_bull is True
    and patLow is not None
    and close_last < patLow
)

sl_short = (
    st.session_state.in_short
    and pattern_bull is False
    and patHigh is not None
    and close_last > patHigh
)

tp_long = (
    st.session_state.in_long
    and (ema_bearish or mom_bearish or close_last < lb_last)
)

tp_short = (
    st.session_state.in_short
    and (ema_bullish or mom_bullish or close_last > lb_last)
)

expired_exit = expired and (st.session_state.in_long or st.session_state.in_short)

exit_long = sl_long or tp_long or expired_exit
exit_short = sl_short or tp_short or expired_exit

if long_signal:
    st.session_state.in_long = True
    st.session_state.in_short = False

if short_signal:
    st.session_state.in_short = True
    st.session_state.in_long = False

if exit_long:
    st.session_state.in_long = False

if exit_short:
    st.session_state.in_short = False

# -----------------------------------------
# DISPLAY SIGNAL COLUMNS
# -----------------------------------------

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

# -----------------------------------------
# DRAW CHART
# -----------------------------------------

fig = plotchart(df_slice, zones, title=f"{ticker} — {tf} SMC FVG View")
st.pyplot(fig)
