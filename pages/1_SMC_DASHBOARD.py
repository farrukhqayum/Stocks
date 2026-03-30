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

    ema_up = (ema20 > ema50) & (ema50 > ema200)
    ema_down = (ema20 < ema50) & (ema50 < ema200)
    lb_up = c > lb
    lb_down = c < lb

    trend_up = (c > ema20) & (ema_up | lb_up)
    trend_down = (c < ema20) & (ema_down | lb_down)

    last_pattern = None
    pattern_bull = None
    pattern_idx = None

    for i in range(2, n):
        body0 = abs(c[i] - o[i])
        body1 = abs(c[i-1] - o[i-1])
        body2 = abs(c[i-2] - o[i-2])
        crange0 = h[i] - l[i]
        wickHigh = h[i] - max(o[i], c[i])
        wickLow = min(o[i], c[i]) - l[i]

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
            pattern_idx = i - 1
            continue

        if isEvening:
            last_pattern = "Evening Star"
            pattern_bull = False
            pattern_idx = i - 1
            continue

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
            pattern_idx = i - 1
            continue

        if tweezerTop:
            last_pattern = "Tweezer Top"
            pattern_bull = False
            pattern_idx = i - 1
            continue

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

def fvg_mask_regime(df):
    """
    FVG Mask Regime Model:
    - Bull regime: price above EMA20, EMA20 > EMA50, and last broken bearish FVG holds as support.
    - Bear regime: price below EMA20, EMA20 < EMA50, and last broken bullish FVG holds as resistance.
    """
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    ema20 = df["ema20"].values
    ema50 = df["ema50"].values

    n = len(df)
    if n < 3:
        return {
            "bull_mask_on": False,
            "bear_mask_on": False,
            "ref_bear_low": None,
            "ref_bear_high": None,
            "ref_bull_low": None,
            "ref_bull_high": None,
        }

    close_last = c[-1]
    ema20_last = ema20[-1]
    ema50_last = ema50[-1]

    bull_trend_mask = (close_last > ema20_last) and (ema20_last > ema50_last)
    bear_trend_mask = (close_last < ema20_last) and (ema20_last < ema50_last)

    ref_bear_low = None
    ref_bear_high = None
    ref_bull_low = None
    ref_bull_high = None

    last_break_bear_idx = None
    last_break_bull_idx = None

    for i in range(2, n):
        is_fvg_up = l[i] > h[i-2]
        is_fvg_dn = h[i] < l[i-2]

        if is_fvg_dn:
            top = h[i]
            bottom = l[i-2]
            for j in range(i, n):
                if c[j] > top:
                    if last_break_bear_idx is None or j >= last_break_bear_idx:
                        last_break_bear_idx = j
                        ref_bear_low = bottom
                        ref_bear_high = top

        if is_fvg_up:
            top = l[i]
            bottom = h[i-2]
            for j in range(i, n):
                if c[j] < bottom:
                    if last_break_bull_idx is None or j >= last_break_bull_idx:
                        last_break_bull_idx = j
                        ref_bull_low = bottom
                        ref_bull_high = top

    bull_mask_on = False
    bear_mask_on = False

    if bull_trend_mask and ref_bear_low is not None and ref_bear_high is not None:
        if close_last > ref_bear_low:
            bull_mask_on = True

    if bear_trend_mask and ref_bull_low is not None and ref_bull_high is not None:
        if close_last < ref_bull_high:
            bear_mask_on = True

    return {
        "bull_mask_on": bull_mask_on,
        "bear_mask_on": bear_mask_on,
        "ref_bear_low": ref_bear_low,
        "ref_bear_high": ref_bear_high,
        "ref_bull_low": ref_bull_low,
        "ref_bull_high": ref_bull_high,
    }

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
        ax.scatter(last_idx, last_close, color="gold", marker="s", s=60, zorder=21)

    if exit_short:
        ax.text(
            last_idx, last_close, "❌",
            color="red", fontsize=16,
            ha="center", va="center",
            fontweight="bold", zorder=21
        )

    legend_text = (
        "▲ BUY THE DIP\n"
        "▼ SELL THE RISE\n"
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
# FVG MASK REGIME + TRADE STATE
# ---------------------------------------------------------
def get_last_broken_fvg(df):
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(df)

    last_broken_bear = None   # bearish FVG broken UP → bull reference
    last_broken_bull = None   # bullish FVG broken DOWN → bear reference

    for i in range(2, n):
        # Bearish FVG (for bulls)
        if h[i] < l[i-2]:
            top = h[i]
            bottom = l[i-2]
            # broken UP?
            for j in range(i, n):
                if c[j] > top:
                    last_broken_bear = {"low": bottom, "high": top, "break_idx": j}

        # Bullish FVG (for bears)
        if l[i] > h[i-2]:
            top = l[i]
            bottom = h[i-2]
            # broken DOWN?
            for j in range(i, n):
                if c[j] < bottom:
                    last_broken_bull = {"low": bottom, "high": top, "break_idx": j}

    return last_broken_bear, last_broken_bull
    
# ---------------------------------------------------------
# FVG MASK REGIME ENGINE
# ---------------------------------------------------------

last_broken_bear, last_broken_bull = get_last_broken_fvg(df_slice)

close_last = df_slice["close"].iloc[-1]
open_last = df_slice["open"].iloc[-1]
ema20_last = df_slice["ema20"].iloc[-1]
ema50_last = df_slice["ema50"].iloc[-1]

bullish_candle = close_last > open_last
bearish_candle = close_last < open_last

# -----------------------------
# TREND MASKS
# -----------------------------
bull_mask = (close_last > ema20_last) and (ema20_last > ema50_last)
bear_mask = (close_last < ema20_last) and (ema20_last < ema50_last)

# -----------------------------
# SESSION STATE INIT
# -----------------------------
if "in_long" not in st.session_state:
    st.session_state.in_long = False
if "in_short" not in st.session_state:
    st.session_state.in_short = False

if "ref_low" not in st.session_state:
    st.session_state.ref_low = None
if "ref_high" not in st.session_state:
    st.session_state.ref_high = None

# -----------------------------
# BULLISH REGIME LOGIC
# -----------------------------
long_entry = False
exit_long = False

if last_broken_bear is not None:
    ref_low = last_broken_bear["low"]
    ref_high = last_broken_bear["high"]

    # If we are NOT in long and bull mask is ON
    if not st.session_state.in_long and bull_mask:
        # 1) Breakout long
        if close_last > ref_high:
            long_entry = True

        # 2) Buy the dip: bullish candle while above FVG low
        elif bullish_candle and close_last > ref_low:
            long_entry = True

        # 3) Re‑entry rule: 2nd candle closes above 1st bullish FVG (5% range)
        fvg_range = ref_high - ref_low
        if bullish_candle and close_last > ref_low + 0.05 * fvg_range:
            long_entry = True

    # If already long → check exits
    if st.session_state.in_long:
        # Structural SL: price breaks below FVG low
        if close_last < ref_low:
            exit_long = True

        # Trend mask breaks
        if not bull_mask:
            exit_long = True

        # Bearish candle exit
        if bearish_candle:
            exit_long = True

# -----------------------------
# BEARISH REGIME LOGIC
# -----------------------------
short_entry = False
exit_short = False

if last_broken_bull is not None:
    ref_low = last_broken_bull["low"]
    ref_high = last_broken_bull["high"]

    if not st.session_state.in_short and bear_mask:
        # 1) Breakdown short
        if close_last < ref_low:
            short_entry = True

        # 2) Sell the rally: bearish candle while below FVG high
        elif bearish_candle and close_last < ref_high:
            short_entry = True

        # 3) Re‑entry rule: 2nd candle closes below 1st bearish FVG (5% range)
        fvg_range = ref_high - ref_low
        if bearish_candle and close_last < ref_high - 0.05 * fvg_range:
            short_entry = True

    # If already short → check exits
    if st.session_state.in_short:
        # Structural SL: price breaks above FVG high
        if close_last > ref_high:
            exit_short = True

        # Trend mask breaks
        if not bear_mask:
            exit_short = True

        # Bullish candle exit
        if bullish_candle:
            exit_short = True

# -----------------------------
# APPLY ENTRIES / EXITS
# -----------------------------
# -----------------------------
# APPLY ENTRIES / EXITS
# -----------------------------

# --- LONG ENTRY ---
if long_entry:
    st.session_state.in_long = True
    st.session_state.in_short = False

# --- SHORT ENTRY ---
if short_entry:
    st.session_state.in_short = True
    st.session_state.in_long = False

# --- LONG EXIT ---
if exit_long:
    st.session_state.in_long = False

    # *** REGIME FLIP: LONG → SHORT ***
    # If price breaks below bearish FVG AND candle is bearish → SHORT
    if last_broken_bear is not None:
        if close_last < last_broken_bear["low"] and bearish_candle:
            st.session_state.in_short = True

# --- SHORT EXIT ---
if exit_short:
    st.session_state.in_short = False

    # *** REGIME FLIP: SHORT → LONG ***
    # If price breaks above bullish FVG AND candle is bullish → LONG
    if last_broken_bull is not None:
        if close_last > last_broken_bull["high"] and bullish_candle:
            st.session_state.in_long = True


# -----------------------------
# UI SIGNALS
# -----------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    if long_entry:
        st.success("📈 LONG (FVG MASK)")
    elif st.session_state.in_long:
        st.info("🟢 LONG ACTIVE")
    else:
        st.info("—")

with c2:
    if short_entry:
        st.error("📉 SHORT (FVG MASK)")
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

fig = plotchart(
    df_slice,
    zones,
    title=f"{ticker} — {tf} SMC FVG View",
    exit_long=exit_long,
    exit_short=exit_short
)

st.pyplot(fig)
