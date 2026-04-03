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

class FVGZone:
    def __init__(self, top, bottom, start_idx, is_bull):
        self.top = top
        self.bottom = bottom
        self.start_idx = start_idx
        self.is_bull = is_bull
        self.is_mitigated = False
        self.mitigated_idx = None
        self.touched = False
        
# ---------------------------------------------------------
# CANDLESTICK ENGINE
# ---------------------------------------------------------
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

        # 3-candle patterns
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
            pattern_idx = i - 1
            continue

        if tweezerTop:
            last_pattern = "Tweezer Top"
            pattern_bull = False
            pattern_idx = i - 1
            continue

        # 1-candle patterns
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

    # Pattern validation
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

    # Trend + momentum
    ema_bullish = ema20[-1] > ema50[-1]
    ema_bearish = ema20[-1] < ema50[-1]

    rsi_last = rsi[-1]
    rsi_ema_last = rsi_ema[-1]
    lb_last = lb[-1]

    mom_bullish = (rsi_last > 51 and rsi_last > rsi_ema_last) or (c[-1] > lb_last * 1.02)
    mom_bearish = (rsi_last < 44 and rsi_last < rsi_ema_last) or (c[-1] < lb_last * 0.98)

    strong_bullish = ema_bullish and c[-1] > lb_last
    strong_bearish = ema_bearish and c[-1] < lb_last

    # Sweeps
    bullSweep = False
    bearSweep = False
    if n >= 2:
        bullSweep = (l[-1] < l[-2]) and (c[-1] > (h[-1] + l[-1]) / 2)
        bearSweep = (h[-1] > h[-2]) and (c[-1] < (h[-1] + l[-1]) / 2)

    # ---------------------------------------------------------
    # TURNING POINT ENGINE (Ultra-Compact Codes)
    # ---------------------------------------------------------
    turning_point = False
    turning_code = None

    if last_pattern is not None and pattern_idx is not None and not expired and not rejected:
        body_last = abs(c[-1] - o[-1])
        range_last = h[-1] - l[-1]
        wick_high_last = h[-1] - max(o[-1], c[-1])
        wick_low_last = min(o[-1], c[-1]) - l[-1]

        # Bearish pattern → look for bullish reversal
        if pattern_bull is False:
            # Bull Reject  → "Reject Lows"
            if (c[-1] > o[-1]) and (wick_low_last > body_last * 1.2):
                turning_point = True
                turning_code = "▲ Rejecting Lows"

            # Bull Engulf → "Bull Shift"
            if (n >= 2 and
                c[-2] < o[-2] and          # prev bearish
                c[-1] > o[-1] and          # curr bullish
                o[-1] <= c[-2] and
                c[-1] >= o[-2]):
                turning_point = True
                turning_code = "▲ Bullish Shift"

            # Bull Body   → "Bull Drive"
            if (c[-1] > o[-1]) and (body_last > 0.55 * range_last):
                turning_point = True
                turning_code = "▲ Bullish Drive"

        # Bullish pattern → look for bearish reversal
        if pattern_bull is True:
            # Bear Reject → "Reject Highs"
            if (c[-1] < o[-1]) and (wick_high_last > body_last * 1.2):
                turning_point = True
                turning_code = "▼ Rejecting Highs"

            # Bear Engulf → "Bear Shift"
            if (n >= 2 and
                c[-2] > o[-2] and          # prev bullish
                c[-1] < o[-1] and          # curr bearish
                o[-1] >= c[-2] and
                c[-1] <= o[-2]):
                turning_point = True
                turning_code = "▼ Bearish Shift"

            # Bear Body   → "Bear Drive"
            if (c[-1] < o[-1]) and (body_last > 0.55 * range_last):
                turning_point = True
                turning_code = "▼ Bearish Drive"


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
        "strong_bearish": strong_bearish,
        "turning_point": turning_point,
        "turning_code": turning_code
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
                    z.mitigated_idx = i

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
def plotchart(df, zones, title="SMC FVG View", glong = False, gshort = False, elong = False, eshort = False):
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

    # -----------------------------
    # CANDLE PLOTTING
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

    # -----------------------------
    # INDICATORS
    # -----------------------------
    ax.plot(x, df["lb_crv"], color="gray", alpha=0.75, linewidth=1.2)
    ax.plot(x, df["ema20"], color="yellow", alpha=0.75, linewidth=1)
    ax.plot(x, df["ema50"], color="red", alpha=0.75, linewidth=1)

    # -----------------------------
    # FVG ZONES
    # -----------------------------
    last_idx = len(df) - 1
    
    for z in zones:
    
        rect_x = z.start_idx - 0.5
    
        if z.is_mitigated:
            end_idx = z.mitigated_idx
        else:
            end_idx = last_idx  # ACTIVE FVG → extend to current bar
    
        rect_width = (end_idx - z.start_idx) + 1
    
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

    # -----------------------------
    # SMC BOX
    # -----------------------------
    draw_smc_box(ax, df, zones)

    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    # -----------------------------
    # RSI PANEL
    # -----------------------------
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

    # -----------------------------
    # EXIT MARKERS
    # -----------------------------
    last_idx = len(df) - 1
    last_close = df["close"].iloc[-1]

    if elong:
        ax.scatter(last_idx, last_close, color="gold", marker="s", s=60, zorder=21)

    if eshort:
        ax.text(
            last_idx, last_close, "❌",
            color="red", fontsize=16,
            ha="center", va="center",
            fontweight="bold", zorder=21
        )

    # -----------------------------
    # LEGEND
    # -----------------------------
    legend_text = "■ EXIT LONG\n❌ EXIT SHORT"
    ax.text(
        0.02, 0.02, legend_text,
        transform=ax.transAxes,
        fontsize=8, color="blue",
        ha="left", va="bottom",
        bbox=dict(facecolor="white", alpha=0.4, edgecolor="none", boxstyle="round,pad=0.3")
    )
    
    # -----------------------------
    # PATTERN + REVERSAL ENGINE
    # -----------------------------
    info = pine_candle_engine(df)
    
    # Draw candlestick pattern label
    plot_pattern_label(
        ax, df,
        info["pattern_idx"],
        info["last_pattern"],
        info["pattern_bull"],
        info["rejected"]
    )
    
    # -----------------------------
    # REVERSAL DETECTION
    # -----------------------------
    bull_reversal = (
        info["last_pattern"] is not None
        and info["pattern_bull"] is True
        and not info["rejected"]
        and not info["expired"]
        and info["mom_bullish"]
        and info["ema_bullish"]
    )
    
    bear_reversal = (
        info["last_pattern"] is not None
        and info["pattern_bull"] is False
        and not info["rejected"]
        and not info["expired"]
        and info["mom_bearish"]
        and info["ema_bearish"]
    )
    
    if bull_reversal:
        reversal_text = "🟢 Bullish Reversal"
        reversal_color = "green"
    elif bear_reversal:
        reversal_text = "🔴 Bearish Reversal"
        reversal_color = "red"
    else:
        reversal_text = None
    
    # -----------------------------
    # DRAW REVERSAL TEXT INSIDE CHART
    # -----------------------------
    if reversal_text:
        ax.text(
            0.5, 0.1, reversal_text,
            transform=ax.transAxes,
            fontsize=10,
            color=reversal_color,
            fontweight="bold",
            bbox=dict(
                facecolor="white",
                alpha=0.5,
                edgecolor=reversal_color,
                boxstyle="round,pad=0.3"
            )
        )

    # -----------------------------
    # ENTRY MARKERS
    # -----------------------------
    for i in range(len(df)):
        price = df["close"].iloc[i]
    
        if df["long_entry_sig"].iloc[i] and glong:
            ax.scatter(i, price, color="lime", marker="^", s=10, zorder=22)
    
        if df["short_entry_sig"].iloc[i] and gshort:
            ax.scatter(i, price, color="red", marker="v", s=10, zorder=22)
    
        if df["exit_long_sig"].iloc[i] and elong:
            ax.scatter(i, price, color="gold", marker="s", s=10, zorder=22)
    
        if df["exit_short_sig"].iloc[i] and eshort:
            ax.text(i, price, "❌", color="red", fontsize=14,
                    ha="center", va="center", zorder=22)
        
    # ---------------------------------------------------------
    # DRAW TURNING POINT ABOVE BAR
    # ---------------------------------------------------------
    tp_flag  = info["turning_point"]
    tp_code  = info["turning_code"]
    
    if tp_flag and tp_code is not None:
        idx = len(df) - 1               
        high_val = df["high"].iloc[idx]

        ax.text(
            idx, high_val * 1.01,
            tp_code,
            color="orange",
            fontsize=8,
            ha="center",
            va="bottom",
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="orange")
        )

    plt.tight_layout()
    return fig

def precompute_signals(df_slice):

    # Make a copy to avoid SettingWithCopy warnings
    df_slice = df_slice.copy()

    # Initialize signal columns
    df_slice["long_entry_sig"] = False
    df_slice["short_entry_sig"] = False
    df_slice["exit_long_sig"] = False
    df_slice["exit_short_sig"] = False

    for i in range(2, len(df_slice)):

        row  = df_slice.iloc[i]
        prev = df_slice.iloc[i - 1]

        close_last = row["close"]
        open_last  = row["open"]
        ema20_last = row["ema20"]
        ema50_last = row["ema50"]
        lb_last    = row["lb_crv"]
        lb_prev    = prev["lb_crv"]

        bullish_candle = close_last > open_last
        bearish_candle = close_last < open_last

        # Trend masks (used only for bias, not for exits)
        bull_mask = (close_last > ema20_last) and (ema20_last > ema50_last)
        bear_mask = (close_last < ema20_last) and (ema20_last < ema50_last)

        # Last broken FVGs up to this bar
        lb_bear, lb_bull = get_last_broken_fvg(df_slice.iloc[:i+1])

        long_entry  = False
        short_entry = False
        exit_long   = False
        exit_short  = False

        # -------------------------------------------------
        # 1) LB-CURVE EARLY ENTRIES
        # -------------------------------------------------
        bull_lb_entry = (
            bullish_candle and
            close_last > lb_last and
            lb_last > lb_prev          # LB turning up
        )

        bear_lb_entry = (
            bearish_candle and
            close_last < lb_last and
            lb_last < lb_prev          # LB turning down
        )

        # -------------------------------------------------
        # 2) FVG-BASED STRUCTURAL ENTRIES
        # -------------------------------------------------
        fvg_bull_entry = False
        fvg_bear_entry = False

        if lb_bear is not None and bull_mask:
            ref_low  = lb_bear["low"]
            ref_high = lb_bear["high"]
            fvg_range = ref_high - ref_low

            # Price accepting above broken bearish FVG
            if bullish_candle and close_last > ref_low + 0.05 * fvg_range:
                fvg_bull_entry = True

        if lb_bull is not None and bear_mask:
            ref_low  = lb_bull["low"]
            ref_high = lb_bull["high"]
            fvg_range = ref_high - ref_low

            # Price accepting below broken bullish FVG
            if bearish_candle and close_last < ref_high - 0.05 * fvg_range:
                fvg_bear_entry = True

        # Combine LB + FVG for entries
        long_entry  = bull_lb_entry or fvg_bull_entry
        short_entry = bear_lb_entry or fvg_bear_entry

        # -------------------------------------------------
        # 3) CLEAN, EARLY EXITS (LB + STRUCTURE ONLY)
        # -------------------------------------------------
        # Exit long: lose LB support OR break FVG low
        if lb_bear is not None:
            ref_low  = lb_bear["low"]
            # Structural break of bullish reference
            fvg_long_break = close_last < ref_low
        else:
            fvg_long_break = False

        exit_long = (
            close_last < lb_last or     # Lose LB support
            lb_last < lb_prev or        # LB turns down
            fvg_long_break              # Structural break
        )

        # Exit short: lose LB resistance OR break FVG high
        if lb_bull is not None:
            ref_high = lb_bull["high"]
            fvg_short_break = close_last > ref_high
        else:
            fvg_short_break = False

        exit_short = (
            close_last > lb_last or     # Lose LB resistance
            lb_last > lb_prev or        # LB turns up
            fvg_short_break             # Structural break
        )

        # -------------------------------------------------
        # 4) WRITE SIGNALS
        # -------------------------------------------------
        idx = df_slice.index[i]
        df_slice.loc[idx, "long_entry_sig"]  = long_entry
        df_slice.loc[idx, "short_entry_sig"] = short_entry
        df_slice.loc[idx, "exit_long_sig"]   = exit_long
        df_slice.loc[idx, "exit_short_sig"]  = exit_short

    return df_slice

# ---------------------------------------------------------
# UI — TIMEFRAME, DATA LOADING, WINDOW MANAGEMENT
# ---------------------------------------------------------

st.sidebar.header("Settings")

# --- BASIC INPUTS ---
ticker = st.sidebar.text_input("Ticker", "AAPL")
first_load = "initialized" not in st.session_state
    
tf = st.sidebar.selectbox(
    "Timeframe",
    ["4H", "1D", "1W", "1M"],
    index=2
)

step = st.sidebar.number_input(
    "Slice Step",
    min_value=1,
    max_value=50,
    value=5,
    step=1
)

# --- SIGNAL TOGGLES ---
glong  = st.sidebar.checkbox("Show Long Entries", value=False)
gshort = st.sidebar.checkbox("Show Short Entries", value=False)
elong  = st.sidebar.checkbox("Show Long Exits", value=False)
eshort = st.sidebar.checkbox("Show Short Exits", value=False)

# --- TIMEFRAME CONFIG ---
TF_CONFIG = {
    "4H": {"days": 180, "interval": "4h"},
    "1D": {"days": 365, "interval": "1d"},
    "1W": {"days": 700, "interval": "1wk"},
    "1M": {"days": 365 * 5, "interval": "1mo"},
}

cfg = TF_CONFIG[tf]
today = datetime.today()
start_date = today - timedelta(days=cfg["days"])
interval = cfg["interval"]

# --- LOAD DATA ---
df = load_data(ticker, start_date, interval)
zones = detect_fvg_zones(df)
df = precompute_signals(df)

if first_load:
    st.session_state.window_start_idx = 0
    st.session_state.window_end_idx = len(df) - 1
    st.session_state.initialized = True
    
if df is None or df.empty:
    st.error("No data found.")
    st.stop()
    
# --- SESSION STATE INIT ---
if "last_tf" not in st.session_state:
    st.session_state.last_tf = tf

if "window_start_idx" not in st.session_state:
    st.session_state.window_start_idx = 0

if "window_end_idx" not in st.session_state:
    st.session_state.window_end_idx = len(df) - 1

# --- RESET WINDOW ON TF CHANGE ---
if st.session_state.last_tf != tf:
    st.session_state.window_start_idx = 0
    st.session_state.window_end_idx = len(df) - 1
    st.session_state.last_tf = tf

# Initialize persistent regime references
if "bull_ref" not in st.session_state:
    st.session_state.bull_ref = None

if "bear_ref" not in st.session_state:
    st.session_state.bear_ref = None

# --- DATA SLICING ---
start_idx = st.session_state.window_start_idx
end_idx   = st.session_state.window_end_idx
start_idx = max(0, min(start_idx, len(df) - 1))
end_idx   = max(0, min(end_idx, len(df) - 1))

if first_load:
    df_slice = df.copy()
else:
    df_slice = df.iloc[start_idx : end_idx + 1 : 1]

df_sig = df_slice.copy()

# Entries are truthful?
long_events = df_sig[["long_entry_sig", "exit_long_sig"]].any(axis=1)
if long_events.any():
    last_long_idx = df_sig[long_events].index[-1]
    last_long_row = df_sig.loc[last_long_idx]
    long_active = bool(last_long_row["long_entry_sig"]) and not bool(last_long_row["exit_long_sig"])
else:
    long_active = False

short_events = df_sig[["short_entry_sig", "exit_short_sig"]].any(axis=1)
if short_events.any():
    last_short_idx = df_sig[short_events].index[-1]
    last_short_row = df_sig.loc[last_short_idx]
    short_active = bool(last_short_row["short_entry_sig"]) and not bool(last_short_row["exit_short_sig"])
else:
    short_active = False


last = df_slice.iloc[-1]

last = df_slice.iloc[-1]
long_entry  = bool(last["long_entry_sig"])
short_entry = bool(last["short_entry_sig"])
exit_long   = bool(last["exit_long_sig"])
exit_short  = bool(last["exit_short_sig"])
c1, c2, c3, c4 = st.columns(4)

with c1:
    if long_entry:
        st.success("📈 LONG ENTRY/HOLD")
    elif long_active:
        st.info("🟢 LONG ACTIVE")
    else:
        st.info("Sideline")

with c2:
    if exit_long and long_active:
        st.warning("🟡 EXIT LONG")
    else:
        st.info("—")

with c3:
    if short_entry:
        st.error("📉 SHORT ENTRY/HOLD")
    elif short_active:
        st.info("🔴 SHORT ACTIVE")
    else:
        st.info("Sideline")

with c4:
    if exit_short and short_active:
        st.warning("❌ EXIT SHORT")
    else:
        st.info("—")
        
# --- NAVIGATION BUTTONS ---
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("⬅️ Previous"):
        st.session_state.window_end_idx = max(1, st.session_state.window_end_idx - step)

with col2:
    if st.button("Next ➡️"):
        st.session_state.window_end_idx = min(len(df) - 1, st.session_state.window_end_idx + step)
        
with col3:
    if len(df_slice) > 0:
        st.write(f"Data from **{df_slice.index[0].date()} → {df_slice.index[-1].date()}**")
    else:
        st.write("Visible Window: —")

visible_zones = [z for z in zones if start_idx <= z.start_idx <= end_idx]
# ---------------------------------------------------------
# DRAW CHART
# ---------------------------------------------------------

fig = plotchart(
    df_slice,
    visible_zones,
    title=f"{ticker} — {tf} SMC FVG Regime View",
    glong=glong,
    gshort=gshort,
    elong=elong,
    eshort=eshort
)
st.pyplot(fig)
