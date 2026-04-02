# =========================================================
# BLOCK 1 — IMPORTS + HELPERS + DATA LOADER
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ---------------------------------------------------------
# DATA LOADER
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data(ticker, start_date, interval):
    try:
        df = yf.download(ticker, start=start_date, interval=interval)
        if df is None or df.empty:
            return None
        df = df.rename(columns=str.lower)
        df.dropna(inplace=True)
        return df
    except Exception:
        return None

# ---------------------------------------------------------
# BASIC INDICATORS
# ---------------------------------------------------------

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def rsi(series, length=14):
    delta = series.diff()

    # Gains (positive changes) and losses (negative changes)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def atr(df, length=14):
    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift()

    tr = pd.concat([
        (high - low),
        (high - close_prev).abs(),
        (low - close_prev).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(length).mean()


# ---------------------------------------------------------
# SWING POINTS
# ---------------------------------------------------------

def pivot_high(df, left=20, right=5):
    highs = df["high"]
    return highs[(highs.shift(right) == highs.rolling(left+right+1).max())]

def pivot_low(df, left=20, right=5):
    lows = df["low"]
    return lows[(lows.shift(right) == lows.rolling(left+right+1).min())]

# ---------------------------------------------------------
# ZONE OBJECT
# ---------------------------------------------------------

class Zone:
    def __init__(self, top, bottom, start_idx, is_bull, is_ob, color):
        self.top = top
        self.bottom = bottom
        self.start_idx = start_idx
        self.is_bull = is_bull
        self.is_ob = is_ob
        self.color = color
        self.is_mitigated = False
        self.taps = 0

    def __repr__(self):
        return f"Zone({self.top}, {self.bottom}, bull={self.is_bull}, ob={self.is_ob}, mitigated={self.is_mitigated})"

# ---------------------------------------------------------
# COLOR HELPERS
# ---------------------------------------------------------

def rgba(hex_color, alpha):
    """Convert hex to RGBA tuple for matplotlib."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255
    g = int(hex_color[2:4], 16) / 255
    b = int(hex_color[4:6], 16) / 255
    return (r, g, b, alpha)

# =========================================================
# BLOCK 2 — FULL ENGINE (ZONES, STRUCTURE, PATTERNS, SIGNALS)
# =========================================================

# This block ports the core of your Pine Script logic into Python.
# It builds:
#   - FVGs (3-candle, gap-aware)
#   - OBs (displacement + gap-based)
#   - Zone merging + mitigation
#   - Market structure (BOS / CHoCH)
#   - Candlestick patterns (core set)
#   - Momentum + SMC bias
#   - Unified "info" dict for plotting / regime logic
#
# Main public functions:
#   apply_pinescript_logic(df) -> (df, zones, info)
#   get_last_broken_fvg(df_or_slice) -> (last_broken_bear, last_broken_bull)


# ---------------------------------------------------------
# ZONE ENGINE HELPERS
# ---------------------------------------------------------

def zones_overlap(top1, bottom1, top2, bottom2):
    return not (bottom1 > top2 or bottom2 > top1)


def merge_overlapping_zones(zones):
    """Merge overlapping zones, keeping the larger one (by height)."""
    if len(zones) <= 1:
        return zones

    zones_sorted = sorted(zones, key=lambda z: z.start_idx)
    merged = [zones_sorted[0]]

    for z in zones_sorted[1:]:
        last = merged[-1]
        if zones_overlap(last.top, last.bottom, z.top, z.bottom):
            size1 = abs(last.top - last.bottom)
            size2 = abs(z.top - z.bottom)
            if size2 > size1:
                merged[-1] = z
        else:
            merged.append(z)

    return merged


def add_zone(zones, top, bottom, is_bull, is_ob, start_idx, color, cond=True):
    if not cond:
        return
    if top is None or bottom is None:
        return
    if np.isnan(top) or np.isnan(bottom):
        return
    if top == bottom:
        return
    zones.append(Zone(top, bottom, start_idx, is_bull, is_ob, color))


# ---------------------------------------------------------
# PATTERN ENGINE (CORE SUBSET)
# ---------------------------------------------------------

def detect_candlestick_patterns(df, lb_crv, body_thresh=0.10,
                                wick_thresh_high=0.55, wick_thresh_low=0.15):
    """
    Returns:
        pattern_name: Series[str]
        pattern_bull: Series[bool]  (True if bullish pattern)
    """
    o = df["open"]
    c = df["close"]
    h = df["high"]
    l = df["low"]

    body = (c - o).abs()
    crange = h - l
    wick_high = h - np.maximum(o, c)
    wick_low = np.minimum(o, c) - l

    # Basic patterns
    bull_engulf = (
        (c > o) &
        (c.shift(1) < o.shift(1)) &
        (c >= h.shift(1)) &
        (o <= l.shift(1))
    )

    bear_engulf = (
        (c < o) &
        (c.shift(1) > o.shift(1)) &
        (c <= l.shift(1)) &
        (o >= h.shift(1))
    )

    doji = (
        (body <= crange * body_thresh) &
        (wick_high <= crange * 0.45) &
        (wick_low <= crange * 0.45) &
        ((wick_high - wick_low).abs() <= crange * 0.15)
    )

    # Gravestone / Dragonfly with LB filter
    gravestone = (
        doji &
        (wick_high >= crange * wick_thresh_high) &
        (wick_low <= crange * wick_thresh_low) &
        (c > lb_crv * 1.02)
    )

    dragonfly = (
        doji &
        (wick_low >= crange * wick_thresh_high) &
        (wick_high <= crange * wick_thresh_low) &
        (c < lb_crv * 0.98)
    )

    # Morning / Evening star (simplified)
    body1 = (c.shift(1) - o.shift(1)).abs()
    body2 = (c.shift(2) - o.shift(2)).abs()

    is_morning = (
        (c.shift(2) < o.shift(2)) &
        (body1 < body2 * 0.4) &
        (c > (o.shift(2) + c.shift(2)) / 2)
    )

    is_evening = (
        (c.shift(2) > o.shift(2)) &
        (body1 < body2 * 0.4) &
        (c < (o.shift(2) + c.shift(2)) / 2)
    )

    # Piercing / Dark Cloud
    bull_pierce = (
        (c.shift(1) < o.shift(1)) &
        (c > (o.shift(1) + c.shift(1)) / 2)
    )

    bear_dark = (
        (c.shift(1) > o.shift(1)) &
        (c < (o.shift(1) + c.shift(1)) / 2)
    )

    # Tweezer top/bottom
    tweezer_bot = (l - l.shift(1)).abs() < df["close"].diff().abs().median() * 0.1
    tweezer_top = (h - h.shift(1)).abs() < df["close"].diff().abs().median() * 0.1

    # Hammer / Shooting star
    is_hammer = (wick_low > body * 2) & (wick_high < body * 0.5)
    is_star = (wick_high > body * 2) & (wick_low < body * 0.5)

    # Priority-based assignment (similar to Pine logic)
    pattern_name = pd.Series("None", index=df.index, dtype="object")
    pattern_bull = pd.Series(False, index=df.index, dtype=bool)

    # Morning Star
    mask = is_morning & (df["close"] < lb_crv * 0.98)
    pattern_name[mask] = "Morning Star"
    pattern_bull[mask] = True

    # Evening Star
    mask = is_evening & (df["close"] > lb_crv * 1.02)
    pattern_name[mask] = "Evening Star"
    pattern_bull[mask] = False

    # Bull Engulfing
    mask = bull_engulf & (df["close"] < lb_crv * 0.98) & (pattern_name == "None")
    pattern_name[mask] = "Bull Engulfing"
    pattern_bull[mask] = True

    # Bear Engulfing
    mask = bear_engulf & (df["close"] > lb_crv * 1.02) & (pattern_name == "None")
    pattern_name[mask] = "Bear Engulfing"
    pattern_bull[mask] = False

    # Piercing
    mask = bull_pierce & (df["close"] < lb_crv * 0.98) & (pattern_name == "None")
    pattern_name[mask] = "Piercing"
    pattern_bull[mask] = True

    # Dark Cloud
    mask = bear_dark & (df["close"] > lb_crv * 1.02) & (pattern_name == "None")
    pattern_name[mask] = "Dark Cloud"
    pattern_bull[mask] = False

    # Tweezer Bottom
    mask = tweezer_bot & (df["close"] < lb_crv * 0.98) & (pattern_name == "None")
    pattern_name[mask] = "Tweezer Bottom"
    pattern_bull[mask] = True

    # Tweezer Top
    mask = tweezer_top & (df["close"] > lb_crv * 1.02) & (pattern_name == "None")
    pattern_name[mask] = "Tweezer Top"
    pattern_bull[mask] = False

    # Hammer
    mask = is_hammer & (df["close"] < lb_crv * 0.98) & (pattern_name == "None")
    pattern_name[mask] = "Hammer"
    pattern_bull[mask] = True

    # Shooting Star
    mask = is_star & (df["close"] > lb_crv * 1.02) & (pattern_name == "None")
    pattern_name[mask] = "Shooting Star"
    pattern_bull[mask] = False

    # Gravestone / Dragonfly
    mask = gravestone & (pattern_name == "None")
    pattern_name[mask] = "Gravestone"
    pattern_bull[mask] = False

    mask = dragonfly & (pattern_name == "None")
    pattern_name[mask] = "Dragonfly"
    pattern_bull[mask] = True

    # Neutral Doji
    mask = doji & (pattern_name == "None")
    pattern_name[mask] = np.where(df["close"] > df["open"], "Bull Doji", "Bear Doji")
    pattern_bull[mask] = df["close"][mask] > df["open"][mask]

    return pattern_name, pattern_bull


# ---------------------------------------------------------
# MARKET STRUCTURE (BOS / CHoCH)
# ---------------------------------------------------------
def compute_structure(df, swing_left=20, swing_right=5, bos_ext_bars=20):
    highs = df["high"].to_numpy().flatten()
    lows  = df["low"].to_numpy().flatten()
    closes = df["close"].to_numpy().flatten()
    n = len(df)

    # Swing points as arrays of floats (NaN where no swing)
    sw_hi = np.full(n, np.nan)
    sw_lo = np.full(n, np.nan)

    for i in range(swing_left, n - swing_right):
        window = highs[i - swing_left : i + swing_right + 1]
        if highs[i] == window.max():
            sw_hi[i] = highs[i]
        if lows[i] == window.min():
            sw_lo[i] = lows[i]

    bos_up = np.zeros(n, dtype=bool)
    bos_dn = np.zeros(n, dtype=bool)
    trend  = np.zeros(n, dtype=bool)

    last_hi = np.nan
    last_lo = np.nan
    is_uptrend = False

    for i in range(n):
        if not np.isnan(sw_hi[i]):
            last_hi = sw_hi[i]
        if not np.isnan(sw_lo[i]):
            last_lo = sw_lo[i]

        close_i = closes[i]

        # BOS up
        if not np.isnan(last_hi) and close_i > last_hi:
            bos_up[i] = True
            is_uptrend = True
            last_hi = np.nan

        # BOS down
        if not np.isnan(last_lo) and close_i < last_lo:
            bos_dn[i] = True
            is_uptrend = False
            last_lo = np.nan

        trend[i] = is_uptrend

    return (
        pd.Series(bos_up, index=df.index),
        pd.Series(bos_dn, index=df.index),
        pd.Series(trend, index=df.index),
    )

# ---------------------------------------------------------
# MAIN ENGINE: APPLY PINE LOGIC TO DF
# ---------------------------------------------------------

def apply_pinescript_logic(df_raw):
    """
    Core function that applies the Pine Script logic to a pandas DataFrame.

    Returns:
        df:   enriched DataFrame (indicators, structure, patterns, etc.)
        zones: list[Zone] (all zones, with mitigation flags)
        info: dict with extra state useful for plotting / regime logic
    """
    df = df_raw.copy()

    # --- INDICATORS ---
    len_ema_short = 20
    len_ema_med = 50
    len_ema_long = 200
    lblen = 10
    rsi_len = 14
    rsi_ema_len = 14
    swing_left = 20
    swing_right = 5
    max_zone_age = 100
    fail_window = 5

    df["ema20"] = ema(df["close"], len_ema_short)
    df["ema50"] = ema(df["close"], len_ema_med)
    df["ema200"] = ema(df["close"], len_ema_long)

    # --- LB CURVE (FINAL FIXED VERSION) ---
    close_arr = df["close"].to_numpy().flatten()
    high_arr  = df["high"].to_numpy().flatten()
    low_arr   = df["low"].to_numpy().flatten()
    
    highest_lb = pd.Series(close_arr).rolling(lblen).max().to_numpy()
    lowest_lb  = pd.Series(close_arr).rolling(lblen).min().to_numpy()
    
    lb_new = close_arr.copy()
    
    for i in range(1, len(close_arr)):
        prev_high = highest_lb[i-1]
        prev_low  = lowest_lb[i-1]
    
        if np.isnan(prev_high) or np.isnan(prev_low):
            lb_new[i] = lb_new[i-1]
            continue
    
        close_i = close_arr[i]
    
        if close_i > prev_high:
            lb_new[i] = (high_arr[i] + close_i) / 2
        elif close_i < prev_low:
            lb_new[i] = (low_arr[i] + close_i) / 2
        else:
            lb_new[i] = lb_new[i-1]
    
    df["lb"] = lb_new
    df["lb_crv"] = pd.Series(lb_new).ewm(span=lblen, adjust=False).mean().to_numpy()

    # RSI + EMA of RSI
    df["rsi"] = rsi(df["close"], rsi_len)
    df["rsi_ema"] = ema(df["rsi"], rsi_ema_len)

    # ATR
    df["atr"] = atr(df, 14)
    atr_mult = 0.2
    df["offset2"] = df["atr"] * atr_mult

    # --- STRUCTURE (BOS / CHoCH) ---
    bos_up, bos_dn, is_uptrend = compute_structure(df, swing_left, swing_right)
    df["bos_up"] = bos_up
    df["bos_dn"] = bos_dn
    df["is_uptrend"] = is_uptrend

    # --- PATTERNS ---
    pattern_name, pattern_bull = detect_candlestick_patterns(df, df["lb_crv"])
    df["pattern_name"] = pattern_name
    df["pattern_bull"] = pattern_bull

    # --- MOMENTUM + SMC ---
    df["ema_bullish"] = df["ema20"] > df["ema50"]
    df["ema_bearish"] = df["ema20"] < df["ema50"]

    df["mom_bullish"] = ((df["rsi"] >= 50) & (df["rsi"] > df["rsi_ema"])) | (df["close"] > df["lb_crv"] * 1.02)
    df["mom_bearish"] = ((df["rsi"] <= 44) & (df["rsi"] < df["rsi_ema"])) | (df["close"] < df["lb_crv"] * 0.98)

    smc_bullish = []
    smc_bearish = []
    bull_state = False
    bear_state = False
    for i in range(len(df)):
        if bos_up.iloc[i]:
            bull_state = True
            bear_state = False
        if bos_dn.iloc[i]:
            bull_state = False
            bear_state = True
        smc_bullish.append(bull_state)
        smc_bearish.append(bear_state)

    df["smc_bullish"] = smc_bullish
    df["smc_bearish"] = smc_bearish

    # --- ZONES (FVG + OB) ---
    zones = []
    max_taps = 2

    # Colors (approx from Pine)
    fvg_bull_col = rgba("35aa18", 0.35)
    fvg_bear_col = rgba("da1313", 0.35)
    ob_bull_col = rgba("008950", 0.35)
    ob_bear_col = rgba("883f0e", 0.35)
    mitigated_col = rgba("808080", 0.05)

    idx = df.index

    # We'll track per-bar zone awareness flags
    has_bull_ob = []
    has_bear_ob = []
    has_bull_fvg = []
    has_bear_fvg = []
    inside_zone = []

    for i in range(len(df)):
        if i < 2:
            has_bull_ob.append(False)
            has_bear_ob.append(False)
            has_bull_fvg.append(False)
            has_bear_fvg.append(False)
            inside_zone.append(False)
            continue

        t = idx[i]
        row = df.iloc[i]
        atr_fvg = df["atr"].iloc[i]
        min_gap = atr_fvg * 0.1 if not np.isnan(atr_fvg) else 0

        # 3-candle FVG
        # Bullish FVG: low(i) > high(i-2) + min_gap
        low_i = df["low"].iloc[i]
        high_i = df["high"].iloc[i]
        low_2 = df["low"].iloc[i-2]
        high_2 = df["high"].iloc[i-2]

        fvg_up3 = low_i > high_2 + min_gap
        fvg_dn3 = high_i < low_2 - min_gap

        if fvg_up3:
            add_zone(zones, top=high_2, bottom=low_i,
                     is_bull=True, is_ob=False,
                     start_idx=i, color=fvg_bull_col, cond=True)

        if fvg_dn3:
            add_zone(zones, top=high_i, bottom=low_2,
                     is_bull=False, is_ob=False,
                     start_idx=i, color=fvg_bear_col, cond=True)

        # 3-candle OB (displacement)
        close_1 = df["close"].iloc[i-1])
        open_1 = float(df["open"].iloc[i-1]
        high_1 = float(df["high"].iloc[i-1])
        low_1 = float(df["low"].iloc[i-1])
        high_2 = float(df["high"].iloc[i-2])
        low_2 = float(df["low"].iloc[i-2])
        close_i = float(df["close"].iloc[i])
        open_i  = float(df["open"].iloc[i])
        high_1  = float(df["high"].iloc[i-1])
        low_1   = float(df["low"].iloc[i-1])


        displacement_up = (close_i > high_1) and (close_i > open_i)
        displacement_dn = (close_i < low_1) and (close_i < open_i)

        bull_ob3 = displacement_up and (low_1 < low_2)
        bear_ob3 = displacement_dn and (high_1 > high_2)

        if bull_ob3:
            add_zone(zones, top=high_1, bottom=low_1,
                     is_bull=True, is_ob=True,
                     start_idx=i, color=ob_bull_col, cond=True)

        if bear_ob3:
            add_zone(zones, top=high_1, bottom=low_1,
                     is_bull=False, is_ob=True,
                     start_idx=i, color=ob_bear_col, cond=True)

        # Gap-based OBs
        open_i = df["open"].iloc[i]
        high_prev = df["high"].iloc[i-1]
        low_prev = df["low"].iloc[i-1]

        gap_up_ob = (open_i > high_prev) and (close_i > open_i)
        gap_dn_ob = (open_i < low_prev) and (close_i < open_i)

        if gap_up_ob:
            add_zone(zones, top=open_i, bottom=low_prev,
                     is_bull=True, is_ob=True,
                     start_idx=i, color=ob_bull_col, cond=True)

        if gap_dn_ob:
            add_zone(zones, top=high_prev, bottom=open_i,
                     is_bull=False, is_ob=True,
                     start_idx=i, color=ob_bear_col, cond=True)

        # Merge overlapping zones (keep larger)
        zones = merge_overlapping_zones(zones)

        # Zone engine: mitigation, age, visibility flags
        bull_ob_flag = False
        bear_ob_flag = False
        bull_fvg_flag = False
        bear_fvg_flag = False
        inside_flag = False

        # Iterate backwards so we can safely remove
        j = len(zones) - 1
        while j >= 0:
            z = zones[j]
            age = i - z.start_idx

            # Failure check (5-bar rule)
            failed = False
            if age <= fail_window:
                if z.is_bull and close_i < z.bottom and close_1 < z.bottom:
                    failed = True
                if (not z.is_bull) and close_i > z.top and close_1 > z.top:
                    failed = True

            # Mitigation / taps
            if not z.is_mitigated:
                # Count taps if price touches zone
                if (high_i > z.bottom) and (low_i < z.top):
                    z.taps += 1

                bull_broken = z.is_bull and (close_i < z.bottom)
                bear_broken = (not z.is_bull) and (close_i > z.top)

                if bull_broken or bear_broken or (z.taps > 5):
                    z.is_mitigated = True

            # Delete old / failed
            if age > max_zone_age or failed:
                zones.pop(j)
                j -= 1
                continue

            # Zone awareness flags
            if z.is_ob and z.is_bull:
                bull_ob_flag = True
            if z.is_ob and not z.is_bull:
                bear_ob_flag = True
            if (not z.is_ob) and z.is_bull:
                bull_fvg_flag = True
            if (not z.is_ob) and (not z.is_bull):
                bear_fvg_flag = True

            if (close_i < z.top) and (close_i > z.bottom):
                inside_flag = True

            j -= 1

        has_bull_ob.append(bull_ob_flag)
        has_bear_ob.append(bear_ob_flag)
        has_bull_fvg.append(bull_fvg_flag)
        has_bear_fvg.append(bear_fvg_flag)
        inside_zone.append(inside_flag)

    df["has_bull_ob"] = has_bull_ob
    df["has_bear_ob"] = has_bear_ob
    df["has_bull_fvg"] = has_bull_fvg
    df["has_bear_fvg"] = has_bear_fvg
    df["inside_zone"] = inside_zone

    # FVG interpretation (compact)
    fvg_state = []
    for i in range(len(df)):
        if df["has_bull_fvg"].iloc[i] and not df["has_bear_fvg"].iloc[i]:
            fvg_state.append("Bull FVG↓ → Pullback then up")
        elif df["has_bear_fvg"].iloc[i] and not df["has_bull_fvg"].iloc[i]:
            fvg_state.append("Bear FVG↑ → Bear rally then down")
        elif df["has_bull_fvg"].iloc[i] and df["has_bear_fvg"].iloc[i]:
            fvg_state.append("FVG↑↓ → Range / Sweep")
        else:
            fvg_state.append("No FVG Bias")
    df["fvg_state"] = fvg_state

    # Info dict for plotting / regime logic
    info = {
        "pattern_name": df["pattern_name"],
        "pattern_bull": df["pattern_bull"],
        "smc_bullish": df["smc_bullish"],
        "smc_bearish": df["smc_bearish"],
        "bos_up": df["bos_up"],
        "bos_dn": df["bos_dn"],
        "is_uptrend": df["is_uptrend"],
        "fvg_state": df["fvg_state"],
    }

    return df, zones, info


# ---------------------------------------------------------
# LAST BROKEN FVG (FOR REGIME ENGINE)
# ---------------------------------------------------------

def get_last_broken_fvg(df_slice):
    """
    Approximation of "last broken FVG" logic:
    - last_broken_bear: last bearish FVG that price has closed above
    - last_broken_bull: last bullish FVG that price has closed below

    Returns:
        last_broken_bear: dict or None  (keys: 'low', 'high', 'index')
        last_broken_bull: dict or None
    """
    last_broken_bear = None
    last_broken_bull = None

    if "has_bear_fvg" not in df_slice.columns or "has_bull_fvg" not in df_slice.columns:
        return None, None

    closes = df_slice["close"]
    highs = df_slice["high"]
    lows = df_slice["low"]

    # We approximate FVG bounds using recent candles where flags are true
    for i in range(2, len(df_slice)):
        idx = df_slice.index[i]

        # Bearish FVG: high(i) < low(i-2)
        if df_slice["has_bear_fvg"].iloc[i]:
            fvg_high = highs.iloc[i]
            fvg_low = lows.iloc[i-2]
            # Broken if close > high
            if closes.iloc[i] > fvg_high:
                last_broken_bear = {
                    "index": idx,
                    "high": float(fvg_high),
                    "low": float(fvg_low),
                }

        # Bullish FVG: low(i) > high(i-2)
        if df_slice["has_bull_fvg"].iloc[i]:
            fvg_low = lows.iloc[i]
            fvg_high = highs.iloc[i-2]
            # Broken if close < low
            if closes.iloc[i] < fvg_low:
                last_broken_bull = {
                    "index": idx,
                    "high": float(fvg_high),
                    "low": float(fvg_low),
                }

    return last_broken_bear, last_broken_bull

# =========================================================
# BLOCK 3 — PLOTTING ENGINE (MATPLOTLIB)
# =========================================================

def plotchart(df_slice, zones, info, title="SMC View",
              exit_long=False, exit_short=False):

    fig, ax = plt.subplots(figsize=(14, 7))

    # -----------------------------------------------------
    # CANDLESTICKS
    # -----------------------------------------------------
    for i in range(len(df_slice)):
        idx = df_slice.index[i]
        o = df_slice["open"].iloc[i]
        c = df_slice["close"].iloc[i]
        h = df_slice["high"].iloc[i]
        l = df_slice["low"].iloc[i]

        color = "green" if c >= o else "red"
        ax.plot([i, i], [l, h], color=color, linewidth=1)
        ax.add_patch(
            plt.Rectangle(
                (i - 0.3, min(o, c)),
                0.6,
                abs(c - o),
                color=color,
                alpha=0.8
            )
        )

    # -----------------------------------------------------
    # EMAs
    # -----------------------------------------------------
    ax.plot(df_slice["ema20"].values, color="yellow", linewidth=1, label="EMA20")
    ax.plot(df_slice["ema50"].values, color="orange", linewidth=1.2, label="EMA50")
    ax.plot(df_slice["ema200"].values, color="purple", linewidth=1.2, label="EMA200")

    # -----------------------------------------------------
    # ACTIVE ZONES ONLY
    # -----------------------------------------------------
    for z in zones:
        if z.is_mitigated:
            continue  # only active zones

        # Only draw if zone is inside visible window
        if z.start_idx < df_slice.index[0] or z.start_idx > df_slice.index[-1]:
            continue

        # Convert absolute index to slice-relative index
        rel_idx = df_slice.index.get_loc(z.start_idx)

        height = z.top - z.bottom
        ax.add_patch(
            plt.Rectangle(
                (rel_idx, z.bottom),
                width=len(df_slice) - rel_idx,
                height=height,
                color=z.color,
                alpha=0.35,
                linewidth=1.2,
                edgecolor=z.color
            )
        )

    # -----------------------------------------------------
    # BOS / CHoCH LABELS
    # -----------------------------------------------------
    for i in range(len(df_slice)):
        if info["bos_up"].iloc[i]:
            ax.text(i, df_slice["high"].iloc[i],
                    "BOS↑", color="lime", fontsize=8, ha="center")

        if info["bos_dn"].iloc[i]:
            ax.text(i, df_slice["low"].iloc[i],
                    "BOS↓", color="red", fontsize=8, ha="center")

    # -----------------------------------------------------
    # PATTERN LABELS
    # -----------------------------------------------------
    for i in range(len(df_slice)):
        name = info["pattern_name"].iloc[i]
        if name != "None":
            y = df_slice["low"].iloc[i] if info["pattern_bull"].iloc[i] else df_slice["high"].iloc[i]
            ax.text(i, y,
                    name,
                    fontsize=7,
                    color="yellow" if info["pattern_bull"].iloc[i] else "white",
                    ha="center")

    # -----------------------------------------------------
    # ENTRY / EXIT MARKERS
    # -----------------------------------------------------
    if exit_long:
        ax.text(len(df_slice)-1, df_slice["close"].iloc[-1],
                "EXIT LONG", color="yellow", fontsize=10)

    if exit_short:
        ax.text(len(df_slice)-1, df_slice["close"].iloc[-1],
                "EXIT SHORT", color="yellow", fontsize=10)

    # -----------------------------------------------------
    # FINAL FORMATTING
    # -----------------------------------------------------
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-1, len(df_slice) + 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig

# =========================================================
# BLOCK 4 — STREAMLIT UI (FULL APP)
# =========================================================

st.set_page_config(page_title="SMC FVG Dashboard", layout="wide")

st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Ticker", "ASML")

tf = st.sidebar.selectbox(
    "Timeframe",
    ["4H", "1D", "1W", "1M"],
    index=2
)

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
elif tf == "1M":
    start_date = today - timedelta(days=365*2)
    interval = "1mo"

# ---------------------------------------------------------
# LOAD DATA + APPLY ENGINE
# ---------------------------------------------------------

df = load_data(ticker, start_date, interval)

if df is None or df.empty:
    st.error("No data found.")
    st.stop()

df, zones, info = apply_pinescript_logic(df)

# ---------------------------------------------------------
# WINDOW MANAGEMENT
# ---------------------------------------------------------

col1, col2, col3 = st.columns(3)

if "last_tf" not in st.session_state:
    st.session_state.last_tf = tf

if st.session_state.last_tf != tf:
    st.session_state.window_start_idx = 0
    st.session_state.window_end_idx = len(df) - 1
    st.session_state.last_tf = tf

if "window_end_idx" not in st.session_state:
    st.session_state.window_end_idx = 50

if "window_start_idx" not in st.session_state:
    st.session_state.window_start_idx = 0

# --- BUTTONS ---
if col1.button("⬅️ Previous"):
    st.session_state.window_end_idx = max(
        st.session_state.window_start_idx + 1,
        st.session_state.window_end_idx - 1
    )

if col2.button("Next ➡️"):
    st.session_state.window_end_idx = min(
        len(df) - 1,
        st.session_state.window_end_idx + 1
    )

# --- FINAL SLICE ---
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

# ---------------------------------------------------------
# ADAPTIVE REGIME ENGINE
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
# BULLISH REGIME
# -----------------------------
if last_broken_bear is not None:
    ref_low = last_broken_bear["low"]
    ref_high = last_broken_bear["high"]
    fvg_range = ref_high - ref_low

    if not st.session_state.in_long and bull_mask:
        if close_last > ref_high and bullish_candle:
            long_entry = True
        elif bullish_candle and close_last > ref_low:
            long_entry = True
        if bullish_candle and close_last > ref_low + 0.05 * fvg_range:
            long_entry = True

    if st.session_state.in_long:
        if bearish_candle:
            exit_long = True
        if close_last < ref_low:
            exit_long = True
        if not bull_mask:
            exit_long = True

# -----------------------------
# BEARISH REGIME
# -----------------------------
if last_broken_bull is not None:
    ref_low = last_broken_bull["low"]
    ref_high = last_broken_bull["high"]
    fvg_range = ref_high - ref_low

    if not st.session_state.in_short and bear_mask:
        if close_last < ref_low and bearish_candle:
            short_entry = True
        elif bearish_candle and close_last < ref_high:
            short_entry = True
        if bearish_candle and close_last < ref_high - 0.05 * fvg_range:
            short_entry = True

    if st.session_state.in_short:
        if bullish_candle:
            exit_short = True
        if close_last > ref_high:
            exit_short = True
        if not bear_mask:
            exit_short = True

# -----------------------------
# APPLY STATE CHANGES
# -----------------------------
if long_entry:
    st.session_state.in_long = True
    st.session_state.in_short = False

if short_entry:
    st.session_state.in_short = True
    st.session_state.in_long = False

if exit_long:
    st.session_state.in_long = False
    if last_broken_bull is not None:
        if close_last < last_broken_bull["low"] and bearish_candle:
            st.session_state.in_short = True

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
        st.success("📈 LONG ENTRY")
    elif st.session_state.in_long:
        st.info("🟢 LONG ACTIVE")
    else:
        st.info("—")

with c2:
    if short_entry:
        st.error("📉 SHORT ENTRY")
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

fig = plotchart(df_slice, zones, info,
                title=f"SMC FVG View — {ticker}",
                exit_long=exit_long,
                exit_short=exit_short)

st.pyplot(fig)
