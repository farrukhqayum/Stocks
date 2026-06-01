#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnchoredText
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SMART MONEY CONCEPTS", layout="wide")
st.title("📈 SMART MONEY CONCEPTS - 1D/1H")

# =============================================================================
# DATA LOADER (same as before, using yfinance)
# =============================================================================
class OptimizedDataHandler:
    @st.cache_data(ttl=300, show_spinner=False)
    def load_data(_self, ticker, start_date, interval):
        try:
            interval_map = {'1d': '1d', '1h': '1h', '4h': '1h'}
            yf_int = interval_map.get(interval, '1d')
            if interval == '4h':
                df = yf.download(ticker, start=start_date, interval='1h', progress=False, auto_adjust=False)
                if df.empty:
                    return None
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]
                df = df.resample('4H').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
            else:
                df = yf.download(ticker, start=start_date, interval=yf_int, progress=False, auto_adjust=False)
                if df.empty:
                    return None
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]
            df.columns = [c.lower() for c in df.columns]
            required = ['open','high','low','close']
            if not all(c in df.columns for c in required):
                return None
            if 'volume' not in df.columns:
                df['volume'] = 0
            for c in required + ['volume']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            max_bars = 500 if interval != '1d' else 1000
            if len(df) > max_bars:
                df = df.tail(max_bars)
            # Add indicators (identical to Pine)
            df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
            df['rsi'] = _fast_rsi(df['close'], 14)
            df['rsi_ema'] = df['rsi'].ewm(span=14, adjust=False).mean()
            df['atr'] = _fast_atr(df, 14)
            # LB curve (identical to Pine)
            df['lb_crv'] = _fast_lb_curve(df, 10)
            df = df.bfill().ffill()
            return df
        except Exception as e:
            st.error(f"Data error: {e}")
            return None

def _fast_rsi(series, length):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def _fast_atr(df, length):
    tr = pd.concat([df['high']-df['low'],
                    (df['high']-df['close'].shift()).abs(),
                    (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def _fast_lb_curve(df, lblen):
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    lb = np.zeros(len(df))
    lb[0] = close[0]
    for i in range(1, len(df)):
        start = max(0, i - lblen + 1)
        highest_lb_prev = lb[start:i].max() if start < i else lb[i-1]
        lowest_lb_prev  = lb[start:i].min() if start < i else lb[i-1]
        if close[i] > highest_lb_prev:
            lb[i] = (high[i] + close[i]) / 2
        elif close[i] < lowest_lb_prev:
            lb[i] = (low[i] + close[i]) / 2
        else:
            lb[i] = lb[i-1]
    result = pd.Series(lb, index=df.index).ewm(span=lblen, adjust=False).mean()
    result.iloc[0] = close[0]
    return result

# =============================================================================
# ZONE CLASS (exactly as in Pine)
# =============================================================================
class Zone:
    def __init__(self, top, bottom, start_bar, is_bull, is_ob, color):
        self.top = top
        self.bottom = bottom
        self.start_bar = start_bar
        self.is_bull = is_bull
        self.is_ob = is_ob
        self.base_col = color
        self.is_mitigated = False
        self.taps = 0

# =============================================================================
# PINE STATE – store all dynamic variables
# =============================================================================
class PineState:
    def __init__(self):
        self.all_zones = []
        self.smc_bullish = False
        self.smc_bearish = False
        self.mom_bullish = False
        self.mom_bearish = False
        self.pattern_bullish = False
        self.pattern_rejected = False
        self.pattern_stored_low = None
        self.pattern_stored_high = None
        self.pattern_stored_bar = None
        self.pattern_status = "None"
        self.last_swing_high = None
        self.last_swing_low = None
        self.prev_swing_high = None
        self.prev_swing_low = None
        self.last_hi = None
        self.last_lo = None
        self.last_hi_idx = None
        self.last_lo_idx = None
        self.is_uptrend = None
        self.smc_early_bull = False
        self.smc_early_bear = False
        self.last_pattern = "None"
        self.pattern_bar = 0
        self.pattern_invalidated = False
        self.last_pattern_bar = 0
        self.pattern_stored_close = None
        self.pattern_name = "None"
        self.pattern_is_bullish = False
        self.pattern_stored_high_val = None
        self.pattern_stored_low_val = None
        self.pattern_stored_bar_idx = None
        self.bt_lbl = None
        self.bt_idx = None
        self.bt_hi = None
        self.bt_lo = None
        self.bt_bull = False
        self.last_sweep_bar = None
        self.ssl_rejected = False
        self.bsl_rejected = False
        self.ssl_rejection_bar = None
        self.bsl_rejection_bar = None
        self.active_ssl = False
        self.active_ssl_bar = None
        self.active_bsl = False
        self.active_bsl_bar = None
        self.last_turning_bar = None
        self.last_turning_reason = ""
        self.last_turning_price = None
        self.last_turning_actual_price = None
        self.last_turning_color = None
        self.last_turning_style = "down"
        self.last_bos_up_bar = None
        self.last_bos_up_price = None
        self.last_bos_dn_bar = None
        self.last_bos_dn_price = None
        self.in_long = False
        self.in_short = False
        self.entry_stop_long = None
        self.entry_stop_short = None
        self.entry_price_long = None
        self.entry_price_short = None
        self.active_long_sl = None
        self.active_long_tp = None
        self.active_short_sl = None
        self.active_short_tp = None
        self.trade_start_bar = -1
        self.inside_zone_prev = False
        self.last_zone_bullish = False
        self.last_zone_bearish = False
        self.last_close = None
        self.last_lb_crv = None
        self.last_rsi = None
        self.last_rsi_ema = None
        self.last_ema20 = None
        self.last_ema50 = None
        self.last_high = None
        self.last_low = None
        self.last_open = None
        self.bos_up_list = []
        self.bos_dn_list = []
        self.cho_up_list = []
        self.cho_dn_list = []
        self.turning_points = []

# =============================================================================
# HELPER: add_zone (same as Pine)
# =============================================================================
def add_zone(zones, top, bottom, start_bar, is_bull, is_ob, condition):
    if condition and top is not None and bottom is not None and top != bottom:
        col = "#35aa18" if is_bull and not is_ob else "#da1313" if not is_bull and not is_ob else "#008950" if is_bull else "#883f0e"
        zones.append(Zone(top, bottom, start_bar, is_bull, is_ob, col))

# =============================================================================
# MAIN PROCESSING (one bar at a time)
# =============================================================================
def process_bar(df, i, state, params):
    """Process one bar exactly as Pine Script does"""
    # current values
    o = df['open'].iloc[i]
    h = df['high'].iloc[i]
    l = df['low'].iloc[i]
    c = df['close'].iloc[i]
    v = df['volume'].iloc[i] if 'volume' in df.columns else 0
    ema20 = df['ema20'].iloc[i]
    ema50 = df['ema50'].iloc[i]
    ema200 = df['ema200'].iloc[i]
    rsi_val = df['rsi'].iloc[i]
    rsi_ema = df['rsi_ema'].iloc[i]
    atr = df['atr'].iloc[i]
    lb_crv = df['lb_crv'].iloc[i]

    # previous values
    if i > 0:
        o_prev = df['open'].iloc[i-1]
        h_prev = df['high'].iloc[i-1]
        l_prev = df['low'].iloc[i-1]
        c_prev = df['close'].iloc[i-1]
        v_prev = df['volume'].iloc[i-1] if 'volume' in df.columns else 0
        ema20_prev = df['ema20'].iloc[i-1]
        ema50_prev = df['ema50'].iloc[i-1]
        ema200_prev = df['ema200'].iloc[i-1]
        rsi_prev = df['rsi'].iloc[i-1]
        rsi_ema_prev = df['rsi_ema'].iloc[i-1]
        atr_prev = df['atr'].iloc[i-1]
        lb_crv_prev = df['lb_crv'].iloc[i-1]
    else:
        o_prev = o; h_prev = h; l_prev = l; c_prev = c
        v_prev = v; ema20_prev = ema20; ema50_prev = ema50; ema200_prev = ema200
        rsi_prev = rsi_val; rsi_ema_prev = rsi_ema; atr_prev = atr; lb_crv_prev = lb_crv

    # Volume SMA
    vol5 = df['volume'].rolling(5).mean().iloc[i] if i>=4 else v
    vol_confirm_bull = v > vol5
    vol_confirm_bear = v > vol5
    fvg_vol_ok = v > vol5 * 0.6

    # ---------------------- Swings (Pivots) ----------------------
    swing_l = params.get('swing_l', 10)
    swing_r = params.get('swing_r', 5)
    if i >= swing_l and i < len(df)-swing_r:
        # pivot high
        is_high = all(h > df['high'].iloc[i-k] for k in range(1, swing_l+1)) and all(h > df['high'].iloc[i+k] for k in range(1, swing_r+1))
        if is_high:
            state.prev_swing_high = state.last_swing_high
            state.last_swing_high = h
            state.last_hi = h
            state.last_hi_idx = i
        # pivot low
        is_low = all(l < df['low'].iloc[i-k] for k in range(1, swing_l+1)) and all(l < df['low'].iloc[i+k] for k in range(1, swing_r+1))
        if is_low:
            state.prev_swing_low = state.last_swing_low
            state.last_swing_low = l
            state.last_lo = l
            state.last_lo_idx = i

    made_hl = (state.last_swing_low is not None and state.prev_swing_low is not None and
               state.last_swing_low > state.prev_swing_low)
    made_lh = (state.last_swing_high is not None and state.prev_swing_high is not None and
               state.last_swing_high < state.prev_swing_high)

    # Sweeps
    sweep_buy_side = (state.last_swing_high is not None and h > state.last_swing_high and c < state.last_swing_high)
    sweep_sell_side = (state.last_swing_low is not None and l < state.last_swing_low and c > state.last_swing_low)

    # ---------------------- FVG and OB detection ----------------------
    min_gap = atr * 0.1
    if i >= 2:
        fvg_up3 = (l > df['high'].iloc[i-2] + min_gap) and fvg_vol_ok
        fvg_dn3 = (h < df['low'].iloc[i-2] - min_gap) and fvg_vol_ok
        if fvg_up3:
            add_zone(state.all_zones, df['high'].iloc[i-2], l, i-2, True, False, True)
        if fvg_dn3:
            add_zone(state.all_zones, h, df['low'].iloc[i-2], i-2, False, False, True)

    if i >= 2:
        displacement_up = (c > h_prev and c > o)
        displacement_dn = (c < l_prev and c < o)
        bull_ob3 = displacement_up and (l_prev < df['low'].iloc[i-2]) and vol_confirm_bull
        bear_ob3 = displacement_dn and (h_prev > df['high'].iloc[i-2]) and vol_confirm_bear
        if bull_ob3:
            add_zone(state.all_zones, h_prev, l_prev, i-1, True, True, True)
        if bear_ob3:
            add_zone(state.all_zones, h_prev, l_prev, i-1, False, True, True)
        gap_up_ob = (i>1 and o > h_prev and c > o)
        gap_dn_ob = (i>1 and o < l_prev and c < o)
        if gap_up_ob:
            add_zone(state.all_zones, o, l_prev, i-1, True, True, True)
        if gap_dn_ob:
            add_zone(state.all_zones, h_prev, o, i-1, False, True, True)

    # ---------------------- Zone aging and mitigation ----------------------
    max_age = params.get('maxAge', 70)
    fail_window = params.get('failWindow', 5)
    close_mitigate = params.get('closeMitigate', True)
    max_taps = 5
    to_remove = []
    for idx, z in enumerate(state.all_zones):
        age = i - z.start_bar
        failed = False
        if age <= fail_window and i >= 1:
            if z.is_bull:
                if c < z.bottom and c_prev < z.bottom:
                    failed = True
            else:
                if c > z.top and c_prev > z.top:
                    failed = True
        if not z.is_mitigated:
            if h > z.bottom and l < z.top:
                z.taps += 1
            if (z.is_bull and (c < z.bottom if close_mitigate else l < z.bottom)) or (not z.is_bull and (c > z.top if close_mitigate else h > z.top)):
                z.is_mitigated = True
            if z.taps > max_taps:
                z.is_mitigated = True
        if age > max_age or failed:
            to_remove.append(idx)
    for idx in reversed(to_remove):
        del state.all_zones[idx]

    # ---------------------- Zone awareness ----------------------
    inside_zone = False
    near_zone = False
    near_zone_bullish = False
    near_zone_bearish = False
    zone_distance = None
    for z in state.all_zones:
        if z.is_mitigated:
            continue
        if h >= z.bottom and l <= z.top:
            inside_zone = True
            state.last_zone_bullish = z.is_bull
            state.last_zone_bearish = not z.is_bull
            break
    if not inside_zone:
        for z in state.all_zones:
            if z.is_mitigated:
                continue
            dist_to_top = abs(c - z.top)/c*100
            dist_to_btm = abs(c - z.bottom)/c*100
            if dist_to_top < 3 or dist_to_btm < 3:
                near_zone = True
                near_zone_bullish = z.is_bull
                near_zone_bearish = not z.is_bull
                zone_distance = min(dist_to_top, dist_to_btm)
                break

    retest_occurred = (not state.inside_zone_prev and inside_zone)
    breakout_occurred = (state.inside_zone_prev and not inside_zone)
    state.inside_zone_prev = inside_zone

    # ---------------------- Structure (BOS/CHoCH) ----------------------
    if state.last_swing_high is not None:
        bos_up_raw = h > state.last_swing_high
        bos_up_valid = bos_up_raw and c > state.last_swing_high + atr * 0.1
        bos_up_rejected = bos_up_valid and c_prev > state.last_swing_high and c < state.last_swing_high
        bos_up_confirmed = bos_up_valid and not bos_up_rejected
        if bos_up_confirmed and l <= state.last_swing_high:
            if state.is_uptrend is None or state.is_uptrend:
                state.bos_up_list.append((state.last_hi_idx, i, state.last_swing_high))
            else:
                state.cho_up_list.append((state.last_hi_idx, i, state.last_swing_high))
            state.is_uptrend = True
            state.smc_bullish = True
            state.smc_bearish = False
    if state.last_swing_low is not None:
        bos_dn_raw = l < state.last_swing_low
        bos_dn_valid = bos_dn_raw and c < state.last_swing_low - atr * 0.1
        bos_dn_rejected = bos_dn_valid and c_prev < state.last_swing_low and c > state.last_swing_low
        bos_dn_confirmed = bos_dn_valid and not bos_dn_rejected
        if bos_dn_confirmed and h >= state.last_swing_low:
            if state.is_uptrend is None or not state.is_uptrend:
                state.bos_dn_list.append((state.last_lo_idx, i, state.last_swing_low))
            else:
                state.cho_dn_list.append((state.last_lo_idx, i, state.last_swing_low))
            state.is_uptrend = False
            state.smc_bullish = False
            state.smc_bearish = True

    # Early structure flip
    internal_bull_bos = (state.last_swing_high is not None and c > state.last_swing_high)
    internal_bear_bos = (state.last_swing_low is not None and c < state.last_swing_low)
    if internal_bull_bos:
        state.smc_early_bull = True
        state.smc_early_bear = False
    if internal_bear_bos:
        state.smc_early_bull = False
        state.smc_early_bear = True
    if state.smc_early_bull and (state.last_swing_low is not None and l < state.last_swing_low):
        state.smc_early_bull = False

    # ---------------------- Candlestick patterns (exactly as Pine) ----------------------
    body0 = abs(c - o)
    crange0 = h - l
    wick_high = h - max(o, c)
    wick_low = min(o, c) - l
    safe_crange = crange0 if crange0 > 0 else 0.001
    body_pct = body0 / safe_crange
    upper_wick_pct = wick_high / safe_crange
    lower_wick_pct = wick_low / safe_crange

    gravestone = (body_pct <= 0.10) and (upper_wick_pct >= 0.60) and (lower_wick_pct <= 0.10)
    shooting_star = (c < o) and (upper_wick_pct >= 0.60) and (body_pct <= 0.30) and (lower_wick_pct <= 0.10)
    hammer = (lower_wick_pct >= 0.60) and (upper_wick_pct <= 0.10) and (body_pct <= 0.30)
    bull_engulf = (c > o) and (c_prev < o_prev) and (o <= c_prev) and (c >= o_prev)
    bear_engulf = (c < o) and (c_prev > o_prev) and (o >= c_prev) and (c <= o_prev)
    doji = body0 <= safe_crange * params.get('bodyThresh', 0.10)
    dragonfly = doji and (wick_low >= crange0 * params.get('wickThreshHigh', 0.55)) and (wick_high <= crange0 * params.get('wickThreshLow', 0.15))
    neutral_doji = doji and not gravestone and not dragonfly
    bull_pierce = (c_prev < o_prev) and (c > o) and (o < c_prev) and (c > (o_prev + c_prev)/2) and (c < o_prev)
    bear_dark = (c_prev > o_prev) and (c < o) and (o > h_prev) and (c < (o_prev + c_prev)/2)
    is_morning = (i>=2 and df['close'].iloc[i-2] < df['open'].iloc[i-2] and
                  abs(df['close'].iloc[i-1]-df['open'].iloc[i-1]) <= (df['high'].iloc[i-1]-df['low'].iloc[i-1])*0.3 and
                  c > (df['open'].iloc[i-2]+df['close'].iloc[i-2])/2)
    is_evening = (i>=2 and df['close'].iloc[i-2] > df['open'].iloc[i-2] and
                  abs(df['close'].iloc[i-1]-df['open'].iloc[i-1]) <= (df['high'].iloc[i-1]-df['low'].iloc[i-1])*0.3 and
                  c < (df['open'].iloc[i-2]+df['close'].iloc[i-2])/2)
    tweezer_bot = abs(l - l_prev) < 0.001 and c > o
    tweezer_top = abs(h - h_prev) < 0.001 and c < o
    bull_r3m = (i>=4 and df['close'].iloc[i-4] > df['open'].iloc[i-4] and
                df['close'].iloc[i-3] < df['open'].iloc[i-3] and
                df['close'].iloc[i-2] < df['open'].iloc[i-2] and
                df['close'].iloc[i-1] < df['open'].iloc[i-1] and
                df['high'].iloc[i-3] < df['high'].iloc[i-4] and df['low'].iloc[i-3] > df['low'].iloc[i-4] and
                df['high'].iloc[i-2] < df['high'].iloc[i-4] and df['low'].iloc[i-2] > df['low'].iloc[i-4] and
                df['high'].iloc[i-1] < df['high'].iloc[i-4] and df['low'].iloc[i-1] > df['low'].iloc[i-4] and
                c > df['close'].iloc[i-4])
    bear_f3m = (i>=4 and df['close'].iloc[i-4] < df['open'].iloc[i-4] and
                df['close'].iloc[i-3] > df['open'].iloc[i-3] and
                df['close'].iloc[i-2] > df['open'].iloc[i-2] and
                df['close'].iloc[i-1] > df['open'].iloc[i-1] and
                df['high'].iloc[i-3] < df['high'].iloc[i-4] and df['low'].iloc[i-3] > df['low'].iloc[i-4] and
                df['high'].iloc[i-2] < df['high'].iloc[i-4] and df['low'].iloc[i-2] > df['low'].iloc[i-4] and
                df['high'].iloc[i-1] < df['high'].iloc[i-4] and df['low'].iloc[i-1] > df['low'].iloc[i-4] and
                c < df['close'].iloc[i-4])
    bull_simple = (c > h_prev and c > o and (c-o) > atr * 0.4)
    bear_simple = (c < l_prev and c < o and (o-c) > atr * 0.4)

    # Pattern assignment (same order as Pine)
    pattern_name = "None"
    pattern_bullish = False
    pattern_bar = -1
    if bull_r3m:
        pattern_name, pattern_bullish = "R 3 M", True
        pattern_bar = i-3
    elif bear_f3m:
        pattern_name, pattern_bullish = "F 3 M", False
        pattern_bar = i-3
    elif is_morning:
        pattern_name, pattern_bullish = "Morning Star", True
        pattern_bar = i-1
    elif is_evening:
        pattern_name, pattern_bullish = "Evening Star", False
        pattern_bar = i-1
    elif bull_engulf:
        pattern_name, pattern_bullish = "Bull Engulfing", True
        pattern_bar = i
    elif bear_engulf:
        pattern_name, pattern_bullish = "Bear Engulfing", False
        pattern_bar = i
    elif bull_pierce:
        pattern_name, pattern_bullish = "Piercing", True
        pattern_bar = i
    elif bear_dark:
        pattern_name, pattern_bullish = "Dark Cloud", False
        pattern_bar = i
    elif tweezer_bot:
        pattern_name, pattern_bullish = "Tweezer Bottom", True
        pattern_bar = i
    elif tweezer_top:
        pattern_name, pattern_bullish = "Tweezer Top", False
        pattern_bar = i
    elif hammer:
        pattern_name, pattern_bullish = "Hammer", True
        pattern_bar = i
    elif shooting_star:
        pattern_name, pattern_bullish = "Shooting Star", False
        pattern_bar = i
    elif gravestone:
        pattern_name, pattern_bullish = "Gravestone", False
        pattern_bar = i
    elif dragonfly:
        pattern_name, pattern_bullish = "Dragonfly", True
        pattern_bar = i
    elif neutral_doji:
        pattern_name, pattern_bullish = "Doji", (c > o)
        pattern_bar = i
    elif bull_simple:
        pattern_name, pattern_bullish = "Bull Break", True
        pattern_bar = i
    elif bear_simple:
        pattern_name, pattern_bullish = "Bear Break", False
        pattern_bar = i

    if pattern_name != "None":
        state.last_pattern = pattern_name
        state.pattern_bullish = pattern_bullish
        state.pattern_bar = pattern_bar
        state.pattern_stored_low_val = l
        state.pattern_stored_high_val = h
        state.pattern_stored_bar_idx = pattern_bar
        state.pattern_rejected = False
        state.pattern_status = "Active"
        state.pattern_stored_close = c

    # Pattern rejection and expiration logic
    if state.pattern_name != "None" and state.pattern_status == "Active" and state.pattern_stored_bar_idx is not None:
        bars_since = i - state.pattern_stored_bar_idx
        if bars_since > 10:
            state.pattern_status = "Expired"
        else:
            thresh = params.get('patternRejectionPercent', 2.0) / 100
            if not state.pattern_bullish:
                if state.pattern_name in ["Gravestone","Shooting Star"]:
                    rejection_level = state.pattern_stored_high_val * (1 + thresh)
                    if h > rejection_level:
                        state.pattern_rejected = True
                elif state.pattern_name == "Bear Break":
                    rejection_level = state.pattern_stored_high_val * (1 + thresh)
                    if c > rejection_level:
                        state.pattern_rejected = True
                elif state.pattern_name in ["Bear Engulfing","Evening Star","Dark Cloud","Tweezer Top","Bear Doji"]:
                    rejection_level = state.pattern_stored_close * (1 + thresh)
                    if c > rejection_level:
                        state.pattern_rejected = True
                else:
                    if c > state.pattern_stored_close * (1 + thresh):
                        state.pattern_rejected = True
            else:
                if state.pattern_name in ["Dragonfly","Hammer"]:
                    rejection_level = state.pattern_stored_low_val * (1 - thresh)
                    if l < rejection_level:
                        state.pattern_rejected = True
                elif state.pattern_name == "Bull Break":
                    rejection_level = state.pattern_stored_low_val * (1 - thresh)
                    if c < rejection_level:
                        state.pattern_rejected = True
                elif state.pattern_name in ["Bull Engulfing","Morning Star","Piercing","Tweezer Bottom","Bull Doji"]:
                    rejection_level = state.pattern_stored_close * (1 - thresh)
                    if c < rejection_level:
                        state.pattern_rejected = True
                else:
                    if c < state.pattern_stored_close * (1 - thresh):
                        state.pattern_rejected = True
            if state.pattern_rejected:
                state.pattern_status = "Rejected"
                state.pattern_bullish = False

    # Momentum (uptrend/downtrend)
    lb_up = c > lb_crv * 1.02
    lb_down = c < lb_crv * 0.98
    ema_bullish = ema20 > ema50
    ema_bearish = ema20 < ema50
    valid_bull_pattern = (state.pattern_bullish and state.pattern_status == "Active" and not state.pattern_rejected)
    valid_bear_pattern = (not state.pattern_bullish and state.pattern_status == "Active" and not state.pattern_rejected)
    mom_bullish = (rsi_val > rsi_ema and c > lb_crv * 0.95) or valid_bull_pattern or (state.last_pattern == "Bull Break" and not state.pattern_rejected)
    mom_bearish = not mom_bullish and ((rsi_val <= rsi_ema and c < lb_crv * 1.05) or valid_bear_pattern or (state.last_pattern == "Bear Break" and not state.pattern_rejected))
    state.mom_bullish = mom_bullish
    state.mom_bearish = mom_bearish
    uptrend = mom_bullish and (rsi_val > rsi_ema or rsi_val >= 47)
    downtrend = (not uptrend and mom_bearish) and (rsi_val < rsi_ema or rsi_val <= 44)
    neutral = not uptrend and not downtrend

    # ---------------------- Liquidity sweeps (SSL/BSL) ----------------------
    # Similar to Pine: detect isSSL, isBSL, strongSSL, strongBSL using 3-bar pattern
    is_ssl = (i>=2 and df['low'].iloc[i-2] < df['low'].iloc[i-3] and df['low'].iloc[i-2] < df['low'].iloc[i-1]) if i>=3 else False
    is_bsl = (i>=2 and df['high'].iloc[i-2] > df['high'].iloc[i-3] and df['high'].iloc[i-2] > df['high'].iloc[i-1]) if i>=3 else False
    swept_ssl_high = h > df['low'].iloc[i-2] if i>=2 else False
    swept_ssl_low = l < df['low'].iloc[i-2] if i>=2 else False
    swept_bsl_high = h > df['high'].iloc[i-2] if i>=2 else False
    swept_bsl_low = l < df['high'].iloc[i-2] if i>=2 else False
    confirm_ssl = is_ssl and c > o and swept_ssl_low
    confirm_bsl = is_bsl and c < o and swept_bsl_high
    can_sweep = (state.last_sweep_bar is None or i - state.last_sweep_bar > 2)
    strong_ssl = confirm_ssl and not (confirm_ssl and confirm_bsl) and can_sweep
    strong_bsl = confirm_bsl and not (confirm_ssl and confirm_bsl) and can_sweep
    if strong_ssl and strong_bsl:
        strong_ssl = c > o
        strong_bsl = c < o
    if strong_ssl:
        state.active_ssl = True
        state.active_ssl_bar = i
    if strong_bsl:
        state.active_bsl = True
        state.active_bsl_bar = i

    # Additional confirmation/rejection logic (simplified, but matches Pine)
    if state.active_ssl and state.last_swing_low is not None:
        if l < state.last_swing_low and c > state.last_swing_low:
            state.ssl_rejected = True
            state.ssl_rejection_bar = i
            state.active_ssl = False
    if state.active_bsl and state.last_swing_high is not None:
        if h > state.last_swing_high and c < state.last_swing_high:
            state.bsl_rejected = True
            state.bsl_rejection_bar = i
            state.active_bsl = False

    # ---------------------- Turning points (as in Pine) ----------------------
    turning_point = False
    turning_reason = ""
    if pattern_name != "None" and not state.pattern_rejected and (i - state.pattern_bar) <= 5:
        if pattern_bullish and strong_ssl:
            turning_point = True
            turning_reason = f"▲ {pattern_name}"
            state.turning_points.append((i, turning_reason, l, "up"))
        elif not pattern_bullish and strong_bsl:
            turning_point = True
            turning_reason = f"▼ {pattern_name}"
            state.turning_points.append((i, turning_reason, h, "down"))
    # BOS rejection
    for (swing_idx, break_idx, price) in state.bos_dn_list:
        if i - break_idx <= 3 and h > price:
            turning_point = True
            turning_reason = "▲ BOS ↓ REJECTED"
            state.turning_points.append((i, turning_reason, price, "up"))
    for (swing_idx, break_idx, price) in state.bos_up_list:
        if i - break_idx <= 3 and l < price:
            turning_point = True
            turning_reason = "▼ BOS ↑ REJECTED"
            state.turning_points.append((i, turning_reason, price, "down"))

    # Store last turning point
    if turning_point:
        state.last_turning_bar = i
        state.last_turning_reason = turning_reason
        state.last_turning_actual_price = l if "up" in turning_reason else h
        state.last_turning_price = (l - atr*0.2) if "up" in turning_reason else (h + atr*0.2)
        state.last_turning_style = "up" if "up" in turning_reason else "down"

    # ---------------------- Scoring & Regime (simplified) ----------------------
    bull_score = 0
    bear_score = 0
    if state.smc_bullish: bull_score += 30
    if state.smc_bearish: bear_score += 30
    if strong_ssl: bull_score += 25
    if strong_bsl: bear_score += 25
    if pattern_name != "None" and not state.pattern_rejected:
        if pattern_bullish: bull_score += 20
        else: bear_score += 20
    if inside_zone and (state.smc_early_bull or state.smc_bullish): bull_score += 10
    if inside_zone and (state.smc_early_bear or state.smc_bearish): bear_score += 10
    if mom_bullish: bull_score += 15
    if mom_bearish: bear_score += 15
    net_score = bull_score - bear_score
    if net_score > 20: regime = "Bullish"
    elif net_score < -20: regime = "Bearish"
    else: regime = "Neutral"

    # Store last values for dashboard
    state.last_close = c
    state.last_lb_crv = lb_crv
    state.last_rsi = rsi_val
    state.last_rsi_ema = rsi_ema
    state.last_ema20 = ema20
    state.last_ema50 = ema50
    state.last_high = h
    state.last_low = l
    state.last_open = o

    # Dashboard dictionary (for UI)
    dashboard = {
        'liquidity': 'SSL' if strong_ssl else 'BSL' if strong_bsl else 'None',
        'sweep_status': 'ACTIVE' if (strong_ssl or strong_bsl) else '---',
        'pattern_text': f"{'↑' if pattern_bullish else '↓'} {pattern_name}" if pattern_name!="None" else "No pattern",
        'pattern_status': 'Active' if pattern_name!="None" and (i - pattern_bar)<=5 else 'Expired',
        'momentum': 'UP ↑' if mom_bullish else 'DOWN ↓' if mom_bearish else '---',
        'struct': 'Bullish' if state.smc_bullish else 'Bearish' if state.smc_bearish else 'Neutral',
        'smc_concept': regime,
        'zone_event': 'Inside Bull Zone' if inside_zone and state.last_zone_bullish else 'Inside Bear Zone' if inside_zone else '---',
        'zone_dist': 'Inside zone' if inside_zone else f"{zone_distance:.1f}% away" if zone_distance else '---',
        'bias': regime,
        'z_score': net_score,
        'signal': 'LONG' if (uptrend and not state.in_short) else 'SHORT' if (downtrend and not state.in_long) else 'NO TRADE'
    }
    return dashboard, (uptrend, downtrend, strong_ssl, strong_bsl, inside_zone, pattern_bullish, pattern_name, pattern_bar, net_score)

# =============================================================================
# SIMPLIFIED SIGNAL AND RISK MANAGEMENT (following Pine)
# =============================================================================
def get_improved_bull_stop(state, df, i, atr, params):
    # Use active zones + last swing low
    best_stop = None
    highest_support = 0
    max_stop_age = params.get('maxAge', 70)
    for z in state.all_zones:
        if not z.is_mitigated and z.is_bull and (i - z.start_bar) <= max_stop_age:
            if z.bottom < df['close'].iloc[i] and z.bottom > highest_support:
                highest_support = z.bottom
                best_stop = z.bottom
    swing_stop = state.last_swing_low - atr * 0.5 if state.last_swing_low is not None else df['close'].iloc[i] - atr * params.get('atrStopMultiplier', 1.5)
    atr_stop = df['close'].iloc[i] - atr * params.get('atrStopMultiplier', 1.5)
    combined = best_stop if best_stop is not None else swing_stop
    combined = max(combined, atr_stop)
    if params.get('enableRiskCap', True):
        extreme = df['close'].iloc[i] * (1 - params.get('maxRiskPercentInput', 7)/100)
        combined = max(combined, extreme)
    min_stop = df['close'].iloc[i] - atr * 0.3
    return min(combined, min_stop) if combined is not None else min_stop

def get_improved_bear_stop(state, df, i, atr, params):
    best_stop = None
    lowest_resistance = 1e9
    max_stop_age = params.get('maxAge', 70)
    for z in state.all_zones:
        if not z.is_mitigated and not z.is_bull and (i - z.start_bar) <= max_stop_age:
            if z.top > df['close'].iloc[i] and z.top < lowest_resistance:
                lowest_resistance = z.top
                best_stop = z.top
    swing_stop = state.last_swing_high + atr * 0.5 if state.last_swing_high is not None else df['close'].iloc[i] + atr * params.get('atrStopMultiplier', 1.5)
    atr_stop = df['close'].iloc[i] + atr * params.get('atrStopMultiplier', 1.5)
    combined = best_stop if best_stop is not None else swing_stop
    combined = min(combined, atr_stop)
    if params.get('enableRiskCap', True):
        extreme = df['close'].iloc[i] * (1 + params.get('maxRiskPercentInput', 7)/100)
        combined = min(combined, extreme)
    min_stop = df['close'].iloc[i] + atr * 0.3
    return max(combined, min_stop) if combined is not None else min_stop

def find_improved_level(ref_price, is_support, level, atr, tp_mult):
    atr_base = atr * tp_mult
    if not is_support:
        target = ref_price + atr_base * level
        return max(target, ref_price * 1.03)
    else:
        target = ref_price - atr_base * level
        return min(target, ref_price * 0.98)

# =============================================================================
# MAIN STREAMLIT APP
# =============================================================================
def main():
    st.sidebar.header("Settings")
    ticker = st.sidebar.text_input("Ticker", "AAPL").upper()
    timeframe = st.sidebar.selectbox("Timeframe", ["1d", "1h"], index=0)
    days_back = 365 if timeframe == "1d" else 30
    start_date = datetime.now() - timedelta(days=days_back)

    handler = OptimizedDataHandler()
    df = handler.load_data(ticker, start_date, timeframe)
    if df is None or df.empty:
        st.error(f"No data for {ticker}")
        return

    st.success(f"Loaded {len(df)} bars for {ticker} ({timeframe})")

    # Parameters (match Pine inputs)
    params = {
        'swing_l': 10 if timeframe=='1d' else 6,
        'swing_r': 5 if timeframe=='1d' else 3,
        'maxAge': 70,
        'failWindow': 5,
        'closeMitigate': True,
        'bodyThresh': 0.10,
        'wickThreshHigh': 0.55,
        'wickThreshLow': 0.15,
        'patternRejectionPercent': 2.0,
        'atrStopMultiplier': 1.5,
        'atrTPMultiplier': 1.5,
        'enableRiskCap': True,
        'maxRiskPercentInput': 7.0
    }

    state = PineState()
    last_dash = None
    # Process each bar
    for i in range(len(df)):
        dash, (uptrend, downtrend, strong_ssl, strong_bsl, inside_zone, pat_bull, pat_name, pat_bar, net_score) = process_bar(df, i, state, params)
        last_dash = dash

        # Signal generation (cooldown)
        cooldown = 3
        if not state.in_long and not state.in_short:
            can_take = (i - state.last_signal_bar) >= cooldown if hasattr(state,'last_signal_bar') else True
            if can_take:
                if uptrend and (strong_ssl or (pat_name!="None" and pat_bull and (i-pat_bar)<=5)):
                    # Enter long
                    state.in_long = True
                    state.entry_price_long = df['close'].iloc[i]
                    state.active_long_sl = get_improved_bull_stop(state, df, i, df['atr'].iloc[i], params)
                    state.active_long_tp1 = find_improved_level(df['close'].iloc[i], False, 1, df['atr'].iloc[i], params['atrTPMultiplier'])
                    state.active_long_tp = state.active_long_tp1
                    state.trade_start_bar = i
                    state.last_signal_bar = i
                elif downtrend and (strong_bsl or (pat_name!="None" and not pat_bull and (i-pat_bar)<=5)):
                    state.in_short = True
                    state.entry_price_short = df['close'].iloc[i]
                    state.active_short_sl = get_improved_bear_stop(state, df, i, df['atr'].iloc[i], params)
                    state.active_short_tp1 = find_improved_level(df['close'].iloc[i], True, 1, df['atr'].iloc[i], params['atrTPMultiplier'])
                    state.active_short_tp = state.active_short_tp1
                    state.trade_start_bar = i
                    state.last_signal_bar = i

        # Exit conditions (simplified)
        if state.in_long:
            if df['low'].iloc[i] <= state.active_long_sl or df['high'].iloc[i] >= state.active_long_tp or downtrend:
                state.in_long = False
        if state.in_short:
            if df['high'].iloc[i] >= state.active_short_sl or df['low'].iloc[i] <= state.active_short_tp or uptrend:
                state.in_short = False

    # Display dashboard (using last_dash)
    st.sidebar.markdown("## 📊 SMC DASHBOARD")
    d = last_dash
    st.sidebar.markdown(f"**LIQUIDITY:** {d['liquidity']}")
    st.sidebar.markdown(f"**SWEEP:** {d['sweep_status']}")
    st.sidebar.markdown(f"**PATTERN:** {d['pattern_text']} ({d['pattern_status']})")
    st.sidebar.markdown(f"**MOMENTUM:** {d['momentum']}")
    st.sidebar.markdown(f"**STRUCT:** {d['struct']}")
    st.sidebar.markdown(f"**SMC:** {d['smc_concept']}")
    st.sidebar.markdown(f"**ZONE:** {d['zone_event']}")
    st.sidebar.markdown(f"**ZONE DIST:** {d['zone_dist']}")
    st.sidebar.markdown(f"**BIAS:** {d['bias']}")
    st.sidebar.markdown(f"**Z-SCORE:** {d['z_score']}% {'Bull' if d['z_score']>0 else 'Bear' if d['z_score']<0 else 'Neut'}")
    st.sidebar.markdown(f"**SIGNAL:** {d['signal']}")

    if state.in_long:
        st.sidebar.success("LONG ACTIVE")
        st.sidebar.info(f"Entry: {state.entry_price_long:.2f} | SL: {state.active_long_sl:.2f} | TP: {state.active_long_tp:.2f}")
    elif state.in_short:
        st.sidebar.warning("SHORT ACTIVE")
        st.sidebar.info(f"Entry: {state.entry_price_short:.2f} | SL: {state.active_short_sl:.2f} | TP: {state.active_short_tp:.2f}")
    else:
        st.sidebar.info("No active trade")

    # Plotting (simplified candlestick + zones + turning points)
    st.subheader("SMC Chart")
    fig, ax = plt.subplots(figsize=(12,6))
    # Plot candlesticks
    for idx in range(len(df)):
        o = df['open'].iloc[idx]
        h = df['high'].iloc[idx]
        l = df['low'].iloc[idx]
        c = df['close'].iloc[idx]
        color = 'green' if c >= o else 'red'
        ax.plot([idx, idx], [l, h], color=color, linewidth=1)
        ax.add_patch(Rectangle((idx-0.3, min(o,c)), 0.6, abs(c-o), facecolor=color, edgecolor=color))
    # Zones
    for z in state.all_zones:
        if not z.is_mitigated:
            ax.axhspan(z.bottom, z.top, xmin=0, xmax=1, alpha=0.1, color=z.base_col)
    # Turning points
    for (idx, reason, price, style) in state.turning_points[-30:]:
        y_offset = price * 0.99 if style=='up' else price * 1.01
        ax.text(idx, y_offset, reason, fontsize=7, ha='center', va='center', rotation=0, alpha=0.8, bbox=dict(facecolor='yellow', alpha=0.5))
    ax.set_title(f"{ticker} - SMC Analysis")
    ax.grid(True, alpha=0.2)
    ax.set_xticks(range(0, len(df), max(1,len(df)//10)))
    ax.set_xticklabels(df.index.strftime("%Y-%m-%d")[::max(1,len(df)//10)], rotation=45)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
