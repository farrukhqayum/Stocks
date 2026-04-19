# =====================================================================
# 1_SMC_DASHBOARD.py (FINAL)
# - Dashboard in sidebar (Pine‑style table, hourly data)
# - Full‑width chart with clipped zones (no future drawing)
# - Daily context filters hourly early entry signals
# =====================================================================

import streamlit as st
st.set_page_config(page_title="SMC Dashboard", layout="wide")
st.title("📈 SMART MONEY CONCEPTS")
from imports import *
# ------------------------------------------------------------
# 1. INDICATORS & HELPERS
# ------------------------------------------------------------
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
    high = df['high']; low = df['low']; close = df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def compute_lb_curve(df, lblen=10):
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    lb = np.zeros(len(df))
    lb[0] = close[0]
    for i in range(1, len(df)):
        start = max(0, i - lblen + 1)
        if start < i:
            highest_lb_prev = lb[start:i].max()
            lowest_lb_prev = lb[start:i].min()
        else:
            highest_lb_prev = lb[i-1]
            lowest_lb_prev = lb[i-1]
        if close[i] > highest_lb_prev:
            lb[i] = (high[i] + close[i]) / 2
        elif close[i] < lowest_lb_prev:
            lb[i] = (low[i] + close[i]) / 2
        else:
            lb[i] = lb[i-1]
    lb_series = pd.Series(lb, index=df.index)
    return lb_series.ewm(span=lblen, adjust=False).mean()

# ------------------------------------------------------------
# 2. ZONE CLASSES
# ------------------------------------------------------------
class FVGZone:
    def __init__(self, top, bottom, start_idx, end_idx, is_bull):
        self.top = top
        self.bottom = bottom
        self.start_idx = start_idx
        self.end_idx = end_idx          # last bar where zone is active (can be updated)
        self.is_bull = is_bull
        self.is_mitigated = False
        self.mitigated_idx = None

class OBZone:
    def __init__(self, top, bottom, start_idx, end_idx, is_bull):
        self.top = top
        self.bottom = bottom
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.is_bull = is_bull
        self.is_mitigated = False
        self.mitigated_idx = None

# ------------------------------------------------------------
# 3. ZONE DETECTION (forward loop, tracks end_idx as current bar)
# ------------------------------------------------------------
def detect_fvg_zones(df, max_age=25, fail_window=5):
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    atr = df['atr'].values
    min_gap = atr * 0.1
    zones = []
    for i in range(len(df)):
        if i >= 2:
            is_fvg_up = (low[i] > high[i-2] + min_gap[i])
            is_fvg_dn = (high[i] < low[i-2] - min_gap[i])
            if is_fvg_up:
                zones.append(FVGZone(high[i-2], low[i], i-2, i, True))
            if is_fvg_dn:
                zones.append(FVGZone(high[i], low[i-2], i-2, i, False))
        # Update existing zones
        to_delete = []
        for j, z in enumerate(zones):
            z.end_idx = i   # extend to current bar
            age = i - z.start_idx
            failed = False
            if age <= fail_window and i >= 1:
                if z.is_bull:
                    if close[i] < z.bottom and close[i-1] < z.bottom:
                        failed = True
                else:
                    if close[i] > z.top and close[i-1] > z.top:
                        failed = True
            if not z.is_mitigated and not failed:
                if z.is_bull and close[i] < z.bottom:
                    z.is_mitigated = True
                    z.mitigated_idx = i
                if not z.is_bull and close[i] > z.top:
                    z.is_mitigated = True
                    z.mitigated_idx = i
            if failed or age > max_age:
                to_delete.append(j)
        for j in reversed(to_delete):
            del zones[j]
    # Return only unmitigated zones
    return [z for z in zones if not z.is_mitigated]

def detect_order_blocks(df, max_age=25, fail_window=5):
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    open_ = df['open'].values
    volume = df['volume'].values if 'volume' in df.columns else None
    if volume is not None:
        vol_sma = pd.Series(volume).rolling(5).mean().values
        vol_sma = np.nan_to_num(vol_sma, nan=np.median(volume) if volume is not None else 1000000)
    else:
        vol_sma = np.ones(len(df)) * 1000000
    zones = []
    for i in range(len(df)):
        if i >= 2:
            vol_ok = volume[i] > vol_sma[i] * 0.6 if volume is not None else True
            displacement_up = (close[i] > high[i-1] and close[i] > open_[i] and low[i-1] < low[i-2])
            if displacement_up and vol_ok:
                zones.append(OBZone(high[i-1], low[i-1], i-1, i, True))
            displacement_down = (close[i] < low[i-1] and close[i] < open_[i] and high[i-1] > high[i-2])
            if displacement_down and vol_ok:
                zones.append(OBZone(high[i-1], low[i-1], i-1, i, False))
            gap_up = (open_[i] > high[i-1] and close[i] > open_[i])
            if gap_up and vol_ok:
                zones.append(OBZone(open_[i], low[i-1], i-1, i, True))
            gap_down = (open_[i] < low[i-1] and close[i] < open_[i])
            if gap_down and vol_ok:
                zones.append(OBZone(high[i-1], open_[i], i-1, i, False))
        # Update existing zones
        to_delete = []
        for j, z in enumerate(zones):
            z.end_idx = i
            age = i - z.start_idx
            failed = False
            if age <= fail_window and i >= 1:
                if z.is_bull:
                    if close[i] < z.bottom and close[i-1] < z.bottom:
                        failed = True
                else:
                    if close[i] > z.top and close[i-1] > z.top:
                        failed = True
            if not z.is_mitigated and not failed:
                if z.is_bull and close[i] < z.bottom:
                    z.is_mitigated = True
                    z.mitigated_idx = i
                if not z.is_bull and close[i] > z.top:
                    z.is_mitigated = True
                    z.mitigated_idx = i
            if failed or age > max_age:
                to_delete.append(j)
        for j in reversed(to_delete):
            del zones[j]
    return [z for z in zones if not z.is_mitigated]

# ------------------------------------------------------------
# 4. SWING POINTS
# ------------------------------------------------------------
def detect_swings(df, left_bars=10, right_bars=4):
    high = df['high'].values
    low = df['low'].values
    swing_highs = []
    swing_lows = []
    for i in range(left_bars, len(df) - right_bars):
        # Pivot high
        is_high = True
        for k in range(1, left_bars+1):
            if high[i] <= high[i-k]:
                is_high = False
                break
        if is_high:
            for k in range(1, right_bars+1):
                if high[i] <= high[i+k]:
                    is_high = False
                    break
        if is_high:
            swing_highs.append({'idx': i, 'price': high[i]})
        # Pivot low
        is_low = True
        for k in range(1, left_bars+1):
            if low[i] >= low[i-k]:
                is_low = False
                break
        if is_low:
            for k in range(1, right_bars+1):
                if low[i] >= low[i+k]:
                    is_low = False
                    break
        if is_low:
            swing_lows.append({'idx': i, 'price': low[i]})
    return swing_highs, swing_lows

# ------------------------------------------------------------
# 5. BOS / CHoCH DETECTION
# ------------------------------------------------------------
def compute_bos_cho_ch(df, swing_highs, swing_lows, atr):
    last_swing_high = None
    last_swing_low = None
    last_high_idx = None
    last_low_idx = None
    is_uptrend = None
    bos_up = []      # (swing_idx, break_idx, price)
    bos_dn = []
    cho_up = []
    cho_dn = []
    for i in range(len(df)):
        for sh in swing_highs:
            if sh['idx'] <= i and (last_swing_high is None or sh['idx'] > last_high_idx):
                last_swing_high = sh['price']
                last_high_idx = sh['idx']
        for sl in swing_lows:
            if sl['idx'] <= i and (last_swing_low is None or sl['idx'] > last_low_idx):
                last_swing_low = sl['price']
                last_low_idx = sl['idx']

        if last_swing_high is not None:
            bos_up_valid = df['close'].iloc[i] > last_swing_high + atr[i] * 0.1
            bos_up_rejected = (i>0 and df['close'].iloc[i-1] > last_swing_high and df['close'].iloc[i] < last_swing_high)
            bos_up_confirmed = bos_up_valid and not bos_up_rejected
            if bos_up_confirmed and df['low'].iloc[i] <= last_swing_high:
                if is_uptrend is None or is_uptrend:
                    bos_up.append((last_high_idx, i, last_swing_high))
                else:
                    cho_up.append((last_high_idx, i, last_swing_high))
                is_uptrend = True

        if last_swing_low is not None:
            bos_dn_valid = df['close'].iloc[i] < last_swing_low - atr[i] * 0.1
            bos_dn_rejected = (i>0 and df['close'].iloc[i-1] < last_swing_low and df['close'].iloc[i] > last_swing_low)
            bos_dn_confirmed = bos_dn_valid and not bos_dn_rejected
            if bos_dn_confirmed and df['high'].iloc[i] >= last_swing_low:
                if is_uptrend is None or not is_uptrend:
                    bos_dn.append((last_low_idx, i, last_swing_low))
                else:
                    cho_dn.append((last_low_idx, i, last_swing_low))
                is_uptrend = False

    return bos_up, bos_dn, cho_up, cho_dn, is_uptrend

# ------------------------------------------------------------
# 6. LIQUIDITY SWEEPS
# ------------------------------------------------------------
def detect_liquidity_sweeps(df, swing_highs, swing_lows):
    last_swing_high = None
    last_swing_low = None
    strong_bsl = []
    strong_ssl = []
    for i in range(len(df)):
        for sh in swing_highs:
            if sh['idx'] <= i:
                last_swing_high = sh['price']
        for sl in swing_lows:
            if sl['idx'] <= i:
                last_swing_low = sl['price']
        if last_swing_high is not None and i>=2:
            is_bsl = (df['high'].iloc[i-2] > df['high'].iloc[i-3] and df['high'].iloc[i-2] > df['high'].iloc[i-1])
            if is_bsl and df['close'].iloc[i] < df['open'].iloc[i] and df['high'].iloc[i] > df['high'].iloc[i-2]:
                strong_bsl.append((i, df['high'].iloc[i-2]))
        if last_swing_low is not None and i>=2:
            is_ssl = (df['low'].iloc[i-2] < df['low'].iloc[i-3] and df['low'].iloc[i-2] < df['low'].iloc[i-1])
            if is_ssl and df['close'].iloc[i] > df['open'].iloc[i] and df['low'].iloc[i] < df['low'].iloc[i-2]:
                strong_ssl.append((i, df['low'].iloc[i-2]))
    return strong_bsl, strong_ssl

# ------------------------------------------------------------
# 7. CANDLESTICK PATTERN
# ------------------------------------------------------------
def detect_candle_pattern(df):
    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    n = len(df)
    if n < 3:
        return None, None, None
    last_pattern = None
    pattern_bull = None
    pattern_idx = None
    for i in range(2, n):
        body0 = abs(c[i] - o[i])
        body1 = abs(c[i-1] - o[i-1])
        body2 = abs(c[i-2] - o[i-2])
        crange0 = h[i] - l[i]
        wick_high = h[i] - max(o[i], c[i])
        wick_low = min(o[i], c[i]) - l[i]
        # Morning / Evening star
        if c[i-2] < o[i-2] and body1 < body2*0.4 and c[i] > (o[i-2]+c[i-2])/2:
            last_pattern, pattern_bull, pattern_idx = "Morning Star", True, i
        elif c[i-2] > o[i-2] and body1 < body2*0.4 and c[i] < (o[i-2]+c[i-2])/2:
            last_pattern, pattern_bull, pattern_idx = "Evening Star", False, i
        # Engulfing
        elif c[i-1] < o[i-1] and c[i] > o[i] and o[i] <= c[i-1] and c[i] >= o[i-1]:
            last_pattern, pattern_bull, pattern_idx = "Bull Engulfing", True, i
        elif c[i-1] > o[i-1] and c[i] < o[i] and o[i] >= c[i-1] and c[i] <= o[i-1]:
            last_pattern, pattern_bull, pattern_idx = "Bear Engulfing", False, i
        # Hammer / Shooting star
        elif wick_low > body0*2 and wick_high < body0*0.5:
            last_pattern, pattern_bull, pattern_idx = "Hammer", True, i
        elif wick_high > body0*2 and wick_low < body0*0.5:
            last_pattern, pattern_bull, pattern_idx = "Shooting Star", False, i
    return last_pattern, pattern_bull, pattern_idx

# ------------------------------------------------------------
# 8. SCORING & REGIME (for hourly)
# ------------------------------------------------------------
def compute_regime_score(smc_bullish, smc_bearish, strong_ssl, strong_bsl,
                         pattern_bull, pattern_rejected, mom_bullish, mom_bearish,
                         inside_zone, last_zone_bullish):
    bull_score, bear_score = 0, 0
    if smc_bullish: bull_score += 30
    if smc_bearish: bear_score += 30
    if strong_ssl: bull_score += 25
    if strong_bsl: bear_score += 25
    if pattern_bull and not pattern_rejected: bull_score += 20
    if pattern_bull is False and not pattern_rejected: bear_score += 20
    if inside_zone and smc_bullish: bull_score += 10
    if inside_zone and smc_bearish: bear_score += 10
    if mom_bullish: bull_score += 15
    if mom_bearish: bear_score += 15
    net_score = bull_score - bear_score
    if net_score > 20: regime = "Bullish"
    elif net_score < -20: regime = "Bearish"
    else: regime = "Neutral"
    return net_score, regime

# ------------------------------------------------------------
# 9. LOAD DATA
# ------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data(ticker, start_date, interval):
    end_date = datetime.today().strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval,
                     auto_adjust=False, progress=False)
    if df is None or df.empty:
        return None
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    if isinstance(df.index, pd.DatetimeIndex):
        df["Date"] = df.index
    else:
        raise ValueError("No datetime index")
    df.set_index("Date", inplace=True)
    df = df.dropna(subset=["open", "high", "low", "close"]).astype(float)
    df['ema20'] = ema(df.close, 20)
    df['ema50'] = ema(df.close, 50)
    df['ema200'] = ema(df.close, 200)
    df['rsi'] = compute_rsi(df.close)
    df['rsi_ema'] = ema(df['rsi'], 14)
    df['atr'] = compute_atr(df, 14)
    df['lb_crv'] = compute_lb_curve(df)
    df = df.bfill().ffill()
    return df

# ------------------------------------------------------------
# 10. DAILY CONTEXT (for entry filtering)
# ------------------------------------------------------------
def analyze_daily_context(df_daily):
    fvg = detect_fvg_zones(df_daily)
    ob = detect_order_blocks(df_daily)
    swing_h, swing_l = detect_swings(df_daily)
    bos_up, bos_dn, cho_up, cho_dn, uptrend = compute_bos_cho_ch(df_daily, swing_h, swing_l, df_daily['atr'].values)
    bsl, ssl = detect_liquidity_sweeps(df_daily, swing_h, swing_l)
    pat, pat_bull, _ = detect_candle_pattern(df_daily)
    lb_up = df_daily['close'].iloc[-1] > df_daily['lb_crv'].iloc[-1] * 1.02
    lb_down = df_daily['close'].iloc[-1] < df_daily['lb_crv'].iloc[-1] * 0.98
    mom_bull = (df_daily['rsi'].iloc[-1] > 51 and df_daily['rsi'].iloc[-1] > df_daily['rsi_ema'].iloc[-1]) or lb_up
    mom_bear = (df_daily['rsi'].iloc[-1] < 44 and df_daily['rsi'].iloc[-1] < df_daily['rsi_ema'].iloc[-1]) or lb_down
    inside = False
    last_bull = None
    for z in fvg+ob:
        if df_daily['high'].iloc[-1] >= z.bottom and df_daily['low'].iloc[-1] <= z.top:
            inside = True
            last_bull = z.is_bull
            break
    net_score, regime = compute_regime_score(uptrend==True, uptrend==False,
                                             len(ssl)>0, len(bsl)>0,
                                             pat_bull, False, mom_bull, mom_bear,
                                             inside, last_bull)
    return {
        'trend': 'BULLISH' if uptrend else 'BEARISH' if uptrend is not None else 'NEUTRAL',
        'net_score': net_score,
        'inside_zone': inside,
        'zone_bullish': last_bull,
        'recent_ssl': len(ssl)>0,
        'recent_bsl': len(bsl)>0,
        'mom_bullish': mom_bull,
        'mom_bearish': mom_bear,
    }

# ------------------------------------------------------------
# 11. HOURLY SIGNAL (early entry, using daily context)
# ------------------------------------------------------------
def get_hourly_signal(df_hourly, daily_ctx):
    df = df_hourly.copy()
    df['lb_crv'] = compute_lb_curve(df)
    df['rsi'] = compute_rsi(df.close)
    df['rsi_ema'] = ema(df['rsi'], 14)
    df['atr'] = compute_atr(df, 14)
    last = df.iloc[-1]
    lb_up = last['close'] > last['lb_crv'] * 1.02
    lb_down = last['close'] < last['lb_crv'] * 0.98
    mom_bull = (last['rsi'] > 51 and last['rsi'] > last['rsi_ema']) or lb_up
    mom_bear = (last['rsi'] < 44 and last['rsi'] < last['rsi_ema']) or lb_down
    pattern, pat_bull, _ = detect_candle_pattern(df)
    fvg = detect_fvg_zones(df, max_age=15)
    ob = detect_order_blocks(df, max_age=15)
    inside = any(last['high'] >= z.bottom and last['low'] <= z.top for z in fvg+ob)
    swing_h, swing_l = detect_swings(df, left_bars=6, right_bars=3)
    bsl, ssl = detect_liquidity_sweeps(df, swing_h, swing_l)
    bos_up, bos_dn, _, _, uptrend = compute_bos_cho_ch(df, swing_h, swing_l, df['atr'].values)

    daily_bull = daily_ctx['trend'] == 'BULLISH' and daily_ctx['net_score'] > 20
    daily_bear = daily_ctx['trend'] == 'BEARISH' and daily_ctx['net_score'] < -20
    can_long = daily_bull or daily_ctx['recent_ssl']
    can_short = daily_bear or daily_ctx['recent_bsl']
    if daily_ctx['inside_zone'] and daily_ctx['zone_bullish'] is not None:
        can_long = can_long and daily_ctx['zone_bullish']
        can_short = can_short and not daily_ctx['zone_bullish']

    long_signal = short_signal = False
    reason = ""
    if can_long and (inside or len(ssl)>0) and (mom_bull or (pattern and pat_bull)):
        long_signal = True
        reason = f"Early long: {pattern if pattern else 'momentum'}"
    elif can_short and (inside or len(bsl)>0) and (mom_bear or (pattern and not pat_bull)):
        short_signal = True
        reason = f"Early short: {pattern if pattern else 'momentum'}"

    atr_val = last['atr']
    if long_signal:
        sl = last['low'] - atr_val * 0.5
        tp = last['close'] + atr_val * 1.5
        risk = last['close'] - sl
        reward = tp - last['close']
        rr = reward/risk if risk>0 else 0
        return {'signal':'LONG','valid':rr>=1.5,'reason':reason,'sl':sl,'tp':tp,'rr':rr,'risk_pct':(risk/last['close'])*100}
    elif short_signal:
        sl = last['high'] + atr_val * 0.5
        tp = last['close'] - atr_val * 1.5
        risk = sl - last['close']
        reward = last['close'] - tp
        rr = reward/risk if risk>0 else 0
        return {'signal':'SHORT','valid':rr>=1.5,'reason':reason,'sl':sl,'tp':tp,'rr':rr,'risk_pct':(risk/last['close'])*100}
    else:
        return {'signal':'NO TRADE','valid':False,'reason':'No early setup'}

# ------------------------------------------------------------
# 12. CHART WITH CLIPPED ZONES (no future drawing)
# ------------------------------------------------------------
def plot_full_chart(df, fvg_zones, ob_zones, bos_up, bos_dn, cho_up, cho_dn,
                    turning_points, title, start_idx_global=0,
                    show_fvg=True, show_ob=True, show_bos=True, show_tp=True):
    """
    df: full dataframe (or slice)
    zones are defined on the same df (global indices). We clip them to the visible slice.
    """
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                   gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    x = np.arange(len(df))
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    width = 0.6
    up_color = "#26a69a"; down_color = "#ef5350"

    # Candles
    for i in range(len(df)):
        color = up_color if c.iloc[i] >= o.iloc[i] else down_color
        ax.vlines(i, l.iloc[i], h.iloc[i], color=color, linewidth=1)
        ax.add_patch(Rectangle((i-width/2, min(o.iloc[i], c.iloc[i])), width,
                               abs(c.iloc[i]-o.iloc[i]) or 0.001,
                               facecolor=color, edgecolor=color))

    # LB curve
    ax.plot(x, df["lb_crv"], color="gray", alpha=0.8, linewidth=1.2)

    # Clip zones to visible range (0 to len(df)-1)
    visible_start = 0
    visible_end = len(df)-1

    if show_fvg:
        for z in fvg_zones:
            # Clip zone to visible window
            start = max(z.start_idx - start_idx_global, visible_start)
            end = min(z.end_idx - start_idx_global, visible_end) if not z.is_mitigated else min(z.mitigated_idx - start_idx_global, visible_end)
            if start > visible_end or end < visible_start:
                continue
            color = "teal" if z.is_bull else "blue"
            rect_x = start - 0.5
            rect_w = (end - start) + 1
            ax.add_patch(Rectangle((rect_x, z.bottom), rect_w, z.top-z.bottom,
                                   facecolor=color, alpha=0.07, edgecolor=color,
                                   linestyle="--", linewidth=1.5))

    if show_ob:
        for z in ob_zones:
            start = max(z.start_idx - start_idx_global, visible_start)
            end = min(z.end_idx - start_idx_global, visible_end) if not z.is_mitigated else min(z.mitigated_idx - start_idx_global, visible_end)
            if start > visible_end or end < visible_start:
                continue
            color = "green" if z.is_bull else "orange"
            rect_x = start - 0.5
            rect_w = (end - start) + 1
            ax.add_patch(Rectangle((rect_x, z.bottom), rect_w, z.top-z.bottom,
                                   facecolor=color, alpha=0.07, edgecolor=color,
                                   linestyle="-", linewidth=2))

    # BOS/CHoCH lines (convert indices to local)
        # BOS/CHoCH lines (from swing bar to break bar only)
    if show_bos:
        for (swing_idx, break_idx, price) in bos_up:
            local_swing = swing_idx - start_idx_global
            local_break = break_idx - start_idx_global
            if local_swing < 0 or local_break > visible_end:
                continue
            ax.plot([local_swing, local_break], [price, price], color="lime", linestyle="--", linewidth=1.5, alpha=0.8)
            # optional label at the break bar
            ax.text(local_break, price, "  BOS ↑", fontsize=8, color="lime", va='bottom')
        for (swing_idx, break_idx, price) in bos_dn:
            local_swing = swing_idx - start_idx_global
            local_break = break_idx - start_idx_global
            if local_swing < 0 or local_break > visible_end:
                continue
            ax.plot([local_swing, local_break], [price, price], color="red", linestyle="--", linewidth=1.5, alpha=0.8)
            ax.text(local_break, price, "  BOS ↓", fontsize=8, color="red", va='top')
        for (swing_idx, break_idx, price) in cho_up:
            local_swing = swing_idx - start_idx_global
            local_break = break_idx - start_idx_global
            if local_swing < 0 or local_break > visible_end:
                continue
            ax.plot([local_swing, local_break], [price, price], color="cyan", linestyle="--", linewidth=1.5, alpha=0.8)
            ax.text(local_break, price, "  CHoCH ↑", fontsize=8, color="cyan", va='bottom')
        for (swing_idx, break_idx, price) in cho_dn:
            local_swing = swing_idx - start_idx_global
            local_break = break_idx - start_idx_global
            if local_swing < 0 or local_break > visible_end:
                continue
            ax.plot([local_swing, local_break], [price, price], color="orange", linestyle="--", linewidth=1.5, alpha=0.8)
            ax.text(local_break, price, "  CHoCH ↓", fontsize=8, color="orange", va='top')

    # Turning points
    if show_tp and turning_points:
        for (idx, reason, price, style) in turning_points:
            local_idx = idx - start_idx_global
            if local_idx < 0 or local_idx > visible_end:
                continue
            y = price * 0.99 if style == "up" else price * 1.01
            va = 'top' if style == "up" else 'bottom'
            ax.text(local_idx, y, reason, color="orange", fontsize=7, ha='center', va=va,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='orange'))

    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.yaxis.tick_right()

    # RSI panel
    rsi = df["rsi"]; rsi_ema = df["rsi_ema"]
    ax2.fill_between(x, rsi, rsi_ema, where=(rsi>rsi_ema), color="green", alpha=0.15)
    ax2.fill_between(x, rsi, rsi_ema, where=(rsi<rsi_ema), color="red", alpha=0.15)
    ax2.plot(x, rsi, color="gray", linewidth=1.2)
    ax2.plot(x, rsi_ema, color="gold", linewidth=1.2)
    for level in [25,50,78]:
        ax2.axhline(level, color="black", linestyle="--", linewidth=0.7, alpha=0.6)
    ax2.set_ylim(0,100)
    ax2.set_ylabel("RSI")
    ax2.grid(alpha=0.2)
    ax2.yaxis.tick_right()

    if isinstance(df.index, pd.DatetimeIndex):
        step = max(1, len(x)//10)
        ax2.set_xticks(x[::step])
        ax2.set_xticklabels(df.index.strftime("%Y-%m-%d %H:%M")[::step], rotation=45, fontsize=8)

    plt.tight_layout()
    return fig

# ------------------------------------------------------------
# 13. MAIN STREAMLIT UI
# ------------------------------------------------------------
st.sidebar.header("Multi‑Timeframe Settings")
ticker = st.sidebar.text_input("Ticker", "AAPL")

# Load data
start_daily = datetime.today() - timedelta(days=365)
df_daily = load_data(ticker, start_daily, "1d")
if df_daily is None:
    st.error("No daily data")
    st.stop()

start_hourly = datetime.today() - timedelta(days=30)
df_hourly = load_data(ticker, start_hourly, "1h")
if df_hourly is None:
    st.warning("Hourly data unavailable, using 4H")
    df_hourly = load_data(ticker, start_hourly, "4h")
    if df_hourly is None:
        st.error("No intraday data")
        st.stop()

# Daily context
with st.spinner("Analyzing daily context..."):
    daily_ctx = analyze_daily_context(df_daily)

# Hourly full analysis (for dashboard)
with st.spinner("Computing hourly SMC state..."):
    # Indicators already in df_hourly (from load_data), but need LB and RSI_ema recomputed? load_data already did.
    fvg_h = detect_fvg_zones(df_hourly)
    ob_h = detect_order_blocks(df_hourly)
    swing_h, swing_l = detect_swings(df_hourly)
    bos_up, bos_dn, cho_up, cho_dn, uptrend_h = compute_bos_cho_ch(df_hourly, swing_h, swing_l, df_hourly['atr'].values)
    bsl_h, ssl_h = detect_liquidity_sweeps(df_hourly, swing_h, swing_l)
    pat_h, pat_bull_h, pat_idx_h = detect_candle_pattern(df_hourly)
    last = df_hourly.iloc[-1]
    lb_up_h = last['close'] > last['lb_crv'] * 1.02
    lb_down_h = last['close'] < last['lb_crv'] * 0.98
    mom_bull_h = (last['rsi'] > 51 and last['rsi'] > last['rsi_ema']) or lb_up_h
    mom_bear_h = (last['rsi'] < 44 and last['rsi'] < last['rsi_ema']) or lb_down_h
    inside_h = any(last['high'] >= z.bottom and last['low'] <= z.top for z in fvg_h+ob_h)
    last_zone_bullish_h = None
    for z in fvg_h+ob_h:
        if last['high'] >= z.bottom and last['low'] <= z.top:
            last_zone_bullish_h = z.is_bull
            break
    net_score_h, regime_h = compute_regime_score(uptrend_h==True, uptrend_h==False,
                                                 len(ssl_h)>0, len(bsl_h)>0,
                                                 pat_bull_h, False, mom_bull_h, mom_bear_h,
                                                 inside_h, last_zone_bullish_h)

    # Turning points for hourly
    turning_points_h = []
    if pat_h is not None and pat_idx_h is not None and (len(df_hourly)-1 - pat_idx_h) <= 5:
        if pat_bull_h and len(ssl_h)>0:
            turning_points_h.append((len(df_hourly)-1, f"▲ {pat_h}", last['low'], "up"))
        elif not pat_bull_h and len(bsl_h)>0:
            turning_points_h.append((len(df_hourly)-1, f"▼ {pat_h}", last['high'], "down"))
    for (idx, price) in bos_dn:
        if len(df_hourly)-1 - idx <= 3 and df_hourly['high'].iloc[-1] > price:
            turning_points_h.append((len(df_hourly)-1, "▲ BOS ↓ REJECTED", price, "up"))
    for (idx, price) in bos_up:
        if len(df_hourly)-1 - idx <= 3 and df_hourly['low'].iloc[-1] < price:
            turning_points_h.append((len(df_hourly)-1, "▼ BOS ↑ REJECTED", price, "down"))

# Hourly entry signal
with st.spinner("Computing hourly entry signal..."):
    hourly_signal = get_hourly_signal(df_hourly, daily_ctx)

# ------------------------------------------------------------
# SIDEBAR DASHBOARD (Pine‑style, using hourly data)
# ------------------------------------------------------------
st.sidebar.markdown("## 📊 SMC DASHBOARD (Hourly)")
# Build rows
liquidity_text = "SSL" if len(ssl_h)>0 else "BSL" if len(bsl_h)>0 else "None"
liquidity_color = "green" if len(ssl_h)>0 else "red" if len(bsl_h)>0 else "gray"
sweep_status = "ACTIVE" if (len(ssl_h)>0 or len(bsl_h)>0) else "---"
pattern_text = f"{'↑' if pat_bull_h else '↓'} {pat_h}" if pat_h else "No pattern"
pattern_status = "Active" if pat_h and (len(df_hourly)-1 - pat_idx_h) <= 5 else "Expired"
mom_text = "UP ↑" if mom_bull_h else "DOWN ↓" if mom_bear_h else "---"
struct_text = "Bullish" if uptrend_h else "Bearish" if uptrend_h is not None else "Neutral"
smc_concept = regime_h
zone_event = "Inside Bull Zone" if inside_h and last_zone_bullish_h else \
             "Inside Bear Zone" if inside_h and last_zone_bullish_h is False else "---"
# Zone distance
zone_dist = ""
if inside_h:
    zone_dist = "Inside zone"
else:
    min_dist = 999
    for z in fvg_h+ob_h:
        if not z.is_mitigated:
            dist = min(abs(last['close']-z.top), abs(last['close']-z.bottom))
            if dist < min_dist:
                min_dist = dist
    if min_dist < 999:
        pct = (min_dist / last['close']) * 100
        zone_dist = f"{pct:.1f}% away"
bias = regime_h
score = net_score_h
signal_text = hourly_signal['signal']
if hourly_signal['signal'] != 'NO TRADE':
    signal_text += f"\nSL:{hourly_signal['sl']:.2f} TP:{hourly_signal['tp']:.2f}\nR/R:{hourly_signal['rr']:.2f} Risk:{hourly_signal['risk_pct']:.1f}%"

# Render HTML table
html = f"""
<style>
.smc-table {{ font-family: monospace; font-size: 14px; border-collapse: collapse; width: 100%; }}
.smc-table td {{ padding: 6px; border: 1px solid #ddd; }}
.green-bg {{ background-color: #2e7d32; color: white; }}
.red-bg {{ background-color: #c62828; color: white; }}
.gray-bg {{ background-color: #4f4f4f; color: white; }}
.yellow-bg {{ background-color: #f9a825; color: black; }}
.blue-bg {{ background-color: #1565c0; color: white; }}
.orange-bg {{ background-color: #ef6c00; color: white; }}
</style>
<table class="smc-table">
<tr><td style="background-color:#1e3a5f; color:white; text-align:center" colspan="2"><b>📊 {ticker} - SMC</b></td></tr>
<tr><td>LIQUIDITY:</td><td class="{liquidity_color}-bg">{liquidity_text}</td></tr>
<tr><td>SWEEP:</td><td class="{'green-bg' if sweep_status=='ACTIVE' else 'gray-bg'}">{sweep_status}</td></tr>
<tr><td>PATTERN:</td><td class="{'green-bg' if pat_bull_h else 'red-bg' if pat_h else 'gray-bg'}">{pattern_text} ({pattern_status})</td></tr>
<tr><td>MOMENTUM:</td><td class="{'green-bg' if mom_bull_h else 'red-bg' if mom_bear_h else 'gray-bg'}">{mom_text}</td></tr>
<tr><td>STRUCT:</td><td class="{'green-bg' if struct_text=='Bullish' else 'red-bg' if struct_text=='Bearish' else 'gray-bg'}">{struct_text}</td></tr>
<tr><td>SMC:</td><td class="{'green-bg' if smc_concept=='Bullish' else 'red-bg' if smc_concept=='Bearish' else 'gray-bg'}">{smc_concept}</td></tr>
<tr><td>ZONE:</td><td class="{'green-bg' if 'Bull' in zone_event else 'red-bg' if 'Bear' in zone_event else 'gray-bg'}">{zone_event}</td></tr>
<tr><td>ZONE DIST:</td><td class="{'yellow-bg' if zone_dist else 'gray-bg'}">{zone_dist if zone_dist else '---'}</td></tr>
<tr><td>BIAS:</td><td class="{'green-bg' if bias=='Bullish' else 'red-bg' if bias=='Bearish' else 'gray-bg'}">{bias}</td></tr>
<tr><td>Z-SCORE:</td><td class="{'green-bg' if score>0 else 'red-bg' if score<0 else 'gray-bg'}">{score}% {'Bull' if score>0 else 'Bear' if score<0 else 'Neut'}</td></tr>
<tr><td>SIGNAL:</td><td class="{'green-bg' if 'LONG' in signal_text else 'red-bg' if 'SHORT' in signal_text else 'gray-bg'}">{signal_text}</td></tr>
</table>
"""
st.sidebar.markdown(html, unsafe_allow_html=True)
if hourly_signal['valid']:
    st.sidebar.success("✅ RECOMMENDATION: TAKE TRADE")
else:
    st.sidebar.info("⛔ RECOMMENDATION: AVOID or wait")

# ------------------------------------------------------------
# MAIN AREA: FULL WIDTH CHART
# ------------------------------------------------------------
st.markdown("## 📈 Hourly SMC Chart")
with st.expander("Chart Overlays", expanded=True):
    show_fvg = st.checkbox("Show FVG Zones", value=True)
    show_ob = st.checkbox("Show Order Blocks", value=True)
    show_bos = st.checkbox("Show BOS/CHoCH Lines", value=True)
    show_tp = st.checkbox("Show Turning Points", value=True)

# Use last 300 bars for visibility, but zones are clipped automatically
slice_df = df_hourly.tail(300)
global_start_idx = len(df_hourly) - len(slice_df)

fig = plot_full_chart(slice_df, fvg_h, ob_h, bos_up, bos_dn, cho_up, cho_dn,
                      turning_points_h, f"{ticker} – Hourly SMC",
                      start_idx_global=global_start_idx,
                      show_fvg=show_fvg, show_ob=show_ob,
                      show_bos=show_bos, show_tp=show_tp)
st.pyplot(fig)

# Optional daily chart
if st.sidebar.checkbox("Show Daily Chart"):
    st.markdown("## 📉 Daily SMC Chart")
    fvg_d = detect_fvg_zones(df_daily)
    ob_d = detect_order_blocks(df_daily)
    swing_hd, swing_ld = detect_swings(df_daily)
    bos_up_d, bos_dn_d, cho_up_d, cho_dn_d, _ = compute_bos_cho_ch(df_daily, swing_hd, swing_ld, df_daily['atr'].values)
    # No turning points for daily in this optional view
    slice_daily = df_daily.tail(150)
    global_start_daily = len(df_daily) - len(slice_daily)
    fig_d = plot_full_chart(slice_daily, fvg_d, ob_d, bos_up_d, bos_dn_d, cho_up_d, cho_dn_d,
                            [], f"{ticker} – Daily",
                            start_idx_global=global_start_daily,
                            show_fvg=show_fvg, show_ob=show_ob, show_bos=show_bos, show_tp=False)
    st.pyplot(fig_d)
