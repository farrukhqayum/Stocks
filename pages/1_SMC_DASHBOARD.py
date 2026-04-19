# =====================================================================
# 1_SMC_DASHBOARD.py (MULTI‑TIMEFRAME + SIMPLIFIED CHART)
# Daily Context + Hourly Entry Signals + Clean Candles + LB + RSI
# =====================================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

st.set_page_config(page_title="SMC Dashboard", layout="wide")
st.title("📈 SMART MONEY CONCEPTS – Daily Context + Hourly Entry")

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
    high = df['high']
    low = df['low']
    close = df['close']
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
# 2. ZONE CLASSES (FVG / OB) – needed for context but not plotted
# ------------------------------------------------------------
class FVGZone:
    def __init__(self, top, bottom, start_idx, is_bull):
        self.top = top
        self.bottom = bottom
        self.start_idx = start_idx
        self.is_bull = is_bull
        self.is_mitigated = False
        self.mitigated_idx = None
        self.touched = False

class OBZone:
    def __init__(self, top, bottom, start_idx, is_bull):
        self.top = top
        self.bottom = bottom
        self.start_idx = start_idx
        self.is_bull = is_bull
        self.is_mitigated = False
        self.mitigated_idx = None
        self.touched = False

# ------------------------------------------------------------
# 3. ZONE DETECTION (FVG & OB) – used for daily context only
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
                zones.append(FVGZone(high[i-2], low[i], i-2, True))
            if is_fvg_dn:
                zones.append(FVGZone(high[i], low[i-2], i-2, False))
        to_delete = []
        for j, z in enumerate(zones):
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
                zones.append(OBZone(high[i-1], low[i-1], i-1, True))
            displacement_down = (close[i] < low[i-1] and close[i] < open_[i] and high[i-1] > high[i-2])
            if displacement_down and vol_ok:
                zones.append(OBZone(high[i-1], low[i-1], i-1, False))
            gap_up = (open_[i] > high[i-1] and close[i] > open_[i])
            if gap_up and vol_ok:
                zones.append(OBZone(open_[i], low[i-1], i-1, True))
            gap_down = (open_[i] < low[i-1] and close[i] < open_[i])
            if gap_down and vol_ok:
                zones.append(OBZone(high[i-1], open_[i], i-1, False))
        to_delete = []
        for j, z in enumerate(zones):
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
# 4. SWING POINTS (pivothigh / pivotlow)
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
# 5. BOS / CHoCH DETECTION (matching Pine logic)
# ------------------------------------------------------------
def compute_bos_cho_ch(df, swing_highs, swing_lows, atr):
    last_swing_high = None
    last_swing_low = None
    last_high_idx = None
    last_low_idx = None
    is_uptrend = None
    bos_up_bars = []
    bos_dn_bars = []
    cho_ch_up_bars = []
    cho_ch_dn_bars = []

    for i in range(len(df)):
        for sh in swing_highs:
            if sh['idx'] <= i:
                if last_swing_high is None or sh['idx'] > last_high_idx:
                    last_swing_high = sh['price']
                    last_high_idx = sh['idx']
        for sl in swing_lows:
            if sl['idx'] <= i:
                if last_swing_low is None or sl['idx'] > last_low_idx:
                    last_swing_low = sl['price']
                    last_low_idx = sl['idx']

        if last_swing_high is not None:
            bos_up_valid = df['close'].iloc[i] > last_swing_high + atr[i] * 0.1
            bos_up_rejected = (i>0 and df['close'].iloc[i-1] > last_swing_high and df['close'].iloc[i] < last_swing_high)
            bos_up_confirmed = bos_up_valid and not bos_up_rejected
            if bos_up_confirmed and df['low'].iloc[i] <= last_swing_high:
                if is_uptrend is None or is_uptrend == True:
                    bos_up_bars.append((i, last_swing_high))
                else:
                    cho_ch_up_bars.append((i, last_swing_high))
                is_uptrend = True

        if last_swing_low is not None:
            bos_dn_valid = df['close'].iloc[i] < last_swing_low - atr[i] * 0.1
            bos_dn_rejected = (i>0 and df['close'].iloc[i-1] < last_swing_low and df['close'].iloc[i] > last_swing_low)
            bos_dn_confirmed = bos_dn_valid and not bos_dn_rejected
            if bos_dn_confirmed and df['high'].iloc[i] >= last_swing_low:
                if is_uptrend is None or is_uptrend == False:
                    bos_dn_bars.append((i, last_swing_low))
                else:
                    cho_ch_dn_bars.append((i, last_swing_low))
                is_uptrend = False

    return bos_up_bars, bos_dn_bars, cho_ch_up_bars, cho_ch_dn_bars, is_uptrend

# ------------------------------------------------------------
# 6. LIQUIDITY SWEEPS (BSL / SSL)
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
# 7. CANDLESTICK PATTERN ENGINE (simplified)
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
        if c[i-2] < o[i-2] and body1 < body2*0.4 and c[i] > (o[i-2]+c[i-2])/2:
            last_pattern = "Morning Star"
            pattern_bull = True
            pattern_idx = i
        elif c[i-2] > o[i-2] and body1 < body2*0.4 and c[i] < (o[i-2]+c[i-2])/2:
            last_pattern = "Evening Star"
            pattern_bull = False
            pattern_idx = i
        elif c[i-1] < o[i-1] and c[i] > o[i] and o[i] <= c[i-1] and c[i] >= o[i-1]:
            last_pattern = "Bull Engulfing"
            pattern_bull = True
            pattern_idx = i
        elif c[i-1] > o[i-1] and c[i] < o[i] and o[i] >= c[i-1] and c[i] <= o[i-1]:
            last_pattern = "Bear Engulfing"
            pattern_bull = False
            pattern_idx = i
        else:
            body0 = abs(c[i] - o[i])
            crange = h[i] - l[i]
            wick_low = min(o[i], c[i]) - l[i]
            wick_high = h[i] - max(o[i], c[i])
            if wick_low > body0*2 and wick_high < body0*0.5:
                last_pattern = "Hammer"
                pattern_bull = True
                pattern_idx = i
            elif wick_high > body0*2 and wick_low < body0*0.5:
                last_pattern = "Shooting Star"
                pattern_bull = False
                pattern_idx = i
    return last_pattern, pattern_bull, pattern_idx

# ------------------------------------------------------------
# 8. SCORING & REGIME
# ------------------------------------------------------------
def compute_regime_score(df, smc_bullish, smc_bearish, strong_ssl, strong_bsl,
                         pattern_bull, pattern_bear, pattern_rejected,
                         mom_bullish, mom_bearish, inside_zone, last_zone_bullish):
    bull_score = 0
    bear_score = 0
    if smc_bullish:
        bull_score += 30
    if smc_bearish:
        bear_score += 30
    if strong_ssl:
        bull_score += 25
    if strong_bsl:
        bear_score += 25
    if pattern_bull and not pattern_rejected:
        bull_score += 20
    if pattern_bear and not pattern_rejected:
        bear_score += 20
    if inside_zone and smc_bullish:
        bull_score += 10
    if inside_zone and smc_bearish:
        bear_score += 10
    if mom_bullish:
        bull_score += 15
    if mom_bearish:
        bear_score += 15
    net_score = bull_score - bear_score
    if net_score > 20:
        regime = "Bullish"
    elif net_score < -20:
        regime = "Bearish"
    else:
        regime = "Neutral"
    return net_score, regime

# ------------------------------------------------------------
# 9. LOAD DATA FOR A GIVEN TIMEFRAME
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
    # Basic indicators
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
# 10. DAILY CONTEXT ANALYSIS
# ------------------------------------------------------------
def analyze_daily_context(df_daily):
    fvg_zones = detect_fvg_zones(df_daily)
    ob_zones = detect_order_blocks(df_daily)
    swing_highs, swing_lows = detect_swings(df_daily, left_bars=10, right_bars=4)
    bos_up, bos_dn, cho_up, cho_dn, is_uptrend = compute_bos_cho_ch(df_daily, swing_highs, swing_lows, df_daily['atr'].values)
    strong_bsl, strong_ssl = detect_liquidity_sweeps(df_daily, swing_highs, swing_lows)
    pattern, pattern_bull, _ = detect_candle_pattern(df_daily)
    pattern_rejected = False
    lb_up = df_daily['close'].iloc[-1] > df_daily['lb_crv'].iloc[-1] * 1.02
    lb_down = df_daily['close'].iloc[-1] < df_daily['lb_crv'].iloc[-1] * 0.98
    mom_bullish = (df_daily['rsi'].iloc[-1] > 51 and df_daily['rsi'].iloc[-1] > df_daily['rsi_ema'].iloc[-1]) or lb_up
    mom_bearish = (df_daily['rsi'].iloc[-1] < 44 and df_daily['rsi'].iloc[-1] < df_daily['rsi_ema'].iloc[-1]) or lb_down
    inside_zone = False
    last_zone_bullish = None
    for z in fvg_zones + ob_zones:
        if df_daily['high'].iloc[-1] >= z.bottom and df_daily['low'].iloc[-1] <= z.top:
            inside_zone = True
            last_zone_bullish = z.is_bull
            break
    net_score, regime = compute_regime_score(df_daily,
                                             smc_bullish=is_uptrend==True,
                                             smc_bearish=is_uptrend==False,
                                             strong_ssl=len(strong_ssl)>0,
                                             strong_bsl=len(strong_bsl)>0,
                                             pattern_bull=pattern_bull==True,
                                             pattern_bear=pattern_bull==False,
                                             pattern_rejected=pattern_rejected,
                                             mom_bullish=mom_bullish,
                                             mom_bearish=mom_bearish,
                                             inside_zone=inside_zone,
                                             last_zone_bullish=last_zone_bullish)
    return {
        'trend': 'BULLISH' if is_uptrend else 'BEARISH' if is_uptrend is not None else 'NEUTRAL',
        'regime': regime,
        'net_score': net_score,
        'inside_zone': inside_zone,
        'zone_bullish': last_zone_bullish,
        'fvg_count': len([z for z in fvg_zones if not z.is_mitigated]),
        'ob_count': len([z for z in ob_zones if not z.is_mitigated]),
        'recent_bsl': len(strong_bsl)>0,
        'recent_ssl': len(strong_ssl)>0,
        'pattern': pattern,
        'pattern_bull': pattern_bull,
        'mom_bullish': mom_bullish,
        'mom_bearish': mom_bearish,
        'lb_up': lb_up,
        'lb_down': lb_down,
        'bos_up_count': len(bos_up),
        'bos_dn_count': len(bos_dn),
        'cho_up_count': len(cho_up),
        'cho_dn_count': len(cho_dn),
        'last_close': df_daily['close'].iloc[-1]
    }

# ------------------------------------------------------------
# 11. HOURLY ENTRY SIGNAL (using daily context)
# ------------------------------------------------------------
def get_hourly_signal(df_hourly, daily_ctx):
    df_hourly = df_hourly.copy()
    df_hourly['ema20'] = ema(df_hourly.close, 20)
    df_hourly['ema50'] = ema(df_hourly.close, 50)
    df_hourly['lb_crv'] = compute_lb_curve(df_hourly)
    df_hourly['rsi'] = compute_rsi(df_hourly.close)
    df_hourly['rsi_ema'] = ema(df_hourly['rsi'], 14)
    df_hourly['atr'] = compute_atr(df_hourly, 14)

    last = df_hourly.iloc[-1]
    lb_up = last['close'] > last['lb_crv'] * 1.02
    lb_down = last['close'] < last['lb_crv'] * 0.98
    mom_bullish = (last['rsi'] > 51 and last['rsi'] > last['rsi_ema']) or lb_up
    mom_bearish = (last['rsi'] < 44 and last['rsi'] < last['rsi_ema']) or lb_down

    pattern, pattern_bull, _ = detect_candle_pattern(df_hourly)

    fvg_h = detect_fvg_zones(df_hourly, max_age=15)
    ob_h = detect_order_blocks(df_hourly, max_age=15)
    inside_zone_h = False
    for z in fvg_h + ob_h:
        if last['high'] >= z.bottom and last['low'] <= z.top:
            inside_zone_h = True
            break

    daily_bullish = daily_ctx['trend'] == 'BULLISH' and daily_ctx['net_score'] > 20
    daily_bearish = daily_ctx['trend'] == 'BEARISH' and daily_ctx['net_score'] < -20

    can_long = daily_bullish
    can_short = daily_bearish

    if daily_ctx['inside_zone'] and daily_ctx['zone_bullish'] is not None:
        can_long = can_long and daily_ctx['zone_bullish']
        can_short = can_short and not daily_ctx['zone_bullish']

    if daily_ctx['recent_ssl']:
        can_long = True
    if daily_ctx['recent_bsl']:
        can_short = True

    long_signal = False
    short_signal = False
    reason = ""

    if can_long and inside_zone_h and mom_bullish and pattern is not None and pattern_bull:
        long_signal = True
        reason = f"Hourly {pattern} inside zone + daily bullish"
    elif can_long and daily_ctx['recent_ssl'] and lb_up and last['close'] > last['ema20']:
        long_signal = True
        reason = "Daily SSL sweep + hourly LB up"
    elif can_short and inside_zone_h and mom_bearish and pattern is not None and not pattern_bull:
        short_signal = True
        reason = f"Hourly {pattern} inside zone + daily bearish"
    elif can_short and daily_ctx['recent_bsl'] and lb_down and last['close'] < last['ema20']:
        short_signal = True
        reason = "Daily BSL sweep + hourly LB down"

    atr_val = last['atr']
    if long_signal:
        stop_loss = last['low'] - atr_val * 0.5
        take_profit = last['close'] + atr_val * 1.5
        risk = last['close'] - stop_loss
        reward = take_profit - last['close']
        rr = reward / risk if risk > 0 else 0
        return {
            'signal': 'LONG',
            'valid': rr >= 1.5,
            'reason': reason,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'rr': rr,
            'risk_pct': (risk / last['close']) * 100
        }
    elif short_signal:
        stop_loss = last['high'] + atr_val * 0.5
        take_profit = last['close'] - atr_val * 1.5
        risk = stop_loss - last['close']
        reward = last['close'] - take_profit
        rr = reward / risk if risk > 0 else 0
        return {
            'signal': 'SHORT',
            'valid': rr >= 1.5,
            'reason': reason,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'rr': rr,
            'risk_pct': (risk / last['close']) * 100
        }
    else:
        return {'signal': 'NO TRADE', 'valid': False, 'reason': 'No valid setup'}

# ------------------------------------------------------------
# 12. SIMPLIFIED CHART (only candles, LB curve, RSI panel)
# ------------------------------------------------------------
def plot_simple_chart(df, title="Price Action (Candles + LB + RSI)"):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                   gridspec_kw={"height_ratios": [3, 1]},
                                   sharex=True)
    x = np.arange(len(df))
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    width = 0.6
    up_color = "#26a69a"
    down_color = "#ef5350"

    # Candles
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

    # LB curve
    ax.plot(x, df["lb_crv"], color="gray", alpha=0.8, linewidth=1.2, label="LB Curve")
    ax.legend(loc="upper left")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    # RSI panel
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
        ax2.set_xticklabels(df.index.strftime("%Y-%m-%d %H:%M")[::max(1, len(x)//10)],
                            rotation=45, fontsize=8)

    plt.tight_layout()
    return fig

# ------------------------------------------------------------
# 13. STREAMLIT UI
# ------------------------------------------------------------
st.sidebar.header("Multi‑Timeframe Settings")
ticker = st.sidebar.text_input("Ticker", "AAPL")

# Daily data (1 year)
start_date_daily = datetime.today() - timedelta(days=365)
df_daily = load_data(ticker, start_date_daily, "1d")
if df_daily is None or df_daily.empty:
    st.error("Could not load daily data.")
    st.stop()

# Hourly data (last 30 days)
start_date_hourly = datetime.today() - timedelta(days=30)
df_hourly = load_data(ticker, start_date_hourly, "1h")
if df_hourly is None or df_hourly.empty:
    st.warning("Hourly data not available. Using 4H instead.")
    df_hourly = load_data(ticker, start_date_hourly, "4h")
    if df_hourly is None:
        st.error("No intraday data available.")
        st.stop()

with st.spinner("Analyzing daily context..."):
    daily_ctx = analyze_daily_context(df_daily)

with st.spinner("Computing hourly entry signal..."):
    hourly_signal = get_hourly_signal(df_hourly, daily_ctx)

# DASHBOARD
st.subheader(f"📊 SMC Dashboard – {ticker}")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔵 Daily Context (1D)")
    st.metric("Trend", daily_ctx['trend'])
    st.metric("Regime Score", f"{daily_ctx['net_score']} pts", delta=f"{daily_ctx['regime']}")
    st.write(f"**Inside Zone:** {daily_ctx['inside_zone']} (Bullish: {daily_ctx['zone_bullish']})")
    st.write(f"**Active FVG/OB:** {daily_ctx['fvg_count']} / {daily_ctx['ob_count']}")
    st.write(f"**Recent Sweeps:** BSL={daily_ctx['recent_bsl']}, SSL={daily_ctx['recent_ssl']}")
    st.write(f"**Last Pattern:** {daily_ctx['pattern']} ({'Bullish' if daily_ctx['pattern_bull'] else 'Bearish'})")
    st.write(f"**Momentum:** Bull={daily_ctx['mom_bullish']}, Bear={daily_ctx['mom_bearish']}")
    st.write(f"**LB status:** Up={daily_ctx['lb_up']}, Down={daily_ctx['lb_down']}")
    st.write(f"**BOS/CHoCH:** BOS↑={daily_ctx['bos_up_count']}, BOS↓={daily_ctx['bos_dn_count']}, CHoCH↑={daily_ctx['cho_up_count']}, CHoCH↓={daily_ctx['cho_dn_count']}")

with col2:
    st.markdown("### 🟢 Hourly Entry Signal (1H)")
    sig = hourly_signal
    if sig['signal'] == 'NO TRADE':
        st.error(f"❌ {sig['signal']}")
        st.write(f"**Reason:** {sig['reason']}")
    else:
        if sig['valid']:
            st.success(f"✅ VALID {sig['signal']} ENTRY")
        else:
            st.warning(f"⚠️ {sig['signal']} SIGNAL – poor risk/reward")
        st.write(f"**Trigger:** {sig['reason']}")
        st.write(f"**Risk/Reward:** {sig['rr']:.2f}")
        st.write(f"**Risk %:** {sig['risk_pct']:.2f}%")
        st.write(f"**Stop Loss:** {sig['stop_loss']:.2f}")
        st.write(f"**Take Profit:** {sig['take_profit']:.2f}")
        if sig['valid']:
            st.info("📌 Recommendation: **Consider taking the trade** with daily context aligned.")
        else:
            st.info("⛔ Recommendation: **Avoid trade** – wait for better R/R or stronger hourly signal.")

# SIMPLIFIED CHART (candles + LB + RSI)
st.subheader("📉 Hourly Price Action (Candles + LB Curve + RSI)")
fig = plot_simple_chart(df_hourly.tail(200), title=f"{ticker} – Hourly (last 200 bars)")
st.pyplot(fig)

# Optional: show mini daily chart
if st.checkbox("Show daily chart (last 100 days)"):
    fig_daily = plot_simple_chart(df_daily.tail(100), title=f"{ticker} – Daily (last 100 bars)")
    st.pyplot(fig_daily)
