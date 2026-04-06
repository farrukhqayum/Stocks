import streamlit as st
st.set_page_config(page_title="SMC Dashboard", layout="wide")
st.title("📈 SMART MONEY CONCEPTS (SMC)")

from imports import *
yf.pdr_override()
yf.cache.clear() 

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

def compute_atr(df, length=14):
    """Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    return tr.rolling(length).mean()

def compute_lb_curve(df, lblen=10):
    """Calculate LB Curve without lookahead bias - uses previous bars only"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    lb = np.zeros(len(df))
    lb[0] = close[0]
    
    for i in range(1, len(df)):
        start = max(0, i - lblen + 1)
        
        # Use PREVIOUS bar's highest/lowest only (no lookahead)
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
    lb_ema = lb_series.ewm(span=lblen, adjust=False).mean()
    return lb_ema

@st.cache_data
def load_data(ticker, start_date, interval):
    end_date = datetime.today().strftime("%Y-%m-%d")
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
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
    df['atr'] = compute_atr(df, 14)
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

class OBZone:
    def __init__(self, top, bottom, start_idx, is_bull):
        self.top = top
        self.bottom = bottom
        self.start_idx = start_idx
        self.is_bull = is_bull
        self.is_mitigated = False
        self.mitigated_idx = None
        self.touched = False

# ---------------------------------------------------------
# FVG DETECTION
# ---------------------------------------------------------

def detect_fvg_zones(df, max_age=25, fail_window=5):
    """
    Detect FVG zones with ATR-based gap tolerance
    Only mitigate when CLOSE breaches the zone
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    open_ = df['open'].values
    
    # Calculate ATR for gap tolerance
    atr = df['atr'].values
    min_gap = atr * 0.1  # 10% of ATR minimum gap
    
    zones = []
    
    for i in range(len(df)):
        if i >= 2:
            # Bullish FVG with gap requirement
            is_fvg_up = (low[i] > high[i-2] + min_gap[i])
            # Bearish FVG with gap requirement
            is_fvg_dn = (high[i] < low[i-2] - min_gap[i])
            
            if is_fvg_up:
                zones.append(FVGZone(high[i-2], low[i], i-2, True))
            
            if is_fvg_dn:
                zones.append(FVGZone(high[i], low[i-2], i-2, False))
        
        # Update existing zones
        to_delete = []
        for j, z in enumerate(zones):
            age = i - z.start_idx
            
            # Check failure within window (requires 2 closes)
            failed = False
            if age <= fail_window and i >= 1:
                if z.is_bull:
                    if close[i] < z.bottom and close[i-1] < z.bottom:
                        failed = True
                else:
                    if close[i] > z.top and close[i-1] > z.top:
                        failed = True
            
            # Mitigation: CLOSE must be beyond zone (not just touch)
            if not z.is_mitigated and not failed:
                if z.is_bull and close[i] < z.bottom:
                    z.is_mitigated = True
                    z.mitigated_idx = i
                if not z.is_bull and close[i] > z.top:
                    z.is_mitigated = True
                    z.mitigated_idx = i
            
            # Track touches (for potential future mitigation)
            if not z.touched and (z.bottom < close[i] < z.top):
                z.touched = True
            
            if failed or age > max_age:
                to_delete.append(j)
        
        for j in reversed(to_delete):
            del zones[j]
    
    return [z for z in zones if not z.is_mitigated]

# ---------------------------------------------------------
# ORDER BLOCK DETECTION
# ---------------------------------------------------------

def detect_order_blocks(df, max_age=25, fail_window=5):
    """
    Detect Order Blocks (3-candle and gap-based)
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    open_ = df['open'].values
    volume = df['volume'].values if 'volume' in df.columns else None
    
    # Volume SMA for confirmation
    if volume is not None:
        vol_sma = pd.Series(volume).rolling(5).mean().values
        vol_sma = np.nan_to_num(vol_sma, nan=np.median(volume))
    else:
        vol_sma = np.ones(len(df)) * 1000000
    
    zones = []
    
    for i in range(len(df)):
        if i >= 2:
            vol_ok = volume[i] > vol_sma[i] * 0.6 if volume is not None else True
            
            # 3-candle Bullish OB (displacement up)
            displacement_up = (close[i] > high[i-1] and 
                              close[i] > open_[i] and 
                              low[i-1] < low[i-2])
            
            if displacement_up and vol_ok:
                zones.append(OBZone(high[i-1], low[i-1], i-1, True))
            
            # 3-candle Bearish OB (displacement down)
            displacement_down = (close[i] < low[i-1] and 
                                close[i] < open_[i] and 
                                high[i-1] > high[i-2])
            
            if displacement_down and vol_ok:
                zones.append(OBZone(high[i-1], low[i-1], i-1, False))
            
            # Gap-based Bullish OB
            gap_up = (open_[i] > high[i-1] and close[i] > open_[i])
            if gap_up and vol_ok:
                zones.append(OBZone(open_[i], low[i-1], i-1, True))
            
            # Gap-based Bearish OB
            gap_down = (open_[i] < low[i-1] and close[i] < open_[i])
            if gap_down and vol_ok:
                zones.append(OBZone(high[i-1], open_[i], i-1, False))
        
        # Update existing zones
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
            "strong_bearish": False,
            "turning_point": False,
            "turning_code": None
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

    # Turning Point Engine
    turning_point = False
    turning_code = None

    if last_pattern is not None and pattern_idx is not None and not expired and not rejected:
        body_last = abs(c[-1] - o[-1])
        range_last = h[-1] - l[-1]
        wick_high_last = h[-1] - max(o[-1], c[-1])
        wick_low_last = min(o[-1], c[-1]) - l[-1]

        # Bearish pattern → look for bullish reversal
        if pattern_bull is False:
            if (c[-1] > o[-1]) and (wick_low_last > body_last * 1.2):
                turning_point = True
                turning_code = "▲ Rejecting Lows"

            if (n >= 2 and
                c[-2] < o[-2] and
                c[-1] > o[-1] and
                o[-1] <= c[-2] and
                c[-1] >= o[-2]):
                turning_point = True
                turning_code = "▲ Bullish Shift"

            if (c[-1] > o[-1]) and (body_last > 0.55 * range_last):
                turning_point = True
                turning_code = "▲ Bullish Drive"

        # Bullish pattern → look for bearish reversal
        if pattern_bull is True:
            if (c[-1] < o[-1]) and (wick_high_last > body_last * 1.2):
                turning_point = True
                turning_code = "▼ Rejecting Highs"

            if (n >= 2 and
                c[-2] > o[-2] and
                c[-1] < o[-1] and
                o[-1] >= c[-2] and
                c[-1] <= o[-2]):
                turning_point = True
                turning_code = "▼ Bearish Shift"

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
# CHART FUNCTION WITH BOS/CHoCH
# ---------------------------------------------------------

def draw_smc_box(ax, df, fvg_zones, ob_zones):
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

    # Count zones by type
    bull_fvg = [z for z in fvg_zones if z.is_bull]
    bear_fvg = [z for z in fvg_zones if not z.is_bull]
    bull_ob = [z for z in ob_zones if z.is_bull]
    bear_ob = [z for z in ob_zones if not z.is_bull]

    def yn(flag): return "green" if flag else "red"

    zone_lines = [
        ("ZONES:", "gray"),
        (f"  BULL FVG: {len(bull_fvg)}", "green" if bull_fvg else "red"),
        (f"  BEAR FVG: {len(bear_fvg)}", "red" if bear_fvg else "green"),
        (f"  BULL OB: {len(bull_ob)}", "green" if bull_ob else "red"),
        (f"  BEAR OB: {len(bear_ob)}", "red" if bear_ob else "green"),
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
            fontsize=7,
            color=color,
            ha="left", va="top"
        )
        y -= 0.03
        
def check_rejection(df, i, swing_level, is_high=True):
    """Check if price rejected from swing level (Pine style)"""
    if i < 1:
        return False
    
    close_curr = df['close'].iloc[i]
    close_prev = df['close'].iloc[i-1]
    
    if is_high:
        # Rejection: prev bar above, current bar below
        return close_prev > swing_level and close_curr < swing_level
    else:
        # Rejection: prev bar below, current bar above
        return close_prev < swing_level and close_curr > swing_level
        
def detect_swings_for_chart(df, left_bars=10, right_bars=4):
    """Match Pine's ta.pivothigh and ta.pivotlow exactly"""
    high = df['high'].values
    low = df['low'].values
    
    swing_highs = []
    swing_lows = []
    
    for i in range(left_bars, len(df) - right_bars):
        # Pivot high: highest in left_bars to the left AND right_bars to the right
        is_pivot_high = True
        for k in range(1, left_bars + 1):
            if high[i] <= high[i - k]:
                is_pivot_high = False
                break
        if is_pivot_high:
            for k in range(1, right_bars + 1):
                if high[i] <= high[i + k]:
                    is_pivot_high = False
                    break
        
        if is_pivot_high:
            swing_highs.append({'idx': i, 'price': high[i]})
        
        # Pivot low
        is_pivot_low = True
        for k in range(1, left_bars + 1):
            if low[i] >= low[i - k]:
                is_pivot_low = False
                break
        if is_pivot_low:
            for k in range(1, right_bars + 1):
                if low[i] >= low[i + k]:
                    is_pivot_low = False
                    break
        
        if is_pivot_low:
            swing_lows.append({'idx': i, 'price': low[i]})
    
    return swing_highs, swing_lows

def plotchart(df, fvg_zones, ob_zones, title="SMC FVG View", glong = False, gshort = False, elong = False, eshort = False):
    df = df.copy()
    if "rsi" not in df.columns:
        df["rsi"] = compute_rsi(df["close"], 14)
    if "rsi_ema" not in df.columns:
        df["rsi_ema"] = ema(df["rsi"], 14)
    if "atr" not in df.columns:
        df["atr"] = compute_atr(df, 14)

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

    # CANDLE PLOTTING
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

    # INDICATORS
    ax.plot(x, df["lb_crv"], color="gray", alpha=0.75, linewidth=1.2)
    ax.plot(x, df["ema20"], color="yellow", alpha=0.75, linewidth=1)
    ax.plot(x, df["ema50"], color="red", alpha=0.75, linewidth=1)

    # FVG ZONES (Teal/Blue, Dashed)
    last_idx = len(df) - 1
    
    for z in fvg_zones:
        rect_x = z.start_idx - 0.5
    
        if z.is_mitigated:
            end_idx = z.mitigated_idx
        else:
            end_idx = last_idx
    
        rect_width = (end_idx - z.start_idx) + 1
    
        color = "teal" if z.is_bull else "blue"
    
        ax.add_patch(Rectangle(
            (rect_x, z.bottom),
            rect_width,
            z.top - z.bottom,
            facecolor=color,
            alpha=0.07,
            edgecolor=color,
            linestyle="--",
            linewidth=1.5
        ))
    
    # ORDER BLOCKS (Green/Orange, Solid)
    for z in ob_zones:
        rect_x = z.start_idx - 0.5
    
        if z.is_mitigated:
            end_idx = z.mitigated_idx
        else:
            end_idx = last_idx
    
        rect_width = (end_idx - z.start_idx) + 1
    
        color = "green" if z.is_bull else "orange"
    
        ax.add_patch(Rectangle(
            (rect_x, z.bottom),
            rect_width,
            z.top - z.bottom,
            facecolor=color,
            alpha=0.07,
            edgecolor=color,
            linestyle="-",
            linewidth=1
        ))

    # SMC BOX
    draw_smc_box(ax, df, fvg_zones, ob_zones)

    # BOS/CHoCH DETECTION (Fixed - Matches Pine Logic)
    swing_highs, swing_lows = detect_swings_for_chart(df)
    close_vals = df['close'].values
    open_vals = df['open'].values
    high_vals = df['high'].values
    low_vals = df['low'].values
    atr_vals = df['atr'].values
    
    # Track last swing levels (Pine style)
    last_swing_high = None
    last_swing_low = None
    last_high_idx = None
    last_low_idx = None
    is_uptrend = False
    
    # Track if BOS was already plotted for this swing
    plotted_highs = set()
    plotted_lows = set()
    
    for i in range(len(df)):
        # Update last swing levels (only from confirmed swings)
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
        
        # Check for BOS up (Pine logic)
        if last_swing_high is not None and last_high_idx not in plotted_highs:
            # Valid BOS: close above swing high + ATR buffer
            bos_up_valid = close_vals[i] > last_swing_high + atr_vals[i] * 0.1
            
            # Check for rejection (price tried to break but failed)
            bos_up_rejected = False
            if i > 0 and bos_up_valid:
                # Rejection if previous bar closed above but current closed below
                if close_vals[i-1] > last_swing_high and close_vals[i] < last_swing_high:
                    bos_up_rejected = True
            
            # Confirmed if valid and not rejected
            bos_up_confirmed = bos_up_valid and not bos_up_rejected
            
            # Mitigated (actual break) - only plot when confirmed AND price held above
            if bos_up_confirmed and i >= 1:
                # Check if price has accepted above (low doesn't retrace below swing high)
                if low_vals[i] <= last_swing_high:
                    # Plot BOS/CHoCH
                    label_text = "CHoCH ↑" if is_uptrend else "BOS ↑"
                    break_col = 'lime'
                    
                    # Draw line from swing index to current bar
                    ax.plot([last_high_idx, i], [last_swing_high, last_swing_high], 
                           color=break_col, linestyle='--', linewidth=1.5, alpha=0.8)
                    
                    if '↑' in label_text:
                        y_offset = last_swing_high + (atr_vals[i] * 0.3)  # Above by 30% of ATR
                        va_position = 'bottom'
                        ax.text(i, y_offset, f"  {label_text}", 
                               fontsize=7, color=break_col, va=va_position, alpha=0.8, fontweight='bold')
                    else:
                        y_offset = last_swing_low - (atr_vals[i] * 0.3)  # Below by 30% of ATR
                        va_position = 'top'
                        ax.text(i, y_offset, f"  {label_text}", 
                               fontsize=7, color=break_col, va=va_position, alpha=0.8, fontweight='bold')
                    
                    plotted_highs.add(last_high_idx)
                    is_uptrend = True
                    # Don't reset last_swing_high immediately - Pine keeps it
        
        # Check for BOS down (Pine logic)
        if last_swing_low is not None and last_low_idx not in plotted_lows:
            bos_down_valid = close_vals[i] < last_swing_low - atr_vals[i] * 0.1
            
            bos_down_rejected = False
            if i > 0 and bos_down_valid:
                if close_vals[i-1] < last_swing_low and close_vals[i] > last_swing_low:
                    bos_down_rejected = True
            
            bos_down_confirmed = bos_down_valid and not bos_down_rejected
            
            if bos_down_confirmed and i >= 1:
                if high_vals[i] >= last_swing_low:
                    label_text = "CHoCH ↓" if not is_uptrend else "BOS ↓"
                    break_col = 'red'
                    
                    ax.plot([last_low_idx, i], [last_swing_low, last_swing_low], 
                           color=break_col, linestyle='--', linewidth=1.5, alpha=0.8)
                    ax.text(i, last_swing_low, f"  {label_text}", 
                           fontsize=7, color=break_col, va='top', alpha=0.8,
                           fontweight='bold')
                    
                    plotted_lows.add(last_low_idx)
                    is_uptrend = False  
                    last_swing_low = None

    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    # RSI PANEL
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

    # EXIT MARKERS
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

    # LEGEND
    legend_text = "■ EXIT LONG\n❌ EXIT SHORT\n🟢 BULL OB\n🟠 BEAR OB\n── BOS/CHoCH"
    ax.text(
        0.02, 0.02, legend_text,
        transform=ax.transAxes,
        fontsize=7, color="blue",
        ha="left", va="bottom",
        bbox=dict(facecolor="white", alpha=0.4, edgecolor="none", boxstyle="round,pad=0.3")
    )
    
    # PATTERN + REVERSAL ENGINE
    info = pine_candle_engine(df)
    
    # Draw candlestick pattern label
    plot_pattern_label(
        ax, df,
        info["pattern_idx"],
        info["last_pattern"],
        info["pattern_bull"],
        info["rejected"]
    )
    
    # REVERSAL DETECTION
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
    
    # DRAW REVERSAL TEXT INSIDE CHART
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

    # ENTRY MARKERS
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
        
    # DRAW TURNING POINT ABOVE BAR
    tp_flag = info["turning_point"]
    tp_code = info["turning_code"]
    
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
    """Update with proper signal logic"""
    
    df_slice = df_slice.copy()

    # Initialize signal columns
    df_slice["long_entry_sig"] = False
    df_slice["short_entry_sig"] = False
    df_slice["exit_long_sig"] = False
    df_slice["exit_short_sig"] = False

    # Track active positions
    in_long = False
    in_short = False

    for i in range(50, len(df_slice)):
        row = df_slice.iloc[i]
        prev = df_slice.iloc[i - 1]

        close_last = row["close"]
        ema20_last = row["ema20"]
        ema50_last = row["ema50"]
        lb_last = row["lb_crv"]
        lb_prev = prev["lb_crv"]
        rsi_last = row["rsi"]
        rsi_ema_last = row["rsi_ema"]

        # LB conditions
        lb_up = close_last > lb_last * 1.02
        lb_down = close_last < lb_last * 0.98
        
        # EMA conditions
        ema_bullish = ema20_last > ema50_last
        ema_bearish = ema20_last < ema50_last
        
        # Momentum
        mom_bullish = (rsi_last >= 50 and rsi_last > rsi_ema_last) or (close_last > lb_last * 1.02)
        mom_bearish = (rsi_last <= 44 and rsi_last < rsi_ema_last) or (close_last < lb_last * 0.98)

        # Simple pattern detection (oversold/overbought with LB)
        has_bull_pattern = rsi_last < 35 and lb_down
        has_bear_pattern = rsi_last > 65 and lb_up

        # Context OK for entries
        bull_context_ok = ema_bullish and mom_bullish and lb_up
        bear_context_ok = ema_bearish and mom_bearish and lb_down

        # Entry signals
        long_entry = False
        short_entry = False
        
        if not in_short and bull_context_ok and has_bull_pattern:
            long_entry = True
            in_long = True
            in_short = False
        
        if not in_long and bear_context_ok and has_bear_pattern:
            short_entry = True
            in_short = True
            in_long = False

        # Exit logic
        exit_long = False
        exit_short = False
        
        if in_long:
            exit_long = (close_last < lb_last * 0.97) or (lb_last < lb_prev)
            if exit_long:
                in_long = False
        
        if in_short:
            exit_short = (close_last > lb_last * 1.05) or (lb_last > lb_prev)
            if exit_short:
                in_short = False

        # Write signals
        idx = df_slice.index[i]
        df_slice.loc[idx, "long_entry_sig"] = long_entry
        df_slice.loc[idx, "short_entry_sig"] = short_entry
        df_slice.loc[idx, "exit_long_sig"] = exit_long
        df_slice.loc[idx, "exit_short_sig"] = exit_short

    return df_slice

# ---------------------------------------------------------
# UI — TIMEFRAME, DATA LOADING, WINDOW MANAGEMENT
# ---------------------------------------------------------

st.sidebar.header("Settings")

# BASIC INPUTS
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

# SIGNAL TOGGLES
glong  = st.sidebar.checkbox("Show Long Entries", value=False)
gshort = st.sidebar.checkbox("Show Short Entries", value=False)
elong  = st.sidebar.checkbox("Show Long Exits", value=False)
eshort = st.sidebar.checkbox("Show Short Exits", value=False)

# TIMEFRAME CONFIG
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

# LOAD DATA
df = load_data(ticker, start_date, interval)

if df is None or df.empty:
    st.error("No data found.")
    st.stop()

# Detect zones
fvg_zones = detect_fvg_zones(df)
ob_zones = detect_order_blocks(df)
df = precompute_signals(df)

if first_load:
    st.session_state.window_start_idx = 0
    st.session_state.window_end_idx = len(df) - 1
    st.session_state.initialized = True
    
# SESSION STATE INIT
if "last_tf" not in st.session_state:
    st.session_state.last_tf = tf

if "window_start_idx" not in st.session_state:
    st.session_state.window_start_idx = 0

if "window_end_idx" not in st.session_state:
    st.session_state.window_end_idx = len(df) - 1

# RESET WINDOW ON TF CHANGE
if st.session_state.last_tf != tf:
    st.session_state.window_start_idx = 0
    st.session_state.window_end_idx = len(df) - 1
    st.session_state.last_tf = tf

# DATA SLICING
start_idx = st.session_state.window_start_idx
end_idx   = st.session_state.window_end_idx
start_idx = max(0, min(start_idx, len(df) - 1))
end_idx   = max(0, min(end_idx, len(df) - 1))

if first_load:
    df_slice = df.copy()
else:
    df_slice = df.iloc[start_idx : end_idx + 1 : 1]
    fvg_zones = detect_fvg_zones(df_slice)
    ob_zones = detect_order_blocks(df_slice)

# Check active positions
long_events = df_slice[["long_entry_sig", "exit_long_sig"]].any(axis=1)
if long_events.any():
    last_long_idx = df_slice[long_events].index[-1]
    last_long_row = df_slice.loc[last_long_idx]
    long_active = bool(last_long_row["long_entry_sig"]) and not bool(last_long_row["exit_long_sig"])
else:
    long_active = False

short_events = df_slice[["short_entry_sig", "exit_short_sig"]].any(axis=1)
if short_events.any():
    last_short_idx = df_slice[short_events].index[-1]
    last_short_row = df_slice.loc[last_short_idx]
    short_active = bool(last_short_row["short_entry_sig"]) and not bool(last_short_row["exit_short_sig"])
else:
    short_active = False

last = df_slice.iloc[-1]
long_entry  = bool(last["long_entry_sig"])
short_entry = bool(last["short_entry_sig"])
exit_long   = bool(last["exit_long_sig"])
exit_short  = bool(last["exit_short_sig"])

c1, c2, c3, c4 = st.columns(4)

with c1:
    if long_entry and not long_active and not short_entry:
        st.success("📈 LONG ENTRY/HOLD")
    elif long_active:
        st.info("🟢 LONG ACTIVE")
    else:
        st.info("—")

with c2:
    if exit_long and long_active:
        st.warning("🟡 EXIT LONG")
    else:
        st.info("—")

with c3:
    if short_entry and not short_active and not long_entry:
        st.error("📉 SHORT ENTRY/HOLD")
    elif short_active:
        st.info("🔴 SHORT ACTIVE")
    else:
        st.info("—")

with c4:
    if exit_short and short_active:
        st.warning("❌ EXIT SHORT")
    else:
        st.info("—")
        
# NAVIGATION BUTTONS
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

# Filter visible zones
visible_fvg = []
for z in fvg_zones:
    if z.start_idx <= end_idx:
        if not z.is_mitigated or (z.mitigated_idx and z.mitigated_idx >= start_idx):
            visible_fvg.append(z)

visible_ob = []
for z in ob_zones:
    if z.start_idx <= end_idx:
        if not z.is_mitigated or (z.mitigated_idx and z.mitigated_idx >= start_idx):
            visible_ob.append(z)

# DRAW CHART
fig = plotchart(
    df_slice,
    visible_fvg,
    visible_ob,
    title=f"{ticker} — {tf} SMC Regime View (FVG + OB + BOS)",
    glong=glong,
    gshort=gshort,
    elong=elong,
    eshort=eshort
)
st.pyplot(fig)
