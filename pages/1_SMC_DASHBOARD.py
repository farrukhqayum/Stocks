# =====================================================================
# SMART MONEY CONCEPTS – Exact Pine Logic + Full Dashboard + Charts
# =====================================================================

import streamlit as st

st.set_page_config(page_title="SMART MONEY CONCEPTS", layout="wide")
st.title("📈 SMART MONEY CONCEPTS - 1D/1H")
from imports import *
# ------------------------------------------------------------
# 1. INDICATORS (exact Pine formulas)
# ------------------------------------------------------------
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def atr(df, length=14):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high-low, abs(high-close.shift()), abs(low-close.shift())], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def lb_curve(df, lblen=10):
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
    return pd.Series(lb, index=df.index).ewm(span=lblen, adjust=False).mean()

# ------------------------------------------------------------
# 2. ZONE CLASSES (exact Pine properties)
# ------------------------------------------------------------
class Zone:
    def __init__(self, top, bottom, startBar, isBull, isOb, col):
        self.top = top
        self.bottom = bottom
        self.startBar = startBar
        self.isBull = isBull
        self.isOb = isOb
        self.baseCol = col
        self.isMitigated = False
        self.taps = 0

# ------------------------------------------------------------
# 3. GLOBAL STATE (Pine 'var' variables)
# ------------------------------------------------------------
class PineState:
    def __init__(self):
        self.allZones = []
        self.smc_bullish = False
        self.smc_bearish = False
        self.mom_bullish = False
        self.mom_bearish = False
        self.pattern_bullish = False
        self.pattern_rejected = False
        self.patternStoredLow = None
        self.patternStoredHigh = None
        self.patternStoredBar = None
        self.pattern_status = "None"
        self.lastSwingHigh = None
        self.lastSwingLow = None
        self.prevSwingHigh = None
        self.prevSwingLow = None
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
        self.lastPatternBar = 0
        self.patternStoredClose = None
        self.patternName = "None"
        self.patternIsBullish = False
        self.patternStoredHighValue = None
        self.patternStoredLowValue = None
        self.patternStoredBarIndex = None
        self.btLbl = None
        self.btIdx = None
        self.btHi = None
        self.btLo = None
        self.btBull = False
        self.lastSweepBar = None
        self.sslRejected = False
        self.bslRejected = False
        self.sslRejectionBar = None
        self.bslRejectionBar = None
        self.activeSSL = False
        self.activeSSLBar = None
        self.activeBSL = False
        self.activeBSLBar = None
        self.lastTurningBar = None
        self.lastTurningReason = ""
        self.lastTurningPrice = None
        self.lastTurningActualPrice = None
        self.lastTurningColor = None
        self.lastTurningStyle = "down"
        self.lastBOSUpBar = None
        self.lastBOSUpPrice = None
        self.lastBOSDnBar = None
        self.lastBOSDnPrice = None
        self.inLong = False
        self.inShort = False
        self.entryStopLong = None
        self.entryStopShort = None
        self.entryPriceLong = None
        self.entryPriceShort = None
        self.activeLongSL = None
        self.activeLongTP = None
        self.activeShortSL = None
        self.activeShortTP = None
        self.tradeStartBar = -1
        self.insideZonePrev = False
        self.lastZoneBullish = False
        self.lastZoneBearish = False
        # Additional for dashboard
        self.last_close = None
        self.last_lb_crv = None
        self.last_rsi = None
        self.last_rsi_ema = None
        self.last_ema20 = None
        self.last_ema50 = None
        self.last_high = None
        self.last_low = None
        self.last_open = None
        self.bos_up_list = []   # (swing_idx, break_idx, price)
        self.bos_dn_list = []
        self.cho_up_list = []
        self.cho_dn_list = []
        self.turning_points = []  # (idx, reason, price, style)

# ------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ------------------------------------------------------------
def get_label_size(lbl_size):
    return 8

def get_style(s):
    return "--" if s == "Dashed" else ":" if s == "Dotted" else "-"

# ------------------------------------------------------------
# 5. LOAD DATA (cached)
# ------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data(ticker, start_date, interval):
    end = datetime.today().strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start_date, end=end, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return None
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df = df.dropna(subset=["open","high","low","close"]).astype(float)
    df['ema20'] = ema(df.close, 20)
    df['ema50'] = ema(df.close, 50)
    df['ema200'] = ema(df.close, 200)
    df['rsi'] = rsi(df.close, 14)
    df['rsi_ema'] = ema(df['rsi'], 14)
    df['atr'] = atr(df, 14)
    df['lb_crv'] = lb_curve(df, 10)
    df = df.bfill().ffill()
    return df

# ------------------------------------------------------------
# 6. CORE PINE LOGIC (processes one bar, updates state)
# ------------------------------------------------------------
def process_bar(df, i, state, inputs):
    """Execute all Pine calculations for bar i, update state, return dashboard dict"""
    # Current values
    open_ = df['open'].iloc[i]
    high = df['high'].iloc[i]
    low = df['low'].iloc[i]
    close = df['close'].iloc[i]
    volume = df['volume'].iloc[i] if 'volume' in df.columns else 0
    ema20 = df['ema20'].iloc[i]
    ema50 = df['ema50'].iloc[i]
    ema200 = df['ema200'].iloc[i]
    rsi_val = df['rsi'].iloc[i]
    rsi_ema_val = df['rsi_ema'].iloc[i]
    atr_val = df['atr'].iloc[i]
    lb_crv = df['lb_crv'].iloc[i]

    # Previous bar values
    if i > 0:
        open_prev = df['open'].iloc[i-1]
        high_prev = df['high'].iloc[i-1]
        low_prev = df['low'].iloc[i-1]
        close_prev = df['close'].iloc[i-1]
        volume_prev = df['volume'].iloc[i-1] if 'volume' in df.columns else 0
        ema20_prev = df['ema20'].iloc[i-1]
        ema50_prev = df['ema50'].iloc[i-1]
        ema200_prev = df['ema200'].iloc[i-1]
        rsi_prev = df['rsi'].iloc[i-1]
        rsi_ema_prev = df['rsi_ema'].iloc[i-1]
        atr_prev = df['atr'].iloc[i-1]
        lb_crv_prev = df['lb_crv'].iloc[i-1]
    else:
        open_prev = open_
        high_prev = high
        low_prev = low
        close_prev = close
        volume_prev = volume
        ema20_prev = ema20
        ema50_prev = ema50
        ema200_prev = ema200
        rsi_prev = rsi_val
        rsi_ema_prev = rsi_ema_val
        atr_prev = atr_val
        lb_crv_prev = lb_crv

    # Volume SMA
    vol5 = df['volume'].rolling(5).mean().iloc[i] if i>=4 else volume
    volConfirmBull = volume > vol5
    volConfirmBear = volume > vol5
    fvgVolOK = volume > vol5 * 0.6

    # ---------- Swings (pivots) ----------
    left = inputs['swing_l']
    right = inputs['swing_r']
    if i >= left and i < len(df)-right:
        # Pivot high
        is_high = True
        for k in range(1, left+1):
            if high <= df['high'].iloc[i-k]:
                is_high = False
                break
        if is_high:
            for k in range(1, right+1):
                if high <= df['high'].iloc[i+k]:
                    is_high = False
                    break
        if is_high:
            state.prevSwingHigh = state.lastSwingHigh
            state.lastSwingHigh = high
            state.last_hi = high
            state.last_hi_idx = i
        # Pivot low
        is_low = True
        for k in range(1, left+1):
            if low >= df['low'].iloc[i-k]:
                is_low = False
                break
        if is_low:
            for k in range(1, right+1):
                if low >= df['low'].iloc[i+k]:
                    is_low = False
                    break
        if is_low:
            state.prevSwingLow = state.lastSwingLow
            state.lastSwingLow = low
            state.last_lo = low
            state.last_lo_idx = i

    madeHL = (state.lastSwingLow is not None and state.prevSwingLow is not None and 
              state.lastSwingLow > state.prevSwingLow)
    madeLH = (state.lastSwingHigh is not None and state.prevSwingHigh is not None and 
              state.lastSwingHigh < state.prevSwingHigh)

    # Sweep detection
    sweepBuySide = (state.lastSwingHigh is not None and high > state.lastSwingHigh and close < state.lastSwingHigh)
    sweepSellSide = (state.lastSwingLow is not None and low < state.lastSwingLow and close > state.lastSwingLow)

    # ---------- FVG and OB detection ----------
    min_gap = atr_val * 0.1
    if i >= 2:
        fvgUp3 = (low > df['high'].iloc[i-2] + min_gap) and fvgVolOK
        fvgDn3 = (high < df['low'].iloc[i-2] - min_gap) and fvgVolOK
        if fvgUp3:
            col = inputs['fvgBull']
            state.allZones.append(Zone(df['high'].iloc[i-2], low, i-2, True, False, col))
        if fvgDn3:
            col = inputs['fvgBear']
            state.allZones.append(Zone(high, df['low'].iloc[i-2], i-2, False, False, col))

    if i >= 2:
        displacementUp = (close > high_prev and close > open_)
        displacementDn = (close < low_prev and close < open_)
        bullOB3 = displacementUp and (low_prev < df['low'].iloc[i-2]) and volConfirmBull
        bearOB3 = displacementDn and (high_prev > df['high'].iloc[i-2]) and volConfirmBear
        if bullOB3:
            state.allZones.append(Zone(high_prev, low_prev, i, True, True, inputs['obBull']))
        if bearOB3:
            state.allZones.append(Zone(high_prev, low_prev, i, False, True, inputs['obBear']))
        gapUpOB = (open_ > high_prev and close > open_)
        gapDnOB = (open_ < low_prev and close < open_)
        if gapUpOB:
            state.allZones.append(Zone(open_, low_prev, i, True, True, inputs['obBull']))
        if gapDnOB:
            state.allZones.append(Zone(high_prev, open_, i, False, True, inputs['obBear']))

    # ---------- Zone aging and mitigation ----------
    maxAge = inputs['maxAge']
    failWindow = inputs['failWindow']
    closeMitigate = inputs['closeMitigate']
    maxTaps = 2
    to_remove = []
    for idx, z in enumerate(state.allZones):
        age = i - z.startBar
        failed = False
        if age <= failWindow and i >= 1:
            if z.isBull:
                if close < z.bottom and close_prev < z.bottom:
                    failed = True
            else:
                if close > z.top and close_prev > z.top:
                    failed = True
        if not z.isMitigated:
            bullBroken = z.isBull and (close < z.bottom if closeMitigate else low < z.bottom)
            bearBroken = (not z.isBull) and (close > z.top if closeMitigate else high > z.top)
            if high > z.bottom and low < z.top:
                z.taps += 1
            if bullBroken or bearBroken or z.taps > maxTaps:
                z.isMitigated = True
        if age > maxAge or failed:
            to_remove.append(idx)
    for idx in reversed(to_remove):
        del state.allZones[idx]

    # ---------- Zone awareness ----------
    insideZone = False
    nearZone = False
    nearZoneBullish = False
    nearZoneBearish = False
    zone_distance = None
    for z in state.allZones:
        if z.isMitigated:
            continue
        if high >= z.bottom and low <= z.top:
            insideZone = True
            state.lastZoneBullish = z.isBull
            state.lastZoneBearish = not z.isBull
            break
    if not insideZone:
        for z in state.allZones:
            if z.isMitigated:
                continue
            dist_to_top = abs(close - z.top)
            dist_to_btm = abs(close - z.bottom)
            if dist_to_top / close * 100 < 3 or dist_to_btm / close * 100 < 3:
                nearZone = True
                nearZoneBullish = z.isBull
                nearZoneBearish = not z.isBull
                zone_distance = min(dist_to_top, dist_to_btm)
                break

    retestOccurred = (not state.insideZonePrev and insideZone)
    breakoutOccurred = (state.insideZonePrev and not insideZone)
    state.insideZonePrev = insideZone

    # ---------- BOS/CHoCH ----------
    if state.lastSwingHigh is not None:
        bos_up_raw = high > state.lastSwingHigh and close > high_prev
        bos_up_valid = bos_up_raw and close > state.lastSwingHigh + atr_val * 0.1
        bos_up_rejected = bos_up_valid and close_prev > state.lastSwingHigh and close < state.lastSwingHigh
        bos_up_confirmed = bos_up_valid and not bos_up_rejected
        if bos_up_confirmed and low <= state.lastSwingHigh:
            if state.is_uptrend is None or state.is_uptrend:
                state.bos_up_list.append((state.last_hi_idx, i, state.lastSwingHigh))
            else:
                state.cho_up_list.append((state.last_hi_idx, i, state.lastSwingHigh))
            state.is_uptrend = True
            state.smc_bullish = True
            state.smc_bearish = False
    if state.lastSwingLow is not None:
        bos_dn_raw = low < state.lastSwingLow and close < low_prev
        bos_dn_valid = bos_dn_raw and close < state.lastSwingLow - atr_val * 0.1
        bos_dn_rejected = bos_dn_valid and close_prev < state.lastSwingLow and close > state.lastSwingLow
        bos_dn_confirmed = bos_dn_valid and not bos_dn_rejected
        if bos_dn_confirmed and high >= state.lastSwingLow:
            if state.is_uptrend is None or not state.is_uptrend:
                state.bos_dn_list.append((state.last_lo_idx, i, state.lastSwingLow))
            else:
                state.cho_dn_list.append((state.last_lo_idx, i, state.lastSwingLow))
            state.is_uptrend = False
            state.smc_bullish = False
            state.smc_bearish = True

    # Early structure flip
    internalBullBOS = (state.lastSwingHigh is not None and close > state.lastSwingHigh)
    internalBearBOS = (state.lastSwingLow is not None and close < state.lastSwingLow)
    if internalBullBOS:
        state.smc_early_bull = True
        state.smc_early_bear = False
    if internalBearBOS:
        state.smc_early_bull = False
        state.smc_early_bear = True
    if state.smc_early_bull and (state.lastSwingLow is not None and low < state.lastSwingLow):
        state.smc_early_bull = False

    # ---------- Candlestick patterns ----------
    lb_up = close > lb_crv * 1.02
    lb_down = close < lb_crv * 0.98
    body0 = abs(close - open_)
    crange0 = high - low
    wickHigh = high - max(open_, close)
    wickLow = min(open_, close) - low
    safeCrange = crange0 if crange0 > 0 else 0.001
    bodyPct = body0 / safeCrange
    upperWickPct = wickHigh / safeCrange
    lowerWickPct = wickLow / safeCrange

    gravestone = (upperWickPct >= 0.50) and (lowerWickPct <= 0.20) and (bodyPct <= 0.40)
    shootingStar = (upperWickPct >= 0.45) and (lowerWickPct <= 0.20) and (bodyPct >= 0.30) and (bodyPct <= 0.70) and (close < open_)
    hammer = lowerWickPct >= 0.60 and upperWickPct <= 0.10 and bodyPct <= 0.30
    bullEngulf = (close > open_ and close_prev < open_prev and close > open_prev and open_ < close_prev and 
                  body0 >= abs(close_prev-open_prev)*1.02)
    bearEngulf = (close < open_ and close_prev > open_prev and close < open_prev and open_ > close_prev and 
                  body0 >= abs(close_prev-open_prev)*1.02)
    doji = body0 <= safeCrange * inputs['bodyThresh']
    dragonfly = doji and wickLow >= crange0 * inputs['wickThreshHigh'] and wickHigh <= crange0 * inputs['wickThreshLow']
    neutralDoji = doji and not gravestone and not dragonfly
    bullPierce = (close_prev < open_prev and open_ < close_prev and close > (open_prev+close_prev)/2 and close < open_prev)
    bearDark = (close_prev > open_prev and open_ > high_prev and close < (open_prev+close_prev)/2 and close > open_prev)
    isMorning = (i>=2 and df['close'].iloc[i-2] < df['open'].iloc[i-2] and 
                 abs(df['close'].iloc[i-1]-df['open'].iloc[i-1]) <= (df['high'].iloc[i-1]-df['low'].iloc[i-1])*0.3 and 
                 close > (df['open'].iloc[i-2]+df['close'].iloc[i-2])/2)
    isEvening = (i>=2 and df['close'].iloc[i-2] > df['open'].iloc[i-2] and 
                 abs(df['close'].iloc[i-1]-df['open'].iloc[i-1]) <= (df['high'].iloc[i-1]-df['low'].iloc[i-1])*0.3 and 
                 close < (df['open'].iloc[i-2]+df['close'].iloc[i-2])/2)
    tweezerBot = abs(low - low_prev) < 0.001 and close > open_
    tweezerTop = abs(high - high_prev) < 0.001 and close < open_
    bull_r3m = (i>=2 and close > open_ and close_prev > open_prev and df['close'].iloc[i-2] > df['open'].iloc[i-2] and 
                close > close_prev and close_prev > df['close'].iloc[i-2])
    bear_f3m = (i>=2 and close < open_ and close_prev < open_prev and df['close'].iloc[i-2] < df['open'].iloc[i-2] and 
                close < close_prev and close_prev < df['close'].iloc[i-2])

    last_pattern = None
    pattern_bull = None
    if bull_r3m:
        last_pattern, pattern_bull = "R 3 M", True
    elif bear_f3m:
        last_pattern, pattern_bull = "F 3 M", False
    elif isMorning:
        last_pattern, pattern_bull = "Morning Star", True
    elif isEvening:
        last_pattern, pattern_bull = "Evening Star", False
    elif bullEngulf:
        last_pattern, pattern_bull = "Bull Engulfing", True
    elif bearEngulf:
        last_pattern, pattern_bull = "Bear Engulfing", False
    elif bullPierce:
        last_pattern, pattern_bull = "Piercing", True
    elif bearDark:
        last_pattern, pattern_bull = "Dark Cloud", False
    elif tweezerBot:
        last_pattern, pattern_bull = "Tweezer Bottom", True
    elif tweezerTop:
        last_pattern, pattern_bull = "Tweezer Top", False
    elif hammer:
        last_pattern, pattern_bull = "Hammer", True
    elif shootingStar:
        last_pattern, pattern_bull = "Shooting Star", False
    elif gravestone:
        last_pattern, pattern_bull = "Gravestone", False
    elif dragonfly:
        last_pattern, pattern_bull = "Dragonfly", True
    elif doji and neutralDoji:
        last_pattern, pattern_bull = ("Bull Doji" if close > open_ else "Bear Doji"), (close > open_)

    # Pattern rejection (simplified: store pattern for later)
    if last_pattern is not None:
        state.last_pattern = last_pattern
        state.pattern_bullish = pattern_bull
        state.pattern_bar = i
        state.patternStoredLowValue = low
        state.patternStoredHighValue = high
        state.patternStoredBarIndex = i
        state.pattern_rejected = False
        # Check rejection after a few bars (not implemented in this loop)
        # For dashboard, we'll assume active for now

    # Momentum
    ema_bullish = ema20 > ema50
    ema_bearish = ema20 < ema50
    mom_bullish = (rsi_val > 51 and rsi_val > rsi_ema_val) or lb_up
    mom_bearish = (rsi_val < 44 and rsi_val < rsi_ema_val) or lb_down
    state.mom_bullish = mom_bullish
    state.mom_bearish = mom_bearish

    # ---------- Liquidity sweeps (strong BSL/SSL) ----------
    strongSSL = False
    strongBSL = False
    if i>=2:
        # BSL
        is_bsl = (df['high'].iloc[i-2] > df['high'].iloc[i-3] and df['high'].iloc[i-2] > df['high'].iloc[i-1]) if i>=3 else False
        if is_bsl and close < open_ and high > df['high'].iloc[i-2]:
            strongBSL = True
        # SSL
        is_ssl = (df['low'].iloc[i-2] < df['low'].iloc[i-3] and df['low'].iloc[i-2] < df['low'].iloc[i-1]) if i>=3 else False
        if is_ssl and close > open_ and low < df['low'].iloc[i-2]:
            strongSSL = True

    # Update active sweeps (simplified)
    if strongSSL:
        state.activeSSL = True
        state.activeSSLBar = i
    if strongBSL:
        state.activeBSL = True
        state.activeBSLBar = i
    if state.activeSSL and (state.lastSwingLow is not None and low < state.lastSwingLow):
        state.sslRejected = True
        state.activeSSL = False
    if state.activeBSL and (state.lastSwingHigh is not None and high > state.lastSwingHigh):
        state.bslRejected = True
        state.activeBSL = False

    # ---------- Turning points (pattern + sweep or BOS rejection) ----------
    turning_points = []
    if last_pattern is not None and not state.pattern_rejected and (i - state.pattern_bar) <= 5:
        if pattern_bull and strongSSL:
            turning_points.append((i, f"▲ {last_pattern}", low, "up"))
            state.turning_points.append((i, f"▲ {last_pattern}", low, "up"))
        elif not pattern_bull and strongBSL:
            turning_points.append((i, f"▼ {last_pattern}", high, "down"))
            state.turning_points.append((i, f"▼ {last_pattern}", high, "down"))
    # BOS rejection turning points
    for (swing_idx, break_idx, price) in state.bos_dn_list:
        if i - break_idx <= 3 and high > price:
            turning_points.append((i, "▲ BOS ↓ REJECTED", price, "up"))
            state.turning_points.append((i, "▲ BOS ↓ REJECTED", price, "up"))
    for (swing_idx, break_idx, price) in state.bos_up_list:
        if i - break_idx <= 3 and low < price:
            turning_points.append((i, "▼ BOS ↑ REJECTED", price, "down"))
            state.turning_points.append((i, "▼ BOS ↑ REJECTED", price, "down"))

    # ---------- Scoring and regime ----------
    bullScore = 0
    bearScore = 0
    if state.smc_bullish: bullScore += 30
    if state.smc_bearish: bearScore += 30
    if strongSSL: bullScore += 25
    if strongBSL: bearScore += 25
    if pattern_bull and not state.pattern_rejected: bullScore += 20
    if pattern_bull is False and not state.pattern_rejected: bearScore += 20
    if insideZone and state.smc_bullish: bullScore += 10
    if insideZone and state.smc_bearish: bearScore += 10
    if mom_bullish: bullScore += 15
    if mom_bearish: bearScore += 15
    netScore = bullScore - bearScore
    if netScore > 20: regime = "Bullish"
    elif netScore < -20: regime = "Bearish"
    else: regime = "Neutral"

    # ---------- Store last values for dashboard ----------
    state.last_close = close
    state.last_lb_crv = lb_crv
    state.last_rsi = rsi_val
    state.last_rsi_ema = rsi_ema_val
    state.last_ema20 = ema20
    state.last_ema50 = ema50
    state.last_high = high
    state.last_low = low
    state.last_open = open_

    # Dashboard data for this bar
    dashboard = {
        'liquidity': 'SSL' if strongSSL else 'BSL' if strongBSL else 'None',
        'sweep_status': 'ACTIVE' if (strongSSL or strongBSL) else '---',
        'pattern_text': f"{'↑' if pattern_bull else '↓'} {last_pattern}" if last_pattern else "No pattern",
        'pattern_status': 'Active' if last_pattern and (i - state.pattern_bar <= 5) else 'Expired',
        'momentum': 'UP ↑' if mom_bullish else 'DOWN ↓' if mom_bearish else '---',
        'struct': 'Bullish' if state.smc_bullish else 'Bearish' if state.smc_bearish else 'Neutral',
        'smc_concept': regime,
        'zone_event': 'Inside Bull Zone' if insideZone and state.lastZoneBullish else 'Inside Bear Zone' if insideZone and not state.lastZoneBullish else '---',
        'zone_dist': 'Inside zone' if insideZone else f"{zone_distance/close*100:.1f}% away" if zone_distance else '---',
        'bias': regime,
        'z_score': netScore,
        'signal': 'LONG' if (state.smc_bullish and insideZone and mom_bullish) else 'SHORT' if (state.smc_bearish and insideZone and mom_bearish) else 'NO TRADE'
    }
    return dashboard

# ------------------------------------------------------------
# 7. CHART PLOTTING (with toggles)
# ------------------------------------------------------------
def plot_full_chart(df, zones, bos_up, bos_dn, cho_up, cho_dn, turning_points,
                    start_idx_global, title, show_fvg=True, show_ob=True, show_bos=True, show_tp=True):
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

    visible_start = 0
    visible_end = len(df)-1

    # Zones
    if show_fvg or show_ob:
        for z in zones:
            if z.isMitigated:
                continue
            start = max(z.startBar - start_idx_global, visible_start)
            end = visible_end
            if start > visible_end:
                continue
            color = "teal" if z.isBull else "blue" if not z.isOb else "green" if z.isBull else "orange"
            rect_x = start - 0.5
            rect_w = (end - start) + 1
            ax.add_patch(Rectangle((rect_x, z.bottom), rect_w, z.top-z.bottom,
                                   facecolor=color, alpha=0.07, edgecolor=color,
                                   linestyle="--" if not z.isOb else "-", linewidth=1.5))

    # BOS/CHoCH lines
    if show_bos:
        for (swing_idx, break_idx, price) in bos_up:
            local_swing = swing_idx - start_idx_global
            local_break = break_idx - start_idx_global
            if local_swing < 0 or local_break > visible_end:
                continue
            ax.plot([local_swing, local_break], [price, price], color="lime", linestyle="--", linewidth=1.5)
            ax.text(local_break, price, "  BOS ↑", fontsize=8, color="lime", va='bottom')
        for (swing_idx, break_idx, price) in bos_dn:
            local_swing = swing_idx - start_idx_global
            local_break = break_idx - start_idx_global
            if local_swing < 0 or local_break > visible_end:
                continue
            ax.plot([local_swing, local_break], [price, price], color="red", linestyle="--", linewidth=1.5)
            ax.text(local_break, price, "  BOS ↓", fontsize=8, color="red", va='top')
        for (swing_idx, break_idx, price) in cho_up:
            local_swing = swing_idx - start_idx_global
            local_break = break_idx - start_idx_global
            if local_swing < 0 or local_break > visible_end:
                continue
            ax.plot([local_swing, local_break], [price, price], color="cyan", linestyle="--", linewidth=1.5)
            ax.text(local_break, price, "  CHoCH ↑", fontsize=8, color="cyan", va='bottom')
        for (swing_idx, break_idx, price) in cho_dn:
            local_swing = swing_idx - start_idx_global
            local_break = break_idx - start_idx_global
            if local_swing < 0 or local_break > visible_end:
                continue
            ax.plot([local_swing, local_break], [price, price], color="orange", linestyle="--", linewidth=1.5)
            ax.text(local_break, price, "  CHoCH ↓", fontsize=8, color="orange", va='top')

    # Turning points
    if show_tp:
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
# 8. DAILY CONTEXT (using same Pine logic, but only for filtering)
# ------------------------------------------------------------
def get_daily_context(df_daily):
    inputs = {
        'swing_l': 10, 'swing_r': 4,
        'maxAge': 26, 'failWindow': 5,
        'closeMitigate': True,
        'bodyThresh': 0.25, 'wickThreshHigh': 0.55, 'wickThreshLow': 0.15,
        'fvgBull': '#35aa18', 'fvgBear': '#da1313',
        'obBull': '#008950', 'obBear': '#883f0e',
        'maxAgeForProximity': 50
    }
    state = PineState()
    for i in range(len(df_daily)):
        dash = process_bar(df_daily, i, state, inputs)
    # Return relevant daily context
    return {
        'trend': 'BULLISH' if state.smc_bullish else 'BEARISH' if state.smc_bearish else 'NEUTRAL',
        'net_score': dash['z_score'],
        'inside_zone': 'Inside Bull Zone' in dash['zone_event'] or 'Inside Bear Zone' in dash['zone_event'],
        'zone_bullish': 'Bull' in dash['zone_event'],
        'recent_ssl': dash['liquidity'] == 'SSL',
        'recent_bsl': dash['liquidity'] == 'BSL',
        'mom_bullish': dash['momentum'] == 'UP ↑',
        'mom_bearish': dash['momentum'] == 'DOWN ↓'
    }

# ------------------------------------------------------------
# 9. HOURLY ENTRY SIGNAL (using stateful Pine logic)
# ------------------------------------------------------------
def get_hourly_signal(df_hourly, daily_ctx):
    inputs = {
        'swing_l': 6, 'swing_r': 3,
        'maxAge': 26, 'failWindow': 5,
        'closeMitigate': True,
        'bodyThresh': 0.25, 'wickThreshHigh': 0.55, 'wickThreshLow': 0.15,
        'fvgBull': '#35aa18', 'fvgBear': '#da1313',
        'obBull': '#008950', 'obBear': '#883f0e',
        'maxAgeForProximity': 50
    }
    state = PineState()
    last_dash = None
    for i in range(len(df_hourly)):
        last_dash = process_bar(df_hourly, i, state, inputs)
    # Use the final state to determine signal
    # Daily filters
    daily_bull = daily_ctx['trend'] == 'BULLISH' and daily_ctx['net_score'] > 20
    daily_bear = daily_ctx['trend'] == 'BEARISH' and daily_ctx['net_score'] < -20
    can_long = daily_bull or daily_ctx['recent_ssl']
    can_short = daily_bear or daily_ctx['recent_bsl']
    if daily_ctx['inside_zone'] and daily_ctx['zone_bullish']:
        can_long = can_long and daily_ctx['zone_bullish']
        can_short = can_short and not daily_ctx['zone_bullish']

    # Use Pine's goLong / goShort conditions (already computed in state via process_bar)
    # We'll compute them again from the last dashboard data
    inside_zone = 'Inside' in last_dash['zone_event']
    strong_ssl = last_dash['liquidity'] == 'SSL'
    strong_bsl = last_dash['liquidity'] == 'BSL'
    mom_bull = last_dash['momentum'] == 'UP ↑'
    mom_bear = last_dash['momentum'] == 'DOWN ↓'
    bullish_candle = state.last_close > state.last_open
    bearish_candle = state.last_close < state.last_open

    goLong = (state.smc_early_bull or state.smc_bullish) and inside_zone and ((bullish_candle and mom_bull) or (mom_bull and strong_ssl)) and state.smc_bullish
    trendLong = state.smc_bullish and inside_zone and mom_bull and (state.last_close > state.last_ema20) and (state.last_close > state.last_lb_crv) and (state.last_rsi > 50)
    earlyLong = can_long and strong_ssl and daily_bull  # simplified
    goShort = (state.smc_early_bear or state.smc_bearish) and inside_zone and ((bearish_candle and mom_bear) or (mom_bear and strong_bsl)) and state.smc_bearish
    trendShort = state.smc_bearish and inside_zone and mom_bear and (state.last_close < state.last_ema20) and (state.last_close < state.last_lb_crv) and (state.last_rsi < 50)
    earlyShort = can_short and strong_bsl and daily_bear

    long_signal = goLong or trendLong or earlyLong
    short_signal = goShort or trendShort or earlyShort

    atr_val = df_hourly['atr'].iloc[-1]
    if long_signal:
        sl = state.last_low - atr_val * 0.5
        tp = state.last_close + atr_val * 1.5
        risk = state.last_close - sl
        reward = tp - state.last_close
        rr = reward/risk if risk>0 else 0
        return {'signal':'LONG','valid':rr>=1.5,'reason':f"Long (goLong={goLong}, trendLong={trendLong}, earlyLong={earlyLong})",
                'sl':sl,'tp':tp,'rr':rr,'risk_pct':(risk/state.last_close)*100}
    elif short_signal:
        sl = state.last_high + atr_val * 0.5
        tp = state.last_close - atr_val * 1.5
        risk = sl - state.last_close
        reward = state.last_close - tp
        rr = reward/risk if risk>0 else 0
        return {'signal':'SHORT','valid':rr>=1.5,'reason':f"Short (goShort={goShort}, trendShort={trendShort}, earlyShort={earlyShort})",
                'sl':sl,'tp':tp,'rr':rr,'risk_pct':(risk/state.last_close)*100}
    else:
        return {'signal':'NO TRADE','valid':False,'reason':'No signal'}

# ------------------------------------------------------------
# 10. STREAMLIT UI
# ------------------------------------------------------------
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker", "AAPL")

# Load daily data for context
start_daily = datetime.today() - timedelta(days=365)
df_daily = load_data(ticker, start_daily, "1d")
if df_daily is None:
    st.error("No daily data")
    st.stop()

# Load hourly data for trading
try:
    # Use period instead of start/end dates for intraday
    df_hourly = yf.download(ticker, period="30d", interval="1h", auto_adjust=False, progress=False)
    
    if df_hourly is None or df_hourly.empty:
        raise Exception("No data")
    
    # Clean up the dataframe (same as your load_data function)
    df_hourly.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df_hourly.columns]
    df_hourly.index = pd.to_datetime(df_hourly.index)
    df_hourly = df_hourly.dropna(subset=["0open","high","low","close"]).astype(float)
    
    # Calculate all the indicators
    df_hourly['ema20'] = ema(df_hourly.close, 20)
    df_hourly['ema50'] = ema(df_hourly.close, 50)
    df_hourly['ema200'] = ema(df_hourly.close, 200)
    df_hourly['rsi'] = rsi(df_hourly.close, 14)
    df_hourly['rsi_ema'] = ema(df_hourly['rsi'], 14)
    df_hourly['atr'] = atr(df_hourly, 14)
    df_hourly['lb_crv'] = lb_curve(df_hourly, 10)
    df_hourly = df_hourly.bfill().ffill()
    
except Exception as e:
    st.warning(f"Hourly data unavailable, using 4H: {e}")
    df_hourly = yf.download(ticker, period="1mo", interval="4h", auto_adjust=False, progress=False)
    
    if df_hourly is None or df_hourly.empty:
        st.error("No intraday data")
        st.stop()
    
    # Clean 4H data
    df_hourly.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df_hourly.columns]
    df_hourly.index = pd.to_datetime(df_hourly.index)
    df_hourly = df_hourly.dropna(subset=["open","high","low","close"]).astype(float)
    
    # Calculate indicators for 4H
    df_hourly['ema20'] = ema(df_hourly.close, 20)
    df_hourly['ema50'] = ema(df_hourly.close, 50)
    df_hourly['ema200'] = ema(df_hourly.close, 200)
    df_hourly['rsi'] = rsi(df_hourly.close, 14)
    df_hourly['rsi_ema'] = ema(df_hourly['rsi'], 14)
    df_hourly['atr'] = atr(df_hourly, 14)
    df_hourly['lb_crv'] = lb_curve(df_hourly, 10)
    df_hourly = df_hourly.bfill().ffill()

# Daily context
with st.spinner("Analyzing daily context..."):
    daily_ctx = get_daily_context(df_daily)

# Hourly full analysis (for dashboard and chart)
with st.spinner("Processing hourly data with Pine logic..."):
    inputs = {
        'swing_l': 6, 'swing_r': 3,
        'maxAge': 26, 'failWindow': 5,
        'closeMitigate': True,
        'bodyThresh': 0.25, 'wickThreshHigh': 0.55, 'wickThreshLow': 0.15,
        'fvgBull': '#35aa18', 'fvgBear': '#da1313',
        'obBull': '#008950', 'obBear': '#883f0e',
        'maxAgeForProximity': 50
    }
    state = PineState()
    last_dashboard = None
    for i in range(len(df_hourly)):
        last_dashboard = process_bar(df_hourly, i, state, inputs)

# Hourly entry signal
with st.spinner("Computing hourly entry signal..."):
    hourly_signal = get_hourly_signal(df_hourly, daily_ctx)

# ------------------------------------------------------------
# SIDEBAR DASHBOARD (colored, same as before)
# ------------------------------------------------------------
st.sidebar.markdown("## 📊 SMC DASHBOARD (Hourly)")
d = last_dashboard
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
<tr><td>LIQUIDITY:</td><td class="{'green-bg' if d['liquidity']=='SSL' else 'red-bg' if d['liquidity']=='BSL' else 'gray-bg'}">{d['liquidity']}</td></tr>
<tr><td>SWEEP:</td><td class="{'green-bg' if d['sweep_status']=='ACTIVE' else 'gray-bg'}">{d['sweep_status']}</td></tr>
<tr><td>PATTERN:</td><td class="{'green-bg' if '↑' in d['pattern_text'] else 'red-bg' if '↓' in d['pattern_text'] else 'gray-bg'}">{d['pattern_text']} ({d['pattern_status']})</td></tr>
<tr><td>MOMENTUM:</td><td class="{'green-bg' if d['momentum']=='UP ↑' else 'red-bg' if d['momentum']=='DOWN ↓' else 'gray-bg'}">{d['momentum']}</td></tr>
<tr><td>STRUCT:</td><td class="{'green-bg' if d['struct']=='Bullish' else 'red-bg' if d['struct']=='Bearish' else 'gray-bg'}">{d['struct']}</td></tr>
<tr><td>SMC:</td><td class="{'green-bg' if d['smc_concept']=='Bullish' else 'red-bg' if d['smc_concept']=='Bearish' else 'gray-bg'}">{d['smc_concept']}</td></tr>
<tr><td>ZONE:</td><td class="{'green-bg' if 'Bull' in d['zone_event'] else 'red-bg' if 'Bear' in d['zone_event'] else 'gray-bg'}">{d['zone_event']}</td></tr>
<tr><td>ZONE DIST:</td><td class="{'yellow-bg' if d['zone_dist']!='---' else 'gray-bg'}">{d['zone_dist']}</td></tr>
<tr><td>BIAS:</td><td class="{'green-bg' if d['bias']=='Bullish' else 'red-bg' if d['bias']=='Bearish' else 'gray-bg'}">{d['bias']}</td></tr>
<tr><td>Z-SCORE:</td><td class="{'green-bg' if d['z_score']>0 else 'red-bg' if d['z_score']<0 else 'gray-bg'}">{d['z_score']}% {'Bull' if d['z_score']>0 else 'Bear' if d['z_score']<0 else 'Neut'}</td></tr>
<tr><td>SIGNAL:</td><td class="{'green-bg' if 'LONG' in hourly_signal['signal'] else 'red-bg' if 'SHORT' in hourly_signal['signal'] else 'gray-bg'}">{hourly_signal['signal']}</td></tr>
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

# Slice last 300 bars for performance
slice_df = df_hourly.tail(300)
global_start_idx = len(df_hourly) - len(slice_df)
# Filter zones that are within the slice
zones_in_slice = [z for z in state.allZones if z.startBar >= global_start_idx]

fig = plot_full_chart(slice_df, zones_in_slice, state.bos_up_list, state.bos_dn_list,
                      state.cho_up_list, state.cho_dn_list, state.turning_points,
                      global_start_idx, f"{ticker} – Hourly SMC",
                      show_fvg, show_ob, show_bos, show_tp)
st.pyplot(fig)

# Optional daily chart
if st.sidebar.checkbox("Show Daily Chart"):
    st.markdown("## 📉 Daily SMC Chart")
    # Process daily again for zones and structure (or reuse state from earlier)
    state_d = PineState()
    for i in range(len(df_daily)):
        _ = process_bar(df_daily, i, state_d, inputs)
    slice_daily = df_daily.tail(150)
    global_start_daily = len(df_daily) - len(slice_daily)
    zones_daily = [z for z in state_d.allZones if z.startBar >= global_start_daily]
    fig_d = plot_full_chart(slice_daily, zones_daily, state_d.bos_up_list, state_d.bos_dn_list,
                            state_d.cho_up_list, state_d.cho_dn_list, state_d.turning_points,
                            global_start_daily, f"{ticker} – Daily",
                            show_fvg, show_ob, show_bos, False)
    st.pyplot(fig_d)
