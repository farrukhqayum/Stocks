# pinescript_engine.py
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class FVGZone:
    top: float
    bottom: float
    start_idx: int
    is_bull: bool
    is_ob: bool = False
    is_mitigated: bool = False
    taps: int = 0


def _zones_overlap(z1: FVGZone, z2: FVGZone) -> bool:
    return not (z1.bottom > z2.top or z2.bottom > z1.top)


def _merge_zones(zones: list[FVGZone]) -> list[FVGZone]:
    if len(zones) <= 1:
        return zones
    zones_sorted = sorted(zones, key=lambda z: z.start_idx)
    merged = [zones_sorted[0]]
    for z in zones_sorted[1:]:
        last = merged[-1]
        if _zones_overlap(last, z):
            size1 = abs(last.top - last.bottom)
            size2 = abs(z.top - z.bottom)
            merged[-1] = last if size1 >= size2 else z
        else:
            merged.append(z)
    return merged


def apply_pinescript_logic(
    df: pd.DataFrame,
    max_age: int = 100,
    fail_window: int = 5,
) -> tuple[pd.DataFrame, list[FVGZone], dict]:
    """
    Pine-style bar-by-bar engine.
    Returns:
      df_enriched, active_zones, info_dict (pattern + turning point etc.)
    """

    df = df.copy()
    n = len(df)

    # Safety: ensure required columns exist
    required = ["open", "high", "low", "close", "ema20", "ema50", "ema200", "lb_crv", "rsi", "rsi_ema"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column in df: {col}")

    zones: list[FVGZone] = []

    last_pattern = None
    pattern_bull = None
    pattern_idx = None
    rejected = False
    expired = True
    turning_point = False
    turning_code = None

    # ------------- MAIN BAR LOOP -------------
    for i in range(n):
        o = df["open"].iloc[i]
        h = df["high"].iloc[i]
        l = df["low"].iloc[i]
        c = df["close"].iloc[i]

        # -----------------------------
        # 1) FVG + simple OB detection (3-candle + gap OB)
        # -----------------------------
        if i >= 2:
            h2 = df["high"].iloc[i - 2]
            l2 = df["low"].iloc[i - 2]
            o1 = df["open"].iloc[i - 1]
            h1 = df["high"].iloc[i - 1]
            l1 = df["low"].iloc[i - 1]
            c1 = df["close"].iloc[i - 1]

            # 3-candle FVG
            is_fvg_up = l > h2
            is_fvg_dn = h < l2

            if is_fvg_up:
                zones.append(FVGZone(top=l, bottom=h2, start_idx=i, is_bull=True))
            if is_fvg_dn:
                zones.append(FVGZone(top=h, bottom=l2, start_idx=i, is_bull=False))

            # 3-candle OB (displacement style)
            displacement_up = c > h1 and c > o
            displacement_dn = c < l1 and c < o

            bull_ob3 = displacement_up and (l1 < l2)
            bear_ob3 = displacement_dn and (h1 > h2)

            if bull_ob3:
                zones.append(FVGZone(top=h1, bottom=l1, start_idx=i - 1, is_bull=True, is_ob=True))
            if bear_ob3:
                zones.append(FVGZone(top=h1, bottom=l1, start_idx=i - 1, is_bull=False, is_ob=True))

            # Gap OB
            if i >= 1:
                o_prev = df["open"].iloc[i - 1]
                h_prev = df["high"].iloc[i - 1]
                l_prev = df["low"].iloc[i - 1]

                gap_up_ob = o > h_prev and c > o
                gap_dn_ob = o < l_prev and c < o

                if gap_up_ob:
                    zones.append(FVGZone(top=o, bottom=l_prev, start_idx=i, is_bull=True, is_ob=True))
                if gap_dn_ob:
                    zones.append(FVGZone(top=h_prev, bottom=o, start_idx=i, is_bull=False, is_ob=True))

        # -----------------------------
        # 2) Zone engine: age, taps, mitigation, failure
        # -----------------------------
        to_delete = []
        for j, z in enumerate(zones):
            age = i - z.start_idx
            failed = False

            if age <= fail_window and i >= 1:
                c_prev = df["close"].iloc[i - 1]
                if z.is_bull and c < z.bottom and c_prev < z.bottom:
                    failed = True
                if (not z.is_bull) and c > z.top and c_prev > z.top:
                    failed = True

            if not z.is_mitigated:
                if (h > z.bottom) and (l < z.top):
                    z.taps += 1

                bull_broken = z.is_bull and c < z.bottom
                bear_broken = (not z.is_bull) and c > z.top

                if bull_broken or bear_broken or (z.taps > 5):
                    z.is_mitigated = True

            if age > max_age or failed:
                to_delete.append(j)

        for j in reversed(to_delete):
            del zones[j]

        zones = _merge_zones(zones)

        # -----------------------------
        # 3) Candlestick pattern engine (similar to your pine_candle_engine)
        # -----------------------------
        if i >= 2:
            o0 = df["open"].iloc[i]
            c0 = df["close"].iloc[i]
            h0 = df["high"].iloc[i]
            l0 = df["low"].iloc[i]

            o1 = df["open"].iloc[i - 1]
            c1 = df["close"].iloc[i - 1]
            h1 = df["high"].iloc[i - 1]
            l1 = df["low"].iloc[i - 1]

            o2 = df["open"].iloc[i - 2]
            c2 = df["close"].iloc[i - 2]
            h2 = df["high"].iloc[i - 2]
            l2 = df["low"].iloc[i - 2]

            body0 = abs(c0 - o0)
            body1 = abs(c1 - o1)
            body2 = abs(c2 - o2)
            crange0 = h0 - l0
            wickHigh = h0 - max(o0, c0)
            wickLow = min(o0, c0) - l0

            ema20 = df["ema20"].iloc[i]
            ema50 = df["ema50"].iloc[i]
            ema200 = df["ema200"].iloc[i]
            lb = df["lb_crv"].iloc[i]

            ema_up = (ema20 > ema50) and (ema50 > ema200)
            ema_down = (ema20 < ema50) and (ema50 < ema200)
            lb_up = c0 > lb * 1.02
            lb_down = c0 < lb * 0.98

            # 3-candle patterns
            isMorning = (c2 < o2) and (body1 < body2 * 0.4) and (c0 > (o2 + c2) / 2)
            isEvening = (c2 > o2) and (body1 < body2 * 0.4) and (c0 < (o2 + c2) / 2)

            # 2-candle patterns
            bullEngulf = (
                c0 > o0 and
                c1 < o1 and
                c0 >= h1 and
                o0 <= l1
            )
            bearEngulf = (
                c0 < o0 and
                c1 > o1 and
                c0 <= l1 and
                o0 >= h1
            )

            bullPierce = (c1 < o1) and (c0 > (o1 + c1) / 2)
            bearDark = (c1 > o1) and (c0 < (o1 + c1) / 2)

            tweezerBot = abs(l0 - l1) < (crange0 * 0.1)
            tweezerTop = abs(h0 - h1) < (crange0 * 0.1)

            isHammer = (wickLow > body0 * 2) and (wickHigh < body0 * 0.5)
            isStar = (wickHigh > body0 * 2) and (wickLow < body0 * 0.5)

            pat = None
            pat_bull = None
            pat_idx = None

            if isMorning and lb_down:
                pat = "Morning Star"; pat_bull = True; pat_idx = i - 1
            elif isEvening and lb_up:
                pat = "Evening Star"; pat_bull = False; pat_idx = i - 1
            elif bullEngulf and lb_down:
                pat = "Bull Engulfing"; pat_bull = True; pat_idx = i
            elif bearEngulf and lb_up:
                pat = "Bear Engulfing"; pat_bull = False; pat_idx = i
            elif bullPierce and lb_down:
                pat = "Piercing"; pat_bull = True; pat_idx = i
            elif bearDark and lb_up:
                pat = "Dark Cloud"; pat_bull = False; pat_idx = i
            elif tweezerBot and lb_down:
                pat = "Tweezer Bottom"; pat_bull = True; pat_idx = i - 1
            elif tweezerTop and lb_up:
                pat = "Tweezer Top"; pat_bull = False; pat_idx = i - 1
            elif isHammer and lb_down:
                pat = "Hammer"; pat_bull = True; pat_idx = i
            elif isStar and lb_up:
                pat = "Shooting Star"; pat_bull = False; pat_idx = i

            if pat is not None:
                last_pattern = pat
                pattern_bull = pat_bull
                pattern_idx = pat_idx

    # -----------------------------
    # 4) Pattern validation + turning point (on last bar)
    # -----------------------------
    if last_pattern is not None and pattern_idx is not None:
        barsAgo = n - 1 - pattern_idx
        expired = barsAgo > 27

        pat_low = df["low"].iloc[pattern_idx]
        pat_high = df["high"].iloc[pattern_idx]
        close_last = df["close"].iloc[-1]

        if pattern_bull:
            rejected = close_last < pat_low
        else:
            rejected = close_last > pat_high

        turning_point = False
        turning_code = None

        if not expired and not rejected:
            o_last = df["open"].iloc[-1]
            h_last = df["high"].iloc[-1]
            l_last = df["low"].iloc[-1]
            c_last = df["close"].iloc[-1]

            body_last = abs(c_last - o_last)
            range_last = h_last - l_last
            wick_high_last = h_last - max(o_last, c_last)
            wick_low_last = min(o_last, c_last) - l_last

            if pattern_bull is False:
                if (c_last > o_last) and (wick_low_last > body_last * 1.2):
                    turning_point = True
                    turning_code = "▲ Rejecting Lows"
                if (c_last > o_last) and (body_last > 0.55 * range_last):
                    turning_point = True
                    turning_code = "▲ Bullish Drive"
            else:
                if (c_last < o_last) and (wick_high_last > body_last * 1.2):
                    turning_point = True
                    turning_code = "▼ Rejecting Highs"
                if (c_last < o_last) and (body_last > 0.55 * range_last):
                    turning_point = True
                    turning_code = "▼ Bearish Drive"

    info = {
        "last_pattern": last_pattern,
        "pattern_bull": pattern_bull,
        "pattern_idx": pattern_idx,
        "rejected": rejected,
        "expired": expired,
        "bull_signal": False,
        "bear_signal": False,
        "bullSweep": False,
        "bearSweep": False,
        "ema_bullish": df["ema20"].iloc[-1] > df["ema50"].iloc[-1],
        "ema_bearish": df["ema20"].iloc[-1] < df["ema50"].iloc[-1],
        "mom_bullish": False,
        "mom_bearish": False,
        "strong_bullish": False,
        "strong_bearish": False,
        "turning_point": turning_point,
        "turning_code": turning_code,
    }

    # Only return active (non-mitigated) zones to plot
    active_zones = [z for z in zones if not z.is_mitigated]

    return df, active_zones, info
