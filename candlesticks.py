import pandas as pd
import numpy as np

# ============================
# Helper Calculations
# ============================

def candle_components(df):
    body = (df['Close'] - df['Open']).abs()
    upper_shadow = np.where(
        df['Close'] >= df['Open'],
        df['High'] - df['Close'],
        df['High'] - df['Open']
    )
    lower_shadow = np.where(
        df['Close'] >= df['Open'],
        df['Open'] - df['Low'],
        df['Close'] - df['Low']
    )
    total_range = df['High'] - df['Low']
    return body, upper_shadow, lower_shadow, total_range

def detect_gravestone(df, doji_mask, wick_high=0.55, wick_low=0.15):
    _, upper, lower, total = candle_components(df)
    doji_mask = np.asarray(doji_mask)
    return ((doji_mask) &
            (upper >= total * wick_high) &
            (lower <= total * wick_low)).astype(int)

def detect_dragonfly(df, doji_mask, wick_high=0.55, wick_low=0.15):
    _, upper, lower, total = candle_components(df)
    doji_mask = np.asarray(doji_mask)
    return ((doji_mask) &
            (lower >= total * wick_high) &
            (upper <= total * wick_low)).astype(int)

def classify_doji(df, doji_mask, gravestone_mask, dragonfly_mask):
    doji_mask = np.asarray(doji_mask)
    gravestone_mask = np.asarray(gravestone_mask)
    dragonfly_mask = np.asarray(dragonfly_mask)

    neutral = doji_mask & (~gravestone_mask) & (~dragonfly_mask)

    bull_doji = neutral & (df['Close'].values > df['Open'].values)
    bear_doji = neutral & (df['Close'].values < df['Open'].values)

    return bull_doji.astype(int), bear_doji.astype(int)

# ============================
# 1. Doji
# ============================

def detect_doji(df, body_thresh=0.10, wick_limit=0.45, symmetry_limit=0.15):
    body, upper, lower, total = candle_components(df)
    total = np.where(total == 0, 1e-9, total)
    body_ok = (body <= total * body_thresh)
    upper_ok = (upper <= total * wick_limit)
    lower_ok = (lower <= total * wick_limit)
    symmetry_ok = (np.abs(upper - lower) <= total * symmetry_limit)

    return (body_ok & upper_ok & lower_ok & symmetry_ok).astype(int)

# ============================
# 2. Hammer
# ============================

def detect_hammer(df):
    body, upper, lower, _ = candle_components(df)
    return ((lower >= 2 * body) & (upper <= body)).astype(int)


# ============================
# 3. Hanging Man
# ============================

def detect_hanging_man(df):
    body, upper, lower, _ = candle_components(df)
    bearish = df['Close'] < df['Open']
    return ((lower >= 2 * body) & (upper <= body) & bearish).astype(int)


# ============================
# 4. Shooting Star
# ============================

def detect_shooting_star(df):
    body, upper, lower, _ = candle_components(df)
    return ((upper >= 2 * body) & (lower <= body)).astype(int)


# ============================
# 5. Morning Star
# ============================

def detect_morning_star(df):
    pattern = np.zeros(len(df), dtype=int)
    for i in range(2, len(df)):
        c1 = df.iloc[i-2]
        c2 = df.iloc[i-1]
        c3 = df.iloc[i]

        body2 = abs(c2['Close'] - c2['Open'])

        if (
            c1['Close'] < c1['Open'] and
            body2 < 0.3 * (c2['High'] - c2['Low']) and
            c3['Close'] > c3['Open'] and
            c3['Close'] > c1['Open']
        ):
            pattern[i] = 1
    return pattern


# ============================
# 6. Evening Star
# ============================

def detect_evening_star(df):
    pattern = np.zeros(len(df), dtype=int)
    for i in range(2, len(df)):
        c1 = df.iloc[i-2]
        c2 = df.iloc[i-1]
        c3 = df.iloc[i]

        body2 = abs(c2['Close'] - c2['Open'])

        if (
            c1['Close'] > c1['Open'] and
            body2 < 0.3 * (c2['High'] - c2['Low']) and
            c3['Close'] < c3['Open'] and
            c3['Close'] < c1['Open']
        ):
            pattern[i] = 1
    return pattern


# ============================
# 7. Three White Soldiers
# ============================

def detect_three_white_soldiers(df):
    pattern = np.zeros(len(df), dtype=int)
    for i in range(2, len(df)):
        c1 = df.iloc[i-2]
        c2 = df.iloc[i-1]
        c3 = df.iloc[i]

        if (
            c1['Close'] > c1['Open'] and
            c2['Close'] > c2['Open'] and
            c3['Close'] > c3['Open'] and
            c2['Open'] > c1['Open'] and
            c3['Open'] > c2['Open'] and
            c3['Close'] > c2['Close']
        ):
            pattern[i] = 1
    return pattern


# ============================
# 8. Three Black Crows
# ============================

def detect_three_black_crows(df):
    pattern = np.zeros(len(df), dtype=int)
    for i in range(2, len(df)):
        c1 = df.iloc[i-2]
        c2 = df.iloc[i-1]
        c3 = df.iloc[i]

        if (
            c1['Close'] < c1['Open'] and
            c2['Close'] < c2['Open'] and
            c3['Close'] < c3['Open'] and
            c2['Open'] < c1['Open'] and
            c3['Open'] < c2['Open'] and
            c3['Close'] < c2['Close']
        ):
            pattern[i] = 1
    return pattern


# ============================
# 9. Bullish Engulfing
# ============================

def detect_bullish_engulfing(df):
    prev_open = df['Open'].shift(1)
    prev_close = df['Close'].shift(1)

    curr_open = df['Open']
    curr_close = df['Close']

    bearish_prev = prev_close < prev_open
    bullish_curr = curr_close > curr_open

    engulf = (curr_close > prev_open) & (curr_open < prev_close)

    return (bearish_prev & bullish_curr & engulf).astype(int)
