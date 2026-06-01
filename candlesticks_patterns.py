import pandas as pd
import numpy as np

def detect_doji(df, tolerance=0.001):
    pattern = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        if abs(df['Open'].iloc[i] - df['Close'].iloc[i]) / df['Open'].iloc[i] < tolerance:
            pattern[i] = 1
    return pd.Series(pattern, index=df.index)

def detect_hammer(df):
    pattern = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        body = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
        if df['Open'].iloc[i] > df['Close'].iloc[i]:
            lower_shadow = df['Open'].iloc[i] - df['Low'].iloc[i]
            upper_shadow = df['High'].iloc[i] - df['Open'].iloc[i]
        else:
            lower_shadow = df['Close'].iloc[i] - df['Low'].iloc[i]
            upper_shadow = df['High'].iloc[i] - df['Close'].iloc[i]
        if lower_shadow > 2 * body and upper_shadow < body:
            pattern[i] = 1
    return pd.Series(pattern, index=df.index)

def detect_hanging_man(df):
    pattern = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        body = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
        if df['Open'].iloc[i] > df['Close'].iloc[i]:
            lower_shadow = df['Open'].iloc[i] - df['Low'].iloc[i]
            upper_shadow = df['High'].iloc[i] - df['Open'].iloc[i]
        else:
            lower_shadow = df['Close'].iloc[i] - df['Low'].iloc[i]
            upper_shadow = df['High'].iloc[i] - df['Close'].iloc[i]
        if lower_shadow > 2 * body and upper_shadow < body and df['Open'].iloc[i] > df['Close'].iloc[i]:
            pattern[i] = 1
    return pd.Series(pattern, index=df.index)

def detect_morning_star(df):
    pattern = np.zeros(len(df), dtype=int)
    for i in range(2, len(df)):
        first = df.iloc[i-2]
        second = df.iloc[i-1]
        third = df.iloc[i]
        if (first['Close'] < first['Open'] and 
            abs(second['Close'] - second['Open']) < 0.002 * second['Open'] and 
            third['Close'] > third['Open'] and 
            third['Close'] > first['Open']):
            pattern[i] = 1
    return pd.Series(pattern, index=df.index)

def detect_evening_star(df):
    pattern = np.zeros(len(df), dtype=int)
    for i in range(2, len(df)):
        first = df.iloc[i-2]
        second = df.iloc[i-1]
        third = df.iloc[i]
        if (first['Close'] > first['Open'] and 
            abs(second['Close'] - second['Open']) < 0.002 * second['Open'] and 
            third['Close'] < third['Open'] and 
            third['Close'] < first['Open']):
            pattern[i] = 1
    return pd.Series(pattern, index=df.index)

def detect_shooting_star(df):
    pattern = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        body = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
        if df['Close'].iloc[i] > df['Open'].iloc[i]:
            upper_shadow = df['High'].iloc[i] - df['Close'].iloc[i]
            lower_shadow = df['Close'].iloc[i] - df['Low'].iloc[i]
        else:
            upper_shadow = df['High'].iloc[i] - df['Open'].iloc[i]
            lower_shadow = df['Open'].iloc[i] - df['Low'].iloc[i]
        if upper_shadow > 2 * body and lower_shadow < body:
            pattern[i] = 1
    return pd.Series(pattern, index=df.index)

def detect_three_white_soldiers(df):
    pattern = np.zeros(len(df), dtype=int)
    for i in range(2, len(df)):
        first = df.iloc[i-2]
        second = df.iloc[i-1]
        third = df.iloc[i]
        if (first['Close'] > first['Open'] and 
            second['Close'] > second['Open'] and 
            third['Close'] > third['Open'] and 
            third['Close'] > second['Close'] and 
            second['Close'] > first['Close']):
            pattern[i] = 1
    return pd.Series(pattern, index=df.index)

def detect_three_black_crows(df):
    pattern = np.zeros(len(df), dtype=int)
    for i in range(2, len(df)):
        first = df.iloc[i-2]
        second = df.iloc[i-1]
        third = df.iloc[i]
        if (first['Close'] < first['Open'] and 
            second['Close'] < second['Open'] and 
            third['Close'] < third['Open'] and 
            third['Close'] < second['Close'] and 
            second['Close'] < first['Close']):
            pattern[i] = 1
    return pd.Series(pattern, index=df.index)

def detect_bullish_engulfing(df):
    print(">>> USING LOOP VERSION OF BULLISH ENGULFING <<<", flush=True)
    pattern = np.zeros(len(df), dtype=int)
    for i in range(1, len(df)):
        prev_open  = df['Open'].iloc[i-1]
        prev_close = df['Close'].iloc[i-1]
        curr_open  = df['Open'].iloc[i]
        curr_close = df['Close'].iloc[i]
        if (prev_open > prev_close and
            curr_close > curr_open and
            curr_close > prev_open and
            curr_open < prev_close):
            pattern[i] = 1
    return pd.Series(pattern, index=df.index)

def detect_bearish_engulfing(df):
    pattern = np.zeros(len(df), dtype=int)
    for i in range(1, len(df)):
        prev_close = df['Close'].iloc[i-1]
        prev_open = df['Open'].iloc[i-1]
        curr_close = df['Close'].iloc[i]
        curr_open = df['Open'].iloc[i]
        if (prev_open < prev_close and 
            curr_close < curr_open and 
            curr_close < prev_open and 
            curr_open > prev_close):
            pattern[i] = 1
    return pd.Series(pattern, index=df.index)
