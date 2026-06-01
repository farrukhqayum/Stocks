import pandas as pd
import numpy as np

print(">>> LOADED vectorized candlesticks_patterns.py FROM:", __file__, flush=True)

def detect_doji(df, tolerance=0.001):
    condition = (abs(df['Open'] - df['Close']) / df['Open']) < tolerance
    return np.where(condition, 1, 0)

def detect_hammer(df):
    body = abs(df['Close'] - df['Open'])
    lower_shadow = np.where(df['Open'] > df['Close'], df['Open'] - df['Low'], df['Close'] - df['Low'])
    upper_shadow = np.where(df['Open'] > df['Close'], df['High'] - df['Open'], df['High'] - df['Close'])
    
    condition = (lower_shadow > 2 * body) & (upper_shadow < body)
    return np.where(condition, 1, 0)

def detect_hanging_man(df):
    body = abs(df['Close'] - df['Open'])
    lower_shadow = np.where(df['Open'] > df['Close'], df['Open'] - df['Low'], df['Close'] - df['Low'])
    upper_shadow = np.where(df['Open'] > df['Close'], df['High'] - df['Open'], df['High'] - df['Close'])
    
    condition = (lower_shadow > 2 * body) & (upper_shadow < body) & (df['Open'] > df['Close'])
    return np.where(condition, 1, 0)

def detect_morning_star(df):
    # Shifted values to match i-2 (first) and i-1 (second)
    first_open, first_close = df['Open'].shift(2), df['Close'].shift(2)
    second_open, second_close = df['Open'].shift(1), df['Close'].shift(1)
    third_open, third_close = df['Open'], df['Close']
    
    condition = (
        (first_close < first_open) & 
        (abs(second_close - second_open) < 0.002 * second_open) & 
        (third_close > third_open) & 
        (third_close > first_open)
    )
    return np.where(condition, 1, 0)

def detect_evening_star(df):
    first_open, first_close = df['Open'].shift(2), df['Close'].shift(2)
    second_open, second_close = df['Open'].shift(1), df['Close'].shift(1)
    third_open, third_close = df['Open'], df['Close']
    
    condition = (
        (first_close > first_open) & 
        (abs(second_close - second_open) < 0.002 * second_open) & 
        (third_close < third_open) & 
        (third_close < first_open)
    )
    return np.where(condition, 1, 0)

def detect_shooting_star(df):
    body = abs(df['Close'] - df['Open'])
    upper_shadow = np.where(df['Close'] > df['Open'], df['High'] - df['Close'], df['High'] - df['Open'])
    lower_shadow = np.where(df['Close'] > df['Open'], df['Close'] - df['Low'], df['Open'] - df['Low'])
    
    condition = (upper_shadow > 2 * body) & (lower_shadow < body)
    return np.where(condition, 1, 0)

def detect_three_white_soldiers(df):
    first_open, first_close = df['Open'].shift(2), df['Close'].shift(2)
    second_open, second_close = df['Open'].shift(1), df['Close'].shift(1)
    third_open, third_close = df['Open'], df['Close']
    
    condition = (
        (first_close > first_open) & 
        (second_close > second_open) & 
        (third_close > third_open) & 
        (third_close > second_close) & 
        (second_close > first_close)
    )
    return np.where(condition, 1, 0)

def detect_three_black_crows(df):
    first_open, first_close = df['Open'].shift(2), df['Close'].shift(2)
    second_open, second_close = df['Open'].shift(1), df['Close'].shift(1)
    third_open, third_close = df['Open'], df['Close']
    
    condition = (
        (first_close < first_open) & 
        (second_close < second_open) & 
        (third_close < third_open) & 
        (third_close < second_close) & 
        (second_close < first_close)
    )
    return np.where(condition, 1, 0)

def detect_bullish_engulfing(df):
    prev_open, prev_close = df['Open'].shift(1), df['Close'].shift(1)
    curr_open, curr_close = df['Open'], df['Close']
    
    condition = (
        (prev_open > prev_close) &
        (curr_close > curr_open) &
        (curr_close > prev_open) &
        (curr_open < prev_close)
    )
    return np.where(condition, 1, 0)

def detect_bearish_engulfing(df):
    prev_open, prev_close = df['Open'].shift(1), df['Close'].shift(1)
    curr_open, curr_close = df['Open'], df['Close']
    
    condition = (
        (prev_open < prev_close) & 
        (curr_close < curr_open) & 
        (curr_close < prev_open) & 
        (curr_open > prev_close)
    )
    return np.where(condition, 1, 0)
