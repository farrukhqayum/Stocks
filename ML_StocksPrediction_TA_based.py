from imports import *
import streamlit as st
import time
import re
import warnings
import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import emoji
import altair as alt
import sys
st.write("### Version Information")
st.write(f"Python: {sys.version}")
st.write(f"Pandas version: {pd.__version__}")
st.write(f"Numpy version: {np.__version__}")

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

st.caption("Data sourced via Yahoo Finance • Updated dynamically")
warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="📈 MAIN - Machine Learning of Stocks")

bold = '\033[1m'
end = '\033[0m'

desc = """  
- Machine learning models train technical indicators
- Trade signals include signal type, hit probability, and direction
- Use tables to identify strong stocks and charts to confirm bullish trends
    - **SEE THE CHART FOR YOUR TICKER OF INTEREST**:
        - **BUY TIMES**: Green areas signal "Buy the Dip" (BTD) opportunities
        - **SELL TIMES**: Red areas signal "Sell the Rally" opportunities
        - **NEUTRAL**: Hold during buy zones; stay in cash otherwise (avoid FOMO/revenge trading)
        - **STRONG BUYS**: Occur when RSI recovers from oversold (<30), crosses above its SMA (yellow line), and price > moving averages
        - **STRONG SELLS**: Occur when RSI <42 and declining (RSI <30 doesn't guarantee rebound)
        - **Buy late rather than early** when chasing 3-10% swing trade gains
- **AVOID CHASING**:
    - Opportunities occur daily/weekly/monthly—don't take every one
    - Focus on double bottoms, weekly/monthly candles; avoid daily noise
    - Trust the system over gut feelings (though intuition has occasional value)
    - Risk ≤5% per trade (increased retail participation post-COVID)
    - Split entries: 2-3 buys, 2-3 sells per position
    - Best trades form on weekly timeframes; enter on 4H/1H confirmation
- **USE DIVERGENCE**: Identifies market swing highs/lows for 4-6 month holds
""" 

HowTo = """
## 📘 Real-Life ML Stock Trading Rules (Simple & Proven)

### 🎯 Objective
Use ML as a **risk filter**, not a prediction engine.  
Trade **few, high-quality setups** with patience and discipline.

---

### 1️⃣ Before You Trade
- Trade only liquid, well-known stocks
- Avoid hype, penny stocks, and thin volume
- Use **daily timeframe only**
- One open trade at a time

---

### 2️⃣ ML Entry Conditions (Must ALL Be True)
- ML confidence **≥ 60%**
- ML signal is **NOT bearish**
  - Allowed: `Neutral`, `TP`, `Hold`
- Expected reward > expected risk
- No existing open trade

---

### 3️⃣ Price Location Filter
- Price is **near its short-term average**
- Avoid extended or parabolic moves
- Enter near balance, not emotional extremes

---

### 4️⃣ Entry Execution
- Enter at **daily close**
- No adding mid-trade unless weekly confidence
- One decision per trade

---

### 5️⃣ Take-Profit (TP) Planning
- Default TP: **3–7%** (best for compounding)
- Allow higher TP only if ML supports it
- Do not force large targets on slow stocks

---

### 6️⃣ Stop-Loss (SL) Planning
- SL must reflect volatility
- Wider SL for strong stocks
- Tighter SL for weak or fast stocks
- If SL feels painful → position is too large

---

### 7️⃣ Exit Rules (No Emotions)
Exit immediately if:
- Stop-loss is hit
- Take-profit is hit
- Maximum holding time is reached

---

### 8️⃣ Time Discipline
- Exit trades that go nowhere
- Capital must stay productive
- Dead money kills compounding

---

### 🔑 Core Principles
- ML filters risk, it does not guarantee outcomes
- Discipline > accuracy
- Protect capital first
- Let winners work, cut losers fast

> “I am not here to predict.  
> I am here to manage probability and risk.”
"""


mistakes = """
- Review **Psychology Tab** for common pitfalls
- Having money ≠ being smart or beating the market
- Wanting 5 wins/week leads to zero-sum losses
- Set realistic **Monthly Goals**
- Master staying sidelined with losing positions
- **Cash is king**: Discipline to hold cash without trading is an art
- **Split positions systematically**:
    - Entry 1: 10-20% (always wrong initially)
    - Entry 2: 30% of remainder  
    - Entry 3: Final 30% (solid stocks reverse)
- **If ML signal fails but stock moves against you**: Stop using this app
  Blaming the system or poor emotion control = biggest mistake
"""

disclaimer = """
---
- Trading involves substantial risk of financial loss  
- Past performance does not predict future results  
- Always conduct your own research  
- Information is educational only—not financial advice  
- **Trade at your own risk**  
---
"""

today = datetime.now().strftime('%Y-%m-%d')

_Nr = 50
YEARS_OF_DATA = 3
end_date = datetime.now()
start_date = end_date - timedelta(days=365 * YEARS_OF_DATA)
PROFIT_TARGET = 0.04
STOP_LOSS = 0.0375
_DAYS = 22
_FWDAYS = 14
windows = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
_window = 9
_tolerance = 1.07
_FIBS = False
_FibLen = 20
_ms = 5

FEATURES = [
    # Price High, Low
    'High', 'Low',
    
    # Technical Indicators
    'RSI', 'RSI_SMA', 'CCI', '+DI', '-DI', 'ADX', 'ATR', 'VI+', 'KCu', 'KCl', 'KCu_outer', 'KCl_outer', 'Kasym', 'Kcount', 'STu', 'STl',

    # Moving Averages & Bands
    'EMA1', 'EMA2', 'EMA3', 'EMA_Ratio', 'Upper_Band', 'Lower_Band', 'Volume_MA20', 'SMIIO', 'SMIIO_Signal', 'SMIIO_Osc', 'MACD', 'Signal_Line',

    # Returns & Volatility
    'return1', 'return2', 'return3', 'Volatility', 'Scaled_Volatility', 'DD',

    # Volume Features
    'sumBuyVol', 'sumSellVol', 'vSpike', 'VPT', 'OBV', 'MFI', 'VWMA', 'CMF',

    # Candlestick Patterns
    'Candlesticks', 'gapStrength',

    # Market Sentiment & Signals
    'Bear', 'Bull', 'Short', 'Hold', 'Neutral', 'StrongBull', 'StrongBear', 'Neutral', 'Exhaustion',

    # PIVOTS
    'PP_Avg', 'R1_Avg', 'R2_Avg', 'S1_Avg', 'S2_Avg'
]


EMOJI = {
    "STRONG BUY": "💎", 
    "Buy": "🟢", 
    "Wait": "⏳", 
    "Short/AVOID": "🚫", 
    "Short the RISE": "🔻", 
    "RISKY BUY": "⚠️", 
    "Monitor": "🔍", 
    "Watch": "👀"
} 
def label(text):
    return f"{EMOJI.get(text, '')} {text}"
    
def optimize_dataframe(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    return df
    
def get_stock_data(ticker, start_date, end_date):
    try:
        # ✅ Add retry logic
        for attempt in range(3):
            try:
                df = yf.download(ticker, start=start_date, end=end_date, 
                    progress=False, auto_adjust=True, threads=False, prepost=False)
                break
            except Exception as e:
                if attempt == 2:
                    raise e
                sleep(1)
        
        if df.empty: 
            return None
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.astype({col: 'float32' for col in df.select_dtypes(include=['float64']).columns})
        if 'Date' in df.columns:
            df = df.set_index('Date')
        
        df.index = pd.to_datetime(df.index)
        return df
        
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return None

def strip_ansi_codes(text):
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

def add_technical_indicators(df):
    close = df.Close
    df['Close'] = df[['Open', 'High', 'Low', 'Close']].mean(axis=1).rolling(2).mean()
    df['EMA1'] = df['Close'].ewm(span=int(_DAYS * 0.5), adjust=False).mean()
    df['EMA2'] = df['Close'].ewm(span=_DAYS, adjust=False).mean()
    df['EMA3'] = df['Close'].ewm(span=int(_DAYS * 2), adjust=False).mean()
    df['EMA_Ratio'] = df['EMA1'] / df['EMA2']
    df['ATR'] = ta.calculate_atr(high=df.High, low=df.Low, close=df.Close)
    df = ta.scaled_volatility(df)
    df = ta.add_candlestickpatterns(df)
    df['RSI']= ta.calculate_rsi(df)
    df['RSI_SMA'] = df['RSI'].rolling(14).mean()
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=24, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['SMIIO'], df['SMIIO_Signal'], df['SMIIO_Osc'] = ta.calculate_smiio(df)
    df['Upper_Band'] = df['EMA1'] + (2 * df['Close'].rolling(20).std())
    df['Lower_Band'] = df['EMA1'] - (2 * df['Close'].rolling(20).std())
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    df['buy_volume'] = (df.Close > df.Close.shift(1)) * df['Volume']
    df['sell_volume'] = (df.Close < df.Close.shift(1)) * df['Volume']
    df['sumBuyVol'] = df['buy_volume'].rolling(window=9).sum()
    df['sumSellVol'] = df['sell_volume'].rolling(window=9).sum()
    df['vSpike'] = np.where(df['Volume'] > 2 * df['Volume_MA20'], np.where(df['Close'] > df['Open'], 1, -1), 0)
    df['VPT'] = df['Volume'].mul((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)).cumsum()
    df['MFI'] = ta.calculate_mfi(df)
    df['CMF'] = ta.chaikin_money_flow(df, window=20)
    df['CCI'] = ta.calculate_cci(df)
    df['OBV'] = ta.calculate_obv(df)
    df[['+DI', '-DI', 'ADX']] = ta.calculate_dmi(df, n=14).rolling(3).mean()
    df['VWMA'] = ta.calculate_vwma(df)
    df[['KCm', 'KCu', 'KCl', 'KCu_outer','KCl_outer', 'Kasym', 'Kcount']] = ta.calculate_keltner(df).rolling(3).mean()
    df[['VI+', 'VI-']] = ta.calculate_vortex(df)
    df[['STu', 'STl']] = ta.calculate_supertrend(df)
    df['DD'] = df['Close'].rolling(14).apply(lambda x: x[-1] - x.max())
    df['return1'] = df['Close'].pct_change(7).rolling(3).mean()
    df['return2'] = df['Close'].pct_change(14).rolling(3).mean()
    df['return3'] = df['Close'].pct_change(21).rolling(3).mean()
    df['Volatility'] = df['Close'].rolling(14).std().rolling(3).mean()
    cols = ['EMA1', 'EMA2', 'RSI', '-DI', 'Close']
    df[cols] = df[cols].ffill().bfill()
    conditions = [
        # 1️⃣ HOLD FIRST (Extended Rally - HIGHEST priority)
        (
            (df['Close'] > df['EMA2']) &
            (df['EMA1'] > df['EMA2']) &
            (df['RSI'].between(50, 90)) &
            (df['ADX'] > 40) &
            (df['+DI'] > df['-DI']) &
            (df['Close'] > df['Close'].shift(5) * 1.02)  # ✅ Rally proof!
        ),
        
        # 2️⃣ BULL (Entry signals)
        (
            (
                ((df['EMA1'] >= df['EMA2']) &
                  (df['RSI'] >= df['RSI_SMA']) &
                  (df['RSI'].between(52, 95)) &
                  ((df['ADX'] > 24) & (df['+DI'] > df['-DI'])))
                |
                (
                    ((df['RSI'] >= df['RSI_SMA']) & (df['RSI'] > 50)) & 
                    ((df['ADX'] > 18) & (df['+DI'] > df['-DI']))
                )
            )
        ),
        
        # 3️⃣ SHORT (Aggressive shorts)
        (
            ((df['Close'] <= df['EMA1']) &
             (df['EMA1'] < df['EMA2']) &
             (df['RSI'].between(50, 85)) &
             (df['ADX'] > 24) & 
             (df['+DI'] < df['-DI']))
        ),
        
        # 4️⃣ BEAR (Bearish entries - LOWEST priority)
        (
            (
                ((df['EMA1'] < df['EMA2']) &
                  (df['RSI'].between(18,60)) &
                  (df['RSI'] < df['RSI_SMA']) &
                  ((df['ADX'] > 18) & (df['+DI'] < df['-DI'])))
                |
                (
                    ((df['RSI'] < df['RSI_SMA']) & 
                     (df['RSI'].between(20, 60)) &
                     ((df['ADX'] > 18) & (df['+DI'] < df['-DI'])))
                )
                |
                (
                    ((df['RSI'] > df['RSI_SMA']) & 
                     (df['RSI_SMA'] < 37))
                )
            )
        )
    ]
    
    choices = ['Hold', 'Bull', 'Short', 'Bear']  # ✅ Priority order!
    df['TI'] = np.select(conditions, choices, default='Neutral')
    df['TI'] = df['TI'].astype('category')
 
    df_encoded = pd.get_dummies(df['TI'], prefix='', prefix_sep='')
    expected_cols = ['Hold', 'Bull', 'Short', 'Bear', 'Neutral']
    for col in expected_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df = pd.concat([df, df_encoded], axis=1)

    strongbull_condition = ((df['RSI'] > 52) & (df['ADX'] > 22) & (df['+DI'] > df['-DI']) & (df['sumBuyVol'] > df['sumSellVol']))
    strongbear_condition = ((df['RSI'] < 40) & (df['ADX'] > 22) & (df['+DI'] < df['-DI']) & (df['sumBuyVol'] < df['sumSellVol']))
    df['StrongBull'] = strongbull_condition.astype(int)
    df['StrongBear'] = strongbear_condition.astype(int)
    df['sNeutral'] = ((df['StrongBull'] == 0) & (df['StrongBear'] == 0)).astype(int)
    df['gapStrength'] = ta.compute_gapStrength(df)
    df = ta.add_exhaustion_indicator(df)
    df.Close = close
    return df

def add_pivot_levels(df, window=_DAYS):
    high = df['High'].rolling(window)
    low = df['Low'].rolling(window)
    close = df['Close'].rolling(window)
    PP = (high.max() + low.min() + close.apply(lambda x: x[-1])).div(3)
    R1 = 2 * PP - low.min()
    S1 = 2 * PP - high.max()
    R2 = PP + (high.max() - low.min())
    S2 = PP - (high.max() - low.min())
    df['PP'] = PP.bfill()
    df['R1'] = R1.bfill()
    df['S1'] = S1.bfill()
    df['R2'] = R2.bfill()
    df['S2'] = S2.bfill()
    return df

def add_pivots(df, win=windows):
    for w in win:
        roll_high = df['High'].rolling(w)
        roll_low = df['Low'].rolling(w)
        roll_close = df['Close'].rolling(w)
        PP = (roll_high.max() + roll_low.min() + roll_close.apply(lambda x: x[-1])).div(3)
        R1 = 2 * PP - roll_low.min()
        S1 = 2 * PP - roll_high.max()
        R2 = PP + (roll_high.max() - roll_low.min())
        S2 = PP - (roll_high.max() - roll_low.min())
        df[f'PP_{w}'] = PP
        df[f'R1_{w}'] = R1
        df[f'S1_{w}'] = S1
        df[f'R2_{w}'] = R2
        df[f'S2_{w}'] = S2
    return df

def average_pivots(df, windows=[5, 10, 14, 20]):
    for level in ['PP', 'R1', 'S1', 'R2', 'S2']:
        cols = [f'{level}_{w}' for w in windows]
        df[f'{level}_Avg'] = df[cols].mean(axis=1)
    return df

def compute_expected_return(df, forward_window=14, r_cols=['R1', 'R2']):
    df['Expected_Return'] = np.nan
    close_prices = df['Close'].values
    pivot_arrays = []
    for col in r_cols:
        if col in df.columns:
            pivot_arrays.append(df[col].values)
        else:
            pivot_arrays.append(np.full(len(df), np.nan))
    for i in range(len(df) - forward_window):
        current_price = close_prices[i]
        pivots = [arr[i] for arr in pivot_arrays if not np.isnan(arr[i])]
        target_level = max(pivots) if pivots else None
        future_window = close_prices[i+1:i+1+forward_window]
        if target_level is not None:
            hit = False
            for future_price in future_window:
                if future_price >= target_level:
                    df.iloc[i, df.columns.get_loc('Expected_Return')] = (target_level - current_price) / current_price
                    hit = True
                    break
            if not hit and future_window.size > 0:
                df.iloc[i, df.columns.get_loc('Expected_Return')] = (np.nanmax(future_window) - current_price) / current_price
        else:
            if future_window.size > 0:
                df.iloc[i, df.columns.get_loc('Expected_Return')] = (np.nanmax(future_window) - current_price) / current_price
            else:
                df.iloc[i, df.columns.get_loc('Expected_Return')] = np.nan
    return df

def compute_expected_loss(df, forward_window=14, s_cols=['S1', 'S2']):
    df['Expected_Loss'] = np.nan
    close_prices = df['Close'].values
    pivot_arrays = []
    for col in s_cols:
        if col in df.columns:
            pivot_arrays.append(df[col].values)
        else:
            pivot_arrays.append(np.full(len(df), np.nan))
    for i in range(len(df) - forward_window):
        current_price = close_prices[i]
        pivots = [arr[i] for arr in pivot_arrays if not np.isnan(arr[i])]
        target_level = min(pivots) if pivots else None
        future_window = close_prices[i+1:i+1+forward_window]
        if target_level is not None:
            hit = False
            for future_price in future_window:
                if future_price <= target_level:
                    df.iloc[i, df.columns.get_loc('Expected_Loss')] = (target_level - current_price) / current_price
                    hit = True
                    break
            if not hit and future_window.size > 0:
                df.iloc[i, df.columns.get_loc('Expected_Loss')] = (np.nanmin(future_window) - current_price) / current_price
        else:
            if future_window.size > 0:
                df.iloc[i, df.columns.get_loc('Expected_Loss')] = (np.nanmin(future_window) - current_price) / current_price
            else:
                df.iloc[i, df.columns.get_loc('Expected_Loss')] = np.nan
    return df


def label_hit_prob_past(
    df,
    window=14,
    profit_target=0.05,
    stop_loss=0.05,
    lookback=60,
    tp_thresh=0.35,
    sl_thresh=0.4
):
    import numpy as np
    
    close_prices = df['Close'].values
    
    bull = (df['TI'] == 'Bull')
    bear = (df['TI'] == 'Bear')
    hold = (df['TI'] == 'Hold')
    short = (df['TI'] == 'Short')
    neutral = (df['TI'] == 'Neutral')

    EMA1 = df['EMA1'].values
    atr = df['ATR'].values
    rsi = df['RSI'].values
    adx = df['ADX'].values

    N = len(close_prices)
    labels = []
    
    for i in range(N):
        current_price = close_prices[i]
        tp = current_price * (1 + profit_target)
        sl = current_price * (1 - stop_loss)
        future_prices = close_prices[i + 1 : min(i + 1 + window, N)]
        tp_hit_idx = next((j for j, price in enumerate(future_prices) if price >= tp), None)
        sl_hit_idx = next((j for j, price in enumerate(future_prices) if price <= sl), None)
        
        lookback_start = max(0, i - lookback)
        history_tp, history_sl = [], []
        for j in range(lookback_start, i):
            hist_price = close_prices[j]
            hist_tp = hist_price * (1 + profit_target)
            hist_sl = hist_price * (1 - stop_loss)
            hist_future = close_prices[j + 1: j + 1 + window]
            
            if bull.iloc[j]:
                hist_tp_hit_idx = next((k for k, p in enumerate(hist_future) if p >= hist_tp), None)
                hist_sl_hit_idx = next((k for k, p in enumerate(hist_future) if p <= hist_sl), None)
                hit = hist_tp_hit_idx is not None and (hist_sl_hit_idx is None or hist_tp_hit_idx < hist_sl_hit_idx)
                history_tp.append(int(hit))
                
            if bear.iloc[j]:
                hist_tp_hit_idx = next((k for k, p in enumerate(hist_future) if p >= hist_tp), None)
                hist_sl_hit_idx = next((k for k, p in enumerate(hist_future) if p <= hist_sl), None)
                hit = hist_sl_hit_idx is not None and (hist_tp_hit_idx is None or hist_sl_hit_idx < hist_tp_hit_idx)
                history_sl.append(int(hit))
        
        tp_prob = np.mean(history_tp) if len(history_tp) >= 3 else min(np.mean(history_tp) if history_tp else 0.5, tp_thresh)
        sl_prob = np.mean(history_sl) if len(history_sl) >= 3 else min(np.mean(history_sl) if history_sl else 0.5, sl_thresh)
        
        # Initial label assignment priority: TP > SL > Hold > Short > Neutral
        if tp_hit_idx is not None and (sl_hit_idx is None or tp_hit_idx < sl_hit_idx) and bull[i] and tp_prob >= tp_thresh:
            labels.append(2)  # TP (bull)
        elif sl_hit_idx is not None and (tp_hit_idx is None or sl_hit_idx < tp_hit_idx) and bear[i] and sl_prob >= sl_thresh:
            labels.append(1)  # SL (bear)
        elif hold[i]:
            # Upgrade Hold to TP if breakout early within window
            if any(p >= tp for p in future_prices):
                labels.append(2)
            else:
                labels.append(3)
        elif short[i]:
            labels.append(4)
        else:
            if i >= N - window:
                if bull[i]:
                    labels.append(2)
                elif bear[i]:
                    labels.append(1)
                else:
                    labels.append(0)
            else:
                labels.append(0)
    
    # Post-process: Trigger SL immediately on price dip below EMA1 or EMA1-ATR buffer with momentum checks for Hold/TP
    for i in range(N):
        if labels[i] in [2, 3]:  # TP or Hold bars
            current_close = close_prices[i]
            EMA1_now = EMA1[i]
            atr_now = atr[i]
            rsi_now = rsi[i]
            adx_now = adx[i]

            future_end = min(i + 1 + window, N)
            future_closes = close_prices[i + 1 : future_end]
            future_EMA1 = EMA1[i + 1 : future_end]

            current_dip = current_close < EMA1_now or current_close < (EMA1_now - 0.5 * atr_now)
            future_dips = any((p < s) or (p < s - 0.5 * atr_now) for p, s in zip(future_closes, future_EMA1))

            bearish_momentum = (rsi_now < 40) and (adx_now > 22)
            fading_bullish = (rsi_now < 50) or (adx_now < 20)
            hold_extreme = (labels[i] == 3) and (rsi_now < 45)

            if (current_dip or future_dips) and (bearish_momentum or fading_bullish or hold_extreme):
                if not ((rsi_now > 52) and (df['Close'].iloc[i] > df['EMA2'].iloc[i])):
                    labels[i] = 1
    
    df['Hit_Label'] = labels
    return df
    
def get_recent_fib_levels(df, left=_FibLen, right=_FibLen):
    highs = df['High']
    lows = df['Low']
    is_pivot_high = highs == highs.rolling(window=left+right+1, center=True).max()
    is_pivot_low = lows == lows.rolling(window=left+right+1, center=True).min()
    is_pivot_high = is_pivot_high.fillna(False)
    is_pivot_low = is_pivot_low.fillna(False)
    pivot_highs = df[is_pivot_high]
    pivot_lows = df[is_pivot_low]
    if pivot_highs.empty or pivot_lows.empty:
        return None, None, None
    last_high_idx = pivot_highs.index[-1]
    last_low_idx = pivot_lows.index[-1]
    high = df.loc[last_high_idx, 'High']
    low = df.loc[last_low_idx, 'Low']
    diff = high - low
    fibs = {
        'F:0': low,
        'F:100': high,
        'F:61.8': high - 0.618 * diff,
        'F:125': high + 1.25 * diff,
        'F:-125': low - 1.25 * diff,
    }
    fib_start = min(last_high_idx, last_low_idx)
    fib_end = max(last_high_idx, last_low_idx)
    return fibs, fib_start, fib_end
    
def extract_emojis(text):
    # Returns a string with only the emoji characters from the input
    return ''.join(c for c in text if c in emoji.EMOJI_DATA)
    
def colored_row(text, color):
    colors = {
        'green': '\033[92m',
        'red': '\033[91m',
        'yellow': '\033[93m',
        'white': '\033[97m',
        'darkred': '\033[38;5;52m'
    }
    reset = '\033[0m'
    color_code = colors.get(color, colors['white'])
    return f"{color_code}{text}{reset}"

def color_signal(row):
    signal = row['Signal']
    if 'Bullish' in signal:
        return '\033[92m' + signal + '\033[0m'
    elif 'Bearish' in signal:
        return '\033[91m' + signal + '\033[0m'
    else:
        return '\033[93m' + signal + '\033[0m'

def safe_format_float(val, fmt="{:7.2f}", na_str="N/A"):
    try:
        return fmt.format(float(val))
    except (ValueError, TypeError):
        return na_str


def plot_confidence_heatmap(df_results):
    st.subheader("🔥 ML Confidence Heatmap")
    
    df_plot = df_results.sort_values(by='Confidence', ascending=False).head(16).copy()
    
    if len(df_plot) == 0:
        st.warning("No data available to plot the heatmap.")
        return

    df_plot['Index'] = range(len(df_plot))
    df_plot['Row'] = (df_plot['Index'] // 5).astype(str) 
    df_plot['Col'] = (df_plot['Index'] % 5).astype(str)
    
    df_plot['Display_Text'] = df_plot.apply(
        lambda row: (
            f"{row['Ticker']} + ({row['_Extremes']})\n"
            f"${row['Price']:.2f}\n"
            f"R/R: {(row['Max (%)'] / abs(row['Loss (%)'])):.1f}\n"
            f"Conf: {row['Confidence']:.0f}%"
        ), 
        axis=1
    )
    
    df_plot['Tooltip_Detail'] = df_plot.apply(
        lambda row: (
            f"{row['Ticker']} + ({row['_Extremes']}) | Price: ${row['Price']:.2f} | "
            f"Gain: {row['Max (%)']:.1f}% | SL: {row['Loss (%)']:.1f}% | "
            f"Conf: {row['Confidence']:.0f}%"
        ), 
        axis=1
    )

    base = alt.Chart(df_plot).encode(
        x=alt.X('Col:N', axis=None),
        y=alt.Y('Row:N', axis=None),
    )
    
    heatmap = base.mark_rect().encode(
        color=alt.Color('Confidence:Q',
            scale=alt.Scale(
                domain=[5, 48, 95],           # Map 5->red, 48->white, 95->green
                range=['red', 'white', 'green'], 
                clamp=True
            ), 
            legend=alt.Legend(title="Confidence %"),
        ),
        tooltip=['Tooltip_Detail:N'] 
    )

    text = base.mark_text(
        align='center', 
        baseline='middle',
        lineBreak='\n' 
    ).encode(
        text=alt.Text('Display_Text:N'),
        color=alt.condition(
            alt.datum.Confidence > 60,
            alt.value('white'), 
            alt.value('black')
        )
    )

    chart = (heatmap + text).properties(
        title='Top ML Confidence: ~High Confidence/BULLISH, Low Confidence/BEARISH',
        width=600,
        height=400
    ).interactive()

    st.altair_chart(chart, use_container_width=False)

def generate_action(ticker, clean_label, conf, will_hit_str):
    colour = 'white'
    bull_case = {
        'TP': "BULLISH",
        'Hold': "HOLD"
    }

    bear_case = {
        'Short': "BEARISH",
        'SL': "BEARISH"
    }

    neutral_case = {
        'None': "NEUTRAL"
    }

    # Default signal text
    signal_text = (
        bull_case.get(clean_label)
        or bear_case.get(clean_label)
        or neutral_case.get(clean_label, "NEUTRAL — monitor for clearer signals.")
    )

    # Confidence-based interpretation
    if clean_label in bull_case and conf >= 80:
        action = (
            f"{ticker}: Prediction is extremely {signal_text}, "
            f"with ML {will_hit_str} & bull confidence ({conf:.0f}%) - BUY THE DIP"
        )
        colour = 'darkgreen'

    elif clean_label in bull_case and 60 <= conf < 80:
        action = (
            f"{ticker}: Prediction is {signal_text}, "
            f"ML {will_hit_str} & bull confidence ({conf:.0f}%) suggests - BUY THE DIP"
        )
        colour = 'green'

    elif clean_label in neutral_case and conf > 60:
        # Neutral label but high confidence → Buy-the-Dip
        confidence_text = f"{conf:.0f}%."
        action = (
            f"{ticker}: Prediction is {signal_text}, "
            f"Despite neutrality, the confidence *{confidence_text}* suggests Buy-the-Dip."
        )
        colour = 'lightgreen'

    elif clean_label in neutral_case and conf <= 20:
        action = (
            f"{ticker}: Prediction is {signal_text}, "
            f"Panic selling, tight SL - else SHORT - "
            f"ML ({will_hit_str}) & bear confidence ({conf:.0f}%) suggests candles/patterns trades only."
        )
        colour = 'white'

    elif clean_label in neutral_case and 40 <= conf <= 60:
        confidence_text = f"."
        action = (
            f"{ticker}: Prediction is {signal_text}, "
            f"ML indicates SIDEWAYS ({conf:.0f}%). Only trade patterns with SL."
        )
        colour = 'orange'

    elif clean_label in bear_case and 21 <= conf < 40:
        action = (
            f"{ticker}: Prediction is {signal_text} - "
            f"ML {will_hit_str} / {conf:.0f}% confidence suggest SHORT or HOLD POSITION/stay on sidelines."
        )
        colour = 'red'

    elif clean_label in bear_case and conf <= 20:
        confidence_text = f" ({conf:.0f}%) confidence indicates SELLERS Market / LONG-TERM up-trend(?) BUY-THE-DIP."
        action = (
            f"{ticker}: Prediction is {signal_text}, "
            f"{will_hit_str}, {confidence_text}"
        )
        colour = 'red'

    else:
        confidence_text = f"NEUTRAL ({conf:.0f}%)."
        action = (
            f"{ticker} is NEUTRAL - Check for Monthly Candle, patterns, divergences"
        )
        colour = 'gray'

    return action, colour

def get_action_label(confidence, 
                     will_hit_raw, 
                     current_price, 
                     ema1, 
                     rsi, 
                     ti_signal, 
                     predicted_return, 
                     predicted_loss):
    """
    ENHANCED ACTION LOGIC from backtest proven rules:
    1. SL/Short OVERRIDES everything (immediate exit risk)
    2. EMA proximity filter (avoid extended moves) 
    3. ML confidence threshold (63% backtest default)
    4. TI trend alignment (Bull/Hold context)
    5. R/R ratio validation
    """
                         
    if will_hit_raw is None or str(will_hit_raw).lower() == "nan":
        base = "None"
    else:
        base = str(will_hit_raw).split()[0]

    c = float(confidence)

    # 🔴 PRIORITY 1: DANGER SIGNALS (Backtest shows SL/Short = immediate exit)
    if base in ("SL", "Short") and c < 42:
        return label("Short/AVOID")

    # 🟢 PRIORITY 2: IDEAL BUY ZONE (Backtest sweet spot)
    ema_proximity = 0.95 <= current_price / ema1 <= 1.05  # Price near EMA1 (±5%)
    good_signal = base in ("None", "TP", "Hold")
    strong_trend = ti_signal in ("Bull", "Hold", "StrongBull")
    
    if (c >= 63 and good_signal and ema_proximity and strong_trend):
        # Bonus: Validate R/R > 1.0 (backtest uses predicted returns)
        rr_ratio = predicted_return / abs(predicted_loss) if predicted_loss != 0 else 0
        if rr_ratio >= 1.0:
            return label("STRONG BUY")
        return label("Buy")

    # 🟡 PRIORITY 3: Medium confidence → Wait (backtest avoids weak signals)
    elif 40 <= c < 63:
        return label("Wait")

    # 🔴 PRIORITY 4: Low confidence bearish → Short (backtest filters these out)
    elif c < 40 and c >= 20 and base in ("Bear", "Short"):
        return label("Short the RISE")

    # ⚠️ PRIORITY 5: Very low confidence → Risky (backtest shows poor performance)
    elif c < 20:
        return label("RISKY BUY")

    if ti_signal == "StrongBull":
        return label("Monitor")
    elif ti_signal in ("Bull", "Hold"):
        return label("Watch")
    
    return label("Wait")
    
#  🟡 PLOT TA
def plot_single_ticker(ticker, df, df_results, _window=14):
    predictions = df_results[df_results['Ticker'] == ticker].iloc[0]
    if predictions.empty:
        st.text(f"No prediction results found for ticker {ticker}, skipping plot.")
        return
    signal = predictions.Signal
    current_price = round(df['Close'].iloc[-1], 2)
    gain = round(predictions['Max (%)'], 1)
    loss = round(predictions['Loss (%)'], 1)
    rrr = abs(gain/loss)
    gain_price = current_price * (1 + gain/100)
    loss_price = current_price * (1 + loss/100)
    hit_prob = predictions.Hit_Prob
    conf = predictions.Confidence
    will_hit_str = df_results.loc[df_results['Ticker'] == ticker, 'Will_Hit'].values[0]
    prob_threshold = 70
    clean_label = re.sub(r'\(.*?\)|[\d\.]+', '', will_hit_str).strip()
    last_date = df.index[-1]
    future_date = last_date + pd.Timedelta(days=_window)
    avg_price = (current_price+loss_price)/2.
    EMA1_ = round(df['EMA1'].iloc[-1], 2)
    EMA2_ = round(df['EMA2'].iloc[-1], 2)
    R1 = round(df['R1_Avg'].iloc[-1], 2)
    R2 = round(df['R2_Avg'].iloc[-1], 2)
    PP = round(df['PP_Avg'].iloc[-1], 2)
    S1 = round(df['S1_Avg'].iloc[-1], 2)
    S2 = round(df['S2_Avg'].iloc[-1], 2)

    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(12, 6),
        dpi=600,
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )
    end_date = df.index[-1]
    start_date = end_date - pd.DateOffset(months=12)
    df = df.loc[start_date:end_date]
    ax1.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)
    df['Signal'] = np.select(
        [df['Bull']==1, df['Bear']==1],
        ['Bull', 'Bear'],
        default='Neutral'
    )
    price = df['Close'].rolling(3).mean()
    price.iloc[-1] = df['Close'].iloc[-1]
    color_map = {'Bull': 'green', 'Bear': 'red', 'Neutral': 'gray'}
    last_signal = df['Signal'].iloc[0]
    start_idx = 0
    for idx, (date, row) in enumerate(df.iterrows()):
        is_last = (idx == len(df) - 1)
        if row['Signal'] != last_signal or is_last:
            seg_idx = slice(start_idx, idx + 1)
            seg_price = price.iloc[seg_idx]
            seg_dates = df.index[seg_idx]
            ax1.plot(seg_dates, seg_price, color=color_map[last_signal], alpha=0.4, linewidth=2)
            start_idx = idx
            last_signal = row['Signal']
    kcount_absmax = df['Kcount'].abs().max()
    df['Kcount_sc'] = df['Kcount'] * (df['EMA1'] / kcount_absmax)
    ax1.plot(df.index, df['EMA1'], label=f'EMA{int(_DAYS*0.5)}', color='gold', alpha=0.7, linewidth=1.2)
    ax1.plot(df.index, df['EMA2'], label=f'EMA{int(_DAYS*2)}', color='red', alpha=0.7, linewidth=1.2, linestyle='--')
    ax1.plot(df.index, df['KCu'], color='blue', alpha=0.3, linestyle='--', linewidth=1)
    ax1.plot(df.index, df['KCl'], color='red', alpha=0.3, linestyle='--', linewidth=1)
    ax1_ = ax1.twinx()
    line_kcount, = ax1_.plot(df.index, df['Kcount_sc'], color='gray', alpha=0.15, linewidth=2, label='KC Cumm. touches', zorder=0)
    ax1_.set_yticks([])
    ax1_.set_ylabel('')
    for line in ax1.lines:
        line.set_zorder(3)
    ta.add_regression_forecast(ax1, df['EMA1'], last_date, color='orange')
    ta.add_regression_forecast(ax1, df['EMA2'], last_date, color='red')
    ax1.fill_between(df.index, df['EMA1'], df['EMA2'], where=(df['EMA1'] > df['EMA2']), facecolor='green', alpha=0.2, interpolate=True, label='BUY-times')
    ax1.fill_between(df.index, df['EMA1'], df['EMA2'], where=(df['EMA1'] <= df['EMA2']), facecolor='red', alpha=0.2, interpolate=True, label='Stay-away')
    if (_FIBS):
        fibs, fib_start, fib_end = get_recent_fib_levels(df)
        fib_colors = {'F:0': 'gray','F:100': 'gray','F:61.8': 'blue','F:125': 'green','F:-125': 'red'}
        for label, value in fibs.items():
            ax1.hlines(value, xmin=fib_start, xmax=fib_end, color=fib_colors[label], linestyle='--', linewidth=1, alpha=0.3)
            ax1.annotate(f'{label}: ${value:.0f}', xy=(df.index[-5], value), xytext=(-5, 0), textcoords='offset points', va='center', fontsize=8, color=fib_colors[label], alpha=0.5)
    ax1.plot([last_date, future_date], [avg_price, gain_price], color='green', linestyle=':', linewidth=1.5, alpha=0.5)
    ax1.plot([last_date, future_date], [avg_price, loss_price], color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    ax1.plot(future_date, gain_price, '^', markersize=_ms, color='green', alpha=0.5, label=f'TP: ${gain_price:.2f}, {gain}%')
    ax1.plot(future_date, loss_price, 'v', markersize=_ms, color='red', alpha=0.5, label=f'SL: ${loss_price:.2f}, {loss}%')
    ax1.plot(last_date, avg_price, 'o', markersize=_ms, color='orange', alpha=0.5, label=f'E: ${avg_price:.2f}')
    ax1.annotate(f'E: ${avg_price:.2f}', xy=(last_date, avg_price), xytext=(10, 0), textcoords='offset points', ha='left', va='center', color='orange', fontsize=9, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    ax1.annotate(f'${current_price:.2f}\t-\t${gain_price:.2f}\n+{predictions["Max (%)"]:.1f}%',
                 xy=(future_date, gain_price), 
                 xytext=(10, 10), 
                 textcoords='offset points', ha='left', va='bottom', color='green', 
                 fontsize=9, fontname='DejaVu Sans', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    ax1.annotate(f'${current_price:.2f}\t-\t${loss_price:.2f}\n{predictions["Loss (%)"]:.1f}%',
                 xy=(future_date, loss_price), 
                 xytext=(10, -10), 
                 textcoords='offset points', ha='left', va='top', color='red', 
                 fontsize=9, fontname='DejaVu Sans', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    signal_color = (
        'green' if 'Bull' in predictions['Signal'] else
        'red' if 'Bear' in predictions['Signal'] else
        'yellow' if 'Hold' in predictions['Signal'] else
        'gray'
    )
    _sigConf = f'{predictions['Signal']} & ML Action: {predictions.Action}, Conf ({conf:.0f}%)'
    ax1.annotate(
        _sigConf,
        xy=(0.7, 0.95),
        xycoords='axes fraction',
        ha='right',
        va='top',
        fontsize=10,
        weight='bold',
        fontname='DejaVu Sans',
        bbox=dict(boxstyle='round', facecolor=signal_color, alpha=0.2, edgecolor=signal_color)
    )
    ax1.text(0.5, 0.5, f'@{ticker}', transform=ax1.transAxes, fontsize=50, color='grey', alpha=0.2, horizontalalignment='center', verticalalignment='center', rotation=0, weight='bold', style='italic')
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    ax1.set_ylabel('Price')
    ax1.set_title(f'{today}:\t{ticker} - {predictions["Signal"]}', fontdict={'fontname': 'DejaVu Sans', 'fontsize': 16}, pad=20)
    ax1.scatter(df.index[df['StrongBull'] == 1], price[df['StrongBull'] == 1], color='lime', marker='^', s=5, alpha=0.4, label='StrongBull', zorder=10)
    ax1.scatter(df.index[df['StrongBear'] == 1], price[df['StrongBear'] == 1], color='red', marker='v', s=5, alpha=0.4, label='StrongBear', zorder=10)

    rsi_ = df['RSI'].rolling(3).mean()
    rsi_sma = df['RSI'].rolling(20).mean()
    ax2.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax2.plot(df.index, rsi_, label='RSI', color='gray', linewidth=1.5, alpha=0.5)
    ax2.plot(df.index, rsi_sma, label='RSI SMA', color='red', linewidth=1.2, alpha=0.35)
    ax2.fill_between(df.index, rsi_, 52, where=(df['RSI'] > 52), facecolor='green', alpha=0.15)
    ax2.fill_between(df.index, rsi_, 40, where=(df['RSI'] < 40), facecolor='red', alpha=0.15)
    ax2.fill_between(df.index, rsi_, rsi_sma, where=((df['RSI'] < df['RSI_SMA']) & (df.EMA1 > df.EMA2)), facecolor='orange', alpha=0.3, label='Dip(?)')
    
    rsi_last = round(df['RSI'].iloc[-1], 1)
    rsi_sma_last = round(df['RSI'].rolling(20).mean().iloc[-1], 1)
    price_vs_EMA1 = 100 * (current_price - EMA1_) / EMA1_ if EMA1_ != 0 else 0
    ax2.scatter(df.index[df['Bull'] == 1], rsi_[df['Bull'] == 1], color='green', marker='^', s=5, alpha=0.4, label='Bull', zorder=7)
    ax2.scatter(df.index[df['Bear'] == 1], rsi_[df['Bear'] == 1], color='red', marker='v', s=5, alpha=0.4, label='Bear', zorder=8)
    ax2.scatter(df.index[df['Short'] == 1], rsi_[df['Short'] == 1], color='red', marker='x', s=5, alpha=0.4, label='Short', zorder=10)
    ax2.scatter(df.index[df['Hold'] == 1], rsi_[df['Hold'] == 1], color='orange', marker='o', s=5, alpha=0.4, label='Hold', zorder=10)
    ax2.axhline(80, color='green', linewidth=1, linestyle='dotted', alpha=0.3)
    ax2.axhline(20, color='red', linewidth=1, linestyle='dotted', alpha=0.3)
    ax2.axhline(40, color='brown', linewidth=1, linestyle='dashed', alpha=0.3)
    ax2.axhline(52, color='gray', linewidth=1.2, linestyle='dashed', alpha=0.3)
    ax2.set_ylim(0, 100)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel('RSI')
    mid_date = df.index[len(df.index)//2]
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize='x-small')
    ax2.legend(loc='upper left', fontsize='x-small')
    bull_div, bear_div, hbull_div, hbear_div = ta.detect_divergences(df, period=20)
    dtop, dbot = ta.find_doubleTopBottom(df, tol=0.5, max_bar_diff=5)
    ta.plot_divergences(df, bull_div, bear_div, hbull_div, hbear_div, dtop, dbot, ax1, ax2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate()
    
    strong_bull = (df['RSI'].iloc[-1] > 52) and (df['ADX'].iloc[-1] > 22) and (df['sumBuyVol'].iloc[-1] > df['sumSellVol'].iloc[-1])
    strong_bear = (df['RSI'].iloc[-1] < 40) and (df['ADX'].iloc[-1] > 22) and (df['sumBuyVol'].iloc[-1] < df['sumSellVol'].iloc[-1])
    
    summary_lines = [
        f"Trend: EMA1 ({EMA1_}) is {'above' if EMA1_ > EMA2_ else 'below'} EMA2 ({EMA2_}) → Market is {'bullish' if EMA1_ > EMA2_ else 'bearish'}.",
        f"Momentum: RSI = {rsi_last} ({'above' if rsi_last > rsi_sma_last else 'below'} its 20-day average of {rsi_sma_last}).",
        f"Price: ${current_price} is {abs(price_vs_EMA1):.2f}% {'above' if price_vs_EMA1 > 0 else 'below'} EMA1.",
        f"Resistance PIVOTS: (${R1}, ${R2}), PIVOT(${PP}), Support PIVOTS ( ${S1}, ${S2}).",
        f"Trend Strength: Strong Bull: {'Yes' if strong_bull else 'No'}, Strong Bear: {'Yes' if strong_bear else 'No'}.",
        f"Model Signal: {signal} | Expected Gain: +{gain}% (${gain_price:.2f}), Loss: {loss}% (${loss_price:.2f}) | Hit Probability: {round(hit_prob, 1)}%."
    ]

    sig_ = f'{signal}\tR/R: {rrr:.1f}\tML Conf: {conf:.0f}%'
    action, cl = generate_action(ticker, clean_label, conf, will_hit_str)
    summary_lines.append(action)

    textbox = AnchoredText(
       action,
       loc='lower left',
       frameon=True,
       borderpad=1.5,
       prop=dict(size=7, color= 'blue', weight='normal')
    )
    
    textbox.patch.set(facecolor= cl, edgecolor='gray', alpha=0.4, boxstyle='round')
    ax1.add_artist(textbox)
  
    plt.tight_layout()
    st.pyplot(fig)
    with st.expander(f'{action}, {sig_}'):
        st.code("\n".join(summary_lines))

                           
#  🟡 Make Predictions (Gain/Loss/Confidence)
def MakePredictions(TICKERS = "AAPL, GOOGL, MSFT"):
    
    n = 1
    dfs = {}
    results = []
    label2str = {0: 'None', 1: 'SL', 2: 'TP', 3: 'Hold', 4: 'Short'}
    expected_classes = [0, 1, 2, 3, 4]
    
    for ticker in TICKERS:
        try:
            df = get_stock_data(ticker, start_date, end_date)
            if not pd.api.types.is_datetime64_any_dtype(df.index):
                if "Date" in df.columns:
                    df = df.set_index("Date")
                else:
                    raise ValueError("DataFrame must have Date as index or column for plotting!")
            df = add_technical_indicators(df)
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            df['BuyTime'] = (
                (df['Bull'] == 1) &
                ((df['Close'] - df['EMA1']) / df['EMA1'] <= 0.02)
            )
            df = add_pivot_levels(df, window=14)
            df = add_pivots(df, windows)
            df = average_pivots(df, windows)
            df = compute_expected_return(df, forward_window=14, r_cols=['R1_Avg', 'R2_Avg'])
            df = compute_expected_loss(df, forward_window=14, s_cols=['S1_Avg', 'S2_Avg'])
            df = label_hit_prob_past(df, window=30, profit_target=PROFIT_TARGET, stop_loss=STOP_LOSS, lookback=120, tp_thresh=0.35, sl_thresh=0.35)
            df['Hit_Label'] = df['Hit_Label'].fillna(0).astype(int)
            
            dfs[ticker] = df
            
            df_model = df.dropna(subset=FEATURES + ['Hit_Label', 'Expected_Return', 'Expected_Loss'])
            if len(df_model) < _Nr:
                st.text(f"Skipping {ticker} due to insufficient data after dropna.")
                continue
            
            # --- Step 1: Train TP Hit Classifier ---
            X_cls = df_model[FEATURES]
            y_cls = df_model['Hit_Label'].astype(int)
            scaler_cls = StandardScaler()
            X_scaled_cls = scaler_cls.fit_transform(X_cls)
            X_train_cls, X_val_cls, y_train_cls, y_val_cls = train_test_split(
                X_scaled_cls, y_cls, test_size=0.2, random_state=42)
            
            model_class = RandomForestClassifier(
                n_estimators=120, 
                max_depth=12, 
                min_samples_split=4,
                min_samples_leaf=3,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42
            )

            model_class.fit(X_train_cls, y_train_cls)
            
            # --- Step 2: Extract Full Class Probabilities as Features ---
            cls_probs = model_class.predict_proba(X_scaled_cls)
            # Extract probability columns for all expected classes safely
            prob_df = pd.DataFrame(0, index=np.arange(len(cls_probs)), columns=[f'Prob_Class_{c}' for c in expected_classes])
            for i, c in enumerate(model_class.classes_):
                if c in expected_classes:
                    prob_df[f'Prob_Class_{c}'] = cls_probs[:, i]
            
            df_model = df_model.reset_index(drop=True)
            df_model = pd.concat([df_model, prob_df], axis=1)
            FEATURES_with_probs = FEATURES + [f'Prob_Class_{c}' for c in expected_classes]
            X_reg = df_model[FEATURES_with_probs]
            
            # --- Step 3: Train Return Model ---
            y_return = df_model['Expected_Return']
            scaler_return = StandardScaler()
            X_scaled_return = scaler_return.fit_transform(X_reg)
            X_train_ret, X_val_ret, y_train_ret, y_val_ret = train_test_split(
                X_scaled_return, y_return, test_size=0.2, random_state=42)

            model_return = RandomForestRegressor(
                n_estimators=120,
                max_depth=14,
                min_samples_leaf=3,
                max_features='sqrt',
                ccp_alpha=0.001,
                random_state=42,
                n_jobs=-1
            )
            model_return.fit(X_train_ret, y_train_ret)
            
            # --- Step 4: Train Loss Model ---
            y_loss = df_model['Expected_Loss']
            scaler_loss = StandardScaler()
            X_scaled_loss = scaler_loss.fit_transform(X_reg)
            X_train_loss, X_val_loss, y_train_loss, y_val_loss = train_test_split(
                X_scaled_loss, y_loss, test_size=0.2, random_state=42)
            model_loss = RandomForestRegressor(
                n_estimators=120,
                max_depth=14,
                min_samples_leaf=3,
                max_features='sqrt',
                ccp_alpha=0.001,
                random_state=42,
                n_jobs=-1
            )
            model_loss.fit(X_train_loss, y_train_loss)
            
            # --- Step 5: Live Prediction ---
            latest = df.iloc[[-1]]
            if latest[FEATURES].isnull().values.any():
                st.text(f"Skipping {ticker} for NULL Features")
                null_features = latest[FEATURES].iloc[0].isnull()
                st.text(f"NaN features for {ticker}: {list(null_features[null_features].index)}")
                continue
            
            latest_scaled_cls = scaler_cls.transform(latest[FEATURES])
            latest_probs_raw = model_class.predict_proba(latest_scaled_cls)[0]
    
            # Compute probabilities for all expected classes
            latest_prob_features = {}
            for c in expected_classes:
                if c in model_class.classes_:
                    idx = model_class.classes_.tolist().index(c)
                    latest_prob_features[f'Prob_Class_{c}'] = latest_probs_raw[idx]
                else:
                    latest_prob_features[f'Prob_Class_{c}'] = 0.0
                    
            # Predict class based on max probability among expected classes
            probs_of_interest = [latest_prob_features[f'Prob_Class_{c}'] for c in expected_classes]
            max_prob_index = probs_of_interest.index(max(probs_of_interest))
            pred_class = expected_classes[max_prob_index]
            
            will_hit = label2str.get(pred_class, "None")
            if pd.isna(will_hit):
                will_hit = "None"
                
            hit_prob = latest_prob_features[f'Prob_Class_{pred_class}']
            
            # Prepare latest features including probability features for regressors
            latest_prob_df = pd.DataFrame([latest_prob_features])
            latest_features_with_probs = pd.concat([latest[FEATURES].reset_index(drop=True), latest_prob_df], axis=1)
            latest_scaled_return = scaler_return.transform(latest_features_with_probs)
            latest_scaled_loss = scaler_loss.transform(latest_features_with_probs)
            
            current_price = latest['Close'].values[0]
            predicted_return = model_return.predict(latest_scaled_return)[0]
            predicted_loss = model_loss.predict(latest_scaled_loss)[0]
            predicted_tp = current_price * (1 + predicted_return)
            predicted_sl = current_price * (1 + predicted_loss)
            entry_price = (current_price + predicted_sl) / 2
            entry_discount_pct = ((current_price - entry_price) / entry_price) * 100

            # Confidence calculation
            p_none  = latest_prob_features.get('Prob_Class_0', 0)
            p_sl    = latest_prob_features.get('Prob_Class_1', 0)
            p_tp    = latest_prob_features.get('Prob_Class_2', 0)
            p_hold  = latest_prob_features.get('Prob_Class_3', 0)
            p_short = latest_prob_features.get('Prob_Class_4', 0)
    
            bullish_prob = p_tp + p_hold
            bearish_prob = p_sl + p_short
        	
            if predicted_loss != 0:
                rr_ratio = predicted_return / abs(predicted_loss)
            else:
                rr_ratio = 0
    
            max_ratio = 10  # cap at 10:1
            log_ratio = np.log1p(rr_ratio)  # log(1+ratio)
            max_log_ratio = np.log1p(max_ratio)
            normalized_rr = log_ratio / max_log_ratio
        
            total_prob = bullish_prob + bearish_prob
            if total_prob > 0:
                prob_confidence = bullish_prob / total_prob 
            else:
                prob_confidence = 0.5
        
            confidence_score = (0.5 * prob_confidence + 0.5 * normalized_rr) * 100

            rsi = latest['RSI'].values[-1]
            signal = "TI: ⚪ Neut"
            _Extremes = "High" if df['Exhaustion'].values[-1] >= 0.9 else ("Low" if df['Exhaustion'].values[-1] <= -0.9 else "--")
                
            entry_signal = True
            sc = 'white'
            lookback_n = 5
            bull_mode = pd.Series(df.Bull.values[-lookback_n:]).mode().iloc[0]
            bear_mode = pd.Series(df.Bear.values[-lookback_n:]).mode().iloc[0]
            neutral_mode = pd.Series(df.Neutral.values[-lookback_n:]).mode().iloc[0]
            hit_price = None

            TI = df.TI.values[-1]

            if TI == 'Bull':
                signal = "TI: ✅ Bullish"
            elif TI == 'Bear':
                signal = "TI: 🔻 Bearish"
            elif TI == 'Hold':
                signal = "TI: 🟡 Hold"
            else: 
                signal = "TI: ⚪ Neutral"
            
            if will_hit == 'TP':
                hit_price = predicted_tp
                sc = 'green'
            elif will_hit == 'Hold':
                hit_price = predicted_tp
                sc = 'orange'
            elif will_hit == 'SL':
                hit_price = predicted_sl
                sc = 'red'
            elif will_hit == 'Short':
                hit_price = predicted_sl
                sc = 'darkred'
            else:
                hit_price = None
                sc = 'white'
    
            def safe_format_float(val, fmt="{:7.2f}", na_str="N/A"):
                try:
                    return fmt.format(float(val))
                except (ValueError, TypeError):
                    return na_str
            
            tp_str = safe_format_float(predicted_tp)
            sl_str = safe_format_float(predicted_sl)
            atr_str = safe_format_float(df['ATR'].iloc[-1], fmt="{:5.1f}")
            
            if hit_price is not None and isinstance(hit_price, (int, float, np.floating)):
                hit_price_str = f"${hit_price:>5.2f}"
            else:
                hit_price_str = "None"
            
            # Derive cleaned will_hit base for logic
            will_hit_base = will_hit if will_hit is not None else "None"
            if isinstance(will_hit_base, str):
                will_hit_base = will_hit_base.split()[0]
            
            # Compute action
            ema1_val = latest['EMA1'].iloc[0]
            rsi_val = latest['RSI'].iloc[0]
            ti_signal = df['TI'].iloc[-1]
            #rr_ratio = predicted_return / abs(predicted_loss) if predicted_loss != 0 else 0
            
            # Enhanced action with backtest-proven filters
            action = get_action_label(
                confidence_score, will_hit_base, 
                current_price, ema1_val, rsi_val, ti_signal,
                predicted_return, predicted_loss
            )
                        
            row_text = (
                f"{extract_emojis(signal):<2} "
                f"{ticker:<7} | "
                f"${current_price:>7.2f} | "
                f"${tp_str:>7} | "
                f"${sl_str:>7} | "
                f"${atr_str:>10} | "
                f"{action:<12} | "
                f"Conf: {confidence_score:>4.0f}% | " 
                f"{_Extremes}"
            )

            st.code(strip_ansi_codes(row_text))
            
            # Append results with formatted Will_Hit string
            if will_hit is None or str(will_hit).lower() == "nan":
                will_hit = "None"
        
            if will_hit == 'TP':
                hit_price_rounded = round(predicted_tp, 2)
            elif will_hit == 'SL':
                hit_price_rounded = round(predicted_sl, 2)
            else:
                hit_price_rounded = None
            
            if hit_price_rounded is not None:
                will_hit_str = f"{will_hit} (${hit_price_rounded})"
            else:
                will_hit_str = will_hit
            
            results.append({
                "Index": n >4,
                "Ticker": ticker,
                "Date": latest.index[-1].date(),
                "Price": round(current_price, 1),
                "Entry": round(entry_price, 1),
                "Dip%": round(entry_discount_pct * -1, 1),
                "TP": round(predicted_tp, 1),
                "Max (%)": round(predicted_return * 100, 1),
                "SL": round(predicted_sl, 1),
                "Loss (%)": round(predicted_loss * 100, 1),
                "Risk": "🔴 High Risk" if (abs(predicted_loss) > STOP_LOSS) else "🟢 Low Risk",
                "Signal": signal,
                "Will_Hit": will_hit_str,
                "Hit_Prob": round(latest_prob_features[f'Prob_Class_{pred_class}'] * 100, 1),
                "Confidence": round(confidence_score, 1),
                "_Extremes": _Extremes,
                "Action" : action
            })
        except Exception as e:
            st.text(f"Error processing {ticker}: {e}")
    df_results = pd.DataFrame(results)
    return dfs, df_results


# ✅ PLOT PREDICTIONS
def PlotPredictions(df_results):
    
    tickers = df_results['Ticker']
    tickers_list = tickers.tolist()
    
    df_plot = df_results
    df_plot = df_plot.sort_values(by="Confidence", ascending=False)
    max_vals = df_plot["Max (%)"].to_numpy()
    norm = mcolors.Normalize(vmin=min(max_vals), vmax=max(max_vals))
    cmap = cm.jet #Inverse of spectral
    custom_colors = cmap(norm(max_vals))
    
    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=300)
    cax = inset_axes(ax1, width="2%", height="60%", loc='center right',
                     bbox_to_anchor=(0.12, 0., 1, 1),
                     bbox_transform=ax1.transAxes,
                     borderpad=0)
    
    # Main bar plot
    ax1.bar(df_plot["Ticker"], max_vals, color=custom_colors, alpha = 0.4)
    ax1.set_ylabel('Max Return (%)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add colorbar at the right of the plot
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax, orientation='vertical', label="Colored by: Max (%)", alpha = 0.4)
    cbar.ax.tick_params(labelsize=8)
    
    # Secondary axis for loss line
    ax2 = ax1.twinx()
    sns.lineplot(x="Ticker", y="Loss (%)", data=df_plot, color='red', marker='o',
                 ax=ax2, linewidth=2, markersize=8, label='Expected Loss')
    ax2.set_ylabel('Expected Loss (%)', fontsize=12, color='red')
    
    combined_min = min(ax1.get_ylim()[0], -ax2.get_ylim()[1])
    combined_max = max(ax1.get_ylim()[1], -ax2.get_ylim()[0])
    ax1.set_ylim(combined_min, combined_max)
    ax2.set_ylim(-combined_max, -combined_min)
    ax2.spines['right'].set_color('red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.invert_yaxis()
    ax1.legend(fontsize='small')
    ax2.legend(fontsize='small') 
    
    # --- ANNOTATIONS ALIGNED BELOW X-TICK LABELS ---
    x_ticks = ax1.get_xticks()
    for i, (_, row) in enumerate(df_plot.iterrows()):
        # Color assignment for signal types
        fcolor = (
            'green' if "Bull" in row.Signal
            else 'red' if "Bear" in row.Signal
            else 'yellow' if "Hold" in row.Signal
            else 'white'
        )
        
        if row.Confidence > 61 and str(row.Will_Hit).split()[0] in ['TP', 'Hold', 'None']:
            ProbColor = 'green'
        elif row.Confidence > 61 and str(row.Will_Hit).split()[0] in ['Hold']:
            ProbColor = 'orange'
        elif "Bear" in row.Signal and row.Confidence < 40 and str(row.Will_Hit).split()[0] in ['SL', 'None']:
            ProbColor = 'red'
        else:
            ProbColor = 'white'
    
        # Top annotations
        ax1.text(i, row["Max (%)"] + 0.5, f'{row["Max (%)"]:.1f}%',
                 ha='center', va='bottom', fontsize=9)
        ax2.text(i, row["Loss (%)"] + 0.5, f'{row["Loss (%)"]:.1f}%',
                 ha='center', va='top', color='red', fontsize=9)
    
        # Bottom annotations: align with x-tick, just below tick label
        #x_tick = x_ticks[i]
        x_coord = i
        x_offset = -0.4 # to fix x-shift if colorbar is added, else put this to zero.
        y_offset1 = -0.275  # Adjust as needed for your plot
        y_offset2 = -0.575  # Stagger if two boxes per tick
    
        ax1.text(
            x_coord + x_offset, y_offset1,
            f'{row["Risk"]}\nP: ${row["Price"]:.2f}\nE: ${row["Entry"]:.2f}\nDip: {row["Dip%"]:.1f}%\n{row["Signal"]}',
            ha='left', va='top', fontsize=7, fontname='DejaVu Sans',
            bbox=dict(facecolor=fcolor, alpha=0.3, linewidth=0.3),
            transform=ax1.get_xaxis_transform(),
            multialignment='left',
            clip_on=False
        )
            
        ax1.text(
            x_coord + x_offset, y_offset2,
            f'TP: ${row["TP"]:.2f}\nSL: ${row["SL"]:.2f}\n\n{str(row.Will_Hit).split()[0]}: {row.Hit_Prob:.0f}%\nConf: {row.Confidence:.0f}%',
            ha='left', va='top', fontsize=7, fontname='DejaVu Sans',
            bbox=dict(facecolor=ProbColor, alpha=0.3, linewidth=0.3),
            transform=ax1.get_xaxis_transform(),
            clip_on=False
        )
        
    # Strategic hint box
    textbox = AnchoredText(
        "Hint: Buy closer to predicted SL to reduce risk\nand increase the chance of success.",
        loc='lower left',
        frameon=True,
        borderpad=1.5,
        prop=dict(size=10, color='gray', weight='bold')
    )
    ax1.add_artist(textbox)
    textbox.set_clip_on(True)
    textbox.set_in_layout(True)
    textbox.set_zorder(100)
    textbox.patch.set_facecolor('honeydew')
    textbox.patch.set_edgecolor('darkgreen')
    textbox.patch.set_alpha(0.8)
    
    # Space management
    plt.title(f'{today} - ML Predictions of Tickers (From Current Price)', fontsize=16, color='black', pad=20)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35)

    st.pyplot(fig)

def is_valid_ticker(ticker):
    try:
        df = yf.Ticker(ticker).history(period="1d")
        return not df.empty
    except Exception:
        return False
        
###### Tabulate Data
def style_rows(row):
    signal = row.Signal
    hits = row.Will_Hit
    hit_prob = row.Hit_Prob
    conf = row.Confidence

    # Exhaustion logic
    extreme = row.get("_Extremes", None)

    # 1️⃣ Exhaustion High → GRAY
    if extreme == "High":
        return ['background-color: rgba(128, 128, 128, 0.5)'] * len(row)

    # 2️⃣ Exhaustion Low → LIGHT RED
    if extreme == "Low":
        return ['background-color: rgba(255, 182, 193, 0.4)'] * len(row)

    # 3️⃣ Hold + Hold → Magenta
    if ('Hold' in signal) and ('Hold' in hits):
        return ['background-color: rgba(255, 0, 255, 0.3)'] * len(row)

    # 4️⃣ Bullish TP / None with good confidence
    if ('Bull' in signal) and (('TP' in hits) or ('None' in hits)) \
       and (conf > 60) and (row['Max (%)'] > abs(row['Loss (%)'])):
        return ['background-color: rgba(144, 238, 144, 0.3)'] * len(row)

    # 5️⃣ Bearish / SL / Short with low confidence
    if (('Bear' in signal) or ('SL' in hits) or ('Short' in signal)) and (conf < 40):
        return ['background-color: rgba(240, 128, 128, 0.3)'] * len(row)

    # 6️⃣ Default → gray text
    return ['color: gray'] * len(row)

def tabular_display(df_results):
    _df = df_results.copy()
    _df['Signal'] = _df['Signal'].str.replace(r'^TI:\s*', '', regex=True)
    _df['Will_Hit'] = _df['Will_Hit'].str.replace(r'\([^)]*\)', '', regex=True)
    _df['Will_Hit'] = _df['Will_Hit'].str.replace(r'[^A-Za-z]+', '', regex=True)

    custom_order = ['TP', 'Hold', 'None', 'SL', 'Short']
    ord_map = {label: i for i, label in enumerate(custom_order)}
    _df['who'] = _df['Will_Hit'].map(lambda x: ord_map.get(x, len(custom_order)))

    _df_sorted = _df.sort_values(
        by=['Confidence', 'who', '_Extremes', 'Signal'],
        ascending=[False, True, False, False]
    ).reset_index(drop=True)
    _df_sorted = _df_sorted.drop(columns=['Index', 'who'], errors='ignore')

    def custom_price_format(x):
        try:
            if x > 1:
                return f"{x:.2f}"
            else:
                return f"{x:.4f}"  # or f"{x:.2f}" if you want 2 decimals always
        except:
            return x  # if x is not a number, return as is
    
    styled_df = _df_sorted.style.apply(style_rows, axis=1).format({
        'Price': custom_price_format,
        'Entry': custom_price_format,
        'TP': custom_price_format,
        'SL': custom_price_format,
        'Dip%' : '{:.1f}',
        'Max (%)': '{:.0f}',
        'Loss (%)': '{:.0f}',
        'Confidence': '{:.0f}',
        'Hit_Prob': '{:.0f}',
        'Confidence': '{:.0f}'
    })

    st.dataframe(styled_df, height=550, use_container_width=True)

def run_app():

    with st.expander("Positional/Swing Trading Guidance"):
        st.write(desc)  
    
    with st.expander("Signals & Stocks Selection"):
        st.write(HowTo)

    with st.expander("Common Mistakes"):
        st.write(mistakes)
    with st.expander("Disclaimer"):
        st.write(disclaimer)

    #st.header("Machine Learning Signals (Technical Analysis)")  
    st.title("📈 Machine Learning Signals (TA)")

    tickers_input = st.text_input("Enter comma-separated tickers (max 15):", placeholder = "e.g., COIN, TSLA, BTC-USD, ETH-USD")
    
    if tickers_input:
        TICKERS = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        
        if len(TICKERS) > 15:
            st.error("You can enter up to 15 tickers only. Please reduce your list.")
        else:
            valid_tickers = []
            invalid_tickers = []
            for t in TICKERS:
                if is_valid_ticker(t):
                    valid_tickers.append(t)
                else:
                    invalid_tickers.append(t)
    
            if not valid_tickers:
                st.error(f"All tickers are invalid: {', '.join(invalid_tickers)}. Please enter valid tickers.")
            else:
                if invalid_tickers:
                    st.warning(f"Ignoring invalid tickers: {', '.join(invalid_tickers)}")
                st.code(f"Valid tickers to process ({len(valid_tickers)}): {', '.join(valid_tickers)}")
                st.code(f"The indicators use OHLC with a mean of 2-days to suppress noise/spikes")
                
        row_text = (
            f'{"#":<2} | '
            f'{"Ticker":<7} | '
            f'{"Price":>7} | '
            f'{"TP":>8} | '
            f'{"SL":>8} | '
            f'{"ATR":>10} | '
            f'{"Action":<12} | '
            f'{"ML(%)":>10} | '
            f'{"Extremes":<10}'
        )

        st.code(row_text)
        dfs, df_results = MakePredictions(TICKERS)
        
        plot_confidence_heatmap(df_results)
        PlotPredictions(df_results)


        with st.expander("Tabular Results"):
            st.write("""
            The results are tabulated that can be manually sorted or downloaded. Look for the stocks with higher confidence >65%, TP/Bullishness and double check the graph and 'is High'.
    
            Avoid buying near tops, or near highs, or ATHs. The first rule is to buy low and the second rule is to buy closer to SL or buy 3-red days or 3-red weeks.
    
            This practice avoids unnecessary chasing or entries.
            """)
                  
            st.markdown(
                """
                Tabular display of results,<br>
                <span style='color:green;'>Bullish (Green),</span><br>
                <span style='color:red;'>Bearish (Red),</span><br>
                <span style='color:gray;'>Neutral (Gray),</span><br>
                stocks touching 3-months high are in <span style='color:darkgray;'>darkgray</span>.<br>
                Use column filters to further fine-tune the stocks to trade/compound & build positions.<br>
                Good luck!
                """, 
                unsafe_allow_html=True
            )
        
        tabular_display(df_results)
        st.session_state['ml_results'] = df_results
        for ticker in TICKERS:
            _df = dfs.get(ticker)
            if _df is None:
                st.text(f"Skipping {ticker}: no preloaded data available")
                continue
            plot_single_ticker(ticker, _df, df_results)  
        
# Call this only in streamlit run mode
if __name__ == "__main__":
    run_app()
