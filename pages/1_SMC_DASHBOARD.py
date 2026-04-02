# streamlit_smc_yfinance.py
"""
Single-file Streamlit app:
- Downloads 2 years of OHLCV via yfinance (cached)
- UI: ticker, timeframe (1D/1W/1M), EMA/RSI/ATR params, swing detection, min_gap_frac
- Computes EMAs, RSI, ATR; detects 3-candle FVG and 3-candle OB using numpy arrays
- Shows 3-column recommendation (Go Long / Hold / Go Short)
- Matplotlib price chart with EMAs and semi-transparent zone rectangles
- Shows market structure (last swing highs/lows and BOS/CHoCH)
- CSV download of sliced data
Run: streamlit run streamlit_smc_yfinance.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import io

st.set_page_config(layout="wide", page_title="SMC Signals (yfinance)")

# -------------------------
# Cached download helper
# -------------------------
@st.cache_data(ttl=300)
def download_yf(ticker: str, period_days: int, interval: str) -> pd.DataFrame:
    """
    Download OHLCV from yfinance and normalize column names.
    period_days: number of days to request (we pass 365*2 for 2 years)
    interval: '1d', '1wk', '1mo'
    """
    period = f"{period_days}d"
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    # unify column names to lowercase
    df = df.rename(columns=lambda c: c.lower())
    # ensure datetime column name
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'datetime'})
    if 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(df.index)
    # keep only required columns
    cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]
    df = df.sort_values('datetime').reset_index(drop=True)
    return df

# -------------------------
# Indicator helpers
# -------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/length, adjust=False).mean()
    ma_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def pivot_high_idxs_np(high_np: np.ndarray, left: int = 20, right: int = 5):
    idxs = []
    n = len(high_np)
    # ensure left/right are within bounds
    left = max(1, int(left))
    right = max(1, int(right))
    for i in range(left, n - right):
        window = high_np[i-left:i+right+1]
        if np.isnan(window).any():
            continue
        if high_np[i] == window.max():
            idxs.append(i)
    return idxs

def pivot_low_idxs_np(low_np: np.ndarray, left: int = 20, right: int = 5):
    idxs = []
    n = len(low_np)
    left = max(1, int(left))
    right = max(1, int(right))
    for i in range(left, n - right):
        window = low_np[i-left:i+right+1]
        if np.isnan(window).any():
            continue
        if low_np[i] == window.min():
            idxs.append(i)
    return idxs

# -------------------------
# Sidebar / Controls
# -------------------------
st.title("SMC Signals Dashboard (yfinance)")

with st.sidebar:
    st.header("Data & Parameters")
    ticker = st.text_input("Ticker", value="AAPL").upper()
    timeframe = st.selectbox("Timeframe", options=["1D","1W","1M"], index=0)
    interval_map = {"1D":"1d","1W":"1wk","1M":"1mo"}
    yf_interval = interval_map[timeframe]
    # always 2 years slice
    period_days = 365 * 2

    st.markdown("**Indicator params**")
    ema_short = st.number_input("EMA Short", value=20, min_value=1)
    ema_med   = st.number_input("EMA Medium", value=50, min_value=1)
    ema_long  = st.number_input("EMA Long", value=200, min_value=1)
    rsi_len   = st.number_input("RSI Length", value=14, min_value=1)
    atr_len   = st.number_input("ATR Length", value=14, min_value=1)

    st.markdown("**Zone / Swing params**")
    swing_left = st.number_input("Swing Left", value=20, min_value=1)
    swing_right = st.number_input("Swing Right", value=5, min_value=1)
    min_gap_frac = st.number_input("Min gap (ATR fraction)", value=0.1, step=0.05, min_value=0.0)
    max_zones = st.number_input("Max Zones to draw", value=200, min_value=1)

# -------------------------
# Data download & compute
# -------------------------
st.info(f"Downloading 2 years of {ticker} @ {timeframe} (interval={yf_interval}) ...")
df = download_yf(ticker, period_days, yf_interval)

if df.empty:
    st.error("No data returned for this ticker/timeframe. Try another ticker or timeframe.")
    st.stop()

# compute indicators (vectorized)
df['ema_short'] = ema(df['close'], ema_short)
df['ema_med']   = ema(df['close'], ema_med)
df['ema_long']  = ema(df['close'], ema_long)
df['rsi'] = rsi(df['close'], rsi_len)

# ATR (True Range then rolling mean)
high = df['high']
low = df['low']
close = df['close']
tr1 = high - low
tr2 = (high - close.shift(1)).abs()
tr3 = (low - close.shift(1)).abs()
df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df['atr'] = df['tr'].rolling(window=atr_len, min_periods=1).mean()

# -------------------------
# Convert to numpy arrays for zone detection and pivots
# -------------------------
dt_np = df['datetime'].to_numpy()
open_np = df['open'].to_numpy()
high_np = df['high'].to_numpy()
low_np = df['low'].to_numpy()
close_np = df['close'].to_numpy()
atr_np = df['atr'].to_numpy()

# -------------------------
# Zone detection using numpy arrays (robust indexing)
# -------------------------
zones = []  # list of dicts: start, end, top, bot, bull(bool), type
n = len(df)
for i in range(2, n):
    # ensure we have required previous indices
    if i-2 < 0 or i-1 < 0:
        continue
    # compute min_gap safely
    min_gap = float(atr_np[i]) * float(min_gap_frac) if not np.isnan(atr_np[i]) else 0.0

    # 3-candle FVG up (bull): low[i] > high[i-2] + min_gap
    if (not np.isnan(low_np[i])) and (not np.isnan(high_np[i-2])) and (low_np[i] > high_np[i-2] + min_gap):
        zones.append({
            'start': int(i-2),
            'end': int(i),
            'top': float(high_np[i-2]),
            'bot': float(low_np[i]),
            'bull': True,
            'type': 'FVG'
        })

    # 3-candle FVG down (bear): high[i] < low[i-2] - min_gap
    if (not np.isnan(high_np[i])) and (not np.isnan(low_np[i-2])) and (high_np[i] < low_np[i-2] - min_gap):
        zones.append({
            'start': int(i-2),
            'end': int(i),
            'top': float(high_np[i]),
            'bot': float(low_np[i-2]),
            'bull': False,
            'type': 'FVG'
        })

    # 3-candle OB displacement (bull)
    if (not np.isnan(close_np[i]) and not np.isnan(high_np[i-1]) and not np.isnan(open_np[i])
        and close_np[i] > high_np[i-1] and close_np[i] > open_np[i] and (not np.isnan(low_np[i-1]) and not np.isnan(low_np[i-2])) and low_np[i-1] < low_np[i-2]):
        zones.append({
            'start': int(i-1),
            'end': int(i),
            'top': float(high_np[i-1]),
            'bot': float(low_np[i-1]),
            'bull': True,
            'type': 'OB'
        })

    # 3-candle OB displacement (bear)
    if (not np.isnan(close_np[i]) and not np.isnan(low_np[i-1]) and not np.isnan(open_np[i])
        and close_np[i] < low_np[i-1] and close_np[i] < open_np[i] and (not np.isnan(high_np[i-1]) and not np.isnan(high_np[i-2])) and high_np[i-1] > high_np[i-2]):
        zones.append({
            'start': int(i-1),
            'end': int(i),
            'top': float(high_np[i-1]),
            'bot': float(low_np[i-1]),
            'bull': False,
            'type': 'OB'
        })

# limit zones drawn
if len(zones) > max_zones:
    zones = zones[-max_zones:]

# -------------------------
# Market structure (swings & BOS/CHoCH) using numpy pivot helpers
# -------------------------
ph_idxs = pivot_high_idxs_np(high_np, left=swing_left, right=swing_right)
pl_idxs = pivot_low_idxs_np(low_np, left=swing_left, right=swing_right)
last_hi_idx = ph_idxs[-1] if ph_idxs else None
last_lo_idx = pl_idxs[-1] if pl_idxs else None
last_hi_price = float(high_np[last_hi_idx]) if last_hi_idx is not None else None
last_lo_price = float(low_np[last_lo_idx]) if last_lo_idx is not None else None

# Detect breakout over last swing high / under last swing low
close_latest = float(close_np[-1])
bos = "No BOS"
if last_hi_idx is not None and (last_hi_price is not None) and close_latest > last_hi_price:
    bos = "BOS Up"
elif last_lo_idx is not None and (last_lo_price is not None) and close_latest < last_lo_price:
    bos = "BOS Down"

# -------------------------
# Signals (middle column)
# -------------------------
latest = df.iloc[-1]
smc_bull = latest['close'] > latest['ema_med']
ema_bull = latest['ema_short'] > latest['ema_med']
mom_bull = latest['rsi'] > 50
bull_signal = bool(smc_bull and ema_bull and mom_bull)

smc_bear = latest['close'] < latest['ema_med']
ema_bear = latest['ema_short'] < latest['ema_med']
mom_bear = latest['rsi'] < 45
bear_signal = bool(smc_bear and ema_bear and mom_bear)

if bull_signal and not bear_signal:
    rec_text = "GO LONG"
    rec_color = "#16a34a"  # green
elif bear_signal and not bull_signal:
    rec_text = "GO SHORT"
    rec_color = "#dc2626"  # red
else:
    rec_text = "HOLD"
    rec_color = "#f59e0b"  # yellow

# -------------------------
# Layout: three columns
# -------------------------
col_left, col_mid, col_right = st.columns([1,1.2,1])

with col_left:
    st.subheader("Numeric Context")
    st.write(f"**Latest Close:** {latest['close']:.6f}")
    st.write(f"**EMA {ema_short}:** {latest['ema_short']:.6f}")
    st.write(f"**EMA {ema_med}:** {latest['ema_med']:.6f}")
    st.write(f"**EMA {ema_long}:** {latest['ema_long']:.6f}")
    st.write(f"**RSI ({rsi_len}):** {latest['rsi']:.2f}")
    st.write(f"**ATR ({atr_len}):** {latest['atr']:.6f}")

with col_mid:
    st.subheader("Recommendation")
    # big colored recommendation box
    st.markdown(
        f"<div style='background:{rec_color};padding:22px;border-radius:8px;text-align:center;'>"
        f"<h1 style='color:white;margin:0;'>{rec_text}</h1></div>",
        unsafe_allow_html=True
    )
    st.write("**Signal logic:** close vs EMA_med, EMA short vs EMA med, RSI threshold")

with col_right:
    st.subheader("Market Structure")
    st.write(f"**Last Swing High idx:** {last_hi_idx if last_hi_idx is not None else 'N/A'}")
    st.write(f"**Last Swing High price:** {last_hi_price:.6f}" if last_hi_price is not None else "**Last Swing High price:** N/A")
    st.write(f"**Last Swing Low idx:** {last_lo_idx if last_lo_idx is not None else 'N/A'}")
    st.write(f"**Last Swing Low price:** {last_lo_price:.6f}" if last_lo_price is not None else "**Last Swing Low price:** N/A")
    st.write(f"**Breakout (BOS/CHoCH):** {bos}")

# -------------------------
# Matplotlib plot (consistent across timeframes)
# -------------------------
st.subheader("Price Chart (EMAs + Zones)")
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(df['datetime'], df['close'], color='black', linewidth=1, label='Close')
ax.plot(df['datetime'], df['ema_short'], color='gold', label=f'EMA{ema_short}')
ax.plot(df['datetime'], df['ema_med'], color='orange', label=f'EMA{ema_med}')
ax.plot(df['datetime'], df['ema_long'], color='purple', label=f'EMA{ema_long}')

# Draw zones as rectangles aligned to dates
for z in zones:
    start_idx = z['start']
    end_idx = z['end']
    # safety checks for indices
    if start_idx < 0 or end_idx >= n or start_idx >= n:
        continue
    x0 = df['datetime'].iat[start_idx]
    x1 = df['datetime'].iat[end_idx]
    # width as timedelta works with matplotlib datetime axis
    width = x1 - x0
    height = z['top'] - z['bot']
    color = (0.0, 0.6, 0.0, 0.18) if z['bull'] else (0.8, 0.0, 0.0, 0.18)
    rect = Rectangle((x0, z['bot']), width, height, color=color, linewidth=0)
    ax.add_patch(rect)
    # draw borders for clarity (use RGB portion)
    ax.plot([x0, x1], [z['top'], z['top']], color=color[:3], alpha=0.6, linewidth=1)
    ax.plot([x0, x1], [z['bot'], z['bot']], color=color[:3], alpha=0.6, linewidth=1)

ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend(loc='upper left')
fig.autofmt_xdate()
st.pyplot(fig)

# -------------------------
# Data tail and download
# -------------------------
st.subheader("Data (tail)")
st.dataframe(df.tail(10).reset_index(drop=True))

# CSV download
csv_buf = io.StringIO()
df.to_csv(csv_buf, index=False)
csv_bytes = csv_buf.getvalue().encode()
st.download_button("Download sliced CSV (2 years)", data=csv_bytes, file_name=f"{ticker}_2y_{timeframe}.csv", mime="text/csv")
