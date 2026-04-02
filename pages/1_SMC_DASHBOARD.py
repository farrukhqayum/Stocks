# streamlit_smc_yfinance.py
"""
Single-file Streamlit app:
- Downloads 2 years of OHLCV via yfinance (cached)
- UI: ticker, timeframe (1D/1W/1M), EMA/RSI/ATR params, swing detection, min_gap_frac
- Computes EMAs, RSI, ATR; detects 3-candle FVG and 3-candle OB (approx)
- Shows 3-column recommendation (Go Long / Hold / Go Short)
- Matplotlib price chart with EMAs and semi-transparent zone rectangles
- Shows market structure (last swing high/low and BOS/CHoCH)
- Provides CSV download of sliced data
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
import io

st.set_page_config(layout="wide", page_title="SMC Signals (yfinance)")

# -------------------------
# Utility functions
# -------------------------
@st.cache_data(ttl=300)
def download_yf(ticker: str, period_days: int, interval: str):
    # yfinance interval mapping already handled by caller
    period = f"{period_days}d"
    data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    data = data.reset_index().rename(columns={'Date': 'datetime'})
    if 'Datetime' in data.columns:
        data = data.rename(columns={'Datetime': 'datetime'})
    data = data[['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].rename(
        columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
    return data

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, length=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/length, adjust=False).mean()
    ma_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def pivot_high_idxs(high, left=20, right=5):
    idxs = []
    for i in range(left, len(high)-right):
        window = high[i-left:i+right+1]
        if high[i] == window.max():
            idxs.append(i)
    return idxs

def pivot_low_idxs(low, left=20, right=5):
    idxs = []
    for i in range(left, len(low)-right):
        window = low[i-left:i+right+1]
        if low[i] == window.min():
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
    # map to yfinance interval
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

# Ensure datetime is datetime type
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# Compute indicators
df['ema_short'] = ema(df['close'], ema_short)
df['ema_med']   = ema(df['close'], ema_med)
df['ema_long']  = ema(df['close'], ema_long)
df['rsi'] = rsi(df['close'], rsi_len)
# ATR simple (high-low rolling mean as approximation)
df['tr'] = np.maximum(df['high'] - df['low'], 
                      np.maximum((df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()))
df['atr'] = df['tr'].rolling(window=atr_len, min_periods=1).mean()

# -------------------------
# Zone detection (loop allowed)
# -------------------------
zones = []  # each zone: dict with start_idx, end_idx, top, bot, bull(bool), type 'FVG'/'OB'
for i in range(2, len(df)):
    min_gap = df['atr'].iat[i] * min_gap_frac
    # 3-candle FVG up (bull)
    if df['low'].iat[i] > df['high'].iat[i-2] + min_gap:
        zones.append({'start': i-2, 'end': i, 'top': df['high'].iat[i-2], 'bot': df['low'].iat[i], 'bull': True, 'type': 'FVG'})
    # 3-candle FVG down (bear)
    if df['high'].iat[i] < df['low'].iat[i-2] - min_gap:
        zones.append({'start': i-2, 'end': i, 'top': df['high'].iat[i], 'bot': df['low'].iat[i-2], 'bull': False, 'type': 'FVG'})
    # 3-candle OB displacement (bull)
    if (df['close'].iat[i] > df['high'].iat[i-1] and df['close'].iat[i] > df['open'].iat[i] and df['low'].iat[i-1] < df['low'].iat[i-2]):
        zones.append({'start': i-1, 'end': i, 'top': df['high'].iat[i-1], 'bot': df['low'].iat[i-1], 'bull': True, 'type': 'OB'})
    # 3-candle OB displacement (bear)
    if (df['close'].iat[i] < df['low'].iat[i-1] and df['close'].iat[i] < df['open'].iat[i] and df['high'].iat[i-1] > df['high'].iat[i-2]):
        zones.append({'start': i-1, 'end': i, 'top': df['high'].iat[i-1], 'bot': df['low'].iat[i-1], 'bull': False, 'type': 'OB'})

# Limit zones drawn
if len(zones) > max_zones:
    zones = zones[-max_zones:]

# -------------------------
# Market structure (swings & BOS/CHoCH)
# -------------------------
ph_idxs = pivot_high_idxs(df['high'].values, left=swing_left, right=swing_right)
pl_idxs = pivot_low_idxs(df['low'].values, left=swing_left, right=swing_right)
last_hi_idx = ph_idxs[-1] if ph_idxs else None
last_lo_idx = pl_idxs[-1] if pl_idxs else None
last_hi_price = df['high'].iat[last_hi_idx] if last_hi_idx is not None else None
last_lo_price = df['low'].iat[last_lo_idx] if last_lo_idx is not None else None

# Detect breakout over last swing high / under last swing low
close_latest = df['close'].iat[-1]
bos = None
if last_hi_idx is not None and close_latest > last_hi_price:
    bos = "BOS Up"
elif last_lo_idx is not None and close_latest < last_lo_price:
    bos = "BOS Down"
else:
    bos = "No BOS"

# -------------------------
# Signals (middle column)
# -------------------------
latest = df.iloc[-1]
smc_bull = latest['close'] > latest['ema_med']
ema_bull = latest['ema_short'] > latest['ema_med']
mom_bull = latest['rsi'] > 50
bull_signal = smc_bull and ema_bull and mom_bull

smc_bear = latest['close'] < latest['ema_med']
ema_bear = latest['ema_short'] < latest['ema_med']
mom_bear = latest['rsi'] < 45
bear_signal = smc_bear and ema_bear and mom_bear

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
    st.write(f"**Latest Close:** {latest['close']:.4f}")
    st.write(f"**EMA {ema_short}:** {latest['ema_short']:.4f}")
    st.write(f"**EMA {ema_med}:** {latest['ema_med']:.4f}")
    st.write(f"**EMA {ema_long}:** {latest['ema_long']:.4f}")
    st.write(f"**RSI ({rsi_len}):** {latest['rsi']:.2f}")
    st.write(f"**ATR ({atr_len}):** {latest['atr']:.4f}")

with col_mid:
    st.subheader("Recommendation")
    st.markdown(f"<div style='background:{rec_color};padding:20px;border-radius:8px;text-align:center;'>"
                f"<h1 style='color:white;margin:0;'>{rec_text}</h1></div>", unsafe_allow_html=True)
    st.write("**Signal logic:** close vs EMA_med, EMA short vs EMA med, RSI threshold")

with col_right:
    st.subheader("Market Structure")
    st.write(f"**Last Swing High idx:** {last_hi_idx if last_hi_idx is not None else 'N/A'}")
    st.write(f"**Last Swing High price:** {last_hi_price:.4f}" if last_hi_price is not None else "**Last Swing High price:** N/A")
    st.write(f"**Last Swing Low idx:** {last_lo_idx if last_lo_idx is not None else 'N/A'}")
    st.write(f"**Last Swing Low price:** {last_lo_price:.4f}" if last_lo_price is not None else "**Last Swing Low price:** N/A")
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
    x0 = df['datetime'].iat[z['start']]
    x1 = df['datetime'].iat[z['end']]
    width = (x1 - x0)
    height = z['top'] - z['bot']
    color = (0.0, 0.6, 0.0, 0.18) if z['bull'] else (0.8, 0.0, 0.0, 0.18)
    rect = Rectangle((x0, z['bot']), width, height, color=color, linewidth=0)
    ax.add_patch(rect)
    # optional border
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
