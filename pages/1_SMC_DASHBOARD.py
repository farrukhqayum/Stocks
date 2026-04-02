import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
import io
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="SMC Signals (yfinance)")

# -------------------------
# Cached download helper (robust)
# -------------------------
@st.cache_data(ttl=300)
def download_yf(ticker: str, period_days: int, interval: str) -> pd.DataFrame:
    """
    Download OHLCV from yfinance and normalize column names.
    This function is defensive: handles MultiIndex columns, missing columns,
    and coerces numeric types safely.
    """
    period = f"{period_days}d"
    raw = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        return pd.DataFrame()

    # If yfinance returns a DataFrame with MultiIndex columns (rare), flatten them
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(c) for c in col]).strip() for col in df.columns.values]

    df = df.reset_index()

    # Normalize column names to lowercase strings
    df.columns = [str(c).lower() for c in df.columns]

    # Ensure we have a datetime column
    if 'date' in df.columns and 'datetime' not in df.columns:
        df = df.rename(columns={'date': 'datetime'})
    if 'datetime' not in df.columns:
        # try common alternatives
        if 'index' in df.columns:
            df = df.rename(columns={'index': 'datetime'})
        else:
            # fallback: create datetime from index
            df['datetime'] = pd.to_datetime(df.index)

    # Keep only required columns, create missing ones as NaN
    required = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    for c in required:
        if c not in df.columns:
            df[c] = np.nan

    # Coerce numeric columns safely (operate on Series only)
    for c in ['open', 'high', 'low', 'close', 'volume']:
        # ensure we have a Series object
        series = df[c] if c in df.columns else pd.Series([np.nan]*len(df))
        # convert to numeric, coerce errors to NaN
        df[c] = pd.to_numeric(series, errors='coerce').astype(float)

    # Ensure datetime column is proper dtype
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    # Drop rows with invalid datetimes
    df = df.dropna(subset=['datetime']).reset_index(drop=True)

    # Sort and reset index
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
    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
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

# ATR (True Range then rolling mean) - robust numeric handling
high = df['high']
low = df['low']
close = df['close']
tr1 = high - low
tr2 = (high - close.shift(1)).abs()
tr3 = (low - close.shift(1)).abs()
df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df['atr'] = df['tr'].rolling(window=atr_len, min_periods=1).mean()

# -------------------------
# Convert to numpy arrays for zone detection and pivots (coerced to float)
# -------------------------
# Ensure numeric arrays (float) and datetime array
dt_np = pd.to_datetime(df['datetime']).to_numpy()
open_np = pd.to_numeric(df['open'], errors='coerce').to_numpy(dtype=float)
high_np = pd.to_numeric(df['high'], errors='coerce').to_numpy(dtype=float)
low_np = pd.to_numeric(df['low'], errors='coerce').to_numpy(dtype=float)
close_np = pd.to_numeric(df['close'], errors='coerce').to_numpy(dtype=float)
atr_np = pd.to_numeric(df['atr'], errors='coerce').to_numpy(dtype=float)

# Precompute matplotlib numeric x values for rectangles (date2num)
# Convert to python datetime objects first to avoid numpy datetime issues
py_dates = pd.to_datetime(df['datetime']).dt.to_pydatetime()
x_nums = mdates.date2num(py_dates)

# -------------------------
# Zone detection using numpy arrays (robust indexing & type-safe)
# -------------------------
zones = []  # list of dicts: start, end, top, bot, bull(bool), type
n = len(df)
for i in range(2, n):
    # ensure we have required previous indices
    if i-2 < 0 or i-1 < 0:
        continue
    # compute min_gap safely (handle nan)
    atr_val = atr_np[i] if not np.isnan(atr_np[i]) else 0.0
    min_gap = float(atr_val) * float(min_gap_frac)

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
    cond_bull_ob = (
        (not np.isnan(close_np[i])) and
        (not np.isnan(high_np[i-1])) and
        (not np.isnan(open_np[i])) and
        (not np.isnan(low_np[i-1])) and
        (not np.isnan(low_np[i-2]))
        and (close_np[i] > high_np[i-1]) and (close_np[i] > open_np[i]) and (low_np[i-1] < low_np[i-2])
    )
    if cond_bull_ob:
        zones.append({
            'start': int(i-1),
            'end': int(i),
            'top': float(high_np[i-1]),
            'bot': float(low_np[i-1]),
            'bull': True,
            'type': 'OB'
        })

    # 3-candle OB displacement (bear)
    cond_bear_ob = (
        (not np.isnan(close_np[i])) and
        (not np.isnan(low_np[i-1])) and
        (not np.isnan(open_np[i])) and
        (not np.isnan(high_np[i-1])) and
        (not np.isnan(high_np[i-2]))
        and (close_np[i] < low_np[i-1]) and (close_np[i] < open_np[i]) and (high_np[i-1] > high_np[i-2])
    )
    if cond_bear_ob:
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
last_hi_price = float(high_np[last_hi_idx]) if last_hi_idx is not None and not np.isnan(high_np[last_hi_idx]) else None
last_lo_price = float(low_np[last_lo_idx]) if last_lo_idx is not None and not np.isnan(low_np[last_lo_idx]) else None

# Detect breakout over last swing high / under last swing low
close_latest = float(close_np[-1]) if not np.isnan(close_np[-1]) else None
bos = "No BOS"
if close_latest is not None:
    if last_hi_price is not None and close_latest > last_hi_price:
        bos = "BOS Up"
    elif last_lo_price is not None and close_latest < last_lo_price:
        bos = "BOS Down"

# -------------------------
# Signals (middle column)
# -------------------------
latest = df.iloc[-1]
# guard against NaNs in latest values
latest_close = float(latest['close']) if not pd.isna(latest['close']) else np.nan
latest_ema_short = float(latest['ema_short']) if not pd.isna(latest['ema_short']) else np.nan
latest_ema_med = float(latest['ema_med']) if not pd.isna(latest['ema_med']) else np.nan
latest_rsi = float(latest['rsi']) if not pd.isna(latest['rsi']) else np.nan

smc_bull = (latest_close > latest_ema_med) if (not np.isnan(latest_close) and not np.isnan(latest_ema_med)) else False
ema_bull = (latest_ema_short > latest_ema_med) if (not np.isnan(latest_ema_short) and not np.isnan(latest_ema_med)) else False
mom_bull = (latest_rsi > 50) if not np.isnan(latest_rsi) else False
bull_signal = bool(smc_bull and ema_bull and mom_bull)

smc_bear = (latest_close < latest_ema_med) if (not np.isnan(latest_close) and not np.isnan(latest_ema_med)) else False
ema_bear = (latest_ema_short < latest_ema_med) if (not np.isnan(latest_ema_short) and not np.isnan(latest_ema_med)) else False
mom_bear = (latest_rsi < 45) if not np.isnan(latest_rsi) else False
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
    st.write(f"**Latest Close:** {latest_close:.6f}" if not np.isnan(latest_close) else "**Latest Close:** N/A")
    st.write(f"**EMA {ema_short}:** {latest_ema_short:.6f}" if not np.isnan(latest_ema_short) else f"**EMA {ema_short}:** N/A")
    st.write(f"**EMA {ema_med}:** {latest_ema_med:.6f}" if not np.isnan(latest_ema_med) else f"**EMA {ema_med}:** N/A")
    st.write(f"**EMA {ema_long}:** {float(latest['ema_long']):.6f}" if not pd.isna(latest['ema_long']) else f"**EMA {ema_long}:** N/A")
    st.write(f"**RSI ({rsi_len}):** {latest_rsi:.2f}" if not np.isnan(latest_rsi) else f"**RSI ({rsi_len}):** N/A")
    st.write(f"**ATR ({atr_len}):** {float(latest['atr']):.6f}" if not pd.isna(latest['atr']) else f"**ATR ({atr_len}):** N/A")

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

# plot using datetime axis
ax.plot(df['datetime'], df['close'], color='black', linewidth=1, label='Close')
ax.plot(df['datetime'], df['ema_short'], color='gold', label=f'EMA{ema_short}')
ax.plot(df['datetime'], df['ema_med'], color='orange', label=f'EMA{ema_med}')
ax.plot(df['datetime'], df['ema_long'], color='purple', label=f'EMA{ema_long}')

# Draw zones as rectangles aligned to date2num coordinates (use ax.transData)
for z in zones:
    start_idx = z['start']
    end_idx = z['end']
    # safety checks for indices
    if start_idx < 0 or end_idx >= n or start_idx >= n:
        continue
    x0_num = x_nums[start_idx]
    x1_num = x_nums[end_idx]
    width_num = x1_num - x0_num
    height = z['top'] - z['bot']
    color = (0.0, 0.6, 0.0, 0.18) if z['bull'] else (0.8, 0.0, 0.0, 0.18)
    # Rectangle in date2num coordinates; set transform to data coordinates
    rect = Rectangle((x0_num, z['bot']), width_num, height, color=color, linewidth=0, transform=ax.transData)
    ax.add_patch(rect)
    # draw borders for clarity (use datetime values)
    ax.plot([df['datetime'].iat[start_idx], df['datetime'].iat[end_idx]], [z['top'], z['top']], color=color[:3], alpha=0.6, linewidth=1)
    ax.plot([df['datetime'].iat[start_idx], df['datetime'].iat[end_idx]], [z['bot'], z['bot']], color=color[:3], alpha=0.6, linewidth=1)

# Improve x-axis formatting
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
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
