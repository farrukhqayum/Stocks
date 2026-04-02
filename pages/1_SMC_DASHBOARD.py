# streamlit_smc_yfinance.py
"""
Streamlit single-file SMC dashboard (yfinance)
- Downloads 2 years of OHLCV via yfinance (cached)
- UI: ticker, timeframe (1D/1W/1M), EMA params, swing params, min_gap_frac
- Plots candlesticks + LB curve + EMAs (EMAs not shown as numeric)
- Detects active 3-candle FVG and 3-candle OB (zones that are not mitigated)
- Shows 4-column recommendation panel:
    Col1: Long / Hold Long
    Col2: Exit Long
    Col3: Short / Hold Short
    Col4: Exit Short
- Y axis placed on the right
- Signals (active zones & pattern signals) plotted as markers above/below candles
- Robust numpy/pandas handling and defensive checks
Run: streamlit run streamlit_smc_yfinance.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
import io
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="SMC Dashboard (yfinance)")

# -------------------------
# Cached download helper
# -------------------------
@st.cache_data(ttl=300)
def download_yf(ticker: str, period_days: int, interval: str) -> pd.DataFrame:
    period = f"{period_days}d"
    raw = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    # flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(map(str, c)).strip() for c in df.columns.values]
    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    if 'date' in df.columns and 'datetime' not in df.columns:
        df = df.rename(columns={'date': 'datetime'})
    if 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(df.index)
    # ensure required columns exist
    for c in ['open','high','low','close','volume']:
        if c not in df.columns:
            df[c] = np.nan
    # coerce numeric
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime']).reset_index(drop=True)
    df = df.sort_values('datetime').reset_index(drop=True)
    return df

# -------------------------
# Small helpers
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
    left = max(1, int(left)); right = max(1, int(right))
    for i in range(left, n - right):
        window = high_np[i-left:i+right+1]
        if np.isnan(window).any(): continue
        if high_np[i] == window.max(): idxs.append(i)
    return idxs

def pivot_low_idxs_np(low_np: np.ndarray, left: int = 20, right: int = 5):
    idxs = []
    n = len(low_np)
    left = max(1, int(left)); right = max(1, int(right))
    for i in range(left, n - right):
        window = low_np[i-left:i+right+1]
        if np.isnan(window).any(): continue
        if low_np[i] == window.min(): idxs.append(i)
    return idxs

# -------------------------
# UI controls
# -------------------------
st.title("SMC Dashboard (yfinance)")

with st.sidebar:
    st.header("Data & Params")
    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
    timeframe = st.selectbox("Timeframe", options=["1D","1W","1M"], index=0)
    interval_map = {"1D":"1d","1W":"1wk","1M":"1mo"}
    yf_interval = interval_map[timeframe]
    period_days = 365 * 2

    st.markdown("**SMC / Indicator params**")
    ema_short = st.number_input("EMA Short", value=20, min_value=1)
    ema_med   = st.number_input("EMA Medium", value=50, min_value=1)
    ema_long  = st.number_input("EMA Long", value=200, min_value=1)
    lblen     = st.number_input("LB Length (line break)", value=10, min_value=2)
    rsi_len   = st.number_input("RSI Length", value=14, min_value=1)
    atr_len   = st.number_input("ATR Length", value=14, min_value=1)

    st.markdown("**Zones & Swings**")
    swing_left = st.number_input("Swing Left", value=20, min_value=1)
    swing_right = st.number_input("Swing Right", value=5, min_value=1)
    min_gap_frac = st.number_input("Min gap (ATR fraction)", value=0.1, step=0.05, min_value=0.0)
    max_zones = st.number_input("Max Zones to draw", value=200, min_value=1)

# -------------------------
# Download data (yfinance)
# -------------------------
st.info(f"Downloading 2 years of {ticker} @ {timeframe} ...")
df = download_yf(ticker, period_days, yf_interval)
if df.empty:
    st.error("No data returned. Try another ticker/timeframe.")
    st.stop()

# -------------------------
# Compute indicators (vectorized)
# -------------------------
df['ema_short'] = ema(df['close'], ema_short)
df['ema_med']   = ema(df['close'], ema_med)
df['ema_long']  = ema(df['close'], ema_long)
df['rsi'] = rsi(df['close'], rsi_len)

# ATR (True Range)
tr1 = df['high'] - df['low']
tr2 = (df['high'] - df['close'].shift(1)).abs()
tr3 = (df['low'] - df['close'].shift(1)).abs()
df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df['atr'] = df['tr'].rolling(window=atr_len, min_periods=1).mean()

# -------------------------
# LB curve (line break curve approximation from Pine logic)
# Implemented with a small loop to replicate Pine's conditional update
# -------------------------
n = len(df)
lb = np.full(n, np.nan, dtype=float)
# initialize lb as close[0]
if n > 0:
    lb[0] = df['close'].iat[0]
for i in range(1, n):
    prev_highest = np.nanmax(df['close'].iloc[max(0, i - lblen):i]) if i - lblen >= 0 else np.nan
    prev_lowest = np.nanmin(df['close'].iloc[max(0, i - lblen):i]) if i - lblen >= 0 else np.nan
    # follow Pine logic: if close > highest[1] then lb := (high + close)/2 ; if close < lowest[1] then lb := (low + close)/2 else keep previous
    if not np.isnan(prev_highest) and df['close'].iat[i] > prev_highest:
        lb[i] = (df['high'].iat[i] + df['close'].iat[i]) / 2.0
    elif not np.isnan(prev_lowest) and df['close'].iat[i] < prev_lowest:
        lb[i] = (df['low'].iat[i] + df['close'].iat[i]) / 2.0
    else:
        lb[i] = lb[i-1]
# smooth LB with EMA
df['lb_crv'] = pd.Series(lb).ewm(span=lblen, adjust=False).mean()

# -------------------------
# Convert to numpy arrays for robust indexing
# -------------------------
open_np = pd.to_numeric(df['open'], errors='coerce').to_numpy(dtype=float)
high_np = pd.to_numeric(df['high'], errors='coerce').to_numpy(dtype=float)
low_np = pd.to_numeric(df['low'], errors='coerce').to_numpy(dtype=float)
close_np = pd.to_numeric(df['close'], errors='coerce').to_numpy(dtype=float)
atr_np = pd.to_numeric(df['atr'], errors='coerce').to_numpy(dtype=float)
dt_py = pd.to_datetime(df['datetime']).dt.to_pydatetime()
x_nums = mdates.date2num(dt_py)

# -------------------------
# Zone detection (3-candle FVG & 3-candle OB) and active check (not mitigated)
# Active = zone exists and price has not closed inside zone since creation
# We'll store zone creation index and check mitigation by any close inside zone after creation
# -------------------------
zones = []  # dict: start,end,top,bot,bull,type,created_idx,mitigated(bool),active(bool)
for i in range(2, n):
    if i-2 < 0: continue
    min_gap = (atr_np[i] if not np.isnan(atr_np[i]) else 0.0) * float(min_gap_frac)
    # FVG bull
    if (not np.isnan(low_np[i]) and not np.isnan(high_np[i-2]) and (low_np[i] > high_np[i-2] + min_gap)):
        start = i-2; end = i
        top = float(high_np[i-2]); bot = float(low_np[i])
        zones.append({'start':start,'end':end,'top':top,'bot':bot,'bull':True,'type':'FVG','created':i,'mitigated':False})
    # FVG bear
    if (not np.isnan(high_np[i]) and not np.isnan(low_np[i-2]) and (high_np[i] < low_np[i-2] - min_gap)):
        start = i-2; end = i
        top = float(high_np[i]); bot = float(low_np[i-2])
        zones.append({'start':start,'end':end,'top':top,'bot':bot,'bull':False,'type':'FVG','created':i,'mitigated':False})
    # OB bull (displacement)
    cond_bull_ob = (
        not np.isnan(close_np[i]) and not np.isnan(high_np[i-1]) and not np.isnan(open_np[i])
        and close_np[i] > high_np[i-1] and close_np[i] > open_np[i] and not np.isnan(low_np[i-1]) and not np.isnan(low_np[i-2]) and low_np[i-1] < low_np[i-2]
    )
    if cond_bull_ob:
        start = i-1; end = i
        top = float(high_np[i-1]); bot = float(low_np[i-1])
        zones.append({'start':start,'end':end,'top':top,'bot':bot,'bull':True,'type':'OB','created':i,'mitigated':False})
    # OB bear
    cond_bear_ob = (
        not np.isnan(close_np[i]) and not np.isnan(low_np[i-1]) and not np.isnan(open_np[i])
        and close_np[i] < low_np[i-1] and close_np[i] < open_np[i] and not np.isnan(high_np[i-1]) and not np.isnan(high_np[i-2]) and high_np[i-1] > high_np[i-2]
    )
    if cond_bear_ob:
        start = i-1; end = i
        top = float(high_np[i-1]); bot = float(low_np[i-1])
        zones.append({'start':start,'end':end,'top':top,'bot':bot,'bull':False,'type':'OB','created':i,'mitigated':False})

# Determine mitigation/active status: if any close after creation closes inside zone -> mitigated
for z in zones:
    created_idx = z['created']
    # check closes from created_idx+1 to end
    for j in range(created_idx+1, n):
        c = close_np[j]
        if np.isnan(c): continue
        if (c < z['top'] and c > z['bot']):  # closed inside zone
            z['mitigated'] = True
            break
    z['active'] = not z['mitigated']

# Keep only active zones for plotting (user asked active FVG and active OB)
active_zones = [z for z in zones if z.get('active', False)]
if len(active_zones) > max_zones:
    active_zones = active_zones[-max_zones:]

# -------------------------
# Market structure (swings & BOS/CHoCH)
# -------------------------
ph_idxs = pivot_high_idxs_np(high_np, left=swing_left, right=swing_right)
pl_idxs = pivot_low_idxs_np(low_np, left=swing_left, right=swing_right)
last_hi_idx = ph_idxs[-1] if ph_idxs else None
last_lo_idx = pl_idxs[-1] if pl_idxs else None
last_hi_price = float(high_np[last_hi_idx]) if last_hi_idx is not None and not np.isnan(high_np[last_hi_idx]) else None
last_lo_price = float(low_np[last_lo_idx]) if last_lo_idx is not None and not np.isnan(low_np[last_lo_idx]) else None
close_latest = float(close_np[-1]) if not np.isnan(close_np[-1]) else None
bos = "No BOS"
if close_latest is not None:
    if last_hi_price is not None and close_latest > last_hi_price:
        bos = "BOS Up"
    elif last_lo_price is not None and close_latest < last_lo_price:
        bos = "BOS Down"

# -------------------------
# Signals & Recommendations (map to 4 columns)
# - Long/Hold Long: when bull_signal true
# - Exit Long: when price crosses below ema_med or bear_signal true
# - Short/Hold Short: when bear_signal true
# - Exit Short: when price crosses above ema_med or bull_signal true
# -------------------------
latest = df.iloc[-1]
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

# Determine column states
col1_state = "HOLD LONG" if bull_signal else "LONG (No)"
col1_color = "#16a34a" if bull_signal else "#94a3b8"
col2_state = "EXIT LONG" if (not bull_signal and smc_bear) or bear_signal else "NO EXIT"
col2_color = "#dc2626" if col2_state == "EXIT LONG" else "#94a3b8"
col3_state = "HOLD SHORT" if bear_signal else "SHORT (No)"
col3_color = "#dc2626" if bear_signal else "#94a3b8"
col4_state = "EXIT SHORT" if (not bear_signal and smc_bull) or bull_signal else "NO EXIT"
col4_color = "#16a34a" if col4_state == "EXIT SHORT" else "#94a3b8"

# -------------------------
# Layout: 4 columns for recommendations
# -------------------------
c1, c2, c3, c4 = st.columns([1,1,1,1])
with c1:
    st.markdown(f"<div style='background:{col1_color};padding:18px;border-radius:8px;text-align:center;'>"
                f"<h3 style='color:white;margin:0;'>{col1_state}</h3></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div style='background:{col2_color};padding:18px;border-radius:8px;text-align:center;'>"
                f"<h3 style='color:white;margin:0;'>{col2_state}</h3></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div style='background:{col3_color};padding:18px;border-radius:8px;text-align:center;'>"
                f"<h3 style='color:white;margin:0;'>{col3_state}</h3></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div style='background:{col4_color};padding:18px;border-radius:8px;text-align:center;'>"
                f"<h3 style='color:white;margin:0;'>{col4_state}</h3></div>", unsafe_allow_html=True)

# -------------------------
# Plot: candlesticks + LB curve + active zones + markers
# - Y-axis on right
# - Candles drawn with rectangles and wicks
# - Active zones shaded (green bull, red bear)
# - Markers: active OB as triangles; active FVG as diamonds; BOS as star
# -------------------------
st.subheader(f"{ticker} Price Chart ({timeframe})")
fig, ax = plt.subplots(figsize=(14,6))

# candlesticks
width = 0.6 * (x_nums[1] - x_nums[0]) if n > 1 else 0.6
for i in range(n):
    if np.isnan(open_np[i]) or np.isnan(close_np[i]) or np.isnan(high_np[i]) or np.isnan(low_np[i]):
        continue
    color = '#16a34a' if close_np[i] >= open_np[i] else '#dc2626'
    # body
    left = x_nums[i] - width/2
    rect = Rectangle((left, min(open_np[i], close_np[i])), width, abs(close_np[i]-open_np[i]), color=color, linewidth=0)
    ax.add_patch(rect)
    # wick
    ax.plot([x_nums[i], x_nums[i]], [low_np[i], high_np[i]], color='black', linewidth=0.6, zorder=2)

# LB curve
ax.plot(df['datetime'], df['lb_crv'], color='#1e90ff', linewidth=1.5, label='LB Curve')

# EMAs (plotted but not shown as numeric panels)
ax.plot(df['datetime'], df['ema_short'], color='#f59e0b', linewidth=0.9, alpha=0.9, label=f'EMA{ema_short}')
ax.plot(df['datetime'], df['ema_med'], color='#ef4444', linewidth=0.9, alpha=0.9, label=f'EMA{ema_med}')
ax.plot(df['datetime'], df['ema_long'], color='#7c3aed', linewidth=0.9, alpha=0.9, label=f'EMA{ema_long}')

# Active zones shading and markers
for z in active_zones:
    s = z['start']; e = z['end']
    if s < 0 or e >= n: continue
    x0 = x_nums[s]; x1 = x_nums[e]
    width_num = x1 - x0 if x1 > x0 else (x_nums[s] - x_nums[max(0, s-1)])*1.1
    height = z['top'] - z['bot']
    color = (0.0, 0.6, 0.0, 0.18) if z['bull'] else (0.8, 0.0, 0.0, 0.18)
    rect = Rectangle((x0, z['bot']), width_num, height, color=color, linewidth=0, transform=ax.transData)
    ax.add_patch(rect)
    # marker: OB -> triangle, FVG -> diamond
    mid_x = (x0 + x0 + width_num) / 2.0
    # convert mid_x back to datetime for scatter plotting
    mid_dt = mdates.num2date(mid_x)
    if z['type'] == 'OB':
        # triangle above (bull) or below (bear)
        if z['bull']:
            ax.scatter(mid_dt, z['top'] + (height * 0.05 if height>0 else 0.001), marker='^', color='#065f46', s=80, zorder=5)
        else:
            ax.scatter(mid_dt, z['bot'] - (abs(height) * 0.05 if height!=0 else 0.001), marker='v', color='#7f1d1d', s=80, zorder=5)
    else:  # FVG
        if z['bull']:
            ax.scatter(mid_dt, z['bot'] - (abs(height) * 0.02 if height!=0 else 0.001), marker='D', color='#10b981', s=60, zorder=5)
        else:
            ax.scatter(mid_dt, z['top'] + (abs(height) * 0.02 if height!=0 else 0.001), marker='D', color='#ef4444', s=60, zorder=5)

# Plot BOS marker if present
if bos != "No BOS":
    if bos == "BOS Up" and last_hi_idx is not None:
        ax.scatter(df['datetime'].iat[last_hi_idx], last_hi_price, marker='*', color='gold', s=140, zorder=6)
    if bos == "BOS Down" and last_lo_idx is not None:
        ax.scatter(df['datetime'].iat[last_lo_idx], last_lo_price, marker='*', color='gold', s=140, zorder=6)

# Formatting: x-axis datetime, y-axis on right
ax.xaxis_date()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.grid(alpha=0.15)
ax.set_title(f"{ticker} — Candles, LB Curve, Active FVG/OB (last {n} bars)")
ax.legend(loc='upper left')

# tighten x-limits
ax.set_xlim([df['datetime'].iat[0], df['datetime'].iat[-1]])
fig.autofmt_xdate()
st.pyplot(fig)
