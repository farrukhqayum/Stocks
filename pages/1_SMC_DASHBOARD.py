# streamlit_smc_yfinance.py
"""
Streamlit SMC Dashboard (yfinance) - Fixed and hardened
- Downloads 2 years of OHLCV via yfinance (cached)
- UI: ticker, timeframe (1D/1W/1M), EMA params, swing params, min_gap_frac
- Plots candlesticks (matplotlib), LB curve, EMAs, active FVG/OB zones
- 4-column recommendation panel: Long/Hold Long | Exit Long | Short/Hold Short | Exit Short
- Y-axis placed on the right; markers for active zones and BOS
- Uses numpy arrays for robust indexing and avoids pandas .iat pitfalls
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
# Cached download helper (defensive)
# -------------------------
@st.cache_data(ttl=300)
def download_yf(ticker: str, period_days: int, interval: str):
    period = f"{period_days}d"

    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=True
        )
    except Exception as e:
        st.error(f"yfinance error: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        st.error("Yahoo returned no data. Try another timeframe or ticker.")
        return pd.DataFrame()

    # Flatten MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(c) for c in col]).strip() for col in df.columns.values]

    df = df.reset_index()

    # Normalize datetime column
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "datetime"})
    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "datetime"})

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    # Ensure numeric OHLC
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("datetime").reset_index(drop=True)

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
    st.header("Data & Parameters")
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
# LB curve (approximate Pine logic)
# -------------------------
n = len(df)
lb = np.full(n, np.nan, dtype=float)
if n > 0:
    lb[0] = df['close'].iat[0]
for i in range(1, n):
    # compute highest/lowest over previous lblen bars (mimic Pine highest/lowest with shift)
    left_idx = max(0, i - lblen)
    prev_window = df['close'].iloc[left_idx:i]
    prev_highest = prev_window.max() if not prev_window.empty else np.nan
    prev_lowest = prev_window.min() if not prev_window.empty else np.nan
    if not np.isnan(prev_highest) and df['close'].iat[i] > prev_highest:
        lb[i] = (df['high'].iat[i] + df['close'].iat[i]) / 2.0
    elif not np.isnan(prev_lowest) and df['close'].iat[i] < prev_lowest:
        lb[i] = (df['low'].iat[i] + df['close'].iat[i]) / 2.0
    else:
        lb[i] = lb[i-1]
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
# Zone detection (3-candle FVG & 3-candle OB) and active check
# -------------------------
zones = []
for i in range(2, n):
    if i-2 < 0: continue
    atr_val = atr_np[i] if not np.isnan(atr_np[i]) else 0.0
    min_gap = float(atr_val) * float(min_gap_frac)

    # FVG bull
    if (not np.isnan(low_np[i])) and (not np.isnan(high_np[i-2])) and (low_np[i] > high_np[i-2] + min_gap):
        zones.append({'start': i-2, 'end': i, 'top': float(high_np[i-2]), 'bot': float(low_np[i]), 'bull': True, 'type': 'FVG', 'created': i, 'mitigated': False})
    # FVG bear
    if (not np.isnan(high_np[i])) and (not np.isnan(low_np[i-2])) and (high_np[i] < low_np[i-2] - min_gap):
        zones.append({'start': i-2, 'end': i, 'top': float(high_np[i]), 'bot': float(low_np[i-2]), 'bull': False, 'type': 'FVG', 'created': i, 'mitigated': False})
    # OB bull
    cond_bull_ob = (
        not np.isnan(close_np[i]) and not np.isnan(high_np[i-1]) and not np.isnan(open_np[i])
        and close_np[i] > high_np[i-1] and close_np[i] > open_np[i]
        and not np.isnan(low_np[i-1]) and not np.isnan(low_np[i-2]) and low_np[i-1] < low_np[i-2]
    )
    if cond_bull_ob:
        zones.append({'start': i-1, 'end': i, 'top': float(high_np[i-1]), 'bot': float(low_np[i-1]), 'bull': True, 'type': 'OB', 'created': i, 'mitigated': False})
    # OB bear
    cond_bear_ob = (
        not np.isnan(close_np[i]) and not np.isnan(low_np[i-1]) and not np.isnan(open_np[i])
        and close_np[i] < low_np[i-1] and close_np[i] < open_np[i]
        and not np.isnan(high_np[i-1]) and not np.isnan(high_np[i-2]) and high_np[i-1] > high_np[i-2]
    )
    if cond_bear_ob:
        zones.append({'start': i-1, 'end': i, 'top': float(high_np[i-1]), 'bot': float(low_np[i-1]), 'bull': False, 'type': 'OB', 'created': i, 'mitigated': False})

# Mitigation check: if any close after creation closes inside zone -> mitigated
for z in zones:
    created_idx = z['created']
    z['mitigated'] = False
    for j in range(created_idx + 1, n):
        c = close_np[j]
        if np.isnan(c): continue
        if (c < z['top']) and (c > z['bot']):
            z['mitigated'] = True
            break
    z['active'] = not z['mitigated']

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
# Signals & Recommendations (4 columns)
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
# Use numeric x (date2num) for all drawing to avoid mixing datetime and numeric coords
# -------------------------
st.subheader(f"{ticker} Price Chart ({timeframe})")
fig, ax = plt.subplots(figsize=(14,6))

# Determine candle width (in date2num units)
if n > 1:
    # median spacing to be robust to irregular intervals
    diffs = np.diff(x_nums)
    candle_width = np.median(diffs) * 0.7
else:
    candle_width = 0.6

# Draw candles using numeric x
for i in range(n):
    o = open_np[i]; c = close_np[i]; h = high_np[i]; l = low_np[i]
    if np.isnan(o) or np.isnan(c) or np.isnan(h) or np.isnan(l):
        continue
    color = '#16a34a' if c >= o else '#dc2626'
    left = x_nums[i] - candle_width / 2.0
    height = abs(c - o)
    bottom = min(o, c)
    # body
    rect = Rectangle((left, bottom), candle_width, height if height > 0 else 0.000001, color=color, linewidth=0, zorder=2)
    ax.add_patch(rect)
    # wick (vertical line)
    ax.plot([x_nums[i], x_nums[i]], [l, h], color='black', linewidth=0.6, zorder=3)

# Plot LB curve and EMAs using numeric x
ax.plot(x_nums, df['lb_crv'].to_numpy(), color='#1e90ff', linewidth=1.5, label='LB Curve', zorder=4)
ax.plot(x_nums, df['ema_short'].to_numpy(), color='#f59e0b', linewidth=0.9, alpha=0.9, label=f'EMA{ema_short}', zorder=4)
ax.plot(x_nums, df['ema_med'].to_numpy(), color='#ef4444', linewidth=0.9, alpha=0.9, label=f'EMA{ema_med}', zorder=4)
ax.plot(x_nums, df['ema_long'].to_numpy(), color='#7c3aed', linewidth=0.9, alpha=0.9, label=f'EMA{ema_long}', zorder=4)

# Active zones shading and markers (use numeric coords)
for z in active_zones:
    s = z['start']; e = z['end']
    if s < 0 or e >= n: continue
    x0 = x_nums[s]; x1 = x_nums[e]
    width_num = x1 - x0 if x1 > x0 else (candle_width * (e - s + 1))
    height = z['top'] - z['bot']
    color = (0.0, 0.6, 0.0, 0.18) if z['bull'] else (0.8, 0.0, 0.0, 0.18)
    rect = Rectangle((x0, z['bot']), width_num, height if height != 0 else 0.0001, color=color, linewidth=0, zorder=1)
    ax.add_patch(rect)
    # marker center (numeric -> convert to datetime for scatter plotting with date formatter)
    mid_num = x0 + width_num / 2.0
    mid_dt = mdates.num2date(mid_num)
    if z['type'] == 'OB':
        if z['bull']:
            ax.scatter(mid_num, z['top'] + abs(height) * 0.02 if height != 0 else z['top'] + 0.001, marker='^', color='#065f46', s=80, zorder=6)
        else:
            ax.scatter(mid_num, z['bot'] - abs(height) * 0.02 if height != 0 else z['bot'] - 0.001, marker='v', color='#7f1d1d', s=80, zorder=6)
    else:  # FVG
        if z['bull']:
            ax.scatter(mid_num, z['bot'] - abs(height) * 0.02 if height != 0 else z['bot'] - 0.001, marker='D', color='#10b981', s=60, zorder=6)
        else:
            ax.scatter(mid_num, z['top'] + abs(height) * 0.02 if height != 0 else z['top'] + 0.001, marker='D', color='#ef4444', s=60, zorder=6)

# Plot BOS marker if present
if bos != "No BOS":
    if bos == "BOS Up" and last_hi_idx is not None:
        ax.scatter(x_nums[last_hi_idx], last_hi_price, marker='*', color='gold', s=140, zorder=7)
    if bos == "BOS Down" and last_lo_idx is not None:
        ax.scatter(x_nums[last_lo_idx], last_lo_price, marker='*', color='gold', s=140, zorder=7)

# Formatting: use date formatter on x-axis, y-axis on right
ax.xaxis_date()
ax.set_xlim([x_nums[0] - candle_width, x_nums[-1] + candle_width])
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.grid(alpha=0.12)
ax.set_title(f"{ticker} — Candles, LB Curve, Active FVG/OB (last {n} bars)")
ax.legend(loc='upper left')

# Convert x-axis ticks back to datetime for display
fig.autofmt_xdate()
st.pyplot(fig)
