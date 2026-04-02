import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# =========================================================
# BLOCK 1 — DATA & INDICATORS
# =========================================================

@st.cache_data(show_spinner=False)
def load_data(ticker, start_date, interval):
    try:
        df = yf.download(ticker, start=start_date, interval=interval)
        if df is None or df.empty: return None
        # Handle potential MultiIndex from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)
        df.dropna(inplace=True)
        return df
    except Exception:
        return None

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def atr(df, length=14):
    high, low, close_prev = df["high"], df["low"], df["close"].shift()
    tr = pd.concat([(high - low), (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()

class Zone:
    def __init__(self, top, bottom, start_idx, is_bull, is_ob, color):
        self.top, self.bottom = top, bottom
        self.start_idx = start_idx # This is the index label (Date)
        self.is_bull, self.is_ob, self.color = is_bull, is_ob, color
        self.is_mitigated = False

# =========================================================
# BLOCK 2 — ENGINE (BOS, PATTERNS, ZONES)
# =========================================================

def compute_structure(df, sw_len=20):
    highs, lows, closes = df["high"].values, df["low"].values, df["close"].values
    n = len(df)
    bos_up, bos_dn = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    last_hi, last_lo = np.nan, np.nan

    for i in range(sw_len, n - 5):
        if highs[i] == highs[i-sw_len : i+6].max(): last_hi = highs[i]
        if lows[i] == lows[i-sw_len : i+6].min(): last_lo = lows[i]
        
        if not np.isnan(last_hi) and closes[i] > last_hi:
            bos_up[i], last_hi = True, np.nan
        if not np.isnan(last_lo) and closes[i] < last_lo:
            bos_dn[i], last_lo = True, np.nan
            
    return pd.Series(bos_up, index=df.index), pd.Series(bos_dn, index=df.index)

def apply_pinescript_logic(df_raw):
    df = df_raw.copy()
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].to_numpy().flatten()

    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["atr"] = atr(df, 14)
    
    bos_up, bos_dn = compute_structure(df)
    
    # Zones Detection
    zones = []
    for i in range(2, len(df)):
        # FVG Detection
        if df["low"].iloc[i] > df["high"].iloc[i-2] + (df["atr"].iloc[i]*0.1):
            zones.append(Zone(df["low"].iloc[i], df["high"].iloc[i-2], df.index[i], True, False, (0.14, 0.44, 0.09, 0.3)))
        if df["high"].iloc[i] < df["low"].iloc[i-2] - (df["atr"].iloc[i]*0.1):
            zones.append(Zone(df["low"].iloc[i-2], df["high"].iloc[i], df.index[i], False, False, (0.55, 0.05, 0.05, 0.3)))

    # Synchronized Info DataFrame for slicing
    info_df = pd.DataFrame({"bos_up": bos_up, "bos_dn": bos_dn}, index=df.index)
    return df, zones, info_df

# =========================================================
# BLOCK 3 — PLOTTING (FIXED COORDINATES)
# =========================================================

def plotchart(df_slice, zones, info_slice, full_df_index):
    fig, ax = plt.subplots(figsize=(14, 7), facecolor='#131722')
    ax.set_facecolor('#131722')
    
    dates = df_slice.index
    for i in range(len(df_slice)):
        o, c, h, l = df_slice["open"].iloc[i], df_slice["close"].iloc[i], df_slice["high"].iloc[i], df_slice["low"].iloc[i]
        color = "#26a69a" if c >= o else "#ef5350"
        ax.plot([i, i], [l, h], color=color, linewidth=1)
        ax.add_patch(plt.Rectangle((i-0.3, min(o, c)), 0.6, abs(c-o), color=color, alpha=0.9))
        
        # Fixed Signal Lookup
        curr_date = dates[i]
        if info_slice.loc[curr_date, "bos_up"]:
            ax.text(i, h, "BOS↑", color="#00ff00", fontsize=9, ha="center", va="bottom", fontweight='bold')
        if info_slice.loc[curr_date, "bos_dn"]:
            ax.text(i, l, "BOS↓", color="#ff0000", fontsize=9, ha="center", va="top", fontweight='bold')

    # Fixed Zone drawing to prevent KeyError
    for z in zones:
        if z.start_idx in dates:
            start_x = dates.get_loc(z.start_idx)
            width = len(df_slice) - start_x
            ax.add_patch(plt.Rectangle((start_x, z.bottom), width, z.top - z.bottom, color=z.color, linewidth=0))

    ax.set_title("SMC Backtest Dashboard", color="white", fontsize=12)
    ax.grid(True, color="#2a2e39", alpha=0.5)
    ax.tick_params(colors='white')
    return fig

# =========================================================
# BLOCK 4 — UI ORCHESTRATION
# =========================================================

st.set_page_config(page_title="SMC Backtester", layout="wide")
ticker = st.sidebar.text_input("Ticker", "NVDA")
df_raw = load_data(ticker, datetime.now()-timedelta(days=365), "1d")

if df_raw is not None:
    df, zones, info_df = apply_pinescript_logic(df_raw)
    
    if "end_idx" not in st.session_state: st.session_state.end_idx = 100
    
    c1, c2, c3 = st.columns([1,1,4])
    if c1.button("⬅️ Prev"): st.session_state.end_idx = max(50, st.session_state.end_idx - 5)
    if c2.button("Next ➡️"): st.session_state.end_idx = min(len(df), st.session_state.end_idx + 5)
    
    # Critical: Slice both Data and Signal Info identically
    start = max(0, st.session_state.end_idx - 100)
    df_slice = df.iloc[start : st.session_state.end_idx]
    info_slice = info_df.iloc[start : st.session_state.end_idx]
    
    st.pyplot(plotchart(df_slice, zones, info_slice, df.index))
else:
    st.error("Select a valid Ticker.")
