import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ---------------------------------------------------------
# DATA LOADER
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(ticker, start_date, interval):
    try:
        df = yf.download(ticker, start=start_date, interval=interval)
        if df is None or df.empty:
            return None
        # Handle MultiIndex if yfinance returns it
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)
        df.dropna(inplace=True)
        return df
    except Exception:
        return None

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def atr(df, length=14):
    high, low, close_prev = df["high"], df["low"], df["close"].shift()
    tr = pd.concat([(high - low), (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()

class Zone:
    def __init__(self, top, bottom, start_idx, is_bull, is_ob, color):
        self.top, self.bottom = top, bottom
        self.start_idx = start_idx
        self.is_bull, self.is_ob, self.color = is_bull, is_ob, color
        self.is_mitigated, self.taps = False, 0

def rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    return (int(hex_color[0:2], 16)/255, int(hex_color[2:4], 16)/255, int(hex_color[4:6], 16)/255, alpha)

# ---------------------------------------------------------
# ENGINES
# ---------------------------------------------------------
def compute_structure(df, swing_left=20, swing_right=5):
    highs, lows, closes = df["high"].values, df["low"].values, df["close"].values
    n = len(df)
    bos_up, bos_dn, trend = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    last_hi, last_lo, is_uptrend = np.nan, np.nan, False

    for i in range(n):
        if i >= swing_left and i < n - swing_right:
            if highs[i] == highs[i-swing_left : i+swing_right+1].max(): last_hi = highs[i]
            if lows[i] == lows[i-swing_left : i+swing_right+1].min(): last_lo = lows[i]
        
        if not np.isnan(last_hi) and closes[i] > last_hi:
            bos_up[i], is_uptrend, last_hi = True, True, np.nan
        if not np.isnan(last_lo) and closes[i] < last_lo:
            bos_dn[i], is_uptrend, last_lo = True, False, np.nan
        trend[i] = is_uptrend
    return pd.Series(bos_up, index=df.index), pd.Series(bos_dn, index=df.index), pd.Series(trend, index=df.index)

def detect_patterns(df, lb_crv):
    o, c, h, l = df["open"], df["close"], df["high"], df["low"]
    pattern_name = pd.Series("None", index=df.index)
    pattern_bull = pd.Series(False, index=df.index)
    
    # Simple Engulfing Logic
    bull_eng = (c > o) & (c.shift(1) < o.shift(1)) & (c >= h.shift(1))
    bear_eng = (c < o) & (c.shift(1) > o.shift(1)) & (c <= l.shift(1))
    
    pattern_name[bull_eng] = "Bull Engulfing"
    pattern_bull[bull_eng] = True
    pattern_name[bear_eng] = "Bear Engulfing"
    pattern_bull[bear_eng] = False
    return pattern_name, pattern_bull

def apply_pinescript_logic(df_raw):
    df = df_raw.copy()
    for col in ["open", "high", "low", "close"]: df[col] = df[col].values.flatten()
    
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["atr"] = atr(df, 14)
    
    # Simple LB Curve
    df["lb_crv"] = df["close"].rolling(10).mean()
    
    bos_up, bos_dn, is_uptrend = compute_structure(df)
    p_name, p_bull = detect_patterns(df, df["lb_crv"])
    
    # Zone Detection
    zones = []
    for i in range(2, len(df)):
        # FVG Detection
        if df["low"].iloc[i] > df["high"].iloc[i-2]:
            zones.append(Zone(df["high"].iloc[i-2], df["low"].iloc[i], i, True, False, rgba("35aa18", 0.3)))
        if df["high"].iloc[i] < df["low"].iloc[i-2]:
            zones.append(Zone(df["high"].iloc[i], df["low"].iloc[i-2], i, False, False, rgba("da1313", 0.3)))

    info_df = pd.DataFrame({
        "pattern_name": p_name, "pattern_bull": p_bull,
        "bos_up": bos_up, "bos_dn": bos_dn, "is_uptrend": is_uptrend
    }, index=df.index)
    
    return df, zones, info_df

# ---------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------
def plotchart(df_slice, zones, info_slice, full_df_index, title="SMC View"):
    fig, ax = plt.subplots(figsize=(14, 7), facecolor='#0e1117')
    ax.set_facecolor('#0e1117')
    
    dates = df_slice.index
    for i in range(len(df_slice)):
        o, c, h, l = df_slice["open"].iloc[i], df_slice["close"].iloc[i], df_slice["high"].iloc[i], df_slice["low"].iloc[i]
        color = "#26a69a" if c >= o else "#ef5350"
        ax.plot([i, i], [l, h], color=color, linewidth=1)
        ax.add_patch(plt.Rectangle((i-0.3, min(o, c)), 0.6, abs(c-o), color=color))
        
        curr_date = dates[i]
        if info_slice.loc[curr_date, "bos_up"]:
            ax.text(i, h, "BOS↑", color="lime", fontsize=8, ha="center", va="bottom")
        if info_slice.loc[curr_date, "bos_dn"]:
            ax.text(i, l, "BOS↓", color="red", fontsize=8, ha="center", va="top")

    for z in zones:
        if z.start_idx < len(full_df_index):
            z_date = full_df_index[z.start_idx]
            if z_date <= dates[-1]:
                start_x = max(0, dates.get_loc(z_date)) if z_date in dates else 0
                ax.add_patch(plt.Rectangle((start_x, z.bottom), len(df_slice)-start_x, z.top-z.bottom, color=z.color, alpha=0.2))

    plt.title(title, color="white")
    ax.tick_params(colors='white')
    return fig

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.set_page_config(page_title="SMC Dashboard", layout="wide")
ticker = st.sidebar.text_input("Ticker", "ASML")
df_raw = load_data(ticker, datetime.now()-timedelta(days=365), "1d")

if df_raw is not None:
    df, zones, info_df = apply_pinescript_logic(df_raw)
    
    if "win_end" not in st.session_state: st.session_state.win_end = len(df)
    
    # Simple navigation
    cc1, cc2 = st.sidebar.columns(2)
    if cc1.button("Prev"): st.session_state.win_end = max(20, st.session_state.win_end - 5)
    if cc2.button("Next"): st.session_state.win_end = min(len(df), st.session_state.win_end + 5)
    
    df_slice = df.iloc[st.session_state.win_end-50 : st.session_state.win_end]
    info_slice = info_df.iloc[st.session_state.win_end-50 : st.session_state.win_end]
    
    st.pyplot(plotchart(df_slice, zones, info_slice, df.index, f"{ticker} Analysis"))
else:
    st.error("Data not found.")
