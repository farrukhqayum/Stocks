import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime, timedelta

# =========================================================
# 1. CORE SMC ENGINE (PINE SCRIPT V6 PORT)
# =========================================================

def apply_smc_v6_logic(df):
    n = len(df)
    high, low, close, open_p = df['high'].values, df['low'].values, df['close'].values, df['open'].values
    
    # --- Indicators ---
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # LB Curve Logic
    lb_val = np.zeros(n)
    lb_val[0] = close[0]
    for i in range(1, n):
        prev_h = np.max(close[max(0, i-10):i])
        prev_l = np.min(close[max(0, i-10):i])
        if close[i] > prev_h: lb_val[i] = (high[i] + close[i]) / 2
        elif close[i] < prev_l: lb_val[i] = (low[i] + close[i]) / 2
        else: lb_val[i] = lb_val[i-1]
    df['lb_crv'] = pd.Series(lb_val, index=df.index).ewm(span=10, adjust=False).mean()

    # --- Market Structure (BOS/CHoCH) ---
    bos_up = np.zeros(n, dtype=bool)
    bos_dn = np.zeros(n, dtype=bool)
    last_hi, last_lo = np.nan, np.nan
    is_uptrend = False
    
    for i in range(20, n-5):
        if high[i] == np.max(high[i-20:i+6]): last_hi = high[i]
        if low[i] == np.min(low[i-20:i+6]): last_lo = low[i]
        if not np.isnan(last_hi) and close[i] > last_hi:
            bos_up[i], last_hi, is_uptrend = True, np.nan, True
        if not np.isnan(last_lo) and close[i] < last_lo:
            bos_dn[i], last_lo, is_uptrend = True, np.nan, False

    # --- 3-Candle FVG & OB Detection ---
    zones = []
    atr = (df['high'] - df['low']).rolling(14).mean().values
    for i in range(2, n):
        # FVG
        if low[i] > high[i-2] + (atr[i] * 0.1):
            zones.append({'t': df.index[i], 'top': low[i], 'btm': high[i-2], 'type': 'bull_fvg'})
        if high[i] < low[i-2] - (atr[i] * 0.1):
            zones.append({'t': df.index[i], 'top': low[i-2], 'btm': high[i], 'type': 'bear_fvg'})
        # OB (Displacement)
        if close[i] > high[i-1] and close[i] > open_p[i] and low[i-1] < low[i-2]:
            zones.append({'t': df.index[i], 'top': high[i-1], 'btm': low[i-1], 'type': 'bull_ob'})

    # --- Candle Patterns ---
    df['pattern'] = ""
    for i in range(2, n):
        if close[i] > open_p[i] and close[i-1] < open_p[i-1] and close[i] >= high[i-1]:
            df.at[df.index[i], 'pattern'] = "Bull Engulfing"
        elif close[i] < open_p[i] and close[i-1] > open_p[i-1] and close[i] <= low[i-1]:
            df.at[df.index[i], 'pattern'] = "Bear Engulfing"

    return df, zones, bos_up, bos_dn, is_uptrend

# =========================================================
# 2. PLOTTING ENGINE (WHITE BACKGROUND + SMC TABLE)
# =========================================================

def draw_smc_dashboard(ax, ticker, trend, pattern, fvg_status):
    # Mimics the Pine Script Table
    text_str = (
        f"SMC MTF — {ticker}\n"
        f"STRUCTURE: {'Bullish' if trend else 'Bearish'}\n"
        f"PATTERN: {pattern if pattern else 'None'}\n"
        f"FVG BIAS: {fvg_status}"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#207e85', linewidth=2)
    ax.text(0.02, 0.95, text_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')

def plot_backtest(df_slice, zones, bos_up_slice, bos_dn_slice, trend):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    ax.set_facecolor('white')
    
    idx = np.arange(len(df_slice))
    dates = df_slice.index
    
    # Candlesticks
    for i in range(len(df_slice)):
        o, c, h, l = df_slice.iloc[i][['open', 'close', 'high', 'low']]
        color = '#26a69a' if c >= o else '#ef5350'
        ax.plot([i, i], [l, h], color=color, linewidth=1)
        ax.add_patch(patches.Rectangle((i-0.3, min(o, c)), 0.6, abs(c-o), color=color, alpha=0.8))
        
        # Annotate Patterns
        pat = df_slice['pattern'].iloc[i]
        if pat:
            ax.text(i, h*1.002 if 'Bear' in pat else l*0.998, f"● {pat}", 
                    color='blue', fontsize=7, ha='center', rotation=45)

    # BOS Annotations
    for i in range(len(df_slice)):
        if bos_up_slice[i]:
            ax.annotate("BOS ↑", xy=(i, df_slice['high'].iloc[i]), xytext=(0, 10),
                        textcoords='offset points', color='green', fontweight='bold', ha='center')
        if bos_dn_slice[i]:
            ax.annotate("BOS ↓", xy=(i, df_slice['low'].iloc[i]), xytext=(0, -15),
                        textcoords='offset points', color='red', fontweight='bold', ha='center')

    # Draw Indicators
    ax.plot(idx, df_slice['ema20'], color='blue', alpha=0.3, label='EMA20')
    ax.plot(idx, df_slice['lb_crv'], color='orange', linestyle='--', alpha=0.5, label='LB Curve')

    # Zones (Active only in window)
    for z in zones:
        if z['t'] in dates:
            z_idx = dates.get_loc(z['t'])
            color = 'green' if 'bull' in z['type'] else 'red'
            ax.add_patch(patches.Rectangle((z_idx, z['btm']), len(df_slice)-z_idx, z['top']-z['btm'], 
                                           color=color, alpha=0.1, linewidth=0))

    # SMC Table
    last_pattern = df_slice['pattern'].replace("", np.nan).ffill().iloc[-1]
    draw_smc_dashboard(ax, ticker, trend, last_pattern, "Bull Bias" if trend else "Bear Bias")

    ax.grid(True, color='#e0e0e0', linestyle='--', alpha=0.5)
    ax.set_ylabel("Price")
    plt.xticks(idx[::10], dates[::10].strftime('%d %b'), rotation=0)
    return fig

# =========================================================
# 3. STREAMLIT UI & DATA PERSISTENCE
# =========================================================

st.set_page_config(layout="wide")
ticker = st.sidebar.text_input("Symbol", "NVDA").upper()

if 'step' not in st.session_state:
    st.session_state.step = 150

# Preload Data
df_raw = yf.download(ticker, period="2y", interval="1d")
if not df_raw.empty:
    if isinstance(df_raw.columns, pd.MultiIndex): df_raw.columns = df_raw.columns.get_level_values(0)
    df_raw.columns = [c.lower() for c in df_raw.columns]
    
    # Precompute All Logic
    df, zones, b_up, b_dn, trend = apply_smc_v6_logic(df_raw)

    # Navigation Controls
    col1, col2, col3, col4 = st.columns([1,1,1,5])
    if col1.button("⬅️ Previous Bar"): st.session_state.step -= 1
    if col2.button("Next Bar ➡️"): st.session_state.step += 1
    if col3.button("Reset"): st.session_state.step = 150

    # Slicing
    end = st.session_state.step
    start = max(0, end - 100)
    df_slice = df.iloc[start:end]
    bos_up_slice = b_up[start:end]
    bos_dn_slice = b_dn[start:end]

    st.pyplot(plot_backtest(df_slice, zones, bos_up_slice, bos_dn_slice, trend))
