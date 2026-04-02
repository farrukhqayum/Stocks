import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime, timedelta

# =========================================================
# 1. THE ENGINE: 1:1 PINE SCRIPT V6 LOGIC
# =========================================================

class SMCZone:
    def __init__(self, top, btm, start_idx, is_bull, is_ob):
        self.top, self.btm = top, btm
        self.start_idx = start_idx
        self.is_bull, self.is_ob = is_bull, is_ob
        self.is_mitigated = False
        self.taps = 0
        self.age = 0
        self.status = "Active" # Active, Rejected, Invalidated

def apply_smc_v6(df):
    df = df.copy()
    high, low, close, open_p = df['high'].values, df['low'].values, df['close'].values, df['open'].values
    n = len(df)
    
    # Indicators
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    
    # LB Curve (1:1 Logic)
    lb = np.zeros(n)
    lb[0] = close[0]
    for i in range(1, n):
        prev_h = np.max(close[max(0, i-10):i])
        prev_l = np.min(close[max(0, i-10):i])
        if close[i] > prev_h: lb[i] = (high[i] + close[i]) / 2
        elif close[i] < prev_l: lb[i] = (low[i] + close[i]) / 2
        else: lb[i] = lb[i-1]
    df['lb_crv'] = pd.Series(lb, index=df.index).ewm(span=10, adjust=False).mean()

    # Structure & Zones
    zones = []
    bos_up = np.zeros(n, dtype=bool)
    bos_dn = np.zeros(n, dtype=bool)
    
    for i in range(2, n):
        # 3-Candle FVG (Gap-Aware)
        min_gap = df['atr'].iloc[i] * 0.1
        if low[i] > high[i-2] + min_gap:
            zones.append(SMCZone(low[i], high[i-2], i, True, False))
        if high[i] < low[i-2] - min_gap:
            zones.append(SMCZone(low[i-2], high[i], i, False, False))
            
        # 3-Candle OB (Displacement)
        if close[i] > high[i-1] and close[i] > open_p[i] and low[i-1] < low[i-2]:
            zones.append(SMCZone(high[i-1], low[i-1], i, True, True))
        if close[i] < low[i-1] and close[i] < open_p[i] and high[i-1] > high[i-2]:
            zones.append(SMCZone(high[i-1], low[i-1], i, False, True))

        # Mitigation & Age Logic (The 5-Bar Rule)
        for z in zones:
            if not z.is_mitigated:
                z.age = i - z.start_idx
                # Mitigation: Close past opposite side
                if (z.is_bull and close[i] < z.btm) or (not z.is_bull and close[i] > z.top):
                    z.is_mitigated = True
                    z.status = "Invalidated"
                # Tap detection
                if high[i] > z.btm and low[i] < z.top:
                    z.taps += 1
                if z.taps > 5: z.is_mitigated = True

    return df, zones

# =========================================================
# 2. DISPLAY: 1:1 WHITE UI & TABLE
# =========================================================

def draw_pine_table(ax, ticker, df_slice):
    # Precise replica of your Pine Script dashboard
    last_row = df_slice.iloc[-1]
    struct = "Bullish" if last_row['close'] > last_row['lb_crv'] else "Bearish"
    
    table_data = [
        ["SMC MTF — " + ticker],
        ["STRUCTURE: " + struct],
        ["MOMENTUM: " + ("Bullish" if last_row['ema20'] > last_row['ema50'] else "Bearish")],
        ["SIGNAL: " + ("ACTIVE" if abs(last_row['close'] - last_row['lb_crv']) < last_row['atr'] else "NONE")]
    ]
    
    # Styling the box exactly like getTablePos
    props = dict(boxstyle='square,pad=0.5', facecolor='white', edgecolor='#207e85', linewidth=1.5)
    ax.text(0.02, 0.05, "\n".join([d[0] for d in table_data]), transform=ax.transAxes, 
            fontsize=9, verticalalignment='bottom', bbox=props, color='#131722', family='monospace')

def plot_1to1(df_slice, zones, start_idx, end_idx, ticker):
    plt.style.use('fast')
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    ax.set_facecolor('white')
    
    x = np.arange(len(df_slice))
    
    # Candlesticks
    for i in range(len(df_slice)):
        row = df_slice.iloc[i]
        color = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
        ax.plot([i, i], [row['low'], row['high']], color=color, linewidth=1)
        ax.add_patch(patches.Rectangle((i-0.3, min(row['open'], row['close'])), 0.6, abs(row['close']-row['open']), color=color))

    # LB Curve
    ax.plot(x, df_slice['lb_crv'], color='gray', alpha=0.4, linewidth=1, label="LB")

    # Zones (Handling start_idx offset for sliding window)
    for z in zones:
        if start_idx <= z.start_idx < end_idx:
            local_x = z.start_idx - start_idx
            width = len(df_slice) - local_x
            
            # Color Matching from your script
            if z.is_ob:
                base_col = '#008950' if z.is_bull else '#883f0e'
            else:
                base_col = '#35aa18' if z.is_bull else '#da1313'
                
            ax.add_patch(patches.Rectangle((local_x, z.btm), width, z.top-z.btm, 
                                           facecolor=base_col, alpha=0.15, edgecolor=base_col, 
                                           linestyle='--' if not z.is_ob else '-'))

    draw_pine_table(ax, ticker, df_slice)
    
    # Formatting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, color='#f0f0f0', linestyle='-')
    ax.set_title(f"{ticker} Backtest - SMC V6", loc='left', color='#131722', fontsize=10)
    
    return fig

# =========================================================
# 3. BACKTEST NAVIGATION (1-BAR SLICING)
# =========================================================

st.set_page_config(layout="wide")

if 'ticker' not in st.session_state: st.session_state.ticker = "AAPL"
input_ticker = st.sidebar.text_input("Ticker", "AAPL").upper()

# Precompute when ticker changes
if input_ticker != st.session_state.ticker or 'full_df' not in st.session_state:
    st.session_state.ticker = input_ticker
    raw = yf.download(input_ticker, period="2y", interval="1d")
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]
    st.session_state.full_df, st.session_state.all_zones = apply_smc_v6(raw)
    st.session_state.bar_pos = 150

# Controls
c1, c2, c3, _ = st.columns([1, 1, 1, 5])
if c1.button("⬅️ Prev Bar"): st.session_state.bar_pos = max(100, st.session_state.bar_pos - 1)
if c2.button("Next Bar ➡️"): st.session_state.bar_pos = min(len(st.session_state.full_df), st.session_state.bar_pos + 1)
if c3.button("RESET"): st.session_state.bar_pos = 150

# Slicing
end = st.session_state.bar_pos
start = end - 100
df_view = st.session_state.full_df.iloc[start:end]

st.pyplot(plot_1to1(df_view, st.session_state.all_zones, start, end, st.session_state.ticker))
