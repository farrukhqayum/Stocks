import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime, timedelta

# =========================================================
# 1. DATA LOADING
# =========================================================
@st.cache_data
def load_data(ticker, start_date, interval):
    try:
        data = yf.download(ticker, start=start_date, interval=interval)
        if data.empty: return None
        # Flatten columns if multi-index (common in newer yfinance versions)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# =========================================================
# 2. SMC LOGIC ENGINE (The "PineScript" Core)
# =========================================================
def apply_pinescript_logic(df):
    # Setup Indicators
    df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # ATR for filters
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(14).mean()

    zones = [] # List to store Zone dictionaries

    # Sequential scan for Zones (FVG & OB)
    for i in range(2, len(df)):
        current_idx = df.index[i]
        
        # --- FVG Detection ---
        min_gap = df['atr'].iloc[i] * 0.1
        # Bull FVG
        if df['Low'].iloc[i] > (df['High'].iloc[i-2] + min_gap):
            zones.append({
                'type': 'FVG', 'is_bull': True, 'top': df['Low'].iloc[i], 
                'btm': df['High'].iloc[i-2], 'start_idx': i, 'mitigated': False
            })
        # Bear FVG
        if df['High'].iloc[i] < (df['Low'].iloc[i-2] - min_gap):
            zones.append({
                'type': 'FVG', 'is_bull': False, 'top': df['Low'].iloc[i-2], 
                'btm': df['High'].iloc[i], 'start_idx': i, 'mitigated': False
            })

        # --- OB Detection (Simplified Displacement) ---
        # Bull OB
        if df['Close'].iloc[i] > df['High'].iloc[i-1] and df['Close'].iloc[i] > df['Open'].iloc[i]:
            zones.append({
                'type': 'OB', 'is_bull': True, 'top': df['High'].iloc[i-1], 
                'btm': df['Low'].iloc[i-1], 'start_idx': i, 'mitigated': False
            })
        # Bear OB
        if df['Close'].iloc[i] < df['Low'].iloc[i-1] and df['Close'].iloc[i] < df['Open'].iloc[i]:
            zones.append({
                'type': 'OB', 'is_bull': False, 'top': df['High'].iloc[i-1], 
                'btm': df['Low'].iloc[i-1], 'start_idx': i, 'mitigated': False
            })

        # --- Mitigation Check (Ongoing) ---
        for z in zones:
            if not z['mitigated'] and i > z['start_idx']:
                if z['is_bull'] and df['Close'].iloc[i] < z['btm']:
                    z['mitigated'] = True
                elif not z['is_bull'] and df['Close'].iloc[i] > z['top']:
                    z['mitigated'] = True

    # Signal Generation (Simplified for Chart Symbols)
    df['long_sig'] = (df['Close'] > df['ema20']) & (df['Close'].shift(1) <= df['ema20'].shift(1))
    df['short_sig'] = (df['Close'] < df['ema20']) & (df['Close'].shift(1) >= df['ema20'].shift(1))
    
    return df, zones

# =========================================================
# 3. PLOTTING ENGINE (Matplotlib)
# =========================================================
def plotchart(df_slice, all_zones, info_df, title="Chart"):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#131722')
    ax.set_facecolor('#131722')
    
    x = np.arange(len(df_slice))
    
    # Plot Candlesticks
    for i in range(len(df_slice)):
        color = '#26a69a' if df_slice['Close'].iloc[i] >= df_slice['Open'].iloc[i] else '#ef5350'
        # Wick
        ax.plot([i, i], [df_slice['Low'].iloc[i], df_slice['High'].iloc[i]], color=color, linewidth=1)
        # Body
        ax.add_patch(patches.Rectangle((i - 0.3, min(df_slice['Open'].iloc[i], df_slice['Close'].iloc[i])), 
                                      0.6, abs(df_slice['Open'].iloc[i] - df_slice['Close'].iloc[i]), 
                                      color=color, zorder=3))

    # Plot Active Zones
    slice_start_date = df_slice.index[0]
    for z in all_zones:
        # Filter: Only show zones created before slice end and not mitigated yet
        if not z['mitigated'] and df.index[z['start_idx']] <= df_slice.index[-1]:
            # Calculate where the zone starts relative to our current slice
            try:
                z_start_pos = df_slice.index.get_loc(df.index[z['start_idx']])
            except KeyError:
                z_start_pos = 0 # It started before this slice
            
            z_color = 'green' if z['is_bull'] else 'red'
            z_alpha = 0.15 if z['type'] == 'OB' else 0.08
            
            rect = patches.Rectangle((z_start_pos, z['btm']), len(df_slice)-z_start_pos, 
                                     z['top']-z['btm'], color=z_color, alpha=z_alpha, label=z['type'])
            ax.add_patch(rect)

    # Plot Signals (Symbols)
    longs = df_slice[df_slice['long_sig']]
    shorts = df_slice[df_slice['short_sig']]
    
    ax.scatter(np.where(df_slice['long_sig'])[0], df_slice.loc[df_slice['long_sig'], 'Low'] * 0.99, 
               marker='^', color='#00ff00', s=100, label='Long', zorder=5)
    ax.scatter(np.where(df_slice['short_sig'])[0], df_slice.loc[df_slice['short_sig'], 'High'] * 1.01, 
               marker='v', color='#ff0000', s=100, label='Short', zorder=5)

    # Styling
    ax.set_title(title, color='white', loc='left', fontsize=14)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.grid(color='#2a2e39', linestyle='--', alpha=0.5)
    plt.xticks(x[::max(1, len(x)//10)], df_slice.index.strftime('%Y-%m-%d')[::max(1, len(x)//10)], rotation=45)
    
    return fig

# =========================================================
# 4. STREAMLIT UI (FULL APP)
# =========================================================
st.set_page_config(page_title="SMC FVG Dashboard", layout="wide")

st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker", "ASML")
tf = st.sidebar.selectbox("Timeframe", ["1D", "1W", "1M"], index=0)

today = datetime.today()
if tf == "1D":
    start_date = today - timedelta(days=365)
    interval = "1d"
elif tf == "1W":
    start_date = today - timedelta(days=365*3)
    interval = "1wk"
elif tf == "1M":
    start_date = today - timedelta(days=365*10)
    interval = "1mo"

df = load_data(ticker, start_date, interval)

if df is None or df.empty:
    st.error("No data found.")
    st.stop()

# Apply logic once to full DF
df, all_zones = apply_pinescript_logic(df)

# Slicing / Window Logic
if "window_end_idx" not in st.session_state:
    st.session_state.window_end_idx = len(df)
if "window_size" not in st.session_state:
    st.session_state.window_size = 100

col1, col2, col3 = st.columns([1, 1, 2])

if col1.button("⬅️ Back"):
    st.session_state.window_end_idx = max(st.session_state.window_size, st.session_state.window_end_idx - 10)

if col2.button("Next ➡️"):
    st.session_state.window_end_idx = min(len(df), st.session_state.window_end_idx + 10)

end_idx = st.session_state.window_end_idx
start_idx = max(0, end_idx - st.session_state.window_size)

df_slice = df.iloc[start_idx : end_idx]

with col3:
    st.write(f"Showing indices **{start_idx}** to **{end_idx}** (Total: {len(df)})")

fig = plotchart(df_slice, all_zones, None, title=f"{ticker} {tf} - SMC Dashboard")
st.pyplot(fig)
