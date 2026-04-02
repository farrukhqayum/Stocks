import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# =========================================================
# 1. THE ENGINE (Your Logic, Fixed for Vectorization)
# =========================================================

def apply_smc_engine(df):
    # Ensure 1D arrays to prevent "Ambiguous Truth" errors
    close = df['close'].values.flatten()
    high = df['high'].values.flatten()
    low = df['low'].values.flatten()
    
    # 20/50/200 EMA
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Structure (BOS) - Looking for local swings
    lb = 15 
    df['hi_max'] = df['high'].rolling(window=lb, center=True).max()
    df['lo_min'] = df['low'].rolling(window=lb, center=True).min()
    
    bos_up = (df['close'] > df['hi_max'].shift(1))
    bos_dn = (df['close'] < df['lo_min'].shift(1))

    # Zone Detection (FVG & OB)
    zones = []
    for i in range(2, len(df)):
        # FVG Bull (Gap between Candle i and i-2)
        if low[i] > high[i-2]:
            zones.append(dict(t=df.index[i], top=low[i], bot=high[i-2], type='bull', label='FVG'))
        # FVG Bear
        elif high[i] < low[i-2]:
            zones.append(dict(t=df.index[i], top=low[i-2], bot=high[i], type='bear', label='FVG'))
            
    return df, bos_up, bos_dn, zones

# =========================================================
# 2. UI CONFIG & DATA LOAD
# =========================================================
st.set_page_config(page_title="SMC Backtester Pro", layout="wide")

st.sidebar.title("🛠️ Backtest Controls")
ticker = st.sidebar.text_input("Ticker", "AAPL").upper()
timeframe = st.sidebar.selectbox("Interval", ["1d", "1h", "4h", "1wk"])
zoom_level = st.sidebar.slider("Visible Candles", 50, 500, 150)

@st.cache_data
def load_data(symbol, inter):
    d = yf.download(symbol, period="2y", interval=inter)
    if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
    d.columns = [c.lower() for c in d.columns]
    return d.dropna()

data = load_data(ticker, timeframe)

if not data.empty:
    df, b_up, b_dn, zones = apply_smc_engine(data)
    
    # Slice for "Backtest View"
    # This mimics your "Next/Prev" behavior but with a smooth slider
    total_len = len(df)
    current_pos = st.sidebar.slider("Timeline Position", zoom_level, total_len, total_len)
    
    df_slice = df.iloc[current_pos - zoom_level : current_pos]
    
    # =========================================================
    # 3. THE PLOT (TradingView Style)
    # =========================================================
    fig = go.Figure()

    # 1. Candlesticks
    fig.add_trace(go.Candlestick(
        x=df_slice.index, open=df_slice['open'], high=df_slice['high'],
        low=df_slice['low'], close=df_slice['close'],
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
        increasing_fillcolor='#26a69a', decreasing_fillcolor='#ef5350',
        name="OHLC"
    ))

    # 2. Indicators (EMAs)
    fig.add_trace(go.Scatter(x=df_slice.index, y=df_slice['ema20'], line=dict(color='#2962ff', width=1.5), name="EMA 20"))
    fig.add_trace(go.Scatter(x=df_slice.index, y=df_slice['ema50'], line=dict(color='#ff9800', width=1.5), name="EMA 50"))

    # 3. Draw Zones (FVGs)
    for z in zones:
        # Only draw if the zone was created before or during the visible slice
        if z['t'] in df_slice.index:
            color = "rgba(38, 166, 154, 0.25)" if z['type'] == 'bull' else "rgba(239, 83, 80, 0.25)"
            fig.add_shape(type="rect",
                x0=z['t'], x1=df_slice.index[-1], y0=z['bot'], y1=z['top'],
                fillcolor=color, line_width=0, layer="below"
            )

    # 4. Structure Breaks (BOS)
    up_idx = df_slice.index[b_up[df_slice.index]]
    dn_idx = df_slice.index[b_dn[df_slice.index]]

    fig.add_trace(go.Scatter(x=up_idx, y=df_slice.loc[up_idx, 'high'], mode='markers+text',
                             text="BOS↑", textposition="top center", marker=dict(symbol='triangle-up', color='lime')))
    
    fig.add_trace(go.Scatter(x=dn_idx, y=df_slice.loc[dn_idx, 'low'], mode='markers+text',
                             text="BOS↓", textposition="bottom center", marker=dict(symbol='triangle-down', color='red')))

    # 5. Professional Layout
    fig.update_layout(
        height=800,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=10, t=30, b=0),
        paper_bgcolor='#131722', # TradingView Dark Blue
        plot_bgcolor='#131722',
        yaxis=dict(side="right", gridcolor='#2a2e39', title="Price"),
        xaxis=dict(gridcolor='#2a2e39', title="Date")
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # Backtest Stats Table
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Structure Break Log")
        st.write(df_slice[b_up | b_dn][['close', 'ema20']].tail(5))
    with col2:
        st.subheader("Active Zones")
        st.info(f"Detected {len(zones)} total SMC zones in history.")

else:
    st.error("Ticker not found. Please verify the symbol.")
