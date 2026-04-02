import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# =========================================================
# 1. CORE STYLE & SETTINGS
# =========================================================
st.set_page_config(page_title="AlphaSMC Terminal", layout="wide")

# Custom CSS for a professional dark look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; color: #26a69a; }
    </style>
    """, unsafe_allow_html=True)

# =========================================================
# 2. DATA & LOGIC (Optimized)
# =========================================================
@st.cache_data(show_spinner=False)
def get_clean_data(ticker, days=365):
    df = yf.download(ticker, start=datetime.now()-timedelta(days=days), interval="1d")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df.dropna()

def apply_smc(df):
    # Basic Indicators
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Simple Structure (BOS)
    df['hi_20'] = df['high'].rolling(20, center=True).max()
    df['lo_20'] = df['low'].rolling(20, center=True).min()
    
    bos_up = (df['close'] > df['hi_20'].shift(1))
    bos_dn = (df['close'] < df['lo_20'].shift(1))
    
    # FVG Detection
    fvgs = []
    for i in range(2, len(df)):
        # Bullish FVG
        if df['low'].iloc[i] > df['high'].iloc[i-2]:
            fvgs.append({'type': 'bull', 'top': df['low'].iloc[i], 'bot': df['high'].iloc[i-2], 'idx': df.index[i]})
        # Bearish FVG
        if df['high'].iloc[i] < df['low'].iloc[i-2]:
            fvgs.append({'type': 'bear', 'top': df['low'].iloc[i-2], 'bot': df['high'].iloc[i], 'idx': df.index[i]})
            
    return df, bos_up, bos_dn, fvgs

# =========================================================
# 3. SIDEBAR & UI CONTROLS
# =========================================================
st.sidebar.title("⚡ AlphaSMC v2")
ticker = st.sidebar.text_input("Symbol", "NVDA").upper()
lookback = st.sidebar.slider("Chart Lookback", 30, 200, 100)

df_raw = get_clean_data(ticker)

if not df_raw.empty:
    df, bos_up, bos_dn, fvgs = apply_smc(df_raw)
    
    # Header Metrics
    last_price = df['close'].iloc[-1]
    change = last_price - df['close'].iloc[-2]
    
    m1, m2, m3 = st.columns(3)
    m1.metric(f"{ticker} Price", f"${last_price:.2f}", f"{change:.2f}")
    m2.metric("Trend", "BULLISH" if df['ema20'].iloc[-1] > df['ema50'].iloc[-1] else "BEARISH")
    m3.metric("Volatility (ATR)", f"{ (df['high']-df['low']).rolling(14).mean().iloc[-1]:.2f}")

    # =========================================================
    # 4. THE PLOT (Plotly Interactive)
    # =========================================================
    df_plot = df.suffix('').tail(lookback)
    
    fig = go.Figure()

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df_plot.index, open=df_plot['open'], high=df_plot['high'],
        low=df_plot['low'], close=df_plot['close'],
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
        name="Price"
    ))

    # EMAs
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['ema20'], line=dict(color='#2962ff', width=1), name="EMA 20"))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['ema50'], line=dict(color='#ff9800', width=1), name="EMA 50"))

    # Add FVGs (Rectangles)
    for f in fvgs:
        if f['idx'] in df_plot.index:
            color = "rgba(38, 166, 154, 0.2)" if f['type'] == 'bull' else "rgba(239, 83, 80, 0.2)"
            fig.add_shape(type="rect",
                x0=f['idx'], x1=df_plot.index[-1], y0=f['bot'], y1=f['top'],
                fillcolor=color, line_width=0, layer="below"
            )

    # Add BOS Labels
    plot_bos_up = bos_up[df_plot.index]
    plot_bos_dn = bos_dn[df_plot.index]
    
    fig.add_trace(go.Scatter(
        x=df_plot.index[plot_bos_up], y=df_plot['high'][plot_bos_up],
        mode='text', text="BOS ↑", textposition="top center",
        textfont=dict(color="lime", size=10), name="Structure Break"
    ))

    # Layout Styling
    fig.update_layout(
        height=700,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(gridcolor='#1e222d', zeroline=False),
        xaxis=dict(gridcolor='#1e222d', zeroline=False),
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
    )

    st.plotly_chart(fig, use_container_width=True)

    # Logic Information
    with st.expander("Show Raw Signal Data"):
        st.dataframe(df_plot.tail(10), use_container_width=True)

else:
    st.error("Could not fetch data. Check the ticker symbol.")
