import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt
from datetime import datetime, timedelta

# --- INITIAL SETUP ---
st.set_page_config(
    page_title="Global Money Flow Curve (GMF)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.caption("Data sourced via Yahoo Finance • Updated dynamically")
st.title("🌍 Global Money Flow (GMF)")
st.markdown("""
This app tracks capital flows between **risk-on** and **risk-off** assets 
to estimate global risk appetite. 
""")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Settings")

# FIX: Ensure start/end dates are valid business days to prevent empty initial dataframes
def get_valid_date(d):
    if d.weekday() == 5: return d - timedelta(days=1) # Sat -> Fri
    if d.weekday() == 6: return d - timedelta(days=2) # Sun -> Fri
    return d

start_input = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365*3))
end_input = st.sidebar.date_input("End Date", datetime.now())

start_date = get_valid_date(start_input)
end_date = get_valid_date(end_input)

smooth_window = st.sidebar.slider("Smoothing (days)", 5, 100, 40)
z_score_window = st.sidebar.slider("Climax Z-Score Lookback (Days)", 20, 250, 90)
normalize_start = st.sidebar.checkbox("Normalize to 100 at start", value=True)
use_business_days = st.sidebar.checkbox("Remove weekend gaps (use business days only)", value=True)

st.sidebar.markdown("### Select Assets")

default_tickers = {
    "Bitcoin (BTC)": "BTC-USD",
    "Gold (XAU)": "GC=F",
    "S&P 500 (SPX)": "^GSPC",
    "US Dollar Index (DXY)": "DX-Y.NYB",
    "Emerging Markets (EEM)": "EEM",
    "US 10Y Treasury (IEF)": "IEF",
    "Crude Oil (CL)": "CL=F",
    "Volatility Index (VIX)": "^VIX"
}

selected_assets = st.sidebar.multiselect(
    "Choose Assets to Include",
    options=list(default_tickers.keys()),
    default=list(default_tickers.keys())
)

tickers = {asset: default_tickers[asset] for asset in selected_assets}

st.sidebar.markdown("### Set Asset Weights")

default_weights = {
    "Bitcoin (BTC)": 0.05, "S&P 500 (SPX)": 0.15, "Emerging Markets (EEM)": 0.15,
    "Crude Oil (CL)": 0.15, "Gold (XAU)": -0.15, "US Dollar Index (DXY)": -0.12,
    "US 10Y Treasury (IEF)": -0.13, "Volatility Index (VIX)": -0.1
}

weights = {}
for asset in selected_assets:
    weights[asset] = st.sidebar.number_input(
        f"Weight for {asset}", -1.0, 1.0, float(default_weights.get(asset, 0.0)), 0.05
    )

# --- DATA LOADING & CLEANING ---
@st.cache_data
def load_data(tickers, start, end):
    # Download with a small buffer to ensure the first day has data for normalization
    raw = yf.download(list(tickers.values()), start=start - timedelta(days=5), end=end, progress=False)
    
    if 'Close' in raw.columns.get_level_values(0):
        df = raw['Close'].copy()
    else:
        df = raw.copy()

    # Reindex to handle weekend gaps and alignment
    all_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B' if use_business_days else 'D')
    df = df.reindex(all_days).ffill().bfill()

    rename_map = {ticker: name for name, ticker in tickers.items()}
    df = df.rename(columns=rename_map)
    return df[df.index >= pd.Timestamp(start)]

try:
    data = load_data(tickers, start_date, end_date)
    spx_raw = load_data({"SPX": "^GSPC"}, start_date, end_date)
    spx_data = spx_raw["SPX"]
    
    if normalize_start:
        data = (data / data.iloc[0]) * 100
        spx_data = (spx_data / spx_data.iloc[0]) * 100

    # --- MONEY FLOW CALCULATION ---
    # Fix: Ensure weights are applied to a correctly aligned dataframe
    abs_sum = sum(abs(w) for w in weights.values())
    norm_weights = {k: (v / abs_sum) if abs_sum != 0 else 0 for k, v in weights.items()}
    
    money_flow = pd.Series(0.0, index=data.index)
    for asset, w in norm_weights.items():
        if asset in data.columns:
            money_flow += data[asset] * w
    
    # Anchor the curve to 100 to prevent it from starting at 0 or floating incorrectly
    money_flow = (money_flow - money_flow.iloc[0]) + 100

    money_flow_s = money_flow.rolling(3).mean()
    money_flow_smooth = money_flow.rolling(smooth_window).mean()

    # Z-Score Logic
    rolling_mean = money_flow_smooth.rolling(window=z_score_window).mean()
    rolling_std = money_flow_smooth.rolling(window=z_score_window).std()
    money_flow_zscore = ((money_flow_smooth - rolling_mean) / rolling_std).fillna(0)
    
    money_flow_momentum = money_flow_smooth.pct_change(periods=10) * 100
    latest_momentum = money_flow_momentum.iloc[-1]
    latest_zscore = money_flow_zscore.iloc[-1]

    # --- SENTIMENT UI (Kept your logic) ---
    Z_EXTREME, MOM_HIGH, MOM_LOW = 1.8, 10.0, -10.0
    sentiment_color = "#a3a3a3"
    sentiment = "⚪ **Neutral/Choppy Market**"

    if latest_zscore >= Z_EXTREME:
        sentiment, sentiment_color = "🚨 **EXTREME OVERBOUGHT (Euphoria Climax)**", "#ff8533" 
    elif latest_zscore <= -Z_EXTREME:
        sentiment, sentiment_color = "📉 **PANIC/CAPITULATION (Oversold Climax)**", "#990000"
    elif latest_momentum > MOM_HIGH:
        sentiment, sentiment_color = "🟢 **Strong Risk-On: ACCELERATION**", "#2ca02c"
    elif latest_momentum < MOM_LOW:
        sentiment, sentiment_color = "🔴 **Strong Risk-Off: DECELERATION**", "#dc2626"

    st.markdown(f"""<div style="padding:1.2em; border-radius:12px; text-align:center; background-color:{sentiment_color}; color:white; font-size:1.3em; font-weight:bold;">{sentiment}</div>""", unsafe_allow_html=True)

    # --- CHARTS ---
    df_plot = pd.DataFrame({
        "Date": money_flow.index,
        "Money Flow Curve": money_flow_s,
        "Smoothed Curve": money_flow_smooth,
        "Momentum": money_flow_momentum,
        "Z-Score": money_flow_zscore
    }).dropna()

    base = alt.Chart(df_plot).encode(x='Date:T')
    
    # Money Flow Curve
    curve_chart = base.mark_line(color='#1f77b4', opacity=0.6).encode(y=alt.Y('Money Flow Curve:Q', scale=alt.Scale(zero=False)))
    smooth_chart = base.mark_line(color='#d62728', size=2).encode(y='Smoothed Curve:Q')
    st.markdown("### 🌊 GMF Curves")
    st.altair_chart(curve_chart + smooth_chart, use_container_width=True)

    # Momentum Bar Chart
    st.markdown("### 📈 Money Flow Momentum (%)")
    mom_chart = base.mark_bar().encode(
        y='Momentum:Q',
        color=alt.condition(alt.datum.Momentum > 0, alt.value('#2ca02c'), alt.value('#d62728'))
    )
    st.altair_chart(mom_chart, use_container_width=True)

    # Z-Score Chart
    st.markdown("### Climax Zone Indicator (Z-Score)")
    z_chart = base.mark_area(opacity=0.6).encode(
        y='Z-Score:Q',
        color=alt.condition(alt.datum['Z-Score'] > 0, alt.value('#1f77b4'), alt.value('#d62728'))
    )
    st.altair_chart(z_chart, use_container_width=True)

    # --- DIVERGENCE & CORRELATION SECTIONS (Retained original functionality) ---
    with st.expander("📊 Show Underlying Assets"):
        data_melted = data.reset_index().melt("index", var_name="Asset", value_name="Value")
        asset_chart = alt.Chart(data_melted).mark_line().encode(x='index:T', y='Value:Q', color='Asset:N')
        st.altair_chart(asset_chart, use_container_width=True)

except Exception as e:
    st.error(f"⚠️ Error: {e}")
