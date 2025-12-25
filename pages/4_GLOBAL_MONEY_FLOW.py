import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt
from datetime import datetime, timedelta

# --- INITIAL SETUP ---
st.set_page_config(page_title="Global Money Flow Curve (GMF)", layout="wide")

def validate_date(date_val):
    """Ensures start/end dates fall on business days to prevent yfinance empty returns."""
    if date_val.weekday() > 4:  # Saturday or Sunday
        return date_val - timedelta(days=date_val.weekday() - 4)
    return date_val

st.title("🌍 Global Money Flow (GMF)")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Settings")
raw_start = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365*3))
raw_end = st.sidebar.date_input("End Date", datetime.now())

# Fix weekend start/end dates immediately
start_date = validate_date(raw_start)
end_date = validate_date(raw_end)

smooth_window = st.sidebar.slider("Smoothing (days)", 5, 100, 40)
z_score_window = st.sidebar.slider("Climax Z-Score Lookback (Days)", 20, 250, 90)

# --- ASSET SELECTION & WEIGHTING ---
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

selected_assets = st.sidebar.multiselect("Assets", options=list(default_tickers.keys()), default=list(default_tickers.keys()))
weights = {asset: st.sidebar.number_input(f"Weight: {asset}", -1.0, 1.0, 0.1) for asset in selected_assets}

@st.cache_data
def load_data(ticker_dict, start, end):
    # Download with a buffer to ensure we have enough data for the first index
    buffer_start = start - timedelta(days=10)
    df = yf.download(list(ticker_dict.values()), start=buffer_start, end=end, progress=False)['Close']
    
    # Rename columns back to human-readable names
    inv_map = {v: k for k, v in ticker_dict.items()}
    df = df.rename(columns=inv_map)
    
    # Handle the "Weekend No Data" issue by forward filling and reindexing to business days
    all_bus_days = pd.date_range(start=buffer_start, end=end, freq='B')
    df = df.reindex(all_bus_days).ffill().bfill()
    
    # Trim back to requested start date
    return df[df.index >= pd.Timestamp(start)]

try:
    data = load_data({k: default_tickers[k] for k in selected_assets}, start_date, end_date)
    
    # --- NORMALIZATION (FIXES THE ZERO FLOW ISSUE) ---
    # We normalize each asset to 100 at the START of the period
    normalized_data = (data / data.iloc[0]) * 100

    # Calculate Weighted Money Flow
    money_flow = pd.Series(0, index=normalized_data.index)
    for asset, w in weights.items():
        if asset in normalized_data.columns:
            money_flow += normalized_data[asset] * w

    # Rescale the curve so it starts at 100 for better indexing visualization
    money_flow = (money_flow - money_flow.iloc[0]) + 100
    
    # --- CALCULATIONS ---
    money_flow_smooth = money_flow.rolling(smooth_window).mean()
    rolling_mean = money_flow_smooth.rolling(window=z_score_window).mean()
    rolling_std = money_flow_smooth.rolling(window=z_score_window).std()
    z_score = (money_flow_smooth - rolling_mean) / rolling_std

    # --- PLOTTING ---
    df_plot = pd.DataFrame({
        "Date": money_flow.index,
        "GMF": money_flow,
        "GMF_Smooth": money_flow_smooth,
        "Z_Score": z_score.fillna(0)
    }).dropna()

    st.subheader("🌊 Global Money Flow Curve")
    c = alt.Chart(df_plot).mark_line().encode(
        x='Date:T',
        y=alt.Y('GMF_Smooth:Q', scale=alt.Scale(zero=False), title="Index Value (Base 100)"),
        tooltip=['Date', 'GMF_Smooth']
    ).interactive()
    st.altair_chart(c, use_container_width=True)

    st.subheader("📊 Climax Zone (Z-Score)")
    z_chart = alt.Chart(df_plot).mark_area(opacity=0.5).encode(
        x='Date:T',
        y='Z_Score:Q',
        color=alt.condition(alt.datum.Z_Score > 0, alt.value("green"), alt.value("red"))
    ).properties(height=200)
    st.altair_chart(z_chart, use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Try adjusting the Start Date to a Business Day or increasing the asset selection.")
