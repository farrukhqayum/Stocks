import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt
from datetime import datetime, timedelta

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Global Money Flow Curve",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- TITLE ---
st.title("🌍 Global Money Flow Curve Dashboard")
st.markdown("""
This app tracks how capital flows between **risk-on** (BTC, S&P 500)
and **risk-off** (Gold, DXY) assets to estimate global risk appetite.
""")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Settings")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365 * 3))
end_date = st.sidebar.date_input("End Date", datetime.now())
smooth_window = st.sidebar.slider("Smoothing (days)", 5, 60, 20)
normalize_start = st.sidebar.checkbox("Normalize to 100 at start", value=True)
use_business_days = st.sidebar.checkbox("Remove weekend gaps (use business days only)", value=True)

# --- ASSET TICKERS ---
st.sidebar.markdown("### Assets Used")
tickers = {
    "Bitcoin (BTC)": "BTC-USD",
    "Gold (XAU)": "GC=F",
    "S&P 500 (SPX)": "^GSPC",
    "US Dollar Index (DXY)": "DX-Y.NYB"
}
for name, t in tickers.items():
    st.sidebar.write(f"- {name} ({t})")

# --- DATA FETCH FUNCTION ---
@st.cache_data
def load_data(tickers, start, end):
    """Download adjusted close data for all tickers, handle multiindex safely."""
    raw = yf.download(list(tickers.values()), start=start, end=end, progress=False)

    # Handle MultiIndex columns (Open, High, Low, Close, Adj Close)
    if isinstance(raw.columns, pd.MultiIndex):
        if 'Adj Close' in raw.columns.get_level_values(0):
            df = raw['Adj Close'].copy()
        elif 'Close' in raw.columns.get_level_values(0):
            df = raw['Close'].copy()
        else:
            raise ValueError("No 'Adj Close' or 'Close' data found.")
    else:
        df = raw.copy()

    # Rename columns to readable names
    rename_map = {}
    for name, ticker in tickers.items():
        if ticker in df.columns:
            rename_map[ticker] = name
        elif name in df.columns:
            rename_map[name] = name
    df = df.rename(columns=rename_map)

    # Drop empty columns
    df = df.dropna(axis=1, how='all')

    return df

# --- LOAD DATA WITH FALLBACK FOR DXY ---
try:
    data = load_data(tickers, start_date, end_date)
except Exception:
    st.warning("⚠️ DX-Y.NYB data unavailable. Using USDOLLAR fallback.")
    tickers["US Dollar Index (DXY)"] = "USDOLLAR"
    data = load_data(tickers, start_date, end_date)

# --- OPTIONAL: RESAMPLE TO BUSINESS DAYS ---
if use_business_days:
    data = data.asfreq('B')  # keep only weekdays
    data = data.fillna(method='ffill')  # forward-fill weekends

# --- NORMALIZE TO 100 ---
if normalize_start:
    data = data / data.iloc[0] * 100

# --- WEIGHTS ---
weights = {
    "Bitcoin (BTC)": 0.3,
    "S&P 500 (SPX)": 0.4,
    "Gold (XAU)": -0.15,
    "US Dollar Index (DXY)": -0.15
}

# --- COMPUTE MONEY FLOW CURVE ---
money_flow = pd.Series(0, index=data.index, name="Money Flow Curve")
for asset, w in weights.items():
    if asset in data.columns:
        money_flow += data[asset] * w

# --- SMOOTHED CURVE ---
money_flow_smooth = money_flow.rolling(smooth_window).mean()

# --- MOMENTUM (RATE OF CHANGE) ---
money_flow_momentum = money_flow_smooth.pct_change() * 100
money_flow_momentum = money_flow_momentum.fillna(0)

# --- MERGE DATA FOR ALTair ---
df_plot = pd.DataFrame({
    "Date": money_flow.index,
    "Money Flow Curve": money_flow,
    "Smoothed Curve": money_flow_smooth,
    "Momentum": money_flow_momentum
}).dropna()

# --- MONEY FLOW CURVE CHART ---
base = alt.Chart(df_plot).encode(x='Date:T')

curve_chart = (
    base.mark_line(color='#1f77b4', opacity=0.6)
    .encode(y=alt.Y('Money Flow Curve:Q', title='Money Flow Index'))
    .properties(title="💰 Global Money Flow Curve")
)

smooth_chart = (
    base.mark_line(color='#d62728', size=2)
    .encode(y='Smoothed Curve:Q')
)

st.altair_chart(curve_chart + smooth_chart, use_container_width=True)

# --- MOMENTUM CHART ---
momentum_chart = (
    alt.Chart(df_plot)
    .mark_bar()
    .encode(
        x='Date:T',
        y=alt.Y('Momentum:Q', title='Flow Momentum (%)'),
        color=alt.condition(
            alt.datum.Momentum > 0,
            alt.value('#2ca02c'),
            alt.value('#d62728')
        ),
        tooltip=['Date:T', 'Momentum:Q']
    )
    .properties(title="📈 Money Flow Momentum (Rate of Change %)")
)
st.altair_chart(momentum_chart, use_container_width=True)

# --- UNDERLYING ASSETS ---
with st.expander("📊 Show Underlying Assets"):
    data_melted = data.reset_index().melt("Date", var_name="Asset", value_name="Value")
    asset_chart = (
        alt.Chart(data_melted)
        .mark_line()
        .encode(
            x='Date:T',
            y='Value:Q',
            color='Asset:N',
            tooltip=['Date:T', 'Asset:N', 'Value:Q']
        )
        .properties(title="Normalized Asset Prices (Indexed)")
    )
    st.altair_chart(asset_chart, use_container_width=True)

# --- INTERPRETATION ---
st.markdown("""
### 🧠 Interpretation
- 📈 **Rising Curve:** Capital flowing into *risk-on* assets → bullish market sentiment.  
- 📉 **Falling Curve:** Money shifting to *safe* assets → defensive / risk-off tone.  
- ⚖️ **Flat Curve:** Neutral or mixed capital rotation.  
- 🟢 **Positive Momentum:** Acceleration of risk-on flows.  
- 🔴 **Negative Momentum:** Acceleration of risk-off flows.
""")

st.caption("Data sourced via Yahoo Finance • Updated dynamically")
