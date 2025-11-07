import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(
    page_title="Global Money Flow Curve",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌍 Global Money Flow Curve Dashboard")
st.markdown("""
This app tracks how capital flows between **risk-on** (BTC, S&P 500) 
and **risk-off** (Gold, DXY) assets to estimate global risk appetite.
""")

# --- Sidebar controls ---
st.sidebar.header("⚙️ Settings")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365 * 3))
end_date = st.sidebar.date_input("End Date", datetime.now())
smooth_window = st.sidebar.slider("Smoothing (days)", 5, 60, 20)
normalize_start = st.sidebar.checkbox("Normalize to 100 at start", value=True)

# --- Fetch Data ---
st.sidebar.markdown("### Assets Used")
tickers = {
    "Bitcoin (BTC)": "BTC-USD",
    "Gold (XAU)": "GC=F",
    "S&P 500 (SPX)": "^GSPC",
    "US Dollar Index (DXY)": "DX-Y.NYB"
}
for name, t in tickers.items():
    st.sidebar.write(f"- {name} ({t})")

@st.cache_data
def load_data(tickers, start, end):
    df = yf.download(list(tickers.values()), start=start, end=end)["Adj Close"]
    df.columns = tickers.keys()
    return df

data = load_data(tickers, start_date, end_date)
if normalize_start:
    data = data / data.iloc[0] * 100

# --- Define Weights ---
weights = {
    "Bitcoin (BTC)": 0.3,
    "S&P 500 (SPX)": 0.4,
    "Gold (XAU)": -0.15,
    "US Dollar Index (DXY)": -0.15
}

# --- Compute Money Flow Curve ---
money_flow = sum(data[c] * w for c, w in weights.items())
money_flow.name = "Money Flow Curve"
money_flow_smooth = money_flow.rolling(smooth_window).mean()

# --- Merge for plotting ---
df_plot = pd.DataFrame({
    "Date": money_flow.index,
    "Money Flow Curve": money_flow,
    "Smoothed Curve": money_flow_smooth
})

# --- Altair Chart ---
base = alt.Chart(df_plot).encode(x='Date:T')

flow_chart = (
    base.mark_line(color='#1f77b4', opacity=0.7)
    .encode(y=alt.Y('Money Flow Curve:Q', title="Money Flow Index"))
    .properties(title="💰 Global Money Flow Curve")
)

smooth_chart = (
    base.mark_line(color='#d62728', size=2)
    .encode(y='Smoothed Curve:Q')
)

st.altair_chart(flow_chart + smooth_chart, use_container_width=True)

# --- Optional: show component assets ---
with st.expander("📊 Show Underlying Assets"):
    data_melted = data.reset_index().melt("Date", var_name="Asset", value_name="Value")
    line_chart = (
        alt.Chart(data_melted)
        .mark_line()
        .encode(
            x='Date:T',
            y='Value:Q',
            color='Asset:N'
        )
        .properties(title="Normalized Asset Prices")
    )
    st.altair_chart(line_chart, use_container_width=True)

# --- Info ---
st.markdown("""
**Interpretation:**
- 📈 Rising curve → money flowing into *risk-on* assets (bullish sentiment).  
- 📉 Falling curve → money moving into *safe* assets (risk-off sentiment).  
- ⚖️ Flat curve → neutral / mixed flow environment.
""")
