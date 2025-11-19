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
st.title("🌍 Global Money Flow")
st.markdown("""
This app tracks capital flows between **risk-on** and **risk-off** assets 
to estimate global risk appetite.  
Includes BTC, S&P 500, Emerging Markets, Gold, US Dollar, Treasury Bonds, Oil, and VIX.
""")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Settings")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365*3))
end_date = st.sidebar.date_input("End Date", datetime.now())
smooth_window = st.sidebar.slider("Smoothing (days)", 5, 100, 50)
normalize_start = st.sidebar.checkbox("Normalize to 100 at start", value=True)
use_business_days = st.sidebar.checkbox("Remove weekend gaps (use business days only)", value=True)

# --- ASSET TICKERS ---
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

# Let user select tickers to include
selected_assets = st.sidebar.multiselect(
    "Choose Assets to Include",
    options=list(default_tickers.keys()),
    default=list(default_tickers.keys())
)

# Filter tickers dictionary based on user selection
tickers = {asset: default_tickers[asset] for asset in selected_assets}

# --- ASSET WEIGHTS ---
st.sidebar.markdown("### Set Asset Weights (Positive=Risk-On, Negative=Risk-Off)")

default_weights = {
    "Bitcoin (BTC)": 0.10,
    "S&P 500 (SPX)": 0.12,
    "Emerging Markets (EEM)": 0.15,
    "Crude Oil (CL)": 0.10,
    "Gold (XAU)": -0.15,
    "US Dollar Index (DXY)": -0.10,
    "US 10Y Treasury (IEF)": -0.10,
    "Volatility Index (VIX)": -0.025
}

weights = {}
for asset in selected_assets:
    default_val = default_weights.get(asset, 0.0)
    weights[asset] = st.sidebar.number_input(
        f"Weight for {asset}",
        min_value=-1.0, max_value=1.0, value=float(default_val), step=0.05,
        format="%.2f"
    )

# --- DATA FETCH FUNCTION ---
@st.cache_data
def load_data(tickers, start, end):
    """Download adjusted close data for all tickers."""
    raw = yf.download(list(tickers.values()), start=start, end=end, progress=False)

    # Handle MultiIndex columns
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

# --- LOAD DATA ---
try:
    data = load_data(tickers, start_date, end_date)
except Exception:
    st.warning("⚠️ Error loading data. Please check ticker availability and date range.")
    st.stop()

# --- OPTIONAL: RESAMPLE TO BUSINESS DAYS ---
if use_business_days:
    data = data.asfreq('B')
    data = data.fillna(method='ffill')

# --- NORMALIZE TO 100 ---
if normalize_start:
    data = data / data.iloc[0] * 100

# --- COMPUTE MONEY FLOW CURVE ---
money_flow = pd.Series(0, index=data.index, name="Money Flow Curve")
for asset, w in weights.items():
    if asset in data.columns:
        money_flow += data[asset] * w

# --- SMOOTHED CURVE ---
money_flow_smooth = money_flow.rolling(smooth_window).mean()

# --- MOMENTUM (RATE OF CHANGE %) ---
money_flow_momentum = money_flow_smooth.pct_change() * 100
money_flow_momentum = money_flow_momentum.fillna(0)

# --- SENTIMENT GAUGE LOGIC ---
latest_momentum = money_flow_momentum.iloc[-1]
if latest_momentum > 0.2:
    sentiment = "🟢 **Risk-On/Bullish**"
    sentiment_color = "#16a34a"
elif latest_momentum < -0.2:
    sentiment = "🔴 **Risk-Off/Defensive**"
    sentiment_color = "#dc2626"
else:
    sentiment = "⚪ **Neutral**"
    sentiment_color = "#a3a3a3"

# --- SENTIMENT DISPLAY ---
st.markdown(f"""
<div style="padding:1.2em; border-radius:12px; text-align:center; background-color:{sentiment_color}; color:white; font-size:1.3em; font-weight:bold;">
{sentiment} — Current Money-flow Sentiment
</div>
""", unsafe_allow_html=True)

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
            color=alt.Color('Asset:N', legend=alt.Legend(
                title="Assets",
                orient='top-left',
                direction='vertical'
            )),
            tooltip=['Date:T', 'Asset:N', 'Value:Q']
        )
        .properties(
            title="Normalized Asset Prices (Indexed)",
            width='container',
            height=400
        )
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
- 🔗 **Correlation Shifts:** Changing relationships between assets highlight regime changes (e.g., BTC aligning with SPX).
""")

# --- PAIRWISE CORRELATION HEATMAP ---
st.sidebar.markdown("### Pairwise Correlation Heatmap")
corr_matrix = data.corr()

# Melt correlation matrix for Altair
corr_melt = corr_matrix.reset_index().melt('Ticker')
corr_melt.columns = ['Asset1', 'Asset2', 'Correlation']

heatmap = (
    alt.Chart(corr_melt)
    .mark_rect()
    .encode(
        x=alt.X('Asset1:N', title=None),
        y=alt.Y('Asset2:N', title=None),
        color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='redblue', domain=(-1,1))),
        tooltip=['Asset1', 'Asset2', 'Correlation']
    )
    .properties(title="🔥 Pairwise Asset Correlation Heatmap")
)

text = (
    alt.Chart(corr_melt)
    .mark_text(baseline='middle', align='center', fontSize=12, color='black')
    .encode(
        x='Asset1:N',
        y='Asset2:N',
        text=alt.Text('Correlation:Q', format=".2f")   # 2‑decimal format
    )
)
st.altair_chart(heatmap + text, use_container_width=True)

st.markdown("""
### 🔎 Heatmap Interpretation
- **Correlation close to +1:** Assets move together (e.g., SPX & EEM).  
- **Correlation close to -1:** Assets move opposite (e.g., Gold vs SPX).  
- **Near 0:** Assets are largely independent.  
""")

st.caption("Data sourced via Yahoo Finance • Updated dynamically")
