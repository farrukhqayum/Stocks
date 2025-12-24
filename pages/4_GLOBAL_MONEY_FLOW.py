import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt
from datetime import datetime, timedelta
import numpy as np

st.caption("Data sourced via Yahoo Finance • Updated dynamically")

st.set_page_config(
    page_title="Global Money Flow Curve (GMF)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌍 Global Money Flow (GMF)")
st.markdown("""
This app tracks capital flows between **risk-on** and **risk-off** assets 
to estimate global risk appetite. 
Includes BTC, S&P 500, Emerging Markets, Gold, US Dollar, Treasury Bonds, Oil, and VIX.
""")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Settings")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365*3))
end_date = st.sidebar.date_input("End Date", datetime.now())
smooth_window = st.sidebar.slider("Smoothing (days)", 5, 100, 40)
z_score_window = st.sidebar.slider("Climax Z-Score Lookback (Days)", 20, 250, 90)
normalize_start = st.sidebar.checkbox("Normalize to 100 at start", value=True)
use_business_days = st.sidebar.checkbox("Remove weekend gaps (use business days only)", value=True)

# =========================
# ### FIX 1 — DATE SAFETY
# =========================
def snap_to_last_trading_day(index, d):
    d = pd.Timestamp(d)
    valid = index[index <= d]
    return valid.max() if len(valid) else index.min()

# =========================
# ### FIX 2 — SPIKE CONTROL
# =========================
def winsorize_returns(s, low=0.01, high=0.99):
    lo = s.quantile(low)
    hi = s.quantile(high)
    return s.clip(lo, hi)

# =========================
# ASSETS
# =========================
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

tickers = {a: default_tickers[a] for a in selected_assets}

st.sidebar.markdown("### Set Asset Weights (Positive=Risk-On, Negative=Risk-Off)")

default_weights = {
    "Bitcoin (BTC)": 0.05,
    "S&P 500 (SPX)": 0.15,
    "Emerging Markets (EEM)": 0.15,
    "Crude Oil (CL)": 0.15,
    "Gold (XAU)": -0.15,
    "US Dollar Index (DXY)": -0.12,
    "US 10Y Treasury (IEF)": -0.13,
    "Volatility Index (VIX)": -0.1
}

weights = {}
for asset in selected_assets:
    weights[asset] = st.sidebar.number_input(
        f"Weight for {asset}",
        min_value=-1.0, max_value=1.0,
        value=float(default_weights.get(asset, 0.0)),
        step=0.05
    )

# Normalize absolute weights
abs_sum = sum(abs(w) for w in weights.values())
if abs_sum:
    weights = {k: v / abs_sum for k, v in weights.items()}

# =========================
# DATA LOAD
# =========================
@st.cache_data
def load_data(tickers, start, end):
    raw = yf.download(list(tickers.values()), start=start, end=end, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw['Adj Close'] if 'Adj Close' in raw else raw['Close']
    raw = raw.rename(columns={v: k for k, v in tickers.items()})
    return raw.dropna(how="all")

data = load_data(tickers, start_date - timedelta(days=7), end_date + timedelta(days=7))

if use_business_days:
    data = data.asfreq("B").ffill()

# Snap dates safely
start_snap = snap_to_last_trading_day(data.index, start_date)
end_snap = snap_to_last_trading_day(data.index, end_date)
data = data.loc[start_snap:end_snap]

# =========================
# ### FIXED GMF CONSTRUCTION
# =========================
returns = data.pct_change().apply(winsorize_returns)

# ✅ CORRECT: aggregate weighted RETURNS
money_flow = pd.Series(0.0, index=returns.index)
for asset, w in weights.items():
    if asset in returns.columns:
        money_flow += returns[asset] * w

# Build index
money_flow = (1 + money_flow).cumprod()

if normalize_start:
    money_flow = money_flow / money_flow.iloc[0] * 100

money_flow_s = money_flow.rolling(3, min_periods=1).mean()
money_flow_smooth = money_flow.rolling(smooth_window, min_periods=1).mean()

# =========================
# Z-SCORE & MOMENTUM
# =========================
rolling_mean = money_flow_smooth.rolling(z_score_window).mean()
rolling_std = money_flow_smooth.rolling(z_score_window).std()
money_flow_zscore = ((money_flow_smooth - rolling_mean) / rolling_std).fillna(0)

money_flow_momentum = money_flow_smooth.pct_change(10) * 100
money_flow_momentum = money_flow_momentum.fillna(0)

# =========================
# PLOTS
# =========================
df_plot = pd.DataFrame({
    "Date": money_flow.index,
    "Money Flow Curve": money_flow_s,
    "Smoothed Curve": money_flow_smooth,
    "Momentum": money_flow_momentum,
    "Z-Score": money_flow_zscore
}).dropna()

base = alt.Chart(df_plot).encode(x='Date:T')

curve_chart = base.mark_line(color='#1f77b4', opacity=0.6).encode(
    y=alt.Y('Money Flow Curve:Q', title='Money Flow Curve')
)

smooth_chart = base.mark_line(color='#d62728', size=2).encode(
    y=alt.Y('Smoothed Curve:Q', title='Smoothed Curve')
)

st.markdown("### 🌊 GMF Curves")
st.altair_chart(curve_chart + smooth_chart, use_container_width=True)

st.markdown("### 📈 Money Flow Momentum")
st.altair_chart(
    alt.Chart(df_plot).mark_bar().encode(
        x='Date:T',
        y='Momentum:Q',
        color=alt.condition(
            alt.datum.Momentum > 0,
            alt.value('#2ca02c'),
            alt.value('#d62728')
        )
    ),
    use_container_width=True
)

st.markdown("### Climax Zone Indicator (Z-Score)")
st.altair_chart(
    alt.Chart(df_plot).mark_area(opacity=0.6).encode(
        x='Date:T',
        y='Z-Score:Q',
        color=alt.condition(
            alt.datum['Z-Score'] > 0,
            alt.value('#1f77b4'),
            alt.value('#d62728')
        )
    ),
    use_container_width=True
)
