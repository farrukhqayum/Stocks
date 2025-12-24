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

# ------------------ UI (UNCHANGED) ------------------

st.sidebar.header("⚙️ Settings")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365*3))
end_date = st.sidebar.date_input("End Date", datetime.now())
smooth_window = st.sidebar.slider("Smoothing (days)", 5, 100, 40)
z_score_window = st.sidebar.slider("Climax Z-Score Lookback (Days)", 20, 250, 90)
normalize_start = st.sidebar.checkbox("Normalize to 100 at start", value=True)
use_business_days = st.sidebar.checkbox("Remove weekend gaps (use business days only)", value=True)

# ------------------ HELPERS (NEW, MINIMAL) ------------------

def snap_to_last_trading_day(index, date):
    """Snap a user-selected date to last available trading day."""
    date = pd.Timestamp(date)
    valid_dates = index[index <= date]
    if len(valid_dates) == 0:
        return index.min()
    return valid_dates.max()

def winsorize(series, lower=0.02, upper=0.98):
    """Remove extreme spikes safely (VIX / shocks)."""
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)

def index_0_100(series):
    """Proper financial index scaling."""
    s = series.copy()
    return (s - s.min()) / (s.max() - s.min()) * 100

# ------------------ ASSETS (UNCHANGED) ------------------

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
weights = {k: v / abs_sum for k, v in weights.items()} if abs_sum else weights

# ------------------ DATA LOAD (UNCHANGED STRUCTURE) ------------------

@st.cache_data
def load_data(tickers, start, end):
    raw = yf.download(list(tickers.values()), start=start, end=end, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw['Adj Close'] if 'Adj Close' in raw else raw['Close']
    raw = raw.rename(columns={v: k for k, v in tickers.items()})
    return raw.dropna(how="all")

data = load_data(tickers, start_date - timedelta(days=5), end_date + timedelta(days=5))

if use_business_days:
    data = data.asfreq("B").ffill()

# --------- FIX 1: SNAP START/END DATES ---------

start_snap = snap_to_last_trading_day(data.index, start_date)
end_snap = snap_to_last_trading_day(data.index, end_date)
data = data.loc[start_snap:end_snap]

# --------- FIX 2: CLEAN RETURNS (SPIKE CONTROL) ---------

returns = data.pct_change().apply(winsorize)
indexed = (1 + returns).cumprod()

if normalize_start:
    indexed = indexed / indexed.iloc[0] * 100

# ------------------ GMF CONSTRUCTION ------------------

money_flow = pd.Series(0.0, index=indexed.index)
for asset, w in weights.items():
    if asset in indexed:
        money_flow += indexed[asset] * w

# Proper indexed curve (NO ZERO COLLAPSE)
money_flow = index_0_100(money_flow)
money_flow_smooth = money_flow.rolling(smooth_window, min_periods=1).mean()

# ------------------ MOMENTUM & Z ------------------

momentum = money_flow_smooth.pct_change(10) * 100
rolling_mean = money_flow_smooth.rolling(z_score_window).mean()
rolling_std = money_flow_smooth.rolling(z_score_window).std()
zscore = (money_flow_smooth - rolling_mean) / rolling_std
zscore = zscore.fillna(0)

# ------------------ PLOTS (UNCHANGED STYLE) ------------------

df_plot = pd.DataFrame({
    "Date": money_flow.index,
    "Money Flow Curve": money_flow,
    "Smoothed Curve": money_flow_smooth,
    "Momentum": momentum,
    "Z-Score": zscore
}).dropna()

base = alt.Chart(df_plot).encode(x="Date:T")

curve = base.mark_line(color="#1f77b4", opacity=0.6).encode(y="Money Flow Curve:Q")
smooth = base.mark_line(color="#d62728", size=2).encode(y="Smoothed Curve:Q")

st.markdown("### 🌊 GMF Curves")
st.altair_chart(curve + smooth, use_container_width=True)

st.markdown("### 📈 Momentum")
st.altair_chart(
    alt.Chart(df_plot).mark_bar().encode(
        x="Date:T", y="Momentum:Q",
        color=alt.condition(alt.datum.Momentum > 0, alt.value("green"), alt.value("red"))
    ),
    use_container_width=True
)

st.markdown("### Climax Zone Indicator (Z-Score)")
st.altair_chart(
    alt.Chart(df_plot).mark_area().encode(
        x="Date:T", y="Z-Score:Q",
        color=alt.condition(alt.datum["Z-Score"] > 0, alt.value("#1f77b4"), alt.value("#d62728"))
    ),
    use_container_width=True
        )
