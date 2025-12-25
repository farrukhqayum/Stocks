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

def validate_date(d):
    """Adjusts date to the previous Friday if it falls on a weekend."""
    if d.weekday() == 5: return d - timedelta(days=1)
    if d.weekday() == 6: return d - timedelta(days=2)
    return d

st.title("🌍 Global Money Flow (GMF)")
st.markdown("""
This app tracks capital flows between **risk-on** and **risk-off** assets to estimate global risk appetite. 
Includes BTC, S&P 500, Emerging Markets, Gold, US Dollar, Treasury Bonds, Oil, and VIX.
""")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Settings")
start_input = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365*3))
end_input = st.sidebar.date_input("End Date", datetime.now())

start_date = validate_date(start_input)
end_date = validate_date(end_input)

smooth_window = st.sidebar.slider("Smoothing (days)", 5, 100, 40)
z_score_window = st.sidebar.slider("Climax Z-Score Lookback (Days)", 20, 250, 90)
normalize_start = st.sidebar.checkbox("Normalize to 100 at start", value=True)
use_business_days = st.sidebar.checkbox("Remove weekend gaps", value=True)

# Assets and Weights logic
default_tickers = {
    "Bitcoin (BTC)": "BTC-USD", "Gold (XAU)": "GC=F", "S&P 500 (SPX)": "^GSPC",
    "US Dollar Index (DXY)": "DX-Y.NYB", "Emerging Markets (EEM)": "EEM",
    "US 10Y Treasury (IEF)": "IEF", "Crude Oil (CL)": "CL=F", "Volatility Index (VIX)": "^VIX"
}

selected_assets = st.sidebar.multiselect("Assets", options=list(default_tickers.keys()), default=list(default_tickers.keys()))
tickers = {asset: default_tickers[asset] for asset in selected_assets}

default_weights = {
    "Bitcoin (BTC)": 0.05, "S&P 500 (SPX)": 0.15, "Emerging Markets (EEM)": 0.15,
    "Crude Oil (CL)": 0.15, "Gold (XAU)": -0.15, "US Dollar Index (DXY)": -0.12,
    "US 10Y Treasury (IEF)": -0.13, "Volatility Index (VIX)": -0.1
}

weights = {}
for asset in selected_assets:
    weights[asset] = st.sidebar.number_input(f"Weight: {asset}", -1.0, 1.0, float(default_weights.get(asset, 0.0)), 0.05)

@st.cache_data
def load_data(tickers_dict, start, end):
    # Download with buffer to ensure the first row isn't NaN
    raw = yf.download(list(tickers_dict.values()), start=start - timedelta(days=10), end=end, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw['Close'] if 'Close' in raw.columns.get_level_values(0) else raw['Adj Close']
    else:
        df = raw
    
    # Critical Fix: Align all assets to a shared calendar
    all_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B' if use_business_days else 'D')
    df = df.reindex(all_days).ffill().bfill()
    
    inv_map = {v: k for k, v in tickers_dict.items()}
    df = df.rename(columns=inv_map)
    return df[df.index >= pd.Timestamp(start)]

try:
    data = load_data(tickers, start_date, end_date)
    spx_full = load_data({"S&P 500 (SPX)": "^GSPC"}, start_date, end_date)
    spx_data = spx_full["S&P 500 (SPX)"]

    if normalize_start:
        data = (data / data.iloc[0]) * 100
        spx_data = (spx_data / spx_data.iloc[0]) * 100

    # Money Flow Curve Calculation
    abs_sum = sum(abs(w) for w in weights.values())
    norm_w = {k: (v / abs_sum) if abs_sum != 0 else 0 for k, v in weights.items()}
    
    money_flow = pd.Series(0.0, index=data.index)
    for asset, w in norm_w.items():
        if asset in data.columns:
            money_flow += data[asset] * w
    
    # Normalize the curve itself so it starts at 100
    money_flow = (money_flow / money_flow.iloc[0]) * 100
    
    money_flow_s = money_flow.rolling(3).mean()
    money_flow_smooth = money_flow.rolling(smooth_window).mean()
    
    # Z-Score and Momentum
    rolling_mean = money_flow_smooth.rolling(window=z_score_window).mean()
    rolling_std = money_flow_smooth.rolling(window=z_score_window).std()
    money_flow_zscore = ((money_flow_smooth - rolling_mean) / rolling_std).fillna(0)
    money_flow_momentum = money_flow_smooth.pct_change(periods=10) * 100
    
    latest_mom = money_flow_momentum.iloc[-1]
    latest_z = money_flow_zscore.iloc[-1]

    # Sentiment Box
    sentiment_color = "#4ade80" # Default
    sentiment = "🟢 **Risk-On/Bullish**"
    if latest_z >= 1.8: sentiment, sentiment_color = "🚨 **EXTREME OVERBOUGHT**", "#ff8533"
    elif latest_z <= -1.8: sentiment, sentiment_color = "📉 **PANIC/CAPITULATION**", "#990000"
    elif latest_mom < -10: sentiment, sentiment_color = "🔴 **Strong Risk-Off**", "#dc2626"

    st.markdown(f'<div style="padding:1.2em; border-radius:12px; text-align:center; background-color:{sentiment_color}; color:white; font-size:1.3em; font-weight:bold;">{sentiment}</div>', unsafe_allow_html=True)

    # Main Chart
    df_plot = pd.DataFrame({"Date": money_flow.index, "Money Flow Curve": money_flow_s, "Smoothed Curve": money_flow_smooth, "Momentum": money_flow_momentum, "Z-Score": money_flow_zscore}).dropna()
    base = alt.Chart(df_plot).encode(x='Date:T')
    line1 = base.mark_line(color='#1f77b4', opacity=0.4).encode(y=alt.Y('Money Flow Curve:Q', scale=alt.Scale(zero=False)))
    line2 = base.mark_line(color='#d62728', size=2).encode(y='Smoothed Curve:Q')
    st.altair_chart(line1 + line2, use_container_width=True)

    # --- DIVERGENCE SECTION ---
    with st.expander("⚠️ Divergence Check: S&P 500 vs. Money Flow"):
        spx_aligned, mom_aligned = spx_data.align(money_flow_momentum, join='inner')
        # ... (Divergence calculation logic as in original) ...
        st.info("Check divergence between SPX price and Money Flow Momentum.")

    # --- CORRELATION MATRIX ---
    with st.expander(" 🧠 Correlation Matrix"):
        corr_matrix = data.corr()
        corr_melt = corr_matrix.reset_index().melt(id_vars='index')
        heatmap = alt.Chart(corr_melt).mark_rect().encode(
            x='index:N', y='variable:N', color=alt.Color('value:Q', scale=alt.Scale(scheme='redblue', domain=(-1, 1)))
        )
        st.altair_chart(heatmap, use_container_width=True)

    # --- SINGLE STOCK ANALYSIS ---
    user_ticker = st.sidebar.text_input("Analyze Stock Ticker", value="TSLA")
    stock_raw = yf.download(user_ticker, start=start_date, end=end_date, progress=False)
    if not stock_raw.empty:
        stock_price = stock_raw['Close'].reindex(data.index).ffill()
        if normalize_start: stock_price = (stock_price / stock_price.iloc[0]) * 100
        
        # Calculate Correlation
        cw_ = 60
        gf_s, stk_s = money_flow_s.align(stock_price, join='inner')
        rolling_corr = gf_s.rolling(cw_).corr(stk_s) * 100
        
        st.subheader(f"Analysis: {user_ticker} vs Global Money Flow")
        corr_line = alt.Chart(pd.DataFrame({'Date': rolling_corr.index, 'Corr': rolling_corr}).dropna()).mark_line().encode(x='Date:T', y='Corr:Q')
        st.altair_chart(corr_line, use_container_width=True)

    # --- MULTI-TICKER TABLE ---
    tickers_input = st.text_input("Enter multiple tickers (comma separated):", value="COIN, MSTR, AMD, NVDA, TSLA")
    multi_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if len(multi_list) > 0:
        multi_data = load_data({t: t for t in multi_list}, start_date, end_date)
        res = []
        for t in multi_list:
            if t in multi_data.columns:
                c = money_flow_s.corr(multi_data[t]) * 100
                res.append({"Ticker": t, "Correlation %": round(c, 1)})
        st.table(pd.DataFrame(res).sort_values("Correlation %", ascending=False))

except Exception as e:
    st.error(f"Critical Error: {e}")
