import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt
from datetime import datetime, timedelta

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
Includes BTC, S&P 500, Emerging Markets, Gold, US Dollar, Treasury Bonds, Oil, and VIX.
""")

st.sidebar.header("⚙️ Settings")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365*3))
end_date = st.sidebar.date_input("End Date", datetime.now())
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
    default_val = default_weights.get(asset, 0.0)
    weights[asset] = st.sidebar.number_input(
        f"Weight for {asset}",
        min_value=-1.0, max_value=1.0, value=float(default_val), step=0.05,
        format="%.2f"
    )

abs_sum = sum(abs(w) for w in weights.values())
if abs_sum != 0:
    weights = {k: (v / abs_sum) for k, v in weights.items()}

@st.cache_data
def load_data(tickers, start, end):
    raw = yf.download(list(tickers.values()), start=start, end=end, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        if 'Adj Close' in raw.columns.get_level_values(0):
            df = raw['Adj Close'].copy()
        elif 'Close' in raw.columns.get_level_values(0):
            df = raw['Close'].copy()
        else:
            raise ValueError("No close data found.")
    else:
        df = raw.copy()
    rename_map = {ticker: name for name, ticker in tickers.items()}
    df = df.rename(columns=rename_map)
    return df.dropna(axis=1, how='all')

try:
    data = load_data(tickers, start_date, end_date)
    spx_raw = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
    if isinstance(spx_raw.columns, pd.MultiIndex):
        spx_data = spx_raw['Adj Close' if 'Adj Close' in spx_raw.columns.get_level_values(0) else 'Close'].squeeze()
    else:
        spx_data = spx_raw['Adj Close' if 'Adj Close' in spx_raw.columns else 'Close'].squeeze()
    spx_data.name = "S&P 500 (SPX)"
except Exception:
    st.warning("⚠️ Error loading data.")
    st.stop()

if use_business_days:
    data = data.asfreq('B').fillna(method='ffill')
    spx_data = spx_data.asfreq('B').fillna(method='ffill')

if normalize_start:
    data = data / data.iloc[0] * 100
    spx_data = spx_data / spx_data.iloc[0] * 100

money_flow = pd.Series(0, index=data.index)
for asset, w in weights.items():
    if asset in data.columns:
        money_flow += data[asset] * w

money_flow = money_flow.clip(lower=0.01)
money_flow_s = money_flow.rolling(3).mean().squeeze()
money_flow_smooth = money_flow.rolling(smooth_window).mean()
money_flow_zscore = ((money_flow_smooth - money_flow_smooth.rolling(z_score_window).mean()) / money_flow_smooth.rolling(z_score_window).std()).fillna(0)
money_flow_momentum = (money_flow_smooth.pct_change(periods=10) * 100).fillna(0)

latest_momentum = money_flow_momentum.iloc[-1]
latest_zscore = money_flow_zscore.iloc[-1]

Z_EXTREME = 1.8
MOM_HIGH = 10.0
MOM_LOW = -10.0
Z_NEUTRAL_UPPER = 0.5
Z_NEUTRAL_LOWER = -0.5

if latest_zscore >= Z_EXTREME:
    sentiment = "🚨 **EXTREME OVERBOUGHT (Euphoria Climax)**"
    sentiment_color = "#ff8533" 
elif latest_zscore <= -Z_EXTREME:
    sentiment = "📉 **PANIC/CAPITULATION (Oversold Climax)**"
    sentiment_color = "#990000"
elif latest_momentum > MOM_HIGH:
    if latest_zscore >= Z_NEUTRAL_UPPER:
        sentiment = "🟡 **Strong Risk-On: ACCELERATION into STRETCHED ZONE**"
        sentiment_color = "#ffcc00"
    else:
        sentiment = "🟢 **Strong Risk-On: ACCELERATION into NORMAL ZONE**"
        sentiment_color = "#2ca02c"
elif latest_momentum < MOM_LOW:
    if latest_zscore <= Z_NEUTRAL_LOWER:
        sentiment = "🟠 **Strong Risk-Off: DEEPER PULLBACK/DECELERATION**"
        sentiment_color = "#ff8533"
    else:
        sentiment = "🔴 **Strong Risk-Off: ACCELERATION out of NORMAL ZONE**"
        sentiment_color = "#dc2626"
elif latest_momentum >= 0:
    if latest_zscore >= Z_NEUTRAL_UPPER:
        sentiment = "🟢 **Risk-On/Bullish (STRETCHED but HOLDING)**"
        sentiment_color = "#16a34a" 
    else:
        sentiment = "🟢 **Risk-On/Bullish (NORMAL ZONE)**"
        sentiment_color = "#4ade80" 
elif latest_momentum < 0:
    if latest_zscore <= Z_NEUTRAL_LOWER:
        sentiment = "🔴 **Risk-Off/Defensive (OVERSOLD but DECELERATING)**"
        sentiment_color = "#dc2626" 
    else:
        sentiment = "🔴 **Risk-Off/Defensive (NORMAL ZONE pullback)**"
        sentiment_color = "#f87171" 
else:
    sentiment = "⚪ **Neutral/Choppy Market**"
    sentiment_color = "#a3a3a3"

st.markdown(f"""
<div style="padding:1.2em; border-radius:12px; text-align:center; background-color:{sentiment_color}; color:white; font-size:1.3em; font-weight:bold;">
{sentiment}
</div>
""", unsafe_allow_html=True)

df_plot = pd.DataFrame({
    "Date": money_flow.index,
    "Money Flow Curve": money_flow_s,
    "Smoothed Curve": money_flow_smooth,
    "Momentum": money_flow_momentum,
    "Z-Score": money_flow_zscore,
    "Above": money_flow_s > money_flow_smooth
}).dropna()

align_cfg = {"axisLeft": {"minExtent": 60}, "axisRight": {"minExtent": 60}}
base = alt.Chart(df_plot).encode(x='Date:T')

c1 = base.mark_line(color='#1f77b4', opacity=0.6).encode(
    y=alt.Y('Money Flow Curve:Q', title='Money Flow Curve (Fast)'),
    tooltip=['Date:T', alt.Tooltip('Money Flow Curve:Q', format='.2f')]
)

c2 = base.mark_line(color='#d62728', size=2).encode(
    y=alt.Y('Smoothed Curve:Q', title='Smoothed Curve (Slow)'),
    tooltip=['Date:T', alt.Tooltip('Smoothed Curve:Q', format='.2f')]
)

f1 = base.mark_area(opacity=0.17).encode(
    y='Money Flow Curve:Q',
    y2='Smoothed Curve:Q',
    color=alt.Color('Above:N', scale=alt.Scale(domain=[True, False], range=['green', 'red']), legend=None)
)

st.markdown("### 🌊 GMF Curves")
chart_main = (f1 + c1 + c2).properties(height=400).configure(
    axisLeft=alt.AxisConfig(minExtent=60),
    axisRight=alt.AxisConfig(minExtent=60)
)
st.altair_chart(chart_main, use_container_width=True)

mom_chart = alt.Chart(df_plot).mark_bar().encode(
    x='Date:T',
    y=alt.Y('Momentum:Q', title='Flow Momentum (%)'),
    color=alt.condition(alt.datum.Momentum > 0, alt.value('#2ca02c'), alt.value('#d62728')),
    tooltip=['Date:T', alt.Tooltip('Momentum:Q', format='.2f')]
).properties(height=200, title="📈 Money Flow Momentum (%)").configure(
    axisLeft=alt.AxisConfig(minExtent=60),
    axisRight=alt.AxisConfig(minExtent=60)
)
st.altair_chart(mom_chart, use_container_width=True)

st.markdown("### Climax Zone Indicator (Z-Score)")
z_chart = alt.Chart(df_plot).mark_area(opacity=0.6).encode(
    x='Date:T',
    y=alt.Y('Z-Score:Q', title='Money Flow Z-Score'),
    color=alt.condition(alt.datum['Z-Score'] > 0, alt.value('#1f77b4'), alt.value('#d62728')),
    tooltip=['Date:T', alt.Tooltip('Z-Score:Q', format='.2f')]
).properties(height=200, title="Climax Zone Indicator (Z-Score of Smoothed Money Flow)").configure(
    axisLeft=alt.AxisConfig(minExtent=60),
    axisRight=alt.AxisConfig(minExtent=60)
)
st.altair_chart(z_chart, use_container_width=True)

with st.expander("⚠️ Divergence Check: S&P 500 vs. Money Flow Momentum"):
    spx_aligned, mom_aligned = spx_data.align(money_flow_momentum, join='inner')
    div_df = pd.DataFrame({'SPX': spx_aligned, 'Mom': mom_aligned}).reset_index().rename(columns={'index':'Date'})
    d_base = alt.Chart(div_df).encode(x='Date:T')
    l1 = d_base.mark_line(color='#1f77b4').encode(y=alt.Y('SPX:Q', axis=alt.Axis(title='S&P 500', orient='left')))
    l2 = d_base.mark_line(color='#d62728', strokeDash=[5,5]).encode(y=alt.Y('Mom:Q', axis=alt.Axis(title='Momentum', orient='right')))
    chart_div = alt.layer(l1, l2).resolve_scale(y='independent').properties(height=400).configure(
        axisLeft=alt.AxisConfig(minExtent=60),
        axisRight=alt.AxisConfig(minExtent=60)
    )
    st.altair_chart(chart_div, use_container_width=True)

with st.expander("📊 Show Underlying Assets"):
    assets_df = data.reset_index().melt("Date", var_name="Asset", value_name="Value")
    a_chart = alt.Chart(assets_df).mark_line().encode(
        x='Date:T', y='Value:Q', color=alt.Color('Asset:N', legend=alt.Legend(orient='top-left'))
    ).properties(height=400, title="Normalized Asset Prices (Indexed)").configure(
        axisLeft=alt.AxisConfig(minExtent=60),
        axisRight=alt.AxisConfig(minExtent=60)
    )
    st.altair_chart(a_chart, use_container_width=True)

with st.expander("🧠 Interpretation"):
    st.markdown("""
    ### How to read the above chart.
    - 📈 **Rising Curve:** Capital flowing into *risk-on* assets → bullish market sentiment.  
    - 📉 **Falling Curve:** Money shifting to *safe* assets → defensive / risk-off tone.  
    - ⚖️ **Flat Curve:** Neutral or mixed capital rotation.  
    - 🟢 **Positive Momentum:** Acceleration of risk-on flows.  
    - 🔴 **Negative Momentum:** Acceleration of risk-off flows.  
    - **Climax Z-Score ($\mathbf{> 2.0}$ or $\mathbf{< -2.0}$):** Signals historically extreme overbought (euphoria) or oversold (panic) conditions, often leading to a reversal.
    - **Divergence:** Price action in S&P 500 not supported by money flow momentum. This is a powerful leading signal.
    """)

with st.expander(" 🧠 Correlation Matrix"):
    st.markdown("### 🧠 Correlation of multiple assets")
    corr_matrix = data.corr()
    corr_matrix.index.name = 'index'
    corr_melt = corr_matrix.reset_index().melt(id_vars='index')
    heatmap = alt.Chart(corr_melt).mark_rect().encode(
        x='index:N', y='variable:N', color=alt.Color('value:Q', scale=alt.Scale(scheme='redblue', domain=(-1, 1)))
    ).properties(title="🔥 Pairwise Asset Correlation Heatmap")
    st.altair_chart(heatmap, use_container_width=True)

st.markdown("### 💹 📈 STUDY STOCKS WITH MONEY FLOW")
user_ticker = st.sidebar.text_input("Enter Stock Ticker to Analyze", value="TSLA")
u_raw = yf.download(user_ticker, start=start_date, end=end_date, progress=False)
u_data = (u_raw['Adj Close'] if 'Adj Close' in u_raw.columns else u_raw['Close']).fillna(method='ffill').squeeze()
if normalize_start: u_data = u_data / u_data.iloc[0] * 100

gf_s, stk_s = money_flow_s.align(u_data, join='inner')
comb_df = pd.DataFrame({"Date": gf_s.index, "GMF": gf_s, "Stock": stk_s})
u_base = alt.Chart(comb_df).encode(x='Date:T')
ul1 = u_base.mark_line(color='#1f77b4', opacity=0.4).encode(y=alt.Y('GMF:Q', axis=alt.Axis(title='Global Money Flow', orient='left')))
ul2 = u_base.mark_line(color='gray', opacity=0.4).encode(y=alt.Y('Stock:Q', axis=alt.Axis(title=f'Normalized {user_ticker} Price', orient='right')))

chart_stock = alt.layer(ul1, ul2).resolve_scale(y='independent').properties(
    height=400, title=f'{user_ticker} Price vs Money Flow Smooth'
).configure(
    axisLeft=alt.AxisConfig(minExtent=60),
    axisRight=alt.AxisConfig(minExtent=60)
)
st.altair_chart(chart_stock, use_container_width=True)

tickers_input = st.text_input("Enter tickers separated by commas (min 5 required):", value="COIN, MSTR, XYZ, CRM, QCOM, AMD, SMCI, BABA, XPEV, NIO, U, INTC, SNAP, UNH")
ticker_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
if len(ticker_list) >= 5:
    all_data = load_data({t: t for t in ticker_list}, start_date, end_date)
    corr_results = []
    for t in ticker_list:
        if t in all_data.columns:
            s = all_data[t].fillna(method='ffill')
            if normalize_start: s = s / s.iloc[0] * 100
            g, st_ = money_flow_s.align(s, join='inner')
            val = g.rolling(60).corr(st_).iloc[-1] * 100
            corr_results.append({'Ticker': t, 'Correlation %': round(val, 1)})
    st.dataframe(pd.DataFrame(corr_results).sort_values('Correlation %'), width=300)
