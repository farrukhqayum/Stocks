import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Global Money Flow Curve",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.caption("Data sourced via Yahoo Finance • Updated dynamically")
st.title("🌍 Global Money Flow")
st.markdown("""
This app tracks capital flows between **risk-on** and **risk-off** assets 
to estimate global risk appetite. 
Includes BTC, S&P 500, Emerging Markets, Gold, US Dollar, Treasury Bonds, Oil, and VIX.
""")

st.sidebar.header("⚙️ Settings")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365*2))
end_date = st.sidebar.date_input("End Date", datetime.now())
smooth_window = st.sidebar.slider("Smoothing (days)", 5, 100, 40)
z_score_window = st.sidebar.slider("Climax Z-Score Lookback (Days)", 20, 250, 90)
normalize_start = st.sidebar.checkbox("Normalize to 100 at start", value=True)
use_business_days = st.sidebar.checkbox("Remove weekend gaps", value=True)

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

default_weights = {
    "Bitcoin (BTC)": 0.05,
    "S&P 500 (SPX)": 0.15,
    "Emerging Markets (EEM)": 0.15,
    "Crude Oil (CL)": 0.12,
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
    rename_map = {}
    for name, ticker in tickers.items():
        if ticker in df.columns:
            rename_map[ticker] = name
        elif name in df.columns:
            rename_map[name] = name
    df = df.rename(columns=rename_map)
    df = df.dropna(axis=1, how='all')
    return df

try:
    data = load_data(tickers, start_date, end_date)
    spx_raw = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
    if isinstance(spx_raw.columns, pd.MultiIndex) and 'Adj Close' in spx_raw.columns.get_level_values(0):
        spx_data = spx_raw['Adj Close'].squeeze()
    elif 'Adj Close' in spx_raw.columns:
        spx_data = spx_raw['Adj Close'].squeeze()
    else:
        spx_data = spx_raw['Close'].squeeze()
    spx_data.name = "S&P 500 (SPX)"
except Exception:
    st.warning("⚠️ Error loading data. Please check connection.")
    st.stop()

if use_business_days:
    data = data.asfreq('B').fillna(method='ffill')
    spx_data = spx_data.asfreq('B').fillna(method='ffill')

if normalize_start:
    data = data / data.iloc[0] * 100
    spx_data = spx_data / spx_data.iloc[0] * 100

money_flow = pd.Series(0, index=data.index, name="Money Flow Curve")
for asset, w in weights.items():
    if asset in data.columns:
        money_flow += data[asset] * w

money_flow = money_flow.clip(lower=0.01)
money_flow_s = money_flow.rolling(3).mean()
money_flow_smooth = money_flow.rolling(smooth_window).mean()
rolling_mean = money_flow_smooth.rolling(window=z_score_window).mean()
rolling_std = money_flow_smooth.rolling(window=z_score_window).std()
money_flow_zscore = ((money_flow_smooth - rolling_mean) / rolling_std).fillna(0)
cw_ = 60
money_flow_s = money_flow_s.squeeze()
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
{sentiment} — Current Money-flow Sentiment
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

base = alt.Chart(df_plot).encode(x='Date:T')
main_color_scale = alt.Scale(domain=['Money Flow (Fast)', 'Smoothed (Slow)'], range=['#1f77b4', '#d62728'])

curve_chart = base.mark_line(opacity=0.6).encode(
    y=alt.Y('Money Flow Curve:Q', title='Value'),
    color=alt.Color('Metric:N', scale=main_color_scale, 
                    legend=alt.Legend(orient='top-left', title=None, fillColor='white', padding=8, strokeColor='gray', cornerRadius=5)),
    tooltip=['Date:T', alt.Tooltip('Money Flow Curve:Q', format='.2f')]
).transform_calculate(Metric="'Money Flow (Fast)'")

smooth_chart = base.mark_line(size=2).encode(
    y=alt.Y('Smoothed Curve:Q'),
    color=alt.Color('Metric:N', scale=main_color_scale, legend=None),
    tooltip=['Date:T', alt.Tooltip('Smoothed Curve:Q', format='.2f')]
).transform_calculate(Metric="'Smoothed (Slow)'")

fill_area = base.mark_area(opacity=0.15).encode(
    y='Money Flow Curve:Q',
    y2='Smoothed Curve:Q',
    color=alt.Color('Above:N', scale=alt.Scale(domain=[True, False], range=['green', 'red']), legend=None)
)

st.markdown("### 🌊 Global Money Flow Curve & Crossover Signal")
st.altair_chart(fill_area + curve_chart + smooth_chart, use_container_width=True)

momentum_chart = alt.Chart(df_plot).mark_bar().encode(
    x='Date:T',
    y=alt.Y('Momentum:Q', title='Flow Momentum (%)'),
    color=alt.condition(alt.datum.Momentum > 0, alt.value('#2ca02c'), alt.value('#d62728')),
    tooltip=['Date:T', alt.Tooltip('Momentum:Q', format='.2f')]
).properties(title="📈 Money Flow Momentum")
st.altair_chart(momentum_chart, use_container_width=True)

zscore_chart = alt.Chart(df_plot).mark_area(opacity=0.6).encode(
    x='Date:T',
    y=alt.Y('Z-Score:Q', title='Z-Score'),
    color=alt.condition(alt.datum['Z-Score'] > 0, alt.value('#1f77b4'), alt.value('#d62728')),
    tooltip=['Date:T', alt.Tooltip('Z-Score:Q', format='.2f')]
).properties(title="Climax Zone Indicator (Z-Score)")
st.altair_chart(zscore_chart, use_container_width=True)

with st.expander("⚠️ Divergence Check: S&P 500 vs. Money Flow"):
    spx_aligned, momentum_aligned = spx_data.align(money_flow_momentum, join='inner')
    divergence_df = pd.DataFrame({'SPX': spx_aligned, 'Momentum': momentum_aligned}).dropna()
    lookback = 60
    if len(divergence_df) >= lookback:
        recent_spx = divergence_df['SPX'].iloc[-lookback:]
        recent_momentum = divergence_df['Momentum'].iloc[-lookback:]
        div_signal = "No significant divergence."
        div_color = "#a3a3a3"
        if (recent_spx.iloc[-1] >= (recent_spx.max() * 0.99)) and (recent_momentum.iloc[-1] < (recent_momentum.max() * 0.5)):
            div_signal = "🚨 BEARISH Divergence: SPX high, Money Flow weak."
            div_color = "#d62728"
        elif (recent_spx.iloc[-1] <= (recent_spx.min() * 1.01)) and (recent_momentum.iloc[-1] > (recent_momentum.min() * 1.5)):
            div_signal = "🟢 BULLISH Divergence: SPX low, Money Flow building."
            div_color = "#2ca02c"
        st.markdown(f"<div style='padding:1em; border-radius:8px; text-align:center; background-color:{div_color}; color:white;'>{div_signal}</div>", unsafe_allow_html=True)

with st.expander("📊 Show Underlying Assets"):
    data_melted = data.reset_index().melt("Date", var_name="Asset", value_name="Value")
    asset_chart = alt.Chart(data_melted).mark_line().encode(
        x='Date:T', y='Value:Q', color='Asset:N', tooltip=['Date:T', 'Asset:N', 'Value:Q']
    ).properties(height=400)
    st.altair_chart(asset_chart, use_container_width=True)

with st.expander(" 🧠 Correlation Matrix"):
    corr_matrix = data.corr().reset_index().melt(id_vars='index')
    heatmap = alt.Chart(corr_matrix).mark_rect().encode(
        x='index:N', y='variable:N', color=alt.Color('value:Q', scale=alt.Scale(scheme='redblue', domain=(-1, 1)))
    )
    st.altair_chart(heatmap, use_container_width=True)

st.markdown("### 💹 📈  STUDY STOCKS WITH MONEY FLOW")
user_ticker = st.sidebar.text_input("Enter Stock Ticker to Analyze", value="TSLA")
raw_u = yf.download(user_ticker, start=start_date, end=end_date, progress=False)
user_stock_data = raw_u['Adj Close'] if 'Adj Close' in raw_u.columns else raw_u['Close']
user_stock_data = user_stock_data.fillna(method='ffill').squeeze()
smoothed_u = user_stock_data.rolling(5).mean()
if normalize_start: smoothed_u = smoothed_u / smoothed_u.iloc[0] * 100

gf_single, stk_single = money_flow_s.align(smoothed_u, join='inner')
latest_corr = gf_single.rolling(cw_).corr(stk_single).iloc[-1] * 100

st.markdown(f"**Current {cw_}D Correlation with {user_ticker}:** **{latest_corr:.1f}%**")

combined_df = pd.DataFrame({"Date": gf_single.index, "Global Money Flow": gf_single, "Stock Price": stk_single})
long_df = combined_df.melt(id_vars='Date', var_name='Series', value_name='Value')

base_u = alt.Chart(long_df).encode(x='Date:T')
u_color_scale = alt.Scale(domain=['Global Money Flow', 'Stock Price'], range=['#1f77b4', 'gray'])

mf_line = base_u.mark_line(color='#1f77b4', opacity=0.4).encode(
    y=alt.Y('Value:Q', axis=alt.Axis(title='Money Flow')),
    color=alt.Color('Series:N', scale=u_color_scale, legend=alt.Legend(orient='top-left', title=None, fillColor='white'))
).transform_filter(alt.datum.Series == 'Global Money Flow')

stk_line = base_u.mark_line(opacity=0.4).encode(
    y=alt.Y('Value:Q', axis=alt.Axis(title='Stock Price', orient='right')),
    color=alt.Color('Series:N', scale=u_color_scale, legend=None)
).transform_filter(alt.datum.Series == 'Stock Price')

st.altair_chart(alt.layer(mf_line, stk_line).resolve_scale(y='independent'), use_container_width=True)

tickers_input = st.text_input("Enter tickers separated by commas:", value="COIN, MSTR, CRM, AMD, SMCI, BABA")
ticker_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
if len(ticker_list) >= 1:
    all_data = load_data({t: t for t in ticker_list}, start_date, end_date)
    corr_results = []
    for ticker in ticker_list:
        if ticker in all_data.columns:
            s = all_data[ticker].fillna(method='ffill')
            if normalize_start: s = s / s.iloc[0] * 100
            g, st_ = money_flow_s.align(s, join='inner')
            val = g.rolling(cw_).corr(st_).iloc[-1] * 100
            corr_results.append({'Ticker': ticker, 'Correlation %': round(val, 1)})
    st.dataframe(pd.DataFrame(corr_results).sort_values('Correlation %', ascending=False))
