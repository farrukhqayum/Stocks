import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt
from datetime import datetime, timedelta

st.caption("Data sourced via Yahoo Finance • Updated dynamically")

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
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365*2))
end_date = st.sidebar.date_input("End Date", datetime.now())
smooth_window = st.sidebar.slider("Smoothing (days)", 5, 100, 40)
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

# Normalize weights 
abs_sum = sum(abs(w) for w in weights.values())
if abs_sum != 0:  # avoid division by zero
    weights = {k: (v / abs_sum) for k, v in weights.items()}
    
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
money_flow_s = money_flow.rolling(3).mean()
money_flow_smooth = money_flow.rolling(smooth_window).mean()

# GLOBAL ROLLING CORRELATION SETTINGS
cw_ = 21 
money_flow_s = money_flow_s.squeeze()  # Ensure Series for ALL uses

# --- MOMENTUM (RATE OF CHANGE %) ---
money_flow_momentum = money_flow_smooth.pct_change(periods=10) * 100
money_flow_momentum = money_flow_momentum.fillna(0)

# --- SENTIMENT GAUGE LOGIC ---
latest_momentum = money_flow_momentum.iloc[-1]
if 0 <= latest_momentum <= 10:
    sentiment = "🟢 **Risk-On/Bullish**"
    sentiment_color = "#16a34a"
elif latest_momentum >= 10:
    sentiment = "🟡 **Risk-off Peaking**"
    sentiment_color = "#ffcc00"
elif -10 <= latest_momentum < 0:
    sentiment = "🔴 **Risk-Off/Defensive**"
    sentiment_color = "#dc2626"
elif latest_momentum < -10:
    sentiment = "🟠 **R=isk-on Bottoming**"
    sentiment_color = "#ff8533"
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
    "Money Flow Curve": money_flow_s,
    "Smoothed Curve": money_flow_smooth,
    "Momentum": money_flow_momentum
}).dropna()

# --- MONEY FLOW CURVE CHART ---
df_plot['Global Money Flow'] = money_flow_smooth
mean_smooth = money_flow_smooth.mean()
base = alt.Chart(df_plot).encode(x='Date:T')
df_plot['Above'] = df_plot['Money Flow Curve'] > df_plot['Smoothed Curve']

base = alt.Chart(df_plot).encode(x='Date:T')

curve_chart = base.mark_line(color='#1f77b4', opacity=0.6).encode(
    y='Money Flow Curve:Q'
)

smooth_chart = base.mark_line(color='#d62728', size=2).encode(
    y='Smoothed Curve:Q'
)

fill_area = base.mark_area(opacity=0.17).encode(
    y='Money Flow Curve:Q',
    y2='Smoothed Curve:Q',
    color=alt.Color(
        'Above:N',
        scale=alt.Scale(domain=[True, False], range=['green', 'red']),
        legend=None   # 🚫 removes the legend
    )
)


final_chart = fill_area + curve_chart + smooth_chart
st.altair_chart(final_chart, use_container_width=True)

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
with st.expander("🧠 Interpretation"):
    st.markdown("""
    ### How to read the above chart.
    - 📈 **Rising Curve:** Capital flowing into *risk-on* assets → bullish market sentiment.  
    - 📉 **Falling Curve:** Money shifting to *safe* assets → defensive / risk-off tone.  
    - ⚖️ **Flat Curve:** Neutral or mixed capital rotation.  
    - 🟢 **Positive Momentum:** Acceleration of risk-on flows.  
    - 🔴 **Negative Momentum:** Acceleration of risk-off flows.  
    - 🔗 **Correlation Shifts:** Changing relationships between assets highlight regime changes (e.g., BTC aligning with SPX).
    """)

# --- CORRELATION ---
with st.expander(" 🧠 Correlation Matrix"):
    st.markdown("""
    ### 🧠 Correlation of multiple assets
    - Study the relationship between various assets.
    """)
    
    corr_matrix = data.corr()
    corr_matrix.index.name = 'Asset1'    
    corr_melt = corr_matrix.reset_index().melt(id_vars='Asset1', var_name='Asset2', value_name='Correlation')
    corr_melt = corr_melt[corr_melt['Asset1'] != corr_melt['Asset2']]
    
    # Heatmap
    heatmap = (
        alt.Chart(corr_melt)
        .mark_rect()
        .encode(
            x=alt.X('Asset1:N', title=None),
            y=alt.Y('Asset2:N', title=None),
            color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='redblue', domain=(-1, 1))),
            tooltip=['Asset1', 'Asset2', alt.Tooltip('Correlation:Q', format='.2f')]
        )
        .properties(title="🔥 Pairwise Asset Correlation Heatmap")
    )
    
    text = (
        alt.Chart(corr_melt)
        .mark_text(baseline='middle', align='center', fontSize=10, color='black')
        .encode(
            x='Asset1:N',
            y='Asset2:N',
            text=alt.Text('Correlation:Q', format=".2f")
        )
    )
    
    st.altair_chart(heatmap + text, use_container_width=True)
    
    with st.expander("🔎 Heatmap Interpretation"):
        st.markdown("""
        ### How to read the heatmap
        - **Correlation close to +1:** Assets move together (e.g., SPX & EEM).  
        - **Correlation close to -1:** Assets move opposite (e.g., Gold vs SPX).  
        - **Near 0:** Assets are largely independent.  
        """)

st.markdown("""
###  💹 📈  STUDY STOCKS WITH MONEY FLOW
- Provide the ticker and study its normalized graph in relation to global money flow.
- Use the left panel to choose and press ENTER.
""")
user_ticker = st.sidebar.text_input("Enter Stock Ticker to Analyze", value="TSLA")

raw = yf.download(user_ticker, start=start_date, end=end_date, progress=False)

# Extract 'Adj Close' or 'Close' safely
if isinstance(raw.columns, pd.MultiIndex):
    if 'Adj Close' in raw.columns.get_level_values(0):
        user_stock_data = raw['Adj Close'].copy()
    elif 'Close' in raw.columns.get_level_values(0):
        user_stock_data = raw['Close'].copy()
    else:
        raise ValueError("No 'Adj Close' or 'Close' data found in downloaded data.")
else:
    if 'Adj Close' in raw.columns:
        user_stock_data = raw['Adj Close'].copy()
    elif 'Close' in raw.columns:
        user_stock_data = raw['Close'].copy()
    else:
        raise ValueError("No 'Adj Close' or 'Close' data found in downloaded data.")

user_stock_data = user_stock_data.fillna(method='ffill')
smoothed = user_stock_data.rolling(window=5, min_periods=1).mean()
smoothed.iloc[-1] = user_stock_data.iloc[-1]  # Override last value

if normalize_start:
    smoothed = smoothed / smoothed.iloc[0] * 100

gf_single, stk_single = money_flow_s.align(smoothed.squeeze(), join='inner')
if len(gf_single) >= cw_:
    rolling_corr_single = gf_single.rolling(cw_, min_periods=cw_//2).corr(stk_single)
    latest_corr = round(rolling_corr_single.iloc[-1] * 100, 1)
else:
    latest_corr = float('nan')

# Create plotting data
combined_df = pd.DataFrame({
    "Date": gf_single.index,
    "Global Money Flow": gf_single,
    "Stock Price": stk_single
})

combined_long_df = combined_df.melt(
    id_vars='Date',
    value_vars=['Global Money Flow', 'Stock Price'],
    var_name='Series',
    value_name='Value'
)

correlation_text = alt.Chart(pd.DataFrame({'x':[0], 'y':[0]})).mark_text(
    align='center',
    baseline='top',
    fontSize=16,
    color='gray'
).encode(
    x=alt.value(400), 
    y=alt.value(10),
    text=alt.value(f'{cw_}D Correlation: {latest_corr:.1f}')
)

base = alt.Chart(combined_long_df).encode(x='Date:T')

color_scale = alt.Scale(
    domain=['Global Money Flow', 'Stock Price'],
    range=['#1f77b4', 'gray']
)

money_flow_line = base.mark_line(color='#1f77b4', opacity=0.4).encode(
    y=alt.Y('Value:Q', axis=alt.Axis(title='Global Money Flow', orient='left')),
    color=alt.Color('Series:N', scale=color_scale, legend=alt.Legend(orient='top-left'))
).transform_filter(
    alt.datum.Series == 'Global Money Flow'
)

stock_price_line = base.mark_line(opacity=0.4).encode(
    y=alt.Y('Value:Q', axis=alt.Axis(title=f'Normalized {user_ticker} Price', orient='right')),
    color=alt.Color('Series:N', scale=color_scale, legend=None)
).transform_filter(
    alt.datum.Series == 'Stock Price'
)

combined_chart = alt.layer(
    money_flow_line,
    stock_price_line
).resolve_scale(
    y='independent'
).properties(
    width=800,
    height=400,
    title='Stock Price vs Money Flow Smooth'
)

final_chart = combined_chart + correlation_text
st.altair_chart(final_chart, use_container_width=True)

tickers_input = st.text_input("Enter tickers separated by commas (min 5 required):", value="COIN, MSTR, XYZ, CRM, QCOM, AMD, SMCI, BABA, XPEV, NIO, U, INTC, SNAP, UNH")

ticker_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if len(ticker_list) < 5:
    st.error("Please enter at least 5 tickers.")
    st.stop()

all_tickers_dict = {t: t for t in ticker_list}
try:
    all_data = load_data(all_tickers_dict, start_date, end_date)
except Exception as e:
    st.error(f"Failed to load data for tickers: {e}")
    st.stop()

corr_results = []
for ticker in ticker_list:
    try:
        if ticker not in all_data.columns:
            st.warning(f"Ticker {ticker} data not found")
            corr_results.append({'Ticker': ticker, 'Correlation %': float('nan')})
            continue

        series = all_data[ticker].fillna(method='ffill')
        smoothed_multi = series.rolling(window=5, min_periods=1).mean()
        smoothed_multi.iloc[-1] = series.iloc[-1]  # Override last value
        
        if normalize_start and not smoothed_multi.isnull().all():
            smoothed_multi = smoothed_multi / smoothed_multi.iloc[0] * 100
            
        gf, stk = money_flow_s.align(smoothed_multi.squeeze(), join='inner')
        if len(gf) >= cw_ and gf.count() > 0 and stk.count() > 0:
            rolling_corr = gf.rolling(cw_, min_periods=cw_//2).corr(stk)
            latest_corr = round(rolling_corr.iloc[-1] * 100, 1)
            corr_results.append({'Ticker': ticker, 'Correlation %': latest_corr})
        else:
            corr_results.append({'Ticker': ticker, 'Correlation %': float('nan')})

    except Exception as e:
        st.warning(f"Error processing ticker {ticker}: {e}")
        corr_results.append({'Ticker': ticker, 'Correlation %': float('nan')})

corr_df = pd.DataFrame(corr_results).dropna()
corr_df = corr_df.sort_values('Correlation %')

st.markdown(f"### {cw_}D - Correlation with Global Money Flow")
st.markdown("""

- 60-100% correlation - Same direction.   
    - When everything is bullish, go with the strongs, follow the trend.
- 10-50% correlation - Directionless but could be in sideways or consolidation phase.
    - Use them the global money is bearish. Easy to follow trends like rectangle, triangles, flags etc.
- Below 0 or negative correlation - Opposite direction to the global money flow.
    - Use them when the global market is down, and these are gaining.

""")
st.dataframe(corr_df, use_container_width=False, height=500, width=300)

