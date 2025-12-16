import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt
from datetime import datetime, timedelta

st.caption("Data sourced via Yahoo Finance • Updated dynamically")

st.set_page_config(
    page_title="Global Money Flow Curve",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
            raise ValueError("No 'Adj Close' or 'Close' data found.")
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
    
    # --- START of SPX Data Extraction Fix ---
    if isinstance(spx_raw.columns, pd.MultiIndex) and 'Adj Close' in spx_raw.columns.get_level_values(0):
        spx_data = spx_raw['Adj Close'].squeeze()
    elif 'Adj Close' in spx_raw.columns:
        spx_data = spx_raw['Adj Close'].squeeze()
    else:
        spx_data = spx_raw['Close'].squeeze()
    # --- END of SPX Data Extraction Fix ---
        
    spx_data.name = "S&P 500 (SPX)"

except Exception:
    st.warning("⚠️ Error loading data. Please check ticker availability and date range.")
    st.stop()

if use_business_days:
    data = data.asfreq('B')
    data = data.fillna(method='ffill')
    spx_data = spx_data.asfreq('B')
    spx_data = spx_data.fillna(method='ffill')

if normalize_start:
    data = data / data.iloc[0] * 100
    spx_data = spx_data / spx_data.iloc[0] * 100

money_flow = pd.Series(0, index=data.index, name="Money Flow Curve")
for asset, w in weights.items():
    if asset in data.columns:
        money_flow += data[asset] * w

money_flow_s = money_flow.rolling(3).mean()
money_flow_smooth = money_flow.rolling(smooth_window).mean()

rolling_mean = money_flow_smooth.rolling(window=z_score_window).mean()
rolling_std = money_flow_smooth.rolling(window=z_score_window).std()
money_flow_zscore = (money_flow_smooth - rolling_mean) / rolling_std
money_flow_zscore = money_flow_zscore.fillna(0)

cw_ = 21 
money_flow_s = money_flow_s.squeeze()

money_flow_momentum = money_flow_smooth.pct_change(periods=10) * 100
money_flow_momentum = money_flow_momentum.fillna(0)

latest_momentum = money_flow_momentum.iloc[-1]
latest_zscore = money_flow_zscore.iloc[-1]

if latest_zscore >= 2.0:
    sentiment = "🚨 **EXTREME RISK-ON (OVERBOUGHT)**"
    sentiment_color = "#ff8533" 
elif latest_zscore <= -2.0:
    sentiment = "📉 **PANIC/CAPITULATION (OVERSOLD)**"
    sentiment_color = "#990000" 
elif 0 <= latest_momentum <= 10:
    sentiment = "🟢 **Risk-On/Bullish**"
    sentiment_color = "#16a34a"
elif latest_momentum > 10:
    sentiment = "🟡 **Strong Risk-On Acceleration**"
    sentiment_color = "#ffcc00"
elif -10 <= latest_momentum < 0:
    sentiment = "🔴 **Risk-Off/Defensive**"
    sentiment_color = "#dc2626"
elif latest_momentum < -10:
    sentiment = "🟠 **Risk-Off Acceleration/Climax**"
    sentiment_color = "#ff8533"
else:
    sentiment = "⚪ **Neutral**"
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
    "Z-Score": money_flow_zscore
}).dropna()

df_plot['Global Money Flow'] = money_flow_smooth
base = alt.Chart(df_plot).encode(x='Date:T')
df_plot['Above'] = df_plot['Money Flow Curve'] > df_plot['Smoothed Curve']

base = alt.Chart(df_plot).encode(x='Date:T')

curve_chart = base.mark_line(color='#1f77b4', opacity=0.6).encode(
    y=alt.Y('Money Flow Curve:Q', title='Money Flow Curve (Fast)'),
    tooltip=['Date:T', alt.Tooltip('Money Flow Curve:Q', format='.2f')]
)

smooth_chart = base.mark_line(color='#d62728', size=2).encode(
    y=alt.Y('Smoothed Curve:Q', title='Smoothed Curve (Slow)'),
    tooltip=['Date:T', alt.Tooltip('Smoothed Curve:Q', format='.2f')]
)

fill_area = base.mark_area(opacity=0.17).encode(
    y='Money Flow Curve:Q',
    y2='Smoothed Curve:Q',
    color=alt.Color(
        'Above:N',
        scale=alt.Scale(domain=[True, False], range=['green', 'red']),
        legend=None
    )
)

final_chart = fill_area + curve_chart + smooth_chart
st.markdown("### 🌊 Global Money Flow Curve & Crossover Signal")
st.altair_chart(final_chart, use_container_width=True)

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
        tooltip=['Date:T', alt.Tooltip('Momentum:Q', format='.2f')]
    )
)

climax_lines_m = alt.Chart(pd.DataFrame({'y': [-10, 10]})).mark_rule(color='gray', strokeDash=[3, 3]).encode(y='y')

final_momentum_chart = (momentum_chart + climax_lines_m).properties(title="📈 Money Flow Momentum (Rate of Change %)")
st.altair_chart(final_momentum_chart, use_container_width=True)

st.markdown("### Climax Zone Indicator (Money Flow Z-Score)")
zscore_chart = (
    alt.Chart(df_plot)
    .mark_area(opacity=0.6)
    .encode(
        x='Date:T',
        y=alt.Y('Z-Score:Q', title='Money Flow Z-Score'),
        color=alt.condition(
            alt.datum['Z-Score'] > 0,
            alt.value('#1f77b4'), 
            alt.value('#d62728')  
        ),
        tooltip=['Date:T', alt.Tooltip('Z-Score:Q', format='.2f')]
    )
)

climax_lines_z = alt.Chart(pd.DataFrame({'y': [-2.0, 2.0]})).mark_rule(color='gray', strokeDash=[3, 3]).encode(y='y')

final_zscore_chart = (zscore_chart + climax_lines_z).properties(
    title="Climax Zone Indicator (Z-Score of Smoothed Money Flow)"
)
st.altair_chart(final_zscore_chart, use_container_width=True)

with st.expander("⚠️ Divergence Check: S&P 500 vs. Money Flow Momentum"):
    
    # --- START of Error Fix Application ---
    # Ensure spx_data and money_flow_momentum are properly aligned Series before combining
    spx_aligned, momentum_aligned = spx_data.align(money_flow_momentum, join='inner')
    
    divergence_df = pd.DataFrame({
        'SPX': spx_aligned, 
        'Money Flow Momentum': momentum_aligned,
    }).dropna().sort_index()
    # --- END of Error Fix Application ---

    lookback = 60
    if len(divergence_df) >= lookback:
        recent_spx = divergence_df['SPX'].iloc[-lookback:]
        recent_momentum = divergence_df['Money Flow Momentum'].iloc[-lookback:]

        divergence_signal = "No significant divergence detected."
        signal_color = "#a3a3a3"

        spx_recent_max = recent_spx.max()
        mom_at_spx_max = recent_momentum[recent_spx.idxmax()] 
        mom_recent_max = recent_momentum.max()
        
        spx_recent_min = recent_spx.min()
        mom_at_spx_min = recent_momentum[recent_spx.idxmin()]
        mom_recent_min = recent_momentum.min()

        if (recent_spx.iloc[-1] >= (spx_recent_max * 0.99)) and (recent_momentum.iloc[-1] < (mom_recent_max * 0.5)) and (mom_recent_max > 5):
            divergence_signal = "🚨 **Potential BEARISH Divergence:** SPX near high, but Money Flow Momentum is weak. (Risk-Off Warning)"
            signal_color = "#d62728"
        elif (recent_spx.iloc[-1] <= (spx_recent_min * 1.01)) and (recent_momentum.iloc[-1] > (mom_recent_min * 1.5)) and (mom_recent_min < -5):
            divergence_signal = "🟢 **Potential BULLISH Divergence:** SPX near low, but Money Flow Momentum is building. (Risk-On Buy Signal)"
            signal_color = "#2ca02c"
        
        st.markdown(f"""
        <div style="padding:1em; border-radius:8px; text-align:center; background-color:{signal_color}; color:white;">
        **{divergence_signal}**
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info(f"Not enough data for a {lookback}-day divergence check.")

    divergence_plot_df = divergence_df.reset_index().rename(columns={'index': 'Date'}).melt(
        id_vars='Date',
        var_name='Series',
        value_name='Value'
    )
    
    momentum_normalized = (divergence_df['Money Flow Momentum'] - divergence_df['Money Flow Momentum'].min()) / (divergence_df['Money Flow Momentum'].max() - divergence_df['Money Flow Momentum'].min()) * (divergence_df['SPX'].max() - divergence_df['SPX'].min()) + divergence_df['SPX'].min()
    momentum_normalized.name = "Momentum (Rescaled)"
    
    dual_plot_df = pd.DataFrame({
        "Date": divergence_df.index,
        "SPX": divergence_df['SPX'],
        "Momentum (Rescaled)": momentum_normalized
    }).melt(id_vars='Date', var_name='Series', value_name='Value')

    base_divergence = alt.Chart(dual_plot_df).encode(x='Date:T')
    
    spx_line = base_divergence.mark_line(color='#1f77b4', opacity=0.8).encode(
        y=alt.Y('Value:Q', axis=alt.Axis(title='S&P 500 (Normalized)', orient='left')),
        tooltip=['Date:T', alt.Tooltip('SPX:Q', format='.2f')]
    ).transform_filter(alt.datum.Series == 'SPX')

    momentum_line = base_divergence.mark_line(color='#d62728', opacity=0.8, strokeDash=[5, 5]).encode(
        y=alt.Y('Value:Q', axis=alt.Axis(title='Momentum (Rescaled for Visual)', orient='right')),
        tooltip=['Date:T', alt.Tooltip('Momentum (Rescaled):Q', format='.2f')]
    ).transform_filter(alt.datum.Series == 'Momentum (Rescaled)')

    divergence_chart = alt.layer(spx_line, momentum_line).resolve_scale(y='independent').properties(title="S&P 500 Price vs. Money Flow Momentum (Rescaled)")
    st.altair_chart(divergence_chart, use_container_width=True)

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
            tooltip=['Date:T', 'Asset:N', alt.Tooltip('Value:Q', format='.2f')]
        )
        .properties(
            title="Normalized Asset Prices (Indexed)",
            width='container',
            height=400
        )
    )
    st.altair_chart(asset_chart, use_container_width=True)

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
    st.markdown("""
    ### 🧠 Correlation of multiple assets
    - Study the relationship between various assets.
    """)
    
    corr_matrix = data.corr()
    corr_matrix.index.name = 'Asset1'    
    corr_melt = corr_matrix.reset_index().melt(id_vars='Asset1', var_name='Asset2', value_name='Correlation')
    corr_melt = corr_melt[corr_melt['Asset1'] != corr_melt['Asset2']]
    
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

if isinstance(raw.columns, pd.MultiIndex):
    if 'Adj Close' in raw.columns.get_level_values(0):
        user_stock_data = raw['Adj Close'].copy()
    elif 'Close' in raw.columns.get_level_values(0):
        user_stock_data = raw['Close'].copy()
    else:
        st.error("No 'Adj Close' or 'Close' data found in downloaded data for single stock.")
        st.stop()
else:
    if 'Adj Close' in raw.columns:
        user_stock_data = raw['Adj Close'].copy()
    elif 'Close' in raw.columns:
        user_stock_data = raw['Close'].copy()
    else:
        st.error("No 'Adj Close' or 'Close' data found in downloaded data for single stock.")
        st.stop()

user_stock_data = user_stock_data.fillna(method='ffill')
smoothed = user_stock_data.rolling(window=5, min_periods=1).mean()
smoothed.iloc[-1] = user_stock_data.iloc[-1]

if normalize_start:
    smoothed = smoothed / smoothed.iloc[0] * 100

combined_df = pd.DataFrame({
    "Date": gf_single.index,
    "Global Money Flow": gf_single,
    "Stock Price": stk_single
})

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
    title=f'{user_ticker} Price vs Money Flow Smooth'
)

final_chart_single = combined_chart + correlation_text
st.altair_chart(final_chart_single, use_container_width=True)

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
            corr_results.append({'Ticker': ticker, 'Correlation %': float('nan')})
            continue

        series = all_data[ticker].fillna(method='ffill')
        smoothed_multi = series.rolling(window=5, min_periods=1).mean()
        smoothed_multi.iloc[-1] = series.iloc[-1] 
        
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
        corr_results.append({'Ticker': ticker, 'Correlation %': float('nan')})

corr_df = pd.DataFrame(corr_results).dropna()
corr_df = corr_df.sort_values('Correlation %')

st.markdown(f"### {cw_}D - Correlation with Global Money Flow")
st.markdown("""
- **60–100% correlation — Same direction** - When the overall market is bullish, align with the strongest assets and follow the trend.
- **10–50% correlation — Weak or sideways relationship** - Often indicates consolidation or range‑bound phases.  
    - Useful when global money flow is bearish, as these patterns are easier to trade (rectangles, triangles, flags, etc.).
- **Below 0 (negative correlation) — Opposite direction** - These assets move against global money flow.  
    - Favor them when the broader market is declining, since they tend to gain in such conditions.
""")
st.dataframe(corr_df, use_container_width=False, height=500, width=300)
