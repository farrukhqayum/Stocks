import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Global Money Flow Curve (GMF)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌍 Global Money Flow (GMF)")
st.markdown("""
This app tracks capital flows between **risk-on** and **risk-off** assets 
to estimate global risk appetite. 
- **Risk-On Assets**: BTC, S&P 500, Emerging Markets, Oil
- **Risk-Off Assets**: Gold, US Dollar, Treasury Bonds, VIX (inverse)
- **GMF Index**: Composite of weighted asset returns showing capital rotation
""")

st.caption("Data sourced via Yahoo Finance • Updated dynamically")

st.sidebar.header("⚙️ Settings")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365*3))
end_date = st.sidebar.date_input("End Date", datetime.now())
smooth_window = st.sidebar.slider("Smoothing (days)", 5, 100, 40)
z_score_window = st.sidebar.slider("Climax Z-Score Lookback (Days)", 20, 250, 60)
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
    "Bitcoin (BTC)": 0.25,      # Strong risk-on
    "S&P 500 (SPX)": 0.25,      # Strong risk-on  
    "Emerging Markets (EEM)": 0.20,  # Risk-on
    "Crude Oil (CL)": 0.10,     # Risk-on
    "Gold (XAU)": -0.20,        # Risk-off
    "US Dollar Index (DXY)": -0.20,  # Risk-off
    "US 10Y Treasury (IEF)": -0.20,  # Risk-off
    "Volatility Index (VIX)": -0.20  # Risk-off (inverse)
}

weights = {}
for asset in selected_assets:
    default_val = default_weights.get(asset, 0.0)
    weights[asset] = st.sidebar.number_input(
        f"Weight for {asset}",
        min_value=-1.0, max_value=1.0, value=float(default_val), step=0.05,
        format="%.2f"
    )

# Show weight sum
weight_sum = sum(weights.values())
st.sidebar.markdown("---")
st.sidebar.metric("Sum of Weights", f"{weight_sum:.3f}")
if abs(weight_sum) < 0.1:
    st.sidebar.warning("⚠️ Weights sum near zero - index may show little variation")

def load_data(tickers, start, end):
    """Load data from Yahoo Finance"""
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

# Load data
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

except Exception as e:
    st.warning(f"⚠️ Error loading data: {e}. Please check ticker availability and date range.")
    st.stop()

if use_business_days:
    data = data.asfreq('B')
    data = data.fillna(method='ffill')
    spx_data = spx_data.asfreq('B')
    spx_data = spx_data.fillna(method='ffill')

# FIXED: GMF Index Construction - Use daily changes, not cumulative returns
def calculate_gmf_index(data, weights):
    """
    Calculate GMF Index as weighted sum of daily percentage changes
    This keeps the index centered around 0 with reasonable volatility
    """
    # Calculate daily percentage changes for each asset
    daily_pct = data.pct_change().fillna(0)
    
    # Apply weights (re-index to match data columns)
    weights_series = pd.Series(weights).reindex(data.columns).fillna(0.0)
    
    # Calculate weighted daily changes
    weighted_daily = daily_pct.multiply(weights_series, axis=1)
    
    # Sum across assets to get daily GMF change
    daily_gmf_change = weighted_daily.sum(axis=1)
    
    # Convert to index starting at 0, then scale to reasonable range
    # Using cumulative sum of normalized changes
    gmf_index = (daily_gmf_change * 100).cumsum()  # Scale up for visibility
    
    return gmf_index

# Calculate GMF Index
gmf_raw = calculate_gmf_index(data, weights)
gmf_index = gmf_raw - gmf_raw.iloc[0]  # Start at 0 for cleaner visualization

# Create smoothed versions
money_flow_raw = gmf_index
money_flow_s = money_flow_raw.rolling(3, min_periods=1).mean()
money_flow_smooth = money_flow_raw.rolling(smooth_window, min_periods=1).mean()

# Calculate Z-Score (using the SMOOTHED series)
rolling_mean = money_flow_smooth.rolling(window=z_score_window, min_periods=5).mean()
rolling_std = money_flow_smooth.rolling(window=z_score_window, min_periods=5).std()
money_flow_zscore = (money_flow_smooth - rolling_mean) / rolling_std
money_flow_zscore = money_flow_zscore.replace([np.inf, -np.inf], 0).fillna(0)

# FIXED: Calculate Momentum properly (rate of change over 30 days)
money_flow_momentum = money_flow_smooth.diff(30) / 30 * 100  # Percentage change per day annualized
money_flow_momentum = money_flow_momentum.fillna(0)

# Get latest values
latest_momentum = money_flow_momentum.iloc[-1] if not money_flow_momentum.empty else 0
latest_zscore = money_flow_zscore.iloc[-1] if not money_flow_zscore.empty else 0

# FIXED: Updated Sentiment Logic with reasonable thresholds
Z_EXTREME = 1.5      # Reduced from 1.8
MOM_HIGH = 0.5       # 0.5% daily change is strong (annualized ~125%)
MOM_LOW = -0.5       # -0.5% daily change is strong negative
Z_NEUTRAL_UPPER = 0.8
Z_NEUTRAL_LOWER = -0.8

# Sentiment determination
if latest_zscore >= Z_EXTREME:
    if latest_momentum > 0:
        sentiment = "🚨 **EXTREME OVERBOUGHT (Euphoria Climax)**"
        sentiment_color = "#ff6b6b"
    else:
        sentiment = "⚠️ **OVERBOUGHT but Losing Momentum**"
        sentiment_color = "#ffa726"
        
elif latest_zscore <= -Z_EXTREME:
    if latest_momentum < 0:
        sentiment = "📉 **EXTREME OVERSOLD (Panic/Capitulation)**"
        sentiment_color = "#5d4037"
    else:
        sentiment = "🔄 **OVERSOLD but Recovering**"
        sentiment_color = "#42a5f5"
        
elif latest_momentum > MOM_HIGH:
    if latest_zscore > 0:
        sentiment = "🚀 **STRONG RISK-ON (Accelerating Higher)**"
        sentiment_color = "#4caf50"
    else:
        sentiment = "🟢 **RISK-ON (Recovering from Lows)**"
        sentiment_color = "#66bb6a"
        
elif latest_momentum < MOM_LOW:
    if latest_zscore < 0:
        sentiment = "🔻 **STRONG RISK-OFF (Accelerating Lower)**"
        sentiment_color = "#f44336"
    else:
        sentiment = "🔴 **RISK-OFF (Pulling Back from Highs)**"
        sentiment_color = "#ef5350"
        
elif latest_momentum > 0:
    if latest_zscore > Z_NEUTRAL_UPPER:
        sentiment = "🟢 **Risk-On (Above Average)**"
        sentiment_color = "#81c784"
    elif latest_zscore < Z_NEUTRAL_LOWER:
        sentiment = "🟡 **Cautiously Recovering (From Oversold)**"
        sentiment_color = "#ffd54f"
    else:
        sentiment = "⚪ **Mildly Risk-On (Neutral Zone)**"
        sentiment_color = "#bdbdbd"
        
elif latest_momentum < 0:
    if latest_zscore < Z_NEUTRAL_LOWER:
        sentiment = "🔴 **Risk-Off (Below Average)**"
        sentiment_color = "#e57373"
    elif latest_zscore > Z_NEUTRAL_UPPER:
        sentiment = "🟠 **Correcting (From Overbought)**"
        sentiment_color = "#ffb74d"
    else:
        sentiment = "⚫ **Mildly Risk-Off (Neutral Zone)**"
        sentiment_color = "#757575"
        
else:
    sentiment = "⚪ **NEUTRAL / SIDEWAYS**"
    sentiment_color = "#9e9e9e"

# Display Index Info
st.markdown(f"""
**GMF Index Construction:**  
`Daily GMF = Σ (Asset_Daily_Return × Weight)`  
`GMF Index = Cumulative Sum of Daily GMF × 100`  
- Positive values: Net risk-on flows  
- Negative values: Net risk-off flows  
- Current weight sum = **{weight_sum:.3f}**
""")

st.markdown(f"""
<div style="padding:1.2em; border-radius:12px; text-align:center; background-color:{sentiment_color}; color:white; font-size:1.3em; font-weight:bold;">
{sentiment}
</div>
""", unsafe_allow_html=True)

# Display metrics
col1, col2, col3 = st.columns(3)
with col1:
    current_gmf = money_flow_raw.iloc[-1] if not money_flow_raw.empty else 0
    st.metric("Current GMF Index", f"{current_gmf:+.2f}")
with col2:
    st.metric("Z-Score", f"{latest_zscore:+.2f}", 
              delta="Extreme" if abs(latest_zscore) > Z_EXTREME else "Normal")
with col3:
    st.metric("30-Day Momentum", f"{latest_momentum:+.3f}%/day",
              delta="Accelerating" if abs(latest_momentum) > MOM_HIGH else "Stable")

# Prepare data for plotting
df_plot = pd.DataFrame({
    "Date": money_flow_raw.index,
    "Money Flow Curve": money_flow_s,
    "Smoothed Curve": money_flow_smooth,
    "Momentum": money_flow_momentum,
    "Z-Score": money_flow_zscore
}).dropna()

df_plot['Above'] = df_plot['Money Flow Curve'] > df_plot['Smoothed Curve']

# Create GMF Chart with zero line
st.markdown("### 🌊 GMF Curves")
base = alt.Chart(df_plot).encode(x='Date:T')

# Add zero line
zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='gray', strokeDash=[3, 3]).encode(y='y')

curve_chart = base.mark_line(color='#1f77b4', opacity=0.6).encode(
    y=alt.Y('Money Flow Curve:Q', title='GMF Index'),
    tooltip=['Date:T', alt.Tooltip('Money Flow Curve:Q', format='.2f')]
)

smooth_chart = base.mark_line(color='#d62728', size=2).encode(
    y=alt.Y('Smoothed Curve:Q', title='GMF Index'),
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

final_chart = alt.layer(zero_line, fill_area, curve_chart, smooth_chart)
st.altair_chart(final_chart, use_container_width=True)

# Momentum Chart
st.markdown("### 📈 GMF Momentum (30-Day Rate of Change)")
momentum_chart = (
    alt.Chart(df_plot)
    .mark_bar(opacity=0.5)
    .encode(
        x='Date:T',
        y=alt.Y('Momentum:Q', title='Daily Rate of Change (%)'),
        color=alt.condition(
            alt.datum.Momentum > 0,
            alt.value('#2ca02c'),
            alt.value('#d62728')
        ),
        tooltip=['Date:T', alt.Tooltip('Momentum:Q', format='.3f')]
    )
)

# Add momentum threshold lines
mom_threshold_lines = alt.Chart(pd.DataFrame({'y': [MOM_LOW, 0, MOM_HIGH]})).mark_rule(
    color='gray', strokeDash=[3, 3]
).encode(y='y')

final_momentum_chart = (momentum_chart + mom_threshold_lines)
st.altair_chart(final_momentum_chart, use_container_width=True)

# Z-Score Chart
st.markdown("### 📊 Climax Zone Indicator (Z-Score)")
zscore_chart = (
    alt.Chart(df_plot)
    .mark_area(opacity=0.6)
    .encode(
        x='Date:T',
        y=alt.Y('Z-Score:Q', title='Z-Score'),
        color=alt.condition(
            alt.datum['Z-Score'] > 0,
            alt.value('#1f77b4'), 
            alt.value('#d62728')  
        ),
        tooltip=['Date:T', alt.Tooltip('Z-Score:Q', format='.2f')]
    )
)

# Add Z-Score threshold lines
z_threshold_lines = alt.Chart(pd.DataFrame({'y': [-Z_EXTREME, -0.5, 0, 0.5, Z_EXTREME]})).mark_rule(
    color='gray', strokeDash=[3, 3]
).encode(y='y')

final_zscore_chart = (zscore_chart + z_threshold_lines).properties(height=300)
st.altair_chart(final_zscore_chart, use_container_width=True)

# Divergence Check
with st.expander("⚠️ Divergence Check: S&P 500 vs. GMF Momentum"):
    # Align S&P 500 with GMF momentum
    spx_pct = spx_data.pct_change().fillna(0) * 100  # Convert to percentage
    spx_aligned, gmf_aligned = spx_pct.align(money_flow_momentum, join='inner')
    
    divergence_df = pd.DataFrame({
        'SPX_Return': spx_aligned, 
        'GMF_Momentum': gmf_aligned,
    }).dropna().sort_index()

    lookback = 60
    if len(divergence_df) >= lookback:
        # Calculate rolling correlation
        rolling_corr = spx_aligned.rolling(lookback).corr(gmf_aligned)
        latest_corr = rolling_corr.iloc[-1] if not rolling_corr.empty else 0
        
        if latest_corr < -0.5:
            divergence_signal = "🚨 **STRONG NEGATIVE CORRELATION**: SPX and GMF moving opposite directions"
            signal_color = "#d62728"
        elif latest_corr > 0.7:
            divergence_signal = "🟢 **STRONG POSITIVE CORRELATION**: SPX and GMF moving together"
            signal_color = "#2ca02c"
        else:
            divergence_signal = "⚪ **MODERATE CORRELATION**: No strong divergence detected"
            signal_color = "#a3a3a3"
        
        st.markdown(f"""
        <div style="padding:1em; border-radius:8px; text-align:center; background-color:{signal_color}; color:white;">
        **{divergence_signal}** (Correlation: {latest_corr:.2f})
        </div>
        """, unsafe_allow_html=True)
        
        # Plot correlation over time
        corr_plot_df = pd.DataFrame({
            'Date': rolling_corr.index,
            'Correlation': rolling_corr
        }).dropna()
        
        if not corr_plot_df.empty:
            corr_chart = alt.Chart(corr_plot_df).mark_line().encode(
                x='Date:T',
                y=alt.Y('Correlation:Q', scale=alt.Scale(domain=[-1, 1])),
                tooltip=['Date:T', alt.Tooltip('Correlation:Q', format='.2f')]
            ).properties(title=f"{lookback}-Day Rolling Correlation")
            
            corr_zero = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='gray').encode(y='y')
            st.altair_chart(corr_chart + corr_zero, use_container_width=True)
    else:
        st.info(f"Not enough data for a {lookback}-day divergence check.")

# Underlying Assets
with st.expander("📊 Show Underlying Asset Returns"):
    # Calculate daily returns for each asset
    asset_returns = data.pct_change().fillna(0) * 100
    
    # Apply weights to show contribution
    weights_series = pd.Series(weights).reindex(asset_returns.columns).fillna(0)
    weighted_returns = asset_returns.multiply(weights_series, axis=1)
    
    # Calculate cumulative contribution
    cumulative_contrib = weighted_returns.cumsum()
    
    cumulative_melted = cumulative_contrib.reset_index().melt("Date", var_name="Asset", value_name="Cumulative Contribution")
    
    asset_chart = (
        alt.Chart(cumulative_melted)
        .mark_area(opacity=0.6)
        .encode(
            x='Date:T',
            y='Cumulative Contribution:Q',
            color='Asset:N',
            tooltip=['Date:T', 'Asset:N', alt.Tooltip('Cumulative Contribution:Q', format='.2f')]
        )
        .properties(
            title="Cumulative Contribution of Each Asset to GMF Index",
            width='container',
            height=400
        )
    )
    st.altair_chart(asset_chart, use_container_width=True)

# Correlation Matrix
with st.expander("🧠 Asset Correlation Matrix"):
    # Calculate correlations of daily returns
    returns_corr = data.pct_change().corr()
    returns_corr.index.name = 'Asset1'    
    corr_melt = returns_corr.reset_index().melt(id_vars='Asset1', var_name='Asset2', value_name='Correlation')
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
        .properties(title="Daily Return Correlation Heatmap")
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

# Single Stock Analysis
# ... [previous code remains the same until the Single Stock Analysis section] ...

# Single Stock Analysis
st.markdown("""
### 💹 Stock Correlation Analysis
Enter a stock ticker to analyze its correlation with Global Money Flow.
""")
user_ticker = st.sidebar.text_input("Enter Stock Ticker to Analyze", value="TSLA")

# Load single stock data
try:
    raw = yf.download(user_ticker, start=start_date, end=end_date, progress=False)
    
    if isinstance(raw.columns, pd.MultiIndex):
        if 'Adj Close' in raw.columns.get_level_values(0):
            user_stock_data = raw['Adj Close'].copy()
        elif 'Close' in raw.columns.get_level_values(0):
            user_stock_data = raw['Close'].copy()
        else:
            st.error("No 'Adj Close' or 'Close' data found.")
            st.stop()
    else:
        if 'Adj Close' in raw.columns:
            user_stock_data = raw['Adj Close'].copy()
        elif 'Close' in raw.columns:
            user_stock_data = raw['Close'].copy()
        else:
            st.error("No 'Adj Close' or 'Close' data found.")
            st.stop()
            
    user_stock_data = user_stock_data.fillna(method='ffill')
    user_stock_data = user_stock_data.squeeze()  # Ensure it's a Series
    
except Exception as e:
    st.error(f"Failed to load data for {user_ticker}: {e}")
    st.stop()

# Calculate correlation with GMF
# FIXED: Ensure both are Series and properly aligned
stock_returns = user_stock_data.pct_change().fillna(0) * 100
gmf_returns = money_flow_raw.diff().fillna(0)  # Daily GMF changes

# Ensure both are Series
stock_returns = stock_returns.squeeze() if isinstance(stock_returns, pd.DataFrame) else stock_returns
gmf_returns = gmf_returns.squeeze() if isinstance(gmf_returns, pd.DataFrame) else gmf_returns

# Align data - FIXED: Use index intersection
common_index = stock_returns.index.intersection(gmf_returns.index)
if len(common_index) == 0:
    st.warning(f"No overlapping data between {user_ticker} and GMF index")
    stock_aligned = pd.Series(dtype=float)
    gmf_aligned = pd.Series(dtype=float)
else:
    stock_aligned = stock_returns.loc[common_index]
    gmf_aligned = gmf_returns.loc[common_index]

# Calculate rolling correlation
corr_window = 60
if len(stock_aligned) >= corr_window and len(gmf_aligned) >= corr_window:
    # Ensure we have enough non-NaN data
    valid_data = pd.DataFrame({
        'stock': stock_aligned,
        'gmf': gmf_aligned
    }).dropna()
    
    if len(valid_data) >= corr_window:
        rolling_corr = valid_data['stock'].rolling(corr_window).corr(valid_data['gmf'])
        latest_corr = rolling_corr.iloc[-1] if not rolling_corr.empty and not pd.isna(rolling_corr.iloc[-1]) else 0
    else:
        rolling_corr = pd.Series(dtype=float, index=stock_aligned.index)
        latest_corr = 0
else:
    rolling_corr = pd.Series(dtype=float, index=stock_aligned.index)
    latest_corr = 0

# Display correlation info
col1, col2 = st.columns(2)
with col1:
    corr_display = f"{latest_corr:.2f}" if not pd.isna(latest_corr) else "N/A"
    corr_label = "Strong" if abs(latest_corr) > 0.5 else "Weak" if not pd.isna(latest_corr) else "Insufficient Data"
    st.metric(f"{user_ticker} - GMF Correlation", 
              corr_display,
              delta=corr_label)

with col2:
    # Calculate performance relative to GMF
    if len(stock_aligned) > 0 and len(gmf_aligned) > 0 and not stock_aligned.empty and not gmf_aligned.empty:
        # Calculate cumulative returns
        stock_cum = (1 + stock_aligned/100).cumprod()
        gmf_cum = (1 + gmf_aligned/100).cumprod()
        
        # Get the last valid values
        if not stock_cum.empty and not gmf_cum.empty:
            stock_final = stock_cum.iloc[-1] - 1
            gmf_final = gmf_cum.iloc[-1] - 1
            relative_perf = (stock_final - gmf_final) * 100
            st.metric("Relative Performance", f"{relative_perf:.1f}%")
        else:
            st.metric("Relative Performance", "N/A")
    else:
        st.metric("Relative Performance", "N/A")

# Plot correlation over time
if not rolling_corr.empty and rolling_corr.notna().any():
    corr_plot_df = pd.DataFrame({
        'Date': rolling_corr.index,
        'Correlation': rolling_corr
    }).dropna()
    
    if not corr_plot_df.empty:
        corr_chart = alt.Chart(corr_plot_df).mark_line(color='purple').encode(
            x='Date:T',
            y=alt.Y('Correlation:Q', scale=alt.Scale(domain=[-1, 1])),
            tooltip=['Date:T', alt.Tooltip('Correlation:Q', format='.2f')]
        ).properties(
            title=f"{user_ticker} - {corr_window}D Rolling Correlation with GMF",
            height=200
        )
        
        corr_zero = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='gray').encode(y='y')
        st.altair_chart(corr_chart + corr_zero, use_container_width=True)
    else:
        st.info("Insufficient data to plot correlation history.")
else:
    st.info(f"Need at least {corr_window} days of overlapping data to calculate correlation.")

# Multi-ticker Analysis
st.markdown("""
### 📊 Multi-Stock Correlation Analysis
Enter multiple tickers to compare their correlation with Global Money Flow.
""")

tickers_input = st.text_input("Enter tickers separated by commas (min 3 required):", 
                             value="AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, JPM, JNJ, V, PG")

ticker_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if len(ticker_list) >= 3:
    # Load multi-ticker data
    all_tickers_dict = {t: t for t in ticker_list}
    try:
        all_data = load_data(all_tickers_dict, start_date, end_date)
        
        # Calculate correlations
        corr_results = []
        for ticker in ticker_list:
            try:
                if ticker not in all_data.columns:
                    corr_results.append({'Ticker': ticker, 'Correlation %': float('nan')})
                    continue

                # Get stock returns
                stock_data = all_data[ticker].fillna(method='ffill')
                stock_ret = stock_data.pct_change().fillna(0) * 100
                
                # Align with GMF returns
                common_idx = stock_ret.index.intersection(gmf_returns.index)
                if len(common_idx) >= corr_window:
                    stock_aligned_multi = stock_ret.loc[common_idx]
                    gmf_aligned_multi = gmf_returns.loc[common_idx]
                    
                    # Calculate rolling correlation
                    valid_data = pd.DataFrame({
                        'stock': stock_aligned_multi,
                        'gmf': gmf_aligned_multi
                    }).dropna()
                    
                    if len(valid_data) >= corr_window:
                        rolling_corr_multi = valid_data['stock'].rolling(corr_window).corr(valid_data['gmf'])
                        if not rolling_corr_multi.empty:
                            latest_corr_multi = rolling_corr_multi.iloc[-1]
                            if not pd.isna(latest_corr_multi):
                                corr_results.append({
                                    'Ticker': ticker, 
                                    'Correlation %': round(latest_corr_multi * 100, 1)
                                })
                                continue
                
                corr_results.append({'Ticker': ticker, 'Correlation %': float('nan')})

            except Exception:
                corr_results.append({'Ticker': ticker, 'Correlation %': float('nan')})

        # Display correlation table
        corr_df = pd.DataFrame(corr_results)
        corr_df = corr_df.sort_values('Correlation %', na_position='last', ascending=False)

        if not corr_df.empty:
            st.markdown(f"### {corr_window}D - Correlation with Global Money Flow")
            
            # Color coding function
            def color_corr(val):
                if pd.isna(val):
                    return 'color: gray'
                elif val >= 50:
                    return 'color: green; font-weight: bold'
                elif val <= -50:
                    return 'color: red; font-weight: bold'
                elif val >= 20:
                    return 'color: lightgreen'
                elif val <= -20:
                    return 'color: lightcoral'
                else:
                    return 'color: black'
            
            # Display with styling
            styled_df = corr_df.style.map(color_corr, subset=['Correlation %'])
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Interpretation
            st.markdown("""
            **Correlation Interpretation:**
            - **> 50%**: Strong positive correlation with risk flows
            - **20-50%**: Moderate positive correlation  
            - **-20 to 20%**: Weak or no correlation
            - **-50 to -20%**: Moderate negative correlation
            - **< -50%**: Strong negative correlation (hedge/defensive)
            """)
        else:
            st.warning("No correlation data available. Check ticker validity and date range.")
            
    except Exception as e:
        st.error(f"Failed to load data for tickers: {e}")
else:
    st.info("Enter at least 3 tickers separated by commas to analyze.")
