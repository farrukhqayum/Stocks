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
    "Crude Oil (CL)": 0.20,     # Risk-on
    "Gold (XAU)": -0.20,        # Risk-off
    "US Dollar Index (DXY)": -0.20,  # Risk-off
    "US 10Y Treasury (IEF)": -0.20,  # Risk-off
    "Volatility Index (VIX)": -0.07  # Risk-off (inverse)
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
    
# Helper function for correlation interpretation
def get_correlation_interpretation(corr_value):
    """Provide interpretation of correlation value"""
    if pd.isna(corr_value):
        return "Insufficient data for correlation analysis."
    
    corr_value = corr_value  # Already in percentage
    
    if corr_value >= 70:
        return "**Strong Positive Correlation**: Stock moves strongly with global risk appetite. When GMF rises, this stock tends to rise even more."
    elif corr_value >= 40:
        return "**Moderate Positive Correlation**: Stock generally moves with global risk flows but may diverge at times."
    elif corr_value >= 10:
        return "**Weak Positive Correlation**: Some relationship with global risk flows, but other factors dominate."
    elif corr_value > -10:
        return "**No Significant Correlation**: Stock price movements are largely independent of global risk flows."
    elif corr_value >= -40:
        return "**Weak Negative Correlation**: Stock shows some tendency to move opposite to risk flows."
    elif corr_value >= -70:
        return "**Moderate Negative Correlation**: Stock acts as partial hedge - tends to rise when risk appetite falls."
    else:
        return "**Strong Negative Correlation**: Stock is a strong hedge/defensive asset. Tends to rise significantly when risk appetite falls."

# Single Stock Analysis
st.markdown("""
### 💹 Single Stock Correlation Analysis
Enter a stock ticker to analyze its correlation with Global Money Flow.
""")

# Get user ticker from sidebar
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

# Smooth the stock data (5-day smoothing)
user_stock_smoothed = user_stock_data.rolling(window=5, min_periods=1).mean()
user_stock_smoothed.iloc[-1] = user_stock_data.iloc[-1]  # Keep latest value actual

# Align GMF and stock data
# Use the smoothed GMF (money_flow_s) for correlation calculations
gf_single = money_flow_s  # This is the 3-day smoothed GMF
stk_single = user_stock_smoothed

# Align the series
gf_aligned, stk_aligned = gf_single.align(stk_single, join='inner')

# Calculate 60-day rolling correlation
cw_ = 60  # Correlation window (60 days)
latest_corr = float('nan')

if len(gf_aligned) >= cw_:
    rolling_corr_single = gf_aligned.rolling(cw_, min_periods=cw_//2).corr(stk_aligned)
    if not rolling_corr_single.empty:
        latest_corr = rolling_corr_single.iloc[-1]
        if pd.notna(latest_corr):
            latest_corr = round(latest_corr * 100, 1)
        else:
            latest_corr = float('nan')
    
    rolling_corr_df = pd.DataFrame({
        "Date": rolling_corr_single.index,
        "Correlation": rolling_corr_single * 100  # Convert to percentage
    }).dropna()
else:
    rolling_corr_df = pd.DataFrame({"Date": [], "Correlation": []})

# 1. Create DataFrames for the dual-axis chart 
# Normalize both series to start at 100 for comparison
if not gf_aligned.empty and not stk_aligned.empty:
    gf_normalized = (gf_aligned / gf_aligned.iloc[0]) * 100
    stk_normalized = (stk_aligned / stk_aligned.iloc[0]) * 100
    
    combined_df = pd.DataFrame({
        "Date": gf_normalized.index,
        "Global Money Flow": gf_normalized,
        "Stock Price": stk_normalized
    })
    
    combined_long_df = combined_df.melt(
        id_vars='Date',
        value_vars=['Global Money Flow', 'Stock Price'],
        var_name='Series',
        value_name='Value'
    )
    
    # 2. Define the Shared X-Axis Scale to ensure perfect alignment
    shared_x_scale = alt.Scale(domain=[gf_normalized.index.min(), gf_normalized.index.max()])
    
    # 3. Create Top Chart (Correlation)
    if not rolling_corr_df.empty:
        corr_chart = alt.Chart(rolling_corr_df).mark_line(color='#1f77b4', opacity=0.6).encode(
            x=alt.X('Date:T', scale=shared_x_scale, title=None), # Hide title on top chart for cleaner look
            y=alt.Y('Correlation:Q', title=f'{user_ticker} - Correlation (%)', 
                   scale=alt.Scale(domain=[-100, 100])),
            tooltip=['Date:T', alt.Tooltip('Correlation:Q', format='.1f')]
        ).properties(height=150)
        
        # Add correlation zero line
        corr_zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='gray', strokeDash=[3, 3]).encode(y='y')
        corr_chart = corr_chart + corr_zero_line
    else:
        # Create empty chart if no correlation data
        corr_chart = alt.Chart(pd.DataFrame({'x': [], 'y': []})).mark_text(
            text="Insufficient data for correlation calculation"
        ).properties(height=150)
    
    # 4. Create Bottom Chart (Price vs Flow)
    base = alt.Chart(combined_long_df).encode(
        x=alt.X('Date:T', scale=shared_x_scale)
    )
    
    # Color scale for the two lines
    color_scale = alt.Scale(domain=['Global Money Flow', 'Stock Price'], 
                           range=['#1f77b4', '#d62728'])  # Blue for GMF, Red for Stock
    
    # GMF line
    money_flow_line = base.mark_line(color='#1f77b4', opacity=0.8).encode(
        y=alt.Y('Value:Q', axis=alt.Axis(title='Global Money Flow', orient='left')),
        color=alt.Color('Series:N', scale=color_scale, legend=alt.Legend(orient='top-left', title=None))
    ).transform_filter(alt.datum.Series == 'Global Money Flow')
    
    # Stock price line
    stock_price_line = base.mark_line(opacity=0.8).encode(
        y=alt.Y('Value:Q', axis=alt.Axis(title=f'Normalized {user_ticker} Price', orient='right')),
        color=alt.Color('Series:N', scale=color_scale, legend=None)
    ).transform_filter(alt.datum.Series == 'Stock Price')
    
    # Add the correlation text overlay in top-right corner
    if pd.notna(latest_corr):
        correlation_text = alt.Chart(pd.DataFrame({'x':[0], 'y':[0]})).mark_text(
            align='right', baseline='top', fontSize=14, fontWeight='bold', color='gray'
        ).encode(
            x=alt.value(700),  # Position from left
            y=alt.value(10),   # Position from top
            text=alt.value(f'{cw_}D Corr: {latest_corr:.1f}%')
        )
    else:
        correlation_text = alt.Chart(pd.DataFrame({'x':[0], 'y':[0]})).mark_text(
            align='right', baseline='top', fontSize=12, color='gray'
        ).encode(
            x=alt.value(700),
            y=alt.value(10),
            text=alt.value(f'{cw_}D Corr: N/A')
        )
    
    # Combine the two price lines with independent y-axes
    combined_price_chart = alt.layer(
        money_flow_line, 
        stock_price_line
    ).resolve_scale(
        y='independent'
    ).properties(height=300)
    
    # Add the correlation text to the price chart
    combined_price_chart = combined_price_chart + correlation_text
    
    # 5. Combine Top and Bottom charts
    final_stacked_chart = alt.vconcat(
        corr_chart,
        combined_price_chart
    ).resolve_scale(
        x='shared'  # Share the x-axis between charts
    ).properties(
        title=f"{user_ticker} Correlation & Price Analysis"
    )
    
    # 6. Display the final unified chart
    st.altair_chart(final_stacked_chart, use_container_width=True)
    
    # Display correlation interpretation
    if pd.notna(latest_corr):
        st.markdown(f"""
        **Interpretation for {user_ticker}:**
        - **{cw_}-Day Correlation with GMF: {latest_corr:.1f}%**
        - {get_correlation_interpretation(latest_corr)}
        """)
else:
    st.warning(f"Insufficient overlapping data between {user_ticker} and GMF index for analysis.")

# Multi-ticker Correlation Analysis (keep your original style)
st.markdown("""
### 📊 Multi-Stock Correlation Analysis
Enter multiple tickers to compare their correlation with Global Money Flow.
""")

# Use a text input in the main area for tickers
tickers_input_main = st.text_input("Enter tickers separated by commas (min 3 required):", 
                                  value="AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, JPM, JNJ, V, PG",
                                  key="multi_ticker_input")

ticker_list = [t.strip().upper() for t in tickers_input_main.split(",") if t.strip()]

if len(ticker_list) >= 3:
    # Load multi-ticker data
    all_tickers_dict = {t: t for t in ticker_list}
    try:
        all_data = load_data(all_tickers_dict, start_date, end_date)
        
        # Calculate correlations for each ticker
        corr_results = []
        for ticker in ticker_list:
            try:
                if ticker not in all_data.columns:
                    corr_results.append({'Ticker': ticker, 'Correlation %': float('nan')})
                    continue

                # Get stock data and smooth it
                stock_data = all_data[ticker].fillna(method='ffill')
                stock_smoothed = stock_data.rolling(window=5, min_periods=1).mean()
                stock_smoothed.iloc[-1] = stock_data.iloc[-1]
                
                # Align with GMF
                stock_aligned, gmf_aligned = stock_smoothed.align(gf_single, join='inner')
                
                if len(stock_aligned) >= cw_ and len(gmf_aligned) >= cw_:
                    # Calculate rolling correlation
                    rolling_corr = stock_aligned.rolling(cw_, min_periods=cw_//2).corr(gmf_aligned)
                    if not rolling_corr.empty:
                        latest_corr_val = rolling_corr.iloc[-1]
                        if pd.notna(latest_corr_val):
                            corr_results.append({
                                'Ticker': ticker, 
                                'Correlation %': round(latest_corr_val * 100, 1)
                            })
                            continue
                
                corr_results.append({'Ticker': ticker, 'Correlation %': float('nan')})

            except Exception as e:
                corr_results.append({'Ticker': ticker, 'Correlation %': float('nan')})

        # Create and display correlation table
        corr_df = pd.DataFrame(corr_results)
        corr_df = corr_df.sort_values('Correlation %', na_position='last', ascending=False)

        if not corr_df.empty:
            st.markdown(f"### {cw_}D Rolling Correlation with Global Money Flow")
            
            # Color coding function
            def color_corr(val):
                if pd.isna(val):
                    return 'color: gray; font-style: italic'
                elif val >= 60:
                    return 'color: #006400; font-weight: bold; background-color: #e6ffe6'  # Dark green
                elif val >= 30:
                    return 'color: #228B22;'  # Forest green
                elif val >= 10:
                    return 'color: #32CD32;'  # Lime green
                elif val <= -60:
                    return 'color: #8B0000; font-weight: bold; background-color: #ffe6e6'  # Dark red
                elif val <= -30:
                    return 'color: #B22222;'  # Firebrick
                elif val <= -10:
                    return 'color: #DC143C;'  # Crimson
                else:
                    return 'color: #696969;'  # Dim gray
            
            # Display with styling
            styled_df = corr_df.style.map(color_corr, subset=['Correlation %'])
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Add interpretation guide
            with st.expander("📈 Correlation Interpretation Guide"):
                st.markdown("""
                ### How to interpret correlation values:
                
                **Positive Correlation (Stock moves WITH risk appetite):**
                - **60-100%**: Very strong correlation with global risk flows
                - **30-60%**: Strong correlation - tends to move with market sentiment
                - **10-30%**: Moderate correlation - influenced by but not dictated by risk flows
                
                **Negative Correlation (Stock moves AGAINST risk appetite - defensive/haven):**
                - **(-60)-(-100)%**: Very strong inverse correlation - acts as strong hedge
                - **(-30)-(-60)%**: Strong inverse correlation - defensive characteristics
                - **(-10)-(-30)%**: Moderate inverse correlation - some hedging properties
                
                **Near Zero (±0-10%):**
                - Stock movements are largely independent of global risk flows
                - Company-specific or sector-specific factors dominate
                
                ### Trading Implications:
                - **High positive correlation**: Buy when GMF is rising, sell when falling
                - **High negative correlation**: Buy when GMF is falling (hedge), sell when rising
                - **Low correlation**: Focus on stock-specific fundamentals
                """)
                
            # Optional: Create bar chart visualization
            if st.checkbox("Show Bar Chart Visualization", value=False):
                bar_df = corr_df.dropna()
                if not bar_df.empty:
                    # Sort for bar chart
                    bar_df = bar_df.sort_values('Correlation %', ascending=True)
                    
                    bar_chart = alt.Chart(bar_df).mark_bar().encode(
                        x=alt.X('Correlation %:Q', title='Correlation %'),
                        y=alt.Y('Ticker:N', sort='-x', title='Ticker'),
                        color=alt.condition(
                            alt.datum['Correlation %'] > 0,
                            alt.value('#2E8B57'),  # Sea green for positive
                            alt.value('#CD5C5C')   # Indian red for negative
                        ),
                        tooltip=['Ticker:N', alt.Tooltip('Correlation %:Q', format='.1f')]
                    ).properties(
                        title=f"{cw_}D Correlation with Global Money Flow",
                        height=400
                    )
                    
                    # Add vertical line at zero
                    zero_line = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(
                        color='gray', strokeDash=[3, 3]
                    ).encode(x='x')
                    
                    st.altair_chart(bar_chart + zero_line, use_container_width=True)
        else:
            st.warning("No correlation data available. Check ticker validity and date range.")
            
    except Exception as e:
        st.error(f"Failed to load data for tickers: {e}")
else:
    st.info("Enter at least 3 tickers separated by commas to analyze.")

# Interpretation Guide
with st.expander("📖 Complete GMF Interpretation Guide"):
    st.markdown("""
    ## Complete Global Money Flow (GMF) Interpretation Guide
    
    ### **GMF Index Construction:**
    ```
    Daily GMF Change = Σ (Asset_Daily_Return × Weight)
    GMF Index = Cumulative Sum of Daily GMF Changes × 100
    ```
    
    ### **Index Values Interpretation:**
    - **Positive Values**: Net capital flowing INTO risk-on assets (bullish sentiment)
    - **Negative Values**: Net capital flowing INTO risk-off assets (bearish/defensive)
    - **Rising Trend**: Increasing risk appetite, bullish for equities
    - **Falling Trend**: Decreasing risk appetite, bearish for equities
    
    ### **Z-Score (Climax Indicator):**
    - **Above +1.5**: Overbought/Euphoric conditions → Potential reversal point
    - **Below -1.5**: Oversold/Panic conditions → Potential bounce opportunity
    - **Between ±0.8**: Normal trading range
    
    ### **Momentum (30-Day Rate of Change):**
    - **Above +0.5%/day**: Strong risk-on acceleration → Trend continuation likely
    - **Below -0.5%/day**: Strong risk-off acceleration → Trend continuation likely
    
    ### **Trading Signals Framework:**
    
    1. **BUY SIGNALS (Risk-On):**
       - Z-Score < -1.5 (oversold) AND Momentum turning positive
       - Z-Score rising from negative to positive territory
       - Strong positive momentum (> +0.5%/day) in neutral zone
    
    2. **SELL SIGNALS (Risk-Off):**
       - Z-Score > +1.5 (overbought) AND Momentum turning negative
       - Z-Score falling from positive to negative territory
       - Strong negative momentum (< -0.5%/day) in neutral zone
    
    3. **TREND FOLLOWING:**
       - High momentum (> ±0.5%/day) in direction of trend
       - Z-Score between ±0.8 with consistent momentum
    
    4. **MEAN REVERSION:**
       - Extreme Z-Score (> ±1.5) with fading momentum
       - Divergence between price and momentum
    
    ### **Asset Correlation Strategy:**
    
    **High Positive Correlation (> 60%):**
    - Trade WITH the GMF trend
    - Use GMF signals for entry/exit timing
    - Good for momentum strategies
    
    **Negative Correlation (< -30%):**
    - Trade AGAINST the GMF trend (hedge)
    - Buy when GMF is falling, sell when rising
    - Portfolio diversification/defensive allocation
    
    **Low Correlation (±0-30%):**
    - Focus on stock-specific factors
    - Less influenced by macro sentiment
    - Good for alpha generation through stock picking
    
    ### **Weight Configuration Strategy:**
    - **Total weight > 0**: Bullish bias in index construction
    - **Total weight < 0**: Bearish bias in index construction
    - Adjust individual weights based on conviction
    - Higher absolute weights = more influence from that asset
    
    ### **Timeframe Considerations:**
    - **Short-term (days)**: Focus on momentum and recent Z-Score
    - **Medium-term (weeks)**: Focus on Z-Score extremes and trend
    - **Long-term (months)**: Focus on overall index direction and correlations
    """)
