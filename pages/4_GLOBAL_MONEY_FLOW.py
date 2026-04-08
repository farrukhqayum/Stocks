import streamlit as st

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
- **Risk-On Assets**: BTC, S&P 500, Emerging Markets, Oil
- **Risk-Off Assets**: Gold, US Dollar, Treasury Bonds, VIX (inverse)
- **GMF Index**: Composite of weighted asset returns showing capital rotation
""")

# ========== SIDEBAR CONFIGURATION ==========
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

# Single stock analysis input
st.sidebar.markdown("---")
st.sidebar.header("💹 Stock Analysis")
user_ticker = st.sidebar.text_input("Enter Stock Ticker", value="TSLA")

# ========== DATA LOADING FUNCTIONS ==========
def load_data(tickers, start, end):
    """Load data from Yahoo Finance"""
    raw = yf.download(list(tickers.values()), start=start, end=end, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        if 'Adj Close' in raw.columns.get_level_values(0):
            df = raw.xs('Adj Close', axis=1, level=0)
        elif 'Close' in raw.columns.get_level_values(0):
            df = raw.xs('Close', axis=1, level=0)
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

# ========== LOAD DATA ==========
try:
    data = load_data(tickers, start_date, end_date)
    spx_raw = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
    
    if isinstance(spx_raw.columns, pd.MultiIndex) and 'Adj Close' in spx_raw.columns.get_level_values(0):
        spx_data = spx_raw['Adj Close']
        spx_data = spx_data.rename("S&P 500 (SPX)")
    elif 'Adj Close' in spx_raw.columns:
        spx_data = spx_raw['Adj Close']
    else:
        spx_data = spx_raw['Close'].squeeze()
    spx_data.name = "S&P 500 (SPX)"

except Exception as e:
    st.warning(f"⚠️ Error loading data: {e}. Please check ticker availability and date range.")
    st.stop()

if use_business_days:
    data = data = data.asfreq('B').ffill()
    spx_data = spx_data.asfreq('B').ffill()

# ========== GMF INDEX CALCULATION ==========
def calculate_gmf_index(data, weights):
    """Calculate GMF Index as weighted sum of daily percentage changes"""
    daily_pct = data.pct_change().fillna(0)
    weights_series = pd.Series(weights).reindex(data.columns).fillna(0.0)
    weighted_daily = daily_pct.multiply(weights_series, axis=1)
    daily_gmf_change = weighted_daily.sum(axis=1)
    gmf_index = (daily_gmf_change * 100).cumsum()
    return gmf_index

gmf_raw = calculate_gmf_index(data, weights)
gmf_index = gmf_raw - gmf_raw.iloc[0]

# Create smoothed versions
money_flow_raw = gmf_index
money_flow_s = money_flow_raw.rolling(3, min_periods=1).mean()
money_flow_smooth = money_flow_raw.rolling(smooth_window, min_periods=1).mean()

# Calculate Z-Score
rolling_mean = money_flow_smooth.rolling(window=z_score_window, min_periods=5).mean()
rolling_std = money_flow_smooth.rolling(z_score_window, min_periods=5).std(ddof=0)

money_flow_zscore = (money_flow_smooth - rolling_mean) / rolling_std
money_flow_zscore = money_flow_zscore.replace([np.inf, -np.inf], 0).fillna(0)

# Calculate Momentum
money_flow_momentum = money_flow_smooth.diff(30) / 30 * 100
money_flow_momentum = money_flow_momentum.fillna(0)

# Get latest values
latest_momentum = money_flow_momentum.iloc[-1] if not money_flow_momentum.empty else 0
latest_zscore = money_flow_zscore.iloc[-1] if not money_flow_zscore.empty else 0

# ========== SENTIMENT LOGIC ==========
Z_EXTREME = 1.5
MOM_HIGH = 0.5
MOM_LOW = -0.5
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

# ========== MAIN DISPLAY ==========
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
    st.metric("30-Day Momentum", f"{latest_momentum:+.0f}%",
              delta="Accelerating" if abs(latest_momentum) > MOM_HIGH else "Stable")

# ===== 2nd ROW: POSITIONING & ROTATION SNAPSHOT =====
pos_col1, pos_col2, pos_col3 = st.columns(3)

# 1) Equity allocation snapshot (reusing logic later in expander)
with pos_col1:
    st.subheader("Equity Bias")
    try:
        # Compute positioning_df once, to reuse in expander as well
        positioning_data = {}

        if 'Crude Oil (CL)' in data.columns and 'Emerging Markets (EEM)' in data.columns:
            commod_em_ratio = data['Crude Oil (CL)'] / data['Emerging Markets (EEM)']
            positioning_data['Commodity/EM_Ratio'] = (commod_em_ratio / commod_em_ratio.iloc[0] * 100)

        if 'US 10Y Treasury (IEF)' in data.columns and 'US Dollar Index (DXY)' in data.columns:
            treasury_dollar_ratio = data['US 10Y Treasury (IEF)'] / data['US Dollar Index (DXY)']
            positioning_data['Treasury/Dollar_Ratio'] = (treasury_dollar_ratio / treasury_dollar_ratio.iloc[0] * 100)

        positioning_df = pd.DataFrame({
            'Date': data.index,
            'GMF_Index': money_flow_smooth,
            'GMF_Momentum': money_flow_momentum,
            **positioning_data
        }).dropna()

        equity_allocation = None
        positioning_label = "N/A"
        sectors_label = "N/A"

        if not positioning_df.empty:
            recent_gmf = positioning_df['GMF_Index'].iloc[-20:].mean()
            recent_mom = positioning_df['GMF_Momentum'].iloc[-20:].mean()

            equity_allocation = 50

            # GMF adjustment
            if recent_gmf > 20:
                equity_allocation += 20
            elif recent_gmf > 0:
                equity_allocation += 10
            elif recent_gmf < -20:
                equity_allocation -= 20
            elif recent_gmf < 0:
                equity_allocation -= 10

            # Momentum adjustment
            if recent_mom > 0.3:
                equity_allocation += 15
            elif recent_mom > 0.1:
                equity_allocation += 5
            elif recent_mom < -0.3:
                equity_allocation -= 15
            elif recent_mom < -0.1:
                equity_allocation -= 5

            # Commodity/EM adjustment
            if 'Commodity/EM_Ratio' in positioning_df.columns:
                recent_commod_em = positioning_df['Commodity/EM_Ratio'].iloc[-1]
                if recent_commod_em > 110:
                    equity_allocation -= 10
                elif recent_commod_em < 90:
                    equity_allocation += 5

            equity_allocation = max(0, min(100, equity_allocation))

            if equity_allocation >= 70:
                positioning_label = "Max Risk-On"
                sectors_label = "Cyclicals, Tech, Small Caps"
            elif equity_allocation >= 60:
                positioning_label = "Risk-On"
                sectors_label = "Tech, Discretionary, Industrials"
            elif equity_allocation >= 40:
                positioning_label = "Neutral"
                sectors_label = "Balanced, Quality Growth"
            elif equity_allocation >= 30:
                positioning_label = "Risk-Off"
                sectors_label = "Defensives, Healthcare"
            else:
                positioning_label = "Max Risk-Off"
                sectors_label = "Cash, Bonds, Defensives"

        st.metric("Equity Allocation", f"{equity_allocation or 0:.0f}%", positioning_label)

    except Exception:
        st.metric("Equity Allocation", "N/A", "Error")

# 2) Sector rotation snapshot
with pos_col2:
    st.subheader("Sector Tilt")
    if 'latest_zscore' in locals() and 'latest_momentum' in locals():
        if latest_zscore < -1.0 and latest_momentum > 0:
            stage = "Late Bear → Early Bull"
            sectors = "Banks, Homebuilders, Small Caps"
        elif latest_zscore < 0 and latest_momentum > 0.2:
            stage = "Early Bull"
            sectors = "Financials, Industrials, Materials"
        elif 0 <= latest_zscore < 1.0 and latest_momentum > 0.1:
            stage = "Mid Bull"
            sectors = "Tech, Comm, Healthcare"
        elif latest_zscore >= 1.0 and latest_momentum > 0:
            stage = "Late Bull"
            sectors = "Energy, Staples, Utilities"
        elif latest_zscore >= 0.5 and latest_momentum < 0:
            stage = "Early Bear"
            sectors = "Staples, Utilities, Gold"
        elif latest_zscore < 0.5 and latest_momentum < -0.1:
            stage = "Mid Bear"
            sectors = "Staples, Bonds, Gold"
        else:
            stage = "Transition"
            sectors = "Quality, Dividends"

        st.metric("Market Stage", stage, sectors)
    else:
        st.metric("Market Stage", "N/A", "")


# 3) Cross-asset quick view
with pos_col3:
    st.subheader("Cross-Asset Strength")
    commod_em_text = "N/A"
    treas_dxy_text = "N/A"
    if 'Commodity/EM_Ratio' in positioning_df.columns:
        commod_em_latest = positioning_df['Commodity/EM_Ratio'].iloc[-1]
        commod_em_text = f"{commod_em_latest:.1f}"
    if 'Treasury/Dollar_Ratio' in positioning_df.columns:
        treas_dxy_latest = positioning_df['Treasury/Dollar_Ratio'].iloc[-1]
        treas_dxy_text = f"{treas_dxy_latest:.1f}"

    st.metric("Commodity/Emerging Markets", commod_em_text)
    st.metric("Bond/Dollar Strength", treas_dxy_text)

# ========== MARKET POSITIONING ==========
st.markdown("---")
st.header("🎯 Market Positioning")

# Cross-Asset Positioning
with st.expander("📈 Cross-Asset Relative Strength & Stock Positioning"):
    
    st.markdown("""
    ### How to Position Based on Cross-Asset Relationships
    
    1. **Commodities vs. Emerging Markets (Risk-On Hierarchy):**
       - Commodities ↑ + EM ↑ = **Maximum Risk-On** → Buy cyclical, materials, industrials
       - Commodities ↑ + EM ↓ = **Stagflation Risk** → Buy energy, materials, avoid EM stocks
       - Commodities ↓ + EM ↑ = **Growth Recovery** → Buy tech, consumer discretionary
       - Commodities ↓ + EM ↓ = **Risk-Off** → Defensive sectors only
    
    2. **Treasuries vs. Dollar (Liquidity Signals):**
       - Treasuries ↑ (yields ↓) + Dollar ↓ = **Liquidity Expansion** → Growth stocks
       - Treasuries ↑ (yields ↓) + Dollar ↑ = **Flight to Quality** → Defensive/quality
       - Treasuries ↓ (yields ↑) + Dollar ↓ = **Reflation Trade** → Value/cyclicals
       - Treasuries ↓ (yields ↑) + Dollar ↑ = **Tightening Risk** → Reduce leverage
    """)
    
    try:
        # Calculate relative strength ratios
        positioning_data = {}
        
        if 'Crude Oil (CL)' in data.columns and 'Emerging Markets (EEM)' in data.columns:
            commod_em_ratio = data['Crude Oil (CL)'] / data['Emerging Markets (EEM)']
            positioning_data['Commodity/EM_Ratio'] = (commod_em_ratio / commod_em_ratio.iloc[0] * 100)
            
        if 'US 10Y Treasury (IEF)' in data.columns and 'US Dollar Index (DXY)' in data.columns:
            treasury_dollar_ratio = data['US 10Y Treasury (IEF)'] / data['US Dollar Index (DXY)']
            positioning_data['Treasury/Dollar_Ratio'] = (treasury_dollar_ratio / treasury_dollar_ratio.iloc[0] * 100)
        
        positioning_df = pd.DataFrame({
            'Date': data.index,
            'GMF_Index': money_flow_smooth,
            'GMF_Momentum': money_flow_momentum,
            **positioning_data
        }).dropna()
        
        if not positioning_df.empty:
            recent_gmf = positioning_df['GMF_Index'].iloc[-20:].mean()
            recent_mom = positioning_df['GMF_Momentum'].iloc[-20:].mean()
            
            # Determine positioning
            equity_allocation = 50
            
            # GMF adjustment
            if recent_gmf > 20:
                equity_allocation += 20
            elif recent_gmf > 0:
                equity_allocation += 10
            elif recent_gmf < -20:
                equity_allocation -= 20
            elif recent_gmf < 0:
                equity_allocation -= 10
            
            # Momentum adjustment
            if recent_mom > 0.3:
                equity_allocation += 15
            elif recent_mom > 0.1:
                equity_allocation += 5
            elif recent_mom < -0.3:
                equity_allocation -= 15
            elif recent_mom < -0.1:
                equity_allocation -= 5
            
            # Commodity/EM adjustment
            if 'Commodity/EM_Ratio' in positioning_df.columns:
                recent_commod_em = positioning_df['Commodity/EM_Ratio'].iloc[-1]
                if recent_commod_em > 110:
                    equity_allocation -= 10
                elif recent_commod_em < 90:
                    equity_allocation += 5
            
            # Clamp between 0 and 100
            equity_allocation = max(0, min(100, equity_allocation))
            
            # Determine positioning strategy
            if equity_allocation >= 70:
                positioning = "**MAXIMUM RISK-ON** - Full equity allocation"
                pos_color = "#4caf50"
                sectors = "Cyclicals, Tech, Small Caps, High Beta"
            elif equity_allocation >= 60:
                positioning = "**RISK-ON** - Above average equity"
                pos_color = "#81c784"
                sectors = "Tech, Consumer Discretionary, Industrials"
            elif equity_allocation >= 40:
                positioning = "**NEUTRAL** - Balanced allocation"
                pos_color = "#ffb74d"
                sectors = "Balanced mix, Quality growth"
            elif equity_allocation >= 30:
                positioning = "**RISK-OFF** - Below average equity"
                pos_color = "#ef5350"
                sectors = "Defensive, Healthcare, Utilities, Consumer Staples"
            else:
                positioning = "**MAXIMUM RISK-OFF** - Minimal equity"
                pos_color = "#d32f2f"
                sectors = "Cash, Bonds, Defensive sectors only"
            
            st.markdown(f"""
            <div style="padding:1.2em; border-radius:12px; text-align:center; background-color:{pos_color}; color:white; font-size:1.3em; font-weight:bold;">
            Recommended Equity Allocation: {equity_allocation:.0f}%<br>
            {positioning}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            **Recommended Sectors:** {sectors}
            
            **Current Readings:**
            - GMF Index: {recent_gmf:+.1f}
            - GMF Momentum: {recent_mom:+.3f}/day
            """)
            
    except Exception as e:
        st.warning(f"Could not calculate positioning: {e}")

# Sector Rotation
with st.expander("🏗️ Sector Rotation Matrix"):
    
    st.markdown("""
    ### Sector Rotation Based on GMF Phase
    
    | GMF Phase | Z-Score Range | Momentum | Recommended Sectors | Avoid |
    |-----------|---------------|----------|---------------------|-------|
    | **Early Bull** | -1.5 to 0 | Turning positive | Cyclicals, Financials, Small Caps | Defensives |
    | **Mid Bull** | 0 to 1.0 | Positive | Tech, Industrials, Materials | Early cyclicals |
    | **Late Bull** | 1.0 to 1.5 | High but peaking | Energy, Staples, Healthcare | High-beta tech |
    | **Early Bear** | 1.5 to 0.5 | Turning negative | Defensives, Utilities, Bonds | Cyclicals |
    | **Mid Bear** | 0.5 to -1.0 | Negative | Consumer Staples, Healthcare, Gold | Growth stocks |
    | **Late Bear** | -1.5 to -1.0 | Negative but slowing | Early cyclicals, Banks | Defensives at highs |
    """)
    
    if 'latest_zscore' in locals() and 'latest_momentum' in locals():
        if latest_zscore < -1.0 and latest_momentum > 0:
            stage = "**LATE BEAR / EARLY BULL TRANSITION**"
            sectors = "Banks, Homebuilders, Consumer Discretionary, Small Caps"
            rationale = "Oversold bounce + improving momentum"
        elif latest_zscore < 0 and latest_momentum > 0.2:
            stage = "**EARLY BULL**"
            sectors = "Financials, Industrials, Materials, Consumer Discretionary"
            rationale = "Risk appetite returning, early cyclicals lead"
        elif 0 <= latest_zscore < 1.0 and latest_momentum > 0.1:
            stage = "**MID BULL**"
            sectors = "Technology, Communications, Healthcare, Industrials"
            rationale = "Sustainable uptrend, growth sectors outperform"
        elif latest_zscore >= 1.0 and latest_momentum > 0:
            stage = "**LATE BULL**"
            sectors = "Energy, Materials, Staples, Utilities"
            rationale = "Late cycle, inflationary pressures, defensive rotation"
        elif latest_zscore >= 0.5 and latest_momentum < 0:
            stage = "**EARLY BEAR**"
            sectors = "Consumer Staples, Utilities, Healthcare, Gold"
            rationale = "Risk-off beginning, defensive positioning"
        elif latest_zscore < 0.5 and latest_momentum < -0.1:
            stage = "**MID BEAR**"
            sectors = "Staples, Utilities, Bonds, Gold Miners"
            rationale = "Full risk-off, capital preservation"
        else:
            stage = "**TRANSITION / CONSOLIDATION**"
            sectors = "Quality Growth, Dividend Payers, Balanced"
            rationale = "Unclear trend, focus on quality"
        
        st.markdown(f"""
        **Current Market Stage:** {stage}
        
        **Recommended Sectors:** {sectors}
        
        **Rationale:** {rationale}
        """)


# ========== GMF CHARTS ==========
st.markdown("---")
st.header("📊 GMF Visualization")

# Prepare data for plotting
df_plot = pd.DataFrame({
    "Date": money_flow_raw.index,
    "Money Flow Curve": money_flow_s,
    "Smoothed Curve": money_flow_smooth,
    "Momentum": money_flow_momentum,
    "Z-Score": money_flow_zscore
}).dropna()

df_plot['Above'] = df_plot['Money Flow Curve'] > df_plot['Smoothed Curve']

# Create GMF Chart
st.subheader("🌊 GMF Curves")
base = alt.Chart(df_plot).encode(
    x=alt.X('Date:T', axis=alt.Axis(format='%d/%m/%Y', title='Date'))
)
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
st.subheader("📈 GMF Momentum (30-Day Rate of Change)")
momentum_chart = (
    alt.Chart(df_plot)
    .mark_bar(opacity=0.5)
    .encode(
        x=alt.X('Date:T', axis=alt.Axis(format='%d/%m/%Y', title='Date')),
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
st.subheader("📊 Climax Zone Indicator (Z-Score)")
zscore_chart = (
    alt.Chart(df_plot)
    .mark_area(opacity=0.6)
    .encode(
        x=alt.X('Date:T', axis=alt.Axis(format='%d/%m/%Y', title='Date')),
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

# ========== ASSET ANALYSIS ==========
st.markdown("---")
st.header("📈 Asset Analysis")

# Underlying Assets
with st.expander("📊 Show Underlying Asset Returns"):
    asset_returns = data.pct_change().fillna(0) * 100
    weights_series = pd.Series(weights).reindex(asset_returns.columns).fillna(0)
    weighted_returns = asset_returns.multiply(weights_series, axis=1)
    cumulative_contrib = weighted_returns.cumsum()
    
    cumulative_melted = cumulative_contrib.reset_index().melt("Date", var_name="Asset", value_name="Cumulative Contribution")
    
    asset_chart = (
        alt.Chart(cumulative_melted)
        .mark_area(opacity=0.6)
        .encode(
            x=alt.X('Date:T', axis=alt.Axis(format='%d/%m/%Y', title='Date')),
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

# Divergence Check
with st.expander("⚠️ Divergence Check: S&P 500 vs. GMF Momentum"):
    spx_pct = spx_data.pct_change().fillna(0) * 100
    spx_aligned, gmf_aligned = spx_pct.align(money_flow_momentum, join='inner')
    
    lookback = 60
    if len(spx_aligned) >= lookback:
        rolling_corr = spx_aligned.rolling(lookback).corr(gmf_aligned.reindex(spx_aligned.index))
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
                x=alt.X('Date:T', axis=alt.Axis(format='%d/%m/%Y', title='Date')),
                y=alt.Y('Correlation:Q', scale=alt.Scale(domain=[-1, 1])),
                tooltip=['Date:T', alt.Tooltip('Correlation:Q', format='.2f')]
            ).properties(title=f"{lookback}-Day Rolling Correlation")
            
            corr_zero = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='gray').encode(y='y')
            st.altair_chart(corr_chart + corr_zero, use_container_width=True)
    else:
        st.info(f"Not enough data for a {lookback}-day divergence check.")

# ========== STOCK CORRELATION ANALYSIS ==========
st.markdown("---")
st.header("💹 Stock Correlation Analysis")

def get_trading_strategy(correlation, gmf_momentum):
    """Generate trading strategy based on correlation and GMF momentum"""
    if pd.isna(correlation) or pd.isna(gmf_momentum):
        return "Insufficient data for strategy generation."
    
    if correlation > 0.6:
        if gmf_momentum > 0:
            return """
            **Strategy:** Strong momentum play
            - Enter on pullbacks to GMF support
            - Use trailing stops (e.g., 10-15% below highs)
            - Consider options for leverage (calls or bull spreads)
            - Target: Ride the trend until correlation breaks below 50%
            """
        else:
            return """
            **Strategy:** Avoid or short
            - High correlation + falling GMF = high risk
            - Consider puts or bear spreads if trend confirms
            - Wait for GMF to stabilize before considering longs
            """
    
    elif correlation < -0.3:
        if gmf_momentum < 0:
            return """
            **Strategy:** Defensive hedge
            - Buy as portfolio protection
            - Size appropriately (10-20% of portfolio for hedging)
            - Hold until GMF shows signs of bottoming
            - Consider covered calls for income
            """
        else:
            return """
            **Strategy:** Reduce hedge exposure
            - Negative correlation + rising GMF = hedge underperforming
            - Trim hedge positions
            - Consider switching to cash or low-correlation assets
            """
    
    else:  # Low/moderate correlation
        return """
        **Strategy:** Stock-specific focus
        - Ignore GMF signals for this stock
        - Focus on company fundamentals
        - Technical analysis on stock chart
        - Options strategies based on volatility
        - Good for pairs trading or relative value
        """

def get_correlation_interpretation(corr_value):
    """Provide interpretation of correlation value"""
    if pd.isna(corr_value):
        return "Insufficient data for correlation analysis."
    
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
st.subheader(f"Single Stock: {user_ticker}")
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
    user_stock_data = user_stock_data.squeeze()
    
except Exception as e:
    st.error(f"Failed to load data for {user_ticker}: {e}")
    st.stop()

user_stock_smoothed = user_stock_data.rolling(window=5, min_periods=1).mean()
user_stock_smoothed.iloc[-1] = user_stock_data.iloc[-1]

gf_single = money_flow_s
stk_single = user_stock_smoothed
gf_aligned, stk_aligned = gf_single.align(stk_single, join='inner')

cw_ = 60
latest_corr = float('nan')
latest_corr_percent = float('nan')

if len(gf_aligned) >= cw_:
    rolling_corr_single = gf_aligned.rolling(cw_, min_periods=cw_//2).corr(stk_aligned)
    if not rolling_corr_single.empty:
        latest_corr = rolling_corr_single.iloc[-1]
        if pd.notna(latest_corr):
            latest_corr_percent = round(latest_corr * 100, 1)
        else:
            latest_corr_percent = float('nan')
    
    rolling_corr_df = pd.DataFrame({
        "Date": rolling_corr_single.index,
        "Correlation": rolling_corr_single * 100
    }).dropna()
else:
    rolling_corr_df = pd.DataFrame({"Date": [], "Correlation": []})

if not gf_aligned.empty and not stk_aligned.empty:
    gf_normalized = (gf_aligned / gf_aligned.iloc[0]) * 100
    stk_normalized = (stk_aligned / stk_aligned.iloc[0]) * 100
    
    combined_df = pd.DataFrame({
        "Date": gf_aligned.index,
        "Global Money Flow": gf_aligned,
        "Stock Price": stk_normalized
    })
    
    combined_long_df = combined_df.melt(
        id_vars='Date',
        value_vars=['Global Money Flow', 'Stock Price'],
        var_name='Series',
        value_name='Value'
    )
    
    shared_x_scale = alt.Scale(domain=[gf_normalized.index.min(), gf_normalized.index.max()])
    
    # Correlation chart
    if not rolling_corr_df.empty:
        corr_chart = alt.Chart(rolling_corr_df).mark_line(color='#1f77b4', opacity=0.6).encode(
            x=alt.X('Date:T', axis=alt.Axis(format='%d/%m/%Y', title='Date')),
            y=alt.Y('Correlation:Q', title=f'{user_ticker} - Correlation (%)', 
                   scale=alt.Scale(domain=[-100, 100])),
            tooltip=['Date:T', alt.Tooltip('Correlation:Q', format='.1f')]
        ).properties(height=150)
        
        corr_zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='gray', strokeDash=[3, 3]).encode(y='y')
        corr_chart = corr_chart + corr_zero_line
    else:
        corr_chart = alt.Chart(pd.DataFrame({'x': [], 'y': []})).mark_text(
            text="Insufficient data for correlation calculation"
        ).properties(height=150)
    
    # Price vs Flow chart
    base = alt.Chart(combined_long_df).encode(
        x=alt.X('Date:T', scale=shared_x_scale)
    )
    
    color_scale = alt.Scale(domain=['Global Money Flow', 'Stock Price'], 
                           range=['#1f77b4', '#d62728'])
    
    money_flow_line = base.mark_line(color='#1f77b4', opacity=0.5).encode(
        x=alt.X('Date:T', axis=None),
        y=alt.Y('Value:Q', axis=alt.Axis(title='Global Money Flow', orient='left')),
        color=alt.Color('Series:N', scale=color_scale, legend=alt.Legend(orient='top-left', title=None))
    ).transform_filter(alt.datum.Series == 'Global Money Flow')
    
    stock_price_line = base.mark_line(opacity=0.5).encode(
        x = alt.X('Date:T', axis=alt.Axis(format='%d/%m/%Y')),
        y=alt.Y('Value:Q', axis=alt.Axis(title=f'Normalized {user_ticker} Price', orient='right')),
        color=alt.Color('Series:N', scale=color_scale, legend=None)
    ).transform_filter(alt.datum.Series == 'Stock Price')
    
    # Add correlation text
    correlation_text = (
        alt.Chart(pd.DataFrame({'x': [0.5], 'y': [0]}))
          .mark_text(
              align='center',
              baseline='top',
              fontSize=14,
              fontWeight='bold',
              color='gray'
          )
          .encode(
              x='x:Q',
              y='y:Q',
              text=alt.value(
                  f'{cw_}D Corr: {latest_corr_percent:.1f}%'
                  if pd.notna(latest_corr_percent)
                  else f'{cw_}D Corr: N/A'
              )
          )
    )
    
    combined_price_chart = alt.layer(
        money_flow_line, 
        stock_price_line
    ).resolve_scale(
        y='independent'
    ).properties(height=300, width="container")
    
    combined_price_chart = combined_price_chart + correlation_text
    
    # Combine charts
    final_stacked_chart = alt.vconcat(
        corr_chart,
        combined_price_chart
    ).resolve_scale(
        x='shared'
    ).properties(
        title=f"{user_ticker} Correlation & Price Analysis"
    )
    
    st.altair_chart(final_stacked_chart, use_container_width=True)
    
    # Display correlation interpretation
    if pd.notna(latest_corr_percent):
        st.markdown(f"""
        **Interpretation for {user_ticker}:**
        - **{cw_}-Day Correlation with GMF: {latest_corr_percent:.1f}%**
        - {get_correlation_interpretation(latest_corr_percent)}
        """)
        
        # Trading strategy
        st.markdown(f"""
        **Trading Strategy:**
        {get_trading_strategy(latest_corr, latest_momentum)}
        """)
else:
    st.warning(f"Insufficient overlapping data between {user_ticker} and GMF index for analysis.")

# ========== STOCK SCREENER ==========
st.markdown("---")
st.header("🔍 Stock Screener")

with st.expander("Run Correlation-Based Screener"):
    
    st.markdown("""
    ### How to Use 60-Day Correlation for Stock Selection
    
    **High Correlation (>60%) with GMF:**
    - **When GMF is rising**: Buy these stocks first (momentum plays)
    - **When GMF is falling**: Sell/avoid these stocks
    - **Best for**: Trend following, momentum strategies
    
    **Negative Correlation (<-30%) with GMF:**
    - **When GMF is falling**: Buy these as hedges (defensive plays)
    - **When GMF is rising**: Reduce exposure
    - **Best for**: Portfolio protection, defensive allocation
    
    **Low Correlation (±0-30%) with GMF:**
    - **Always**: Focus on stock-specific factors
    - **Best for**: Alpha generation, diversification, range-bound markets
    """)
    
screener_tickers = st.text_area(
    "Enter tickers to screen (one per line or comma separated):",
    value="""COIN, MSTR, XYZ, CRM, QCOM, AMD, SMCI, BABA, XPEV, NIO, U, INTC, SNAP, UNH""",
    height=150
)
    
# Parse tickers
tickers_list = []
for line in screener_tickers.split('\n'):
    if ',' in line:
        tickers_list.extend([t.strip().upper() for t in line.split(',') if t.strip()])
    else:
        if line.strip():
            tickers_list.append(line.strip().upper())

# Remove duplicates
seen = set()
tickers_list = [x for x in tickers_list if not (x in seen or seen.add(x))]

if len(tickers_list) > 50:
    st.warning(f"Limiting to first 50 tickers (you entered {len(tickers_list)})")
    tickers_list = tickers_list[:50]

# Define styling functions
def style_screener_corr(val):
    """Style the correlation percentage column"""
    if pd.isna(val):
        return 'color: gray; font-style: italic'
    elif val > 60:
        return 'background-color: #006400; color: white; font-weight: bold'
    elif val > 30:
        return 'background-color: #228B22; color: white'
    elif val > 10:
        return 'background-color: #32CD32; color: black'
    elif val < -60:
        return 'background-color: #8B0000; color: white; font-weight: bold'
    elif val < -30:
        return 'background-color: #B22222; color: white'
    elif val < -10:
        return 'background-color: #DC143C; color: white'
    else:
        return 'background-color: #696969; color: white'

def style_screener_signal(val):
    """Style the Signal emoji column"""
    if "🟢" in val:
        return 'background-color: #155724; color: white; font-weight: bold'
    elif "🔴" in val:
        return 'background-color: #721c24; color: white; font-weight: bold'
    elif "🟡" in val:
        return 'background-color: #856404; color: white'
    else:
        return 'background-color: #6c757d; color: white'

if tickers_list and st.button("Run 60-Day Correlation Screening"):
    with st.spinner("Analyzing 60-day correlations with GMF..."):
        try:
            progress_bar = st.progress(0)
            
            screener_data = {}
            for idx, ticker in enumerate(tickers_list):
                try:
                    raw = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    if not raw.empty:
                        if isinstance(raw.columns, pd.MultiIndex):
                            if 'Adj Close' in raw.columns.get_level_values(0):
                                screener_data[ticker] = raw['Adj Close'].squeeze()
                            elif 'Close' in raw.columns.get_level_values(0):
                                screener_data[ticker] = raw['Close'].squeeze()
                        else:
                            if 'Adj Close' in raw.columns:
                                screener_data[ticker] = raw['Adj Close']
                            elif 'Close' in raw.columns:
                                screener_data[ticker] = raw['Close']
                except:
                    continue
                
                progress_bar.progress((idx + 1) / len(tickers_list))
            
            screener_results = []
            correlation_history = {}
            
            if screener_data:
                for idx, (ticker, prices) in enumerate(screener_data.items()):
                    try:
                        if isinstance(prices, pd.Series) and not prices.empty:
                            prices_clean = prices.fillna(method='ffill').dropna()
                            if len(prices_clean) < 60:
                                continue
                            
                            stock_smooth = prices_clean.rolling(5, min_periods=1).mean()
                            stock_smooth.iloc[-1] = prices_clean.iloc[-1]
                            
                            stock_aligned, gmf_aligned = stock_smooth.align(gf_single, join='inner')
                            
                            if len(stock_aligned) >= 60:
                                corr_series = stock_aligned.rolling(60, min_periods=30).corr(gmf_aligned)
                                
                                if not corr_series.empty:
                                    latest_corr = corr_series.iloc[-1]
                                    if pd.notna(latest_corr):
                                        correlation_history[ticker] = corr_series
                                        
                                        gmf_momentum_current = money_flow_momentum.iloc[-1] if not money_flow_momentum.empty else 0
                                        
                                        if latest_corr > 0.6:
                                            if gmf_momentum_current > 0:
                                                rec = "STRONG BUY (High Correlation + GMF Rising)"
                                                rec_color = "🟢"
                                                signal_score = 9
                                            else:
                                                rec = "SELL/AVOID (High Correlation + GMF Falling)"
                                                rec_color = "🔴"
                                                signal_score = 1
                                        elif latest_corr > 0.3:
                                            if gmf_momentum_current > 0:
                                                rec = "BUY (Moderate Correlation + GMF Rising)"
                                                rec_color = "🟢"
                                                signal_score = 7
                                            else:
                                                rec = "NEUTRAL (Moderate Correlation + GMF Falling)"
                                                rec_color = "⚪"
                                                signal_score = 4
                                        elif latest_corr < -0.3:
                                            if gmf_momentum_current < 0:
                                                rec = "BUY HEDGE (Negative Correlation + GMF Falling)"
                                                rec_color = "🟢"
                                                signal_score = 8
                                            else:
                                                rec = "REDUCE HEDGE (Negative Correlation + GMF Rising)"
                                                rec_color = "🟡"
                                                signal_score = 3
                                        else:
                                            rec = "NEUTRAL (Low Correlation)"
                                            rec_color = "⚪"
                                            signal_score = 5
                                        
                                        screener_results.append({
                                            'Ticker': ticker,
                                            '60D Corr %': round(latest_corr * 100, 1),
                                            'Signal': rec_color,
                                            'Recommendation': rec,
                                            'Signal Score': signal_score,
                                            'GMF Momentum': f"{gmf_momentum_current:+.3f}"
                                        })
                    except:
                        continue
            
            if screener_results:
                screener_df = pd.DataFrame(screener_results)
                screener_df = screener_df.sort_values('Signal Score', ascending=False)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    high_corr = len([x for x in screener_results if x['60D Corr %'] > 60])
                    st.metric("High Correlation (>60%)", high_corr)
                with col2:
                    neg_corr = len([x for x in screener_results if x['60D Corr %'] < -30])
                    st.metric("Negative Correlation (<-30%)", neg_corr)
                with col3:
                    low_corr = len([x for x in screener_results if -30 <= x['60D Corr %'] <= 60])
                    st.metric("Low/Moderate Correlation", low_corr)
                with col4:
                    current_gmf_momentum = money_flow_momentum.iloc[-1] if not money_flow_momentum.empty else 0
                    st.metric("GMF Momentum", f"{current_gmf_momentum:+.3f}/day")
                
                # Create display dataframe with Signal Score for gradient
                display_df_with_score = screener_df[['Ticker', '60D Corr %', 'Signal', 'Recommendation', 'GMF Momentum', 'Signal Score']].copy()
                
                # Apply gradient based on Signal Score
                styled_df = display_df_with_score.style.background_gradient(
                    subset=['Signal Score'], 
                    cmap='Greys',  # Black to white gradient
                    vmin=1, vmax=9  # Signal Score range
                )
                
                # Hide the Signal Score column after applying gradient
                styled_df = styled_df.hide_columns(['Signal Score'])

                # Apply individual column styling
                styled_df = styled_df.map(style_screener_corr, subset=['60D Corr %'])
                styled_df = styled_df.map(style_screener_signal, subset=['Signal'])
                
                st.markdown(f"### 📋 Screening Results ({len(display_df_with_score)} stocks)")
                st.dataframe(styled_df, use_container_width=True, height=400)
                
                # Download option
                csv = screener_df.to_csv(index=False)
                st.download_button(
                    label="Download Screening Results (CSV)",
                    data=csv,
                    file_name=f"gmf_screener_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
                
            else:
                st.warning("No correlation data could be calculated for the entered tickers.")
                
            progress_bar.empty()
            
        except Exception as e:
            st.error(f"Error in screener: {str(e)}")

# ========== HELP & GUIDES ==========
st.markdown("---")
st.header("📚 Guides & Interpretation")

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

with st.expander("🎯 Correlation-Based Trading Rules"):
    st.markdown("""
    ### 🎯 **Correlation-Based Trading Rules**
    
    **For High Correlation Stocks (>60%):**
    1. **Entry:** Wait for GMF > 0 AND rising momentum
    2. **Exit:** GMF < 0 OR correlation drops below 50%
    3. **Position Sizing:** Full size when conditions align
    4. **Stop Loss:** Below recent GMF support levels
    
    **For Negative Correlation Stocks (<-30%):**
    1. **Entry:** When GMF < 0 AND falling (portfolio hedge)
    2. **Exit:** When GMF > 0 AND rising (remove hedge)
    3. **Position Sizing:** 10-20% of portfolio as hedge
    4. **Stop Loss:** Use wider stops (hedges can be volatile)
    
    **For Low Correlation Stocks (±0-30%):**
    1. **Ignore GMF** for these stocks
    2. **Focus on:** Fundamentals, technicals, sector trends
    3. **Use for:** Diversification, alpha generation
    4. **Best in:** Range-bound or uncertain markets
    """)

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

st.markdown("---")
st.caption("Global Money Flow Analysis Tool • Data from Yahoo Finance")
