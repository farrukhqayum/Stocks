import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta

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
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365 * 3))
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
    "Bitcoin (BTC)": 0.25,           # Strong risk-on
    "S&P 500 (SPX)": 0.25,           # Strong risk-on
    "Emerging Markets (EEM)": 0.20,  # Risk-on
    "Crude Oil (CL)": 0.20,          # Risk-on
    "Gold (XAU)": -0.20,             # Risk-off
    "US Dollar Index (DXY)": -0.20,  # Risk-off
    "US 10Y Treasury (IEF)": -0.20,  # Risk-off
    "Volatility Index (VIX)": -0.07  # Risk-off (inverse)
}

weights = {}
for asset in selected_assets:
    default_val = default_weights.get(asset, 0.0)
    weights[asset] = st.sidebar.number_input(
        f"Weight for {asset}",
        min_value=-1.0,
        max_value=1.0,
        value=float(default_val),
        step=0.05,
        format="%.2f"
    )

# Show weight sum
weight_sum = sum(weights.values())
st.sidebar.markdown("---")
st.sidebar.metric("Sum of Weights", f"{weight_sum:.3f}")
if abs(weight_sum) < 0.1:
    st.sidebar.warning("⚠️ Weights sum near zero - index may show little variation")

# Single stock analysis input (placeholder for future use)
st.sidebar.markdown("---")
st.sidebar.header("💹 Stock Analysis")
user_ticker = st.sidebar.text_input("Enter Stock Ticker", value="TSLA")

# ========== DATA LOADING FUNCTIONS ==========
def validate(df, name):
    if df is None or df.empty:
        st.error(f"❌ No data returned for {name}. Check ticker or date range.")
        return None
    if df.isna().all().all():
        st.error(f"❌ All values missing for {name}.")
        return None
    return df


def load_data(tickers_dict, start, end, label="Assets"):
    """Load data from Yahoo Finance with robust handling."""
    if not tickers_dict:
        st.error(f"❌ No tickers selected for {label}.")
        return pd.DataFrame()

    raw = yf.download(list(tickers_dict.values()), start=start, end=end, progress=False)

    if raw is None or raw.empty:
        st.error(f"❌ No data returned for {label}. Check tickers or date range.")
        return pd.DataFrame()

    # Handle MultiIndex vs single-level columns
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        if "Adj Close" in lvl0:
            df = raw["Adj Close"].copy()
        elif "Close" in lvl0:
            df = raw["Close"].copy()
        else:
            st.error(f"❌ No 'Adj Close' or 'Close' data found for {label}.")
            return pd.DataFrame()
    else:
        if "Adj Close" in raw.columns:
            df = raw["Adj Close"].copy()
        elif "Close" in raw.columns:
            df = raw["Close"].copy()
        else:
            df = raw.copy()

    # Rename columns back to friendly asset names
    rename_map = {}
    for name, ticker in tickers_dict.items():
        if ticker in df.columns:
            rename_map[ticker] = name
        elif name in df.columns:
            rename_map[name] = name
    df = df.rename(columns=rename_map)

    # Drop columns that are entirely NaN
    df = df.dropna(axis=1, how="all")

    if df.empty:
        st.error(f"❌ All series for {label} are empty after cleaning.")
        return pd.DataFrame()

    return df


# ========== LOAD DATA ==========
data = load_data(tickers, start_date, end_date, label="Selected Assets")

try:
    spx_raw = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
    if spx_raw is None or spx_raw.empty:
        st.error("❌ No data for S&P 500 (SPX).")
        st.stop()

    if isinstance(spx_raw.columns, pd.MultiIndex) and "Adj Close" in spx_raw.columns.get_level_values(0):
        spx_data = spx_raw["Adj Close"].squeeze()
    elif "Adj Close" in spx_raw.columns:
        spx_data = spx_raw["Adj Close"].squeeze()
    else:
        spx_data = spx_raw["Close"].squeeze()

    spx_data.name = "S&P 500 (SPX)"

except Exception as e:
    st.warning(f"⚠️ Error loading data: {e}. Please check ticker availability and date range.")
    st.stop()

if data is None or data.empty:
    st.error("❌ No valid asset data available after download.")
    st.stop()

if use_business_days:
    data = data.asfreq("B")
    if not data.empty:
        data = data.ffill()

    spx_data = spx_data.asfreq("B")
    if not spx_data.empty:
        spx_data = spx_data.ffill()

# ========== GMF INDEX CALCULATION ==========
def calculate_gmf_index(data_df, weights_dict):
    """Calculate GMF Index as weighted sum of daily percentage changes."""
    if data_df is None or data_df.empty:
        return pd.Series(dtype=float)

    daily_pct = data_df.pct_change().fillna(0)
    weights_series = pd.Series(weights_dict).reindex(data_df.columns).fillna(0.0)
    weighted_daily = daily_pct.multiply(weights_series, axis=1)
    daily_gmf_change = weighted_daily.sum(axis=1)
    gmf_index_series = (daily_gmf_change * 100).cumsum()
    return gmf_index_series


gmf_raw = calculate_gmf_index(data, weights)
if gmf_raw.empty:
    st.error("❌ GMF index could not be computed (no valid data).")
    st.stop()

# Normalize to start at 0
gmf_index = gmf_raw - gmf_raw.iloc[0]

# Create smoothed versions
money_flow_raw = gmf_index
money_flow_s = money_flow_raw.rolling(3, min_periods=1).mean()
money_flow_smooth = money_flow_raw.rolling(smooth_window, min_periods=1).mean()

# Calculate Z-Score
rolling_mean = money_flow_smooth.rolling(window=z_score_window, min_periods=5).mean()
rolling_std = money_flow_smooth.rolling(window=z_score_window, min_periods=5).std()
money_flow_zscore = (money_flow_smooth - rolling_mean) / rolling_std
money_flow_zscore = money_flow_zscore.replace([np.inf, -np.inf], 0).fillna(0)

# Calculate Momentum (30-day ROC)
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

st.markdown(
    f"""
<div style="padding:1.2em; border-radius:12px; text-align:center; background-color:{sentiment_color}; color:white; font-size:1.3em; font-weight:bold;">
{sentiment}
</div>
""",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    current_gmf = money_flow_raw.iloc[-1] if not money_flow_raw.empty else 0
    st.metric("Current GMF Index", f"{current_gmf:+.2f}")
with col2:
    st.metric(
        "Z-Score",
        f"{latest_zscore:+.2f}",
        delta="Extreme" if abs(latest_zscore) > Z_EXTREME else "Normal",
    )
with col3:
    st.metric(
        "30-Day Momentum",
        f"{latest_momentum:+.0f}%",
        delta="Accelerating" if abs(latest_momentum) > MOM_HIGH else "Stable",
    )

# ===== 2nd ROW: POSITIONING & ROTATION SNAPSHOT =====
pos_col1, pos_col2, pos_col3 = st.columns(3)

# 1) Equity allocation snapshot
with pos_col1:
    st.subheader("Equity Bias")
    try:
        positioning_data = {}

        if "Crude Oil (CL)" in data.columns and "Emerging Markets (EEM)" in data.columns:
            commod_em_ratio = data["Crude Oil (CL)"] / data["Emerging Markets (EEM)"]
            positioning_data["Commodity/EM_Ratio"] = (
                commod_em_ratio / commod_em_ratio.iloc[0] * 100
            )

        if "US 10Y Treasury (IEF)" in data.columns and "US Dollar Index (DXY)" in data.columns:
            treasury_dollar_ratio = data["US 10Y Treasury (IEF)"] / data["US Dollar Index (DXY)"]
            positioning_data["Treasury/Dollar_Ratio"] = (
                treasury_dollar_ratio / treasury_dollar_ratio.iloc[0] * 100
            )

        positioning_df = pd.DataFrame(
            {
                "Date": data.index,
                "GMF_Index": money_flow_smooth,
                "GMF_Momentum": money_flow_momentum,
                **positioning_data,
            }
        ).dropna()

        equity_allocation = None
        positioning_label = "N/A"
        sectors_label = "N/A"

        if not positioning_df.empty:
            recent_gmf = positioning_df["GMF_Index"].iloc[-20:].mean()
            recent_mom = positioning_df["GMF_Momentum"].iloc[-20:].mean()

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
            if "Commodity/EM_Ratio" in positioning_df.columns:
                recent_commod_em = positioning_df["Commodity/EM_Ratio"].iloc[-1]
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

        st.metric("Equity Allocation", f"{(equity_allocation or 0):.0f}%", positioning_label)

    except Exception:
        st.metric("Equity Allocation", "N/A", "Error")

# 2) Sector rotation snapshot
with pos_col2:
    st.subheader("Sector Tilt")
    if "latest_zscore" in locals() and "latest_momentum" in locals():
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
    try:
        if "Commodity/EM_Ratio" in positioning_df.columns:
            commod_em_latest = positioning_df["Commodity/EM_Ratio"].iloc[-1]
            commod_em_text = f"{commod_em_latest:.1f}"
        if "Treasury/Dollar_Ratio" in positioning_df.columns:
            treas_dxy_latest = positioning_df["Treasury/Dollar_Ratio"].iloc[-1]
            treas_dxy_text = f"{treas_dxy_latest:.1f}"
    except Exception:
        pass

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
        positioning_data = {}

        if "Crude Oil (CL)" in data.columns and "Emerging Markets (EEM)" in data.columns:
            commod_em_ratio = data["Crude Oil (CL)"] / data["Emerging Markets (EEM)"]
            positioning_data["Commodity/EM_Ratio"] = (
                commod_em_ratio / commod_em_ratio.iloc[0] * 100
            )

        if "US 10Y Treasury (IEF)" in data.columns and "US Dollar Index (DXY)" in data.columns:
            treasury_dollar_ratio = data["US 10Y Treasury (IEF)"] / data["US Dollar Index (DXY)"]
            positioning_data["Treasury/Dollar_Ratio"] = (
                treasury_dollar_ratio / treasury_dollar_ratio.iloc[0] * 100
            )

        positioning_df = pd.DataFrame(
            {
                "Date": data.index,
                "GMF_Index": money_flow_smooth,
                "GMF_Momentum": money_flow_momentum,
                **positioning_data,
            }
        ).dropna()

        if not positioning_df.empty:
            recent_gmf = positioning_df["GMF_Index"].iloc[-20:].mean()
            recent_mom = positioning_df["GMF_Momentum"].iloc[-20:].mean()

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
            if "Commodity/EM_Ratio" in positioning_df.columns:
                recent_commod_em = positioning_df["Commodity/EM_Ratio"].iloc[-1]
                if recent_commod_em > 110:
                    equity_allocation -= 10
                elif recent_commod_em < 90:
                    equity_allocation += 5

            equity_allocation = max(0, min(100, equity_allocation))

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

            st.markdown(
                f"""
            <div style="padding:1.2em; border-radius:12px; text-align:center; background-color:{pos_color}; color:white; font-size:1.3em; font-weight:bold;">
            Recommended Equity Allocation: {equity_allocation:.0f}%<br>
            {positioning}
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
            **Recommended Sectors:** {sectors}
            
            **Current Readings:**
            - GMF Index: {recent_gmf:+.1f}
            - GMF Momentum: {recent_mom:+.3f}/day
            """
            )

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

    if "latest_zscore" in locals() and "latest_momentum" in locals():
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

        st.markdown(
            f"""
        **Current Market Stage:** {stage}
        
        **Recommended Sectors:** {sectors}
        
        **Rationale:** {rationale}
        """
        )

# ========== GMF CHARTS ==========
st.markdown("---")
st.header("📊 GMF Visualization")

df_plot = pd.DataFrame(
    {
        "Date": money_flow_raw.index,
        "Money Flow Curve": money_flow_s,
        "Smoothed Curve": money_flow_smooth,
        "Momentum": money_flow_momentum,
        "Z-Score": money_flow_zscore,
    }
).dropna()

df_plot["Above"] = df_plot["Money Flow Curve"] > df_plot["Smoothed Curve"]

st.subheader("🌊 GMF Curves")
base = alt.Chart(df_plot).encode(
    x=alt.X("Date:T", axis=alt.Axis(format="%d/%m/%Y", title="Date"))
)
zero_line = (
    alt.Chart(pd.DataFrame({"y": [0]}))
    .mark_rule(color="gray", strokeDash=[3, 3])
    .encode(y="y")
)

curve_chart = base.mark_line(color="#1f77b4", opacity=0.6).encode(
    y=alt.Y("Money Flow Curve:Q", title="GMF Index"),
    tooltip=["Date:T", alt.Tooltip("Money Flow Curve:Q", format=".2f")],
)

smooth_chart = base.mark_line(color="#d62728", size=2).encode(
    y=alt.Y("Smoothed Curve:Q", title="GMF Index"),
    tooltip=["Date:T", alt.Tooltip("Smoothed Curve:Q", format=".2f")],
)

fill_area = base.mark_area(opacity=0.17).encode(
    y="Money Flow Curve:Q",
    y2="Smoothed Curve:Q",
    color=alt.Color(
        "Above:N",
        scale=alt.Scale(domain=[True, False], range=["green", "red"]),
        legend=None,
    ),
)

final_chart = alt.layer(zero_line, fill_area, curve_chart, smooth_chart)
st.altair_chart(final_chart, use_container_width=True)

# Momentum Chart
st.subheader("📈 GMF Momentum (30-Day Rate of Change)")
momentum_chart = (
    alt.Chart(df_plot)
    .mark_bar(opacity=0.5)
    .encode(
        x=alt.X("Date:T", axis=alt.Axis(format="%d/%m/%Y", title="Date")),
        y=alt.Y("Momentum:Q", title="Daily Rate of Change (%)"),
        color=alt.condition(
            alt.datum.Momentum > 0, alt.value("#2ca02c"), alt.value("#d62728")
        ),
        tooltip=["Date:T", alt.Tooltip("Momentum:Q", format=".3f")],
    )
)

mom_threshold_lines = (
    alt.Chart(pd.DataFrame({"y": [MOM_LOW, 0, MOM_HIGH]}))
    .mark_rule(color="gray", strokeDash=[3, 3])
    .encode(y="y")
)

final_momentum_chart = momentum_chart + mom_threshold_lines
st.altair_chart(final_momentum_chart, use_container_width=True)

# Z-Score Chart
st.subheader("📊 Climax Zone Indicator (Z-Score)")
zscore_chart = (
    alt.Chart(df_plot)
    .mark_area(opacity=0.6)
    .encode(
        x=alt.X("Date:T", axis=alt.Axis(format="%d/%m/%Y", title="Date")),
        y=alt.Y("Z-Score:Q", title="Z-Score"),
        color=alt.condition(
            alt.datum["Z-Score"] > 0, alt.value("#1f77b4"), alt.value("#d62728")
        ),
        tooltip=["Date:T", alt.Tooltip("Z-Score:Q", format=".2f")],
    )
)

z_threshold_lines = (
    alt.Chart(pd.DataFrame({"y": [-Z_EXTREME, -0.5, 0, 0.5, Z_EXTREME]}))
    .mark_rule(color="gray", strokeDash=[3, 3])
    .encode(y="y")
)

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

    cumulative_melted = (
        cumulative_contrib.reset_index()
        .melt("Date", var_name="Asset", value_name="Cumulative Contribution")
    )

    asset_chart = (
        alt.Chart(cumulative_melted)
        .mark_area(opacity=0.6)
        .encode(
            x=alt.X("Date:T", axis=alt.Axis(format="%d/%m/%Y", title="Date")),
            y="Cumulative Contribution:Q",
            color="Asset:N",
            tooltip=[
                "Date:T",
                "Asset:N",
                alt.Tooltip("Cumulative Contribution:Q", format=".2f"),
            ],
        )
        .properties(
            title="Cumulative Contribution of Each Asset to GMF Index",
            width="container",
            height=400,
        )
    )
    st.altair_chart(asset_chart, use_container_width=True)

# Correlation Matrix
with st.expander("🧠 Asset Correlation Matrix"):
    returns_corr = data.pct_change().corr()
    returns_corr.index.name = "Asset1"
    corr_melt = returns_corr.reset_index().melt(
        id_vars="Asset1", var_name="Asset2", value_name="Correlation"
    )
    corr_melt = corr_melt[corr_melt["Asset1"] != corr_melt["Asset2"]]

    heatmap = (
        alt.Chart(corr_melt)
        .mark_rect()
        .encode(
            x=alt.X("Asset1:N", title=None),
            y=alt.Y("Asset2:N", title=None),
            color=alt.Color(
                "Correlation:Q",
                scale=alt.Scale(scheme="redblue", domain=(-1, 1)),
            ),
            tooltip=[
                "Asset1",
                "Asset2",
                alt.Tooltip("Correlation:Q", format=".2f"),
            ],
        )
        .properties(title="Daily Return Correlation Heatmap", width="container", height=400)
    )

    text = (
        alt.Chart(corr_melt)
        .mark_text(baseline="middle", fontSize=10)
        .encode(
            x="Asset1:N",
            y="Asset2:N",
            text=alt.Text("Correlation:Q", format=".2f"),
            color=alt.value("black"),
        )
    )

    st.altair_chart(heatmap + text, use_container_width=True)
