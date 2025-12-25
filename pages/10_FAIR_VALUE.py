import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ============================================
# 1️⃣ User inputs and Assumptions
# ============================================
target_ticker = st.sidebar.text_input("Enter Ticker Symbol", "COIN")
industry_peers = st.sidebar.multiselect("Select Industry Peers",
                                        ["COIN", "CME", "ICE", "HOOD", "MKTX", "NDAQ"],
                                        default=["COIN", "CME", "ICE", "HOOD", "MKTX", "NDAQ"])
rolling_eps_years = st.sidebar.slider("Years for Normalized EPS", 1, 10, 5)
forecast_years = st.sidebar.slider("DCF Forecast Period (Years)", 3, 10, 5)

st.sidebar.subheader("Financial Projection Assumptions")
revenue_growth_rate = st.sidebar.slider("Revenue Growth Rate (%)", 0.0, 0.20, 0.05, 0.01)
operating_margin = st.sidebar.slider("Operating Margin (%)", 0.0, 0.30, 0.15, 0.01)
tax_rate = st.sidebar.slider("Tax Rate (%)", 0.0, 0.50, 0.21, 0.01)
capex_as_pct_revenue = st.sidebar.slider("CapEx as % of Revenue (%)", 0.0, 0.10, 0.02, 0.005)
depreciation_as_pct_revenue = st.sidebar.slider("Depreciation as % of Revenue (%)", 0.0, 0.10, 0.015, 0.005)
nwc_as_pct_revenue = st.sidebar.slider("NWC as % of Revenue (%)", 0.0, 0.20, 0.10, 0.01)
perpetual_growth_rate = st.sidebar.slider("Perpetual Growth Rate (%)", 0.0, 0.05, 0.02, 0.005)

st.sidebar.subheader("WACC Assumptions")
risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.01, 0.05, 0.03, 0.005)
market_risk_premium = st.sidebar.slider("Market Risk Premium (%)", 0.03, 0.07, 0.05, 0.005)
cost_of_debt = st.sidebar.slider("Cost of Debt (%)", 0.03, 0.10, 0.06, 0.005)

# Convert percentages to decimals
revenue_growth_rate /= 100
operating_margin /= 100
tax_rate /= 100
capex_as_pct_revenue /= 100
depreciation_as_pct_revenue /= 100
nwc_as_pct_revenue /= 100
perpetual_growth_rate /= 100
risk_free_rate /= 100
market_risk_premium /= 100
cost_of_debt /= 100

# ============================================
# 2️⃣ Helper Functions
# ============================================
def get_financials(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    pe = info.get("trailingPE", np.nan)
    eps = info.get("trailingEps", np.nan)
    fcf = info.get("freeCashflow", np.nan)
    revenue_growth = info.get("revenueGrowth", 0)
    op_margin = info.get("operatingMargins", 0)
    beta = info.get("beta", 1)
    debt_to_equity = info.get("debtToEquity", 0)
    return {
        "ticker": ticker,
        "PE": pe,
        "EPS": eps,
        "FCF": fcf,
        "RevenueGrowth": revenue_growth,
        "OpMargin": op_margin,
        "Beta": beta,
        "DebtEquity": debt_to_equity
    }

def compute_adjustment(row, industry_df):
    z_growth = (row['RevenueGrowth'] - industry_df['RevenueGrowth'].mean()) / industry_df['RevenueGrowth'].std() if industry_df['RevenueGrowth'].std() != 0 else 0
    z_margin = (row['OpMargin'] - industry_df['OpMargin'].mean()) / industry_df['OpMargin'].std() if industry_df['OpMargin'].std() != 0 else 0
    z_fcf = (row['FCF'] - industry_df['FCF'].mean()) / industry_df['FCF'].std() if industry_df['FCF'].std() != 0 else 0
    z_leverage = (row['DebtEquity'] - industry_df['DebtEquity'].mean()) / industry_df['DebtEquity'].std() if industry_df['DebtEquity'].std() != 0 else 0
    z_volatility = (row['Beta'] - industry_df['Beta'].mean()) / industry_df['Beta'].std() if industry_df['Beta'].std() != 0 else 0

    adj_factor = 0.35 * z_growth + 0.25 * z_margin + 0.2 * z_fcf - 0.1 * z_volatility - 0.1 * z_leverage
    return adj_factor

def get_normalized_eps(ticker, years=5):
    hist = yf.Ticker(ticker).financials.T
    if 'Diluted EPS' in hist.columns:
        eps_series = hist['Diluted EPS']
    else:
        eps_series = hist['Net Income'] / hist['Diluted Average Shares']

    eps_norm = eps_series.tail(years).mean()
    return eps_norm

def get_last_historical_value(df, metric_name, default_value=0):
    if metric_name in df.index and not df.iloc[:, 0].isna().all():
        value = df.loc[metric_name].iloc[0]
        if pd.isna(value):
            return default_value
        return value
    return default_value

# ============================================
# 3️⃣ Data Fetching and Calculations
# ============================================
st.header(f"Valuation Analysis for {target_ticker}")

# --- Industry-Anchored Fair Value ---
with st.spinner("Fetching industry data..."):
    peer_data = pd.DataFrame([get_financials(t) for t in industry_peers])
    peer_data = peer_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['PE','EPS'])
    industry_median_PE = peer_data["PE"].median()

    peer_data['AdjFactor'] = peer_data.apply(lambda x: compute_adjustment(x, peer_data), axis=1)
    peer_data['Adj_PE'] = industry_median_PE * (1 + peer_data['AdjFactor'])
    peer_data['Adj_PE'] = peer_data['Adj_PE'].clip(lower=industry_median_PE*0.6, upper=industry_median_PE*1.6)

    peer_data['Norm_EPS'] = peer_data['ticker'].apply(lambda t: get_normalized_eps(t, rolling_eps_years))
    peer_data['FairValue'] = peer_data['Norm_EPS'] * peer_data['Adj_PE']

    relative_valuation_sigma = peer_data['PE'].std() / industry_median_PE
    price_sigma = 0.1
    band_weight = 0.6 * relative_valuation_sigma + 0.4 * price_sigma

    peer_data['FairUpper'] = peer_data['FairValue'] * (1 + band_weight)
    peer_data['FairLower'] = peer_data['FairValue'] * (1 - band_weight)

ticker_row = peer_data[peer_data['ticker']==target_ticker].iloc[0]
current_price = yf.Ticker(target_ticker).history(period="1d", auto_adjust=True)['Close'].iloc[0]

# --- DCF Intrinsic Value ---
with st.spinner("Performing DCF analysis..."):
    stock_data = yf.Ticker(target_ticker)
    income_statement = stock_data.financials.copy()
    balance_sheet = stock_data.balance_sheet.copy()
    cash_flow = stock_data.cashflow.copy()

    last_historical_year = income_statement.columns[0] # Assuming most recent year is first column

    hist_revenue = get_last_historical_value(income_statement, 'Total Revenue')
    hist_op_income = get_last_historical_value(income_statement, 'Operating Income')
    hist_net_income = get_last_historical_value(income_statement, 'Net Income')
    hist_cash = get_last_historical_value(balance_sheet, 'Cash And Cash Equivalents', default_value=0)
    hist_total_assets = get_last_historical_value(balance_sheet, 'Total Assets', default_value=0)
    hist_total_liabilities = get_last_historical_value(balance_sheet, 'Total Liabilities Net Minority Interest', default_value=0)
    hist_equity = get_last_historical_value(balance_sheet, 'Total Equity Gross Minority Interest', default_value=0)
    hist_capex = get_last_historical_value(cash_flow, 'Capital Expenditure', default_value=0)
    hist_depreciation = get_last_historical_value(cash_flow, 'Depreciation And Amortization', default_value=0)

    projected_income_statement = {}
    projected_balance_sheet = {}
    projected_cash_flow = {}

    # Projections
    for year in range(1, forecast_years + 1):
        current_year_proj = last_historical_year.year + year
        projected_income_statement[current_year_proj] = {}
        projected_balance_sheet[current_year_proj] = {}
        projected_cash_flow[current_year_proj] = {}

        # Income Statement
        if year == 1:
            projected_income_statement[current_year_proj]['Total Revenue'] = hist_revenue * (1 + revenue_growth_rate)
        else:
            projected_income_statement[current_year_proj]['Total Revenue'] = projected_income_statement[current_year_proj - 1]['Total Revenue'] * (1 + revenue_growth_rate)

        projected_income_statement[current_year_proj]['Operating Income'] = projected_income_statement[current_year_proj]['Total Revenue'] * operating_margin
        projected_income_statement[current_year_proj]['EBT'] = projected_income_statement[current_year_proj]['Operating Income']
        projected_income_statement[current_year_proj]['Tax Expense'] = projected_income_statement[current_year_proj]['EBT'] * tax_rate
        projected_income_statement[current_year_proj]['Net Income'] = projected_income_statement[current_year_proj]['EBT'] - projected_income_statement[current_year_proj]['Tax Expense']

        # Balance Sheet (partial for NWC and CapEx)
        projected_balance_sheet[current_year_proj]['Capital Expenditure'] = projected_income_statement[current_year_proj]['Total Revenue'] * capex_as_pct_revenue
        projected_balance_sheet[current_year_proj]['Depreciation'] = projected_income_statement[current_year_proj]['Total Revenue'] * depreciation_as_pct_revenue
        projected_balance_sheet[current_year_proj]['Net Working Capital'] = projected_income_statement[current_year_proj]['Total Revenue'] * nwc_as_pct_revenue

        # Cash Flow Statement
        net_income = projected_income_statement[current_year_proj]['Net Income']
        depreciation = projected_balance_sheet[current_year_proj]['Depreciation']
        capex = projected_balance_sheet[current_year_proj]['Capital Expenditure']

        current_nwc = projected_balance_sheet[current_year_proj]['Net Working Capital']
        if year == 1:
            hist_nwc_value = hist_revenue * nwc_as_pct_revenue if hist_revenue != 0 else 0
            change_in_nwc = current_nwc - hist_nwc_value
        else:
            previous_nwc = projected_balance_sheet[current_year_proj - 1]['Net Working Capital']
            change_in_nwc = current_nwc - previous_nwc

        projected_cash_flow[current_year_proj]['Net Income'] = net_income
        projected_cash_flow[current_year_proj]['Depreciation'] = depreciation
        projected_cash_flow[current_year_proj]['Change in Net Working Capital'] = -change_in_nwc # Increase in NWC is a use of cash
        projected_cash_flow[current_year_proj]['Operating Cash Flow'] = net_income + depreciation - change_in_nwc
        projected_cash_flow[current_year_proj]['Capital Expenditures'] = -capex # CapEx is a use of cash
        projected_cash_flow[current_year_proj]['Investing Cash Flow'] = -capex
        projected_cash_flow[current_year_proj]['Financing Cash Flow'] = 0 # Simplified

        net_change_in_cash = projected_cash_flow[current_year_proj]['Operating Cash Flow'] + \
                             projected_cash_flow[current_year_proj]['Investing Cash Flow'] + \
                             projected_cash_flow[current_year_proj]['Financing Cash Flow']
        projected_cash_flow[current_year_proj]['Net Change in Cash'] = net_change_in_cash

        if year == 1:
            projected_balance_sheet[current_year_proj]['Cash'] = hist_cash + net_change_in_cash
        else:
            projected_balance_sheet[current_year_proj]['Cash'] = projected_balance_sheet[current_year_proj - 1]['Cash'] + net_change_in_cash

    # Convert projections to DataFrames for easier access and display
    projected_income_statement_df = pd.DataFrame(projected_income_statement)
    projected_balance_sheet_df = pd.DataFrame(projected_balance_sheet)
    projected_cash_flow_df = pd.DataFrame(projected_cash_flow)

    # Calculate FCFF
    projected_fcff = {}
    for year_fcff in range(1, forecast_years + 1):
        current_year_fcff = last_historical_year.year + year_fcff
        ebit = projected_income_statement[current_year_fcff]['Operating Income']
        depreciation = projected_cash_flow[current_year_fcff]['Depreciation']
        capex_cf_impact = projected_cash_flow[current_year_fcff]['Capital Expenditures'] # Already negative
        change_in_nwc_cf_impact = projected_cash_flow[current_year_fcff]['Change in Net Working Capital'] # Already negative if cash outflow

        fcff = ebit * (1 - tax_rate) + depreciation + capex_cf_impact + change_in_nwc_cf_impact
        projected_fcff[current_year_fcff] = fcff
    projected_fcff_df = pd.DataFrame(list(projected_fcff.items()), columns=['Year', 'FCFF'])
    projected_fcff_df.set_index('Year', inplace=True)

    # Calculate Terminal Value
    last_fcff = projected_fcff_df['FCFF'].iloc[-1]
    wacc_placeholder = 0.10 # This will be overwritten by calculated WACC
    terminal_value = (last_fcff * (1 + perpetual_growth_rate)) / (wacc_placeholder - perpetual_growth_rate)

    # Determine WACC
    target_beta = peer_data[peer_data['ticker'] == target_ticker]['Beta'].iloc[0]
    cost_of_equity = risk_free_rate + target_beta * market_risk_premium

    total_debt = hist_total_liabilities
    total_equity = hist_equity
    total_capital = total_debt + total_equity

    if total_capital <= 0:
        debt_proportion = 0.5
        equity_proportion = 0.5
    else:
        debt_proportion = total_debt / total_capital
        equity_proportion = total_equity / total_capital

    wacc = (equity_proportion * cost_of_equity) + (debt_proportion * cost_of_debt * (1 - tax_rate))

    # Recalculate Terminal Value with actual WACC
    if wacc > perpetual_growth_rate:
        terminal_value = (last_fcff * (1 + perpetual_growth_rate)) / (wacc - perpetual_growth_rate)
    else:
        terminal_value = np.nan
        st.warning("Warning: WACC must be greater than perpetual growth rate for Terminal Value calculation.")

    # Calculate Intrinsic Value (DCF)
    present_values_fcff = []
    last_hist_year_int = last_historical_year.year

    for year_pv, fcff_value in projected_fcff.items():
        year_number = year_pv - last_hist_year_int
        discount_factor = 1 / ((1 + wacc) ** year_number)
        present_value = fcff_value * discount_factor
        present_values_fcff.append(present_value)

    discount_factor_tv = 1 / ((1 + wacc) ** forecast_years)
present_value_terminal_value = terminal_value * discount_factor_tv

total_intrinsic_value = sum(present_values_fcff) + present_value_terminal_value

if 'Ordinary Shares Number' in balance_sheet.index:
    total_outstanding_shares = balance_sheet.loc['Ordinary Shares Number'].iloc[0]
else:
    stock_info = yf.Ticker(target_ticker).info
    total_outstanding_shares = stock_info.get('sharesOutstanding', 100_000_000)
    if total_outstanding_shares <=0:
        total_outstanding_shares = 100_000_000

intrinsic_value_per_share = total_intrinsic_value / total_outstanding_shares

# ============================================
# 4️⃣ Plotting and Summary
# ============================================

st.subheader("Valuation Comparison")

# Get fair value components from ticker_row
fair_value_center = ticker_row['FairValue']
fair_value_lower = ticker_row['FairLower']
fair_value_upper = ticker_row['FairUpper']

# Fetch historical prices for plotting
stock_hist = yf.download(target_ticker, period="1y", auto_adjust=True)['Close']

fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(stock_hist.index, stock_hist.values, label=f"{target_ticker} Price", color='blue', linewidth=2)

# Annotate intrinsic and fair value on the chart
ax.axhline(intrinsic_value_per_share, color='red', linestyle='-.', label='DCF Intrinsic Value')
ax.axhline(fair_value_center, color='green', linestyle='--', label='Industry Fair Value')
ax.fill_between(stock_hist.index, fair_value_lower, fair_value_upper, color='green', alpha=0.1, label='Industry Fair Value Band')

# Current price line for visual comparison
ax.axhline(current_price, color='purple', linestyle=':', label='Current Market Price')

ax.set_title(f"{target_ticker} Valuation Comparison (1-Year Historical Price)", fontsize=16)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Price ($)", fontsize=12)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.6)

# Add stats as a text box on the plot
stats_text = (
    f"Current Price: ${current_price:,.2f}\n"
    f"DCF Intrinsic Value: ${intrinsic_value_per_share:,.2f}\n"
    f"Industry Fair Value: ${fair_value_center:,.2f}\n"
    f"Fair Value Band: ${fair_value_lower:,.2f} - ${fair_value_upper:,.2f}\n"
    f"Price / Fair Value (Industry): {current_price / fair_value_center:,.2f}\n"
    f"Price / Fair Value (DCF): {current_price / intrinsic_value_per_share:,.2f}"
)
fig.text(0.15, 0.95, stats_text, horizontalalignment='left', verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8, ec="lightgray"), fontsize=10)

fig.tight_layout(rect=[0, 0, 1, 0.9]) # Adjust layout to make space for text box
st.pyplot(fig)

st.subheader("Valuation Metrics")
st.write(f"**Current Price:** ${current_price:,.2f}")
st.write(f"**DCF Intrinsic Value:** ${intrinsic_value_per_share:,.2f}")
st.write(f"**Industry Fair Value:** ${fair_value_center:,.2f}")
st.write(f"**Industry Fair Value Band:** ${fair_value_lower:,.2f} - ${fair_value_upper:,.2f}")
st.write(f"**Price / Fair Value (Industry):** {current_price / fair_value_center:,.2f}")
st.write(f"**Price / Fair Value (DCF):** {current_price / intrinsic_value_per_share:,.2f}")

# ============================================
# 5️⃣ Important 5-Year Projections
# ============================================
st.subheader("Key 5-Year Financial Projections")

st.write("**Income Statement Projections (Millions $)**")
st.dataframe(projected_income_statement_df.loc[['Total Revenue', 'Net Income']].apply(lambda x: x / 1_000_000).round(2))

st.write("**Balance Sheet Projections (Millions $)**")
st.dataframe(projected_balance_sheet_df.loc[['Cash', 'Capital Expenditure', 'Net Working Capital']].apply(lambda x: x / 1_000_000).round(2))

st.write("**Cash Flow Projections (Millions $)**")
st.dataframe(projected_cash_flow_df.loc[['Operating Cash Flow', 'Capital Expenditures', 'Net Change in Cash']].apply(lambda x: x / 1_000_000).round(2))

st.write("**Free Cash Flow to Firm (Millions $)**")
st.dataframe(projected_fcff_df.apply(lambda x: x / 1_000_000).round(2))
