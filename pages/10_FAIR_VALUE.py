import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
import time

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
revenue_growth_rate = st.sidebar.slider("Revenue Growth Rate (%)", 0.0, 20.0, 5.0, 1.0) / 100
operating_margin = st.sidebar.slider("Operating Margin (%)", 0.0, 30.0, 15.0, 1.0) / 100
tax_rate = st.sidebar.slider("Tax Rate (%)", 0.0, 50.0, 21.0, 1.0) / 100
capex_as_pct_revenue = st.sidebar.slider("CapEx as % of Revenue (%)", 0.0, 10.0, 2.0, 0.5) / 100
depreciation_as_pct_revenue = st.sidebar.slider("Depreciation as % of Revenue (%)", 0.0, 10.0, 1.5, 0.5) / 100
nwc_as_pct_revenue = st.sidebar.slider("NWC as % of Revenue (%)", 0.0, 20.0, 10.0, 1.0) / 100
perpetual_growth_rate = st.sidebar.slider("Perpetual Growth Rate (%)", 0.0, 5.0, 2.0, 0.5) / 100

st.sidebar.subheader("WACC Assumptions")
risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 1.0, 5.0, 3.0, 0.5) / 100
market_risk_premium = st.sidebar.slider("Market Risk Premium (%)", 3.0, 7.0, 5.0, 0.5) / 100
cost_of_debt = st.sidebar.slider("Cost of Debt (%)", 3.0, 10.0, 6.0, 0.5) / 100


# ============================================
# 2️⃣ Helper Functions with Proper Error Handling
# ============================================

def safe_get_info(stock, key, default=np.nan):
    """Safely get info with fallback"""
    try:
        return stock.info.get(key, default)
    except Exception:
        return default

def safe_get_financials(stock):
    """Safely fetch financials with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            time.sleep(0.5)  # Rate limiting
            financials = stock.financials.T
            return financials
        except Exception as e:
            if attempt == max_retries - 1:
                st.warning(f"Could not fetch financials after {max_retries} attempts")
                return pd.DataFrame()
            time.sleep(1)
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_and_process_peer_data(ticker, rolling_eps_years):
    """Fetch peer data with comprehensive error handling"""
    try:
        stock = yf.Ticker(ticker)
        
        # Safely get info
        pe = safe_get_info(stock, "trailingPE", np.nan)
        eps_trailing = safe_get_info(stock, "trailingEps", np.nan)
        fcf = safe_get_info(stock, "freeCashflow", np.nan)
        revenue_growth = safe_get_info(stock, "revenueGrowth", 0)
        op_margin = safe_get_info(stock, "operatingMargins", 0)
        beta = safe_get_info(stock, "beta", 1)
        debt_to_equity = safe_get_info(stock, "debtToEquity", 0)

        # Normalized EPS calculation
        eps_norm = eps_trailing  # Default fallback
        try:
            financials_df = safe_get_financials(stock)
            if not financials_df.empty:
                if 'Diluted EPS' in financials_df.columns:
                    eps_series = financials_df['Diluted EPS'].dropna()
                elif 'Net Income' in financials_df.columns:
                    eps_series = financials_df['Net Income'].dropna()
                    if 'Diluted Average Shares' in financials_df.columns:
                        shares = financials_df['Diluted Average Shares'].dropna()
                        eps_series = eps_series / shares
                else:
                    eps_series = pd.Series([eps_trailing])
                
                if len(eps_series) > 0:
                    eps_norm = eps_series.tail(rolling_eps_years).mean()
        except Exception as e:
            st.warning(f"Could not calculate normalized EPS for {ticker}: {str(e)}")

        return {
            "ticker": ticker,
            "PE": pe,
            "EPS": eps_trailing,
            "FCF": fcf,
            "RevenueGrowth": revenue_growth,
            "OpMargin": op_margin,
            "Beta": beta,
            "DebtEquity": debt_to_equity,
            "Norm_EPS": eps_norm
        }
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def compute_adjustment(row, industry_df):
    """Compute adjustment factor with safe division"""
    def safe_z_score(value, mean, std):
        if pd.isna(value) or pd.isna(mean) or pd.isna(std) or std == 0:
            return 0
        return (value - mean) / std
    
    z_growth = safe_z_score(row['RevenueGrowth'], industry_df['RevenueGrowth'].mean(), industry_df['RevenueGrowth'].std())
    z_margin = safe_z_score(row['OpMargin'], industry_df['OpMargin'].mean(), industry_df['OpMargin'].std())
    z_fcf = safe_z_score(row['FCF'], industry_df['FCF'].mean(), industry_df['FCF'].std())
    z_leverage = safe_z_score(row['DebtEquity'], industry_df['DebtEquity'].mean(), industry_df['DebtEquity'].std())
    z_volatility = safe_z_score(row['Beta'], industry_df['Beta'].mean(), industry_df['Beta'].std())

    adj_factor = 0.35 * z_growth + 0.25 * z_margin + 0.2 * z_fcf - 0.1 * z_volatility - 0.1 * z_leverage
    return adj_factor


def get_last_historical_value(df, metric_name, default_value=0):
    """Safely get last historical value"""
    try:
        if metric_name in df.index and not df.iloc[:, 0].isna().all():
            value = df.loc[metric_name].iloc[0]
            if pd.isna(value):
                return default_value
            return value
    except Exception:
        pass
    return default_value

@st.cache_data(ttl=3600)
def get_historical_stock_data(ticker, period="1y"):
    """Fetch historical data with error handling"""
    try:
        time.sleep(0.3)  # Rate limiting
        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if 'Close' in data.columns:
            return data['Close']
        return data
    except Exception as e:
        st.error(f"Error fetching historical data for {ticker}: {str(e)}")
        return pd.Series()

@st.cache_data(ttl=3600)
def fetch_dcf_base_data(ticker):
    """Fetch DCF base data with error handling"""
    try:
        stock = yf.Ticker(ticker)
        time.sleep(0.5)  # Rate limiting
        
        income_statement = stock.financials.copy() if hasattr(stock, 'financials') else pd.DataFrame()
        time.sleep(0.3)
        balance_sheet = stock.balance_sheet.copy() if hasattr(stock, 'balance_sheet') else pd.DataFrame()
        time.sleep(0.3)
        cash_flow = stock.cashflow.copy() if hasattr(stock, 'cashflow') else pd.DataFrame()
        
        return income_statement, balance_sheet, cash_flow
    except Exception as e:
        st.error(f"Error fetching DCF data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# ============================================
# 3️⃣ Data Fetching and Calculations
# ============================================
st.header(f"Valuation Analysis for {target_ticker}")

# --- Industry-Anchored Fair Value ---
with st.spinner("Fetching industry data..."):
    peer_processed_data_list = []
    for i, t in enumerate(industry_peers):
        with st.spinner(f"Fetching {t} ({i+1}/{len(industry_peers)})..."):
            data = fetch_and_process_peer_data(t, rolling_eps_years)
            if data is not None:
                peer_processed_data_list.append(data)
    
    if not peer_processed_data_list:
        st.error("Could not fetch any peer data. Please try again later.")
        st.stop()
    
    peer_data = pd.DataFrame(peer_processed_data_list)
    
    # Clean data
    peer_data = peer_data.replace([np.inf, -np.inf], np.nan)
    valid_peers = peer_data.dropna(subset=['PE','EPS'])
    
    if len(valid_peers) == 0:
        st.error("No valid peer data available. Please try different tickers or try again later.")
        st.stop()
    
    peer_data = valid_peers
    industry_median_PE = peer_data["PE"].median()

    peer_data['AdjFactor'] = peer_data.apply(lambda x: compute_adjustment(x, peer_data), axis=1)
    peer_data['Adj_PE'] = industry_median_PE * (1 + peer_data['AdjFactor'])
    peer_data['Adj_PE'] = peer_data['Adj_PE'].clip(lower=industry_median_PE*0.6, upper=industry_median_PE*1.6)

    peer_data['FairValue'] = peer_data['Norm_EPS'] * peer_data['Adj_PE']

    relative_valuation_sigma = peer_data['PE'].std() / industry_median_PE if peer_data['PE'].std() > 0 else 0.1
    price_sigma = 0.1
    band_weight = 0.6 * relative_valuation_sigma + 0.4 * price_sigma

    peer_data['FairUpper'] = peer_data['FairValue'] * (1 + band_weight)
    peer_data['FairLower'] = peer_data['FairValue'] * (1 - band_weight)

# Get target ticker data
target_data = peer_data[peer_data['ticker']==target_ticker]
if target_data.empty:
    st.error(f"Could not fetch data for {target_ticker}. Please check the ticker symbol.")
    st.stop()

ticker_row = target_data.iloc[0]

# Get current price
stock_hist = get_historical_stock_data(target_ticker, period="1d")
if stock_hist.empty:
    st.error(f"Could not fetch current price for {target_ticker}")
    st.stop()

current_price = float(stock_hist.iloc[-1])

# --- DCF Intrinsic Value ---
with st.spinner("Performing DCF analysis..."):
    income_statement, balance_sheet, cash_flow = fetch_dcf_base_data(target_ticker)
    
    if income_statement.empty or balance_sheet.empty or cash_flow.empty:
        st.warning("Some financial statements are missing. DCF calculation may be limited.")
        intrinsic_value_per_share = np.nan
    else:
        try:
            last_historical_year = income_statement.columns[0]

            hist_revenue = get_last_historical_value(income_statement, 'Total Revenue')
            hist_op_income = get_last_historical_value(income_statement, 'Operating Income')
            hist_net_income = get_last_historical_value(income_statement, 'Net Income')
            hist_cash = get_last_historical_value(balance_sheet, 'Cash And Cash Equivalents', default_value=0)
            hist_total_assets = get_last_historical_value(balance_sheet, 'Total Assets', default_value=0)
            hist_total_liabilities = get_last_historical_value(balance_sheet, 'Total Liabilities Net Minority Interest', default_value=0)
            hist_equity = get_last_historical_value(balance_sheet, 'Total Equity Gross Minority Interest', default_value=0)
            hist_capex = get_last_historical_value(cash_flow, 'Capital Expenditure', default_value=0)
            hist_depreciation = get_last_historical_value(cash_flow, 'Depreciation And Amortization', default_value=0)

            if hist_revenue == 0:
                st.warning("Historical revenue is zero. Using default assumptions.")
                hist_revenue = 1_000_000_000  # Default 1B revenue

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

                # Balance Sheet
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
                projected_cash_flow[current_year_proj]['Change in Net Working Capital'] = -change_in_nwc
                projected_cash_flow[current_year_proj]['Operating Cash Flow'] = net_income + depreciation - change_in_nwc
                projected_cash_flow[current_year_proj]['Capital Expenditures'] = -capex
                projected_cash_flow[current_year_proj]['Investing Cash Flow'] = -capex
                projected_cash_flow[current_year_proj]['Financing Cash Flow'] = 0

                net_change_in_cash = projected_cash_flow[current_year_proj]['Operating Cash Flow'] + \
                                     projected_cash_flow[current_year_proj]['Investing Cash Flow'] + \
                                     projected_cash_flow[current_year_proj]['Financing Cash Flow']
                projected_cash_flow[current_year_proj]['Net Change in Cash'] = net_change_in_cash

                if year == 1:
                    projected_balance_sheet[current_year_proj]['Cash'] = hist_cash + net_change_in_cash
                else:
                    projected_balance_sheet[current_year_proj]['Cash'] = projected_balance_sheet[current_year_proj - 1]['Cash'] + net_change_in_cash

            # Convert projections to DataFrames
            projected_income_statement_df = pd.DataFrame(projected_income_statement)
            projected_balance_sheet_df = pd.DataFrame(projected_balance_sheet)
            projected_cash_flow_df = pd.DataFrame(projected_cash_flow)

            # Calculate FCFF
            projected_fcff = {}
            for year_fcff in range(1, forecast_years + 1):
                current_year_fcff = last_historical_year.year + year_fcff
                ebit = projected_income_statement[current_year_fcff]['Operating Income']
                depreciation = projected_cash_flow[current_year_fcff]['Depreciation']
                capex_cf_impact = projected_cash_flow[current_year_fcff]['Capital Expenditures']
                change_in_nwc_cf_impact = projected_cash_flow[current_year_fcff]['Change in Net Working Capital']

                fcff = ebit * (1 - tax_rate) + depreciation + capex_cf_impact + change_in_nwc_cf_impact
                projected_fcff[current_year_fcff] = fcff
            projected_fcff_df = pd.DataFrame(list(projected_fcff.items()), columns=['Year', 'FCFF'])
            projected_fcff_df.set_index('Year', inplace=True)

            # Calculate Terminal Value and WACC
            target_beta = ticker_row['Beta']
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

            last_fcff = projected_fcff_df['FCFF'].iloc[-1]
            if wacc > perpetual_growth_rate:
                terminal_value = (last_fcff * (1 + perpetual_growth_rate)) / (wacc - perpetual_growth_rate)
            else:
                terminal_value = 0
                st.warning("Warning: WACC must be greater than perpetual growth rate for Terminal Value calculation.")

            # Calculate Intrinsic Value
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

            # Get shares outstanding
            try:
                if 'Ordinary Shares Number' in balance_sheet.index:
                    total_outstanding_shares = balance_sheet.loc['Ordinary Shares Number'].iloc[0]
                else:
                    stock_info = yf.Ticker(target_ticker).info
                    total_outstanding_shares = stock_info.get('sharesOutstanding', 100_000_000)
                
                if pd.isna(total_outstanding_shares) or total_outstanding_shares <= 0:
                    total_outstanding_shares = 100_000_000
            except Exception:
                total_outstanding_shares = 100_000_000

            intrinsic_value_per_share = total_intrinsic_value / total_outstanding_shares
            
        except Exception as e:
            st.error(f"Error in DCF calculation: {str(e)}")
            intrinsic_value_per_share = np.nan

# ============================================
# 4️⃣ Plotting and Summary
# ============================================

st.subheader("Valuation Comparison")

fair_value_center = ticker_row['FairValue']
fair_value_lower = ticker_row['FairLower']
fair_value_upper = ticker_row['FairUpper']

# Fetch historical prices for plotting
stock_hist_year = get_historical_stock_data(target_ticker, period="1y")

if not stock_hist_year.empty:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(stock_hist_year.index, stock_hist_year.values, label=f"{target_ticker} Price", color='blue', linewidth=2)

    if not pd.isna(intrinsic_value_per_share):
        ax.axhline(intrinsic_value_per_share, color='red', linestyle='-.', label='DCF Intrinsic Value')
    
    ax.axhline(fair_value_center, color='green', linestyle='--', label='Industry Fair Value')
    ax.fill_between(stock_hist_year.index, fair_value_lower, fair_value_upper, color='green', alpha=0.1, label='Industry Fair Value Band')
    ax.axhline(current_price, color='purple', linestyle=':', label='Current Market Price')

    ax.set_title(f"{target_ticker} Valuation Comparison (1-Year Historical Price)", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price ($)", fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Stats text
    if not pd.isna(intrinsic_value_per_share):
        stats_text = (
            f"Current Price: ${current_price:,.2f}\n"
            f"DCF Intrinsic Value: ${intrinsic_value_per_share:,.2f}\n"
            f"Industry Fair Value: ${fair_value_center:,.2f}\n"
            f"Fair Value Band: ${fair_value_lower:,.2f} - ${fair_value_upper:,.2f}\n"
            f"Price / Fair Value (Industry): {current_price / fair_value_center:.2f}\n"
            f"Price / Fair Value (DCF): {current_price / intrinsic_value_per_share:.2f}"
        )
    else:
        stats_text = (
            f"Current Price: ${current_price:,.2f}\n"
            f"Industry Fair Value: ${fair_value_center:,.2f}\n"
            f"Fair Value Band: ${fair_value_lower:,.2f} - ${fair_value_upper:,.2f}\n"
            f"Price / Fair Value (Industry): {current_price / fair_value_center:.2f}"
        )
    
    fig.text(0.15, 0.95, stats_text, horizontalalignment='left', verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8, ec="lightgray"), fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.9])
    st.pyplot(fig)

st.subheader("Valuation Metrics")
st.write(f"**Current Price:** ${current_price:,.2f}")
if not pd.isna(intrinsic_value_per_share):
    st.write(f"**DCF Intrinsic Value:** ${intrinsic_value_per_share:,.2f}")
    st.write(f"**Price / Fair Value (DCF):** {current_price / intrinsic_value_per_share:.2f}")
else:
    st.write("**DCF Intrinsic Value:** Not available")
st.write(f"**Industry Fair Value:** ${fair_value_center:,.2f}")
st.write(f"**Industry Fair Value Band:** ${fair_value_lower:,.2f} - ${fair_value_upper:,.2f}")
st.write(f"**Price / Fair Value (Industry):** {current_price / fair_value_center:.2f}")

# ============================================
# 5️⃣ Financial Projections
# ============================================
if not pd.isna(intrinsic_value_per_share):
    st.subheader("Key 5-Year Financial Projections")

    st.write("**Income Statement Projections (Millions $)**")
    st.dataframe(projected_income_statement_df.loc[['Total Revenue', 'Net Income']].apply(lambda x: x / 1_000_000).round(2))

    st.write("**Balance Sheet Projections (Millions $)**")
    st.dataframe(projected_balance_sheet_df.loc[['Cash', 'Capital Expenditure', 'Net Working Capital']].apply(lambda x: x / 1_000_000).round(2))

    st.write("**Cash Flow Projections (Millions $)**")
    st.dataframe(projected_cash_flow_df.loc[['Operating Cash Flow', 'Capital Expenditures', 'Net Change in Cash']].apply(lambda x: x / 1_000_000).round(2))

    st.write("**Free Cash Flow to Firm (Millions $)**")
    st.dataframe(projected_fcff_df.apply(lambda x: x / 1_000_000).round(2))
