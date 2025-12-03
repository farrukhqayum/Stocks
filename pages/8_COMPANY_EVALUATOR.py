import streamlit as st
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
import numpy as np
import pandas as pd
import time
# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Company Evaluator",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Company Evaluator & Hold Score")

@st.cache_data(ttl=3600, show_spinner=False)
def get_financials(ticker):
    for attempt in range(3):
        try:
            t = yf.Ticker(ticker)
            return getattr(t, 'financials', pd.DataFrame()).T
        except YFRateLimitError:
            if attempt < 2:
                time.sleep(2 ** attempt * 2)
            else:
                return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def get_cashflow(ticker):
    for attempt in range(3):
        try:
            t = yf.Ticker(ticker)
            return getattr(t, 'cashflow', pd.DataFrame()).T
        except YFRateLimitError:
            if attempt < 2:
                time.sleep(2 ** attempt * 2)
            else:
                return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def get_info(ticker):
    for attempt in range(3):
        try:
            t = yf.Ticker(ticker)
            return getattr(t, 'info', {})
        except YFRateLimitError:
            if attempt < 2:
                time.sleep(2 ** attempt * 2)
            else:
                return {}

@st.cache_data(ttl=3600, show_spinner=False)
def get_history(ticker, period="3y"):
    for attempt in range(3):
        try:
            t = yf.Ticker(ticker)
            return t.history(period=period)
        except YFRateLimitError:
            if attempt < 2:
                time.sleep(2 ** attempt * 2)
            else:
                return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def get_sp500_history(period="3y"):
    for attempt in range(3):
        try:
            return yf.download("^GSPC", period=period, progress=False)
        except YFRateLimitError:
            if attempt < 2:
                time.sleep(2 ** attempt * 2)
            else:
                return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_company_data(ticker):
    try:
        income = get_financials(ticker)
        cashflow = get_cashflow(ticker)
        info = get_info(ticker)
        stock_hist = get_history(ticker)
        sp_hist = get_sp500_history()

        # Revenue CAGR
        cagr = 0
        if not income.empty and "Total Revenue" in income.columns:
            rev = income["Total Revenue"].dropna()
            if len(rev) >= 2:
                cagr = (rev.iloc[0] / rev.iloc[-1]) ** (1/len(rev)) - 1

        # Gross margin
        margin = 0
        if (not income.empty and 
            "Gross Profit" in income.columns and 
            "Total Revenue" in income.columns):
            gp = income["Gross Profit"].dropna()
            rev = income["Total Revenue"].dropna()
            if len(gp) > 0 and len(rev) > 0:
                margin = (gp / rev).mean()

        # Free Cash Flow
        fcf = 0
        if not cashflow.empty and "Free Cash Flow" in cashflow.columns:
            fcf_values = cashflow["Free Cash Flow"].dropna()
            if len(fcf_values) > 0:
                fcf = fcf_values.mean()

        # 3-Year Performance - Extract SCALAR values
        stock_return = 0.0
        if isinstance(stock_hist, pd.DataFrame) and len(stock_hist) > 1:
            try:
                start_price = float(stock_hist["Close"].iloc[0])
                end_price = float(stock_hist["Close"].iloc[-1])
                stock_return = (end_price / start_price - 1) if start_price > 0 else 0
            except:
                stock_return = 0.0

        sp_return = 0.0
        if isinstance(sp_hist, pd.DataFrame) and len(sp_hist) > 1:
            try:
                sp_start = float(sp_hist["Close"].iloc[0])
                sp_end = float(sp_hist["Close"].iloc[-1])
                sp_return = (sp_end / sp_start - 1) if sp_start > 0 else 0
            except:
                sp_return = 0.0

        # Safe info extraction - convert to float where possible
        beta = float(info.get("beta", 0)) if info.get("beta") else None
        trailing_pe = float(info.get("trailingPE", 0)) if info.get("trailingPE") else None
        forward_pe = float(info.get("forwardPE", 0)) if info.get("forwardPE") else None
        total_debt = float(info.get("totalDebt", 0)) if info.get("totalDebt") else None
        total_assets = float(info.get("totalAssets", 0)) if info.get("totalAssets") else None
        debt_to_equity = float(info.get("debtToEquity", 0)) if info.get("debtToEquity") else None

        # Debt ratio
        debt_ratio = None
        if debt_to_equity is not None and not pd.isna(debt_to_equity):
            debt_ratio = debt_to_equity
        elif total_debt is not None and total_assets is not None and total_assets != 0:
            debt_ratio = total_debt / total_assets

        return {
            "ticker": ticker,
            "cagr": round(float(cagr) * 100, 2),
            "margin": round(float(margin) * 100, 2),
            "fcf": round(float(fcf), 2),
            "stock_3yr": round(float(stock_return) * 100, 2),
            "sp_3yr": round(float(sp_return) * 100, 2),
            "beta": beta,
            "trailing_pe": trailing_pe,
            "forward_pe": forward_pe,
            "debt_ratio": debt_ratio,
        }

    except Exception as e:
        st.error(f"Failed to process data for {ticker}: {str(e)}")
        return None

# Volatility classification
def classify_volatility(beta):
    if beta is None or pd.isna(beta):
        return "Unknown", "⚠️"
    
    beta = float(beta)
    if beta < 0.8:
        return "Stable", "✅"
    elif beta <= 1.2:
        return "Normal", "⚪"
    elif beta <= 1.8:
        return "Volatile", "⚠️"
    elif beta <= 2.1:
        return "Highly Volatile", "🔴"
    else:
        return "Emotionally Destructive", "🔴"

# FIXED Scoring system - ALL values are scalars
def calculate_hold_score(data, tailwind, leader):
    score = 0
    breakdown = {}

    # Safe numeric extraction
    cagr = float(data.get("cagr", 0))
    margin = float(data.get("margin", 0))
    fcf = float(data.get("fcf", 0))
    stock_3yr = float(data.get("stock_3yr", 0))
    sp_3yr = float(data.get("sp_3yr", 0))
    beta = float(data.get("beta", 0)) if data.get("beta") is not None else None
    trailing_pe = float(data.get("trailing_pe", 0)) if data.get("trailing_pe") is not None else None
    debt_ratio = float(data.get("debt_ratio", 0)) if data.get("debt_ratio") is not None else None

    # 1. Revenue growth
    if cagr > 15:
        score += 2
        breakdown["Revenue CAGR"] = f"{cagr}% ✅"
    elif cagr > 8:
        score += 1
        breakdown["Revenue CAGR"] = f"{cagr}% ⚠️"
    else:
        breakdown["Revenue CAGR"] = f"{cagr}% ❌"

    # 2. Gross margin
    if margin > 40:
        score += 2
        breakdown["Gross Margin"] = f"{margin}% ✅"
    elif margin > 25:
        score += 1
        breakdown["Gross Margin"] = f"{margin}% ⚠️"
    else:
        breakdown["Gross Margin"] = f"{margin}% ❌"

    # 3. Free cash flow
    if fcf > 0:
        score += 2
        breakdown["Free Cash Flow"] = f"${fcf:,.0f} ✅"
    else:
        breakdown["Free Cash Flow"] = f"${fcf:,.0f} ❌"

    # 4. 3Y vs S&P - NOW SAFE SCALAR COMPARISON
    if stock_3yr > sp_3yr:
        score += 2
        breakdown["3Y vs S&P"] = f"{stock_3yr}% > {sp_3yr}% ✅"
    else:
        breakdown["3Y vs S&P"] = f"{stock_3yr}% ≤ {sp_3yr}% ❌"

    # 5. Tailwind
    if tailwind == "Yes":
        score += 1
        breakdown["Sector Tailwind"] = "✅ Yes"
    elif tailwind == "Uncertain":
        breakdown["Sector Tailwind"] = "⚠️ Uncertain"
    else:
        breakdown["Sector Tailwind"] = "❌ No"

    # 6. Leader
    if leader == "Yes":
        score += 1
        breakdown["Market Leader"] = "✅ Yes"
    elif leader == "Uncertain":
        breakdown["Market Leader"] = "⚠️ Uncertain"
    else:
        breakdown["Market Leader"] = "❌ No"

    # 7. Beta
    vol_label, icon = classify_volatility(beta)
    breakdown["Beta / Volatility"] = f"{beta or 'N/A'} → {vol_label} {icon}"
    if beta and beta > 1.6:
        score -= 1

    # 8. P/E
    if trailing_pe is not None and trailing_pe >= 0:
        if trailing_pe < 25:
            score += 1
            breakdown["P/E (Trailing)"] = f"{trailing_pe:.1f} ✅"
        elif trailing_pe <= 40:
            breakdown["P/E (Trailing)"] = f"{trailing_pe:.1f} ⚠️"
        else:
            score -= 1
            breakdown["P/E (Trailing)"] = f"{trailing_pe:.1f} ❌"
    else:
        breakdown["P/E (Trailing)"] = "N/A ⚠️"

    # 9. Debt ratio
    if debt_ratio is not None and debt_ratio >= 0:
        if debt_ratio < 0.5:
            score += 1
            breakdown["Debt Ratio"] = f"{debt_ratio:.2f} ✅"
        elif debt_ratio <= 1.5:
            breakdown["Debt Ratio"] = f"{debt_ratio:.2f} ⚠️"
        else:
            score -= 1
            breakdown["Debt Ratio"] = f"{debt_ratio:.2f} ❌"
    else:
        breakdown["Debt Ratio"] = "N/A ⚠️"

    return score, breakdown


# ==================================================
# ================= STREAMLIT UI ===================
# ==================================================

ticker = st.text_input("Enter stock ticker (ex: AAPL, COIN, TSLA)").upper()

tailwind = st.selectbox(
    "Is industry in a long-term tailwind?",
    ["Yes", "No", "Uncertain"]
)

leader = st.selectbox(
    "Is the company an industry leader?",
    ["Yes", "No", "Uncertain"]
)

if st.button("Analyze stock"):

    if ticker == "":
        st.warning("Enter a stock symbol")
    else:
        data = get_company_data(ticker)

        if data is None:
            st.error("No data found for this ticker")
        else:

            score, breakdown = calculate_hold_score(data, tailwind, leader)

            st.subheader(f"📊 Final Hold Score: {score} / 10")

            if score >= 8:
                st.success("STRONG LONG-TERM HOLD")
            elif score >= 5:
                st.warning("CONDITIONAL HOLD - MONITOR ANNUALLY")
            else:
                st.error("NOT SUITABLE FOR LONG-TERM HOLD")

            # Volatility Highlight
            vol = breakdown.get("Beta / Volatility")
            st.subheader("⚡ Volatility Level")
            st.info(vol)

            # Details section
            st.subheader("📌 Breakdown")
            df = pd.DataFrame(breakdown.items(), columns=["Metric", "Status"])
            st.dataframe(df, height = 400, width = 500)
