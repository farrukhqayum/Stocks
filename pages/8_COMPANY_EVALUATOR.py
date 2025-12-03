import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Company Evaluator",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Company Evaluator & Hold Score")


# ----------------------------
# DATA FETCH
# ----------------------------
def get_company_data(ticker):
    try:
        stock = yf.Ticker(ticker)

        income = stock.financials.T
        cashflow = stock.cashflow.T
        info = stock.info
        sp500 = yf.Ticker("^GSPC")

        # Revenue CAGR
        if "Total Revenue" in income:
            rev = income["Total Revenue"].dropna()
            if len(rev) >= 2:
                cagr = (rev.iloc[0] / rev.iloc[-1]) ** (1/len(rev)) - 1
            else:
                cagr = 0
        else:
            cagr = 0

        # Gross margin
        if "Gross Profit" in income and "Total Revenue" in income:
            margin = (income["Gross Profit"] / income["Total Revenue"]).mean()
        else:
            margin = 0

        # Free Cash Flow
        if "Free Cash Flow" in cashflow:
            fcf = cashflow["Free Cash Flow"].mean()
        else:
            fcf = 0

        # 3-Year Performance
        stock_hist = stock.history(period="3y")
        sp_hist = sp500.history(period="3y")

        if len(stock_hist) > 0 and len(sp_hist) > 0:
            stock_return = (stock_hist["Close"][-1] / stock_hist["Close"][0]) - 1
            sp_return = (sp_hist["Close"][-1] / sp_hist["Close"][0]) - 1
        else:
            stock_return = 0
            sp_return = 0

        # Beta
        beta = info.get("beta", None)

        # P/E ratios
        trailing_pe = info.get("trailingPE", None)
        forward_pe = info.get("forwardPE", None)

        # Debt metrics (simple proxies)
        total_debt = info.get("totalDebt", None)            # latest total debt
        total_assets = info.get("totalAssets", None)        # sometimes available
        debt_to_equity = info.get("debtToEquity", None)     # Yahoo’s D/E if present

        # Prefer Yahoo’s D/E; else compute debt/assets if both present
        if debt_to_equity is not None:
            debt_ratio = debt_to_equity
        elif total_debt is not None and total_assets:
            debt_ratio = total_debt / total_assets if total_assets != 0 else None
        else:
            debt_ratio = None

        return {
            "ticker": ticker,
            "cagr": round(cagr * 100, 2),
            "margin": round(margin * 100, 2),
            "fcf": round(fcf, 2),
            "stock_3yr": round(stock_return * 100, 2),
            "sp_3yr": round(sp_return * 100, 2),
            "beta": beta,
            "trailing_pe": trailing_pe,
            "forward_pe": forward_pe,
            "debt_ratio": debt_ratio,
        }

    except Exception:
        return None



# ----------------------------
# VOLATILITY CLASSIFICATION
# ----------------------------
def classify_volatility(beta):

    if beta is None:
        return "Unknown", "⚠️"

    if beta < 0.8:
        return "Stable", "✅"
    elif beta <= 1.2:
        return "Normal", "⚪"
    elif beta <= 1.8:
        return "Volatile", "⚠️"
    elif beta <=2.1:
        return "Highly Volatile", "🔴"
    else:
        return "Emotionally Destructive", "🔴"


# ----------------------------
# SCORING SYSTEM
# ----------------------------
def calculate_hold_score(data, tailwind, leader):

    score = 0
    breakdown = {}

    # 1. Revenue growth
    if data["cagr"] > 15:
        score += 2
        breakdown["Revenue CAGR"] = f"{data['cagr']}% ✅"
    elif data["cagr"] > 8:
        score += 1
        breakdown["Revenue CAGR"] = f"{data['cagr']}% ⚠️"
    else:
        breakdown["Revenue CAGR"] = f"{data['cagr']}% ❌"

    # 2. Gross margin
    if data["margin"] > 40:
        score += 2
        breakdown["Gross Margin"] = f"{data['margin']}% ✅"
    elif data["margin"] > 25:
        score += 1
        breakdown["Gross Margin"] = f"{data['margin']}% ⚠️"
    else:
        breakdown["Gross Margin"] = f"{data['margin']}% ❌"

    # 3. Free cash flow
    if data["fcf"] > 0:
        score += 2
        breakdown["Free Cash Flow"] = f"${data['fcf']:,.0f} ✅"
    else:
        breakdown["Free Cash Flow"] = f"${data['fcf']:,.0f} ❌"

    # 4. 3Y performance vs S&P
    if data["stock_3yr"] > data["sp_3yr"]:
        score += 2
        breakdown["3Y vs S&P"] = f"{data['stock_3yr']}% > {data['sp_3yr']}% ✅"
    else:
        breakdown["3Y vs S&P"] = f"{data['stock_3yr']}% ≤ {data['sp_3yr']}% ❌"

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

    # 7. Beta (Volatility)
    vol_label, icon = classify_volatility(data["beta"])
    breakdown["Beta / Volatility"] = f"{data['beta']} → {vol_label} {icon}"
    if data["beta"] and data["beta"] > 1.6:
        score -= 1

    # 8. Valuation (P/E)
    pe = data.get("trailing_pe")
    if pe is not None:
        if pe < 25:
            score += 1
            breakdown["P/E (Trailing)"] = f"{pe:.1f} ✅"
        elif pe <= 40:
            breakdown["P/E (Trailing)"] = f"{pe:.1f} ⚠️"
        else:
            score -= 1
            breakdown["P/E (Trailing)"] = f"{pe:.1f} ❌"
    else:
        breakdown["P/E (Trailing)"] = "N/A ⚠️"

    # 9. Leverage (Debt ratio)
    dr = data.get("debt_ratio")
    if dr is not None:
        if dr < 0.5:
            score += 1
            breakdown["Debt Ratio (D/E or Debt/Assets)"] = f"{dr:.2f} ✅"
        elif dr <= 1.5:
            breakdown["Debt Ratio (D/E or Debt/Assets)"] = f"{dr:.2f} ⚠️"
        else:
            score -= 1
            breakdown["Debt Ratio (D/E or Debt/Assets)"] = f"{dr:.2f} ❌"
    else:
        breakdown["Debt Ratio (D/E or Debt/Assets)"] = "N/A ⚠️"

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
