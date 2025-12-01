import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd

def get_company_data(ticker):
    try:
        stock = yf.Ticker(ticker)

        income = stock.financials.T
        cashflow = stock.cashflow.T
        info = stock.info
        sp500 = yf.Ticker("^GSPC")

        # Revenue CAGR (last 5 yrs)
        rev = income["Total Revenue"].dropna()
        if len(rev) >= 2:
            cagr = (rev[-1] / rev[0]) ** (1/len(rev)) - 1
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

        # 3-year return
        stock_hist = stock.history(period="3y")
        sp_hist = sp500.history(period="3y")

        stock_return = (stock_hist["Close"][-1] / stock_hist["Close"][0]) - 1
        sp_return = (sp_hist["Close"][-1] / sp_hist["Close"][0]) - 1

        # ✅ Get beta
        beta = info.get("beta", None)

        return {
            "ticker": ticker,
            "cagr": round(cagr * 100,2),
            "margin": round(margin * 100,2),
            "fcf": round(fcf,2),
            "stock_3yr": round(stock_return * 100,2),
            "sp_3yr": round(sp_return * 100,2),
            "beta": beta
        }

    except:
        return None

def classify_volatility(beta):

    if beta is None:
        return "Unknown", "⚠️"

    if beta < 0.8:
        return "Stable", "✅"
    elif beta <= 1.2:
        return "Normal", "⚪"
    elif beta <= 1.6:
        return "Risky", "⚠️"
    else:
        return "Very Volatile", "🔴"

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
        breakdown["Sector tailwind"] = "✅ Yes"
    elif tailwind == "Uncertain":
        summary = "⚠️ Uncertain"
        breakdown["Sector tailwind"] = summary
    else:
        breakdown["Sector tailwind"] = "❌ No"

    # 6. Leader
    if leader == "Yes":
        score += 1
        breakdown["Market leader"] = "✅ Yes"
    elif leader == "Uncertain":
        breakdown["Market leader"] = "⚠️ Uncertain"
    else:
        breakdown["Market leader"] = "❌ No"

    from scoring import classify_volatility

    vol_label, icon = classify_volatility(data["beta"])
    breakdown["Beta / Volatility"] = f"{data['beta']}  →  {vol_label} {icon}"
    vol = breakdown.get("Beta / Volatility")

    st.subheader("⚡ Volatility Level")
    st.info(vol)
    return score, breakdown


st.set_page_config(page_title="10-Year Hold Analyzer", layout="wide")
st.title("🤖 Company Quality & Hold Analyzer")

ticker = st.text_input("Enter stock ticker (ex: AAPL, COIN, TSLA)").upper()

tailwind = st.selectbox("Is industry in a long-term tailwind?",
                        ["Yes", "No", "Uncertain"])

leader = st.selectbox("Is the company an industry leader?",
                       ["Yes", "No", "Uncertain"])

if st.button("Analyze stock"):

    if ticker == "":
        st.warning("Enter a stock symbol")
    else:
        data = get_company_data(ticker)

        if data is None:
            st.error("No data found")
        else:
            score, breakdown = calculate_hold_score(data, tailwind, leader)

            st.subheader(f"✅ Final Hold Score: {score} / 10")

            if score >= 8:
                st.success("STRONG LONG-TERM HOLD")
            elif score >= 5:
                st.warning("CONDITIONAL HOLD - MONITOR ANNUALLY")
            else:
                st.error("NOT SUITABLE FOR LONG-TERM HOLD")

            st.subheader("📌 Details")
            for k, v in breakdown.items():
                st.write(f"**{k}:** {v}")
