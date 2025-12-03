import streamlit as st
import yfinance as yf
import pandas as pd

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Company Evaluator",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Company Strength & Hold Score")

def calculate_beta(stock_ticker, benchmark_ticker="^GSPC", period="5y"):
    try:
        stock = yf.Ticker(stock_ticker)
        index = yf.Ticker(benchmark_ticker)

        stock_hist = stock.history(period=period)[["Close"]]
        index_hist = index.history(period=period)[["Close"]]

        if stock_hist.empty or index_hist.empty:
            return None
        
        # Align dates
        df = stock_hist.join(index_hist, lsuffix="_stock", rsuffix="_index", how="inner")

        # Daily returns
        df["stock_ret"] = df["Close_stock"].pct_change().rolling(window=30).mean()
        df["index_ret"] = df["Close_index"].pct_change().rolling(window=30).mean()

        df = df.dropna()

        # Compute beta
        cov = df["stock_ret"].cov(df["index_ret"])
        var = df["index_ret"].var()

        if var == 0:
            return None

        beta = cov / var
        return round(beta, 2)

    except Exception:
        return None

# ----------------------------
# SAFE DATA FETCHING FUNCTION
# ----------------------------

def get_company_data(ticker):
    stock = yf.Ticker(ticker)

    # Validate ticker
    try:
        test = stock.history(period="1d")
        if test.empty:
            return None
    except:
        return None

    # Try loading financial statements
    try:
        income = stock.financials.T
    except:
        income = pd.DataFrame()

    try:
        cashflow = stock.cashflow.T
    except:
        cashflow = pd.DataFrame()

    # Load company info safely
    try:
        info = stock.get_info()
    except:
        info = {}

    # ----------------------------
    # Revenue CAGR
    # ----------------------------
    try:
        rev = income.get("Total Revenue", pd.Series()).dropna()

        if len(rev) >= 2:
            first = rev.iloc[-1]
            last = rev.iloc[0]
            years = len(rev) - 1
            if first > 0 and last > 0:
                cagr = (last / first) ** (1 / years) - 1
            else:
                cagr = 0
        else:
            cagr = 0
    except:
        cagr = 0

    # ----------------------------
    # Gross Margin
    # ----------------------------
    try:
        gp = income["Gross Profit"]
        tr = income["Total Revenue"]
        margin = (gp / tr).mean()
    except:
        margin = 0

    # ----------------------------
    # Free Cash Flow
    # ----------------------------
    try:
        fcf = cashflow["Free Cash Flow"].mean()
    except:
        fcf = 0

    # ----------------------------
    # 5-Year Performance vs S&P 500
    # ----------------------------
    try:
        hist = stock.history(period="3y")
        sp = yf.Ticker("^GSPC").history(period="3y")

        if not hist.empty and not sp.empty:
            stock_return = (hist["Close"][-1] / hist["Close"][0]) - 1
            sp_return = (sp["Close"][-1] / sp["Close"][0]) - 1
        else:
            stock_return = 0
            sp_return = 0
    except:
        stock_return = 0
        sp_return = 0

    # ----------------------------
    # Beta (from info)
    # ----------------------------
    beta = calculate_beta(ticker)

    return {
        "ticker": ticker,
        "cagr": round(cagr * 100, 2),
        "margin": round(margin * 100, 2),
        "fcf": round(fcf, 2),
        "stock_3yr": round(stock_return * 100, 2),
        "sp_3yr": round(sp_return * 100, 2),
        "beta": beta
    }


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
    elif beta <= 2.1:
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
        breakdown["Sector Tailwind"] = "Yes ✅"
    elif tailwind == "Uncertain":
        breakdown["Sector Tailwind"] = "Uncertain ⚠️"
    else:
        breakdown["Sector Tailwind"] = "No ❌"

    # 6. Leader
    if leader == "Yes":
        score += 1
        breakdown["Market Leader"] = "Yes ✅"
    elif leader == "Uncertain":
        breakdown["Market Leader"] = "Uncertain ⚠️"
    else:
        breakdown["Market Leader"] = "No ❌"

    # 7. Beta (Volatility)
    vol_label, icon = classify_volatility(data["beta"])
    beta_display = data['beta'] if data['beta'] is not None else "Unknown"
    breakdown["Beta / Volatility"] = f"{beta_display} → {vol_label} {icon}"

    # Optional penalty
    if data["beta"] and data["beta"] > 1.6:
        score -= 1

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
            st.error("❌ No data found or invalid ticker")
        else:
            score, breakdown = calculate_hold_score(data, tailwind, leader)

            st.subheader(f"📊 Final Hold Score: {score} / 10")

            if score >= 8:
                st.success("STRONG LONG-TERM HOLD")
            elif score >= 5:
                st.warning("CONDITIONAL HOLD — Monitor Annually")
            else:
                st.error("NOT SUITABLE FOR LONG-TERM HOLD")

            # Volatility Highlight
            st.subheader("⚡ Volatility Level")
            st.info(breakdown["Beta / Volatility"])

            # Breakdown Table
            st.subheader("📌 Breakdown")
            df = pd.DataFrame(breakdown.items(), columns=["Metric", "Status"])
            st.table(df)
