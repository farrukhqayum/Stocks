import streamlit as st
import yfinance as yf
import pandas as pd

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
        
        df = stock_hist.join(index_hist, lsuffix="_stock", rsuffix="_index", how="inner")

        df["stock_ret"] = df["Close_stock"].pct_change().rolling(window=21).mean()
        df["index_ret"] = df["Close_index"].pct_change().rolling(window=21).mean()

        df = df.dropna()

        cov = df["stock_ret"].cov(df["index_ret"])
        var = df["index_ret"].var()

        if var == 0:
            return None

        beta = cov / var
        return round(beta, 2)

    except Exception:
        return None

def get_company_data(ticker):
    stock = yf.Ticker(ticker)

    try:
        test = stock.history(period="1d")
        if test.empty:
            return None
    except:
        return None

    try:
        income = stock.financials.T
    except:
        income = pd.DataFrame()

    try:
        cashflow = stock.cashflow.T
    except:
        cashflow = pd.DataFrame()

    try:
        info = stock.get_info()
    except:
        info = {}

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

    try:
        gp = income["Gross Profit"]
        tr = income["Total Revenue"]
        margin = (gp / tr).mean()
    except:
        margin = 0

    try:
        fcf = cashflow["Free Cash Flow"].mean()
    except:
        fcf = 0

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

def classify_fundamental_and_speculation(data, score):
    speculative_flags = []

    strong_fund = (
        data["cagr"] >= 8 and
        data["margin"] >= 25 and
        data["fcf"] > 0
    )
    weak_fund = (
        data["cagr"] <= 0 or
        data["margin"] < 20 or
        data["fcf"] <= 0
    )

    if strong_fund:
        fundamental = "Fundamentally Strong"
    elif weak_fund:
        fundamental = "Fundamentally Weak / Speculative"
    else:
        fundamental = "Mixed Fundamentals"

    if data["stock_3yr"] > data["sp_3yr"] + 40 and weak_fund:
        speculative_flags.append("Price far ahead of fundamentals (meme/speculative behavior)")

    if data["beta"] is not None and data["beta"] >= 1.8 and weak_fund:
        speculative_flags.append("High volatility with weak fundamentals (avoid as long-term hold)")

    if not speculative_flags:
        speculative_label = "No strong meme/speculative signs"
    else:
        speculative_label = " / ".join(speculative_flags)

    return fundamental, speculative_label

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

def calculate_hold_score(data, tailwind, leader):
    score = 0
    breakdown = {}

    if data["cagr"] > 15:
        score += 2
        breakdown["Revenue CAGR"] = f"{data['cagr']}% ✅"
    elif data["cagr"] > 8:
        score += 1
        breakdown["Revenue CAGR"] = f"{data['cagr']}% ⚠️"
    else:
        breakdown["Revenue CAGR"] = f"{data['cagr']}% ❌"

    if data["margin"] > 40:
        score += 2
        breakdown["Gross Margin"] = f"{data['margin']}% ✅"
    elif data["margin"] > 25:
        score += 1
        breakdown["Gross Margin"] = f"{data['margin']}% ⚠️"
    else:
        breakdown["Gross Margin"] = f"{data['margin']}% ❌"

    if data["fcf"] > 0:
        score += 2
        breakdown["Free Cash Flow"] = f"${data['fcf']:,.0f} ✅"
    else:
        breakdown["Free Cash Flow"] = f"${data['fcf']:,.0f} ❌"

    if data["stock_3yr"] > data["sp_3yr"]:
        score += 2
        breakdown["3Y vs S&P"] = f"{data['stock_3yr']}% > {data['sp_3yr']}% ✅"
    else:
        breakdown["3Y vs S&P"] = f"{data['stock_3yr']}% ≤ {data['sp_3yr']}% ❌"

    if tailwind == "Yes":
        score += 1
        breakdown["Sector Tailwind"] = "Yes ✅"
    elif tailwind == "Uncertain":
        breakdown["Sector Tailwind"] = "Uncertain ⚠️"
    else:
        breakdown["Sector Tailwind"] = "No ❌"

    if leader == "Yes":
        score += 1
        breakdown["Market Leader"] = "Yes ✅"
    elif leader == "Uncertain":
        breakdown["Market Leader"] = "Uncertain ⚠️"
    else:
        breakdown["Market Leader"] = "No ❌"

    vol_label, icon = classify_volatility(data["beta"])
    beta_display = data['beta'] if data['beta'] is not None else "Unknown"
    breakdown["Beta / Volatility"] = f"{beta_display} → {vol_label} {icon}"

    if data["beta"] and data["beta"] > 1.6:
        score -= 1

    return score, breakdown

raw = st.text_input(
    "Enter up to 10 stock tickers, separated by commas (ex: AAPL, MSFT, TSLA)"
)
tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]

if len(tickers) > 10:
    st.warning("Only the first 10 tickers will be analyzed.")
    tickers = tickers[:10]

tailwind = st.selectbox(
    "Industry tailwind (applies to all stocks)?",
    ["Yes", "No", "Uncertain"]
)

leader = st.selectbox(
    "Company leadership status (applies to all stocks)?",
    ["Yes", "No", "Uncertain"]
)

if st.button("Analyze stocks"):
    if not tickers:
        st.warning("Enter at least one stock symbol")
    else:
        results_summary = []
        detailed_sections = []

        for ticker in tickers:
            data = get_company_data(ticker)

            if data is None:
                st.error(f"❌ No data found or invalid ticker: {ticker}")
                continue

            score, breakdown = calculate_hold_score(data, tailwind, leader)
            fundamental, speculative = classify_fundamental_and_speculation(data, score)
            vol_label, vol_icon = classify_volatility(data["beta"])

            results_summary.append({
                "Ticker": ticker,
                "Score": score,
                "Rev CAGR %": data["cagr"],
                "Gross Margin %": data["margin"],
                "FCF": data["fcf"],
                "3Y Stock %": data["stock_3yr"],
                "3Y S&P %": data["sp_3yr"],
                "Beta": data["beta"],
                "Volatility": f"{vol_label} {vol_icon}",
                "Fundamental": fundamental,
                "Speculative": speculative
            })

            detailed_sections.append((ticker, score, breakdown, fundamental, speculative))

        if not results_summary:
            st.error("No valid data found for the provided tickers.")
        else:
            st.subheader("📊 Multi-Stock Summary")

            def color_row(fundamental):
                if "Strong" in fundamental:
                    return 'style="background-color: rgba(0, 255, 0, 0.1); border-left: 4px solid rgba(40, 167, 69, 0.3);"'
                elif "Weak" in fundamental:
                    return 'style="background-color: rgba(255, 0, 0, 0.1); border-left: 4px solid rgba(220, 53, 69, 0.3);"'
                else:
                    return 'style="background-color: rgba(255, 193, 7, 0.1); border-left: 4px solid rgba(255, 193, 7, 0.3);"'
            
            html_table = """
            <table class="summary-table" style="width:100%; border-collapse: collapse; font-size: 14px;">
                <thead>
                    <tr style="background: none !important;">
            """
            df_summary = pd.DataFrame(results_summary)
            columns = df_summary.columns
            for col in columns:
                html_table += f"<th style='padding: 12px; text-align: left; border-bottom: 1px solid rgba(0,0,0,0.1); font-weight: bold;'>{col}</th>"
            html_table += "</tr></thead><tbody>"
            
            for idx, row in df_summary.iterrows():
                row_style = color_row(row['Fundamental'])
                html_table += f"<tr {row_style}>"
                for col in columns:
                    value = row[col]
                    if pd.isna(value):
                        display_value = "N/A"
                    elif isinstance(value, float):
                        display_value = f"{value:.1f}" if value % 1 != 0 else f"{int(value)}"
                    else:
                        display_value = str(value)
                    
                    if col == "FCF" and isinstance(value, (int, float)):
                        display_value = f"${value:,.0f}"
                    
                    html_table += f"<td style='padding: 12px; border-bottom: 1px solid rgba(0,0,0,0.05);'>{display_value}</td>"
                html_table += "</tr>"
            
            html_table += """
                </tbody>
            </table>
            <style>
            .summary-table th, .summary-table td {
                vertical-align: top;
            }
            .summary-table thead tr,
            .summary-table thead tr:hover {
                background: none !important;
            }
            .summary-table tbody tr:hover {
                background-color: rgba(0,0,0,0.03) !important;
            }
            </style>
            """
            
            st.markdown(html_table, unsafe_allow_html=True)


            for ticker, score, breakdown, fundamental, speculative in detailed_sections:
                st.markdown("---")
                st.subheader(f"🔍 Details for {ticker}")
                st.markdown(f"**Final Hold Score:** {score} / 10")

                if score >= 8:
                    st.success("STRONG LONG-TERM HOLD")
                elif score >= 5:
                    st.warning("CONDITIONAL HOLD — Monitor Annually")
                else:
                    st.error("NOT SUITABLE FOR LONG-TERM HOLD")

                st.subheader("⚡ Volatility Level")
                st.info(breakdown["Beta / Volatility"])

                st.subheader("🧬 Company Type")
                st.info(f"**{fundamental}**")
                st.warning(f"**{speculative}**")

                st.subheader("📌 Breakdown")
                df_breakdown = pd.DataFrame(breakdown.items(), columns=["Metric", "Status"])
                st.table(df_breakdown)
