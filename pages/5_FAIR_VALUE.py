import streamlit as st
import yfinance as yf

st.title("Fair Value Estimator")

ticker = st.text_input("Enter stock ticker", "AAPL")
margin_of_safety = st.slider("Margin of Safety (%)", 10, 50, 30) / 100

if ticker:
    stock = yf.Ticker(ticker)
    cashflow = stock.cashflow  # quarterly or annual cash flow
    capex = stock.cashflow.loc['Capital Expenditures'][0]
    shares_outstanding = stock.info.get('sharesOutstanding', None)
    
    if cashflow is not None and capex is not None and shares_outstanding:
        owner_earnings = cashflow.iloc[0] + capex.iloc[0]  # approximate net cash flow
        fair_value = owner_earnings / shares_outstanding
        fair_value_discounted = fair_value * (1 - margin_of_safety)
        
        st.write(f"Owner Earnings per Share (approx): ${fair_value:,.2f}")
        st.write(f"Fair Value after {int(margin_of_safety * 100)}% Margin of Safety: ${fair_value_discounted:,.2f}")
    else:
        st.write("Fundamental data unavailable for this ticker.")
