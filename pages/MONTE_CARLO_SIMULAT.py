import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

@st.cache_data(ttl=1200)
def get_stock_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1),
                         interval='1d', auto_adjust=False, progress=False)
    except Exception:
        return None
    if df.empty:
        return None
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df.dropna()
    if df.empty:
        return None
    return df

def monte_carlo_simulation(start_price, tp_price, sl_price, mu, sigma, days, simulations):
    results = []
    for _ in range(simulations):
        price = start_price
        hit_tp = False
        hit_sl = False
        for day in range(days):
            price = price * np.exp(np.random.normal(mu, sigma))
            if price >= tp_price:
                hit_tp = True
                break
            if price <= sl_price:
                hit_sl = True
                break
        results.append((hit_tp, hit_sl))
    return results

def main():
    st.title("Monte Carlo Simulation on ML Predicted TP and SL")

    ticker = st.text_input("Ticker Symbol (e.g. AAPL)", value="AAPL")
    start_date = st.date_input("Start Date for Historical Data", datetime.now() - timedelta(days=365))
    end_date = st.date_input("End Date for Historical Data", datetime.now())
    predicted_tp = st.number_input("Predicted Take Profit Price", min_value=0.0, format="%.2f")
    predicted_sl = st.number_input("Predicted Stop Loss Price", min_value=0.0, format="%.2f")
    simulation_days = st.number_input("Simulation Length (days)", min_value=1, max_value=252, value=20)
    simulation_runs = st.number_input("Number of Simulations", min_value=100, max_value=10000, value=1000)

    if st.button("Run Simulation"):
        df = get_stock_data(ticker, start_date, end_date)

        if df is None:
            st.error("Failed to fetch data for the given ticker and dates.")
            return

        last_price = df['Close'].iloc[-1]
        st.write(f"Last Close Price: {last_price:.2f}")

        # Calculate daily log returns
        df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
        mu = df['LogReturn'].mean()
        sigma = df['LogReturn'].std()

        # Run Monte Carlo Simulations
        results = monte_carlo_simulation(last_price, predicted_tp, predicted_sl, mu, sigma, simulation_days, simulation_runs)

        hit_tp_count = sum(1 for r in results if r[0])
        hit_sl_count = sum(1 for r in results if r[1])
        neither_count = simulation_runs - hit_tp_count - hit_sl_count

        st.write(f"Out of {simulation_runs} simulations:")
        st.write(f"- Hit TP first: {hit_tp_count} times ({hit_tp_count / simulation_runs * 100:.2f}%)")
        st.write(f"- Hit SL first: {hit_sl_count} times ({hit_sl_count / simulation_runs * 100:.2f}%)")
        st.write(f"- Neither hit: {neither_count} times ({neither_count / simulation_runs * 100:.2f}%)")

        # Optional: Plot histogram of results
        labels = ['Hit TP', 'Hit SL', 'Neither']
        counts = [hit_tp_count, hit_sl_count, neither_count]

        fig, ax = plt.subplots()
        ax.bar(labels, counts, color=['green', 'red', 'gray'])
        ax.set_ylabel('Number of Simulations')
        ax.set_title('Monte Carlo Simulation Results')
        st.pyplot(fig)

if __name__ == "__main__":
    main()
