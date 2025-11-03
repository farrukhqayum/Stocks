import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from streamlit_lightweight_charts import renderLightweightCharts

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
    hit_tp_counts = 0
    hit_sl_counts = 0
    neither_counts = 0
    sim_paths = []

    for _ in range(simulations):
        price = start_price
        path = []
        hit_tp = False
        hit_sl = False
        for day in range(days):
            price = price * np.exp(np.random.normal(mu, sigma))
            path.append({'time': str(day), 'value': price})
            if price >= tp_price:
                hit_tp = True
                break
            if price <= sl_price:
                hit_sl = True
                break

        sim_paths.append(path)

        if hit_tp:
            hit_tp_counts += 1
        elif hit_sl:
            hit_sl_counts += 1
        else:
            neither_counts += 1

    return hit_tp_counts, hit_sl_counts, neither_counts, sim_paths

def main():
    st.title("Monte Carlo Simulation with Lightweight Financial Chart")

    ticker = st.text_input("Ticker Symbol (e.g. AAPL)", value="AAPL")
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
    end_date = st.date_input("End Date", datetime.now())
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

        df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
        mu = df['LogReturn'].mean()
        sigma = df['LogReturn'].std()

        hit_tp_count, hit_sl_count, neither_count, sim_paths = monte_carlo_simulation(
            last_price, predicted_tp, predicted_sl, mu, sigma, simulation_days, simulation_runs)

        st.write(f"Out of {simulation_runs} simulations:")
        st.write(f"- Hit TP first: {hit_tp_count} times ({hit_tp_count / simulation_runs * 100:.2f}%)")
        st.write(f"- Hit SL first: {hit_sl_count} times ({hit_sl_count / simulation_runs * 100:.2f}%)")
        st.write(f"- Neither hit: {neither_count} times ({neither_count / simulation_runs * 100:.2f}%)")

        # Flatten simulation paths for overlay plotting
        overlay_series = []
        for path in sim_paths[:50]:  # limit to first 50 paths for performance
            overlay_series.append({
                "type": "Line",
                "data": path,
                "options": {
                    "lineColor": "rgba(0, 150, 136, 0.3)",
                    "lineWidth": 1,
                },
            })

        # Add horizontal lines for TP and SL
        markers = [
            {"time": str(0), "position": "aboveBar", "color": "green", "shape": "circle", "text": "TP"},
            {"time": str(simulation_days), "position": "belowBar", "color": "red", "shape": "circle", "text": "SL"}
        ]
        series = overlay_series + [
            {
                "type": "Baseline",
                "data": [
                    {"time": str(i), "value": predicted_tp} for i in range(simulation_days + 1)
                ],
                "options": {"topLineColor": "green", "bottomLineColor": "transparent"},
                "markers": [markers[0]],
            },
            {
                "type": "Baseline",
                "data": [
                    {"time": str(i), "value": predicted_sl} for i in range(simulation_days + 1)
                ],
                "options": {"topLineColor": "red", "bottomLineColor": "transparent"},
                "markers": [markers[1]],
            }
        ]

        chart_options = {
            "height": 400,
            "layout": {"background": {"type": "solid", "color": "#fff"}, "textColor": "#000"},
            "timeScale": {"timeVisible": True},
            "grid": {"vertLines": {"color": "#eee"}, "horzLines": {"color": "#eee"}},
            "crosshair": {"mode": 1},
        }

        renderLightweightCharts(
            [{
                "chart": chart_options,
                "series": series
            }],
            key="simulation_chart"
        )

if __name__ == "__main__":
    main()
