import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import altair as alt

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

def monte_carlo_simulation(
    start_price, tp_price, sl_price, mu, sigma, days, simulations
):
    simulation_results = []
    hit_tp_counts = 0
    hit_sl_counts = 0
    neither_counts = 0

    for _ in range(simulations):
        price = start_price
        path = []
        hit_tp = False
        hit_sl = False
        for day in range(days):
            price = price * np.exp(np.random.normal(mu, sigma))
            path.append({"Day": day, "Price": price})
            if price >= tp_price:
                hit_tp = True
                break
            if price <= sl_price:
                hit_sl = True
                break

        simulation_results.extend(path)

        if hit_tp:
            hit_tp_counts += 1
        elif hit_sl:
            hit_sl_counts += 1
        else:
            neither_counts += 1

    return hit_tp_counts, hit_sl_counts, neither_counts, pd.DataFrame(simulation_results)

def main():
    st.title("Monte Carlo Simulation with Altair Chart")

    ticker = st.text_input("Ticker Symbol (e.g. AAPL)", value="AAPL")
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
    end_date = st.date_input("End Date", datetime.now())
    predicted_tp = st.number_input("Predicted Take Profit Price", min_value=0.0, format="%.2f")
    predicted_sl = st.number_input("Predicted Stop Loss Price", min_value=0.0, format="%.2f")
    simulation_days = st.number_input(
        "Simulation Length (days)", min_value=1, max_value=252, value=20
    )
    simulation_runs = st.number_input(
        "Number of Simulations", min_value=100, max_value=100000, value=1000
    )

    if st.button("Run Simulation"):
        df = get_stock_data(ticker, start_date, end_date)
        if df is None:
            st.error("Failed to fetch data for the given ticker and dates.")
            return

        last_price = df["Close"].iloc[-1]
        st.write(f"Last Close Price of {ticker}: {last_price:.2f}")

        df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
        mu = df["LogReturn"].mean()
        sigma = df["LogReturn"].std()

        hit_tp_count, hit_sl_count, neither_count, sim_df = monte_carlo_simulation(
            last_price,
            predicted_tp,
            predicted_sl,
            mu,
            sigma,
            simulation_days,
            simulation_runs,
        )

        st.write(f"Out of {simulation_runs} simulations:")
        st.write(
            f"- Hit TP ${predicted_tp} first: {hit_tp_count} times ({hit_tp_count / simulation_runs * 100:.2f}%)"
        )
        st.write(
            f"- Hit SL ${predicted_sl} first: {hit_sl_count} times ({hit_sl_count / simulation_runs * 100:.2f}%)"
        )
        st.write(
            f"- Neither hit: {neither_count} times ({neither_count / simulation_runs * 100:.2f}%)"
        )

        # Prepare data for histogram
        hist_data = pd.DataFrame({
            "Outcome": ["Hit TP", "Hit SL", "Neither"],
            "Count": [hit_tp_count, hit_sl_count, neither_count]
        })
        
        # Create Altair bar chart
        hist_chart = alt.Chart(hist_data).mark_bar().encode(
            x=alt.X('Outcome:N', title='Simulation Outcome'),
            y=alt.Y('Count:Q', title='Number of Simulations'),
            color=alt.Color('Outcome:N', scale=alt.Scale(domain=["Hit TP", "Hit SL", "Neither"], range=["green", "red", "gray"]))
        ).properties(
            width=400,
            height=300,
            title="Monte Carlo Simulation Outcome Counts"
        )
        
        # Display in Streamlit
        st.altair_chart(hist_chart, use_container_width=True)


        # Limit paths to 50 for performance
        sim_subset = sim_df.groupby(sim_df.index // simulation_days).head(simulation_days).copy()
        sim_subset["Simulation"] = (sim_subset.index // simulation_days) + 1
        sim_subset = sim_subset[sim_subset["Simulation"] <= 50]

        # Altair line chart for simulation paths
        line_chart = (
            alt.Chart(sim_subset)
            .mark_line(opacity=0.3)
            .encode(
                x="Day:Q",
                y="Price:Q",
                detail="Simulation:N",
                color=alt.value("turquoise"),
                tooltip=["Day:Q", "Price:Q", "Simulation:N"],
            )
            .properties(width=700, height=400)
        )

        # Horizontal lines for TP and SL
        tp_line = (
            alt.Chart(pd.DataFrame({"y": [predicted_tp]}))
            .mark_rule(color="green", size=2, strokeDash=[5, 5])
            .encode(y="y:Q")
        )
        sl_line = (
            alt.Chart(pd.DataFrame({"y": [predicted_sl]}))
            .mark_rule(color="red", size=2, strokeDash=[5, 5])
            .encode(y="y:Q")
        )

        st.altair_chart(line_chart + tp_line + sl_line, use_container_width=True)


if __name__ == "__main__":
    main()
