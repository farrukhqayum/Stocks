import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

st.header("Establish a MONTHLY GOAL!!!")
st.markdown("""
If you plan to compound, you have to be a disciplined trader either bi-monthly, quarterly, or monthly.
You need to track how much profit you need to achieve to compound effectively.
Below is a strategic projection of your potential growth.
""")

def calculate_investment_growth(P, r, months, max_trades):
    investment_curves = {}
    for n_trades in range(1, max_trades + 1):
        investment_values = [P]
        for m in range(1, months + 1):
            monthly_profit = investment_values[-1] * (r * n_trades)
            new_value = investment_values[-1] + monthly_profit
            investment_values.append(round(new_value))
        investment_curves[n_trades] = investment_values
    return investment_curves

def create_investment_dataframe(investment_curves, months):
    month_range = np.arange(0, months + 1)
    df = pd.DataFrame(investment_curves, index=month_range)
    df.index.name = 'Period'
    return df.reset_index()

# User inputs
P = st.number_input("Initial Investment ($)", min_value=0.0, value=1000.0, step=100.0)
r = st.number_input("Profit Rate per Trade (%)", min_value=0.0, value=3.75, step=0.01) / 100.0
months = st.number_input("Number of Periods", min_value=1, value=12, step=1)
max_trades = st.number_input("Maximum Wins per Period", min_value=1, value=7, step=1)
eff_monthly = ((1 + r) ** max_trades - 1) * 100

if st.button("Calculate Investment Growth"):
    st.subheader(f"Effective Win Rate per Period: {eff_monthly:.2f}%")
    
    investment_curves = calculate_investment_growth(P, r, months, max_trades)
    df_investments = create_investment_dataframe(investment_curves, months)
    st.dataframe(df_investments)

    # Reshape data for Altair line chart: Investment Growth per number of wins
    df_melted = df_investments.melt(id_vars=['Period'], var_name='Wins Per Period', value_name='Investment Value')

    # Investment Growth line chart
    growth_chart = alt.Chart(df_melted).mark_line(point=True).encode(
        x=alt.X('Period:O', title='Period (e.g. Months)'),
        y=alt.Y('Investment Value:Q', title='Investment Value ($)', scale=alt.Scale(zero=False)),
        color=alt.Color('Wins Per Period:N',
                        legend=alt.Legend(
                            orient='top-left',
                            legendX=20,
                            legendY=20,
                            fillColor='white',
                            strokeColor='black',
                            padding=5,
                            cornerRadius=3
                        )),
        tooltip=['Period', 'Wins Per Period', 'Investment Value']
    ).properties(
        width=700,
        height=400,
        title=f'Investment Growth @ {r*100:.2f}% Profit Rate'
    ).interactive()

    st.altair_chart(growth_chart, use_container_width=True)

    # Calculate average dollar gain per period (per month)
    df_investments_sorted = df_investments.sort_v('Period').set_index('Period')
    avg_dollar_gain_per_period = df_investments_sorted.diff().mean(axis=1).reset_index(name='Avg Dollar Gain')

    # Average Dollar Gain line chart
    avg_gain_chart = alt.Chart(avg_dollar_gain_per_period).mark_line(point=True, color='crimson').encode(
        x=alt.X('Period:O', title='Period (e.g. Months)'),
        y=alt.Y('Avg Dollar Gain:Q', title='Average Dollar Gain ($)'),
        tooltip=['Period', alt.Tooltip('Avg Dollar Gain', format=',.2f')]
    ).properties(
        width=700,
        height=300,
        title='Average Dollar Gain per Period'
    ).interactive()

    st.altair_chart(avg_gain_chart, use_container_width=True)
