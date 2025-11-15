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

def calculate_investment_growth(P, r, months, max_trades, tax_rate=0.0):
    investment_curves = {}
    for n_trades in range(1, max_trades + 1):
        investment_values = [P]
        for m in range(1, months + 1):
            previous_value = investment_values[-1]
            monthly_profit = previous_value * (r * n_trades)
            gross_value = previous_value + monthly_profit
            fee = gross_value * tax_rate
            new_value = gross_value - fee
            investment_values.append(round(new_value))
        investment_curves[n_trades] = investment_values
    return investment_curves

def create_investment_dataframe(investment_curves, months):
    month_range = np.arange(0, months + 1)
    df = pd.DataFrame(investment_curves, index=month_range)
    df.index.name = 'Period'
    return df.reset_index()

# User inputs
P = st.number_input("Initial Investment ($)", min_value=0.0, value=1000.0, max_value=10000000.0, step=100.0)
r = st.number_input("Profit Rate per Trade (%)", min_value=0.0, value=3.75, max_value=50.0, step=0.01) / 100.0
fee = st.number_input("Fee/Tax Rate Per Period (%)", min_value=0.0, value=1, max_value=50.0, step=1) / 100.0
months = st.number_input("Number of Periods", min_value=1, value=12, max_value=50, step=1)
max_trades = st.number_input("Maximum Wins per Period", min_value=1, value=7, max_value=20, step=1)
eff_monthly = ((1 + r) ** max_trades - 1) * 100

if st.button("Calculate Investment Growth"):
    st.subheader(f"Effective Win Rate per Period: {eff_monthly:.2f}%")

    investment_curves = calculate_investment_growth(P, r, months, max_trades, fee)
    df_investments = create_investment_dataframe(investment_curves, months)

    # Reshape data for Altair line chart: Investment Growth per number of wins
    df_melted = df_investments.melt(id_vars=['Period'], var_name='Wins Per Period', value_name='Investment Value')

    # Investment Growth line chart with y-axis on the right
    growth_chart = alt.Chart(df_melted).mark_line(point=True, size=0.5).encode(
        x=alt.X('Period:O', title='Period (e.g. Months)'),
        y=alt.Y('Investment Value:Q', title='Investment Value ($)', axis=alt.Axis(orient='right', format='~s', labelExpr="replace(datum.label, 'G', 'B')"), scale=alt.Scale(zero=False)),
        color=alt.Color(
            'Wins Per Period:N',
            legend=alt.Legend(
                orient='top-left',
                legendX=20,
                legendY=20,
                labelColor='white',
                titleColor='white',
                padding=5,
                cornerRadius=3
            )
        ),
        tooltip=[
            'Period', 
            'Wins Per Period', 
            alt.Tooltip('Investment Value', format='~s')
        ]

    ).properties(
        width=700,
        height=400,
        title=f'Investment Growth @ {r*100:.2f}% Profit Rate'
    )

    st.altair_chart(growth_chart, use_container_width=True)

    # Calculate average dollar gain per period (per month)
    df_investments_sorted = df_investments.sort_values('Period').set_index('Period')
    avg_dollar_gain_per_period = df_investments_sorted.diff().mean(axis=1).reset_index(name='Avg Dollar Gain')

    # Conditional annotation filtering based on number of periods
    if months >= 13 and months <= 24:
        annotations_df = avg_dollar_gain_per_period[avg_dollar_gain_per_period['Period'] % 2 == 0]
    elif months > 24:
        annotations_df = avg_dollar_gain_per_period[avg_dollar_gain_per_period['Period'] % 3 == 0]
    else:
        annotations_df = avg_dollar_gain_per_period

    base = alt.Chart(annotations_df).encode(
        x=alt.X('Period:O', title='Period (e.g. Months)'),
        y=alt.Y('Avg Dollar Gain:Q', title='Average Dollar Gain ($)',
                axis=alt.Axis(orient='right', format='~s', labelExpr="replace(datum.label, 'G', 'B')")),
        tooltip=['Period', alt.Tooltip('Avg Dollar Gain', format='~s')]
    )

    line = base.mark_line(
        point=alt.OverlayMarkDef(filled=True, fill='white', size=10),
        color='green'
    )
    text = base.mark_text(
        align='center',
        baseline='bottom',
        dy=-8,
        color='white',
        fontWeight='bold'
    ).encode(
        text=alt.Text('Avg Dollar Gain:Q', format='.2s')
    )

    avg_gain_chart = (line + text).properties(
        width=700,
        height=300,
        title='Average Dollar Gain per Period'
    )

    st.altair_chart(avg_gain_chart, use_container_width=True)
