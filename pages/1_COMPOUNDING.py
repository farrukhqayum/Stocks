import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.cache_data.clear()
st.cache_resource.clear()

st.header("Just Keep Winning!!!")
comp_text = """
Compounding is the process where the returns you earn are reinvested to generate their own returns. 
This effect causes your capital to grow exponentially over time, not just linearly.
Even small percentage gains consistently accumulated can turn modest initial capital into significant wealth.
Keep winning trades and staying disciplined to harness the power of compounding — patience and persistence are key to long-term trading success.
Remember, consistent small wins build up to large gains as profits generate more profits.
"""
st.markdown(comp_text)

# Streamlit inputs
with st.form(key='compound_form'):
    initial_capital = st.number_input("Initial Capital ($)", min_value=0.0, value=10000.0, step=100.0)
    win_pct = st.number_input("Avg. Win (%)", min_value=0.0, value=3.75, step=0.1) / 100.0
    tax_pct_input = st.number_input("Tax/Fee (%)", min_value=0.0, value=0.0, step=0.1)
    tax_rate = tax_pct_input / 100.0
    num_wins = st.number_input("Number of Wins", min_value=0, value=75, step=1)
    std_dev = st.number_input("Standard Deviation (fraction)", min_value=0.0, max_value=0.3, value=0.1, step=0.01, format="%.2f")
    submitted = st.form_submit_button("Calculate Growth")

def compound_growth(initial_capital, gain_pct, num_wins, tax_rate):
    effective_gain = gain_pct * (1 - tax_rate)
    final_capital = initial_capital * (1 + effective_gain) ** num_wins
    return final_capital

if submitted:
    if num_wins <= 0:
        st.warning("Please enter a positive number of wins.")
    else:
        base_gain = win_pct * (1 - tax_rate)
        upper_gain_pct = win_pct * (1 + std_dev)
        lower_gain_pct = max(win_pct * (1 - std_dev), 0)
        effective_upper_gain = upper_gain_pct * (1 - tax_rate)
        effective_lower_gain = lower_gain_pct * (1 - tax_rate)

        capitals = [initial_capital * (1 + base_gain) ** i for i in range(num_wins + 1)]
        upper_bound = [initial_capital * (1 + effective_upper_gain) ** i for i in range(num_wins + 1)]
        lower_bound = [initial_capital * (1 + effective_lower_gain) ** i for i in range(num_wins + 1)]

        final_capital = capitals[-1]
        pct_growth_final = ((final_capital - initial_capital) / initial_capital) * 100
        st.write(f"After {num_wins} wins, your capital grows to: **${final_capital:,.0f}** ({pct_growth_final:.0f}%)")

        # Prepare DataFrame for Altair
        df = pd.DataFrame({
            'Trade Number': np.arange(num_wins + 1),
            'Base': capitals,
            'Upper': upper_bound,
            'Lower': lower_bound
        })
        df_melt = df.melt('Trade Number', var_name='Series', value_name='Capital')

        color_map = {
            "Base": "#ffffff",
            "Upper": "#FF0000",
            "Lower": "#00FF00"
        }

        chart = alt.Chart(df_melt).mark_line().encode(
            x=alt.X('Trade Number', title='Trade Number'),
            y=alt.Y('Capital', title='Capital ($)'),
            color=alt.Color('Series', scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())),
                            legend=alt.Legend(title="Series")),
        ).properties(
            width=800, height=400,
            title='Capital Growth Over Trades (Std Dev Bounds)'
        )

        # Add annotation for final capital
        annotation = alt.Chart(pd.DataFrame({'x': [num_wins], 'y': [final_capital]})).mark_text(
            text=f"${final_capital:,.0f}", dx=-50, dy=-20, fontSize=15, color='grey', opacity=0.7
        ).encode(
            x='x:Q',
            y='y:Q'
        )

        st.altair_chart(chart + annotation, use_container_width=True)
