import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Generate your data
initial_capital = st.number_input("Initial Capital ($)", min_value=0.0, value=1000.0, step=100.0)
win_pct = st.number_input("Avg. Win (%)", min_value=0.0, value=3.75, step=0.1) / 100.0
tax_pct_input = st.number_input("Tax/Fee (%)", min_value=0.0, value=0.0, step=0.1)
tax_rate = tax_pct_input / 100.0
num_wins = st.number_input("Number of Wins", min_value=0, value=75, step=1)
std_dev = st.number_input("Standard Deviation (fraction)", min_value=0.0, max_value=0.3, value=0.1, step=0.01, format="%.2f")

base_gain = win_pct * (1 - tax_rate)
upper_gain_pct = win_pct * (1 + std_dev)
lower_gain_pct = max(win_pct * (1 - std_dev), 0)
effective_upper_gain = upper_gain_pct * (1 - tax_rate)
effective_lower_gain = lower_gain_pct * (1 - tax_rate)

capitals = [initial_capital * (1 + base_gain) ** i for i in range(num_wins + 1)]
upper_bound = [initial_capital * (1 + effective_upper_gain) ** i for i in range(num_wins + 1)]
lower_bound = [initial_capital * (1 + effective_lower_gain) ** i for i in range(num_wins + 1)]

df = pd.DataFrame({
    'Trade Number': np.arange(num_wins + 1),
    'Base': capitals,
    'Upper': upper_bound,
    'Lower': lower_bound
})
df_melt = df.melt('Trade Number', var_name='Series', value_name='Capital')

color_map = {
    "Base": "#ffffff",      # white
    "Upper": "#FF0000",     # red
    "Lower": "#00FF00"      # green
}

chart = alt.Chart(df_melt).mark_line().encode(
    x='Trade Number',
    y='Capital',
    color=alt.Color('Series', scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())))
).properties(width=800, height=400)

st.altair_chart(chart, use_container_width=True)
