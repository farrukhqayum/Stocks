import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Generate your data
initial_capital = 1000
win_pct = 0.0375
tax_rate = 0.0
num_wins = 75
std_dev = 0.1

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
