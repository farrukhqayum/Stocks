import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(layout="centered")

st.header("Just Keep Winning!!!")
comp_text = """
Compounding is the process where the returns you earn are reinvested to generate their own returns. 
This effect causes your capital to grow exponentially over time, not just linearly.
Even small percentage gains consistently accumulated can turn modest initial capital into significant wealth.
Keep winning trades and staying disciplined to harness the power of compounding — patience and persistence are key to long-term trading success.
Remember, consistent small wins build up to large gains as profits generate more profits.
"""
st.markdown(comp_text)

with st.form(key='compound_form'):
    initial_capital = st.number_input("Initial Capital ($)", min_value=0.0, value=1000.0, step=100.0)
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
    # Clear caches dynamically on each new calculation
    st.cache_data.clear()
    st.cache_resource.clear()

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
        st.success(f"After {num_wins} wins, your capital grows to: **${final_capital:,.0f}** ({pct_growth_final:.0f}%)")

        label_upper = f'Upper ({upper_gain_pct*100:.2f}%)'
        label_lower = f'Lower ({lower_gain_pct*100:.2f}%)'
        label_base = f'Base ({win_pct*100:.2f}%)'
        
        df = pd.DataFrame({
            'Trade Number': np.arange(num_wins + 1),
            label_base: capitals,
            label_upper: upper_bound,
            label_lower: lower_bound
        })
        df_melt = df.melt('Trade Number', var_name='Series', value_name='Capital')

        color_map = {
            label_base: "#ffffff",
            label_upper: "#FF0000",
            label_lower: "#00FF00"
        }

        chart = alt.Chart(df_melt).mark_line(size=2).encode(
            x=alt.X('Trade Number', axis=alt.Axis(labelAngle=-45, labelFontSize=10, labelFlush=False, title='Trade Number')),
            y=alt.Y('Capital', axis=alt.Axis(format=".2~s", orient="right", title='Capital ($ Millions)')),
            color=alt.Color('Series', scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())),
                            legend=alt.Legend(title="Legend", orient='top-left'))
        ).properties(
            title=f'Capital Growth with {num_wins} Wins',
                width=400,
                height=300
        ).interactive()


        annotation = alt.Chart(pd.DataFrame({
            'x': [num_wins],
            'y': [final_capital]
        })).mark_text(
            text=f"${final_capital:,.0f}",
            align='right',
            dx=-10, dy=-10,
            fontSize=13,
            color='grey',
            opacity=0.8
        ).encode(x='x', y='y')

        st.altair_chart(chart + annotation, use_container_width=True)
