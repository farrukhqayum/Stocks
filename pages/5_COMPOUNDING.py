import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Compounded Growth", layout="wide")

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
    tax_pct_input = st.number_input(
        "Fee per Trade (%)", min_value=0.0, value=0.0, step=0.1,
        help="Enter your broker, exchange or trading fee as a percent of each trade amount."
    )
    tax_rate = tax_pct_input / 100.0  # fee per trade, not tax on gains
    num_wins = st.number_input("Number of Wins", min_value=0, value=75, step=1)
    std_dev = st.number_input("Standard Deviation (fraction)", min_value=0.0, max_value=0.4, value=0.20, step=0.01, format="%.2f")
    submitted = st.form_submit_button("Calculate Growth")

def compound_growth(initial_capital, gain_pct, num_wins, tax_rate):
    capital = initial_capital
    total_fee = 0
    for _ in range(num_wins):
        fee = capital * tax_rate
        capital = capital * (1 + gain_pct) - fee
        total_fee += fee
    return capital, total_fee

def format_large_number(x):
    if x >= 1_000_000:
        return f"${x/1_000_000:.2f}M"
    elif x >= 1_000:
        return f"${x/1_000:.2f}K"
    else:
        return f"${x:.2f}"

if submitted:
    if num_wins <= 0:
        st.warning("Please enter a positive number of wins.")
    else:
        # Growth series for chart
        def grow_series(gain_pct):
            caps = [initial_capital]
            capital = initial_capital
            for _ in range(num_wins):
                fee = capital * tax_rate
                capital = capital * (1 + gain_pct) - fee
                caps.append(capital)
            return caps

        upper_gain_pct = win_pct * (1 + std_dev)
        lower_gain_pct = max(win_pct * (1 - std_dev), 0)

        capitals = grow_series(win_pct)
        upper_bound = grow_series(upper_gain_pct)
        lower_bound = grow_series(lower_gain_pct)

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
            x=alt.X('Trade Number', axis=alt.Axis(labelAngle=-45, labelFontSize=10, title='Trade Number')),
            y=alt.Y('Capital', axis=alt.Axis(format=".2s", title='Capital ($)')),
            color=alt.Color('Series', scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())),
                            legend=alt.Legend(title="Legend", orient='top-left'))
        ).properties(
            title=f'Capital Growth with {num_wins} Wins',
            width=700,
            height=400
        )

        st.altair_chart(chart, use_container_width=True)

        # Table with fee breakdown for 3 splits
        splits = [0.33, 0.34, 0.33]
        expected_gains = [win_pct, win_pct, win_pct]

        results = [
            compound_growth(initial_capital * sp, gain, num_wins, tax_rate)
            for sp, gain in zip(splits, expected_gains)
        ]
        final_caps = [fc for fc, fee in results]
        total_fees = [fee for fc, fee in results]

        df_split = pd.DataFrame({
            "Stock": ["Stock 1", "Stock 2", "Stock 3"],
            "Allocation (%)": [sp * 100 for sp in splits],
            "Capital Allocated ($)": [initial_capital * sp for sp in splits],
            "Expected Gain per Win (%)": [g * 100 for g in expected_gains],
            f"Capital after {num_wins} Wins ($)": final_caps,
            "Total Fee ($)": total_fees
        })

        st.dataframe(df_split.style.format({
            "Allocation (%)": "{:.2f}%",
            "Capital Allocated ($)": format_large_number,
            "Expected Gain per Win (%)": "{:.2f}%",
            f"Capital after {num_wins} Wins ($)": format_large_number,
            "Total Fee ($)": format_large_number
        }), use_container_width=True)
