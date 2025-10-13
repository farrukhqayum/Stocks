import streamlit as st
import numpy as np
import pandas as pd

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

def compound_growth(initial_capital, gain_pct, num_wins, tax_rate):
    effective_gain = gain_pct * (1 - tax_rate)
    final_capital = initial_capital * (1 + effective_gain) ** num_wins
    return final_capital

initial_capital = st.number_input("Initial Capital ($)", min_value=0.0, value=1000.0, step=100.0)
win_pct = st.number_input("Avg. Win (%)", min_value=0.0, value=3.75, step=0.1) / 100.0
tax_pct_input = st.number_input("Tax/Fee (%)", min_value=0.0, value=0.0, step=0.1)
tax_rate = tax_pct_input / 100.0
num_wins = st.number_input("Number of Wins", min_value=0, value=75, step=1)
std_dev = st.number_input("Standard Deviation (fraction)", min_value=0.0, max_value=0.3, value=0.1, step=0.01, format="%.2f")

if st.button("Calculate Growth"):
    try:
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

            # Create a DataFrame for the lines
            df = pd.DataFrame({
                'Base': capitals,
                f'Upper ({upper_gain_pct*100:.2f}%)': upper_bound,
                f'Lower ({lower_gain_pct*100:.2f}%)': lower_bound
            })

            st.line_chart(df)

    except Exception as e:
        st.error(f"Error calculating growth: {e}")
