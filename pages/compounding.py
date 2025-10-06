import streamlit as st

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

def compound_growth(initial_capital, win_pct, num_wins, tax_rate):
    effective_gain = win_pct * (1 - tax_rate)
    final_capital = initial_capital * (1 + effective_gain) ** num_wins
    return final_capital

# Add fields
initial_capital = st.number_input("Initial Capital ($)", min_value=0.0, value=1000.0, step=100.0)
win_pct = st.number_input("Avg. Win (%)", min_value=0.0, value=3.75, step=0.1) / 100.0
tax_pct_input = st.number_input("Tax (%)", min_value=0.0, value=0.0, step=0.1)
tax_rate = tax_pct_input / 100.0
num_wins = st.number_input("Number of Trade Wins", min_value=0, value=75, step=1)

if st.button("Calculate Growth"):
    try:
        effective_win_pct = win_pct * (1 - tax_rate)
        if num_wins <= 0:
            st.warning("Please enter a positive number of wins.")
        else:
            final_capital = compound_growth(initial_capital, win_pct, num_wins, tax_rate)
            pct_growth_final = ((final_capital - initial_capital) / initial_capital) * 100
            st.write(f"After {num_wins} consecutive wins, your capital grows to: **${final_capital:,.0f}** "
                f"({pct_growth_final:.2f}%)")
            #st.write(f"After {num_wins} consecutive wins, your capital grows to: **${final_capital:,.0f}**")
            capitals = [initial_capital * (1 + effective_win_pct) ** i for i in range(num_wins + 1)]
            st.line_chart(capitals)
    except Exception as e:
        st.error(f"Error calculating growth: {e}")
