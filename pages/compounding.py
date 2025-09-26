import streamlit as st

st.header("Just Keep Winning!!!")
comp_text = """
Compounding is the process where the returns you earn are reinvested to generate their own returns. 
This effect causes your capital to grow exponentially over time, not just linearly.
Even small percentage gains consistently accumulated can turn modest initial capital into significant wealth.
Keep winning trades and staying disciplined to harness the power of compounding — patience and persistence are key to long-term trading success.
Remember, consistent small wins build up to large gains as profits generate more profits.
"""
st.markdown(comp_text)

# Add fields
initial_capital = st.number_input("Initial Capital ($)", min_value=0.0, value=1000.0, step=100.0)
win_pct = st.number_input("Avg. Win (%)", min_value=0.0, value=3.75, step=0.1) / 100.0
tax_pct = st.number_input("Tax (%)", min_value=0.0, value=0.0, step=0.1) / 100.0
num_wins = st.number_input("Number of Trade Wins", min_value=0, value=75, step=1)

if st.button("Calculate Growth"):
    # Make sure compound_growth function is defined/imported
    final_capital = compound_growth(initial_capital, win_pct, num_wins, tax_pct)
    st.write(f"After {num_wins} consecutive wins, your capital grows to: **${final_capital:,.2f}**")
    # Show growth over each trade
    capitals = [initial_capital * (1 + win_pct) ** i for i in range(num_wins + 1)]
    st.line_chart(capitals)
