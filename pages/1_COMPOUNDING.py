import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

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
tax_pct_input = st.number_input("Tax/Fee (%)", min_value=0.0, value=0.0, step=0.1)
tax_rate = tax_pct_input / 100.0
num_wins = st.number_input("Number of Trade Wins", min_value=0, value=75, step=1)

std_dev = 0.10  # 10% standard deviation for up and down cases

if st.button("Calculate Growth"):
    try:
        effective_win_pct = win_pct * (1 - tax_rate)
        if num_wins <= 0:
            st.warning("Please enter a positive number of wins.")
        else:
            capitals = np.array([initial_capital * (1 + effective_win_pct) ** i for i in range(num_wins + 1)])

            # Calculate upper and lower bounds with 10% standard deviation
            upper_bound = capitals * (1 + std_dev)  # 10% above base case
            lower_bound = capitals * (1 - std_dev)  # 10% below base case

            final_capital = capitals[-1]
            pct_growth_final = ((final_capital - initial_capital) / initial_capital) * 100
            st.write(f"After {num_wins} wins, your capital grows to: **${final_capital:,.0f}** "
                     f"({pct_growth_final:.0f}%)")

            fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

            # Plot base capital growth
            ax.plot(capitals, color='black', linewidth=2, linestyle='solid', alpha=0.8, label='Capital (Base)')

            # Plot upper & lower bounds as dotted lines
            ax.plot(upper_bound, color='red', linewidth=1.5, linestyle='dotted', label='Upper Bound (+10% std)')
            ax.plot(lower_bound, color='green', linewidth=1.5, linestyle='dotted', label='Lower Bound (-10% std)')

            # Fill between bounds and base capital
            ax.fill_between(range(num_wins + 1), capitals, upper_bound, where=(upper_bound > capitals),
                            facecolor='red', alpha=0.3, interpolate=True)
            ax.fill_between(range(num_wins + 1), lower_bound, capitals, where=(lower_bound < capitals),
                            facecolor='green', alpha=0.3, interpolate=True)

            # Add centered annotation text
            ax.text(0.5, 0.5, f'@{round(win_pct*100, 2)}% Profit',
                    transform=ax.transAxes, fontsize=25, color='grey', alpha=0.2,
                    horizontalalignment='center', verticalalignment='center',
                    rotation=0, weight='bold', style='italic')

            ax.set_xlabel('Trade Number')   # X-axis label
            ax.set_ylabel('Capital ($)')    # Y-axis label
            ax.set_title('Capital Growth Over Trades with Std Dev Bounds')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.legend(fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error calculating growth: {e}")
