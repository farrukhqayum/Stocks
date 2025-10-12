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

def compound_growth(initial_capital, gain_pct, num_wins, tax_rate):
    effective_gain = gain_pct * (1 - tax_rate)
    final_capital = initial_capital * (1 + effective_gain) ** num_wins
    return final_capital

initial_capital = st.number_input("Initial Capital ($)", min_value=0.0, value=1000.0, step=100.0)
win_pct = st.number_input("Avg. Win (%)", min_value=0.0, value=7.0, step=0.1) / 100.0
tax_pct_input = st.number_input("Tax/Fee (%)", min_value=0.0, value=0.0, step=0.1)
tax_rate = tax_pct_input / 100.0
num_wins = st.number_input("Number of Trade Wins", min_value=0, value=75, step=1)
std_dev = st.number_input("Standard Deviation (fraction)", min_value=0.0, max_value=0.3, value=0.1, step=0.01, format="%.2f")

if st.button("Calculate Growth"):
    try:
        if num_wins <= 0:
            st.warning("Please enter a positive number of wins.")
        else:
            # Calculate gains after tax for base, upper and lower bounds
            base_gain = win_pct * (1 - tax_rate)
            upper_gain_pct = win_pct * (1 + std_dev)
            lower_gain_pct = max(win_pct * (1 - std_dev), 0)  # Avoid negative growth pct
            
            effective_upper_gain = upper_gain_pct * (1 - tax_rate)
            effective_lower_gain = lower_gain_pct * (1 - tax_rate)
            
            capitals = np.array([initial_capital * (1 + base_gain) ** i for i in range(num_wins + 1)])
            upper_bound = np.array([initial_capital * (1 + effective_upper_gain) ** i for i in range(num_wins + 1)])
            lower_bound = np.array([initial_capital * (1 + effective_lower_gain) ** i for i in range(num_wins + 1)])

            final_capital = capitals[-1]
            pct_growth_final = ((final_capital - initial_capital) / initial_capital) * 100
            st.write(f"After {num_wins} wins, your capital grows to: **${final_capital:,.0f}** ({pct_growth_final:.0f}%)")

            label_upper = f'Upper: ({upper_gain_pct*100:.2f}%)'
            label_lower = f'Lower: ({lower_gain_pct*100:.2f}%)'
            
            fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

            # Plot main capital growth on left y-axis
            ax.plot(capitals, color='black', linewidth=2, linestyle='solid', alpha=0.8, label='Capital (Base)')
            
            # Create secondary y-axis on right
            ax2 = ax.twinx()
            ax2.plot(upper_bound, color='red', linewidth=0.5, linestyle='dotted', label=label_upper)
            ax2.plot(lower_bound, color='green', linewidth=0.5, linestyle='dotted', label=label_lower)
            
            # Set labels for both y-axes
            #ax.set_ylabel('Capital ($) - Left')
            ax2.set_ylabel('Capital ($)')
            
            # Optional: synchronize limits if needed
            ax2.set_ylim(ax.get_ylim())
            
            fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

            # Plot base capital on primary axis
            ax.plot(capitals, color='black', linewidth=2, linestyle='solid', alpha=0.8, label='Capital (Base)')
            
            # Create and plot on secondary axis
            ax2 = ax.twinx()
            ax2.plot(upper_bound, color='red', linewidth=0.5, linestyle='dotted', label=label_upper)
            ax2.plot(lower_bound, color='green', linewidth=0.5, linestyle='dotted', label=label_lower)
            
            # Synchronize y-limits
            ax2.set_ylim(ax.get_ylim())
            
            # Plot fills on secondary axis (ax2), not on ax
            ax2.fill_between(range(num_wins + 1), capitals, upper_bound, 
                             where=(upper_bound > capitals), facecolor='red', alpha=0.1, interpolate=True)
            ax2.fill_between(range(num_wins + 1), lower_bound, capitals, 
                             where=(lower_bound < capitals), facecolor='green', alpha=0.1, interpolate=True)
            
            # Set right y-axis label
            ax2.set_ylabel('Capital ($)')
            
            # Set left axis xlabel & title
            ax.set_xlabel('Trade Number', fontsize=8)
            ax.set_title('Capital Growth Over Trades')
            
            # Customize grid and ticks on primary axis (ax)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax2.tick_params(axis='both', which='major', labelsize=7)
            
            # Combine legends from both axes
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, fontsize=7)
            
            plt.tight_layout()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error calculating growth: {e}")
