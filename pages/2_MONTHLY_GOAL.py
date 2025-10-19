import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.header("Establish a MONTHLY GOAL!!!")
comp_text = """
If you plan to compound, you have to be a discipline trader for bi-monthly, or quarterly or per month. You need to know how much do you need to win to compound profits.

Here is a strategic chart to plan accordingly.
"""

def calculate_investment_growth(P, r, months, max_trades):
    investment_curves = {}
    for n_trades in range(1, max_trades + 1):
        investment_values = [P]
        for m in range(1, months + 1):
            monthly_profit = investment_values[-1] * (r * n_trades)
            new_value = investment_values[-1] + monthly_profit
            investment_values.append(round(new_value))
        investment_curves[n_trades] = investment_values
    return investment_curves

def plot_investment_growth(investment_curves, r, months):
    month_range = np.arange(0, months + 1)
    fig, ax_left = plt.subplots(figsize=(10, 6), dpi=100)
    for n_trades, values in investment_curves.items():
        ax_left.plot(month_range, values, marker='o', markersize=3, alpha=0.5, label=f'{n_trades} Wins/Month')
    y_min = min(min(values) for values in investment_curves.values())
    y_max = max(max(values) for values in investment_curves.values())
    y_padding = 0.1 * (y_max - y_min)
    ax_left.set_ylim(y_min - y_padding, y_max + y_padding)
    ax_left.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${int(x):,}'))
    ax_left.text(0.5, 0.5, f'@{round(r*100, 2)}% Profit', transform=ax_left.transAxes, fontsize=50,
                 color='grey', alpha=0.2, ha='center', va='center', weight='bold', style='italic')
    ax_left.yaxis.tick_right()
    ax_left.set_title(f'Investment Growth @{round(r*100, 2)}% Profit')
    ax_left.set_xlabel('Months')
    ax_left.set_ylabel('Investment Value ($)')
    ax_left.set_xticks(month_range)
    ax_left.grid(alpha=0.25)
    ax_left.legend(title='Successful Trades')
    plt.tight_layout()
    st.pyplot(fig)

def create_investment_dataframe(investment_curves, months):
    month_range = np.arange(0, months + 1)
    df_investments = pd.DataFrame(investment_curves, index=month_range)
    df_investments.index.name = 'Period'
    return df_investments

# Streamlit inputs for parameters
P = st.number_input("Initial Investment ($)", min_value=0.0, value=1000.0, step=100.0)
r = st.number_input("Profit Rate/Trade (%)", min_value=0.0, value=3.75, step=0.01) / 100.0
months = st.number_input("Period (e.g. Weeks/Months/Years)", min_value=1, value=12, step=1)
max_trades = st.number_input("Maximum Wins Per Period", min_value=1, value=7, step=1)
eff_monthly = ((1 + r) ** max_trades - 1)*100

if st.button("Calculate Investment Growth"):
    st.write(f'Effective Win per given period: {eff_monthly:.2f} (%)')
    investment_curves = calculate_investment_growth(P, r, months, max_trades)
    plot_investment_growth(investment_curves, r, months)
    df_investments = create_investment_dataframe(investment_curves, months)
    st.dataframe(df_investments)
    
    # Calculate and plot effective average gains per period
    avg_gains = df_investments.pct_change().mean() * 100  # Average growth % per column (n_trades)
    
    fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=100)
    ax2.bar(avg_gains.index, avg_gains.values, color='teal', alpha=0.7)
    ax2.set_title('Effective Average Gains per Period')
    ax2.set_xlabel('Successful Trades per Period')
    ax2.set_ylabel('Average Gain (%)')
    for i, v in enumerate(avg_gains.values):
        ax2.text(i, v, f'{v:.2f}%', ha='center', va='bottom')
    ax2.grid(alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig2)
        # Calculate the average dollar gain for each month
    avg_dollar_gain_per_period = df_investments.diff(axis=0).mean(axis=1)  # Row-wise mean difference per period

    # Plot the average dollar gain per month as a curve
    fig4, ax4 = plt.subplots(figsize=(10, 5), dpi=100)
    ax4.plot(avg_dollar_gain_per_period.index, avg_dollar_gain_per_period.values,
             marker='o', color='crimson', linewidth=2)
    ax4.fill_between(avg_dollar_gain_per_period.index, avg_dollar_gain_per_period.values,
                     color='crimson', alpha=0.2)
    ax4.set_title('Average Dollar Gain per Period')
    ax4.set_xlabel('Period (e.g. Month)')
    ax4.set_ylabel('Average Gain ($)')
    for i, v in enumerate(avg_dollar_gain_per_period.values):
        ax4.text(i, v, f'${v:,.0f}', ha='center', va='bottom', fontsize=8)
    ax4.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig4)
