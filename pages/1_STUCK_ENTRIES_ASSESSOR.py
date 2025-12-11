import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings

# Suppress matplotlib deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Page config
st.set_page_config(page_title="Stock Position Assessor", layout="wide")

st.title("Stock Position Assessment Tool")

# a) Ticker input and data loading
ticker = st.sidebar.text_input("Enter Stock Ticker", value="COIN")
start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", value=datetime.now())

def historical_outcome_probabilities(data, entries_df, horizon=90, gain_threshold=0.05):
    """
    data: price dataframe with 'Close'
    entries_df: dataframe with 'date', 'shares', 'price'
    horizon: number of days to look forward
    gain_threshold: % gain above avg cost to count as 'gain'
    """
    results = []
    for _, entry in entries_df.iterrows():
        entry_date = pd.to_datetime(entry['date'])
        avg_cost = entry['price']
        
        # Slice horizon window
        window = data.loc[entry_date: entry_date + pd.Timedelta(days=horizon)]
        if window.empty: 
            continue
        
        # Check outcomes
        breakeven = (window['Close'] >= avg_cost).any()
        gain = (window['Close'] >= avg_cost * (1 + gain_threshold)).any()
        loss = not breakeven  # if never broke even
        
        results.append({
            'entry_date': entry_date,
            'breakeven': breakeven,
            'gain': gain,
            'loss': loss
        })
    
    # Aggregate probabilities
    df = pd.DataFrame(results)
    probs = {
        'Chance of Breakeven': df['breakeven'].mean() * 100,
        'Chance of Retaining Losses': df['loss'].mean() * 100,
        'Chance of Gaining a Bit': df['gain'].mean() * 100
    }
    return probs

@st.cache_data
def load_stock_data(ticker, start_date, end_date):
    """Get stock data from Yahoo Finance"""
    df = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1), 
                     progress=False)
    if df.empty:
        return None
    
    # Clean column names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    df.index = pd.to_datetime(df.index)
    df = df.dropna()
    return df

if ticker:
    data = load_stock_data(ticker, start_date, end_date)
    if data is not None and not data.empty:
        current_price = data['Close'].iloc[-1]
        st.metric("Current Price", f"${current_price:.2f}")  # Fixed: data loaded successfully

        # Price chart with matplotlib
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data['Close'], linewidth=2, color='blue', label='Close Price')
        ax.fill_between(data.index, data['Low'], data['High'], alpha=0.3, color='gray', label='High-Low')
        ax.set_title(f"{ticker} Price Chart", fontsize=16, fontweight='bold')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # b) Position entries input
        st.header("Your Positions")
        num_entries = st.number_input("Number of Recent Entries", min_value=1, max_value=10, value=3)
        
        entries = []
        for i in range(num_entries):
            col1, col2, col3 = st.columns(3)
            with col1:
                date = st.date_input(f"Entry {i+1} Date", value=datetime.now() - timedelta(days=30*(i+1)), key=f"date_{i}")
            with col2:
                shares = st.number_input(f"Entry {i+1} Shares", min_value=1.0, value=100.0, key=f"shares_{i}")
            with col3:
                price = st.number_input(f"Entry {i+1} Price", min_value=0.1, value=float(current_price), key=f"price_{i}")
            entries.append({'date': date, 'shares': shares, 'price': price})
        
        entries_df = pd.DataFrame(entries)
        total_shares = entries_df['shares'].sum()
        total_invested = (entries_df['shares'] * entries_df['price']).sum()
        avg_cost = total_invested / total_shares if total_shares > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Shares", f"{total_shares:.0f}")
        col2.metric("Total Invested", f"${total_invested:.2f}")
        col3.metric("Avg Cost Basis", f"${avg_cost:.2f}")
        col4.metric("Current Value", f"${total_shares * current_price:.2f}")
        pnl_pct = ((current_price - avg_cost) / avg_cost * 100) if avg_cost > 0 else 0
        st.metric("PnL %", f"{pnl_pct:.2f}%")

        col1, col2 = st.columns([1, 2])
        with col1:
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            weights = entries_df['shares'].tolist()  # Each entry's shares as slice size
            labels = [f"Entry {i+1}\n${(s * p):.0f}" for i, (s, p) in enumerate(zip(entries_df['shares'], entries_df['price']))]
            colors = plt.cm.Set3(np.linspace(0, 1, len(weights)))  # Distinct colors for wedges
        
            ax2.pie(weights, labels=labels, colors=colors,
                    autopct='%1.1f%%', startangle=90,
                    wedgeprops=dict(width=0.6, edgecolor='white'))
            ax2.set_title("Position Breakdown by Entry", fontsize=14, fontweight='bold')
            st.pyplot(fig2)
            plt.close(fig2)

        # c) Position assessment and Monte Carlo
        st.header("Position Assessment & Monte Carlo Simulation")
        
        # Historical returns for simulation params
        returns = data['Close'].pct_change().dropna()
        mu = returns.mean() * 252  # Annualized drift
        sigma = returns.std() * np.sqrt(252)  # Annualized volatility
        
        col1, col2 = st.columns(2)
        col1.metric("Annualized Return", f"{mu*100:.1f}%")
        col2.metric("Annualized Volatility", f"{sigma*100:.1f}%")
        
        st.info(f"Historical stats used for Monte Carlo simulation")
        
        # Monte Carlo parameters
        days = st.slider("Forecast Days", 30, 365, 90)
        num_sims = st.slider("Monte Carlo Simulations", 1000, 10000, 5000)
        
        @st.cache_data
        def monte_carlo_forecast(current_price, mu, sigma, days, num_sims):
            dt = 1/252
            paths = np.zeros((days+1, num_sims))
            paths[0] = current_price
            for t in range(1, days+1):
                rand = np.random.standard_normal(num_sims)
                paths[t] = paths[t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*rand)
            return paths
        
        paths = monte_carlo_forecast(current_price, mu, sigma, days, num_sims)
        
        # Monte Carlo plot
        fig3, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
        
        # Price paths (sample 50)
        sample_paths = min(50, num_sims)
        for i in range(sample_paths):
            ax3.plot(range(days+1), paths[:, i], color='gray', alpha=0.3, linewidth=0.5)
        ax3.plot(range(days+1), paths[:, -1], color='blue', linewidth=2, label='Sample Path')
        
        # Percentiles
        percentiles = np.percentile(paths[-1], [5, 25, 50, 75, 95])
        ax3.fill_between([days, days], percentiles[0], percentiles[4], alpha=0.2, color='green', label='5-95% Range')
        ax3.fill_between([days, days], percentiles[1], percentiles[3], alpha=0.4, color='blue', label='25-75% Range')
        ax3.axhline(percentiles[2], color='red', linestyle='--', linewidth=2, label=f'Median: ${percentiles[2]:.2f}')
        
        ax3.set_title("Monte Carlo Price Simulation Paths", fontsize=14, fontweight='bold')
        ax3.set_xlabel("Days")
        ax3.set_ylabel("Price ($)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Final distribution histogram
        ax4.hist(paths[-1], bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
        ax4.axvline(percentiles[2], color='red', linestyle='--', linewidth=2, label=f'Median: ${percentiles[2]:.2f}')
        ax4.set_title("Final Price Distribution (Density)")
        ax4.set_xlabel("Price ($)")
        ax4.legend()
        
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        # d) Action scenarios
        st.header("Action Recommendations")
        target_price = percentiles[2]  # Median forecast
        recovery_needed = (avg_cost - current_price) / current_price * 100 if avg_cost > current_price else 0
        
        if pnl_pct > 20:
            st.success(f"**HOLD or TRIM**: +{pnl_pct:.1f}% gain. Median forecast ${target_price:.2f}.")
        elif pnl_pct > 0:
            st.info(f"**HOLD**: +{pnl_pct:.1f}% gain. Forecast upside: {((target_price-current_price)/current_price*100):+.1f}%.")
        elif pnl_pct > -10:
            st.warning(f"**HOLD or AVERAGE DOWN**: -{abs(pnl_pct):.1f}% loss. Needs {recovery_needed:.1f}% recovery.")
        else:
            st.error(f"**CUT LOSSES**: -{abs(pnl_pct):.1f}% loss. Forecast recovery uncertain.")
        
        # Portfolio allocation (user's $20k context)
        portfolio_size = 20000
        position_value = total_shares * current_price
        allocation_pct = (position_value / portfolio_size) * 100
        st.metric("Portfolio Allocation", f"{allocation_pct:.1f}% of $20K")

        st.caption("Monte Carlo: $$S(t+Δt) = S(t) × exp[(μ - ½σ²)Δt + σ√Δt × Z]$$ where Z ~ N(0,1)")
    else:
        st.warning("No data loaded. Please check ticker symbol and date range.")

probs = historical_outcome_probabilities(data, entries_df)
st.metric("Chance of Breakeven", f"{probs['Chance of Breakeven']:.1f}%")
st.metric("Chance of Retaining Losses", f"{probs['Chance of Retaining Losses']:.1f}%")
st.metric("Chance of Gaining a Bit", f"{probs['Chance of Gaining a Bit']:.1f}%")

