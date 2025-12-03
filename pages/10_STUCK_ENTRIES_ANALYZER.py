import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="Stock Position Assessor", layout="wide")

st.title("Stock Position Assessment Tool")

# a) Ticker input and data loading
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", value=datetime.now())

@st.cache_data
def load_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

if ticker:
    data = load_stock_data(ticker, start_date, end_date)
    current_price = data['Close'].iloc[-1]
    st.metric("Current Price", f"${current_price:.2f}")

    # Price chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], 
                                 low=data['Low'], close=data['Close'], name=ticker))
    fig.update_layout(title=f"{ticker} Price Chart", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    # b) Position entries input
    st.header("Your Positions")
    num_entries = st.number_input("Number of Recent Entries", min_value=1, max_value=10, value=3)
    
    entries = []
    for i in range(num_entries):
        col1, col2, col3 = st.columns(3)
        with col1:
            date = st.date_input(f"Entry {i+1} Date", value=datetime.now() - timedelta(days=30*(i+1)))
        with col2:
            shares = st.number_input(f"Entry {i+1} Shares", min_value=1.0, value=100.0)
        with col3:
            price = st.number_input(f"Entry {i+1} Price", min_value=0.1, value=float(current_price))
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

    # c) Position assessment and scenarios
    st.header("Position Assessment & Scenarios")
    
    # Historical returns for simulation params [web:1][web:3]
    returns = data['Close'].pct_change().dropna()
    mu = returns.mean() * 252  # Annualized drift
    sigma = returns.std() * np.sqrt(252)  # Annualized volatility
    
    st.info(f"Historical Annualized Return: {mu*100:.1f}%, Volatility: {sigma*100:.1f}% [web:1]")
    
    # Monte Carlo Simulation [web:1][web:3]
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
    
    # Plot simulation
    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(x=list(range(days+1)), y=paths[:, -1], 
                                 name='Sample Path', line=dict(color='rgba(0,100,80,0.9)')))
    percentiles = np.percentile(paths[-1], [5, 25, 50, 75, 95])
    fig_sim.add_trace(go.Scatter(x=[days, days], y=[percentiles[0], percentiles[4]], 
                                 fill='toself', fillcolor='rgba(0,100,80,0.2)', 
                                 line_color='rgba(255,255,255,0)', name='5-95% Range'))
    fig_sim.add_trace(go.Scatter(x=[days, days], y=[percentiles[1], percentiles[3]], 
                                 fill='toself', fillcolor='rgba(0,176,246,0.2)', 
                                 line_color='rgba(255,255,255,0)', name='25-75% Range'))
    fig_sim.add_trace(go.Scatter(x=[days, days], y=[percentiles[2], percentiles[2]], 
                                 mode='markers+text', marker=dict(size=10), 
                                 name=f'Median: ${percentiles[2]:.2f}'))
    fig_sim.update_layout(title="Monte Carlo Price Forecast", xaxis_title="Days", yaxis_title="Price")
    st.plotly_chart(fig_sim, use_container_width=True)
    
    # Scenario recommendations
    target_price = percentiles[2]  # Median forecast
    position_pct = pnl_pct
    
    st.subheader("Action Scenarios")
    if position_pct > 20:
        st.success("**Hold or Trim**: Position up >20%. Consider trimming 20-30% if forecast median < current price.")
    elif position_pct > 0:
        st.info("**Hold**: Small gain. Monitor if forecast shows >10% upside potential.")
    elif position_pct > -10:
        st.warning("**Hold or Average Down**: Minor loss. If forecast median > avg cost, consider adding cautiously.")
    else:
        st.error("**Accept Loss or Restructure**: Deep loss. Evaluate if forecast supports recovery; otherwise cut losses.")
    
    st.caption("Scenarios based on PnL and Monte Carlo median forecast. Use for assessment only [web:1][web:3].")
    
    # Portfolio context (user has ~$20k total capital)
    portfolio_size = 20000  # From user history
    position_value = total_shares * current_price
    allocation = (position_value / portfolio_size) * 100
    st.metric("Position Allocation", f"{allocation:.1f}% of $20k portfolio")

# Sidebar info
with st.sidebar:
    st.markdown("---")
    st.markdown("**Monte Carlo uses geometric Brownian motion:**")
    st.latex(r"S(t+\Delta t) = S(t) \times \exp\left[(\mu - \frac{\sigma^2}{2})\Delta t + \sigma\sqrt{\Delta t} \times Z\right] [web:3]")
    st.markdown("Where Z ~ N(0,1)")
