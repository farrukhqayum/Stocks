import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import yfinance as yf

tickers = ['^TNX', '^GSPC']
data = yf.download(tickers, period='2y', group_by='ticker')

# Now access Adj Close per ticker:
adj_close = pd.DataFrame({
    'US 10Y Treasury Yield': data['^TNX']['Adj Close'],
    'S&P 500': data['^GSPC']['Adj Close']
})

# Calculate 100-period moving averages
data['US 10Y MA100'] = data['US 10Y Treasury Yield'].rolling(window=100).mean()
data['S&P 500 MA100'] = data['S&P 500'].rolling(window=100).mean()

# Reset index for Altair plotting
df = data.reset_index()

# Melt dataframe for Altair multi-line plotting
df_melted = df.melt(id_vars=['Date'], value_vars=['US 10Y Treasury Yield', 'US 10Y MA100', 'S&P 500', 'S&P 500 MA100'],
                    var_name='Series', value_name='Value')

# Chart height setup
chart_height = 300

# Create separate charts for Treasury Yield and S&P 500 with their MAs, stacked vertically
base = alt.Chart(df_melted).encode(x='Date:T', y='Value:Q', color='Series:N')

yield_chart = base.transform_filter(
    alt.FieldOneOfPredicate(field='Series', oneOf=['US 10Y Treasury Yield', 'US 10Y MA100'])
).mark_line().properties(height=chart_height, title='US 10Y Treasury Yield and 100-period MA')

sp500_chart = base.transform_filter(
    alt.FieldOneOfPredicate(field='Series', oneOf=['S&P 500', 'S&P 500 MA100'])
).mark_line().properties(height=chart_height, title='S&P 500 and 100-period MA')

# Combine the two charts vertically
final_chart = alt.vconcat(yield_chart, sp500_chart).resolve_scale(x='shared')

# Display with Streamlit
st.title("US 10Y Treasury Yield and S&P 500 with 100-period Moving Average")
st.altair_chart(final_chart, use_container_width=True)
