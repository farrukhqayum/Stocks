import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import yfinance as yf
tickers = ['^TNX', '^GSPC']

# Download data without group_by to inspect structure
data = yf.download(tickers, period='2y', group_by='ticker')

# If group_by='ticker' does not yield expected level names, try without grouping
# data = yf.download(tickers, period='2y')

# Inspect columns to understand exact labels
print(data.columns)

# If columns are MultiIndex with first level as ticker and second as attribute:
if isinstance(data.columns, pd.MultiIndex):
    # Extract 'Adj Close' for each ticker safely
    adj_close = pd.DataFrame({ticker: data[ticker]['Adj Close'] for ticker in tickers})
else:
    # If no multiindex columns, just select 'Adj Close'
    adj_close = data['Adj Close']

# Rename columns for clarity
adj_close.columns = ['US 10Y Treasury Yield', 'S&P 500']

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
