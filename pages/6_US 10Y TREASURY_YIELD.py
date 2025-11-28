import streamlit as st
import pandas as pd
import altair as alt
import yfinance as yf

tickers = ['^TNX', '^GSPC']
data = yf.download(tickers, period='2y')

# Flatten columns if MultiIndex
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [f"{col[1]} {col[0]}" for col in data.columns]

# Now you can safely do:
adj_close = pd.DataFrame({
    'US 10Y Treasury Yield': data['^TNX Adj Close'],
    'S&P 500': data['^GSPC Adj Close']
})


# Access Adj Close correctly from MultiIndex
adj_close = pd.DataFrame({
    'US 10Y Treasury Yield': data['Adj Close']['^TNX'],
    'S&P 500': data['Adj Close']['^GSPC']
})

# Calculate 100-period moving averages
adj_close['US 10Y MA100'] = adj_close['US 10Y Treasury Yield'].rolling(window=100).mean()
adj_close['S&P 500 MA100'] = adj_close['S&P 500'].rolling(window=100).mean()

# Reset index for Altair plotting
df = adj_close.reset_index()

# Melt dataframe for Altair multi-line plotting
df_melted = df.melt(id_vars=['Date'], 
                    value_vars=['US 10Y Treasury Yield', 'US 10Y MA100', 'S&P 500', 'S&P 500 MA100'],
                    var_name='Series', value_name='Value')

# Chart height setup
chart_height = 300

# Create charts
base = alt.Chart(df_melted).encode(x='Date:T', y='Value:Q', color='Series:N')

yield_chart = base.transform_filter(
    alt.FieldOneOfPredicate(field='Series', oneOf=['US 10Y Treasury Yield', 'US 10Y MA100'])
).mark_line().properties(height=chart_height, title='US 10Y Treasury Yield and 100-period MA')

sp500_chart = base.transform_filter(
    alt.FieldOneOfPredicate(field='Series', oneOf=['S&P 500', 'S&P 500 MA100'])
).mark_line().properties(height=chart_height, title='S&P 500 and 100-period MA')

final_chart = alt.vconcat(yield_chart, sp500_chart).resolve_scale(x='shared')

st.title("US 10Y Treasury Yield and S&P 500 with 100-period Moving Average")
st.altair_chart(final_chart, use_container_width=True)
