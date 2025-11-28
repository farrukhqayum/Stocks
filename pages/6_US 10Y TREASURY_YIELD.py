import streamlit as st
import pandas as pd
import altair as alt
import yfinance as yf

tickers = ['^TNX', '^GSPC']
data = yf.download(tickers, period='2y')

def get_adj_close(df, ticker):
    # Case 1: MultiIndex (standard stocks)
    try:
        return df['Adj Close'][ticker]
    except:
        pass

    # Case 2: Single-level index (common for ^TNX, ^VIX, etc.)
    for col in df.columns:
        if ticker in col and 'Adj Close' in col:
            return df[col]
        if ticker in col and 'Close' in col:   # fallback
            return df[col]

    # If still not found
    raise KeyError(f"Adj Close column not found for {ticker}. Columns: {df.columns}")

adj_close = pd.DataFrame({
    'US 10Y Treasury Yield': get_adj_close(data, '^TNX'),
    'S&P 500': get_adj_close(data, '^GSPC')
})


# Calculate 100-period moving averages
adj_close['US 10Y MA100'] = adj_close['US 10Y Treasury Yield'].rolling(100).mean()
adj_close['S&P 500 MA100'] = adj_close['S&P 500'].rolling(100).mean()

# Reset index for Altair plotting
df = adj_close.reset_index()

# Melt dataframe for Altair multi-line plotting
df_melted = df.melt(
    id_vars=['Date'],
    value_vars=[
        'US 10Y Treasury Yield', 'US 10Y MA100',
        'S&P 500', 'S&P 500 MA100'
    ],
    var_name='Series',
    value_name='Value'
)

# Chart height
chart_height = 300

# Shared chart base
base = alt.Chart(df_melted).encode(
    x='Date:T',
    y='Value:Q',
    color='Series:N'
)

# Yield chart
yield_chart = base.transform_filter(
    alt.FieldOneOfPredicate(
        field='Series',
        oneOf=['US 10Y Treasury Yield', 'US 10Y MA100']
    )
).mark_line().properties(height=chart_height, title='US 10Y Treasury Yield and 100-period MA')

# S&P 500 chart
sp500_chart = base.transform_filter(
    alt.FieldOneOfPredicate(
        field='Series',
        oneOf=['S&P 500', 'S&P 500 MA100']
    )
).mark_line().properties(height=chart_height, title='S&P 500 and 100-period MA')

# Combine vertically
final_chart = alt.vconcat(yield_chart, sp500_chart).resolve_scale(x='shared')

st.title("US 10Y Treasury Yield and S&P 500")
st.altair_chart(final_chart, use_container_width=True)
