import streamlit as st
import pandas as pd
import altair as alt
import yfinance as yf

tickers = ['^TNX', '^GSPC']
data = yf.download(tickers, period='10y')

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
adj_close['US 10Y MA200'] = adj_close['US 10Y Treasury Yield'].rolling(200).mean()
adj_close['S&P 500 MA200'] = adj_close['S&P 500'].rolling(200).mean()

# Reset index for Altair plotting
df = adj_close.reset_index()

# Melt dataframe for Altair multi-line plotting
df_melted = df.melt(
    id_vars=['Date'],
    value_vars=[
        'US 10Y Treasury Yield', 'US 10Y MA200',
        'S&P 500', 'S&P 500 MA200'
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
        oneOf=['US 10Y Treasury Yield', 'US 10Y MA200']
    )
).mark_line().properties(height=chart_height, title='US 10Y Treasury Yield and 200-period MA')

# S&P 500 chart
sp500_chart = base.transform_filter(
    alt.FieldOneOfPredicate(
        field='Series',
        oneOf=['S&P 500', 'S&P 500 MA200']
    )
).mark_line().properties(height=chart_height, title='S&P 500 and 200-period MA')

# Combine vertically
final_chart = alt.vconcat(yield_chart, sp500_chart).resolve_scale(x='shared')

st.title("US 10Y Treasury Yield and S&P 500")
st.altair_chart(final_chart, use_container_width=True)

st.markdown("""
### ✅ US 10-Year Treasury Yield — Simple Explanation

What is the US 10-Year Treasury Yield?
- The US 10-Year Treasury Yield (^TNX) represents the interest rate the US government pays to borrow money for 10 years. It is one of the most important financial indicators in the world.

### ✅ Why the 10-Year Yield Matters

***1. Reflects Inflation & Economic Expectations***

-- Rising yields → traders expect higher inflation or stronger economic growth.

-- Falling yields → markets expect slower growth or lower inflation.

***2. Drives Mortgage Rates & Loans***

***The 10Y yield is used as a benchmark:***

- Mortgage rates

- Auto loans

- Business lending

***When the yield rises, borrowing becomes more expensive.***

3. Strong Impact on Stock Market Valuation

The 10-year yield acts like a "discount rate":

- Higher yields → lower stock valuations
(especially tech and growth stocks)

***Lower yields → higher stock prices

This is why the S&P 500 and the 10Y yield often move in opposite directions.

### ✅ Relationship to S&P 500

- When yields rise quickly → stocks often drop (higher borrowing cost & lower valuations).

- When yields fall → stocks often rally (cheap borrowing & higher valuations).

- Slow, steady moves usually have a mild effect; big spikes create sharp volatility.

""")



