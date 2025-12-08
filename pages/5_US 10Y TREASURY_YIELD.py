import streamlit as st
import pandas as pd
import altair as alt
import yfinance as yf

years = st.number_input(
    "Enter number of years of data to fetch:",
    min_value=1,
    max_value=50,
    value=5,
    step=1
)
p = f"{years}y"

tickers = ['^TNX', '^GSPC', '^VIX']
data = yf.download(tickers, period=p)

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
    'S&P 500': get_adj_close(data, '^GSPC'),
    'VIX': get_adj_close(data, '^VIX')
})


# Calculate moving averages
adj_close['US 10Y MA200'] = adj_close['US 10Y Treasury Yield'].rolling(200).mean()
adj_close['S&P 500 MA200'] = adj_close['S&P 500'].rolling(200).mean()
adj_close['VIX MA200'] = adj_close['VIX'].rolling(200).mean()
adj_close['VIX']= adj_close['VIX'].rolling(5).mean()
adj_close['US 10Y Treasury Yield']= adj_close['US 10Y Treasury Yield'].rolling(5).mean() 
df = adj_close.reset_index()

# Melt dataframe for Altair multi-line plotting
df_melted = df.melt(
    id_vars=['Date'],
    value_vars=[
        'US 10Y Treasury Yield', 'US 10Y MA200',
        'S&P 500', 'S&P 500 MA200',
        'VIX', 'VIX MA200'
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
    color=alt.Color(
        "Series:N",
        legend=alt.Legend(orient="top-left")  # puts legend inside, top-left
    )
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

# VIX chart
vix_chart = base.transform_filter(
    alt.FieldOneOfPredicate(
        field='Series',
        oneOf=['VIX', 'VIX MA200']
    )
).mark_line().properties(height=chart_height, title='VIX and 200-period MA')

# Combine vertically
final_chart = alt.vconcat(yield_chart, sp500_chart, vix_chart).resolve_scale(x='shared')
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

st.markdown("""
### ✅ VIX — Simple Explanation

**What is the VIX?**
- The VIX (CBOE Volatility Index) measures expected S&P 500 volatility over the next 30 days
- Known as the "fear index" — spikes during market crashes, drops during calm periods

### ✅ Why VIX Matters for Trading

**1. Inverse Relationship with S&P 500**
- VIX rises → S&P 500 typically falls (fear drives selling)
- VIX falls → S&P 500 typically rises (confidence returns)

**2. Key Trading Signals**
- VIX > 30: Extreme fear (potential buying opportunity)
- VIX < 15: Complacency (potential risk of correction)
- VIX spikes often precede S&P 500 bottoms

**3. Perfect Companion to Your Current Charts**
- Shows when S&P 500 moves are "panicky" vs "orderly"
- VIX + Yield + S&P 500 = complete market regime picture
""")


