import streamlit as st

st.set_page_config(page_title="Technical Indicators Flashcards", layout="centered")

st.title("Technical Indicators Flashcards")

flashcards = {
    "Simple Moving Average (SMA)": 
        """**What is it?**  
        The average closing price over a set number of days.  
        **Usage:**  
        Smooths out short-term price fluctuations to identify trends. Bullish signals occur when short-term SMA crosses above long-term SMA, and bearish vice versa.
        Use 12-period SMA and 40-period SMA on daily or weekly to understand the trends (Or use any period that you like).
        Buy when the price is safely above these averages and/or retrests.
        Goal should be 3-7% compounding when SMA12 > SMA40.
        Stay-away or short when SMA40 > SMA12 or Price is below these averages.
        Avoid at any cost to buy below or I've missed the bottom.""",

    "Relative Strength Index (RSI)": 
        """**What is it?**  
        A momentum oscillator ranging 0-100, measuring speed and change of recent price movements.  
        **Usage:**  
        RSI above 70 indicates overbought (possible pullback). Below 30 shows oversold (potential bounce). Helps spot reversals and divergences with price.
        RSI can stay above 50 for a long-period of time, means, you have many opportunities, combine it with SMA displays.
        RSI can stay long below 30 just like above. So, remain bearish.
        You are only bullish when RSI is above 50 and SMAs are normally ordered i.e. SMA12 > SMA40.""",

    "Moving Average Convergence Divergence (MACD)":
        """**What is it?**  
        Difference between 12-day and 26-day EMAs, with a 9-day signal line.  
        **Usage:**  
        MACD crossing above signal line signals bullish momentum; crossing below signals bearish. Useful for trend and momentum shifts.""",

    "Average True Range (ATR)": 
        """**What is it?**  
        Measures market volatility by averaging the price range over a period.  
        **Usage:**  
        Higher ATR means higher volatility. Traders use ATR to set wider stop-losses in volatile markets.
        Use this to define the targets and SLs. If SMA12 > SMA40, use ATR to define targets e.g. 1.5 times or 2.0 times.
        When the market is sideways, ATR will be low for two to three months, hence lower down your expectations and use 0.5 of ATR as a target if it fits.""",

    "Directional Movement Index (DI+, DI-, ADX)":
        """**What is it?**  
        DI+ measures upward movement, DI- downward, ADX trend strength.  
        **Usage:**  
        Rising ADX indicates strengthening trend. DI+ above DI- signals uptrend and vice versa.""",

    "Keltner Channels": 
        """**What is it?**  
        Volatility bands plotted around an EMA using ATR.  
        **Usage:**  
        Price touching upper/lower bands may indicate overbought/oversold or breakout conditions.""",

    "SuperTrend Indicator": 
        """**What is it?**  
        Trend-following indicator using ATR bands.  
        **Usage:**  
        Price above SuperTrend is bullish; below is bearish.""",

    "Volume Indicators (OBV, MFI, CMF, etc.)":
        """**What are they?**  
        Metrics analyzing volume flow to confirm trends.  
        **Usage:**  
        Rising volume metrics with rising price confirm strength. Divergences warn of reversal.""",

    "Candlestick Patterns": 
        """**What are they?**  
        Price bar formations showing market psychology.  
        **Usage:**  
        Patterns like hammer, engulfing suggest trend reversals or continuation.""",

    "Pivot Points (PP, R1, R2, S1, S2)": 
        """**What are they?**  
        Calculated support and resistance price levels.  
        **Usage:**  
        Used for setting entry/exit targets and stops.""",

    "Returns and Volatility":
        """**What are they?**  
        Price changes over time and variability.  
        **Usage:**  
        Assess risk and reward potential.""",

    "Exhaustion Indicator":
        """**What is it?**  
        Detects extreme market conditions signaling potential trend exhaustion.  
        **Usage:**  
        Indicates probable turning points or pauses in trend."""
}

for key, value in flashcards.items():
    with st.expander(key):
        st.write(value)
