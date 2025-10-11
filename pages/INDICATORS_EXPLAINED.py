import streamlit as st

st.set_page_config(page_title="Technical Indicators Flashcards", layout="centered")

st.title("Stock Market Technical Indicators Flashcards")

flashcards = {
    "Simple Moving Average (SMA)": 
        """**What is it?**  
        The average closing price over a set number of days.  
        **Usage:**  
        Smooths out short-term price fluctuations to identify trends. Bullish signals occur when short-term SMA crosses above long-term SMA, and bearish vice versa.""",

    "Relative Strength Index (RSI)": 
        """**What is it?**  
        A momentum oscillator ranging 0-100, measuring speed and change of recent price movements.  
        **Usage:**  
        RSI above 70 indicates overbought (possible pullback). Below 30 shows oversold (potential bounce). Helps spot reversals and divergences with price.""",

    "Moving Average Convergence Divergence (MACD)":
        """**What is it?**  
        Difference between 12-day and 26-day EMAs, with a 9-day signal line.  
        **Usage:**  
        MACD crossing above signal line signals bullish momentum; crossing below signals bearish. Useful for trend and momentum shifts.""",

    "Average True Range (ATR)": 
        """**What is it?**  
        Measures market volatility by averaging the price range over a period.  
        **Usage:**  
        Higher ATR means higher volatility. Traders use ATR to set wider stop-losses in volatile markets.""",

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
