import streamlit as st

st.header("About This App")
st.write(
  """
  This Streamlit app provides a high-level platform for predicting stock trading opportunities using machine learning, with an emphasis on practical trading decisions to achieve 3–10% compounded growth for swing traders.
  """)

st.header("Overview")

st.write(
"The app allows the user to input up to 20 tickers. For each ticker, it fetches historical data, applies advanced technical analysis, and generates trading signals using machine learning models. Predictions include hit probabilities, expected gains, loss forecasts, and actionable labels (such as Buy, Hold, Sell, or Short)."

st.header("Step-by-Step Process")
st.write(
"""
1. Data Entry and Setup
The user enters a comma-separated list of up to 20 stock symbols (tickers).

The app fetches historical and fundamental data for each symbol, ensuring robust data preparation and cleaning before analysis.

2. Feature Engineering
Computes numerous technical indicators: various SMAs, RSI, CCI, ADX, ATR, MACD, volume features, volatility, pivot points, and more.

Integrates market sentiment signals and candlestick patterns to enrich the signal generation process.

3. Prediction Engine
Machine learning models (RandomForest and XGBoost) predict the likelihood of the next swing reaching a 3–10% gain or risking a 4–5% loss.

The script classifies each future event as Take-Profit, Stop-Loss, Hold, or Neutral, attaching probabilities and confidence scores to each forecast.

4. Visualization and Summaries
Key results are displayed in colored tables and annotated charts:

Green = buy opportunities (“Buy the Dip”).

Red = sell/short warnings (“Sell the Rise”).

Gray/Magenta = hold zones or exhaustion, to avoid trading.

For each ticker, the app shows a summary of predicted return, risk, hit probability, and signal direction, supporting easy comparison.

Usage Guidelines
Focus trades on signals marked “Bullish” with high confidence and a strong probability to reach target gains.

Prioritize entries where:

The RSI recovers above its moving average or is above 52.

Price is above key moving averages.

Strong volume accompanies the move.

Avoid chasing weak signals or entering when the risk-to-reward is not favorable. The app will highlight opportunities where expected gain is high and expected loss is low.

Ideal positions build on weekly timeframes, with tactical 1H/4H entries for precision. Split orders for risk management, and use divergence or double bottom patterns as confirmation.

Use the built-in tables and filters to identify optimal tickers to focus on, aiming for compounded gains in the target range. It’s better to enter late on confirmation than prematurely.

Risk Disclaimer
Trading involves substantial risk. Results are for educational purposes, and past performance does not guarantee future results. Always perform independent research and stick to personal risk limits.

Overall, the app is designed for swing traders seeking systematic, machine-learning-informed entries and exits, supporting robust decision-making to compound moderate gains while minimizing downside risk.
""")
