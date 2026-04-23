import requests
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
import time

API_KEY = None  # Will be set by the app

def configure(api_key):
    global _API_KEY
    API_KEY = api_key

@st.cache_data(ttl=3600)  # Cache for 1 hour to respect rate limits
def get_stock_data(ticker, start_date, end_date, max_retries=2):
    """
    Fetch daily adjusted stock data from Alpha Vantage.
    Returns DataFrame with columns: Open, High, Low, Close, Volume
    Index is datetime.
    """
    if API_KEY is None:
        raise ValueError("Alpha Vantage API key not configured. Call configure() first.")
    
    # Alpha Vantage endpoint for daily adjusted
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "apikey": API_KEY,
        "outputsize": "full"  # Get full history
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=15)
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                st.error(f"Alpha Vantage error for {ticker}: {data['Error Message']}")
                return None
            if "Note" in data:  # Rate limit message
                st.warning(f"Rate limit hit for {ticker}. Waiting 60 seconds...")
                time.sleep(60)
                continue
                
            time_series = data.get("Time Series (Daily)")
            if not time_series:
                st.error(f"No time series data for {ticker}. Response: {data}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient="index")
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns to standard OHLCV
            df = df.rename(columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. adjusted close": "Close",  # Use adjusted close as primary
                "6. volume": "Volume"
            })
            
            # Keep only needed columns (adjusted close overrides close)
            df["Close"] = df.get("5. adjusted close", df["Close"])
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            
            # Convert to numeric
            df = df.apply(pd.to_numeric)
            
            # Filter by date range
            mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
            df = df.loc[mask]
            
            if df.empty:
                st.warning(f"No data in date range for {ticker}")
                return None
                
            return df
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            st.error(f"Failed to fetch {ticker}: {e}")
            return None
    return None
