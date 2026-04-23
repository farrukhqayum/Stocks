import requests
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
import time

# Private global variable
_API_KEY = None

def configure(api_key):
    """Set the Alpha Vantage API key (call once at app startup)."""
    global _API_KEY
    _API_KEY = api_key

@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data(ticker, start_date, end_date, interval='1d', max_retries=2):
    """
    Fetch stock data from Alpha Vantage.
    Supports daily and weekly intervals.
    
    Parameters:
    - ticker: str (e.g., 'AAPL')
    - start_date, end_date: datetime or date objects
    - interval: '1d' or '1wk'
    - max_retries: int
    
    Returns DataFrame with columns: Open, High, Low, Close, Volume
    Index is datetime.
    """
    if _API_KEY is None:
        st.error("Alpha Vantage API key not configured. Call configure() first.")
        return None
    
    ticker = ticker.strip().upper()
    
    # Map interval to Alpha Vantage function
    if interval.lower() in ('1wk', '1w'):
        function = 'TIME_SERIES_WEEKLY_ADJUSTED'
        ts_key = 'Weekly Adjusted Time Series'
    else:
        function = 'TIME_SERIES_DAILY_ADJUSTED'
        ts_key = 'Time Series (Daily)'
    
    url = "https://www.alphavantage.co/query"
    params = {
        "function": function,
        "symbol": ticker,
        "apikey": _API_KEY,
        "outputsize": "full"
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=15)
            data = response.json()
            
            if "Error Message" in data:
                st.error(f"Alpha Vantage error for {ticker}: {data['Error Message']}")
                return None
            if "Note" in data:  # Rate limit
                st.warning(f"Rate limit hit for {ticker}. Waiting 60 seconds...")
                time.sleep(60)
                continue
            
            time_series = data.get(ts_key)
            if not time_series:
                st.error(f"No time series data for {ticker}")
                return None
            
            df = pd.DataFrame.from_dict(time_series, orient="index")
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            # Rename columns to standard OHLCV
            df = df.rename(columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. adjusted close": "Close",
                "6. volume": "Volume"
            })
            
            # Keep only required columns
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            df = df.apply(pd.to_numeric)
            
            # Filter by date range
            mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
            df = df.loc[mask]
            
            if df.empty:
                st.warning(f"No data in date range for {ticker}")
                return None
            
            # Convert to float32 for memory efficiency
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = df[col].astype('float32')
            
            return df
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            st.error(f"Failed to fetch {ticker}: {e}")
            return None
    
    return None
