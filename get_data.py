# get_data.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time

def load_data(tickers, start, end):
    """
    Loads OHLCV data for multiple tickers.
    
    Args:
        tickers: dict {friendly_name: yahoo_ticker_symbol}
        start, end: dates
    
    Returns:
        DataFrame with MultiIndex columns (ticker, attribute)
        where attribute is one of: Open, High, Low, Close, Adj Close, Volume
    """
    data_dict = {}  # {ticker_name: DataFrame with OHLCV}
    
    for name, ticker in tickers.items():
        try:
            raw = yf.download(
                ticker, 
                start=start, 
                end=end, 
                progress=False,
                auto_adjust=False,   # keeps Open, High, Low, Close, Adj Close, Volume
                group_by='ticker' if len(tickers) > 1 else None
            )
            
            if raw.empty:
                st.warning(f"⚠️ No data for {ticker}. Creating placeholder.")
                date_range = pd.date_range(start=start, end=end, freq='B')
                empty_df = pd.DataFrame(index=date_range,
                                        columns=['Open','High','Low','Close','Adj Close','Volume'])
                data_dict[name] = empty_df
                continue
            
            # Ensure we have the necessary columns
            if 'Close' not in raw.columns:
                # Some futures data may not have Close; take first column as price
                raw['Close'] = raw.iloc[:, 0]
            if 'Open' not in raw.columns:
                raw['Open'] = raw['Close']
            if 'High' not in raw.columns:
                raw['High'] = raw['Close']
            if 'Low' not in raw.columns:
                raw['Low'] = raw['Close']
            if 'Volume' not in raw.columns:
                raw['Volume'] = 0
                
            # Keep only relevant columns
            keep_cols = ['Open','High','Low','Close','Adj Close','Volume']
            raw = raw[[c for c in keep_cols if c in raw.columns]]
            
            data_dict[name] = raw
            time.sleep(0.1)
            
        except Exception as e:
            st.warning(f"⚠️ Error loading {ticker}: {str(e)}")
            date_range = pd.date_range(start=start, end=end, freq='B')
            empty_df = pd.DataFrame(index=date_range,
                                    columns=['Open','High','Low','Close','Adj Close','Volume'])
            data_dict[name] = empty_df
    
    # Combine into a single DataFrame with MultiIndex columns
    combined = pd.concat(data_dict, axis=1)
    return combined
