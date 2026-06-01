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
        DataFrame with MultiIndex columns (ticker, price_type)
    """
    data_dict = {}  # will hold {ticker_name: DataFrame with OHLCV}
    
    for name, ticker in tickers.items():
        try:
            raw = yf.download(
                ticker, 
                start=start, 
                end=end, 
                progress=False,
                auto_adjust=False,   # keeps Open, High, Low, Close, Adj Close, Volume
                group_by='column'    # ensures MultiIndex if multiple tickers, but we do one by one
            )
            
            if raw.empty:
                st.warning(f"⚠️ No data for {ticker}. Creating placeholder.")
                # Create empty DataFrame with expected columns
                date_range = pd.date_range(start=start, end=end, freq='B')
                empty_df = pd.DataFrame(index=date_range,
                                        columns=['Open','High','Low','Close','Adj Close','Volume'])
                data_dict[name] = empty_df
                continue
            
            # raw has columns: Open, High, Low, Close, Adj Close, Volume
            # Ensure we have at least Close
            if 'Close' not in raw.columns:
                st.warning(f"⚠️ No Close price for {ticker}. Using first column.")
                raw['Close'] = raw.iloc[:, 0]
            
            # Keep all relevant columns
            keep_cols = ['Open','High','Low','Close','Adj Close','Volume']
            raw = raw[[c for c in keep_cols if c in raw.columns]]
            
            data_dict[name] = raw
            time.sleep(0.1)
            
        except Exception as e:
            st.warning(f"⚠️ Error loading {ticker}: {str(e)}")
            # Create empty placeholder
            date_range = pd.date_range(start=start, end=end, freq='B')
            empty_df = pd.DataFrame(index=date_range,
                                    columns=['Open','High','Low','Close','Adj Close','Volume'])
            data_dict[name] = empty_df
    
    # Combine all tickers into a MultiIndex DataFrame
    # Stack along columns: (ticker1, Open), (ticker1, High), ...
    combined = pd.concat(data_dict, axis=1)   # results in MultiIndex columns
    return combined
