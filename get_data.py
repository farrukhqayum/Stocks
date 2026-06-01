# get_data.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time

def load_data(tickers, start, end):
    """
    tickers: dict {friendly_name: yahoo_symbol}
    returns: dict {friendly_name: DataFrame with OHLCV columns}
    """
    data_dict = {}
    for name, symbol in tickers.items():
        try:
            df = yf.download(
                symbol,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False
            )
            if df.empty:
                st.warning(f"No data for {symbol}. Creating placeholder.")
                date_range = pd.date_range(start=start, end=end, freq='B')
                df = pd.DataFrame(index=date_range,
                                  columns=['Open','High','Low','Close','Adj Close','Volume'])
            # Ensure required columns exist
            if 'Open' not in df:
                df['Open'] = df['Close']
            if 'High' not in df:
                df['High'] = df['Close']
            if 'Low' not in df:
                df['Low'] = df['Close']
            if 'Volume' not in df:
                df['Volume'] = 0
            # Keep only needed columns
            keep = ['Open','High','Low','Close','Adj Close','Volume']
            df = df[[c for c in keep if c in df.columns]]
            data_dict[name] = df
            time.sleep(0.1)
        except Exception as e:
            st.warning(f"Error loading {symbol}: {e}")
            date_range = pd.date_range(start=start, end=end, freq='B')
            df = pd.DataFrame(index=date_range,
                              columns=['Open','High','Low','Close','Adj Close','Volume'])
            data_dict[name] = df
    return data_dict   # returns dict, not DataFrame
