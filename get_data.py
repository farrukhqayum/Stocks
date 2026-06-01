def load_data(tickers, start, end):
    """Load data from Yahoo Finance - handles stocks, futures, indices"""
    
    ticker_list = list(tickers.values())
    data_dict = {}
    
    for name, ticker in tickers.items():
        try:
            raw = yf.download(
                ticker, 
                start=start, 
                end=end, 
                progress=False,
                auto_adjust=False
            )
            
            if raw.empty:
                st.warning(f"⚠️ No data for {ticker}. Creating placeholder.")
                date_range = pd.date_range(start=start, end=end, freq='B')
                data_dict[name] = pd.Series(np.nan, index=date_range, name=name)
                continue
            
            if 'Adj Close' in raw.columns:
                price_series = raw['Adj Close']
            elif 'Close' in raw.columns:
                price_series = raw['Close']
            else:
                price_series = raw.iloc[:, 0] if raw.shape[1] > 0 else pd.Series()
            if isinstance(price_series, pd.DataFrame):
                price_series = price_series.iloc[:, 0]
            
            price_series.name = name
            data_dict[name] = price_series
            time.sleep(0.1)
            
        except Exception as e:
            st.warning(f"⚠️ Error loading {ticker}: {str(e)}")
            date_range = pd.date_range(start=start, end=end, freq='B')
            data_dict[name] = pd.Series(np.nan, index=date_range, name=name)

    df = pd.DataFrame(data_dict)
    df = df.dropna(axis=1, how='all')
    
    return df
