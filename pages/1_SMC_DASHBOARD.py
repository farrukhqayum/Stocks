st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker", "AAPL")

# Add a refresh button
col1, col2 = st.sidebar.columns([3, 1])
with col2:
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

# Load daily data for context
start_daily = datetime.today() - timedelta(days=365)
df_daily = load_data(ticker, start_daily, "1d")
if df_daily is None:
    st.error("No daily data")
    st.stop()

# Load hourly data using period method
try:
    df_hourly = yf.download(
        ticker, 
        period="10d",  # Get last 10 days including today
        interval="1h", 
        auto_adjust=False, 
        progress=False,
        prepost=False,
        repair=True,
        rounding=True
    )
    
    if df_hourly is None or df_hourly.empty:
        raise Exception("No hourly data from period method")
    
    # Clean up columns
    df_hourly.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df_hourly.columns]
    df_hourly.index = pd.to_datetime(df_hourly.index)
    df_hourly = df_hourly.dropna(subset=["open","high","low","close"]).astype(float)
    
    # Calculate indicators for hourly
    df_hourly['ema20'] = ema(df_hourly.close, 20)
    df_hourly['ema50'] = ema(df_hourly.close, 50)
    df_hourly['ema200'] = ema(df_hourly.close, 200)
    df_hourly['rsi'] = rsi(df_hourly.close, 14)
    df_hourly['rsi_ema'] = ema(df_hourly['rsi'], 14)
    df_hourly['atr'] = atr(df_hourly, 14)
    df_hourly['lb_crv'] = lb_curve(df_hourly, 10)
    df_hourly = df_hourly.bfill().ffill()
    
except Exception as e:
    st.warning(f"Hourly data unavailable, using 4H: {e}")
    df_hourly = yf.download(ticker, period="1mo", interval="4h", progress=False, repair=True)
    
    if df_hourly is None or df_hourly.empty:
        st.error("No intraday data")
        st.stop()
    
    # Clean 4H data
    df_hourly.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df_hourly.columns]
    df_hourly.index = pd.to_datetime(df_hourly.index)
    df_hourly = df_hourly.dropna(subset=["open","high","low","close"]).astype(float)
    
    # Calculate indicators for 4H
    df_hourly['ema20'] = ema(df_hourly.close, 20)
    df_hourly['ema50'] = ema(df_hourly.close, 50)
    df_hourly['ema200'] = ema(df_hourly.close, 200)
    df_hourly['rsi'] = rsi(df_hourly.close, 14)
    df_hourly['rsi_ema'] = ema(df_hourly['rsi'], 14)
    df_hourly['atr'] = atr(df_hourly, 14)
    df_hourly['lb_crv'] = lb_curve(df_hourly, 10)
    df_hourly = df_hourly.bfill().ffill()

# Display data info in sidebar - FIXED VERSION
if df_hourly is not None and not df_hourly.empty:
    last_date = df_hourly.index[-1]
    # Convert pandas Timestamp to datetime for subtraction
    last_date_dt = last_date.to_pydatetime() if hasattr(last_date, 'to_pydatetime') else pd.Timestamp(last_date).to_pydatetime()
    today = datetime.now()
    
    st.sidebar.info(f"📅 Hourly data up to: {last_date.strftime('%Y-%m-%d %H:%M')}")
    
    # Calculate hours difference
    time_diff = today - last_date_dt
    hours_old = time_diff.total_seconds() / 3600
    
    if hours_old > 2:
        st.sidebar.warning(f"⚠️ Data is {hours_old:.1f} hours old")
    else:
        st.sidebar.success(f"✅ Data is {hours_old:.1f} hours old")
    
    # Show if we have today's data
    if last_date.date() == today.date():
        st.sidebar.success("✅ Including today's data")
    else:
        st.sidebar.warning(f"⚠️ No data for today ({today.strftime('%Y-%m-%d')})")
else:
    st.error("Failed to load hourly data")
    st.stop()
