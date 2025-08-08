#!/usr/bin/env python
# coding: utf-8

# In[1]:


from imports import *


# In[2]:


# GLOBAL PARAMETERS
today = datetime.now().strftime('%Y-%m-%d') # For printing/filenames
path = r'C:\Users\Farrukh\jupyter-Notebooks\STOCKS\ML_TP_SL_Figures' # CHECK THIS PATH / CREATE THE FOLDER
pdf_path = os.path.join(path, f'{today}_ML_TA_MultipleStocks.pdf')
pred_file = os.path.join(path, "tp_sl_daily.xlsx")

plt.rcParams['font.family'] = 'Segoe UI Emoji' # Matplotlib Font Family for windows.

##### STOCKS ##########
TICKERS = ["COIN", "TSLA", "GOOGL", "NVDA", "AAPL", "NKE", "SMCI", "BABA", "XPEV", "NIO", "XYZ", "U"]

##### CRYPTOS ##########
#TICKERS = ["BTC-USD","ETH-USD", "XRP-USD", "SOL-USD", "ADA-USD", "DOGE-USD", "LTC-USD", "BCH-USD"]

isStockCrypto = "CRYPTO"

if "BTC-USD" in TICKERS:
    isStockCrypto = "CRYPTO"
else:
    isStockCrypto = "STOCKS"

_Nr = 50 # Skip model if the length is this
YEARS_OF_DATA = 2
PROFIT_TARGET = 0.07
STOP_LOSS = 0.07
_DAYS = 20 # Used for SMA and training
_FWDAYS = 14 # Forward days to plot stored data
windows = [3, 5, 7, 10, 13, 15, 20, 30, 40, 50] # For calculating returns
_window = 9  # Backtesting
tolerance = 1.07
_FIBS = False
_FibLen = 20 # Scan pivots for fibonacci levels
_ms = 7 # global marker size for matplotlib

bold = '\033[1m'
end = '\033[0m'

# Time window
end_date = datetime.now()
start_date = end_date - timedelta(days=365 * YEARS_OF_DATA)

# Shared model components

FEATURES = [
    # Technical Indicators
    'RSI', 'RSI_SMA', 'CCI', 'OBV', '+DI', '-DI', 'ADX', 'ATR', 'VWMA', 'VI+', 'KCu', 'KCl', 'STu', 'STl', 'MFI',

    # Moving Averages & Bands
    'SMA1', 'SMA2', 'SMA3', 'SMA_Ratio', 'Upper_Band', 'Lower_Band', 'Volume_MA20',

    # Returns & Volatility
    'return1', 'return2', 'return3', 'Volatility', 'Scaled_Volatility', 'DD',

    # Volume Features
    'sumBuyVol', 'sumSellVol', 'vSpike',

    # Candlestick Patterns
    'Candlesticks', 'gapStrength',

    # Market Sentiment & Signals
    'Bear', 'Bull', 'Neutral', 'StrongBull', 'StrongBear', 'Neutral',

    # PIVOTS
    'PP_Avg', 'R1_Avg', 'R2_Avg', 'S1_Avg', 'S2_Avg'
]

results = []


# In[3]:


# Functions
def get_stock_data(ticker, start_date, end_date):
    #print("Getting data for:   ", ticker)
    df = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1), 
                     interval='1d', auto_adjust=False, progress=False)
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.dropna()
    return df

def get_fundamentals(ticker: str):
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # Extract fundamentals safely with defaults if missing
    fundamentals = {
        'Market Cap': info.get('marketCap', 'N/A'),
        'Net Profit Margin': info.get('netMargins', 'N/A'),
        'PE Ratio': info.get('trailingPE', 'N/A'),
        'Quick Ratio': info.get('quickRatio', 'N/A'),
        'Long Term Debt': info.get('longTermDebt', 'N/A'),
        'Free Cash Flow': info.get('freeCashflow', 'N/A')
    }
    
    # Format numeric values for better readability
    def fmt(value):
        if isinstance(value, (int, float)) and value != 'N/A':
            if abs(value) > 1e9:
                return f"{value/1e9:.2f}B"
            elif abs(value) > 1e6:
                return f"{value/1e6:.2f}M"
            elif abs(value) > 1e3:
                return f"{value/1e3:.0f}K"
            else:
                return f"{value:.2f}"
        return value
    
    return {k: fmt(v) for k, v in fundamentals.items()}

def add_fundamentals_table(ax, fundamentals, loc='upper left', alpha=0.4):
    # Prepare data: rows = one column headers + all rows
    col_labels = ['Metric', 'Value']
    cell_text = [[k, v] for k, v in fundamentals.items()]

    # Choose bbox location in axes coordinates (x0, y0, width, height)
    # Tune these numbers to position nicely
    bbox_dict = {
        'upper left': [0.01, 0.95, 0.3, 0.25],
        'upper right': [0.68, 0.95, 0.3, 0.25],
        'lower left': [0.01, 0.05, 0.3, 0.25],
        'lower right': [0.68, 0.05, 0.3, 0.25]
    }
    bbox = bbox_dict.get(loc, [0.02, 0.95, 0.3, 0.25])

    # Build cell colours with alpha transparency
    base_color = [1, 1, 1, alpha]  # white with alpha
    header_color = [0.8, 0.8, 0.8, alpha]  # light gray with alpha

    cell_colours = [  # One row per data row (no header here)
        [base_color, base_color] for _ in cell_text
    ]

    # Insert header background colors (first row does not exist here, colLabels handled separately)

    # Add the table to the axes
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='left',
        colLoc='left',
        colWidths=[0.4, 0.6],
        bbox=bbox,
        cellColours=cell_colours,
        edges='open'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell._loc = 'center'   # <- this line centers the header text
            cell.set_text_props(weight='bold')
            cell.set_facecolor(header_color)
        else:
            cell.set_edgecolor('none')

    return table

def add_technical_indicators(df):
    df['SMA1'] = df['Close'].rolling(window=int(_DAYS*0.5)).mean()
    df['SMA2'] = df['Close'].rolling(window=_DAYS).mean()
    df['SMA3'] = df['Close'].rolling(window=int(_DAYS*2)).mean()
    df['SMA_Ratio'] = df['SMA1'] / df['SMA2']

    df['RSI']= ta.calculate_rsi(df)
    df['RSI_SMA'] = df['RSI'] / df['RSI'].rolling(14).mean()
    '''
    df['Bear'] = (df['SMA1'] < df['SMA2']).astype(int)
    df['Bull'] = (df['SMA2'] < df['SMA1']).astype(int)
    '''
    
    bull_condition = (df['SMA1'] > df['SMA2']) & (df['RSI'] > 52) & (df['RSI_SMA'] < df['RSI'])
    bear_condition = (df['Close'] < df['SMA2']) & (df['RSI'] < 42)

    df['Bull'] = bull_condition.astype(int)
    df['Bear'] = bear_condition.astype(int)
    df['Neutral'] = (~(bull_condition | bear_condition)).astype(int)

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=24, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Upper_Band'] = df['SMA1'] + (2 * df['Close'].rolling(20).std())
    df['Lower_Band'] = df['SMA1'] - (2 * df['Close'].rolling(20).std())
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    df['buy_volume'] = (df.Close > df.Close.shift(1)) * df['Volume']
    df['sell_volume'] = (df.Close < df.Close.shift(1)) * df['Volume']
    df['sumBuyVol'] = df['buy_volume'].rolling(window=9).sum()
    df['sumSellVol'] = df['sell_volume'].rolling(window=9).sum()
    df['vSpike'] = (df['Volume'] > 2 * df['Volume_MA20']).astype(int)
    df['MFI'] = ta.calculate_mfi(df)
    
    df['CCI'] = ta.calculate_cci(df)
    df['OBV'] = ta.calculate_obv(df)
    df[['+DI', '-DI', 'ADX']] = ta.calculate_dmi(df, n=14)
    df['ATR'] = ta.calculate_atr(high=df.High, low=df.Low, close=df.Close)
    
    df['VWMA'] = ta.calculate_vwma(df)
    df[['KCm', 'KCu', 'KCl']] = ta.calculate_keltner(df)
    df[['VI+', 'VI-']] = ta.calculate_vortex(df)
    df[['STu', 'STl']] = ta.calculate_supertrend(df)
    
    df = ta.add_candlestickpatterns(df)
    
    df['DD'] = df['Close'].where(df['Close'] < df['Close'].shift(1)).std()

    df['return1'] = df['Close'].pct_change(9)
    df['return2'] = df['Close'].pct_change(20)
    df['return3'] = df['Close'].pct_change(50)
    df['Volatility'] = df['Close'].rolling(20).std()
    df = ta.scaled_volatility(df)
    # StrongBull, StrongBear, and Neutral as features
    df['StrongBull'] = ((df['RSI'] > 52) & (df['ADX'] > 22) & (df['sumBuyVol'] > df['sumSellVol'])).astype(int)
    df['StrongBear'] = ((df['RSI'] < 40) & (df['ADX'] > 22) & (df['sumBuyVol'] < df['sumSellVol'])).astype(int)
    df['Neutral'] = (~(df['StrongBull'].astype(bool) | df['StrongBear'].astype(bool))).astype(int)
    df['gapStrength'] = ta.compute_gapStrength(df)

    return df

def add_pivot_levels(df, window=_DAYS):
    # Compute rolling high/low/close over the window
    high = df['High'].rolling(window)
    low = df['Low'].rolling(window)
    close = df['Close'].rolling(window)
    # Classic floor trader pivots (you can adjust formulas as needed)
    PP = (high.max() + low.min() + close.apply(lambda x: x[-1])).div(3)
    R1 = 2 * PP - low.min()
    S1 = 2 * PP - high.max()
    R2 = PP + (high.max() - low.min())
    S2 = PP - (high.max() - low.min())
    # Assign to DataFrame
    df['PP'] = PP
    df['R1'] = R1
    df['S1'] = S1
    df['R2'] = R2
    df['S2'] = S2
    return df

def add_pivots(df, win=windows):
    for w in win:
        roll_high = df['High'].rolling(w)
        roll_low = df['Low'].rolling(w)
        roll_close = df['Close'].rolling(w)
        # Calculate rolling pivots
        PP = (roll_high.max() + roll_low.min() + roll_close.apply(lambda x: x[-1])).div(3)
        R1 = 2 * PP - roll_low.min()
        S1 = 2 * PP - roll_high.max()
        R2 = PP + (roll_high.max() - roll_low.min())
        S2 = PP - (roll_high.max() - roll_low.min())
        # Store in DataFrame
        df[f'PP_{w}'] = PP
        df[f'R1_{w}'] = R1
        df[f'S1_{w}'] = S1
        df[f'R2_{w}'] = R2
        df[f'S2_{w}'] = S2
    return df

def average_pivots(df, windows=[5, 10, 14, 20]):
    for level in ['PP', 'R1', 'S1', 'R2', 'S2']:
        cols = [f'{level}_{w}' for w in windows]
        # Take row-wise mean, ignore NaN for early rows
        df[f'{level}_Avg'] = df[cols].mean(axis=1)
    return df
    
def compute_expected_return(df, forward_window=14, r_cols=['R1', 'R2']):
    df['Expected_Return'] = np.nan
    close_prices = df['Close'].values
    for i in range(len(df) - forward_window):
        current_price = close_prices[i]
        # Find resistance pivots for this row (skip if all NaN)
        pivots = [df.iloc[i][col] for col in r_cols if col in df.columns and not pd.isnull(df.iloc[i][col])]
        target_level = None
        if pivots:  # Only if we have any non-NaN pivots
            target_level = max(pivots)  # Or use your preferred pivot selection logic

        future_window = close_prices[i+1:i+1+forward_window]
        if target_level is not None:
            # Check if we hit target pivot in the window
            for j, future_price in enumerate(future_window):
                if future_price >= target_level:
                    df.iloc[i, df.columns.get_loc('Expected_Return')] = (target_level - current_price) / current_price
                    break
            else:  # If not hit
                if future_window.size > 0:
                    future_max = np.nanmax(future_window)
                    df.iloc[i, df.columns.get_loc('Expected_Return')] = (future_max - current_price) / current_price
        else:
            # No valid pivots, fall back to window max logic
            if future_window.size > 0:
                future_max = np.nanmax(future_window)
                df.iloc[i, df.columns.get_loc('Expected_Return')] = (future_max - current_price) / current_price
            else:
                df.iloc[i, df.columns.get_loc('Expected_Return')] = np.nan
    return df

def compute_expected_loss(df, forward_window=14, s_cols=['S1', 'S2']):
    df['Expected_Loss'] = np.nan
    close_prices = df['Close'].values
    for i in range(len(df) - forward_window):
        current_price = close_prices[i]
        # Find support pivots (skip if all NaN)
        pivots = [df.iloc[i][col] for col in s_cols if col in df.columns and not pd.isnull(df.iloc[i][col])]
        target_level = None
        if pivots:
            target_level = min(pivots)  # Or your support logic

        future_window = close_prices[i+1:i+1+forward_window]
        if target_level is not None:
            # Check if we hit pivot support in window
            for j, future_price in enumerate(future_window):
                if future_price <= target_level:
                    df.iloc[i, df.columns.get_loc('Expected_Loss')] = (target_level - current_price) / current_price
                    break
            else:
                if future_window.size > 0:
                    future_min = np.nanmin(future_window)
                    df.iloc[i, df.columns.get_loc('Expected_Loss')] = (future_min - current_price) / current_price
        else:
            # No valid pivots, fallback to window min logic
            if future_window.size > 0:
                future_min = np.nanmin(future_window)
                df.iloc[i, df.columns.get_loc('Expected_Loss')] = (future_min - current_price) / current_price
            else:
                df.iloc[i, df.columns.get_loc('Expected_Loss')] = np.nan
    return df

def initialize_XGBR():
    model = XGBRegressor(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42
    )
    return model

def label_hit2(df, window=14, profit_target=0.03, stop_loss=0.03):
    labels = []
    close_prices = df['Close'].values

    for i in range(len(close_prices) - window):
        current_price = close_prices[i]
        tp = current_price * (1 + profit_target)
        sl = current_price * (1 - stop_loss)
        future_prices = close_prices[i + 1:i + 1 + window]

        tp_hit_idx = next((j for j, price in enumerate(future_prices) if price >= tp), None)
        sl_hit_idx = next((j for j, price in enumerate(future_prices) if price <= sl), None)

        if tp_hit_idx is not None and (sl_hit_idx is None or tp_hit_idx < sl_hit_idx):
            labels.append(2)  # TP hit before SL
        elif sl_hit_idx is not None and (tp_hit_idx is None or sl_hit_idx < tp_hit_idx):
            labels.append(1)  # SL hit before TP
        else:
            labels.append(0)  # Neither hit

    labels += [np.nan] * window
    df['Hit_Label'] = labels
    return df

def label_hit3(df, window=14, profit_target=0.03, stop_loss=0.03):
    labels = []
    close_prices = df['Close'].values

    for i in range(window, len(close_prices)):
        entry_price = close_prices[i]
        tp = entry_price * (1 + profit_target)
        sl = entry_price * (1 - stop_loss)
        # Look BACKWARD: previous `window` bars
        past_highs = df['High'].values[i-window:i]
        past_lows = df['Low'].values[i-window:i]

        tp_hit_idx = next((j for j, price in enumerate(reversed(past_highs)) if price >= tp), None)
        sl_hit_idx = next((j for j, price in enumerate(reversed(past_lows)) if price <= sl), None)

        if tp_hit_idx is not None and (sl_hit_idx is None or tp_hit_idx < sl_hit_idx):
            labels.append(1)
        else:
            labels.append(0)

    # Pad the beginning to align with df length
    labels = [np.nan]*window + labels
    df['Hit_Label'] = labels
    return df

def compute_optimal_entry(df, _DAYS=10, profit_target=0.05, stop_loss=-0.03):
    optimal_entries = []
    
    for i in range(len(df) - _DAYS):
        entry_price = df['Close'].iloc[i]
        future_data = df.iloc[i+1:i+1+_DAYS]
        
        min_price = future_data['Low'].min()  # Best possible entry
        max_price = future_data['High'].max()  # Highest possible gain

        # Check if TP or SL hit
        tp_price = entry_price * (1 + profit_target)
        sl_price = entry_price * (1 + stop_loss)

        tp_hit = (future_data['High'] >= tp_price).any()
        sl_hit = (future_data['Low'] <= sl_price).any()

        if tp_hit and not sl_hit:
            optimal_entry = min_price  # You had time to enter lower before TP
        elif sl_hit and not tp_hit:
            optimal_entry = entry_price  # Didn't get a better chance
        elif tp_hit and sl_hit:
            # Whichever came first
            first_tp_idx = future_data[future_data['High'] >= tp_price].index[0]
            first_sl_idx = future_data[future_data['Low'] <= sl_price].index[0]
            optimal_entry = min_price if first_tp_idx < first_sl_idx else entry_price
        else:
            optimal_entry = entry_price  # No TP/SL hit, assume flat

        optimal_entries.append(optimal_entry)

    # Align with DataFrame length
    df['Optimal_Entry'] = [np.nan]*_DAYS + optimal_entries
    return df

def compute_expected_entry(df, n=3):
    df['Expected_Entry'] = df['Low'].rolling(window=n, min_periods=1).min().shift(-n)
    return df
    
def label_hit(df, window=14, profit_target=0.03, stop_loss=0.03):
    """
    Label each row:
    1 = TP hit before SL
    0 = SL hit before TP or neither hit
    """
    labels = []
    close_prices = df['Close'].values

    for i in range(len(close_prices) - window):
        current_price = close_prices[i]
        tp = current_price * (1 + profit_target)
        sl = current_price * (1 - stop_loss)
        future_prices = close_prices[i + 1:i + 1 + window]

        tp_hit_idx = next((j for j, price in enumerate(future_prices) if price >= tp), None)
        sl_hit_idx = next((j for j, price in enumerate(future_prices) if price <= sl), None)

        if tp_hit_idx is not None and (sl_hit_idx is None or tp_hit_idx < sl_hit_idx):
            labels.append(1)
        else:
            labels.append(0)

    # Fill remaining with NaN to keep alignment
    labels += [np.nan] * window
    df['Hit_Label'] = labels
    return df

def get_recent_fib_levels(df, left=_FibLen, right=_FibLen):
    # Step 1: Find pivot highs/lows
    highs = df['High']
    lows = df['Low']
    is_pivot_high = highs == highs.rolling(window=left+right+1, center=True).max()
    is_pivot_low = lows == lows.rolling(window=left+right+1, center=True).min()
    is_pivot_high = is_pivot_high.fillna(False)
    is_pivot_low = is_pivot_low.fillna(False)

    # Step 2: Get most recent swing high and low
    pivot_highs = df[is_pivot_high]
    pivot_lows = df[is_pivot_low]
    if pivot_highs.empty or pivot_lows.empty:
        return None, None, None  # Not enough data

    last_high_idx = pivot_highs.index[-1]
    last_low_idx = pivot_lows.index[-1]
    high = df.loc[last_high_idx, 'High']
    low = df.loc[last_low_idx, 'Low']

    # Step 3: Calculate Fib levels
    diff = high - low
    fibs = {
        'F:0': low,
        'F:100': high,
        'F:61.8': high - 0.618 * diff,
        'F:125': high + 1.25 * diff,
        'F:-125': low - 1.25 * diff,
    }
    # For plotting, use the range between the pivots
    fib_start = min(last_high_idx, last_low_idx)
    fib_end = max(last_high_idx, last_low_idx)
    return fibs, fib_start, fib_end

def del_old_files (directory, days, exclude_extensions=None, dry_run=False):
    """Delete files older than `days` without returning a list."""
    if exclude_extensions is None:
        exclude_extensions = []

    cutoff_time = datetime.now() - timedelta(days=days)
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isdir(filepath) or any(filename.lower().endswith(ext.lower()) for ext in exclude_extensions):
            continue
        file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
        if file_mtime < cutoff_time:
            if dry_run:
                print(f"[Dry Run] Would delete: {filepath}")
            else:
                try:
                    os.remove(filepath)
                    print(f"Deleted: {filepath}")
                except Exception as e:
                    print(f"Error deleting {filepath}: {e}")
       
def append_pred(df, fpath):
    cols = ['Ticker', 'Date', 'Price', 'TP', 'SL']
    new_data = df[cols].copy()
    new_data['Date'] = pd.to_datetime(new_data['Date'])
    
    if os.path.exists(fpath):
        old_data = pd.read_excel(fpath)
        old_data['Date'] = pd.to_datetime(old_data['Date'])
        # Filter out rows from old_data that are duplicated in new_data by Ticker+Date
        mask = old_data.set_index(['Ticker', 'Date']).index.isin(new_data.set_index(['Ticker', 'Date']).index)
        old_data = old_data[~mask]
        # Combine without duplicates (new_data overwrites)
        combined = pd.concat([old_data, new_data], ignore_index=True)
    else:
        combined = new_data
    
    # Keep only latest 20 rows per Ticker, sorted by Date
    combined = (
        combined
        .sort_values(['Ticker', 'Date'])
        .groupby('Ticker', as_index=False)
        .tail(20)
        .reset_index(drop=True)
    )
    
    combined.to_excel(fpath, index=False)

def colored_row(text, color):
    colors = {
        'green': '\033[92m',
        'red': '\033[91m',
        'white': '\033[97m'
    }
    reset = '\033[0m'
    color_code = colors.get(color, colors['white'])
    return f"{color_code}{text}{reset}"

def color_signal(row):
    signal = row['Signal']
    if 'Bullish' in signal:
        return '\033[92m' + signal + '\033[0m'  # Green
    elif 'Bearish' in signal:
        return '\033[91m' + signal + '\033[0m'  # Red
    else:
        return '\033[93m' + signal + '\033[0m'  # Yellow for Neutral


# In[4]:


# PRICE CHARTS
def plot_single_ticker(ticker, df, df_results, _window=14):
    # Get predictions
    predictions = df_results[df_results['Ticker'] == ticker].iloc[0]

    ## --- Technical Market Summary ---    
    signal = predictions.Signal
    current_price = round(df['Close'].iloc[-1], 2)
    gain = round(predictions['Max (%)'], 1)
    loss = round(predictions['Loss (%)'], 1)
    gain_price = current_price * (1 + gain/100)
    loss_price = current_price * (1 + loss/100)
    hit_prob = predictions.Hit_Prob
    
    last_date = df.index[-1]
    future_date = last_date + pd.Timedelta(days=_window)
    avg_price = (current_price+loss_price)/2.
    sma1_ = round(df['SMA1'].iloc[-1], 2)
    sma2_ = round(df['SMA2'].iloc[-1], 2)
    
    # Create figure with white background
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), dpi = 300, height_ratios=[3, 1], sharex=True)
    #fig.patch.set_facecolor('white')
    
    # Get true trailing 12 months of data (not calendar YTD)
    end_date = df.index[-1]
    start_date = end_date - pd.DateOffset(months=12)
    df = df.loc[start_date:end_date]
    
    # ===== 1. PRICE PLOT =====
    # Configure plot style
    #ax1.set_facecolor('white')
    ax1.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)

    # Signal for color
    df['Signal'] = np.select(
    [df['Bull']==1, df['Bear']==1],
    ['Bull', 'Bear'],
    default='Neutral'
    )
    
    # Smooth the price (3-periods) to remove outliers, the last price may also be not visible
    price = df['Close'].rolling(3).mean()
    price.iloc[-1] = df['Close'].iloc[-1]
    
    color_map = {'Bull': 'green', 'Bear': 'red', 'Neutral': 'gray'}
    last_signal = df['Signal'].iloc[0]
    start_idx = 0
    
    for idx, (date, row) in enumerate(df.iterrows()):
        is_last = (idx == len(df) - 1)
        
        if row['Signal'] != last_signal or is_last:
            seg_idx = slice(start_idx, idx + 1)
            seg_price = price.iloc[seg_idx]
            seg_dates = df.index[seg_idx]
            ax1.plot(seg_dates, seg_price, color=color_map[last_signal], alpha=0.4, linewidth=2)
            start_idx = idx
            last_signal = row['Signal']

    # Historical data
    # Price is 3-days mean to avoid noise
    #ax1.plot(df.index, price, label= f'Price: ${current_price}', color='gray', alpha=0.7, linewidth=1.5)
    ax1.plot(df.index, df['SMA1'], label=f'SMA{int(_DAYS*0.5)}: ${sma1_}', color='gold', alpha=0.7, linewidth=1.2)
    ax1.plot(df.index, df['SMA2'], label=f'SMA{int(_DAYS*2)}: ${sma2_}', color='red', alpha=0.7, linewidth=1.2, linestyle='--')

    ta.add_regression_forecast(ax1, df['SMA1'], last_date, _DAYS, color='gold')

    ta.add_regression_forecast(ax1, df['SMA2'], last_date, _DAYS, color='red')
    
    # Fill between SMAs - green when SMA1 > SMA2, red otherwise
    ax1.fill_between(df.index, df['SMA1'], df['SMA2'],
                    where=(df['SMA1'] > df['SMA2']),
                    facecolor='green', alpha=0.1, interpolate=True)
    
    ax1.fill_between(df.index, df['SMA1'], df['SMA2'],
                    where=(df['SMA1'] <= df['SMA2']),
                    facecolor='red', alpha=0.1, interpolate=True)

    
    # Add stock's fundamental info box
    fundamentals = get_fundamentals(ticker)
    add_fundamentals_table(ax1, fundamentals, loc='lower left', alpha=0.6)

    # PLOT STORED PAST PREDICTIONS
    
    try:
        if 'df_hist' not in locals():
            df_hist = pd.read_excel(pred_file)
            df_hist['Date'] = pd.to_datetime(df_hist['Date'])
        points = df_hist[(df_hist['Ticker'] == ticker)].sort_values('Date')
        if not points.empty:
            points = points.copy()
            points['Date_Lagged'] = points['Date'] + pd.Timedelta(days=_FWDAYS)
            #ax1.scatter(points['Date_Lagged'], points['Price'], color='gray', marker='o', s=_ms, zorder=10, alpha=0.4)
            ax1.scatter(points['Date_Lagged'], points['TP'], color='green', marker='^', s=_ms, zorder=10, alpha=0.4)
            ax1.scatter(points['Date_Lagged'], points['SL'], color='red', marker='v', s=_ms, zorder=10, alpha=0.4)
    except Exception as e:
        print(f"Prediction plotting error for {ticker}: {e}")

    # Add earning date to the bottom
    data_ymin = df['Close'].min()
    data_ymax = df['Close'].max()
    price_margin = (data_ymax - data_ymin) * 0.1
    ymin_fixed = data_ymin - price_margin * 0.3
    ymax_fixed = data_ymax + price_margin * 0.2
    ax1.set_ylim(ymin_fixed, ymax_fixed)  # Set limits manually so they won't be autoscaled later

    first_date = df.index[0]
    last_date = df.index[-1]
    earnings_date = ta.get_next_earnings_date(ticker)
    
    if earnings_date is not None:
        extended_range = pd.Timedelta(days=7)
        
        if (first_date <= earnings_date <= last_date) or (last_date < earnings_date <= last_date + extended_range):
    
            y_pos = data_ymin + 0.05 * (data_ymax - data_ymin)
    
            # If just outside the range, pin label to right edge of chart
            x_pos = earnings_date if earnings_date <= last_date else last_date
    
            ax1.text(
                x_pos,
                y_pos,
                'E',
                fontsize=10,
                fontweight='bold',
                ha='center',
                va='bottom',
                color='blue',
                bbox=dict(boxstyle='round,pad=0.2', fc='lightblue', ec='blue', lw=0.5, alpha=0.5),
                zorder=10
            )

    
    # --- Add Fibonacci Levels ---
    if (_FIBS):
        fibs, fib_start, fib_end = get_recent_fib_levels(df)
        fib_colors = {
            'F:0': 'gray',
            'F:100': 'gray',
            'F:61.8': 'blue',
            'F:125': 'green',
            'F:-125': 'red',
        }
        for label, value in fibs.items():
            ax1.hlines(value, xmin=fib_start, xmax=fib_end, color=fib_colors[label], linestyle='--', linewidth=1, alpha=0.3)
            ax1.annotate(f'{label}: ${value:.0f}', xy=(df.index[-5], value), 
                         xytext=(-5, 0), textcoords='offset points', 
                         va='center', fontsize=8, color=fib_colors[label], alpha=0.5)
    
    
    # Connect lines
    ax1.plot([last_date, future_date], [avg_price, gain_price], 
             color='green', linestyle=':', linewidth=1.5, alpha=0.5)
    ax1.plot([last_date, future_date], [avg_price, loss_price], 
             color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    
    # Markers For Key levels
    ax1.plot(future_date, gain_price, '^', markersize=_ms, color='green', alpha=0.5, label=f'Projected Gain: {gain}%')
    ax1.plot(future_date, loss_price, 'v', markersize=_ms, color='red', alpha=0.5, label=f'Projected Loss: {loss}%')
    ax1.plot(last_date, avg_price, 'o', markersize=_ms, color='orange', alpha=0.5, label='Entry')

    ax1.annotate(f'Avg: ${avg_price:.2f}', 
                xy=(last_date, avg_price),
                xytext=(10, 0),  # 10 points to the right of the marker
                textcoords='offset points',
                ha='left', 
                va='center',
                color='orange',
                fontsize=9,
                bbox=dict(facecolor='white', 
                         alpha=0.5, 
                         edgecolor='none'))
    
    ax1.annotate(f'${current_price}\t-\t${gain_price:.2f}\n+{predictions["Max (%)"]:.1f}%', 
                xy=(future_date, gain_price),
                xytext=(10, 10), textcoords='offset points',
                ha='left', va='bottom', color='green', fontsize=9, 
                fontname='Segoe UI Emoji',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    
    ax1.annotate(f'${current_price}\t-\t${loss_price:.2f}\n{predictions["Loss (%)"]:.1f}%', 
                xy=(future_date, loss_price),
                xytext=(10, -10), textcoords='offset points',
                ha='left', va='top', color='red', fontsize=9,
                fontname='Segoe UI Emoji',
                bbox=dict(facecolor='white', alpha=0.4, edgecolor='none'))

    signal_color = 'green' if 'Bullish' in predictions['Signal'] else 'red' if 'Bearish' in predictions['Signal'] else 'gray'

    #_sigConf = f'{predictions.Signal}, {predictions.Risk}, Hit Prob: {predictions.Hits}, {int(predictions.Hit_Prob)}%'

    _sigConf = f'{predictions.Signal}, {predictions.Risk}, Hit Prob: {int(predictions.Hit_Prob)}%, Will Hit: {predictions.Will_Hit}'

    ax1.annotate(_sigConf,
                 xy=(0.7, 0.95), xycoords='axes fraction',
                 ha='right', va='top',
                 fontsize=12, weight='bold',
                 fontname='Segoe UI Emoji',
                 bbox=dict(boxstyle='round',
                          facecolor=signal_color,
                          alpha=0.2,
                          edgecolor=signal_color))

    # Add ticker name in the middle
    ax1.text(0.5, 0.5, f'@{ticker}', transform=ax1.transAxes, 
                 fontsize=50, color='grey', alpha=0.2,
                 horizontalalignment='center', verticalalignment='center',
                 rotation=0, weight='bold', style='italic')    
        
    # Move y-axis to right
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    ax1.set_ylabel('Price')
    ax1.set_title(
        f'{today}:\t{ticker} - {predictions["Signal"]}',
        fontdict={'fontname': 'Segoe UI Emoji', 'fontsize': 16},
        pad=20
    )
    ax1.legend(loc='upper left')
    
    # ===== 2. RSI PLOT =====
    #ax2.set_facecolor('white')
    rsi_ = df['RSI'].rolling(3).mean()
    rsi_sma = df['RSI'].rolling(20).mean()
    ax2.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax2.plot(df.index, rsi_, label='RSI', color='gray', linewidth=1.5, alpha=0.5)
    ax2.plot(df.index, rsi_sma, label='RSI SMA', color='gold', linewidth=1.2, alpha=0.7)

    # Fill RSI above 52 (green) and below 40 (red)
    ax2.fill_between(df.index, rsi_, 52,
                    where=(df['RSI'] > 52),
                    facecolor='green', alpha=0.1)
    ax2.fill_between(df.index, rsi_, 40,
                    where=(df['RSI'] < 40),
                    facecolor='red', alpha=0.1)

    rsi_last = round(df['RSI'].iloc[-1], 1)
    rsi_sma_last = round(df['RSI'].rolling(20).mean().iloc[-1], 1)
    price_vs_sma1 = 100 * (current_price - sma1_) / sma1_ if sma1_ != 0 else 0
    
    # Trend Strength Indicators    
    strong_Bull = (df['RSI'] > 52) & (df['ADX'] > 22) & (df['sumBuyVol'] > df['sumSellVol'])
    ax2.scatter(df.index[strong_Bull], rsi_[strong_Bull], color='lime', marker='^', s=5, label='Bullish', zorder=10)
    
    strong_Bear = (df['RSI'] < 40) & (df['ADX'] > 22) & (df['sumBuyVol'] < df['sumSellVol'])
    ax2.scatter(df.index[strong_Bear], rsi_[strong_Bear], color='red', marker='v', s=5, label='Bearish', zorder=10)

    
    ax2.axhline(70, color='green', linewidth=1, linestyle='--', alpha=0.2)
    ax2.axhline(30, color='red', linewidth=1, linestyle='--', alpha=0.2)
    ax2.axhline(50, color='gray', linewidth=1, linestyle='-', alpha=0.2)
    ax2.set_ylim(0, 100)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel('RSI')

    ax1.legend(loc='upper left', fontsize='small')
    ax2.legend(loc='upper left', fontsize='small')
    
    # Formatting
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate()

    strong_bull = (df['RSI'].iloc[-1] > 52) and \
                  (df['ADX'].iloc[-1] > 22) and \
                  (df['sumBuyVol'].iloc[-1] > df['sumSellVol'].iloc[-1])
    strong_bear = (df['RSI'].iloc[-1] < 40) and \
                  (df['ADX'].iloc[-1] > 22) and \
                  (df['sumBuyVol'].iloc[-1] < df['sumSellVol'].iloc[-1])
    
    summary_lines = [
        f"==== Market Technical Summary for {ticker} ====",
        f"Trend: SMA1 ({sma1_}) is {'above' if sma1_ > sma2_ else 'below'} SMA2 ({sma2_}) → Market is {'bullish' if sma1_ > sma2_ else 'bearish'}.",
        f"Momentum: RSI = {rsi_last} ({'above' if rsi_last > rsi_sma_last else 'below'} its 20-day average of {rsi_sma_last}).",
        f"Price: ${current_price} is {abs(price_vs_sma1):.2f}% {'above' if price_vs_sma1 > 0 else 'below'} SMA1.",
        f"\n"
        f"Trend Strength: Strong Bull: {'Yes' if strong_bull else 'No'}, Strong Bear: {'Yes' if strong_bear else 'No'}.",
        f"\n"
        f"Model Signal: {signal} | Expected Gain: +{gain}% (${gain_price:.2f}), Loss: {loss}% (${loss_price:.2f}) | Hit Probability: {round(hit_prob, 1)}%.",
        f"\n"
    ]
    action = ""
    # Actionable suggestion based on indicators and model confidence
    if signal == "TI: ✅ Bullish" and hit_prob > 50 and strong_bull and predicted_return > abs(predicted_loss):
        action = f"{ticker} is BULLISH: Consider buying or holding; good chance for positive return."
    elif signal == "TI: 🔻 Bearish" or hit_prob < 40 or strong_bear:
        action = f"{ticker} is BEARISH: Exercise caution or consider selling; risk of loss is higher."
    else:
        action = f"{ticker} is NEUTRAL; monitor market for clearer signals."
    
    summary_lines.append(action)

    textbox = AnchoredText(
       action,
       loc='lower right',
       frameon=True,
       borderpad=1.5,
       prop=dict(size=7, color='blue', weight='bold')
    )
    # Set the box properties through the patch
    textbox.patch.set(facecolor='white', edgecolor='gray', alpha=0.5, boxstyle='round')
    
    ax1.add_artist(textbox)
  
    # Save the figure to disk
    fname = f'{today}_{ticker}_TPSL.png'
    fpath = os.path.join(path, fname)
    plt.savefig(fpath, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
      
    print("\n".join(summary_lines))


# In[ ]:


# Make Predictions (Gain/Loss/Confidence)
n = 1
dfs = {}
for ticker in TICKERS:
    try:
        df = get_stock_data(ticker, start_date, end_date)
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            if "Date" in df.columns:
                df = df.set_index("Date")
            else:
                raise ValueError("DataFrame must have Date as index or column for plotting!")

        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df = add_technical_indicators(df)
        df = add_pivot_levels(df, window=14)
        df = add_pivots(df, windows)
        df = average_pivots(df, windows)
        df = compute_expected_return(df, forward_window=14, r_cols=['R1', 'R2'])
        df = compute_expected_loss(df, forward_window=14, s_cols=['S1', 'S2'])
        df = label_hit2(df, window=_DAYS, profit_target=PROFIT_TARGET, stop_loss=STOP_LOSS)
        dfs[ticker] = df
        
        df_model = df.dropna(subset=FEATURES + ['Hit_Label', 'Expected_Return', 'Expected_Loss'])

        if len(df_model) < _Nr:
            print(f"Skipping {ticker} due to insufficient data after dropna.")
            continue

        # --- Step 1: Train TP Hit Classifier ---
        X_cls = df_model[FEATURES]
        y_cls = df_model['Hit_Label'].astype(int)

        scaler_cls = StandardScaler()
        X_scaled_cls = scaler_cls.fit_transform(X_cls)

        X_train_cls, X_val_cls, y_train_cls, y_val_cls = train_test_split(
            X_scaled_cls, y_cls, test_size=0.2, random_state=42)

        model_class = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42)
        model_class.fit(X_train_cls, y_train_cls)

        # --- Step 2: Extract Full Class Probabilities as Features ---
        cls_probs = model_class.predict_proba(X_scaled_cls)
        expected_classes = [0, 1, 2]
        prob_df = pd.DataFrame(0, index=np.arange(len(cls_probs)), columns=[f'Prob_Class_{c}' for c in expected_classes])

        for i, c in enumerate(model_class.classes_):
            prob_df[f'Prob_Class_{c}'] = cls_probs[:, i]

        df_model = df_model.reset_index(drop=True)
        df_model = pd.concat([df_model, prob_df], axis=1)

        FEATURES_with_probs = FEATURES + [f'Prob_Class_{c}' for c in expected_classes]
        X_reg = df_model[FEATURES_with_probs]

        # --- Step 3: Train Return Model ---
        y_return = df_model['Expected_Return']
        scaler_return = StandardScaler()
        X_scaled_return = scaler_return.fit_transform(X_reg)
        X_train_ret, X_val_ret, y_train_ret, y_val_ret = train_test_split(
            X_scaled_return, y_return, test_size=0.2, random_state=42)

        model_return = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            max_features='sqrt', ccp_alpha=0.01)
        model_return.fit(X_train_ret, y_train_ret)

        # --- Step 4: Train Loss Model ---
        y_loss = df_model['Expected_Loss']
        scaler_loss = StandardScaler()
        X_scaled_loss = scaler_loss.fit_transform(X_reg)
        X_train_loss, X_val_loss, y_train_loss, y_val_loss = train_test_split(
            X_scaled_loss, y_loss, test_size=0.2, random_state=42)

        model_loss = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            max_features='sqrt', ccp_alpha=0.01)
        model_loss.fit(X_train_loss, y_train_loss)
        
        # --- Step 5: Live Prediction ---
        latest = df.iloc[[-1]]

        if latest[FEATURES].isnull().values.any():
            print(f"Skipping {ticker} for NULL Features")
            null_features = latest[FEATURES].iloc[0].isnull()
            print(f"NaN features for {ticker}: {list(null_features[null_features].index)}")
            continue
        
        label2str = {2: 'TP', 1: 'SL', 0: 'None'}

        # Predict class probabilities for latest sample
        latest_scaled_cls = scaler_cls.transform(latest[FEATURES])
        latest_probs_raw = model_class.predict_proba(latest_scaled_cls)[0]
        
        # Find index of highest probability class and corresponding class label
        pred_idx = latest_probs_raw.argmax()
        pred_class = model_class.classes_[pred_idx]  # 0,1,2
        will_hit = label2str[pred_class]
        
        # Hit probability for predicted event only
        hit_prob = latest_probs_raw[pred_idx]
        
        # Prepare feature DataFrame with probabilities for return/loss prediction
        # Note: We only keep the full prob vector here for return and loss scaling,
        # though storing all probabilities is removed from final results
        expected_classes = [0, 1, 2]
        latest_prob_features = {}
        for c in expected_classes:
            if c in model_class.classes_:
                latest_prob_features[f'Prob_Class_{c}'] = latest_probs_raw[model_class.classes_.tolist().index(c)]
            else:
                latest_prob_features[f'Prob_Class_{c}'] = 0.0
        latest_prob_df = pd.DataFrame([latest_prob_features])
        latest_features_with_probs = pd.concat([latest[FEATURES].reset_index(drop=True), latest_prob_df], axis=1)
        
        # Scale features for return and loss models
        latest_scaled_return = scaler_return.transform(latest_features_with_probs)
        latest_scaled_loss = scaler_loss.transform(latest_features_with_probs)
        
        # Predict expected return and loss
        predicted_return = model_return.predict(latest_scaled_return)[0]
        predicted_loss = model_loss.predict(latest_scaled_loss)[0]
        
        current_price = latest['Close'].values[0]
        predicted_tp = current_price * (1 + predicted_return)
        predicted_sl = current_price * (1 + predicted_loss)
        entry_price = (current_price + predicted_sl) / 2
        entry_discount_pct = ((current_price - entry_price) / entry_price) * 100
        
        confidence_score = hit_prob * max(predicted_return / abs(predicted_loss), 0)
        
        # Trading signal logic (unchanged)
        sma1 = latest['SMA1'].values[0]
        sma2 = latest['SMA2'].values[0]
        rsi = latest['RSI'].values[0]
        signal = "TI: ⚠️ Neut"
        entry_signal = False
        if (current_price >= sma1 and sma1 >= sma2 and rsi >= 52):
            signal = "TI: ✅ Bullish"
            if predicted_return > abs(predicted_loss) and hit_prob > 0.5:
                entry_signal = True
        elif (current_price <= sma1 and sma1 <= sma2 or rsi <= 42):
            signal = "TI: 🔻 Bearish"
            entry_signal = False
        
        # Color for printing rows
        sc = 'green' if (signal == "TI: ✅ Bullish" and will_hit == "TP" and hit_prob >= 0.4) else \
            'red' if (signal == "TI: 🔻 Bearish" and will_hit == "SL" and hit_prob>=0.4) else 'white'
        
        row_text = (
            f"{n:>3} "
            f"{bold}{ticker:>7} "
            f"Price: ${current_price:>7.2f} "
            f"TP: ${predicted_tp:>7.2f} ({predicted_return*100:>5.2f}%) "
            f"{will_hit:>4} (${predicted_sl:>7.2f}) "
            f"Prob: {int(hit_prob*100):>3}% "
            f"{signal:>7}{end}"
        )
        print(colored_row(row_text, sc))
        n += 1
        # Append results with only Will_Hit and its 
        if will_hit == 'TP':
            hit_price = round(predicted_tp, 2)
        elif will_hit == 'SL':
            hit_price = round(predicted_sl, 2)
        else:  # None
            hit_price = None

        if hit_price is not None:
            will_hit_str = f"{will_hit} (${hit_price})"
        else:
            will_hit_str = will_hit  # e.g. "None"

        results.append({
            "Ticker": ticker,
            "Date": latest.index[-1].date(),
            "Price": round(current_price, 1),
            "Entry": round(entry_price, 1),
            "Dip%": round(entry_discount_pct * -1, 1),
            "Max (%)": round(predicted_return * 100, 1),
            "TP": round(predicted_tp, 1),
            "SL": round(predicted_sl, 1),
            "Loss (%)": round(predicted_loss * 100, 1),
            "Signal": signal,
            "Risk": "🔴 High Risk" if (abs(predicted_loss) > STOP_LOSS) else "🟢 Low Risk",
            "Will_Hit": will_hit_str,
            "Hit_Prob": round(hit_prob * 100, 1),
            "Confidence": round(confidence_score * 100, 1),
        })

    except Exception as e:
        print(f"Error processing {ticker}: {e}")

df_results = pd.DataFrame(results)
append_pred(df_results, pred_file)


# In[ ]:


# Tabulate Data
#_df = pd.DataFrame(results).sort_values(by=["Confidence", "Will_Hit"], ascending=False)
def wrap_row_with_color(row, color_code):
    return [f"{color_code}{str(cell)}\033[0m" for cell in row]

colored_rows = []

_df = df_results.copy()
_df['Hits'] = _df['Will_Hit'].str.strip().str.split().str[0]
hit_order = ['TP', 'SL', 'None']
_df['Hits'] = pd.Categorical(_df['Hits'], categories=hit_order, ordered=True)

_df_sorted = _df.sort_values(by=['Hits',  "Signal", 'Confidence', "Hit_Prob"], ascending=[True, False, False, False]).reset_index(drop=True)
_df_sorted = _df_sorted.drop(columns=['Will_Hit'])
headers = _df_sorted.columns.tolist()


for _, row in _df_sorted.iterrows():
    signal = row.Signal
    hit_prob = row.Hit_Prob
    
    if ('Bullish' in row.Signal) and (row.Hit_Prob > 40) and (row['Max (%)'] > abs(row['Loss (%)'])):
        color = '\033[92m'  # Green
    elif ('Bearish' in row.Signal) and (row.Hit_Prob > 40) and (row['Max (%)'] > abs(row['Loss (%)'])):
        color = '\033[91m'  # Red
    else:
        color = '\033[38;5;251m'  # light gray
    colored_rows.append(wrap_row_with_color(row.values, color))


# Print colored table
print("\n=== Prediction Table (Signal, Hit Probability and Maximum returns) ===\n")
print(tabulate(colored_rows, headers=headers, floatfmt=".1f", tablefmt='orgtbl'))


# In[ ]:


# ✅ PLOT PREDICTIONS

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

df_plot = df_results
#df_plot = df_plot.sort_values(by="Max (%)", ascending=False)
max_vals = df_plot["Max (%)"].to_numpy()
norm = mcolors.Normalize(vmin=min(max_vals), vmax=max(max_vals))
cmap = cm.Spectral_r #Inverse of spectral
custom_colors = cmap(norm(max_vals))

fig, ax1 = plt.subplots(figsize=(12, 6), zorder=1, dpi=200)
cax = inset_axes(ax1, width="2%", height="60%", loc='center right',
                 bbox_to_anchor=(0.12, 0., 1, 1),
                 bbox_transform=ax1.transAxes,
                 borderpad=0)


# Main bar plot
ax1.bar(df_plot["Ticker"], max_vals, color=custom_colors, alpha = 0.7)
ax1.set_ylabel('Max Return (%)', fontsize=12)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

# Add colorbar at the right of the plot
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = plt.colorbar(sm, cax=cax, orientation='vertical', label="Colored by: Max (%)", alpha = 0.6)
cbar.ax.tick_params(labelsize=8)

# Secondary axis for loss line
ax2 = ax1.twinx()
sns.lineplot(x="Ticker", y="Loss (%)", data=df_plot, color='red', marker='o',
             ax=ax2, linewidth=2, markersize=8, label='Expected Loss')
ax2.set_ylabel('Expected Loss (%)', fontsize=12, color='red')

combined_min = min(ax1.get_ylim()[0], -ax2.get_ylim()[1])
combined_max = max(ax1.get_ylim()[1], -ax2.get_ylim()[0])
ax1.set_ylim(combined_min, combined_max)
ax2.set_ylim(-combined_max, -combined_min)
ax2.spines['right'].set_color('red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.invert_yaxis()
ax1.legend(fontsize='small')
ax2.legend(fontsize='small') 


# --- ANNOTATIONS ALIGNED BELOW X-TICK LABELS ---
x_ticks = ax1.get_xticks()
for i, (_, row) in enumerate(df_plot.iterrows()):
    # Color assignment for signal types
    fcolor = (
        'green' if row.Signal == "TI: ✅ Bullish"
        else 'red' if row.Signal == "TI: 🔻 Bearish"
        else 'yellow'
    )
    ProbColor = 'green' if (row.Signal == "TI: ✅ Bullish" and row.Confidence > 40 and str(row.Will_Hit).split()[0] == 'TP') else 'white'

    if row.Signal == "TI: ✅ Bullish" and row.Confidence > 50 and str(row.Will_Hit).split()[0] == 'TP':
        ProbColor = 'green'
    elif row.Signal == "TI: 🔻 Bearish" and row.Confidence > 50 and str(row.Will_Hit).split()[0] == 'SL':
        ProbColor = 'red'
    else:
        ProbColor = 'white'


    # Top annotations (unchanged)
    ax1.text(i, row["Max (%)"] + 0.5, f'{row["Max (%)"]:.1f}%',
             ha='center', va='bottom', fontsize=9)
    ax2.text(i, row["Loss (%)"] + 0.5, f'{row["Loss (%)"]:.1f}%',
             ha='center', va='top', color='red', fontsize=9)

    # Bottom annotations: align with x-tick, just below tick label
    x_tick = x_ticks[i]
    x_offset = -0.4 # to fix x-shift if colorbar is added, else put this to zero.
    y_offset1 = -0.275  # Adjust as needed for your plot
    y_offset2 = -0.575  # Stagger if two boxes per tick

    ax1.text(
        x_tick+x_offset, y_offset1,
        f'{row["Risk"]}\nP: ${row["Price"]:.2f}\nE: ${row["Entry"]:.2f}\nDip: {row["Dip%"]:.1f}%\n{row["Signal"]}',
        ha='left', va='top', fontsize=7, fontname='Segoe UI Emoji',
        bbox=dict(facecolor=fcolor, alpha=0.3, linewidth=0.3),
        transform=ax1.get_xaxis_transform(),
        multialignment='left',
        clip_on=False
    )

    ax1.text(
        x_tick + x_offset, y_offset2,
        f'TP: ${row["TP"]:.2f}\nSL: ${row["SL"]:.2f}\n\n{str(row.Will_Hit).split()[0]}: {row.Hit_Prob:.0f}%\nConf: {row.Confidence:.0f}%',
        ha='left', va='top', fontsize=7, fontname='Segoe UI Emoji',
        bbox=dict(facecolor=ProbColor, alpha=0.3, linewidth=0.3),
        transform=ax1.get_xaxis_transform(),
        clip_on=False
    )


# Strategic hint box
textbox = AnchoredText(
    "Hint: Buy closer to predicted SL to reduce risk\nand increase the chance of success.",
    loc='lower left',
    frameon=True,
    borderpad=1.5,
    prop=dict(size=10, color='gray', weight='bold')
)
ax1.add_artist(textbox)
textbox.set_clip_on(True)
textbox.set_in_layout(True)
textbox.set_zorder(100)
textbox.patch.set_facecolor('honeydew')
textbox.patch.set_edgecolor('darkgreen')
textbox.patch.set_alpha(0.8)

# Space management
plt.title(f'{today} - ML Predictions of {isStockCrypto} (From Current Price)', fontsize=16, color='black', pad=20)
plt.tight_layout()
plt.subplots_adjust(bottom=0.35)  # Increase if needed for annotation visibility

# Save and show
fname = f'{today}_ML_PNL_Multi{isStockCrypto}.png'
fpath = os.path.join(path, fname)
plt.savefig(fpath, bbox_inches='tight')
plt.show()


# In[ ]:


# PLOT STOCK TA with Predictions
for ticker in TICKERS:
    df = dfs.get(ticker)
    if df is None:
        print(f"Skipping {ticker}: no preloaded data available")
        continue
    plot_single_ticker(ticker, df, df_results)
    
del_old_files(path, 14)


# In[ ]:


# CREATE A PYTHON FILE BACKUP
get_ipython().system('jupyter nbconvert --to script FixedProfit_ML_MultiStocksV4.ipynb')


# In[ ]:




