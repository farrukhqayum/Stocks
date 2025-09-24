from imports import *
import streamlit as st
import re
import warnings
warnings.filterwarnings("ignore")

disclaimer = """
---
**Disclaimer:**

- Trading involves substantial risk and may result in significant financial loss.
- Past performance is not indicative of future results.
- Always do your own research before making any investment or trading decisions.
- The information provided is for educational and informational purposes only.
- Trade at your own risk.
---
"""
power_of_compounding = """

Compounding is the process where the returns you earn are reinvested to generate their own returns. 
This effect causes your capital to grow exponentially over time, not just linearly.

Even small percentage gains consistently accumulated can turn modest initial capital into significant wealth.

Keep winning trades and staying disciplined to harness the power of compounding — patience and persistence are key to long-term trading success.

Remember, consistent small wins build up to large gains as profits generate more profits.
"""


# GLOBAL PARAMETERS
today = datetime.now().strftime('%Y-%m-%d') # For printing/filenames
path = 'ML_TP_SL_Figures' # CHECK THIS PATH / CREATE THE FOLDER
pdf_path = os.path.join(path, f'{today}_ML_TA_MultipleStocks.pdf')
pred_file = os.path.join(path, "tp_sl_daily.xlsx")
plt.rcParams['font.family'] = 'Segoe UI Emoji'

_Nr = 50 # Skip model if the length is this
YEARS_OF_DATA = 3
PROFIT_TARGET = 0.08
STOP_LOSS = 0.07
_DAYS = 22 # Used for SMA and training
_FWDAYS = 14 # Forward days to plot stored data
windows = [3, 5, 7, 9, 13, 15, 19, 29, 39, 49, 59] # For calculating returns
_window = 9  # Backtesting
tolerance = 1.07
_FIBS = False
_FibLen = 20 # Scan pivots for fibonacci levels
_ms = 5 # global marker size for matplotlib

bold = '\033[1m'
end = '\033[0m'

# Time window
end_date = datetime.now()
start_date = end_date - timedelta(days=365 * YEARS_OF_DATA)

# Shared model components

FEATURES = [
    # Technical Indicators
    'RSI', 'RSI_SMA', 'CCI', '+DI', '-DI', 'ADX', 'ATR', 'VI+', 'KCu', 'KCl', 'Kasym', 'Kcount', 'STu', 'STl',

    # Moving Averages & Bands
    'SMA1', 'SMA2', 'SMA3', 'SMA_Ratio', 'Upper_Band', 'Lower_Band', 'Volume_MA20', 'SMIIO', 'SMIIO_Signal', 'SMIIO_Osc', 'MACD', 'Signal_Line',

    # Returns & Volatility
    'return1', 'return2', 'return3', 'Volatility', 'Scaled_Volatility', 'DD',

    # Volume Features
    'sumBuyVol', 'sumSellVol', 'vSpike', 'VPT', 'OBV', 'MFI', 'VWMA', 'CMF',

    # Candlestick Patterns
    'Candlesticks', 'gapStrength',

    # Market Sentiment & Signals
    'Bear', 'Bull', 'Short', 'Hold', 'Neutral', 'StrongBull', 'StrongBear', 'Neutral', 'Exhaustion',

    # PIVOTS
    'PP_Avg', 'R1_Avg', 'R2_Avg', 'S1_Avg', 'S2_Avg'
]

results = []


# In[3]:


# Functions
@st.cache_data(ttl=600)
def get_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1), 
                     interval='1d', auto_adjust=False, progress=False)
    if df.empty:
        return None  # Explicitly return None if no data
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df.dropna()  # Assign back the dropped NA rows
    if df.empty:
        st.text(f"No data for {ticker}, skipping.")
        return None
    return df

def get_fundamentals(ticker: str, df=None):
    stock = yf.Ticker(ticker)
    info = stock.info
    
    atr_value = None
    if df is not None and 'ATR' in df.columns:
        atr_value = f"${df['ATR'].iloc[-1]:.2f}"

    fundamentals = {}
    if atr_value is not None:
        fundamentals['ATR'] = atr_value

    fundamentals.update({
            'Market Cap': info.get('marketCap', 'N/A'),
            'Net Profit Margin': info.get('netMargins', 'N/A'),
            'PE Ratio': info.get('trailingPE', 'N/A'),
            'Quick Ratio': info.get('quickRatio', 'N/A'),
            'Long Term Debt': info.get('longTermDebt', 'N/A'),
            'Free Cash Flow': info.get('freeCashflow', 'N/A')
        })
    
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
    
def strip_ansi_codes(text):
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)
    
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
    table.set_fontsize(6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell._loc = 'center'   # <- this line centers the header text
            cell.set_text_props(weight='bold')
            cell.set_facecolor(header_color)
        else:
            cell.set_edgecolor('none')

    return table

def add_technical_indicators(df):
    df['SMA1'] = df['Close'].ewm(span=int(_DAYS * 0.5), adjust=False).mean()
    df['SMA2'] = df['Close'].ewm(span=_DAYS, adjust=False).mean()
    df['SMA3'] = df['Close'].ewm(span=int(_DAYS * 2), adjust=False).mean()
    df['SMA_Ratio'] = df['SMA1'] / df['SMA2']
        
    df['ATR'] = ta.calculate_atr(high=df.High, low=df.Low, close=df.Close)
    df = ta.scaled_volatility(df)
    df = ta.add_candlestickpatterns(df)

    df['RSI']= ta.calculate_rsi(df)
    df['RSI_SMA'] = df['RSI'].rolling(14).mean()

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=24, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df['SMIIO'], df['SMIIO_Signal'], df['SMIIO_Osc'] = ta.calculate_smiio(df)

    df['Upper_Band'] = df['SMA1'] + (2 * df['Close'].rolling(20).std())
    df['Lower_Band'] = df['SMA1'] - (2 * df['Close'].rolling(20).std())
    
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    df['buy_volume'] = (df.Close > df.Close.shift(1)) * df['Volume']
    df['sell_volume'] = (df.Close < df.Close.shift(1)) * df['Volume']
    df['sumBuyVol'] = df['buy_volume'].rolling(window=9).sum()
    df['sumSellVol'] = df['sell_volume'].rolling(window=9).sum()
    df['vSpike'] = np.where(df['Volume'] > 2 * df['Volume_MA20'],
                        np.where(df['Close'] > df['Open'], 1, -1), 0)
    df['VPT'] = df['Volume'].mul((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)).cumsum()
    
    df['MFI'] = ta.calculate_mfi(df)
    df['CMF'] = ta.chaikin_money_flow(df, window=20)
    df['CCI'] = ta.calculate_cci(df)
    df['OBV'] = ta.calculate_obv(df)
    df[['+DI', '-DI', 'ADX']] = ta.calculate_dmi(df, n=14)

    
    df['VWMA'] = ta.calculate_vwma(df)
    df[['KCm', 'KCu', 'KCl', 'Kasym', 'Kcount']] = ta.calculate_keltner(df)
    df[['VI+', 'VI-']] = ta.calculate_vortex(df)
    df[['STu', 'STl']] = ta.calculate_supertrend(df)
    
    df['DD'] = df['Close'].where(df['Close'] < df['Close'].shift(1)).std()

    df['return1'] = df['Close'].pct_change(7)
    df['return2'] = df['Close'].pct_change(14)
    df['return3'] = df['Close'].pct_change(21)
    
    df['Volatility'] = df['Close'].rolling(14).std()
    df['Bull'] = ((df['SMA1'] > df['SMA2']) & (df['RSI'] > df['RSI_SMA']) & (df['RSI'] > 52)).astype(int)    
    df['Bear'] = ((df['SMA1'] < df['SMA2']) & (df['RSI'] < df['RSI_SMA']) & (df['RSI'] < 42)).astype(int)    
    df['Hold'] = (((df['Close'] >= df['SMA1']) & (df['RSI'] < df['RSI_SMA']) & (df['Bull'] == 0) & (df['Bear'] == 0))).astype(int)    
    df['Short'] = (((df['SMA1'] <= df['SMA2']) & df['RSI'].between(25, 42) & (df['Bear'] == 0))).astype(int)    
    df['Neutral'] = ((df['Bull'] == 0) & (df['Bear'] == 0) & (df['Hold'] == 0) & (df['Short'] == 0)).astype(int)

    strongbull_condition = ((df['RSI'] > 52) & (df['ADX'] > 22) & 
                           (df['+DI'] > df['-DI']) & (df['sumBuyVol'] > df['sumSellVol']))
    strongbear_condition = ((df['RSI'] < 40) & (df['ADX'] > 22) & 
                           (df['+DI'] < df['-DI']) & (df['sumBuyVol'] < df['sumSellVol']))
    
    df['StrongBull'] = strongbull_condition.astype(int)
    df['StrongBear'] = strongbear_condition.astype(int)
    df['sNeutral'] = ((df['StrongBull'] == 0) & (df['StrongBear'] == 0)).astype(int)

    df['gapStrength'] = ta.compute_gapStrength(df)
    df = ta.add_exhaustion_indicator(df)

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
    df['PP'] = PP.fillna(method='bfill')
    df['R1'] = R1.fillna(method='bfill')
    df['S1'] = S1.fillna(method='bfill')
    df['R2'] = R2.fillna(method='bfill')
    df['S2'] = S2.fillna(method='bfill')
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
    
    # Pre-extract pivot arrays
    pivot_arrays = []
    for col in r_cols:
        if col in df.columns:
            pivot_arrays.append(df[col].values)
        else:
            pivot_arrays.append(np.full(len(df), np.nan))
    
    for i in range(len(df) - forward_window):
        current_price = close_prices[i]
        
        # Gather valid pivot values for this row
        pivots = [arr[i] for arr in pivot_arrays if not np.isnan(arr[i])]
        target_level = max(pivots) if pivots else None
        
        future_window = close_prices[i+1:i+1+forward_window]
        
        if target_level is not None:
            # Check if future price hits the pivot level
            hit = False
            for future_price in future_window:
                if future_price >= target_level:
                    df.iloc[i, df.columns.get_loc('Expected_Return')] = (target_level - current_price) / current_price
                    hit = True
                    break
            if not hit and future_window.size > 0:
                df.iloc[i, df.columns.get_loc('Expected_Return')] = (np.nanmax(future_window) - current_price) / current_price
        else:
            if future_window.size > 0:
                df.iloc[i, df.columns.get_loc('Expected_Return')] = (np.nanmax(future_window) - current_price) / current_price
            else:
                df.iloc[i, df.columns.get_loc('Expected_Return')] = np.nan
    return df

def compute_expected_loss(df, forward_window=14, s_cols=['S1', 'S2']):
    df['Expected_Loss'] = np.nan
    close_prices = df['Close'].values
    
    pivot_arrays = []
    for col in s_cols:
        if col in df.columns:
            pivot_arrays.append(df[col].values)
        else:
            pivot_arrays.append(np.full(len(df), np.nan))
    
    for i in range(len(df) - forward_window):
        current_price = close_prices[i]
        
        pivots = [arr[i] for arr in pivot_arrays if not np.isnan(arr[i])]
        target_level = min(pivots) if pivots else None
        
        future_window = close_prices[i+1:i+1+forward_window]
        
        if target_level is not None:
            hit = False
            for future_price in future_window:
                if future_price <= target_level:
                    df.iloc[i, df.columns.get_loc('Expected_Loss')] = (target_level - current_price) / current_price
                    hit = True
                    break
            if not hit and future_window.size > 0:
                df.iloc[i, df.columns.get_loc('Expected_Loss')] = (np.nanmin(future_window) - current_price) / current_price
        else:
            if future_window.size > 0:
                df.iloc[i, df.columns.get_loc('Expected_Loss')] = (np.nanmin(future_window) - current_price) / current_price
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

def label_hit_prob_past(
    df,
    window=14,
    profit_target=0.08,
    stop_loss=0.08,
    lookback=60,
    tp_thresh=0.4,
    sl_thresh=0.4
):
    import numpy as np
    
    close_prices = df['Close'].values
    bull = df['Bull'].fillna(0).astype(int).values
    bear = df['Bear'].fillna(0).astype(int).values
    hold = df['Hold'].fillna(0).astype(int).values
    short = df['Short'].fillna(0).astype(int).values
    
    N = len(close_prices)
    labels = []
    
    for i in range(N):
        current_price = close_prices[i]
        tp = current_price * (1 + profit_target)
        sl = current_price * (1 - stop_loss)
        
        # Adjust window for tail of series
        future_prices = close_prices[i + 1:i + 1 + window] if i + 1 < N else np.array([])
        tp_hit_idx = next((j for j, price in enumerate(future_prices) if price >= tp), None)
        sl_hit_idx = next((j for j, price in enumerate(future_prices) if price <= sl), None)
        
        # Lookback for probability
        lookback_start = max(0, i - lookback)
        history_tp, history_sl = [], []
        for j in range(lookback_start, i):
            hist_price = close_prices[j]
            hist_tp = hist_price * (1 + profit_target)
            hist_sl = hist_price * (1 - stop_loss)
            hist_future = close_prices[j + 1: j + 1 + window]
            
            if bull[j]:
                hist_tp_hit_idx = next((k for k, p in enumerate(hist_future) if p >= hist_tp), None)
                hist_sl_hit_idx = next((k for k, p in enumerate(hist_future) if p <= hist_sl), None)
                hit = hist_tp_hit_idx is not None and (hist_sl_hit_idx is None or hist_tp_hit_idx < hist_sl_hit_idx)
                history_tp.append(int(hit))
                
            if bear[j]:
                hist_tp_hit_idx = next((k for k, p in enumerate(hist_future) if p >= hist_tp), None)
                hist_sl_hit_idx = next((k for k, p in enumerate(hist_future) if p <= hist_sl), None)
                hit = hist_sl_hit_idx is not None and (hist_tp_hit_idx is None or hist_sl_hit_idx < hist_tp_hit_idx)
                history_sl.append(int(hit))
        
        # Dynamic fallback for short history
        tp_prob = np.mean(history_tp) if len(history_tp) >= 3 else min(np.mean(history_tp) if history_tp else 0.5, tp_thresh)
        sl_prob = np.mean(history_sl) if len(history_sl) >= 3 else min(np.mean(history_sl) if history_sl else 0.5, sl_thresh)
        
        # Label assignment priority: TP > SL > Hold > Short > Neutral
        if tp_hit_idx is not None and (sl_hit_idx is None or tp_hit_idx < sl_hit_idx) and bull[i] and tp_prob >= tp_thresh:
            labels.append(2)  # TP (bull)
        elif sl_hit_idx is not None and (tp_hit_idx is None or sl_hit_idx < tp_hit_idx) and bear[i] and sl_prob >= sl_thresh:
            labels.append(1)  # SL (bear)
        elif hold[i]:
            labels.append(3)  # Hold
        elif short[i]:
            labels.append(4)  # Short
        else:
            # If recent days with incomplete future, fallback to Hold/Short if Bull/Bear active
            if i >= N - window:
                if bull[i]:
                    labels.append(2)
                elif bear[i]:
                    labels.append(1)
                else:
                    labels.append(0)
            else:
                labels.append(0)
    
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

def del_old_files(directory, days, exclude_extensions=None, dry_run=False):
    if not os.path.isdir(directory):
        print(f"Warning: directory {directory} does not exist, skipping deletion.")
        return
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
                st.text(f"[Dry Run] Would delete: {filepath}")
            else:
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"Error deleting {filepath}: {e}")
       
def append_pred(df, fpath):
    cols = ['Ticker', 'Date', 'Price', 'TP', 'SL', 'Will_Hit','Signal']
    new_data = df[cols].copy()
    new_data['Date'] = pd.to_datetime(new_data['Date'])
    
    if os.path.exists(fpath):
        old_data = pd.read_excel(fpath)
        old_data['Date'] = pd.to_datetime(old_data['Date'])
        mask = old_data.set_index(['Ticker', 'Date']).index.isin(new_data.set_index(['Ticker', 'Date']).index)
        old_data = old_data[~mask]

        combined = pd.concat([old_data, new_data], ignore_index=True)
    else:
        combined = new_data

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
        'yellow': '\033[93m',
        'white': '\033[97m',
        'darkred': '\033[38;5;52m'
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

def compound_growth(initial_capital, win_pct, num_wins, tax_rate):
    effective_gain = win_pct * (1 - tax_rate)
    final_capital = initial_capital * (1 + effective_gain) ** num_wins
    return final_capital


def safe_format_float(val, fmt="{:7.2f}", na_str="N/A"):
    try:
        return fmt.format(float(val))
    except (ValueError, TypeError):
        return na_str

# PRICE CHARTS
def plot_single_ticker(ticker, df, df_results, _window=14):
    # Get predictions
    predictions = df_results[df_results['Ticker'] == ticker].iloc[0]
    if predictions.empty:
        st.text(f"No prediction results found for ticker {ticker}, skipping plot.")
        return
    
    import re
    ## --- Technical Market Summary ---    
    signal = predictions.Signal
    current_price = round(df['Close'].iloc[-1], 2)
    gain = round(predictions['Max (%)'], 1)
    loss = round(predictions['Loss (%)'], 1)
    gain_price = current_price * (1 + gain/100)
    loss_price = current_price * (1 + loss/100)
    hit_prob = predictions.Hit_Prob
    conf = predictions.Confidence
    summary_lines = []
    action = "N/A"
    will_hit_str = df_results.loc[df_results['Ticker'] == ticker, 'Will_Hit'].values[0]
    prob_threshold = 40
    clean_label = re.sub(r'\(.*?\)|[\d\.]+', '', will_hit_str).strip()
    
    last_date = df.index[-1]
    future_date = last_date + pd.Timedelta(days=_window)
    avg_price = (current_price+loss_price)/2.
    sma1_ = round(df['SMA1'].iloc[-1], 2)
    sma2_ = round(df['SMA2'].iloc[-1], 2)

    # Create figure with white background
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(12, 6),
    dpi=600,
    sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
    )

    #fig.patch.set_facecolor('white')
    
    # Get true trailing 12 months of data (not calendar YTD)
    end_date = df.index[-1]
    start_date = end_date - pd.DateOffset(months=12)
    df = df.loc[start_date:end_date]
    
    # ===== 1. PRICE PLOT =====
    ax1.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)
    # Signal for color
    df['Signal'] = np.select(
    [df['Bull']==1, df['Bear']==1],
    ['Bull', 'Bear'],
    default='Neutral'
    )
    
    # Smooth the price (3-periods) to remove outliers, the last price may also not be visible
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
    kcount_absmax = df['Kcount'].abs().max()
    df['Kcount_sc'] = df['Kcount'] * (df['SMA1'] / kcount_absmax)

    ax1.plot(df.index, df['SMA1'], label=f'SMA{int(_DAYS*0.5)}', color='gold', alpha=0.7, linewidth=1.2)
    ax1.plot(df.index, df['SMA2'], label=f'SMA{int(_DAYS*2)}', color='red', alpha=0.7, linewidth=1.2, linestyle='--')

    ax1.plot(df.index, df['KCu'], color='blue', alpha=0.3, linestyle='--', linewidth=1)
    ax1.plot(df.index, df['KCl'], color='red', alpha=0.3, linestyle='--', linewidth=1)

    ax1_ = ax1.twinx()
    line_kcount, = ax1_.plot(df.index, df['Kcount_sc'], color='gray', alpha=0.15, linewidth=2, label='KC Cumm. touches', zorder=0)
    ax1_.set_yticks([])
    ax1_.set_ylabel('')

    for line in ax1.lines:
        line.set_zorder(3)
    
    ta.add_regression_forecast(ax1, df['SMA1'], last_date, color='orange')
    ta.add_regression_forecast(ax1, df['SMA2'], last_date, color='red')
    
    # Fill between SMAs - green when SMA1 > SMA2, red otherwise
    ax1.fill_between(df.index, df['SMA1'], df['SMA2'],
                     where=(df['SMA1'] > df['SMA2']),
                     facecolor='green', alpha=0.2, interpolate=True,
                     label='BUY-times')
    
    ax1.fill_between(df.index, df['SMA1'], df['SMA2'],
                    where=(df['SMA1'] <= df['SMA2']),
                    facecolor='red', alpha=0.2, interpolate=True,
                    label='Stay-away')

    # Add stock's fundamental info box
    fundamentals = get_fundamentals(ticker, df)
    add_fundamentals_table(ax1, fundamentals, loc='lower left', alpha=0.6)

    # Add the earning date to the bottom
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
        extended_range = pd.Timedelta(days=_DAYS)
        
        if (first_date <= earnings_date <= last_date) or (last_date < earnings_date <= last_date + extended_range):
    
            y_pos = data_ymin + 0.05 * (data_ymax - data_ymin)
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
    ax1.plot(future_date, gain_price, '^', markersize=_ms, color='green', alpha=0.5, label=f'TP: ${gain_price:.2f}, {gain}%')
    ax1.plot(future_date, loss_price, 'v', markersize=_ms, color='red', alpha=0.5, label=f'SL: ${loss_price:.2f}, {loss}%')
    ax1.plot(last_date, avg_price, 'o', markersize=_ms, color='orange', alpha=0.5, label=f'E: ${avg_price:.2f}')

    ax1.annotate(f'E: ${avg_price:.2f}', 
                xy=(last_date, avg_price),
                xytext=(10, 0),
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


    signal_color = (
    'green' if 'Bullish' in predictions['Signal'] else
    'red' if 'Bearish' in predictions['Signal'] else
    'yellow' if 'Exh' in predictions['Signal'] else
    'gray'
    )

    _sigConf = f'{predictions.Signal}, {predictions.Risk}, Will Hit: {predictions.Will_Hit} [{int(predictions.Hit_Prob)}%]'

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
        
    # Move y-axis to the right
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    ax1.set_ylabel('Price')
    ax1.set_title(
        f'{today}:\t{ticker} - {predictions["Signal"]}',
        fontdict={'fontname': 'Segoe UI Emoji', 'fontsize': 16},
        pad=20
    )
    ax1.scatter(df.index[df['StrongBull'] == 1], price[df['StrongBull'] == 1], color='lime', marker='^', s=5, alpha=0.4, label='StrongBull', zorder=10)
    ax1.scatter(df.index[df['StrongBear'] == 1], price[df['StrongBear'] == 1], color='red', marker='v', s=5, alpha=0.4, label='StrongBear', zorder=10)
    
    # ======================================   RSI  PLOT    ==================================================== #
    #ax2.set_facecolor('white')
    rsi_ = df['RSI'].rolling(3).mean()
    rsi_sma = df['RSI'].rolling(20).mean()
    ax2.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax2.plot(df.index, rsi_, label='RSI', color='gray', linewidth=1.5, alpha=0.5)
    ax2.plot(df.index, rsi_sma, label='RSI SMA', color='gold', linewidth=1.2, alpha=0.7)

    # Fill RSI above 52 (green) and below 40 (red)
    ax2.fill_between(df.index, rsi_, 52,
                    where=(df['RSI'] > 52),
                    facecolor='green', alpha=0.15)
    ax2.fill_between(df.index, rsi_, 40,
                    where=(df['RSI'] < 40),
                    facecolor='red', alpha=0.15)

    rsi_last = round(df['RSI'].iloc[-1], 1)
    rsi_sma_last = round(df['RSI'].rolling(20).mean().iloc[-1], 1)
    price_vs_sma1 = 100 * (current_price - sma1_) / sma1_ if sma1_ != 0 else 0
    
    # Trend Strength Indicators    
    ax2.scatter(df.index[df['Bull'] == 1], rsi_[df['Bull'] == 1], color='green', marker='^', s=5, alpha = 0.4, label = 'Bull', zorder=7)
    ax2.scatter(df.index[df['Bear'] == 1], rsi_[df['Bear'] == 1], color='red', marker='v', s=5, alpha = 0.4, label = 'Bear',   zorder=8)
    ax2.scatter(df.index[df['Short'] == 1], rsi_[df['Short'] == 1], color='red', marker='x', s=5, alpha = 0.4, label = 'Short',  zorder=10)
    ax2.scatter(df.index[df['Hold'] == 1], rsi_[df['Hold'] == 1], color='orange', marker='o', s=5, alpha = 0.4, label = 'Hold',  zorder=10)
    
    # Horizontal RSI Levels
    ax2.axhline(80, color='green', linewidth=1, linestyle='dotted', alpha=0.3)
    ax2.axhline(20, color='red', linewidth=1, linestyle='dotted', alpha=0.3)
    ax2.axhline(40, color='brown', linewidth=1, linestyle='dashed', alpha=0.3)
    ax2.axhline(52, color='gray', linewidth=1.2, linestyle='dashed', alpha=0.3)
    ax2.set_ylim(0, 100)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel('RSI')

    mid_date = df.index[len(df.index)//2]
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize='x-small')
    ax2.legend(loc='upper left', fontsize='x-small')

    # PLOT Divergences
    
    bull_div, bear_div, hbull_div, hbear_div = ta.detect_divergences(df, period=20)
    dtop, dbot = ta.find_doubleTopBottom(df, tol=0.5, max_bar_diff=5)
    ta.plot_divergences(df,
                        bull_div,
                        bear_div,
                        hbull_div,
                        hbear_div,
                        dtop,
                        dbot,
                        ax1,
                        ax2
                        )
    
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

    import re

    hit_interp = {
        'TP': "bullish — consider buying or holding",
        'SL': "bearish — exercise caution or consider selling",
        'Hold': "hold current position — no immediate action",
        'Short': "bearish short position — be cautious",
        'None': "neutral — monitor market for clearer signals",
    }

    if clean_label in hit_interp and conf >= prob_threshold:
        action = (
            f"{ticker} is {hit_interp[clean_label]} "
            f"with confidence {conf:.1f}%."
        )
    else:
        action = f"{ticker} is neutral; monitor for clearer signals."
    
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

    plt.tight_layout()
    st.pyplot(fig)
    st.text("\n".join(summary_lines))


# In[5]:


# Make Predictions (Gain/Loss/Confidence)
def MakePredictions(TICKERS = "AAPL, GOOGL, MSFT"):
    
    n = 1
    dfs = {}
    results = []
    label2str = {0: 'None', 1: 'SL', 2: 'TP', 3: 'Hold', 4: 'Short'}
    expected_classes = [0, 1, 2, 3, 4]
    
    for ticker in TICKERS:
        try:
            df = get_stock_data(ticker, start_date, end_date)
            if not pd.api.types.is_datetime64_any_dtype(df.index):
                if "Date" in df.columns:
                    df = df.set_index("Date")
                else:
                    raise ValueError("DataFrame must have Date as index or column for plotting!")
            df = add_technical_indicators(df)
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            df['BuyTime'] = (
                (df['Bull'] == 1) &
                ((df['Close'] - df['SMA1']) / df['SMA1'] <= 0.02)
            )
            df = add_pivot_levels(df, window=14)
            df = add_pivots(df, windows)
            df = average_pivots(df, windows)
            df = compute_expected_return(df, forward_window=14, r_cols=['R1_Avg', 'R2_Avg'])
            df = compute_expected_loss(df, forward_window=14, s_cols=['S1_Avg', 'S2_Avg'])
            df = label_hit_prob_past(df, window=30, profit_target=PROFIT_TARGET, stop_loss=STOP_LOSS, lookback=120, tp_thresh=0.35, sl_thresh=0.35)
            df['Hit_Label'] = df['Hit_Label'].fillna(0).astype(int)
            
            dfs[ticker] = df
            
            df_model = df.dropna(subset=FEATURES + ['Hit_Label', 'Expected_Return', 'Expected_Loss'])
            if len(df_model) < _Nr:
                st.text(f"Skipping {ticker} due to insufficient data after dropna.")
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
            
            # Extract probability columns for all expected classes safely
            prob_df = pd.DataFrame(0, index=np.arange(len(cls_probs)), columns=[f'Prob_Class_{c}' for c in expected_classes])
            for i, c in enumerate(model_class.classes_):
                if c in expected_classes:
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
                st.text(f"Skipping {ticker} for NULL Features")
                null_features = latest[FEATURES].iloc[0].isnull()
                st.text(f"NaN features for {ticker}: {list(null_features[null_features].index)}")
                continue
            
            latest_scaled_cls = scaler_cls.transform(latest[FEATURES])
            latest_probs_raw = model_class.predict_proba(latest_scaled_cls)[0]
    
            # Compute probabilities for all expected classes
            latest_prob_features = {}
            for c in expected_classes:
                if c in model_class.classes_:
                    idx = model_class.classes_.tolist().index(c)
                    latest_prob_features[f'Prob_Class_{c}'] = latest_probs_raw[idx]
                else:
                    latest_prob_features[f'Prob_Class_{c}'] = 0.0
                    
            # Predict class based on max probability among expected classes
            probs_of_interest = [latest_prob_features[f'Prob_Class_{c}'] for c in expected_classes]
            max_prob_index = probs_of_interest.index(max(probs_of_interest))
            pred_class = expected_classes[max_prob_index]
            
            will_hit = label2str.get(pred_class, "None")
            if pd.isna(will_hit):
                will_hit = "None"
                
            hit_prob = latest_prob_features[f'Prob_Class_{pred_class}']
            
            # Prepare latest features including probability features for regressors
            latest_prob_df = pd.DataFrame([latest_prob_features])
            latest_features_with_probs = pd.concat([latest[FEATURES].reset_index(drop=True), latest_prob_df], axis=1)
            latest_scaled_return = scaler_return.transform(latest_features_with_probs)
            latest_scaled_loss = scaler_loss.transform(latest_features_with_probs)
            
            current_price = latest['Close'].values[0]
            predicted_return = model_return.predict(latest_scaled_return)[0]
            predicted_loss = model_loss.predict(latest_scaled_loss)[0]
            predicted_tp = current_price * (1 + predicted_return)
            predicted_sl = current_price * (1 + predicted_loss)
            entry_price = (current_price + predicted_sl) / 2
            entry_discount_pct = ((current_price - entry_price) / entry_price) * 100

            # Safely calculate confidence score
            try:
                ratio = predicted_return / abs(predicted_loss) if predicted_loss != 0 else 0
                ratio = max(ratio, 0)
                confidence_score = max(hit_prob * ratio, 0)
            except Exception as e:
                confidence_score = 0 
    
            sma1 = latest['SMA1'].values[0]
            sma2 = latest['SMA2'].values[0]
            rsi = latest['RSI'].values[0]
            signal = "TI: ⚪ Neut"
            _90DHigh = df['Exhaustion'].values[0] > 0.05
            entry_signal = True
            sc = 'white'
            lookback_n = 5
            bull_mode = pd.Series(df.Bull.values[-lookback_n:]).mode().iloc[0]
            bear_mode = pd.Series(df.Bear.values[-lookback_n:]).mode().iloc[0]
            neutral_mode = pd.Series(df.Neutral.values[-lookback_n:]).mode().iloc[0]
            hit_price = None
            
            window_high = df['High'].rolling(window=90).max().iloc[-1]
            if (rsi > 78) or (current_price >= window_high *1.02):
                _90DHigh = True
    
            if will_hit == 'TP':
                hit_price = predicted_tp
                signal = "TI: ✅ Bullish"
                sc = 'green'
            elif will_hit == 'Hold':
                hit_price = predicted_tp
                signal = "TI: 🟡 Hold"
                sc = 'magenta'
            elif will_hit == 'SL':
                hit_price = predicted_sl
                signal = "TI: 🔻 Bearish"
                sc = 'red'
            elif will_hit == 'Short':
                hit_price = predicted_sl
                signal = "TI: 🔻 Bearish"
                sc = 'darkred'
            else:
                hit_price = None
                signal = "TI: ⚪ Neut"
                sc = 'white'
    
            def safe_format_float(val, fmt="{:7.2f}", na_str="N/A"):
                try:
                    return fmt.format(float(val))
                except (ValueError, TypeError):
                    return na_str
            
            tp_str = safe_format_float(predicted_tp)
            sl_str = safe_format_float(predicted_sl)
            atr_str = safe_format_float(df['ATR'].iloc[-1], fmt="{:5.1f}")
            
            if hit_price is not None and isinstance(hit_price, (int, float, np.floating)):
                hit_price_str = f"${hit_price:>5.2f}"
            else:
                hit_price_str = "None"
            
            row_text = (
                f"{ticker:<7} | "
                f"${current_price:>7.2f} | "
                f"TP: ${tp_str:>8}({predicted_return*100:5.2f}%) | "
                f"SL: ${sl_str:>8}({predicted_loss*100:5.2f}%) | "
                f"{will_hit:<5} | "
                f"{int(latest_prob_features[f'Prob_Class_{pred_class}']*100):>3}% | "
                f"{signal[3]:<2}{signal[4:]:<10} | "
                f"{_90DHigh}{end}"
            )
            
            st.code(strip_ansi_codes(row_text))
            
            # Append results with formatted Will_Hit string
            if will_hit is None or str(will_hit).lower() == "nan":
                will_hit = "None"
        
            if will_hit == 'TP':
                hit_price_rounded = round(predicted_tp, 2)
            elif will_hit == 'SL':
                hit_price_rounded = round(predicted_sl, 2)
            else:
                hit_price_rounded = None
            
            if hit_price_rounded is not None:
                will_hit_str = f"{will_hit} (${hit_price_rounded})"
            else:
                will_hit_str = will_hit
            
            results.append({
                "Index": n >4,
                "Ticker": ticker,
                "Date": latest.index[-1].date(),
                "Price": round(current_price, 1),
                "Entry": round(entry_price, 1),
                "Dip%": round(entry_discount_pct * -1, 1),
                "TP": round(predicted_tp, 1),
                "Max (%)": round(predicted_return * 100, 1),
                "SL": round(predicted_sl, 1),
                "Loss (%)": round(predicted_loss * 100, 1),
                "Risk": "🔴 High Risk" if (abs(predicted_loss) > STOP_LOSS) else "🟢 Low Risk",
                "Signal": signal,
                "Will_Hit": will_hit_str,
                "Hit_Prob": round(latest_prob_features[f'Prob_Class_{pred_class}'] * 100, 1),
                "Confidence": round(confidence_score * 100, 1),
                "_90DHigh": _90DHigh
            })
        except Exception as e:
            st.text(f"Error processing {ticker}: {e}")
    df_results = pd.DataFrame(results)
    append_pred(df_results, pred_file)
    return dfs, df_results


# In[6]:


###### Tabulate Data
def style_rows(row):
    signal = row.Signal
    hit_prob = row.Hit_Prob
    exhaustion = row.get("_90DHigh", False)

    if exhaustion:
        return ['background-color: rgba(255, 255, 0, 0.3)'] * len(row)  # Yellow semi-transparent
    elif 'Hold' in signal:
        return ['background-color: rgba(238, 130, 238, 0.3)'] * len(row)  # Violet semi-transparent
    elif ('Bullish' in signal) and (hit_prob > 40) and (row['Max (%)'] > abs(row['Loss (%)'])):
        return ['background-color: rgba(144, 238, 144, 0.3)'] * len(row)  # LightGreen semi-transparent
    elif (('Bearish' in signal) or ('Short' in signal)) and (hit_prob > 40):
        return ['background-color: rgba(240, 128, 128, 0.3)'] * len(row)  # LightCoral semi-transparent
    else:
        return ['color: lightgray'] * len(row)

def streamlit_display(df_results):
    _df = df_results.copy()
    _df['Signal'] = _df['Signal'].str.replace(r'^TI:\s*', '', regex=True)
    _df['Will_Hit'] = _df['Will_Hit'].str.replace(r'\([^)]*\)', '', regex=True)
    _df['Will_Hit'] = _df['Will_Hit'].str.replace(r'[^A-Za-z]+', '', regex=True)

    custom_order = ['TP', 'Hold', 'SL', 'Short', 'None']
    ord_map = {label: i for i, label in enumerate(custom_order)}
    _df['who'] = _df['Will_Hit'].map(lambda x: ord_map.get(x, len(custom_order)))

    _df_sorted = _df.sort_values(
        by=['who', '_90DHigh', 'Signal', 'Confidence', "Hit_Prob"],
        ascending=[True, True, False, False, False]
    ).reset_index(drop=True)
    _df_sorted = _df_sorted.drop(columns=['Index', 'who'], errors='ignore')

    styled_df = _df_sorted.style.apply(style_rows, axis=1).format({
        'Max (%)': '{:.1f}',
        'Loss (%)': '{:.1f}',
        'Confidence': '{:.1f}',
        'Hit_Prob': '{:.0f}'
    })

    st.dataframe(styled_df, height=600)

# Usage in app
# streamlit_display(df_results)


# ✅ PLOT PREDICTIONS
def PlotPredictions(df_results):
    
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    tickers = df_results['Ticker']
    tickers_list = tickers.tolist()
    
    df_plot = df_results
    #df_plot = df_plot.sort_values(by="Max (%)", ascending=False)
    max_vals = df_plot["Max (%)"].to_numpy()
    norm = mcolors.Normalize(vmin=min(max_vals), vmax=max(max_vals))
    cmap = cm.jet #Inverse of spectral
    custom_colors = cmap(norm(max_vals))
    
    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=600)
    cax = inset_axes(ax1, width="2%", height="60%", loc='center right',
                     bbox_to_anchor=(0.12, 0., 1, 1),
                     bbox_transform=ax1.transAxes,
                     borderpad=0)
    
    # Main bar plot
    ax1.bar(df_plot["Ticker"], max_vals, color=custom_colors, alpha = 0.4)
    ax1.set_ylabel('Max Return (%)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add colorbar at the right of the plot
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax, orientation='vertical', label="Colored by: Max (%)", alpha = 0.4)
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
    plt.title(f'{today} - ML Predictions of Tickers (From Current Price)', fontsize=16, color='black', pad=20)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35)  # Increase if needed for annotation visibility
    
    # Save and show
    fname = f'{today}_ML_PNL_Multi.png'
    fpath = os.path.join(path, fname)
    plt.savefig(fpath, bbox_inches='tight', dpi=300)
    st.pyplot(fig)


def run_app():
    desc = """  
    - Machine learning/training of technical indicators
    - Trade signals include (Signals, hit-probability, and hit direction)
    - Use tables to find the strong stocks, and use the chart to stay in bullish trend. 
        - SEE THE CHART FOR THE TICKER YOU ARE INTERESTED IN:
            - BUY-TIMES: Colored green to BTD-BUY THE DIP
            - SELL-TIMES: Colored red to SELL-THE-RISE
            - NEUTRAL: Hold if in the buy times, else stay side-lines, avoid revenge trading/FOMO.
            - STRONG BUYS: Dominate when RSI recovers from bearish zone and is above its SMA (RSI) in yellow and price is above averages.
            - STRONG SELLS: Dominate when RSI is below 42 and falls below.
    - USE DIVERGENCE: For market swings (lows, tops) if you plan to trade for 4-6 months hold
    """
    
    st.header("Positional/Swing Trading Guidance")
    st.markdown(desc)
    
    st.header("Just Keep Winning!!!")
    
    st.markdown(power_of_compounding)
    initial_capital = st.number_input("Initial Capital ($)", min_value=0.0, value=1000.0, step=100.0)
    win_pct = st.number_input("Avg. Win (%)", min_value=0.0, value=3.75, step=0.1) / 100.0
    tax_pct = st.number_input("Tax (%)", min_value=0.0, value=0.0, step=0.1) / 100.0
    num_wins = st.number_input("Number of Trade Wins", min_value=0, value=75, step=1)
    if st.button("Calculate Growth"):
        final_capital = compound_growth(initial_capital, win_pct, num_wins, tax_pct)
        st.write(f"After {num_wins} consecutive wins, your capital grows to: **${final_capital:,.2f}**")
        # Show growth over each trade
        capitals = [initial_capital * (1 + win_pct) ** i for i in range(num_wins + 1)]
        st.line_chart(capitals)

    st.markdown(disclaimer)
    st.header("Stocks Signal Forecasting via Machine Learning")
    st.markdown("""
    <style>
    .stTextInput input[aria-label="Enter comma-separated tickers (max 20):"] {
        background-color: #f0fff0 !important;   /* Light green background */
        color: #003300;                        /* Dark green text */
    }
    </style>
    """, unsafe_allow_html=True)
    tickers_input = st.text_input("Enter comma-separated tickers (max 20):")
    
    if tickers_input:
        TICKERS = [t.strip() for t in tickers_input.split(",") if t.strip()]
        if len(TICKERS) > 20:
            st.error("You can enter up to 20 tickers only. Please reduce your list.")
            return 
        row_text = (
            f'{"Ticker":<7} | '
            f'{"Price":>7} | '
            f'{"Take-profit (%)":>15} | '
            f'{"Stop-loss (%)":>15} | '
            f'{"Will Hit":>8} | '
            f'{"Probability (%)":<11} | '
            f'{"Signal (TI)":<8} | '
            f'{"Is High":<7}'
        )

        st.code(row_text)
        dfs, df_results = MakePredictions(TICKERS)
        PlotPredictions(df_results)
        streamlit_display(df_results)
        for ticker in TICKERS:
            _df = dfs.get(ticker)
            if _df is None:
                st.text(f"Skipping {ticker}: no preloaded data available")
                continue

            plot_single_ticker(ticker, _df, df_results)  
        #del_old_files(path, 14)
        
# Call this only in streamlit run mode
if __name__ == "__main__":
    run_app()

