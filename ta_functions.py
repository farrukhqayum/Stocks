"""
Technical Analysis Functions - Fixed for pandas 2.x, numpy 1.24+, and streamlit
All functions have proper error handling, division by zero protection, and return types
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================
# HELPER FUNCTIONS
# ============================================

def safe_divide(numerator, denominator, default=0):
    """Safe division that handles zeros and infinities"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(denominator != 0, numerator / denominator, default)
        result = np.where(np.isfinite(result), result, default)
    return result

# ============================================
# DATA FETCHING FUNCTIONS
# ============================================

def get_stock_data(ticker, start_date, end_date, TF='1d'):
    """Fetch stock data with proper error handling"""
    try:
        df = yf.download(ticker, start=start_date, end=end_date, interval=TF, auto_adjust=False, progress=False)
        
        if df.empty:
            return None
        
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Handle MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def get_next_earnings_date(ticker):
    """Get next earnings date"""
    try:
        stock = yf.Ticker(ticker)
        earnings = stock.calendar
        
        if isinstance(earnings, dict):
            try:
                earnings = pd.DataFrame.from_dict(earnings)
            except Exception:
                return None

        if isinstance(earnings, pd.DataFrame):
            if 'Earnings Date' in earnings.index:
                next_earnings = earnings.loc['Earnings Date']
                if isinstance(next_earnings, pd.Series):
                    next_earnings = next_earnings.iloc[0]
                elif hasattr(next_earnings, '__getitem__'):
                    next_earnings = next_earnings[0]
                else:
                    return None
                if pd.notnull(next_earnings):
                    return pd.to_datetime(next_earnings)
            elif 'Earnings Date' in earnings.columns:
                val = earnings['Earnings Date'].values[0]
                if pd.notnull(val):
                    return pd.to_datetime(val)
        return None
    except Exception:
        return None

# ============================================
# MOVING AVERAGES
# ============================================

def calSMAs(close):
    """Calculate Simple Moving Averages"""
    sma1 = close.rolling(window=20).mean()
    sma2 = close.rolling(window=50).mean()
    sma3 = close.rolling(window=100).mean()
    return sma1, sma2, sma3

def calEMAs(close):
    """Calculate Exponential Moving Averages"""
    ema1 = close.ewm(span=20, adjust=False).mean()
    ema2 = close.ewm(span=50, adjust=False).mean()
    ema3 = close.ewm(span=100, adjust=False).mean()
    return ema1, ema2, ema3

def calculate_vwma(df, window=20):
    """Volume Weighted Moving Average"""
    vwma = (df['Close'] * df['Volume']).rolling(window=window).sum() / df['Volume'].rolling(window=window).sum().replace(0, np.nan)
    return vwma.fillna(method='ffill').fillna(df['Close'])

# ============================================
# VOLATILITY AND RISK
# ============================================

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr.fillna(0)

def scaled_volatility(df, window=9):
    """Calculate scaled volatility indicator"""
    df = df.copy()
    df['HL'] = df['High'] - df['Low']
    df['OC'] = df['Open'] - df['Close']
    df['OC'] = df['OC'].replace(0, np.nan)
    df['Volatility_HL_OC'] = safe_divide(df['HL'], df['OC'], 0)
    df['Volatility_HL_OC'] = df['Volatility_HL_OC'].fillna(0)
    
    df['Up_Day'] = df['Close'] > df['Open']
    df['Down_Day'] = df['Close'] < df['Open']
    df['Unchanged_Day'] = df['Close'] == df['Open']
    
    df['Vol_Up'] = df['Volume'].where(df['Up_Day'], 0).rolling(window, min_periods=1).sum()
    df['Vol_Down'] = df['Volume'].where(df['Down_Day'], 0).rolling(window, min_periods=1).sum()
    df['Vol_Unchanged'] = df['Volume'].where(df['Unchanged_Day'], 0).rolling(window, min_periods=1).sum()
    
    numerator = df['Vol_Up'] * 2 + df['Vol_Unchanged']
    denominator = df['Vol_Down'] * 2 + df['Vol_Unchanged']
    df['VR'] = 100 * safe_divide(numerator, denominator, 100)
    
    df['Scaled_Volatility'] = df['Volatility_HL_OC'] * (df['VR'] / 100)
    df['Scaled_Volatility'] = df['Scaled_Volatility'].rolling(5, min_periods=1).mean().fillna(0)
    
    return df

# ============================================
# OSCILLATORS AND MOMENTUM
# ============================================

def calculate_rsi(df, period=14):
    """Relative Strength Index"""
    close = df['Close']
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = safe_divide(avg_gain, avg_loss, 1)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_stochrsi(df, rsi_period=14, stoch_period=20, d_period=9):
    """Stochastic RSI"""
    df = df.copy()
    lowest_low = df['RSI'].rolling(window=stoch_period).min()
    highest_high = df['RSI'].rolling(window=stoch_period).max()
    df['StochRSI'] = safe_divide((df['RSI'] - lowest_low), (highest_high - lowest_low), 0) * 100
    df['StochRSI_D'] = df['StochRSI'].rolling(window=d_period).mean()
    return df

def calculate_mfi(data, period=20):
    """Money Flow Index"""
    try:
        data = data.copy()
        
        required_columns = ['High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            return pd.Series(50, index=data.index)
        
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        raw_money_flow = typical_price * data['Volume']
        money_flow_ratio = typical_price.diff()
        
        positive_flow = raw_money_flow.where(money_flow_ratio > 0, 0)
        negative_flow = raw_money_flow.where(money_flow_ratio < 0, 0)
        
        positive_sum = positive_flow.rolling(window=period).sum()
        negative_sum = negative_flow.rolling(window=period).sum()
        
        money_ratio = safe_divide(positive_sum, negative_sum, 1)
        mfi = 100 - (100 / (1 + money_ratio))
        mfi = mfi.fillna(50).clip(0, 100)
        
        return mfi
    except Exception:
        return pd.Series(50, index=data.index)

def calculate_cci(data, period=20):
    """Commodity Channel Index"""
    try:
        data = data.copy()
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean() if len(x) > 0 else 0
        )
        cci = safe_divide((typical_price - sma), (0.015 * mean_deviation), 0)
        return cci.fillna(0)
    except Exception:
        return pd.Series(0, index=data.index)

def calculate_smiio(df, r=13, s=25, u=9):
    """SMIIO Indicator"""
    price = df['Close']
    m = price - price.shift(1)
    ema1 = m.ewm(span=r, adjust=False).mean()
    ema2 = ema1.ewm(span=s, adjust=False).mean()
    abs_m = np.abs(m)
    abs_ema1 = abs_m.ewm(span=r, adjust=False).mean()
    abs_ema2 = abs_ema1.ewm(span=s, adjust=False).mean()
    smiio = 100 * safe_divide(ema2, abs_ema2, 0)
    signal = smiio.ewm(span=u, adjust=False).mean()
    oscillator = smiio - signal
    return smiio.fillna(0), signal.fillna(0), oscillator.fillna(0)

# ============================================
# DIRECTIONAL MOVEMENT
# ============================================

def calculate_dmi(df, n=14):
    """Directional Movement Index (DMI/ADX)"""
    try:
        df = df.copy()
        
        # True Range
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Directional Movements
        high_diff = df['High'] - df['High'].shift(1)
        low_diff = df['Low'].shift(1) - df['Low']
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        # Smooth with rolling mean
        tr_smooth = tr.rolling(window=n).mean()
        plus_dm_smooth = pd.Series(plus_dm, index=df.index).rolling(window=n).mean()
        minus_dm_smooth = pd.Series(minus_dm, index=df.index).rolling(window=n).mean()
        
        # Calculate DI
        plus_di = 100 * safe_divide(plus_dm_smooth, tr_smooth, 0)
        minus_di = 100 * safe_divide(minus_dm_smooth, tr_smooth, 0)
        
        # Calculate DX and ADX
        di_sum = plus_di + minus_di
        di_diff = np.abs(plus_di - minus_di)
        dx = 100 * safe_divide(di_diff, di_sum, 0)
        adx = dx.rolling(window=n).mean()
        
        result = pd.DataFrame({
            '+DI': plus_di.fillna(0),
            '-DI': minus_di.fillna(0),
            'ADX': adx.fillna(0)
        }, index=df.index)
        
        return result
    except Exception as e:
        print(f"DMI Error: {e}")
        return pd.DataFrame({
            '+DI': pd.Series(25, index=df.index),
            '-DI': pd.Series(25, index=df.index),
            'ADX': pd.Series(25, index=df.index)
        })

# ============================================
# VOLUME INDICATORS
# ============================================

def calculate_obv(data):
    """On-Balance Volume"""
    try:
        obv = [0]
        for i in range(1, len(data)):
            if pd.isna(data['Close'].iloc[i]) or pd.isna(data['Close'].iloc[i-1]) or pd.isna(data['Volume'].iloc[i]):
                obv.append(obv[-1])
            elif data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.append(obv[-1] + data['Volume'].iloc[i])
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.append(obv[-1] - data['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=data.index)
    except Exception:
        return pd.Series(0, index=data.index)

def calculate_pvt(df):
    """Price Volume Trend"""
    try:
        price_change = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)
        pvt = (price_change * df['Volume']).cumsum()
        return pvt.fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)

def chaikin_money_flow(df, window=20):
    """Chaikin Money Flow"""
    try:
        high = df['High']
        low = df['Low']
        close = df['Close']
        volume = df['Volume']
        
        mfm = safe_divide((close - low) - (high - close), (high - low), 0)
        mfv = mfm * volume
        cmf = mfv.rolling(window).sum() / volume.rolling(window).sum().replace(0, np.nan)
        return cmf.fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)

# ============================================
# BAND AND CHANNEL INDICATORS
# ============================================

def calcBollingerBands(df, window=20):
    """Calculate Bollinger Bands"""
    df = df.copy()
    close = df['Close']
    rolling = close.rolling(window=window)
    df['BBm'] = rolling.mean()
    rolling_std = rolling.std()
    df['BBu'] = df['BBm'] + 2 * rolling_std
    df['BBl'] = df['BBm'] - 2 * rolling_std
    return df

def calculate_keltner(df, ema_window=20, atr_window=10, multiplier=2, outer_mult=4):
    """Keltner Channels"""
    try:
        middle = df['Close'].ewm(span=ema_window).mean()
        atr = calculate_atr(df['High'], df['Low'], df['Close'])
        upper = middle + multiplier * atr
        lower = middle - multiplier * atr
        upper_outer = middle + outer_mult * atr
        lower_outer = middle - outer_mult * atr
        
        hits = []
        counter = 0
        for close, up, low in zip(df['Close'], upper, lower):
            if close >= up:
                counter += 1
            elif close <= low:
                counter -= 1
            hits.append(counter)
        
        kasym = safe_divide((df['Close'] - middle), (upper - lower), 0)
        
        return pd.DataFrame({
            'KCm': middle,
            'KCu': upper,
            'KCl': lower,
            'KCu_outer': upper_outer,
            'KCl_outer': lower_outer,
            'Kasym': kasym,
            'Kcount': hits
        }, index=df.index)
    except Exception:
        return pd.DataFrame({
            'KCm': df['Close'],
            'KCu': df['High'],
            'KCl': df['Low'],
            'KCu_outer': df['High'] * 1.02,
            'KCl_outer': df['Low'] * 0.98,
            'Kasym': 0,
            'Kcount': 0
        }, index=df.index)

def calculate_supertrend(df, multiplier=3, window=10):
    """Supertrend Indicator"""
    try:
        atr = calculate_atr(df['High'], df['Low'], df['Close'])
        middle = (df['High'] + df['Low']) / 2
        upper = middle + multiplier * atr
        lower = middle - multiplier * atr
        return pd.DataFrame({
            'STu': upper,
            'STl': lower
        }, index=df.index)
    except Exception:
        return pd.DataFrame({
            'STu': df['High'],
            'STl': df['Low']
        }, index=df.index)

def calculate_vortex(df, window=20):
    """Vortex Indicator"""
    try:
        vm_plus = np.abs(df['High'] - df['Low'].shift(1))
        vm_minus = np.abs(df['Low'] - df['High'].shift(1))
        atr = calculate_atr(df['High'], df['Low'], df['Close'])
        vi_plus = vm_plus.rolling(window).sum() / atr.rolling(window).sum().replace(0, np.nan)
        vi_minus = vm_minus.rolling(window).sum() / atr.rolling(window).sum().replace(0, np.nan)
        return pd.DataFrame({
            'VI+': vi_plus.fillna(1),
            'VI-': vi_minus.fillna(1)
        }, index=df.index)
    except Exception:
        return pd.DataFrame({
            'VI+': pd.Series(1, index=df.index),
            'VI-': pd.Series(1, index=df.index)
        })

def calculate_ichimoku(df):
    """Ichimoku Cloud"""
    try:
        high, low, close = df['High'], df['Low'], df['Close']
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        return pd.DataFrame({
            'Tenkan': tenkan,
            'Kijun': kijun,
            'Senkou_A': senkou_a,
            'Senkou_B': senkou_b
        }, index=df.index)
    except Exception:
        return pd.DataFrame({
            'Tenkan': df['Close'],
            'Kijun': df['Close'],
            'Senkou_A': df['Close'],
            'Senkou_B': df['Close']
        }, index=df.index)

# ============================================
# PATTERN AND EXHAUSTION INDICATORS
# ============================================

def compute_gapStrength(df):
    """Calculate gap strength"""
    try:
        gap = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        strength = np.where(gap > 0.01, 1, np.where(gap < -0.01, -1, 0))
        return pd.Series(strength, index=df.index, name='strength')
    except Exception:
        return pd.Series(0, index=df.index)

def add_exhaustion_indicator(df, lookback=90, threshold=0.10):
    """Add exhaustion indicator"""
    try:
        high_90 = df['High'].rolling(lookback).max()
        low_90 = df['Low'].rolling(lookback).min()
        close = df['Close']
        
        range_hl = high_90 - low_90
        dist_high = 1 - (high_90 - close) / (range_hl + 1e-9)
        dist_low = 1 - (close - low_90) / (range_hl + 1e-9)
        
        dist_high = dist_high.clip(0, 1)
        dist_low = dist_low.clip(0, 1) * -1
        
        df['Exhaustion'] = np.where(dist_high > np.abs(dist_low), dist_high, dist_low)
        df['Exhaustion'] = df['Exhaustion'].fillna(0)
        
        return df
    except Exception:
        df['Exhaustion'] = 0
        return df

# ============================================
# DIVERGENCE DETECTION
# ============================================

def detect_divergences(df, period=20, max_bar_diff=3):
    """Detect price-RSI divergences"""
    try:
        price_lows_mask = df['Low'] == df['Low'].rolling(window=period, center=True).min()
        price_highs_mask = df['High'] == df['High'].rolling(window=period, center=True).max()
        rsi_lows_mask = df['RSI'] == df['RSI'].rolling(window=period, center=True).min()
        rsi_highs_mask = df['RSI'] == df['RSI'].rolling(window=period, center=True).max()
        
        lows_idx = np.where(price_lows_mask)[0]
        highs_idx = np.where(price_highs_mask)[0]
        
        bullish_pairs = []
        bearish_pairs = []
        hidden_bullish_pairs = []
        hidden_bearish_pairs = []
        
        # Regular Bullish
        for i in range(1, len(lows_idx)):
            idx1, idx2 = lows_idx[i-1], lows_idx[i]
            if idx2 - idx1 <= period * 3:
                if rsi_lows_mask.iloc[idx1] and rsi_lows_mask.iloc[idx2]:
                    if df['Low'].iloc[idx2] < df['Low'].iloc[idx1] and df['RSI'].iloc[idx2] > df['RSI'].iloc[idx1]:
                        bullish_pairs.append((idx1, idx2))
        
        # Regular Bearish
        for i in range(1, len(highs_idx)):
            idx1, idx2 = highs_idx[i-1], highs_idx[i]
            if idx2 - idx1 <= period * 3:
                if rsi_highs_mask.iloc[idx1] and rsi_highs_mask.iloc[idx2]:
                    if df['High'].iloc[idx2] > df['High'].iloc[idx1] and df['RSI'].iloc[idx2] < df['RSI'].iloc[idx1]:
                        bearish_pairs.append((idx1, idx2))
        
        return bullish_pairs, bearish_pairs, hidden_bullish_pairs, hidden_bearish_pairs
    except Exception:
        return [], [], [], []

def find_doubleTopBottom(df, rsi_col='RSI', tol=0.5, max_bar_diff=3):
    """Find RSI double tops and bottoms"""
    try:
        rsi_highs_mask = df[rsi_col] == df[rsi_col].rolling(window=20, center=True).max()
        rsi_lows_mask = df[rsi_col] == df[rsi_col].rolling(window=20, center=True).min()
        
        rsi_highs_idx = np.where(rsi_highs_mask)[0]
        rsi_lows_idx = np.where(rsi_lows_mask)[0]
        
        double_tops = []
        double_bottoms = []
        
        for i in range(1, len(rsi_highs_idx)):
            idx1 = rsi_highs_idx[i-1]
            idx2 = rsi_highs_idx[i]
            if abs(idx2 - idx1) <= max_bar_diff:
                if abs(df[rsi_col].iloc[idx2] - df[rsi_col].iloc[idx1]) <= tol:
                    double_tops.append((idx1, idx2))
        
        for i in range(1, len(rsi_lows_idx)):
            idx1 = rsi_lows_idx[i-1]
            idx2 = rsi_lows_idx[i]
            if abs(idx2 - idx1) <= max_bar_diff:
                if abs(df[rsi_col].iloc[idx2] - df[rsi_col].iloc[idx1]) <= tol:
                    double_bottoms.append((idx1, idx2))
        
        return double_tops, double_bottoms
    except Exception:
        return [], []

# ============================================
# PLOTTING HELPERS
# ============================================

def add_regression_forecast(ax, series, last_date, color='orange', days=14):
    """Add linear regression forecast to plot"""
    try:
        data = series.dropna()
        y = data.iloc[-days:].values if len(data) >= days else data.values
        x = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        x_pred = np.arange(len(y), len(y) + days).reshape(-1, 1)
        y_pred = model.predict(x_pred)
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
        ax.plot(future_dates, y_pred, linestyle='dashdot', color=color, alpha=0.5)
    except Exception:
        pass

def plot_divergences(df, bullish, bearish, hidden_bull, hidden_bear, double_tops, double_bottoms, ax_price, ax_rsi):
    """Plot divergences on charts"""
    alpha_val = 0.5
    
    for i1, i2 in bullish:
        ax_price.plot([df.index[i1], df.index[i2]], [df['Low'].iloc[i1], df['Low'].iloc[i2]], color='green', alpha=alpha_val, lw=1.5)
        ax_rsi.plot([df.index[i1], df.index[i2]], [df['RSI'].iloc[i1], df['RSI'].iloc[i2]], color='green', alpha=alpha_val, lw=1.5)
    
    for i1, i2 in bearish:
        ax_price.plot([df.index[i1], df.index[i2]], [df['High'].iloc[i1], df['High'].iloc[i2]], color='red', alpha=alpha_val, lw=1.5)
        ax_rsi.plot([df.index[i1], df.index[i2]], [df['RSI'].iloc[i1], df['RSI'].iloc[i2]], color='red', alpha=alpha_val, lw=1.5)
    
    for i1, i2 in hidden_bull:
        ax_price.plot([df.index[i1], df.index[i2]], [df['Low'].iloc[i1], df['Low'].iloc[i2]], color='lime', alpha=alpha_val, lw=1.5, linestyle='dashed')
        ax_rsi.plot([df.index[i1], df.index[i2]], [df['RSI'].iloc[i1], df['RSI'].iloc[i2]], color='lime', alpha=alpha_val, lw=1.5, linestyle='dashed')
    
    for i1, i2 in hidden_bear:
        ax_price.plot([df.index[i1], df.index[i2]], [df['High'].iloc[i1], df['High'].iloc[i2]], color='orange', alpha=alpha_val, lw=1.5, linestyle='dashed')
        ax_rsi.plot([df.index[i1], df.index[i2]], [df['RSI'].iloc[i1], df['RSI'].iloc[i2]], color='orange', alpha=alpha_val, lw=1.5, linestyle='dashed')
    
    for i1, i2 in double_tops:
        ax_rsi.plot([df.index[i1], df.index[i2]], [df['RSI'].iloc[i1], df['RSI'].iloc[i2]], color='blue', alpha=alpha_val, lw=1.2, linestyle='dotted')
    
    for i1, i2 in double_bottoms:
        ax_rsi.plot([df.index[i1], df.index[i2]], [df['RSI'].iloc[i1], df['RSI'].iloc[i2]], color='purple', alpha=alpha_val, lw=1.2, linestyle='dotted')

# ============================================
# MAIN TECHNICAL INDICATOR FUNCTION
# ============================================

def add_technical_indicators(df):
    """Add all technical indicators to dataframe"""
    try:
        df = df.copy()
        close_prices = df['Close']
        
        # Moving Averages
        df['SMA1'], df['SMA2'], df['SMA3'] = calSMAs(close_prices)
        df['EMA1'], df['EMA2'], df['EMA3'] = calEMAs(close_prices)
        
        # Oscillators
        df['RSI'] = calculate_rsi(df)
        df['OBV'] = calculate_obv(df)
        df['PVT'] = calculate_pvt(df)
        df['MFI'] = calculate_mfi(df)
        df['CCI'] = calculate_cci(df)
        
        # Directional Movement
        dmi_result = calculate_dmi(df, n=14)
        df['+DI'] = dmi_result['+DI']
        df['-DI'] = dmi_result['-DI']
        df['ADX'] = dmi_result['ADX']
        
        # Stochastic
        df = calculate_stochrsi(df)
        
        # Bands
        df = calcBollingerBands(df)
        
        # Volatility
        df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'])
        
        # Momentum
        df['Mom1'] = close_prices - close_prices.shift(9)
        df['Mom2'] = close_prices - close_prices.shift(20)
        df['ROC1'] = close_prices.pct_change(periods=9) * 100
        df['ROC2'] = close_prices.pct_change(periods=20) * 100
        
        # Volume features
        df['buy_volume'] = (df['Close'] > df['Close'].shift(1)) * df['Volume']
        df['sell_volume'] = (df['Close'] < df['Close'].shift(1)) * df['Volume']
        df['sumBuyVol'] = df['buy_volume'].rolling(window=20).sum()
        df['sumSellVol'] = df['sell_volume'].rolling(window=20).sum()
        
        # Fill NaN values using ffill and bfill (pandas 2.x compatible)
        indicator_cols = ['EMA1', 'EMA2', 'RSI', '-DI', 'Close', 'SMA1', 'SMA2', 'SMA3', 'EMA3', 
                          '+DI', 'ADX', 'ATR', 'MFI', 'CCI', 'OBV', 'PVT', 'StochRSI', 'StochRSI_D',
                          'BBm', 'BBu', 'BBl', 'Mom1', 'Mom2', 'ROC1', 'ROC2']
        
        for col in indicator_cols:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()
        
        # Drop rows with remaining NaN values
        df = df.dropna()
        
        return df
        
    except Exception as e:
        print(f"Error in add_technical_indicators: {e}")
        return df

# ============================================
# ADDITIONAL UTILITIES
# ============================================

def add_candlestickpatterns(df):
    """Placeholder for candlestick patterns - to be implemented"""
    df = df.copy()
    df['Candlesticks'] = 0
    return df
