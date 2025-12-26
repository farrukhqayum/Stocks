from imports import *
import candlesticks as cs
from time import sleep
from sklearn.linear_model import LinearRegression

w10 = 10
w20 = 20
w30 = 30
w40 = 40
w50 = 50
w100 = 100
w200 = 200

def get_stock_data(ticker, start_date, end_date, TF = '1d'):
    df = yf.download(ticker, start=start_date, end=end_date, interval = TF, auto_adjust=False, progress=False)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df

def get_next_earnings_date(ticker):
    stock = yf.Ticker(ticker)
    earnings = stock.calendar
    if isinstance(earnings, dict):
        try:
            earnings = pd.DataFrame.from_dict(earnings)
        except:
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

def get_fed_rates(start_date, end_date):
    rates = web.DataReader('FEDFUNDS', 'fred', start_date, end_date)
    return rates

def add_technical_indicators(df):
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values
    n = len(close)
    
    df['SMA1'] = pd.Series(np.convolve(close, np.ones(20)/20, mode='same'), index=df.index)
    df['SMA2'] = pd.Series(np.convolve(close, np.ones(50)/50, mode='same'), index=df.index)
    df['SMA3'] = pd.Series(np.convolve(close, np.ones(100)/100, mode='same'), index=df.index)
    
    df['EMA1'] = pd.Series(pd.Series(close).ewm(span=20, adjust=False).mean().values, index=df.index)
    df['EMA2'] = pd.Series(pd.Series(close).ewm(span=50, adjust=False).mean().values, index=df.index)
    df['EMA3'] = pd.Series(pd.Series(close).ewm(span=100, adjust=False).mean().values, index=df.index)
    
    df['RSI'] = calculate_rsi(df)
    df['OBV'] = calculate_obv(df)
    df['PVT'] = calculate_pvt(df)
    df['MFI'] = calculate_mfi(df)
    df['CCI'] = calculate_cci(df)
    df[['+DI', '-DI', 'ADX']] = calculate_dmi(df, n=14)
    
    df = calculate_stochrsi(df)
    df = calcBollingerBands(df)
    df['ATR'] = calculate_atr(df)
    
    df['Mom1'] = pd.Series(close - np.roll(close, 9), index=df.index)
    df['Mom2'] = pd.Series(close - np.roll(close, 20), index=df.index)
    
    df['ROC1'] = pd.Series((close / np.roll(close, 9) - 1) * 100, index=df.index)
    df['ROC2'] = pd.Series((close / np.roll(close, 20) - 1) * 100, index=df.index)
    
    close_shifted = np.roll(close, 1)
    buy_mask = close > close_shifted
    df['buy_volume'] = pd.Series(np.where(buy_mask, volume, 0), index=df.index)
    df['sell_volume'] = pd.Series(np.where(~buy_mask, volume, 0), index=df.index)
    
    df['sumBuyVol'] = pd.Series(np.convolve(df['buy_volume'].values, np.ones(20)/20, mode='same'), index=df.index)
    df['sumSellVol'] = pd.Series(np.convolve(df['sell_volume'].values, np.ones(20)/20, mode='same'), index=df.index)
    
    return df.dropna()

def calSMAs(close):
    sma1 = pd.Series(np.convolve(close.values, np.ones(20)/20, mode='same'), index=close.index)
    sma2 = pd.Series(np.convolve(close.values, np.ones(50)/50, mode='same'), index=close.index)
    sma3 = pd.Series(np.convolve(close.values, np.ones(100)/100, mode='same'), index=close.index)
    return sma1, sma2, sma3

def calEMAs(close):
    ema1 = pd.Series(pd.Series(close).ewm(span=20, adjust=False).mean().values, index=close.index)
    ema2 = pd.Series(pd.Series(close).ewm(span=50, adjust=False).mean().values, index=close.index)
    ema3 = pd.Series(pd.Series(close).ewm(span=100, adjust=False).mean().values, index=close.index)
    return ema1, ema2, ema3

def calculate_vwma(df, window=20):
    close_vol = df['Close'] * df['Volume']
    vwma = close_vol.rolling(window=window).sum() / df['Volume'].rolling(window=window).sum()
    return vwma

def compute_gapStrength(df):
    gap = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    strength = np.select([gap > 0.01, gap < -0.01], [1, -1], default=0)
    return pd.Series(strength, index=df.index)

def calculate_keltner(df, ema_window=20, atr_window=10, multiplier=2, outer_mult=4):
    middle = df['Close'].ewm(span=ema_window).mean()
    atr = calculate_atr(df)
    upper = middle + multiplier * atr
    lower = middle - multiplier * atr
    upper_outer = middle + outer_mult * atr
    lower_outer = middle - outer_mult * atr
    
    close_arr = df['Close'].values
    upper_arr = upper.values
    lower_arr = lower.values
    hits = np.zeros(len(close_arr))
    counter = 0
    
    for i in range(len(close_arr)):
        if close_arr[i] >= upper_arr[i]:
            counter += 1
        elif close_arr[i] <= lower_arr[i]:
            counter -= 1
        hits[i] = counter
    
    kasym = (df['Close'] - middle) / (upper - lower)
    
    return pd.DataFrame({
        'KCm': middle,
        'KCu': upper,
        'KCl': lower,
        'KCu_outer': upper_outer,
        'KCl_outer': lower_outer,
        'Kasym': kasym,
        'Kcount': hits
    }, index=df.index)

def calculate_vortex(df, window=20):
    high = df['High'].values
    low = df['Low'].values
    
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    atr = calculate_atr(df).values
    
    vi_plus = pd.Series(np.convolve(vm_plus, np.ones(window)/window, mode='same') / 
                       np.convolve(atr, np.ones(window)/window, mode='same'), index=df.index)
    vi_minus = pd.Series(np.convolve(vm_minus, np.ones(window)/window, mode='same') / 
                        np.convolve(atr, np.ones(window)/window, mode='same'), index=df.index)
    
    return pd.DataFrame({'VI+': vi_plus, 'VI-': vi_minus}, index=df.index)

def calculate_ichimoku(df):
    high = df['High'].values
    low = df['Low'].values
    
    tenkan = np.zeros_like(high)
    kijun = np.zeros_like(high)
    senkou_a = np.zeros_like(high)
    senkou_b = np.zeros_like(high)
    
    for i in range(len(high)):
        start_9 = max(0, i-8)
        start_26 = max(0, i-25)
        start_52 = max(0, i-51)
        
        tenkan[i] = (np.max(high[start_9:i+1]) + np.min(low[start_9:i+1])) / 2
        kijun[i] = (np.max(high[start_26:i+1]) + np.min(low[start_26:i+1])) / 2
        senkou_b[i] = (np.max(high[start_52:i+1]) + np.min(low[start_52:i+1])) / 2
    
    senkou_a = ((tenkan + kijun) / 2)
    
    df['Tenkan'] = tenkan
    df['Kijun'] = kijun
    df['Senkou_A'] = pd.Series(senkou_a, index=df.index).shift(26)
    df['Senkou_B'] = pd.Series(senkou_b, index=df.index).shift(26)
    
    return df[['Tenkan', 'Kijun', 'Senkou_A', 'Senkou_B']]

def calculate_supertrend(df, multiplier=3, window=10):
    atr = calculate_atr(df)
    df['Upper'] = (df['High'] + df['Low']) / 2 + multiplier * atr
    df['Lower'] = (df['High'] + df['Low']) / 2 - multiplier * atr
    return df[['Upper', 'Lower']]

def calcBollingerBands(df):
    close = df['Close'].values
    window = 20
    
    bb_m = pd.Series(np.convolve(close, np.ones(window)/window, mode='same'), index=df.index)
    
    std_arr = np.zeros_like(close)
    for i in range(len(close)):
        start = max(0, i - window + 1)
        std_arr[i] = np.std(close[start:i+1]) if i-start+1 > 1 else 0
    
    df['BBm'] = bb_m
    df['BBu'] = bb_m + 2 * pd.Series(std_arr, index=df.index)
    df['BBl'] = bb_m - 2 * pd.Series(std_arr, index=df.index)
    
    return df

def calculate_rsi(df, w=14):
    close = df['Close'].values
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    
    avg_gain[:w] = np.mean(gain[:w])
    avg_loss[:w] = np.mean(loss[:w])
    
    for i in range(w, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (w-1) + gain[i]) / w
        avg_loss[i] = (avg_loss[i-1] * (w-1) + loss[i]) / w
    
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    return pd.Series(rsi, index=df.index)

def calculate_stochrsi(df, rsi_period=14, stoch_period=20, d_period=9):
    rsi = df['RSI'].values
    lowest = np.zeros_like(rsi)
    highest = np.zeros_like(rsi)
    
    for i in range(len(rsi)):
        start = max(0, i - stoch_period + 1)
        lowest[i] = np.min(rsi[start:i+1])
        highest[i] = np.max(rsi[start:i+1])
    
    stoch_rsi = 100 * (rsi - lowest) / np.where(highest - lowest == 0, 1e-10, highest - lowest)
    stoch_rsi_d = pd.Series(stoch_rsi).rolling(window=d_period).mean().values
    
    df['StochRSI'] = pd.Series(stoch_rsi, index=df.index)
    df['StochRSI_D'] = pd.Series(stoch_rsi_d, index=df.index)
    
    return df

def calculate_atr(df):
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    
    hl = high - low
    hc = np.abs(high - np.roll(close, 1))
    lc = np.abs(low - np.roll(close, 1))
    
    hc[0] = 0
    lc[0] = 0
    
    tr = np.maximum(np.maximum(hl, hc), lc)
    atr = pd.Series(tr).rolling(window=14).mean().values
    
    return pd.Series(atr, index=df.index)

def scaled_volatility(df, window=9):
    hl = df['High'] - df['Low']
    oc = df['Open'] - df['Close']
    oc = oc.replace(0, np.nan)
    
    volatility = hl / oc
    volatility = volatility.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    up_day = df['Close'] > df['Open']
    down_day = df['Close'] < df['Open']
    unchanged_day = df['Close'] == df['Open']
    
    vol_up = df['Volume'].where(up_day, 0).rolling(window, min_periods=1).sum()
    vol_down = df['Volume'].where(down_day, 0).rolling(window, min_periods=1).sum()
    vol_unchanged = df['Volume'].where(unchanged_day, 0).rolling(window, min_periods=1).sum()
    
    numerator = vol_up * 2 + vol_unchanged
    denominator = vol_down * 2 + vol_unchanged
    denominator = denominator.replace(0, np.nan)
    
    vr = 100 * numerator / denominator
    vr = vr.replace([np.inf, -np.inf], np.nan).fillna(100)
    
    df['Scaled_Volatility'] = volatility * (vr / 100)
    df['Scaled_Volatility'] = df['Scaled_Volatility'].rolling(5, min_periods=1).mean().fillna(0)
    
    return df

def calculate_obv(df):
    close = df['Close'].values
    volume = df['Volume'].values
    
    obv = np.zeros_like(close)
    obv[0] = 0
    
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    
    return pd.Series(obv, index=df.index)

def calculate_pvt(df):
    close = df['Close'].values
    volume = df['Volume'].values
    
    pct_change = (close - np.roll(close, 1)) / np.roll(close, 1)
    pct_change[0] = 0
    
    pvt = pct_change * volume
    pvt_cumsum = np.cumsum(pvt)
    
    return pd.Series(pvt_cumsum, index=df.index)

def chaikin_money_flow(df, window=20):
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values
    
    mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
    mfm = np.nan_to_num(mfm)
    
    mfv = mfm * volume
    mfv_sum = pd.Series(mfv).rolling(window).sum()
    volume_sum = df['Volume'].rolling(window).sum()
    
    cmf = -mfv_sum / volume_sum
    return cmf

def calculate_mfi(df, period=20):
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    volume = df['Volume'].values
    
    tp = (high + low + close) / 3
    rmf = tp * volume
    tp_diff = np.diff(tp)
    tp_diff = np.insert(tp_diff, 0, 0)
    
    positive_mf = np.where(tp_diff > 0, rmf, 0)
    negative_mf = np.where(tp_diff < 0, rmf, 0)
    
    pos_sum = pd.Series(positive_mf).rolling(window=period).sum().values
    neg_sum = pd.Series(negative_mf).rolling(window=period).sum().values
    
    mfr = pos_sum / np.where(neg_sum == 0, 1e-10, neg_sum)
    mfi = 100 - (100 / (1 + mfr))
    mfi = np.where(neg_sum == 0, 100, mfi)
    
    return pd.Series(mfi, index=df.index)

def calculate_smiio(df, r=13, s=25, u=9):
    price = df['Close'].values
    m = price - np.roll(price, 1)
    m[0] = 0
    
    ema1 = pd.Series(m).ewm(span=r, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=s, adjust=False).mean().values
    
    abs_m = np.abs(m)
    abs_ema1 = pd.Series(abs_m).ewm(span=r, adjust=False).mean().values
    abs_ema2 = pd.Series(abs_ema1).ewm(span=s, adjust=False).mean().values
    
    smiio = 100 * (ema2 / np.where(abs_ema2 == 0, 1e-10, abs_ema2))
    signal = pd.Series(smiio).ewm(span=u, adjust=False).mean().values
    oscillator = smiio - signal
    
    return (pd.Series(smiio, index=df.index), 
            pd.Series(signal, index=df.index), 
            pd.Series(oscillator, index=df.index))

def calculate_cci(df, period=20):
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    
    tp = (high + low + close) / 3
    sma = pd.Series(tp).rolling(window=period).mean().values
    
    mean_dev = np.zeros_like(tp)
    for i in range(len(tp)):
        start = max(0, i - period + 1)
        mean_dev[i] = np.mean(np.abs(tp[start:i+1] - sma[i])) if i-start+1 > 0 else 0
    
    cci = (tp - sma) / (0.015 * np.where(mean_dev == 0, 1e-10, mean_dev))
    
    return pd.Series(cci, index=df.index)

def calculate_dmi(df, n=14):
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    
    hl = high - low
    hc = np.abs(high - np.roll(close, 1))
    lc = np.abs(low - np.roll(close, 1))
    hc[0] = 0
    lc[0] = 0
    
    tr = np.maximum(np.maximum(hl, hc), lc)
    
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       high - np.roll(high, 1), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)),
                        np.roll(low, 1) - low, 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr_smooth = pd.Series(tr).rolling(window=n).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=n).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=n).mean().values
    
    plus_di = 100 * (plus_dm_smooth / np.where(tr_smooth == 0, 1e-10, tr_smooth))
    minus_di = 100 * (minus_dm_smooth / np.where(tr_smooth == 0, 1e-10, tr_smooth))
    
    dx = 100 * (np.abs(plus_di - minus_di) / np.where(plus_di + minus_di == 0, 1e-10, plus_di + minus_di))
    adx = pd.Series(dx).rolling(window=n).mean().values
    
    result = pd.DataFrame({
        '+DI': plus_di,
        '-DI': minus_di,
        'ADX': adx
    }, index=df.index)
    
    return result

def add_exhaustion_indicator(df, lookback=90, threshold=0.10):
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    
    high_90 = np.zeros_like(close)
    low_90 = np.zeros_like(close)
    
    for i in range(len(close)):
        start = max(0, i - lookback + 1)
        high_90[i] = np.max(high[start:i+1])
        low_90[i] = np.min(low[start:i+1])
    
    dist_high = 1 - (high_90 - close) / (high_90 - low_90 + 1e-9)
    dist_low = 1 - (close - low_90) / (high_90 - low_90 + 1e-9)
    
    dist_high = np.clip(dist_high, 0, 1)
    dist_low = np.clip(dist_low, 0, 1)
    dist_low = -dist_low
    
    exhaustion = np.where(dist_high > np.abs(dist_low), dist_high, dist_low)
    df['Exhaustion'] = pd.Series(exhaustion, index=df.index)
    
    return df

def add_regression_forecast(ax, series, last_date, color):
    data = series.dropna()
    _DAYS = 14
    y = data.iloc[-_DAYS:].values if len(data) >= _DAYS else data.values
    x = np.arange(len(y)).reshape(-1,1)
    model = LinearRegression().fit(x, y)
    x_pred = np.arange(len(y), len(y)+_DAYS).reshape(-1,1)
    y_pred = model.predict(x_pred)
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=_DAYS)
    ax.plot(future_dates, y_pred, linestyle='dashdot', color=color, alpha=0.5)

def prepare_ml_data(df):
    features = ['SMA1', 'SMA2', 'SMA3', 'EMA1', 'EMA2', 'EMA3', 'RSI', 'RSI2', 
                'BBm', 'BBu', 'BBl', 'Mom1', 'Mom2', 'ROC1', 'ROC2', 
                'Candlesticks', 'Volume', 'ATR', 'MFI', 'CCI', '+DI', '-DI', 'ADX',
                'StochRSI', 'StochRSI_D', 'sumBuyVol', 'sumSellVol']
    
    df = df.dropna()
    X = df[features].values
    y = df['Close'].values
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_model(X_train, y_train, nest=1000, md=6):
    model = RandomForestRegressor(n_estimators=nest, random_state=42, max_depth=md, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def train_booster(X_train, y_train, nest=1000, lr=0.001, md=6, ss=0.8, ra=0.1, rl=1):
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=nest, 
                             learning_rate=lr, max_depth=md, subsample=ss,
                             reg_alpha=ra, reg_lambda=rl, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = round(np.sqrt(np.mean((y_test - y_pred) ** 2)), 3)
    r2 = round(1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2), 3)
    return rmse, r2

def generate_signal(predicted_prices, current_price, df):
    last_row = df.iloc[-1]
    SMA2 = last_row['SMA2']
    SMA3 = last_row['SMA3']
    EMA1 = last_row['EMA1']
    EMA2 = last_row['EMA2']
    rsi = last_row['RSI']
    BBl = last_row['BBl']
    BBu = last_row['BBu']
    
    SMA2_threshold = 0.02
    SMA3_threshold = 0.02
    EMA1_threshold = 0.02
    EMA2_threshold = 0.02
    rsi_threshold_buy = 35
    rsi_threshold_sell = 65
    bb_threshold = 0.02
    
    conditions = np.array([
        current_price > (1 + SMA2_threshold) * SMA2,
        current_price < (1 - SMA2_threshold) * SMA2,
        current_price > (1 + SMA3_threshold) * SMA3,
        current_price < (1 - SMA3_threshold) * SMA3,
        current_price > (1 + EMA1_threshold) * EMA1,
        current_price < (1 - EMA1_threshold) * EMA1,
        current_price > (1 + EMA2_threshold) * EMA2,
        current_price < (1 - EMA2_threshold) * EMA2,
        rsi < rsi_threshold_buy,
        rsi > rsi_threshold_sell,
        current_price < (1 - bb_threshold) * BBl,
        current_price > (1 + bb_threshold) * BBu
    ])
    
    buy_score = conditions[::2].sum()
    sell_score = conditions[1::2].sum()
    
    if buy_score > sell_score:
        return "BUY"
    elif sell_score > buy_score:
        return "SELL"
    else:
        return "HODL / SIDELINES"

def predict_prices(model, data, scaler, num_days=5, window_size=300):
    features = ['SMA1', 'SMA2', 'SMA3', 'EMA1', 'EMA2', 'EMA3', 'RSI', 'RSI2',
                'BBm', 'BBu', 'BBl', 'Mom1', 'Mom2', 'ROC1', 'ROC2', 
                'Candlesticks', 'Volume', 'ATR', 'MFI', 'CCI', '+DI', '-DI', 'ADX',
                'StochRSI', 'StochRSI_D', 'sumBuyVol', 'sumSellVol']
    
    last_data = data.copy()
    predicted_prices = []
    wt = 0.25
    
    for i in range(num_days):
        xdf = last_data.iloc[-window_size:].copy()
        
        close_arr = xdf['Close'].values
        xdf['SMA1'] = pd.Series(np.convolve(close_arr, np.ones(20)/20, mode='same'), index=xdf.index)
        xdf['SMA2'] = pd.Series(np.convolve(close_arr, np.ones(50)/50, mode='same'), index=xdf.index)
        xdf['SMA3'] = pd.Series(np.convolve(close_arr, np.ones(100)/100, mode='same'), index=xdf.index)
        
        xdf['EMA1'] = pd.Series(pd.Series(close_arr).ewm(span=20, adjust=False).mean().values, index=xdf.index)
        xdf['EMA2'] = pd.Series(pd.Series(close_arr).ewm(span=50, adjust=False).mean().values, index=xdf.index)
        xdf['EMA3'] = pd.Series(pd.Series(close_arr).ewm(span=100, adjust=False).mean().values, index=xdf.index)
        
        xdf['RSI'] = calculate_rsi(xdf)
        xdf['RSI2'] = xdf['RSI'].rolling(window=14).mean()
        xdf['MFI'] = calculate_mfi(xdf)
        xdf['CCI'] = calculate_cci(xdf)
        
        di_result = calculate_dmi(xdf, n=14)
        xdf['+DI'] = di_result['+DI']
        xdf['-DI'] = di_result['-DI']
        xdf['ADX'] = di_result['ADX']
        
        xdf = calcBollingerBands(xdf)
        xdf = calculate_stochrsi(xdf)
        
        xdf['Mom1'] = pd.Series(close_arr - np.roll(close_arr, 9), index=xdf.index)
        xdf['Mom2'] = pd.Series(close_arr - np.roll(close_arr, 20), index=xdf.index)
        
        xdf['ROC1'] = pd.Series((close_arr / np.roll(close_arr, 9) - 1) * 100, index=xdf.index)
        xdf['ROC2'] = pd.Series((close_arr / np.roll(close_arr, 20) - 1) * 100, index=xdf.index)
        
        inData = xdf[features].iloc[-1:].values
        inData_scaled = scaler.transform(inData)
        
        predicted_price = model.predict(inData_scaled)[0]
        last_actual = last_data['Close'].iloc[-1]
        wtPrice = wt * predicted_price + (1 - wt) * last_actual
        
        predicted_prices.append(wtPrice)
        
        next_index = pd.bdate_range(last_data.index[-1], periods=2)[-1]
        last_data.loc[next_index] = np.nan
        last_data.at[next_index, 'Close'] = wtPrice
        
        new_row = pd.DataFrame({'Close': [wtPrice]}, index=[next_index])
        last_data = pd.concat([last_data, new_row])
    
    return predicted_prices

def add_candlestickpatterns(df):
    df = df.copy()
    
    patterns_data = {
        'Bullish_Engulfing': cs.detect_bullish_engulfing(df),
        'Doji': cs.detect_doji(df),
        'Hammer': cs.detect_hammer(df),
        'Hanging_Man': cs.detect_hanging_man(df),
        'Morning_Star': cs.detect_morning_star(df),
        'Evening_Star': cs.detect_evening_star(df),
        'Shooting_Star': cs.detect_shooting_star(df),
        'Three_White_Soldiers': cs.detect_three_white_soldiers(df),
        'Three_Black_Crows': cs.detect_three_black_crows(df)
    }
    
    for name, pattern in patterns_data.items():
        df[name] = pattern
    
    pattern_weights = np.array([
        df['Doji'].values * 1,
        df['Hammer'].values * 2,
        df['Hanging_Man'].values * 3,
        df['Morning_Star'].values * 4,
        df['Evening_Star'].values * 5,
        df['Shooting_Star'].values * 6,
        df['Three_White_Soldiers'].values * 7,
        df['Three_Black_Crows'].values * 8,
        df['Bullish_Engulfing'].values * 9
    ])
    
    df['Candlesticks'] = pd.Series(pattern_weights.sum(axis=0), index=df.index)
    
    return df

def detect_divergences(df, period=14, max_bar_diff=3):
    price_lows = df['Low'].values
    price_highs = df['High'].values
    rsi = df['RSI'].values
    n = len(rsi)
    
    bullish_pairs = []
    bearish_pairs = []
    hidden_bullish_pairs = []
    hidden_bearish_pairs = []
    
    for i in range(period, n):
        start = i - period
        price_window = price_lows[start:i]
        rsi_window = rsi[start:i]
        
        price_min_idx = start + np.argmin(price_window)
        rsi_min_idx = start + np.argmin(rsi_window)
        
        if abs(price_min_idx - rsi_min_idx) <= max_bar_diff:
            if price_lows[price_min_idx] < price_lows[rsi_min_idx] and rsi[rsi_min_idx] > rsi[price_min_idx]:
                bullish_pairs.append((price_min_idx, rsi_min_idx))
            elif price_lows[price_min_idx] > price_lows[rsi_min_idx] and rsi[rsi_min_idx] < rsi[price_min_idx]:
                hidden_bullish_pairs.append((price_min_idx, rsi_min_idx))
    
    for i in range(period, n):
        start = i - period
        price_window = price_highs[start:i]
        rsi_window = rsi[start:i]
        
        price_max_idx = start + np.argmax(price_window)
        rsi_max_idx = start + np.argmax(rsi_window)
        
        if abs(price_max_idx - rsi_max_idx) <= max_bar_diff:
            if price_highs[price_max_idx] > price_highs[rsi_max_idx] and rsi[rsi_max_idx] < rsi[price_max_idx]:
                bearish_pairs.append((price_max_idx, rsi_max_idx))
            elif price_highs[price_max_idx] < price_highs[rsi_max_idx] and rsi[rsi_max_idx] > rsi[price_max_idx]:
                hidden_bearish_pairs.append((price_max_idx, rsi_max_idx))
    
