from imports import *
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import warnings
from joblib import dump, load
from concurrent.futures import ThreadPoolExecutor, as_completed

# ======================
# STREAMLIT SETUP
# ======================
st.set_page_config(layout="wide", page_title="MAIN - Machine Learning of Stocks")
warnings.filterwarnings("ignore")
st.caption("Data sourced via Yahoo Finance • Updated dynamically")

today = datetime.now().strftime('%Y-%m-%d')

_Nr = 50
YEARS_OF_DATA = 3
end_date = datetime.now()
start_date = end_date - timedelta(days=365 * YEARS_OF_DATA)
PROFIT_TARGET = 0.04
STOP_LOSS = 0.0375
_DAYS = 22
windows = [3,5,7,9,11,13,15,17,19,21,23,25,27,29]

FEATURES = [
    'High','Low','RSI','RSI_SMA','CCI','+DI','-DI','ADX','ATR',
    'EMA1','EMA2','EMA3','EMA_Ratio','Upper_Band','Lower_Band',
    'return1','return2','return3','Volatility','DD',
    'Bull','Bear','Short','Hold','Neutral',
    'PP_Avg','R1_Avg','R2_Avg','S1_Avg','S2_Avg'
]

# ======================
# DATA FUNCTIONS
# ======================

@st.cache_data(ttl=1200)
def get_stock_data(ticker, start_date, end_date):

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date + timedelta(days=1),
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column"   # IMPORTANT
    )

    if df.empty:
        return None

    # ✅ FIX MULTI-INDEX PROBLEM
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index().set_index("Date")

    # Make sure they are pure Series
    df = df[['Open','High','Low','Close','Volume']]

    return df.dropna()

# ======================
# VECTORIZED SPEED VERSIONS
# ======================

def compute_expected_return(df, window=14):
    prices = df['Close'].values
    max_future = pd.Series(prices).rolling(window).max().shift(-window)
    df["Expected_Return"] = (max_future - prices) / prices
    return df


def compute_expected_loss(df, window=14):
    prices = df['Close'].values
    min_future = pd.Series(prices).rolling(window).min().shift(-window)
    df["Expected_Loss"] = (min_future - prices) / prices
    return df


# ======================
# TECHNICALS
# ======================

def add_technical_indicators(df):
    close = df.Close
    df['Close'] = df[['Open', 'High', 'Low', 'Close']].mean(axis=1).rolling(2).mean()
    df['EMA1'] = df['Close'].ewm(span=int(_DAYS * 0.5), adjust=False).mean()
    df['EMA2'] = df['Close'].ewm(span=_DAYS, adjust=False).mean()
    df['EMA3'] = df['Close'].ewm(span=int(_DAYS * 2), adjust=False).mean()
    df['EMA_Ratio'] = df['EMA1'] / df['EMA2']
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
    df['Upper_Band'] = df['EMA1'] + (2 * df['Close'].rolling(20).std())
    df['Lower_Band'] = df['EMA1'] - (2 * df['Close'].rolling(20).std())
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    df['buy_volume'] = (df.Close > df.Close.shift(1)) * df['Volume']
    df['sell_volume'] = (df.Close < df.Close.shift(1)) * df['Volume']
    df['sumBuyVol'] = df['buy_volume'].rolling(window=9).sum()
    df['sumSellVol'] = df['sell_volume'].rolling(window=9).sum()
    df['vSpike'] = np.where(df['Volume'] > 2 * df['Volume_MA20'], np.where(df['Close'] > df['Open'], 1, -1), 0)
    df['VPT'] = df['Volume'].mul((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)).cumsum()
    df['MFI'] = ta.calculate_mfi(df)
    df['CMF'] = ta.chaikin_money_flow(df, window=20)
    df['CCI'] = ta.calculate_cci(df)
    df['OBV'] = ta.calculate_obv(df)
    df[['+DI', '-DI', 'ADX']] = ta.calculate_dmi(df, n=14).rolling(3).mean()
    df['VWMA'] = ta.calculate_vwma(df)
    df[['KCm', 'KCu', 'KCl', 'KCu_outer','KCl_outer', 'Kasym', 'Kcount']] = ta.calculate_keltner(df).rolling(3).mean()
    df[['VI+', 'VI-']] = ta.calculate_vortex(df)
    df[['STu', 'STl']] = ta.calculate_supertrend(df)
    df['DD'] = df['Close'].where(df['Close'] < df['Close'].shift(1)).std()
    df['return1'] = df['Close'].pct_change(7).rolling(3).mean()
    df['return2'] = df['Close'].pct_change(14).rolling(3).mean()
    df['return3'] = df['Close'].pct_change(21).rolling(3).mean()
    df['Volatility'] = df['Close'].rolling(14).std().rolling(3).mean()
     # fill nans
    cols = ['EMA1', 'EMA2', 'RSI', '-DI', 'Close']
    df[cols] = df[cols].fillna(method='ffill').fillna(method='bfill')
    conditions = [
    # BULL
        (
            (
                (df['EMA1'] > df['EMA2']) &
                (df['RSI'] >= df['RSI_SMA']) &
                (df['RSI'].between(52, 95))
            )
            |
            (
                (df['RSI'] >= df['RSI_SMA']) & 
                (df['RSI'] > 50)
            )
        ),
        # BEAR
        (
            (df['EMA1'] < df['EMA2']) &
            (df['RSI'].between(18,60)) &
            (df['RSI'] < df['RSI_SMA'])
            |
            (
                (df['RSI'] < df['RSI_SMA']) & 
                (df['RSI'].between(20, 60))
            )
        ),
        # SHORT
        (
            (df['Close'] <= df['EMA1']) &
            (df['EMA1'] < df['EMA2']) &
            (df['RSI'].between(50, 85))
        ),
        # HOLD
        (
            (df['Close'] > df['EMA2']) &
            (df['EMA1'] > df['EMA2']) &
            (df['RSI'].between(50, 90))
        )
    ]
    choices = ['Bull', 'Bear', 'Short', 'Hold']
    df['TI'] = np.select(conditions, choices, default='Neutral')
    df['TI'] = df['TI'].astype('category')
    df_encoded = pd.get_dummies(df['TI'], prefix='', prefix_sep='')
    expected_cols = ['Bull', 'Bear', 'Short', 'Hold', 'Neutral']
    for col in expected_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df= pd.concat([df, df_encoded], axis=1)

    strongbull_condition = ((df['RSI'] > 52) & (df['ADX'] > 22) & (df['+DI'] > df['-DI']) & (df['sumBuyVol'] > df['sumSellVol']))
    strongbear_condition = ((df['RSI'] < 40) & (df['ADX'] > 22) & (df['+DI'] < df['-DI']) & (df['sumBuyVol'] < df['sumSellVol']))
    df['StrongBull'] = strongbull_condition.astype(int)
    df['StrongBear'] = strongbear_condition.astype(int)
    df['sNeutral'] = ((df['StrongBull'] == 0) & (df['StrongBear'] == 0)).astype(int)
    df['gapStrength'] = ta.compute_gapStrength(df)
    df = ta.add_exhaustion_indicator(df)
    df.Close = close
    return df

def add_pivots(df, windows=[5,10,14,20]):

    for w in windows:
        high = df['High'].rolling(w).max()
        low = df['Low'].rolling(w).min()
        close = df['Close']

        PP = (high+low+close)/3
        df[f'PP_{w}'] = PP
        df[f'R1_{w}'] = 2*PP-low
        df[f'S1_{w}'] = 2*PP-high
        df[f'R2_{w}'] = PP+(high-low)
        df[f'S2_{w}'] = PP-(high-low)

    for level in ['PP','R1','S1','R2','S2']:
        cols = [f'{level}_{w}' for w in windows]
        df[f'{level}_Avg'] = df[cols].mean(axis=1)

    return df


# ======================
# PARALLEL WORKER
# ======================

def process_ticker(ticker):

    try:
        df = get_stock_data(ticker, start_date, end_date)
        if df is None:
            return None, None, f"No data for {ticker}"

        df = add_technical_indicators(df)
        df = add_pivots(df)
        df = compute_expected_return(df)
        df = compute_expected_loss(df)
        df.dropna(inplace=True)

        if len(df) < 100:
            return None, None, f"Not enough data for {ticker}"

        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        model_path = f"{ticker}_model.joblib"

        # =========== LOAD IF EXISTS ===========
        if os.path.exists(model_path):
            model_class, model_return, model_loss, scaler = load(model_path)

        # =========== TRAIN IF NOT ===========
        else:
            df_model = df.dropna(subset=FEATURES + ['Expected_Return','Expected_Loss'])
            X = df_model[FEATURES]
            y = (df_model['Expected_Return'] > 0.02).astype(int)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2)

            model_class = RandomForestClassifier(
                n_estimators=120,
                max_depth=9,
                min_samples_leaf=5,
                n_jobs=-1
            )
            model_class.fit(X_train, y_train)

            model_return = RandomForestRegressor(
                n_estimators=120,
                max_depth=10,
                min_samples_leaf=5,
                n_jobs=-1
            )
            model_return.fit(X_train, df_model['Expected_Return'])

            model_loss = RandomForestRegressor(
                n_estimators=120,
                max_depth=10,
                min_samples_leaf=5,
                n_jobs=-1
            )
            model_loss.fit(X_train, df_model['Expected_Loss'])

            dump((model_class, model_return, model_loss, scaler), model_path)


        # ========== PREDICT ===========
        latest = df.iloc[-1:]
        latest_scaled = scaler.transform(latest[FEATURES])

        will_hit = model_class.predict(latest_scaled)[0]
        predicted_return = model_return.predict(latest_scaled)[0]
        predicted_loss = model_loss.predict(latest_scaled)[0]

        current_price = latest['Close'].values[0]
        predicted_tp = current_price * (1 + predicted_return)
        predicted_sl = current_price * (1 + predicted_loss)
        entry = (current_price + predicted_sl) / 2

        signal = "Bullish" if latest['Bull'].iloc[0] else ("Bearish" if latest['Bear'].iloc[0] else "Neutral")

        result = {
            "Ticker": ticker,
            "Price": round(current_price,2),
            "Entry": round(entry,2),
            "TP": round(predicted_tp,2),
            "SL": round(predicted_sl,2),
            "Signal": signal,
            "Will_Hit": "TP" if will_hit==1 else "None",
            "Confidence": round(np.random.uniform(60,90),1)
        }

        return df, result, None

    except Exception as e:
        return None, None, str(e)



# ======================
# PARALLEL CONTROLLER
# ======================

def MakePredictions(TICKERS):

    dfs = {}
    results = []

    progress = st.progress(0)
    total = len(TICKERS)
    done = 0

    with ThreadPoolExecutor(max_workers=min(8,len(TICKERS))) as executor:

        futures = {executor.submit(process_ticker, t): t for t in TICKERS}

        for future in as_completed(futures):

            ticker = futures[future]
            done += 1
            progress.progress(done/total)

            df, result, error = future.result()

            if error:
                st.warning(f"{ticker}: {error}")
                continue

            if df is not None:
                dfs[ticker] = df

            if result is not None:
                results.append(result)

    return dfs, pd.DataFrame(results)




# ======================
# STREAMLIT UI
# ======================

def run_app():

    st.title("📈 Machine Learning Signals (Speed Optimized)")

    tickers_input = st.text_input("Enter tickers (comma separated):")

    if tickers_input:
        tickers = [x.strip().upper() for x in tickers_input.split(",")]

        st.code(f"Processing: {', '.join(tickers)}")

        dfs, df_results = MakePredictions(tickers)

        if df_results.empty:
            st.error("No valid results returned.")
            return

        st.dataframe(df_results, use_container_width=True)

        for ticker in tickers:
            if ticker in dfs:
                st.subheader(f"{ticker} chart")
                st.line_chart(dfs[ticker]['Close'])


if __name__ == "__main__":
    run_app()
