#!/usr/bin/env python3
# coding: utf-8
"""
Entry Position Analyzer — Auto-refresh (60s), session_state-safe, fixed pivots, preserved plotting style.

Requirements:
- yfinance
- scikit-learn
- matplotlib
- pandas, numpy
- your 'imports' module that exposes `ta` helpers used earlier (recommended)
"""

from imports import *   # keep if you have your 'ta' helpers (calculate_rsi, calculate_atr, etc.)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ======= CONFIG =======
st.set_page_config(page_title="Advanced Analyzer", layout="wide")
AUTO_REFRESH_SECONDS = 60  # as requested

YEARS_OF_DATA = {'4H': 1, '1D': 2, '1W': 5}
MIN_TRAIN_ROWS = {'4H': 50, '1D': 30, '1W': 10}
_DAYS = 21
WINDOWS = [3,5,7,9,11,13,15,17,19,21]
EXPECTED_CLASSES = [0,1,2,3,4]
label2str = {0:'None',1:'SL',2:'TP',3:'Hold',4:'Short'}

FEATURES = [
    'High','Low','Close','Volume',
    'SMA1','SMA2','SMA3','SMA_Ratio',
    'RSI','RSI_SMA','CCI','+DI','-DI','ADX','ATR',
    'Upper_Band','Lower_Band','Volume_MA20','SMIIO','SMIIO_Signal','SMIIO_Osc','MACD','Signal_Line',
    'return1','return2','return3','Volatility','Scaled_Volatility','DD',
    'sumBuyVol','sumSellVol','vSpike','VPT','OBV','MFI','VWMA','CMF',
    'Bull','Bear','Short','Hold','Neutral','StrongBull','StrongBear','sNeutral','gapStrength',
    'KCu','KCl','KCu_outer','KCl_outer','Kasym','Kcount','STu','STl'
]

# ======= Helpers: yfinance + session caching =======
def fetch_latest_price(ticker):
    """Fast single-value fetch (used to update current price quickly)"""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period='1d', interval='1m', progress=False, auto_adjust=True)
        if hist is None or hist.empty:
            # fallback to daily last close
            hist2 = t.history(period='1d', progress=False, auto_adjust=True)
            return float(hist2['Close'].iloc[-1]) if not hist2.empty else None
        return float(hist['Close'].iloc[-1])
    except Exception:
        return None

def download_ohlc(ticker, start, end, interval):
    try:
        interval_map = {'4H':'4h','1D':'1d','1W':'1wk'}
        yf_interval = interval_map.get(interval, interval)
        df = yf.download(ticker, start=start, end=end, interval=yf_interval, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        df = df.reset_index()
        # Date normalization
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        elif 'Datetime' in df.columns:
            df['Date'] = pd.to_datetime(df['Datetime'])
            df.set_index('Date', inplace=True)
            df.drop(columns=['Datetime'], inplace=True, errors='ignore')
        # flatten MultiIndex columns if any
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        required = ['Open','High','Low','Close','Volume']
        if not all(c in df.columns for c in required):
            return None
        df = df[required].dropna()
        if df.empty:
            return None
        return df
    except Exception:
        return None

# ======= Technical indicator wrapper (assumes your ta module available) =======
def add_technical_indicators(df, timeframe='1D'):
    try:
        close_backup = df['Close'].copy()
        df['Close'] = df[['Open','High','Low','Close']].mean(axis=1).rolling(3, min_periods=1).mean()

        if timeframe == '1W':
            sma_multiplier = 1
        elif timeframe == '4H':
            sma_multiplier = 3
        else:
            sma_multiplier = 3

        df['SMA1'] = df['Close'].ewm(span=int(_DAYS*0.5*sma_multiplier), adjust=False).mean()
        df['SMA2'] = df['Close'].ewm(span=_DAYS*sma_multiplier, adjust=False).mean()
        df['SMA3'] = df['Close'].ewm(span=int(_DAYS*2*sma_multiplier), adjust=False).mean()
        df['SMA_Ratio'] = df['SMA1'] / df['SMA2']

        # call functions from your ta helper object (adjust names if your ta differs)
        df['ATR'] = ta.calculate_atr(high=df.High, low=df.Low, close=df.Close)
        df = ta.scaled_volatility(df)
        df = ta.add_candlestickpatterns(df)
        df['RSI'] = ta.calculate_rsi(df)
        df['RSI_SMA'] = df['RSI'].rolling(14, min_periods=1).mean()

        ema_short = 9 if timeframe == '1W' else 12
        ema_long = 22 if timeframe == '1W' else 26
        ema_s = df['Close'].ewm(span=ema_short, adjust=False).mean()
        ema_l = df['Close'].ewm(span=ema_long, adjust=False).mean()
        df['MACD'] = ema_s - ema_l
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        df['SMIIO'], df['SMIIO_Signal'], df['SMIIO_Osc'] = ta.calculate_smiio(df)
        df['Upper_Band'] = df['SMA1'] + (2 * df['Close'].rolling(20, min_periods=1).std())
        df['Lower_Band'] = df['SMA1'] - (2 * df['Close'].rolling(20, min_periods=1).std())
        df['Volume_MA20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        df['buy_volume'] = (df.Close > df.Close.shift(1)) * df['Volume']
        df['sell_volume'] = (df.Close < df.Close.shift(1)) * df['Volume']
        df['sumBuyVol'] = df['buy_volume'].rolling(window=9, min_periods=1).sum()
        df['sumSellVol'] = df['sell_volume'].rolling(window=9, min_periods=1).sum()
        df['vSpike'] = np.where(df['Volume'] > 2 * df['Volume_MA20'], np.where(df['Close'] > df['Open'], 1, -1), 0)
        df['VPT'] = df['Volume'].mul((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)).cumsum()
        df['MFI'] = ta.calculate_mfi(df)
        df['CMF'] = ta.chaikin_money_flow(df, window=20)
        df['CCI'] = ta.calculate_cci(df)
        df['OBV'] = ta.calculate_obv(df)
        dmi = ta.calculate_dmi(df, n=14)
        if isinstance(dmi, pd.DataFrame):
            df['+DI'] = dmi['+DI']; df['-DI'] = dmi['-DI']; df['ADX'] = dmi['ADX']
        df['VWMA'] = ta.calculate_vwma(df)
        kelt = ta.calculate_keltner(df)
        if isinstance(kelt, pd.DataFrame):
            for c in ['KCu','KCl','Kasym','Kcount','KCu_outer','KCl_outer']:
                if c in kelt.columns:
                    df[c] = kelt[c]
        vortex = ta.calculate_vortex(df)
        if isinstance(vortex, pd.DataFrame) and 'VI+' in vortex.columns:
            df['VI+'] = vortex['VI+']; df['VI-'] = vortex['VI-']
        stf = ta.calculate_supertrend(df)
        if isinstance(stf, pd.DataFrame) and 'STu' in stf.columns:
            df['STu'] = stf['STu']; df['STl'] = stf['STl']

        df['Volatility'] = df['Close'].rolling(14, min_periods=1).std().rolling(3, min_periods=1).mean()
        df[['SMA1','SMA2','RSI','-DI','Close']] = df[['SMA1','SMA2','RSI','-DI','Close']].fillna(method='ffill').fillna(method='bfill')

        # TI classification (keeps your original logic)
        rsi_lower = 25 if timeframe == '1W' else 18
        conditions = [
            ((df['SMA1'] > df['SMA2']) & (df['RSI'] >= df['RSI_SMA']) & (df['RSI'].between(52,95)) & (df['+DI'] > df['-DI']) & (df['+DI'].between(18,55)) & (df['Close'] > df['SMA1'])),
            ((df['SMA1'] < df['SMA2']) & (df['RSI'].between(rsi_lower,60)) & (df['RSI'] < df['RSI_SMA']) & (df['+DI'] < df['-DI']) & (df['-DI'].between(18,55))),
            ((df['SMA1'] < df['SMA2']) & (df['RSI'].between(25,50)) & (df['-DI'].between(30,55)) & (df['Close'] > df['SMA1'])),
            (((df['SMA1'] > df['SMA2']) & (df['RSI'] >= 50)) | ((df['RSI'] < df['RSI_SMA']) & (df['ADX'].between(40,75))))
        ]
        choices = ['Bull','Bear','Short','Hold']
        df['TI'] = np.select(conditions, choices, default='Neutral')
        df['TI'] = df['TI'].astype('category')
        df = pd.concat([df, pd.get_dummies(df['TI'])], axis=1)

        df['StrongBull'] = (((df['RSI']>52) & (df['ADX']>22) & (df['+DI']>df['-DI']) & (df['sumBuyVol']>df['sumSellVol']))).astype(int)
        df['StrongBear'] = (((df['RSI']<40) & (df['ADX']>22) & (df['+DI']<df['-DI']) & (df['sumBuyVol']<df['sumSellVol']))).astype(int)
        df['sNeutral'] = ((df['StrongBull']==0) & (df['StrongBear']==0)).astype(int)
        df['gapStrength'] = ta.compute_gapStrength(df)
        df = ta.add_exhaustion_indicator(df)
        df['Close'] = close_backup
        return df
    except Exception:
        return None

# ======= Safe pivot functions (fixed KeyError) =======
def add_pivots(df, win=WINDOWS):
    for w in win:
        roll_high = df['High'].rolling(window=w)
        roll_low = df['Low'].rolling(window=w)
        roll_close = df['Close'].rolling(window=w)
        # safe apply using positional access
        PP = (roll_high.max() + roll_low.min() + roll_close.apply(lambda x: x.values[-1] if len(x) > 0 else np.nan)).div(3)
        R1 = 2 * PP - roll_low.min()
        S1 = 2 * PP - roll_high.max()
        R2 = PP + (roll_high.max() - roll_low.min())
        S2 = PP - (roll_high.max() - roll_low.min())
        df[f'PP_{w}'] = PP
        df[f'R1_{w}'] = R1
        df[f'S1_{w}'] = S1
        df[f'R2_{w}'] = R2
        df[f'S2_{w}'] = S2
    return df

def average_pivots(df, windows=[5,10,14,20]):
    for level in ['PP','R1','S1','R2','S2']:
        cols = [f'{level}_{w}' for w in windows if f'{level}_{w}' in df.columns]
        if cols:
            df[f'{level}_Avg'] = df[cols].mean(axis=1)
    return df

# ======= Expected return/loss and labeling (unchanged) =======
def compute_expected_return(df, forward_window=14, r_cols=['R1_Avg','R2_Avg']):
    df['Expected_Return'] = np.nan
    close = df['Close'].values
    for i in range(len(df) - forward_window):
        cur = close[i]
        future = close[i+1:i+1+forward_window]
        pivots = []
        for c in r_cols:
            if c in df.columns and not np.isnan(df[c].iloc[i]):
                pivots.append(df[c].iloc[i])
        target = max(pivots) if pivots else np.nan
        if not np.isnan(target):
            hit = False
            for p in future:
                if p >= target:
                    df.at[df.index[i],'Expected_Return'] = (target - cur)/cur
                    hit = True; break
            if not hit and future.size > 0:
                df.at[df.index[i],'Expected_Return'] = (np.nanmax(future) - cur)/cur
        else:
            if future.size > 0:
                df.at[df.index[i],'Expected_Return'] = (np.nanmax(future) - cur)/cur
    return df

def compute_expected_loss(df, forward_window=14, s_cols=['S1_Avg','S2_Avg']):
    df['Expected_Loss'] = np.nan
    close = df['Close'].values
    for i in range(len(df) - forward_window):
        cur = close[i]
        future = close[i+1:i+1+forward_window]
        pivots = []
        for c in s_cols:
            if c in df.columns and not np.isnan(df[c].iloc[i]):
                pivots.append(df[c].iloc[i])
        target = min(pivots) if pivots else np.nan
        if not np.isnan(target):
            hit = False
            for p in future:
                if p <= target:
                    df.at[df.index[i],'Expected_Loss'] = (target - cur)/cur
                    hit = True; break
            if not hit and future.size > 0:
                df.at[df.index[i],'Expected_Loss'] = (np.nanmin(future) - cur)/cur
        else:
            if future.size > 0:
                df.at[df.index[i],'Expected_Loss'] = (np.nanmin(future) - cur)/cur
    return df

def label_hit_prob_past(df, window=14, profit_target=0.05, stop_loss=0.05):
    close = df['Close'].values
    N = len(close)
    labels = np.zeros(N, dtype=int)
    for i in range(N):
        if i >= N - window:
            labels[i] = 0; continue
        cur = close[i]
        tp = cur * (1 + profit_target)
        sl = cur * (1 - stop_loss)
        future = close[i+1:i+1+window]
        tp_idx = None; sl_idx = None
        for j,p in enumerate(future):
            if tp_idx is None and p >= tp: tp_idx = j
            if sl_idx is None and p <= sl: sl_idx = j
            if tp_idx is not None and sl_idx is not None: break
        if tp_idx is not None and (sl_idx is None or tp_idx < sl_idx): labels[i] = 2
        elif sl_idx is not None and (tp_idx is None or sl_idx < tp_idx): labels[i] = 1
        else: labels[i] = 0
    df['Hit_Label'] = labels
    return df

# ======= Training models (with safeguards) =======
def train_models(df, timeframe):
    required_cols = [c for c in FEATURES if c in df.columns] + ['Hit_Label','Expected_Return','Expected_Loss']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        # warn but continue — model training needs present subset
        st.warning(f"Missing cols for modeling ({timeframe}): {missing[:6]}{'...' if len(missing)>6 else ''}")
        return None, None, None, None, None, None

    df_model = df.dropna(subset=required_cols)
    if len(df_model) < MIN_TRAIN_ROWS.get(timeframe, 20):
        st.warning(f"Insufficient data for {timeframe}: {len(df_model)} rows (need {MIN_TRAIN_ROWS.get(timeframe)})")
        return None, None, None, None, None, None

    X_cls = df_model[[c for c in FEATURES if c in df_model.columns]]
    y_cls = df_model['Hit_Label'].astype(int)

    scaler_cls = StandardScaler()
    X_scaled_cls = scaler_cls.fit_transform(X_cls)

    model_class = RandomForestClassifier(n_estimators=60, max_depth=8, random_state=42, n_jobs=-1)
    model_class.fit(X_scaled_cls, y_cls)

    cls_probs = model_class.predict_proba(X_scaled_cls)
    prob_df = pd.DataFrame(0, index=df_model.index, columns=[f'Prob_Class_{c}' for c in EXPECTED_CLASSES])
    for i, c in enumerate(model_class.classes_):
        if c in EXPECTED_CLASSES:
            prob_df[f'Prob_Class_{c}'] = cls_probs[:, i]

    FEATURES_with_probs = list(X_cls.columns) + [f'Prob_Class_{c}' for c in EXPECTED_CLASSES]
    X_reg = pd.concat([df_model[X_cls.columns].reset_index(drop=True), prob_df.reset_index(drop=True)], axis=1)

    scaler_return = StandardScaler()
    X_scaled_return = scaler_return.fit_transform(X_reg[FEATURES_with_probs])

    model_return = RandomForestRegressor(n_estimators=60, max_depth=8, random_state=42, n_jobs=-1)
    model_return.fit(X_scaled_return, df_model['Expected_Return'].values)

    scaler_loss = StandardScaler()
    X_scaled_loss = scaler_loss.fit_transform(X_reg[FEATURES_with_probs])

    model_loss = RandomForestRegressor(n_estimators=60, max_depth=8, random_state=42, n_jobs=-1)
    model_loss.fit(X_scaled_loss, df_model['Expected_Loss'].values)

    return model_class, model_return, model_loss, scaler_cls, scaler_return, scaler_loss

# ======= Prediction & decision logic (blended confidence + gating) =======
def make_prediction(model_class, model_return, model_loss, scaler_cls, scaler_return, scaler_loss, latest_data):
    try:
        feats = [c for c in FEATURES if c in latest_data.columns]
        if latest_data[feats].isnull().values.any():
            missing_feats = latest_data[feats].columns[latest_data[feats].isnull().any()].tolist()
            st.warning(f"Missing features in latest row: {missing_feats}")
            return None

        latest_scaled = scaler_cls.transform(latest_data[feats])
        probs = model_class.predict_proba(latest_scaled)[0]
        prob_map = {c:0.0 for c in EXPECTED_CLASSES}
        for i,c in enumerate(model_class.classes_):
            prob_map[c] = probs[i]

        prob_tp = prob_map.get(2, 0.0)
        prob_sl = prob_map.get(1, 0.0)

        prob_features = {f'Prob_Class_{c}': prob_map[c] for c in EXPECTED_CLASSES}
        latest_reg = pd.concat([latest_data[feats].reset_index(drop=True), pd.DataFrame([prob_features])], axis=1)

        reg_feats = [f for f in FEATURES if f in latest_reg.columns] + [f'Prob_Class_{c}' for c in EXPECTED_CLASSES]
        for pcol in [f'Prob_Class_{c}' for c in EXPECTED_CLASSES]:
            if pcol not in latest_reg.columns:
                latest_reg[pcol] = 0.0

        Xr = scaler_return.transform(latest_reg[reg_feats])
        Xl = scaler_loss.transform(latest_reg[reg_feats])

        pred_return = model_return.predict(Xr)[0]
        pred_loss = model_loss.predict(Xl)[0]

        current_price = float(latest_data['Close'].values[0])
        predicted_tp = current_price * (1 + pred_return)
        predicted_sl = current_price * (1 + pred_loss)

        tp_pct = (predicted_tp - current_price) / current_price * 100
        sl_pct = (predicted_sl - current_price) / current_price * 100

        trend_score = 1.0 if (latest_data['SMA1'].values[0] > latest_data['SMA2'].values[0] and latest_data['Close'].values[0] > latest_data['SMA1'].values[0]) else 0.0
        atr_pct = latest_data['ATR'].values[0] / latest_data['Close'].values[0] * 100 if latest_data['ATR'].values[0] and latest_data['Close'].values[0] else 0.0
        vol_score = max(0.0, 1.0 - (atr_pct / 10.0))

        blended = 0.6 * prob_tp + 0.2 * trend_score + 0.2 * vol_score
        blended_pct = blended * 100

        rsi = latest_data['RSI'].values[0]
        uptrend = trend_score == 1.0
        rsi_ok = (rsi < 65) if uptrend else (rsi < 55)

        avoid_buy = False
        if (rsi > 70) or ('Upper_Band' in latest_data.columns and latest_data['Close'].values[0] > latest_data['Upper_Band'].values[0]) or (latest_data['Close'].values[0] > latest_data['SMA1'].values[0] * 1.02):
            avoid_buy = True

        rr = None
        if (current_price - predicted_sl) != 0:
            rr = (predicted_tp - current_price) / (current_price - predicted_sl)

        decision = "Hold / Wait"
        will_hit = 'None'
        if prob_tp > 0.65 and blended > 0.45 and rsi_ok and not avoid_buy and rr is not None and rr >= 1.5:
            decision = "Strong Buy"; will_hit = 'TP'
        elif prob_tp > 0.5 and blended > 0.4 and rr is not None and rr >= 1.2 and not avoid_buy:
            decision = "Buy (Cautious)"; will_hit = 'TP'
        elif prob_sl > 0.6:
            decision = "Avoid / Short bias"; will_hit = 'SL'
        else:
            pred_label = int(model_class.predict(latest_scaled)[0])
            will_hit = label2str.get(pred_label, 'None')
            decision = "Hold / Wait"

        return {
            'will_hit': will_hit,
            'prob_tp': prob_tp,
            'prob_sl': prob_sl,
            'blended_confidence_pct': blended_pct,
            'predicted_tp': predicted_tp,
            'predicted_sl': predicted_sl,
            'tp_pct': tp_pct,
            'sl_pct': sl_pct,
            'rr': rr,
            'decision_label': decision,
            'avoid_buy': avoid_buy,
            'current_price': current_price
        }
    except Exception:
        return None

# ======= Plotting (keeps your visual style) =======
def plot_analysis(ticker, df, entry_price, timeframe, assessment, prediction=None):
    try:
        fig, (ax1, ax2) = plt.subplots(2,1,figsize=(12,8), gridspec_kw={'height_ratios':[3,1]}, sharex=True)
        price = df['Close'].rolling(2, min_periods=1).mean()
        ax1.plot(df.index, price, label='Price', color='gray', alpha=0.5, linewidth=1)
        if 'SMA1' in df.columns: ax1.plot(df.index, df['SMA1'], label=f'SMA{int(_DAYS*0.5)}', color='orange', alpha=0.4, linewidth=1)
        if 'SMA2' in df.columns: ax1.plot(df.index, df['SMA2'], label=f'SMA{int(_DAYS*2)}', color='red', alpha=0.4, linewidth=1)
        if 'SMA1' in df.columns and 'SMA2' in df.columns:
            ax1.fill_between(df.index, df.SMA1, df.SMA2, where=(df.SMA1 > df.SMA2), facecolor='green', alpha=0.15)
            ax1.fill_between(df.index, df.SMA1, df.SMA2, where=(df.SMA1 < df.SMA2), facecolor='red', alpha=0.15)

        last_date = df.index[-1]
        ax1.plot(last_date, entry_price, 'o', markersize=5, color='black', alpha=0.3, label=f'Entry: ${entry_price:.2f}')

        if prediction:
            tp_price = prediction.get('predicted_tp', None)
            sl_price = prediction.get('predicted_sl', None)
            future_date = last_date + timedelta(days=20)
            if tp_price is not None and np.isfinite(tp_price):
                ax1.plot(future_date, tp_price, '^', markersize=6, color='blue')
                ax1.annotate(f'TP: ${tp_price:.2f}', xy=(future_date, tp_price), xytext=(5,5), textcoords='offset points', ha='left', va='center', color='blue')
                ax1.axhline(y=tp_price, color='blue', linestyle='--', alpha=0.3, linewidth=1.2)
            if sl_price is not None and np.isfinite(sl_price):
                ax1.plot(future_date, sl_price, 'v', markersize=6, color='red')
                ax1.annotate(f'SL: ${sl_price:.2f}', xy=(future_date, sl_price), xytext=(5,-5), textcoords='offset points', ha='left', va='center', color='red')
                ax1.axhline(y=sl_price, color='red', linestyle='--', alpha=0.3, linewidth=1.2)

        ax1.yaxis.tick_right(); ax1.yaxis.set_label_position("right"); ax1.set_ylabel('Price')
        ax1.legend(loc='upper left', fontsize='x-small'); ax1.grid(True, alpha=0.5)

        hint_text = "Hint: Buy closer to predicted SL to reduce risk\nand increase the chance of success."
        ax1.text(0.01, 0.02, hint_text, transform=ax1.transAxes, fontsize=10, color='gray',
                 bbox=dict(facecolor='honeydew', edgecolor='darkgreen', alpha=0.85, boxstyle='round,pad=0.6'))

        color_map = {'Strong Buy':'green','Buy (Cautious)':'orange','Avoid / Short bias':'red','Hold / Wait':'gray'}
        assessment_color = color_map.get(assessment, 'gray')
        ax1.annotate(f'Assessment: {assessment}', xy=(0.5,0.95), xycoords='axes fraction', ha='center', fontsize=12, weight='bold',
                     bbox=dict(boxstyle='round', facecolor=assessment_color, alpha=0.4))

        ax1.text(0.5, 0.5, f'@{ticker}', transform=ax1.transAxes, fontsize=50, color='grey', alpha=0.2, ha='center', va='center', style='italic', weight='bold')

        if 'RSI' in df.columns:
            rsi_ = df['RSI'].rolling(3, min_periods=1).mean()
            rsi_sma = df['RSI'].rolling(20, min_periods=1).mean()
            ax2.plot(df.index, rsi_, label='RSI', color='gray', linewidth=1.5, alpha=0.4)
            ax2.plot(df.index, rsi_sma, label='RSI SMA', color='red', linewidth=1.5, alpha=0.45)
            ax2.fill_between(df.index, rsi_, 52, where=(df['RSI'] > 52), facecolor='green', alpha=0.15)
            ax2.fill_between(df.index, rsi_, 40, where=(df['RSI'] < 40), facecolor='red', alpha=0.15)
            ax2.fill_between(df.index, rsi_, rsi_sma, where=((df['RSI'] < df['RSI_SMA']) & (df.SMA1 > df.SMA2)), facecolor='orange', alpha=0.14)
            ax2.axhline(70, color='red', linestyle='--', alpha=0.4); ax2.axhline(30, color='green', linestyle='--', alpha=0.4); ax2.axhline(50, color='gray', linestyle='-', alpha=0.4)
            if 'Bull' in df.columns: ax2.scatter(df.index[df['Bull'] == 1], rsi_[df['Bull'] == 1], color='green', marker='^', s=6, alpha=0.4)
            if 'Bear' in df.columns: ax2.scatter(df.index[df['Bear'] == 1], rsi_[df['Bear'] == 1], color='red', marker='v', s=6, alpha=0.4)
            if 'Short' in df.columns: ax2.scatter(df.index[df['Short'] == 1], rsi_[df['Short'] == 1], color='red', marker='x', s=6, alpha=0.4)
            if 'Hold' in df.columns: ax2.scatter(df.index[df['Hold'] == 1], rsi_[df['Hold'] == 1], color='orange', marker='o', s=6, alpha=0.4)
            ax2.yaxis.set_label_position("right"); ax2.yaxis.tick_right(); ax2.set_ylabel('RSI'); ax2.set_ylim(0,100); ax2.legend(loc='lower left', fontsize='x-small')
        else:
            ax2.text(0.5, 0.5, 'RSI data not available', ha='center', va='center', transform=ax2.transAxes)

        ax2.grid(True, alpha=0.3)
        plt.title(f'{timeframe} Analysis - {assessment}')
        plt.tight_layout()
        return fig
    except Exception as e:
        fig, ax = plt.subplots(figsize=(10,6))
        ax.text(0.5,0.5, f'Plot error: {e}', ha='center', va='center')
        return fig

# ======= Aggregate & UI helpers =======
def aggregate_timeframes(results):
    votes = {'Strong Buy':0, 'Buy (Cautious)':0, 'Avoid / Short bias':0, 'Hold / Wait':0}
    for tf, val in results.items():
        label = val.get('decision_label')
        votes[label] = votes.get(label, 0) + 1
    buy_votes = votes['Strong Buy'] + votes['Buy (Cautious)']
    short_votes = votes['Avoid / Short bias']
    if buy_votes >= 2:
        return 'BUY', votes
    elif short_votes >= 2:
        return 'SHORT/AVOID', votes
    else:
        return 'HOLD/WAIT', votes

# ======= Session-state utility =======
def init_session():
    # initialize keys we will use
    if 'last_fetch' not in st.session_state:
        st.session_state.last_fetch = None
    if 'data' not in st.session_state:
        st.session_state.data = {}   # dict keyed by timeframe -> df
    if 'models' not in st.session_state:
        st.session_state.models = {} # dict keyed by timeframe -> models tuple
    if 'predictions' not in st.session_state:
        st.session_state.predictions = {}
    if 'current_price' not in st.session_state:
        st.session_state.current_price = 0.0

# ======= Main App =======
def main():
    init_session()
    # meta refresh (browser reload every AUTO_REFRESH_SECONDS)
    st.markdown(f'<meta http-equiv="refresh" content="{AUTO_REFRESH_SECONDS}">', unsafe_allow_html=True)

    st.title("📊 Entry Position Analyzer — Auto Refresh (60s)")
    st.write("Auto-refreshes every 60 seconds. Use 'Refresh Data' to force immediate update.")

    cols = st.columns([2,1,1])
    with cols[0]:
        ticker = st.text_input("Ticker Symbol (e.g., TSLA or BTC-USD):", value="TSLA").upper()
    with cols[1]:
        entry_price = st.number_input("Entry Price ($)", min_value=0.0, value=float(st.session_state.get('entry_price', 0.0)), step=0.1)
    with cols[2]:
        user_gain = st.number_input("Expected Gain (%)", min_value=0.1, max_value=50.0, value=5.0, step=0.1)
        user_loss = st.number_input("Expected Loss (%)", min_value=0.1, max_value=50.0, value=4.5, step=0.1)

    refresh_col, manual_col = st.columns([1,1])
    with refresh_col:
        refresh_now = st.button("Refresh Data")
    with manual_col:
        # small label to indicate auto-refresh interval
        st.write(f"Auto refresh every {AUTO_REFRESH_SECONDS}s")

    if not ticker:
        st.warning("Please enter a ticker.")
        st.stop()

    # quick price update (fast)
    cur_price = fetch_latest_price(ticker)
    if cur_price:
        st.session_state.current_price = cur_price
        if st.session_state.get('entry_price', 0.0) == 0.0:
            st.session_state.entry_price = cur_price
    else:
        # fallback to stored
        cur_price = st.session_state.get('current_price', 0.0)

    # If user pressed refresh or nothing cached, redownload everything
    do_full_fetch = refresh_now or (st.session_state.last_fetch is None) or (datetime.now() - st.session_state.last_fetch > timedelta(seconds=AUTO_REFRESH_SECONDS))
    results = {}

    timeframes = [('4H','4H'), ('1D','1D'), ('1W','1W')]
    end_date = datetime.now()

    for timeframe, interval in timeframes:
        years = YEARS_OF_DATA[timeframe]
        start_date = end_date - timedelta(days=365 * years)

        # download if needed or forced refresh
        if do_full_fetch or (timeframe not in st.session_state.data):
            df = download_ohlc(ticker, start_date, end_date, interval)
            if df is None:
                st.warning(f"No data for {timeframe} ({ticker})")
                st.session_state.data.pop(timeframe, None)
                st.session_state.models.pop(timeframe, None)
                st.session_state.predictions.pop(timeframe, None)
                continue
            st.session_state.data[timeframe] = df
        else:
            df = st.session_state.data.get(timeframe)

        # check minimum raw rows
        if df is None or len(df) < MIN_TRAIN_ROWS.get(timeframe, 10):
            st.warning(f"Insufficient raw data for {timeframe}: {0 if df is None else len(df)} rows")
            continue

        # compute indicators each run (cheap relative to model training)
        df_ind = add_technical_indicators(df.copy(), timeframe)
        if df_ind is None:
            st.warning(f"Indicator calc failed for {timeframe}")
            continue

        # pivots and averages (safe)
        df_ind = add_pivots(df_ind, WINDOWS)
        df_ind = average_pivots(df_ind, [5,10,14,20])

        # expected returns and labeling
        df_ind = compute_expected_return(df_ind, forward_window=14, r_cols=['R1_Avg','R2_Avg'])
        df_ind = compute_expected_loss(df_ind, forward_window=14, s_cols=['S1_Avg','S2_Avg'])
        df_ind = label_hit_prob_past(df_ind, window=14, profit_target=user_gain/100.0, stop_loss=user_loss/100.0)

        # cache the indicators
        st.session_state.data[timeframe] = df_ind

        # if models not trained yet or we forced refresh, train
        if do_full_fetch or (timeframe not in st.session_state.models):
            with st.spinner(f"Training models for {timeframe}..."):
                models = train_models(df_ind, timeframe)
                if models[0] is None:
                    st.warning(f"Could not train models for {timeframe}")
                    st.session_state.models.pop(timeframe, None)
                    continue
                st.session_state.models[timeframe] = models
        else:
            models = st.session_state.models[timeframe]

        model_class, model_return, model_loss, scaler_cls, scaler_return, scaler_loss = models

        # make prediction on latest row
        latest_data = df_ind.iloc[[-1]]
        prediction = make_prediction(model_class, model_return, model_loss, scaler_cls, scaler_return, scaler_loss, latest_data)
        if prediction is None:
            st.warning(f"Prediction failed for {timeframe}")
            continue

        st.session_state.predictions[timeframe] = prediction
        results[timeframe] = prediction

        # display per-timeframe compact header + metrics
        st.subheader(f"{timeframe} analysis — {ticker}")
        left, right = st.columns(2)
        with left:
            st.metric("Current Price", f"${prediction['current_price']:.2f}")
            st.metric("Will Hit", prediction['will_hit'])
            st.metric("TP Prob", f"{prediction['prob_tp']*100:.1f}%")
        with right:
            st.metric("Predicted TP", f"${prediction['predicted_tp']:.2f}", delta=f"{prediction['tp_pct']:.2f}%")
            st.metric("Predicted SL", f"${prediction['predicted_sl']:.2f}", delta=f"{prediction['sl_pct']:.2f}%")
            st.metric("Confidence", f"{prediction['blended_confidence_pct']:.1f}%")

        # classification color
        if prediction['decision_label'] == "Strong Buy":
            st.success(prediction['decision_label'])
        elif prediction['decision_label'] == "Buy (Cautious)":
            st.warning(prediction['decision_label'])
        else:
            st.info(prediction['decision_label'])

        # plot (preserve your plotting style)
        fig = plot_analysis(ticker, df_ind, entry_price if entry_price > 0 else prediction['current_price'], timeframe, prediction['decision_label'], prediction)
        st.pyplot(fig)

        st.write("---")

    # aggregate summary columns (final)
    if results:
        overall, votes = aggregate_timeframes(results)
        col_overall, col_votes, col_quick = st.columns([1,1,2])
        with col_overall:
            if overall == 'BUY':
                st.success(f"🎯 Overall: {overall}")
            elif overall == 'SHORT/AVOID':
                st.error(f"🎯 Overall: {overall}")
            else:
                st.info(f"🎯 Overall: {overall}")
        with col_votes:
            st.write("**Votes**")
            st.write(votes)
        with col_quick:
            st.write("**Per-timeframe summary**")
            tfcols = st.columns(len(results))
            for i,(tf,pred) in enumerate(results.items()):
                with tfcols[i]:
                    st.subheader(tf)
                    st.write(f"Decision: **{pred['decision_label']}**")
                    st.write(f"Conf: {pred['blended_confidence_pct']:.1f}%")
                    st.write(f"TP: ${pred['predicted_tp']:.2f} ({pred['tp_pct']:.2f}%)")
                    st.write(f"SL: ${pred['predicted_sl']:.2f} ({pred['sl_pct']:.2f}%)")
            # quick advice
            st.write("**Quick Advice**")
            if overall == 'BUY':
                st.write("Consider staggered entries near support/predicted SL; size appropriately.")
            elif overall == 'SHORT/AVOID':
                st.write("Avoid longs; consider short bias or wait for clearer setup.")
            else:
                st.write("No multi-timeframe conviction — wait or reduce size.")

    else:
        st.info("No analysis results available (check ticker, data availability, or internet).")

    # update last_fetch timestamp
    st.session_state.last_fetch = datetime.now()

if __name__ == "__main__":
    main()
