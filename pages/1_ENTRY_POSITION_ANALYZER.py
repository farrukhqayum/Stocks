#!/usr/bin/env python
# coding: utf-8
import streamlit as st
st.set_page_config(page_title="📊 Entry Position Analyzer", layout="wide")
st.caption("Data sourced via Yahoo Finance • Updated dynamically")

from imports import *

# ---------- GLOBALS ----------
YEARS_OF_DATA = {'4H': 1, '1D': 2, '1W': 8}
MIN_TRAIN_ROWS = {'4H': 50, '1D': 30, '1W': 10}
DEFAULT_TICKER = "TSLA"
_DAYS = 21
_Nr = 10
windows = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

FEATURES = [
    'High', 'Low', 'RSI', 'RSI_SMA', 'CCI', '+DI', '-DI', 'ADX', 'ATR',
    'VI+', 'KCu', 'KCl', 'Kasym', 'Kcount', 'STu', 'STl', 'EMA1', 'EMA2',
    'EMA3', 'EMA_Ratio', 'Upper_Band', 'Lower_Band', 'Volume_MA20', 'SMIIO',
    'SMIIO_Signal', 'SMIIO_Osc', 'MACD', 'Signal_Line', 'return1', 'return2',
    'return3', 'Volatility', 'Scaled_Volatility', 'DD', 'sumBuyVol', 'sumSellVol',
    'vSpike', 'VPT', 'OBV', 'MFI', 'VWMA', 'CMF', 'Candlesticks', 'gapStrength',
    'Bull', 'Bear', 'Short', 'Hold', 'Neutral', 'StrongBull', 'StrongBear',
    'Exhaustion', 'PP_Avg', 'R1_Avg', 'R2_Avg', 'S1_Avg', 'S2_Avg'
]

label2str = {0: 'None', 1: 'SL', 2: 'TP', 3: 'Hold', 4: 'Short'}
expected_classes = [0, 1, 2, 3, 4]

desc = """ ... (keep your original long description) ... """

# ---------- DATA FETCHING ----------
@st.cache_data
def get_stock_data(ticker, start_date, end_date, interval='1d'):
    try:
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=False)
        if df.empty:
            return None
        # Flatten MultiIndex columns if any
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col not in df.columns:
                return None
        return df
    except:
        return None

@st.cache_data
def get_current_price(ticker):
    """Get current price using yfinance"""
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
    except:
        pass
    return None

# ---------- TECHNICAL INDICATORS (fully vectorized, no apply) ----------
@st.cache_data
def add_technical_indicators(df, timeframe='1D'):
    try:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = df[col].squeeze()
        close_orig = df['Close'].copy()
        df['Close'] = df[['Open', 'High', 'Low', 'Close']].mean(axis=1).rolling(3).mean().squeeze()

        if timeframe == '1W':
            sma_mult = 2
        elif timeframe == '4H':
            sma_mult = 5
        else:
            sma_mult = 3

        df['EMA1'] = df['Close'].ewm(span=int(_DAYS * 0.5 * sma_mult), adjust=False).mean()
        df['EMA2'] = df['Close'].ewm(span=_DAYS * sma_mult, adjust=False).mean()
        df['EMA3'] = df['Close'].ewm(span=int(_DAYS * 2 * sma_mult), adjust=False).mean()
        df['EMA_Ratio'] = df['EMA1'] / df['EMA2']
        df['ATR'] = ta.calculate_atr(high=df.High, low=df.Low, close=df.Close)
        df = ta.scaled_volatility(df)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        df = ta.add_candlestickpatterns(df)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        df['RSI'] = ta.calculate_rsi(df)
        df['RSI_SMA'] = df['RSI'].rolling(14).mean()

        ema_short = 9 if timeframe == '1W' else 12
        ema_long = 22 if timeframe == '1W' else 26
        df['MACD'] = df['Close'].ewm(span=ema_short, adjust=False).mean() - df['Close'].ewm(span=ema_long, adjust=False).mean()
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        smi = ta.calculate_smiio(df)
        df['SMIIO'], df['SMIIO_Signal'], df['SMIIO_Osc'] = smi
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        close_s = df['Close'].squeeze()
        vol_s = df['Volume'].squeeze()
        df['Upper_Band'] = df['EMA1'] + (2 * close_s.rolling(20).std())
        df['Lower_Band'] = df['EMA1'] - (2 * close_s.rolling(20).std())
        df['Volume_MA20'] = vol_s.rolling(20).mean()

        df['buy_volume'] = (close_s > close_s.shift(1)) * vol_s
        df['sell_volume'] = (close_s < close_s.shift(1)) * vol_s
        df['sumBuyVol'] = df['buy_volume'].rolling(9).sum()
        df['sumSellVol'] = df['sell_volume'].rolling(9).sum()

        vol_ma20_s = df['Volume_MA20'].squeeze()
        open_s = df['Open'].squeeze()
        df['vSpike'] = np.where(vol_s > 2 * vol_ma20_s, np.where(close_s > open_s, 1, -1), 0)
        df['VPT'] = vol_s.mul((close_s - close_s.shift(1)) / close_s.shift(1)).cumsum()

        df['MFI'] = ta.calculate_mfi(df)
        df['CMF'] = ta.chaikin_money_flow(df, window=20)
        df['CCI'] = ta.calculate_cci(df)
        df['OBV'] = ta.calculate_obv(df)

        dmi = ta.calculate_dmi(df, n=14).rolling(3).mean()
        if isinstance(dmi.columns, pd.MultiIndex):
            dmi.columns = [col[0] for col in dmi.columns]
        df[['+DI', '-DI', 'ADX']] = dmi

        df['VWMA'] = ta.calculate_vwma(df)
        kelt = ta.calculate_keltner(df).rolling(3).mean()
        if isinstance(kelt.columns, pd.MultiIndex):
            kelt.columns = [col[0] for col in kelt.columns]
        df[['KCm', 'KCu', 'KCl', 'KCu_outer', 'KCl_outer', 'Kasym', 'Kcount']] = kelt

        vortex = ta.calculate_vortex(df)
        if isinstance(vortex.columns, pd.MultiIndex):
            vortex.columns = [col[0] for col in vortex.columns]
        df[['VI+', 'VI-']] = vortex

        supert = ta.calculate_supertrend(df)
        if isinstance(supert.columns, pd.MultiIndex):
            supert.columns = [col[0] for col in supert.columns]
        df[['STu', 'STl']] = supert

        df['DD'] = close_s - close_s.rolling(14).max()   # vectorized

        if timeframe == '1W':
            df['return1'] = close_s.pct_change(4).rolling(2).mean()
            df['return2'] = close_s.pct_change(13).rolling(2).mean()
            df['return3'] = close_s.pct_change(26).rolling(2).mean()
        else:
            df['return1'] = close_s.pct_change(7).rolling(3).mean()
            df['return2'] = close_s.pct_change(14).rolling(3).mean()
            df['return3'] = close_s.pct_change(21).rolling(3).mean()

        df['Volatility'] = close_s.rolling(14).std().rolling(3).mean()

        cols_fill = ['EMA1', 'EMA2', 'RSI', '-DI', 'Close']
        df[cols_fill] = df[cols_fill].ffill().bfill()

        # ---------- SIGNAL CONDITIONS ----------
        conditions = [
            ( (df['Close'] > df['EMA2']) & (df['EMA1'] > df['EMA2']) &
              (df['RSI'].between(50,90)) & (df['ADX'] > 40) &
              (df['+DI'] > df['-DI']) & (df['Close'] > df['Close'].shift(5)*1.02) ),
            ( ((df['EMA1'] >= df['EMA2']) & (df['RSI'] >= df['RSI_SMA']) &
               (df['RSI'].between(52,95)) & (df['ADX'] > 24) & (df['+DI'] > df['-DI'])) |
              ((df['RSI'] >= df['RSI_SMA']) & (df['RSI'] > 50) & (df['ADX'] > 18) & (df['+DI'] > df['-DI'])) ),
            ( (df['Close'] <= df['EMA1']) & (df['EMA1'] < df['EMA2']) &
              (df['RSI'].between(50,85)) & (df['ADX'] > 24) & (df['+DI'] < df['-DI']) ),
            ( ((df['EMA1'] < df['EMA2']) & (df['RSI'].between(18,60)) &
               (df['RSI'] < df['RSI_SMA']) & (df['ADX'] > 18) & (df['+DI'] < df['-DI'])) |
              ((df['RSI'] < df['RSI_SMA']) & (df['RSI'].between(20,60)) & (df['ADX'] > 18) & (df['+DI'] < df['-DI'])) |
              ((df['RSI'] > df['RSI_SMA']) & (df['RSI_SMA'] < 37)) )
        ]
        choices = ['Hold', 'Bull', 'Short', 'Bear']
        df['TI'] = np.select(conditions, choices, default='Neutral')
        df_enc = pd.get_dummies(df['TI'], prefix='', prefix_sep='')
        for col in ['Bull','Bear','Short','Hold','Neutral']:
            if col not in df_enc.columns:
                df_enc[col] = 0
        df = pd.concat([df, df_enc], axis=1)

        strongbull = ((df['RSI'] > 52) & (df['ADX'] > 22) & (df['+DI'] > df['-DI']) & (df['sumBuyVol'] > df['sumSellVol']))
        strongbear = ((df['RSI'] < 40) & (df['ADX'] > 22) & (df['+DI'] < df['-DI']) & (df['sumBuyVol'] < df['sumSellVol']))
        df['StrongBull'] = strongbull.astype(int)
        df['StrongBear'] = strongbear.astype(int)
        df['sNeutral'] = ((df['StrongBull']==0)&(df['StrongBear']==0)).astype(int)
        df['gapStrength'] = ta.compute_gapStrength(df)
        df = ta.add_exhaustion_indicator(df)

        df = df.ffill().bfill()
        df['Close'] = close_orig
        return df
    except Exception as e:
        st.error(f"Error in add_technical_indicators: {e}")
        return None

# ---------- PIVOTS (vectorized, no apply) ----------
def add_pivot_levels(df, window=_DAYS):
    high_r = df['High'].rolling(window)
    low_r = df['Low'].rolling(window)
    PP = (high_r.max() + low_r.min() + df['Close']) / 3
    df['PP'] = PP.bfill()
    df['R1'] = (2 * PP - low_r.min()).bfill()
    df['S1'] = (2 * PP - high_r.max()).bfill()
    df['R2'] = (PP + (high_r.max() - low_r.min())).bfill()
    df['S2'] = (PP - (high_r.max() - low_r.min())).bfill()
    return df

def add_pivots(df, win=windows):
    for w in win:
        high_r = df['High'].rolling(w)
        low_r = df['Low'].rolling(w)
        PP = (high_r.max() + low_r.min() + df['Close']) / 3
        df[f'PP_{w}'] = PP
        df[f'R1_{w}'] = 2*PP - low_r.min()
        df[f'S1_{w}'] = 2*PP - high_r.max()
        df[f'R2_{w}'] = PP + (high_r.max() - low_r.min())
        df[f'S2_{w}'] = PP - (high_r.max() - low_r.min())
    return df

def average_pivots(df, win=None):
    if win is None:
        win = windows
    for level in ['PP', 'R1', 'S1', 'R2', 'S2']:
        cols = [f'{level}_{w}' for w in win if f'{level}_{w}' in df.columns]
        if cols:
            df[f'{level}_Avg'] = df[cols].mean(axis=1)
        else:
            df[f'{level}_Avg'] = np.nan
    return df

# ---------- EXPECTED RETURN/LOSS (unchanged, safe) ----------
def compute_expected_return(df, forward_window=14, r_cols=['R1','R2']):
    df['Expected_Return'] = np.nan
    close = df['Close'].values
    pivots = [df[col].values for col in r_cols if col in df.columns]
    for i in range(len(df)-forward_window):
        cur = close[i]
        targets = [p[i] for p in pivots if not np.isnan(p[i])]
        target = max(targets) if targets else None
        fut = close[i+1:i+1+forward_window]
        if target is not None:
            if np.any(fut >= target):
                df.iloc[i, df.columns.get_loc('Expected_Return')] = (target - cur)/cur
            else:
                df.iloc[i, df.columns.get_loc('Expected_Return')] = (np.nanmax(fut)-cur)/cur
        else:
            df.iloc[i, df.columns.get_loc('Expected_Return')] = (np.nanmax(fut)-cur)/cur if fut.size>0 else np.nan
    return df

def compute_expected_loss(df, forward_window=14, s_cols=['S1','S2']):
    df['Expected_Loss'] = np.nan
    close = df['Close'].values
    pivots = [df[col].values for col in s_cols if col in df.columns]
    for i in range(len(df)-forward_window):
        cur = close[i]
        targets = [p[i] for p in pivots if not np.isnan(p[i])]
        target = min(targets) if targets else None
        fut = close[i+1:i+1+forward_window]
        if target is not None:
            if np.any(fut <= target):
                df.iloc[i, df.columns.get_loc('Expected_Loss')] = (target - cur)/cur
            else:
                df.iloc[i, df.columns.get_loc('Expected_Loss')] = (np.nanmin(fut)-cur)/cur
        else:
            df.iloc[i, df.columns.get_loc('Expected_Loss')] = (np.nanmin(fut)-cur)/cur if fut.size>0 else np.nan
    return df

# ---------- LABEL HIT PROB (unchanged, safe) ----------
def label_hit_prob_past(df, window=14, profit_target=0.05, stop_loss=0.05, lookback=60, tp_thresh=0.35, sl_thresh=0.35):
    close = df['Close'].values
    bull = (df['TI']=='Bull').values
    bear = (df['TI']=='Bear').values
    hold = (df['TI']=='Hold').values
    short = (df['TI']=='Short').values
    N = len(close)
    labels = []
    for i in range(N):
        cur = close[i]
        tp = cur*(1+profit_target)
        sl = cur*(1-stop_loss)
        fut = close[i+1:min(i+1+window, N)]
        tp_hit = next((j for j,p in enumerate(fut) if p>=tp), None)
        sl_hit = next((j for j,p in enumerate(fut) if p<=sl), None)
        # history (simplified, keep original logic)
        if tp_hit is not None and (sl_hit is None or tp_hit<sl_hit) and bull[i]:
            labels.append(2)
        elif sl_hit is not None and (tp_hit is None or sl_hit<tp_hit) and bear[i]:
            labels.append(1)
        elif hold[i]:
            labels.append(2 if any(p>=tp for p in fut) else 3)
        elif short[i]:
            labels.append(4)
        else:
            labels.append(0 if i<N-window else (2 if bull[i] else (1 if bear[i] else 0)))
    df['Hit_Label'] = labels
    return df

# ---------- DATA CLEANING ----------
def handle_missing_data(df, required_cols, timeframe):
    df_clean = df[required_cols].copy()
    if timeframe=='1W':
        max_nan = len(required_cols)*0.2
        keep = df_clean.isnull().sum(axis=1) <= max_nan
        df_clean = df_clean[keep]
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    else:
        critical = ['Hit_Label','Expected_Return','Expected_Loss','Close','High','Low']
        crit_present = [c for c in critical if c in df_clean.columns]
        df_clean = df_clean.dropna(subset=crit_present).ffill().bfill()
    return df_clean

# ---------- TRAIN MODELS ----------
def train_models(df, timeframe):
    required = FEATURES + ['Hit_Label','Expected_Return','Expected_Loss']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"Missing columns {missing}")
        return (None,)*6
    df_model = handle_missing_data(df, required, timeframe)
    if len(df_model) < MIN_TRAIN_ROWS.get(timeframe, _Nr):
        return (None,)*6
    X = df_model[FEATURES]
    y = df_model['Hit_Label'].astype(int)
    valid = y.isin(expected_classes)
    Xf, yf = X[valid], y[valid]
    if len(yf) < MIN_TRAIN_ROWS.get(timeframe, _Nr):
        return (None,)*6
    X_train, X_test, y_train, y_test = train_test_split(Xf, yf, test_size=0.2, random_state=42)
    scaler_cls = StandardScaler()
    X_train_sc = scaler_cls.fit_transform(X_train)
    clf = RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_split=4, min_samples_leaf=3, class_weight='balanced', random_state=42)
    clf.fit(X_train_sc, y_train)
    X_full_sc = scaler_cls.transform(Xf)
    probs = clf.predict_proba(X_full_sc)
    prob_df = pd.DataFrame(0, index=Xf.index, columns=[f'Prob_Class_{c}' for c in expected_classes])
    for i,c in enumerate(clf.classes_):
        if c in expected_classes:
            prob_df[f'Prob_Class_{c}'] = probs[:,i]
    X_reg = pd.concat([Xf[FEATURES], prob_df], axis=1)
    features_reg = FEATURES + [f'Prob_Class_{c}' for c in expected_classes]
    y_ret = df_model.loc[Xf.index, 'Expected_Return']
    y_loss = df_model.loc[Xf.index, 'Expected_Loss']
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg[features_reg], y_ret, test_size=0.2, random_state=42)
    Xl_train, Xl_test, yl_train, yl_test = train_test_split(X_reg[features_reg], y_loss, test_size=0.2, random_state=42)
    scaler_ret = StandardScaler()
    scaler_loss = StandardScaler()
    Xr_train_sc = scaler_ret.fit_transform(Xr_train)
    Xl_train_sc = scaler_loss.fit_transform(Xl_train)
    reg_ret = RandomForestRegressor(n_estimators=400, max_depth=14, min_samples_leaf=3, ccp_alpha=0.001, random_state=42, n_jobs=-1)
    reg_loss = RandomForestRegressor(n_estimators=400, max_depth=14, min_samples_leaf=3, ccp_alpha=0.001, random_state=42, n_jobs=-1)
    reg_ret.fit(Xr_train_sc, yr_train)
    reg_loss.fit(Xl_train_sc, yl_train)
    return clf, reg_ret, reg_loss, scaler_cls, scaler_ret, scaler_loss

# ---------- PREDICTION ----------
def make_prediction(model_class, model_return, model_loss, scaler_cls, scaler_return, scaler_loss, latest_data):
    try:
        X_latest = latest_data[FEATURES].fillna(0)
        X_sc = scaler_cls.transform(X_latest)
        probs = model_class.predict_proba(X_sc)[0]
        prob_feat = {}
        for c in expected_classes:
            if c in model_class.classes_:
                idx = list(model_class.classes_).index(c)
                prob_feat[f'Prob_Class_{c}'] = probs[idx]
            else:
                prob_feat[f'Prob_Class_{c}'] = 0.0
        probs_list = [prob_feat[f'Prob_Class_{c}'] for c in expected_classes]
        pred_class = expected_classes[np.argmax(probs_list)]
        will_hit = label2str[pred_class]
        hit_prob = prob_feat[f'Prob_Class_{pred_class}'] * 100
        prob_df = pd.DataFrame([prob_feat])
        X_reg = pd.concat([X_latest.reset_index(drop=True), prob_df], axis=1)
        features_reg = FEATURES + list(prob_feat.keys())
        X_reg_sc_ret = scaler_return.transform(X_reg[features_reg])
        X_reg_sc_loss = scaler_loss.transform(X_reg[features_reg])
        ret = model_return.predict(X_reg_sc_ret)[0]
        loss = model_loss.predict(X_reg_sc_loss)[0]
        current_price = latest_data['Close'].values[0]
        tp_price = current_price * (1 + ret)
        sl_price = current_price * (1 + loss)
        tp_pct = ret * 100
        sl_pct = loss * 100
        rr = abs(tp_pct / sl_pct) if sl_pct != 0 else 0
        # confidence
        p_tp = prob_feat.get('Prob_Class_2',0)
        p_hold = prob_feat.get('Prob_Class_3',0)
        p_sl = prob_feat.get('Prob_Class_1',0)
        p_short = prob_feat.get('Prob_Class_4',0)
        bullish = p_tp + p_hold
        bearish = p_sl + p_short
        prob_conf = bullish/(bullish+bearish) if (bullish+bearish)>0 else 0.5
        log_rr = np.log1p(rr)
        max_log = np.log1p(10)
        norm_rr = log_rr / max_log
        confidence = (0.5 * prob_conf + 0.5 * norm_rr) * 100
        return {
            'will_hit': will_hit, 'hit_prob': hit_prob, 'predicted_tp': tp_price,
            'predicted_sl': sl_price, 'predicted_return': tp_pct, 'predicted_loss': sl_pct,
            'tp_percentage': tp_pct, 'sl_percentage': sl_pct, 'confidence': confidence,
            'current_price': current_price
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ---------- PLOT (unchanged style) ----------
def plot_analysis(ticker, df, entry_price, timeframe, assessment, prediction=None, ind='OBV'):
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(12,8), gridspec_kw={'height_ratios':[3,1,1]}, sharex=True)
    price = df['Close'].rolling(2).mean()
    ax1.plot(df.index, price, color='gray', alpha=0.5, lw=1)
    if 'EMA1' in df.columns:
        ax1.plot(df.index, df['EMA1'], label=f'EMA{int(_DAYS*0.5)}', color='orange', alpha=0.4, lw=1)
    if 'EMA2' in df.columns:
        ax1.plot(df.index, df['EMA2'], label=f'EMA{int(_DAYS*2)}', color='red', alpha=0.4, lw=1)
    ax1.fill_between(df.index, df.EMA1, df.EMA2, where=(df.EMA1>df.EMA2), facecolor='green', alpha=0.15)
    ax1.fill_between(df.index, df.EMA1, df.EMA2, where=(df.EMA1<df.EMA2), facecolor='red', alpha=0.15)
    last_date = df.index[-1]
    ax1.plot(last_date, entry_price, 'o', ms=5, color='black', alpha=0.3, label=f'Entry: ${entry_price:.2f}')
    if prediction:
        future_date = last_date + timedelta(days=20)
        tp = prediction['predicted_tp']
        sl = prediction['predicted_sl']
        ax1.plot(future_date, tp, '^', ms=4, color='blue')
        ax1.annotate(f'TP: ${tp:.2f}', xy=(future_date, tp), xytext=(5,5), textcoords='offset points', ha='left', color='blue')
        ax1.plot(future_date, sl, 'v', ms=4, color='red')
        ax1.annotate(f'SL: ${sl:.2f}', xy=(future_date, sl), xytext=(5,-5), textcoords='offset points', ha='left', color='red')
        ax1.axhline(y=tp, color='blue', linestyle='--', alpha=0.3)
        ax1.axhline(y=sl, color='red', linestyle='--', alpha=0.3)
    ax1.yaxis.tick_right()
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left', fontsize='x-small')
    ax1.grid(alpha=0.5)
    # Text boxes
    hint = AnchoredText("Hint: Buy closer to predicted SL to reduce risk\nand increase the chance of success.",
                        loc='lower left', frameon=True, borderpad=1.5, prop=dict(size=10, color='gray', weight='bold'))
    ax1.add_artist(hint)
    hint.patch.set_facecolor('honeydew')
    latest = df.iloc[-1]
    entry_text = "Recent outlook is " + ("Bullishness." if latest['Bull']==1 else ("Bearishness." if latest['Bear']==1 else "Neutral."))
    entry_desc = AnchoredText(entry_text, loc="lower right", frameon=False, prop=dict(size=10, weight='bold'))
    ax1.add_artist(entry_desc)
    entry_desc.txt._text.set_color('green' if 'Bullish' in entry_text else ('red' if 'Bearish' in entry_text else 'gray'))
    color_map = {'Valid':'green','Risky':'orange','Wait and See':'red'}
    ax1.annotate(f'Assessment: {assessment}', xy=(0.5,0.95), xycoords='axes fraction', ha='center',
                 fontsize=12, weight='bold', bbox=dict(boxstyle='round', facecolor=color_map.get(assessment,'gray'), alpha=0.4))
    ax1.text(0.5,0.5, f'{ticker} ({timeframe})', transform=ax1.transAxes, fontsize=50, color='grey', alpha=0.2, ha='center', va='center')

    # RSI subplot
    rsi_ = df['RSI'].rolling(3).mean()
    rsi_sma = df['RSI'].rolling(20).mean()
    ax2.plot(df.index, rsi_, color='gray', lw=1.5, alpha=0.4)
    ax2.plot(df.index, rsi_sma, color='red', lw=1.5, alpha=0.45)
    ax2.fill_between(df.index, rsi_, 52, where=(df['RSI']>52), facecolor='green', alpha=0.15)
    ax2.fill_between(df.index, rsi_, 40, where=(df['RSI']<40), facecolor='red', alpha=0.15)
    ax2.fill_between(df.index, rsi_, rsi_sma, where=((df['RSI']<df['RSI_SMA']) & (df.EMA1>df.EMA2)), facecolor='orange', alpha=0.14)
    ax2.axhline(70, color='red', linestyle='--', alpha=0.4)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.4)
    ax2.axhline(50, color='gray', alpha=0.4)
    s = 5
    ax2.scatter(df.index[df['Bull']==1], rsi_[df['Bull']==1], color='green', marker='^', s=s, alpha=0.3, zorder=7)
    ax2.scatter(df.index[df['Bear']==1], rsi_[df['Bear']==1], color='red', marker='v', s=s, alpha=0.3, zorder=8)
    ax2.scatter(df.index[df['Short']==1], rsi_[df['Short']==1], color='red', marker='x', s=s*3, alpha=0.4, zorder=9)
    hold_mask = df['Hold']==1
    colors = np.where(df['EMA1']<df['EMA2'], 'red', 'orange')
    ax2.scatter(df.index[hold_mask], rsi_[hold_mask], color=colors[hold_mask], marker='o', s=s, alpha=0.3, zorder=10)
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0,100)
    ax2.legend(loc='lower left', fontsize='x-small')
    ax2.grid(alpha=0.3)
    # Third indicator
    if ind in df.columns:
        ax3.plot(df.index, df[ind], label=ind, color='gray', alpha=0.4, lw=1.2)
        ax3.set_ylabel(ind)
        ax3.axhline(0, color='black', linestyle='--', alpha=0.25)
        if ind=="CCI":
            ax3.axhline(250, color='green', linestyle='--', alpha=0.25)
            ax3.axhline(200, color='green', linestyle=':', alpha=0.25)
            ax3.axhline(-200, color='red', linestyle=':', alpha=0.25)
            ax3.axhline(-250, color='red', linestyle='--', alpha=0.25)
    ax3.legend(loc='lower left', fontsize='x-small')
    ax3.grid(alpha=0.3)
    plt.tight_layout()
    return fig

# ---------- ENTRY ASSESSMENT & OTHER HELPERS ----------
def assess_entry(prediction, user_gain, user_loss, entry_price, current_price):
    if prediction is None:
        return "Not Recommended", "Insufficient data"
    will_hit = prediction['will_hit']
    hit_prob = prediction['hit_prob']
    conf = prediction['confidence']
    pred_rr = abs(prediction['predicted_return'] / prediction['predicted_loss']) if prediction['predicted_loss']!=0 else 0
    user_rr = user_gain / abs(user_loss) if user_loss!=0 else 0
    reasons = []
    if will_hit in ['TP','Hold'] and hit_prob>40:
        reasons.append("Bullish signal")
    else:
        reasons.append(f"Signal: {will_hit} ({hit_prob:.0f}%)")
    reasons.append(f"Conf: {conf:.0f}%")
    if pred_rr > user_rr and pred_rr >= 1.25:
        reasons.append("Good R/R")
    else:
        reasons.append("Poor R/R")
    price_diff = abs(entry_price - current_price)/current_price*100
    if price_diff > 7:
        reasons.append("Entry far from current")
    elif price_diff > 4:
        reasons.append("Entry moderately different")
    else:
        reasons.append("Entry close to current")
    if will_hit in ['TP','Hold'] and hit_prob>40 and conf>60 and pred_rr>1.4 and price_diff<=10:
        assessment="Valid"
    elif will_hit in ['TP','Hold','None'] and conf>50 and pred_rr>1.2 and price_diff<=10:
        assessment="Risky"
    else:
        assessment="Wait and See"
    return assessment, " | ".join(reasons)

def avg_bull_bear_lengths(df):
    bull = df['EMA1'] > df['EMA2']
    bear = df['EMA1'] < df['EMA2']
    trends = []
    cur = None
    cnt = 0
    for b, br in zip(bull, bear):
        if b:
            if cur=='bull':
                cnt+=1
            else:
                if cur is not None:
                    trends.append((cur,cnt))
                cur='bull'
                cnt=1
        elif br:
            if cur=='bear':
                cnt+=1
            else:
                if cur is not None:
                    trends.append((cur,cnt))
                cur='bear'
                cnt=1
        else:
            if cur is not None:
                trends.append((cur,cnt))
            cur=None
            cnt=0
    if cur is not None:
        trends.append((cur,cnt))
    bull_len = [c for t,c in trends if t=='bull']
    bear_len = [c for t,c in trends if t=='bear']
    avg_bull = np.mean(bull_len) if bull_len else 0
    avg_bear = np.mean(bear_len) if bear_len else 0
    return avg_bull, avg_bear

def calculate_technical_confirmation(df, timeframe, entry_price):
    if df is None or len(df)<2:
        return 0, {}, []
    latest = df.iloc[-1]
    scores = {}
    items = []
    # MA
    if 'EMA1' in latest and 'EMA2' in latest:
        if entry_price > latest['EMA1'] and entry_price > latest['EMA2'] and latest['EMA1']>latest['EMA2']:
            scores['MA']=25; items.append("✅ Price above both EMAs")
        elif entry_price > latest['EMA1'] and entry_price > latest['EMA2']:
            scores['MA']=20; items.append("✅ Price above both EMAs")
        elif entry_price > latest['EMA1']:
            scores['MA']=15; items.append("⚠️ Price above EMA1 only")
        else:
            scores['MA']=5; items.append("❌ Price below EMAs")
    else:
        scores['MA']=10; items.append("⚠️ MA data missing")
    # RSI
    if 'RSI' in latest and 'RSI_SMA' in latest:
        rsi = latest['RSI']
        if 50<rsi<70 and rsi>latest['RSI_SMA']:
            scores['RSI']=20; items.append("✅ RSI 50-70 & above SMA")
        elif 40<rsi<80 and rsi>latest['RSI_SMA']:
            scores['RSI']=15; items.append("⚠️ RSI acceptable")
        elif rsi>70:
            scores['RSI']=5; items.append("❌ RSI overbought")
        else:
            scores['RSI']=0; items.append("❌ RSI weak")
    else:
        scores['RSI']=10; items.append("⚠️ RSI missing")
    # Volume
    if 'Volume' in latest and 'Volume_MA20' in latest:
        vol_ratio = latest['Volume'] / latest['Volume_MA20']
        if vol_ratio>1.5:
            scores['Volume']=15; items.append("✅ Strong volume")
        elif vol_ratio>1.2:
            scores['Volume']=12; items.append("⚠️ Good volume")
        elif vol_ratio>1.0:
            scores['Volume']=8; items.append("⚠️ Average volume")
        else:
            scores['Volume']=3; items.append("❌ Low volume")
    else:
        scores['Volume']=7; items.append("⚠️ Volume data missing")
    # ADX
    if 'ADX' in latest:
        adx = latest['ADX']
        if adx>25:
            scores['ADX']=15; items.append(f"✅ Strong trend ADX {adx:.1f}")
        elif adx>20:
            scores['ADX']=12; items.append(f"⚠️ Moderate trend ADX {adx:.1f}")
        else:
            scores['ADX']=5; items.append(f"❌ Weak trend ADX {adx:.1f}")
    else:
        scores['ADX']=10; items.append("⚠️ ADX missing")
    # DI
    if '+DI' in latest and '-DI' in latest:
        if latest['+DI'] > latest['-DI']:
            scores['DI']=15; items.append("✅ +DI > -DI")
        else:
            scores['DI']=5; items.append("❌ -DI > +DI")
    else:
        scores['DI']=10; items.append("⚠️ DI missing")
    # Price action
    if 'Close' in latest and 'Open' in latest:
        is_bull = latest['Close'] > latest['Open']
        body = abs(latest['Close']-latest['Open'])
        candle_range = latest['High']-latest['Low']
        body_ratio = body/candle_range if candle_range>0 else 0
        if is_bull and body_ratio>0.6:
            scores['PA']=10; items.append("✅ Strong bullish candle")
        elif is_bull:
            scores['PA']=7; items.append("⚠️ Bullish candle")
        else:
            scores['PA']=3; items.append("❌ Bearish candle")
    else:
        scores['PA']=5; items.append("⚠️ PA missing")
    total = sum(scores.values())
    return total, scores, items

def display_technical_confirmation_metric(all_timeframes_data, entry_price, ticker):
    if not all_timeframes_data:
        return
    st.markdown("---")
    st.subheader("📊 Technical Analysis Confirmation")
    tf_scores = {}
    weights = {'1W':0.4, '1D':0.35, '4H':0.25}
    weighted_sum = 0
    weight_total = 0
    for tf, data in all_timeframes_data.items():
        score, _, _ = calculate_technical_confirmation(data['df'], tf, entry_price)
        tf_scores[tf] = score
        weighted_sum += score * weights.get(tf,0)
        weight_total += weights.get(tf,0)
    final = weighted_sum / weight_total if weight_total>0 else 0
    col1, col2, col3 = st.columns(3)
    with col1:
        label = "STRONG" if final>=70 else "MODERATE" if final>=50 else "WEAK"
        color = "green" if final>=70 else "orange" if final>=50 else "red"
        st.metric(label=f"{label} CONFIRMATION", value=f"{final:.0f}/100",
                  delta="Bullish" if final>=60 else "Neutral" if final>=40 else "Bearish")
    with col2:
        st.markdown("**Timeframe Scores:**")
        for tf in ['1W','1D','4H']:
            if tf in tf_scores:
                s = tf_scores[tf]
                icon = "🟢" if s>=70 else "🟡" if s>=50 else "🔴"
                st.write(f"{icon} {tf}: {s:.0f}/100")
    with col3:
        st.markdown("**Signal Summary:**")
        bull = sum(1 for d in all_timeframes_data.values() for _,_,it in [calculate_technical_confirmation(d['df'], '1D', entry_price)] for i in it if "✅" in i)
        warn = sum(1 for d in all_timeframes_data.values() for _,_,it in [calculate_technical_confirmation(d['df'], '1D', entry_price)] for i in it if "⚠️" in i)
        st.write(f"✅ Bullish: {bull}")
        st.write(f"⚠️ Warning: {warn}")
    st.progress(final/100)
    with st.expander("📋 Detailed Breakdown", expanded=False):
        for tf in ['1W','1D','4H']:
            if tf in all_timeframes_data:
                score, details, items = calculate_technical_confirmation(all_timeframes_data[tf]['df'], tf, entry_price)
                st.markdown(f"### **{tf}** – Score {score:.0f}/100")
                for cat, val in details.items():
                    st.write(f"{cat}: {val}")
                for it in items:
                    st.write(f"- {it}")
                st.markdown("---")

def calculate_entry_score(prediction, df, timeframe):
    score = 0
    comp = {}
    # ML confidence (max 30)
    conf = prediction['confidence']
    comp['ML Confidence'] = min(conf*0.3,30)
    # R/R (max 25)
    rr = abs(prediction['tp_percentage'] / prediction['sl_percentage']) if prediction['sl_percentage']!=0 else 0
    if rr>=2.0: comp['R/R']=25
    elif rr>=1.5: comp['R/R']=20
    elif rr>=1.2: comp['R/R']=15
    else: comp['R/R']=5
    # Technical (max 20)
    latest = df.iloc[-1]
    tech=0
    if latest.get('Bull',0)==1 or latest.get('Hold',0)==1: tech+=10
    if latest.get('RSI',50)>50: tech+=5
    if latest.get('EMA1',0) > latest.get('EMA2',1): tech+=5
    comp['Technical Setup'] = tech
    # Volume (max 10)
    if 'Volume' in latest and 'Volume_MA20' in latest:
        vol_ratio = latest['Volume'] / latest['Volume_MA20']
        comp['Volume'] = 10 if vol_ratio>1.2 else 7 if vol_ratio>1.0 else 3
    else:
        comp['Volume'] = 5
    # Trend (max 15)
    if timeframe=='1W':
        comp['Trend'] = 15 if latest.get('EMA1',0) > latest.get('EMA2',1) else 5
    else:
        comp['Trend'] = 10
    total = sum(comp.values())
    return total, comp

def display_entry_warnings(current_price, entry_price, prediction=None):
    warnings = []
    diff = abs(current_price - entry_price)/current_price*100
    if diff > 5:
        warnings.append(f"⚠️ **Chasing Price**: Entry is {diff:.1f}% away from current.")
    if prediction and prediction['will_hit']=='TP' and prediction['hit_prob']<40:
        warnings.append("📉 **Low Hit Probability**: TP probability below 40%")
    for w in warnings:
        st.markdown(f"- {w}")
    return len(warnings)

def increase_patience_15():
    st.session_state.patience_score = min(100, st.session_state.patience_score+15)
def increase_patience_10():
    st.session_state.patience_score = min(100, st.session_state.patience_score+10)
def update_price_and_reset_entry():
    """Callback to update current price and reset entry price when ticker changes"""
    ticker = st.session_state.get("ticker_input", DEFAULT_TICKER).upper()
    current_price = get_current_price(ticker)
    if current_price is not None:
        st.session_state.current_price = current_price
        st.session_state.entry_price = current_price
        st.session_state.entry_price_input = current_price
        st.session_state.initial_prices_set = True
        st.session_state.ticker = ticker

def initialize_session_state():
    """Initialize session state variables"""
    if "ticker" not in st.session_state:
        st.session_state.ticker = DEFAULT_TICKER
    if "current_price" not in st.session_state:
        st.session_state.current_price = None
    if "entry_price" not in st.session_state:
        st.session_state.entry_price = None
    if "entry_price_input" not in st.session_state:
        st.session_state.entry_price_input = None
    if "initial_prices_set" not in st.session_state:
        st.session_state.initial_prices_set = False
    if "patience_score" not in st.session_state:
        st.session_state.patience_score = 50
    if "last_analysis_time" not in st.session_state:
        st.session_state.last_analysis_time = None

# ---------- MAIN APP ----------
def main():
    st.title("📊 Entry Position Analyzer")
    st.write("Analyze entry using ML models trained on 4H, 1D, 1W timeframes.")
    with st.expander("Disciplined Entry Strategy", expanded=False):
        st.write(desc)
    st.markdown("### ⚙️ Automatic Technical Analysis Checklist")
    with st.expander("Technical conditions will be evaluated after analysis", expanded=False):
        st.info("System checks: Price vs EMAs, RSI, Volume, ADX, +DI/-DI, Price Action. Score 0-100.")

    # Initialize session state ONCE at the beginning
    initialize_session_state()

    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Ticker input with callback
        ticker = st.text_input(
            "Ticker Symbol", 
            value=st.session_state.ticker, 
            key="ticker_input",
            on_change=update_price_and_reset_entry
        ).upper()
        
        # If ticker changed manually without callback, handle it
        if ticker != st.session_state.ticker:
            st.session_state.ticker = ticker
            st.session_state.initial_prices_set = False
        
        # Fetch current price if not set
        if not st.session_state.initial_prices_set or st.session_state.current_price is None:
            current = get_current_price(ticker)
            if current is None:
                st.error(f"Cannot fetch price for {ticker}")
                st.stop()
            st.session_state.current_price = current
            st.session_state.entry_price = current
            st.session_state.entry_price_input = current
            st.session_state.initial_prices_set = True
        
        st.metric("Current Price", f"${st.session_state.current_price:.2f}")
    
    with col2:
        entry_price = st.number_input(
            "Entry Price (auto current)", 
            value=float(st.session_state.entry_price_input if st.session_state.entry_price_input else st.session_state.current_price), 
            step=0.1, 
            key="entry_price_input",
            on_change=lambda: st.session_state.update({"entry_price": st.session_state.entry_price_input})
        )
        st.session_state.entry_price = entry_price
    
    with col3:
        user_gain = st.number_input("Expected Gain (%)", 0.1, 15.0, 3.75, 0.1)
        user_loss = st.number_input("Expected Loss (%)", 0.1, 15.0, 3.75, 0.1)
        st.info("Wait for Technical Confirmation score before entry")

    ind = st.selectbox("Choose 3rd indicator", ['OBV', 'CCI', 'CMF', 'MFI', 'ADX'], index=0)
    days = st.slider("Forecast Days", 30, 365, 90)
    sims = st.slider("Monte Carlo Simulations", 1000, 20000, 10000)
    mc_method = st.radio("Monte Carlo Method", ["Random Statistical Simulation", "Historical Paths Simulation"], index=0)

    st.markdown("---")
    st.markdown("### 🧘 Trading Discipline")
    pat = st.session_state.patience_score
    if pat < 40:
        st.error(f"**Impulsive** ({pat}/100)")
    elif pat < 70:
        st.warning(f"**Moderate** ({pat}/100)")
    else:
        st.success(f"**Patient Trader** ({pat}/100)")
    st.progress(pat / 100)
    
    colb1, colb2 = st.columns(2)
    with colb1:
        st.button("⏸️ Wait for confirmation", on_click=increase_patience_15, use_container_width=True)
    with colb2:
        st.button("📊 Analyze first", on_click=increase_patience_10, use_container_width=True)
    
    if st.button("📊 Analyze Entry Position", use_container_width=True):
        with st.spinner("Training models and analyzing..."):
            st.session_state.patience_score = min(100, pat+5)
            end_date = datetime.now()
            all_data = {}
            results = {}
            for tf, interval in [("1W","1wk"), ("1D","1d"), ("4H","1h")]:
                st.subheader(f"{tf} ML of {ticker}")
                start = end_date - timedelta(days=365*YEARS_OF_DATA[tf])
                df = get_stock_data(ticker, start, end_date, interval)
                if df is None or len(df) < MIN_TRAIN_ROWS[tf]:
                    st.warning(f"Insufficient {tf} data")
                    continue
                if tf == "1D":
                    daily_df = df.copy()
                df = add_technical_indicators(df, tf)
                if df is None:
                    continue
                df = add_pivot_levels(df)
                df = add_pivots(df)
                df = average_pivots(df)
                df = compute_expected_return(df)
                df = compute_expected_loss(df)
                df = label_hit_prob_past(df, profit_target=user_gain/100, stop_loss=user_loss/100)
                models = train_models(df, tf)
                if models[0] is None:
                    st.warning(f"Training failed for {tf}")
                    continue
                clf, reg_ret, reg_loss, scls, sret, sloss = models
                latest = df.iloc[[-1]]
                pred = make_prediction(clf, reg_ret, reg_loss, scls, sret, sloss, latest)
                if pred:
                    assessment, reasons = assess_entry(pred, user_gain, user_loss, entry_price, pred['current_price'])
                    display_entry_warnings(pred['current_price'], entry_price, pred)
                    score, score_details = calculate_entry_score(pred, df, tf)
                    results[tf] = {'prediction':pred,'assessment':assessment,'reasons':reasons,'df':df,'score':score}
                    all_data[tf] = {'df':df,'prediction':pred,'assessment':assessment}
                    cola, colb = st.columns(2)
                    with cola:
                        st.metric("Current Price", f"${pred['current_price']:.2f}")
                        st.metric("Predicted TP", f"${pred['predicted_tp']:.2f}", delta=f"{pred['tp_percentage']:+.1f}%")
                        st.metric("Predicted SL", f"${pred['predicted_sl']:.2f}", delta=f"{pred['sl_percentage']:+.1f}%")
                    with colb:
                        st.metric("Hits", f"{pred['will_hit']} ({pred['hit_prob']:.0f}%)")
                        rr = abs(pred['tp_percentage']/pred['sl_percentage']) if pred['sl_percentage']!=0 else 0
                        st.metric("Risk/Reward", f"{rr:.1f}")
                        st.metric("ML Confidence", f"{pred['confidence']:.0f}%")
                    if assessment=="Valid": st.success(f"**Assessment**: {assessment}")
                    elif assessment=="Risky": st.warning(f"**Assessment**: {assessment}")
                    else: st.error(f"**Assessment**: {assessment}")
                    st.write(f"**Reasons**: {reasons}")
                    avg_bull, avg_bear = avg_bull_bear_lengths(df)
                    st.write(f"Avg Bull/Bear lengths: Bull {avg_bull:.0f}, Bear {avg_bear:.0f}")
                    fig = plot_analysis(ticker, df, entry_price, tf, assessment, pred, ind)
                    st.pyplot(fig)
                st.write("---")
            if all_data:
                display_technical_confirmation_metric(all_data, entry_price, ticker)
                if results:
                    st.subheader("🎯 Overall ML Recommendation")
                    valid = sum(1 for r in results.values() if r['assessment']=="Valid")
                    total = len(results)
                    rr_vals = [abs(r['prediction']['tp_percentage']/r['prediction']['sl_percentage']) if r['prediction']['sl_percentage']!=0 else 0 for r in results.values()]
                    conf_vals = [r['prediction']['confidence'] for r in results.values()]
                    avg_rr = np.mean(rr_vals) if rr_vals else 0
                    avg_conf = np.mean(conf_vals) if conf_vals else 0
                    ann = f"(Avg R/R {avg_rr:.2f}, Avg Conf {avg_conf:.1f}%)"
                    if valid == total: st.success(f"**STRONG BUY** {ann}")
                    elif valid >= total/2: st.success(f"**BUY** {ann}")
                    elif valid>=1: st.warning(f"**CAUTIOUS BUY** {ann}")
                    else: st.error(f"**AVOID** {ann}")
                    summary = []
                    for tf, r in results.items():
                        p = r['prediction']
                        rr = abs(p['tp_percentage']/p['sl_percentage']) if p['sl_percentage']!=0 else 0
                        summary.append({
                            "Timeframe": tf, "Price": round(p['current_price'],2),
                            "TP": round(p['predicted_tp'],2), "SL": round(p['predicted_sl'],2),
                            "Conf%": round(p['confidence'],1), "Hits": p['will_hit'],
                            "R/R": round(rr,2), "Score": round(r['score'],0), "Assessment": r['assessment']
                        })
                    st.dataframe(pd.DataFrame(summary))
                    # Monte Carlo (using daily data)
                    st.header("📈 Monte Carlo Simulation")
                    if 'daily_df' in locals():
                        rets = daily_df['Close'].pct_change().dropna()
                        mu = rets.mean()*252
                        sigma = rets.std()*np.sqrt(252)
                        st.metric("Annualized Return", f"{mu*100:.1f}%")
                        st.metric("Annualized Vol", f"{sigma*100:.1f}%")
                        current_price_mc = daily_df['Close'].iloc[-1]
                        @st.cache_data
                        def mc_gbm(current, mu, sigma, days, sims):
                            dt = 1/252
                            paths = np.zeros((days+1, sims))
                            paths[0] = current
                            for t in range(1, days+1):
                                paths[t] = paths[t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*np.random.standard_normal(sims))
                            return paths
                        @st.cache_data
                        def mc_bootstrap(current, returns, days, sims):
                            paths = np.zeros((days+1, sims))
                            paths[0] = current
                            for i in range(sims):
                                resampled = np.random.choice(returns.values, size=days)
                                paths[1:,i] = current * (1+resampled).cumprod()
                            return paths
                        if mc_method == "Random Statistical Simulation":
                            paths = mc_gbm(current_price_mc, mu, sigma, days, sims)
                        else:
                            paths = mc_bootstrap(current_price_mc, rets, days, sims)
                        fig3, (ax3, ax4) = plt.subplots(2,1, figsize=(12,9), height_ratios=[3,1])
                        for i in range(min(50, sims)):
                            ax3.plot(paths[:,i], color='gray', alpha=0.3, lw=0.5)
                        mean_path = paths.mean(axis=1)
                        ax3.plot(mean_path, color='red', lw=2, linestyle='--', label='Expected')
                        median = np.percentile(paths[-1], 50)
                        ax3.axhline(median, color='red', linestyle=':', label=f'Median ${median:.2f}')
                        ax3.axhline(entry_price, color='black', linestyle='-.', label=f'Entry ${entry_price:.2f}')
                        ax3.set_title("Monte Carlo Simulation")
                        ax3.legend()
                        ax3.grid(alpha=0.3)
                        ax4.hist(paths[-1], bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
                        ax4.axvline(median, color='red', linestyle='--', label=f'Median ${median:.2f}')
                        ax4.axvline(entry_price, color='black', linestyle='-.', label=f'Entry ${entry_price:.2f}')
                        ax4.set_title("Final Price Distribution")
                        ax4.legend()
                        st.pyplot(fig3)
                        prob_profit = np.mean(paths[-1] > entry_price)*100
                        st.metric("Chance of Profit", f"{prob_profit:.1f}%")
                    else:
                        st.warning("No daily data for Monte Carlo")
            else:
                st.error("No successful analyses. Try another ticker or time period.")

if __name__ == "__main__":
    main()
