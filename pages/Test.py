#!/usr/bin/env python

coding: utf-8

""" Enhanced Entry Position Analyzer Includes improvements:

TP-before-SL classifier priority

ADX + ATR volatility filters

Dynamic RSI thresholds by trend

2-of-3 timeframe consensus (4H, 1D, 1W)

Required R:R >= 1.5 for confirmed entries

Blended confidence score (classifier, trend, volatility)

Entry timing delay rules (avoid buying into local tops)


NOTE: This file depends on your existing imports module which must provide ta (technical functions) and any helper imports you had. It also uses yfinance and scikit-learn. Adapt paths as needed. """

from imports import * import streamlit as st import pandas as pd import numpy as np import yfinance as yf from datetime import datetime, timedelta import matplotlib.pyplot as plt import matplotlib.dates as mdates from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor from sklearn.preprocessing import StandardScaler from sklearn.model_selection import train_test_split import warnings warnings.filterwarnings('ignore')

--- Configuration ---

st.set_page_config(page_title="Entry Position Analyzer (Enhanced)", layout="wide")

YEARS_OF_DATA = {'4H': 1, '1D': 2, '1W': 5} MIN_TRAIN_ROWS = {'4H': 150, '1D': 120, '1W': 40} _DAYS = 21 windows = [3,5,7,9,11,13,15,17,19,21] EXPECTED_CLASSES = [0,1,2,3,4] label2str = {0:'None',1:'SL',2:'TP',3:'Hold',4:'Short'}

FEATURES minimal subset (extend if you have more)

FEATURES = [ 'Close','High','Low','Volume', 'SMA1','SMA2','SMA3','SMA_Ratio', 'RSI','RSI_SMA','ATR','Volatility', 'ADX','+DI','-DI', 'Upper_Band','Lower_Band','Volume_MA20', 'sumBuyVol','sumSellVol','vSpike','VPT','OBV','MFI','VWMA','CMF', 'SMIIO','SMIIO_Signal','SMIIO_Osc','MACD','Signal_Line', 'CCI','KCu','KCl','Kasym','Kcount','STu','STl' ]

--- Utilities ---

def get_current_price(ticker): try: t = yf.Ticker(ticker) data = t.history(period='1d', progress=False, auto_adjust=True) return float(data['Close'][-1]) except Exception: return None

def get_stock_data(ticker, start_date, end_date, interval='1d'): try: interval_map = {'4H':'4h','1D':'1d','1W':'1wk'} yf_interval = interval_map.get(interval, interval) df = yf.download(ticker, start=start_date, end=end_date, interval=yf_interval, progress=False, auto_adjust=True) if df.empty: return None df = df.reset_index() if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date']) df.set_index('Date', inplace=True) df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns] req = ['Open','High','Low','Close','Volume'] if not all(c in df.columns for c in req): return None df = df[req].dropna() return df except Exception: return None

--- Indicator calculation (wraps your ta module) ---

def add_technical_indicators(df, timeframe='1D'): # Keep original close copy close = df['Close'].copy() df['Close'] = df[['Open','High','Low','Close']].mean(axis=1).rolling(3,min_periods=1).mean()

if timeframe == '1W':
    sma_multiplier = 1
    atr_period = 7
    rsi_period = 9
elif timeframe == '4H':
    sma_multiplier = 3
    atr_period = 50
    rsi_period = 50
else:
    sma_multiplier = 3
    atr_period = 14
    rsi_period = 14

df['SMA1'] = df['Close'].ewm(span=int(_DAYS*0.5*sma_multiplier),adjust=False).mean()
df['SMA2'] = df['Close'].ewm(span=_DAYS*sma_multiplier,adjust=False).mean()
df['SMA3'] = df['Close'].ewm(span=int(_DAYS*2*sma_multiplier),adjust=False).mean()
df['SMA_Ratio'] = df['SMA1']/df['SMA2']

# TA helpers - assume ta namespace available
df['ATR'] = ta.calculate_atr(high=df.High, low=df.Low, close=df.Close)
df = ta.scaled_volatility(df)
df = ta.add_candlestickpatterns(df)
df['RSI'] = ta.calculate_rsi(df)
df['RSI_SMA'] = df['RSI'].rolling(14).mean()

ema_short = 9 if timeframe=='1W' else 12
ema_long = 22 if timeframe=='1W' else 26
ema_s = df['Close'].ewm(span=ema_short,adjust=False).mean()
ema_l = df['Close'].ewm(span=ema_long,adjust=False).mean()
df['MACD'] = ema_s - ema_l
df['Signal_Line'] = df['MACD'].ewm(span=9,adjust=False).mean()

df['SMIIO'],df['SMIIO_Signal'],df['SMIIO_Osc'] = ta.calculate_smiio(df)
df['Upper_Band'] = df['SMA1'] + (2*df['Close'].rolling(20).std())
df['Lower_Band'] = df['SMA1'] - (2*df['Close'].rolling(20).std())
df['Volume_MA20'] = df['Volume'].rolling(20).mean()

df['buy_volume'] = (df.Close > df.Close.shift(1)) * df['Volume']
df['sell_volume'] = (df.Close < df.Close.shift(1)) * df['Volume']
df['sumBuyVol'] = df['buy_volume'].rolling(9).sum()
df['sumSellVol'] = df['sell_volume'].rolling(9).sum()
df['vSpike'] = np.where(df['Volume'] > 2*df['Volume_MA20'], np.where(df['Close']>df['Open'],1,-1),0)
df['VPT'] = df['Volume'].mul((df['Close'] - df['Close'].shift(1))/df['Close'].shift(1)).cumsum()
df['MFI'] = ta.calculate_mfi(df)
df['CMF'] = ta.chaikin_money_flow(df, window=20)
df['CCI'] = ta.calculate_cci(df)
df['OBV'] = ta.calculate_obv(df)
dmi = ta.calculate_dmi(df, n=14)
if isinstance(dmi, pd.DataFrame):
    df['+DI'] = dmi['+DI']
    df['-DI'] = dmi['-DI']
    df['ADX'] = dmi['ADX']
df['VWMA'] = ta.calculate_vwma(df)
kelt = ta.calculate_keltner(df)
if isinstance(kelt, pd.DataFrame):
    for c in ['KCu','KCl','Kasym','Kcount','KCu_outer','KCl_outer']:
        if c in kelt.columns:
            df[c] = kelt[c]
vx = ta.calculate_vortex(df)
if isinstance(vx, pd.DataFrame):
    if 'VI+' in vx.columns:
        df['VI+'] = vx['VI+']
        df['VI-'] = vx['VI-']
stf = ta.calculate_supertrend(df)
if isinstance(stf, pd.DataFrame):
    if 'STu' in stf.columns:
        df['STu'] = stf['STu']
        df['STl'] = stf['STl']

df['Volatility'] = df['Close'].rolling(14).std().rolling(3).mean()
df[ ['SMA1','SMA2','RSI','-DI','Close'] ] = df[ ['SMA1','SMA2','RSI','-DI','Close'] ].fillna(method='ffill').fillna(method='bfill')

# TI label similar to earlier - simplified
rsi_lower = 25 if timeframe=='1W' else 18
rsi_upper = 60 if timeframe=='1W' else 55
conditions = [
    ((df['SMA1']>df['SMA2']) & (df['RSI']>=df['RSI_SMA']) & (df['RSI'].between(52,95)) & (df['+DI']>df['-DI']) & (df['+DI'].between(18,55)) & (df['Close']>df['SMA1'])),
    ((df['SMA1']<df['SMA2']) & (df['RSI'].between(rsi_lower,60)) & (df['RSI']<df['RSI_SMA']) & (df['+DI']<df['-DI']) & (df['-DI'].between(18,55))),
    ((df['SMA1']<df['SMA2']) & (df['RSI'].between(25,50)) & (df['-DI'].between(30,55)) & (df['Close']>df['SMA1'])),
    (((df['SMA1']>df['SMA2']) & (df['RSI']>=50)) | ((df['RSI']<df['RSI_SMA']) & (df['ADX'].between(40,75))))
]
choices = ['Bull','Bear','Short','Hold']
df['TI'] = np.select(conditions, choices, default='Neutral')
df['TI'] = df['TI'].astype('category')
df = pd.concat([df, pd.get_dummies(df['TI'])], axis=1)

# Strong bull/bear
df['StrongBull'] = (((df['RSI']>52) & (df['ADX']>22) & (df['+DI']>df['-DI']) & (df['sumBuyVol']>df['sumSellVol']))).astype(int)
df['StrongBear'] = (((df['RSI']<40) & (df['ADX']>22) & (df['+DI']<df['-DI']) & (df['sumBuyVol']<df['sumSellVol']))).astype(int)
df['sNeutral'] = ((df['StrongBull']==0) & (df['StrongBear']==0)).astype(int)

df['gapStrength'] = ta.compute_gapStrength(df)
df = ta.add_exhaustion_indicator(df)
df['Close'] = close
return df

--- Pivot helpers (kept simple) ---

def add_pivots(df, win=windows): for w in win: roll_high = df['High'].rolling(w) roll_low = df['Low'].rolling(w) roll_close = df['Close'].rolling(w) PP = (roll_high.max() + roll_low.min() + roll_close.apply(lambda x: x[-1])).div(3) df[f'PP_{w}'] = PP df[f'R1_{w}'] = 2PP - roll_low.min() df[f'S1_{w}'] = 2PP - roll_high.max() df[f'R2_{w}'] = PP + (roll_high.max() - roll_low.min()) df[f'S2_{w}'] = PP - (roll_high.max() - roll_low.min()) return df

def average_pivots(df, windows=[5,10,14,20]): for level in ['PP','R1','S1','R2','S2']: cols = [f'{level}{w}' for w in windows if f'{level}{w}' in df.columns] if cols: df[f'{level}_Avg'] = df[cols].mean(axis=1) return df

--- Expected return/loss computation (same idea) ---

def compute_expected_return(df, forward_window=14, r_cols=['R1_Avg','R2_Avg']): df['Expected_Return'] = np.nan close = df['Close'].values for i in range(len(df)-forward_window): cur = close[i] future = close[i+1:i+1+forward_window] pivots = [] for c in r_cols: if c in df.columns and not np.isnan(df[c].iloc[i]): pivots.append(df[c].iloc[i]) target = max(pivots) if pivots else np.nan if not np.isnan(target): # first hit hit = False for p in future: if p >= target: df.at[df.index[i],'Expected_Return'] = (target-cur)/cur hit = True break if not hit and future.size>0: df.at[df.index[i],'Expected_Return'] = (np.nanmax(future)-cur)/cur else: if future.size>0: df.at[df.index[i],'Expected_Return'] = (np.nanmax(future)-cur)/cur return df

def compute_expected_loss(df, forward_window=14, s_cols=['S1_Avg','S2_Avg']): df['Expected_Loss'] = np.nan close = df['Close'].values for i in range(len(df)-forward_window): cur = close[i] future = close[i+1:i+1+forward_window] pivots = [] for c in s_cols: if c in df.columns and not np.isnan(df[c].iloc[i]): pivots.append(df[c].iloc[i]) target = min(pivots) if pivots else np.nan if not np.isnan(target): hit = False for p in future: if p <= target: df.at[df.index[i],'Expected_Loss'] = (target-cur)/cur hit = True break if not hit and future.size>0: df.at[df.index[i],'Expected_Loss'] = (np.nanmin(future)-cur)/cur else: if future.size>0: df.at[df.index[i],'Expected_Loss'] = (np.nanmin(future)-cur)/cur return df

--- Labeling: TP before SL ---

def label_hit_prob_past(df, window=14, profit_target=0.05, stop_loss=0.05): close = df['Close'].values N = len(close) labels = np.zeros(N, dtype=int) for i in range(N): if i >= N-window: labels[i] = 0 continue cur = close[i] tp = cur*(1+profit_target) sl = cur*(1-stop_loss) future = close[i+1:i+1+window] tp_hit = None sl_hit = None for j,p in enumerate(future): if tp_hit is None and p>=tp: tp_hit = j if sl_hit is None and p<=sl: sl_hit = j if tp_hit is not None and sl_hit is not None: break if tp_hit is not None and (sl_hit is None or tp_hit < sl_hit): labels[i] = 2 elif sl_hit is not None and (tp_hit is None or sl_hit < tp_hit): labels[i] = 1 else: labels[i] = 0 df['Hit_Label'] = labels return df

--- Training models (kept similar but ensure classifier is TP-before-SL primary) ---

def train_models(df, timeframe): required = [c for c in FEATURES if c in df.columns] + ['Hit_Label','Expected_Return','Expected_Loss'] df_model = df.dropna(subset=required) if len(df_model) < MIN_TRAIN_ROWS.get(timeframe, 50): return None,None,None,None,None,None

X = df_model[[c for c in FEATURES if c in df_model.columns]]
y_cls = df_model['Hit_Label'].astype(int)

scaler_cls = StandardScaler()
Xs = scaler_cls.fit_transform(X)

clf = RandomForestClassifier(n_estimators=80, max_depth=10, random_state=42, n_jobs=-1)
clf.fit(Xs,y_cls)

# Get probs for adding to regressors
probs = clf.predict_proba(Xs)
prob_df = pd.DataFrame(0, index=df_model.index, columns=[f'Prob_Class_{c}' for c in EXPECTED_CLASSES])
for i,c in enumerate(clf.classes_):
    if c in EXPECTED_CLASSES:
        prob_df[f'Prob_Class_{c}'] = probs[:,i]

X_reg = pd.concat([X.reset_index(drop=True), prob_df.reset_index(drop=True)], axis=1)
X_reg_cols = X_reg.columns.tolist()

scaler_ret = StandardScaler()
Xr = scaler_ret.fit_transform(X_reg)
y_ret = df_model['Expected_Return'].values
model_ret = RandomForestRegressor(n_estimators=80, max_depth=10, random_state=42, n_jobs=-1)
model_ret.fit(Xr, y_ret)

scaler_loss = StandardScaler()
Xl = scaler_loss.fit_transform(X_reg)
y_loss = df_model['Expected_Loss'].values
model_loss = RandomForestRegressor(n_estimators=80, max_depth=10, random_state=42, n_jobs=-1)
model_loss.fit(Xl, y_loss)

return clf, model_ret, model_loss, scaler_cls, scaler_ret, scaler_loss

--- Blended confidence & decision logic ---

def make_prediction(clf, model_ret, model_loss, scaler_cls, scaler_ret, scaler_loss, latest_row): # latest_row: single-row DataFrame # Ensure required features present feats = [c for c in FEATURES if c in latest_row.columns] if any(latest_row[feats].isnull().any()): return None Xlatest = scaler_cls.transform(latest_row[feats]) probs = clf.predict_proba(Xlatest)[0] # build prob dict prob_map = {c:0.0 for c in EXPECTED_CLASSES} for i,c in enumerate(clf.classes_): prob_map[c] = probs[i]

prob_tp = prob_map.get(2,0.0)
prob_sl = prob_map.get(1,0.0)

# Prepare features for regressors
prob_features = {f'Prob_Class_{c}':prob_map[c] for c in EXPECTED_CLASSES}
Xreg = pd.concat([latest_row[feats].reset_index(drop=True), pd.DataFrame([prob_features])], axis=1)

Xr_scaled = scaler_ret.transform(Xreg)
Xl_scaled = scaler_loss.transform(Xreg)

pred_return = model_ret.predict(Xr_scaled)[0]
pred_loss = model_loss.predict(Xl_scaled)[0]

cur_price = float(latest_row['Close'].values[0])
predicted_tp = cur_price * (1 + pred_return)
predicted_sl = cur_price * (1 + pred_loss)

tp_pct = (predicted_tp - cur_price)/cur_price*100
sl_pct = (predicted_sl - cur_price)/cur_price*100

# Trend score
trend_score = 1.0 if (latest_row['SMA1'].values[0] > latest_row['SMA2'].values[0] and latest_row['Close'].values[0] > latest_row['SMA1'].values[0]) else 0.0
# Volatility score - lower ATR% => higher score
atr_pct = latest_row['ATR'].values[0] / latest_row['Close'].values[0] * 100
vol_score = max(0.0, 1.0 - (atr_pct/10.0))

# Blended confidence
blended = 0.6*prob_tp + 0.2*trend_score + 0.2*vol_score
blended_pct = blended * 100

# Dynamic RSI thresholds
rsi = latest_row['RSI'].values[0]
uptrend = trend_score==1.0
rsi_ok = False
if uptrend:
    rsi_ok = (rsi < 65)
else:
    rsi_ok = (rsi < 55)

# Entry timing rules
avoid_buy = False
if rsi > 70:
    avoid_buy = True
if latest_row['Close'].values[0] > latest_row['Upper_Band'].values[0]:
    avoid_buy = True
if latest_row['Close'].values[0] > latest_row['SMA1'].values[0] * 1.02:
    avoid_buy = True

# Risk:Reward
rr = None
if predicted_sl is not None and predicted_tp is not None and predicted_sl != cur_price:
    rr = (predicted_tp - cur_price) / (cur_price - predicted_sl)

# Primary decision logic prioritizing TP-before-SL classifier
will_hit = 'None'
if prob_tp > 0.65 and blended > 0.45 and rsi_ok and not avoid_buy and rr is not None and rr >= 1.5:
    will_hit = 'TP'
    decision = 'Strong Buy'
elif prob_tp > 0.5 and blended > 0.4 and rr is not None and rr >= 1.2 and not avoid_buy:
    will_hit = 'TP'
    decision = 'Buy (Cautious)'
elif prob_sl > 0.6:
    will_hit = 'SL'
    decision = 'Avoid / Short bias'
else:
    will_hit = label2str.get(int(clf.predict(Xlatest)[0]), 'None')
    decision = 'Hold / Wait'

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
    'cur_price': cur_price
}

--- Assess multi-timeframe consensus ---

def aggregate_timeframes(results): # results: dict of timeframe->prediction dict votes = {'Strong Buy':0,'Buy (Cautious)':0,'Avoid / Short bias':0,'Hold / Wait':0} for tf,p in results.items(): lab = p.get('decision_label') votes[lab] = votes.get(lab,0)+1 # Simple rule: if >=2 strong buy/buy then BUY buy_votes = votes['Strong Buy'] + votes['Buy (Cautious)'] short_votes = votes['Avoid / Short bias'] if buy_votes >= 2: overall = 'BUY' elif short_votes >= 2: overall = 'SHORT/AVOID' else: overall = 'HOLD/WAIT' return overall, votes

--- Plotting (kept compact) ---

def plot_analysis(ticker, df, entry_price, timeframe, prediction, assessment): fig, (ax1, ax2) = plt.subplots(2,1,figsize=(12,8),sharex=True, gridspec_kw={'height_ratios':[3,1]}) ax1.plot(df.index, df['Close'], label='Close') if 'SMA1' in df.columns: ax1.plot(df.index, df['SMA1'], label='SMA1') if 'SMA2' in df.columns: ax1.plot(df.index, df['SMA2'], label='SMA2') last = df.iloc[-1] ax1.scatter(df.index[-1], entry_price, marker='o', label='Entry') if prediction: ax1.axhline(y=prediction['predicted_tp'], color='blue', linestyle='--', alpha=0.6) ax1.axhline(y=prediction['predicted_sl'], color='red', linestyle='--', alpha=0.6) ax1.legend() if 'RSI' in df.columns: ax2.plot(df.index, df['RSI'], label='RSI') ax2.axhline(70, linestyle='--') ax2.axhline(30, linestyle='--') plt.suptitle(f"{ticker} {timeframe} - {assessment}") plt.tight_layout() return fig

--- Streamlit UI ---

def main(): st.title("📊 Entry Position Analyzer — Enhanced") col1,col2,col3 = st.columns(3) with col1: ticker = st.text_input('Ticker', value='TSLA').upper() with col2: entry_price = st.number_input('Entry Price ($)', min_value=0.0, value=0.0, step=0.1) with col3: user_gain = st.number_input('Expected Gain (%)', min_value=0.1, max_value=50.0, value=5.0) user_loss = st.number_input('Expected Loss (%)', min_value=0.1, max_value=50.0, value=4.5)

if st.button('Analyze'):
    end_date = datetime.now()
    results = {}
    timeframes = [('4H','4H'),('1D','1D'),('1W','1W')]
    for timeframe, interval in timeframes:
        st.subheader(f"{timeframe} analysis for {ticker}")
        years = YEARS_OF_DATA[timeframe]
        start_date = end_date - timedelta(days=365*years)
        df = get_stock_data(ticker, start_date, end_date, interval)
        if df is None or len(df) < 10:
            st.warning(f"Insufficient data for {timeframe}")
            continue
        df = add_technical_indicators(df, timeframe)
        df = add_pivots(df, windows)
        df = average_pivots(df)
        df = compute_expected_return(df, forward_window=14)
        df = compute_expected_loss(df, forward_window=14)
        df = label_hit_prob_past(df, window=14, profit_target=user_gain/100.0, stop_loss=user_loss/100.0)

        models = train_models(df, timeframe)
        if models[0] is None:
            st.warning(f"Could not train models for {timeframe}")
            continue
        clf, model_ret, model_loss, scaler_cls, scaler_ret, scaler_loss = models
        latest = df.iloc[[-1]]
        pred = make_prediction(clf, model_ret, model_loss, scaler_cls, scaler_ret, scaler_loss, latest)
        if pred is None:
            st.warning('Prediction failed')
            continue
        assessment = pred['decision_label']
        results[timeframe] = pred

        # Display metrics
        c1,c2,c3 = st.columns(3)
        c1.metric('Current Price', f"${pred['cur_price']:.2f}")
        c2.metric('Predicted TP', f"${pred['predicted_tp']:.2f}", delta=f"{pred['tp_pct']:.2f}%")
        c3.metric('Predicted SL', f"${pred['predicted_sl']:.2f}", delta=f"{pred['sl_pct']:.2f}%")
        st.metric('Decision', assessment)
        st.text(f"Blended confidence: {pred['blended_confidence_pct']:.1f}% | TP prob: {pred['prob_tp']:.2f}")

        fig = plot_analysis(ticker, df, entry_price if entry_price>0 else pred['cur_price'], timeframe, pred, assessment)
        st.pyplot(fig)

    if results:
        overall, votes = aggregate_timeframes(results)
        st.header('Overall Recommendation')
        if overall == 'BUY':
            st.success('✅ BUY — multiple timeframes agree')
        elif overall == 'SHORT/AVOID':
            st.error('⚠️ SHORT / AVOID — multi-timeframe bearish')
        else:
            st.info('⏸ HOLD / WAIT — no clear multi-timeframe conviction')
        st.write('Vote breakdown:', votes)
    else:
        st.error('No successful analyses — try a different ticker or timeframe')

if name == 'main': main()
