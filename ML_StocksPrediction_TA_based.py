# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.offsetbox import AnchoredText
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import altair as alt
import warnings
import re
import sys
import emoji
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Import your fixed ta_functions
import ta_functions as ta

# Suppress warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(layout="wide", page_title="📈 MAIN - Machine Learning of Stocks")
st.caption("Data sourced via Yahoo Finance • Updated dynamically")

# Display version info
st.write("### Version Information")
st.write(f"Python: {sys.version}")
st.write(f"Pandas version: {pd.__version__}")
st.write(f"Numpy version: {np.__version__}")

# Constants
bold = '\033[1m'
end = '\033[0m'
today = datetime.now().strftime('%Y-%m-%d')

_Nr = 50
YEARS_OF_DATA = 3
end_date = datetime.now()
start_date = end_date - timedelta(days=365 * YEARS_OF_DATA)
PROFIT_TARGET = 0.04
STOP_LOSS = 0.0375
_DAYS = 22
_FWDAYS = 14
windows = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
_window = 9
_tolerance = 1.07
_FIBS = False
_FibLen = 20
_ms = 5

FEATURES = [
    'High', 'Low', 'RSI', 'RSI_SMA', 'CCI', '+DI', '-DI', 'ADX', 'ATR', 
    'VI+', 'KCu', 'KCl', 'KCu_outer', 'KCl_outer', 'Kasym', 'Kcount', 'STu', 'STl',
    'EMA1', 'EMA2', 'EMA3', 'EMA_Ratio', 'Upper_Band', 'Lower_Band', 'Volume_MA20', 
    'SMIIO', 'SMIIO_Signal', 'SMIIO_Osc', 'MACD', 'Signal_Line',
    'return1', 'return2', 'return3', 'Volatility', 'Scaled_Volatility', 'DD',
    'sumBuyVol', 'sumSellVol', 'vSpike', 'VPT', 'OBV', 'MFI', 'VWMA', 'CMF',
    'Candlesticks', 'gapStrength',
    'Bear', 'Bull', 'Short', 'Hold', 'Neutral', 'StrongBull', 'StrongBear', 'Exhaustion',
    'PP_Avg', 'R1_Avg', 'R2_Avg', 'S1_Avg', 'S2_Avg'
]

EMOJI = {
    "STRONG BUY": "💎", 
    "Buy": "🟢", 
    "Wait": "⏳", 
    "Short/AVOID": "🚫", 
    "Short the RISE": "🔻", 
    "RISKY BUY": "⚠️", 
    "Monitor": "🔍", 
    "Watch": "👀"
}

# ============================================
# HELPER FUNCTIONS
# ============================================

def label(text):
    return f"{EMOJI.get(text, '')} {text}"

def strip_ansi_codes(text):
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

def extract_emojis(text):
    return ''.join(c for c in text if c in emoji.EMOJI_DATA)

def safe_format_float(val, fmt="{:7.2f}", na_str="N/A"):
    try:
        return fmt.format(float(val))
    except (ValueError, TypeError):
        return na_str

def optimize_dataframe(df):
    df = df.copy()
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

def get_stock_data(ticker, start_date, end_date):
    """Fetch stock data with proper error handling"""
    try:
        df = yf.download(
            ticker, 
            start=start_date, 
            end=end_date + timedelta(days=1),
            progress=False,
            auto_adjust=True, 
            actions=False
        )
        
        if df.empty:
            return None
            
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
            
        df = df.dropna()
        return optimize_dataframe(df)
        
    except Exception as e:
        st.warning(f"Error fetching {ticker}: {e}")
        return None

def is_valid_ticker(ticker):
    try:
        df = yf.Ticker(ticker).history(period="1d")
        return not df.empty
    except Exception:
        return False

# ============================================
# TECHNICAL INDICATORS (Using ta_functions)
# ============================================

def add_technical_indicators(df):
    """Add all technical indicators using ta_functions module"""
    try:
        df = df.copy()
        close = df['Close']
        
        # Smooth price
        df['Close_smooth'] = df[['Open', 'High', 'Low', 'Close']].mean(axis=1).rolling(2).mean()
        
        # Moving Averages
        df['EMA1'] = df['Close_smooth'].ewm(span=int(_DAYS * 0.5), adjust=False).mean()
        df['EMA2'] = df['Close_smooth'].ewm(span=_DAYS, adjust=False).mean()
        df['EMA3'] = df['Close_smooth'].ewm(span=int(_DAYS * 2), adjust=False).mean()
        df['EMA_Ratio'] = df['EMA1'] / df['EMA2']
        
        # Volatility
        df['ATR'] = ta.calculate_atr(df['High'], df['Low'], df['Close'])
        df = ta.scaled_volatility(df)
        
        # Patterns
        df = ta.add_candlestickpatterns(df)
        
        # RSI
        df['RSI'] = ta.calculate_rsi(df)
        df['RSI_SMA'] = df['RSI'].rolling(14).mean()
        
        # MACD
        ema12 = df['Close_smooth'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close_smooth'].ewm(span=24, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # SMIIO
        smiio_result = ta.calculate_smiio(df)
        df['SMIIO'] = smiio_result[0]
        df['SMIIO_Signal'] = smiio_result[1]
        df['SMIIO_Osc'] = smiio_result[2]
        
        # Bands
        df['Upper_Band'] = df['EMA1'] + (2 * df['Close_smooth'].rolling(20).std())
        df['Lower_Band'] = df['EMA1'] - (2 * df['Close_smooth'].rolling(20).std())
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        
        # Volume features
        df['buy_volume'] = (df['Close'] > df['Close'].shift(1)) * df['Volume']
        df['sell_volume'] = (df['Close'] < df['Close'].shift(1)) * df['Volume']
        df['sumBuyVol'] = df['buy_volume'].rolling(window=9).sum()
        df['sumSellVol'] = df['sell_volume'].rolling(window=9).sum()
        
        # Volume spikes
        df['vSpike'] = np.where(
            df['Volume'] > 2 * df['Volume_MA20'], 
            np.where(df['Close'] > df['Open'], 1, -1), 
            0
        )
        
        # VPT
        df['VPT'] = df['Volume'].mul((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)).cumsum()
        
        # Money Flow Index
        df['MFI'] = ta.calculate_mfi(df)
        
        # Chaikin Money Flow
        df['CMF'] = ta.chaikin_money_flow(df, window=20)
        
        # CCI
        df['CCI'] = ta.calculate_cci(df)
        
        # OBV
        df['OBV'] = ta.calculate_obv(df)
        
        # DMI
        dmi_result = ta.calculate_dmi(df, n=14)
        df['+DI'] = dmi_result['+DI'].rolling(3).mean()
        df['-DI'] = dmi_result['-DI'].rolling(3).mean()
        df['ADX'] = dmi_result['ADX'].rolling(3).mean()
        
        # VWMA
        df['VWMA'] = ta.calculate_vwma(df)
        
        # Keltner Channels
        keltner_result = ta.calculate_keltner(df)
        for col in ['KCm', 'KCu', 'KCl', 'KCu_outer', 'KCl_outer', 'Kasym', 'Kcount']:
            if col in keltner_result.columns:
                df[col] = keltner_result[col].rolling(3).mean()
            else:
                df[col] = 0
        
        # Vortex
        vortex_result = ta.calculate_vortex(df)
        df['VI+'] = vortex_result['VI+']
        df['VI-'] = vortex_result['VI-']
        
        # Supertrend
        supertrend_result = ta.calculate_supertrend(df)
        df['STu'] = supertrend_result['STu']
        df['STl'] = supertrend_result['STl']
        
        # Drawdown
        df['DD'] = df['Close'] - df['Close'].rolling(14).max()
        df['DD'] = df['DD'].shift(1)
        
        # Returns
        df['return1'] = df['Close'].pct_change(7).rolling(3).mean()
        df['return2'] = df['Close'].pct_change(14).rolling(3).mean()
        df['return3'] = df['Close'].pct_change(21).rolling(3).mean()
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(14).std().rolling(3).mean()
        
        # Fill NaN values using ffill/bfill (pandas 2.x compatible)
        cols_to_fill = ['EMA1', 'EMA2', 'RSI', '-DI', 'Close', 'ADX', '+DI']
        for col in cols_to_fill:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()
        
        # Generate TI signals
        conditions = [
            (
                (df['Close'] > df['EMA2']) &
                (df['EMA1'] > df['EMA2']) &
                (df['RSI'].between(50, 90)) &
                (df['ADX'] > 40) &
                (df['+DI'] > df['-DI']) &
                (df['Close'] > df['Close'].shift(5) * 1.02)
            ),
            (
                ((df['EMA1'] >= df['EMA2']) & (df['RSI'] >= df['RSI_SMA']) &
                 (df['RSI'].between(52, 95)) & ((df['ADX'] > 24) & (df['+DI'] > df['-DI']))) |
                (((df['RSI'] >= df['RSI_SMA']) & (df['RSI'] > 50)) & 
                 ((df['ADX'] > 18) & (df['+DI'] > df['-DI'])))
            ),
            (
                (df['Close'] <= df['EMA1']) & (df['EMA1'] < df['EMA2']) &
                (df['RSI'].between(50, 85)) & (df['ADX'] > 24) & (df['+DI'] < df['-DI'])
            ),
            (
                ((df['EMA1'] < df['EMA2']) & (df['RSI'].between(18, 60)) &
                 (df['RSI'] < df['RSI_SMA']) & ((df['ADX'] > 18) & (df['+DI'] < df['-DI']))) |
                ((df['RSI'] < df['RSI_SMA']) & (df['RSI'].between(20, 60)) &
                 ((df['ADX'] > 18) & (df['+DI'] < df['-DI']))) |
                ((df['RSI'] > df['RSI_SMA']) & (df['RSI_SMA'] < 37))
            )
        ]
        
        choices = ['Hold', 'Bull', 'Short', 'Bear']
        df['TI'] = np.select(conditions, choices, default='Neutral')
        df['TI'] = df['TI'].astype('category')
        
        # One-hot encode TI
        df_encoded = pd.get_dummies(df['TI'], prefix='', prefix_sep='')
        expected_cols = ['Hold', 'Bull', 'Short', 'Bear', 'Neutral']
        for col in expected_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df = pd.concat([df, df_encoded], axis=1)
        
        # Strong signals
        df['StrongBull'] = ((df['RSI'] > 52) & (df['ADX'] > 22) & 
                           (df['+DI'] > df['-DI']) & (df['sumBuyVol'] > df['sumSellVol'])).astype(int)
        df['StrongBear'] = ((df['RSI'] < 40) & (df['ADX'] > 22) & 
                           (df['+DI'] < df['-DI']) & (df['sumBuyVol'] < df['sumSellVol'])).astype(int)
        df['sNeutral'] = ((df['StrongBull'] == 0) & (df['StrongBear'] == 0)).astype(int)
        
        # Gap strength and exhaustion
        df['gapStrength'] = ta.compute_gapStrength(df)
        df = ta.add_exhaustion_indicator(df)
        
        # Restore original close
        df['Close'] = close
        
        return df
        
    except Exception as e:
        st.error(f"Error adding technical indicators: {e}")
        raise

# ============================================
# PIVOT FUNCTIONS
# ============================================

def add_pivot_levels(df, window=_DAYS):
    high = df['High'].rolling(window)
    low = df['Low'].rolling(window)
    close = df['Close'].rolling(window)
    PP = (high.max() + low.min() + close.apply(lambda x: x[-1])).div(3)
    R1 = 2 * PP - low.min()
    S1 = 2 * PP - high.max()
    R2 = PP + (high.max() - low.min())
    S2 = PP - (high.max() - low.min())
    df['PP'] = PP.bfill()
    df['R1'] = R1.bfill()
    df['S1'] = S1.bfill()
    df['R2'] = R2.bfill()
    df['S2'] = S2.bfill()
    return df

def add_pivots(df, win=windows):
    for w in win:
        roll_high = df['High'].rolling(w)
        roll_low = df['Low'].rolling(w)
        roll_close = df['Close'].rolling(w)
        PP = (roll_high.max() + roll_low.min() + roll_close.apply(lambda x: x[-1])).div(3)
        R1 = 2 * PP - roll_low.min()
        S1 = 2 * PP - roll_high.max()
        R2 = PP + (roll_high.max() - roll_low.min())
        S2 = PP - (roll_high.max() - roll_low.min())
        df[f"PP_{w}"] = PP
        df[f"R1_{w}"] = R1
        df[f"S1_{w}"] = S1
        df[f"R2_{w}"] = R2
        df[f"S2_{w}"] = S2
    return df

def average_pivots(df, windows=[5, 10, 14, 20]):
    for level in ['PP', 'R1', 'S1', 'R2', 'S2']:
        cols = [f"{level}_{w}" for w in windows if f"{level}_{w}" in df.columns]
        if cols:
            df[f"{level}_Avg"] = df[cols].mean(axis=1)
    return df

def compute_expected_return(df, forward_window=14, r_cols=['R1', 'R2']):
    df = df.copy()
    df['Expected_Return'] = np.nan
    close_prices = df['Close'].values
    
    for i in range(len(df) - forward_window):
        current_price = close_prices[i]
        future_prices = close_prices[i+1:i+1+forward_window]
        if len(future_prices) > 0:
            max_future = np.nanmax(future_prices)
            df.iloc[i, df.columns.get_loc('Expected_Return')] = (max_future - current_price) / current_price
    return df

def compute_expected_loss(df, forward_window=14, s_cols=['S1', 'S2']):
    df = df.copy()
    df['Expected_Loss'] = np.nan
    close_prices = df['Close'].values
    
    for i in range(len(df) - forward_window):
        current_price = close_prices[i]
        future_prices = close_prices[i+1:i+1+forward_window]
        if len(future_prices) > 0:
            min_future = np.nanmin(future_prices)
            df.iloc[i, df.columns.get_loc('Expected_Loss')] = (min_future - current_price) / current_price
    return df

# ============================================
# LABELING FUNCTIONS
# ============================================

def label_hit_prob_past(df, window=14, profit_target=0.05, stop_loss=0.05, 
                        lookback=60, tp_thresh=0.35, sl_thresh=0.4):
    df = df.copy()
    close_prices = df['Close'].values
    bull = (df['TI'] == 'Bull')
    bear = (df['TI'] == 'Bear')
    
    N = len(close_prices)
    labels = []
    
    for i in range(N):
        current_price = close_prices[i]
        tp = current_price * (1 + profit_target)
        sl = current_price * (1 - stop_loss)
        future_prices = close_prices[i+1:min(i+1+window, N)]
        
        tp_hit = any(p >= tp for p in future_prices)
        sl_hit = any(p <= sl for p in future_prices)
        
        if tp_hit and bull[i]:
            labels.append(2)  # TP
        elif sl_hit and bear[i]:
            labels.append(1)  # SL
        else:
            labels.append(0)  # None
    
    df['Hit_Label'] = labels
    return df

# ============================================
# ACTION FUNCTIONS
# ============================================

def get_action_label(confidence, will_hit_raw, current_price, ema1, rsi, ti_signal, predicted_return, predicted_loss):
    if will_hit_raw is None or str(will_hit_raw).lower() == "nan":
        base = "None"
    else:
        base = str(will_hit_raw).split()[0]
    
    c = float(confidence)
    
    if base in ("SL", "Short") and c < 42:
        return label("Short/AVOID")
    
    ema_proximity = 0.95 <= current_price / ema1 <= 1.05
    good_signal = base in ("None", "TP", "Hold")
    strong_trend = ti_signal in ("Bull", "Hold", "StrongBull")
    
    if c >= 63 and good_signal and ema_proximity and strong_trend:
        rr_ratio = predicted_return / abs(predicted_loss) if predicted_loss != 0 else 0
        if rr_ratio >= 1.0:
            return label("STRONG BUY")
        return label("Buy")
    elif 40 <= c < 63:
        return label("Wait")
    elif c < 40 and c >= 20 and base in ("Bear", "Short"):
        return label("Short the RISE")
    elif c < 20:
        return label("RISKY BUY")
    elif ti_signal == "StrongBull":
        return label("Monitor")
    elif ti_signal in ("Bull", "Hold"):
        return label("Watch")
    
    return label("Wait")

def generate_action(ticker, clean_label, conf, will_hit_str):
    colour = 'white'
    bull_case = {'TP': "BULLISH", 'Hold': "HOLD"}
    bear_case = {'Short': "BEARISH", 'SL': "BEARISH"}
    neutral_case = {'None': "NEUTRAL"}
    
    signal_text = bull_case.get(clean_label) or bear_case.get(clean_label) or neutral_case.get(clean_label, "NEUTRAL")
    
    if clean_label in bull_case and conf >= 80:
        action = f"{ticker}: Prediction is extremely {signal_text}, with ML {will_hit_str} & bull confidence ({conf:.0f}%) - BUY THE DIP"
        colour = 'darkgreen'
    elif clean_label in bull_case and 60 <= conf < 80:
        action = f"{ticker}: Prediction is {signal_text}, ML {will_hit_str} & bull confidence ({conf:.0f}%) suggests - BUY THE DIP"
        colour = 'green'
    elif clean_label in neutral_case and conf > 60:
        action = f"{ticker}: Prediction is {signal_text}, Despite neutrality, the confidence {conf:.0f}% suggests Buy-the-Dip."
        colour = 'lightgreen'
    elif clean_label in neutral_case and conf <= 20:
        action = f"{ticker}: Prediction is {signal_text}, Panic selling, tight SL - else SHORT - ML ({will_hit_str}) & bear confidence ({conf:.0f}%)"
        colour = 'white'
    elif clean_label in neutral_case and 40 <= conf <= 60:
        action = f"{ticker}: Prediction is {signal_text}, ML indicates SIDEWAYS ({conf:.0f}%). Only trade patterns with SL."
        colour = 'orange'
    elif clean_label in bear_case and 21 <= conf < 40:
        action = f"{ticker}: Prediction is {signal_text} - ML {will_hit_str} / {conf:.0f}% confidence suggest SHORT or HOLD"
        colour = 'red'
    elif clean_label in bear_case and conf <= 20:
        action = f"{ticker}: Prediction is {signal_text}, {will_hit_str}, ({conf:.0f}%) confidence indicates SELLERS Market"
        colour = 'red'
    else:
        action = f"{ticker} is NEUTRAL - Check for Monthly Candle, patterns, divergences"
        colour = 'gray'
    
    return action, colour

# ============================================
# PLOTTING FUNCTIONS
# ============================================

def plot_confidence_heatmap(df_results):
    st.subheader("🔥 ML Confidence Heatmap")
    
    df_plot = df_results.sort_values(by='Confidence', ascending=False).head(16).copy()
    
    if len(df_plot) == 0:
        st.warning("No data available to plot the heatmap.")
        return
    
    df_plot['Index'] = range(len(df_plot))
    df_plot['Row'] = (df_plot['Index'] // 5).astype(str)
    df_plot['Col'] = (df_plot['Index'] % 5).astype(str)
    
    df_plot['Display_Text'] = df_plot.apply(
        lambda row: f"{row['Ticker']} + ({row['_Extremes']})\n${row['Price']:.2f}\nR/R: {(row['Max (%)'] / abs(row['Loss (%)'])):.1f}\nConf: {row['Confidence']:.0f}%",
        axis=1
    )
    
    base = alt.Chart(df_plot).encode(
        x=alt.X('Col:N', axis=None),
        y=alt.Y('Row:N', axis=None),
    )
    
    heatmap = base.mark_rect().encode(
        color=alt.Color('Confidence:Q',
            scale=alt.Scale(domain=[5, 48, 95], range=['red', 'white', 'green'], clamp=True),
            legend=alt.Legend(title="Confidence %"),
        ),
        tooltip=['Ticker:N', 'Confidence:Q']
    )
    
    text = base.mark_text(align='center', baseline='middle', lineBreak='\n').encode(
        text=alt.Text('Display_Text:N'),
        color=alt.condition(alt.datum.Confidence > 60, alt.value('white'), alt.value('black'))
    )
    
    chart = (heatmap + text).properties(title='Top ML Confidence', width=600, height=400).interactive()
    st.altair_chart(chart, use_container_width=False)

def plot_single_ticker(ticker, df, df_results, _window=14):
    predictions = df_results[df_results['Ticker'] == ticker].iloc[0]
    if predictions.empty:
        st.text(f"No prediction results found for ticker {ticker}")
        return
    
    signal = predictions['Signal']
    current_price = round(df['Close'].iloc[-1], 2)
    gain = round(predictions['Max (%)'], 1)
    loss = round(predictions['Loss (%)'], 1)
    gain_price = current_price * (1 + gain/100)
    loss_price = current_price * (1 + loss/100)
    conf = predictions['Confidence']
    will_hit_str = df_results.loc[df_results['Ticker'] == ticker, 'Will_Hit'].values[0]
    clean_label = re.sub(r'\(.*?\)|[\d\.]+', '', will_hit_str).strip()
    last_date = df.index[-1]
    future_date = last_date + pd.Timedelta(days=_window)
    avg_price = (current_price + loss_price) / 2
    
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), dpi=600, sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Filter last 12 months
    end_date = df.index[-1]
    start_date = end_date - pd.DateOffset(months=12)
    df_plot = df.loc[start_date:end_date].copy()
    
    ax1.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Plot price with signal coloring
    price = df_plot['Close'].rolling(3).mean()
    ax1.plot(df_plot.index, price, color='blue', alpha=0.6, linewidth=1.5, label='Price')
    
    # Plot EMAs
    ax1.plot(df_plot.index, df_plot['EMA1'], label=f'EMA{int(_DAYS*0.5)}', color='gold', alpha=0.7, linewidth=1.2)
    ax1.plot(df_plot.index, df_plot['EMA2'], label=f'EMA{_DAYS}', color='red', alpha=0.7, linewidth=1.2, linestyle='--')
    
    # Fill between EMAs
    ax1.fill_between(df_plot.index, df_plot['EMA1'], df_plot['EMA2'], 
                     where=(df_plot['EMA1'] > df_plot['EMA2']), 
                     facecolor='green', alpha=0.2, label='BUY-times')
    ax1.fill_between(df_plot.index, df_plot['EMA1'], df_plot['EMA2'], 
                     where=(df_plot['EMA1'] <= df_plot['EMA2']), 
                     facecolor='red', alpha=0.2, label='Stay-away')
    
    # TP/SL lines
    ax1.plot([last_date, future_date], [avg_price, gain_price], color='green', linestyle=':', linewidth=1.5, alpha=0.5)
    ax1.plot([last_date, future_date], [avg_price, loss_price], color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    ax1.plot(future_date, gain_price, '^', markersize=_ms, color='green', alpha=0.5, label=f'TP: ${gain_price:.2f}')
    ax1.plot(future_date, loss_price, 'v', markersize=_ms, color='red', alpha=0.5, label=f'SL: ${loss_price:.2f}')
    
    # RSI subplot
    ax2.plot(df_plot.index, df_plot['RSI'], label='RSI', color='gray', linewidth=1.5, alpha=0.5)
    ax2.axhline(70, color='red', linewidth=1, linestyle='dashed', alpha=0.5)
    ax2.axhline(30, color='green', linewidth=1, linestyle='dashed', alpha=0.5)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('RSI')
    
    # Annotations
    signal_color = 'green' if 'Bull' in str(signal) else ('red' if 'Bear' in str(signal) else 'gray')
    _sigConf = f"{signal} & ML Action: {predictions['Action']}, Conf ({conf:.0f}%)"
    ax1.annotate(_sigConf, xy=(0.7, 0.95), xycoords='axes fraction', ha='right', va='top',
                 fontsize=10, weight='bold', bbox=dict(boxstyle='round', facecolor=signal_color, alpha=0.2))
    
    action, cl = generate_action(ticker, clean_label, conf, will_hit_str)
    textbox = AnchoredText(action, loc='lower left', frameon=True, borderpad=1.5,
                          prop=dict(size=7, color='blue', weight='normal'))
    textbox.patch.set(facecolor=cl, edgecolor='gray', alpha=0.4)
    ax1.add_artist(textbox)
    
    plt.tight_layout()
    st.pyplot(fig)

def PlotPredictions(df_results):
    if df_results.empty:
        st.warning("No results to plot")
        return
    
    df_plot = df_results.sort_values(by="Confidence", ascending=False)
    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=300)
    
    ax1.bar(df_plot["Ticker"], df_plot["Max (%)"], alpha=0.4, color='steelblue')
    ax1.set_ylabel('Max Return (%)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    ax2 = ax1.twinx()
    ax2.plot(df_plot["Ticker"], df_plot["Loss (%)"], color='red', marker='o', linewidth=2, markersize=8, label='Expected Loss')
    ax2.set_ylabel('Expected Loss (%)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title(f"{today} - ML Predictions", fontsize=16, pad=20)
    plt.tight_layout()
    st.pyplot(fig)

# ============================================
# MAIN PREDICTION FUNCTION
# ============================================

def MakePredictions(TICKERS):
    dfs = {}
    results = []
    label2str = {0: 'None', 1: 'SL', 2: 'TP', 3: 'Hold', 4: 'Short'}
    expected_classes = [0, 1, 2, 3, 4]
    
    for ticker in TICKERS:
        try:
            st.write(f"🔍 Processing {ticker}...")
            
            df = get_stock_data(ticker, start_date, end_date)
            if df is None or df.empty:
                st.warning(f"No data for {ticker}")
                continue
            
            df = add_technical_indicators(df)
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            
            df = add_pivot_levels(df, window=14)
            df = add_pivots(df, windows)
            df = average_pivots(df, windows)
            df = compute_expected_return(df, forward_window=14)
            df = compute_expected_loss(df, forward_window=14)
            df = label_hit_prob_past(df, window=30, profit_target=PROFIT_TARGET, stop_loss=STOP_LOSS)
            df['Hit_Label'] = df['Hit_Label'].fillna(0).astype(int)
            
            dfs[ticker] = df
            
            # Prepare data for ML
            df_model = df.dropna(subset=FEATURES + ['Hit_Label', 'Expected_Return', 'Expected_Loss'])
            if len(df_model) < _Nr:
                st.warning(f"Skipping {ticker} due to insufficient data")
                continue
            
            # Train classifier
            X_cls = df_model[FEATURES]
            y_cls = df_model['Hit_Label'].astype(int)
            scaler_cls = StandardScaler()
            X_scaled_cls = scaler_cls.fit_transform(X_cls)
            
            model_class = RandomForestClassifier(n_estimators=120, max_depth=12, 
                                                min_samples_split=4, random_state=42)
            model_class.fit(X_scaled_cls, y_cls)
            
            # Latest prediction
            latest = df.iloc[[-1]]
            if latest[FEATURES].isnull().values.any():
                st.warning(f"Null features for {ticker}")
                continue
            
            latest_scaled = scaler_cls.transform(latest[FEATURES])
            pred_class = model_class.predict(latest_scaled)[0]
            will_hit = label2str.get(pred_class, "None")
            
            # Get probabilities
            probs = model_class.predict_proba(latest_scaled)[0]
            hit_prob = max(probs) * 100
            
            # Simple confidence score
            confidence_score = hit_prob
            
            current_price = latest['Close'].values[0]
            predicted_return = df['Expected_Return'].iloc[-1] if not pd.isna(df['Expected_Return'].iloc[-1]) else 0.03
            predicted_loss = df['Expected_Loss'].iloc[-1] if not pd.isna(df['Expected_Loss'].iloc[-1]) else -0.03
            
            # Get TI signal
            ti_signal = df['TI'].iloc[-1] if 'TI' in df.columns else 'Neutral'
            ema1_val = latest['EMA1'].iloc[0] if 'EMA1' in latest.columns else current_price
            
            # Generate action
            action = get_action_label(confidence_score, will_hit, current_price, ema1_val, 50, ti_signal, predicted_return, predicted_loss)
            
            results.append({
                "Ticker": ticker,
                "Price": round(current_price, 2),
                "Max (%)": round(predicted_return * 100, 1),
                "Loss (%)": round(predicted_loss * 100, 1),
                "Signal": ti_signal,
                "Will_Hit": will_hit,
                "Hit_Prob": round(hit_prob, 1),
                "Confidence": round(confidence_score, 1),
                "Action": action,
                "_Extremes": df['Exhaustion'].iloc[-1] if 'Exhaustion' in df.columns else 0
            })
            
            st.write(f"✅ {ticker}: {action} (Conf: {confidence_score:.0f}%)")
            
        except Exception as e:
            st.error(f"Error processing {ticker}: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    df_results = pd.DataFrame(results)
    return dfs, df_results

# ============================================
# TABULAR DISPLAY
# ============================================

def tabular_display(df_results):
    if df_results.empty:
        st.warning("No results to display")
        return
    
    styled_df = df_results.style.format({
        'Price': '{:.2f}',
        'Max (%)': '{:.1f}',
        'Loss (%)': '{:.1f}',
        'Confidence': '{:.0f}',
        'Hit_Prob': '{:.0f}'
    })
    
    st.dataframe(styled_df, height=550, use_container_width=True)

def style_rows(row):
    if 'Bull' in str(row.get('Signal', '')):
        return ['background-color: rgba(144, 238, 144, 0.3)'] * len(row)
    elif 'Bear' in str(row.get('Signal', '')):
        return ['background-color: rgba(240, 128, 128, 0.3)'] * len(row)
    else:
        return ['color: gray'] * len(row)

# ============================================
# MAIN APP
# ============================================

desc = """  
- Machine learning models train technical indicators
- Trade signals include signal type, hit probability, and direction
- Use tables to identify strong stocks and charts to confirm bullish trends
"""

HowTo = """
## 📘 Real-Life ML Stock Trading Rules

### 🎯 Objective
Use ML as a **risk filter**, not a prediction engine.

### 2️⃣ ML Entry Conditions
- ML confidence **≥ 60%**
- ML signal is **NOT bearish**
- Expected reward > expected risk
"""

mistakes = """
- Review Psychology Tab for common pitfalls
- Cash is king: Discipline to hold cash without trading is an art
- Split positions systematically
"""

disclaimer = """
---
- Trading involves substantial risk of financial loss  
- Past performance does not predict future results  
- Trade at your own risk  
---
"""

def run_app():
    with st.expander("Positional/Swing Trading Guidance"):
        st.write(desc)
    
    with st.expander("Signals & Stocks Selection"):
        st.write(HowTo)
    
    with st.expander("Common Mistakes"):
        st.write(mistakes)
    
    with st.expander("Disclaimer"):
        st.write(disclaimer)
    
    st.title("📈 Machine Learning Signals (TA)")
    
    tickers_input = st.text_input("Enter comma-separated tickers (max 15):", 
                                   placeholder="e.g., AAPL, MSFT, GOOGL, COIN")
    
    if tickers_input:
        TICKERS = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        
        if len(TICKERS) > 15:
            st.error("You can enter up to 15 tickers only.")
        else:
            valid_tickers = []
            invalid_tickers = []
            for t in TICKERS:
                if is_valid_ticker(t):
                    valid_tickers.append(t)
                else:
                    invalid_tickers.append(t)
            
            if not valid_tickers:
                st.error(f"All tickers are invalid: {', '.join(invalid_tickers)}")
            else:
                if invalid_tickers:
                    st.warning(f"Ignoring invalid tickers: {', '.join(invalid_tickers)}")
                
                st.code(f"Processing {len(valid_tickers)} tickers: {', '.join(valid_tickers)}")
                
                with st.spinner("Analyzing tickers..."):
                    dfs, df_results = MakePredictions(valid_tickers)
                
                if not df_results.empty:
                    plot_confidence_heatmap(df_results)
                    PlotPredictions(df_results)
                    
                    with st.expander("Tabular Results"):
                        tabular_display(df_results)
                    
                    st.session_state['ml_results'] = df_results
                    
                    # Plot individual tickers
                    for ticker in valid_tickers:
                        if ticker in dfs and dfs[ticker] is not None:
                            plot_single_ticker(ticker, dfs[ticker], df_results)
                else:
                    st.error("No results generated. Please check your inputs.")

if __name__ == "__main__":
    run_app()
