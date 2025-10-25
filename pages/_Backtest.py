#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from imports import *   # expects ta.calculate_rsi(df) and ta.calculate_atr(...)
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="ML Daily Entry with Weekly Trend Filter", layout="wide")

st.title("🤖 ML Daily Entry — Weekly Trend Filter with 7% TP/SL")
st.markdown("""
Strategy:
- **Weekly Trend Filter**: Only take trades when SMA10 > SMA50 on weekly timeframe
- **Daily ML Entries**: Use ML predictions for entry signals on daily timeframe  
- **Entry**: At daily close when ML predicts upward movement
- **Exit**: 7% TP or 7% SL based on daily price action
- **Risk Management**: Close trade if price jumps ±7% intraday or next day open
- **Non-overlapping**: Wait for current trade to close before taking next signal
""")

# -------------------------
# User inputs
# -------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    ticker = st.text_input("Ticker", value="COIN")
with col2:
    period = st.selectbox("History period", ["2y", "3y", "5y", "7y"], index=2)
with col3:
    TP_pct = st.number_input("TP (%)", value=7.0, step=0.5)
with col4:
    SL_pct = st.number_input("SL (%)", value=7.0, step=0.5)

# ML prediction settings
col5, col6 = st.columns(2)
with col5:
    ml_confidence_threshold = st.number_input("ML Confidence Threshold", value=0.6, step=0.1, 
                                            help="Minimum confidence score for ML prediction")
with col6:
    max_holding_days = st.number_input("Max Holding Days", value=30, step=5,
                                      help="Maximum days to hold a trade")

if st.button("Run ML Strategy Backtest"):
    st.write(f"Downloading {period} data for {ticker} and running ML strategy...")
    
    # Download both weekly and daily data
    df_daily = yf.download(ticker, period=period, interval="1d", progress=False)
    df_weekly = yf.download(ticker, period=period, interval="1wk", progress=False)
    
    if df_daily.empty or df_weekly.empty:
        st.error("No data returned from Yahoo Finance for that ticker/period.")
        st.stop()

    # --- Data cleaning ---
    for df in [df_daily, df_weekly]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]

    # --- Weekly trend analysis ---
    df_weekly['SMA10'] = df_weekly['Close'].rolling(10).mean()
    df_weekly['SMA50'] = df_weekly['Close'].rolling(50).mean()
    df_weekly['trend_up'] = df_weekly['SMA10'] > df_weekly['SMA50']

    # --- ML Prediction Simulation ---
    # In a real implementation, you would replace this with your actual ML model
    st.write("Generating simulated ML predictions...")
    
    # Simulate ML predictions (replace this with actual ML model inference)
    df_daily = simulate_ml_predictions(df_daily)
    
    # Ensure we have ML prediction column
    if 'ml_prediction' not in df_daily.columns:
        st.error("ML predictions not found. Please ensure your ML model generates 'ml_prediction' column.")
        st.stop()

    # --- Backtest Logic ---
    trades = []
    in_trade = False
    current_trade = {}
    
    # Align weekly and daily data
    weekly_dates = df_weekly.index
    daily_dates = df_daily.index
    
    for i, current_date in enumerate(daily_dates):
        # Find corresponding weekly date (last weekly candle before or on current_date)
        weekly_mask = weekly_dates <= current_date
        if not weekly_mask.any():
            continue
            
        latest_weekly_date = weekly_dates[weekly_mask][-1]
        weekly_trend_up = df_weekly.loc[latest_weekly_date, 'trend_up'] if pd.notna(df_weekly.loc[latest_weekly_date, 'trend_up']) else False
        
        # Skip if weekly trend is down
        if not weekly_trend_up:
            if in_trade:
                # Check if we need to exit due to trend change
                pass  # You might want to add trend-based exits
            continue
        
        current_ml_signal = df_daily.loc[current_date, 'ml_prediction']
        current_ml_confidence = df_daily.loc[current_date, 'ml_confidence'] if 'ml_confidence' in df_daily.columns else 1.0
        
        # ENTRY LOGIC: Not in trade + ML buy signal + sufficient confidence
        if (not in_trade and 
            current_ml_signal == 1 and 
            current_ml_confidence >= ml_confidence_threshold):
            
            entry_price = float(df_daily.loc[current_date, 'Close'])
            TP_price = entry_price * (1 + TP_pct / 100.0)
            SL_price = entry_price * (1 - SL_pct / 100.0)
            
            current_trade = {
                'entry_date': current_date,
                'entry_price': entry_price,
                'tp_price': TP_price,
                'sl_price': SL_price,
                'entry_week': latest_weekly_date
            }
            in_trade = True
            
            st.write(f"📈 Entry on {current_date.strftime('%Y-%m-%d')} at {entry_price:.2f}")
        
        # EXIT LOGIC: Check if we're in a trade and need to exit
        elif in_trade:
            entry_date = current_trade['entry_date']
            entry_price = current_trade['entry_price']
            TP_price = current_trade['tp_price']
            SL_price = current_trade['sl_price']
            
            # Calculate days in trade
            days_in_trade = (current_date - entry_date).days
            
            # Get current day's prices
            current_open = float(df_daily.loc[current_date, 'Open'])
            current_high = float(df_daily.loc[current_date, 'High'])
            current_low = float(df_daily.loc[current_date, 'Low'])
            current_close = float(df_daily.loc[current_date, 'Close'])
            
            exit_reason = None
            exit_price = None
            
            # 1) Check for Stop Loss hit (intraday low <= SL)
            if current_low <= SL_price:
                exit_reason = 'SL'
                exit_price = SL_price
            
            # 2) Check for Take Profit hit (intraday high >= TP)
            elif current_high >= TP_price:
                exit_reason = 'TP'
                exit_price = TP_price
            
            # 3) Check for gap moves at open
            elif current_open <= SL_price:
                exit_reason = 'Gap_SL'
                exit_price = min(current_open, SL_price)
            elif current_open >= TP_price:
                exit_reason = 'Gap_TP'
                exit_price = max(current_open, TP_price)
            
            # 4) Max holding period reached
            elif days_in_trade >= max_holding_days:
                exit_reason = 'Max_Hold'
                exit_price = current_close
            
            # Exit trade if any condition met
            if exit_reason:
                return_pct = (exit_price / entry_price - 1) * 100.0
                
                trades.append({
                    'EntryDate': entry_date,
                    'ExitDate': current_date,
                    'EntryPrice': entry_price,
                    'ExitPrice': exit_price,
                    'Outcome': exit_reason,
                    'Return_%': return_pct,
                    'HoldingDays': days_in_trade,
                    'ML_Confidence': current_trade.get('ml_confidence', 1.0)
                })
                
                st.write(f"📉 Exit on {current_date.strftime('%Y-%m-%d')} at {exit_price:.2f} ({exit_reason})")
                in_trade = False
                current_trade = {}

    # Handle any open trade at the end
    if in_trade:
        last_date = daily_dates[-1]
        exit_price = float(df_daily.loc[last_date, 'Close'])
        return_pct = (exit_price / current_trade['entry_price'] - 1) * 100.0
        
        trades.append({
            'EntryDate': current_trade['entry_date'],
            'ExitDate': last_date,
            'EntryPrice': current_trade['entry_price'],
            'ExitPrice': exit_price,
            'Outcome': 'Open',
            'Return_%': return_pct,
            'HoldingDays': (last_date - current_trade['entry_date']).days,
            'ML_Confidence': current_trade.get('ml_confidence', 1.0)
        })

    # --- Results Analysis ---
    results = pd.DataFrame(trades)
    
    if results.empty:
        st.warning("No trades executed. Check ML predictions and weekly trend conditions.")
        st.stop()

    # Calculate cumulative returns
    initial_cap = 1.0
    results['Return_factor'] = 1 + results['Return_%'] / 100.0
    results['Cumulative'] = initial_cap * results['Return_factor'].cumprod()
    equity_ts = pd.Series(data=results['Cumulative'].values, index=pd.to_datetime(results['ExitDate']))

    # --- Performance Metrics ---
    total_trades = len(results)
    wins = results['Return_%'] > 0
    win_rate = 100.0 * wins.sum() / total_trades if total_trades > 0 else 0
    avg_return = results['Return_%'].mean()
    net_return_pct = (results['Cumulative'].iloc[-1] - initial_cap) / initial_cap * 100.0
    
    # ML-specific metrics
    successful_ml_predictions = len(results[results['Return_%'] > 0])
    ml_accuracy = 100.0 * successful_ml_predictions / total_trades if total_trades > 0 else 0

    st.subheader("📊 ML Strategy Performance Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades", total_trades)
    c2.metric("Win Rate", f"{win_rate:.1f}%")
    c3.metric("ML Prediction Accuracy", f"{ml_accuracy:.1f}%")
    c4.metric("Net Return", f"{net_return_pct:.2f}%")

    # Trade outcomes breakdown
    st.subheader("Trade Outcomes")
    outcome_counts = results['Outcome'].value_counts()
    st.write(outcome_counts)

    st.subheader("Recent Trades")
    st.dataframe(results.sort_values('EntryDate', ascending=False).head(20))

    # --- Visualization ---
    st.subheader("Charts")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Price chart with trades
    ax1.plot(df_daily.index, df_daily['Close'], label='Price', color='black', linewidth=1)
    
    # Mark entries and exits
    for _, trade in results.iterrows():
        color = 'green' if trade['Return_%'] > 0 else 'red'
        ax1.scatter(trade['EntryDate'], trade['EntryPrice'], color='blue', marker='^', s=80, zorder=5)
        ax1.scatter(trade['ExitDate'], trade['ExitPrice'], color=color, marker='o', s=60, zorder=5)
        
        # Draw trade line
        ax1.plot([trade['EntryDate'], trade['ExitDate']], 
                [trade['EntryPrice'], trade['ExitPrice']], color=color, alpha=0.3)

    ax1.set_title(f"{ticker} - ML Strategy Trades (Weekly Trend Filter)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Equity curve
    ax2.plot(equity_ts.index, equity_ts.values, color='green', linewidth=2)
    ax2.set_title("Equity Curve")
    ax2.set_ylabel("Growth (1.0 = Start)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # --- Export Results ---
    if st.button("Export Results to CSV"):
        csv = results.to_csv(index=False)
        st.download_button(
            label="Download Trade Log",
            data=csv,
            file_name=f"ml_strategy_results_{ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    st.success("Backtest complete!")

def simulate_ml_predictions(df_daily, lookback_days=5):
    """
    Simulate ML predictions for demonstration.
    Replace this function with your actual ML model inference.
    """
    # Simple simulation: predict up if recent momentum is positive
    df = df_daily.copy()
    
    # Calculate some features (simulate what your ML model might use)
    df['returns_1d'] = df['Close'].pct_change(1)
    df['returns_5d'] = df['Close'].pct_change(5)
    df['volume_ma'] = df['Volume'].rolling(5).mean()
    df['volatility'] = df['Close'].rolling(5).std()
    
    # Simulate ML predictions (1 = buy, 0 = hold/no action)
    # This is a simple rule-based simulation - replace with actual ML
    df['ml_prediction'] = 0
    df['ml_confidence'] = 0.0
    
    # Simple rules to simulate ML (replace with actual model)
    buy_conditions = (
        (df['returns_5d'] > -0.02) &  # Not in strong downtrend
        (df['volume_ma'] > df['Volume'].rolling(20).mean()) &  # Above average volume
        (df['volatility'] < df['volatility'].rolling(20).mean() * 1.5)  # Not too volatile
    )
    
    df.loc[buy_conditions, 'ml_prediction'] = 1
    df.loc[buy_conditions, 'ml_confidence'] = np.random.uniform(0.6, 0.9, sum(buy_conditions))
    
    # Add some randomness to make it more realistic
    random_signals = np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])
    random_mask = (random_signals == 1) & (df['ml_prediction'] == 0)
    df.loc[random_mask, 'ml_prediction'] = 1
    df.loc[random_mask, 'ml_confidence'] = np.random.uniform(0.5, 0.7, sum(random_mask))
    
    return df

# Add this if you want to run standalone
if __name__ == "__main__":
    pass
