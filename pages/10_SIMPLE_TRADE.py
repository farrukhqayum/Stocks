import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =========================
# EARLY ENTRY Parameters
# =========================
TICKER = "COIN"

# Faster indicators = Earlier signals
EMA_FAST = 8   # Even faster (was 9)
EMA_SLOW = 21
RSI_LEN = 14

# Aggressive Risk Management
INITIAL_STOP_LOSS = 0.06  # Tighter since entering earlier (6%)
TRAIL_MULT = 2.5  # Still give room
PARTIAL_TP = 0.35  # Lower target for earlier entry (35%)
PARTIAL_SIZE = 0.50

# Relaxed entry filters
MIN_ADX = 12  # Lower bar
USE_PULLBACK_ENTRY = True  # NEW: Enter on pullbacks to EMA

INITIAL_CAPITAL = 1000

# =========================
# Data Download
# =========================
def robust_download(ticker, period="2y"):
    print(f"🔄 Downloading {ticker}...")
    session = requests.Session()
    retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    yf._session = session

    methods = [
        lambda: yf.download(ticker, period=period, progress=False, threads=False),
        lambda: yf.download(ticker, start="2023-01-01", progress=False, threads=False),
        lambda: yf.Ticker(ticker).history(period=period),
    ]

    for i, method in enumerate(methods, 1):
        try:
            df = method()
            if not df.empty and len(df) > 50:
                print(f"✅ Method {i} worked! {len(df)} rows")
                return df
        except Exception as e:
            print(f"❌ Method {i} failed: {str(e)[:50]}")
            time.sleep(1)

    print("🔄 Using fallback data...")
    return yf.Ticker("AAPL").history(period="2y")

def RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs)).fillna(50)

def ADX(df, period=14):
    high, low, close = df["High"].squeeze(), df["Low"].squeeze(), df["Close"].squeeze()
    tr1, tr2, tr3 = high-low, abs(high-close.shift()), abs(low-close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    atr = tr.rolling(window=period, min_periods=1).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()
    return adx, plus_di, minus_di

# =========================
# MAIN EXECUTION
# =========================
print("🚀 EARLY ENTRY STRATEGY - Catch Moves Sooner!")
print("=" * 50)

df = robust_download(TICKER)
df = df.dropna()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
if 'Adj Close' in df.columns:
    df['Close'] = df['Adj Close']

close = df['Close'].squeeze()
high = df['High'].squeeze()
low = df['Low'].squeeze()

print(f"📊 {len(df)} rows loaded for {TICKER}")

# Calculate indicators
print("\n📈 Calculating indicators...")
df['EMA_FAST'] = close.ewm(span=EMA_FAST, adjust=False).mean()
df['EMA_SLOW'] = close.ewm(span=EMA_SLOW, adjust=False).mean()
df["RSI"] = RSI(close, RSI_LEN)
df["ADX"], df["PLUS_DI"], df["MINUS_DI"] = ADX(df)

# ATR
tr = pd.concat([high-low, abs(high-close.shift()), abs(low-close.shift())], axis=1).max(axis=1)
df["ATR"] = tr.rolling(14, min_periods=1).mean()

# Price momentum
df['PRICE_CHANGE'] = close.pct_change(1) * 100  # 1-day % change

df = df.dropna()

# =========================
# MULTIPLE ENTRY STRATEGIES - Catch More Moves!
# =========================

# Strategy 1: BREAKOUT ENTRY (catch early surge)
df["BREAKOUT"] = (
    (df['EMA_FAST'] > df['EMA_SLOW']) &  # Trend up
    (df['Close'] > df['EMA_FAST'].shift(1)) &  # Just crossed above
    (df['Close'].shift(1) <= df['EMA_FAST'].shift(1)) &  # Was below yesterday
    (df['ADX'] > MIN_ADX)
)

# Strategy 2: PULLBACK ENTRY (buy dips in uptrend)
df["PULLBACK"] = (
    (df['EMA_FAST'] > df['EMA_SLOW']) &  # Trend up
    (df['Close'] < df['EMA_FAST']) &  # Price pulled back below EMA
    (df['Close'] > df['EMA_FAST'] * 0.97) &  # But not too far (within 3%)
    (df['RSI'] < 55) &  # RSI cooled off
    (df['RSI'] > 40) &  # But not oversold
    (df['PRICE_CHANGE'] > -2) &  # Not a crash
    (df['ADX'] > MIN_ADX)
)

# Strategy 3: MOMENTUM CONTINUATION (ride the wave)
df["MOMENTUM"] = (
    (df['EMA_FAST'] > df['EMA_SLOW']) &  # Trend up
    (df['Close'] > df['EMA_FAST']) &  # Above fast EMA
    (df['RSI'] > 50) &  # Bullish RSI
    (df['RSI'] < 70) &  # Not overbought
    (df['PRICE_CHANGE'] > 0.5) &  # Positive momentum today
    (df['ADX'] > MIN_ADX)
)

# COMBINED SIGNAL - Enter on ANY strategy
df["BULL"] = df["BREAKOUT"] | df["PULLBACK"] | df["MOMENTUM"]

df["Next_Low"] = df["Low"].shift(-1)
df["Next_High"] = df["High"].shift(-1)
df["Next_Close"] = df["Close"].shift(-1)
df = df.dropna()

# =========================
# BACKTEST
# =========================
capital = INITIAL_CAPITAL
equity_curve = []
trade_records = []

in_trade = False
entry_price = 0
original_entry = 0
initial_stop = 0
trail_stop = 0
position_size = 1.0
took_partial = False
highest_price = 0
entry_strategy = ""

print("\n⚡ Running backtest with EARLY entries...")
for i in range(len(df) - 1):
    row = df.iloc[i]
    current_close = row["Close"]
    next_low = row["Next_Low"]
    next_close = row["Next_Close"]

    current_equity = capital

    if not in_trade:
        if row["BULL"]:
            entry_price = current_close
            original_entry = current_close
            highest_price = current_close
            in_trade = True
            took_partial = False
            position_size = 1.0
            
            # Determine which strategy triggered
            if row["BREAKOUT"]:
                entry_strategy = "BREAKOUT"
            elif row["PULLBACK"]:
                entry_strategy = "PULLBACK"
            else:
                entry_strategy = "MOMENTUM"
            
            # Set stop
            initial_stop = entry_price * (1 - INITIAL_STOP_LOSS)
            trail_stop = initial_stop
            
            current_trade = {
                'EntryDate': df.index[i],
                'EntryPrice': entry_price,
                'Strategy': entry_strategy,
                'ADX': row['ADX'],
                'RSI': row['RSI'],
                'Partial': False
            }

    else:
        # Track highest
        if current_close > highest_price:
            highest_price = current_close
        
        # Partial profit
        if not took_partial and current_close >= original_entry * (1 + PARTIAL_TP):
            partial_profit = PARTIAL_SIZE * (current_close - original_entry) / original_entry
            capital *= (1 + partial_profit)
            position_size *= (1 - PARTIAL_SIZE)
            took_partial = True
            current_trade['Partial'] = True

        # Trailing stop
        new_trail = highest_price - TRAIL_MULT * row["ATR"]
        trail_stop = max(trail_stop, new_trail)

        # Exit conditions
        exit_triggered = False
        exit_price = None
        exit_reason = None
        
        # 1. Stop hit
        if next_low <= trail_stop:
            exit_price = max(next_close, trail_stop)
            exit_reason = "STOP"
            exit_triggered = True
        
        # 2. Trend broken
        elif (df['EMA_FAST'].iloc[i] < df['EMA_SLOW'].iloc[i] and 
              current_close < df['EMA_FAST'].iloc[i]):
            exit_price = next_close
            exit_reason = "TREND_BREAK"
            exit_triggered = True
        
        # 3. RSI extreme weakness (divergence)
        elif row['RSI'] < 35 and current_close < df['EMA_FAST'].iloc[i]:
            exit_price = next_close
            exit_reason = "RSI_WEAK"
            exit_triggered = True

        if exit_triggered:
            pnl = position_size * (exit_price - original_entry) / original_entry
            capital *= (1 + pnl)
            
            max_gain = (highest_price - original_entry) / original_entry
            
            current_trade['ExitDate'] = df.index[i+1]
            current_trade['ExitPrice'] = exit_price
            current_trade['HighestPrice'] = highest_price
            current_trade['MaxGain%'] = max_gain * 100
            current_trade['Return%'] = pnl * 100
            current_trade['Reason'] = exit_reason
            trade_records.append(current_trade.copy())
            
            in_trade = False

        if in_trade:
            unrealized = position_size * (current_close - original_entry) / original_entry
            current_equity = capital * (1 + unrealized)

    equity_curve.append(current_equity)

# Close open position
if in_trade:
    exit_price = df['Close'].iloc[-1]
    pnl = position_size * (exit_price - original_entry) / original_entry
    capital *= (1 + pnl)
    current_trade['ExitDate'] = df.index[-1]
    current_trade['ExitPrice'] = exit_price
    current_trade['Return%'] = pnl * 100
    current_trade['Reason'] = "END"
    trade_records.append(current_trade)

equity_curve += [capital] * (len(df) - len(equity_curve))
equity_series = pd.Series(equity_curve, index=df.index)

# =========================
# RESULTS WITH STRATEGY BREAKDOWN
# =========================
trade_df = pd.DataFrame(trade_records)

print("\n" + "="*70)
print(f"🎯 EARLY ENTRY RESULTS: {TICKER}")
print("="*70)
print(f"Total Trades: {len(trade_df)}")
print(f"Initial Capital: ${INITIAL_CAPITAL:,.0f}")
print(f"Final Capital: ${capital:,.2f}")
print(f"Total Return: {(capital/INITIAL_CAPITAL-1)*100:.2f}%")

if not trade_df.empty:
    wins = trade_df[trade_df['Return%'] > 0]
    losses = trade_df[trade_df['Return%'] <= 0]
    
    print(f"\n📊 Performance:")
    print(f"Winners: {len(wins)} | Losers: {len(losses)}")
    print(f"Win Rate: {len(wins)/len(trade_df)*100:.1f}%")
    
    if len(wins) > 0:
        print(f"Avg Winner: {wins['Return%'].mean():.2f}%")
        print(f"Largest Win: {wins['Return%'].max():.2f}%")
    
    if len(losses) > 0:
        print(f"Avg Loser: {losses['Return%'].mean():.2f}%")
        print(f"Largest Loss: {losses['Return%'].min():.2f}%")
    
    # Profit Factor
    if len(losses) > 0 and losses['Return%'].sum() != 0:
        gross_profit = wins['Return%'].sum()
        gross_loss = abs(losses['Return%'].sum())
        profit_factor = gross_profit / gross_loss
        print(f"\n💎 Profit Factor: {profit_factor:.2f}")
    
    # Strategy breakdown
    print(f"\n🎯 Entry Strategy Performance:")
    for strategy in ['BREAKOUT', 'PULLBACK', 'MOMENTUM']:
        strat_trades = trade_df[trade_df['Strategy'] == strategy]
        if len(strat_trades) > 0:
            strat_wins = strat_trades[strat_trades['Return%'] > 0]
            win_rate = len(strat_wins)/len(strat_trades)*100
            avg_return = strat_trades['Return%'].mean()
            print(f"  {strategy:12} - {len(strat_trades):2} trades | WR: {win_rate:5.1f}% | Avg: {avg_return:+6.2f}%")
    
    print(f"\n📋 Exit Reasons:")
    print(trade_df['Reason'].value_counts())
    
    partials = len(trade_df[trade_df['Partial'] == True])
    print(f"\n💰 Reached partial target: {partials}/{len(trade_df)} ({partials/len(trade_df)*100:.1f}%)")

# =========================
# VISUALIZATION
# =========================
fig, axes = plt.subplots(4, 1, figsize=(16, 12), height_ratios=[3, 1, 1, 1.5])
fig.patch.set_facecolor('white')

# Price Chart
ax1 = axes[0]
ax1.plot(df.index, df['Close'], color='#2C3E50', linewidth=1.5, label='Close', alpha=0.8)
ax1.plot(df.index, df['EMA_FAST'], color='#3498DB', linewidth=1.5, label=f'EMA {EMA_FAST}', alpha=0.7)
ax1.plot(df.index, df['EMA_SLOW'], color='#E74C3C', linewidth=1.5, label=f'EMA {EMA_SLOW}', alpha=0.7)

if not trade_df.empty:
    # Color entries by strategy
    for strategy, color in [('BREAKOUT', '#FF6B6B'), ('PULLBACK', '#4ECDC4'), ('MOMENTUM', '#95E1D3')]:
        strat_entries = trade_df[trade_df['Strategy'] == strategy]
        if len(strat_entries) > 0:
            ax1.scatter(strat_entries['EntryDate'], strat_entries['EntryPrice'], 
                       marker='^', color=color, s=120, alpha=0.9, 
                       label=f'{strategy}', zorder=5, edgecolors='white', linewidths=1)
    
    # Exits
    exits = trade_df[trade_df['ExitDate'].notna()].copy()
    winners_exits = exits[exits['Return%'] > 0]
    losers_exits = exits[exits['Return%'] <= 0]
    
    if len(winners_exits) > 0:
        ax1.scatter(winners_exits['ExitDate'], winners_exits['ExitPrice'],
                   marker='v', color='#27AE60', s=120, alpha=0.9, 
                   label='Exit (Win)', zorder=5, edgecolors='white', linewidths=1)
    if len(losers_exits) > 0:
        ax1.scatter(losers_exits['ExitDate'], losers_exits['ExitPrice'],
                   marker='v', color='#E74C3C', s=120, alpha=0.9, 
                   label='Exit (Loss)', zorder=5, edgecolors='white', linewidths=1)

ax1.set_title(f'{TICKER} - Early Entry Multi-Strategy', fontsize=16, fontweight='bold', pad=15)
ax1.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', framealpha=0.9, fontsize=9, ncol=2)
ax1.grid(True, alpha=0.2, linestyle='--')
ax1.set_facecolor('#F8F9FA')

# RSI
ax2 = axes[1]
ax2.plot(df.index, df['RSI'], color='#9B59B6', linewidth=1.5, label='RSI')
ax2.axhline(y=70, color='#E74C3C', linestyle='--', alpha=0.5, linewidth=1)
ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.3, linewidth=1)
ax2.axhline(y=30, color='#27AE60', linestyle='--', alpha=0.5, linewidth=1)
ax2.set_ylabel('RSI', fontsize=11, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.2)
ax2.set_ylim(0, 100)
ax2.set_facecolor('#F8F9FA')

# ADX
ax3 = axes[2]
ax3.plot(df.index, df['ADX'], color='#E67E22', linewidth=1.5, label='ADX')
ax3.axhline(y=MIN_ADX, color='#E74C3C', linestyle='--', alpha=0.5, linewidth=1, label=f'Min={MIN_ADX}')
ax3.set_ylabel('ADX', fontsize=11, fontweight='bold')
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.2)
ax3.set_facecolor('#F8F9FA')

# Equity
ax4 = axes[3]
ax4.plot(equity_series.index, equity_series, color='#27AE60', linewidth=2.5, label='Portfolio')
ax4.fill_between(equity_series.index, INITIAL_CAPITAL, equity_series, 
                alpha=0.3, color='#27AE60' if capital > INITIAL_CAPITAL else '#E74C3C')
ax4.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax4.set_ylabel('Equity ($)', fontsize=11, fontweight='bold')
ax4.set_xlabel('Date', fontsize=12, fontweight='bold')
ax4.legend(loc='upper left', fontsize=9)
ax4.grid(True, alpha=0.2)
ax4.set_facecolor('#F8F9FA')

return_pct = (capital/INITIAL_CAPITAL-1)*100
perf_text = f'Return: {return_pct:+.1f}% | Trades: {len(trade_df)} | Win Rate: {len(wins)/len(trade_df)*100:.0f}%'
fig.text(0.5, 0.02, perf_text, ha='center', fontsize=13, 
        fontweight='bold', color='#27AE60' if return_pct > 0 else '#E74C3C', 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.show()

print("\n✅ Multi-strategy backtest complete!")
print("\n💡 STRATEGY COMPARISON:")
print("   BREAKOUT: Catches initial surge (highest reward, moderate risk)")
print("   PULLBACK: Enters on dips (better entry, lower risk)")
print("   MOMENTUM: Rides trends (consistent but can be late)")
print("\n   Using ALL THREE = More opportunities + better timing!")
