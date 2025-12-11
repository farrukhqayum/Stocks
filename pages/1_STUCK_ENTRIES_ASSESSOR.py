import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ------------------------
# Page config
# ------------------------
st.set_page_config(page_title="Stock Position Assessor (Enhanced)", layout="wide")
st.title("Stock Position Assessment Tool – Enhanced Breakeven Analysis")

# ------------------------
# Sidebar: inputs
# ------------------------
ticker = st.sidebar.text_input("Enter Stock Ticker", value="COIN")
start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365 * 5))
end_date = st.sidebar.date_input("End Date", value=datetime.now())

# ------------------------
# Data loader
# ------------------------
@st.cache_data
def load_stock_data(ticker, start_date, end_date):
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date + timedelta(days=1),
        progress=False
    )
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df = df.dropna()
    return df

data = None
if ticker:
    data = load_stock_data(ticker, start_date, end_date)

if (data is None) or data.empty:
    st.warning("No data loaded. Please check ticker symbol and date range.")
    st.stop()

# ------------------------
# Current price & chart
# ------------------------
current_price = data["Close"].iloc[-1]
st.metric("Current Price", f"${current_price:.2f}")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data.index, data["Close"], linewidth=2, color="blue", label="Close")
ax.fill_between(data.index, data["Low"], data["High"], alpha=0.3, color="gray", label="High–Low")
ax.set_title(f"{ticker} Price Chart", fontsize=16, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Price ($)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ------------------------
# Positions input (same style as yours)
# ------------------------
st.header("Your Positions")

num_entries = st.number_input("Number of Recent Entries", min_value=1, max_value=10, value=3)

entries = []
for i in range(num_entries):
    c1, c2, c3 = st.columns(3)
    with c1:
        date = st.date_input(
            f"Entry {i+1} Date",
            value=datetime.now() - timedelta(days=30 * (i + 1)),
            key=f"date_{i}"
        )
    with c2:
        shares = st.number_input(
            f"Entry {i+1} Shares",
            min_value=1.0,
            value=100.0,
            key=f"shares_{i}"
        )
    with c3:
        price = st.number_input(
            f"Entry {i+1} Price",
            min_value=0.1,
            value=float(current_price),
            key=f"price_{i}"
        )
    entries.append({"date": pd.to_datetime(date), "shares": shares, "price": price})

entries_df = pd.DataFrame(entries)

total_shares = entries_df["shares"].sum()
total_invested = (entries_df["shares"] * entries_df["price"]).sum()
avg_cost = total_invested / total_shares if total_shares > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Shares", f"{total_shares:.0f}")
c2.metric("Total Invested", f"${total_invested:.2f}")
c3.metric("Avg Cost Basis", f"${avg_cost:.2f}")
c4.metric("Current Value", f"${total_shares * current_price:.2f}")

pnl_pct = ((current_price - avg_cost) / avg_cost * 100) if avg_cost > 0 else 0
st.metric("PnL %", f"{pnl_pct:.2f}%")

# Position breakdown pie chart
c1, c2 = st.columns([1, 2])
with c1:
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    weights = entries_df["shares"].tolist()
    labels = [
        f"Entry {i+1}\n${(s * p):.0f}"
        for i, (s, p) in enumerate(zip(entries_df["shares"], entries_df["price"]))
    ]
    colors = plt.cm.Set3(np.linspace(0, 1, len(weights)))
    ax2.pie(
        weights,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.6, edgecolor="white"),
    )
    ax2.set_title("Position Breakdown by Entry", fontsize=14, fontweight="bold")
    st.pyplot(fig2)
    plt.close(fig2)

# ------------------------
# Helper: simple outcome probabilities (your original idea)
# ------------------------
def historical_outcome_probabilities(data, entries_df, horizon=700, gain_threshold=0.05):
    results = []
    for _, entry in entries_df.iterrows():
        entry_date = pd.to_datetime(entry["date"])
        avg_cost_local = entry["price"]

        if entry_date < data.index[0] or entry_date > data.index[-1]:
            continue

        window = data.loc[entry_date : entry_date + pd.Timedelta(days=horizon)]
        if window.empty:
            continue

        breakeven = (window["Close"] >= avg_cost_local).any()
        gain = (window["Close"] >= avg_cost_local * (1 + gain_threshold)).any()
        loss = not breakeven

        results.append(
            {
                "entry_date": entry_date,
                "breakeven": breakeven,
                "gain": gain,
                "loss": loss,
            }
        )

    if not results:
        return {
            "Chance of Breakeven": 0.0,
            "Chance of Retaining Losses": 0.0,
            "Chance of Gaining a Bit": 0.0,
        }

    df_res = pd.DataFrame(results)
    probs = {
        "Chance of Breakeven": df_res["breakeven"].mean() * 100,
        "Chance of Retaining Losses": df_res["loss"].mean() * 100,
        "Chance of Gaining a Bit": df_res["gain"].mean() * 100,
    }
    return probs

# ------------------------
# New: full-history bootstrap breakeven probabilities
# ------------------------
def full_history_bootstrap_probabilities(
    data,
    horizon=180,
    gain_threshold=0.05,
    step=1,
):
    """
    For every possible historical entry date (with stride `step`),
    check if a buy-and-hold breaks even / gains within `horizon` days.
    """
    results = []
    idx = data.index

    # last possible entry date so that we have horizon days ahead
    last_entry_index = len(idx) - 1
    for i in range(0, last_entry_index, step):
        entry_date = idx[i]
        entry_price = data.loc[entry_date, "Close"]

        window_end_date = entry_date + pd.Timedelta(days=horizon)
        window = data.loc[entry_date:window_end_date]
        if window.empty:
            continue

        breakeven = (window["Close"] >= entry_price).any()
        gain = (window["Close"] >= entry_price * (1 + gain_threshold)).any()
        loss = not breakeven

        results.append((breakeven, gain, loss))

    if not results:
        return {
            "Chance of Breakeven": 0.0,
            "Chance of Retaining Losses": 0.0,
            "Chance of Gaining a Bit": 0.0,
        }

    arr = np.array(results, dtype=int)
    return {
        "Chance of Breakeven": arr[:, 0].mean() * 100,
        "Chance of Retaining Losses": arr[:, 2].mean() * 100,
        "Chance of Gaining a Bit": arr[:, 1].mean() * 100,
    }

# ------------------------
# New: regime labels (simple filters)
# ------------------------
def compute_regimes(data, lookback=60):
    """
    Simple regime classifier using rolling return and volatility:
      - bull: positive rolling return
      - bear: negative rolling return
      - high_vol: top 30% volatility
      - low_vol: bottom 30% volatility
    """
    df = data.copy()
    returns = df["Close"].pct_change()
    df["roll_ret"] = returns.rolling(lookback).mean()
    df["roll_vol"] = returns.rolling(lookback).std()
    df["regime"] = "neutral"

    df.loc[df["roll_ret"] > 0, "regime"] = "bull"
    df.loc[df["roll_ret"] < 0, "regime"] = "bear"

    vol_thresh_high = df["roll_vol"].quantile(0.7)
    vol_thresh_low = df["roll_vol"].quantile(0.3)
    df.loc[df["roll_vol"] >= vol_thresh_high, "vol_regime"] = "high_vol"
    df.loc[df["roll_vol"] <= vol_thresh_low, "vol_regime"] = "low_vol"
    df["vol_regime"] = df["vol_regime"].fillna("mid_vol")
    return df

def regime_conditioned_bootstrap(data, horizon=180, gain_threshold=0.05):
    """
    Run full-history bootstrap probabilities separately for bull/bear
    and high/low vol regimes (entry-day regime).
    """
    df_reg = compute_regimes(data)
    results = {}

    for regime_label in ["bull", "bear"]:
        mask = df_reg["regime"] == regime_label
        subset = df_reg.loc[mask]
        if subset.empty:
            results[f"{regime_label}_breakeven"] = np.nan
            results[f"{regime_label}_loss"] = np.nan
            results[f"{regime_label}_gain"] = np.nan
            continue

        sub_probs = full_history_bootstrap_probabilities(
            subset, horizon=horizon, gain_threshold=gain_threshold, step=1
        )
        results[f"{regime_label}_breakeven"] = sub_probs["Chance of Breakeven"]
        results[f"{regime_label}_loss"] = sub_probs["Chance of Retaining Losses"]
        results[f"{regime_label}_gain"] = sub_probs["Chance of Gaining a Bit"]

    for regime_label in ["high_vol", "low_vol"]:
        mask = df_reg["vol_regime"] == regime_label
        subset = df_reg.loc[mask]
        if subset.empty:
            results[f"{regime_label}_breakeven"] = np.nan
            results[f"{regime_label}_loss"] = np.nan
            results[f"{regime_label}_gain"] = np.nan
            continue

        sub_probs = full_history_bootstrap_probabilities(
            subset, horizon=horizon, gain_threshold=gain_threshold, step=1
        )
        results[f"{regime_label}_breakeven"] = sub_probs["Chance of Breakeven"]
        results[f"{regime_label}_loss"] = sub_probs["Chance of Retaining Losses"]
        results[f"{regime_label}_gain"] = sub_probs["Chance of Gaining a Bit"]

    return results

# ------------------------
# Historical probability section
# ------------------------
st.header("Historical Breakeven Probabilities")

horizon_days = st.slider("Look-ahead Horizon (days)", 30, 720, 365)
gain_threshold = st.slider("Gain threshold (%) for 'Gaining a Bit'", 1.0, 50.0, 5.0) / 100.0

with st.expander("Per-entry probabilities (your actual entries)"):
    probs_entries = historical_outcome_probabilities(
        data, entries_df, horizon=horizon_days, gain_threshold=gain_threshold
    )
    for k, v in probs_entries.items():
        st.metric(k, f"{v:.1f}%")

with st.expander("Full-history bootstrap (all historical entry dates)"):
    probs_full = full_history_bootstrap_probabilities(
        data, horizon=horizon_days, gain_threshold=gain_threshold, step=1
    )
    for k, v in probs_full.items():
        st.metric(f"{k} (Full History)", f"{v:.1f}%")

with st.expander("Regime-conditioned bootstrap (bull/bear & vol regimes)"):
    reg_probs = regime_conditioned_bootstrap(
        data, horizon=horizon_days, gain_threshold=gain_threshold
    )
    # show as a small table for clarity
    reg_df = pd.DataFrame(
        {
            "Regime": ["bull", "bear", "high_vol", "low_vol"],
            "Breakeven %": [
                reg_probs.get("bull_breakeven"),
                reg_probs.get("bear_breakeven"),
                reg_probs.get("high_vol_breakeven"),
                reg_probs.get("low_vol_breakeven"),
            ],
            "Retain Loss %": [
                reg_probs.get("bull_loss"),
                reg_probs.get("bear_loss"),
                reg_probs.get("high_vol_loss"),
                reg_probs.get("low_vol_loss"),
            ],
            "Gain a Bit %": [
                reg_probs.get("bull_gain"),
                reg_probs.get("bear_gain"),
                reg_probs.get("high_vol_gain"),
                reg_probs.get("low_vol_gain"),
            ],
        }
    )
    st.dataframe(reg_df.style.format("{:.1f}"))

# ------------------------
# Monte Carlo: historical (GBM) and block bootstrap
# ------------------------
st.header("Position Assessment & Monte Carlo Simulation")

returns = data["Close"].pct_change().dropna()
mu = returns.mean() * 252
sigma = returns.std() * np.sqrt(252)

c1, c2 = st.columns(2)
c1.metric("Annualized Return (GBM)", f"{mu*100:.1f}%")
c2.metric("Annualized Volatility", f"{sigma*100:.1f}%")

days = st.slider("Forecast Days", 30, 365, 90)
num_sims = st.slider("Monte Carlo Simulations", 1000, 20000, 5000)

@st.cache_data
def mc_gbm_paths(current_price, mu, sigma, days, num_sims):
    dt = 1 / 252
    paths = np.zeros((days + 1, num_sims))
    paths[0] = current_price
    for t in range(1, days + 1):
        rand = np.random.standard_normal(num_sims)
        paths[t] = paths[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rand)
    return paths

@st.cache_data
def mc_block_bootstrap_paths(data, days, num_sims, block=5):
    """
    Block-bootstrap Monte Carlo using historical returns.
    Preserves short-term autocorrelation by resampling blocks of returns. [web:23]
    """
    rets = data["Close"].pct_change().dropna().values
    n = len(rets)
    paths = np.zeros((days + 1, num_sims))
    paths[0] = data["Close"].iloc[-1]

    for j in range(num_sims):
        resampled = []
        while len(resampled) < days:
            start = np.random.randint(0, max(1, n - block))
            resampled.extend(rets[start : start + block])
        resampled = np.array(resampled[:days])
        prices = paths[0, j] * np.cumprod(1 + resampled)
        paths[1:, j] = prices

    return paths

mc_method = st.radio(
    "Monte Carlo Method",
    ["Geometric Brownian Motion (GBM)", "Block-Bootstrap (Historical Paths)"],
)

if mc_method == "Geometric Brownian Motion (GBM)":
    paths = mc_gbm_paths(current_price, mu, sigma, days, num_sims)
else:
    paths = mc_block_bootstrap_paths(data, days, num_sims, block=5)

# Plot paths and distribution
fig3, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 9), height_ratios=[3, 1])

sample_paths = min(50, num_sims)
for i in range(sample_paths):
    ax3.plot(range(days + 1), paths[:, i], color="gray", alpha=0.3, linewidth=0.5)
ax3.plot(range(days + 1), paths[:, -1], color="blue", linewidth=2, label="Sample Path")

percentiles = np.percentile(paths[-1], [5, 25, 50, 75, 95])
ax3.axhline(percentiles[2], color="red", linestyle="--", linewidth=2, label=f"Median: ${percentiles[2]:.2f}")
ax3.set_title(f"Monte Carlo Price Simulation ({mc_method})", fontsize=14, fontweight="bold")
ax3.set_xlabel("Days")
ax3.set_ylabel("Price ($)")
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4.hist(paths[-1], bins=50, alpha=0.7, color="skyblue", edgecolor="black", density=True)
ax4.axvline(percentiles[2], color="red", linestyle="--", linewidth=2, label=f"Median: ${percentiles[2]:.2f}")
ax4.set_title("Final Price Distribution (Density)")
ax4.set_xlabel("Price ($)")
ax4.legend()

plt.tight_layout()
st.pyplot(fig3)
plt.close(fig3)

# ------------------------
# Action scenarios
# ------------------------
st.header("Action Recommendations")

target_price = percentiles[2]
recovery_needed = (avg_cost - current_price) / current_price * 100 if avg_cost > current_price else 0

if pnl_pct > 20:
    st.success(f"**HOLD or TRIM**: +{pnl_pct:.1f}% gain. Median forecast ${target_price:.2f}.")
elif pnl_pct > 0:
    upside_pct = (target_price - current_price) / current_price * 100
    st.info(f"**HOLD**: +{pnl_pct:.1f}% gain. Forecast upside: {upside_pct:+.1f}%.")
elif pnl_pct > -10:
    st.warning(f"**HOLD or AVERAGE DOWN**: -{abs(pnl_pct):.1f}% loss. Needs {recovery_needed:.1f}% recovery.")
else:
    st.error(f"**CUT LOSSES**: -{abs(pnl_pct):.1f}% loss. Forecast recovery uncertain.")

portfolio_size = 20000
position_value = total_shares * current_price
allocation_pct = (position_value / portfolio_size) * 100 if portfolio_size > 0 else 0
st.metric("Portfolio Allocation", f"{allocation_pct:.1f}% of $20K")

st.caption(
    "GBM Monte Carlo uses the Geometric Brownian Motion update "
    "S(t+Δt) = S(t) × exp[(μ − ½σ²)Δt + σ√Δt × Z] where Z ~ N(0,1). [web:16][web:19]"
)
st.caption(
    "Block-bootstrap Monte Carlo resamples historical return blocks to preserve short-term autocorrelation in returns. [web:23]"
)
