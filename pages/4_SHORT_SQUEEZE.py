import os
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf
import streamlit as st


# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="Short Squeeze & Fear–Greed Dashboard",
    layout="wide",
)


# -----------------------------
# Helpers – data
# -----------------------------
@st.cache_data(show_spinner=False)
def get_price_history(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    if data.empty:
        return data
    data = data.rename(columns=str.lower)
    return data


@st.cache_data(show_spinner=False)
def get_fear_greed_history(start_date: str = "2021-01-01") -> pd.DataFrame:
    """
    Fetch CNN Fear & Greed Index history using public JSON endpoint.
    """
    base_url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    url = f"{base_url}/{start_date}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    fg = pd.DataFrame(data["fear_and_greed_historical"]["data"])
    # CNN uses ms since epoch; convert to date
    fg["date"] = pd.to_datetime(fg["x"] // 1000, unit="s")
    fg = fg.rename(columns={"y": "fear_greed"})
    fg = fg[["date", "fear_greed"]].sort_values("date")
    fg.set_index("date", inplace=True)
    return fg


def classify_fear_greed(value: int) -> str:
    if value <= 25:
        return "Extreme Fear"
    if value <= 44:
        return "Fear"
    if value <= 55:
        return "Neutral"
    if value <= 74:
        return "Greed"
    return "Extreme Greed"


# -----------------------------
# Placeholder: short-interest API
# -----------------------------
@st.cache_data(show_spinner=False)
def get_short_interest_stub(ticker: str) -> dict:
    """
    Replace this stub with a real short-interest provider (e.g., Fintel, ORTEX, FINRA + your logic).
    Must return:
        {
            "short_float_pct": float (0-100),
            "days_to_cover": float,
            "borrow_fee_pct": float or None
        }
    """
    # TODO: integrate your provider here.
    # For now, return dummy values so the app runs.
    return {
        "short_float_pct": None,
        "days_to_cover": None,
        "borrow_fee_pct": None,
    }

def compute_squeeze_score(row,
                          w_short=0.4,
                          w_dtc=0.3,
                          w_mom=0.2,
                          w_vol=0.1) -> float:
    """
    Composite short squeeze score 0-100 with safe scalar extraction
    """

    def safe_scalar(val):
        # Convert single-element Series to scalar or return val unchanged
        if isinstance(val, pd.Series):
            if len(val) == 1:
                return val.item()
            else:
                # Multiple elements - not expected, fallback to NaN
                return float("nan")
        return val

    short_float_pct = safe_scalar(row.get("short_float_pct"))
    days_to_cover = safe_scalar(row.get("days_to_cover"))
    ret_10d = safe_scalar(row.get("ret_10d"))
    rel_volume = safe_scalar(row.get("rel_volume"))

    score = 0.0

    if pd.notna(short_float_pct):
        score += w_short * min(short_float_pct, 100)

    if pd.notna(days_to_cover):
        score += w_dtc * min(days_to_cover * 5, 100)  # scaling as before

    if pd.notna(ret_10d):
        score += w_mom * max(ret_10d * 100, -100)  # convert fraction to percentage points

    if pd.notna(rel_volume):
        score += w_vol * min(rel_volume * 20, 100)  # 2x volume boosts score

    return round(score, 2)


# -----------------------------
# Layout
# -----------------------------
st.title("Short Squeeze & Fear–Greed Dashboard")

tab1, tab2 = st.tabs(["Short Squeeze Scanner", "Fear–Greed & Momentum"])


# -----------------------------
# Tab 1 – Short Squeeze Scanner
# -----------------------------
with tab1:
    st.header("Short Squeeze Scanner")

    st.markdown(
        "Paste a list of stock tickers (e.g., `GME, AMC, TSLA`) and rank them by short-squeeze potential. "
        "Short-interest fields are wired to a stub for now – plug in your provider where indicated in the code."
    )
    
    tickers_input = st.text_input(
        "Tickers (comma separated, no quotes):",
        value="GME, AMC, TSLA",
    )

    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
    with col_w1:
        w_short = st.slider("Weight – Short % Float", 0.0, 1.0, 0.4, 0.05)
    with col_w2:
        w_dtc = st.slider("Weight – Days to Cover", 0.0, 1.0, 0.3, 0.05)
    with col_w3:
        w_mom = st.slider("Weight – 10D Momentum", 0.0, 1.0, 0.2, 0.05)
    with col_w4:
        w_vol = st.slider("Weight – Rel Volume", 0.0, 1.0, 0.1, 0.05)

    # Normalise weights so they sum to 1 (unless all zeros)
    total_w = w_short + w_dtc + w_mom + w_vol
    if total_w > 0:
        w_short, w_dtc, w_mom, w_vol = [w / total_w for w in (w_short, w_dtc, w_mom, w_vol)]

    if st.button("Run Scan"):
        raw_tickers = [t.strip().upper() for t in tickers_input.replace(",", " ").split()]
        tickers = [t for t in raw_tickers if t]

        if not tickers:
            st.warning("Please provide at least one ticker.")
        else:
            rows = []
            for t in tickers:
                price = get_price_history(t, period="3mo", interval="1d")
                if price.empty:
                    rows.append(
                        dict(
                            ticker=t,
                            short_float_pct=None,
                            days_to_cover=None,
                            borrow_fee_pct=None,
                            ret_10d=None,
                            rel_volume=None,
                        )
                    )
                    continue

                # 10-day momentum
                if len(price) >= 11:
                    ret_10d = price["close"].iloc[-1] / price["close"].iloc[-11] - 1
                else:
                    ret_10d = None

                # Relative volume (last day vs 20D avg)
                vol_20d = price["volume"].tail(21)[:-1].mean()
                if isinstance(vol_20d, pd.Series):
                    vol_20d = vol_20d.item()  # convert 1-element series to scalar
                
                if vol_20d is None or vol_20d == 0:
                    rel_vol = None
                else:
                    rel_vol = float(price["volume"].iloc[-1] / vol_20d)

                si = get_short_interest_stub(t)

                rows.append(
                    dict(
                        ticker=t,
                        short_float_pct=si.get("short_float_pct"),
                        days_to_cover=si.get("days_to_cover"),
                        borrow_fee_pct=si.get("borrow_fee_pct"),
                        ret_10d=ret_10d,
                        rel_volume=rel_vol,
                    )
                )

            df = pd.DataFrame(rows)
            if not df.empty:
                df["squeeze_score"] = df.apply(
                    lambda r: compute_squeeze_score(
                        r, w_short=w_short, w_dtc=w_dtc, w_mom=w_mom, w_vol=w_vol
                    ),
                    axis=1,
                )
                df = df.sort_values("squeeze_score", ascending=False)

                st.subheader("Ranked Short-Squeeze Candidates")
                st.table(df.style.format({
                    "short_float_pct": "{:.1f}",
                    "days_to_cover": "{:.2f}",
                    "borrow_fee_pct": "{:.2f}",
                    "ret_10d": "{:.2%}",
                    "rel_volume": "{:.2f}",
                    "squeeze_score": "{:.2f}",
                }))

            else:
                st.info("No data returned for the given tickers.")


# -----------------------------
# Tab 2 – Fear–Greed & Momentum
# -----------------------------
with tab2:
    st.header("Fear–Greed Index & Benchmark Momentum")

    st.markdown(
        "This tab shows CNN's stock-market Fear & Greed Index history alongside price momentum "
        "for a chosen benchmark (e.g., S&P 500 [finance:S&P 500] via SPY [finance:SPDR S&P 500 ETF Trust])."
    )

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        benchmark = st.text_input("Benchmark ticker", value="SPY")
    with col_b2:
        fg_start = st.date_input(
            "Fear–Greed history from",
            value=datetime.today().date() - timedelta(days=365),
        )

    if st.button("Load Sentiment & Momentum"):
        fg_df = get_fear_greed_history(start_date=fg_start.strftime("%Y-%m-%d"))
        bench_df = get_price_history(benchmark, period="1y", interval="1d")

        if fg_df.empty:
            st.warning("No Fear–Greed data returned.")
        else:
            current_fg = int(fg_df["fear_greed"].iloc[-1])
            fg_label = classify_fear_greed(current_fg)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Current Fear–Greed", value=current_fg, delta=None)
            with c2:
                st.metric("Regime", value=fg_label)
            with c3:
                st.metric("Last Update", value=fg_df.index[-1].strftime("%Y-%m-%d"))

            st.subheader("Fear–Greed Index History")
            st.line_chart(fg_df["fear_greed"])

        if bench_df.empty:
            st.warning(f"No price data for benchmark {benchmark}.")
        else:
            # Momentum windows
            last_price = bench_df["close"].iloc[-1]
            def pct_ret(days: int) -> float | None:
                if len(bench_df) > days:
                    return last_price / bench_df["close"].iloc[-days - 1] - 1
                return None

            m_1m = pct_ret(21)
            m_3m = pct_ret(63)
            m_6m = pct_ret(126)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("1M Return", value=f"{m_1m:.2%}" if m_1m is not None else "N/A")
            with c2:
                st.metric("3M Return", value=f"{m_3m:.2%}" if m_3m is not None else "N/A")
            with c3:
                st.metric("6M Return", value=f"{m_6m:.2%}" if m_6m is not None else "N/A")

            st.subheader(f"{benchmark} Price History")
            st.line_chart(bench_df["close"])
