from pathlib import Path
from typing import Dict, List

import math
import pandas as pd
import streamlit as st
import hashlib

st.set_page_config(
    page_title="Portfolio Tracker & Analysis",
    page_icon=":bar_chart:",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"

# -----------------------------------------------------------------------------
# Data loading


@st.cache_data
def load_ticker_universe() -> Dict[str, List[str]]:
    """Load static ticker universes from local CSV files."""
    universes = {}
    for name, file_name in {
        "CDAX": "cdax_tickers.csv",
        "Nasdaq-100": "nasdaq100_tickers.csv",
    }.items():
        df = pd.read_csv(DATA_DIR / file_name)
        universes[name] = df["Ticker"].dropna().astype(str).unique().tolist()
    return universes


@st.cache_data
def get_gdp_data() -> pd.DataFrame:
    raw_gdp_df = pd.read_csv(DATA_DIR / "gdp_data.csv")

    min_year = 1960
    max_year = 2022
    gdp_df = raw_gdp_df.melt(
        ["Country Code"],
        [str(x) for x in range(min_year, max_year + 1)],
        "Year",
        "GDP",
    )
    gdp_df["Year"] = pd.to_numeric(gdp_df["Year"])

    return gdp_df


# -----------------------------------------------------------------------------
# Stock scoring logic

SCORING_RULES = {
    "Revenue Growth 3Y %": [5, 10, 15, 20],
    "EPS Growth 3Y %": [5, 10, 15, 20],
    "ROE %": [10, 15, 20, 25],
    "Gross Margin %": [30, 40, 50, 60],
    "FCF Margin %": [5, 10, 15, 20],
    "Debt / Equity": [2.0, 1.2, 0.8, 0.4],
    "Forward PE": [40, 30, 22, 15],
}

SCORING_WEIGHTS = {
    "Revenue Growth 3Y %": 0.2,
    "EPS Growth 3Y %": 0.2,
    "ROE %": 0.15,
    "Gross Margin %": 0.1,
    "FCF Margin %": 0.15,
    "Debt / Equity": 0.1,
    "Forward PE": 0.1,
}


def compute_growth_from_series(series: pd.Series, years: int = 3) -> float | None:
    series = series.dropna().sort_index()
    if len(series) < years + 1:
        return None

    start = series.iloc[-(years + 1)]
    end = series.iloc[-1]
    if start <= 0:
        return None

    return (end / start - 1) * 100


def metric_to_score(metric: str, value: float | None) -> float:
    if value is None or pd.isna(value):
        return 0

    thresholds = SCORING_RULES[metric]
    if metric in {"Debt / Equity", "Forward PE"}:  # lower is better
        if value <= thresholds[3]:
            return 5
        if value <= thresholds[2]:
            return 4
        if value <= thresholds[1]:
            return 3
        if value <= thresholds[0]:
            return 2
        return 1

    if value >= thresholds[3]:
        return 5
    if value >= thresholds[2]:
        return 4
    if value >= thresholds[1]:
        return 3
    if value >= thresholds[0]:
        return 2
    return 1


def score_to_grade(total_score: float) -> str:
    if total_score >= 4.5:
        return "A+"
    if total_score >= 4.0:
        return "A"
    if total_score >= 3.5:
        return "B"
    if total_score >= 3.0:
        return "C"
    if total_score >= 2.0:
        return "D"
    return "E"


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_scored_stocks(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Build a scored table using deterministic offline metrics per ticker.

    In environments without market-data API access, this keeps the app usable.
    Replace this with a live provider integration later if needed.
    """
    rows = []

    for ticker in tickers:
        seed = int(hashlib.sha256(ticker.encode()).hexdigest()[:8], 16)

        def scale(min_v: float, max_v: float, shift: int) -> float:
            value = ((seed >> shift) & 1023) / 1023
            return min_v + (max_v - min_v) * value

        row = {
            "Ticker": ticker,
            "Name": ticker,
            "Revenue Growth 3Y %": round(scale(-5, 35, 0), 2),
            "EPS Growth 3Y %": round(scale(-10, 40, 3), 2),
            "ROE %": round(scale(2, 35, 6), 2),
            "Gross Margin %": round(scale(15, 75, 9), 2),
            "FCF Margin %": round(scale(-5, 30, 12), 2),
            "Debt / Equity": round(scale(0.0, 2.5, 15), 2),
            "Forward PE": round(scale(8, 55, 18), 2),
            "Market Cap": int(scale(1e9, 3e12, 21)),
        }

        metric_scores = []
        for metric, weight in SCORING_WEIGHTS.items():
            s = metric_to_score(metric, row.get(metric))
            row[f"Score:{metric}"] = s
            metric_scores.append(s * weight)

        total = sum(metric_scores)
        row["Score"] = round(total, 2)
        row["Grade"] = score_to_grade(total)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values(by=["Score", "Market Cap"], ascending=[False, False])


# -----------------------------------------------------------------------------
# UI

st.title("Stock Portfolio Tracker & Analysis")
st.caption(
    "Page 1 shows CDAX and Nasdaq-100 tables, graded and sorted with a transparent KPI model inspired by Turbo-Depot style ranking."
)

stock_tab, gdp_tab = st.tabs(["Stock Ranking", "GDP Dashboard (legacy)"])

with stock_tab:
    st.subheader("Universe ranking")
    st.write(
        "The score combines growth, quality, profitability and valuation metrics. You can tune the universe size to reduce loading time."
    )

    universes = load_ticker_universe()
    c1, c2 = st.columns(2)
    with c1:
        cdax_limit = st.slider("CDAX tickers to load", min_value=10, max_value=len(universes["CDAX"]), value=30)
    with c2:
        ndx_limit = st.slider("Nasdaq-100 tickers to load", min_value=10, max_value=len(universes["Nasdaq-100"]), value=40)

    with st.expander("Scoring model"):
        st.json({"thresholds": SCORING_RULES, "weights": SCORING_WEIGHTS})
        st.caption("Note: In this offline environment, metrics are deterministic demo values derived from ticker symbols. Connect a live data source for production use.")

    cdax_df = get_scored_stocks(tuple(universes["CDAX"][:cdax_limit]))
    ndx_df = get_scored_stocks(tuple(universes["Nasdaq-100"][:ndx_limit]))

    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown("#### CDAX")
        if cdax_df.empty:
            st.warning("No CDAX rows could be loaded. Data provider may be temporarily unavailable.")
        else:
            st.dataframe(
                cdax_df[[
                    "Grade",
                    "Score",
                    "Ticker",
                    "Name",
                    "Revenue Growth 3Y %",
                    "EPS Growth 3Y %",
                    "ROE %",
                    "Gross Margin %",
                    "FCF Margin %",
                    "Debt / Equity",
                    "Forward PE",
                ]],
                width="stretch",
                hide_index=True,
            )

    with tc2:
        st.markdown("#### Nasdaq-100")
        if ndx_df.empty:
            st.warning("No Nasdaq-100 rows could be loaded. Data provider may be temporarily unavailable.")
        else:
            st.dataframe(
                ndx_df[[
                    "Grade",
                    "Score",
                    "Ticker",
                    "Name",
                    "Revenue Growth 3Y %",
                    "EPS Growth 3Y %",
                    "ROE %",
                    "Gross Margin %",
                    "FCF Margin %",
                    "Debt / Equity",
                    "Forward PE",
                ]],
                width="stretch",
                hide_index=True,
            )

with gdp_tab:
    gdp_df = get_gdp_data()

    min_value = gdp_df["Year"].min()
    max_value = gdp_df["Year"].max()

    from_year, to_year = st.slider(
        "Which years are you interested in?",
        min_value=min_value,
        max_value=max_value,
        value=[min_value, max_value],
        key="gdp_year",
    )

    countries = gdp_df["Country Code"].unique()
    selected_countries = st.multiselect(
        "Which countries would you like to view?",
        countries,
        ["DEU", "FRA", "GBR", "BRA", "MEX", "JPN"],
    )

    filtered_gdp_df = gdp_df[
        (gdp_df["Country Code"].isin(selected_countries))
        & (gdp_df["Year"] <= to_year)
        & (from_year <= gdp_df["Year"])
    ]

    st.header("GDP over time", divider="gray")
    st.line_chart(filtered_gdp_df, x="Year", y="GDP", color="Country Code")

    first_year = gdp_df[gdp_df["Year"] == from_year]
    last_year = gdp_df[gdp_df["Year"] == to_year]

    st.header(f"GDP in {to_year}", divider="gray")
    cols = st.columns(4)

    for i, country in enumerate(selected_countries):
        col = cols[i % len(cols)]

        with col:
            first_gdp = first_year[first_year["Country Code"] == country]["GDP"].iat[0] / 1000000000
            last_gdp = last_year[last_year["Country Code"] == country]["GDP"].iat[0] / 1000000000

            if math.isnan(first_gdp):
                growth = "n/a"
                delta_color = "off"
            else:
                growth = f"{last_gdp / first_gdp:,.2f}x"
                delta_color = "normal"

            st.metric(
                label=f"{country} GDP",
                value=f"{last_gdp:,.0f}B",
                delta=growth,
                delta_color=delta_color,
            )
