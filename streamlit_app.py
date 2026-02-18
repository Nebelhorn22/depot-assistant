from pathlib import Path
from typing import Dict, List

import hashlib
import math

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Portfolio Tracker & Analysis",
    page_icon=":bar_chart:",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"
TRADING_DAYS_52W = 252


@st.cache_data
def load_ticker_universe() -> Dict[str, List[str]]:
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


def make_demo_price_history(ticker: str, periods: int = 330) -> pd.Series:
    """Deterministic pseudo-price series for offline/demo mode.

    This keeps the app functional where external market APIs are blocked.
    """
    seed = int(hashlib.sha256(ticker.encode()).hexdigest()[:8], 16)
    start_price = 20 + (seed % 140)
    drift = ((seed >> 5) % 40 - 12) / 10_000

    values = [float(start_price)]
    for i in range(1, periods):
        cyc = math.sin((i + (seed % 17)) / 17) * 0.007
        noise = (((seed >> (i % 20)) & 31) - 15) / 10_000
        ret = drift + cyc + noise
        values.append(max(1.0, values[-1] * (1 + ret)))

    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=periods)
    return pd.Series(values, index=idx, name=ticker)


def calc_turbo_metrics(close: pd.Series) -> dict[str, float | None]:
    """Compute the three Turbo-Depot metrics from a close series.

    - SMA-Ratio = (SMA20 / SMA200 - 1) * 100
    - Delta-SMA = (SMA200_today / SMA200_20d_ago - 1) * 100
    - 52-WHTR = (Close / 52W_High) + 0.25 * (Close / 52W_Low)
      (distance to 52W-Low intentionally weighted at 25%)
    """
    if close.dropna().shape[0] < 220:
        return {"SMA-Ratio %": None, "Delta-SMA %": None, "52-WHTR": None}

    sma20 = close.rolling(20).mean()
    sma200 = close.rolling(200).mean()

    last_close = close.iloc[-1]
    last_sma20 = sma20.iloc[-1]
    last_sma200 = sma200.iloc[-1]
    sma200_20d_ago = sma200.shift(20).iloc[-1]

    high_52w = close.tail(TRADING_DAYS_52W).max()
    low_52w = close.tail(TRADING_DAYS_52W).min()

    sma_ratio = None
    if pd.notna(last_sma20) and pd.notna(last_sma200) and last_sma200 != 0:
        sma_ratio = (last_sma20 / last_sma200 - 1) * 100

    delta_sma = None
    if pd.notna(last_sma200) and pd.notna(sma200_20d_ago) and sma200_20d_ago != 0:
        delta_sma = (last_sma200 / sma200_20d_ago - 1) * 100

    wht_ratio = None
    if pd.notna(last_close) and pd.notna(high_52w) and pd.notna(low_52w) and high_52w > 0 and low_52w > 0:
        wht_ratio = (last_close / high_52w) + 0.25 * (last_close / low_52w)

    return {
        "SMA-Ratio %": sma_ratio,
        "Delta-SMA %": delta_sma,
        "52-WHTR": wht_ratio,
    }


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def build_turbo_ranking(tickers: tuple[str, ...]) -> pd.DataFrame:
    rows = []

    for ticker in tickers:
        close = make_demo_price_history(ticker)
        metrics = calc_turbo_metrics(close)
        rows.append(
            {
                "Ticker": ticker,
                "Close": round(float(close.iloc[-1]), 2),
                "SMA20": round(float(close.rolling(20).mean().iloc[-1]), 2),
                "SMA200": round(float(close.rolling(200).mean().iloc[-1]), 2),
                "52W High": round(float(close.tail(TRADING_DAYS_52W).max()), 2),
                "52W Low": round(float(close.tail(TRADING_DAYS_52W).min()), 2),
                **metrics,
            }
        )

    df = pd.DataFrame(rows)

    for metric in ["SMA-Ratio %", "Delta-SMA %", "52-WHTR"]:
        df[f"Rank {metric}"] = df[metric].rank(ascending=False, method="min")

    df["Avg Rank"] = df[["Rank SMA-Ratio %", "Rank Delta-SMA %", "Rank 52-WHTR"]].mean(axis=1)
    df["Final Rank"] = df["Avg Rank"].rank(ascending=True, method="min").astype(int)

    return df.sort_values(["Final Rank", "Avg Rank", "Ticker"], ascending=[True, True, True])


st.title("Stock Portfolio Tracker & Analysis")
st.caption(
    "Erste Seite mit CDAX- und Nasdaq-100-Ranglisten nach der Turbo-Depot-Logik: "
    "SMA-Ratio, Delta-SMA und 52-WHTR, dann Durchschnittsrang und finaler Rang."
)

stock_tab, gdp_tab = st.tabs(["Stock Ranking", "GDP Dashboard (legacy)"])

with stock_tab:
    st.subheader("Turbo-Depot Rangliste")
    st.write(
        "Pro Index werden alle Aktien relativ bewertet: je Kennzahl ein Rang, dann Durchschnittsrang, "
        "anschließend finaler Rang (1 = beste Aktie)."
    )

    universes = load_ticker_universe()
    c1, c2 = st.columns(2)
    with c1:
        cdax_limit = st.slider("CDAX Titel", min_value=10, max_value=len(universes["CDAX"]), value=40)
    with c2:
        ndx_limit = st.slider("Nasdaq-100 Titel", min_value=10, max_value=len(universes["Nasdaq-100"]), value=40)

    with st.expander("Formeln & Hinweise"):
        st.markdown(
            """
            - **SMA-Ratio** = \((SMA20 / SMA200) - 1\) \* 100
            - **Delta-SMA** = \((SMA200\_heute / SMA200\_vor\_20T) - 1\) \* 100
            - **52-WHTR** = \((Kurs / 52W\_Hoch) + 0.25 \* (Kurs / 52W\_Tief)\)

            Danach je Kennzahl Rang (absteigend), dann Durchschnittsrang und finaler Rang.
            """
        )
        st.caption(
            "Hinweis: Externe Kursdaten sind in dieser Umgebung nicht erreichbar. "
            "Daher werden reproduzierbare Demo-Kursreihen pro Ticker genutzt."
        )

    cdax_df = build_turbo_ranking(tuple(universes["CDAX"][:cdax_limit]))
    ndx_df = build_turbo_ranking(tuple(universes["Nasdaq-100"][:ndx_limit]))

    display_cols = [
        "Final Rank",
        "Avg Rank",
        "Ticker",
        "SMA-Ratio %",
        "Rank SMA-Ratio %",
        "Delta-SMA %",
        "Rank Delta-SMA %",
        "52-WHTR",
        "Rank 52-WHTR",
        "Close",
        "SMA20",
        "SMA200",
        "52W High",
        "52W Low",
    ]

    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown("#### CDAX")
        st.dataframe(cdax_df[display_cols], width="stretch", hide_index=True)

    with tc2:
        st.markdown("#### Nasdaq-100")
        st.dataframe(ndx_df[display_cols], width="stretch", hide_index=True)

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
