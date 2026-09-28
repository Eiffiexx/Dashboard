from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = ["date", "ror", "type", "cnt", "aumbn"]
COLUMN_ALIASES = {
    "date": ["date", "month", "period", "as_of_date", "month_end", "monthend"],
    "ror": ["ror", "return", "returns", "monthly_return", "monthly_returns", "performance"],
    "type": ["type", "index", "index_name", "strategy", "category", "name"],
    "cnt": [
        "cnt",
        "count",
        "observations",
        "observation_count",
        "num_observations",
        "managers",
        "manager_count",
    ],
    "aumbn": ["aumbn", "aum_bn", "aum", "assets_bn", "assets", "aum_billion"],
}
REQUIRED_COLUMNS = {"date", "ror", "type"}
MAIN_INDICES = ["CTA", "Crypto", "Equity LS", "Market Neutral", "Risk Parity"]


def normalize_column_name(column_name: str) -> str:
    return (
        str(column_name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized_to_original = {
        normalize_column_name(column): column for column in df.columns
    }
    rename_map: dict[str, str] = {}

    for canonical_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalized_to_original:
                rename_map[normalized_to_original[alias]] = canonical_name
                break

    standardized = df.rename(columns=rename_map).copy()
    missing = REQUIRED_COLUMNS.difference(standardized.columns)
    if missing:
        accepted = {name: COLUMN_ALIASES[name] for name in sorted(missing)}
        raise ValueError(
            "The CSV is missing required columns. "
            f"Missing: {sorted(missing)}. Accepted names: {accepted}"
        )

    if "cnt" not in standardized.columns:
        standardized["cnt"] = np.nan
    if "aumbn" not in standardized.columns:
        standardized["aumbn"] = np.nan

    return standardized[CANONICAL_COLUMNS]


def clean_index_data(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for column in ["ror", "cnt", "aumbn"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    valid_returns = df["ror"].abs().dropna()
    if not valid_returns.empty and valid_returns.median() > 1:
        df["ror"] = df["ror"] / 100

    df["type"] = df["type"].astype(str).str.strip()
    df = df.dropna(subset=["date", "type", "ror"])
    df = df[df["type"].ne("") & df["type"].str.lower().ne("nan")]
    df = df.drop_duplicates(subset=["date", "type"], keep="last")
    df = df.sort_values(["type", "date"]).reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df


def load_index_data(path: str | Path) -> pd.DataFrame:
    return clean_index_data(pd.read_csv(path))


def cumulative_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return math.nan
    return float((1 + returns).prod() - 1)


def annualized_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return math.nan
    years = len(returns) / 12
    compounded = float((1 + returns).prod())
    if compounded <= 0 or years <= 0:
        return math.nan
    return float(compounded ** (1 / years) - 1)


def annualized_volatility(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) < 2:
        return math.nan
    return float(returns.std(ddof=1) * math.sqrt(12))


def drawdown_series(returns: pd.Series) -> pd.Series:
    cumulative = (1 + returns.fillna(0)).cumprod()
    return cumulative / cumulative.cummax() - 1


def max_drawdown(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return math.nan
    return float(drawdown_series(returns).min())


def build_summary_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for index_name, group in df.groupby("type", sort=True):
        group = group.sort_values("date")
        latest_date = group["date"].max()
        latest_row = group.tail(1).iloc[0]
        prior_rows = group[group["date"] < latest_date]
        prior_row = prior_rows.tail(1).iloc[0] if not prior_rows.empty else None
        current_year = int(latest_date.year)
        ytd_returns = group[group["date"].dt.year == current_year]["ror"]
        trailing_12m = group[group["date"] > latest_date - pd.DateOffset(months=12)]["ror"]
        ann_return = annualized_return(group["ror"])
        ann_vol = annualized_volatility(group["ror"])
        latest_obs = latest_row["cnt"]
        latest_aum = latest_row["aumbn"]
        prior_obs = prior_row["cnt"] if prior_row is not None else math.nan
        prior_aum = prior_row["aumbn"] if prior_row is not None else math.nan

        rows.append(
            {
                "Index": index_name,
                "Latest date": latest_date,
                "Latest": latest_row["ror"],
                "YTD": cumulative_return(ytd_returns),
                "12M": cumulative_return(trailing_12m),
                "Ann. return": ann_return,
                "Ann. volatility": ann_vol,
                "Return / risk": (
                    ann_return / ann_vol
                    if not pd.isna(ann_vol) and ann_vol != 0
                    else math.nan
                ),
                "Max drawdown": max_drawdown(group["ror"]),
                "Managers": latest_obs,
                "Manager change": (
                    latest_obs / prior_obs - 1
                    if not pd.isna(latest_obs)
                    and not pd.isna(prior_obs)
                    and prior_obs != 0
                    else math.nan
                ),
                "AUM ($bn)": latest_aum,
                "AUM change": (
                    latest_aum / prior_aum - 1
                    if not pd.isna(latest_aum)
                    and not pd.isna(prior_aum)
                    and prior_aum != 0
                    else math.nan
                ),
                "Months": len(group),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary = summary.sort_values("YTD", ascending=False).reset_index(drop=True)
    summary.insert(0, "YTD rank", range(1, len(summary) + 1))
    return summary


def format_percent(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "—"
    return f"{value * 100:.{digits}f}%"


def format_number(value: float, digits: int = 0) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:,.{digits}f}"


def format_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary

    display = summary.copy()
    percent_columns = [
        "Latest",
        "YTD",
        "12M",
        "Ann. return",
        "Ann. volatility",
        "Max drawdown",
        "Manager change",
        "AUM change",
    ]
    for column in percent_columns:
        display[column] = display[column].map(format_percent)
    display["Return / risk"] = display["Return / risk"].map(
        lambda value: "—" if pd.isna(value) else f"{value:.2f}"
    )
    display["Managers"] = display["Managers"].map(format_number)
    display["AUM ($bn)"] = display["AUM ($bn)"].map(
        lambda value: format_number(value, 1)
    )
    display["Latest date"] = pd.to_datetime(display["Latest date"]).dt.strftime("%b %Y")
    return display


def monthly_return_table(group: pd.DataFrame) -> pd.DataFrame:
    if group.empty:
        return pd.DataFrame()

    table = group.pivot_table(index="year", columns="month", values="ror", aggfunc="last")
    table = table.reindex(columns=range(1, 13))
    table.columns = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    ytd = group.groupby("year")["ror"].apply(cumulative_return)
    managers = group.groupby("year")["cnt"].mean()
    table["YTD"] = ytd
    table["Avg managers"] = managers
    table = table.sort_index(ascending=False).reset_index().rename(columns={"year": "Year"})

    for column in [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
        "YTD",
    ]:
        table[column] = table[column].map(format_percent)
    table["Avg managers"] = table["Avg managers"].map(format_number)
    return table


def make_sample_data(seed: int = 17) -> pd.DataFrame:
    """Create deterministic demo data so the app runs before private data is added."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-31", periods=104, freq="ME")
    profiles = {
        "CTA": (0.006, 0.035, 580, 34.0),
        "Crypto": (0.013, 0.115, 175, 12.5),
        "Equity LS": (0.007, 0.041, 920, 82.0),
        "Market Neutral": (0.004, 0.020, 410, 28.0),
        "Risk Parity": (0.005, 0.028, 240, 19.0),
        "Global Macro": (0.006, 0.032, 360, 31.0),
        "Fixed Income Arbitrage": (0.004, 0.018, 205, 22.0),
    }
    market_factor = rng.normal(0.003, 0.025, len(dates))
    rows: list[dict] = []

    for position, (name, (mean, volatility, managers, aum)) in enumerate(profiles.items()):
        loading = 0.15 + position * 0.08
        idiosyncratic = rng.normal(0, volatility, len(dates))
        returns = mean + loading * market_factor + idiosyncratic
        reporting_cycle = 1 - 0.07 * np.sin(np.linspace(0, 8 * np.pi, len(dates)))
        counts = np.maximum(
            25,
            managers * reporting_cycle + rng.normal(0, managers * 0.025, len(dates)),
        ).round()
        aum_path = aum * np.cumprod(1 + np.clip(returns * 0.25, -0.08, 0.08))
        aum_path *= 1 + rng.normal(0, 0.015, len(dates))

        for date, monthly_return, count, aum_value in zip(
            dates, returns, counts, aum_path, strict=True
        ):
            rows.append(
                {
                    "date": date,
                    "ror": float(np.clip(monthly_return, -0.45, 0.65)),
                    "type": name,
                    "cnt": float(count),
                    "aumbn": float(max(aum_value, 0.1)),
                }
            )

    return clean_index_data(pd.DataFrame(rows))
