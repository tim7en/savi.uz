"""Compare SPY drawdown with large-cap names that have usable earnings history.

The comparison universe is the locally cached trad-FI ticker list.  A company is
kept when its latest cached Alpha Vantage overview reports market capitalisation
of at least $10 billion, its earnings file contains at least 20 usable quarterly
EPS reports, and its monthly adjusted-price file contains at least 60 months.

The output is descriptive, not a point-in-time backtest: the market-cap screen
and universe membership are current snapshots and therefore contain look-ahead
and survivorship bias.  Drawdowns are computed from each instrument's own
monthly adjusted-close high within the locally available sample.  SPY monthly
adjusted prices are read from Yahoo's chart endpoint and retained with the
generated results.
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from pathlib import Path

import pandas as pd


YAHOO_CHART = (
    "https://query1.finance.yahoo.com/v8/finance/chart/SPY"
    "?period1=915148800&period2=1893456000&interval=1mo"
    "&events=div%2Csplits&includeAdjustedClose=true"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe", type=Path,
        default=Path("data/intraday/tradfi_universe.json"),
    )
    parser.add_argument(
        "--fundamentals", type=Path,
        default=Path("data/data/sp500_data"),
    )
    parser.add_argument("--min-market-cap", type=float, default=10_000_000_000)
    parser.add_argument("--min-quarters", type=int, default=20)
    parser.add_argument("--min-months", type=int, default=60)
    parser.add_argument(
        "--out", type=Path,
        default=Path("out/strategy/spy_large_cap_drawdown"),
    )
    return parser.parse_args(argv)


def payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["data"]


def usable_quarters(path: Path) -> list[dict]:
    rows = payload(path).get("quarterlyEarnings", [])
    out = []
    for row in rows:
        day = str(row.get("reportedDate", ""))[:10]
        eps = str(row.get("reportedEPS", "")).strip().lower()
        if len(day) == 10 and eps not in {"", "-", "none", "null"}:
            out.append(row)
    return out


def monthly_prices(path: Path) -> pd.Series:
    rows = payload(path)["Monthly Adjusted Time Series"]
    values = {}
    for day, row in rows.items():
        try:
            price = float(row["5. adjusted close"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(price) and price > 0:
            values[pd.Period(day[:10], freq="M")] = price
    return pd.Series(values, dtype=float).sort_index()


def spy_monthly() -> pd.Series:
    request = urllib.request.Request(
        YAHOO_CHART, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        chart = json.loads(response.read())["chart"]["result"][0]
    timestamps = pd.to_datetime(chart["timestamp"], unit="s", utc=True)
    indicators = chart["indicators"]
    adjusted = indicators.get("adjclose", [{}])[0].get("adjclose")
    if adjusted is None:
        adjusted = indicators["quote"][0]["close"]
    values = {}
    for stamp, price in zip(timestamps, adjusted):
        if price is not None and math.isfinite(float(price)) and float(price) > 0:
            values[stamp.tz_localize(None).to_period("M")] = float(price)
    return pd.Series(values, dtype=float).sort_index()


def current_partial_month(series: pd.Series, source_path: Path) -> pd.Period | None:
    """Return the last period when the source's latest observation is partial."""
    rows = payload(source_path)["Monthly Adjusted Time Series"]
    latest = max(pd.Timestamp(day[:10]) for day in rows)
    return latest.to_period("M") if latest.day < 25 else None


def main(argv=None) -> int:
    args = parse_args(argv)
    tickers = json.loads(args.universe.read_text(encoding="utf-8"))["tickers"]

    coverage = {"overview": 0, "earnings": 0, "monthly_prices": 0}
    selected = []
    price_series: dict[str, pd.Series] = {}
    partial_periods = []

    for ticker in tickers:
        overview_path = args.fundamentals / f"{ticker}_overview.json"
        earnings_path = args.fundamentals / f"{ticker}_earnings.json"
        prices_path = args.fundamentals / f"{ticker}_time_series_monthly.json"
        coverage["overview"] += overview_path.exists()
        coverage["earnings"] += earnings_path.exists()
        coverage["monthly_prices"] += prices_path.exists()
        if not (overview_path.exists() and earnings_path.exists()
                and prices_path.exists()):
            continue
        try:
            overview = payload(overview_path)
            market_cap = float(overview.get("MarketCapitalization") or 0)
            quarters = usable_quarters(earnings_path)
            prices = monthly_prices(prices_path)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if (market_cap < args.min_market_cap
                or len(quarters) < args.min_quarters
                or len(prices) < args.min_months):
            continue
        selected.append({
            "ticker": ticker,
            "name": overview.get("Name"),
            "market_cap": market_cap,
            "earnings_quarters": len(quarters),
            "earnings_first": quarters[-1].get("fiscalDateEnding"),
            "earnings_last": quarters[0].get("fiscalDateEnding"),
            "price_months": len(prices),
            "price_first": str(prices.index.min()),
            "price_last": str(prices.index.max()),
        })
        price_series[ticker] = prices
        partial = current_partial_month(prices, prices_path)
        if partial is not None:
            partial_periods.append(partial)

    if not selected:
        raise RuntimeError("No tickers passed the requested coverage filters")

    spy = spy_monthly()
    last_complete = min(partial_periods) - 1 if partial_periods else spy.index.max()
    first_period = min(series.index.min() for series in price_series.values())
    periods = pd.period_range(first_period, min(last_complete, spy.index.max()), freq="M")

    drawdowns = pd.DataFrame(index=periods)
    for ticker, prices in price_series.items():
        values = prices.reindex(periods)
        drawdowns[ticker] = values / values.cummax() - 1.0
    spy_values = spy.reindex(periods)
    spy_drawdown = spy_values / spy_values.cummax() - 1.0

    comparison = pd.DataFrame(index=periods.to_timestamp("M"))
    comparison["spy_adjusted_close"] = spy_values.to_numpy()
    comparison["spy_drawdown"] = spy_drawdown.to_numpy()
    comparison["universe_median"] = drawdowns.median(axis=1).to_numpy()
    comparison["universe_p10"] = drawdowns.quantile(0.10, axis=1).to_numpy()
    comparison["universe_p90"] = drawdowns.quantile(0.90, axis=1).to_numpy()
    member_count = drawdowns.notna().sum(axis=1)
    comparison["breadth_below_10"] = (
        (drawdowns <= -0.10).sum(axis=1) / member_count
    ).to_numpy()
    comparison["breadth_below_20"] = (
        (drawdowns <= -0.20).sum(axis=1) / member_count
    ).to_numpy()
    comparison["members"] = member_count.to_numpy()
    comparison = comparison[comparison["members"] > 0]

    joint = comparison.dropna(subset=["spy_drawdown", "universe_median"])
    worst_spy_date = joint["spy_drawdown"].idxmin()
    worst_median_date = joint["universe_median"].idxmin()
    deep_spy = joint[joint["spy_drawdown"] <= -0.10]

    result = {
        "assumptions": {
            "universe": str(args.universe),
            "fundamentals": str(args.fundamentals),
            "market_cap_snapshot": "latest cached Alpha Vantage overview",
            "min_market_cap": args.min_market_cap,
            "min_earnings_quarters": args.min_quarters,
            "min_price_months": args.min_months,
            "drawdown": (
                "monthly adjusted close from each instrument's own high within "
                "the locally available sample"
            ),
            "spy_source": YAHOO_CHART,
            "bias_warning": (
                "Current universe membership and current market cap are used; "
                "this is descriptive and has survivorship/look-ahead bias."
            ),
        },
        "coverage": {
            "universe_tickers": len(tickers),
            **coverage,
            "qualified": len(selected),
            "comparison_start": joint.index.min().date().isoformat(),
            "comparison_end": joint.index.max().date().isoformat(),
        },
        "summary": {
            "spy_worst_month_end": worst_spy_date.date().isoformat(),
            "spy_worst_drawdown": float(joint.loc[worst_spy_date, "spy_drawdown"]),
            "universe_median_at_spy_worst": float(
                joint.loc[worst_spy_date, "universe_median"]
            ),
            "universe_worst_median_month_end": worst_median_date.date().isoformat(),
            "universe_worst_median_drawdown": float(
                joint.loc[worst_median_date, "universe_median"]
            ),
            "spy_vs_universe_median_correlation": float(
                joint["spy_drawdown"].corr(joint["universe_median"])
            ),
            "median_stock_drawdown_when_spy_below_10": float(
                deep_spy["universe_median"].median()
            ),
            "median_breadth_below_20_when_spy_below_10": float(
                deep_spy["breadth_below_20"].median()
            ),
        },
        "qualified": sorted(selected, key=lambda row: row["ticker"]),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.out / "comparison.csv", index_label="date")
    pd.DataFrame(selected).sort_values("market_cap", ascending=False).to_csv(
        args.out / "universe.csv", index=False
    )
    (args.out / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps({"coverage": result["coverage"],
                      "summary": result["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
