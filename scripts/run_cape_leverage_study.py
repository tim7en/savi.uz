"""Test Shiller-CAPE leverage caps and audit mega-seven excess returns."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import run_quality_compounder_v2 as strategy
import run_spy_quality_rotation as core
from run_contribution_quality_strategy import performance_metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="1993-01-29")
    parser.add_argument("--initial", type=float, default=10_000.0)
    parser.add_argument("--monthly-contribution", type=float, default=10_000.0)
    parser.add_argument("--spread", type=float, default=0.01)
    parser.add_argument("--trade-bp", type=float, default=5.0)
    parser.add_argument("--fundamentals", type=Path, default=Path("data/sp500_data"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/yahoo_daily"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/cape_leverage"))
    return parser.parse_args(argv)


def load_cape() -> pd.Series:
    database = Path("data/equity/equity.db")
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT obs_date,cape FROM shiller_monthly WHERE cape IS NOT NULL ORDER BY obs_date"
    ).fetchall()
    connection.close()
    return pd.Series(
        {pd.Timestamp(day): float(value) for day, value in rows},
        dtype=float, name="cape",
    ).sort_index()


def known_cape_daily(cape: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Conservatively make a month's observation usable the next month."""
    available = cape.copy()
    available.index = available.index + pd.offsets.MonthBegin(1)
    return available.reindex(index, method="ffill")


def percentile_caps(cape: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    values = []
    for stamp, value in cape.items():
        prior = cape.loc[:stamp].dropna()
        rank = float((prior < value).mean())
        values.append(1.0 if rank >= 0.95 else 2.0 if rank >= 0.80 else 3.0)
    monthly = pd.Series(values, index=cape.index, dtype=float)
    monthly.index = monthly.index + pd.offsets.MonthBegin(1)
    return monthly.reindex(index, method="ffill")


def fixed_cap(cape_daily: pd.Series, two_at: float, one_at: float) -> pd.Series:
    result = pd.Series(3.0, index=cape_daily.index)
    result.loc[cape_daily >= two_at] = 2.0
    result.loc[cape_daily >= one_at] = 1.0
    return result


def cashflow_path(spy: pd.Series, rates: pd.Series, args, *,
                  core_share: float, leverage: pd.Series) -> pd.DataFrame:
    leverage = leverage.reindex(spy.index).ffill().bfill()
    schedule = strategy.monthly_schedule(spy.index, args.monthly_contribution)
    returns = spy.pct_change().fillna(0.0)
    days = spy.index.to_series().diff().dt.days.fillna(0.0)
    main, reserve = args.initial * core_share, args.initial * (1.0 - core_share)
    perf, high, previous = 1.0, 1.0, args.initial
    rows = []
    for position, stamp in enumerate(spy.index):
        if position:
            elapsed = float(days.iloc[position]) / 365.0
            known_rate = float(rates.iloc[position - 1])
            level = float(leverage.iloc[position - 1])
            main = max(main * (1.0 + level * float(returns.iloc[position])
                       - max(level - 1.0, 0.0) * (known_rate + args.spread) * elapsed), 0.0)
            reserve *= 1.0 + known_rate * elapsed
            pre_flow = main + reserve
            perf *= pre_flow / previous
        contribution = float(schedule.iloc[position])
        main += contribution * core_share
        reserve += contribution * (1.0 - core_share)
        total = main + reserve
        previous = total
        high = max(high, perf)
        rows.append({"wealth": total, "performance_index": perf,
                     "contribution": contribution,
                     "flow_adjusted_drawdown": perf / high - 1.0})
    return pd.DataFrame(rows, index=spy.index)


def nearest_date(index: pd.DatetimeIndex, target: pd.Timestamp) -> pd.Timestamp | None:
    if target > index[-1]:
        return None
    position = index.searchsorted(target, side="right") - 1
    return index[position] if position >= 0 else None


def mega_seven_audit(prices: pd.DataFrame, events: list[dict]) -> list[dict]:
    records = []
    deploys = [event for event in events if event["kind"] == "deploy_mega_seven"]
    for event in deploys:
        entry = pd.Timestamp(event["date"])
        weights = {}
        for piece in event["detail"].split("; "):
            ticker, weight = piece.rsplit(" ", 1)
            weights[ticker] = float(weight.rstrip("%")) / 100.0
        weight_total = sum(weights.values())
        weights = {ticker: weight / weight_total for ticker, weight in weights.items()}
        entry_position = prices.index.get_loc(entry)
        prior_high = float(prices["SPY"].iloc[:entry_position].max())
        recovery_values = prices["SPY"].iloc[entry_position:]
        recovery_hits = recovery_values.loc[recovery_values >= prior_high]
        recovery = recovery_hits.index[0] if not recovery_hits.empty else None
        horizons = {
            "one_year": nearest_date(prices.index, entry + pd.DateOffset(years=1)),
            "three_year": nearest_date(prices.index, entry + pd.DateOffset(years=3)),
            "five_year": nearest_date(prices.index, entry + pd.DateOffset(years=5)),
            "spy_recovery": recovery,
            "sample_end": prices.index[-1],
        }
        for horizon, end in horizons.items():
            if end is None or end <= entry:
                continue
            basket_multiple = sum(
                weight * float(prices.loc[end, ticker] / prices.loc[entry, ticker])
                for ticker, weight in weights.items()
            )
            spy_multiple = float(prices.loc[end, "SPY"] / prices.loc[entry, "SPY"])
            years = (end - entry).days / 365.2425
            records.append({
                "entry": entry.date().isoformat(), "end": end.date().isoformat(),
                "horizon": horizon, "names": list(weights),
                "basket_return": basket_multiple - 1.0,
                "spy_return": spy_multiple - 1.0,
                "excess_return": basket_multiple / spy_multiple - 1.0,
                "basket_cagr": basket_multiple ** (1.0 / years) - 1.0,
                "spy_cagr": spy_multiple ** (1.0 / years) - 1.0,
            })
        path = pd.Series(0.0, index=prices.index[entry_position:])
        for ticker, weight in weights.items():
            path += weight * prices[ticker].iloc[entry_position:] / prices.loc[entry, ticker]
        spy_path = prices["SPY"].iloc[entry_position:] / prices.loc[entry, "SPY"]
        records.append({
            "entry": entry.date().isoformat(), "end": prices.index[-1].date().isoformat(),
            "horizon": "risk_audit", "names": list(weights),
            "basket_max_drawdown": float((path / path.cummax() - 1.0).min()),
            "spy_max_drawdown": float((spy_path / spy_path.cummax() - 1.0).min()),
        })
    return records


def main(argv=None):
    args = parse_args(argv)
    load_args = strategy.parse_args([])
    for name in ("start", "initial", "monthly_contribution", "spread", "trade_bp",
                 "fundamentals", "cache", "refresh", "out"):
        setattr(load_args, name, getattr(args, name))
    prices, raw, shares, coverage = strategy.load_market_data(load_args)
    full_prices = prices.copy()
    cape = load_cape()
    cape_end = cape.index[-1] + pd.offsets.MonthEnd(1)
    prices = prices.loc[:cape_end]
    raw = raw.loc[prices.index]
    rates = core.load_rates(prices.index)
    known_cape = known_cape_daily(cape, prices.index)
    caps = {
        "no_cape_cap": pd.Series(3.0, index=prices.index),
        "cape_30_40": fixed_cap(known_cape, 30.0, 40.0),
        "cape_25_35": fixed_cap(known_cape, 25.0, 35.0),
        "cape_20_25": fixed_cap(known_cape, 20.0, 25.0),
        "cape_15_20": fixed_cap(known_cape, 15.0, 20.0),
        "cape_percentile_80_95": percentile_caps(cape, prices.index),
    }

    paths, summaries = {}, {}
    for name, cap in caps.items():
        path, events, holdings = strategy.simulate(
            prices, raw, shares, rates, load_args, reserve_share=0.20,
            contribution_mode="immediate", name=name, leverage_cap=cap,
        )
        paths[name] = path
        summary = performance_metrics(path, rates, args.initial)
        summary.update({
            "mean_applied_leverage": float(path["leverage"].mean()),
            "time_at_1x": float((path["leverage"] == 1.0).mean()),
            "time_at_2x": float((path["leverage"] == 2.0).mean()),
            "time_at_3x": float((path["leverage"] == 3.0).mean()),
            "cape_cap_binding_share": float(np.isclose(path["leverage"], cap).mean()),
        })
        summaries[name] = summary

    spy_path = strategy.simulate_spy(prices, load_args)
    summaries["spy_1x"] = performance_metrics(spy_path, rates, args.initial)
    three = pd.Series(3.0, index=prices.index)
    decomposition = {}
    for name, core_share, cap in [
        ("all_equity_constant_3x", 1.0, three),
        ("eighty_twenty_constant_3x", 0.80, three),
        ("all_equity_cape_25_35", 1.0, caps["cape_25_35"]),
    ]:
        path = cashflow_path(prices["SPY"], rates, args,
                             core_share=core_share, leverage=cap)
        decomposition[name] = performance_metrics(path, rates, args.initial)

    existing = json.loads(Path(
        "out/strategy/quality_compounder_v2/results.json"
    ).read_text(encoding="utf-8"))
    mega_audit = mega_seven_audit(
        full_prices, existing["events"]["immediate_20"]
    )
    result = {
        "sample": {"start": prices.index[0].date().isoformat(),
                   "end": prices.index[-1].date().isoformat(),
                   "cape_source_end": cape.index[-1].date().isoformat()},
        "cape_timing": "monthly CAPE becomes usable at the next month start",
        "strategy_variants": summaries,
        "decomposition": decomposition,
        "mega_seven_audit": mega_audit,
        "coverage": coverage,
        "warnings": [
            "CAPE is tested as a leverage cap, not a short-term return signal.",
            "The CAPE sample ends with the local Shiller observation dated 2024-09.",
            "Mega-seven selection uses a survivor-biased fixed historical-leader union.",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    daily = pd.DataFrame(index=prices.index)
    daily["cape_known"] = known_cape
    daily["spy_1x_wealth"] = spy_path["wealth"]
    daily["spy_1x_performance"] = spy_path["performance_index"]
    for name, path in paths.items():
        daily[f"{name}_wealth"] = path["wealth"]
        daily[f"{name}_performance"] = path["performance_index"]
        daily[f"{name}_leverage"] = path["leverage"]
    daily.to_csv(args.out / "daily.csv", index_label="date")
    print(json.dumps({"sample": result["sample"],
                      "strategy_variants": summaries,
                      "decomposition": decomposition,
                      "mega_seven_audit": mega_audit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
