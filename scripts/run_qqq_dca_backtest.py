"""Backtest the daily-dashboard policy with QQQ as the core asset.

The initial $10,000 is 100% QQQ at 1x.  Later contributions follow the
previously specified $10,000 annual plus $30,000 every third year schedule and
are split 80% QQQ / 20% Treasury.  Only a fresh contribution made during a
flow-adjusted NAV drawdown can be levered: at most 2x from -10% and 3x from
-20%, further capped by lagged Shiller CAPE and a prior-close 60-session VXN
percentile.  A leveraged contribution resets to 1x at NAV recovery.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import run_cape_leverage_study as cape_data
import run_quality_compounder_v2 as market
import run_quality_ladder_cape_leverage as guard
import run_quality_ladder_harvest as ladder
import run_spy_quality_rotation as core
from run_contribution_quality_strategy import contribution_schedule, performance_metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="1999-03-10")
    parser.add_argument("--end", default=None)
    parser.add_argument("--initial", type=float, default=10_000.0)
    parser.add_argument("--annual-contribution", type=float, default=10_000.0)
    parser.add_argument("--triennial-contribution", type=float, default=30_000.0)
    parser.add_argument("--spy-share", type=float, default=0.80)
    parser.add_argument("--spread", type=float, default=0.01)
    parser.add_argument("--trade-bp", type=float, default=5.0)
    parser.add_argument("--relative-step", type=float, default=0.20)
    parser.add_argument("--harvest-share", type=float, default=0.05)
    parser.add_argument("--cape-excessive", type=float, default=35.0)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--fundamentals", type=Path, default=Path("data/sp500_data"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/yahoo_daily"))
    parser.add_argument("--macro-db", type=Path, default=Path("data/macro/macro.db"))
    parser.add_argument("--out", type=Path, default=Path("out/strategy/qqq_dca_backtest"))
    return parser.parse_args(argv)


def monthly_equivalent_schedule(index: pd.DatetimeIndex, annual: float,
                                triennial: float) -> pd.Series:
    """Spread the annual contribution over 12 months, preserving triennial cash."""
    result = pd.Series(0.0, index=index)
    years = sorted(set(index.year))
    for number, year in enumerate(years[1:], start=1):
        year_days = index[index.year == year]
        first_sessions = year_days.to_series().groupby(year_days.to_period("M")).first()
        for stamp in first_sessions:
            result.loc[stamp] += annual / 12.0
        if number % 3 == 0:
            result.loc[first_sessions.iloc[0]] += triennial
    return result


def drawdown_episode(path: pd.DataFrame, peak_cutoff: str = "2000-03-31") -> dict:
    perf = path["performance_index"].astype(float)
    before = perf.loc[:peak_cutoff]
    peak_date = before.idxmax()
    peak = float(before.loc[peak_date])
    future = perf.loc[peak_date:]
    drawdown = future / peak - 1.0
    trough_date = drawdown.idxmin()
    after_trough = future.loc[trough_date:]
    recovered = after_trough[after_trough >= peak - 1e-12]
    recovery_date = recovered.index[0] if not recovered.empty else None
    return {
        "peak_date": peak_date.date().isoformat(),
        "trough_date": trough_date.date().isoformat(),
        "max_drawdown": float(drawdown.loc[trough_date]),
        "recovery_date": recovery_date.date().isoformat() if recovery_date is not None else None,
        "calendar_days_to_recovery": (
            int((recovery_date - peak_date).days) if recovery_date is not None else None
        ),
    }


def load_qqq_market(args):
    load_args = market.parse_args([])
    for field in ("start", "end", "initial", "refresh", "fundamentals", "cache"):
        setattr(load_args, field, getattr(args, field))
    prices, raw, shares, coverage = market.load_market_data(load_args)
    qqq = core.yahoo_series("QQQ", args.start, args.end, args.cache, args.refresh)

    cape_monthly = cape_data.load_cape()
    sample_end = cape_monthly.index[-1] + pd.offsets.MonthEnd(1)
    if args.end:
        sample_end = min(sample_end, pd.Timestamp(args.end))
    qqq = qqq.loc[:sample_end]
    prices = prices.reindex(qqq.index)
    prices["SPY"] = qqq
    for column in prices.columns:
        if column != "SPY":
            prices[column] = prices[column].ffill()
    raw = raw.reindex(qqq.index).ffill()
    return prices, raw, shares, coverage, cape_monthly


def summarize(path: pd.DataFrame, rates: pd.Series, initial: float) -> dict:
    result = performance_metrics(path, rates, initial)
    for column, name in (
        ("treasury", "ending_treasury"),
        ("quality", "ending_quality"),
    ):
        result[name] = float(path[column].iloc[-1]) if column in path else 0.0
    result.update({
        "ending_treasury_weight": float(path["treasury"].iloc[-1] / path["wealth"].iloc[-1])
        if "treasury" in path else 0.0,
        "mean_gross_exposure": float(path["gross_exposure"].mean())
        if "gross_exposure" in path else 1.0,
        "max_gross_exposure": float(path["gross_exposure"].max())
        if "gross_exposure" in path else 1.0,
        "financing_cost": float(path["financing_cost"].sum())
        if "financing_cost" in path else 0.0,
        "dotcom": drawdown_episode(path),
    })
    return result


def main(argv=None) -> int:
    args = parse_args(argv)
    prices, raw, shares, coverage, cape_monthly = load_qqq_market(args)
    index = prices.index
    rates = core.load_rates(index)
    known_cape = cape_data.known_cape_daily(cape_monthly, index)
    cape_ceiling = guard.cape_leverage(known_cape)
    known_vxn = guard.trailing_vix_percentile(
        index, args.macro_db, smooth_sessions=60, series_id="VXNCLS"
    )
    # VXN starts in 2001 and needs a full prior-year rank. Unknown means no
    # permission to leverage; it is not silently replaced with hindsight.
    known_vxn_conservative = known_vxn.fillna(1.0)
    annual = contribution_schedule(
        index, args.annual_contribution, args.triennial_contribution
    )
    monthly = monthly_equivalent_schedule(
        index, args.annual_contribution, args.triennial_contribution
    )
    one = pd.Series(1.0, index=index)

    common = dict(
        rungs_enabled=True,
        harvest_enabled=True,
        cape_enabled=True,
        core_leverage=one,
        injection_leverage=cape_ceiling,
        injection_nav_drawdown=0.10,
        injection_nav_tiered=True,
        injection_vix_percentile=known_vxn_conservative,
        injection_vix_mode="brake",
        initial_spy_share=1.0,
    )
    variants = {
        "dashboard_quality_annual": dict(
            **common, quality_enabled=True, contribution_series=annual
        ),
        "dashboard_qqq_only_annual": dict(
            **common, quality_enabled=False, all_spy_rungs=True,
            contribution_series=annual,
        ),
        "dashboard_quality_monthly": dict(
            **common, quality_enabled=True, contribution_series=monthly
        ),
        "quality_ladder_no_leverage": dict(
            rungs_enabled=True, quality_enabled=True, harvest_enabled=True,
            cape_enabled=True, core_leverage=one, contribution_series=annual,
            initial_spy_share=1.0,
        ),
        "qqq_ladder_no_leverage": dict(
            rungs_enabled=True, quality_enabled=False, harvest_enabled=False,
            cape_enabled=False, all_spy_rungs=True, core_leverage=one,
            contribution_series=annual, initial_spy_share=1.0,
        ),
        "static_qqq_treasury": dict(
            rungs_enabled=False, quality_enabled=False, harvest_enabled=False,
            cape_enabled=False, core_leverage=one, contribution_series=annual,
            initial_spy_share=1.0,
        ),
    }

    paths, events, holdings, metrics = {}, {}, {}, {}
    for name, settings in variants.items():
        path, event_set, open_holdings = ladder.simulate(
            prices, raw, shares, rates, known_cape, args, name=name, **settings
        )
        paths[name] = path
        events[name] = event_set
        holdings[name] = open_holdings
        metrics[name] = summarize(path, rates, args.initial)
        metrics[name]["leveraged_contributions"] = sum(
            event.kind == "contribution" and "injection at 1x" not in event.detail
            and "80% SPY /" not in event.detail
            for event in event_set
        )
        metrics[name]["vxn_leverage_changes"] = sum(
            event.kind == "vix_injection_leverage_change" for event in event_set
        )

    qqq_path = guard.simulate_spy(prices["SPY"], annual, args.initial)
    qqq_path["gross_exposure"] = 1.0
    qqq_path["financing_cost"] = 0.0
    metrics["qqq_1x"] = summarize(qqq_path, rates, args.initial)
    paths["qqq_1x"] = qqq_path

    three_args = argparse.Namespace(**vars(args))
    three_args.spy_share = 1.0
    three_path, three_events, _ = ladder.simulate(
        prices, raw, shares, rates, known_cape, three_args,
        name="qqq_3x_theoretical", rungs_enabled=False, quality_enabled=False,
        harvest_enabled=False, cape_enabled=False, contribution_series=annual,
        core_leverage=pd.Series(3.0, index=index), initial_spy_share=1.0,
    )
    paths["qqq_3x_theoretical"] = three_path
    events["qqq_3x_theoretical"] = three_events
    metrics["qqq_3x_theoretical"] = summarize(three_path, rates, args.initial)

    rolling_10y = []
    final_year = index[-1].year
    for start_year in range(2000, final_year - 9):
        cohort_start = index[index >= pd.Timestamp(start_year, 1, 1)]
        if cohort_start.empty:
            continue
        cohort_end = cohort_start[0] + pd.DateOffset(years=10)
        cohort_index = index[(index >= cohort_start[0]) & (index <= cohort_end)]
        if len(cohort_index) < 252 * 9:
            continue
        cohort_contributions = contribution_schedule(
            cohort_index, args.annual_contribution, args.triennial_contribution
        )
        cohort_path, _, _ = ladder.simulate(
            prices.loc[cohort_index], raw.reindex(cohort_index), shares,
            rates.loc[cohort_index], known_cape.loc[cohort_index], args,
            name=f"cohort_{start_year}", rungs_enabled=True,
            quality_enabled=True, harvest_enabled=True, cape_enabled=True,
            core_leverage=one.loc[cohort_index],
            injection_leverage=cape_ceiling.loc[cohort_index],
            injection_nav_drawdown=0.10, injection_nav_tiered=True,
            injection_vix_percentile=known_vxn_conservative.loc[cohort_index],
            injection_vix_mode="brake", initial_spy_share=1.0,
            contribution_series=cohort_contributions,
        )
        cohort_qqq = guard.simulate_spy(
            prices.loc[cohort_index, "SPY"], cohort_contributions, args.initial
        )
        strategy_stats = performance_metrics(
            cohort_path, rates.loc[cohort_index], args.initial
        )
        qqq_stats = performance_metrics(
            cohort_qqq, rates.loc[cohort_index], args.initial
        )
        rolling_10y.append({
            "start": cohort_index[0].date().isoformat(),
            "end": cohort_index[-1].date().isoformat(),
            "strategy_terminal": strategy_stats["terminal_wealth"],
            "qqq_terminal": qqq_stats["terminal_wealth"],
            "terminal_ratio": strategy_stats["terminal_wealth"] / qqq_stats["terminal_wealth"],
            "strategy_xirr": strategy_stats["xirr"],
            "qqq_xirr": qqq_stats["xirr"],
            "strategy_max_drawdown": strategy_stats["max_flow_adjusted_drawdown"],
            "qqq_max_drawdown": qqq_stats["max_flow_adjusted_drawdown"],
        })

    result = {
        "sample": {
            "start": index[0].date().isoformat(),
            "end": index[-1].date().isoformat(),
            "sessions": len(index),
            "cape_source_end": cape_monthly.index[-1].date().isoformat(),
            "vxn_first_observation": "2001-02-02",
        },
        "cash_flows": {
            "initial": args.initial,
            "annual_contribution": args.annual_contribution,
            "triennial_contribution": args.triennial_contribution,
            "annual_total_contributed": metrics["qqq_1x"]["total_contributed"],
        },
        "rules": {
            "core": "initial $10,000 fully in QQQ at 1x",
            "contributions": "annual $10,000 plus $30,000 every third year; 80% QQQ / 20% Treasury",
            "fresh_capital_nav": "new QQQ contribution at 1x above -10% NAV DD, max 2x from -10%, max 3x from -20%",
            "cape": "prior-known monthly Shiller CAPE: max 3x below 25, 2x at 25-35, 1x above 35",
            "vxn": "prior-close 60-session VXN ranked vs prior 252 sessions: max 3x below 70th pct, 2x at 70-90th, 1x above 90th; unavailable history forces 1x",
            "reset": "leveraged contribution lots return to 1x at flow-adjusted NAV recovery",
            "treasury_ladder": "20% quality at -10% QQQ DD; 30% QQQ at -20%; 30% quality at -30%; final 20% QQQ at -50%",
            "funding": f"prior-known DGS3MO + {args.spread:.1%}",
            "trade_cost": f"{args.trade_bp:g} bp on ladder and harvest trades",
        },
        "variants": metrics,
        "rolling_10y": rolling_10y,
        "holdings": holdings,
        "events": {name: [asdict(event) for event in rows] for name, rows in events.items()},
        "coverage": coverage,
        "warnings": [
            "QQQ history begins in March 1999, immediately before the dot-com peak.",
            "VXN begins in February 2001; the 60/252 rank is unavailable until 2002, and the test conservatively caps leverage at 1x while unknown.",
            "Shiller CAPE measures the S&P 500, not the Nasdaq-100; it is only a broad-market ceiling.",
            "The quality universe is survivor-selected and early market-cap/share history is incomplete; skipped quality rungs are reported rather than invented.",
            "Taxes, leverage switching costs, margin calls, ETF tracking differences, and forced liquidation are omitted.",
        ],
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    daily = pd.DataFrame(index=index)
    for name, path in paths.items():
        for column in ("wealth", "performance_index", "flow_adjusted_drawdown",
                       "contribution",
                       "treasury", "quality", "spy_weight", "legacy_core_spy",
                       "rescue_spy", "injection_core_spy",
                       "injection_weighted_leverage", "gross_exposure",
                       "financing_cost"):
            if column in path:
                daily[f"{name}_{column}"] = path[column]
    daily["vxn_sma60_percentile"] = known_vxn
    daily["cape_known"] = known_cape
    daily.to_csv(args.out / "daily.csv", index_label="date")
    print(json.dumps({"sample": result["sample"], "variants": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
