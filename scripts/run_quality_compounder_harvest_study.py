"""Compare point-in-time quality-stock harvest and compounder exit policies.

The policy decisions in this study are temporally clean:

* a stock-price threshold observed at one close is executed at the next close;
* earnings are gated by report time: before-open reports may be used that day,
  while after-close and unknown-time reports become eligible the next day;
* the rolling five-year CAGR uses only total-return prices then available; and
* a post-five-year sale requires both CAGR below the threshold and a mechanical
  earnings break.

The candidate basket is still the present-day illustrative basket defined in
``run_spy_quality_rotation.py``.  Therefore this is an exit-policy experiment,
not a fully survivorship-bias-free stock-selection backtest.  The output carries
that distinction in a machine-readable bias audit.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import pandas as pd

import run_spy_quality_rotation as core


ROOT = Path(__file__).resolve().parents[1]
GRID = (
    (0.05, 0.20), (0.10, 0.20),
    (0.05, 0.25), (0.10, 0.25),
    (0.05, 0.30), (0.10, 0.30),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="1993-01-29")
    parser.add_argument("--end", default=None)
    parser.add_argument("--initial", type=float, default=100_000.0)
    parser.add_argument("--spy-share", type=float, default=0.80)
    parser.add_argument("--annual-harvest", type=float, default=0.10)
    parser.add_argument("--spread", type=float, default=0.01)
    parser.add_argument("--risk-per-stock", type=float, default=0.01)
    parser.add_argument("--stock-tail-loss", type=float, default=0.79)
    parser.add_argument("--trade-bp", type=float, default=5.0)
    parser.add_argument("--max-quality-hold-years", type=float, default=5.0)
    parser.add_argument("--compounder-cagr", type=float, default=0.05)
    parser.add_argument("--rolling-years", type=int, default=20)
    parser.add_argument(
        "--cohort-frequency", choices=("annual", "quarterly", "monthly"),
        default="annual",
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--fundamentals", type=Path, default=Path("data/data/sp500_data")
    )
    parser.add_argument(
        "--cache", type=Path, default=Path(".cache/yahoo_daily")
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("out/strategy/quality_compounder_harvest"),
    )
    return parser.parse_args(argv)


def core_args(args):
    values = core.parse_args([])
    values.start = args.start
    values.end = args.end
    values.initial = args.initial
    values.spy_share = args.spy_share
    values.harvest = args.annual_harvest
    values.spread = args.spread
    values.risk_per_stock = args.risk_per_stock
    values.stock_tail_loss = args.stock_tail_loss
    values.trade_bp = args.trade_bp
    values.max_quality_hold_years = args.max_quality_hold_years
    values.compounder_cagr = args.compounder_cagr
    values.rolling_years = args.rolling_years
    values.cohort_frequency = args.cohort_frequency
    values.refresh = args.refresh
    values.fundamentals = args.fundamentals
    values.cache = args.cache
    values.out = args.out
    return values


def variant_name(harvest_share: float, price_step: float) -> str:
    return f"harvest_{harvest_share:.0%}_step_{price_step:.0%}".replace("%", "")


def event_summary(events):
    counts = Counter(event.kind for event in events)
    amounts = defaultdict(float)
    for event in events:
        amounts[event.kind] += event.amount
    return {
        kind: {"events": counts[kind], "amount": amounts[kind]}
        for kind in sorted(counts)
    }


def simulate_variant(prices, rates, args, earnings, name, share, step):
    path, events = core.simulate(
        prices, rates, args, name,
        leverage_policy="step_3_2_1",
        staging=True,
        quality_at_40=True,
        harvest_share=args.harvest,
        signal_source="portfolio",
        exit_policy="compounder_guardrail",
        earnings_histories=earnings,
        quality_harvest_share=share,
        quality_harvest_step=step,
        compounder_cagr=args.compounder_cagr,
    )
    stats = core.metrics(path["wealth"], rates)
    stats.update({
        "ending_reserve": float(path["reserve"].iloc[-1]),
        "ending_quality": float(path["quality_sleeve"].iloc[-1]),
        "quality_lots": path.attrs["quality_lots"],
        "events": event_summary(events),
    })
    return path, events, stats


def rolling_study(prices, rates, args, earnings):
    starts = prices.index.to_series().groupby(prices.index.to_period("M")).first()
    stride = {"monthly": 1, "quarterly": 3, "annual": 12}[args.cohort_frequency]
    starts = starts.iloc[::stride]
    records = []
    for start in starts:
        target = start + pd.DateOffset(years=args.rolling_years)
        if target > prices.index[-1]:
            continue
        end_position = prices.index.searchsorted(target, side="right") - 1
        end = prices.index[end_position]
        window_prices = prices.loc[start:end]
        window_rates = rates.loc[start:end]
        spy = args.initial * window_prices["SPY"] / window_prices["SPY"].iloc[0]
        spy_stats = core.metrics(spy, window_rates)
        for share, step in GRID:
            name = variant_name(share, step)
            path, events, stats = simulate_variant(
                window_prices, window_rates, args, earnings,
                f"{name}_rolling", share, step,
            )
            records.append({
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
                "variant": name,
                "harvest_share": share,
                "price_step": step,
                "cagr": stats["cagr"],
                "max_drawdown": stats["max_drawdown"],
                "terminal": stats["terminal"],
                "spy_cagr": spy_stats["cagr"],
                "spy_max_drawdown": spy_stats["max_drawdown"],
                "spy_terminal": spy_stats["terminal"],
                "quality_deployments": sum(
                    event.kind == "deploy_quality" for event in events
                ),
                "profit_harvests": sum(
                    event.kind == "quality_profit_harvest" for event in events
                ),
                "compounder_exits": sum(
                    event.kind == "quality_compounder_exit" for event in events
                ),
                "ending_quality": float(path["quality_sleeve"].iloc[-1]),
            })
    frame = pd.DataFrame(records)
    summaries = {}
    for name, group in frame.groupby("variant"):
        active = group.loc[group["quality_deployments"] > 0].copy()
        active["excess_cagr"] = active["cagr"] - active["spy_cagr"]
        summaries[name] = {
            "cohorts": len(group),
            "cagr": core.quantiles(group["cagr"]),
            "max_drawdown": core.quantiles(group["max_drawdown"]),
            "beats_spy_share": float(
                (group["terminal"] > group["spy_terminal"]).mean()
            ),
            "quality_active_cohorts": len(active),
            "quality_active_median_cagr": (
                float(active["cagr"].median()) if len(active) else None
            ),
            "quality_active_median_excess_cagr": (
                float(active["excess_cagr"].median()) if len(active) else None
            ),
            "quality_active_beats_spy_share": (
                float((active["terminal"] > active["spy_terminal"]).mean())
                if len(active) else None
            ),
            "median_profit_harvests": float(group["profit_harvests"].median()),
            "total_compounder_exits": int(group["compounder_exits"].sum()),
        }
    return frame, summaries


def main(argv=None) -> int:
    args = parse_args(argv)
    sim_args = core_args(args)
    prices, coverage = core.load_prices(sim_args)
    rates = core.load_rates(prices.index)
    earnings = core.load_earnings_histories(
        sim_args.fundamentals,
        [ticker for ticker in prices.columns if ticker != "SPY"],
    )

    full_paths = {}
    full_events = {}
    full_summaries = {}
    for share, step in GRID:
        name = variant_name(share, step)
        path, events, stats = simulate_variant(
            prices, rates, sim_args, earnings, name, share, step,
        )
        full_paths[name] = path
        full_events[name] = events
        full_summaries[name] = stats

    rolling_frame, rolling_summaries = rolling_study(
        prices, rates, sim_args, earnings,
    )
    spy_path = args.initial * prices["SPY"] / prices["SPY"].iloc[0]
    spy_summary = core.metrics(spy_path, rates)
    ranking = sorted(
        ({
            "variant": name,
            "median_20y_cagr": values["cagr"]["median"],
            "median_20y_max_drawdown": values["max_drawdown"]["median"],
            "beats_spy_share": values["beats_spy_share"],
            "quality_active_cohorts": values["quality_active_cohorts"],
            "quality_active_median_excess_cagr": (
                values["quality_active_median_excess_cagr"]
            ),
            "quality_active_beats_spy_share": (
                values["quality_active_beats_spy_share"]
            ),
            "full_period_cagr": full_summaries[name]["cagr"],
            "full_period_excess_cagr": (
                full_summaries[name]["cagr"] - spy_summary["cagr"]
            ),
            "full_period_max_drawdown": full_summaries[name]["max_drawdown"],
        } for name, values in rolling_summaries.items()),
        key=lambda row: row["quality_active_median_excess_cagr"], reverse=True,
    )

    bias_audit = {
        "decision_rules_use_future_prices": False,
        "price_signal_execution": "prior close signal; next close execution",
        "earnings_availability_field": "reportedDate + reportTime",
        "after_close_or_unknown_report_timing": (
            "conservatively available on the next calendar day"
        ),
        "future_earnings_filtered": True,
        "historical_fundamental_vintages_available": False,
        "vendor_revision_risk": (
            "Historical reported EPS may reflect later vendor corrections"
        ),
        "rolling_cagr_uses_future_prices": False,
        "fundamental_missing_data_forces_sale": False,
        "point_in_time_universe_membership": False,
        "point_in_time_market_cap_screen": False,
        "delisted_security_return_coverage": False,
        "static_present_day_candidate_basket": True,
        "overall_strategy_claim_is_survivorship_bias_free": False,
        "valid_interpretation": (
            "Temporally clean comparison of exit policies on fixed purchase "
            "candidates; not an unbiased estimate of the stock-selection alpha."
        ),
    }

    result = {
        "sample": {
            "start": prices.index[0].date().isoformat(),
            "end": prices.index[-1].date().isoformat(),
            "sessions": len(prices),
        },
        "policy": {
            "initial": args.initial,
            "starting_spy_share": args.spy_share,
            "starting_reserve_share": 1.0 - args.spy_share,
            "profit_harvest": (
                "sell the configured fraction of current shares after each "
                "configured multiplicative increase from the last harvest level"
            ),
            "profit_harvest_signal": (
                "adjusted-close total-return price at prior close"
            ),
            "maximum_holding_years_before_review": args.max_quality_hold_years,
            "compounder_threshold": args.compounder_cagr,
            "post_review_exit": (
                "sell only when trailing five-year total-return CAGR is below "
                "threshold and the point-in-time earnings guardrail is broken"
            ),
            "earnings_break": (
                "TTM EPS <= 0, or current TTM EPS below both prior-year and "
                "three-years-earlier TTM EPS; minimum 16 reported quarters"
            ),
        },
        "grid": [
            {"variant": variant_name(share, step),
             "harvest_share": share, "price_step": step}
            for share, step in GRID
        ],
        "coverage": coverage,
        "bias_audit": bias_audit,
        "full_period": full_summaries,
        "benchmark": {"spy_1x": spy_summary},
        "rolling": {
            "years": args.rolling_years,
            "cohort_frequency": args.cohort_frequency,
            "summaries": rolling_summaries,
        },
        "ranking": ranking,
        "events": {
            name: [asdict(event) for event in events]
            for name, events in full_events.items()
        },
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    pd.DataFrame(ranking).to_csv(args.out / "grid_ranking.csv", index=False)
    rolling_frame.to_csv(args.out / "rolling_20y.csv", index=False)
    all_events = [
        asdict(event) for events in full_events.values() for event in events
    ]
    pd.DataFrame(all_events).to_csv(args.out / "events.csv", index=False)
    daily = pd.DataFrame(index=prices.index)
    daily["spy_1x"] = spy_path
    for name, path in full_paths.items():
        daily[name] = path["wealth"]
        daily[f"{name}_reserve"] = path["reserve"]
        daily[f"{name}_quality"] = path["quality_sleeve"]
    daily.to_csv(args.out / "daily.csv", index_label="date")

    print(json.dumps({
        "sample": result["sample"],
        "bias_audit": bias_audit,
        "ranking": ranking,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
