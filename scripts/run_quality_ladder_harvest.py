"""Backtest an unlevered SPY/Treasury ladder with quality-stock harvesting.

The strategy begins 80% SPY / 20% Treasury.  Fractions of the Treasury
snapshot at the preceding SPY total-return high are deployed once at -10%,
-20%, -30%, and -50% SPY drawdowns.  The -10% and -30% rungs buy the seven
largest date-ranked companies; the other rungs buy SPY.  Quality lots are
benchmarked against SPY from their own entry dates and sell 5% of original
shares for every new 20 percentage points of relative wealth outperformance.

The historical top-seven universe is date-ranked but survivor-biased because
the local data do not contain a complete point-in-time listed/delisted universe.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import run_cape_leverage_study as cape_data
import run_quality_compounder_v2 as market
import run_spy_quality_rotation as core
from run_contribution_quality_strategy import Event, performance_metrics


RUNGS = (
    (0.10, 0.20, "quality"),
    (0.20, 0.30, "spy"),
    (0.30, 0.30, "quality"),
    (0.50, 0.20, "spy"),
)


@dataclass
class QualityLot:
    ticker: str
    shares: float
    original_shares: float
    cost: float
    entry_date: pd.Timestamp
    entry_price: float
    entry_spy: float
    harvest_bands: int = 0
    harvested_shares: float = 0.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="1993-01-29")
    parser.add_argument("--end", default=None)
    parser.add_argument("--initial", type=float, default=10_000.0)
    parser.add_argument("--monthly-contribution", type=float, default=10_000.0)
    parser.add_argument("--spy-share", type=float, default=0.80)
    parser.add_argument("--cape-excessive", type=float, default=35.0)
    parser.add_argument("--relative-step", type=float, default=0.20)
    parser.add_argument("--harvest-share", type=float, default=0.05)
    parser.add_argument("--trade-bp", type=float, default=5.0)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--fundamentals", type=Path, default=Path("data/sp500_data"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/yahoo_daily"))
    parser.add_argument(
        "--out", type=Path, default=Path("out/strategy/quality_ladder_harvest")
    )
    return parser.parse_args(argv)


def relative_excess(stock_now: float, stock_entry: float,
                    spy_now: float, spy_entry: float) -> float:
    """Relative total-return wealth since a lot's entry date."""
    return (stock_now / stock_entry) / (spy_now / spy_entry) - 1.0


def new_harvest_bands(excess: float, already_harvested: int,
                      step: float = 0.20) -> int:
    """Count newly crossed positive relative-return milestones."""
    if not math.isfinite(excess) or excess < step:
        return 0
    reached = int(math.floor((excess + 1e-12) / step))
    return max(reached - already_harvested, 0)


def summarize_events(events: list[Event]) -> dict:
    counts = Counter(event.kind for event in events)
    amounts = defaultdict(float)
    for event in events:
        amounts[event.kind] += event.amount
    return {
        kind: {"count": counts[kind], "amount": amounts[kind]}
        for kind in sorted(counts)
    }


def quality_value(lots: list[QualityLot], row: pd.Series) -> float:
    return float(sum(
        lot.shares * float(row.get(lot.ticker, np.nan))
        for lot in lots
        if lot.shares > 1e-12 and pd.notna(row.get(lot.ticker, np.nan))
    ))


def _buy_spy(amount: float, price: float, cost_rate: float) -> float:
    return amount * (1.0 - cost_rate) / price if amount > 0 else 0.0


def simulate(
    prices: pd.DataFrame,
    raw: pd.DataFrame,
    shares: dict[str, pd.Series],
    rates: pd.Series,
    cape: pd.Series,
    args,
    *,
    name: str,
    rungs_enabled: bool,
    quality_enabled: bool,
    harvest_enabled: bool,
    cape_enabled: bool,
    all_spy_rungs: bool = False,
) -> tuple[pd.DataFrame, list[Event], list[dict]]:
    spy = prices["SPY"].dropna()
    prices = prices.reindex(spy.index)
    raw = raw.reindex(spy.index)
    rates = rates.reindex(spy.index).ffill().bfill()
    cape = cape.reindex(spy.index).ffill()
    contributions = market.monthly_schedule(spy.index, args.monthly_contribution)
    days = spy.index.to_series().diff().dt.days.fillna(0.0)
    spy_high = spy.cummax()
    spy_drawdown = spy / spy_high - 1.0
    cost_rate = args.trade_bp / 10_000.0

    spy_shares = args.initial * args.spy_share / float(spy.iloc[0])
    reserve = args.initial * (1.0 - args.spy_share)
    lots: list[QualityLot] = []
    events: list[Event] = []
    rows = []
    fired: set[float] = set()
    episode_reserve = reserve
    reserve_target = reserve
    performance_index = 1.0
    performance_high = 1.0
    previous_total = args.initial
    max_quality_weight = 0.0
    max_spy_exposure = args.spy_share
    harvest_to_reserve_total = 0.0
    harvest_to_spy_total = 0.0
    cape_incremental_reserve_total = 0.0

    for position, stamp in enumerate(spy.index):
        signal_position = max(position - 1, 0)
        signal_date = spy.index[signal_position]
        signal_dd = float(spy_drawdown.iloc[signal_position])
        signal_cape = float(cape.iloc[signal_position]) if pd.notna(cape.iloc[signal_position]) else np.nan
        current_row = prices.iloc[position]
        new_quarter = bool(
            position and stamp.to_period("Q") != signal_date.to_period("Q")
        )

        if position:
            elapsed = float(days.iloc[position]) / 365.0
            reserve *= 1.0 + float(rates.iloc[position - 1]) * elapsed

        pre_flow_total = (
            spy_shares * float(spy.iloc[position])
            + reserve
            + quality_value(lots, current_row)
        )
        if position and previous_total > 0:
            performance_index *= pre_flow_total / previous_total

        contribution = float(contributions.iloc[position])
        if contribution:
            spy_cash = contribution * args.spy_share
            reserve_cash = contribution - spy_cash
            spy_shares += spy_cash / float(spy.iloc[position])
            reserve += reserve_cash
            reserve_target += reserve_cash
            events.append(Event(
                stamp.date().isoformat(), "contribution", contribution, signal_dd,
                f"{args.spy_share:.0%} SPY / {1.0 - args.spy_share:.0%} Treasury",
            ))

        # A recovered/new SPY high re-arms the ladder.  It does not force a sale
        # of SPY to refill Treasury; the actual reserve becomes the next budget.
        if signal_dd >= -1e-12:
            if fired:
                events.append(Event(
                    stamp.date().isoformat(), "episode_reset", 0.0, signal_dd,
                    "SPY total-return high recovered; ladder re-armed",
                ))
            fired.clear()
            episode_reserve = reserve
            reserve_target = max(reserve_target, reserve)

        triggered = []
        if rungs_enabled:
            triggered = [
                (threshold, fraction, destination)
                for threshold, fraction, destination in RUNGS
                if threshold not in fired and signal_dd <= -threshold + 1e-12
            ]

        # Check existing lots at quarter boundaries and every new ladder signal.
        if harvest_enabled and (new_quarter or triggered):
            for lot in lots:
                if lot.shares <= 1e-12:
                    continue
                signal_stock = prices[lot.ticker].iloc[signal_position]
                execution_stock = current_row.get(lot.ticker, np.nan)
                if pd.isna(signal_stock) or pd.isna(execution_stock):
                    continue
                excess = relative_excess(
                    float(signal_stock), lot.entry_price,
                    float(spy.iloc[signal_position]), lot.entry_spy,
                )
                bands = new_harvest_bands(
                    excess, lot.harvest_bands, args.relative_step
                )
                if bands <= 0:
                    continue
                shares_to_sell = min(
                    lot.shares,
                    lot.original_shares * args.harvest_share * bands,
                )
                gross = shares_to_sell * float(execution_stock)
                proceeds = gross * (1.0 - cost_rate)
                lot.shares -= shares_to_sell
                lot.harvested_shares += shares_to_sell
                lot.harvest_bands += bands

                total_before_routing = (
                    spy_shares * float(spy.iloc[position])
                    + reserve + proceeds
                    + quality_value(lots, current_row)
                )
                base_target = reserve_target
                target = base_target
                if cape_enabled and pd.notna(signal_cape) and signal_cape >= args.cape_excessive:
                    target = max(target, 0.20 * total_before_routing)
                to_reserve = min(proceeds, max(target - reserve, 0.0))
                base_to_reserve = min(
                    proceeds, max(base_target - reserve, 0.0)
                )
                to_spy = proceeds - to_reserve
                harvest_to_reserve_total += to_reserve
                harvest_to_spy_total += to_spy
                cape_incremental_reserve_total += max(
                    to_reserve - base_to_reserve, 0.0
                )
                reserve += to_reserve
                spy_shares += _buy_spy(
                    to_spy, float(spy.iloc[position]), cost_rate
                )
                events.append(Event(
                    stamp.date().isoformat(), "quality_relative_harvest", proceeds,
                    signal_dd,
                    f"{lot.ticker}; relative excess {excess:.1%}; "
                    f"{bands} new band(s); Treasury {to_reserve:.2f}; SPY {to_spy:.2f}",
                ))

        for threshold, fraction, destination in triggered:
            fired.add(threshold)
            planned = episode_reserve * fraction
            amount = min(planned, reserve)
            if amount <= 0:
                events.append(Event(
                    stamp.date().isoformat(), "rung_unfunded", 0.0, signal_dd,
                    f"-{threshold:.0%}; planned {planned:.2f}",
                ))
                continue
            actual_destination = "spy" if all_spy_rungs else destination
            if actual_destination == "spy" or not quality_enabled:
                reserve -= amount
                spy_shares += _buy_spy(
                    amount, float(spy.iloc[position]), cost_rate
                )
                events.append(Event(
                    stamp.date().isoformat(), "deploy_treasury_spy", amount,
                    signal_dd, f"-{threshold:.0%}; {fraction:.0%} of episode reserve",
                ))
                continue

            selected = market.mega_seven(
                signal_date, signal_position, prices, raw, shares
            )
            if len(selected) < 7:
                events.append(Event(
                    stamp.date().isoformat(), "quality_rung_skipped", 0.0,
                    signal_dd,
                    f"-{threshold:.0%}; only {len(selected)} market caps available",
                ))
                continue
            total_cap = sum(item["market_cap"] for item in selected)
            deployed = 0.0
            names = []
            for item in selected:
                cash = min(
                    amount * item["market_cap"] / total_cap,
                    reserve,
                )
                execution_price = current_row.get(item["ticker"], np.nan)
                if cash <= 0 or pd.isna(execution_price):
                    continue
                bought = cash * (1.0 - cost_rate) / float(execution_price)
                reserve -= cash
                deployed += cash
                lots.append(QualityLot(
                    ticker=item["ticker"], shares=bought,
                    original_shares=bought, cost=cash, entry_date=stamp,
                    entry_price=float(execution_price),
                    entry_spy=float(spy.iloc[position]),
                ))
                names.append(f"{item['ticker']} {item['market_cap'] / total_cap:.1%}")
            events.append(Event(
                stamp.date().isoformat(), "deploy_quality_seven", deployed,
                signal_dd,
                f"-{threshold:.0%}; " + "; ".join(names),
            ))

        quality = quality_value(lots, current_row)
        spy_value = spy_shares * float(spy.iloc[position])
        total = spy_value + reserve + quality
        investable_after_flow = total - contribution
        if pre_flow_total > 0 and investable_after_flow > 0:
            performance_index *= investable_after_flow / pre_flow_total
        performance_high = max(performance_high, performance_index)
        flow_dd = performance_index / performance_high - 1.0
        quality_weight = quality / total if total > 0 else 0.0
        spy_exposure = spy_value / total if total > 0 else 0.0
        max_quality_weight = max(max_quality_weight, quality_weight)
        max_spy_exposure = max(max_spy_exposure, spy_exposure)
        previous_total = total
        rows.append({
            "wealth": total,
            "performance_index": performance_index,
            "contribution": contribution,
            "flow_adjusted_drawdown": flow_dd,
            "spy": spy_value,
            "treasury": reserve,
            "quality": quality,
            "quality_weight": quality_weight,
            "spy_weight": spy_exposure,
            "spy_drawdown": float(spy_drawdown.iloc[position]),
            "episode_reserve": episode_reserve,
            "reserve_target": reserve_target,
            "cape_known": signal_cape,
        })

    path = pd.DataFrame(rows, index=spy.index)
    holdings = []
    for ticker in sorted({lot.ticker for lot in lots if lot.shares > 1e-12}):
        ticker_lots = [
            lot for lot in lots if lot.ticker == ticker and lot.shares > 1e-12
        ]
        value = sum(lot.shares for lot in ticker_lots) * float(prices[ticker].iloc[-1])
        holdings.append({
            "ticker": ticker,
            "market_value": float(value),
            "portfolio_weight": float(value / path["wealth"].iloc[-1]),
            "lots": len(ticker_lots),
            "harvest_bands": sum(lot.harvest_bands for lot in ticker_lots),
            "harvested_original_share": float(
                sum(lot.harvested_shares for lot in ticker_lots)
                / sum(lot.original_shares for lot in ticker_lots)
            ),
            "entry_dates": sorted({
                lot.entry_date.date().isoformat() for lot in ticker_lots
            }),
        })
    holdings.sort(key=lambda row: row["market_value"], reverse=True)
    path.attrs["max_quality_weight"] = max_quality_weight
    path.attrs["max_spy_weight"] = max_spy_exposure
    path.attrs["harvest_to_reserve"] = harvest_to_reserve_total
    path.attrs["harvest_to_spy"] = harvest_to_spy_total
    path.attrs["cape_incremental_reserve"] = cape_incremental_reserve_total
    return path, events, holdings


def main(argv=None):
    args = parse_args(argv)
    load_args = market.parse_args([])
    for field in (
        "start", "end", "initial", "monthly_contribution", "trade_bp",
        "refresh", "fundamentals", "cache",
    ):
        setattr(load_args, field, getattr(args, field))
    prices, raw, shares, coverage = market.load_market_data(load_args)
    cape_monthly = cape_data.load_cape()
    cape_end = cape_monthly.index[-1] + pd.offsets.MonthEnd(1)
    if args.end:
        cape_end = min(cape_end, pd.Timestamp(args.end))
    prices = prices.loc[:cape_end]
    raw = raw.loc[prices.index]
    rates = core.load_rates(prices.index)
    known_cape = cape_data.known_cape_daily(cape_monthly, prices.index)

    variants = {
        "quality_ladder": dict(
            rungs_enabled=True, quality_enabled=True, harvest_enabled=True,
            cape_enabled=True,
        ),
        "quality_ladder_no_cape": dict(
            rungs_enabled=True, quality_enabled=True, harvest_enabled=True,
            cape_enabled=False,
        ),
        "quality_ladder_no_harvest": dict(
            rungs_enabled=True, quality_enabled=True, harvest_enabled=False,
            cape_enabled=False,
        ),
        "spy_ladder": dict(
            rungs_enabled=True, quality_enabled=False, harvest_enabled=False,
            cape_enabled=False, all_spy_rungs=True,
        ),
        "static_80_20": dict(
            rungs_enabled=False, quality_enabled=False, harvest_enabled=False,
            cape_enabled=False,
        ),
    }
    paths, event_sets, holdings, metrics = {}, {}, {}, {}
    for name, settings in variants.items():
        path, events, open_holdings = simulate(
            prices, raw, shares, rates, known_cape, args,
            name=name, **settings,
        )
        paths[name] = path
        event_sets[name] = events
        holdings[name] = open_holdings
        summary = performance_metrics(path, rates, args.initial)
        summary.update({
            "ending_treasury": float(path["treasury"].iloc[-1]),
            "ending_treasury_weight": float(
                path["treasury"].iloc[-1] / path["wealth"].iloc[-1]
            ),
            "ending_quality_weight": float(path["quality_weight"].iloc[-1]),
            "max_quality_weight": float(path["quality_weight"].max()),
            "ending_reserve_target": float(path["reserve_target"].iloc[-1]),
            "ending_reserve_shortfall": float(max(
                path["reserve_target"].iloc[-1] - path["treasury"].iloc[-1], 0.0
            )),
            "harvest_to_reserve": float(path.attrs["harvest_to_reserve"]),
            "harvest_to_spy": float(path.attrs["harvest_to_spy"]),
            "cape_incremental_reserve": float(
                path.attrs["cape_incremental_reserve"]
            ),
            "events": summarize_events(events),
        })
        metrics[name] = summary

    spy_path = market.simulate_spy(prices, load_args)
    metrics["spy_1x"] = performance_metrics(spy_path, rates, args.initial)
    result = {
        "sample": {
            "start": prices.index[0].date().isoformat(),
            "end": prices.index[-1].date().isoformat(),
            "sessions": len(prices),
            "cape_source_end": cape_monthly.index[-1].date().isoformat(),
        },
        "cash_flows": {
            "initial": args.initial,
            "monthly_contribution": args.monthly_contribution,
        },
        "rules": {
            "starting_allocation": f"{args.spy_share:.0%} SPY / {1-args.spy_share:.0%} Treasury; no leverage",
            "rungs": "20% Treasury to quality at -10% SPY DD; 30% to SPY at -20%; 30% to quality at -30%; 20% to SPY at -50%",
            "drawdown": "SPY adjusted-total-return drawdown; first crossing per peak-to-recovery episode; prior-close signal, next-close trade",
            "quality_selection": "top seven date-ranked market-cap estimates from a fixed historical-leader union; market-cap weighted",
            "harvest": f"quarter-end and rung checks; sell {args.harvest_share:.0%} of original lot shares for every new {args.relative_step:.0%} relative-wealth band versus SPY",
            "routing": f"rebuild pre-fall Treasury target, then SPY; when known CAPE >= {args.cape_excessive:g}, rebuild to at least 20% current NAV first",
            "dividends": "reinvested through adjusted total returns",
            "treasury": "prior-known DGS3MO",
            "trade_cost": f"{args.trade_bp:g} bp on drawdown and harvest trades",
        },
        "variants": metrics,
        "holdings": holdings,
        "events": {
            name: [asdict(event) for event in events]
            for name, events in event_sets.items()
        },
        "coverage": coverage,
        "warnings": [
            "The point-in-time market-cap union contains current survivors and omits delisted historical leaders.",
            "Share-count history is unavailable for the early sample, so affected quality rungs are skipped rather than backfilled with hindsight.",
            "CAPE is lagged one month and the common test ends with the local September 2024 Shiller observation.",
            "Taxes, bid/ask spread beyond the fixed trade cost, market impact, and stock-specific delisting gaps are omitted.",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    daily = pd.DataFrame(index=prices.index)
    daily["spy_1x_wealth"] = spy_path["wealth"]
    daily["spy_1x_performance"] = spy_path["performance_index"]
    for name, path in paths.items():
        for column in (
            "wealth", "performance_index", "treasury", "quality",
            "quality_weight", "spy_weight", "spy_drawdown",
            "episode_reserve", "reserve_target",
        ):
            daily[f"{name}_{column}"] = path[column]
    daily.to_csv(args.out / "daily.csv", index_label="date")
    print(json.dumps({
        "sample": result["sample"],
        "cash_flows": result["cash_flows"],
        "variants": metrics,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
