"""Stage a SPY reverse-strategy reserve at 20/30/50/80% account drawdowns.

The reserve receives 10% of positive calendar-year trading P&L and optional
external contributions.  It earns DGS3MO while parked.  Within each drawdown
episode, reserve deployment is split into four approximately equal tranches:
25% of available cash at -20%, one third of what remains at -30%, one half at
-50%, and all remaining cash at -80%.  Thresholds rearm only after the
flow-adjusted strategy NAV regains its previous high.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_spy_drawdown_hysteresis import backtest, drawdown_rungs, load_rates, load_spy


THRESHOLDS = (0.20, 0.30, 0.50, 0.80)
DEPLOY_FRACTIONS = (0.25, 1.0 / 3.0, 0.50, 1.0)


@dataclass
class Event:
    date: str
    kind: str
    amount: float
    threshold: float | None
    strategy_drawdown: float


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="1993-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--initial", type=float, default=10_000.0)
    parser.add_argument("--harvest", type=float, default=0.10)
    parser.add_argument("--monthly", type=float, default=100.0,
                        help="monthly contribution used in the cadence comparison")
    parser.add_argument("--spread", type=float, default=0.01)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/spy_staged_funding"))
    return parser.parse_args(argv)


def contribution_schedule(index: pd.DatetimeIndex, cadence: str,
                          monthly_budget: float) -> pd.Series:
    """Last-session contributions; bimonthly uses 2x to hold annual cash fixed."""
    schedule = pd.Series(0.0, index=index)
    if cadence == "none":
        return schedule
    month_end = index.to_series().dt.to_period("M").ne(
        index.to_series().shift(-1).dt.to_period("M"))
    if cadence == "monthly":
        schedule.loc[month_end.values] = monthly_budget
    elif cadence == "bimonthly":
        eligible = month_end.values & (index.month % 2 == 0)
        schedule.loc[eligible] = monthly_budget * 2.0
    else:
        raise ValueError(f"unknown cadence: {cadence}")
    return schedule


def xirr(cash_flows: list[tuple[pd.Timestamp, float]]) -> float:
    """Money-weighted annual return for negative contributions and final wealth."""
    origin = cash_flows[0][0]

    def present_value(rate: float) -> float:
        return sum(value / (1.0 + rate) ** ((stamp - origin).days / 365.2425)
                   for stamp, value in cash_flows)

    low, high = -0.9999, 1.0
    while present_value(high) > 0.0 and high < 1_000.0:
        high *= 2.0
    for _ in range(100):
        middle = (low + high) / 2.0
        if present_value(middle) > 0.0:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def simulate(strategy_return: pd.Series, rates: pd.Series,
             contributions: pd.Series, initial: float,
             harvest_share: float) -> tuple[pd.DataFrame, list[Event], dict]:
    index = strategy_return.index
    stamps = index.to_series()
    year_end = stamps.dt.year.ne(stamps.shift(-1).dt.year) & stamps.dt.month.eq(12)
    days = stamps.diff().dt.days.fillna(0.0)
    known_rate = rates.shift(1).ffill().bfill()

    main, reserve = initial, 0.0
    strategy_nav, strategy_peak = 1.0, 1.0
    annual_profit = 0.0
    fired: set[float] = set()
    events: list[Event] = []
    cash_flows = [(index[0], -initial)]
    rows = []

    for i, stamp in enumerate(index):
        if i:
            reserve *= 1.0 + float(known_rate.iloc[i]) * float(days.iloc[i]) / 365.0
            profit = main * float(strategy_return.iloc[i])
            main += profit
            annual_profit += profit
            strategy_nav *= 1.0 + float(strategy_return.iloc[i])

        contribution = float(contributions.iloc[i])
        if contribution > 0.0:
            reserve += contribution
            cash_flows.append((stamp, -contribution))

        if strategy_nav >= strategy_peak * (1.0 - 1e-12):
            strategy_peak = max(strategy_peak, strategy_nav)
            fired.clear()
        drawdown = strategy_nav / strategy_peak - 1.0

        for threshold, fraction in zip(THRESHOLDS, DEPLOY_FRACTIONS):
            if threshold in fired or drawdown > -threshold:
                continue
            fired.add(threshold)
            amount = reserve * fraction
            if amount > 0.0:
                reserve -= amount
                main += amount
                events.append(Event(stamp.date().isoformat(), "deploy", amount,
                                    threshold, drawdown))

        if bool(year_end.iloc[i]):
            amount = min(max(annual_profit, 0.0) * harvest_share, max(main, 0.0))
            if amount > 0.0:
                main -= amount
                reserve += amount
                events.append(Event(stamp.date().isoformat(), "harvest", amount,
                                    None, drawdown))
            annual_profit = 0.0

        rows.append({"main_account": main, "reserve": reserve,
                     "combined_wealth": main + reserve,
                     "strategy_nav": strategy_nav,
                     "strategy_drawdown": drawdown,
                     "external_contribution": contribution})

    cash_flows.append((index[-1], main + reserve))
    path = pd.DataFrame(rows, index=index)
    # External deposits are cash flows, not returns.  Remove them before
    # constructing the combined sleeve's time-weighted NAV and drawdown.
    combined_return = ((path["combined_wealth"] - path["external_contribution"])
                       / path["combined_wealth"].shift() - 1.0)
    combined_return.iloc[0] = 0.0
    path["combined_flow_adjusted_nav"] = (1.0 + combined_return).cumprod()
    combined_dd = (path["combined_flow_adjusted_nav"]
                   / path["combined_flow_adjusted_nav"].cummax() - 1.0)
    years = (index[-1] - index[0]).days / 365.2425
    stats = {
        "terminal": float(path["combined_wealth"].iloc[-1]),
        "total_external_contributions": float(contributions.sum()),
        "total_cash_including_initial": float(initial + contributions.sum()),
        "xirr": float(xirr(cash_flows)),
        "simple_cagr_if_no_external_flows": (
            float((path["combined_wealth"].iloc[-1] / initial) ** (1.0 / years) - 1.0)
            if contributions.sum() == 0.0 else None),
        "max_combined_drawdown_flow_adjusted": float(combined_dd.min()),
        "ending_reserve": float(path["reserve"].iloc[-1]),
        "total_harvested": float(sum(e.amount for e in events if e.kind == "harvest")),
        "total_deployed": float(sum(e.amount for e in events if e.kind == "deploy")),
        "deployments": int(sum(e.kind == "deploy" for e in events)),
    }
    return path, events, stats


def threshold_history(drawdown: pd.Series, spy_wealth: pd.Series) -> list[dict]:
    fired: set[float] = set()
    events: list[tuple[pd.Timestamp, float]] = []
    for stamp, value in drawdown.items():
        if value >= -1e-12:
            fired.clear()
        for threshold in THRESHOLDS:
            if threshold not in fired and value <= -threshold:
                fired.add(threshold)
                events.append((stamp, threshold))

    rows = []
    for threshold in THRESHOLDS:
        dates = [stamp for stamp, level in events if level == threshold]
        item = {"threshold": threshold, "count": len(dates),
                "dates": [stamp.date().isoformat() for stamp in dates]}
        for horizon in (1, 3, 5):
            returns = []
            for stamp in dates:
                target = stamp + pd.Timedelta(days=365.2425 * horizon)
                location = spy_wealth.index.searchsorted(target)
                if location < len(spy_wealth):
                    returns.append((spy_wealth.iloc[location] / spy_wealth.loc[stamp])
                                   ** (1.0 / horizon) - 1.0)
            item[f"forward_{horizon}y_n"] = len(returns)
            item[f"forward_{horizon}y_median_annualized"] = (
                float(np.median(returns)) if returns else None)
            item[f"forward_{horizon}y_positive_share"] = (
                float(np.mean(np.asarray(returns) > 0.0)) if returns else None)
        rows.append(item)
    return rows


def make_plot(paths: dict[str, pd.DataFrame], baseline: pd.Series,
              hold: pd.Series, events: list[Event], out: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0]}, constrained_layout=True)
    top.plot(baseline.index, baseline, color="#64748b", lw=1.5,
             label="Reverse, no reserve")
    top.plot(paths["none"].index, paths["none"]["combined_wealth"],
             color="#0f766e", lw=2.0, label="Staged reserve, no external cash")
    top.plot(paths["monthly"].index, paths["monthly"]["combined_wealth"],
             color="#2563eb", lw=2.0, label="+$100 monthly to reserve")
    top.plot(paths["bimonthly"].index, paths["bimonthly"]["combined_wealth"],
             color="#7c3aed", lw=1.4, ls="--", label="+$200 every 2 months")
    top.plot(hold.index, hold, color="#111827", lw=1.2, label="SPY 1x")
    top.set_yscale("log")
    top.set_ylabel("Combined wealth, USD (log scale)")
    top.set_title("Staged reserve mobilization at 20/30/50/80% strategy drawdowns")
    top.legend(loc="upper left", ncol=2)

    dd = paths["none"]["strategy_drawdown"] * 100.0
    bottom.fill_between(dd.index, dd, 0.0, color="#fecaca", alpha=0.55)
    bottom.plot(dd.index, dd, color="#b91c1c", lw=1.2)
    colors = {0.20: "#eab308", 0.30: "#f97316", 0.50: "#dc2626", 0.80: "#7f1d1d"}
    for threshold in THRESHOLDS:
        bottom.axhline(-threshold * 100.0, color=colors[threshold], lw=0.9,
                       ls="--", alpha=0.8, label=f"-{threshold:.0%} rung")
    for event in events:
        if event.kind == "deploy":
            bottom.scatter(pd.Timestamp(event.date), event.strategy_drawdown * 100.0,
                           color=colors[event.threshold], s=22, zorder=4)
    bottom.set_ylabel("Strategy drawdown")
    bottom.set_xlabel("Date")
    bottom.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    bottom.legend(loc="lower right", ncol=2)
    figure.savefig(out / "staged_funding.png", dpi=170)
    figure.savefig(out / "staged_funding.svg")
    plt.close(figure)


def main(argv=None) -> int:
    args = parse_args(argv)
    prices = load_spy(args.start, args.end)
    rates = load_rates(prices.index)
    asset_return = prices["Adj Close"].pct_change().fillna(0.0)
    rungs, _, _ = drawdown_rungs(prices["Adj Close"])
    reverse_target = rungs.map({0: 3.0, 1: 2.0, 2: 1.0})
    reverse = backtest(asset_return, reverse_target, rates, args.spread,
                       initial=args.initial)
    hold = backtest(asset_return, pd.Series(1.0, index=prices.index), rates,
                    args.spread, initial=args.initial)

    paths, event_sets, stats = {}, {}, {}
    for cadence in ("none", "monthly", "bimonthly"):
        schedule = contribution_schedule(prices.index, cadence, args.monthly)
        paths[cadence], event_sets[cadence], stats[cadence] = simulate(
            reverse["strategy_return"], rates, schedule,
            args.initial, args.harvest)

    frequency = threshold_history(paths["none"]["strategy_drawdown"], hold["wealth"])
    report = {
        "assumptions": {
            "initial": args.initial, "annual_profit_harvest_share": args.harvest,
            "funding_rungs": list(THRESHOLDS),
            "monthly_case": args.monthly,
            "bimonthly_case": args.monthly * 2.0,
            "vault_yield": "DGS3MO",
            "leverage_financing": f"DGS3MO + {args.spread:.2%}",
        },
        "sample": {"start": prices.index[0].date().isoformat(),
                   "end": prices.index[-1].date().isoformat()},
        "results": stats,
        "threshold_history": frequency,
        "events_no_external_contributions": [e.__dict__ for e in event_sets["none"]],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for cadence, path in paths.items():
        path.to_csv(args.out / f"daily_{cadence}.csv", index_label="date")
    pd.DataFrame({
        "reverse_no_reserve": reverse["wealth"],
        "spy_hold": hold["wealth"],
        "staged_no_external": paths["none"]["combined_wealth"],
        "monthly": paths["monthly"]["combined_wealth"],
        "bimonthly": paths["bimonthly"]["combined_wealth"],
        "strategy_drawdown": paths["none"]["strategy_drawdown"],
    }).to_csv(args.out / "comparison.csv", index_label="date")
    make_plot(paths, reverse["wealth"], hold["wealth"], event_sets["none"], args.out)
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.out / 'staged_funding.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
