"""Money management for the Turtle trade list: stepped risk, reserve, and DCA.

The backtest elsewhere in this project sizes trades as a continuous fraction of
equity.  That is not how anyone actually trades.  This script models the scheme
the way it gets run in practice:

* A cash **reserve** is held back, so only a share of equity backs risk.
* Risk is **stepped, not continuous**: a fixed amount per R that stays put until
  equity crosses a doubling threshold, then doubles.  This avoids resizing on
  every trade and matches how the original Turtles periodically re-struck unit
  size rather than recomputing it constantly.
* **Contributions** arrive on a schedule, which raises a question the doubling
  rule alone does not answer: does deposited cash count towards the next
  doubling, or should only trading profit earn a step up?

Because contributions distort final equity, results are reported on a unitised
NAV -- the fund convention.  A deposit buys units at the current NAV and so
leaves NAV untouched; drawdown and growth are then properties of the strategy
rather than of the deposit schedule.  Money-weighted return (IRR) is reported
alongside, because that is what the depositor actually earns.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", type=Path,
                        default=Path("out/strategy/turtle_trades.csv"))
    parser.add_argument("--interval", default="daily")
    parser.add_argument("--system", default="System 2 (55/20)")
    parser.add_argument("--direction", choices=("both", "long", "short"), default="both")
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--initial", type=float, default=1000.0)
    parser.add_argument("--reserve", type=float, default=0.30)
    parser.add_argument("--risk-per-1000", type=float, default=2.0,
                        help="dollars risked per R for each $1000 of backing capital")
    parser.add_argument("--monthly", type=float, default=100.0,
                        help="regular contribution, 0 to disable")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/turtle_moneymgmt.json"))
    return parser.parse_args(argv)


def load_trades(path: Path, interval: str, system: str, direction: str):
    with path.open(encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["interval"] == interval and row["system"] == system
        ]
    for row in rows:
        row["net_r"] = float(row["net_r"])
        row["direction"] = int(row["direction"])
    if direction == "long":
        rows = [r for r in rows if r["direction"] > 0]
    elif direction == "short":
        rows = [r for r in rows if r["direction"] < 0]
    return rows


def cap(rows, limit, rng=None):
    shuffled = list(rows)
    if rng is not None:
        rng.shuffle(shuffled)
    ordered = sorted(shuffled, key=lambda row: row["entry_timestamp"])
    live, taken = [], []
    for row in ordered:
        live = [x for x in live if x["exit_timestamp"] > row["entry_timestamp"]]
        if len(live) >= limit:
            continue
        live.append(row)
        taken.append(row)
    return taken


def every_day(first: str, last: str) -> list[str]:
    """Every calendar date in the span.

    The walk has to cover days with no trades, otherwise a deposit dated to the
    first of a month simply never happens: month starts are usually weekends or
    quiet days, and keying the loop to trade dates silently dropped most of the
    contribution schedule.
    """
    start, end = date.fromisoformat(first), date.fromisoformat(last)
    out, day = [], start
    while day <= end:
        out.append(day.isoformat())
        day += timedelta(days=1)
    return out


def contribution_dates(first: str, last: str, monthly: float):
    """First day of each month after the start."""
    if monthly <= 0:
        return set()
    start = date.fromisoformat(first)
    end = date.fromisoformat(last)
    days = set()
    year, month = start.year, start.month
    while True:
        month += 1
        if month > 12:
            month, year = 1, year + 1
        stamp = date(year, month, 1)
        if stamp > end:
            break
        days.add(stamp.isoformat())
    return days


def simulate(trades, *, initial, reserve, risk_per_1000, monthly, policy,
             step_down, calendar):
    """Walk the calendar applying trade results, deposits and risk steps.

    ``policy`` decides what a deposit does to the doubling threshold:
      "count"    deposits count as equity, so they can trigger a step up
      "earn"     the threshold rises with deposits, so only profit steps up
      "annual"   risk is re-struck once a year from whatever equity exists
    ``step_down`` halves risk when equity falls back through a threshold.
    """
    by_day = defaultdict(float)
    for row in trades:
        by_day[row["exit_timestamp"][:10]] += row["net_r"]
    deposits = contribution_dates(calendar[0], calendar[-1], monthly)

    equity = initial
    contributed = initial
    nav = 1.0
    units = initial
    base_equity = initial          # equity level the current risk was struck at
    risk = risk_per_1000 * (initial * (1.0 - reserve)) / 1000.0
    peak_nav, max_dd = nav, 0.0
    ruined = False
    flows = [(calendar[0], -initial)]
    nav_path, equity_path, risk_path = [], [], []
    last_reset_year = calendar[0][:4]

    for day in calendar:
        if day in deposits and monthly > 0:
            equity += monthly
            contributed += monthly
            units += monthly / nav if nav > 0 else 0.0
            flows.append((day, -monthly))
            if policy == "earn":
                base_equity += monthly   # deposit does not shorten the path

        gain = by_day.get(day, 0.0) * risk
        if gain:
            before = equity
            equity = max(0.0, equity + gain)
            if before > 0 and units > 0:
                nav *= equity / before if before else 1.0

        if policy == "annual":
            if day[:4] != last_reset_year:
                last_reset_year = day[:4]
                risk = risk_per_1000 * (equity * (1.0 - reserve)) / 1000.0
        else:
            while equity >= base_equity * 2:
                base_equity *= 2
                risk *= 2
            if step_down:
                while equity < base_equity / 2 and risk > 1e-9:
                    base_equity /= 2
                    risk /= 2

        peak_nav = max(peak_nav, nav)
        max_dd = min(max_dd, nav / peak_nav - 1.0 if peak_nav > 0 else 0.0)
        if equity <= initial * 0.02:
            ruined = True
        nav_path.append(nav)
        equity_path.append(equity)
        risk_path.append(risk)

    flows.append((calendar[-1], equity))
    return {
        "final_equity": equity,
        "contributed": contributed,
        "nav": nav,
        "max_drawdown": max_dd,
        "ruined": ruined,
        "irr": irr(flows),
        "final_risk": risk,
        "nav_path": nav_path,
        "equity_path": equity_path,
        "risk_path": risk_path,
    }


def irr(flows):
    """Annualised money-weighted return from dated cash flows."""
    base = date.fromisoformat(flows[0][0])
    years = [(date.fromisoformat(d) - base).days / 365.25 for d, _ in flows]
    amounts = [a for _, a in flows]

    def npv(rate):
        return sum(a / (1.0 + rate) ** t for a, t in zip(amounts, years))

    low, high = -0.95, 5.0
    if npv(low) * npv(high) > 0:
        return None
    for _ in range(200):
        mid = (low + high) / 2
        if npv(low) * npv(mid) <= 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2


def main(argv=None):
    args = parse_args(argv)
    rows = load_trades(args.trades, args.interval, args.system, args.direction)
    stamps = sorted({r["exit_timestamp"][:10] for r in rows}
                    | {r["entry_timestamp"][:10] for r in rows})
    calendar = every_day(stamps[0], stamps[-1])

    policies = [
        ("Deposits count towards the next double", "count", True),
        ("Only profit earns a double", "earn", True),
        ("Re-struck once a year", "annual", False),
        ("Deposits count, risk never steps down", "count", False),
    ]

    report = {"settings": {
        "initial": args.initial, "reserve": args.reserve,
        "risk_per_1000": args.risk_per_1000, "monthly": args.monthly,
        "direction": args.direction, "max_positions": args.max_positions,
        "trials": args.trials,
    }, "policies": {}}

    for label, policy, step_down in policies:
        finals, navs, dds, irrs, ruins = [], [], [], [], 0
        for seed in range(args.trials):
            taken = cap(rows, args.max_positions, random.Random(seed))
            out = simulate(
                taken, initial=args.initial, reserve=args.reserve,
                risk_per_1000=args.risk_per_1000, monthly=args.monthly,
                policy=policy, step_down=step_down, calendar=calendar,
            )
            finals.append(out["final_equity"])
            navs.append(out["nav"])
            dds.append(out["max_drawdown"])
            if out["irr"] is not None:
                irrs.append(out["irr"])
            ruins += out["ruined"]
        pick = lambda xs, f: sorted(xs)[int(f * len(xs))]
        report["policies"][label] = {
            "final_p05": pick(finals, .05), "final_median": pick(finals, .5),
            "final_p95": pick(finals, .95),
            "nav_median": pick(navs, .5),
            "dd_median": pick(dds, .5), "dd_worst": min(dds),
            "irr_median": pick(irrs, .5) if irrs else None,
            "ruin_rate": ruins / args.trials,
        }

    contributed = args.initial + args.monthly * 12 * 9.4
    print(f"{args.direction} | ${args.initial:,.0f} start, ${args.monthly:,.0f}/month, "
          f"{args.reserve:.0%} reserve, ${args.risk_per_1000:.2f} per R per $1000")
    print(f"total contributed over the period: ~${contributed:,.0f}\n")
    head = f"  {'policy':40s} {'median $':>11s} {'5-95%':>21s} {'IRR':>7s} {'maxDD':>7s} {'ruin':>6s}"
    print(head)
    for label, data in report["policies"].items():
        irr_s = f"{data['irr_median']:.1%}" if data["irr_median"] is not None else "n/a"
        print(f"  {label:40s} ${data['final_median']:>10,.0f} "
              f"${data['final_p05']:>9,.0f}-${data['final_p95']:>9,.0f} "
              f"{irr_s:>7s} {data['dd_median']:>7.1%} {data['ruin_rate']:>6.0%}")

    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
