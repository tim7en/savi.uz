"""Why 20x ruins a strategy that a matched drawdown says is worth 0.20x.

The vol-stretch fade scores 1.68 Sharpe and still destroys an account at 20x.
That is not a contradiction and it is not really about leverage, so this pulls
the number apart.

The identity underneath everything here is that a position's risk is its size
times its stop distance.  Fixing the size therefore *floats* the risk, and the
stop distance in this setup is not a constant: it is two implied daily moves
below a limit already two moves under a crash close, which on a quiet name is a
few percent of price and on a broken one is a third of it.  A single leverage
number applied across that range is the whole problem.

Three things are reported, in the order that makes the argument.

*The stop distribution*, because it is the denominator.  If it spans an order
of magnitude then no single leverage can be right for both ends of it.

*The fixed-leverage ladder*, which is the proposal: the same notional multiple
on every trade.  Ruin is reported at each rung along with the drawdown, and the
rung where it starts is the answer to "how much can I use".

*The fixed-fractional ladder*, which is the alternative: risk a constant share
of equity and let the size float instead.  This is what the rest of the
programme does, and it is why the banked book's leverage is an output.  The
gross leverage it implies is reported per trade so the two can be compared on
the same axis rather than on their own terms.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_vol_stretch_zones as study  # noqa: E402

LEVERAGE_RUNGS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0)
RISK_RUNGS = (0.0025, 0.005, 0.01, 0.02, 0.03)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vol-source", choices=("implied", "realized"),
                        default="realized")
    parser.add_argument("--stretch", type=float, default=2.0)
    parser.add_argument("--offset", type=float, default=2.0)
    parser.add_argument("--max-positions", type=int, default=6)
    return parser.parse_args(argv)


def equity_path(taken, weight):
    """Compounded equity from per-session net R, weighted into equity terms."""
    by_day = defaultdict(float)
    for trade in taken:
        by_day[trade["exit"]] += trade["r"] * weight(trade)
    days = sorted(by_day)
    nav, peak, worst, trough_day = 1000.0, 1000.0, 0.0, days[0] if days else ""
    for day in days:
        nav = max(0.0, nav + by_day[day] * nav)
        peak = max(peak, nav)
        drop = nav / peak - 1.0 if peak > 0 else 0.0
        if drop < worst:
            worst, trough_day = drop, day
        if nav <= 0.0:
            return 0.0, -1.0, day, days
    return nav, worst, trough_day, days


def summarise(nav, worst, trough_day, days):
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25
    cagr = (nav / 1000.0) ** (1 / years) - 1 if nav > 0 else -1.0
    return cagr, worst, trough_day


def main(argv=None) -> int:
    options = parse_args(argv)
    args = study.parse_args(["--vol-source", options.vol_source])
    args.max_positions = options.max_positions

    book = study.load(args)
    panel = (study.realized_moves(book) if options.vol_source == "realized"
             else study.implied_moves(args.options, sorted(book), args.start))
    trades = study.build(book, panel, args, options.stretch, options.offset,
                         "vol", "volume")
    taken = study.cap(trades, args.max_positions, random.Random(0))
    print(f"{len(trades):,d} trades offered, {len(taken):,d} taken under a "
          f"{args.max_positions}-slot cap, {options.vol_source} vol, "
          f"{options.stretch:g}x stretch, {options.offset:g} moves below\n")

    # ---- the denominator -------------------------------------------------
    stops = sorted(t["stop_pct"] for t in taken)
    def at(share):
        return stops[min(int(share * len(stops)), len(stops) - 1)]
    print("########## the stop distance, which is the denominator ##########")
    print("  percent of price risked per share, per trade")
    for label, value in (("5th percentile", at(0.05)), ("25th", at(0.25)),
                         ("median", at(0.50)), ("75th", at(0.75)),
                         ("95th", at(0.95)), ("worst", stops[-1])):
        print(f"    {label:16s} {value:>7.2%}")
    print(f"  the widest stop is {stops[-1] / at(0.05):.0f}x the narrowest, so one "
          f"leverage cannot fit both")

    by_name = defaultdict(list)
    for trade in taken:
        by_name[trade["ticker"]].append(trade["stop_pct"])
    ranked = sorted(by_name.items(), key=lambda kv: -statistics.median(kv[1]))
    print("\n  widest and narrowest names, by median stop")
    for ticker, values in ranked[:4]:
        print(f"    {ticker:6s} {statistics.median(values):>7.2%}  ({len(values)} trades)")
    print("    ...")
    for ticker, values in ranked[-3:]:
        print(f"    {ticker:6s} {statistics.median(values):>7.2%}  ({len(values)} trades)")

    # ---- the proposal ----------------------------------------------------
    print(f"\n########## fixed leverage: the same notional on every trade ##########")
    print(f"  {'gross':>7s} {'per slot':>9s} {'risk median':>12s} {'risk worst':>11s} "
          f"{'max DD':>9s} {'CAGR':>10s}  outcome")
    for gross in LEVERAGE_RUNGS:
        per_slot = gross / args.max_positions
        def weight(trade, size=per_slot):
            return size * trade["stop_pct"]
        nav, worst, trough, days = equity_path(taken, weight)
        cagr, worst, trough = summarise(nav, worst, trough, days)
        risks = sorted(per_slot * t["stop_pct"] for t in taken)
        verdict = ("RUINED" if nav <= 0 else
                   "unholdable" if worst < -0.60 else
                   "survivable" if worst < -0.25 else "comfortable")
        print(f"  {gross:>6.2f}x {per_slot:>8.2f}x {statistics.median(risks):>11.2%} "
              f"{risks[-1]:>10.2%} {worst:>9.1%} {cagr:>9.1%}  {verdict}")

    # ---- the alternative -------------------------------------------------
    print(f"\n########## fixed fractional: the same risk, floating size ##########")
    print(f"  {'risk':>6s} {'max DD':>9s} {'CAGR':>10s} {'median lev':>11s} "
          f"{'95th lev':>9s} {'worst lev':>10s}")
    for fraction in RISK_RUNGS:
        nav, worst, trough, days = equity_path(taken, lambda t, f=fraction: f)
        cagr, worst, trough = summarise(nav, worst, trough, days)
        levers = sorted(fraction / t["stop_pct"] for t in taken)
        print(f"  {fraction:>5.2%} {worst:>9.1%} {cagr:>9.1%} "
              f"{statistics.median(levers):>10.2f}x "
              f"{levers[int(0.95 * len(levers))]:>8.2f}x {levers[-1]:>9.2f}x")

    # ---- where it broke --------------------------------------------------
    print(f"\n########## the trades that do the damage at 20x ##########")
    per_slot = 20.0 / args.max_positions
    worst_trades = sorted(taken, key=lambda t: t["r"] * per_slot * t["stop_pct"])[:6]
    print(f"  {'ticker':7s} {'entry':11s} {'stop%':>7s} {'equity at risk':>15s} "
          f"{'R':>7s} {'equity hit':>11s}")
    for trade in worst_trades:
        risk = per_slot * trade["stop_pct"]
        print(f"  {trade['ticker']:7s} {trade['entry']:11s} "
              f"{trade['stop_pct']:>7.2%} {risk:>14.1%} {trade['r']:>+7.2f} "
              f"{trade['r'] * risk:>10.1%}")

    nav, worst, trough, days = equity_path(
        taken, lambda t: per_slot * t["stop_pct"])
    print(f"\n  deepest drawdown at 20x: {worst:.1%}, trough {trough}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
