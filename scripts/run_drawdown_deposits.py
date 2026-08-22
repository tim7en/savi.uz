"""Adding $1,000 when the account is down, and whether the timing earns anything.

This is not the reserve-deployment study, which failed.  That one held capital
back and mobilised it into a drawdown -- a martingale on a fixed pot, where more
exposure after losses is the whole risk.  This adds *outside* money, and the
distinction matters because the arithmetic is different: fresh capital deposited
during a decline buys more units, so if the strategy ever recovers, terminal
wealth rises.  That is not an edge.  It is what compounding does to a larger
principal.

So the question has to be posed with a denominator that cannot be gamed, and
against controls that isolate the *timing* from the *contributing*.

*Money-weighted return.*  Terminal value is meaningless when the arms contribute
different amounts at different times, so every arm is scored by the internal
rate of return over its own actual cash flows.  An arm that ends richer purely
because it put more in shows no improvement here.

*A fixed-schedule control.*  The same number of deposits, evenly spaced, with no
reference to the equity curve at all.  This is ordinary paying-in, and it is the
bar the drawdown rule has to clear.  If they tie, "buy the dip with new money"
is just "pay in regularly" wearing a costume.

*The reversal.*  Deposit at new equity highs instead.  A timing rule that
performs no better than its own opposite is not a timing rule.

The drawdown rule arms once per episode: it deposits on first crossing the
trigger and does not deposit again until the account has recovered to within 10%
of its peak.  Without that, an account sitting at -25% for a year would deposit
every session and the test would measure deposit frequency instead of timing.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_earnings_strategy as base  # noqa: E402
import run_vol_stretch_zones as shared  # noqa: E402

TRIGGERS = (0.20, 0.30, 0.40)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path,
                        default=Path("data/13f/alphavantage_daily.db"))
    parser.add_argument("--equity", type=Path, default=Path("data/equity/equity.db"))
    parser.add_argument("--earnings", type=Path, default=Path("data/sp500_data"))
    parser.add_argument("--split", default="2013-01-01")
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--maker-bp", type=float, default=2.5)
    parser.add_argument("--taker-bp", type=float, default=5.0)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--start-cash", type=float, default=10_000.0)
    parser.add_argument("--deposit", type=float, default=1_000.0)
    parser.add_argument("--risk", type=float, default=0.01,
                        help="risk budget per trade, as a share of equity")
    parser.add_argument("--hold", type=int, default=40)
    parser.add_argument("--stop-mult", type=float, default=3.0)
    parser.add_argument("--slots", type=int, default=12)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/drawdown_deposits.json"))
    return parser.parse_args(argv)


def daily_stream(taken, fraction, cap=20.0):
    """Per-session fractional return of the book, before any cash flow."""
    per_day = defaultdict(float)
    for t in taken:
        lever = min(fraction / t["stop_pct"], cap)
        per_day[t["exit"]] += t["r"] * lever * t["stop_pct"]
    return sorted(per_day), per_day


def simulate(days, per_day, args, rule):
    """Walk the equity path, applying ``rule`` to decide each session's deposit."""
    nav = args.start_cash
    peak = nav
    worst = 0.0
    armed = True
    flows = [(days[0], -args.start_cash)]
    deposits = []
    for index, day in enumerate(days):
        nav *= (1.0 + per_day[day])
        peak = max(peak, nav)
        drop = nav / peak - 1.0 if peak > 0 else 0.0
        worst = min(worst, drop)
        amount = rule(index, day, drop, nav, peak, armed)
        if amount:
            nav += amount
            flows.append((day, -amount))
            deposits.append({"day": day, "drawdown": drop, "amount": amount})
            armed = False
        if drop > -0.10:
            armed = True
    flows.append((days[-1], nav))
    return nav, worst, flows, deposits


def xirr(flows, low=-0.95, high=5.0):
    """Money-weighted return over irregular cash flows."""
    start = date.fromisoformat(flows[0][0])

    def npv(rate):
        total = 0.0
        for day, amount in flows:
            years = (date.fromisoformat(day) - start).days / 365.25
            total += amount / ((1.0 + rate) ** years)
        return total

    if npv(low) * npv(high) > 0:
        return float("nan")
    for _ in range(200):
        mid = (low + high) / 2
        if npv(low) * npv(mid) <= 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2


def main(argv=None) -> int:
    args = parse_args(argv)
    book, dropped = base.load_book(args)
    earnings = base.load_earnings(args.earnings, sorted(book))
    events, _ = base.build_events(book, earnings, args)
    chosen = [e for e in events
              if e["surprise_pct"] > 0 and e["reaction"] > 0
              and e["day"] >= args.split]
    trades = base.run(book, chosen, args.hold, args.stop_mult, args)
    taken = shared.cap(trades, args.slots, random.Random(0))
    days, per_day = daily_stream(taken, args.risk)
    span = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25
    print(f"beat-and-rose book, out of sample {days[0]} to {days[-1]} "
          f"({span:.1f} years)")
    print(f"{len(taken):,d} trades taken, {args.risk:.1%} risk per trade, "
          f"starting ${args.start_cash:,.0f}, deposits of ${args.deposit:,.0f}\n")

    def hold_rule(*_):
        return 0.0

    plain_nav, plain_dd, plain_flows, _ = simulate(days, per_day, args, hold_rule)
    plain_irr = xirr(plain_flows)
    report = {"span_years": span, "trades": len(taken),
              "baseline": {"terminal": plain_nav, "max_drawdown": plain_dd,
                           "contributed": args.start_cash, "irr": plain_irr}}
    print(f"  {'arm':38s} {'deposits':>9s} {'paid in':>10s} {'ending':>12s} "
          f"{'money-wtd':>10s} {'max DD':>8s}")
    print(f"  {'no deposits (baseline)':38s} {0:>9d} "
          f"${args.start_cash:>9,.0f} ${plain_nav:>11,.0f} {plain_irr:>10.1%} "
          f"{plain_dd:>8.1%}")

    results = {}
    for trigger in TRIGGERS:
        def rule(index, day, drop, nav, peak, armed, t=trigger):
            return args.deposit if armed and drop <= -t else 0.0
        nav, dd, flows, deposits = simulate(days, per_day, args, rule)
        paid = args.start_cash + sum(d["amount"] for d in deposits)
        rate = xirr(flows)
        results[f"drawdown {trigger:.0%}"] = {
            "deposits": len(deposits), "contributed": paid, "terminal": nav,
            "irr": rate, "max_drawdown": dd,
            "dates": [d["day"] for d in deposits]}
        print(f"  {f'deposit at {trigger:.0%} drawdown':38s} {len(deposits):>9d} "
              f"${paid:>9,.0f} ${nav:>11,.0f} {rate:>10.1%} {dd:>8.1%}")

    # tiered: one deposit at each threshold within the same episode
    def tiered(index, day, drop, nav, peak, armed, seen=set()):
        for t in TRIGGERS:
            if drop <= -t and t not in seen:
                seen.add(t)
                return args.deposit
        if drop > -0.10:
            seen.clear()
        return 0.0

    nav, dd, flows, deposits = simulate(days, per_day, args, tiered)
    paid = args.start_cash + sum(d["amount"] for d in deposits)
    results["tiered 20/30/40"] = {"deposits": len(deposits), "contributed": paid,
                                  "terminal": nav, "irr": xirr(flows),
                                  "max_drawdown": dd}
    print(f"  {'tiered, one at each of 20/30/40%':38s} {len(deposits):>9d} "
          f"${paid:>9,.0f} ${nav:>11,.0f} {xirr(flows):>10.1%} {dd:>8.1%}")

    # controls, matched on the number of deposits the 20% rule made
    count = results["drawdown 20%"]["deposits"]
    print()
    if count:
        step = max(1, len(days) // (count + 1))
        schedule = {days[min(step * (i + 1), len(days) - 1)] for i in range(count)}

        def spaced(index, day, drop, nav, peak, armed):
            return args.deposit if day in schedule else 0.0

        nav, dd, flows, deposits = simulate(days, per_day, args, spaced)
        paid = args.start_cash + sum(d["amount"] for d in deposits)
        results["evenly spaced (control)"] = {
            "deposits": len(deposits), "contributed": paid, "terminal": nav,
            "irr": xirr(flows), "max_drawdown": dd}
        print(f"  {'same count, evenly spaced (control)':38s} "
              f"{len(deposits):>9d} ${paid:>9,.0f} ${nav:>11,.0f} "
              f"{xirr(flows):>10.1%} {dd:>8.1%}")

        highs = {"n": 0}

        def at_highs(index, day, drop, nav, peak, armed):
            if drop > -0.005 and highs["n"] < count and index % 21 == 0:
                highs["n"] += 1
                return args.deposit
            return 0.0

        nav, dd, flows, deposits = simulate(days, per_day, args, at_highs)
        paid = args.start_cash + sum(d["amount"] for d in deposits)
        results["at new highs (reversal)"] = {
            "deposits": len(deposits), "contributed": paid, "terminal": nav,
            "irr": xirr(flows), "max_drawdown": dd}
        print(f"  {'same count, at new highs (reversal)':38s} "
              f"{len(deposits):>9d} ${paid:>9,.0f} ${nav:>11,.0f} "
              f"{xirr(flows):>10.1%} {dd:>8.1%}")

    report["arms"] = results
    dates = results.get("drawdown 20%", {}).get("dates", [])
    if dates:
        print(f"\n  the 20% rule fired on: {', '.join(dates)}")
        print(f"  that is {len(dates)} deposits in {span:.1f} years, "
              f"one roughly every {span / len(dates):.1f} years")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
