"""Adding to winners, and lifting the position cap, on the earnings book.

Two changes to the same book, tested together because they push the same lever.

*Pyramiding.*  The turtle rule, ported: a unit is added each time price advances
half a unit of risk above the last fill, up to ``max_units``, and the stop for
the whole position moves to one unit of risk below the newest fill.  That last
clause is what makes the rule survivable -- a fully pyramided position risks
about the same one unit it started with, because the stop has climbed with it.
It is also what makes a *partly* pyramided position risk more than one unit, and
that is reported rather than assumed away.

*The cap.*  Twelve concurrent positions was inherited from a book that traded a
different signal.  Earnings arrive in four dense clusters a year and more than
half of all qualifying triggers were being refused for space, so the cap is
swept from six to unlimited.

The two interact, which is the reason to test them together rather than one at a
time.  Pyramiding raises the exposure of each position; lifting the cap raises
the number of positions.  Doing both multiplies, and a book that looks better on
return while quietly running at six times equity has not been improved, it has
been levered.  So gross exposure is reported in the same table as the return,
and the account that has to hold it is a real one with a Reg-T limit.

Everything is chosen on the first half.  The second half is untouched until the
configuration is frozen, and the drift null runs on the second half only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_earnings_strategy as base  # noqa: E402
import run_vol_stretch_zones as shared  # noqa: E402

UNITS = (1, 2, 4)
CAPS = (6, 12, 24, 48, 999)
ADD_FRACTION = 0.5
RISK = 0.01
LEVERAGE_CAP = 20.0
REG_T = 2.0


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
    parser.add_argument("--hold", type=int, default=40)
    parser.add_argument("--stop-mult", type=float, default=3.0)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--null-draws", type=int, default=30)
    parser.add_argument("--max-positions", type=int, default=12)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/earnings_pyramid.json"))
    return parser.parse_args(argv)


def run_pyramid(book, chosen, max_units, args):
    """One entry, then adds every half-unit in favour, stop trailing the last fill."""
    trades = []
    for event in chosen:
        bars = book[event["ticker"]]
        start = event["index"] + 1
        if start >= len(bars):
            continue
        fill = bars[start].open
        unit = args.stop_mult * event["sigma"] * fill
        if fill <= 0 or unit <= 0:
            continue
        fills = [fill]
        stop = fill - unit
        next_add = fill + ADD_FRACTION * unit
        last = min(start + args.hold, len(bars) - 1)
        exit_price, reason, when = bars[last].close, "time", bars[last].timestamp
        for i in range(start, last + 1):
            bar = bars[i]
            if bar.low <= stop:
                exit_price, reason, when = stop, "stop", bar.timestamp
                break
            while len(fills) < max_units and bar.high >= next_add:
                fills.append(next_add)
                stop = next_add - unit          # the turtle clause
                next_add = next_add + ADD_FRACTION * unit
        gross = sum(exit_price - p for p in fills) / unit
        legs = 2 * len(fills)                    # every unit pays in and out
        cost = legs * args.taker_bp / 10_000 * fill / unit
        trades.append({"ticker": event["ticker"], "entry": event["day"],
                       "exit": when, "r": gross - cost, "reason": reason,
                       "stop_pct": unit / fill, "units": len(fills)})
    trades.sort(key=lambda t: t["entry"])
    return trades


def exposure(taken):
    """Gross notional as a multiple of equity, across the whole timeline."""
    if not taken:
        return {}
    days = sorted({t["entry"] for t in taken} | {t["exit"] for t in taken})
    d0, d1 = date.fromisoformat(days[0]), date.fromisoformat(days[-1])
    calendar, cur = [], d0
    while cur <= d1:
        if cur.weekday() < 5:
            calendar.append(cur.isoformat())
        cur += timedelta(days=1)
    gross = defaultdict(float)
    for t in taken:
        lever = min(RISK / t["stop_pct"], LEVERAGE_CAP) * t["units"]
        for day in calendar:
            if t["entry"] <= day < t["exit"]:
                gross[day] += lever
    series = [gross.get(d, 0.0) for d in calendar]
    series_sorted = sorted(series)
    return {"median": series_sorted[len(series) // 2],
            "p95": series_sorted[int(0.95 * (len(series) - 1))],
            "max": max(series),
            "share_over_regt": sum(1 for g in series if g > REG_T) / len(series)}


def compound(taken, fraction=RISK):
    per_day = defaultdict(float)
    for t in taken:
        lever = min(fraction / t["stop_pct"], LEVERAGE_CAP)
        per_day[t["exit"]] += t["r"] * lever * t["stop_pct"]
    days = sorted(per_day)
    nav, peak, worst = 1000.0, 1000.0, 0.0
    for d in days:
        nav = max(0.0, nav + per_day[d] * nav)
        peak = max(peak, nav)
        worst = min(worst, nav / peak - 1.0)
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25
    return {"cagr": (nav / 1000.0) ** (1 / years) - 1 if nav > 0 else -1.0,
            "max_drawdown": worst, "terminal": nav / 1000.0}


def main(argv=None) -> int:
    args = parse_args(argv)
    book, _ = base.load_book(args)
    earnings = base.load_earnings(args.earnings, sorted(book))
    events, quiet = base.build_events(book, earnings, args)
    chosen = [e for e in events if e["surprise_pct"] > 0 and e["reaction"] > 0]
    print(f"{len(book)} names, {len(chosen):,d} beat-and-rose triggers")
    print(f"adds every {ADD_FRACTION:g} unit in favour, stop trails to one unit "
          f"below the newest fill")
    print(f"in sample to {args.split}, out of sample after\n")
    report = {"grid": {}}

    print("########## in sample: units against the position cap ##########")
    print(f"  {'units':>6s} {'cap':>6s} {'taken':>8s} {'avg units':>10s} "
          f"{'Sharpe':>7s} {'gross med':>10s} {'gross max':>10s}")
    best, best_score = None, -99.0
    cache = {}
    for units in UNITS:
        trades = run_pyramid(book, chosen, units, args)
        for cap in CAPS:
            early = [t for t in trades if t["entry"] < args.split]
            args.max_positions = cap
            taken = shared.cap(early, cap, random.Random(0))
            if len(taken) < 200:
                continue
            result = shared.assess(early, args)
            if not result:
                continue
            grip = exposure(taken)
            cache[(units, cap)] = trades
            report["grid"][f"{units}u|{cap}"] = {
                "sharpe": result["sharpe"], "taken": result["taken"],
                "avg_units": statistics.fmean(t["units"] for t in taken),
                **{f"gross_{k}": v for k, v in grip.items()}}
            if result["sharpe"] > best_score:
                best_score, best = result["sharpe"], (units, cap)
            print(f"  {units:>6d} {cap if cap < 999 else 'none':>6} "
                  f"{result['taken']:>8,d} "
                  f"{statistics.fmean(t['units'] for t in taken):>10.2f} "
                  f"{result['sharpe']:>7.2f} {grip['median']:>9.2f}x "
                  f"{grip['max']:>9.2f}x", flush=True)

    units, cap = best
    print(f"\n  chosen in sample: {units} unit(s), "
          f"cap {cap if cap < 999 else 'unlimited'}  (IS Sharpe {best_score:.2f})")

    print(f"\n########## out of sample ##########")
    print(f"  {'arm':30s} {'taken':>8s} {'Sharpe':>7s} {'[5-95%]':>14s} "
          f"{'CAGR':>8s} {'max DD':>8s} {'gross max':>10s}")
    rows = {}
    for label, u, c in (("no pyramid, cap 12", 1, 12),
                        ("no pyramid, chosen cap", 1, cap),
                        (f"{units} units, cap 12", units, 12),
                        (f"chosen: {units} units, cap {cap if cap<999 else 'none'}",
                         units, cap)):
        trades = run_pyramid(book, chosen, u, args)
        outside = [t for t in trades if t["entry"] >= args.split]
        args.max_positions = c
        result = shared.assess(outside, args)
        if not result:
            continue
        taken = shared.cap(outside, c, random.Random(0))
        grip, money = exposure(taken), compound(taken)
        rows[label] = {**result, **money,
                       **{f"gross_{k}": v for k, v in grip.items()}}
        print(f"  {label:30s} {result['taken']:>8,d} {result['sharpe']:>7.2f} "
              f"{('[%.2f-%.2f]' % (result['sharpe_p05'], result['sharpe_p95'])):>14s} "
              f"{money['cagr']:>+8.1%} {money['max_drawdown']:>8.1%} "
              f"{grip['max']:>9.2f}x", flush=True)
    report["out_of_sample"] = rows

    nulls = []
    for draw in range(args.null_draws):
        want = defaultdict(int)
        for e in chosen:
            want[e["ticker"]] += 1
        pool = defaultdict(list)
        for q in quiet:
            pool[q["ticker"]].append(q)
        picked = []
        for ticker, count in want.items():
            here = pool.get(ticker, [])
            if here:
                rng = random.Random(88_000 + 137 * draw
                                    + base.ticker_seed(ticker))
                picked.extend(rng.sample(here, min(count, len(here))))
        drawn = run_pyramid(book, sorted(picked, key=lambda e: e["day"]),
                            units, args)
        outside = [t for t in drawn if t["entry"] >= args.split]
        if len(outside) < 200:
            continue
        args.max_positions = cap
        outcome = shared.assess(outside, args)
        if outcome:
            nulls.append(outcome["sharpe"])
    if nulls:
        nulls.sort()
        key = f"chosen: {units} units, cap {cap if cap<999 else 'none'}"
        edge = rows.get(key, {}).get("sharpe", math.nan)
        above = sum(1 for x in nulls if x >= edge) / len(nulls)
        report["drift_null"] = {"median": statistics.median(nulls),
                                "low": nulls[0], "high": nulls[-1], "p": above}
        print(f"  {'ordinary sessions (drift null)':30s} {'':>8s} "
              f"{statistics.median(nulls):>7.2f} "
              f"{('[%.2f-%.2f]' % (nulls[0], nulls[-1])):>14s}")
        print(f"  -> p = {above:.2f}, "
              f"{'clears' if above <= 0.05 else 'inside'} its null")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
