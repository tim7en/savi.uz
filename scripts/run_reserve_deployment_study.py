"""Holding 30% back and deploying it into a 10% drawdown.

The proposal: run the book at 70% of full size, and mobilise the remaining 30%
once the equity curve is 10% below its peak.

It is a martingale -- it adds exposure after losses -- and the mechanism argues
against it here.  A breakout book loses money in chop, not in trends, so the
reserve deploys into precisely the regime where the edge is weakest.  That is
the same reasoning that made negative dealer gamma look attractive and then
failed: raising or lowering risk always flatters a comparison that is not
exposure-matched.

So the test is built around the two controls that decide it.

*Matched drawdown.*  Every arm is scaled to the same median peak-to-trough loss
before its return is read.  Without that, an arm that carries more risk shows a
higher CAGR and the comparison measures leverage rather than the rule.

*Constant exposure.*  The rule has to beat running flat at its own average
exposure.  If a constant book matches it, the conditionality is machinery that
earns nothing, and the honest description is "run at 85%" rather than "deploy a
reserve".  Its reversal -- cutting to 70% in a drawdown instead of raising to
100% -- is run alongside, because a rule that performs no better than its
opposite carries no information.

The binding limit is reported first: the number of times the equity curve
actually crosses the trigger.  Over eleven years that is a single-digit count,
and no Sharpe comparison built on single-digit episodes settles anything.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import TurtleConfig, run_turtle  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

FIXED = dict(entry_window=55, exit_window=20, atr_window=20,
             skip_after_winner=False, use_channel_exit=False, chandelier_atr=3.0)

LEVERED_MARKERS = ("2X", "3X", "1.5X", "BULL", "BEAR", "LEVERAGED", "INVERSE",
                   "ULTRA", "DAILY ", "SHORT ")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/intraday/bars_av.db"))
    parser.add_argument("--source-frequency", default="5min")
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--cost-bp", type=float, default=5.0)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trigger", type=float, default=0.10,
                        help="drawdown at which the reserve is mobilised")
    parser.add_argument("--base", type=float, default=0.70)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/reserve_deployment.json"))
    return parser.parse_args(argv)


def load(args):
    connection = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT ticker, name FROM symbols WHERE name IS NOT NULL").fetchall()
        drop = {t for t, n in rows if any(m in n.upper() for m in LEVERED_MARKERS)}
    except sqlite3.OperationalError:
        drop = set()
    tickers = [r[0] for r in connection.execute(
        "SELECT DISTINCT ticker FROM bars WHERE frequency=? ORDER BY ticker",
        (args.source_frequency,)) if r[0] not in drop]
    if args.limit:
        tickers = tickers[:args.limit]
    book = {}
    for ticker in tickers:
        raw = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency=? AND ts>=? ORDER BY ts",
            (ticker, args.source_frequency, args.start)).fetchall()
        if len(raw) < 4000:
            continue
        bars = [Bar(*r) for r in raw]
        if args.minutes != 5:
            bars = resample_regular_session(bars, minutes=args.minutes)
        if len(bars) >= 800:
            book[ticker] = bars
    connection.close()
    return book


def cap(trades, limit, rng):
    shuffled = list(trades)
    rng.shuffle(shuffled)
    live, taken = [], []
    for trade in sorted(shuffled, key=lambda t: t["entry"]):
        live = [x for x in live if x["exit"] > trade["entry"]]
        if len(live) >= limit:
            continue
        live.append(trade)
        taken.append(trade)
    return taken


def marked_map(taken, closes_by_ticker):
    by_day = defaultdict(float)
    for trade in taken:
        closes = closes_by_ticker[trade["ticker"]]
        entry_day, exit_day = trade["entry"][:10], trade["exit"][:10]
        previous = 0.0
        for day in (d for d in closes if entry_day <= d < exit_day):
            live = [u for u in trade["units"] if u.timestamp[:10] <= day]
            if not live:
                continue
            open_r = sum(trade["dir"] * (closes[day] - u.price) / u.n for u in live)
            by_day[day] += open_r - previous
            previous = open_r
        by_day[exit_day] += trade["r"] - previous
    return by_day


def walk(values, risk, schedule, trigger):
    """NAV path under an exposure schedule; also counts trigger crossings."""
    nav = peak = 1000.0
    worst = 0.0
    deployed = False
    crossings = 0
    daily = []
    for value in values:
        drawdown = nav / peak - 1.0
        now = drawdown <= -trigger
        if now and not deployed:
            crossings += 1
        deployed = now
        exposure = schedule(deployed)
        step = value * risk * exposure
        nav = max(0.0, nav + step * nav)
        daily.append(step)
        peak = max(peak, nav)
        worst = min(worst, nav / peak - 1.0)
    return nav, worst, crossings, daily


def solve_risk(series, schedule, trigger, target, lo=1e-6, hi=0.30):
    def dd(risk):
        return statistics.median(
            abs(walk(v, risk, schedule, trigger)[1]) for _, v in series)
    if dd(hi) < target:
        return hi
    for _ in range(32):
        mid = math.sqrt(lo * hi)
        if dd(mid) < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def cagr_of(nav, days):
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25
    return (nav / 1000.0) ** (1 / years) - 1 if years > 0 and nav > 0 else -1.0


def sharpe(stream):
    sd = statistics.pstdev(stream)
    return statistics.fmean(stream) / sd * math.sqrt(252) if sd > 0 else float("nan")


def main(argv=None) -> int:
    args = parse_args(argv)
    book = load(args)
    closes_by_ticker = {t: {b.timestamp[:10]: b.close for b in bars}
                        for t, bars in book.items()}
    config = TurtleConfig(**FIXED, directions=(1,),
                          round_trip_cost=args.cost_bp / 10_000)
    pooled = []
    for ticker, bars in book.items():
        trades, _ = run_turtle(bars, config=config)
        pooled.extend({"ticker": ticker, "entry": t.entry_timestamp,
                       "exit": t.exit_timestamp, "r": t.net_r,
                       "dir": t.direction, "units": t.unit_entries}
                      for t in trades)
    marks = [marked_map(cap(pooled, args.max_positions, random.Random(s)),
                        closes_by_ticker) for s in range(args.trials)]
    series = [(sorted(m), [m[d] for d in sorted(m)]) for m in marks]
    print(f"{len(book)} instruments, {len(pooled):,} breakouts, "
          f"{args.trials} capacity orderings, {args.cost_bp:g}bp\n")

    base, trigger = args.base, args.trigger
    average = (base + 1.0) / 2
    schedules = {
        f"constant {base:.0%}": lambda d: base,
        f"constant {average:.0%} (average exposure)": lambda d: average,
        "constant 100%": lambda d: 1.0,
        f"{base:.0%} -> 100% on -{trigger:.0%}": lambda d: 1.0 if d else base,
        f"100% -> {base:.0%} on -{trigger:.0%} (reversal)":
            lambda d: base if d else 1.0,
    }

    print(f"  {'exposure schedule':40s} {'Sharpe':>7s} {'CAGR':>8s} "
          f"{'risk/R':>8s} {'crossings':>10s}")
    report = {}
    for label, schedule in schedules.items():
        risk = solve_risk(series, schedule, trigger, args.target_dd)
        results = [walk(v, risk, schedule, trigger) for _, v in series]
        navs = [r[0] for r in results]
        crossings = [r[2] for r in results]
        sharpes = [sharpe(r[3]) for r in results]
        cagrs = [cagr_of(n, d) for n, (d, _) in zip(navs, series)]
        entry = {
            "sharpe": statistics.median(sharpes),
            "cagr": statistics.median(cagrs),
            "risk_per_r": risk,
            "trigger_crossings_median": statistics.median(crossings),
            "trigger_crossings_min": min(crossings),
            "trigger_crossings_max": max(crossings),
        }
        report[label] = entry
        print(f"  {label:40s} {entry['sharpe']:>7.2f} {entry['cagr']:>8.1%} "
              f"{risk*10_000:>6.1f}bp {entry['trigger_crossings_median']:>10.0f}",
              flush=True)

    conditional = report[f"{base:.0%} -> 100% on -{trigger:.0%}"]
    constant = report[f"constant {average:.0%} (average exposure)"]
    reversal = report[f"100% -> {base:.0%} on -{trigger:.0%} (reversal)"]
    print(f"\n  episodes behind the rule: median "
          f"{conditional['trigger_crossings_median']:.0f} crossings "
          f"(range {conditional['trigger_crossings_min']}-"
          f"{conditional['trigger_crossings_max']})")
    print(f"  vs constant {average:.0%}: "
          f"{conditional['sharpe'] - constant['sharpe']:+.3f} Sharpe")
    print(f"  vs its own reversal: "
          f"{conditional['sharpe'] - reversal['sharpe']:+.3f} Sharpe")
    print("  all arms matched to the same median drawdown before comparison")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
