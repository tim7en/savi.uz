"""Does the strategy survive Binance's fee schedule, and does its short side work?

Binance now lists perpetual futures on individual US equities, and 39 of the 42
instruments in the book are among them.  That matters for one reason only: those
contracts short as easily as they go long -- no borrow, no locate, no recall, no
uptick rule and no pattern-day-trader minimum -- which is the first venue in this
programme where the short side is a genuine option rather than a paper exercise.

It arrives with a bill.  Every comparison so far has been scored at two basis
points round trip, which is roughly what a retail equity broker charges on a
liquid name.  Binance's USD-M taker fee is five basis points *a side*, so the
round trip is ten, and the measured cost curve is steep exactly there: Sharpe
2.64 at two points, 1.51 at six.  The question this script answers is not whether
the venue is convenient but whether anything is left after paying for it.

Three things are varied and nothing else:

*Direction.*  Long only, short only, and both.  The short side has never been
tested at a cost it could actually be traded at.

*Commission.*  Two basis points through twenty, so the answer is a curve rather
than a verdict at one point.  Ten is Binance taker; fourteen allows a tick of
slippage on a stop fill, which is what a breakout entry actually is.

*Funding.*  Perpetuals charge a carry every eight hours, and it is measured here
rather than assumed -- per symbol, from the venue's own history.  Two cautions
travel with that number.  It covers at most six months of 2026, so applying it to
a nine-year backtest assumes a stationarity nothing here can check; and the
pooled mean is negative, meaning longs were *paid*, which is an artefact of two
thin contracts (TBT at -79% a year on 72 observations, CAT at -30%) rather than a
property of the product.  It is therefore run three ways -- off, as measured, and
at a flat one basis point a day against the position -- and the honest reading is
the one where all three agree.

Scoring is matched-drawdown throughout, because a cost that shortens holding
periods also changes risk, and comparing raw returns across cost levels would
credit the cheap variant twice.
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

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import TurtleConfig, run_turtle  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

#: The house configuration, unchanged from every other script in the programme
#: so the numbers here sit beside the existing ones.  Note that the walk-forward
#: validation preferred a 2N trail in six folds of six; 3N is kept for
#: comparability, and the cost conclusion does not turn on trail width.
BASE = dict(entry_window=55, exit_window=20, atr_window=20,
            skip_after_winner=False, use_channel_exit=False, chandelier_atr=3.0)

COSTS = (0.0002, 0.0005, 0.0010, 0.0014, 0.0020)
MODES = (("long only", (1,)), ("short only", (-1,)), ("both sides", (1, -1)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--funding", type=Path,
                        default=Path("out/strategy/binance_funding.json"))
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--min-sessions", type=int, default=400)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--all-names", action="store_true",
                        help="use all 42 rather than the 39 Binance lists")
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/binance_venue.json"))
    return parser.parse_args(argv)


def load_book(args, keep):
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    book = {}
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker"):
        if keep is not None and ticker not in keep:
            continue
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='5min' ORDER BY ts", (ticker,)).fetchall()
        five = adjust_bars([Bar(*r) for r in rows], splits.get(ticker, []))
        if len({b.timestamp[:10] for b in five}) >= args.min_sessions:
            book[ticker] = resample_regular_session(five, minutes=args.minutes)
    connection.close()
    return book


def cap(trades, limit, rng):
    """Six slots, ties broken at random; the same rule as every other study."""
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


def build(book, config, funding_rate):
    """Pool every trade, marked daily, with funding already deducted.

    Funding is charged on notional, and the engine's unit of account is R, so the
    conversion is the same one the commission uses: a unit occupies ``price / N``
    of notional per unit of risk, and a carry of ``f`` per day on that notional
    costs ``f * price / N`` in R.  Longs pay a positive rate and shorts receive
    it, which is the sign convention the venue publishes.
    """
    pooled = []
    for ticker, bars in book.items():
        closes = {}
        for bar in bars:
            closes[bar.timestamp[:10]] = bar.close
        rate = funding_rate(ticker)
        for trade in run_turtle(bars, config=config)[0]:
            entry_day = trade.entry_timestamp[:10]
            exit_day = trade.exit_timestamp[:10]
            marks, carry, held = [], 0.0, 0
            for day in closes:
                if not (entry_day <= day <= exit_day):
                    continue
                live = [u for u in trade.unit_entries if u.timestamp[:10] <= day]
                if not live:
                    continue
                held += 1
                notional_r = sum(closes[day] / u.n for u in live)
                carry += trade.direction * rate * notional_r
                if day < exit_day:
                    marks.append((day, sum(trade.direction * (closes[day] - u.price)
                                           / u.n for u in live)))
            pooled.append({"entry": trade.entry_timestamp,
                           "exit": trade.exit_timestamp,
                           "r": trade.net_r - carry, "marks": marks,
                           "days": held, "carry": carry,
                           "direction": trade.direction})
    return pooled


def daily_r(taken):
    by_day = defaultdict(float)
    for trade in taken:
        previous = 0.0
        for day, open_r in trade["marks"]:
            by_day[day] += open_r - previous
            previous = open_r
        by_day[trade["exit"][:10]] += trade["r"] - previous
    return by_day


def walk(by_day, calendar, risk):
    nav, peak, worst = 1.0, 1.0, 0.0
    for day in calendar:
        nav = max(1e-12, nav * (1.0 + by_day.get(day, 0.0) * risk))
        peak = max(peak, nav)
        worst = min(worst, nav / peak - 1.0)
    years = (date.fromisoformat(calendar[-1])
             - date.fromisoformat(calendar[0])).days / 365.25
    return nav, worst, nav ** (1 / years) - 1


def solve(maps, calendar, target):
    lo, hi = 1e-7, 0.5
    for _ in range(38):
        mid = math.sqrt(lo * hi)
        dd = statistics.median(abs(walk(m, calendar, mid)[1]) for m in maps)
        if dd < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def sharpe(by_day, calendar):
    series = [by_day.get(d, 0.0) for d in calendar]
    sd = statistics.pstdev(series)
    return statistics.fmean(series) / sd * math.sqrt(252) if sd else 0.0


def score(pooled, calendar, args):
    if len(pooled) < 30:
        return None
    maps = [daily_r(cap(pooled, args.max_positions, random.Random(s)))
            for s in range(args.trials)]
    risk = solve(maps, calendar, args.target_dd)
    results = [walk(m, calendar, risk) for m in maps]
    cagrs = sorted(r[2] for r in results)
    return {"trades": len(pooled),
            "sharpe": statistics.median(sharpe(m, calendar) for m in maps),
            "cagr": cagrs[len(cagrs) // 2],
            "risk": risk,
            "median_days": statistics.median(t["days"] for t in pooled),
            "carry_r": statistics.fmean(t["carry"] for t in pooled)}


def main(argv=None):
    args = parse_args(argv)
    measured = json.loads(args.funding.read_text(encoding="utf-8"))
    keep = None if args.all_names else set(measured)
    book = load_book(args, keep)
    calendar = sorted({b.timestamp[:10] for bars in book.values() for b in bars})
    pooled_rate = statistics.fmean(v["per_day"] for v in measured.values())

    schemes = (("no funding", lambda t: 0.0),
               ("measured", lambda t: measured.get(t, {}).get("per_day", pooled_rate)),
               ("flat 1bp/day", lambda t: 0.0001))

    print(f"{len(book)} instruments, {calendar[0]} -> {calendar[-1]}, "
          f"{len(calendar):,} sessions")
    print(f"scored at a matched {args.target_dd:.0%} drawdown; Sharpe is scale-free\n")

    report = {}
    for funding_label, rate_of in schemes:
        print(f"=== funding: {funding_label} ===")
        header = "  ".join(f"{c * 1e4:>4.0f}bp" for c in COSTS)
        print(f"  {'direction':<12s} {'metric':<7s}  {header}")
        for mode_label, directions in MODES:
            row_sharpe, row_cagr, row_meta = [], [], []
            for cost in COSTS:
                config = TurtleConfig(**{**BASE, "directions": directions,
                                         "round_trip_cost": cost})
                pooled = build(book, config, rate_of)
                item = score(pooled, calendar, args)
                report[f"{funding_label}|{mode_label}|{cost}"] = item
                row_sharpe.append("   n/a" if not item else f"{item['sharpe']:>6.2f}")
                row_cagr.append("   n/a" if not item
                                else f"{item['cagr'] * 100:>5.1f}%")
                row_meta.append(item)
            print(f"  {mode_label:<12s} {'Sharpe':<7s}  " + "  ".join(row_sharpe))
            print(f"  {'':<12s} {'CAGR':<7s}  " + "  ".join(row_cagr))
            live = [m for m in row_meta if m]
            if live:
                print(f"  {'':<12s} {'trades':<7s}  "
                      + "  ".join(f"{m['trades']:>6,d}" for m in live))
        print(flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"universe": sorted(book), "pooled_funding_per_day": pooled_rate,
         "results": report}, indent=1), encoding="utf-8")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
