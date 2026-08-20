"""How many breakouts fire at once, and how much the six-slot cap decides.

The portfolio holds at most six positions and breaks ties at random, which every
study in this programme has treated as a scoring detail.  It is not one.  If
signals arrive alone, the cap never binds and the random tie-break is harmless.
If twenty fire in the same half hour, then six of twenty are taken essentially by
coin toss, and the result an operator actually gets is one draw from a
distribution rather than the median of forty.

Three measurements.

*Arrival.*  Breakouts per thirty-minute bar and per session, across the book.
Clustering matters more than the average: market-wide moves fire many names at
once, so the six taken on a busy bar are correlated with each other, and the
diversification the six-slot book appears to provide is partly illusory.

*Binding.*  How often a signal is refused for want of a slot.  This is the
capacity the rules discard, and it bounds what a larger position limit could add.

*Dispersion.*  The spread of final outcome across tie-break seeds.  A wide spread
means the median understates what can go wrong, and that the choice of *which*
six to take is itself a decision the backtest has been making at random.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import (TurtleConfig, rolling_extremes,  # noqa: E402
                            run_turtle)
from savi_uz.volume_profile import Bar  # noqa: E402

BASE = dict(entry_window=55, exit_window=20, atr_window=20, skip_after_winner=False,
            use_channel_exit=False, chandelier_atr=3.0, directions=(1,))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--funding", type=Path,
                        default=Path("out/strategy/binance_funding.json"))
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--min-sessions", type=int, default=400)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--cost", type=float, default=0.0005)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/concurrency.json"))
    return parser.parse_args(argv)


def load_book(args, keep):
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    book = {}
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker"):
        if ticker not in keep:
            continue
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='5min' ORDER BY ts", (ticker,)).fetchall()
        five = adjust_bars([Bar(*r) for r in rows], splits.get(ticker, []))
        if len({b.timestamp[:10] for b in five}) >= args.min_sessions:
            book[ticker] = resample_regular_session(five, minutes=args.minutes)
    connection.close()
    return book


def breakout_bars(bars, window):
    """Bar timestamps where a fresh long breakout fires, matching the engine.

    The level is ``rolling_extremes`` over the ``window`` bars strictly before
    the bar, so it is known when the bar opens; the signal is the bar whose high
    exceeds it, having not exceeded the previous level on the bar before.
    """
    highs = [b.high for b in bars]
    levels = rolling_extremes(highs, window, True)
    out = []
    previous = False
    for index in range(len(bars)):
        level = levels[index]
        if math.isnan(level):
            continue
        beyond = bars[index].high > level
        if beyond and not previous:
            out.append(bars[index].timestamp)
        previous = beyond
    return out


def cap(trades, limit, rng):
    shuffled = list(trades)
    rng.shuffle(shuffled)
    live, taken, refused = [], [], 0
    for trade in sorted(shuffled, key=lambda t: t["entry"]):
        live = [x for x in live if x["exit"] > trade["entry"]]
        if len(live) >= limit:
            refused += 1
            continue
        live.append(trade)
        taken.append(trade)
    return taken, refused


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
    for _ in range(36):
        mid = math.sqrt(lo * hi)
        if statistics.median(abs(walk(m, calendar, mid)[1]) for m in maps) < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def main(argv=None):
    args = parse_args(argv)
    keep = set(json.loads(args.funding.read_text(encoding="utf-8")))
    book = load_book(args, keep)
    calendar = sorted({b.timestamp[:10] for bars in book.values() for b in bars})

    per_bar = Counter()
    per_session = Counter()
    for ticker, bars in book.items():
        for stamp in breakout_bars(bars, BASE["entry_window"]):
            per_bar[stamp] += 1
            per_session[stamp[:10]] += 1

    total = sum(per_bar.values())
    bars_seen = len({b.timestamp for bars in book.values() for b in bars})
    print(f"{len(book)} instruments, {len(calendar):,} sessions, "
          f"{bars_seen:,} distinct 30-minute bars")
    print(f"{total:,} long breakouts in total, "
          f"{total / len(calendar):.1f} a session\n")

    counts = sorted(per_bar.values())
    print("breakouts arriving in the same 30-minute bar")
    print(f"  bars with at least one signal   {len(counts):,} "
          f"({len(counts) / bars_seen:.0%} of all bars)")
    print(f"  median                          {counts[len(counts) // 2]}")
    print(f"  90th percentile                 {counts[int(0.9 * len(counts))]}")
    print(f"  99th percentile                 {counts[int(0.99 * len(counts))]}")
    print(f"  maximum                         {counts[-1]}")
    over = sum(1 for c in counts if c > args.max_positions)
    print(f"  bars with more than {args.max_positions} at once     {over:,} "
          f"({over / len(counts):.1%} of signalling bars)")
    share = sum(c for c in counts if c > args.max_positions) / total
    print(f"  share of all signals arriving in such bars  {share:.0%}")

    sessions = sorted(per_session.values())
    print(f"\nbreakouts per session")
    print(f"  median {sessions[len(sessions) // 2]}, "
          f"90th pct {sessions[int(0.9 * len(sessions))]}, "
          f"max {sessions[-1]}")

    # Now the cap, and the dispersion it creates.
    config = TurtleConfig(**{**BASE, "round_trip_cost": args.cost})
    pooled = []
    for ticker, bars in book.items():
        closes = {}
        for bar in bars:
            closes[bar.timestamp[:10]] = bar.close
        for trade in run_turtle(bars, config=config)[0]:
            entry_day, exit_day = (trade.entry_timestamp[:10],
                                   trade.exit_timestamp[:10])
            marks = []
            for day in closes:
                if entry_day <= day < exit_day:
                    live = [u for u in trade.unit_entries
                            if u.timestamp[:10] <= day]
                    if live:
                        marks.append((day, sum(trade.direction
                                               * (closes[day] - u.price) / u.n
                                               for u in live)))
            pooled.append({"entry": trade.entry_timestamp,
                           "exit": trade.exit_timestamp,
                           "r": trade.net_r, "marks": marks})

    results = [cap(pooled, args.max_positions, random.Random(s))
               for s in range(args.trials)]
    refused = statistics.fmean(r[1] for r in results)
    print(f"\nthe six-slot cap")
    print(f"  {len(pooled):,} trades generated, "
          f"{statistics.fmean(len(r[0]) for r in results):,.0f} taken on average")
    print(f"  {refused:,.0f} refused for want of a slot "
          f"({refused / len(pooled):.0%} of all signals)")

    maps = [daily_r(r[0]) for r in results]
    risk = solve(maps, calendar, args.target_dd)
    outcomes = sorted(walk(m, calendar, risk)[2] for m in maps)
    dds = sorted(abs(walk(m, calendar, risk)[1]) for m in maps)
    print(f"\ndispersion across {args.trials} tie-break draws, at matched risk")
    print(f"  CAGR      worst {outcomes[0]:.1%}, "
          f"10th {outcomes[int(0.1 * len(outcomes))]:.1%}, "
          f"median {outcomes[len(outcomes) // 2]:.1%}, "
          f"90th {outcomes[int(0.9 * len(outcomes))]:.1%}, "
          f"best {outcomes[-1]:.1%}")
    print(f"  drawdown  best {dds[0]:.1%}, "
          f"median {dds[len(dds) // 2]:.1%}, worst {dds[-1]:.1%}")
    spread = outcomes[-1] / outcomes[0] if outcomes[0] > 0 else float("nan")
    print(f"  best-to-worst CAGR ratio {spread:.2f}x -- this is the cost of "
          f"choosing at random\n  among simultaneous signals, and an operator "
          f"gets one draw, not the median")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"total_signals": total, "per_bar": dict(Counter(counts)),
         "refused_share": refused / len(pooled),
         "cagr_spread": outcomes, "dd_spread": dds}, indent=1), encoding="utf-8")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
