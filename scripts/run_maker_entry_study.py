"""Can the breakout be entered as a maker, and what does waiting for it cost?

Binance charges nothing to a maker on these contracts, which is worth two and a
half basis points a side against the taker rate -- meaningful for a book that
turns over four hundred times a year.  The obstacle is geometric rather than
commercial.

A Donchian long buys when price exceeds the fifty-five bar high.  At the moment
the order would be placed, that level sits *above* the market, so a limit there
is marketable and post-only rejects it.  The only way to rest a bid at the
breakout level is to place it after price has already traded through, and then
the fill requires a pullback.  That is not a cheaper breakout entry; it is a
retest entry, and it selects against the trades worth having, because a breakout
that never looks back is the one that trends.

So the fee saving and the adverse selection point in opposite directions, and the
question is which is larger.  Three arms, identical exits:

*Taker breakout.*  The current system.  Filled on the breakout bar at the level
or the open, whichever is worse, and charged the full taker round trip.

*Maker retest.*  After a breakout, a bid rests at the level for a window of bars.
It fills only if price returns, and at the level exactly.  Trades that never
retest are simply not taken.

*Hybrid.*  The retest bid rests for the window, and if it has not filled the
trade is taken at market afterwards, paying taker.  This keeps the runaways
instead of discarding them.

Two honest limits on the saving, stated rather than buried.  Pyramid units are
buy stops above the market and cannot be maker either, so with 2.84 units to a
trade the first-unit saving is diluted.  And the exit is a trailing stop, which
is a taker by construction.  A round trip is therefore never zero; the best
realistic case is maker in and taker out.

The arms are separated first at *equal* cost, which isolates the selection effect
from the fee effect, and only then re-scored at the real fees.
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

BASE = dict(entry_window=55, exit_window=20, atr_window=20, skip_after_winner=False,
            use_channel_exit=False, chandelier_atr=3.0, directions=(1,))

WINDOWS = (2, 4, 8, 13, 26)
TAKER_SIDE = 0.00025


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--funding", type=Path,
                        default=Path("out/strategy/binance_funding.json"))
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--min-sessions", type=int, default=400)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/maker_entry.json"))
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


def donchian(bars, window):
    """Channel high as of each bar's close, usable from the next bar on."""
    highs = []
    for index in range(len(bars)):
        if index < window:
            highs.append(float("nan"))
        else:
            highs.append(max(b.high for b in bars[index - window:index]))
    return highs


def signals(bars, window, retest_window):
    """Breakout bars, and where a resting bid at the level would have filled.

    Returns three entry maps keyed by bar index: the taker fill on the breakout
    bar, the maker fill on the retest bar, and the hybrid, which falls back to a
    market entry once the window has passed without a fill.
    """
    highs = donchian(bars, window)
    taker, maker, hybrid = {}, {}, {}
    index = window + 1
    # A breakout is the bar that *crosses* the channel, not every bar that sits
    # above it.  Without this the signal fires on almost every bar of a trend --
    # 106,666 of them across the book, one every four bars -- which is a
    # different and flattering strategy rather than the one being tested.
    beyond_previous = False
    while index < len(bars):
        level = highs[index - 1]
        if math.isnan(level):
            index += 1
            continue
        beyond = bars[index].high > level
        if not beyond or beyond_previous:
            beyond_previous = beyond
            index += 1
            continue
        beyond_previous = True
        # A stop at the level fills there, or worse if the bar gapped past it.
        taker[index] = max(level, bars[index].open)
        filled = None
        for ahead in range(index + 1, min(index + 1 + retest_window, len(bars))):
            if bars[ahead].low <= level:
                filled = ahead
                break
        if filled is not None and filled not in maker:
            maker[filled] = level
            hybrid[filled] = level
        else:
            fallback = index + retest_window
            if fallback < len(bars) and fallback not in hybrid:
                hybrid[fallback] = bars[fallback].open
        # Skip past this breakout so one signal does not spawn many entries.
        index += 1
    return taker, maker, hybrid


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


def run_arm(book, cost, retest_window, which):
    """Pool trades for one execution style at one cost."""
    config = TurtleConfig(**{**BASE, "round_trip_cost": cost})
    pooled = []
    for ticker, bars in book.items():
        taker, maker, hybrid = signals(bars, BASE["entry_window"], retest_window)
        chosen = {"taker": taker, "maker": maker, "hybrid": hybrid}[which]
        if not chosen:
            continue
        entries = {i: 1 for i in chosen}
        closes = {}
        for bar in bars:
            closes[bar.timestamp[:10]] = bar.close
        trades, _ = run_turtle(bars, config=config, entries=entries,
                               entry_prices=chosen)
        for trade in trades:
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
    return pooled


def score(pooled, calendar, args):
    if len(pooled) < 50:
        return None
    maps = [daily_r(cap(pooled, args.max_positions, random.Random(s)))
            for s in range(args.trials)]
    risk = solve(maps, calendar, args.target_dd)
    results = [walk(m, calendar, risk) for m in maps]
    cagrs = sorted(r[2] for r in results)
    series = []
    for m in maps:
        values = [m.get(d, 0.0) for d in calendar]
        sd = statistics.pstdev(values)
        series.append(statistics.fmean(values) / sd * math.sqrt(252) if sd else 0.0)
    return {"trades": len(pooled), "sharpe": statistics.median(series),
            "cagr": cagrs[len(cagrs) // 2]}


def main(argv=None):
    args = parse_args(argv)
    keep = set(json.loads(args.funding.read_text(encoding="utf-8")))
    book = load_book(args, keep)
    calendar = sorted({b.timestamp[:10] for bars in book.values() for b in bars})

    # How often does a breakout come back to its level at all?
    print(f"{len(book)} instruments, {calendar[0]} -> {calendar[-1]}\n")
    print("retest frequency: share of breakouts that return to the level")
    print(f"  {'window':>8s} {'bars':>6s} {'breakouts':>11s} {'retested':>10s} "
          f"{'share':>7s}")
    fills = {}
    for window in WINDOWS:
        total = retested = 0
        for ticker, bars in book.items():
            taker, maker, _ = signals(bars, BASE["entry_window"], window)
            total += len(taker)
            retested += len(maker)
        fills[window] = retested / total if total else 0.0
        hours = window * args.minutes / 60
        print(f"  {hours:>6.1f}h {window:>6d} {total:>11,d} {retested:>10,d} "
              f"{fills[window]:>7.1%}")

    print(f"\nstep 1 -- equal cost (5bp both arms), which isolates selection "
          f"from fees")
    print(f"  {'arm':<22s} {'trades':>8s} {'Sharpe':>8s} {'CAGR':>8s}")
    report = {}
    base_taker = run_arm(book, 0.0005, 8, "taker")
    item = score(base_taker, calendar, args)
    report["taker @5bp"] = item
    print(f"  {'taker breakout':<22s} {item['trades']:>8,d} "
          f"{item['sharpe']:>8.2f} {item['cagr']:>7.1%}")
    for window in WINDOWS:
        pooled = run_arm(book, 0.0005, window, "maker")
        item = score(pooled, calendar, args)
        report[f"maker w{window} @5bp"] = item
        label = f"maker retest {window}b"
        if item:
            print(f"  {label:<22s} {item['trades']:>8,d} "
                  f"{item['sharpe']:>8.2f} {item['cagr']:>7.1%}")
        else:
            print(f"  {label:<22s} {'too few':>8s}")

    print(f"\nstep 2 -- real fees: taker pays 5bp round trip, maker pays 2.5bp "
          f"(maker in, taker out)")
    print(f"  {'arm':<22s} {'cost':>6s} {'trades':>8s} {'Sharpe':>8s} {'CAGR':>8s}")
    item = score(run_arm(book, 0.0005, 8, "taker"), calendar, args)
    report["taker real"] = item
    print(f"  {'taker breakout':<22s} {'5.0bp':>6s} {item['trades']:>8,d} "
          f"{item['sharpe']:>8.2f} {item['cagr']:>7.1%}")
    for window in WINDOWS:
        pooled = run_arm(book, TAKER_SIDE, window, "maker")
        item = score(pooled, calendar, args)
        report[f"maker w{window} real"] = item
        label = f"maker retest {window}b"
        if item:
            print(f"  {label:<22s} {'2.5bp':>6s} {item['trades']:>8,d} "
                  f"{item['sharpe']:>8.2f} {item['cagr']:>7.1%}")
    for window in (4, 8, 13):
        pooled = run_arm(book, 0.0004, window, "hybrid")
        item = score(pooled, calendar, args)
        report[f"hybrid w{window}"] = item
        label = f"hybrid {window}b"
        if item:
            print(f"  {label:<22s} {'4.0bp':>6s} {item['trades']:>8,d} "
                  f"{item['sharpe']:>8.2f} {item['cagr']:>7.1%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"retest_share": fills, "results": report},
                                   indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
