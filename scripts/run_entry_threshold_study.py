"""Does demanding a deeper penetration of the level improve the breakout?

The proposal was to rest the entry a basis point beyond the trigger.  As a fee
device it cannot work -- a buy order provides liquidity only below the market, so
a bid above it is marketable and pays taker anyway -- but as a *confirmation
threshold* it is a real question, and a different one: does insisting that price
travel a little further before committing separate the breakouts that go from the
breakouts that tick through and reverse?

Two forces oppose each other.  A threshold buys a worse entry on every trade that
still happens, which is a certain cost.  In exchange it declines the marginal
breakouts, which is an uncertain benefit.  Whether the trade is worth making
depends entirely on how much of the return lives in the marginal breakouts, and
that is measurable rather than arguable.

Scale matters more than direction here.  A basis point is between one and five
per cent of a single thirty-minute bar's range across this book, so almost every
bar that touches the level also touches a point beyond it, and the threshold
should decline almost nothing while charging its cost on everything.  Thresholds
are therefore also expressed in N, where a quarter and a half of the daily range
are large enough to actually refuse trades.

Offsets in basis points are of the level; offsets in N use the Wilder average
computed on the bar before the signal, so nothing is read before it happens.
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
from savi_uz.turtle import (TurtleConfig, rolling_extremes,  # noqa: E402
                            run_turtle, wilder_atr)
from savi_uz.volume_profile import Bar  # noqa: E402

BASE = dict(entry_window=55, exit_window=20, atr_window=20, skip_after_winner=False,
            use_channel_exit=False, chandelier_atr=3.0, directions=(1,))

#: (label, kind, size).  Three families.
#:
#: "bp"/"n" push the *trigger* further beyond the channel, so the threshold must
#: be penetrated before anything is bought -- a confirmation filter that charges
#: a worse entry on every trade it still takes.
#:
#: "cap" is the stop-limit: the trigger stays at the level, but the limit price
#: refuses any fill worse than the cap.  A bar that gaps past it simply does not
#: trade.  This is the honest version of the current model, which fills every gap
#: at the open and assumes no slippage -- the assumption is most generous exactly
#: when the breakout is most violent, which is where the return is.
OFFSETS = (("stop-market (baseline)", "bp", 0.0),
           ("trigger +1bp", "bp", 0.0001),
           ("trigger +5bp", "bp", 0.0005),
           ("trigger +25bp", "bp", 0.0025),
           ("trigger +0.25N", "n", 0.25),
           ("trigger +0.50N", "n", 0.50),
           ("stop-limit cap +0bp", "cap", 0.0),
           ("stop-limit cap +1bp", "cap", 0.0001),
           ("stop-limit cap +2bp", "cap", 0.0002),
           ("stop-limit cap +5bp", "cap", 0.0005),
           ("stop-limit cap +10bp", "cap", 0.0010),
           ("stop-limit cap +25bp", "cap", 0.0025),
           ("stop-limit cap +0.25N", "capn", 0.25))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--funding", type=Path,
                        default=Path("out/strategy/binance_funding.json"))
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--min-sessions", type=int, default=400)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--cost", type=float, default=0.0005)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/entry_threshold.json"))
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


def signals(bars, window, atr_window, kind, size):
    """Entry index -> fill price, for a threshold set beyond the channel."""
    highs = [b.high for b in bars]
    levels = rolling_extremes(highs, window, True)
    atr = wilder_atr(bars, atr_window)
    out, previous, penalty = {}, False, []
    missed = 0
    for index in range(len(bars)):
        level = levels[index]
        if math.isnan(level):
            continue
        n = atr[index - 1] if index else float("nan")
        if kind in ("n", "capn") and math.isnan(n):
            continue
        if kind == "bp":
            threshold = level * (1.0 + size)
        elif kind == "n":
            threshold = level + size * n
        else:
            # stop-limit: the trigger is the level itself; the offset is the
            # worst price the limit will accept.
            threshold = level
        beyond = bars[index].high > threshold
        if beyond and not previous:
            fill = max(threshold, bars[index].open)
            if kind in ("cap", "capn"):
                limit = (level * (1.0 + size) if kind == "cap"
                         else level + size * n)
                if fill > limit:
                    # Price opened beyond the limit, so the order rests unfilled
                    # rather than chasing.  The trade does not happen.
                    missed += 1
                    previous = beyond
                    continue
            out[index] = fill
            if level > 0:
                penalty.append(fill / level - 1.0)
        previous = beyond
    return out, penalty, missed


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


def main(argv=None):
    args = parse_args(argv)
    keep = set(json.loads(args.funding.read_text(encoding="utf-8")))
    book = load_book(args, keep)
    calendar = sorted({b.timestamp[:10] for bars in book.values() for b in bars})
    config = TurtleConfig(**{**BASE, "round_trip_cost": args.cost})

    print(f"{len(book)} instruments, {calendar[0]} -> {calendar[-1]}, "
          f"{args.cost * 1e4:.1f}bp round trip\n")
    print(f"  {'threshold':<18s} {'signals':>8s} {'kept':>6s} {'trades':>8s} "
          f"{'Sharpe':>8s} {'CAGR':>8s} {'entry cost':>11s}")

    report, reference = {}, None
    for label, kind, size in OFFSETS:
        pooled, signal_count, penalties, skipped = [], 0, [], 0
        for ticker, bars in book.items():
            entries, penalty, missed = signals(bars, BASE["entry_window"],
                                              BASE["atr_window"], kind, size)
            signal_count += len(entries)
            penalties += penalty
            skipped += missed
            if not entries:
                continue
            closes = {}
            for bar in bars:
                closes[bar.timestamp[:10]] = bar.close
            trades, _ = run_turtle(bars, config=config,
                                   entries={i: 1 for i in entries},
                                   entry_prices=entries)
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
        if reference is None:
            reference = signal_count
        maps = [daily_r(cap(pooled, args.max_positions, random.Random(s)))
                for s in range(args.trials)]
        risk = solve(maps, calendar, args.target_dd)
        results = [walk(m, calendar, risk) for m in maps]
        cagrs = sorted(r[2] for r in results)
        series = []
        for m in maps:
            values = [m.get(d, 0.0) for d in calendar]
            sd = statistics.pstdev(values)
            series.append(statistics.fmean(values) / sd * math.sqrt(252)
                          if sd else 0.0)
        item = {"signals": signal_count, "kept": signal_count / reference,
                "unfilled": skipped,
                "trades": len(pooled), "sharpe": statistics.median(series),
                "cagr": cagrs[len(cagrs) // 2],
                "entry_cost": statistics.fmean(penalties) if penalties else 0.0}
        report[label] = item
        print(f"  {label:<18s} {signal_count:>8,d} {item['kept']:>5.0%} "
              f"{len(pooled):>8,d} {item['sharpe']:>8.2f} {item['cagr']:>7.1%} "
              f"{item['entry_cost'] * 1e4:>10.1f}bp", flush=True)

    best = max(report.items(), key=lambda kv: kv[1]["sharpe"])
    base = report["level (baseline)"]
    print(f"\n  baseline Sharpe {base['sharpe']:.2f}; best is {best[0]} at "
          f"{best[1]['sharpe']:.2f}")
    if best[0] == "level (baseline)":
        print(f"  no threshold beats entering at the level itself")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
