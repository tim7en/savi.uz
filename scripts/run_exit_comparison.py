"""Which exit rule is doing the work?

The stability study found that random entries through these exit rules capture
63-75% of the strategy's result, which makes the exit the largest untested
surface in the system.  Across eight overlay experiments nobody varied it once.

The comparison has to be clean, and the obvious version is not: letting each
variant generate its own breakouts means the variants see different signals,
because a position that exits sooner frees the instrument to break out again.
Instead the baseline's entry bars are captured once and *offered to every
variant* through the engine's explicit-entry path.  Each rule then sees an
identical opportunity set and differs only in how it manages what it takes --
including how many of those offers it is free to accept, which is a genuine
consequence of the exit rather than a confound.

Every variant is scored over the same capacity cap and the same randomised
tie-breaks, so nothing here rests on which signal happened to win a slot.
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

BASE = dict(entry_window=55, exit_window=20, atr_window=20,
            skip_after_winner=False, directions=(1,))

VARIANTS = [
    ("channel 20 (baseline)", {}),
    ("channel 10 (faster)", dict(exit_window=10)),
    ("channel 50 (slower)", dict(exit_window=50)),
    ("chandelier 5N only", dict(use_channel_exit=False, chandelier_atr=5.0)),
    ("chandelier 3N only", dict(use_channel_exit=False, chandelier_atr=3.0)),
    ("chandelier 8N only", dict(use_channel_exit=False, chandelier_atr=8.0)),
    ("channel 20 + chandelier 5N", dict(chandelier_atr=5.0)),
    ("channel 20 + breakeven 1N", dict(breakeven_trigger_n=1.0)),
    ("channel 20 + breakeven 2N", dict(breakeven_trigger_n=2.0)),
    ("wider hard stop 3N", dict(stop_atr=3.0)),
    ("tighter hard stop 1.5N", dict(stop_atr=1.5)),
    ("no pyramid (1 unit)", dict(max_units=1)),
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--min-sessions", type=int, default=400)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--risk", type=float, default=0.0005)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/exit_comparison.json"))
    return parser.parse_args(argv)


def load_book(args):
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    names = [r[0] for r in connection.execute(
        "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker")]
    book = {}
    for ticker in names:
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='5min' ORDER BY ts", (ticker,)).fetchall()
        if not rows:
            continue
        five = adjust_bars([Bar(*r) for r in rows], splits.get(ticker, []))
        if len({b.timestamp[:10] for b in five}) < args.min_sessions:
            continue
        book[ticker] = resample_regular_session(five, minutes=args.minutes)
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


def wealth(taken, risk):
    by_day = defaultdict(float)
    for trade in taken:
        by_day[trade["exit"][:10]] += trade["r"]
    days = sorted(by_day)
    nav, peak, worst = 1000.0, 1000.0, 0.0
    for day in days:
        nav = max(0.0, nav + by_day[day] * risk * nav)
        peak = max(peak, nav)
        worst = min(worst, nav / peak - 1.0)
    if len(days) < 2:
        return nav, worst, 0.0
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25
    cagr = (nav / 1000.0) ** (1 / years) - 1 if years > 0 and nav > 0 else -1.0
    return nav, worst, cagr


def main(argv=None):
    args = parse_args(argv)
    book = load_book(args)
    print(f"{len(book)} instruments at {args.minutes}-minute bars", flush=True)

    base_config = TurtleConfig(**BASE)
    offers = {}
    total = 0
    for ticker, bars in book.items():
        index_of = {bar.timestamp: i for i, bar in enumerate(bars)}
        trades, _ = run_turtle(bars, config=base_config)
        offers[ticker] = {index_of[t.entry_timestamp]: t.direction for t in trades}
        total += len(offers[ticker])
    print(f"{total:,} baseline entry bars offered to every variant\n", flush=True)

    pick = lambda xs, f: sorted(xs)[int(f * len(xs))]
    report, rows = {}, []
    print(f"  {'exit rule':30s} {'taken':>7s} {'meanR':>7s} {'PF':>5s} {'win':>6s} "
          f"{'totalR':>9s} {'median $':>10s} {'maxDD':>8s} {'Calmar':>7s}")
    for label, overrides in VARIANTS:
        config = TurtleConfig(**{**BASE, **overrides})
        pooled = []
        for ticker, bars in book.items():
            trades, _ = run_turtle(bars, config=config, entries=offers[ticker])
            pooled.extend({"entry": t.entry_timestamp, "exit": t.exit_timestamp,
                           "r": t.net_r} for t in trades)
        rs = [t["r"] for t in pooled]
        gains = sum(r for r in rs if r > 0)
        losses = -sum(r for r in rs if r < 0)
        finals, dds, cagrs = [], [], []
        for seed in range(args.trials):
            taken = cap(pooled, args.max_positions, random.Random(seed))
            nav, dd, cagr = wealth(taken, args.risk)
            finals.append(nav)
            dds.append(dd)
            cagrs.append(cagr)
        median_dd = pick(dds, .5)
        calmar = pick(cagrs, .5) / abs(median_dd) if median_dd else math.nan
        print(f"  {label:30s} {len(pooled):>7,d} {statistics.mean(rs):>+7.3f} "
              f"{gains / losses if losses else math.inf:>5.2f} "
              f"{sum(1 for r in rs if r > 0) / len(rs):>6.1%} {sum(rs):>+9,.0f} "
              f"${pick(finals, .5):>9,.0f} {median_dd:>8.1%} {calmar:>7.2f}")
        entry = {"label": label, "trades": len(pooled),
                 "mean_r": statistics.mean(rs),
                 "pf": gains / losses if losses else None, "total_r": sum(rs),
                 "win": sum(1 for r in rs if r > 0) / len(rs),
                 "final_median": pick(finals, .5), "final_p05": pick(finals, .05),
                 "final_p95": pick(finals, .95), "maxdd": median_dd,
                 "calmar": calmar}
        rows.append(entry)
        report[label] = entry

    good = [r for r in rows if r["calmar"] == r["calmar"]]
    best = max(good, key=lambda r: r["calmar"])
    base = report["channel 20 (baseline)"]
    print(f"\n  best Calmar : {best['label']} ({best['calmar']:.2f} "
          f"vs baseline {base['calmar']:.2f})")
    print(f"  best total R: {max(rows, key=lambda r: r['total_r'])['label']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
