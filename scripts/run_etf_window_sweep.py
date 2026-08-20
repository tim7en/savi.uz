"""Do ETF breakouts want a shorter window, and which side should they trade?

The 55-bar entry channel was chosen on single US equities.  ETFs are baskets:
they move less, they mean-revert more at the index level, and a channel tuned to
single-name dispersion may simply be too slow for them.  That is a testable
claim rather than a stylistic preference.

Direction is swept alongside, because the two interact.  A long-only breakout on
a basket in a downtrend is a different proposition from the same rule on a single
name, and duration and volatility ETFs spend long stretches trending down in a
way the equity book never did over 2017-2026.

The trap here is that a sweep is a search, and a search over enough cells always
returns something.  Two defences.  Every cell is scored on a training half and
then on a held-out half it had no part in choosing, and the reported figure is
the held-out one.  And the selection lesson from validating the exit applies
again: where several settings perform comparably, the median of the leaders is
reported beside the argmax, because the argmax was not reproducible last time.

Costs are shown at 2bp and 8bp.  The equity book's Sharpe falls from 2.64 to 1.01
across that range, so a window that only works at 2bp has not been shown to work.
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

WINDOWS = (15, 20, 30, 40, 55, 80)
DIRECTIONS = (("long", (1,)), ("short", (-1,)), ("both", (1, -1)))
SPLIT = "2015-01-01"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etf", type=Path,
                        default=Path("data/cross_assets/etf_30min.db"))
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--trail", type=float, default=3.0)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/etf_window_sweep.json"))
    return parser.parse_args(argv)


def load_etf(args):
    connection = sqlite3.connect(f"file:{args.etf}?mode=ro", uri=True)
    book = {}
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars ORDER BY ticker"):
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? "
            "ORDER BY ts", (ticker,)).fetchall()
        if len(rows) >= 2000:
            book[ticker] = resample_regular_session([Bar(*r) for r in rows],
                                                    minutes=args.minutes)
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


def sharpe_of(trades, args, lo=None, hi=None):
    window = [t for t in trades
              if (lo is None or t["entry"] >= lo) and (hi is None or t["entry"] < hi)]
    if len(window) < 60:
        return float("nan")
    scores = []
    for seed in range(args.trials):
        by_day = defaultdict(float)
        for trade in cap(window, args.max_positions, random.Random(seed)):
            previous = 0.0
            for day, open_r in trade["marks"]:
                if lo and day < lo[:10]:
                    continue
                by_day[day] += open_r - previous
                previous = open_r
            by_day[trade["exit"][:10]] += trade["r"] - previous
        days = sorted(by_day)
        if len(days) < 30:
            continue
        span = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days
        stream = ([by_day[d] for d in days]
                  + [0.0] * max(0, int(span * 252 / 365.25) - len(days)))
        sd = statistics.pstdev(stream)
        if sd > 0:
            scores.append(statistics.fmean(stream) / sd * math.sqrt(252))
    return statistics.median(scores) if scores else float("nan")


def build(book, window, directions, cost, trail):
    config = TurtleConfig(entry_window=window, exit_window=max(5, window // 3),
                          atr_window=20, skip_after_winner=False,
                          directions=directions, use_channel_exit=False,
                          chandelier_atr=trail, round_trip_cost=cost)
    pooled = []
    for ticker, bars in book.items():
        closes = {b.timestamp[:10]: b.close for b in bars}
        for trade in run_turtle(bars, config=config)[0]:
            marks = []
            for day in (d for d in closes
                        if trade.entry_timestamp[:10] <= d < trade.exit_timestamp[:10]):
                live = [u for u in trade.unit_entries if u.timestamp[:10] <= day]
                if live:
                    marks.append((day, sum(trade.direction * (closes[day] - u.price)
                                           / u.n for u in live)))
            pooled.append({"ticker": ticker, "entry": trade.entry_timestamp,
                           "exit": trade.exit_timestamp, "r": trade.net_r,
                           "dir": trade.direction, "marks": marks})
    return pooled


def main(argv=None):
    args = parse_args(argv)
    book = load_etf(args)
    print(f"{len(book)} ETFs, trail {args.trail}N, "
          f"train before {SPLIT}, held out after\n", flush=True)

    report = {}
    for cost in (0.0002, 0.0008):
        print(f"{'=' * 76}\nround-trip {cost * 1e4:.0f}bp   "
              f"(held-out Sharpe; train in brackets)")
        header = "  " + f"{'window':>7s}" + "".join(
            f"{name:>18s}" for name, _ in DIRECTIONS)
        print(header)
        table = {}
        for window in WINDOWS:
            cells = []
            for name, directions in DIRECTIONS:
                pooled = build(book, window, directions, cost, args.trail)
                train = sharpe_of(pooled, args, hi=SPLIT)
                test = sharpe_of(pooled, args, lo=SPLIT)
                table[(window, name)] = {"train": train, "test": test,
                                         "trades": len(pooled)}
                cells.append(f"{test:>10.2f} [{train:>5.2f}]")
            print(f"  {window:>7d}" + "".join(cells), flush=True)
        report[f"{cost * 1e4:.0f}bp"] = {f"{w}/{d}": v for (w, d), v in table.items()}

        usable = {k: v for k, v in table.items() if v["train"] == v["train"]}
        if usable:
            best = max(usable, key=lambda k: usable[k]["train"])
            ranked = sorted(usable, key=lambda k: usable[k]["train"], reverse=True)
            top = [k for k in ranked[:3]]
            median_window = statistics.median(w for w, _ in top)
            print(f"\n    training best: window {best[0]}, {best[1]} "
                  f"(train {usable[best]['train']:.2f} -> "
                  f"held out {usable[best]['test']:.2f})")
            print(f"    top three by training: "
                  + ", ".join(f"{w}/{d}" for w, d in top)
                  + f"   median window {median_window:.0f}")
            baseline = usable.get((55, "long"))
            if baseline:
                print(f"    the equity book's setting (55/long): "
                      f"train {baseline['train']:.2f} -> "
                      f"held out {baseline['test']:.2f}")
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
