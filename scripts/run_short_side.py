"""Does the system work short, and did it ever take a short?

Every number in this programme comes from ``directions=(1,)``.  Not "mostly long"
-- long only, by configuration, so the answer to "were some positions short" is
no: none were ever taken.  That leaves the short side entirely unmeasured across
a sample that happened to be a bull market, which is the largest untested surface
in the strategy.

Three books are run under the validated configuration -- fixed 3N chandelier,
identical entry and sizing rules -- differing only in which directions they are
allowed to take.  The both-sides book is the interesting one: shorts can help a
portfolio even when they lose money standalone, because they pay off when the
longs are hurting, and matched-drawdown scoring is what reveals that.  A book
that loses a little on shorts but earns the right to carry more size is ahead.

Scoring is mark-to-market at matched drawdown, per calendar year, as everywhere
else.  This is measurement rather than selection, so no variant is being chosen
and there is no kill criterion -- an unprofitable short side is a finding worth
having, not a failure.
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
            use_channel_exit=False, chandelier_atr=3.0)

BOOKS = [("long only (banked)", (1,)),
         ("short only", (-1,)),
         ("both sides", (1, -1))]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--min-sessions", type=int, default=400)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/short_side.json"))
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


def session_closes(bars):
    out = {}
    for bar in bars:
        out[bar.timestamp[:10]] = bar.close
    return out


def trade_marks(trade, closes):
    entry_day, exit_day = trade.entry_timestamp[:10], trade.exit_timestamp[:10]
    marks = []
    for day in (d for d in closes if entry_day <= d < exit_day):
        live = [u for u in trade.unit_entries if u.timestamp[:10] <= day]
        if live:
            marks.append((day, sum(trade.direction * (closes[day] - u.price) / u.n
                                   for u in live)))
    return tuple(marks)


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


def marked_series(taken):
    by_day = defaultdict(float)
    for trade in taken:
        previous = 0.0
        for day, open_r in trade["marks"]:
            by_day[day] += open_r - previous
            previous = open_r
        by_day[trade["exit"][:10]] += trade["r"] - previous
    days = sorted(by_day)
    return days, [by_day[d] for d in days]


def path_metrics(days, values, risk):
    nav, peak, worst = 1000.0, 1000.0, 0.0
    for value in values:
        nav = max(0.0, nav + value * risk * nav)
        peak = max(peak, nav)
        worst = min(worst, nav / peak - 1.0)
    if len(days) < 2:
        return nav, worst, 0.0
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25
    cagr = (nav / 1000.0) ** (1 / years) - 1 if years > 0 and nav > 0 else -1.0
    return nav, worst, cagr


def solve_risk(series, target, lo=1e-6, hi=0.05):
    def dd(risk):
        return statistics.median(abs(path_metrics(d, v, risk)[1]) for d, v in series)
    if dd(hi) < target:
        return hi
    for _ in range(35):
        mid = math.sqrt(lo * hi)
        if dd(mid) < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def sharpe(series):
    scores = []
    for days, values in series:
        if len(days) < 30:
            continue
        span = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days
        stream = values + [0.0] * max(0, int(span * 252 / 365.25) - len(values))
        sd = statistics.pstdev(stream)
        if sd > 0:
            scores.append(statistics.fmean(stream) / sd * math.sqrt(252))
    return statistics.median(scores) if scores else math.nan


def main(argv=None):
    args = parse_args(argv)
    book = load_book(args)
    print(f"{len(book)} instruments at {args.minutes}-minute bars\n", flush=True)

    report = {}
    print(f"  {'book':22s} {'trades':>8s} {'shorts':>7s} {'meanR':>8s} {'win':>6s} "
          f"{'Sharpe':>7s} {'lev':>8s} {'CAGR':>8s}")
    for label, directions in BOOKS:
        config = TurtleConfig(**BASE, directions=directions)
        pooled = []
        for ticker, bars in book.items():
            trades, _ = run_turtle(bars, config=config)
            closes = session_closes(bars)
            pooled.extend({"entry": t.entry_timestamp, "exit": t.exit_timestamp,
                           "r": t.net_r, "dir": t.direction,
                           "marks": trade_marks(t, closes)} for t in trades)
        if not pooled:
            continue
        rs = [t["r"] for t in pooled]
        shorts = [t for t in pooled if t["dir"] < 0]
        series = [marked_series(cap(pooled, args.max_positions, random.Random(s)))
                  for s in range(args.trials)]
        risk = solve_risk(series, args.target_dd)
        cagr = statistics.median(path_metrics(d, v, risk)[2] for d, v in series)
        years = sorted({d[:4] for d in series[0][0]})
        by_year = {}
        for year in years:
            sliced = [([d for d in days if d[:4] == year],
                       [v for d, v in zip(days, values) if d[:4] == year])
                      for days, values in series]
            by_year[year] = sharpe([s for s in sliced if len(s[0]) >= 30])
        print(f"  {label:22s} {len(pooled):>8,d} {len(shorts) / len(pooled):>7.1%} "
              f"{statistics.fmean(rs):>+8.3f} "
              f"{sum(1 for r in rs if r > 0) / len(rs):>6.1%} "
              f"{sharpe(series):>7.2f} {risk:>8.4%} {cagr:>8.1%}", flush=True)
        report[label] = {"trades": len(pooled), "short_share": len(shorts) / len(pooled),
                         "mean_r": statistics.fmean(rs), "total_r": sum(rs),
                         "sharpe": sharpe(series), "risk": risk, "cagr": cagr,
                         "years": by_year}

    if "short only" in report:
        shorts_only = report["short only"]
        print(f"\n  short side standalone: {shorts_only['total_r']:+,.0f}R over "
              f"{shorts_only['trades']:,} trades "
              f"({shorts_only['mean_r']:+.3f}R mean)")
    if {"long only (banked)", "both sides"} <= set(report):
        long_book, both = report["long only (banked)"], report["both sides"]
        years = sorted(long_book["years"])
        print(f"\n  Sharpe by year:")
        print("    " + "".join(f"{y:>8s}" for y in years))
        for key, tag in (("long only (banked)", "long"), ("both sides", "both"),
                         ("short only", "short")):
            if key in report:
                print("    " + "".join(f"{report[key]['years'].get(y, float('nan')):>8.2f}"
                                       for y in years) + f"   {tag}")
        wins = sum(1 for y in years if both["years"][y] > long_book["years"][y])
        print(f"\n  both sides beats long only in {wins}/{len(years)} years; "
              f"CAGR {both['cagr']:.1%} vs {long_book['cagr']:.1%} at matched drawdown")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
