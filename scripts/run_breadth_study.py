"""Does adding instruments still pay once the exit is fixed?

The raw R book kept climbing with breadth -- 6,913 to 14,037R going from 10 to 42
instruments -- but raw R is the wrong yardstick twice over.  More instruments mean
more trades, so R rises even if each one is no better; and the earlier reading was
taken under the Donchian exit, which the matched-risk study has since shown to be
the deteriorating part of the system.

So the question is asked properly here: at each universe size, lever the book to
the same drawdown and read off the return.  If breadth is real, matched-risk return
keeps rising because the diversification lets the same pain carry more size.  If it
is just trade count, the curve flattens.

Both exits are run at every size, because the two may not want the same breadth:
a rule that exits sooner frees capacity, which is worth more when there are more
instruments competing for it.

Instruments are drawn at random and averaged over many draws, so no single lucky
name decides a point on the curve.  Accounting is mark-to-market throughout -- open
positions hit the equity curve on the day they move, not on the day they close.
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

EXITS = [("channel 20 (old)", {}),
         ("chandelier 3N (new)", dict(use_channel_exit=False, chandelier_atr=3.0))]

SIZES = (5, 10, 15, 20, 25, 30, 35, 42)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--min-sessions", type=int, default=400)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--cost", type=float, default=0.0002)
    parser.add_argument("--draws", type=int, default=25,
                        help="random instrument subsets per universe size")
    parser.add_argument("--trials", type=int, default=20,
                        help="capacity tie-breaks per subset")
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/breadth_study.json"))
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
    """Open P&L in R at each session close held through, exit day excluded."""
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
        return statistics.median(
            abs(path_metrics(d, v, risk)[1]) for d, v in series)
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
    BASE["round_trip_cost"] = args.cost
    book = load_book(args)
    tickers = sorted(book)
    sizes = [n for n in SIZES if n <= len(tickers)]
    print(f"{len(tickers)} instruments at {args.minutes}-minute bars, "
          f"sizes {sizes}\n", flush=True)

    per_exit = {}
    for label, overrides in EXITS:
        config = TurtleConfig(**{**BASE, **overrides})
        per_ticker = {}
        for ticker, bars in book.items():
            trades, _ = run_turtle(bars, config=config)
            closes = session_closes(bars)
            per_ticker[ticker] = [
                {"entry": t.entry_timestamp, "exit": t.exit_timestamp,
                 "r": t.net_r, "marks": trade_marks(t, closes)} for t in trades]
        per_exit[label] = per_ticker

    report = {}
    for label, _ in EXITS:
        per_ticker = per_exit[label]
        print(f"  {label}")
        print(f"    {'names':>6s} {'trades':>8s} {'Sharpe':>7s} {'lev':>7s} "
              f"{'CAGR':>8s} {'CAGR/name':>10s}")
        curve = []
        for size in sizes:
            cagrs, sharpes, levs, counts = [], [], [], []
            for draw in range(args.draws):
                rng = random.Random(10_000 * size + draw)
                picked = rng.sample(tickers, size) if size < len(tickers) else tickers
                pooled = [t for name in picked for t in per_ticker[name]]
                if len(pooled) < 100:
                    continue
                series = [marked_series(cap(pooled, args.max_positions,
                                            random.Random(seed)))
                          for seed in range(args.trials)]
                risk = solve_risk(series, args.target_dd)
                cagrs.append(statistics.median(
                    path_metrics(d, v, risk)[2] for d, v in series))
                sharpes.append(sharpe(series))
                levs.append(risk)
                counts.append(len(pooled))
                if size == len(tickers):
                    break
            if not cagrs:
                continue
            row = {"size": size, "trades": statistics.median(counts),
                   "sharpe": statistics.median(sharpes),
                   "risk": statistics.median(levs),
                   "cagr": statistics.median(cagrs)}
            curve.append(row)
            print(f"    {size:>6d} {row['trades']:>8,.0f} {row['sharpe']:>7.2f} "
                  f"{row['risk']:>7.4%} {row['cagr']:>8.1%} "
                  f"{row['cagr'] / size:>10.2%}")
        report[label] = curve
        if len(curve) >= 3:
            last, mid = curve[-1], curve[len(curve) // 2]
            gain = (last["cagr"] / mid["cagr"] - 1) if mid["cagr"] > 0 else float("nan")
            print(f"    -> {mid['size']} to {last['size']} names: "
                  f"{gain:+.0%} matched-risk return, "
                  f"Sharpe {mid['sharpe']:.2f} -> {last['sharpe']:.2f}\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
