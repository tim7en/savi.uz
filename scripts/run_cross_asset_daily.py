"""The validated system on the non-equity daily book, 2007-2026.

Twenty-one ETFs spanning metals, energy, broad commodity, duration, credit, FX
and volatility, on daily bars reaching back to 2002 in places.  Two things this
answers that the 30-minute equity book cannot.

First, consistency: the equity book runs ten years, eight of them bullish, and
its worst year is the only stress it has ever seen.  This book covers 2008, the
2011 and 2015 commodity collapses and the 2013 taper, so a year-by-year table
here means something the equity one does not.

Second, whether the rules transfer.  The engine is interval agnostic by design
and 55/20 on daily bars is the original Turtle specification, so if the edge is
real it should survive the move -- and if it only works on 30-minute US equities
selected in 2026, that is worth knowing before any of it is traded.

Long and short are both run: the case for shorts was never tested anywhere with
a genuine bear market in it, and commodities and duration spend years trending
down in a way US equities did not over 2017-2026.
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

from savi_uz.turtle import TurtleConfig, run_turtle  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

BASE = dict(entry_window=55, exit_window=20, atr_window=20, skip_after_winner=False,
            use_channel_exit=False, chandelier_atr=3.0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path,
                        default=Path("data/cross_assets/etf_daily.db"))
    parser.add_argument("--start", default="2007-01-01")
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/cross_asset_daily.json"))
    return parser.parse_args(argv)


def load(args):
    connection = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    sleeves = dict(connection.execute("SELECT ticker, sleeve FROM venue_map"))
    book = {}
    for (ticker,) in connection.execute(
        "SELECT DISTINCT ticker FROM bars WHERE frequency='daily' ORDER BY ticker"
    ).fetchall():
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='daily' AND ts>=? ORDER BY ts", (ticker, args.start)).fetchall()
        if len(rows) >= 400:
            book[ticker] = [Bar(*r) for r in rows]
    connection.close()
    return book, sleeves


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
    """Daily R, marking open positions at each session close."""
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


def solve_risk(series, target, lo=1e-6, hi=0.08):
    def dd(risk):
        return statistics.median(abs(path_metrics(d, v, risk)[1]) for d, v in series)
    if dd(hi) < target:
        return hi
    for _ in range(32):
        mid = math.sqrt(lo * hi)
        if dd(mid) < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def sharpe(stream):
    sd = statistics.pstdev(stream)
    return statistics.fmean(stream) / sd * math.sqrt(252) if sd > 0 else float("nan")


def main(argv=None):
    args = parse_args(argv)
    book, sleeves = load(args)
    closes_by_ticker = {t: {b.timestamp[:10]: b.close for b in bars}
                        for t, bars in book.items()}
    calendar = sorted({d for c in closes_by_ticker.values() for d in c})
    print(f"{len(book)} instruments on daily bars, "
          f"{calendar[0]} -> {calendar[-1]} ({len(calendar):,} sessions)\n", flush=True)

    books = {}
    for label, directions in (("long only", (1,)), ("both sides", (1, -1))):
        config = TurtleConfig(**BASE, directions=directions)
        pooled = []
        for ticker, bars in book.items():
            trades, _ = run_turtle(bars, config=config)
            pooled.extend({"ticker": ticker, "entry": t.entry_timestamp,
                           "exit": t.exit_timestamp, "r": t.net_r,
                           "dir": t.direction, "units": t.unit_entries}
                          for t in trades)
        books[label] = pooled

    report = {}
    print(f"  {'book':12s} {'trades':>7s} {'shorts':>7s} {'Sharpe':>7s} {'CAGR':>8s} "
          f"{'worst yr':>9s} {'yr sd':>7s} {'neg yrs':>8s}")
    for label, pooled in books.items():
        caps = [cap(pooled, args.max_positions, random.Random(s))
                for s in range(args.trials)]
        # Mark each tie-break once.  Rebuilding the map inside the year loop
        # recomputed every open position's daily excursion twenty times over.
        marks = [marked_map(taken, closes_by_ticker) for taken in caps]
        series = [(sorted(m), [m[d] for d in sorted(m)]) for m in marks]
        risk = solve_risk(series, args.target_dd)
        years = sorted({d[:4] for d in calendar})
        by_year = {}
        for year in years:
            window = [d for d in calendar if d[:4] == year]
            if len(window) < 60:
                continue
            scores = [sharpe([m.get(d, 0.0) for d in window]) for m in marks[:12]]
            scores = [s for s in scores if s == s]
            if scores:
                by_year[year] = statistics.median(scores)
        live = list(by_year.values())
        shorts = sum(1 for x in pooled if x["dir"] < 0) / max(len(pooled), 1)
        entry = {"trades": len(pooled), "short_share": shorts,
                 "sharpe": statistics.median(
                     sharpe([m.get(d, 0.0) for d in calendar]) for m in marks[:12]),
                 "cagr": statistics.median(
                     path_metrics(d, v, risk)[2] for d, v in series),
                 "risk": risk, "years": by_year, "worst_year": min(live),
                 "year_sd": statistics.pstdev(live),
                 "negative_years": sum(1 for v in live if v < 0)}
        report[label] = entry
        print(f"  {label:12s} {entry['trades']:>7,d} {shorts:>7.1%} "
              f"{entry['sharpe']:>7.2f} {entry['cagr']:>8.1%} "
              f"{entry['worst_year']:>9.2f} {entry['year_sd']:>7.2f} "
              f"{entry['negative_years']:>4d}/{len(live):<3d}", flush=True)

    years = sorted(report["long only"]["years"])
    print(f"\n  Sharpe by year:")
    for chunk in range(0, len(years), 10):
        slab = years[chunk:chunk + 10]
        print("    " + "".join(f"{y:>7s}" for y in slab))
        for label in books:
            print("    " + "".join(f"{report[label]['years'].get(y, float('nan')):>7.2f}"
                                   for y in slab) + f"   {label}")
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {k: {i: j for i, j in v.items()} for k, v in report.items()},
        indent=1, default=str), encoding="utf-8")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
