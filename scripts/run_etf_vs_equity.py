"""The validated strategy on ETFs and on single names, entry and exit held fixed.

Same 55-bar Donchian entry, same 3N chandelier trail, same sizing, same capacity
cap.  Only the universe changes, so any difference belongs to the instruments
rather than to the rules.

Three comparisons, because each answers something the others cannot.

*The overlap.*  Both books restricted to 2017-2026.  This is the only clean
head-to-head: the ETF history reaches back to 2002 and the equity history does
not, so comparing full samples would confound the universe with the era.

*The full ETF history.*  2002-2026 on its own, which is the point of having
fetched it.  It contains 2008, 2011, 2015 and 2020 -- the regimes the equity book
has never seen, and where a long-only breakout system should struggle most.

*Cost.*  Every measure is reported at 2bp and at 15bp.  A previous pass on twelve
months of this data found thirteen of twenty-one names taking no trades at all at
15bp, because the N floor is five times the round trip and the duration and FX
names are too quiet to clear it.  That verdict rested on one calm year; this book
spans two decades and settles whether the cost floor is structural or was simply
a feature of 2025.

The correlation between the two books' daily streams is reported alongside,
since the case for holding ETFs at all was that they diversify rather than that
they outperform.
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
            directions=(1,), use_channel_exit=False, chandelier_atr=3.0)
OVERLAP = "2017-01-01"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--etf", type=Path,
                        default=Path("data/cross_assets/etf_30min.db"))
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/etf_vs_equity.json"))
    return parser.parse_args(argv)


def load_equity(args):
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
        if len({b.timestamp[:10] for b in five}) < 400:
            continue
        book[ticker] = resample_regular_session(five, minutes=args.minutes)
    connection.close()
    return book


def load_etf(args):
    connection = sqlite3.connect(f"file:{args.etf}?mode=ro", uri=True)
    names = [r[0] for r in connection.execute(
        "SELECT DISTINCT ticker FROM bars ORDER BY ticker")]
    book = {}
    for ticker in names:
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? "
            "ORDER BY ts", (ticker,)).fetchall()
        if len(rows) < 2000:
            continue
        # Already 30-minute and session anchored; running it through the same
        # resampler is idempotent and applies the identical session filter, so
        # both universes are shaped by one code path rather than two.
        book[ticker] = resample_regular_session([Bar(*r) for r in rows],
                                                minutes=args.minutes)
    connection.close()
    return book


def clip(book, start=None):
    if start is None:
        return book
    return {t: [b for b in bars if b.timestamp >= start]
            for t, bars in book.items()
            if len([b for b in bars if b.timestamp >= start]) > 400}


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


def marked_map(taken):
    by_day = defaultdict(float)
    for trade in taken:
        previous = 0.0
        for day, open_r in trade["marks"]:
            by_day[day] += open_r - previous
            previous = open_r
        by_day[trade["exit"][:10]] += trade["r"] - previous
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
    for _ in range(30):
        mid = math.sqrt(lo * hi)
        if dd(mid) < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def sharpe_stream(stream):
    sd = statistics.pstdev(stream)
    return statistics.fmean(stream) / sd * math.sqrt(252) if sd > 0 else float("nan")


def evaluate(book, args, cost, label):
    config = TurtleConfig(**{**BASE, "round_trip_cost": cost})
    pooled, traded = [], set()
    for ticker, bars in book.items():
        closes = session_closes(bars)
        found, _ = run_turtle(bars, config=config)
        if found:
            traded.add(ticker)
        pooled.extend({"entry": t.entry_timestamp, "exit": t.exit_timestamp,
                       "r": t.net_r, "marks": trade_marks(t, closes)} for t in found)
    if len(pooled) < 100:
        return {"label": label, "names": len(book), "traded": len(traded),
                "trades": len(pooled), "sharpe": float("nan"),
                "cagr": float("nan"), "worst_year": float("nan"), "years": {},
                "marks": {}}
    caps = [cap(pooled, args.max_positions, random.Random(s))
            for s in range(args.trials)]
    marks = [marked_map(c) for c in caps]
    series = [(sorted(m), [m[d] for d in sorted(m)]) for m in marks]
    risk = solve_risk(series, args.target_dd)
    calendar = sorted({d for m in marks for d in m})
    years = sorted({d[:4] for d in calendar})
    by_year = {}
    for year in years:
        window = [d for d in calendar if d[:4] == year]
        if len(window) < 60:
            continue
        scores = [sharpe_stream([m.get(d, 0.0) for d in window]) for m in marks[:10]]
        scores = [s for s in scores if s == s]
        if scores:
            by_year[year] = statistics.median(scores)
    live = list(by_year.values())
    return {"label": label, "names": len(book), "traded": len(traded),
            "trades": len(pooled),
            "sharpe": statistics.median(
                sharpe_stream([m.get(d, 0.0) for d in calendar]) for m in marks[:10]),
            "cagr": statistics.median(path_metrics(d, v, risk)[2] for d, v in series),
            "years": by_year, "worst_year": min(live) if live else float("nan"),
            "negative_years": sum(1 for v in live if v < 0),
            "marks": marks[0]}


def main(argv=None):
    args = parse_args(argv)
    equity = load_equity(args)
    etf = load_etf(args)
    per_session = statistics.median(
        statistics.median(
            len([b for b in bars if b.timestamp[:10] == d])
            for d in sorted({x.timestamp[:10] for x in bars})[:50])
        for bars in list(etf.values())[:5])
    print(f"equity {len(equity)} names, ETF {len(etf)} names "
          f"({per_session:.0f} bars per ETF session)\n", flush=True)

    books = [("equity 42, 2017-26", clip(equity, OVERLAP)),
             ("ETF 21, 2017-26", clip(etf, OVERLAP)),
             ("ETF 21, full 2002-26", etf),
             ("both, 2017-26", {**clip(equity, OVERLAP), **clip(etf, OVERLAP)})]

    report = {}
    for cost in (0.0002, 0.0015):
        print(f"{'=' * 82}\nround-trip cost {cost * 1e4:.0f}bp")
        print(f"  {'book':24s} {'names':>6s} {'traded':>7s} {'trades':>8s} "
              f"{'Sharpe':>7s} {'CAGR':>8s} {'worst yr':>9s} {'neg yrs':>8s}")
        for label, book in books:
            result = evaluate(book, args, cost, label)
            report[f"{label} @{cost * 1e4:.0f}bp"] = {
                k: v for k, v in result.items() if k != "marks"}
            if result["trades"] < 100:
                print(f"  {label:24s} {result['names']:>6d} {result['traded']:>7d} "
                      f"{result['trades']:>8,d}   (too few trades to score)")
                continue
            print(f"  {label:24s} {result['names']:>6d} {result['traded']:>7d} "
                  f"{result['trades']:>8,d} {result['sharpe']:>7.2f} "
                  f"{result['cagr']:>8.1%} {result['worst_year']:>9.2f} "
                  f"{result['negative_years']:>4d}/{len(result['years']):<3d}",
                  flush=True)
            if cost == 0.0002 and label in ("equity 42, 2017-26", "ETF 21, 2017-26"):
                report.setdefault("_marks", {})[label] = result["marks"]
        print()

    marks = report.pop("_marks", {})
    if len(marks) == 2:
        a, b = marks["equity 42, 2017-26"], marks["ETF 21, 2017-26"]
        days = sorted(set(a) | set(b))
        xs = [a.get(d, 0.0) for d in days]
        ys = [b.get(d, 0.0) for d in days]
        rho = statistics.correlation(xs, ys)
        down = [(x, y) for x, y in zip(xs, ys) if x < 0]
        rho_down = (statistics.correlation([x for x, _ in down], [y for _, y in down])
                    if len(down) > 30 else float("nan"))
        print(f"  daily R correlation, equity vs ETF (2017-26): {rho:+.3f}")
        print(f"  on the equity book's down days:               {rho_down:+.3f}"
              f"   ({len(down):,} days)")
        report["correlation"] = {"all": rho, "equity_down": rho_down}

    full = report.get("ETF 21, full 2002-26 @2bp", {})
    if full.get("years"):
        years = sorted(full["years"])
        print(f"\n  ETF book by year, full history at 2bp:")
        for chunk in range(0, len(years), 13):
            slab = years[chunk:chunk + 13]
            print("    " + "".join(f"{y[2:]:>6s}" for y in slab))
            print("    " + "".join(f"{full['years'][y]:>6.2f}" for y in slab))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
