"""The banked rules on the full trad-FI universe, with volume that means something.

Two things could not be asked before this database existed.

*The universe.* The book was 42 names chosen in 2026. Binance now lists 137 US
trad-FI perpetuals, and Alpha Vantage supplies 5-minute history from 2015 for
all of them, so the rules can be run on a universe that was not picked by
hindsight over the same window.

*The volume.* The old 5-minute store was Tiingo, whose US intraday feed is IEX
only: 1.89% of consolidated volume, no volume at all on 13.5% of bars, and a
rank correlation of 0.625 with the real tape on which bars were busy. Both
volume overlays in the programme were rejected on that series. This re-asks the
question on consolidated volume.

The volume filter needs a close-confirmed entry, because a stop order fills
before the breakout bar's volume is known. That is a real cost -- it gives up
the channel-edge fill -- so the control is close-confirm *without* the filter,
never the stop-entry baseline. Comparing a filtered close-confirm book against
the stop-entry book would credit the filter with the entry change.

Costs are swept rather than assumed. The programme's binding fact is that the
cross-asset book scores 2.57 at 2bp and 0.15 at 15bp; no arm here is quoted at
a single cost.
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

BASE = dict(entry_window=55, exit_window=20, atr_window=20, skip_after_winner=False,
            use_channel_exit=False, chandelier_atr=3.0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/intraday/bars_av.db"))
    parser.add_argument("--source-frequency", default="5min")
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--costs-bp", type=float, nargs="+", default=(2.0, 15.0))
    parser.add_argument("--volume-thresholds", type=float, nargs="+",
                        default=(1.5, 2.0))
    parser.add_argument("--limit", type=int, default=0, help="first N tickers only")
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/tradfi_volume_study.json"))
    return parser.parse_args(argv)


def load(args) -> dict[str, list[Bar]]:
    connection = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    tickers = [r[0] for r in connection.execute(
        "SELECT DISTINCT ticker FROM bars WHERE frequency=? ORDER BY ticker",
        (args.source_frequency,))]
    if args.limit:
        tickers = tickers[:args.limit]
    book = {}
    for ticker in tickers:
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency=? AND ts>=? ORDER BY ts",
            (ticker, args.source_frequency, args.start)).fetchall()
        if len(rows) < 4000:
            continue
        bars = resample_regular_session([Bar(*r) for r in rows], minutes=args.minutes)
        if len(bars) >= 400:
            book[ticker] = bars
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


def marked_map(taken, closes_by_ticker):
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


def evaluate(pooled, closes_by_ticker, calendar, args):
    if not pooled:
        return {"trades": 0}
    caps = [cap(pooled, args.max_positions, random.Random(s)) for s in range(args.trials)]
    marks = [marked_map(taken, closes_by_ticker) for taken in caps]
    series = [(sorted(m), [m[d] for d in sorted(m)]) for m in marks]
    risk = solve_risk(series, args.target_dd)
    by_year = {}
    for year in sorted({d[:4] for d in calendar}):
        window = [d for d in calendar if d[:4] == year]
        if len(window) < 60:
            continue
        scores = [sharpe([m.get(d, 0.0) for d in window]) for m in marks[:8]]
        scores = [s for s in scores if s == s]
        if scores:
            by_year[year] = statistics.median(scores)
    live = list(by_year.values())
    return {
        "trades": len(pooled),
        "sharpe": statistics.median(
            sharpe([m.get(d, 0.0) for d in calendar]) for m in marks[:8]),
        "cagr": statistics.median(path_metrics(d, v, risk)[2] for d, v in series),
        "risk": risk,
        "years": by_year,
        "worst_year": min(live) if live else float("nan"),
        "negative_years": sum(1 for v in live if v < 0),
        "year_count": len(live),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    book = load(args)
    if not book:
        raise SystemExit(f"error: no usable {args.source_frequency} bars in {args.db}")
    closes_by_ticker = {t: {b.timestamp[:10]: b.close for b in bars}
                        for t, bars in book.items()}
    calendar = sorted({d for c in closes_by_ticker.values() for d in c})
    print(f"{len(book)} instruments on {args.minutes}-minute bars, "
          f"{calendar[0]} -> {calendar[-1]} ({len(calendar):,} sessions)\n", flush=True)

    arms = [("stop entry, no volume filter", "stop", 0.0),
            ("close confirm, no volume filter", "close confirm", 0.0)]
    arms += [(f"close confirm, volume >= {v:g}x", "close confirm", v)
             for v in args.volume_thresholds]

    report = {}
    print(f"  {'cost':>6s}  {'arm':34s} {'trades':>8s} {'Sharpe':>7s} {'CAGR':>8s} "
          f"{'worst yr':>9s} {'neg yrs':>8s}")
    for bp in args.costs_bp:
        for label, mode, threshold in arms:
            config = TurtleConfig(**BASE, directions=(1,),
                                  round_trip_cost=bp / 10_000,
                                  entry_mode=mode, min_relative_volume=threshold)
            pooled = []
            for ticker, bars in book.items():
                trades, _ = run_turtle(bars, config=config)
                pooled.extend({"ticker": ticker, "entry": t.entry_timestamp,
                               "exit": t.exit_timestamp, "r": t.net_r,
                               "dir": t.direction, "units": t.unit_entries}
                              for t in trades)
            entry = evaluate(pooled, closes_by_ticker, calendar, args)
            report[f"{bp:g}bp|{label}"] = {"cost_bp": bp, "arm": label, **entry}
            if entry["trades"]:
                print(f"  {bp:>4g}bp  {label:34s} {entry['trades']:>8,d} "
                      f"{entry['sharpe']:>7.2f} {entry['cagr']:>8.1%} "
                      f"{entry['worst_year']:>9.2f} "
                      f"{entry['negative_years']:>3d}/{entry['year_count']:<3d}", flush=True)
            else:
                print(f"  {bp:>4g}bp  {label:34s} {0:>8d}   no trades", flush=True)
        print(flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
