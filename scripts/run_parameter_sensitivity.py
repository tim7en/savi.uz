"""Entry-window and volume sensitivity at Binance execution costs.

Three questions this answers that the programme has been carrying unexamined.

*Where did 55 come from?*  Nowhere defensible.  It is the original Turtle System
2 specification, written for daily bars, where 55 bars is eleven weeks.  On
30-minute bars it is 4.2 sessions and on 5-minute bars 35 minutes, which is a
different instrument wearing the same number.  A parameter inherited rather than
chosen has to be shown to sit on a plateau, not a spike -- otherwise the banked
result is a fit to one cell of a grid nobody swept.

*What does it cost to trade?*  Binance charges 10bp round trip taking and 5bp
making.  Every prior figure in this programme assumed 2bp, which is not a price
anyone here can get.  Both real levels are swept and neither is hidden.

*Does volume help once it is measured properly?*  The old volume series was IEX,
1.89% of the tape with a 0.625 rank correlation to it.  On consolidated volume
the filter is re-asked across two rolling-mean windows and two thresholds --
``relative_volume`` compares each bar against the mean of the ``volume_window``
bars before it, so the window is as much a free parameter as the threshold and
is swept rather than fixed at its default.

The exit is held at the banked chandelier 3N throughout; the channel exit is off.
The whole surface is printed, never the best cell, because the point of a
sensitivity sweep is to see whether the neighbourhood holds up.
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

#: Everything except the entry window and the volume rule is the banked config.
FIXED = dict(atr_window=20, skip_after_winner=False,
             use_channel_exit=False, chandelier_atr=3.0)

#: Binance: 10bp round trip taking liquidity, 5bp making it.
COSTS = {"maker 5bp": 5.0, "taker 10bp": 10.0}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/intraday/bars_av.db"))
    parser.add_argument("--source-frequency", default="5min")
    parser.add_argument("--minutes", type=int, default=30,
                        help="resample target; 5 keeps the native bars")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--entry-windows", type=int, nargs="+",
                        default=(20, 34, 55, 89, 144))
    parser.add_argument("--volume-windows", type=int, nargs="+", default=(20, 50))
    parser.add_argument("--volume-thresholds", type=float, nargs="+", default=(1.5, 2.0))
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--exclude-levered", action="store_true",
                        help="drop levered and inverse wrappers")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/parameter_sensitivity.json"))
    return parser.parse_args(argv)


LEVERED_MARKERS = ("2X", "3X", "1.5X", "BULL", "BEAR", "LEVERAGED", "INVERSE",
                   "ULTRA", "DAILY ", "SHORT ")


def levered(connection: sqlite3.Connection) -> set[str]:
    try:
        rows = connection.execute(
            "SELECT ticker, name FROM symbols WHERE name IS NOT NULL").fetchall()
    except sqlite3.OperationalError:
        return set()
    return {t for t, n in rows if any(m in n.upper() for m in LEVERED_MARKERS)}


def load(args) -> dict[str, list[Bar]]:
    connection = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    drop = levered(connection) if args.exclude_levered else set()
    tickers = [r[0] for r in connection.execute(
        "SELECT DISTINCT ticker FROM bars WHERE frequency=? ORDER BY ticker",
        (args.source_frequency,)) if r[0] not in drop]
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
        bars = [Bar(*r) for r in rows]
        if args.minutes != 5:
            bars = resample_regular_session(bars, minutes=args.minutes)
        if len(bars) >= 800:
            book[ticker] = bars
    connection.close()
    if drop:
        print(f"excluded {len(drop)} levered or inverse wrappers")
    return book


def cap(trades, limit, rng):
    """Fill ``limit`` slots in entry order, breaking same-moment ties at random.

    Returns the taken trades and the refusal count. The refusal count is the
    diagnostic that matters once the universe is large: with 131 symbols the cap
    binds almost continuously, so the book is a random six-of-many draw and the
    dispersion across seeds is a property of the sampling rather than the rules.
    """
    shuffled = list(trades)
    rng.shuffle(shuffled)
    live, taken = [], []
    refused = 0
    for trade in sorted(shuffled, key=lambda t: t["entry"]):
        live = [x for x in live if x["exit"] > trade["entry"]]
        if len(live) >= limit:
            refused += 1
            continue
        live.append(trade)
        taken.append(trade)
    return taken, refused


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
    for _ in range(28):
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
    if len(pooled) < 50:
        return {"trades": len(pooled)}
    capped = [cap(pooled, args.max_positions, random.Random(s))
              for s in range(args.trials)]
    caps = [taken for taken, _ in capped]
    refusals = sorted(count for _, count in capped)
    marks = [marked_map(taken, closes_by_ticker) for taken in caps]
    series = [(sorted(m), [m[d] for d in sorted(m)]) for m in marks]
    risk = solve_risk(series, args.target_dd)
    by_year = {}
    for year in sorted({d[:4] for d in calendar}):
        window = [d for d in calendar if d[:4] == year]
        if len(window) < 60:
            continue
        scores = [sharpe([m.get(d, 0.0) for d in window]) for m in marks[:6]]
        scores = [s for s in scores if s == s]
        if scores:
            by_year[year] = statistics.median(scores)
    live = list(by_year.values())
    spread = sorted(sharpe([m.get(d, 0.0) for d in calendar]) for m in marks)
    taken = statistics.median(len(t) for t in caps)
    return {
        "trades_offered": len(pooled),
        "trades_taken_median": taken,
        "refused_median": statistics.median(refusals),
        "refusal_rate": statistics.median(refusals) / len(pooled),
        "sharpe": statistics.median(spread),
        "sharpe_p05": spread[int(.05 * len(spread))],
        "sharpe_p95": spread[min(int(.95 * len(spread)), len(spread) - 1)],
        "cagr": statistics.median(path_metrics(d, v, risk)[2] for d, v in series),
        "years": by_year,
        "worst_year": min(live) if live else float("nan"),
        "negative_years": sum(1 for v in live if v < 0),
        "year_count": len(live),
    }


def arms(args):
    """(label, entry_mode, volume_window, threshold) for every volume rule."""
    out = [("no volume filter", "stop", 20, 0.0),
           ("close confirm, no filter", "close confirm", 20, 0.0)]
    for window in args.volume_windows:
        for threshold in args.volume_thresholds:
            out.append((f"vol>={threshold:g}x of {window}-bar mean",
                        "close confirm", window, threshold))
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    book = load(args)
    if not book:
        raise SystemExit("error: no usable bars")
    closes_by_ticker = {t: {b.timestamp[:10]: b.close for b in bars}
                        for t, bars in book.items()}
    calendar = sorted({d for c in closes_by_ticker.values() for d in c})
    print(f"{len(book)} instruments, {args.minutes}-minute bars, "
          f"{calendar[0]} -> {calendar[-1]} ({len(calendar):,} sessions)")
    print(f"entry windows {list(args.entry_windows)}; exit = chandelier 3N; "
          f"costs {list(COSTS)}\n", flush=True)

    report: dict[str, dict] = {}
    for cost_label, bp in COSTS.items():
        print(f"=== {cost_label} ===")
        print(f"  {'arm':30s}" + "".join(f"{w:>9d}" for w in args.entry_windows)
              + "     Sharpe, and [5-95% across random tie-breaks]")
        for arm_label, mode, volume_window, threshold in arms(args):
            cells = []
            spreads = []
            refusals = []
            for entry_window in args.entry_windows:
                config = TurtleConfig(
                    **FIXED, entry_window=entry_window,
                    exit_window=min(20, entry_window - 1),
                    directions=(1,), round_trip_cost=bp / 10_000,
                    entry_mode=mode, min_relative_volume=threshold,
                    volume_window=volume_window,
                )
                pooled = []
                for ticker, bars in book.items():
                    trades, _ = run_turtle(bars, config=config)
                    pooled.extend({"ticker": ticker, "entry": t.entry_timestamp,
                                   "exit": t.exit_timestamp, "r": t.net_r,
                                   "dir": t.direction, "units": t.unit_entries}
                                  for t in trades)
                result = evaluate(pooled, closes_by_ticker, calendar, args)
                report[f"{cost_label}|{arm_label}|{entry_window}"] = {
                    "cost_bp": bp, "arm": arm_label, "entry_window": entry_window,
                    "volume_window": volume_window, "threshold": threshold, **result}
                cells.append(result.get("sharpe"))
                spreads.append((result.get("sharpe_p05"), result.get("sharpe_p95")))
                refusals.append(result.get("refusal_rate"))
            print(f"  {arm_label:30s}" + "".join(
                f"{c:>9.2f}" if c is not None else f"{'--':>9s}" for c in cells),
                flush=True)
            band = "".join(
                f"{('[%.2f-%.2f]' % s):>9s}" if s[0] is not None else f"{'':>9s}"
                for s in spreads)
            drop = "".join(
                f"{r:>9.0%}" if r is not None else f"{'':>9s}" for r in refusals)
            print(f"  {'  band':30s}{band}", flush=True)
            print(f"  {'  refused':30s}{drop}\n", flush=True)
        print(flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
