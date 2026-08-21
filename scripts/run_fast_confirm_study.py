"""A slow channel confirmed on a fast bar, across breakout intervals.

Close-confirmed entry lost 0.73 Sharpe to a plain stop because of *where* it
fills, not when. A stop fills at the channel edge the instant price crosses;
confirmation fills wherever the bar happened to close, which on the same
breakout events is a median 0.401N above the level -- a fifth of the way to a
2N stop before the trade has started.

Running everything on faster bars does not fix that: measured give-up is
0.424N at 5 minutes and 0.401N at 30, because N shrinks with the interval and
the ratio is scale-invariant. What does help is the *mismatch* -- keeping the
slow channel and accepting the first fast bar that closes beyond it. On the
same 10,809 events that halves the median give-up to 0.196N and is cheaper in
64% of cases.

This tests whether that saving is worth enough to matter, at breakout intervals
of 5, 15, 30 and 60 minutes, with four arms each:

* **stop entry** -- fills at the channel edge, the banked behaviour and the
  number every other arm has to beat;
* **same-interval close confirm** -- the expensive version already measured;
* **fast confirm** -- the first 5-minute close beyond the level, filled there;
* **fast confirm, filtered** -- the same, admitted only when that 5-minute bar
  closed in the upper part of its own range.

The trade is managed on the channel's own bars throughout; only the fill price
changes, so this is an execution change rather than a different strategy. One
consequence is worth naming: the engine evaluates the entry bar's whole range
against the stop, including the part that preceded the fast fill. That is
pessimistic rather than optimistic, which is the safe direction for an entry
improvement to err in.
"""

from __future__ import annotations

import argparse
import bisect
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
from savi_uz.turtle import TurtleConfig, rolling_extremes, run_turtle  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

FIXED = dict(entry_window=55, exit_window=20, atr_window=20,
             skip_after_winner=False, use_channel_exit=False, chandelier_atr=3.0)

LEVERED_MARKERS = ("2X", "3X", "1.5X", "BULL", "BEAR", "LEVERAGED", "INVERSE",
                   "ULTRA", "DAILY ", "SHORT ")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/intraday/bars_av.db"))
    parser.add_argument("--intervals", type=int, nargs="+", default=(5, 15, 30, 60))
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--costs-bp", type=float, nargs="+", default=(10.0, 5.0))
    parser.add_argument("--location-threshold", type=float, default=0.62)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/fast_confirm.json"))
    return parser.parse_args(argv)


def load_five(args):
    connection = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT ticker, name FROM symbols WHERE name IS NOT NULL").fetchall()
        drop = {t for t, n in rows if any(m in n.upper() for m in LEVERED_MARKERS)}
    except sqlite3.OperationalError:
        drop = set()
    tickers = [r[0] for r in connection.execute(
        "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker")
        if r[0] not in drop]
    if args.limit:
        tickers = tickers[:args.limit]
    book = {}
    for ticker in tickers:
        raw = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='5min' AND ts>=? ORDER BY ts",
            (ticker, args.start)).fetchall()
        if len(raw) >= 4000:
            book[ticker] = [Bar(*r) for r in raw]
    connection.close()
    print(f"excluded {len(drop)} levered or inverse wrappers")
    return book


def entry_plans(bars, five, minutes, threshold):
    """Explicit entries for the confirmation arms, keyed by channel-bar index."""
    highs = rolling_extremes([b.high for b in bars], FIXED["entry_window"], True)
    stamps = [b.timestamp for b in five]
    same_e, same_p = {}, {}
    fast_e, fast_p = {}, {}
    filt_e, filt_p = {}, {}
    for index in range(1, len(bars) - 1):
        bar, level = bars[index], highs[index]
        if math.isnan(level) or bar.high <= level:
            continue
        if bar.close > level:
            same_e[index] = 1
            same_p[index] = bar.close
        if minutes == 5:
            continue
        lo = bisect.bisect_left(stamps, bars[index].timestamp)
        hi = bisect.bisect_left(stamps, bars[index + 1].timestamp)
        for k in range(lo, hi):
            candidate = five[k]
            if candidate.close <= level:
                continue
            fast_e[index] = 1
            fast_p[index] = candidate.close
            span = candidate.high - candidate.low
            location = (candidate.close - candidate.low) / span if span > 0 else 0.5
            if location >= threshold:
                filt_e[index] = 1
                filt_p[index] = candidate.close
            break
    return (same_e, same_p), (fast_e, fast_p), (filt_e, filt_p)


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


def path(values, risk):
    nav = peak = 1000.0
    worst = 0.0
    for value in values:
        nav = max(0.0, nav + value * risk * nav)
        peak = max(peak, nav)
        worst = min(worst, nav / peak - 1.0)
    return nav, worst


def solve_risk(series, target, lo=1e-6, hi=0.40):
    def dd(risk):
        return statistics.median(abs(path(v, risk)[1]) for _, v in series)
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


def assess(pooled, closes_by_ticker, args):
    if len(pooled) < 200:
        return None
    caps = [cap(pooled, args.max_positions, random.Random(s))
            for s in range(args.trials)]
    marks = [marked_map(t, closes_by_ticker) for t in caps]
    series = [(sorted(m), [m[d] for d in sorted(m)]) for m in marks]
    risk = solve_risk(series, args.target_dd)
    cagrs, sharpes = [], []
    for days, values in series:
        nav, _ = path(values, risk)
        years = (date.fromisoformat(days[-1])
                 - date.fromisoformat(days[0])).days / 365.25
        cagrs.append((nav / 1000.0) ** (1 / years) - 1 if nav > 0 else -1.0)
        sharpes.append(sharpe([v * risk for v in values]))
    spread = sorted(sharpes)
    return {"offered": len(pooled),
            "sharpe": statistics.median(spread),
            "sharpe_p05": spread[int(.05 * len(spread))],
            "sharpe_p95": spread[min(int(.95 * len(spread)), len(spread) - 1)],
            "cagr": statistics.median(cagrs)}


def main(argv=None) -> int:
    args = parse_args(argv)
    five_by_ticker = load_five(args)
    report = {}

    for minutes in args.intervals:
        book, plans = {}, {}
        for ticker, five in five_by_ticker.items():
            bars = five if minutes == 5 else resample_regular_session(
                five, minutes=minutes)
            if len(bars) < 800:
                continue
            book[ticker] = bars
            plans[ticker] = entry_plans(bars, five, minutes,
                                        args.location_threshold)
        closes_by_ticker = {t: {b.timestamp[:10]: b.close for b in bars}
                            for t, bars in book.items()}
        print(f"\n########## {minutes}-minute breakout channel "
              f"({len(book)} instruments) ##########", flush=True)

        for bp in args.costs_bp:
            config = TurtleConfig(**FIXED, directions=(1,),
                                  round_trip_cost=bp / 10_000)
            print(f"  --- {bp:g}bp ---")
            print(f"  {'arm':34s} {'offered':>9s} {'Sharpe':>7s} "
                  f"{'[5-95%]':>15s} {'CAGR':>8s}")
            arms = {"stop entry (channel edge)": None,
                    f"{minutes}m close confirm": 0,
                    "fast confirm, first 5m close": 1,
                    f"fast confirm, close >= {args.location_threshold:.2f}": 2}
            baseline = None
            for label, which in arms.items():
                if which is not None and minutes == 5 and which > 0:
                    continue
                pooled = []
                for ticker, bars in book.items():
                    if which is None:
                        trades, _ = run_turtle(bars, config=config)
                    else:
                        entries, prices = plans[ticker][which]
                        if not entries:
                            continue
                        trades, _ = run_turtle(bars, config=config,
                                               entries=entries,
                                               entry_prices=prices)
                    pooled.extend({"ticker": ticker, "entry": t.entry_timestamp,
                                   "exit": t.exit_timestamp, "r": t.net_r,
                                   "dir": t.direction, "units": t.unit_entries}
                                  for t in trades)
                result = assess(pooled, closes_by_ticker, args)
                if result is None:
                    continue
                report[f"{minutes}m|{bp:g}bp|{label}"] = {
                    "interval": minutes, "cost_bp": bp, "arm": label, **result}
                if baseline is None:
                    baseline = result["sharpe"]
                delta = ("" if baseline is None or label.startswith("stop")
                         else f"  ({result['sharpe'] - baseline:+.2f})")
                print(f"  {label:34s} {result['offered']:>9,d} "
                      f"{result['sharpe']:>7.2f} "
                      f"{('[%.2f-%.2f]' % (result['sharpe_p05'], result['sharpe_p95'])):>15s} "
                      f"{result['cagr']:>8.1%}{delta}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
