"""Does a close-location filter pay for the entry it has to give up?

The breakout-quality study found the largest conditional effect in the
programme: bottom-quintile close location returns -1.203R and stops out 86.8% of
the time, against +1.140R and 67.5% for the top quintile, monotone across five
buckets at p < 0.001.

That is not the same as a tradeable rule. Where a bar closes is final only when
the bar is, while a stop order fills mid-bar at the channel edge. Acting on the
feature therefore requires close-confirmed entry, which surrenders the channel
fill and cost 0.37 Sharpe at 5bp when it was measured on its own.

So the question is arithmetic, not statistical: is the filter worth more than
the entry it costs? Volume answered no -- it recovered roughly what
close-confirm gave up and netted to a wash against plain stop entry. Close
location is a far larger effect, so it gets its own answer.

Entries are explicit. A breakout at bar i becomes an entry at bar i+1's open,
which is what a close-confirmed rule can actually achieve, and the filter is
applied to the breakout bar's own close location. The comparison that decides it
is the filtered close-confirm book against plain stop entry -- not against
unfiltered close-confirm, which would credit the filter with beating a handicap
it created.
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
from savi_uz.turtle import TurtleConfig, rolling_extremes, run_turtle  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

FIXED = dict(entry_window=55, exit_window=20, atr_window=20,
             skip_after_winner=False, use_channel_exit=False, chandelier_atr=3.0)

LEVERED_MARKERS = ("2X", "3X", "1.5X", "BULL", "BEAR", "LEVERAGED", "INVERSE",
                   "ULTRA", "DAILY ", "SHORT ")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/intraday/bars_av.db"))
    parser.add_argument("--source-frequency", default="5min")
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--costs-bp", type=float, nargs="+", default=(5.0, 10.0))
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=(0.34, 0.62, 0.80, 0.92))
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/close_location_filter.json"))
    return parser.parse_args(argv)


def load(args):
    connection = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT ticker, name FROM symbols WHERE name IS NOT NULL").fetchall()
        drop = {t for t, n in rows if any(m in n.upper() for m in LEVERED_MARKERS)}
    except sqlite3.OperationalError:
        drop = set()
    tickers = [r[0] for r in connection.execute(
        "SELECT DISTINCT ticker FROM bars WHERE frequency=? ORDER BY ticker",
        (args.source_frequency,)) if r[0] not in drop]
    if args.limit:
        tickers = tickers[:args.limit]
    book = {}
    for ticker in tickers:
        raw = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency=? AND ts>=? ORDER BY ts",
            (ticker, args.source_frequency, args.start)).fetchall()
        if len(raw) < 4000:
            continue
        bars = [Bar(*r) for r in raw]
        if args.minutes != 5:
            bars = resample_regular_session(bars, minutes=args.minutes)
        if len(bars) >= 800:
            book[ticker] = bars
    connection.close()
    print(f"excluded {len(drop)} levered or inverse wrappers")
    return book


def confirmed_entries(bars: list[Bar], minimum: float):
    """Breakouts whose bar closed at or above ``minimum`` of its own range,
    entered at the following bar's open."""
    highs = rolling_extremes([b.high for b in bars], FIXED["entry_window"], True)
    entries: dict[int, int] = {}
    prices: dict[int, float] = {}
    for index in range(len(bars) - 1):
        bar, level = bars[index], highs[index]
        # Confirmation means the bar *closed* beyond the channel, matching the
        # engine's own close-confirm rule. Requiring only that the high breached
        # it admits every failed breakout that poked through and closed back
        # inside, which is a different and much worse strategy.
        if math.isnan(level) or bar.close <= level:
            continue
        span = bar.high - bar.low
        location = (bar.close - bar.low) / span if span > 0 else 0.5
        if location < minimum:
            continue
        entries[index + 1] = 1
        prices[index + 1] = bars[index + 1].open
    return entries, prices


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
    for _ in range(30):
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
            "taken_median": statistics.median(len(t) for t in caps),
            "sharpe": statistics.median(spread),
            "sharpe_p05": spread[int(.05 * len(spread))],
            "sharpe_p95": spread[min(int(.95 * len(spread)), len(spread) - 1)],
            "cagr": statistics.median(cagrs)}


def collect(book, config, builder):
    pooled = []
    for ticker, bars in book.items():
        if builder is None:
            trades, _ = run_turtle(bars, config=config)
        else:
            entries, prices = builder(bars)
            if not entries:
                continue
            trades, _ = run_turtle(bars, config=config, entries=entries,
                                   entry_prices=prices)
        pooled.extend({"ticker": ticker, "entry": t.entry_timestamp,
                       "exit": t.exit_timestamp, "r": t.net_r,
                       "dir": t.direction, "units": t.unit_entries}
                      for t in trades)
    return pooled


def main(argv=None) -> int:
    args = parse_args(argv)
    book = load(args)
    closes_by_ticker = {t: {b.timestamp[:10]: b.close for b in bars}
                        for t, bars in book.items()}
    print(f"{len(book)} instruments, matched to {args.target_dd:.0%} "
          f"median drawdown\n", flush=True)

    report = {}
    for bp in args.costs_bp:
        config = TurtleConfig(**FIXED, directions=(1,), round_trip_cost=bp / 10_000)
        print(f"=== {bp:g}bp round trip ===")
        print(f"  {'arm':38s} {'offered':>9s} {'taken':>8s} {'Sharpe':>7s} "
              f"{'[5-95%]':>15s} {'CAGR':>8s}")
        arms = [("stop entry, every breakout", None),
                ("close confirm, every breakout",
                 lambda b: confirmed_entries(b, 0.0))]
        arms += [(f"close confirm, close >= {t:.2f} of range",
                  (lambda t: (lambda b: confirmed_entries(b, t)))(t))
                 for t in args.thresholds]
        baseline = None
        for label, builder in arms:
            pooled = collect(book, config, builder)
            result = assess(pooled, closes_by_ticker, args)
            if result is None:
                print(f"  {label:38s}      too few trades")
                continue
            report[f"{bp:g}bp|{label}"] = {"cost_bp": bp, "arm": label, **result}
            if baseline is None:
                baseline = result["sharpe"]
            print(f"  {label:38s} {result['offered']:>9,d} "
                  f"{result['taken_median']:>8,.0f} {result['sharpe']:>7.2f} "
                  f"{('[%.2f-%.2f]' % (result['sharpe_p05'], result['sharpe_p95'])):>15s} "
                  f"{result['cagr']:>8.1%}", flush=True)
        best = max((v for k, v in report.items()
                    if v["cost_bp"] == bp and "close >=" in v["arm"]),
                   key=lambda v: v["sharpe"], default=None)
        if best and baseline is not None:
            verdict = ("beats plain stop entry by "
                       f"{best['sharpe'] - baseline:+.2f}"
                       if best["sharpe"] > baseline else
                       "does NOT beat plain stop entry "
                       f"({best['sharpe'] - baseline:+.2f})")
            print(f"  -> best filtered arm {verdict}\n", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
