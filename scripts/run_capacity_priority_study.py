"""Does relative strength deserve the scarce slot?

The book caps at six concurrent positions and, when more names break out than
there is room for, picks among them at random.  On 42 instruments that was an
occasional coin flip.  On 120 it decides a large share of what is actually
traded, which makes the tie-break rule a first-order design choice rather than
an implementation detail.

The proposal is to give the slot to the instrument with the stronger trailing
relative performance.  It is worth separating from the eleven rejected overlays
on two grounds: cross-sectional momentum is measured from price rather than from
a vendor series, and this is a *tie-break* rather than a filter -- it removes no
trade, it only orders the ones already competing.

The nearest internal precedent nevertheless failed.  Theme trend strength scored
a top-minus-bottom quintile of -0.104R against a +1.762R baseline, on the
reasoning that a 55-bar breakout already conditions on trend and measuring trend
twice adds noise.  That was a filter, so it does not settle this, but it is the
reason the reversal control below is the decisive one rather than a formality.

Four arms, and the third is the one that matters:

* random -- the current behaviour, and the baseline;
* strongest first -- the proposal;
* weakest first -- its exact reversal.  If ranking worst-first performs as well
  as best-first, the ordering carries no information and any gap between the
  proposal and random is noise that happened to fall the right way;
* the same comparison held at matched drawdown, because a rule that
  systematically prefers higher-volatility names would otherwise be credited
  for carrying more risk.

Scores are trailing total return over a lookback ending at the bar *before*
entry, so nothing in the ranking is knowable only after the decision.
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
from savi_uz.turtle import TurtleConfig, run_turtle  # noqa: E402
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
    parser.add_argument("--lookback-sessions", type=int, nargs="+",
                        default=(63, 126, 252))
    parser.add_argument("--bars-per-session", type=int, default=13)
    parser.add_argument("--cost-bp", type=float, default=5.0)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/capacity_priority.json"))
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


def strength_series(bars: list[Bar], lookback_bars: int) -> dict[str, float]:
    """Trailing return ending one bar before each timestamp."""
    out: dict[str, float] = {}
    closes = [b.close for b in bars]
    for index in range(lookback_bars + 1, len(bars)):
        past = closes[index - 1 - lookback_bars]
        if past > 0:
            out[bars[index].timestamp] = closes[index - 1] / past - 1.0
    return out


def cap(trades, limit, rng, order: str, scores):
    shuffled = list(trades)
    rng.shuffle(shuffled)
    if order == "random":
        key = lambda t: t["entry"]  # noqa: E731
    else:
        sign = -1.0 if order == "strongest" else 1.0
        key = lambda t: (t["entry"], sign * scores.get(  # noqa: E731
            (t["ticker"], t["entry"]), 0.0))
    live, taken, refused = [], [], 0
    for trade in sorted(shuffled, key=key):
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


def evaluate(pooled, closes_by_ticker, calendar, args, order, scores):
    capped = [cap(pooled, args.max_positions, random.Random(s), order, scores)
              for s in range(args.trials)]
    marks = [marked_map(t, closes_by_ticker) for t, _ in capped]
    series = [(sorted(m), [m[d] for d in sorted(m)]) for m in marks]
    risk = solve_risk(series, args.target_dd)
    spread = sorted(sharpe([m.get(d, 0.0) for d in calendar]) for m in marks)
    by_year = {}
    for year in sorted({d[:4] for d in calendar}):
        window = [d for d in calendar if d[:4] == year]
        if len(window) < 60:
            continue
        scored = [sharpe([m.get(d, 0.0) for d in window]) for m in marks[:8]]
        scored = [s for s in scored if s == s]
        if scored:
            by_year[year] = statistics.median(scored)
    live = list(by_year.values())
    return {
        "sharpe": statistics.median(spread),
        "sharpe_p05": spread[int(.05 * len(spread))],
        "sharpe_p95": spread[min(int(.95 * len(spread)), len(spread) - 1)],
        "cagr": statistics.median(path_metrics(d, v, risk)[2] for d, v in series),
        "taken_median": statistics.median(len(t) for t, _ in capped),
        "refused_median": statistics.median(r for _, r in capped),
        "years": by_year,
        "negative_years": sum(1 for v in live if v < 0),
        "year_count": len(live),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    book = load(args)
    closes_by_ticker = {t: {b.timestamp[:10]: b.close for b in bars}
                        for t, bars in book.items()}
    calendar = sorted({d for c in closes_by_ticker.values() for d in c})
    config = TurtleConfig(**FIXED, directions=(1,),
                          round_trip_cost=args.cost_bp / 10_000)
    pooled = []
    for ticker, bars in book.items():
        trades, _ = run_turtle(bars, config=config)
        pooled.extend({"ticker": ticker, "entry": t.entry_timestamp,
                       "exit": t.exit_timestamp, "r": t.net_r,
                       "dir": t.direction, "units": t.unit_entries}
                      for t in trades)
    print(f"{len(book)} instruments, {args.minutes}-minute bars, "
          f"{calendar[0]} -> {calendar[-1]}")
    print(f"{len(pooled):,} breakouts offered to {args.max_positions} slots "
          f"at {args.cost_bp:g}bp\n", flush=True)

    report = {}
    for sessions in args.lookback_sessions:
        lookback = sessions * args.bars_per_session
        scores: dict[tuple[str, str], float] = {}
        for ticker, bars in book.items():
            for timestamp, value in strength_series(bars, lookback).items():
                scores[(ticker, timestamp)] = value
        covered = sum(1 for t in pooled if (t["ticker"], t["entry"]) in scores)
        print(f"=== relative strength over {sessions} sessions "
              f"({lookback:,} bars); {covered/len(pooled):.0%} of breakouts scored ===")
        print(f"  {'tie-break':16s} {'Sharpe':>7s} {'[5-95%]':>16s} {'CAGR':>8s} "
              f"{'taken':>7s} {'refused':>8s} {'neg yrs':>8s}")
        for order in ("random", "strongest", "weakest"):
            result = evaluate(pooled, closes_by_ticker, calendar, args, order, scores)
            report[f"{sessions}|{order}"] = {"lookback_sessions": sessions,
                                             "order": order, **result}
            print(f"  {order:16s} {result['sharpe']:>7.2f} "
                  f"{('[%.2f-%.2f]' % (result['sharpe_p05'], result['sharpe_p95'])):>16s} "
                  f"{result['cagr']:>8.1%} {result['taken_median']:>7,.0f} "
                  f"{result['refused_median']:>8,.0f} "
                  f"{result['negative_years']:>3d}/{result['year_count']:<3d}", flush=True)
        strong = report[f"{sessions}|strongest"]["sharpe"]
        weak = report[f"{sessions}|weakest"]["sharpe"]
        rand = report[f"{sessions}|random"]["sharpe"]
        verdict = ("no information: reversal matches or beats it"
                   if weak >= strong - 0.02 else
                   f"strongest beats weakest by {strong - weak:+.2f} "
                   f"and random by {strong - rand:+.2f}")
        print(f"  -> {verdict}\n", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
