"""Half size below a moving average, full size above it.

The higher-timeframe trend filter was already tested and rejected in this
programme, but as a *gate*: it removed trades, and it removed +1,205R of winning
longs against only -281R of losing shorts, with windows non-monotonic enough
that SMA-100 finished below no filter at all. Sizing is a different instrument.
It removes nothing and only scales, so the earlier rejection does not settle it.

What does bear on it is the reserve result. Constant exposure at 70%, 85% and
100% produced identical Sharpe once every arm was matched to the same drawdown,
because an exposure level is a choice of leverage rather than a strategy. A
sizing rule can therefore only earn its place through the *timing* of its
changes -- whether the regime label knows something about what comes next.

Four arms decide that:

* full size always -- the banked book;
* constant average size -- the same mean exposure with no conditioning, which is
  what the rule has to beat to be worth its machinery;
* half below the average, full above -- the proposal;
* full below, half above -- its exact reversal. A rule that performs no better
  than its opposite carries no information, whatever the headline gap.

Every arm is scaled to the same median drawdown before its return is read, and
the average is swept over several windows because a result that lives at one
window length and dies at the next is a fit rather than a finding.
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
    parser.add_argument("--ma-sessions", type=int, nargs="+", default=(50, 100, 200))
    parser.add_argument("--bars-per-session", type=int, default=13)
    parser.add_argument("--reduced", type=float, default=0.5)
    parser.add_argument("--cost-bp", type=float, default=5.0)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/ma_regime_sizing.json"))
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


def above_ma(bars: list[Bar], window: int) -> dict[str, bool]:
    """Was the close above its trailing mean at the bar before each timestamp?"""
    closes = [b.close for b in bars]
    out: dict[str, bool] = {}
    running = 0.0
    for index, close in enumerate(closes):
        running += close
        if index >= window:
            running -= closes[index - window]
            # Compare the previous close against the mean ending there, so the
            # label is knowable before the bar it is attached to opens.
            out[bars[index].timestamp] = closes[index - 1] > running / window
    return out


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


def marked_map(taken, closes_by_ticker, weight):
    by_day = defaultdict(float)
    for trade in taken:
        scale = weight(trade)
        if scale == 0.0:
            continue
        closes = closes_by_ticker[trade["ticker"]]
        entry_day, exit_day = trade["entry"][:10], trade["exit"][:10]
        previous = 0.0
        for day in (d for d in closes if entry_day <= d < exit_day):
            live = [u for u in trade["units"] if u.timestamp[:10] <= day]
            if not live:
                continue
            open_r = sum(trade["dir"] * (closes[day] - u.price) / u.n for u in live)
            by_day[day] += (open_r - previous) * scale
            previous = open_r
        by_day[exit_day] += (trade["r"] - previous) * scale
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


def assess(caps, closes_by_ticker, weight, args):
    marks = [marked_map(t, closes_by_ticker, weight) for t in caps]
    series = [(sorted(m), [m[d] for d in sorted(m)]) for m in marks]
    risk = solve_risk(series, args.target_dd)
    cagrs, sharpes = [], []
    for days, values in series:
        nav, _ = path(values, risk)
        years = (date.fromisoformat(days[-1])
                 - date.fromisoformat(days[0])).days / 365.25
        cagrs.append((nav / 1000.0) ** (1 / years) - 1 if nav > 0 else -1.0)
        sharpes.append(sharpe([v * risk for v in values]))
    return {"sharpe": statistics.median(sharpes),
            "cagr": statistics.median(cagrs),
            "risk_per_r": risk}


def main(argv=None) -> int:
    args = parse_args(argv)
    book = load(args)
    closes_by_ticker = {t: {b.timestamp[:10]: b.close for b in bars}
                        for t, bars in book.items()}
    config = TurtleConfig(**FIXED, directions=(1,),
                          round_trip_cost=args.cost_bp / 10_000)
    pooled = []
    for ticker, bars in book.items():
        trades, _ = run_turtle(bars, config=config)
        pooled.extend({"ticker": ticker, "entry": t.entry_timestamp,
                       "exit": t.exit_timestamp, "r": t.net_r,
                       "dir": t.direction, "units": t.unit_entries}
                      for t in trades)
    caps = [cap(pooled, args.max_positions, random.Random(s))
            for s in range(args.trials)]
    print(f"{len(book)} instruments, {len(pooled):,} breakouts, "
          f"{args.cost_bp:g}bp, all arms matched to "
          f"{args.target_dd:.0%} median drawdown\n", flush=True)

    report = {}
    low = args.reduced
    for sessions in args.ma_sessions:
        window = sessions * args.bars_per_session
        labels: dict[tuple[str, str], bool] = {}
        for ticker, bars in book.items():
            for timestamp, flag in above_ma(bars, window).items():
                labels[(ticker, timestamp)] = flag
        share = statistics.fmean(
            1.0 if labels.get((t["ticker"], t["entry"]), True) else 0.0
            for t in pooled)
        average = low + (1.0 - low) * share
        arms = {
            "full size always": lambda t: 1.0,
            f"constant {average:.2f}x (same mean exposure)": lambda t: average,
            f"{low:g}x below MA, full above": lambda t: (
                1.0 if labels.get((t["ticker"], t["entry"]), True) else low),
            f"full below MA, {low:g}x above (reversal)": lambda t: (
                low if labels.get((t["ticker"], t["entry"]), True) else 1.0),
        }
        print(f"=== {sessions}-session moving average ({window:,} bars); "
              f"{share:.0%} of breakouts fire above it ===")
        print(f"  {'arm':44s} {'Sharpe':>7s} {'CAGR':>8s} {'risk/R':>8s}")
        for label, weight in arms.items():
            result = assess(caps, closes_by_ticker, weight, args)
            report[f"{sessions}|{label}"] = {"ma_sessions": sessions,
                                             "arm": label, **result}
            print(f"  {label:44s} {result['sharpe']:>7.2f} "
                  f"{result['cagr']:>8.1%} {result['risk_per_r']*10_000:>6.1f}bp",
                  flush=True)
        proposal = report[f"{sessions}|{low:g}x below MA, full above"]["sharpe"]
        control = report[f"{sessions}|constant {average:.2f}x "
                         f"(same mean exposure)"]["sharpe"]
        reverse = report[f"{sessions}|full below MA, {low:g}x above "
                         f"(reversal)"]["sharpe"]
        verdict = ("no information: reversal matches or beats it"
                   if reverse >= proposal - 0.02 else
                   f"beats reversal {proposal - reverse:+.2f}, "
                   f"constant {proposal - control:+.2f}")
        print(f"  -> {verdict}\n", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
