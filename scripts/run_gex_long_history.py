"""Re-test the GEX risk overlay on fifteen years of open dealer-gamma data.

The overlay was validated on 250 sessions of a purpose-computed SPY gamma
series, giving roughly 29 independent regime episodes and no crisis.  The
SqueezeMetrics DIX/GEX file covers 2011-2026, which includes February 2018,
Q4 2018, COVID and the 2022 bear market.

Two things have to be handled for the series to be usable:

* **Calibration, not sign.**  The two gamma series rank alike (Spearman +0.81)
  but disagree on where zero sits: the computed SPY series prints negative on
  49% of sessions, the open series on 4%.  A raw sign threshold is therefore not
  portable.  The regime is defined by *percentile rank within a trailing
  window*, which is comparable across both.

* **Look-ahead, twice over.**  Gamma for a date is published after that date's
  close, so it is lagged one session.  The percentile it is ranked against is
  also computed only from earlier sessions -- ranking against the full-sample
  distribution would quietly use the future to decide what counts as "low"
  today.
"""

from __future__ import annotations

import argparse
import bisect
import csv
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

ETFS = {"SPY", "QQQ", "IWM", "GLD", "EWJ", "EWT", "EWY", "KWEB"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--gex", type=Path, default=Path("data/DIX.csv"))
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--window", type=int, default=252,
                        help="trailing sessions the percentile is measured against")
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--risk", type=float, default=0.0005)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/gex_long_history.json"))
    return parser.parse_args(argv)


def load_gex(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = float(row["gex"])
            if value == 0.0:            # a single missing session prints as 0
                continue
            rows.append((row["date"], value, float(row["dix"])))
    rows.sort()
    return rows


def trailing_percentile(rows, window):
    """Rank each session's gamma within the `window` sessions before it.

    The value for session i is ranked against i-window .. i-1 only, so nothing
    from the present or future decides what counts as a low reading.
    """
    result = {}
    values = [value for _, value, _ in rows]
    for index in range(window, len(rows)):
        history = sorted(values[index - window:index])
        position = bisect.bisect_left(history, values[index])
        result[rows[index][0]] = position / len(history)
    return result


def lag_one(flags, sessions):
    ordered = sorted(sessions)
    return {day: flags.get(ordered[i - 1]) if i else None
            for i, day in enumerate(ordered)}


def build_trades(args):
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    names = [r[0] for r in connection.execute(
        "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker")]
    config = TurtleConfig(entry_window=55, exit_window=20, atr_window=20,
                          skip_after_winner=False, directions=(1,))
    trades = []
    for ticker in names:
        if ticker in ETFS:
            continue
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='5min' AND ts>=? AND ts<? ORDER BY ts",
            (ticker, args.start, args.end)).fetchall()
        if not rows:
            continue
        series = resample_regular_session(
            adjust_bars([Bar(*r) for r in rows], splits.get(ticker, [])),
            minutes=args.minutes)
        for trade in run_turtle(series, config=config)[0]:
            trades.append({"day": trade.entry_timestamp[:10],
                           "entry": trade.entry_timestamp,
                           "exit": trade.exit_timestamp, "r": trade.net_r})
    connection.close()
    return trades


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


def walk(taken, days, risk, flag):
    nav, peak, worst = 1000.0, 1000.0, 0.0
    by_day = defaultdict(list)
    for trade in taken:
        by_day[trade["exit"][:10]].append(trade)
    path = []
    for day in days:
        for trade in by_day.get(day, ()):
            size = 0.5 if flag.get(trade["day"]) is True else 1.0
            nav = max(0.0, nav + trade["r"] * risk * size * nav)
        peak = max(peak, nav)
        worst = min(worst, nav / peak - 1.0)
        path.append(nav)
    rets = [path[i] / path[i - 1] - 1.0
            for i in range(1, len(path)) if path[i - 1] > 0]
    sd = statistics.stdev(rets) if len(rets) > 1 else 0.0
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25
    return {
        "final": nav, "maxdd": worst,
        "cagr": (nav / 1000.0) ** (1 / years) - 1 if years > 0 and nav > 0 else -1.0,
        "sharpe": (statistics.mean(rets) / sd * math.sqrt(252.0)) if sd else 0.0,
    }


def episodes(flag, days):
    seq = [flag.get(d) for d in days if flag.get(d) is not None]
    if not seq:
        return 0, 0.0
    runs = 1 + sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    on = sum(1 for v in seq if v)
    return runs, on / len(seq)


def main(argv=None):
    args = parse_args(argv)
    gex = load_gex(args.gex)
    pct = trailing_percentile(gex, args.window)
    sessions = sorted(pct)
    print(f"open gamma series: {len(gex):,} sessions {gex[0][0]} -> {gex[-1][0]}; "
          f"{len(pct):,} carry a trailing-{args.window} percentile")

    trades = build_trades(args)
    days = sorted({t["exit"][:10] for t in trades})
    days = [d for d in days if args.start <= d <= args.end]
    print(f"{len(trades):,} long {args.minutes}-minute stock trades over "
          f"{days[0]} -> {days[-1]} ({len(days):,} sessions)\n")

    raw = {d: v for d, v, _ in gex}
    dix = {d: x for d, _, x in gex}
    definitions = {
        "gamma in bottom 20% (trailing)": {d: pct[d] < 0.20 for d in sessions},
        "gamma in bottom 50% (trailing)": {d: pct[d] < 0.50 for d in sessions},
        "gamma raw negative": {d: raw[d] < 0 for d in sessions},
        "DIX in bottom 20% (trailing)": None,
    }
    dix_rows = [(d, x, 0.0) for d, _, x in gex]
    dix_pct = trailing_percentile(dix_rows, args.window)
    definitions["DIX in bottom 20% (trailing)"] = {
        d: dix_pct[d] < 0.20 for d in sorted(dix_pct)}

    flags = {name: lag_one(f, sessions) for name, f in definitions.items()}
    never = {d: False for d in days}

    results, report = {}, {"trades": len(trades), "sessions": len(days)}
    for name, flag in [("baseline (no overlay)", never)] + list(flags.items()):
        got = defaultdict(list)
        for seed in range(args.trials):
            taken = cap(trades, args.max_positions, random.Random(seed))
            for key, value in walk(taken, days, args.risk, flag).items():
                got[key].append(value)
        results[name] = got
    # persistence-matched null: rotate the best regime, preserving every run length
    best = flags["gamma in bottom 20% (trailing)"]
    keys = sorted(best)
    got = defaultdict(list)
    for seed in range(args.trials):
        taken = cap(trades, args.max_positions, random.Random(seed))
        offset = random.Random(7000 + seed).randrange(200, len(keys) - 200)
        shifted = {keys[i]: best[keys[(i + offset) % len(keys)]]
                   for i in range(len(keys))}
        for key, value in walk(taken, days, args.risk, shifted).items():
            got[key].append(value)
    results["CONTROL: circular-shifted gamma"] = got

    base = results["baseline (no overlay)"]
    pick = lambda xs, f: sorted(xs)[int(f * len(xs))]
    print(f"  {'policy':34s} {'median $':>10s} {'CAGR':>7s} {'maxDD':>8s} "
          f"{'Sharpe':>7s} | {'wins':>8s} {'dCAGR':>8s} | {'episodes':>9s} {'on':>5s}")
    for name, got in results.items():
        wins = sum(1 for a, b in zip(got["final"], base["final"]) if a > b)
        dc = statistics.median([a - b for a, b in zip(got["cagr"], base["cagr"])])
        flag = flags.get(name, never if "baseline" in name else best)
        runs, share = episodes(flag, days)
        print(f"  {name:34s} ${pick(got['final'], .5):>9,.0f} "
              f"{pick(got['cagr'], .5):>7.1%} {pick(got['maxdd'], .5):>8.1%} "
              f"{pick(got['sharpe'], .5):>7.2f} | {wins:>4d}/{args.trials:<3d} "
              f"{dc:>+8.1%} | {runs:>9,d} {share:>5.0%}")
        report[name] = {"median_final": pick(got["final"], .5),
                        "median_cagr": pick(got["cagr"], .5),
                        "median_maxdd": pick(got["maxdd"], .5),
                        "median_sharpe": pick(got["sharpe"], .5),
                        "wins": wins, "d_cagr": dc,
                        "episodes": runs, "share_on": share}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
