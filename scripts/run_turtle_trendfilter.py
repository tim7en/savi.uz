"""Does a higher-timeframe trend filter earn its place, or just delete shorts?

In a nine-year equity bull market almost every instrument sits above its long
mean almost all the time, so a trend filter mostly refuses short signals.  That
alone would improve results here, but it is not evidence the filter selects
better trades -- it is the long-only result arriving by a different route.

The control is the long side.  If the filter is doing real work it should also
improve trades it does not remove: filtered longs versus unfiltered longs.  If
those two are the same, the filter is an expensive way to stop shorting.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sqlite3
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import TurtleConfig, run_turtle  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

BASE = dict(entry_window=55, exit_window=20, atr_window=20, skip_after_winner=False)

VARIANTS = [
    ("Unfiltered, both sides", dict(BASE)),
    ("Trend filter SMA-200", dict(BASE, trend_filter="sma", trend_window=200)),
    ("Trend filter SMA-100", dict(BASE, trend_filter="sma", trend_window=100)),
    ("Trend filter SMA-50", dict(BASE, trend_filter="sma", trend_window=50)),
    ("Long only, unfiltered", dict(BASE, directions=(1,))),
    ("Long only + SMA-200", dict(BASE, directions=(1,), trend_filter="sma",
                                 trend_window=200)),
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--split", default="2023-01-01")
    parser.add_argument("--min-sessions", type=int, default=500)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--outdir", type=Path, default=Path("out/strategy"))
    return parser.parse_args(argv)


def universe(db: Path, start: str, end: str, min_sessions: int):
    splits = load_splits(db)
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    names = [row[0] for row in connection.execute(
        "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker")]
    out = []
    for ticker in names:
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='5min' AND ts>=? AND ts<? ORDER BY ts", (ticker, start, end),
        ).fetchall()
        if not rows:
            continue
        bars = adjust_bars([Bar(*row) for row in rows], splits.get(ticker, []))
        daily = resample_regular_session(bars, minutes=390)
        if len(daily) < min_sessions:
            continue
        out.append((ticker, daily))
    connection.close()
    return out


def cap(rows, limit, rng):
    shuffled = list(rows)
    rng.shuffle(shuffled)
    ordered = sorted(shuffled, key=lambda row: row["entry_timestamp"])
    live, taken = [], []
    for row in ordered:
        live = [x for x in live if x["exit_timestamp"] > row["entry_timestamp"]]
        if len(live) >= limit:
            continue
        live.append(row)
        taken.append(row)
    return taken


def equity(rows, risk=0.0020):
    by_day = defaultdict(float)
    for row in rows:
        by_day[row["exit_timestamp"][:10]] += row["net_r"]
    value = peak = 1.0
    worst = 0.0
    for day in sorted(by_day):
        value *= max(0.0, 1.0 + risk * by_day[day])
        peak = max(peak, value)
        worst = min(worst, value / peak - 1.0)
    return value, worst


def profit_factor(values):
    gains = sum(v for v in values if v > 0)
    losses = -sum(v for v in values if v < 0)
    return gains / losses if losses else float("inf")


def main(argv=None):
    args = parse_args(argv)
    book = universe(args.db, args.start, args.end, args.min_sessions)
    print(f"{len(book)} instruments", flush=True)

    results = {}
    for label, overrides in VARIANTS:
        config = TurtleConfig(**overrides)
        pooled, breakouts, refused = [], 0, 0
        for _, daily in book:
            trades, audit = run_turtle(daily, config=config)
            pooled.extend(asdict(t) for t in trades)
            breakouts += audit.breakouts
            refused += audit.skipped_against_trend
        results[label] = {"trades": pooled, "breakouts": breakouts, "refused": refused}
        print(f"  {label:26s} {len(pooled):5d} trades "
              f"({refused:,} refused by trend)", flush=True)

    report = {}
    for label, blob in results.items():
        rows = blob["trades"]
        stats = {"signals": len(rows), "refused_by_trend": blob["refused"]}
        finals, dds, totals, pfs = [], [], [], []
        for seed in range(args.trials):
            taken = cap(rows, args.max_positions, random.Random(seed))
            value, worst = equity(taken)
            finals.append(value)
            dds.append(worst)
            totals.append(sum(r["net_r"] for r in taken))
            pfs.append(profit_factor([r["net_r"] for r in taken]))
        pick = lambda xs, f: sorted(xs)[int(f * len(xs))]
        longs = [r for r in rows if r["direction"] > 0]
        shorts = [r for r in rows if r["direction"] < 0]
        stats.update({
            "final_median": pick(finals, .5), "final_p05": pick(finals, .05),
            "final_p95": pick(finals, .95),
            "dd_median": pick(dds, .5), "dd_worst": min(dds),
            "total_r_median": pick(totals, .5), "pf_median": pick(pfs, .5),
            "long_trades": len(longs), "short_trades": len(shorts),
            "long_r": sum(r["net_r"] for r in longs),
            "short_r": sum(r["net_r"] for r in shorts),
            "long_pf": profit_factor([r["net_r"] for r in longs]) if longs else None,
            "long_win": (sum(1 for r in longs if r["net_r"] > 0) / len(longs))
                        if longs else None,
        })
        report[label] = stats

    lines = [f"  {'variant':26s} {'signals':>8s} {'L/S':>10s} {'PF':>5s} "
             f"{'$1000@0.2%':>11s} {'maxDD':>7s}"]
    for label, s in report.items():
        lines.append(
            f"  {label:26s} {s['signals']:8,d} "
            f"{s['long_trades']:>4d}/{s['short_trades']:<5d} {s['pf_median']:5.2f} "
            f"${1000 * s['final_median']:>10,.0f} {s['dd_median']:>7.1%}")
    print("\n" + "\n".join(lines))

    a = report["Long only, unfiltered"]
    b = report["Long only + SMA-200"]
    print("\n  CONTROL - does the filter improve the side it does not delete?")
    print(f"    long only, unfiltered : {a['long_trades']:4d} trades  "
          f"PF {a['long_pf']:.2f}  win {a['long_win']:.1%}  {a['long_r']:+.0f}R")
    print(f"    long only + SMA-200   : {b['long_trades']:4d} trades  "
          f"PF {b['long_pf']:.2f}  win {b['long_win']:.1%}  {b['long_r']:+.0f}R")

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "turtle_trendfilter.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    csv_path = args.outdir / "turtle_trendfilter_trades.csv"
    rows = [{"variant": label, **t}
            for label, blob in results.items() for t in blob["trades"]]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  wrote {out} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
