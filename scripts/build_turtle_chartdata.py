"""Compact chart data for the Turtle report artifact.

Emits one small JSON: equity paths on a shared calendar, drawdown, yearly
results, the direction split, and the interval comparison.  Curves are thinned
to roughly weekly points because the page inlines them.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sqlite3
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

RISK = 0.0020  # the "2x" case: 0.20% of equity per R


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--trades", type=Path,
                        default=Path("out/strategy/turtle_trades.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("out/strategy/turtle_intervals.md"))
    parser.add_argument("--start", default="2017-03-21")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/turtle_chartdata.json"))
    return parser.parse_args(argv)


def load_trades(path: Path):
    with path.open(encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["interval"] == "daily" and row["system"] == "System 2 (55/20)"
        ]
    for row in rows:
        row["net_r"] = float(row["net_r"])
        row["direction"] = int(row["direction"])
        row["units"] = int(row["units"])
    return rows


def cap(rows, limit, rng=None):
    shuffled = list(rows)
    if rng is not None:
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


def median_trial(rows, limit, trials=200):
    """Most entries share a date, so which one gets a free slot is arbitrary.

    Charting a single arbitrary ordering would overstate the precision of the
    curve, so the run whose total sits at the median of many shuffles is used.
    """
    runs = []
    for seed in range(trials):
        sample = cap(rows, limit, random.Random(seed))
        runs.append((sum(row["net_r"] for row in sample), seed, sample))
    runs.sort(key=lambda item: item[0])
    return runs[len(runs) // 2][2]


def curve(rows, calendar, risk):
    by_day = defaultdict(float)
    for row in rows:
        by_day[row["exit_timestamp"][:10]] += row["net_r"]
    equity, peak = 1.0, 1.0
    values, draws = [], []
    for day in calendar:
        equity *= max(0.0, 1.0 + risk * by_day.get(day, 0.0))
        peak = max(peak, equity)
        values.append(equity)
        draws.append(equity / peak - 1.0)
    return values, draws


def benchmark(db: Path, ticker: str, start: str, end: str):
    splits = load_splits(db)
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
        "frequency='5min' AND ts>=? AND ts<? ORDER BY ts", (ticker, start, end),
    ).fetchall()
    connection.close()
    daily = resample_regular_session(
        adjust_bars([Bar(*row) for row in rows], splits.get(ticker, [])), minutes=390
    )
    base = daily[0].open
    return {bar.timestamp[:10]: bar.close / base for bar in daily}


def interval_table(path: Path):
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or "System 2 (55/20)" not in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 11 or cells[0] not in {
            "daily", "30-minute", "15-minute", "5-minute"
        }:
            continue
        result.append({
            "interval": cells[0],
            "trades": int(cells[2].replace(",", "")),
            "win": float(cells[4].rstrip("%")),
            "pf": float(cells[5]),
            "mean_r": float(cells[6]),
            "cost_r": float(cells[8]),
            "cost_share": float(cells[9].rstrip("%")),
        })
    return result


def main(argv=None):
    args = parse_args(argv)
    trades = load_trades(args.trades)
    marks = benchmark(args.db, "SPY", args.start, args.end)
    calendar = sorted(day for day in marks if args.start <= day <= args.end)

    both = median_trial(trades, args.max_positions)
    longs = median_trial([t for t in trades if t["direction"] > 0], args.max_positions)
    shorts = median_trial([t for t in trades if t["direction"] < 0], args.max_positions)

    both_eq, both_dd = curve(both, calendar, RISK)
    long_eq, long_dd = curve(longs, calendar, RISK)

    step = max(1, len(calendar) // 320)
    keep = list(range(0, len(calendar), step))
    if keep[-1] != len(calendar) - 1:
        keep.append(len(calendar) - 1)

    yearly = defaultdict(float)
    for row in both:
        yearly[row["exit_timestamp"][:4]] += row["net_r"]

    def side(rows):
        wins = [r["net_r"] for r in rows if r["net_r"] > 0]
        losses = [r["net_r"] for r in rows if r["net_r"] <= 0]
        return {
            "trades": len(rows), "wins": len(wins), "losses": len(losses),
            "win_rate": len(wins) / len(rows) if rows else 0.0,
            "pf": (sum(wins) / -sum(losses)) if losses and sum(losses) else None,
            "total_r": sum(r["net_r"] for r in rows),
            "mean_win": sum(wins) / len(wins) if wins else 0.0,
            "mean_loss": sum(losses) / len(losses) if losses else 0.0,
        }

    buckets = [(-99, -8), (-8, -5), (-5, -3), (-3, -1), (-1, 0),
               (0, 5), (5, 15), (15, 40), (40, 9999)]
    histogram = []
    for low, high in buckets:
        histogram.append({
            "low": low, "high": high,
            "count": sum(1 for r in both if low <= r["net_r"] < high),
        })

    payload = {
        "risk_per_r": RISK,
        "max_positions": args.max_positions,
        "start": calendar[0], "end": calendar[-1],
        "dates": [calendar[i] for i in keep],
        "both_equity": [round(both_eq[i], 4) for i in keep],
        "long_equity": [round(long_eq[i], 4) for i in keep],
        "spy_equity": [round(marks[calendar[i]], 4) for i in keep],
        "both_drawdown": [round(both_dd[i], 4) for i in keep],
        "long_drawdown": [round(long_dd[i], 4) for i in keep],
        "yearly_r": {k: round(yearly[k], 1) for k in sorted(yearly)},
        "sides": {"both": side(both), "long": side(longs), "short": side(shorts)},
        "histogram": histogram,
        "intervals": interval_table(args.report),
        "units_per_trade": {
            str(n): sum(1 for r in both if r["units"] == n) for n in (1, 2, 3, 4)
        },
        "exit_reasons": {
            reason: sum(1 for r in both if r["exit_reason"] == reason)
            for reason in {r["exit_reason"] for r in both}
        },
    }
    args.out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {args.out}  ({len(payload['dates'])} points, "
          f"{payload['sides']['both']['trades']} trades)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
