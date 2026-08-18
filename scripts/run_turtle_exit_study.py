"""Sweep leakage-safe exits while holding Turtle System 2 entries fixed."""

from __future__ import annotations

import argparse
import csv
import math
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import TurtleConfig, run_turtle  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402


VARIANTS = (
    ("Channel 10", dict(exit_window=10)),
    ("Channel 20 (baseline)", dict(exit_window=20)),
    ("Channel 30", dict(exit_window=30)),
    ("Channel 40", dict(exit_window=40)),
    ("Channel 50", dict(exit_window=50)),
    ("Chandelier 2N", dict(use_channel_exit=False, chandelier_atr=2.0)),
    ("Chandelier 3N", dict(use_channel_exit=False, chandelier_atr=3.0)),
    ("Chandelier 4N", dict(use_channel_exit=False, chandelier_atr=4.0)),
    ("Chandelier 5N", dict(use_channel_exit=False, chandelier_atr=5.0)),
    ("Channel 20 + chandelier 3N", dict(chandelier_atr=3.0)),
    ("Channel 20 + chandelier 4N", dict(chandelier_atr=4.0)),
    ("Channel 20 + break-even at 1N", dict(breakeven_trigger_n=1.0)),
    ("Channel 20 + break-even at 2N", dict(breakeven_trigger_n=2.0)),
    ("Channel 20 + initial stop 3N", dict(stop_atr=3.0)),
    ("Channel 20 + initial stop 4N", dict(stop_atr=4.0)),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--split", default="2023-01-01")
    parser.add_argument("--min-sessions", type=int, default=500)
    parser.add_argument("--outdir", type=Path, default=Path("out/strategy"))
    return parser.parse_args(argv)


def load_bars(path: Path, ticker: str, start: str, end: str):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? "
            "AND frequency='5min' AND ts>=? AND ts<? ORDER BY ts",
            (ticker, start, end),
        ).fetchall()
    finally:
        connection.close()
    return [Bar(*row) for row in rows]


def names(path: Path):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return [row[0] for row in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker"
        )]
    finally:
        connection.close()


def metrics(rows):
    if not rows:
        return dict(n=0, win=math.nan, pf=math.nan, mean=math.nan, median=math.nan,
                    total=0.0, loss=math.nan, top5=math.nan, held=math.nan)
    values = [row.net_r for row in rows]
    gains = [value for value in values if value > 0]
    losses = [value for value in values if value < 0]
    tail_count = max(1, math.ceil(len(values) * 0.05))
    top = sorted(values, reverse=True)[:tail_count]
    return {
        "n": len(rows), "win": len(gains) / len(rows),
        "pf": sum(gains) / -sum(losses) if losses else math.inf,
        "mean": statistics.mean(values), "median": statistics.median(values),
        "total": sum(values), "loss": statistics.mean(losses) if losses else math.nan,
        "top5": sum(top) / sum(gains) if gains else math.nan,
        "held": statistics.mean(row.bars_held for row in rows),
    }


def fmt(value, places=2):
    if math.isnan(value):
        return "--"
    if math.isinf(value):
        return "inf"
    return f"{value:.{places}f}"


def main(argv=None) -> int:
    args = parse_args(argv)
    splits = load_splits(args.db)
    universe = []
    for ticker in names(args.db):
        source = adjust_bars(
            load_bars(args.db, ticker, args.start, args.end), splits.get(ticker, [])
        )
        if len({bar.timestamp[:10] for bar in source}) >= args.min_sessions:
            universe.append((ticker, source))
    series = {
        "daily": [(ticker, resample_regular_session(rows, minutes=390))
                  for ticker, rows in universe],
        "30-minute": [(ticker, resample_regular_session(rows, minutes=30))
                      for ticker, rows in universe],
    }
    print(f"{len(universe)} instruments", flush=True)

    results = {}
    ledger = []
    base = dict(entry_window=55, exit_window=20, atr_window=20,
                skip_after_winner=False)
    for interval, book in series.items():
        for label, overrides in VARIANTS:
            config = TurtleConfig(**{**base, **overrides})
            rows = []
            for ticker, bars in book:
                trades, _ = run_turtle(bars, config=config)
                rows.extend(trades)
                ledger.extend({"interval": interval, "variant": label, "ticker": ticker,
                               **asdict(trade)} for trade in trades)
            results[(interval, label)] = rows
            print(f"  {interval:10s} {label:34s} {len(rows):7,d}", flush=True)

    lines = [
        "# Turtle System 2 exit study", "",
        f"Universe: **{len(universe)}** instruments, **{args.start}** through "
        f"**{args.end}**; split at **{args.split}**. Every row holds the original "
        "55-bar stop entry, prior-bar Wilder N, half-N pyramiding to four units, "
        "and 2 bp/unit constant. Only the exit changes.", "",
        "Chandelier and break-even levels are calculated after a bar completes and become "
        "active on the following bar. This prevents a bar's final high/low from creating a "
        "stop that is then claimed to have filled earlier inside the same bar.", "",
        "## Chronological results", "",
        "| Interval | Exit | Period | Side | Trades | Win | PF | Mean R | Median R | "
        "Avg loss R | Total R | Top 5% / gains | Bars held |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for interval in series:
        for label, _ in VARIANTS:
            all_rows = results[(interval, label)]
            for period, period_rows in (
                ("train", [row for row in all_rows if row.entry_timestamp[:10] < args.split]),
                ("test", [row for row in all_rows if row.entry_timestamp[:10] >= args.split]),
            ):
                for side, rows in (
                    ("both", period_rows),
                    ("long", [row for row in period_rows if row.direction > 0]),
                    ("short", [row for row in period_rows if row.direction < 0]),
                ):
                    item = metrics(rows)
                    lines.append(
                        f"| {interval} | {label} | {period} | {side} | {item['n']:,} | "
                        f"{fmt(item['win'] * 100, 1)}% | {fmt(item['pf'])} | "
                        f"{fmt(item['mean'], 3)} | {fmt(item['median'], 3)} | "
                        f"{fmt(item['loss'], 3)} | {item['total']:+,.0f} | "
                        f"{fmt(item['top5'] * 100, 1)}% | {fmt(item['held'], 1)} |"
                    )

    lines += ["", "## Exit attribution", "",
              "| Interval | Exit | Reason | Trades | Share |",
              "|---|---|---|---:|---:|"]
    for interval in series:
        for label, _ in VARIANTS:
            rows = results[(interval, label)]
            counts = Counter(row.exit_reason for row in rows)
            for reason, count in sorted(counts.items()):
                lines.append(
                    f"| {interval} | {label} | {reason} | {count:,} | "
                    f"{count / len(rows):.1%} |"
                )

    lines += ["", "## Cross-symbol robustness", "",
              "A ticker counts as improved only if its filtered exit PF exceeds its own "
              "Channel-20 baseline in the same period.", "",
              "| Interval | Exit | Period | Common tickers | PF improved |",
              "|---|---|---|---:|---:|"]
    for interval in series:
        baseline = results[(interval, "Channel 20 (baseline)")]
        for label, _ in VARIANTS:
            if label == "Channel 20 (baseline)":
                continue
            candidate = results[(interval, label)]
            for period, before in (("train", True), ("test", False)):
                grouped = {}
                for key, rows in (("base", baseline), ("candidate", candidate)):
                    by_ticker = defaultdict(list)
                    for ticker, trade in (
                        (row["ticker"], row) for row in ledger
                        if row["interval"] == interval and row["variant"] == (
                            "Channel 20 (baseline)" if key == "base" else label
                        ) and (row["entry_timestamp"][:10] < args.split) == before
                    ):
                        by_ticker[ticker].append(trade["net_r"])
                    grouped[key] = by_ticker
                common = sorted(set(grouped["base"]) & set(grouped["candidate"]))
                better = 0
                for ticker in common:
                    base_item = metrics_from_values(grouped["base"][ticker])
                    candidate_item = metrics_from_values(grouped["candidate"][ticker])
                    better += candidate_item > base_item
                lines.append(
                    f"| {interval} | {label} | {period} | {len(common)} | "
                    f"{better}/{len(common)} |"
                )

    args.outdir.mkdir(parents=True, exist_ok=True)
    report = args.outdir / "turtle_exit_study.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    csv_path = args.outdir / "turtle_exit_trades.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(ledger[0]) if ledger else ["interval"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ledger)
    print(f"wrote {report} and {csv_path}")
    return 0


def metrics_from_values(values):
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    return gains / losses if losses else math.inf


if __name__ == "__main__":
    raise SystemExit(main())
