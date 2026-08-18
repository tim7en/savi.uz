"""Test volume/profile confirmation on daily Turtle System 2 breakouts."""

from __future__ import annotations

import argparse
import csv
import math
import sqlite3
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle_volume import (  # noqa: E402
    ConfirmedTurtleConfig,
    VolumeFilter,
    build_breakout_volumes,
    build_intraday_breakout_volumes,
    run_volume_turtle,
)
from savi_uz.volume_profile import Bar  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--split", default="2023-01-01")
    parser.add_argument("--min-sessions", type=int, default=500)
    parser.add_argument("--interval", choices=("daily", "5min"), default="daily")
    parser.add_argument("--outdir", type=Path, default=Path("out/strategy"))
    return parser.parse_args(argv)


def load_source(path: Path, ticker: str, start: str, end: str) -> list[Bar]:
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


def ticker_names(path: Path) -> list[str]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return [row[0] for row in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker"
        )]
    finally:
        connection.close()


def stats(rows) -> dict[str, float]:
    if not rows:
        return {"count": 0, "win": math.nan, "pf": math.nan, "mean": math.nan,
                "total": 0.0, "units": 0}
    values = [row.trade.net_r for row in rows]
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    return {
        "count": len(rows),
        "win": sum(value > 0 for value in values) / len(values),
        "pf": gains / losses if losses else math.inf,
        "mean": sum(values) / len(values),
        "total": sum(values),
        "units": sum(row.trade.units for row in rows),
    }


def fmt(value: float, places: int = 2) -> str:
    if math.isnan(value):
        return "—"
    if math.isinf(value):
        return "∞"
    return f"{value:.{places}f}"


def result_row(label: str, period: str, side: str, rows) -> str:
    item = stats(rows)
    return (
        f"| {label} | {period} | {side} | {item['count']:,} | {item['units']:,} | "
        f"{fmt(item['win'] * 100, 1)}% | {fmt(item['pf'])} | "
        f"{fmt(item['mean'], 3)} | {item['total']:+,.1f} |"
    )


def slice_stats(results, label: str, split: str, period: str, side: str = "both"):
    rows = results[label]
    rows = [
        row for row in rows
        if (row.signal_timestamp[:10] < split) == (period == "train")
    ]
    if side == "long":
        rows = [row for row in rows if row.direction > 0]
    elif side == "short":
        rows = [row for row in rows if row.direction < 0]
    return stats(rows)


def flatten(label, row):
    trade = asdict(row.trade)
    return {
        "variant": label,
        "ticker": row.ticker,
        "signal_timestamp": row.signal_timestamp,
        "direction": row.direction,
        "signal_close": row.signal_close,
        "channel_level": row.channel_level,
        "volume_ratio": row.volume_ratio,
        "rising_volume": row.rising_volume,
        "poc": row.poc,
        "value_low": row.value_low,
        "value_high": row.value_high,
        "outside_value": row.outside_value,
        "poc_distance_n": row.poc_distance_n,
        **trade,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    splits = load_splits(args.db)
    book = []
    for ticker in ticker_names(args.db):
        source = adjust_bars(
            load_source(args.db, ticker, args.start, args.end), splits.get(ticker, [])
        )
        if len({row.timestamp[:10] for row in source}) < args.min_sessions:
            continue
        series = (
            resample_regular_session(source, minutes=390)
            if args.interval == "daily" else source
        )
        features = (
            build_breakout_volumes(source)
            if args.interval == "daily" else build_intraday_breakout_volumes(source)
        )
        if features:
            book.append((ticker, series, features))
            print(f"  {ticker:6s} {len(series):7,d} bars  {len(features):7,d} features", flush=True)
    print(f"{len(book)} volume-ready instruments", flush=True)

    base = ConfirmedTurtleConfig()
    combined = VolumeFilter(volume_floor=1.25, require_outside_value=True)
    variants = [
        ("Matched close-confirmed control", VolumeFilter(), base),
        ("Outside prior value area", VolumeFilter(require_outside_value=True), base),
        ("RVOL >= 1.0", VolumeFilter(volume_floor=1.0), base),
        ("RVOL >= 1.25", VolumeFilter(volume_floor=1.25), base),
        ("RVOL >= 1.5", VolumeFilter(volume_floor=1.5), base),
        ("RVOL >= 1.0 and rising", VolumeFilter(
            volume_floor=1.0, require_rising_volume=True
        ), base),
        ("Outside value + RVOL >= 1.0", VolumeFilter(
            volume_floor=1.0, require_outside_value=True
        ), base),
        ("Outside value + RVOL >= 1.25", combined, base),
        ("POC distance >= 1N", VolumeFilter(minimum_poc_distance_n=1.0), base),
        ("POC >= 1N + outside + RVOL >= 1.25", VolumeFilter(
            volume_floor=1.25, require_outside_value=True,
            minimum_poc_distance_n=1.0,
        ), base),
        ("Long only control", VolumeFilter(), replace(base, directions=(1,))),
        ("Long only combined", combined, replace(base, directions=(1,))),
        ("Short only control", VolumeFilter(), replace(base, directions=(-1,))),
        ("Short only combined", combined, replace(base, directions=(-1,))),
    ]

    results = {}
    audits = {}
    for label, rule, config in variants:
        pooled = []
        counts = {"confirmed": 0, "missing": 0, "rejected": 0, "small_n": 0}
        for ticker, series, features in book:
            trades, audit = run_volume_turtle(
                ticker, series, features, rule=rule, config=config
            )
            pooled.extend(trades)
            counts["confirmed"] += audit.confirmed_breakouts
            counts["missing"] += audit.missing_volume_profile
            counts["rejected"] += audit.rejected_by_filter
            counts["small_n"] += audit.skipped_small_n
        results[label], audits[label] = pooled, counts
        print(f"  {label:40s} {len(pooled):5,d} trades", flush=True)

    lines = [
        "# Volume-confirmed Turtle System 2", "",
        f"Universe: **{len(book)}** volume-ready instruments, **{args.start}** through "
        f"**{args.end}**; split **{args.split}**. Core: {args.interval} 55-bar breakout, 20-bar exit, "
        "2N stop, half-N pyramiding to four units, 2 bp per unit.", "",
        "## Why entry is delayed", "",
        "The published Turtle stop enters during the breakout bar. That bar's closing price and "
        "total volume do not exist yet, so using them to approve that fill would leak. Every row "
        "below instead requires a close beyond the prior 55-bar channel and enters the next "
        f"{args.interval} "
        "open. The matched control uses the same delay and the same valid-volume/profile sample.", "",
        ("Relative volume is the breakout session's completed regular-hours volume divided by the "
         "median of its preceding 20 valid sessions." if args.interval == "daily" else
         "Relative volume is the completed five-minute signal volume divided by the median for "
         "the same time-of-day slot over 20 preceding complete sessions."),
        "The volume profile is built from five completed "
        "sessions strictly before the signal, using 30 price bins and a 70% value area.", "",
        "## Chronological results", "",
        "| Variant | Period | Side | Trades | Units | Win | PF | Mean R | Total R |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    main_labels = [label for label, _, _ in variants[:10]]
    for label in main_labels:
        rows = results[label]
        for period, subset in (
            ("train", [row for row in rows if row.signal_timestamp[:10] < args.split]),
            ("test", [row for row in rows if row.signal_timestamp[:10] >= args.split]),
        ):
            lines.append(result_row(label, period, "both", subset))
            lines.append(result_row(label, period, "long", [r for r in subset if r.direction > 0]))
            lines.append(result_row(label, period, "short", [r for r in subset if r.direction < 0]))

    lines += [
        "", "## Standalone long and short books", "",
        "These reruns prevent one side's open position from blocking signals on the other side.", "",
        "| Variant | Period | Side | Trades | Units | Win | PF | Mean R | Total R |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("Long only control", "Long only combined",
                  "Short only control", "Short only combined"):
        rows = results[label]
        for period, subset in (
            ("train", [row for row in rows if row.signal_timestamp[:10] < args.split]),
            ("test", [row for row in rows if row.signal_timestamp[:10] >= args.split]),
        ):
            lines.append(result_row(label, period, "long" if "Long" in label else "short", subset))

    control = results["Matched close-confirmed control"]
    lines += [
        "", "## Direct volume-burst test inside the matched control", "",
        "| Period | Side | RVOL bucket | Trades | PF | Mean R | Total R |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    buckets = (
        ("<1.0", lambda value: value < 1.0),
        ("1.0-1.25", lambda value: 1.0 <= value < 1.25),
        ("1.25-1.5", lambda value: 1.25 <= value < 1.5),
        (">=1.5", lambda value: value >= 1.5),
    )
    for period, period_rows in (
        ("train", [row for row in control if row.signal_timestamp[:10] < args.split]),
        ("test", [row for row in control if row.signal_timestamp[:10] >= args.split]),
    ):
        for side, side_rows in (
            ("both", period_rows),
            ("long", [row for row in period_rows if row.direction > 0]),
            ("short", [row for row in period_rows if row.direction < 0]),
        ):
            for bucket, predicate in buckets:
                item = stats([row for row in side_rows if predicate(row.volume_ratio)])
                lines.append(
                    f"| {period} | {side} | {bucket} | {item['count']:,} | "
                    f"{fmt(item['pf'])} | {fmt(item['mean'], 3)} | {item['total']:+,.1f} |"
                )

    control_train = slice_stats(
        results, "Matched close-confirmed control", args.split, "train"
    )
    control_test = slice_stats(
        results, "Matched close-confirmed control", args.split, "test"
    )
    outside_train = slice_stats(results, "Outside prior value area", args.split, "train")
    outside_test = slice_stats(results, "Outside prior value area", args.split, "test")
    poc_train = slice_stats(results, "POC distance >= 1N", args.split, "train")
    poc_test = slice_stats(results, "POC distance >= 1N", args.split, "test")
    burst_train = slice_stats(results, "RVOL >= 1.5", args.split, "train")
    burst_test = slice_stats(results, "RVOL >= 1.5", args.split, "test")
    long_test = slice_stats(
        results, "Matched close-confirmed control", args.split, "test", "long"
    )
    short_test = slice_stats(
        results, "Matched close-confirmed control", args.split, "test", "short"
    )
    lines += [
        "", "## Interpretation", "",
        f"- Matched-control PF was **{fmt(control_train['pf'])}** in train and "
        f"**{fmt(control_test['pf'])}** in test.",
        f"- Requiring price outside the prior value area changed PF to "
        f"**{fmt(outside_train['pf'])}** / **{fmt(outside_test['pf'])}** "
        "(train/test).",
        f"- Requiring at least 1N directional displacement from POC changed PF to "
        f"**{fmt(poc_train['pf'])}** / **{fmt(poc_test['pf'])}**.",
        f"- A raw RVOL >= 1.5 burst changed PF to **{fmt(burst_train['pf'])}** / "
        f"**{fmt(burst_test['pf'])}**. This is the direct test of whether a large "
        "volume burst improves breakout continuation.",
        f"- In test, the unfiltered long/short PF split was **{fmt(long_test['pf'])}** "
        f"versus **{fmt(short_test['pf'])}**. Side conclusions should therefore not be "
        "hidden inside the pooled result.",
    ]

    audit = audits["Matched close-confirmed control"]
    outside_share = sum(row.outside_value for row in control) / len(control) if control else math.nan
    lines += [
        "", "## Signal and leakage audit", "",
        f"- Confirmed channel closes considered while flat: **{audit['confirmed']:,}**",
        f"- Missing valid volume/profile history: **{audit['missing']:,}**",
        f"- Collapsed-N exclusions: **{audit['small_n']:,}**",
        f"- Matched-control trades: **{len(control):,}**",
        f"- Control trades already outside the prior value area: **{outside_share:.1%}**",
        f"- Signal volume and close are used only after that {args.interval} bar has completed; "
        "entry is the following bar's observed open.",
        "- Profiles use only the five sessions preceding the signal. The signal day's volume never "
        "enters its own POC or value area.",
        "- Split adjustments restate both historical prices and volumes before features are built.",
        "- The 2023+ results have now been inspected for this threshold set and are no longer a "
        "pristine holdout for subsequent tuning.",
    ]

    args.outdir.mkdir(parents=True, exist_ok=True)
    report = args.outdir / f"turtle_volume_{args.interval}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    csv_path = args.outdir / f"turtle_volume_{args.interval}_trades.csv"
    csv_rows = [flatten(label, row) for label, rows in results.items() for row in rows]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(csv_rows[0].keys()) if csv_rows else ["variant"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"wrote {report} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
