"""Test 3/5-session volume-profile breaks, including overnight holding.

Usage:
    PYTHONPATH=src python scripts/run_composite_breakout_study.py --tickers SPY QQQ GLD
"""

from __future__ import annotations

import argparse
import csv
import math
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.composite_breakout import (  # noqa: E402
    CompositeEvent,
    build_events,
    complete_sessions,
    non_overlapping_results,
    summarise_trades,
)
from savi_uz.volume_profile import Bar  # noqa: E402


HORIZONS = (
    ("close", "close_return"),
    ("next open", "next_open_return"),
    ("next close", "next_close_return"),
    ("close +3", "close_3_return"),
    ("close +5", "close_5_return"),
)

CORE_CONFIGS = (
    (3, "value", 1.0, None),
    (5, "value", 1.0, None),
    (3, "range", 1.5, 0.25),
    (5, "range", 1.5, 0.25),
)

TRADE_VARIANTS = (
    ("EOD stop2.5", 2.5, None, 2.0, 0),
    ("next close stop2.5", 2.5, None, 2.0, 1),
    ("next close stop4", 4.0, None, 2.0, 1),
    ("next close stop6", 6.0, None, 2.0, 1),
    ("next close stop8", 8.0, None, 2.0, 1),
    ("next close no stop", None, None, 2.0, 1),
    ("3-close no stop", None, None, 2.0, 3),
    ("5-close no stop", None, None, 2.0, 5),
    ("next close wide trail", 4.0, 2.0, 4.0, 1),
    ("3-close wide trail", 4.0, 2.0, 4.0, 3),
    ("5-close wide trail", 4.0, 2.0, 4.0, 5),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "GLD"])
    parser.add_argument("--frequency", default="5min")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--split", default="2023-01-01")
    parser.add_argument("--profile-coverage", type=float, default=0.90)
    parser.add_argument("--outdir", type=Path, default=Path("out/strategy"))
    return parser.parse_args(argv)


def load_bars(db: Path, ticker: str, frequency: str) -> list[Bar]:
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars "
            "WHERE ticker=? AND frequency=? ORDER BY ts", (ticker, frequency)
        ).fetchall()
    finally:
        connection.close()
    return [Bar(*row) for row in rows]


def _pf(values: list[float]) -> float:
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    return gains / losses if losses else math.inf


def _horizon(events: list[CompositeEvent], attribute: str, cost: float = 0.0002):
    values = [getattr(event, attribute) for event in events]
    net = [value - cost for value in values if value is not None]
    if not net:
        return 0, math.nan, math.nan, math.nan, math.nan
    return (
        len(net),
        sum(net) / len(net),
        median(net),
        sum(value > 0 for value in net) / len(net),
        _pf(net),
    )


def _label(window: int, boundary: str, floor: float, compression: float | None) -> str:
    compressed = "narrow25" if compression is not None else "all"
    return f"{window}d {boundary} vol{floor:g} {compressed}"


def _event_table(lines: list[str], configurations, split: str) -> None:
    lines += [
        "", "## Directional event study", "",
        "Each holding-period return follows every event independently, so 3/5-session horizons may overlap.",
        "Returns are directional and net of a 2 bp round trip. `close` is the signal session close.",
        "", "| Configuration | Period | Horizon | n | Mean bp | Median bp | Win | PF |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for config, events in configurations:
        for period, subset in (
            ("train", [event for event in events if event.session < split]),
            ("test", [event for event in events if event.session >= split]),
        ):
            for horizon, attribute in HORIZONS:
                count, mean, med, win, factor = _horizon(subset, attribute)
                lines.append(
                    f"| {config} | {period} | {horizon} | {count:,} | "
                    f"{mean * 10_000:+.2f} | {med * 10_000:+.2f} | "
                    f"{win * 100:.1f}% | {factor:.2f} |"
                )


def _strategy_table(lines: list[str], configurations, sessions, split: str) -> None:
    lines += [
        "", "## Non-overlapping strategy simulations", "",
        "Only one position may be open. Fixed stops from 2.5-8 ATR and a no-stop benchmark are",
        "compared. The wide trail uses a 4 ATR initial stop, activates after a 4 ATR favorable",
        "excursion, and trails by 2 ATR after completed bars. Overnight gaps through",
        "the stop fill at the next regular-session open. Results include the same 2 bp round trip.",
        "", "| Configuration | Variant | Period | n | Mean bp | Win | PF | $100 -> | CAGR | Max DD | Stop | Gap stop | Avg hold |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config, events in configurations:
        if not any(config.startswith(f"{w}d {b} vol{v:g} {('narrow25' if c else 'all')}")
                   for w, b, v, c in CORE_CONFIGS):
            continue
        for variant, stop, trail, activation, hold in TRADE_VARIANTS:
            for period, subset in (
                ("train", [event for event in events if event.session < split]),
                ("test", [event for event in events if event.session >= split]),
            ):
                results = non_overlapping_results(
                    subset, sessions, stop_atr=stop, trail_atr=trail,
                    activation_atr=activation, max_hold_sessions=hold,
                    round_trip_cost=0.0002,
                )
                summary = summarise_trades(results)
                lines.append(
                    f"| {config} | {variant} | {period} | {summary.count:,} | "
                    f"{summary.mean_return * 10_000:+.2f} | {summary.win_rate * 100:.1f}% | "
                    f"{summary.profit_factor:.2f} | ${summary.ending_equity * 100:.2f} | "
                    f"{summary.cagr * 100:+.2f}% | {summary.max_drawdown * 100:.1f}% | "
                    f"{summary.stop_rate * 100:.1f}% | {summary.gap_stop_rate * 100:.1f}% | "
                    f"{summary.mean_holding_sessions:.2f} |"
                )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    for ticker in [value.upper() for value in args.tickers]:
        bars = load_bars(args.db, ticker, args.frequency)
        sessions = complete_sessions(bars, start=args.start, expected_bars=78)
        configurations = []
        all_events = []
        for window in (3, 5):
            for boundary in ("value", "range"):
                for floor in (1.0, 1.5):
                    for compression in (None, 0.25):
                        events = build_events(
                            bars, window, boundary, floor, start=args.start,
                            expected_bars=78,
                            min_profile_coverage=args.profile_coverage,
                            compression_quantile=compression,
                        )
                        label = _label(window, boundary, floor, compression)
                        configurations.append((label, events))
                        all_events.extend(events)
                        print(f"{ticker:<4} {label:<31} {len(events):>4} events")

        lines = [
            f"# Composite volume-profile breakout - {ticker} {args.frequency}", "",
            f"Complete sessions: **{len(sessions):,}**; split: **{args.split}**; "
            f"minimum prior-session volume coverage: **{args.profile_coverage:.0%}**.", "",
            "Profiles use only the immediately preceding 3 or 5 completed sessions. Value boundaries",
            "are the composite 70% volume area; range boundaries are the full composite high-low.",
            "`narrow25` requires the prior range/daily-ATR ratio to be in the trailing 25th percentile",
            "using only older observations. Relative volume compares the signal bar with the same",
            "five-minute slot over the prior 20 sessions. Entry is the next five-minute open.",
        ]
        _event_table(lines, configurations, args.split)
        _strategy_table(lines, configurations, sessions, args.split)

        stem = f"composite_breakout_{ticker}_{args.frequency}"
        report = args.outdir / f"{stem}.md"
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        event_file = args.outdir / f"{stem}.csv"
        with event_file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(all_events[0]).keys()))
            writer.writeheader()
            for event in all_events:
                writer.writerow(asdict(event))
        print(f"wrote {report} and {event_file} ({len(all_events):,} rows)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
