"""Compare 5/10-session Donchian breaks across relative-volume floors.

Usage:
    PYTHONPATH=src python scripts/run_donchian_study.py --ticker SPY --frequency 5min
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.donchian_study import build_events, summarise  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--frequency", default="5min")
    parser.add_argument("--windows", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--volume-floors", nargs="+", type=float,
                        default=[0.0, 1.0, 1.25, 1.5, 2.0])
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--split", default="2023-01-01")
    parser.add_argument("--outdir", type=Path, default=Path("out/strategy"))
    return parser.parse_args(argv)


def load_bars(db: Path, ticker: str, frequency: str) -> list[Bar]:
    if not db.is_file():
        raise SystemExit(f"error: {db} not found")
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT ts, open, high, low, close, volume FROM bars "
            "WHERE ticker = ? AND frequency = ? ORDER BY ts", (ticker, frequency)
        ).fetchall()
    finally:
        connection.close()
    return [Bar(*row) for row in rows]


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _row(window: int, floor: float, period: str, events) -> str:
    summary = summarise(events)
    return (
        f"| {window} | {floor:.2f}x | {period} | {summary.count:,} | "
        f"{summary.mean_r:+.3f} | {summary.profit_factor:.2f} | "
        f"{_pct(summary.target_rate)} | {_pct(summary.sustainable_rate)} | "
        f"{_pct(summary.accepted_60m_rate)} | {_pct(summary.reentry_30m_rate)} | "
        f"{_pct(summary.whipsaw_30m_rate)} | {summary.median_mfe_r:.2f} |"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bars = load_bars(args.db, args.ticker.upper(), args.frequency)
    if not bars:
        raise SystemExit(f"error: no {args.ticker} {args.frequency} bars")

    all_events = []
    lines = [
        f"# Donchian breakout volume sweep - {args.ticker.upper()} {args.frequency}", "",
        f"Prior-session windows: **{', '.join(map(str, args.windows))}**; "
        f"chronological split: **{args.split}**.", "",
        "The channel uses completed prior sessions only. A breakout is observed at a bar close and",
        "entered at the next bar open. Relative volume compares the signal bar with the median",
        "of the same five-minute slot over the previous 20 clean sessions. Sessions must contain",
        "78 positive-volume bars. `Sustainable` means no close back inside during the next 30",
        "minutes and still outside after 60 minutes. The trade outcome uses a 2 ATR stop and 2R",
        "target, conservatively charging a stop if both are touched in one bar.", "",
        "| Sessions | Vol floor | Period | n | Mean R | PF | Target | Sustainable | Outside 60m | Re-enter 30m | Whipsaw 30m | Median MFE |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for window in args.windows:
        for floor in args.volume_floors:
            events = build_events(
                bars, window, floor, start=args.start,
                expected_bars=78 if args.frequency == "5min" else None,
            )
            all_events.extend(events)
            train = [event for event in events if event.session < args.split]
            test = [event for event in events if event.session >= args.split]
            lines.append(_row(window, floor, "train", train))
            lines.append(_row(window, floor, "test", test))

            for direction, label in ((1, "long"), (-1, "short")):
                side = [event for event in test if event.direction == direction]
                summary = summarise(side)
                print(
                    f"{window:>2}d {floor:>4.2f}x test {label:<5} n={summary.count:>3} "
                    f"sustain={summary.sustainable_rate*100:>5.1f}% "
                    f"whip={summary.whipsaw_30m_rate*100:>5.1f}% "
                    f"meanR={summary.mean_r:+.3f} PF={summary.profit_factor:.2f}"
                )

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = f"donchian_{args.ticker.upper()}_{args.frequency}"
    report = args.outdir / f"{stem}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    csv_path = args.outdir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(all_events[0]).keys()))
        writer.writeheader()
        for event in all_events:
            writer.writerow(asdict(event))
    print(f"\nwrote {report} and {csv_path} ({len(all_events):,} threshold-event rows)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        traceback.print_exc(limit=0)
        raise SystemExit(130)

