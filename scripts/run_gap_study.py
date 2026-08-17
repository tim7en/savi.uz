"""How far does SPY move before the open, and does the move hold?

The feed covers the regular session only, so the pre-market *path* is not
visible. Its net result is: the gap from one close to the next open contains
every after-hours and pre-market tick.

Usage:
    PYTHONPATH=src python scripts/run_gap_study.py --ticker SPY --frequency 5min
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

from savi_uz.gap_study import (  # noqa: E402
    GapBucket,
    bucket_by_size,
    build_gaps,
    group_sessions,
    mean,
    median,
    summarise,
)
from savi_uz.volume_profile import Bar  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--frequency", default="5min")
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument("--min-gap-bp", type=float, default=5.0)
    parser.add_argument("--outdir", type=Path, default=Path("out/strategy"))
    return parser.parse_args(argv)


def load_bars(db: Path, ticker: str, frequency: str) -> list[Bar]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT ts, open, high, low, close, volume FROM bars "
        "WHERE ticker = ? AND frequency = ? ORDER BY ts",
        (ticker, frequency),
    ).fetchall()
    return [Bar(*row) for row in rows]


def table(rows: list[GapBucket], header: str) -> list[str]:
    out = [
        "", f"### {header}", "",
        "| Bucket | n | Mean gap | Median retained | Filled | Extended | Reversed | Value overlap | Opening volume |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for b in rows:
        out.append(
            f"| {b.label} | {b.count:,} | {b.mean_gap_bp:.0f}bp | {b.median_retained:.2f} "
            f"| {b.fill_rate*100:.0f}% | {b.extend_rate*100:.0f}% | {b.reverse_rate*100:.0f}% "
            f"| {b.mean_overlap:.2f} | {b.mean_volume_ratio:.2f}x |"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bars = load_bars(args.db, args.ticker, args.frequency)
    if not bars:
        raise SystemExit(f"error: no {args.frequency} bars for {args.ticker}")

    sessions = group_sessions(bars)
    gaps = build_gaps(sessions, bins=args.bins, min_gap_bp=args.min_gap_bp)
    if not gaps:
        raise SystemExit("error: no gaps built")

    ups = [g for g in gaps if g.direction > 0]
    downs = [g for g in gaps if g.direction < 0]
    overall = summarise(gaps, "all gaps")

    print(f"{args.ticker} {args.frequency}: {len(sessions):,} sessions, "
          f"{len(gaps):,} with a gap over {args.min_gap_bp:.0f}bp")
    print(f"  {gaps[0].session} -> {gaps[-1].session}")
    print(f"\nmedian absolute gap {median([abs(g.gap_bp) for g in gaps]):.0f}bp, "
          f"mean {mean([abs(g.gap_bp) for g in gaps]):.0f}bp, "
          f"largest {max(abs(g.gap_bp) for g in gaps):.0f}bp")
    print(f"the gap is {mean([abs(g.gap_bp) for g in gaps]) / mean([g.range_bp for g in gaps]) * 100:.0f}% "
          f"the size of the average whole session range ({mean([g.range_bp for g in gaps]):.0f}bp)\n")

    print("is it sustained?")
    print(f"  median retained at the close : {overall.median_retained:.2f} of the gap")
    print(f"  filled at some point         : {overall.fill_rate*100:.0f}%")
    print(f"  extended beyond the gap      : {overall.extend_rate*100:.0f}%")
    print(f"  reversed through prior close : {overall.reverse_rate*100:.0f}%")
    print(f"  value area overlap with prior: {overall.mean_overlap:.2f}")
    print(f"  opening volume vs session    : {overall.mean_volume_ratio:.2f}x\n")

    lines = [
        f"# Overnight gaps - {args.ticker} {args.frequency}",
        "",
        f"- Sessions: **{len(sessions):,}**, of which **{len(gaps):,}** opened more than "
        f"{args.min_gap_bp:.0f}bp from the prior close",
        f"- Span: {gaps[0].session} to {gaps[-1].session}",
        f"- Median absolute gap **{median([abs(g.gap_bp) for g in gaps]):.0f}bp**, "
        f"mean {mean([abs(g.gap_bp) for g in gaps]):.0f}bp, largest "
        f"{max(abs(g.gap_bp) for g in gaps):.0f}bp",
        "",
        "The feed is regular-session only, 09:30 to 16:00 ET, so the pre-market path is",
        "not observable here. The gap is its net result and contains every after-hours",
        "and pre-market tick.",
        "",
        "`Retained` is where the session closed along the gap: 1.00 means the whole move",
        "was held, 0.00 that it was given back exactly, above 1 that it extended, below 0",
        "that price crossed back through the prior close. `Value overlap` is how much of",
        "the new session's value area sits on the old one -- 0 is a new distribution,",
        "1 is the market trading right back where it was.",
    ]
    lines += table([overall], "All gaps")
    lines += table(bucket_by_size(gaps), "By gap size")
    lines += table([summarise(ups, "gap up"), summarise(downs, "gap down")], "By direction")

    args.outdir.mkdir(parents=True, exist_ok=True)
    report = args.outdir / f"gaps_{args.ticker}_{args.frequency}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    csv_path = args.outdir / f"gaps_{args.ticker}_{args.frequency}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(gaps[0]).keys()))
        writer.writeheader()
        for gap in gaps:
            writer.writerow(asdict(gap))

    print("by gap size")
    for b in bucket_by_size(gaps):
        print(f"  {b.label:<12} n={b.count:>5,}  retained {b.median_retained:>5.2f}  "
              f"filled {b.fill_rate*100:>3.0f}%  overlap {b.mean_overlap:.2f}  "
              f"open vol {b.mean_volume_ratio:.2f}x")
    print(f"\nwrote {report} and {csv_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        traceback.print_exc(limit=0)
        raise SystemExit(130)
