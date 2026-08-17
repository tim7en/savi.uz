"""Test whether a session's volume-profile shape says anything about the next bar.

The question: at the close of an intraday bar, does the volume profile built
from that session so far predict how far price travels in the *next* bar?

The discipline that makes the answer worth reading:

- A feature row at bar t reads only bars 1..t. Both price and volume are known
  only at a bar's close, so anything else would be reading the answer.
- The target reads only bar t+1.
- The last bar of each session is dropped, so no target is an overnight gap.
- The split is chronological. A random split leaks, because adjacent bars in one
  session share almost the same profile and would land on both sides.
- The headline metric is the *absolute* next-bar move. "Just before a breakout"
  is a claim about magnitude; testing direction would be a much stronger claim
  and this data does not support it.

Usage:
    PYTHONPATH=src python scripts/run_breakout_study.py --ticker SPY --frequency 5min
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

from savi_uz.breakout_study import (  # noqa: E402
    Bucket,
    bucket_by,
    bucket_numeric,
    build_samples,
    mean,
    quantile_edges,
    quantile_labeller,
    split_by_date,
    stratified_buckets,
)
from savi_uz.volume_profile import SHAPE_NAMES, Bar  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--frequency", default="1hour")
    parser.add_argument("--bins", type=int, default=24, help="price bins in the profile")
    parser.add_argument("--min-prefix", type=int, default=3,
                        help="closed bars required before a profile is formed")
    parser.add_argument("--split", default="2023-01-01", help="train/test cutoff date")
    parser.add_argument("--outdir", type=Path, default=Path("out/strategy"))
    return parser.parse_args(argv)


def load_bars(db: Path, ticker: str, frequency: str) -> list[Bar]:
    if not db.is_file():
        raise SystemExit(f"error: {db} not found; run download_intraday_history.py first")
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT ts, open, high, low, close, volume FROM bars "
        "WHERE ticker = ? AND frequency = ? ORDER BY ts",
        (ticker, frequency),
    ).fetchall()
    return [Bar(*row) for row in rows]


def _fmt_bp(value: float) -> str:
    return f"{value * 10_000:.1f}"


def table(rows: list[Bucket], header: str) -> list[str]:
    out = [
        "",
        f"### {header}",
        "",
        "| Bucket | n | Next-bar \\|move\\| (bp) | Next-bar range (bp) | Lift | t |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for b in sorted(rows, key=lambda r: -r.lift):
        out.append(
            f"| {b.label} | {b.count:,} | {_fmt_bp(b.mean_abs)} | {_fmt_bp(b.mean_range)} "
            f"| {b.lift:.3f} | {b.t_stat:+.1f} |"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bars = load_bars(args.db, args.ticker, args.frequency)
    if not bars:
        raise SystemExit(f"error: no {args.frequency} bars for {args.ticker}")

    with_volume = sum(1 for b in bars if b.volume)
    samples = build_samples(bars, bins=args.bins, min_prefix=args.min_prefix)
    if not samples:
        raise SystemExit("error: no samples built; check volume coverage and bar count")

    train, test = split_by_date(samples, args.split)
    sessions = len({s.session for s in samples})
    base_abs = mean([s.forward_abs for s in samples])

    print(f"{args.ticker} {args.frequency}: {len(bars):,} bars, {with_volume:,} with volume")
    print(f"samples {len(samples):,} across {sessions:,} sessions "
          f"({len(train):,} train / {len(test):,} test at {args.split})")
    print(f"baseline next-bar |move| {_fmt_bp(base_abs)} bp\n")

    shape_labels = ["B", "D", "P", "b"]
    features = [
        ("Profile shape", lambda s: s.shape, shape_labels, None),
        ("Value-area width", None, None, lambda s: s.value_width),
        ("Volume concentration", None, None, lambda s: s.concentration),
        ("Close vs POC", None, None, lambda s: s.close_vs_poc),
        ("Close in range", None, None, lambda s: s.close_position),
        ("Session range so far", None, None, lambda s: s.range_pct),
        ("Volume vs session mean", None, None, lambda s: s.volume_ratio),
        ("Bars elapsed", lambda s: f"bar {s.bars_elapsed}", None, None),
    ]

    lines = [
        f"# Volume-profile breakout study - {args.ticker} {args.frequency}",
        "",
        f"- Bars: **{len(bars):,}** ({with_volume:,} with volume)",
        f"- Decision points: **{len(samples):,}** across **{sessions:,}** sessions",
        f"- Train / test: **{len(train):,}** before {args.split} / **{len(test):,}** after",
        f"- Baseline next-bar absolute move: **{_fmt_bp(base_abs)} bp**",
        "",
        "Every feature is computed from bars 1..t of the session; every target reads",
        "bar t+1 only. The last bar of each session is dropped so no target is an",
        "overnight gap. `Lift` is the bucket's mean absolute next-bar move divided by",
        "the mean over all other samples, so 1.000 is no edge.",
        "",
        "## Full sample",
    ]

    all_rows: list[tuple[str, Bucket]] = []
    for name, key, labels, numeric in features:
        if key is not None and labels is not None:
            rows = bucket_by(samples, key, labels)
        elif key is not None:
            present = sorted({key(s) for s in samples})
            rows = bucket_by(samples, key, present)
        else:
            rows = bucket_numeric(samples, numeric, name)
        lines += table(rows, name)
        all_rows += [(name, r) for r in rows]

    # Control for the two effects that dominate the raw table: how far the
    # session has already travelled, and how late in it we are. Both are
    # well-documented volatility effects, and any feature correlated with them
    # inherits their lift without adding information of its own.
    range_bucket = quantile_labeller(samples, lambda s: s.range_pct, 5)
    controls = [range_bucket, lambda s: s.bars_elapsed]
    lines += ["", "## Controlled for session range and time of day", "",
              "The two strongest raw signals -- range travelled so far and bars elapsed --",
              "are volatility clustering and the intraday U-shape, both long known and",
              "neither a profile insight. Here each bucket is compared only against samples",
              "in the same range quintile *and* the same bar of the session, so a feature",
              "cannot win by proxying for them. Lift is pooled across strata by weight.",
              ""]
    controlled: list[tuple[str, Bucket]] = []
    for name, key, labels, numeric in features:
        if name in ("Session range so far", "Bars elapsed"):
            continue
        if key is not None and labels is not None:
            rows = stratified_buckets(samples, key, labels, controls)
        elif numeric is not None:
            lab = quantile_labeller(samples, numeric, 5)
            names = [f"{name} Q{i + 1}" for i in range(5)]
            rows = stratified_buckets(samples, lambda s, f=lab, n=name: f"{n} Q{f(s) + 1}", names, controls)
        else:
            continue
        lines += table(rows, name + " (controlled)")
        controlled += [(name, r) for r in rows]

    lines += ["", "## Does it hold out of sample?", "",
              f"Same buckets, fitted on data before {args.split} and re-measured after it.",
              "A lift that survives the split is worth a second look; one that flips is noise.",
              "", "| Feature | Bucket | Train lift | Test lift | Holds |",
              "|---|---|---:|---:|---|"]

    for name, key, labels, numeric in features:
        if numeric is None and key is None:
            continue
        if key is not None and labels is not None:
            tr = {b.label: b for b in bucket_by(train, key, labels)}
            te = {b.label: b for b in bucket_by(test, key, labels)}
        elif key is not None:
            present = sorted({key(s) for s in samples})
            tr = {b.label: b for b in bucket_by(train, key, present)}
            te = {b.label: b for b in bucket_by(test, key, present)}
        else:
            # Thresholds are fitted on train and reused on test. Refitting them
            # on test would let the bucket boundaries be chosen with knowledge
            # of the data being scored -- a quiet look-ahead in the evaluation
            # even though the features themselves are clean.
            fitted = quantile_edges([numeric(s) for s in train], 5)
            tr = {b.label: b for b in bucket_numeric(train, numeric, name, edges=fitted)}
            te = {b.label: b for b in bucket_numeric(test, numeric, name, edges=fitted)}
        for label in tr:
            if label not in te:
                continue
            a, b = tr[label].lift, te[label].lift
            holds = "yes" if (a - 1) * (b - 1) > 0 and abs(b - 1) > 0.02 else "no"
            lines.append(f"| {name} | {label} | {a:.3f} | {b:.3f} | {holds} |")

    lines += ["", "## Shape vocabulary", ""]
    for code in shape_labels:
        lines.append(f"- **{code}** &mdash; {SHAPE_NAMES[code]}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    report = args.outdir / f"breakout_{args.ticker}_{args.frequency}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    sample_csv = args.outdir / f"samples_{args.ticker}_{args.frequency}.csv"
    with sample_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(samples[0]).keys()))
        writer.writeheader()
        for s in samples:
            writer.writerow(asdict(s))

    print("strongest RAW buckets (uncontrolled - mostly volatility clustering)")
    for name, b in sorted(all_rows, key=lambda r: -r[1].lift)[:5]:
        print(f"  {name:<24}{b.label:<26}lift {b.lift:.3f}  n={b.count:>6,}  t={b.t_stat:+.1f}")
    print("\nstrongest CONTROLLED buckets (same range quintile and bar of session)")
    for name, b in sorted(controlled, key=lambda r: -r[1].lift)[:6]:
        print(f"  {name:<24}{b.label:<26}lift {b.lift:.3f}  n={b.count:>6,}  t={b.t_stat:+.1f}")
    print("\nweakest CONTROLLED buckets")
    for name, b in sorted(controlled, key=lambda r: r[1].lift)[:3]:
        print(f"  {name:<24}{b.label:<26}lift {b.lift:.3f}  n={b.count:>6,}  t={b.t_stat:+.1f}")
    print(f"\nwrote {report} and {sample_csv}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        traceback.print_exc(limit=0)
        raise SystemExit(130)
