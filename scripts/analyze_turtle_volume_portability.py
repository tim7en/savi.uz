"""Measure whether Turtle volume/profile filters improve results across tickers."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


VARIANTS = (
    "Outside prior value area",
    "RVOL >= 1.5",
    "POC distance >= 1N",
)
CONTROL = "Matched close-confirmed control"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("out/strategy"))
    parser.add_argument("--split", default="2023-01-01")
    return parser.parse_args(argv)


def profit_factor(values: list[float]) -> float:
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    return gains / losses if losses else math.inf


def summarize(path: Path, split: str):
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["variant"] not in (CONTROL, *VARIANTS):
                continue
            period = "train" if row["signal_timestamp"][:10] < split else "test"
            grouped[(row["variant"], period, row["ticker"])].append(
                float(row["net_r"])
            )

    result = []
    for variant in VARIANTS:
        for period in ("train", "test"):
            control_tickers = {
                ticker for label, item_period, ticker in grouped
                if label == CONTROL and item_period == period
            }
            variant_tickers = {
                ticker for label, item_period, ticker in grouped
                if label == variant and item_period == period
            }
            common = sorted(control_tickers & variant_tickers)
            pf_better = 0
            mean_better = 0
            for ticker in common:
                control = grouped[(CONTROL, period, ticker)]
                filtered = grouped[(variant, period, ticker)]
                pf_better += profit_factor(filtered) > profit_factor(control)
                mean_better += (
                    sum(filtered) / len(filtered) > sum(control) / len(control)
                )
            result.append((variant, period, len(common), pf_better, mean_better))
    return result


def main(argv=None) -> int:
    args = parse_args(argv)
    sources = (
        ("daily", args.outdir / "turtle_volume_daily_trades.csv"),
        ("5min", args.outdir / "turtle_volume_5min_trades.csv"),
    )
    lines = [
        "# Turtle volume/profile cross-symbol portability", "",
        "A filter counts as an improvement for a ticker only when its own filtered "
        "profit factor or mean R exceeds that ticker's matched close-confirmed control. "
        "Only tickers with trades in both samples are compared.", "",
        "| Interval | Filter | Period | Common tickers | PF improved | Mean R improved |",
        "|---|---|---|---:|---:|---:|",
    ]
    for interval, path in sources:
        if not path.exists() and interval == "daily":
            path = args.outdir / "turtle_volume_trades.csv"
        if not path.exists():
            continue
        for variant, period, count, pf_better, mean_better in summarize(path, args.split):
            lines.append(
                f"| {interval} | {variant} | {period} | {count} | "
                f"{pf_better}/{count} | {mean_better}/{count} |"
            )
    output = args.outdir / "turtle_volume_portability.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
