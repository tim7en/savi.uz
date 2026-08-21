"""What the IEX volume in the old bar store was actually measuring.

The 5-minute book was built from Tiingo, whose US intraday feed is IEX only.
IEX is one venue of roughly a dozen and prints a low single-digit share of
consolidated tape, so its volume is not a scaled-down copy of real volume --
it is a sample, and a thin one. Alpha Vantage serves consolidated volume.

This measures the difference on the bars the two stores share, because two
programme results were decided on IEX volume: the volume-burst overlay and the
volume-profile location study, both rejected. A rejection measured on a 2%
sample of the tape is not the same evidence as a rejection measured on the tape.

Reported per ticker and pooled: how often each source prints no volume at all,
the ratio of medians, and the rank correlation of the two volume series on
matched timestamps. Rank correlation is the one that matters -- a constant
scale factor would be harmless, since every volume rule here thresholds on a
relative measure. Disagreement about which bars were busy is not harmless.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iex", type=Path,
                        default=Path("data/data/intraday/bars.db"))
    parser.add_argument("--consolidated", type=Path,
                        default=Path("data/intraday/bars_av.db"))
    parser.add_argument("--frequency", default="5min")
    parser.add_argument("--start", default="2017-01-03")
    parser.add_argument("--end", default="2026-07-31")
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/volume_source_comparison.json"))
    return parser.parse_args(argv)


def series(path: Path, ticker: str, frequency: str, start: str, end: str):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {
            row[0]: row[1] for row in connection.execute(
                "SELECT ts, volume FROM bars WHERE ticker=? AND frequency=? "
                "AND ts>=? AND ts<=? ORDER BY ts",
                (ticker, frequency, start, end),
            )
        }
    finally:
        connection.close()


def tickers(path: Path, frequency: str) -> set[str]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {r[0] for r in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency=?", (frequency,))}
    finally:
        connection.close()


def spearman(left: list[float], right: list[float]) -> float:
    """Rank correlation; ties averaged."""
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        index = 0
        while index < len(order):
            stop = index
            while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
                stop += 1
            average = (index + stop) / 2 + 1
            for position in range(index, stop + 1):
                out[order[position]] = average
            index = stop + 1
        return out

    a, b = ranks(left), ranks(right)
    n = len(a)
    mean_a, mean_b = statistics.fmean(a), statistics.fmean(b)
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    den = (sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b)) ** 0.5
    return num / den if den else float("nan")


def main(argv=None) -> int:
    args = parse_args(argv)
    shared = sorted(tickers(args.iex, args.frequency)
                    & tickers(args.consolidated, args.frequency))
    if not shared:
        raise SystemExit("error: the two stores share no tickers")
    print(f"{len(shared)} shared tickers, {args.start} to {args.end}, "
          f"{args.frequency} bars\n")
    print(f"  {'ticker':7s} {'matched':>9s} {'iex 0-vol':>10s} {'cons 0-vol':>11s} "
          f"{'iex/cons':>9s} {'spearman':>9s}")

    report = {}
    pooled_zero_iex = pooled_zero_cons = pooled_bars = 0
    shares, correlations = [], []
    for ticker in shared:
        left = series(args.iex, ticker, args.frequency, args.start, args.end)
        right = series(args.consolidated, ticker, args.frequency, args.start, args.end)
        common = sorted(left.keys() & right.keys())
        if len(common) < 500:
            continue
        iex = [float(left[t] or 0.0) for t in common]
        cons = [float(right[t] or 0.0) for t in common]
        zero_iex = sum(1 for v in iex if v <= 0)
        zero_cons = sum(1 for v in cons if v <= 0)
        both = [(a, b) for a, b in zip(iex, cons) if a > 0 and b > 0]
        share = (statistics.median([a for a, _ in both])
                 / statistics.median([b for _, b in both])) if both else float("nan")
        rho = spearman([a for a, _ in both], [b for _, b in both]) if both else float("nan")
        pooled_zero_iex += zero_iex
        pooled_zero_cons += zero_cons
        pooled_bars += len(common)
        shares.append(share)
        correlations.append(rho)
        report[ticker] = {
            "matched_bars": len(common),
            "iex_zero_volume_share": zero_iex / len(common),
            "consolidated_zero_volume_share": zero_cons / len(common),
            "median_volume_ratio": share,
            "spearman": rho,
        }
        print(f"  {ticker:7s} {len(common):>9,d} {zero_iex/len(common):>9.1%} "
              f"{zero_cons/len(common):>10.1%} {share:>9.3f} {rho:>9.3f}")

    summary = {
        "tickers": len(report),
        "matched_bars": pooled_bars,
        "iex_zero_volume_share": pooled_zero_iex / pooled_bars,
        "consolidated_zero_volume_share": pooled_zero_cons / pooled_bars,
        "median_volume_ratio_median": statistics.median(shares),
        "spearman_median": statistics.median(correlations),
        "spearman_min": min(correlations),
        "spearman_max": max(correlations),
    }
    print(f"\n  pooled over {summary['tickers']} tickers, "
          f"{summary['matched_bars']:,} matched bars")
    print(f"    IEX bars with no volume          {summary['iex_zero_volume_share']:.1%}")
    print(f"    consolidated bars with no volume {summary['consolidated_zero_volume_share']:.1%}")
    print(f"    IEX share of consolidated volume {summary['median_volume_ratio_median']:.2%}")
    print(f"    rank correlation, median         {summary['spearman_median']:.3f} "
          f"(min {summary['spearman_min']:.3f}, max {summary['spearman_max']:.3f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "by_ticker": report}, indent=1),
                        encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
