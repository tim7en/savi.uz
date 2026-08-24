"""What entry depth does 3x or 5x actually require?

The question inverted.  Every previous test asked what return a given drawdown
entry produces; this asks what drawdown a given leverage survives, which is the
only version that decides whether a position can be held at all.

The arithmetic is fixed and unkind.  At leverage L a further fall of 1/L takes
the whole position: 33% at 3x, 20% at 5x.  Whether the stop is explicit or the
broker supplies it makes no difference to the threshold, only to who closes the
trade -- so the same number answers both "how often am I wiped out" and "how
often am I stopped out", and the choice between them is a choice about who does
the selling.

So for every entry depth the distribution of the *further* fall is reported, from
the median out to the first percentile, and beside it the leverage that would
have survived each point of that distribution.  Then the probability that 3x and
5x are closed out, at each depth.

The measurement is the worst mark against the entry over the following three
years, taken from daily lows, so an intraday spike that would have triggered a
margin call is counted.  Equity ETFs only: futures-backed and volatility products
decay for reasons that have nothing to do with drawdown, and currencies, bonds
and metals have no earnings for a margin of safety to be measured against.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

HORIZON = 756
LOOKBACK = 252
BUCKETS = ((0.00, 0.05), (0.05, 0.12), (0.12, 0.20), (0.20, 0.30),
           (0.30, 0.45), (0.45, 1.01))
LEVERAGES = (2.0, 3.0, 5.0)
DECAYING = ("USO", "UNG", "GSG", "DBC", "DBA", "BNO", "VXX")
NON_EQUITY = ("FXB", "FXE", "FXY", "UUP", "HYG", "IEF", "LQD", "SHY",
              "TIP", "TLT", "GLD", "SLV", "PPLT")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path,
                        default=Path("data/cross_assets/etf_daily.db"))
    parser.add_argument("--stocks", type=Path,
                        default=Path("data/13f/alphavantage_daily.db"))
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/leverage_entry_depth.json"))
    known, _ = parser.parse_known_args(argv)
    return known


def load(path, drop=True):
    splits = load_splits(path)
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    book = {}
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency='daily' "
            "ORDER BY ticker"):
        if drop and (ticker in DECAYING or ticker in NON_EQUITY):
            continue
        rows = connection.execute(
            "SELECT ts,open,high,low,close FROM bars WHERE ticker=? AND "
            "frequency='daily' ORDER BY ts", (ticker,)).fetchall()
        bars = adjust_bars([Bar(r[0][:10], r[1], r[2], r[3], r[4], None)
                            for r in rows], splits.get(ticker, []))
        if len(bars) >= 1000:
            book[ticker] = bars
    connection.close()
    return book


def entries(book):
    """Every fresh drawdown crossing, with the worst mark that followed it."""
    rows = []
    for ticker, bars in book.items():
        closes = [b.close for b in bars]
        lows = [b.low for b in bars]
        peak, running = [], []
        for i in range(len(bars)):
            running.append(closes[i])
            if len(running) > LOOKBACK:
                running.pop(0)
            peak.append(max(running))
        seen = set()
        for i in range(LOOKBACK, len(bars) - 60):
            if bars[i].timestamp[:7] in seen or peak[i] <= 0 or closes[i] <= 0:
                continue
            seen.add(bars[i].timestamp[:7])
            end = min(i + HORIZON, len(bars) - 1)
            floor = min(lows[i + 1:end + 1] or [closes[i]])
            rows.append({"ticker": ticker, "day": bars[i].timestamp,
                         "depth": -(closes[i] / peak[i] - 1.0),
                         "further": floor / closes[i] - 1.0})
    return rows


def percentile(values, share):
    if not values:
        return float("nan")
    return values[min(int(share * len(values)), len(values) - 1)]


def report_universe(rows, title, report, key):
    print()
    print(f"########## {title} ##########")
    print(f"  {len(rows):,d} entries. 'further' is the worst mark below the entry")
    print("  over the next three years, taken from daily lows.")
    print(f"  {'depth at entry':18s} {'n':>6s} {'median':>8s} {'75th':>8s} "
          f"{'90th':>8s} {'95th':>8s} {'99th':>8s} {'safe lev':>9s}")
    report[key] = {}
    for low, high in BUCKETS:
        chunk = sorted(r["further"] for r in rows if low <= r["depth"] < high)
        if len(chunk) < 30:
            continue
        label = f"{low:.0%} to {high:.0%}" if high < 1 else f"{low:.0%}+"
        p95 = percentile(chunk, 0.05)
        safe = 1.0 / abs(p95) if p95 < 0 else float("inf")
        report[key][label] = {
            "n": len(chunk), "median": statistics.median(chunk),
            "p75": percentile(chunk, 0.25), "p90": percentile(chunk, 0.10),
            "p95": p95, "p99": percentile(chunk, 0.01), "safe_leverage": safe}
        print(f"  {label:18s} {len(chunk):>6,d} "
              f"{statistics.median(chunk):>8.1%} {percentile(chunk, 0.25):>8.1%} "
              f"{percentile(chunk, 0.10):>8.1%} {p95:>8.1%} "
              f"{percentile(chunk, 0.01):>8.1%} {safe:>8.2f}x")

    print()
    print("  probability the position is closed out, by leverage")
    print(f"  {'depth at entry':18s} {'n':>6s} " +
          " ".join(f"{'at ' + format(L, '.0f') + 'x':>10s}" for L in LEVERAGES))
    report[key + "_wipeout"] = {}
    for low, high in BUCKETS:
        chunk = [r["further"] for r in rows if low <= r["depth"] < high]
        if len(chunk) < 30:
            continue
        label = f"{low:.0%} to {high:.0%}" if high < 1 else f"{low:.0%}+"
        shares = []
        for leverage in LEVERAGES:
            threshold = -1.0 / leverage
            shares.append(sum(1 for f in chunk if f <= threshold) / len(chunk))
        report[key + "_wipeout"][label] = dict(
            zip((f"{L:.0f}x" for L in LEVERAGES), shares))
        print(f"  {label:18s} {len(chunk):>6,d} " +
              " ".join(f"{s:>10.0%}" for s in shares))


def main(argv=None) -> int:
    args = parse_args(argv)
    report = {}
    print("At leverage L a further fall of 1/L closes the position:")
    for leverage in LEVERAGES:
        print(f"  {leverage:.0f}x survives a further fall of "
              f"{100.0 / leverage:.0f}% and no more")

    etfs = load(args.bars)
    report_universe(entries(etfs), f"equity ETFs ({len(etfs)} names)",
                    report, "etf")

    if args.stocks.exists():
        stocks = load(args.stocks, drop=False)
        report_universe(entries(stocks), f"single stocks ({len(stocks)} names)",
                        report, "stock")

    print()
    print("########## what this means for an entry rule ##########")
    etf_rows = entries(etfs)
    for leverage in LEVERAGES:
        threshold = -1.0 / leverage
        overall = sum(1 for r in etf_rows if r["further"] <= threshold) / len(etf_rows)
        deep = [r for r in etf_rows if r["depth"] >= 0.30]
        deep_share = (sum(1 for r in deep if r["further"] <= threshold) / len(deep)
                      if deep else float("nan"))
        print(f"  {leverage:.0f}x  closed out on {overall:.0%} of all entries, "
              f"{deep_share:.0%} of entries already 30%+ down")
    report["summary"] = {"entries": len(etf_rows)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print()
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
