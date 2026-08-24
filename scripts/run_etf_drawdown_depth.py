"""How far should an index fall before you buy it? Asked of ETFs, over 27 years.

Two reasons to move the question off single stocks.

*ETFs cannot be permanently impaired.*  A company can go to zero and a
drawdown-buying rule will keep buying it all the way down; that is the value trap
the stock tests kept running into.  An index replaces its failures automatically,
so a decline is much closer to what the rule assumes it is -- a price change
rather than a change in what you own.

*The history is twice as long.*  Company fundamentals here begin in 2009, which
gave the stock work a window containing no sustained bear market and roughly four
independent three-year periods.  These ETFs begin in 1999 and contain the
dot-com unwind, the financial crisis, 2020 and 2022.

The question is deliberately absolute rather than cross-sectional.  For a single
stock the useful question is *which one*; for an index it is *whether now*, so
what matters is the forward return itself, not the return relative to peers.  The
cross-sectional version is reported beside it because the two answer different
things and confusing them is easy.

Depth is swept finely instead of tested at one threshold, because the shape is
the point: if buying deeper is better there should be an ordering, and if there
is an optimum it should be visible as a peak rather than asserted.  And the sweep
is split -- the shape is read on 1999-2012 and checked on 2013-2026 -- because a
peak located on all the data is just the deepest bucket that happened to precede
a recovery.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sqlite3
import sys
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

HORIZONS = (252, 504, 756)
LABELS = {252: "1y", 504: "2y", 756: "3y"}
BUCKETS = ((0.00, 0.03), (0.03, 0.07), (0.07, 0.12), (0.12, 0.18),
           (0.18, 0.25), (0.25, 0.35), (0.35, 0.50), (0.50, 1.01))
LOOKBACKS = (252, 500)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path,
                        default=Path("data/cross_assets/etf_daily.db"))
    parser.add_argument("--split", default="2013-01-01")
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/etf_drawdown_depth.json"))
    known, _ = parser.parse_known_args(argv)
    return known


def load(args):
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    book = {}
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency='daily' "
            "ORDER BY ticker"):
        rows = connection.execute(
            "SELECT ts,open,high,low,close FROM bars WHERE ticker=? AND "
            "frequency='daily' ORDER BY ts", (ticker,)).fetchall()
        bars = adjust_bars([Bar(r[0][:10], r[1], r[2], r[3], r[4], None)
                            for r in rows], splits.get(ticker, []))
        if len(bars) >= 1500:
            book[ticker] = bars
    connection.close()
    return book


def build(book, lookback):
    """One observation per ETF per month: depth now, return later."""
    rows = []
    for ticker, bars in book.items():
        days = [b.timestamp for b in bars]
        closes = [b.close for b in bars]
        peak, running = [], []
        for i in range(len(bars)):
            running.append(closes[i])
            if len(running) > lookback:
                running.pop(0)
            peak.append(max(running))
        seen = set()
        for i in range(lookback, len(bars) - min(HORIZONS) - 1):
            if days[i][:7] in seen:
                continue
            seen.add(days[i][:7])
            if peak[i] <= 0 or closes[i] <= 0:
                continue
            row = {"ticker": ticker, "day": days[i],
                   "depth": -(closes[i] / peak[i] - 1.0)}
            for horizon in HORIZONS:
                ahead = i + horizon
                row[f"fwd_{horizon}"] = (closes[ahead] / closes[i] - 1.0
                                         if ahead < len(bars) else None)
            rows.append(row)
    # cross-sectional excess, per month and per horizon
    by_month = defaultdict(list)
    for r in rows:
        by_month[r["day"][:7]].append(r)
    for group in by_month.values():
        for horizon in HORIZONS:
            have = [r for r in group if r[f"fwd_{horizon}"] is not None]
            if len(have) < 8:
                for r in group:
                    r[f"xs_{horizon}"] = None
                continue
            average = statistics.fmean(r[f"fwd_{horizon}"] for r in have)
            for r in group:
                r[f"xs_{horizon}"] = (r[f"fwd_{horizon}"] - average
                                      if r[f"fwd_{horizon}"] is not None else None)
    rows.sort(key=lambda r: r["day"])
    return rows


def stats(chunk, horizon, key="fwd"):
    values = [r[f"{key}_{horizon}"] for r in chunk
              if r.get(f"{key}_{horizon}") is not None]
    if len(values) < 25:
        return None
    return {"n": len(values), "median": statistics.median(values),
            "mean": statistics.fmean(values),
            "win": sum(1 for v in values if v > 0) / len(values)}


def table(rows, key, title, note):
    print(f"\n########## {title} ##########")
    print(f"  {note}")
    print(f"  {'drawdown at entry':22s} {'n':>7s} " +
          " ".join(f"{'med ' + LABELS[h]:>10s}" for h in HORIZONS) +
          f" {'mean 3y':>9s} {'win 3y':>8s}")
    out, medians = {}, []
    for low, high in BUCKETS:
        chunk = [r for r in rows if low <= r["depth"] < high]
        cells, three = [], None
        for horizon in HORIZONS:
            got = stats(chunk, horizon, key)
            cells.append(f"{got['median']:>+10.1%}" if got else f"{'-':>10s}")
            if horizon == 756:
                three = got
        if three is None:
            continue
        label = f"{low:.0%} to {high:.0%}" if high < 1 else f"{low:.0%}+"
        medians.append((label, three["median"]))
        out[label] = three
        print(f"  {label:22s} {three['n']:>7,d} " + " ".join(cells) +
              f" {three['mean']:>+9.1%} {three['win']:>8.0%}")
    if medians:
        best = max(medians, key=lambda kv: kv[1])
        print(f"  -> deepest median at {best[0]} ({best[1]:+.1%})")
        out["peak"] = best[0]
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    book = load(args)
    rows = build(book, args.lookback)
    span = (rows[0]["day"], rows[-1]["day"])
    print(f"{len(book)} ETFs, {len(rows):,d} ETF-months, {span[0]} to {span[1]}")
    print(f"drawdown measured against the trailing {args.lookback}-session high")
    depths = sorted(r["depth"] for r in rows)
    print(f"depth: median {statistics.median(depths):.1%}, "
          f"90th percentile {depths[int(.9 * len(depths))]:.1%}, "
          f"deepest {depths[-1]:.1%}")
    report = {"etfs": len(book), "observations": len(rows), "span": span}

    report["absolute_all"] = table(
        rows, "fwd", "absolute forward return by drawdown depth, all years",
        "The question is whether to be invested, so this is the raw return.")

    inside = [r for r in rows if r["day"] < args.split]
    outside = [r for r in rows if r["day"] >= args.split]
    report["absolute_in"] = table(
        inside, "fwd", f"the same, {span[0][:4]} to {args.split[:4]} only",
        "Where the shape gets read.")
    report["absolute_out"] = table(
        outside, "fwd", f"the same, {args.split[:4]} onward",
        "Where it gets checked. A peak that moves between these two is noise.")

    report["cross_sectional"] = table(
        rows, "xs", "cross-sectional: which ETF, not whether",
        "Excess over an equal-weight hold of the same ETFs that month.")

    print("\n########## how often each depth is even available ##########")
    print("  A rule that waits for 35% down is out of the market most of the time.")
    months = len({r["day"][:7] for r in rows})
    print(f"  {'threshold':22s} {'ETF-months':>12s} {'share':>8s} "
          f"{'months with any':>17s}")
    report["availability"] = {}
    for threshold in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        hit = [r for r in rows if r["depth"] >= threshold]
        distinct = len({r["day"][:7] for r in hit})
        report["availability"][f"{threshold:.0%}"] = {
            "observations": len(hit), "share": len(hit) / len(rows),
            "months": distinct, "month_share": distinct / months}
        print(f"  down {threshold:.0%} or more{'':7s} {len(hit):>12,d} "
              f"{len(hit)/len(rows):>8.1%} {distinct/months:>16.0%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
