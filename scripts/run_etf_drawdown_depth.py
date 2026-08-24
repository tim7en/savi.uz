"""How far should an index fall before you buy it, and what does it cost to be wrong?

Two reasons the question moved off single stocks.  An index replaces its failures
automatically, so a decline is closer to a price change than to a change in what
you own -- the value trap the stock tests kept hitting.  And this data begins in
1999, so it contains the dot-com unwind, the financial crisis, 2020 and 2022,
where the company-fundamentals work had none of them.

But the first version of this study made a claim in its own docstring that turned
out to be false: that an ETF cannot be permanently impaired.  A futures-backed
commodity fund or a volatility product bleeds from roll and decay regardless of
what its underlying does, and those instruments filled the deepest drawdown
buckets and produced a cliff that looked like a finding about depth.  They are
excluded here by construction, and the cliff is re-examined without them.

Three things are measured.

*The shape.*  Forward return by drawdown depth, swept finely, because the
question is where the optimum sits and an optimum should be visible as a peak
rather than asserted.  Split in half so a peak located on one era can be checked
on the other.

*The risk of being early.*  A depth is only a margin of safety if the further
fall it exposes you to is smaller than the recovery it buys.  For every entry the
worst subsequent drawdown is recorded, at the median and at the fifth percentile,
which is the number that decides whether a position can actually be held.

*The ratio between them.*  Median three-year gain over median further fall, which
is the margin of safety expressed as something checkable rather than as a
sentiment.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sqlite3
import sys
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

# Futures-backed and volatility products. Their price decays from roll and
# compounding whatever the underlying does, so a deep drawdown in one of them is
# not the cyclical decline this study is about.
DECAYING = ("USO", "UNG", "GSG", "DBC", "DBA", "BNO", "VXX")

# A margin-of-safety framework asks what a thing is worth against what it earns.
# Currencies and physical metals earn nothing, and a bond fund's drawdown is a
# rate move rather than a discount to intrinsic value, so none of them can be
# cheap in the sense the framework means. Equity only.
NON_EQUITY = ("FXB", "FXE", "FXY", "UUP",
              "HYG", "IEF", "LQD", "SHY", "TIP", "TLT",
              "GLD", "SLV", "PPLT")
RISK_RUNGS = (0.005, 0.01, 0.02)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path,
                        default=Path("data/cross_assets/etf_daily.db"))
    parser.add_argument("--split", default="2013-01-01")
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--keep-decaying", action="store_true",
                        help="leave the futures and volatility products in")
    parser.add_argument("--all-asset-classes", action="store_true",
                        help="keep currencies, bonds and physical metals")
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/etf_drawdown_depth.json"))
    known, _ = parser.parse_known_args(argv)
    return known


def load(args):
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    book, dropped = {}, []
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency='daily' "
            "ORDER BY ticker"):
        if not args.keep_decaying and ticker in DECAYING:
            dropped.append(ticker)
            continue
        if not args.all_asset_classes and ticker in NON_EQUITY:
            dropped.append(ticker)
            continue
        rows = connection.execute(
            "SELECT ts,open,high,low,close FROM bars WHERE ticker=? AND "
            "frequency='daily' ORDER BY ts", (ticker,)).fetchall()
        bars = adjust_bars([Bar(r[0][:10], r[1], r[2], r[3], r[4], None)
                            for r in rows], splits.get(ticker, []))
        if len(bars) >= 1500:
            book[ticker] = bars
    connection.close()
    return book, dropped


def market_forward(book, ticker="DIA"):
    """What the broad index did over each forward window, for comparison.

    The relevant alternative to buying a fallen sector is not cash, it is simply
    holding the index, so every entry is scored against what the index itself
    returned across the identical dates."""
    bars = book.get(ticker)
    if not bars:
        return {}
    days = [b.timestamp for b in bars]
    closes = [b.close for b in bars]
    out = {}
    for i, day in enumerate(days):
        entry = {}
        for horizon in HORIZONS:
            ahead = i + horizon
            entry[horizon] = (closes[ahead] / closes[i] - 1.0
                              if ahead < len(bars) and closes[i] > 0 else None)
        out[day] = entry
    return out


def build(book, lookback, market=None):
    rows = []
    for ticker, bars in book.items():
        days = [b.timestamp for b in bars]
        closes = [b.close for b in bars]
        lows = [b.low for b in bars]
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
            # how far below the entry it went before it recovered
            end = min(i + max(HORIZONS), len(bars) - 1)
            row["further"] = min(lows[i + 1:end + 1] or [closes[i]]) / closes[i] - 1.0
            if market is not None:
                reference = market.get(days[i], {})
                for horizon in HORIZONS:
                    row[f"mkt_{horizon}"] = reference.get(horizon)
            rows.append(row)
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


def buckets_of(rows):
    for low, high in BUCKETS:
        chunk = [r for r in rows if low <= r["depth"] < high]
        label = f"{low:.0%} to {high:.0%}" if high < 1 else f"{low:.0%}+"
        yield label, chunk


def shape(rows, title, note, key="fwd"):
    print()
    print(f"########## {title} ##########")
    print(f"  {note}")
    print(f"  {'drawdown at entry':22s} {'n':>7s} " +
          " ".join(f"{'med ' + LABELS[h]:>10s}" for h in HORIZONS) + f" {'win 3y':>8s}")
    out, peak = {}, []
    for label, chunk in buckets_of(rows):
        cells, three = [], None
        for horizon in HORIZONS:
            got = stats(chunk, horizon, key)
            cells.append(f"{got['median']:>+10.1%}" if got else f"{'-':>10s}")
            if horizon == 756:
                three = got
        if three is None:
            continue
        out[label] = three
        peak.append((label, three["median"]))
        print(f"  {label:22s} {three['n']:>7,d} " + " ".join(cells) +
              f" {three['win']:>8.0%}")
    if peak:
        best = max(peak, key=lambda kv: kv[1])
        print(f"  -> deepest median at {best[0]} ({best[1]:+.1%})")
        out["peak"] = best[0]
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    book, dropped = load(args)
    market = market_forward(book)
    rows = build(book, args.lookback, market)
    span = (rows[0]["day"], rows[-1]["day"])
    print(f"{len(book)} ETFs, {len(rows):,d} ETF-months, {span[0]} to {span[1]}")
    if dropped:
        print(f"excluded {len(dropped)} futures and volatility products: "
              f"{', '.join(dropped)}")
    print(f"drawdown against the trailing {args.lookback}-session high")
    report = {"etfs": len(book), "excluded": dropped,
              "observations": len(rows), "span": span}

    report["all"] = shape(rows, "forward return by drawdown depth, all years",
                          "Absolute return: the question is whether to be invested.")
    report["in"] = shape([r for r in rows if r["day"] < args.split],
                         f"{span[0][:4]} to {args.split[:4]}", "Where the shape is read.")
    report["out"] = shape([r for r in rows if r["day"] >= args.split],
                          f"{args.split[:4]} onward",
                          "Where it is checked. A peak that moves is noise.")

    print()
    print("########## the risk of being early ##########")
    print("  How much further it fell below the entry before recovering, and what")
    print("  the recovery paid for that. Margin of safety is the ratio.")
    print(f"  {'drawdown at entry':22s} {'n':>7s} {'further, median':>16s} "
          f"{'worst 5%':>10s} {'gain 3y':>9s} {'gain/risk':>10s}")
    report["risk"] = {}
    for label, chunk in buckets_of(rows):
        further = sorted(r["further"] for r in chunk)
        if len(further) < 25:
            continue
        three = stats(chunk, 756)
        if not three:
            continue
        median_fall = statistics.median(further)
        worst = further[int(0.05 * len(further))]
        ratio = three["median"] / abs(median_fall) if median_fall < 0 else float("nan")
        report["risk"][label] = {"n": len(further), "further_median": median_fall,
                                 "further_p05": worst, "gain_3y": three["median"],
                                 "gain_over_risk": ratio}
        print(f"  {label:22s} {len(further):>7,d} {median_fall:>15.1%} "
              f"{worst:>10.1%} {three['median']:>+9.1%} {ratio:>10.2f}")

    print()
    print("########## against simply holding the index ##########")
    print("  The alternative is not cash, it is DIA. Beat = the fallen sector")
    print("  returned more over the same window. Double = it returned at least")
    print("  twice as much, counted only where the index itself rose.")
    print(f"  {'drawdown at entry':22s} {'n':>7s} {'median excess':>14s} "
          f"{'beat index':>11s} {'doubled it':>11s}")
    report["versus_index"] = {}
    for label, chunk in buckets_of(rows):
        have = [r for r in chunk if r.get("mkt_756") is not None
                and r.get("fwd_756") is not None]
        if len(have) < 25:
            continue
        excess = [r["fwd_756"] - r["mkt_756"] for r in have]
        beat = sum(1 for e in excess if e > 0) / len(excess)
        rose = [r for r in have if r["mkt_756"] > 0]
        doubled = (sum(1 for r in rose if r["fwd_756"] >= 2 * r["mkt_756"])
                   / len(rose)) if len(rose) >= 20 else float("nan")
        report["versus_index"][label] = {
            "n": len(have), "median_excess": statistics.median(excess),
            "beat": beat, "doubled": doubled}
        print(f"  {label:22s} {len(have):>7,d} "
              f"{statistics.median(excess):>+14.1%} {beat:>11.0%} "
              f"{doubled:>11.0%}")

    print()
    print("########## risk sizing: what you can hold ##########")
    print("  Size is set by the tail, not the median. The worst 5% further fall")
    print("  is the loss a position has to survive, so position size is the risk")
    print("  budget divided by it, and the slot count is what that budget buys.")
    print(f"  {'drawdown at entry':22s} {'survive':>9s} " +
          " ".join(f"{'at ' + format(f, '.1%'):>12s}" for f in RISK_RUNGS) +
          f" {'slots at 1%':>12s}")
    report["sizing"] = {}
    for label, chunk in buckets_of(rows):
        further = sorted(r["further"] for r in chunk)
        if len(further) < 25:
            continue
        tail = abs(further[int(0.05 * len(further))])
        if tail <= 0:
            continue
        sizes = [budget / tail for budget in RISK_RUNGS]
        report["sizing"][label] = {"tail": -tail,
                                   **{f"{f:.1%}": s for f, s in zip(RISK_RUNGS, sizes)}}
        print(f"  {label:22s} {tail:>9.1%} " +
              " ".join(f"{s:>11.1%}" for s in sizes) +
              f" {1.0 / sizes[1]:>11.0f}")

    print()
    print("########## how often each depth is available ##########")
    months = len({r["day"][:7] for r in rows})
    print(f"  {'threshold':24s} {'ETF-months':>12s} {'share':>8s} "
          f"{'months with any':>17s}")
    report["availability"] = {}
    for threshold in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
        hit = [r for r in rows if r["depth"] >= threshold]
        distinct = len({r["day"][:7] for r in hit})
        report["availability"][f"{threshold:.0%}"] = {
            "observations": len(hit), "share": len(hit) / len(rows),
            "month_share": distinct / months}
        print(f"  down {threshold:.0%} or more{'':10s} {len(hit):>12,d} "
              f"{len(hit)/len(rows):>8.1%} {distinct/months:>16.0%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print()
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
