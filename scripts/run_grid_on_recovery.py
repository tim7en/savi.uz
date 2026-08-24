"""Does a narrow grid add to a deep-drawdown recovery, or eat it?

The objection was that a grid and a value entry want opposite things: the grid is
paid for oscillation, the entry is paid for a one-way recovery, and every level
the price re-crosses on the way up is a level the grid already sold.  That
objection is only correct for a grid that sells its whole position.  A grid with
a held core sells nothing it needs to keep, and harvests whatever chop happens
around the recovery it is already long.

Whether that actually adds anything is arithmetic, and this measures it against
the path each recovery really took.

The construction.  At entry the book buys a core it never sells, plus one grid
unit.  Below the last fill it adds a unit every ``spacing`` percent, up to a cap.
Above the last fill it sells a unit at the same interval.  The core rides the
recovery; the grid units churn.  At the horizon everything still held is marked
to market and added to what the churn realised.

Costs are charged as **maker on both legs**, which is the grid's one real
structural advantage and the reason it deserves its own test: its buys are
resting limits below the market and its sells are resting limits above, so unlike
every breakout entry in this programme neither leg has to cross the spread.

Three things get reported, because a grid can improve the return and still be the
wrong choice: what it adds over simply holding, how many round trips it needs to
do it, and how much inventory it accumulates -- since a grid that helps by
carrying four units at the bottom is a leverage decision wearing a strategy
costume.
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
SPACINGS = (0.02, 0.035, 0.05, 0.08, 0.12)
LOOKBACK = 252
DECAYING = ("USO", "UNG", "GSG", "DBC", "DBA", "BNO", "VXX")
NON_EQUITY = ("FXB", "FXE", "FXY", "UUP", "HYG", "IEF", "LQD", "SHY",
              "TIP", "TLT", "GLD", "SLV", "PPLT")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path,
                        default=Path("data/cross_assets/etf_daily.db"))
    parser.add_argument("--depth", type=float, default=0.25,
                        help="how far below the high the entry is taken")
    parser.add_argument("--core", type=float, default=1.0,
                        help="units held and never sold")
    parser.add_argument("--max-units", type=int, default=4,
                        help="grid units on top of the core")
    parser.add_argument("--maker-bp", type=float, default=2.5)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/grid_on_recovery.json"))
    known, _ = parser.parse_known_args(argv)
    return known


def load(args):
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    book = {}
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency='daily' "
            "ORDER BY ticker"):
        if ticker in DECAYING or ticker in NON_EQUITY:
            continue
        rows = connection.execute(
            "SELECT ts,open,high,low,close FROM bars WHERE ticker=? AND "
            "frequency='daily' ORDER BY ts", (ticker,)).fetchall()
        bars = adjust_bars([Bar(r[0][:10], r[1], r[2], r[3], r[4], None)
                            for r in rows], splits.get(ticker, []))
        if len(bars) >= 1500:
            book[ticker] = bars
    connection.close()
    return book


def entries(bars, depth):
    """One entry per fresh crossing below ``depth`` under the trailing high."""
    closes = [b.close for b in bars]
    peak, running, out = [], [], []
    for i in range(len(bars)):
        running.append(closes[i])
        if len(running) > LOOKBACK:
            running.pop(0)
        peak.append(max(running))
    armed = True
    for i in range(LOOKBACK, len(bars) - min(HORIZONS) - 1):
        if peak[i] <= 0:
            continue
        drop = closes[i] / peak[i] - 1.0
        if drop > -depth * 0.6:
            armed = True
        if armed and drop <= -depth:
            armed = False
            out.append(i)
    return out


def simulate(bars, start, horizon, spacing, args):
    """Core plus grid, walked along the real path. Returns total return on the
    capital actually committed, and what the churn cost and carried."""
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    end = min(start + horizon, len(bars) - 1)
    entry = closes[start]
    if entry <= 0:
        return None
    fee = args.maker_bp / 10_000

    core = args.core
    units = [entry]                      # grid inventory, one unit at entry
    last = entry
    realised, trips, spent, peak_units = 0.0, 0, args.core + 1.0, 1
    for i in range(start + 1, end + 1):
        # additions first: a bar that falls through several levels fills them all
        while len(units) < args.max_units and lows[i] <= last * (1 - spacing):
            last = last * (1 - spacing)
            units.append(last)
            spent += 1.0
            peak_units = max(peak_units, len(units))
        # then reductions, at the same interval above the newest fill
        while units and highs[i] >= last * (1 + spacing):
            price = last * (1 + spacing)
            bought = units.pop()
            realised += (price - bought) / entry - 2 * fee
            trips += 1
            last = price
    held = sum((closes[end] - u) / entry for u in units)
    core_return = core * (closes[end] / entry - 1.0)
    total = core_return + realised + held
    committed = args.core + peak_units
    return {"total": total / committed,
            "hold": (closes[end] / entry - 1.0),
            "realised": realised / committed,
            "trips": trips, "peak_units": peak_units,
            "committed": committed}


def main(argv=None) -> int:
    args = parse_args(argv)
    book = load(args)
    events = [(t, i) for t, bars in book.items() for i in entries(bars, args.depth)]
    print(f"{len(book)} equity ETFs, {len(events):,d} entries at "
          f"{args.depth:.0%} below the trailing high")
    print(f"core {args.core:g} unit held throughout, up to {args.max_units} grid "
          f"units, maker {args.maker_bp:g}bp a leg\n")
    report = {"etfs": len(book), "entries": len(events), "depth": args.depth}

    print("########## does the grid add to the recovery? ##########")
    print("  Buy-and-hold is the same entry with no grid at all. Both are")
    print("  returns on the capital actually committed, so the comparison is")
    print("  not one unit against five.")
    print(f"  {'grid spacing':16s} " +
          " ".join(f"{'hold ' + LABELS[h]:>11s} {'grid ' + LABELS[h]:>11s}"
                   for h in (756,)) +
          f" {'edge':>8s} {'trips':>7s} {'peak':>6s}")
    for spacing in SPACINGS:
        holds, grids, trips, peaks = [], [], [], []
        for ticker, start in events:
            got = simulate(book[ticker], start, 756, spacing, args)
            if got is None:
                continue
            holds.append(got["hold"])
            grids.append(got["total"])
            trips.append(got["trips"])
            peaks.append(got["peak_units"])
        if len(holds) < 30:
            continue
        hold_median = statistics.median(holds)
        grid_median = statistics.median(grids)
        report.setdefault("spacings", {})[f"{spacing:.1%}"] = {
            "n": len(holds), "hold": hold_median, "grid": grid_median,
            "edge": grid_median - hold_median,
            "trips": statistics.median(trips),
            "peak_units": statistics.median(peaks)}
        print(f"  {spacing:>15.1%} {hold_median:>11.1%} {grid_median:>11.1%} "
              f"{grid_median - hold_median:>+8.1%} "
              f"{statistics.median(trips):>7.0f} "
              f"{statistics.median(peaks):>6.1f}")

    print()
    print("########## where the grid's return comes from ##########")
    print("  Realised churn against the core it is riding. If the churn is small")
    print("  the grid is a sizing decision, not a strategy.")
    best = None
    for spacing in SPACINGS:
        parts = [simulate(book[t], i, 756, spacing, args) for t, i in events]
        parts = [p for p in parts if p]
        if len(parts) < 30:
            continue
        churn = statistics.median(p["realised"] for p in parts)
        total = statistics.median(p["total"] for p in parts)
        share = churn / total if total else float("nan")
        committed = statistics.median(p["committed"] for p in parts)
        report.setdefault("attribution", {})[f"{spacing:.1%}"] = {
            "churn": churn, "total": total, "share": share,
            "committed": committed}
        print(f"  spacing {spacing:>5.1%}: churn {churn:>+7.1%} of a total "
              f"{total:>+7.1%}  ({share:>5.0%} of it), "
              f"{committed:.1f} units committed")
        if best is None or total > best[1]:
            best = (spacing, total)

    print()
    print("########## the same at other horizons ##########")
    if best:
        spacing = best[0]
        print(f"  at the best spacing found above, {spacing:.1%}")
        print(f"  {'horizon':16s} {'hold':>10s} {'grid':>10s} {'edge':>9s} "
              f"{'grid wins':>10s}")
        report["horizons"] = {}
        for horizon in HORIZONS:
            parts = [(simulate(book[t], i, horizon, spacing, args))
                     for t, i in events]
            parts = [p for p in parts if p]
            if len(parts) < 30:
                continue
            holds = [p["hold"] for p in parts]
            grids = [p["total"] for p in parts]
            wins = sum(1 for h, g in zip(holds, grids) if g > h) / len(parts)
            report["horizons"][LABELS[horizon]] = {
                "hold": statistics.median(holds), "grid": statistics.median(grids),
                "win": wins}
            print(f"  {LABELS[horizon]:16s} {statistics.median(holds):>10.1%} "
                  f"{statistics.median(grids):>10.1%} "
                  f"{statistics.median(grids) - statistics.median(holds):>+9.1%} "
                  f"{wins:>10.0%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print()
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
