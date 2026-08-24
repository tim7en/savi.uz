"""A tight trailing grid, outside crisis conditions, at the leverage proposed.

Two changes from the previous grid test, both requested and both material.

*The extremes come out.*  The earlier result leaned on 2008-09 and 2020, where
everything fell together and then recovered in a V.  Those are excluded by rule
rather than by naming dates: any entry taken while the market itself was more
than 20% below its own trailing high is dropped.  What remains is a name that
fell on its own account while the market was behaving normally, which is the
situation the strategy is actually for.

*The grid trails.*  A grid anchored at entry sells its whole position into a
recovery and then watches from the sidelines.  A trailing grid ratchets its
anchor upward: when price rises a level above the anchor, the anchor follows, so
the book keeps a position and keeps harvesting instead of being emptied by the
first sustained move.  That is the difference between a range trade and something
that can ride a recovery.

The spacing tested is deliberately tight -- 0.3% to 0.5% with twenty to thirty
levels -- which spans only six to fifteen percent of price.  Whether that band is
wide enough to matter is the point of the test: a grid that fills completely in
the first week is not harvesting oscillation, it is a leveraged long that took a
few days to establish itself.  So the fill statistics are reported beside the
returns, because they decide which of those two things it is.

Costs are maker on both legs.  Grid buys rest below the market and grid sells
rest above it, so neither has to cross the spread -- the one structural advantage
this family of strategies has over every breakout entry in this programme, and at
a 0.4% turnaround it is the difference between keeping seven-eighths of each turn
and keeping none of it.
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
SPACINGS = (0.003, 0.004, 0.005, 0.010)
LEVELS = (20, 30)
LEVERAGES = (3.0, 5.0)
LOOKBACK = 252
DECAYING = ("USO", "UNG", "GSG", "DBC", "DBA", "BNO", "VXX")
NON_EQUITY = ("FXB", "FXE", "FXY", "UUP", "HYG", "IEF", "LQD", "SHY",
              "TIP", "TLT", "GLD", "SLV", "PPLT")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path,
                        default=Path("data/cross_assets/etf_daily.db"))
    parser.add_argument("--market-ticker", default="DIA")
    parser.add_argument("--depth", type=float, default=0.25)
    parser.add_argument("--crisis", type=float, default=0.20,
                        help="market drawdown above which an entry is excluded")
    parser.add_argument("--maker-bp", type=float, default=2.5)
    parser.add_argument("--keep-crises", action="store_true")
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/trailing_grid.json"))
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


def drawdown_series(bars):
    closes = [b.close for b in bars]
    peak, running, out = [], [], []
    for i in range(len(bars)):
        running.append(closes[i])
        if len(running) > LOOKBACK:
            running.pop(0)
        peak.append(max(running))
        out.append(closes[i] / peak[i] - 1.0 if peak[i] > 0 else 0.0)
    return out


def trailing_grid(bars, start, horizon, spacing, levels, args):
    """Anchor ratchets up; buys rest below it, sells rest above the last fill."""
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    end = min(start + horizon, len(bars) - 1)
    entry = closes[start]
    if entry <= 0:
        return None
    fee = args.maker_bp / 10_000

    anchor = entry
    units = [entry]
    realised, trips, full_days, empty_days = 0.0, 0, 0, 0
    peak_units = 1
    for i in range(start + 1, end + 1):
        # Fills are taken off the CLOSE, not the high and low. A daily bar on an
        # equity ETF spans one to two percent, so a grid this tight would fill
        # several levels inside a single bar -- and crediting both the buys at
        # the low and the sells at the high assumes the price went down first
        # and then up, every single day. That assumption is worth several
        # hundred percent over three years and none of it is real.
        price = closes[i]
        while len(units) < levels and price <= anchor * (1 - spacing * len(units)):
            units.append(anchor * (1 - spacing * len(units)))
            peak_units = max(peak_units, len(units))
        while units and price >= units[-1] * (1 + spacing):
            bought = units.pop()
            realised += (price - bought) / entry - 2 * fee
            trips += 1
            anchor = max(anchor, price)
        if len(units) >= levels:
            full_days += 1
        if not units:
            empty_days += 1
    held = sum((closes[end] - u) / entry for u in units)
    days = max(end - start, 1)
    # Divide by the capital the grid actually had to fund. A thirty-level grid
    # deploys up to thirty units; summing their profits against one unit's price
    # reports a thirty-fold return on money that was never committed.
    committed = max(peak_units, 1)
    return {"realised": realised / committed, "held": held / committed,
            "total": (realised + held) / committed, "trips": trips,
            "peak_units": peak_units, "full": full_days / days,
            "empty": empty_days / days,
            "hold": closes[end] / entry - 1.0}


def main(argv=None) -> int:
    args = parse_args(argv)
    book = load(args)
    market = book.get(args.market_ticker)
    market_dd = {}
    if market:
        series = drawdown_series(market)
        market_dd = {b.timestamp: d for b, d in zip(market, series)}

    events, dropped = [], 0
    for ticker, bars in book.items():
        depth = drawdown_series(bars)
        armed = True
        for i in range(LOOKBACK, len(bars) - min(HORIZONS) - 1):
            if depth[i] > -args.depth * 0.6:
                armed = True
            if not armed or depth[i] > -args.depth:
                continue
            armed = False
            if not args.keep_crises:
                if market_dd.get(bars[i].timestamp, 0.0) <= -args.crisis:
                    dropped += 1
                    continue
            events.append((ticker, i))
    print(f"{len(book)} equity ETFs, {len(events):,d} entries at "
          f"{args.depth:.0%} below the trailing high")
    print(f"excluded {dropped} taken while the market itself was more than "
          f"{args.crisis:.0%} down")
    if events:
        years = sorted({book[t][i].timestamp[:4] for t, i in events})
        print(f"entry years: {', '.join(years)}")
    report = {"entries": len(events), "excluded_crisis": dropped}

    print()
    print("########## the tight trailing grid, outside crises ##########")
    print("  'full' is the share of days every level was already filled -- a grid")
    print("  that is always full is a leveraged long, not a harvest.")
    print(f"  {'spacing':>8s} {'levels':>7s} {'band':>7s} {'hold 3y':>9s} "
          f"{'grid 3y':>9s} {'trips':>7s} {'full':>7s} {'empty':>7s}")
    best = None
    for spacing in SPACINGS:
        for levels in LEVELS:
            parts = [trailing_grid(book[t], i, 756, spacing, levels, args)
                     for t, i in events]
            parts = [p for p in parts if p]
            if len(parts) < 20:
                continue
            hold = statistics.median(p["hold"] for p in parts)
            total = statistics.median(p["total"] for p in parts)
            row = {"n": len(parts), "hold": hold, "grid": total,
                   "trips": statistics.median(p["trips"] for p in parts),
                   "full": statistics.fmean(p["full"] for p in parts),
                   "empty": statistics.fmean(p["empty"] for p in parts),
                   "band": spacing * levels}
            report.setdefault("grid", {})[f"{spacing:.1%}x{levels}"] = row
            print(f"  {spacing:>8.1%} {levels:>7d} {spacing*levels:>7.1%} "
                  f"{hold:>9.1%} {total:>9.1%} {row['trips']:>7.0f} "
                  f"{row['full']:>7.0%} {row['empty']:>7.0%}")
            if best is None or total > best[2]:
                best = (spacing, levels, total)

    if best:
        spacing, levels, _ = best
        print()
        print(f"########## at {spacing:.1%} x {levels} levels, with leverage "
              f"##########")
        print("  Return is on committed capital; leverage multiplies both the")
        print("  return and the worst mark against you.")
        print(f"  {'leverage':>9s} {'grid 3y':>10s} {'hold 3y':>10s} "
              f"{'grid worst':>11s} {'hold worst':>11s}")
        report["leverage"] = {}
        parts = [(t, i, trailing_grid(book[t], i, 756, spacing, levels, args))
                 for t, i in events]
        parts = [(t, i, p) for t, i, p in parts if p]
        worst_hold = []
        for ticker, start, _ in parts:
            bars = book[ticker]
            end = min(start + 756, len(bars) - 1)
            entry = bars[start].close
            worst_hold.append(min(b.low for b in bars[start:end + 1]) / entry - 1.0)
        for leverage in LEVERAGES:
            grid_return = statistics.median(p["total"] for _, _, p in parts) * leverage
            hold_return = statistics.median(p["hold"] for _, _, p in parts) * leverage
            grid_worst = statistics.median(worst_hold) * leverage
            report["leverage"][f"{leverage:g}x"] = {
                "grid": grid_return, "hold": hold_return, "worst": grid_worst}
            print(f"  {leverage:>8.0f}x {grid_return:>10.1%} {hold_return:>10.1%} "
                  f"{grid_worst:>11.1%} {grid_worst:>11.1%}")
        tail = sorted(worst_hold)[int(0.05 * len(worst_hold))]
        print()
        print(f"  worst 5% of paths fall {tail:.1%} below entry before recovering")
        for leverage in LEVERAGES:
            print(f"    at {leverage:g}x that is {tail * leverage:.0%} of the "
                  f"position{'  -- wiped out' if tail * leverage <= -1 else ''}")
        report["tail"] = {"worst_5pct": tail}

    print()
    print("########## and the same grid with crises left in ##########")
    if best and not args.keep_crises:
        args.keep_crises = True
        spacing, levels, _ = best
        crisis_events = []
        for ticker, bars in book.items():
            depth = drawdown_series(bars)
            armed = True
            for i in range(LOOKBACK, len(bars) - min(HORIZONS) - 1):
                if depth[i] > -args.depth * 0.6:
                    armed = True
                if armed and depth[i] <= -args.depth:
                    armed = False
                    crisis_events.append((ticker, i))
        parts = [trailing_grid(book[t], i, 756, spacing, levels, args)
                 for t, i in crisis_events]
        parts = [p for p in parts if p]
        if parts:
            print(f"  all {len(parts)} entries: hold "
                  f"{statistics.median(p['hold'] for p in parts):.1%}, grid "
                  f"{statistics.median(p['total'] for p in parts):.1%}")
            report["with_crises"] = {
                "n": len(parts),
                "hold": statistics.median(p["hold"] for p in parts),
                "grid": statistics.median(p["total"] for p in parts)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print()
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
