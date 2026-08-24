"""A centred grid: does farming the turns actually buy you margin?

The previous grid model was wrong in a way that mattered.  It placed every level
below the entry, so the book committed its whole capital into the fall and never
sold anything on the way up.  That is not a grid, it is a scaled entry.

A centred grid holds a base position with half its levels below and half above.
Buys rest under the market, sells rest over it, and the position oscillates
around the base rather than accumulating in one direction.  Two consequences,
and the second is the one under test:

*Peak inventory is roughly half.*  Only the lower half of the ladder can fill, so
the capital at risk when everything goes wrong is much smaller than a
fully-below grid of the same level count.

*Every completed turn banks realised profit.*  That cash sits in the account and
cushions the unrealised mark, so the further fall required to close the position
grows with every turn farmed.  This is the claim: farming adjusts the margin.

So the comparison is a leveraged buy-and-hold against a centred grid at the same
gross leverage and the same entry, and what is measured is not return but
survival -- the worst equity mark, and how often the account is closed out.

Fills come off the close, never off the high and low of the same bar.  A tight
grid on daily data would otherwise buy every low and sell every high, which
assumes the price fell then rose on every single day and is worth several hundred
percent of imaginary return.  That convention understates the harvest, so the
cushion measured here is a floor rather than an estimate.
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
LEVERAGES = (2.0, 3.0, 5.0)
SPACINGS = (0.003, 0.005, 0.010)
LEVELS = 30
DECAYING = ("USO", "UNG", "GSG", "DBC", "DBA", "BNO", "VXX")
NON_EQUITY = ("FXB", "FXE", "FXY", "UUP", "HYG", "IEF", "LQD", "SHY",
              "TIP", "TLT", "GLD", "SLV", "PPLT")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path,
                        default=Path("data/cross_assets/etf_daily.db"))
    parser.add_argument("--depth", type=float, default=0.20)
    parser.add_argument("--levels", type=int, default=LEVELS)
    parser.add_argument("--maker-bp", type=float, default=2.5)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/centred_grid_margin.json"))
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


def entry_points(bars, depth):
    closes = [b.close for b in bars]
    peak, running, out = [], [], []
    for i in range(len(bars)):
        running.append(closes[i])
        if len(running) > LOOKBACK:
            running.pop(0)
        peak.append(max(running))
    armed = True
    for i in range(LOOKBACK, len(bars) - 60):
        if peak[i] <= 0:
            continue
        drop = closes[i] / peak[i] - 1.0
        if drop > -depth * 0.6:
            armed = True
        if armed and drop <= -depth:
            armed = False
            out.append(i)
    return out


def run_grid(bars, start, spacing, levels, leverage, args):
    """Centred grid at ``leverage`` gross. Equity is capital plus realised plus
    unrealised; the account is closed the first time equity reaches zero."""
    closes = [b.close for b in bars]
    end = min(start + HORIZON, len(bars) - 1)
    entry = closes[start]
    if entry <= 0:
        return None
    fee = args.maker_bp / 10_000
    half = levels // 2

    # capital 1.0; gross notional at full inventory is `leverage`
    unit = leverage / levels          # notional per slot, as a share of capital
    held = [entry] * half             # base position: the upper half can be sold
    low_ref = entry
    realised, trips = 0.0, 0
    worst_equity, closed_on = 1.0, None

    for i in range(start + 1, end + 1):
        price = closes[i]
        # buys rest below the lowest fill, sells rest above the highest
        # low_ref survives an empty ladder: once every unit has been sold the
        # grid still needs a level to buy back at, and deriving it from `held`
        # crashes exactly when the grid has worked best.
        while len(held) < levels and price <= low_ref * (1 - spacing):
            low_ref = low_ref * (1 - spacing)
            held.append(low_ref)
        while held and price >= max(held) * (1 + spacing):
            bought = max(held)
            held.remove(bought)
            low_ref = min(held) if held else bought
            realised += unit * ((price - bought) / bought - 2 * fee)
            trips += 1
        unrealised = sum(unit * (price - u) / u for u in held)
        equity = 1.0 + realised + unrealised
        if equity < worst_equity:
            worst_equity = equity
        if equity <= 0 and closed_on is None:
            closed_on = i - start
    final = closes[end]
    unrealised = sum(unit * (final - u) / u for u in held)
    return {"worst_equity": worst_equity, "closed": closed_on is not None,
            "realised": realised, "trips": trips,
            "final": 1.0 + realised + unrealised,
            "peak_units": levels if len(held) >= levels else max(len(held), half)}


def run_hold(bars, start, leverage):
    closes = [b.close for b in bars]
    lows = [b.low for b in bars]
    end = min(start + HORIZON, len(bars) - 1)
    entry = closes[start]
    floor = min(lows[start + 1:end + 1] or [entry])
    worst = 1.0 + leverage * (floor / entry - 1.0)
    return {"worst_equity": worst, "closed": worst <= 0,
            "final": 1.0 + leverage * (closes[end] / entry - 1.0)}


def main(argv=None) -> int:
    args = parse_args(argv)
    book = load(args)
    events = [(t, i) for t, bars in book.items()
              for i in entry_points(bars, args.depth)]
    print(f"{len(book)} equity ETFs, {len(events):,d} entries at "
          f"{args.depth:.0%} below the trailing high")
    print(f"centred grid: {args.levels} levels, {args.levels // 2} below the "
          f"entry and {args.levels // 2} above, maker {args.maker_bp:g}bp a leg")
    print("fills off the close only, so the harvest is a floor not an estimate")
    report = {"entries": len(events), "levels": args.levels}

    print()
    print("########## closed out: leveraged hold against a centred grid ##########")
    print(f"  {'leverage':>9s} {'spacing':>9s} {'hold closed':>12s} "
          f"{'grid closed':>12s} {'hold worst':>11s} {'grid worst':>11s} "
          f"{'turns':>7s}")
    for leverage in LEVERAGES:
        holds = [run_hold(book[t], i, leverage) for t, i in events]
        hold_closed = sum(1 for h in holds if h["closed"]) / len(holds)
        hold_worst = statistics.median(h["worst_equity"] for h in holds)
        for spacing in SPACINGS:
            grids = [run_grid(book[t], i, spacing, args.levels, leverage, args)
                     for t, i in events]
            grids = [g for g in grids if g]
            if not grids:
                continue
            closed = sum(1 for g in grids if g["closed"]) / len(grids)
            worst = statistics.median(g["worst_equity"] for g in grids)
            turns = statistics.median(g["trips"] for g in grids)
            report.setdefault("survival", {})[f"{leverage:g}x|{spacing:.1%}"] = {
                "hold_closed": hold_closed, "grid_closed": closed,
                "hold_worst": hold_worst, "grid_worst": worst, "turns": turns}
            print(f"  {leverage:>8.0f}x {spacing:>9.1%} {hold_closed:>12.0%} "
                  f"{closed:>12.0%} {hold_worst:>11.2f} {worst:>11.2f} "
                  f"{turns:>7.0f}")

    print()
    print("########## what the farming actually banks ##########")
    print("  Realised cash as a share of capital, which is the cushion under the")
    print("  mark. If it is small the grid has not adjusted any margin.")
    print(f"  {'leverage':>9s} {'spacing':>9s} {'realised':>10s} "
          f"{'turns':>7s} {'final equity':>13s}")
    report["farming"] = {}
    for leverage in LEVERAGES:
        for spacing in SPACINGS:
            grids = [run_grid(book[t], i, spacing, args.levels, leverage, args)
                     for t, i in events]
            grids = [g for g in grids if g]
            if not grids:
                continue
            report["farming"][f"{leverage:g}x|{spacing:.1%}"] = {
                "realised": statistics.median(g["realised"] for g in grids),
                "turns": statistics.median(g["trips"] for g in grids),
                "final": statistics.median(g["final"] for g in grids)}
            print(f"  {leverage:>8.0f}x {spacing:>9.1%} "
                  f"{statistics.median(g['realised'] for g in grids):>10.1%} "
                  f"{statistics.median(g['trips'] for g in grids):>7.0f} "
                  f"{statistics.median(g['final'] for g in grids):>13.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print()
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
