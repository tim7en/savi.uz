"""What surrounds a 5% drawdown: the option surface, and everything else.

An event study rather than a backtest. Take every occasion a symbol first falls
5% below its running high, line the episodes up on that day, and average what
happened in the month either side.

Two halves with different reach, and it matters which is which.

*The option surface* can only be read over 2025-08 to 2026-07, because that is
all the chain history there is. Whatever it shows is one year of a benign tape
with no crisis in it, so a null result there means "not in this year" rather
than "not ever".

*The cross-asset and macro response* runs on the full 2015-2026 record, using the
21 non-equity ETFs -- metals, energy, broad commodity, duration, credit, FX and
volatility -- plus VIX. That window contains February 2018, the fourth quarter of
2018, March 2020 and 2022, so it can answer the question the option half cannot:
when equities fall, what actually rises.

Episodes are de-overlapped. A symbol in a long decline crosses -5% once and then
keeps falling, and counting each subsequent day as a fresh event would weight one
bear market as hundreds of observations. A minimum gap between events per symbol
keeps the count closer to the number of independent episodes -- the correction
that the CFTC study needed and the earnings-lag study got right.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

LEVERED = ("2X", "3X", "1.5X", "BULL", "BEAR", "LEVERAGED", "INVERSE",
           "ULTRA", "DAILY ", "SHORT ")

OPTION_COLUMNS = ["net_gex", "atm_iv", "iv_term_slope", "skew_25delta",
                  "put_call_oi", "put_call_volume", "gamma_flip_distance",
                  "zero_dte_share", "total_volume"]

SLEEVES = {"GLD": "metals", "SLV": "metals", "GDX": "metals", "PPLT": "metals",
           "USO": "energy", "UNG": "energy", "BNO": "energy",
           "DBC": "commodity", "DBA": "commodity", "GSG": "commodity",
           "TLT": "duration", "IEF": "duration", "SHY": "duration",
           "TIP": "duration", "LQD": "credit", "HYG": "credit",
           "UUP": "fx", "FXE": "fx", "FXY": "fx", "FXB": "fx",
           "VXX": "volatility"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path,
                        default=Path("data/intraday/bars_av.db"))
    parser.add_argument("--options", type=Path,
                        default=Path("data/options/alphavantage.db"))
    parser.add_argument("--cross", type=Path,
                        default=Path("data/data/cross_assets/etf_daily.db"))
    parser.add_argument("--macro", type=Path,
                        default=Path("data/data/macro/macro.db"))
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--window", type=int, default=21)
    parser.add_argument("--min-gap", type=int, default=42,
                        help="sessions between events on the same symbol")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--market", action="store_true",
                        help="key events on SPY drawdowns, not per-symbol ones")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/drawdown_events.json"))
    return parser.parse_args(argv)


def daily_closes(path, start):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        named = dict(connection.execute(
            "SELECT ticker, name FROM symbols WHERE name IS NOT NULL"))
    except sqlite3.OperationalError:
        named = {}
    drop = {t for t, n in named.items() if any(m in n.upper() for m in LEVERED)}
    series = defaultdict(dict)
    for ticker, ts, close in connection.execute(
        "SELECT ticker, ts, close FROM bars WHERE frequency='5min' AND ts>=? "
        "ORDER BY ticker, ts", (start,)):
        if ticker not in drop and close is not None:
            series[ticker][ts[:10]] = float(close)
    connection.close()
    return series


def find_events(closes, threshold, gap):
    """First crossing of -threshold from a running high, de-overlapped."""
    days = sorted(closes)
    events = []
    peak = closes[days[0]]
    armed = True
    last = -10 ** 9
    for index, day in enumerate(days):
        price = closes[day]
        if price >= peak:
            peak, armed = price, True
            continue
        if armed and price / peak - 1.0 <= -threshold and index - last >= gap:
            events.append((day, index))
            armed = False
            last = index
    return events, days


def main(argv=None) -> int:
    args = parse_args(argv)
    equities = daily_closes(args.bars, args.start)
    all_events = {}
    total = 0
    if args.market:
        closes = equities.get(args.benchmark)
        if not closes:
            raise SystemExit(f"error: no {args.benchmark} bars")
        events, days = find_events(closes, args.threshold, args.min_gap)
        all_events[args.benchmark] = (events, days)
        total = len(events)
        print(f"{args.benchmark}: {total} de-overlapped {args.threshold:.0%} "
              f"market drawdowns (minimum {args.min_gap} sessions apart)")
        print("  " + ", ".join(d for d, _ in events) + "\n")
    else:
        for ticker, closes in equities.items():
            events, days = find_events(closes, args.threshold, args.min_gap)
            if events:
                all_events[ticker] = (events, days)
                total += len(events)
        print(f"{len(all_events)} symbols, {total:,} de-overlapped "
              f"{args.threshold:.0%} drawdown events "
              f"(minimum {args.min_gap} sessions apart)\n")
    report = {"events": total, "symbols": len(all_events)}

    # ---- option surface around the event, 2025-08 to 2026-07 ----
    connection = sqlite3.connect(f"file:{args.options}?mode=ro", uri=True)
    options = defaultdict(dict)
    for row in connection.execute(
        "SELECT symbol, observation_date, " + ",".join(OPTION_COLUMNS)
            + " FROM av_daily"):
        options[row[0]][row[1]] = dict(zip(OPTION_COLUMNS, row[2:]))
    connection.close()

    offsets = list(range(-args.window, args.window + 1))
    paths = {c: {o: [] for o in offsets} for c in OPTION_COLUMNS}
    used = 0
    for ticker, (events, days) in all_events.items():
        table = options.get(ticker)
        if not table:
            continue
        for _, index in events:
            base = table.get(days[index])
            if base is None:
                continue
            used += 1
            for offset in offsets:
                j = index + offset
                if 0 <= j < len(days):
                    row = table.get(days[j])
                    if row:
                        for column in OPTION_COLUMNS:
                            value, anchor = row[column], base[column]
                            if value is None or anchor in (None, 0):
                                continue
                            paths[column][offset].append(value / anchor - 1.0)

    if used:
        print(f"=== option surface around the event ({used} events with chains) ===")
        print(f"  {'offset':>7s}" + "".join(
            f"{c.replace('_',' ')[:9]:>11s}" for c in
            ("atm_iv", "skew_25delta", "put_call_oi", "net_gex", "total_volume")))
        for offset in (-21, -10, -5, -1, 0, 1, 5, 10, 21):
            cells = []
            for column in ("atm_iv", "skew_25delta", "put_call_oi", "net_gex",
                           "total_volume"):
                values = paths[column][offset]
                cells.append(statistics.median(values) if values else float("nan"))
            print(f"  {offset:>+7d}" + "".join(f"{c:>10.1%} " for c in cells))
        report["option_paths"] = {
            c: {o: (statistics.median(v) if v else None)
                for o, v in paths[c].items()} for c in OPTION_COLUMNS}
    else:
        print("no option coverage overlapping these events")

    # ---- cross-asset response, full history ----
    connection = sqlite3.connect(f"file:{args.cross}?mode=ro", uri=True)
    cross = defaultdict(dict)
    for ticker, ts, close in connection.execute(
        "SELECT ticker, ts, close FROM bars WHERE frequency='daily' AND ts>=? "
        "ORDER BY ticker, ts", (args.start,)):
        if close is not None:
            cross[ticker][ts[:10]] = float(close)
    connection.close()

    dates = sorted({day for _, (events, _) in all_events.items()
                    for day, _ in events})
    print(f"\n=== cross-asset move over the month after the event "
          f"({len(dates):,} event dates, 2015-2026) ===")
    print(f"  {'sleeve':12s} {'ticker':>7s} {'-21d':>8s} {'-5d':>8s} "
          f"{'+5d':>8s} {'+21d':>8s} {'hit rate +21d':>14s}")
    rows = []
    for ticker, sleeve in sorted(SLEEVES.items(), key=lambda kv: (kv[1], kv[0])):
        table = cross.get(ticker)
        if not table:
            continue
        days = sorted(table)
        position = {d: i for i, d in enumerate(days)}
        buckets = {o: [] for o in (-21, -5, 5, 21)}
        for day in dates:
            i = position.get(day)
            if i is None:
                continue
            for offset in buckets:
                j = i + offset
                if 0 <= j < len(days):
                    if offset < 0:
                        buckets[offset].append(table[days[i]] / table[days[j]] - 1.0)
                    else:
                        buckets[offset].append(table[days[j]] / table[days[i]] - 1.0)
        if not buckets[21]:
            continue
        forward = buckets[21]
        row = {"ticker": ticker, "sleeve": sleeve,
               **{f"d{o}": statistics.median(v) if v else None
                  for o, v in buckets.items()},
               "hit_rate": sum(1 for v in forward if v > 0) / len(forward),
               "n": len(forward)}
        rows.append(row)
        print(f"  {sleeve:12s} {ticker:>7s} {row['d-21']:>7.1%} {row['d-5']:>7.1%} "
              f"{row['d5']:>7.1%} {row['d21']:>7.1%} {row['hit_rate']:>13.0%}")
    report["cross_asset"] = rows

    # ---- macro: VIX ----
    connection = sqlite3.connect(f"file:{args.macro}?mode=ro", uri=True)
    vix = {d: float(v) for d, v in connection.execute(
        "SELECT obs_date, value FROM observations WHERE series_id='VIXCLS' "
        "AND value IS NOT NULL")}
    connection.close()
    vdays = sorted(vix)
    vpos = {d: i for i, d in enumerate(vdays)}
    print(f"\n=== VIX around the event ===")
    print(f"  {'offset':>7s} {'median VIX':>12s} {'change vs t-21':>16s}")
    base_levels = []
    for offset in (-21, -5, -1, 0, 1, 5, 21):
        levels = []
        for day in dates:
            i = vpos.get(day)
            if i is None:
                continue
            j = i + offset
            if 0 <= j < len(vdays):
                levels.append(vix[vdays[j]])
        if not levels:
            continue
        median = statistics.median(levels)
        if offset == -21:
            base_levels = median
        print(f"  {offset:>+7d} {median:>12.1f} "
              f"{(median / base_levels - 1) if base_levels else 0:>15.1%}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
