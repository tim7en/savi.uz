"""The whole rule, end to end: entry, exit, tail, compounding, and what kind of fall.

Everything measured so far has answered one piece at a time and held the rest
fixed.  Entry depth was swept while the exit stayed at three years.  Returns were
reported per episode and never compounded.  The tail was summarised at a single
percentile.  And no test asked *why* the thing had fallen, which is probably the
most important question of the lot.

Five things, in the order they decide whether this is a strategy.

*What kind of fall it is.*  A name that doubled and then gave back 30% is a
bubble deflating; one that fell 30% from an ordinary level is an overcorrection.
Those should not behave alike, and the run-up over the two years before the peak
separates them cleanly and in advance.

*The exit.*  Fixed holds are an assumption, not a rule.  Recovering to the prior
high, a trailing stop, and holding on are compared on identical entries.

*The tail.*  Not one percentile but the whole left side, because "worst 5%" hides
whether the losses are merely bad or unrecoverable.

*The compounding.*  A portfolio simulation rather than an average of episodes:
hold whatever qualifies each month, equally weighted, at leverage, and pay
financing on the borrowed part -- which every previous number in this programme
omitted and which at 3x is worth more than the entire grid harvest.

*The frequency.*  How often the opportunity exists at all, since a rule that
fires twice a decade is a different instrument from one that fires monthly.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sqlite3
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

LOOKBACK = 252
RUNUP = 504
HORIZON = 756
DECAYING = ("USO", "UNG", "GSG", "DBC", "DBA", "BNO", "VXX")
NON_EQUITY = ("FXB", "FXE", "FXY", "UUP", "HYG", "IEF", "LQD", "SHY",
              "TIP", "TLT", "GLD", "SLV", "PPLT")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path,
                        default=Path("data/cross_assets/etf_daily.db"))
    parser.add_argument("--depth", type=float, default=0.18)
    parser.add_argument("--leverage", type=float, default=3.0)
    parser.add_argument("--financing", type=float, default=0.05,
                        help="annual cost of the borrowed portion")
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/etf_rule_complete.json"))
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


def events(book, depth):
    """Entries, each carrying the run-up that preceded the peak it fell from."""
    out = []
    for ticker, bars in book.items():
        closes = [b.close for b in bars]
        lows = [b.low for b in bars]
        peak, running, peak_at = [], [], []
        for i in range(len(bars)):
            running.append((closes[i], i))
            if len(running) > LOOKBACK:
                running.pop(0)
            best = max(running)
            peak.append(best[0])
            peak_at.append(best[1])
        armed = True
        for i in range(LOOKBACK + RUNUP, len(bars) - 60):
            if peak[i] <= 0:
                continue
            drop = closes[i] / peak[i] - 1.0
            if drop > -depth * 0.6:
                armed = True
            if not armed or drop > -depth:
                continue
            armed = False
            top = peak_at[i]
            before = top - RUNUP
            if before < 0 or closes[before] <= 0:
                continue
            out.append({"ticker": ticker, "index": i, "day": bars[i].timestamp,
                        "depth": -drop,
                        "runup": closes[top] / closes[before] - 1.0,
                        "peak_price": peak[i], "peak_index": top})
    return out


def outcomes(bars, event, args):
    """Every exit rule scored on the same entry and the same path."""
    closes = [b.close for b in bars]
    lows = [b.low for b in bars]
    start = event["index"]
    entry = closes[start]
    end = min(start + HORIZON, len(bars) - 1)
    out = {"hold_3y": closes[end] / entry - 1.0}

    # exit on regaining the high it fell from
    target = event["peak_price"]
    out["to_prior_high"] = out["hold_3y"]
    for i in range(start + 1, end + 1):
        if closes[i] >= target:
            out["to_prior_high"] = closes[i] / entry - 1.0
            out["days_to_high"] = i - start
            break

    # 25% trailing stop from the running high after entry
    best, stopped = entry, None
    for i in range(start + 1, end + 1):
        best = max(best, closes[i])
        if closes[i] <= best * 0.75:
            stopped = closes[i] / entry - 1.0
            break
    out["trailing_25"] = stopped if stopped is not None else out["hold_3y"]

    floor = min(lows[start + 1:end + 1] or [entry])
    out["worst"] = floor / entry - 1.0
    return out


def describe(values):
    values = sorted(values)
    def at(share):
        return values[min(int(share * len(values)), len(values) - 1)]
    return {"n": len(values), "p01": at(0.01), "p05": at(0.05), "p25": at(0.25),
            "median": statistics.median(values), "p75": at(0.75),
            "p95": at(0.95), "mean": statistics.fmean(values)}


def main(argv=None) -> int:
    args = parse_args(argv)
    book = load(args)
    rows = events(book, args.depth)
    for r in rows:
        r.update(outcomes(book[r["ticker"]], r, args))
    print(f"{len(book)} equity ETFs, {len(rows):,d} entries at "
          f"{args.depth:.0%} below the 252-day high, "
          f"{rows[0]['day'][:7]} to {rows[-1]['day'][:7]}")
    report = {"entries": len(rows), "depth": args.depth}

    print()
    print("########## overcorrection or bubble? ##########")
    print("  Split by the two-year run-up into the peak it then fell from.")
    print(f"  {'run-up before the peak':26s} {'n':>6s} {'med 3y':>9s} "
          f"{'worst 5%':>10s} {'win':>7s}")
    ranked = sorted(rows, key=lambda r: r["runup"])
    quarter = len(ranked) // 4
    report["runup"] = {}
    for q in range(4):
        chunk = ranked[q * quarter:(q + 1) * quarter]
        if len(chunk) < 15:
            continue
        span = f"{chunk[0]['runup']:+.0%} to {chunk[-1]['runup']:+.0%}"
        got = describe([r["hold_3y"] for r in chunk])
        wins = sum(1 for r in chunk if r["hold_3y"] > 0) / len(chunk)
        tail = describe([r["worst"] for r in chunk])["p05"]
        report["runup"][span] = {**got, "win": wins, "worst_p05": tail}
        print(f"  {span:26s} {len(chunk):>6d} {got['median']:>+9.1%} "
              f"{tail:>10.1%} {wins:>7.0%}")

    print()
    print("########## the exit rule ##########")
    print("  Same entries, same paths, three ways out.")
    print(f"  {'exit':26s} {'median':>9s} {'mean':>9s} {'p05':>9s} {'win':>7s}")
    report["exits"] = {}
    for key, text in (("hold_3y", "hold three years"),
                      ("to_prior_high", "sell at the prior high"),
                      ("trailing_25", "25% trailing stop")):
        values = [r[key] for r in rows]
        got = describe(values)
        wins = sum(1 for v in values if v > 0) / len(values)
        report["exits"][text] = {**got, "win": wins}
        print(f"  {text:26s} {got['median']:>+9.1%} {got['mean']:>+9.1%} "
              f"{got['p05']:>+9.1%} {wins:>7.0%}")
    reached = [r for r in rows if "days_to_high" in r]
    if reached:
        print(f"  -> regained the prior high in {len(reached)/len(rows):.0%} of "
              f"cases, median {statistics.median(r['days_to_high'] for r in reached):.0f} "
              f"sessions")

    print()
    print("########## the tail ##########")
    print("  Three-year outcome and worst mark, unlevered and at "
          f"{args.leverage:g}x.")
    print(f"  {'':20s} {'p01':>9s} {'p05':>9s} {'p25':>9s} {'median':>9s} "
          f"{'p75':>9s} {'p95':>9s}")
    for key, text in (("hold_3y", "3-year outcome"), ("worst", "worst mark")):
        got = describe([r[key] for r in rows])
        report.setdefault("tail", {})[text] = got
        print(f"  {text:20s} {got['p01']:>+9.1%} {got['p05']:>+9.1%} "
              f"{got['p25']:>+9.1%} {got['median']:>+9.1%} {got['p75']:>+9.1%} "
              f"{got['p95']:>+9.1%}")
    worst = describe([r["worst"] for r in rows])
    print(f"  at {args.leverage:g}x the worst mark becomes "
          f"p05 {worst['p05'] * args.leverage:.0%}, "
          f"p01 {worst['p01'] * args.leverage:.0%}")

    print()
    print("########## how often the market is displaced ##########")
    calendar = sorted({b.timestamp for bars in book.values() for b in bars})
    months = sorted({d[:7] for d in calendar})
    depth_by_day = {}
    for ticker, bars in book.items():
        closes = [b.close for b in bars]
        running = []
        for i, bar in enumerate(bars):
            running.append(closes[i])
            if len(running) > LOOKBACK:
                running.pop(0)
            top = max(running)
            if top > 0:
                depth_by_day.setdefault(bar.timestamp, {})[ticker] = \
                    closes[i] / top - 1.0
    print(f"  {'threshold':16s} {'share of name-days':>20s} "
          f"{'months with any':>17s}")
    report["frequency"] = {}
    for threshold in (0.10, 0.18, 0.25, 0.35, 0.50):
        total = sum(len(v) for v in depth_by_day.values())
        hit = sum(1 for v in depth_by_day.values()
                  for d in v.values() if d <= -threshold)
        with_any = len({day[:7] for day, v in depth_by_day.items()
                        if any(d <= -threshold for d in v.values())})
        report["frequency"][f"{threshold:.0%}"] = {
            "share": hit / total, "months": with_any / len(months)}
        print(f"  down {threshold:.0%} or more{'':2s} {hit/total:>19.1%} "
              f"{with_any/len(months):>16.0%}")

    print()
    print(f"########## compounding it, {args.leverage:g}x with financing "
          f"##########")
    print(f"  Hold every qualifying name, equal weight, up to {args.slots} slots,")
    print(f"  paying {args.financing:.0%} a year on the borrowed portion.")
    per_day = defaultdict(float)
    for ticker, bars in book.items():
        for i in range(1, len(bars)):
            if bars[i - 1].close > 0:
                per_day[bars[i].timestamp] = per_day.get(bars[i].timestamp, 0.0)
    days = sorted(depth_by_day)
    report["compounding"] = {}
    for leverage in (1.0, 2.0, args.leverage):
        nav, peak, worst = 1.0, 1.0, 0.0
        prices = {t: {b.timestamp: b.close for b in bars}
                  for t, bars in book.items()}
        for a, b in zip(days, days[1:]):
            qualifying = [t for t, d in depth_by_day.get(a, {}).items()
                          if d <= -args.depth][:args.slots]
            if qualifying:
                moves = []
                for t in qualifying:
                    p0, p1 = prices[t].get(a), prices[t].get(b)
                    if p0 and p1 and p0 > 0:
                        moves.append(p1 / p0 - 1.0)
                gross = statistics.fmean(moves) if moves else 0.0
                carry = (leverage - 1.0) * args.financing / 252.0
                nav *= (1.0 + leverage * gross - carry)
            peak = max(peak, nav)
            worst = min(worst, nav / peak - 1.0)
        years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25
        cagr = nav ** (1 / years) - 1 if nav > 0 else -1.0
        report["compounding"][f"{leverage:g}x"] = {
            "cagr": cagr, "max_drawdown": worst, "terminal": nav}
        print(f"  {leverage:>4.0f}x   CAGR {cagr:>+7.1%}   max drawdown "
              f"{worst:>7.1%}   {nav:>8.2f}x money")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print()
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
