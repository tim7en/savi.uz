"""The rule as a portfolio: correct holds, risk-based sizing, real exposure.

The previous simulation tested a rule nobody proposed.  It held a name only while
that name was still 18% below its high, so it sold the moment the drawdown closed
-- cutting off exactly the recovery the episode tests were measuring -- and then
rotated into whatever else was falling.  That produced a -80% drawdown at 1x and
it deserved to.

This holds what the rule says to hold.  A position opens on a fresh crossing and
stays open until the price regains the high it fell from or three years pass,
whichever comes first, which is what the exit comparison found best.

Sizing is by risk rather than by equal weight.  The worst mark after entry sits
near -60% at the fifth percentile, so a position sized to lose ``risk`` of the
account in that case is ``risk / 0.60`` of NAV.  At 1% that is under two percent
of the book per position, which is what the tail arithmetic permits and is far
smaller than intuition suggests.

Gross exposure is then an output, not a setting, and it is reported alongside the
return because a strategy that reaches 40% invested is a different instrument
from one that reaches 300%, whatever their CAGRs look like.  Financing is charged
on any borrowed portion.

The entry carries the run-up filter the previous run found: a fall from a peak
that had itself risen more than about a third in two years is a bubble deflating
rather than an overcorrection, and it returned less with a worse tail.
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
MAX_HOLD = 756
TAIL = 0.60                       # p05 worst mark, the sizing denominator
RISK_RUNGS = (0.01, 0.02, 0.03)
DECAYING = ("USO", "UNG", "GSG", "DBC", "DBA", "BNO", "VXX")
NON_EQUITY = ("FXB", "FXE", "FXY", "UUP", "HYG", "IEF", "LQD", "SHY",
              "TIP", "TLT", "GLD", "SLV", "PPLT")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path,
                        default=Path("data/cross_assets/etf_daily.db"))
    parser.add_argument("--depth", type=float, default=0.18)
    parser.add_argument("--max-runup", type=float, default=0.35)
    parser.add_argument("--financing", type=float, default=0.05)
    parser.add_argument("--no-runup-filter", action="store_true")
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/etf_portfolio.json"))
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


def signals(book, args):
    """Fresh crossings, with the peak to recover to and the run-up before it."""
    out = defaultdict(list)
    for ticker, bars in book.items():
        closes = [b.close for b in bars]
        peak, running = [], []
        for i in range(len(bars)):
            running.append((closes[i], i))
            if len(running) > LOOKBACK:
                running.pop(0)
            peak.append(max(running))
        armed = True
        for i in range(LOOKBACK + RUNUP, len(bars)):
            top, top_at = peak[i]
            if top <= 0:
                continue
            drop = closes[i] / top - 1.0
            if drop > -args.depth * 0.6:
                armed = True
            if not armed or drop > -args.depth:
                continue
            armed = False
            before = top_at - RUNUP
            if before < 0 or closes[before] <= 0:
                continue
            runup = closes[top_at] / closes[before] - 1.0
            if not args.no_runup_filter and runup > args.max_runup:
                continue
            out[bars[i].timestamp].append(
                {"ticker": ticker, "index": i, "target": top, "runup": runup})
    return out


def simulate(book, triggers, risk, args):
    """Walk the calendar; open on signal, close at the prior high or three years."""
    prices = {t: {b.timestamp: b.close for b in bars} for t, bars in book.items()}
    calendar = sorted({b.timestamp for bars in book.values() for b in bars})
    size = risk / TAIL                      # share of NAV per position

    nav, peak, worst = 1.0, 1.0, 0.0
    live, exposures, counts = [], [], []
    opened = 0
    for a, b in zip(calendar, calendar[1:]):
        # mark what is held
        gross = sum(p["size"] for p in live)
        move = 0.0
        for position in live:
            p0, p1 = prices[position["ticker"]].get(a), prices[position["ticker"]].get(b)
            if p0 and p1 and p0 > 0:
                move += position["size"] * (p1 / p0 - 1.0)
        carry = max(gross - 1.0, 0.0) * args.financing / 252.0
        nav *= (1.0 + move - carry)
        peak = max(peak, nav)
        worst = min(worst, nav / peak - 1.0)
        exposures.append(gross)
        counts.append(len(live))

        # close what has finished
        kept = []
        for position in live:
            price = prices[position["ticker"]].get(b)
            position["age"] += 1
            done = (price is not None and price >= position["target"]) \
                or position["age"] >= MAX_HOLD
            if not done:
                kept.append(position)
        live = kept

        # open what has signalled
        for signal in triggers.get(b, []):
            if any(p["ticker"] == signal["ticker"] for p in live):
                continue
            live.append({"ticker": signal["ticker"], "target": signal["target"],
                         "size": size, "age": 0})
            opened += 1

    years = (date.fromisoformat(calendar[-1])
             - date.fromisoformat(calendar[0])).days / 365.25
    return {"cagr": nav ** (1 / years) - 1 if nav > 0 else -1.0,
            "terminal": nav, "max_drawdown": worst, "opened": opened,
            "exposure_median": statistics.median(exposures),
            "exposure_p95": sorted(exposures)[int(0.95 * len(exposures))],
            "exposure_max": max(exposures),
            "positions_median": statistics.median(counts),
            "positions_max": max(counts),
            "invested_share": sum(1 for e in exposures if e > 0) / len(exposures)}


def main(argv=None) -> int:
    args = parse_args(argv)
    book = load(args)
    triggers = signals(book, args)
    total = sum(len(v) for v in triggers.values())
    calendar = sorted({b.timestamp for bars in book.values() for b in bars})
    print(f"{len(book)} equity ETFs, {calendar[0][:7]} to {calendar[-1][:7]}")
    print(f"{total:,d} entry signals at {args.depth:.0%} below the 252-day high"
          + ("" if args.no_runup_filter
             else f", run-up under {args.max_runup:.0%}"))
    print(f"exit at the prior high or {MAX_HOLD} sessions; sizing risk / "
          f"{TAIL:.0%} tail; financing {args.financing:.0%}\n")
    report = {"etfs": len(book), "signals": total, "tail": TAIL}

    print("########## returns and exposure by risk per trade ##########")
    print(f"  {'risk':>6s} {'per position':>13s} {'CAGR':>8s} {'max DD':>9s} "
          f"{'x money':>9s} {'gross med':>10s} {'gross p95':>10s} "
          f"{'gross max':>10s} {'held':>6s}")
    for risk in RISK_RUNGS:
        got = simulate(book, triggers, risk, args)
        report.setdefault("risk", {})[f"{risk:.0%}"] = got
        print(f"  {risk:>5.0%} {risk / TAIL:>12.1%} {got['cagr']:>+8.1%} "
              f"{got['max_drawdown']:>9.1%} {got['terminal']:>8.2f}x "
              f"{got['exposure_median']:>10.0%} {got['exposure_p95']:>10.0%} "
              f"{got['exposure_max']:>10.0%} {got['positions_median']:>6.0f}")

    print()
    print("########## for reference ##########")
    market = book.get("DIA")
    if market:
        first, last = market[0].close, market[-1].close
        years = (date.fromisoformat(market[-1].timestamp)
                 - date.fromisoformat(market[0].timestamp)).days / 365.25
        closes = [b.close for b in market]
        peak, worst = closes[0], 0.0
        for c in closes:
            peak = max(peak, c)
            worst = min(worst, c / peak - 1.0)
        print(f"  DIA bought and held: CAGR {(last/first) ** (1/years) - 1:+.1%}, "
              f"max drawdown {worst:.1%}, {last/first:.2f}x money")
        report["benchmark"] = {"cagr": (last / first) ** (1 / years) - 1,
                               "max_drawdown": worst}

    print()
    print("########## without the run-up filter ##########")
    args.no_runup_filter = True
    plain = signals(book, args)
    print(f"  {sum(len(v) for v in plain.values()):,d} signals instead of "
          f"{total:,d}")
    print(f"  {'risk':>6s} {'CAGR':>8s} {'max DD':>9s} {'x money':>9s} "
          f"{'gross med':>10s}")
    for risk in RISK_RUNGS:
        got = simulate(book, plain, risk, args)
        report.setdefault("no_filter", {})[f"{risk:.0%}"] = got
        print(f"  {risk:>5.0%} {got['cagr']:>+8.1%} {got['max_drawdown']:>9.1%} "
              f"{got['terminal']:>8.2f}x {got['exposure_median']:>10.0%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print()
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
