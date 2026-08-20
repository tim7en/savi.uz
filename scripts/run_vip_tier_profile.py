"""Corrected fee tiers: what the venue costs, what it exposes, what tier it reaches.

An earlier run of this analysis charged the standard USD-M taker fee of five basis
points a side.  Binance prices its TradFi contracts at half that -- 2.5bp a side
at VIP 0, 2.0bp from VIP 2 -- so the round trip is five basis points rather than
ten, and the conclusion drawn from the wrong number ("the venue does not clear its
own costs") does not survive the right one.  This script replaces it.

Four rates are run.  Five basis points is the VIP 0 and VIP 1 round trip; 4.5
applies the ten per cent BNB discount to it; four is the VIP 2 round trip and 3.6
is that with BNB.  The spread between best and worst here is 1.4 basis points,
which sounds like nothing and is not, because the strategy turns its equity over
several hundred times a year.

Two questions beyond the fee itself.

*What tier the account actually reaches.*  VIP 1 wants five million dollars of
thirty-day volume and VIP 2 wants ten.  Turnover is proportional to equity and to
the risk multiple, so the account size implied by each threshold falls straight
out, and it is the honest way to read a fee table -- not "what would I pay at VIP
2" but "what would I have to be trading to get there".

*What exposure is actually carried.*  The risk multiple is not the leverage.
Trebling the risk fraction trebles the notional a unit occupies, but the notional
itself depends on price over N, which varies enormously across instruments -- a
quiet index fund takes a position several times the size of a violent single
name for the same risk.  Gross notional is therefore measured rather than
assumed, and compared against the twenty-times cap the contracts impose.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import statistics
import sys
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import TurtleConfig, run_turtle  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

BASE = dict(entry_window=55, exit_window=20, atr_window=20, skip_after_winner=False,
            use_channel_exit=False, chandelier_atr=3.0, directions=(1,))

#: label -> round-trip cost.  TradFi taker is half the standard USD-M rate.
RATES = (("VIP 0/1, no BNB", 0.00050),
         ("VIP 0/1, with BNB", 0.00045),
         ("VIP 2, no BNB", 0.00040),
         ("VIP 2, with BNB", 0.00036))

TIERS = (("VIP 1", 5_000_000, 5), ("VIP 2", 10_000_000, 25))
MAX_LEVERAGE = 20.0
MULTIPLES = (1.0, 2.0, 3.0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--funding", type=Path,
                        default=Path("out/strategy/binance_funding.json"))
    parser.add_argument("--equity", type=float, default=1000.0)
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--min-sessions", type=int, default=400)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/vip_tiers.json"))
    return parser.parse_args(argv)


def bnb_price():
    try:
        request = urllib.request.Request(
            "https://api.binance.com/api/v3/ticker/price?symbol=BNBUSDT",
            headers={"User-Agent": "research/1.0"})
        with urllib.request.urlopen(request, timeout=20) as response:
            return float(json.loads(response.read().decode())["price"])
    except Exception:
        return None


def load_book(args, keep):
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    book = {}
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker"):
        if ticker not in keep:
            continue
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='5min' ORDER BY ts", (ticker,)).fetchall()
        five = adjust_bars([Bar(*r) for r in rows], splits.get(ticker, []))
        if len({b.timestamp[:10] for b in five}) >= args.min_sessions:
            book[ticker] = resample_regular_session(five, minutes=args.minutes)
    connection.close()
    return book


def cap(trades, limit, rng):
    shuffled = list(trades)
    rng.shuffle(shuffled)
    live, taken = [], []
    for trade in sorted(shuffled, key=lambda t: t["entry"]):
        live = [x for x in live if x["exit"] > trade["entry"]]
        if len(live) >= limit:
            continue
        live.append(trade)
        taken.append(trade)
    return taken


def build(book, config):
    pooled = []
    for ticker, bars in book.items():
        closes = {}
        for bar in bars:
            closes[bar.timestamp[:10]] = bar.close
        for trade in run_turtle(bars, config=config)[0]:
            entry_day, exit_day = (trade.entry_timestamp[:10],
                                   trade.exit_timestamp[:10])
            marks, notional_days, held = [], 0.0, 0
            for day in closes:
                if not (entry_day <= day <= exit_day):
                    continue
                live = [u for u in trade.unit_entries if u.timestamp[:10] <= day]
                if not live:
                    continue
                held += 1
                notional_days += sum(closes[day] / u.n for u in live)
                if day < exit_day:
                    marks.append((day, sum(trade.direction * (closes[day] - u.price)
                                           / u.n for u in live)))
            pooled.append({"entry": trade.entry_timestamp,
                           "exit": trade.exit_timestamp, "r": trade.net_r,
                           "marks": marks, "basis": trade.cost_basis_r,
                           "days": held, "notional_days": notional_days})
    return pooled


def daily_r(taken):
    by_day = defaultdict(float)
    for trade in taken:
        previous = 0.0
        for day, open_r in trade["marks"]:
            by_day[day] += open_r - previous
            previous = open_r
        by_day[trade["exit"][:10]] += trade["r"] - previous
    return by_day


def walk(by_day, calendar, risk):
    nav, peak, worst = 1.0, 1.0, 0.0
    for day in calendar:
        nav = max(1e-12, nav * (1.0 + by_day.get(day, 0.0) * risk))
        peak = max(peak, nav)
        worst = min(worst, nav / peak - 1.0)
    years = (date.fromisoformat(calendar[-1])
             - date.fromisoformat(calendar[0])).days / 365.25
    return nav, worst, nav ** (1 / years) - 1


def solve(maps, calendar, target):
    lo, hi = 1e-7, 0.5
    for _ in range(38):
        mid = math.sqrt(lo * hi)
        if statistics.median(abs(walk(m, calendar, mid)[1]) for m in maps) < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def exposure(taken, calendar):
    per_day = defaultdict(float)
    for trade in taken:
        entry_day, exit_day = trade["entry"][:10], trade["exit"][:10]
        span = [d for d in calendar if entry_day <= d < exit_day]
        if not span or not trade["notional_days"]:
            continue
        share = trade["notional_days"] / len(span)
        for day in span:
            per_day[day] += share
    return [per_day.get(d, 0.0) for d in calendar]


def main(argv=None):
    args = parse_args(argv)
    keep = set(json.loads(args.funding.read_text(encoding="utf-8")))
    book = load_book(args, keep)
    calendar = sorted({b.timestamp[:10] for bars in book.values() for b in bars})
    years = (date.fromisoformat(calendar[-1])
             - date.fromisoformat(calendar[0])).days / 365.25
    price = bnb_price()

    print(f"{len(book)} instruments, {years:.1f} years, long only")
    print(f"TradFi taker is half the standard USD-M rate: 2.5bp a side at VIP 0/1, "
          f"2.0bp from VIP 2\n")

    report = {}
    for label, cost in RATES:
        config = TurtleConfig(**{**BASE, "round_trip_cost": cost})
        pooled = build(book, config)
        caps = [cap(pooled, args.max_positions, random.Random(s))
                for s in range(args.trials)]
        maps = [daily_r(c) for c in caps]
        base = solve(maps, calendar, args.target_dd)
        taken = caps[0]
        turn = sum(2.0 * t["basis"] for t in taken) / years
        gross = exposure(taken, calendar)

        print(f"=== {label}  ({cost * 1e4:.1f}bp round trip) ===")
        print(f"  {'lev':>4s} {'median DD':>10s} {'CAGR':>7s} {'turnover/yr':>12s} "
              f"{'mean gross':>11s} {'95th':>7s} {'max':>7s} {'fees/yr':>8s}")
        levels = {}
        for multiple in MULTIPLES:
            risk = base * multiple
            results = [walk(m, calendar, risk) for m in maps]
            dds = sorted(abs(r[1]) for r in results)
            cagrs = sorted(r[2] for r in results)
            mid = len(dds) // 2
            scaled = sorted(v * risk for v in gross)
            turnover = turn * risk
            levels[f"{multiple:g}x"] = {
                "risk": risk, "median_dd": dds[mid], "cagr": cagrs[mid],
                "turnover": turnover, "mean_gross": statistics.fmean(scaled),
                "p95_gross": scaled[int(0.95 * len(scaled))],
                "max_gross": scaled[-1],
                "fee_share": turnover * cost}
            print(f"  {multiple:>3g}x {dds[mid]:>10.1%} {cagrs[mid]:>7.1%} "
                  f"{turnover:>11.0f}x {statistics.fmean(scaled):>10.2f}x "
                  f"{scaled[int(0.95 * len(scaled))]:>6.2f}x {scaled[-1]:>6.2f}x "
                  f"{turnover * cost:>7.1%}")
        report[label] = {"cost": cost, "base_risk": base, "levels": levels}
        print(flush=True)

    # Account size implied by each VIP threshold, at the entry rate.
    entry = report["VIP 0/1, with BNB"]["levels"]
    print("account size needed to reach each tier, at 4.5bp (VIP 0 with BNB)")
    print(f"  a 30-day window is {30 / 365.25:.4f} of a year, so monthly volume "
          f"is turnover/12.2\n")
    print(f"  {'lev':>4s} {'turnover/yr':>12s} {'per 30 days':>12s} "
          + "  ".join(f"{name:>14s}" for name, _, _ in TIERS))
    sizes = {}
    for multiple in MULTIPLES:
        item = entry[f"{multiple:g}x"]
        monthly = item["turnover"] * 30 / 365.25
        cells = []
        for name, threshold, _ in TIERS:
            need = threshold / monthly
            sizes[f"{multiple:g}x|{name}"] = need
            cells.append(f"${need:>12,.0f}")
        print(f"  {multiple:>3g}x {item['turnover']:>11.0f}x {monthly:>11.1f}x "
              + "  ".join(cells))

    print(f"\n  the BNB holding requirement is separate from the volume one"
          + (f", at ${price:,.0f} a coin:" if price else ":"))
    for name, threshold, coins in TIERS:
        value = f"${coins * price:,.0f}" if price else "n/a"
        print(f"    {name}: {coins} BNB = {value}")
    if price:
        need_1x = sizes["1x|VIP 1"]
        print(f"\n  note the 5 BNB for VIP 1 (${5 * price:,.0f}) against the "
              f"${need_1x:,.0f} account that")
        print(f"  reaches its volume threshold at 1x -- the coin requirement is "
              f"{5 * price / need_1x:.0%} of the account,")
        print(f"  and it is held, not spent, so it carries BNB price risk for as "
              f"long as the tier is wanted.")

    print(f"\nexposure: the risk multiple is not the leverage")
    entry_gross = entry["3x"]
    print(f"  at 3x the book carries {entry_gross['mean_gross']:.2f}x equity on "
          f"average, {entry_gross['p95_gross']:.2f}x at the 95th percentile,")
    print(f"  and peaks at {entry_gross['max_gross']:.2f}x against a "
          f"{MAX_LEVERAGE:.0f}x contract cap "
          f"({'clears' if entry_gross['max_gross'] < MAX_LEVERAGE else 'BREACHES'}"
          f" it).")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"rates": report, "account_sizes": sizes, "bnb_price": price},
        indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
