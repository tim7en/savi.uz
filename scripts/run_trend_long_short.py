"""Trend following both ways, capped at five times, and what its null becomes.

Every long-only arm tested in this programme has failed the same way: it scored
respectably and its drift null scored as much or more.  That is not a coincidence
and it is not bad luck.  A long book in these names is a bet on equity drift, and
so is a control that buys the same names on random days.  The control inherits
the edge, so the test can never separate them.

A long *and* short book is different in exactly that respect.  Its null is random
longs and random shorts in matched proportion, which nets most of the drift out
and collapses toward zero.  If trend following contains anything beyond drift,
this is the configuration where it becomes visible -- and if it contains nothing,
this is where that shows up as a book that cannot beat zero either.

So three arms, each against its own matched-direction null:

* long only, which is the incumbent and the one whose null eats it;
* short only, run without any borrow cost at all -- generous on purpose, because
  the programme's earlier finding was that breakout shorts fail at 10bp *with
  borrow removed entirely*, and a short arm should be allowed to fail on its own
  terms rather than on a financing assumption;
* both, which is the proposition.

Sizing is fixed fractional with a hard five-times cap per position.  Unlike the
twenty-times work, five is reachable: a trade whose stop sits half a percent from
entry wants ten times at a 5% budget, so the cap binds and the share of trades it
binds on is reported.  Exposure is reported gross, because a long/short book can
carry a large gross while running a small net, and those are different risks.

The entry is the banked one -- 55-bar Donchian, 2N stop, half-N pyramid to four
units, 3N chandelier -- so this tests the direction and the sizing, not a new
signal.  Parameters are frozen; only direction and the cap vary.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import statistics
import sys
import zlib
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.turtle import TurtleConfig, run_turtle  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

import run_vol_stretch_zones as shared  # noqa: E402

FIXED = dict(entry_window=55, exit_window=20, atr_window=20, stop_atr=2.0,
             add_atr=0.5, max_units=4, skip_after_winner=False,
             use_channel_exit=False, chandelier_atr=3.0)
RISK_RUNGS = (0.0005, 0.001, 0.002, 0.003, 0.005)
ARMS = (("long only", (1,)), ("short only", (-1,)), ("both ways", (1, -1)))
REG_T = 2.0


def ticker_seed(t):
    return zlib.crc32(t.encode("utf-8")) % 10_000


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path,
                        default=Path("data/13f/alphavantage_daily.db"))
    parser.add_argument("--split", default="2013-01-01")
    parser.add_argument("--cost-bp", type=float, default=10.0)
    parser.add_argument("--max-leverage", type=float, default=5.0)
    parser.add_argument("--max-positions", type=int, default=12)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--null-draws", type=int, default=30)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/trend_long_short.json"))
    return parser.parse_args(argv)


def load_book(args):
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    book = {}
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency='daily' "
            "ORDER BY ticker"):
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='daily' ORDER BY ts", (ticker,)).fetchall()
        bars = adjust_bars([Bar(r[0][:10], r[1], r[2], r[3], r[4], r[5])
                            for r in rows], splits.get(ticker, []))
        if len(bars) >= 750:
            book[ticker] = bars
    connection.close()
    return book


def collect(book, directions, args, null_seed=None):
    """The banked turtle, or a random-entry null with the same direction mix."""
    config = TurtleConfig(**FIXED, directions=directions,
                          round_trip_cost=args.cost_bp / 10_000)
    out = []
    for ticker, bars in book.items():
        if null_seed is None:
            trades, _ = run_turtle(bars, config=config)
        else:
            real, _ = run_turtle(bars, config=config)
            if not real:
                continue
            index_of = {b.timestamp: i for i, b in enumerate(bars)}
            wanted = [index_of.get(t.entry_timestamp) for t in real]
            wanted = [w for w in wanted if w is not None]
            rng = random.Random(null_seed + ticker_seed(ticker))
            floor = max(FIXED["entry_window"], FIXED["atr_window"]) + 2
            pool = range(floor, len(bars) - 30)
            if len(pool) <= len(wanted):
                continue
            picks = rng.sample(list(pool), len(wanted))
            mix = [t.direction for t in real]
            rng.shuffle(mix)
            entries = {i: d for i, d in zip(sorted(picks), mix)}
            trades, _ = run_turtle(bars, config=config, entries=entries)
        for t in trades:
            stop_pct = FIXED["stop_atr"] * t.n_at_entry / t.entry if t.entry else 0
            if stop_pct <= 0:
                continue
            out.append({"ticker": ticker, "entry": t.entry_timestamp[:10],
                        "exit": t.exit_timestamp[:10], "r": t.net_r,
                        "dir": t.direction, "units": t.units,
                        "stop_pct": stop_pct, "units_tuple": t.unit_entries})
    out.sort(key=lambda t: t["entry"])
    return out


def window(trades, lo=None, hi=None):
    return [t for t in trades
            if (lo is None or t["entry"] >= lo) and (hi is None or t["entry"] < hi)]


def sized(taken, fraction, cap):
    """Compound the book, and measure the gross and net exposure it carries."""
    per_day, levers, capped = defaultdict(float), [], 0
    for t in taken:
        # A pyramided position holds ``units`` lots, so the venue cap applies to
        # their sum, not to each one. Capping per unit would let a four-unit
        # trade carry four times the stated maximum.
        want = t["units"] * fraction / t["stop_pct"]
        lever = min(want, cap)
        capped += want > cap
        levers.append(lever)
        per_day[t["exit"]] += t["r"] * (lever / want) * fraction if want else 0.0
    days = sorted(per_day)
    if not days:
        return None
    nav, peak, worst = 1000.0, 1000.0, 0.0
    for d in days:
        nav = max(0.0, nav + per_day[d] * nav)
        peak = max(peak, nav)
        worst = min(worst, nav / peak - 1.0)
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25

    calendar, cur = [], date.fromisoformat(taken[0]["entry"])
    stop_at = date.fromisoformat(max(t["exit"] for t in taken))
    while cur <= stop_at:
        if cur.weekday() < 5:
            calendar.append(cur.isoformat())
        cur += timedelta(days=1)
    gross, net = defaultdict(float), defaultdict(float)
    for t, lever in zip(taken, levers):
        weight = lever
        for d in calendar:
            if t["entry"] <= d < t["exit"]:
                gross[d] += weight
                net[d] += weight * t["dir"]
    gseries = sorted(gross.get(d, 0.0) for d in calendar)
    nseries = [net.get(d, 0.0) for d in calendar]
    return {"cagr": (nav / 1000.0) ** (1 / years) - 1 if nav > 0 else -1.0,
            "max_drawdown": worst, "terminal": nav / 1000.0,
            "share_capped": capped / len(levers),
            "gross_median": gseries[len(gseries) // 2],
            "gross_max": gseries[-1],
            "net_median": statistics.median(nseries),
            "share_over_regt": sum(1 for g in gross.values() if g > REG_T)
                               / max(len(calendar), 1)}


def main(argv=None) -> int:
    args = parse_args(argv)
    book = load_book(args)
    print(f"{len(book)} names, banked turtle 55/20 with the 3N chandelier, "
          f"{args.cost_bp:g}bp round trip")
    print(f"sizing capped at {args.max_leverage:g}x per position, "
          f"{args.max_positions} slots, no borrow charged on shorts")
    print(f"in sample to {args.split}, out of sample after\n")
    report = {"names": len(book), "arms": {}}

    print(f"  {'arm':26s} {'period':>13s} {'trades':>8s} {'taken':>7s} "
          f"{'Sharpe':>7s} {'[5-95%]':>14s}")
    books = {}
    for label, directions in ARMS:
        trades = collect(book, directions, args)
        books[label] = trades
        for period, lo, hi in (("in sample", None, args.split),
                               ("out of sample", args.split, None)):
            pooled = window(trades, lo, hi)
            result = shared.assess(pooled, args)
            if not result:
                continue
            report["arms"].setdefault(label, {})[period] = result
            print(f"  {label:26s} {period:>13s} {len(pooled):>8,d} "
                  f"{result['taken']:>7,d} {result['sharpe']:>7.2f} "
                  f"{('[%.2f-%.2f]' % (result['sharpe_p05'], result['sharpe_p95'])):>14s}",
                  flush=True)

    print(f"\n  the nulls, out of sample only "
          f"(random entries, same direction mix)")
    print(f"  {'arm':26s} {'book':>7s} {'null':>7s} {'[low-high]':>16s} {'p':>6s}")
    for label, directions in ARMS:
        edge = report["arms"].get(label, {}).get("out of sample", {}).get("sharpe")
        if edge is None:
            continue
        nulls = []
        for draw in range(args.null_draws):
            drawn = collect(book, directions, args, null_seed=95_000 + 137 * draw)
            pooled = window(drawn, args.split, None)
            if len(pooled) < 200:
                continue
            outcome = shared.assess(pooled, args)
            if outcome:
                nulls.append(outcome["sharpe"])
        if not nulls:
            continue
        nulls.sort()
        above = sum(1 for x in nulls if x >= edge) / len(nulls)
        report["arms"][label]["null"] = {
            "median": statistics.median(nulls), "low": nulls[0],
            "high": nulls[-1], "p": above}
        flag = "clears" if above <= 0.05 else "inside"
        print(f"  {label:26s} {edge:>7.2f} {statistics.median(nulls):>7.2f} "
              f"{('[%.2f-%.2f]' % (nulls[0], nulls[-1])):>16s} {above:>6.2f}  {flag}")

    print(f"\n  sizing out of sample, capped at {args.max_leverage:g}x per position")
    print(f"  {'arm':16s} {'risk':>6s} {'CAGR':>8s} {'max DD':>8s} {'gross med':>10s} "
          f"{'gross max':>10s} {'net med':>8s} {'capped':>7s} {'>RegT':>7s}")
    for label, _ in ARMS:
        outside = window(books[label], args.split, None)
        taken = shared.cap(outside, args.max_positions, random.Random(0))
        if not taken:
            continue
        rows = []
        for fraction in RISK_RUNGS:
            row = sized(taken, fraction, args.max_leverage)
            if not row:
                continue
            rows.append({"risk": fraction, **row})
            print(f"  {label:16s} {fraction:>5.1%} {row['cagr']:>+8.1%} "
                  f"{row['max_drawdown']:>8.1%} {row['gross_median']:>9.2f}x "
                  f"{row['gross_max']:>9.2f}x {row['net_median']:>+7.2f}x "
                  f"{row['share_capped']:>7.1%} {row['share_over_regt']:>7.1%}")
        report["arms"].setdefault(label, {})["sizing"] = rows
        matched = shared.assess(outside, args)
        if matched and matched.get("risk_fraction"):
            frac = matched["risk_fraction"]
            implied = statistics.median(
                t["units"] * frac / t["stop_pct"] for t in taken)
            report["arms"][label]["matched_dd"] = {
                "risk_fraction": frac, "implied_gross": implied}
            print(f"  {label:16s} {frac:>5.3%} <- the risk budget an 18% median "
                  f"drawdown allows, implying {implied:.2f}x gross")
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
