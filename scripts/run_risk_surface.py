"""How much risk can the book actually carry, and how much is it carrying?

Every comparison in this programme has been scored at a matched eighteen per cent
drawdown, which is the right way to rank rules and the wrong way to understand
exposure: it silently answers "how much risk would you have to take for these to
be comparable" rather than "how much are you taking".

Three questions here, all in absolute terms.

*What drawdown comes with what risk setting.*  The risk fraction is swept from a
half to five times the level that produces eighteen per cent, and the drawdown is
read off rather than solved for.  Both the median tie-break and the near-worst
are reported: an investor experiences one path, not the median of thirty.

*Whether the book is fully invested.*  Position size is ``risk / N``, so the
notional a unit occupies is ``risk x price / N`` -- large for a quiet instrument,
small for a violent one.  Summed across open positions this gives the gross
exposure actually carried, which nothing so far has measured.

*Where ruin begins.*  Compounding a fixed fraction has a level beyond which a bad
run cannot be recovered from.  The sweep continues until the worst path loses
half its capital, which is the practical answer to "maximum".
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import TurtleConfig, run_turtle  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

BASE = dict(entry_window=55, exit_window=20, atr_window=20, skip_after_winner=False,
            directions=(1,), use_channel_exit=False, chandelier_atr=3.0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--cost", type=float, default=0.0002)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/risk_surface.json"))
    return parser.parse_args(argv)


def load_book(args):
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    book = {}
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker"):
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='5min' ORDER BY ts", (ticker,)).fetchall()
        five = adjust_bars([Bar(*r) for r in rows], splits.get(ticker, []))
        if len({b.timestamp[:10] for b in five}) >= 400:
            book[ticker] = resample_regular_session(five, minutes=30)
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
    ruined = False
    for day in calendar:
        nav *= 1.0 + by_day.get(day, 0.0) * risk
        if nav <= 0:
            nav, ruined = 1e-12, True
            break
        peak = max(peak, nav)
        worst = min(worst, nav / peak - 1.0)
    years = (date.fromisoformat(calendar[-1])
             - date.fromisoformat(calendar[0])).days / 365.25
    cagr = nav ** (1 / years) - 1 if nav > 0 else -1.0
    return nav, worst, cagr, ruined


def solve_risk(maps, calendar, target, lo=1e-7, hi=0.5):
    def dd(risk):
        return statistics.median(abs(walk(m, calendar, risk)[1]) for m in maps)
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        if dd(mid) < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def gross_exposure(taken, calendar):
    """Notional carried as a multiple of equity, per unit of risk fraction.

    Computed on the *capped* book, which is the whole point: an earlier version
    summed every trade the engine generated and so counted positions the six-slot
    limit would never have allowed simultaneously, overstating exposure roughly
    sevenfold.  A unit occupies ``risk x price / N`` of capital, so summing
    ``price / N`` over live units and multiplying by the risk fraction gives the
    notional actually held.
    """
    per_day = defaultdict(float)
    live_units = defaultdict(int)
    positions = defaultdict(int)
    for trade in taken:
        closes = trade["closes"]
        entry_day, exit_day = trade["entry"][:10], trade["exit"][:10]
        for day in (d for d in closes if entry_day <= d < exit_day):
            units = [u for u in trade["units"] if u.timestamp[:10] <= day]
            if not units:
                continue
            per_day[day] += sum(closes[day] / u.n for u in units)
            live_units[day] += len(units)
            positions[day] += 1
    return per_day, live_units, positions


def main(argv=None):
    args = parse_args(argv)
    book = load_book(args)
    calendar = sorted({b.timestamp[:10] for bars in book.values() for b in bars})
    config = TurtleConfig(**{**BASE, "round_trip_cost": args.cost})

    pooled = []
    for ticker, bars in book.items():
        closes = {b.timestamp[:10]: b.close for b in bars}
        for trade in run_turtle(bars, config=config)[0]:
            marks = []
            for day in (d for d in closes
                        if trade.entry_timestamp[:10] <= d < trade.exit_timestamp[:10]):
                live = [u for u in trade.unit_entries if u.timestamp[:10] <= day]
                if live:
                    marks.append((day, sum(trade.direction * (closes[day] - u.price) / u.n
                                           for u in live)))
            pooled.append({"entry": trade.entry_timestamp,
                           "exit": trade.exit_timestamp, "r": trade.net_r,
                           "marks": marks, "units": trade.unit_entries,
                           "closes": closes})
    caps = [cap(pooled, args.max_positions, random.Random(s))
            for s in range(args.trials)]
    maps = [daily_r(c) for c in caps]
    base = solve_risk(maps, calendar, args.target_dd)
    print(f"{len(book)} instruments, {len(pooled):,} trades")
    print(f"risk fraction giving an {args.target_dd:.0%} median drawdown: "
          f"{base:.4%} of equity per 1N move\n", flush=True)

    print(f"  {'risk':>7s} {'per trade':>10s} {'median DD':>10s} {'worst DD':>9s} "
          f"{'CAGR':>9s} {'$1000 ->':>14s} {'ruined':>7s}")
    rows = {}
    for multiple in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0):
        risk = base * multiple
        results = [walk(m, calendar, risk) for m in maps]
        dds = sorted(abs(r[1]) for r in results)
        navs = sorted(r[0] for r in results)
        cagrs = sorted(r[2] for r in results)
        ruined = sum(1 for r in results if r[3])
        mid = len(dds) // 2
        worst = dds[min(int(0.95 * len(dds)), len(dds) - 1)]
        rows[f"{multiple:g}x"] = {
            "risk": risk, "median_dd": dds[mid], "worst_dd": worst,
            "cagr": cagrs[mid], "final": navs[mid] * 1000, "ruined": ruined}
        print(f"  {multiple:>6g}x {risk:>10.4%} {dds[mid]:>10.1%} {worst:>9.1%} "
              f"{cagrs[mid]:>9.1%} {navs[mid] * 1000:>14,.0f} "
              f"{ruined:>4d}/{len(results)}", flush=True)

    per_unit, live_units, positions = gross_exposure(caps[0], calendar)
    values = [per_unit.get(d, 0.0) for d in calendar]
    print(f"\ngross notional carried, as a multiple of equity")
    print(f"  {'risk setting':>14s} {'mean':>8s} {'median':>8s} "
          f"{'95th pct':>9s} {'maximum':>9s}")
    exposures = {}
    for multiple in (1.0, 2.0, 3.0):
        scaled = [v * base * multiple for v in values]
        ordered = sorted(scaled)
        entry = {"mean": statistics.fmean(scaled),
                 "median": ordered[len(ordered) // 2],
                 "p95": ordered[int(0.95 * len(ordered))], "max": max(scaled)}
        exposures[f"{multiple:g}x"] = entry
        print(f"  {multiple:>13g}x {entry['mean']:>8.2f} {entry['median']:>8.2f} "
              f"{entry['p95']:>9.2f} {entry['max']:>9.2f}")

    invested = sum(1 for d in calendar if positions.get(d, 0) > 0) / len(calendar)
    full = sum(1 for d in calendar
               if positions.get(d, 0) >= args.max_positions) / len(calendar)
    print(f"\nsessions with any position open : {invested:.0%}")
    print(f"  sessions with all six slots used: {full:.0%}")
    print(f"  average positions open          : "
          f"{statistics.fmean(positions.get(d, 0) for d in calendar):.2f} of "
          f"{args.max_positions}")
    print(f"  average units open              : "
          f"{statistics.fmean(live_units.get(d, 0) for d in calendar):.2f} of "
          f"{args.max_positions * 4}")

    report = {"base_risk": base, "levels": rows, "exposure": exposures,
              "invested_share": invested,
              "full_share": full / len(calendar)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
