"""What running this actually looks like: leverage, turnover, holding period, size.

The venue table reported one risk setting -- the level that produces an
eighteen per cent median drawdown, which every comparison in this programme uses
so that rules can be ranked without leverage confounding them.  That is a
scoring convention, not a recommendation, and it answers none of the questions an
operator asks before funding an account.

Four are answered here.

*What two and three times that risk costs.*  Drawdown does not scale linearly
under compounding, so the multiples are walked rather than extrapolated, and both
the median tie-break and the near-worst path are reported.  An investor
experiences one path, not the median of forty.

*How much trading it generates.*  A unit occupies ``risk x price / N`` of
notional, which is the same quantity the commission is charged on, so turnover
falls out of the cost basis directly: entry and exit together are twice it.
Reported as a multiple of equity per year, which is the number that decides
whether ten basis points is a nuisance or the whole result.

*How long positions are actually held.*  Sessions with units live, per trade.
This is what makes funding legible -- a carry of one basis point a day is
irrelevant over three sessions and material over forty.

*How large a fund could do this.*  Position notional is compared against each
instrument's own median daily dollar volume, capping participation at one per
cent.  This is an order-of-magnitude estimate and is stated as one: it ignores
that a breakout is precisely when everyone else wants the same side, which is
where real capacity is lost.

Long only throughout.  The short side was measured at every tradeable cost in the
venue test and loses money at all of them, so leverage on it is not a question
worth asking.
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
            use_channel_exit=False, chandelier_atr=3.0, directions=(1,))

COSTS = (0.0002, 0.0010)
MULTIPLES = (1.0, 2.0, 3.0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--funding", type=Path,
                        default=Path("out/strategy/binance_funding.json"))
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--min-sessions", type=int, default=400)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--participation", type=float, default=0.01,
                        help="share of an instrument's daily dollar volume a "
                             "position may occupy, for the capacity estimate")
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/venue_leverage.json"))
    return parser.parse_args(argv)


def load_book(args, keep):
    """Thirty-minute bars, and each instrument's median daily dollar volume."""
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    book, adv = {}, {}
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker"):
        if ticker not in keep:
            continue
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='5min' ORDER BY ts", (ticker,)).fetchall()
        five = adjust_bars([Bar(*r) for r in rows], splits.get(ticker, []))
        if len({b.timestamp[:10] for b in five}) < args.min_sessions:
            continue
        book[ticker] = resample_regular_session(five, minutes=args.minutes)
        # Dollar volume is computed on the raw rows rather than the adjusted
        # ones: a split restates price and share count in opposite directions,
        # so the traded value is unchanged and adjusting it would be wrong.
        by_day = defaultdict(float)
        for row in rows:
            by_day[row[0][:10]] += (row[5] or 0.0) * (row[4] or 0.0)
        adv[ticker] = statistics.median(by_day.values()) if by_day else 0.0
    connection.close()
    return book, adv


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


def build(book, config, rate_of):
    pooled = []
    for ticker, bars in book.items():
        closes = {}
        for bar in bars:
            closes[bar.timestamp[:10]] = bar.close
        rate = rate_of(ticker)
        for trade in run_turtle(bars, config=config)[0]:
            entry_day, exit_day = (trade.entry_timestamp[:10],
                                   trade.exit_timestamp[:10])
            marks, carry, held, notional_days = [], 0.0, 0, 0.0
            for day in closes:
                if not (entry_day <= day <= exit_day):
                    continue
                live = [u for u in trade.unit_entries if u.timestamp[:10] <= day]
                if not live:
                    continue
                held += 1
                notional_r = sum(closes[day] / u.n for u in live)
                notional_days += notional_r
                carry += trade.direction * rate * notional_r
                if day < exit_day:
                    marks.append((day, sum(trade.direction * (closes[day] - u.price)
                                           / u.n for u in live)))
            pooled.append({
                "ticker": ticker, "entry": trade.entry_timestamp,
                "exit": trade.exit_timestamp, "r": trade.net_r - carry,
                "marks": marks, "days": held, "units": len(trade.unit_entries),
                # sum(price / N) over units -- the notional the commission is
                # charged on, and therefore also the turnover per leg
                "basis": trade.cost_basis_r,
                "notional_days": notional_days})
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
    ruined = False
    for day in calendar:
        nav *= 1.0 + by_day.get(day, 0.0) * risk
        if nav <= 1e-9:
            nav, ruined = 1e-9, True
            break
        peak = max(peak, nav)
        worst = min(worst, nav / peak - 1.0)
    years = (date.fromisoformat(calendar[-1])
             - date.fromisoformat(calendar[0])).days / 365.25
    return nav, worst, nav ** (1 / years) - 1, ruined


def solve(maps, calendar, target):
    lo, hi = 1e-7, 0.5
    for _ in range(38):
        mid = math.sqrt(lo * hi)
        dd = statistics.median(abs(walk(m, calendar, mid)[1]) for m in maps)
        if dd < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def exposure(taken, calendar):
    """Gross notional per unit of risk fraction, by session."""
    per_day = defaultdict(float)
    for trade in taken:
        entry_day, exit_day = trade["entry"][:10], trade["exit"][:10]
        span = [d for d in calendar if entry_day <= d < exit_day]
        if not span or not trade["notional_days"]:
            continue
        share = trade["notional_days"] / max(len(span), 1)
        for day in span:
            per_day[day] += share
    return [per_day.get(d, 0.0) for d in calendar]


def main(argv=None):
    args = parse_args(argv)
    measured = json.loads(args.funding.read_text(encoding="utf-8"))
    book, adv = load_book(args, set(measured))
    calendar = sorted({b.timestamp[:10] for bars in book.values() for b in bars})
    years = (date.fromisoformat(calendar[-1])
             - date.fromisoformat(calendar[0])).days / 365.25

    print(f"{len(book)} instruments, {calendar[0]} -> {calendar[-1]}, "
          f"{len(calendar):,} sessions ({years:.1f} years), long only\n")

    report = {}
    for cost in COSTS:
        config = TurtleConfig(**{**BASE, "round_trip_cost": cost})
        pooled = build(book, config, lambda t: 0.0)
        caps = [cap(pooled, args.max_positions, random.Random(s))
                for s in range(args.trials)]
        maps = [daily_r(c) for c in caps]
        base = solve(maps, calendar, args.target_dd)

        print(f"=== {cost * 1e4:.0f}bp round trip "
              f"{'(retail broker)' if cost <= 0.0002 else '(Binance taker)'} ===")
        print(f"  1x is {base:.4%} of equity risked per 1N move, which is the "
              f"level giving an {args.target_dd:.0%} median drawdown\n")
        print(f"  {'leverage':>9s} {'risk/trade':>11s} {'median DD':>10s} "
              f"{'worst DD':>9s} {'CAGR':>8s} {'$100k ->':>13s} {'ruin':>6s}")
        levels = {}
        for multiple in MULTIPLES:
            risk = base * multiple
            results = [walk(m, calendar, risk) for m in maps]
            dds = sorted(abs(r[1]) for r in results)
            cagrs = sorted(r[2] for r in results)
            navs = sorted(r[0] for r in results)
            mid = len(dds) // 2
            worst = dds[min(int(0.95 * len(dds)), len(dds) - 1)]
            ruined = sum(1 for r in results if r[3])
            levels[f"{multiple:g}x"] = {
                "risk": risk, "median_dd": dds[mid], "worst_dd": worst,
                "cagr": cagrs[mid], "final": navs[mid] * 100_000, "ruined": ruined}
            print(f"  {multiple:>8g}x {risk:>11.4%} {dds[mid]:>10.1%} "
                  f"{worst:>9.1%} {cagrs[mid]:>8.1%} "
                  f"{navs[mid] * 100_000:>13,.0f} {ruined:>3d}/{len(results)}")

        # Holding period and turnover, from the median-sized capped book.
        taken = caps[0]
        held = sorted(t["days"] for t in taken)
        units = statistics.fmean(t["units"] for t in taken)
        print(f"\n  holding period   median {held[len(held) // 2]:>3d} sessions, "
              f"mean {statistics.fmean(held):>4.1f}, "
              f"90th pct {held[int(0.9 * len(held))]:>3d}, "
              f"max {held[-1]:,}")
        print(f"  trades           {len(taken):,} taken of {len(pooled):,} "
              f"generated, {len(taken) / years:.0f} a year, "
              f"{units:.2f} units each")

        gross = exposure(taken, calendar)
        turn = sum(2.0 * t["basis"] for t in taken) / years
        print(f"\n  {'leverage':>9s} {'turnover/yr':>12s} {'mean gross':>11s} "
              f"{'95th pct':>9s} {'max gross':>10s}")
        for multiple in MULTIPLES:
            risk = base * multiple
            scaled = sorted(v * risk for v in gross)
            levels[f"{multiple:g}x"]["turnover"] = turn * risk
            levels[f"{multiple:g}x"]["mean_gross"] = statistics.fmean(scaled)
            levels[f"{multiple:g}x"]["max_gross"] = scaled[-1]
            print(f"  {multiple:>8g}x {turn * risk:>11.1f}x "
                  f"{statistics.fmean(scaled):>10.2f}x "
                  f"{scaled[int(0.95 * len(scaled))]:>8.2f}x {scaled[-1]:>9.2f}x")

        # Capacity: position notional against the instrument's own dollar volume.
        by_ticker = defaultdict(list)
        for trade in taken:
            by_ticker[trade["ticker"]].append(trade["basis"])
        caps_usd = {}
        for ticker, bases in by_ticker.items():
            if not adv.get(ticker):
                continue
            typical = statistics.median(bases)  # notional per unit of risk
            caps_usd[ticker] = args.participation * adv[ticker] / (typical * base)
        if caps_usd:
            ordered = sorted(caps_usd.values())
            tight = sorted(caps_usd.items(), key=lambda kv: kv[1])[:3]
            print(f"\n  capacity at {args.participation:.0%} of daily volume, 1x: "
                  f"median ${ordered[len(ordered) // 2] / 1e6:,.0f}M, "
                  f"tightest ${ordered[0] / 1e6:,.0f}M")
            print(f"    binding names: "
                  + ", ".join(f"{t} ${v / 1e6:,.0f}M" for t, v in tight))
            print(f"    (a 3x book trades three times the notional, so divide by 3)")
        report[f"{cost}"] = {"base_risk": base, "levels": levels,
                             "median_days": held[len(held) // 2],
                             "mean_days": statistics.fmean(held),
                             "trades_per_year": len(taken) / years,
                             "capacity_usd": caps_usd}
        print(flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
