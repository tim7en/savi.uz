"""Time spent underwater, and whether the stocks did the work rather than the rules.

Two questions an investor asks that a Sharpe ratio conceals.

*How it feels.*  An eighteen per cent drawdown says how deep the hole is and
nothing about how long one sits in it.  Time below the previous peak, and time
below a fifty-day average of the equity curve, describe the experience of holding
the thing.

*Whose achievement it is.*  The universe was chosen in 2026 and contains names
that multiplied many times over.  A long-only breakout system on such a list will
look impressive whatever its rules, so the comparison that matters is against
simply buying the same forty-two and holding them.  Each is bought at its own
first available date, since the nine that listed later could not have been held
sooner -- which is the honest version of the comparison rather than the flattering
one.

Both books are put on the same footing by scoring the strategy at the risk level
that gives it the same drawdown as buy-and-hold, so neither wins by carrying more.
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
ETFS = {"SPY", "QQQ", "IWM", "GLD", "SLV", "XLE", "EWJ", "EWT", "EWY", "KWEB",
        "TMF", "TBT", "UVXY"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--cost", type=float, default=0.0002)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/underwater.json"))
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


def curve_from_r(by_day, calendar, risk):
    nav, out = 1.0, []
    for day in calendar:
        nav = max(1e-12, nav * (1.0 + by_day.get(day, 0.0) * risk))
        out.append(nav)
    return out


def equal_weight_curve(book, calendar):
    """Buy each name on its first available session and hold, equally weighted."""
    closes = {t: {b.timestamp[:10]: b.close for b in bars} for t, bars in book.items()}
    returns = {}
    for ticker, series in closes.items():
        days = sorted(series)
        returns[ticker] = {days[i]: series[days[i]] / series[days[i - 1]] - 1.0
                           for i in range(1, len(days)) if series[days[i - 1]] > 0}
    nav, out = 1.0, []
    for day in calendar:
        live = [returns[t][day] for t in returns if day in returns[t]]
        nav *= 1.0 + (statistics.fmean(live) if live else 0.0)
        out.append(nav)
    return out


def underwater(curve, calendar):
    """Depth and, more tellingly, duration below the previous high."""
    peak, worst = curve[0], 0.0
    spells, current = [], None
    below = 0
    for i, value in enumerate(curve):
        if value >= peak:
            peak = value
            if current is not None:
                spells.append(current)
                current = None
        else:
            below += 1
            worst = min(worst, value / peak - 1.0)
            if current is None:
                current = {"from": calendar[i], "days": 1, "depth": value / peak - 1.0}
            else:
                current["days"] += 1
                current["depth"] = min(current["depth"], value / peak - 1.0)
    if current is not None:
        spells.append(current)
    window = 50
    below_ma = 0
    for i in range(window, len(curve)):
        if curve[i] < statistics.fmean(curve[i - window:i]):
            below_ma += 1
    longest = max(spells, key=lambda s: s["days"]) if spells else None
    return {"max_drawdown": worst,
            "share_below_peak": below / len(curve),
            "share_below_50d": below_ma / max(len(curve) - window, 1),
            "spells": len(spells),
            "longest_days": longest["days"] if longest else 0,
            "longest_from": longest["from"] if longest else None,
            "median_spell": statistics.median([s["days"] for s in spells])
            if spells else 0,
            "spells_over_60": sum(1 for s in spells if s["days"] >= 60)}


def stats(curve, calendar):
    years = (date.fromisoformat(calendar[-1])
             - date.fromisoformat(calendar[0])).days / 365.25
    rets = [curve[i] / curve[i - 1] - 1.0 for i in range(1, len(curve))]
    sd = statistics.pstdev(rets)
    return {"cagr": curve[-1] ** (1 / years) - 1,
            "sharpe": statistics.fmean(rets) / sd * math.sqrt(252) if sd else None,
            **underwater(curve, calendar)}


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
                           "marks": marks})
    maps = [daily_r(cap(pooled, args.max_positions, random.Random(s)))
            for s in range(args.trials)]

    hold = equal_weight_curve(book, calendar)
    hold_stats = stats(hold, calendar)
    target = abs(hold_stats["max_drawdown"])

    lo, hi = 1e-7, 0.5
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        dd = statistics.median(
            abs(underwater(curve_from_r(m, calendar, mid), calendar)["max_drawdown"])
            for m in maps)
        if dd < target:
            lo = mid
        else:
            hi = mid
    matched = math.sqrt(lo * hi)

    curves = [curve_from_r(m, calendar, matched) for m in maps]
    picked = sorted(range(len(curves)), key=lambda i: curves[i][-1])[len(curves) // 2]
    strategy_stats = stats(curves[picked], calendar)

    print(f"{len(book)} instruments, {calendar[0]} -> {calendar[-1]}, "
          f"{len(calendar):,} sessions")
    print(f"strategy scored at {matched:.4%} risk, which matches buy-and-hold's "
          f"{target:.1%} drawdown\n")
    print(f"  {'':30s} {'strategy':>12s} {'buy and hold':>14s}")
    for key, label, fmt in (
            ("cagr", "Annual return", "{:.1%}"),
            ("sharpe", "Sharpe", "{:.2f}"),
            ("max_drawdown", "Worst drawdown", "{:.1%}"),
            ("share_below_peak", "Share of days below peak", "{:.0%}"),
            ("share_below_50d", "Share below its 50-day mean", "{:.0%}"),
            ("longest_days", "Longest spell underwater", "{:,.0f} days"),
            ("median_spell", "Median spell", "{:,.0f} days"),
            ("spells_over_60", "Spells beyond 60 days", "{:,.0f}")):
        a = fmt.format(strategy_stats[key])
        b = fmt.format(hold_stats[key])
        print(f"  {label:30s} {a:>12s} {b:>14s}")
    print(f"\n  longest underwater spell began "
          f"{strategy_stats['longest_from']} (strategy), "
          f"{hold_stats['longest_from']} (buy and hold)")

    # How much of the result belongs to the names rather than the rules?
    singles = {t: b for t, b in book.items() if t not in ETFS}
    etfs = {t: b for t, b in book.items() if t in ETFS}
    print(f"\n  buy and hold, by slice of the universe:")
    slices = {}
    for label, subset in (("all 42", book), ("29 single names", singles),
                          ("13 ETFs", etfs)):
        if not subset:
            continue
        curve = equal_weight_curve(subset, calendar)
        item = stats(curve, calendar)
        slices[label] = item
        print(f"    {label:16s} {item['cagr']:>7.1%} a year, worst fall "
              f"{item['max_drawdown']:>6.1%}, Sharpe {item['sharpe']:.2f}")

    report = {"matched_risk": matched, "strategy": strategy_stats,
              "buy_and_hold": hold_stats, "slices": slices}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
