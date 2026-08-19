"""Compare exit rules at matched drawdown, not matched risk setting.

The raw comparison flattered every defensive exit: a chandelier that gives back
less also carries less risk, so it wins on Calmar while earning half the R.  The
honest question is what each rule returns for the *same pain*, which means
levering each one up or down until they all sit at the same drawdown and then
reading off the return.

This is the control whose absence made theme sub-accounts look bad, and it is
the same trap the capacity study fell into before matched-risk columns were
added.  Drawdown does not scale linearly with risk under compounding, so the
level is found by bisection rather than by multiplying.

Entries are held identical across variants: the baseline's breakout bars are
offered to every rule through the engine's explicit-entry path, so the only
thing that differs is management.
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

BASE = dict(entry_window=55, exit_window=20, atr_window=20,
            skip_after_winner=False, directions=(1,))

VARIANTS = [
    ("channel 20 (baseline)", {}),
    ("channel 10 (faster)", dict(exit_window=10)),
    ("channel 50 (slower)", dict(exit_window=50)),
    ("chandelier 3N only", dict(use_channel_exit=False, chandelier_atr=3.0)),
    ("chandelier 5N only", dict(use_channel_exit=False, chandelier_atr=5.0)),
    ("chandelier 8N only", dict(use_channel_exit=False, chandelier_atr=8.0)),
    ("channel 20 + chandelier 5N", dict(chandelier_atr=5.0)),
    ("channel 20 + breakeven 1N", dict(breakeven_trigger_n=1.0)),
    ("wider hard stop 3N", dict(stop_atr=3.0)),
    ("no pyramid (1 unit)", dict(max_units=1)),
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--min-sessions", type=int, default=400)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--cost", type=float, default=0.0002,
                        help="round-trip cost as a fraction of notional")
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/exit_matched_risk.json"))
    return parser.parse_args(argv)


def load_book(args):
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    names = [r[0] for r in connection.execute(
        "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker")]
    book = {}
    for ticker in names:
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='5min' ORDER BY ts", (ticker,)).fetchall()
        if not rows:
            continue
        five = adjust_bars([Bar(*r) for r in rows], splits.get(ticker, []))
        if len({b.timestamp[:10] for b in five}) < args.min_sessions:
            continue
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


def session_closes(bars):
    """Last close of each session, in order."""
    out = {}
    for bar in bars:
        out[bar.timestamp[:10]] = bar.close
    return out


def trade_marks(trade, closes):
    """Open P&L in R at each session close the trade was held through.

    Only units already entered by that day count, so a pyramid that adds on day
    three does not retroactively inflate day one.  The exit day is excluded --
    the caller books the realised result there, costs included.
    """
    entry_day, exit_day = trade.entry_timestamp[:10], trade.exit_timestamp[:10]
    days = [d for d in closes if entry_day <= d < exit_day]
    if not days:
        return ()
    marks = []
    for day in days:
        close = closes[day]
        live = [u for u in trade.unit_entries if u.timestamp[:10] <= day]
        if not live:
            continue
        marks.append((day, sum(trade.direction * (close - u.price) / u.n
                               for u in live)))
    return tuple(marks)


def daily_series(taken, marked=False):
    """Collapse a capped trade list to (ordered days, R per day).

    Realised accounting credits the whole result on the exit day, which hides
    every open position's excursion and understates drawdown most for the rules
    that hold longest.  ``marked`` instead spreads each trade across the days it
    was actually open, using the daily closes, so a position sitting 4R underwater
    shows up as 4R underwater on the day it happens.  Costs still land at exit.
    """
    by_day = defaultdict(float)
    for trade in taken:
        if not marked or not trade.get("marks"):
            by_day[trade["exit"][:10]] += trade["r"]
            continue
        previous = 0.0
        for day, open_r in trade["marks"]:
            by_day[day] += open_r - previous
            previous = open_r
        by_day[trade["exit"][:10]] += trade["r"] - previous
    days = sorted(by_day)
    return days, [by_day[d] for d in days]


def path_metrics(days, values, risk):
    nav, peak, worst = 1000.0, 1000.0, 0.0
    for value in values:
        nav = max(0.0, nav + value * risk * nav)
        peak = max(peak, nav)
        worst = min(worst, nav / peak - 1.0)
    if len(days) < 2:
        return nav, worst, 0.0
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25
    cagr = (nav / 1000.0) ** (1 / years) - 1 if years > 0 and nav > 0 else -1.0
    return nav, worst, cagr


def quantile_dd(series, risk, q):
    """The q-th worst drawdown across tie-breaks (q=0.5 median, q=0.95 near-worst)."""
    dds = sorted(abs(path_metrics(days, values, risk)[1]) for days, values in series)
    return dds[min(int(q * len(dds)), len(dds) - 1)]


def solve_risk(series, target, q=0.5, lo=1e-6, hi=0.05):
    """Bisect for the risk level whose q-th drawdown hits the target."""
    if quantile_dd(series, hi, q) < target:
        return hi
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        if quantile_dd(series, mid, q) < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def sharpe(series):
    """Annualised Sharpe of the median tie-break's daily R stream.

    Scale-free, so it settles the ranking without any leverage assumption and
    without the compounding arithmetic that makes matched-drawdown NAVs absurd.
    Zero-R days are carried explicitly: a rule that is flat more often should not
    be rewarded for the quiet days it sits out.
    """
    scores = []
    for days, values in series:
        if len(days) < 30:
            continue
        span = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days
        idle = max(0, int(span * 252 / 365.25) - len(values))
        stream = values + [0.0] * idle
        sd = statistics.pstdev(stream)
        if sd > 0:
            scores.append(statistics.fmean(stream) / sd * math.sqrt(252))
    return statistics.median(scores) if scores else math.nan


def period_sharpe(series, year):
    """Sharpe restricted to one calendar year, so a single regime cannot carry it."""
    sliced = [([d for d in days if d[:4] == year],
               [v for d, v in zip(days, values) if d[:4] == year])
              for days, values in series]
    return sharpe([s for s in sliced if len(s[0]) >= 30])


def main(argv=None):
    args = parse_args(argv)
    book = load_book(args)
    print(f"{len(book)} instruments at {args.minutes}-minute bars", flush=True)

    BASE["round_trip_cost"] = args.cost
    base_config = TurtleConfig(**BASE)
    offers = {}
    for ticker, bars in book.items():
        index_of = {bar.timestamp: i for i, bar in enumerate(bars)}
        trades, _ = run_turtle(bars, config=base_config)
        offers[ticker] = {index_of[t.entry_timestamp]: t.direction for t in trades}
    print(f"{sum(len(v) for v in offers.values()):,} entry bars offered to every "
          f"variant\ntarget drawdown: {args.target_dd:.0%}\n", flush=True)

    report = {}
    print(f"  {'exit rule':28s} {'taken':>7s} {'Shp.exit':>9s} {'Shp.mtm':>8s} "
          f"{'lev':>6s} {'CAGR':>8s} {'@p95dd':>8s} {'hidden DD':>10s}")
    baseline_risk = None
    for label, overrides in VARIANTS:
        config = TurtleConfig(**{**BASE, **overrides})
        pooled = []
        for ticker, bars in book.items():
            trades, _ = run_turtle(bars, config=config, entries=offers[ticker])
            closes = session_closes(bars)
            pooled.extend({"entry": t.entry_timestamp, "exit": t.exit_timestamp,
                           "r": t.net_r, "marks": trade_marks(t, closes)}
                          for t in trades)
        caps = [cap(pooled, args.max_positions, random.Random(s))
                for s in range(args.trials)]
        booked = [daily_series(c) for c in caps]
        marked = [daily_series(c, marked=True) for c in caps]
        risk = solve_risk(marked, args.target_dd, q=0.5)
        safe = solve_risk(marked, args.target_dd, q=0.95)
        if baseline_risk is None:
            baseline_risk = risk
        mid = args.trials // 2
        cagrs = sorted(path_metrics(d, v, risk)[2] for d, v in marked)
        safe_cagrs = sorted(path_metrics(d, v, safe)[2] for d, v in marked)
        navs = sorted(path_metrics(d, v, risk)[0] for d, v in marked)
        booked_dd = quantile_dd(booked, risk, 0.5)
        ratio, booked_ratio = sharpe(marked), sharpe(booked)
        print(f"  {label:28s} {len(pooled):>7,d} {booked_ratio:>9.2f} "
              f"{ratio:>8.2f} {risk / baseline_risk:>5.2f}x {cagrs[mid]:>8.1%} "
              f"{safe_cagrs[mid]:>8.1%} {args.target_dd - booked_dd:>10.1%}")
        report[label] = {"years": {y: period_sharpe(marked, y)
                                   for y in sorted({d[:4] for d in marked[0][0]})},
                         "total_r": sum(t["r"] for t in pooled),
                         "trades": len(pooled), "sharpe": ratio, "sharpe_booked": booked_ratio,
                         "booked_dd_at_matched_risk": booked_dd,
                         "risk": risk, "risk_vs_baseline": risk / baseline_risk,
                         "risk_p95dd": safe, "cagr": cagrs[mid],
                         "cagr_p95dd": safe_cagrs[mid],
                         "final_median": navs[mid],
                         "final_p05": navs[int(.05 * len(navs))]}

    ranked = sorted(report.items(), key=lambda kv: -kv[1]["sharpe"])
    base = report["channel 20 (baseline)"]
    print(f"\n  at matched {args.target_dd:.0%} drawdown, ranked by return:")
    for i, (label, data) in enumerate(ranked[:4], 1):
        gap = data["sharpe"] - base["sharpe"]
        print(f"    {i}. {label:28s} Sharpe {data['sharpe']:>5.2f} "
              f"({gap:+.2f} vs baseline), carries {data['risk_vs_baseline']:.2f}x "
              f"the risk at equal drawdown")
    years = sorted(report["channel 20 (baseline)"]["years"])
    print("\n  Sharpe by calendar year (marked accounting):")
    print("    " + f"{'exit rule':28s}" + "".join(f"{y:>8s}" for y in years))
    for label in [r[0] for r in ranked[:3]] + ["channel 20 (baseline)"]:
        cells = "".join(f"{report[label]['years'].get(y, float('nan')):>8.2f}"
                        for y in years)
        print(f"    {label:28s}{cells}")
    beat = {y: sum(1 for lbl in ranked[:3]
                   if report[lbl[0]]["years"].get(y, -9) >
                   report["channel 20 (baseline)"]["years"].get(y, 9))
            for y in years}
    print(f"    -> top-3 variants beating baseline: "
          f"{sum(beat.values())}/{3 * len(years)} year-slots")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
