"""How far to pyramid, and whether the edge survives on the names that collapsed.

Two questions that belong together, because both are about where the return
actually comes from.

*Pyramiding.* The one controlled test in the literature -- Zarattini's, over 40
futures markets since 1980, varying only the sizing rule -- found pyramiding
lifted IRR from 11.5% to 20.0% while pushing max drawdown from 25.7% to 48.7%
and lowering the Sharpe ratio. Average trade P&L doubled while the median trade
got worse: a lottery distribution where a few trades carry everything. That is
a CAGR intervention, not a risk-adjusted one, and since optimal leverage is
Sharpe/sigma, pyramiding cuts safe leverage from both directions. So the sweep
reports the trade-off explicitly rather than a single number -- Sharpe at matched
drawdown, and CAGR and drawdown at fixed risk, because the second pair is what
actually changes.

*Breadth and survivorship.* This universe is the Binance trad-FI list as it
stands today, which is a hindsight selection: names that survived to be listed.
The strategy's headline numbers inherit that. Splitting the universe by each
instrument's own buy-and-hold outcome over the period tests it directly -- if the
edge exists only among names that went up, it is a restatement of drift with
extra steps. If it holds among the names that collapsed, it is a trading result.

The second question also bears on breadth. IR = IC x sqrt(breadth), and 118
correlated US equities are nowhere near 118 independent bets. A strategy that
works only on the winners has an effective breadth close to one.
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

from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import TurtleConfig, run_turtle  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

BASE = dict(entry_window=55, exit_window=20, atr_window=20,
            skip_after_winner=False, use_channel_exit=False, chandelier_atr=3.0)

LEVERED = ("2X", "3X", "1.5X", "BULL", "BEAR", "LEVERAGED", "INVERSE",
           "ULTRA", "DAILY ", "SHORT ")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/intraday/bars_av.db"))
    parser.add_argument("--minutes", type=int, default=240)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--cost-bp", type=float, default=10.0)
    parser.add_argument("--max-units", type=int, nargs="+", default=(1, 2, 4, 6, 8))
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--fixed-risk", type=float, default=0.0020,
                        help="risk per R for the unmatched CAGR comparison")
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/pyramid_breadth.json"))
    return parser.parse_args(argv)


def load(args):
    connection = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT ticker, name FROM symbols WHERE name IS NOT NULL").fetchall()
        drop = {t for t, n in rows if any(m in n.upper() for m in LEVERED)}
    except sqlite3.OperationalError:
        drop = set()
    tickers = [r[0] for r in connection.execute(
        "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker")
        if r[0] not in drop]
    if args.limit:
        tickers = tickers[:args.limit]
    book = {}
    for ticker in tickers:
        raw = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='5min' AND ts>=? ORDER BY ts", (ticker, args.start)).fetchall()
        if len(raw) < 4000:
            continue
        bars = resample_regular_session([Bar(*r) for r in raw], minutes=args.minutes)
        if len(bars) >= 400:
            book[ticker] = bars
    connection.close()
    print(f"excluded {len(drop)} levered or inverse wrappers")
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


def marked_map(taken, closes_by_ticker):
    by_day = defaultdict(float)
    for trade in taken:
        closes = closes_by_ticker[trade["ticker"]]
        entry_day, exit_day = trade["entry"][:10], trade["exit"][:10]
        previous = 0.0
        for day in (d for d in closes if entry_day <= d < exit_day):
            live = [u for u in trade["units"] if u.timestamp[:10] <= day]
            if not live:
                continue
            open_r = sum(trade["dir"] * (closes[day] - u.price) / u.n for u in live)
            by_day[day] += open_r - previous
            previous = open_r
        by_day[exit_day] += trade["r"] - previous
    return by_day


def path(values, risk):
    nav = peak = 1000.0
    worst = 0.0
    for value in values:
        nav = max(0.0, nav + value * risk * nav)
        peak = max(peak, nav)
        worst = min(worst, nav / peak - 1.0)
    return nav, worst


def solve_risk(series, target, lo=1e-6, hi=0.40):
    def dd(risk):
        return statistics.median(abs(path(v, risk)[1]) for _, v in series)
    if dd(hi) < target:
        return hi
    for _ in range(28):
        mid = math.sqrt(lo * hi)
        if dd(mid) < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def sharpe(stream):
    sd = statistics.pstdev(stream)
    return statistics.fmean(stream) / sd * math.sqrt(252) if sd > 0 else float("nan")


def cagr_of(nav, days):
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25
    return (nav / 1000.0) ** (1 / years) - 1 if years > 0 and nav > 0 else -1.0


def evaluate(pooled, closes_by_ticker, args):
    if len(pooled) < 100:
        return None
    caps = [cap(pooled, args.max_positions, random.Random(s)) for s in range(args.trials)]
    marks = [marked_map(t, closes_by_ticker) for t in caps]
    series = [(sorted(m), [m[d] for d in sorted(m)]) for m in marks]
    risk = solve_risk(series, args.target_dd)
    matched_sharpe, matched_cagr = [], []
    fixed_cagr, fixed_dd = [], []
    for days, values in series:
        nav, _ = path(values, risk)
        matched_sharpe.append(sharpe([v * risk for v in values]))
        matched_cagr.append(cagr_of(nav, days))
        nav2, worst2 = path(values, args.fixed_risk)
        fixed_cagr.append(cagr_of(nav2, days))
        fixed_dd.append(worst2)
    every = [t["r"] for t in pooled]
    return {
        "trades": len(pooled),
        "sharpe_matched": statistics.median(matched_sharpe),
        "cagr_matched": statistics.median(matched_cagr),
        "cagr_fixed": statistics.median(fixed_cagr),
        "drawdown_fixed": statistics.median(fixed_dd),
        "mean_r": statistics.fmean(every),
        "median_r": statistics.median(every),
        "share_above_3r": sum(1 for r in every if r > 3.0) / len(every),
        "share_stopped": sum(1 for r in every if r <= -0.99) / len(every),
        "risk_matched": risk,
    }


def pooled_for(book, config, closes_by_ticker):
    out = []
    for ticker, bars in book.items():
        trades, _ = run_turtle(bars, config=config)
        out.extend({"ticker": ticker, "entry": t.entry_timestamp,
                    "exit": t.exit_timestamp, "r": t.net_r, "dir": t.direction,
                    "units": t.unit_entries} for t in trades)
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    book = load(args)
    closes_by_ticker = {t: {b.timestamp[:10]: b.close for b in bars}
                        for t, bars in book.items()}
    print(f"{len(book)} instruments, {args.minutes}-minute bars, "
          f"{args.cost_bp:g}bp\n", flush=True)
    report = {}

    print("=== how far to pyramid ===")
    print(f"  {'units':>6s} {'trades':>8s} | matched to 18% DD      | at fixed "
          f"{args.fixed_risk*10_000:.0f}bp risk | distribution")
    print(f"  {'':>6s} {'':>8s} | {'Sharpe':>7s} {'CAGR':>8s} | "
          f"{'CAGR':>8s} {'maxDD':>8s} | {'medR':>7s} {'>+3R':>6s} {'stop':>6s}")
    for units in args.max_units:
        config = TurtleConfig(**BASE, directions=(1,), max_units=units,
                              round_trip_cost=args.cost_bp / 10_000)
        pooled = pooled_for(book, config, closes_by_ticker)
        result = evaluate(pooled, closes_by_ticker, args)
        if not result:
            continue
        report[f"units_{units}"] = {"max_units": units, **result}
        print(f"  {units:>6d} {result['trades']:>8,d} | "
              f"{result['sharpe_matched']:>7.2f} {result['cagr_matched']:>8.1%} | "
              f"{result['cagr_fixed']:>8.1%} {result['drawdown_fixed']:>8.1%} | "
              f"{result['median_r']:>7.2f} {result['share_above_3r']:>6.1%} "
              f"{result['share_stopped']:>6.1%}", flush=True)

    # --- does the edge survive on the names that did badly? ---
    outcome = {}
    for ticker, bars in book.items():
        first, last = bars[0].close, bars[-1].close
        if first > 0:
            outcome[ticker] = last / first - 1.0
    ordered = sorted(outcome, key=lambda t: outcome[t])
    third = len(ordered) // 3
    buckets = {"collapsed (worst third)": ordered[:third],
               "middle third": ordered[third:2 * third],
               "trended (best third)": ordered[2 * third:]}

    print(f"\n=== does it work on the names that fell? "
          f"(4 units, {args.cost_bp:g}bp) ===")
    print(f"  {'bucket':26s} {'names':>6s} {'buy&hold':>10s} {'trades':>8s} "
          f"{'Sharpe':>7s} {'CAGR':>8s} {'medR':>7s} {'>+3R':>6s}")
    config = TurtleConfig(**BASE, directions=(1,), max_units=4,
                          round_trip_cost=args.cost_bp / 10_000)
    for label, names in buckets.items():
        subset = {t: book[t] for t in names}
        subcloses = {t: closes_by_ticker[t] for t in names}
        pooled = pooled_for(subset, config, subcloses)
        result = evaluate(pooled, subcloses, args)
        if not result:
            continue
        hold = statistics.median(outcome[t] for t in names)
        report[f"bucket_{label}"] = {"bucket": label, "names": len(names),
                                     "median_buy_hold": hold, **result}
        print(f"  {label:26s} {len(names):>6d} {hold:>9.0%} "
              f"{result['trades']:>8,d} {result['sharpe_matched']:>7.2f} "
              f"{result['cagr_matched']:>8.1%} {result['median_r']:>7.2f} "
              f"{result['share_above_3r']:>6.1%}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
