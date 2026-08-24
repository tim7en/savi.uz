"""Margin of safety as a measurable thing: buying at a discount to the recent high.

The thesis, written down before the test rather than after it: within a fixed
universe, entries taken at a larger discount to the name's own recent high
outperform entries taken at a smaller one, and the effect is **monotone in the
discount**.  Monotonicity is the part that matters.  A single bucket beating the
rest is what a search finds in noise; an ordering that holds across four buckets
is much harder to produce by accident, and it is a prediction that can fail
cleanly.

Three guards, each answering a way this could fool us.

*The window is swept, not chosen.*  50, 100, 200 and 252 sessions.  Naming a
range in advance -- "100 to 200 days" -- is how the 3x threshold leaked into a
result earlier in this programme, and the fix is to hand the sweep a wider range
than the intuition and let the first half pick.

*Survivorship is confronted, not avoided.*  A drawdown rule buys things that are
falling, and some of them are dying.  This universe is the right place to find
that out because it contains the names that never came back -- BHC, CHGG, RIG,
TDOC -- rather than a list assembled from things that worked.

*Clusters are capped.*  A drawdown rule fires across a whole cluster at once:
all six energy names fell together in 2020.  Without a per-cluster limit twelve
slots become one bet.  The clusters are measured on the in-sample window only
and applied forward, so the grouping cannot see the period it is used in.

The reversal control is the exact opposite rule -- enter at a *new high*, no
discount at all.  If buying strength does as well as buying weakness, there is no
margin of safety here, only trading.
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
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from savi_uz.risk_clustering import (  # noqa: E402
    average_linkage, correlation_matrix, distance_for_correlation, log_returns,
    resample_weekly,
)
from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

import run_vol_stretch_zones as shared  # noqa: E402

WINDOWS = (50, 100, 200, 252)
DEPTHS = (0.10, 0.20, 0.30)
HOLDS = (20, 40, 60)
DEPTH_BUCKETS = ((0.05, 0.15), (0.15, 0.25), (0.25, 0.40), (0.40, 1.01))
HORIZONS = (10, 20, 40, 60)
RISK_RUNGS = (0.0025, 0.005, 0.01, 0.02)
LEVERAGE_CAP = 5.0


def ticker_seed(t):
    return zlib.crc32(t.encode("utf-8")) % 10_000


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path,
                        default=Path("data/13f/alphavantage_daily.db"))
    parser.add_argument("--split", default="2013-01-01")
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--stop-mult", type=float, default=3.0)
    parser.add_argument("--taker-bp", type=float, default=5.0)
    parser.add_argument("--max-positions", type=int, default=12)
    parser.add_argument("--per-cluster", type=int, default=2)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--null-draws", type=int, default=30)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/margin_of_safety.json"))
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


def cluster_map(book, args):
    """Correlation clusters measured on the in-sample window only."""
    frame = {}
    for ticker, bars in book.items():
        rows = {b.timestamp: b.close for b in bars if b.timestamp < args.split}
        if len(rows) >= 250:
            frame[ticker] = pd.Series(rows, dtype=float)
    prices = pd.DataFrame(frame).sort_index()
    prices.index = pd.to_datetime(prices.index)
    returns = log_returns(resample_weekly(prices)).dropna(how="all")
    keep = returns.columns[returns.notna().sum() >= 60]
    corr, _ = correlation_matrix(returns[keep], min_periods=40, shrinkage=0.10)
    groups = average_linkage(corr).cut(distance_for_correlation(0.45))
    label = {}
    for index, members in enumerate(groups):
        for name in members:
            label[name] = index
    print(f"  clusters measured on {args.split[:4]}-and-earlier data only: "
          f"{len(groups)} groups over {len(keep)} names")
    return label


def build(book, args):
    """Every drawdown crossing, with forward moves and the depth at entry."""
    events = defaultdict(list)
    for ticker, bars in book.items():
        closes = [b.close for b in bars]
        opens = [b.open for b in bars]
        lows = [b.low for b in bars]
        days = [b.timestamp for b in bars]
        returns, sigma = [], {}
        for i in range(1, len(bars)):
            a, b = closes[i - 1], closes[i]
            if a > 0 and b > 0:
                returns.append(math.log(b / a))
            if len(returns) > args.vol_window:
                returns.pop(0)
            if len(returns) == args.vol_window:
                sigma[days[i]] = statistics.pstdev(returns)
        for window in WINDOWS:
            peak, armed = [], {d: True for d in DEPTHS}
            running = []
            for i in range(len(bars)):
                running.append(closes[i])
                if len(running) > window:
                    running.pop(0)
                peak.append(max(running))
            for depth in DEPTHS:
                armed = True
                for i in range(max(window, args.vol_window) + 1,
                               len(bars) - max(HORIZONS) - 2):
                    if peak[i] <= 0:
                        continue
                    drop = closes[i] / peak[i] - 1.0
                    if drop > -depth * 0.5:
                        armed = True
                    if not armed or drop > -depth:
                        continue
                    armed = False
                    s = sigma.get(days[i])
                    entry = opens[i + 1]
                    if not s or s <= 0 or entry <= 0:
                        continue
                    row = {"ticker": ticker, "index": i, "day": days[i],
                           "depth": -drop, "sigma": s, "entry": entry}
                    for h in HORIZONS:
                        row[f"fwd_{h}"] = (closes[i + 1 + h] - entry) / (s * entry)
                    events[(window, depth)].append(row)
    return events


def baseline(book, args):
    """Unconditional forward move on the same names, same normalisation."""
    out = defaultdict(list)
    for ticker, bars in book.items():
        closes = [b.close for b in bars]
        opens = [b.open for b in bars]
        days = [b.timestamp for b in bars]
        returns, sigma = [], {}
        for i in range(1, len(bars)):
            a, b = closes[i - 1], closes[i]
            if a > 0 and b > 0:
                returns.append(math.log(b / a))
            if len(returns) > args.vol_window:
                returns.pop(0)
            if len(returns) == args.vol_window:
                sigma[days[i]] = statistics.pstdev(returns)
        for i in range(args.vol_window + 2, len(bars) - max(HORIZONS) - 2):
            if (i * 2654435761) % 100 >= 20:
                continue
            s, entry = sigma.get(days[i]), opens[i + 1]
            if not s or s <= 0 or entry <= 0:
                continue
            for h in HORIZONS:
                out[h].append((closes[i + 1 + h] - entry) / (s * entry))
    return {h: statistics.fmean(v) for h, v in out.items()}


def trade(book, rows, hold, args):
    out = []
    for row in rows:
        bars = book[row["ticker"]]
        start = row["index"] + 1
        fill = bars[start].open
        risk = args.stop_mult * row["sigma"] * fill
        if fill <= 0 or risk <= 0:
            continue
        stop = fill - risk
        last = min(start + hold, len(bars) - 1)
        price, reason, when = bars[last].close, "time", bars[last].timestamp
        for i in range(start, last + 1):
            if bars[i].low <= stop:
                price, reason, when = stop, "stop", bars[i].timestamp
                break
        cost = 2 * args.taker_bp / 10_000 * fill / risk
        out.append({"ticker": row["ticker"], "entry": row["day"], "exit": when,
                    "r": (price - fill) / risk - cost, "reason": reason,
                    "stop_pct": risk / fill, "depth": row["depth"]})
    out.sort(key=lambda t: t["entry"])
    return out


def cap_clustered(trades, limit, per_cluster, labels, rng):
    shuffled = list(trades)
    rng.shuffle(shuffled)
    live, taken = [], []
    for t in sorted(shuffled, key=lambda x: x["entry"]):
        live = [x for x in live if x["exit"] > t["entry"]]
        if len(live) >= limit:
            continue
        if per_cluster:
            group = labels.get(t["ticker"])
            if group is not None and sum(
                    1 for x in live if labels.get(x["ticker"]) == group) >= per_cluster:
                continue
        live.append(t)
        taken.append(t)
    return taken


def compound(taken, fraction, cap=LEVERAGE_CAP):
    per_day, levers = defaultdict(float), []
    for t in taken:
        lever = min(fraction / t["stop_pct"], cap)
        levers.append(lever)
        per_day[t["exit"]] += t["r"] * lever * t["stop_pct"]
    days = sorted(per_day)
    if not days:
        return None
    nav, peak, worst = 1000.0, 1000.0, 0.0
    for d in days:
        nav = max(0.0, nav + per_day[d] * nav)
        peak = max(peak, nav)
        worst = min(worst, nav / peak - 1.0)
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25
    return {"cagr": (nav / 1000.0) ** (1 / years) - 1 if nav > 0 else -1.0,
            "max_drawdown": worst, "median_leverage": statistics.median(levers)}


def main(argv=None) -> int:
    args = parse_args(argv)
    book = load_book(args)
    labels = cluster_map(book, args)
    events = build(book, args)
    base = baseline(book, args)
    print(f"{len(book)} names, drawdown crossings built for "
          f"{len(WINDOWS)}x{len(DEPTHS)} window/depth pairs")
    print(f"in sample to {args.split}, out of sample after\n")
    report = {"baseline": base}

    print("########## the prediction: is the discount monotone? (in sample) ##########")
    print("  Forward move from the next open, in the name's own daily deviations.")
    print(f"  {'unconditional':22s} " +
          " ".join(f"{'+'+str(h):>9s}" for h in HORIZONS))
    print(f"  {'all sessions':22s} " +
          " ".join(f"{base[h]:>+9.3f}" for h in HORIZONS))
    pooled = [r for (w, d), rows in events.items() if w == 200 and d == 0.10
              for r in rows if r["day"] < args.split]
    print(f"\n  by depth at entry, 200-session high, {len(pooled):,d} crossings")
    print(f"  {'depth below the high':22s} {'n':>7s} " +
          " ".join(f"{'+'+str(h):>9s}" for h in HORIZONS))
    mono = []
    for lo, hi in DEPTH_BUCKETS:
        chunk = [r for r in pooled if lo <= r["depth"] < hi]
        if len(chunk) < 60:
            continue
        means = [statistics.fmean(r[f"fwd_{h}"] for r in chunk) for h in HORIZONS]
        mono.append(means[2])
        band = f"{lo:.0%} to {hi:.0%}" if hi < 1 else f"{lo:.0%}+"
        print(f"  {band:22s} {len(chunk):>7,d} " +
              " ".join(f"{m:>+9.3f}" for m in means))
        report.setdefault("monotonicity", {})[band] = {
            "n": len(chunk), "forward": dict(zip(map(str, HORIZONS), means))}
    if len(mono) >= 3:
        steps = sum(1 for a, b in zip(mono, mono[1:]) if b > a)
        print(f"\n  deeper is better in {steps} of {len(mono)-1} steps at +40 "
              f"sessions  -> {'monotone' if steps == len(mono)-1 else 'NOT monotone'}")
        report["monotone_steps"] = f"{steps}/{len(mono)-1}"

    print("\n########## selection on the first half ##########")
    best, best_score, cache = None, -99.0, {}
    for (window, depth), rows in events.items():
        for hold in HOLDS:
            trades = trade(book, rows, hold, args)
            early = [t for t in trades if t["entry"] < args.split]
            if len(early) < 250:
                continue
            result = shared.assess(early, args)
            if result:
                cache[(window, depth, hold)] = trades
                if result["sharpe"] > best_score:
                    best_score, best = result["sharpe"], (window, depth, hold)
    if best is None:
        print("  nothing cleared the minimum trade count")
        return 1
    print(f"  {'window':>8s} {'depth':>7s} {'hold':>6s} {'trades':>8s} {'Sharpe':>7s}")
    for key in sorted(cache, key=lambda k: -shared.assess(
            [t for t in cache[k] if t["entry"] < args.split], args)["sharpe"])[:6]:
        early = [t for t in cache[key][0:] if t["entry"] < args.split]
        mark = " <- chosen" if key == best else ""
        print(f"  {key[0]:>8d} {key[1]:>6.0%} {key[2]:>6d} {len(early):>8,d} "
              f"{shared.assess(early, args)['sharpe']:>7.2f}{mark}")
    window, depth, hold = best
    report["chosen"] = {"window": window, "depth": depth, "hold": hold,
                        "in_sample_sharpe": best_score}
    print(f"\n  frozen: {depth:.0%} below the {window}-session high, held {hold}")

    print(f"\n########## out of sample ##########")
    trades = cache[best]
    outside = [t for t in trades if t["entry"] >= args.split]
    print(f"  {'arm':34s} {'offered':>8s} {'taken':>7s} {'Sharpe':>7s} "
          f"{'[5-95%]':>14s}")

    def score(label, pooled, per_cluster=0):
        if len(pooled) < 150:
            print(f"  {label:34s} {len(pooled):>8,d}   too few")
            return None
        caps = [cap_clustered(pooled, args.max_positions, per_cluster, labels,
                              random.Random(s)) for s in range(args.trials)]
        marks = []
        for taken in caps:
            per_day = defaultdict(float)
            for t in taken:
                per_day[t["exit"]] += t["r"]
            days = sorted(per_day)
            marks.append((days, [per_day[d] for d in days]))
        risk = shared.solve_risk(marks, args.target_dd)
        sharpes, counts = [], []
        for (days, values), taken in zip(marks, caps):
            sharpes.append(shared.sharpe([v * risk for v in values]))
            counts.append(len(taken))
        sharpes.sort()
        out = {"offered": len(pooled), "taken": int(statistics.median(counts)),
               "sharpe": statistics.median(sharpes),
               "p05": sharpes[int(.05 * len(sharpes))],
               "p95": sharpes[min(int(.95 * len(sharpes)), len(sharpes) - 1)]}
        report.setdefault("arms", {})[label] = out
        print(f"  {label:34s} {out['offered']:>8,d} {out['taken']:>7,d} "
              f"{out['sharpe']:>7.2f} "
              f"{('[%.2f-%.2f]' % (out['p05'], out['p95'])):>14s}", flush=True)
        return out

    book_result = score("the discount book", outside)
    score(f"+ max {args.per_cluster} per cluster", outside, args.per_cluster)

    # the reversal: buy new highs instead of discounts
    highs = []
    for ticker, bars in book.items():
        closes = [b.close for b in bars]
        days = [b.timestamp for b in bars]
        returns, sigma = [], {}
        for i in range(1, len(bars)):
            a, b = closes[i - 1], closes[i]
            if a > 0 and b > 0:
                returns.append(math.log(b / a))
            if len(returns) > args.vol_window:
                returns.pop(0)
            if len(returns) == args.vol_window:
                sigma[days[i]] = statistics.pstdev(returns)
        run, armed = [], True
        for i in range(len(bars)):
            run.append(closes[i])
            if len(run) > window:
                run.pop(0)
            if i < max(window, args.vol_window) + 1 or i >= len(bars) - 70:
                continue
            at_high = closes[i] >= max(run) - 1e-9
            if not at_high:
                armed = True
                continue
            if not armed:
                continue
            armed = False
            s = sigma.get(days[i])
            if s and s > 0:
                highs.append({"ticker": ticker, "index": i, "day": days[i],
                              "sigma": s, "depth": 0.0})
    rev = trade(book, highs, hold, args)
    score("new highs instead (reversal)", [t for t in rev if t["entry"] >= args.split])

    if book_result:
        nulls = []
        for draw in range(args.null_draws):
            picked = []
            want = defaultdict(int)
            for t in outside:
                want[t["ticker"]] += 1
            for ticker, count in want.items():
                bars = book[ticker]
                rng = random.Random(31_500 + 137 * draw + ticker_seed(ticker))
                floor = max(WINDOWS) + args.vol_window + 2
                pool = range(floor, len(bars) - 70)
                if len(pool) <= count:
                    continue
                for i in rng.sample(list(pool), count):
                    returns = [math.log(bars[j].close / bars[j - 1].close)
                               for j in range(i - args.vol_window, i)
                               if bars[j].close > 0 and bars[j - 1].close > 0]
                    if len(returns) < 5:
                        continue
                    picked.append({"ticker": ticker, "index": i,
                                   "day": bars[i].timestamp,
                                   "sigma": statistics.pstdev(returns),
                                   "depth": 0.0})
            drawn = trade(book, sorted(picked, key=lambda r: r["day"]), hold, args)
            pooled = [t for t in drawn if t["entry"] >= args.split]
            if len(pooled) < 150:
                continue
            outcome = shared.assess(pooled, args)
            if outcome:
                nulls.append(outcome["sharpe"])
        if nulls:
            nulls.sort()
            above = sum(1 for x in nulls if x >= book_result["sharpe"]) / len(nulls)
            report["drift_null"] = {"median": statistics.median(nulls),
                                    "low": nulls[0], "high": nulls[-1], "p": above}
            print(f"  {'random days (drift null)':34s} {'':>8s} {'':>7s} "
                  f"{statistics.median(nulls):>7.2f} "
                  f"{('[%.2f-%.2f]' % (nulls[0], nulls[-1])):>14s}")
            print(f"  -> p = {above:.2f}, "
                  f"{'clears' if above <= 0.05 else 'inside'} its null")

    print(f"\n########## compounding, capped at {LEVERAGE_CAP:g}x ##########")
    taken = cap_clustered(outside, args.max_positions, args.per_cluster, labels,
                          random.Random(0))
    print(f"  {'risk':>6s} {'CAGR':>9s} {'max DD':>9s} {'median lev':>11s}")
    for fraction in RISK_RUNGS:
        row = compound(taken, fraction)
        if row:
            report.setdefault("compounding", []).append({"risk": fraction, **row})
            print(f"  {fraction:>5.2%} {row['cagr']:>+9.1%} "
                  f"{row['max_drawdown']:>9.1%} {row['median_leverage']:>10.2f}x")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
