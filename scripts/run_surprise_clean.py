"""The surprise book with nothing chosen after the fact.

The previous version of this test leaked, and the leak was not obvious.  Its
3x threshold came from an event study computed over the whole span, including
the years it was then evaluated on, and the sweep was handed only the values
that study had already endorsed.  A split cannot fix a search space that was
narrowed by the answer.

Here every choice is made on the first half and nothing else is looked at:

* the threshold is swept from 1.0 deviations upward, so the sweep has to find
  its own cliff rather than being pointed at one;
* the stop, the target and the holding period are swept with it;
* **direction is not selected.**  All four combinations of trigger sign and
  response -- follow an up move, follow a down move, fade an up move, fade a
  down move -- get their own in-sample search and their own out-of-sample
  number.  Picking the best direction after the fact is the same error as
  picking the threshold after the fact, and reporting all four is the fix.

The volatility normaliser is a trailing realised deviation.  No option chain is
read, which makes the same code run on the wide daily universe where no chain
exists, and lets the two universes be compared on identical terms.

What is reported at the end is money, not only ratio.  A Sharpe says nothing
about whether the equity curve is holdable, so each surviving arm is sized
fixed-fractional -- a risk budget divided by the stop, capped at the venue
maximum -- and the drawdown, the compound return, and the leverage that budget
actually produces are reported together.

Bracket resolution charges the stop whenever one bar touches both levels.  On
the intraday universe that bar is five minutes wide; on the daily universe it is
a whole session, which binds far more often and biases those numbers down.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

import run_vol_stretch_zones as shared  # noqa: E402

STRETCHES = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
STOPS = (2.0, 3.0)
TARGETS = (2.0, 3.0)
HOLDS = (5, 10)
MAX_HOLD = max(HOLDS)
RISK_RUNGS = (0.005, 0.01, 0.02, 0.03)
ARMS = (("follow", 1, "long"), ("follow", -1, "short"),
        ("fade", 1, "short"), ("fade", -1, "long"))


def ticker_seed(ticker):
    return zlib.crc32(ticker.encode("utf-8")) % 10_000


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", choices=("intraday", "daily"),
                        default="intraday")
    parser.add_argument("--bars", type=Path, default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--split", default=None,
                        help="first out-of-sample session")
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--quiet-share", type=int, default=12,
                        help="percent of ordinary sessions kept for the null pool")
    parser.add_argument("--maker-bp", type=float, default=2.5)
    parser.add_argument("--taker-bp", type=float, default=5.0)
    parser.add_argument("--max-leverage", type=float, default=20.0)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--null-draws", type=int, default=50)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.universe == "intraday":
        args.bars = args.bars or Path("data/intraday/bars.db")
        args.start = args.start or "2017-01-01"
        args.split = args.split or "2022-01-01"
        args.out = args.out or Path("out/strategy/surprise_clean_intraday.json")
    else:
        args.bars = args.bars or Path("data/13f/alphavantage_daily.db")
        args.start = args.start or "1999-11-01"
        args.split = args.split or "2013-01-01"
        args.out = args.out or Path("out/strategy/surprise_clean_daily.json")
    return args


# ---------------------------------------------------------------------------
# data


def daily_from(bars):
    groups = defaultdict(list)
    for bar in bars:
        groups[bar.timestamp[:10]].append(bar)
    out = []
    for day in sorted(groups):
        rows = groups[day]
        out.append(Bar(day, rows[0].open, max(r.high for r in rows),
                       min(r.low for r in rows), rows[-1].close, None))
    return out


def load(args):
    """Per name: the series fills resolve on, and one bar per session."""
    frequency = "5min" if args.universe == "intraday" else "daily"
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    book = {}
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency=? ORDER BY ticker",
            (frequency,)):
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency=? AND ts>=? ORDER BY ts",
            (ticker, frequency, args.start)).fetchall()
        if args.universe == "intraday":
            raw = adjust_bars([Bar(*r) for r in rows], splits.get(ticker, []))
            fills = resample_regular_session(raw, minutes=5)
            if len({b.timestamp[:10] for b in fills}) < 400:
                continue
            book[ticker] = (fills, daily_from(fills))
        else:
            bars = adjust_bars([Bar(r[0][:10], *r[1:]) for r in rows],
                               splits.get(ticker, []))
            if len(bars) < 750:
                continue
            book[ticker] = (bars, bars)
    connection.close()
    return book


def deviations(book, window):
    panel = {}
    for ticker, (_, daily) in book.items():
        table, returns = {}, []
        for index in range(1, len(daily)):
            a, b = daily[index - 1].close, daily[index].close
            if a > 0 and b > 0:
                returns.append(math.log(b / a))
            if len(returns) > window:
                returns.pop(0)
            if len(returns) == window:
                table[daily[index].timestamp[:10]] = statistics.pstdev(returns)
        if len(table) >= 200:
            panel[ticker] = table
    return panel


def session_spans(bars):
    spans, first = [], 0
    while first < len(bars):
        day = bars[first].timestamp[:10]
        last = first
        while last + 1 < len(bars) and bars[last + 1].timestamp[:10] == day:
            last += 1
        spans.append((day, first, last))
        first = last + 1
    return spans


def build_paths(book, panel, args):
    """Every trigger, with the forward path its bracket will be walked along.

    Extracted once.  The path does not depend on the stop, the target or the
    holding period, so the sweep can be vectorised over it instead of walking
    the same bars a few hundred times.
    """
    meta, highs, lows, ends = [], [], [], []
    for ticker, (fills, daily) in book.items():
        history = panel.get(ticker)
        if not history:
            continue
        spans = session_spans(fills)
        day_at = {day: i for i, (day, _, _) in enumerate(spans)}
        closes = [b.close for b in daily]
        for index in range(1, len(daily) - 1):
            day = daily[index].timestamp[:10]
            deviation = history.get(daily[index - 1].timestamp[:10])
            base = closes[index - 1]
            if not deviation or deviation <= 0 or base <= 0:
                continue
            change = (closes[index] - base) / base
            if change == 0:
                continue
            position = day_at.get(day)
            if position is None or position + 1 >= len(spans):
                continue
            start = spans[position + 1][1]
            stop_session = min(position + MAX_HOLD, len(spans) - 1)
            finish = spans[stop_session][2]
            path = fills[start:finish + 1]
            if len(path) < 2:
                continue
            z = abs(change) / deviation
            if z < 1.0 and (index * 2654435761) % 100 >= args.quiet_share:
                continue   # the null only needs a sample of ordinary sessions
            end_of = []
            for h in range(1, MAX_HOLD + 1):
                s = min(position + h, len(spans) - 1)
                end_of.append(min(spans[s][2] - start, len(path) - 1))
            end_days = []
            for h in range(1, MAX_HOLD + 1):
                sess = min(position + h, len(spans) - 1)
                end_days.append(spans[sess][0])
            meta.append({"ticker": ticker, "day": day,
                         "direction": 1 if change > 0 else -1,
                         "z": abs(change) / deviation,
                         "move": deviation * base,
                         "fill": path[0].open,
                         "ends": end_of, "end_days": end_days,
                         "closes": [path[e].close for e in end_of]})
            highs.append(np.array([b.high for b in path], dtype=np.float32))
            lows.append(np.array([b.low for b in path], dtype=np.float32))

    width = max(len(a) for a in highs)
    H = np.full((len(highs), width), np.nan, dtype=np.float32)
    L = np.full((len(lows), width), np.nan, dtype=np.float32)
    for i, (a, b) in enumerate(zip(highs, lows)):
        H[i, :len(a)] = a
        L[i, :len(b)] = b
    return meta, H, L


def resolve(meta, H, L, mask, side, stop_mult, target_r, hold, args):
    """Vectorised bracket: first stop, first target, else the clock."""
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return []
    fill = np.array([meta[i]["fill"] for i in idx], dtype=np.float64)
    move = np.array([meta[i]["move"] for i in idx], dtype=np.float64)
    last = np.array([meta[i]["ends"][hold - 1] for i in idx])
    risk = stop_mult * move
    stop = fill - side * risk
    target = fill + side * target_r * risk

    width = H.shape[1]
    grid = np.arange(width)[None, :]
    live = grid <= last[:, None]
    hi, lo = H[idx], L[idx]
    if side > 0:
        hit_stop, hit_target = (lo <= stop[:, None]), (hi >= target[:, None])
    else:
        hit_stop, hit_target = (hi >= stop[:, None]), (lo <= target[:, None])
    hit_stop &= live
    hit_target &= live
    big = width + 1
    first_stop = np.where(hit_stop.any(1), hit_stop.argmax(1), big)
    first_target = np.where(hit_target.any(1), hit_target.argmax(1), big)

    stopped = first_stop <= first_target          # ties charge the stop
    hit = np.minimum(first_stop, first_target) < big
    price = np.where(hit, np.where(stopped, stop, target),
                     np.array([meta[i]["closes"][hold - 1] for i in idx]))
    reason = np.where(~hit, 2, np.where(stopped, 0, 1))   # 0 stop 1 target 2 time
    exit_at = np.where(hit, np.minimum(first_stop, first_target), last)

    exit_leg = np.where(reason == 1, args.maker_bp, args.taker_bp)
    cost = (args.taker_bp + exit_leg) / 10_000 * fill / risk
    r = side * (price - fill) / risk - cost

    out = []
    for k, i in enumerate(idx):
        out.append({"ticker": meta[i]["ticker"], "entry": meta[i]["day"],
                    "exit_step": int(exit_at[k]), "r": float(r[k]),
                    "reason": ("stop", "target", "time")[int(reason[k])],
                    "stop_pct": float(risk[k] / fill[k]), "meta": i})
    return out


def arm_trades(meta, H, L, args, action, trigger, stretch, stop_mult, target_r,
               hold, null_seed=None):
    side = trigger if action == "follow" else -trigger
    z = np.array([m["z"] for m in meta])
    d = np.array([m["direction"] for m in meta])
    if null_seed is None:
        mask = (d == trigger) & (z >= stretch)
    else:
        wanted = defaultdict(int)
        for i, m in enumerate(meta):
            if m["direction"] == trigger and m["z"] >= stretch:
                wanted[m["ticker"]] += 1
        quiet = defaultdict(list)
        for i, m in enumerate(meta):
            if m["z"] < 1.0:
                quiet[m["ticker"]].append(i)
        mask = np.zeros(len(meta), dtype=bool)
        for ticker, count in wanted.items():
            pool = quiet.get(ticker, [])
            if not pool:
                continue
            rng = random.Random(null_seed + ticker_seed(ticker))
            for i in rng.sample(pool, min(count, len(pool))):
                mask[i] = True
    raw = resolve(meta, H, L, mask, side, stop_mult, target_r, hold, args)

    # one position per name at a time, and an exit date for the daily marking
    by_ticker = defaultdict(list)
    for t in raw:
        by_ticker[t["ticker"]].append(t)
    trades = []
    for ticker, rows in by_ticker.items():
        rows.sort(key=lambda t: t["entry"])
        open_until = ""
        for t in rows:
            if t["entry"] < open_until:
                continue
            m = meta[t["meta"]]
            step, day = t["exit_step"], m["end_days"][-1]
            for slot, boundary in enumerate(m["ends"]):
                if step <= boundary:
                    day = m["end_days"][slot]
                    break
            t["exit"] = day
            open_until = t["exit"]
            trades.append(t)
    trades.sort(key=lambda t: t["entry"])
    return trades


def describe(trades):
    outcomes = [t["r"] for t in trades]
    reasons = defaultdict(int)
    for t in trades:
        reasons[t["reason"]] += 1
    n = len(trades)
    return {"trades": n, "mean_r": statistics.fmean(outcomes),
            "target_rate": reasons["target"] / n, "stop_rate": reasons["stop"] / n,
            "time_rate": reasons["time"] / n,
            "median_stop_pct": statistics.median(t["stop_pct"] for t in trades)}


def window(trades, start=None, end=None):
    return [t for t in trades
            if (start is None or t["entry"] >= start)
            and (end is None or t["entry"] < end)]


def sizing(taken, args):
    rows = []
    for fraction in RISK_RUNGS:
        per_day, levers, capped = defaultdict(float), [], 0
        for t in taken:
            wanted = fraction / t["stop_pct"]
            lever = min(wanted, args.max_leverage)
            capped += wanted > args.max_leverage
            levers.append(lever)
            per_day[t["exit"]] += t["r"] * lever * t["stop_pct"]
        days = sorted(per_day)
        nav, peak, worst = 1000.0, 1000.0, 0.0
        for day in days:
            nav = max(0.0, nav + per_day[day] * nav)
            peak = max(peak, nav)
            worst = min(worst, nav / peak - 1.0)
        years = (date.fromisoformat(days[-1])
                 - date.fromisoformat(days[0])).days / 365.25
        rows.append({"risk": fraction, "max_drawdown": worst,
                     "cagr": (nav / 1000.0) ** (1 / years) - 1 if nav > 0 else -1.0,
                     "median_leverage": statistics.median(levers),
                     "max_leverage": max(levers),
                     "share_capped": capped / len(levers)})
    return rows


def main(argv=None) -> int:
    args = parse_args(argv)
    book = load(args)
    panel = deviations(book, args.vol_window)
    meta, H, L = build_paths(book, panel, args)

    print(f"{args.universe} universe: {len(book)} names, {len(meta):,d} triggers, "
          f"{meta[0]['day']} to {meta[-1]['day'] if meta else '?'}")
    print(f"in sample to {args.split}, out of sample after; realised vol, "
          f"no option chain\n")
    report = {"universe": args.universe, "names": len(book),
              "triggers": len(meta), "arms": {}}

    for action, trigger, side_name in ARMS:
        label = f"{action} {'up' if trigger > 0 else 'down'} ({side_name})"
        best, best_score = None, -99.0
        for stretch in STRETCHES:
            for stop_mult in STOPS:
                for target_r in TARGETS:
                    for hold in HOLDS:
                        trades = arm_trades(meta, H, L, args, action, trigger,
                                            stretch, stop_mult, target_r, hold)
                        early = window(trades, end=args.split)
                        if len(early) < 150:
                            continue
                        result = shared.assess(early, args)
                        if result and result["sharpe"] > best_score:
                            best_score, best = result["sharpe"], (
                                stretch, stop_mult, target_r, hold)
        if best is None:
            print(f"{label:26s}  nothing cleared the in-sample minimum")
            continue
        stretch, stop_mult, target_r, hold = best
        trades = arm_trades(meta, H, L, args, action, trigger, *best)
        outside = window(trades, start=args.split)
        entry = {"chosen": {"stretch": stretch, "stop": stop_mult,
                            "target": target_r, "hold": hold},
                 "in_sample_sharpe": best_score}
        print(f"### {label}")
        print(f"    chosen in sample: {stretch:g} deviations, stop {stop_mult:g}, "
              f"target {target_r:g}R, hold {hold}  (IS Sharpe {best_score:.2f})")
        if len(outside) < 120:
            print(f"    out of sample: only {len(outside)} trades\n")
            report["arms"][label] = entry
            continue
        stats = describe(outside)
        result = shared.assess(outside, args)
        entry.update({**stats, **(result or {})})
        print(f"    out of sample: {stats['trades']:,d} trades, "
              f"target {stats['target_rate']:.1%}, stop {stats['stop_rate']:.1%}, "
              f"mean {stats['mean_r']:+.3f}R, Sharpe "
              f"{result['sharpe']:.2f} [{result['sharpe_p05']:.2f}-"
              f"{result['sharpe_p95']:.2f}]")

        nulls = []
        for draw in range(args.null_draws):
            drawn = arm_trades(meta, H, L, args, action, trigger, *best,
                               null_seed=52_000 + 137 * draw)
            pooled = window(drawn, start=args.split)
            if len(pooled) < 120:
                continue
            outcome = shared.assess(pooled, args)
            if outcome:
                nulls.append(outcome["sharpe"])
        if nulls:
            nulls.sort()
            above = sum(1 for x in nulls if x >= result["sharpe"])
            entry["null"] = {"median": statistics.median(nulls), "low": nulls[0],
                             "high": nulls[-1], "p": above / len(nulls),
                             "draws": nulls}
            verdict = ("clears its null" if above / len(nulls) <= 0.05
                       else "inside its null")
            print(f"    drift null:    {statistics.median(nulls):.2f} "
                  f"[{nulls[0]:.2f}-{nulls[-1]:.2f}], "
                  f"p = {above / len(nulls):.2f}  -> {verdict}")
        taken = shared.cap(outside, args.max_positions, random.Random(0))
        entry["sizing"] = sizing(taken, args)
        row = entry["sizing"][1]
        print(f"    at 1% risk:    {row['cagr']:+.1%} CAGR, "
              f"{row['max_drawdown']:.1%} drawdown, "
              f"{row['median_leverage']:.2f}x median leverage, "
              f"{row['share_capped']:.0%} capped\n")
        report["arms"][label] = entry

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
