"""The surprise book on 143 names and twenty-seven years, threshold and all.

The intraday version of this test had two problems and this fixes both.

*Sample.*  230 out-of-sample trades over four and a half years put the Sharpe's
own standard error near 1.0, which is why its drift null ran from -1.13 to 3.21.
No amount of care with the control fixes a sample that thin.  The 13F daily
universe is 143 names back to 1999 -- roughly an order of magnitude more
sessions, spanning the dot-com unwind, the financial crisis and 2022 rather than
one bull market with two interruptions.

*Leak.*  The 3x threshold was not chosen out of sample.  It came from an event
study computed over the whole 2017-2026 span, so the search had already seen the
period it was later tested on.  Here the threshold is swept from 1.5 to 4.0 and
chosen on the first half only, and the second half is untouched until the whole
configuration is frozen.

No option chain is used or needed.  The fade study measured a trailing realised
deviation doing the chain's job at least as well (1.68 against 1.11), the
threshold only ever needed a magnitude normaliser, and this universe has no
chain to read.  That makes the result portable rather than dependent on a vendor
snapshot.

Everything resolves on daily bars, which is the cost of the wider universe.  A
single bar that touches both the stop and the target is unknowable, so the stop
is charged -- the same convention the intraday version used, but binding far
more often here, which biases every number below downward.  Entry is the next
session's open and pays taker; a target exit rests as a limit and pays maker.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

import run_vol_stretch_zones as shared  # noqa: E402

STRETCHES = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
STOPS = (1.5, 2.0, 3.0)
TARGETS = (1.5, 2.0, 3.0)
HOLDS = (5, 10, 20)
RISK_RUNGS = (0.005, 0.01, 0.02, 0.03)
HORIZONS = (1, 5, 10, 20)


def ticker_seed(ticker):
    return zlib.crc32(ticker.encode("utf-8")) % 10_000


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path,
                        default=Path("data/13f/alphavantage_daily.db"))
    parser.add_argument("--start", default="1999-11-01")
    parser.add_argument("--split", default="2013-01-01",
                        help="first out-of-sample session")
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--min-sessions", type=int, default=750)
    parser.add_argument("--maker-bp", type=float, default=2.5)
    parser.add_argument("--taker-bp", type=float, default=5.0)
    parser.add_argument("--max-leverage", type=float, default=20.0)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--null-draws", type=int, default=50)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/surprise_daily.json"))
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------


def load(args):
    """Daily bars per name, split-adjusted where the database records splits."""
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    book = {}
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency='daily' "
            "ORDER BY ticker"):
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='daily' AND ts>=? ORDER BY ts",
            (ticker, args.start)).fetchall()
        bars = adjust_bars([Bar(r[0][:10], *r[1:]) for r in rows],
                           splits.get(ticker, []))
        if len(bars) >= args.min_sessions:
            book[ticker] = bars
    connection.close()
    return book


def deviations(book, window):
    """Trailing log-return deviation per name, readable the following session."""
    panel = {}
    for ticker, bars in book.items():
        table, returns = {}, []
        for index in range(1, len(bars)):
            previous, current = bars[index - 1].close, bars[index].close
            if previous > 0 and current > 0:
                returns.append(math.log(current / previous))
            if len(returns) > window:
                returns.pop(0)
            if len(returns) == window:
                table[bars[index].timestamp] = statistics.pstdev(returns)
        if len(table) >= 200:
            panel[ticker] = table
    return panel


def events(book, panel):
    """Sessions whose close moved at least one deviation, with forward moves."""
    out = []
    for ticker, bars in book.items():
        history = panel.get(ticker)
        if not history:
            continue
        closes = [b.close for b in bars]
        opens = [b.open for b in bars]
        for index in range(1, len(bars) - max(HORIZONS) - 1):
            deviation = history.get(bars[index - 1].timestamp)
            base = closes[index - 1]
            if not deviation or deviation <= 0 or base <= 0:
                continue
            change = (closes[index] - base) / base
            if change == 0:
                continue
            z = abs(change) / deviation
            if z < 1.0:
                pass
            row = {"ticker": ticker, "index": index,
                   "day": bars[index].timestamp,
                   "direction": 1 if change > 0 else -1, "z": z,
                   "move": deviation * base, "close": closes[index]}
            for horizon in HORIZONS:
                row[f"open_{horizon}"] = (
                    closes[index + horizon] - opens[index + 1]) / (deviation * base)
            out.append(row)
    return out


def run_trade(bars, index, side, move, stop_mult, target_r, hold, args):
    """Enter at the next open; walk daily bars until the bracket or the clock."""
    start = index + 1
    if start >= len(bars):
        return None
    fill = bars[start].open
    risk = stop_mult * move
    if fill <= 0 or risk <= 0:
        return None
    stop = fill - side * risk
    target = fill + side * target_r * risk
    last = min(start + hold - 1, len(bars) - 1)
    for step in range(start, last + 1):
        bar = bars[step]
        # Both touched inside one bar is unknowable, so the stop is charged.
        if (bar.low <= stop) if side > 0 else (bar.high >= stop):
            return fill, stop, "stop", bar.timestamp, risk
        if (bar.high >= target) if side > 0 else (bar.low <= target):
            return fill, target, "target", bar.timestamp, risk
    return fill, bars[last].close, "time", bars[last].timestamp, risk


def book_trades(book, rows, args, stretch, stop_mult, target_r, hold,
                direction=1, null_seed=None):
    index_of = {t: {b.timestamp: i for i, b in enumerate(bars)}
                for t, bars in book.items()}
    by_ticker = defaultdict(list)
    if null_seed is None:
        for row in rows:
            if row["direction"] == direction and row["z"] >= stretch:
                by_ticker[row["ticker"]].append(row)
    else:
        wanted = defaultdict(int)
        for row in rows:
            if row["direction"] == direction and row["z"] >= stretch:
                wanted[row["ticker"]] += 1
        quiet = defaultdict(list)
        for row in rows:
            if row["z"] < 1.0:
                quiet[row["ticker"]].append(row)
        for ticker, count in wanted.items():
            pool = quiet.get(ticker, [])
            if pool:
                rng = random.Random(null_seed + ticker_seed(ticker))
                by_ticker[ticker] = sorted(
                    rng.sample(pool, min(count, len(pool))),
                    key=lambda r: r["day"])

    trades = []
    for ticker, here in by_ticker.items():
        bars = book[ticker]
        open_until = ""
        for row in here:
            if row["day"] < open_until:
                continue
            index = index_of[ticker].get(row["day"])
            if index is None:
                continue
            outcome = run_trade(bars, index, direction, row["move"],
                                stop_mult, target_r, hold, args)
            if outcome is None:
                continue
            fill, price, reason, stamp, risk = outcome
            open_until = stamp
            exit_leg = args.maker_bp if reason == "target" else args.taker_bp
            trades.append({
                "ticker": ticker, "entry": row["day"], "exit": stamp,
                "r": direction * (price - fill) / risk
                     - (args.taker_bp + exit_leg) / 10_000 * fill / risk,
                "reason": reason, "stop_pct": risk / fill})
    trades.sort(key=lambda t: t["entry"])
    return trades


def window(trades, start=None, end=None):
    return [t for t in trades
            if (start is None or t["entry"] >= start)
            and (end is None or t["entry"] < end)]


def describe(trades):
    outcomes = [t["r"] for t in trades]
    reasons = defaultdict(int)
    for trade in trades:
        reasons[trade["reason"]] += 1
    total = len(trades)
    return {"trades": total, "mean_r": statistics.fmean(outcomes),
            "target_rate": reasons["target"] / total,
            "stop_rate": reasons["stop"] / total,
            "time_rate": reasons["time"] / total,
            "median_stop_pct": statistics.median(t["stop_pct"] for t in trades)}


def main(argv=None) -> int:
    args = parse_args(argv)
    book = load(args)
    panel = deviations(book, args.vol_window)
    rows = events(book, panel)
    span = (min(r["day"] for r in rows), max(r["day"] for r in rows))
    print(f"{len(book)} names, {len(rows):,d} sessions, {span[0]} to {span[1]}")
    print(f"in sample to {args.split}, out of sample after; realised vol only, "
          f"no option chain\n")
    report = {"universe": len(book), "sessions": len(rows), "span": span}

    print("########## the event study, in sample only ##########")
    print("  Raw forward move, not signed by the trigger, in deviations.")
    inside = [r for r in rows if r["day"] < args.split]
    base = [statistics.fmean(r["direction"] * r[f"open_{h}"] for r in inside)
            for h in HORIZONS]
    print(f"  {'bucket':16s} {'n':>8s} " +
          " ".join(f"{'+' + str(h) + 's':>9s}" for h in HORIZONS) + "   vs drift +10s")
    print(f"  {'every session':16s} {len(inside):>8,d} " +
          " ".join(f"{m:>+9.3f}" for m in base))
    report["event_study"] = {}
    for low, high in ((1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, 99.0)):
        for sign, name in ((1, "up"), (-1, "down")):
            chunk = [r for r in inside
                     if low <= r["z"] < high and r["direction"] == sign]
            if len(chunk) < 100:
                continue
            means = [statistics.fmean(r["direction"] * r[f"open_{h}"] for r in chunk)
                     for h in HORIZONS]
            label = f"{name} {low:g}-{high:g}" if high < 90 else f"{name} {low:g}+"
            report["event_study"][label] = {
                "n": len(chunk), "forward": dict(zip(map(str, HORIZONS), means))}
            print(f"  {label:16s} {len(chunk):>8,d} " +
                  " ".join(f"{m:>+9.3f}" for m in means) +
                  f"   {means[3] - base[3]:>+9.3f}")

    print("\n########## selection on the first half ##########")
    best, best_score, cache = None, -99.0, {}
    for stretch in STRETCHES:
        for stop_mult in STOPS:
            for target_r in TARGETS:
                for hold in HOLDS:
                    trades = book_trades(book, rows, args, stretch, stop_mult,
                                         target_r, hold)
                    early = window(trades, end=args.split)
                    if len(early) < 300:
                        continue
                    result = shared.assess(early, args)
                    if result and result["sharpe"] > best_score:
                        best_score, best = result["sharpe"], (
                            stretch, stop_mult, target_r, hold)
                    if result:
                        cache[(stretch, stop_mult, target_r, hold)] = (
                            trades, result["sharpe"])
    if best is None:
        print("  nothing cleared the minimum trade count")
        return 1
    print(f"  {'stretch':>8s} {'stop':>6s} {'target':>7s} {'hold':>5s} "
          f"{'trades':>8s} {'meanR':>8s} {'Sharpe':>7s}")
    for key in sorted(cache, key=lambda k: -cache[k][1])[:6]:
        early = window(cache[key][0], end=args.split)
        marker = " <- chosen" if key == best else ""
        print(f"  {key[0]:>8.1f} {key[1]:>6.1f} {key[2]:>7.1f} {key[3]:>5d} "
              f"{len(early):>8,d} {describe(early)['mean_r']:>+8.3f} "
              f"{cache[key][1]:>7.2f}{marker}")
    report["chosen"] = dict(zip(("stretch", "stop_mult", "target_r", "hold"), best))
    stretch, stop_mult, target_r, hold = best
    print(f"\n  frozen: {stretch:g} deviations, stop {stop_mult:g}, "
          f"target {target_r:g}R, held {hold} sessions")

    trades = cache[best][0]
    outside = window(trades, start=args.split)
    print(f"\n########## out of sample, from {args.split} ##########")
    print(f"  {'arm':34s} {'trades':>8s} {'target':>7s} {'stop':>7s} "
          f"{'meanR':>8s} {'Sharpe':>7s} {'[5-95%]':>15s}")

    def show(label, pooled):
        if len(pooled) < 100:
            print(f"  {label:34s} {len(pooled):>8,d}   too few")
            return None
        stats, result = describe(pooled), shared.assess(pooled, args)
        if not result:
            return None
        report.setdefault("out_of_sample", {})[label] = {**stats, **result}
        print(f"  {label:34s} {stats['trades']:>8,d} {stats['target_rate']:>7.1%} "
              f"{stats['stop_rate']:>7.1%} {stats['mean_r']:>+8.3f} "
              f"{result['sharpe']:>7.2f} "
              f"{('[%.2f-%.2f]' % (result['sharpe_p05'], result['sharpe_p95'])):>15s}")
        return result

    book_result = show("the book (long follow)", outside)
    short = book_trades(book, rows, args, stretch, stop_mult, target_r, hold,
                        direction=-1)
    show("the mirror (short follow)", window(short, start=args.split))

    nulls = []
    for draw in range(args.null_draws):
        drawn = book_trades(book, rows, args, stretch, stop_mult, target_r, hold,
                            null_seed=31_000 + 137 * draw)
        pooled = window(drawn, start=args.split)
        if len(pooled) < 100:
            continue
        result = shared.assess(pooled, args)
        if result:
            nulls.append(result["sharpe"])
    if nulls and book_result:
        nulls.sort()
        above = sum(1 for x in nulls if x >= book_result["sharpe"])
        print(f"  {'ordinary sessions (drift null)':34s} {'':>8s} {'':>7s} "
              f"{'':>7s} {'':>8s} {statistics.median(nulls):>7.2f} "
              f"{('[%.2f-%.2f]' % (nulls[0], nulls[-1])):>15s}")
        print(f"  -> {len(nulls) - above} of {len(nulls)} null draws fall below "
              f"the book; empirical p = {above / len(nulls):.3f}")
        report["drift_null"] = {"median": statistics.median(nulls),
                                "low": nulls[0], "high": nulls[-1],
                                "draws": nulls, "p": above / len(nulls)}

    taken = shared.cap(outside, args.max_positions, random.Random(0))
    stops = sorted(t["stop_pct"] for t in taken)
    print(f"\n########## sizing ({len(taken):,d} trades taken) ##########")
    print(f"  {'risk':>6s} {'max DD':>9s} {'CAGR':>9s} {'median lev':>11s} "
          f"{'max lev':>8s} {'capped':>8s}")
    for fraction in RISK_RUNGS:
        per_day, levers, capped = defaultdict(float), [], 0
        for trade in taken:
            wanted = fraction / trade["stop_pct"]
            lever = min(wanted, args.max_leverage)
            capped += wanted > args.max_leverage
            levers.append(lever)
            per_day[trade["exit"]] += trade["r"] * lever * trade["stop_pct"]
        days = sorted(per_day)
        nav, peak, worst = 1000.0, 1000.0, 0.0
        for day in days:
            nav = max(0.0, nav + per_day[day] * nav)
            peak = max(peak, nav)
            worst = min(worst, nav / peak - 1.0)
        years = (date.fromisoformat(days[-1])
                 - date.fromisoformat(days[0])).days / 365.25
        cagr = (nav / 1000.0) ** (1 / years) - 1 if nav > 0 else -1.0
        report.setdefault("sizing", {})[f"{fraction:.3f}"] = {
            "max_drawdown": worst, "cagr": cagr,
            "median_leverage": statistics.median(levers),
            "max_leverage": max(levers), "share_capped": capped / len(levers)}
        print(f"  {fraction:>5.1%} {worst:>9.1%} {cagr:>9.1%} "
              f"{statistics.median(levers):>10.2f}x {max(levers):>7.2f}x "
              f"{capped / len(levers):>8.1%}")
    print(f"\n  median stop {statistics.median(stops):.2%} of price; "
          f"{args.max_leverage:g}x on a median trade would risk "
          f"{args.max_leverage * statistics.median(stops):.1%} of the account")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
