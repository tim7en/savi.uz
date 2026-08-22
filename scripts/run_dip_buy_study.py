"""Buying dips below VWAP while the longer trend still holds.

A different family from everything else here. The breakout book buys strength
with a stop order; this buys weakness with a resting limit, and that distinction
is worth more than it first appears. A stop order is a taker and pays 10bp on
this venue. A limit resting below the market is a maker and pays 5bp. Given that
the interval study found cost to be the binding constraint everywhere -- the
30-minute book scores 0.19 at taker cost against 0.95 at four hours -- an entry
that halves the cost deserves its own test rather than an inherited verdict.

The rule: with price above its long moving average, rest a limit at the trailing
VWAP. If the bar trades down to it, the fill is there, not at the close.
"Accelerating" the dip is tested as size scaling with depth below VWAP in units
of N, capped, so a deeper dip buys more.

Three controls, and the second is the one that decides it.

*The breakout book at the same interval and cost*, so the comparison is against
the incumbent rather than against nothing.

*Random entries in the same regime.* Equities drift upward, so any long taken
above a rising average makes money, and a dip rule inherits that drift for free.
Random bars in the identical regime, with identical exits and the same trade
count, separate the signal from the drift. When this control was run against the
breakout book, random entries captured 63-75% of its result -- so the bar is
high and known.

*The regime removed*, isolating what the moving-average condition contributes
against dip-buying with no trend filter at all, which is the version that
catches falling knives.

Exits stay the banked machinery -- 2N stop, half-N pyramid to four units, 3N
chandelier -- so the entry is the only thing that changes.
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
from savi_uz.turtle import TurtleConfig, run_turtle, wilder_atr  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

FIXED = dict(entry_window=55, exit_window=20, atr_window=20,
             skip_after_winner=False, use_channel_exit=False, chandelier_atr=3.0)

LEVERED_MARKERS = ("2X", "3X", "1.5X", "BULL", "BEAR", "LEVERAGED", "INVERSE",
                   "ULTRA", "DAILY ", "SHORT ")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/intraday/bars_av.db"))
    parser.add_argument("--minutes", type=int, default=240,
                        help="bar size; 240 is where the breakout book scored best")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--vwap-window", type=int, default=13,
                        help="bars in the trailing VWAP")
    parser.add_argument("--ma-window", type=int, default=200)
    parser.add_argument("--costs-bp", type=float, nargs="+", default=(5.0, 7.5, 10.0))
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--null-draws", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/dip_buy.json"))
    return parser.parse_args(argv)


def load(args):
    connection = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT ticker, name FROM symbols WHERE name IS NOT NULL").fetchall()
        drop = {t for t, n in rows if any(m in n.upper() for m in LEVERED_MARKERS)}
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
            "frequency='5min' AND ts>=? ORDER BY ts",
            (ticker, args.start)).fetchall()
        if len(raw) < 4000:
            continue
        bars = resample_regular_session([Bar(*r) for r in raw], minutes=args.minutes)
        if len(bars) >= 400:
            book[ticker] = bars
    connection.close()
    print(f"excluded {len(drop)} levered or inverse wrappers")
    return book


def levels(bars, args):
    """Trailing VWAP and long mean, both ending at the bar *before* each index."""
    typical = [(b.high + b.low + b.close) / 3 for b in bars]
    volumes = [max(float(b.volume or 0.0), 0.0) for b in bars]
    closes = [b.close for b in bars]
    vwap, mean = [math.nan] * len(bars), [math.nan] * len(bars)
    pv = vv = 0.0
    running = 0.0
    for index in range(len(bars)):
        if index > 0:
            pv += typical[index - 1] * volumes[index - 1]
            vv += volumes[index - 1]
            running += closes[index - 1]
            if index > args.vwap_window:
                j = index - 1 - args.vwap_window
                pv -= typical[j] * volumes[j]
                vv -= volumes[j]
            if index > args.ma_window:
                running -= closes[index - 1 - args.ma_window]
        if index > args.vwap_window and vv > 0:
            vwap[index] = pv / vv
        if index > args.ma_window:
            mean[index] = running / args.ma_window
    return vwap, mean


def dip_entries(bars, args, use_regime=True):
    """A limit resting at trailing VWAP, filled when the bar trades down to it."""
    vwap, mean = levels(bars, args)
    atr = wilder_atr(bars, FIXED["atr_window"])
    entries, prices, depth = {}, {}, {}
    start = max(args.vwap_window, args.ma_window) + 2
    for index in range(start, len(bars)):
        level, trend, n = vwap[index], mean[index], atr[index - 1]
        if math.isnan(level) or math.isnan(trend):
            continue
        if not n or math.isnan(n) or n <= 0:
            continue
        bar = bars[index]
        if use_regime and bar.close <= trend:
            continue
        if bar.low > level:                      # the limit never traded
            continue
        fill = min(level, bar.open)              # a gap below fills at the open
        entries[index] = 1
        prices[index] = fill
        depth[index] = (level - fill) / n
    return entries, prices, depth


def regime_bars(bars, args):
    """Indexes where the trend condition holds, for the matched random null."""
    _, mean = levels(bars, args)
    out = []
    start = max(args.vwap_window, args.ma_window) + 2
    for index in range(start, len(bars)):
        if not math.isnan(mean[index]) and bars[index].close > mean[index]:
            out.append(index)
    return out


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


def marked_map(taken, closes_by_ticker, weight=None):
    by_day = defaultdict(float)
    for trade in taken:
        scale = weight(trade) if weight else 1.0
        closes = closes_by_ticker[trade["ticker"]]
        entry_day, exit_day = trade["entry"][:10], trade["exit"][:10]
        previous = 0.0
        for day in (d for d in closes if entry_day <= d < exit_day):
            live = [u for u in trade["units"] if u.timestamp[:10] <= day]
            if not live:
                continue
            open_r = sum(trade["dir"] * (closes[day] - u.price) / u.n for u in live)
            by_day[day] += (open_r - previous) * scale
            previous = open_r
        by_day[exit_day] += (trade["r"] - previous) * scale
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


def assess(pooled, closes_by_ticker, args, weight=None):
    if len(pooled) < 200:
        return None
    caps = [cap(pooled, args.max_positions, random.Random(s))
            for s in range(args.trials)]
    marks = [marked_map(t, closes_by_ticker, weight) for t in caps]
    series = [(sorted(m), [m[d] for d in sorted(m)]) for m in marks]
    risk = solve_risk(series, args.target_dd)
    cagrs, sharpes = [], []
    for days, values in series:
        nav, _ = path(values, risk)
        years = (date.fromisoformat(days[-1])
                 - date.fromisoformat(days[0])).days / 365.25
        cagrs.append((nav / 1000.0) ** (1 / years) - 1 if nav > 0 else -1.0)
        sharpes.append(sharpe([v * risk for v in values]))
    spread = sorted(sharpes)
    return {"offered": len(pooled), "sharpe": statistics.median(spread),
            "sharpe_p05": spread[int(.05 * len(spread))],
            "sharpe_p95": spread[min(int(.95 * len(spread)), len(spread) - 1)],
            "cagr": statistics.median(cagrs)}


def pool(book, config, builder):
    out = []
    for ticker, bars in book.items():
        entries, prices = builder(ticker, bars)
        if not entries:
            continue
        trades, _ = run_turtle(bars, config=config, entries=entries,
                               entry_prices=prices)
        out.extend({"ticker": ticker, "entry": t.entry_timestamp,
                    "exit": t.exit_timestamp, "r": t.net_r, "dir": t.direction,
                    "units": t.unit_entries} for t in trades)
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    book = load(args)
    closes_by_ticker = {t: {b.timestamp[:10]: b.close for b in bars}
                        for t, bars in book.items()}
    plans = {t: dip_entries(b, args, True) for t, b in book.items()}
    loose = {t: dip_entries(b, args, False) for t, b in book.items()}
    pockets = {t: regime_bars(b, args) for t, b in book.items()}
    depths = {t: plans[t][2] for t in plans}
    print(f"{len(book)} instruments, {args.minutes}-minute bars, "
          f"VWAP({args.vwap_window}) limit under MA({args.ma_window}) regime\n",
          flush=True)

    report = {}
    for bp in args.costs_bp:
        config = TurtleConfig(**FIXED, directions=(1,), round_trip_cost=bp / 10_000)
        print(f"=== {bp:g}bp round trip ===")
        print(f"  {'arm':40s} {'offered':>9s} {'Sharpe':>7s} {'[5-95%]':>15s} {'CAGR':>8s}")

        arms = {
            "breakout, stop entry (incumbent)": None,
            "dip: VWAP limit, above MA": lambda t, b: plans[t][:2],
            "dip: VWAP limit, no regime filter": lambda t, b: loose[t][:2],
        }
        results = {}
        for label, builder in arms.items():
            if builder is None:
                pooled = []
                for ticker, bars in book.items():
                    trades, _ = run_turtle(bars, config=config)
                    pooled.extend({"ticker": ticker, "entry": t.entry_timestamp,
                                   "exit": t.exit_timestamp, "r": t.net_r,
                                   "dir": t.direction, "units": t.unit_entries}
                                  for t in trades)
            else:
                pooled = pool(book, config, builder)
            result = assess(pooled, closes_by_ticker, args)
            if result is None:
                continue
            results[label] = result
            report[f"{bp:g}bp|{label}"] = {"cost_bp": bp, "arm": label, **result}
            print(f"  {label:40s} {result['offered']:>9,d} {result['sharpe']:>7.2f} "
                  f"{('[%.2f-%.2f]' % (result['sharpe_p05'], result['sharpe_p95'])):>15s} "
                  f"{result['cagr']:>8.1%}", flush=True)

        # accelerated sizing: deeper below VWAP buys more, capped at 2x
        def accelerate(trade):
            table = depths[trade["ticker"]]
            for index, value in table.items():
                pass
            return 1.0
        depth_by_key = {}
        for ticker, bars in book.items():
            for index, value in depths[ticker].items():
                depth_by_key[(ticker, bars[index].timestamp)] = value
        pooled = pool(book, config, lambda t, b: plans[t][:2])
        weight = lambda tr: 1.0 + min(  # noqa: E731
            depth_by_key.get((tr["ticker"], tr["entry"]), 0.0), 1.0)
        result = assess(pooled, closes_by_ticker, args, weight=weight)
        if result:
            results["dip: accelerated by depth"] = result
            report[f"{bp:g}bp|dip: accelerated by depth"] = {
                "cost_bp": bp, "arm": "dip: accelerated by depth", **result}
            print(f"  {'dip: accelerated by depth':40s} {result['offered']:>9,d} "
                  f"{result['sharpe']:>7.2f} "
                  f"{('[%.2f-%.2f]' % (result['sharpe_p05'], result['sharpe_p95'])):>15s} "
                  f"{result['cagr']:>8.1%}", flush=True)

        # the control that decides it: random entries in the same regime
        target = {t: len(plans[t][0]) for t in plans}
        nulls = []
        for draw in range(args.null_draws):
            rng = random.Random(9000 + draw)

            def random_builder(ticker, bars, rng=rng):
                pocket = pockets[ticker]
                want = min(target[ticker], len(pocket))
                if want == 0:
                    return {}, {}
                picks = rng.sample(pocket, want)
                return ({i: 1 for i in picks},
                        {i: bars[i].close for i in picks})

            drawn = pool(book, config, random_builder)
            outcome = assess(drawn, closes_by_ticker, args)
            if outcome:
                nulls.append(outcome["sharpe"])
        if nulls:
            nulls.sort()
            median = statistics.median(nulls)
            dip = results.get("dip: VWAP limit, above MA", {}).get("sharpe")
            beat = sum(1 for x in nulls if x >= dip) / len(nulls) if dip else float("nan")
            print(f"  {'random entries, same regime (null)':40s} {'':>9s} "
                  f"{median:>7.2f} {('[%.2f-%.2f]' % (nulls[0], nulls[-1])):>15s}")
            print(f"  -> dip beats the drift null in "
                  f"{1 - beat:.0%} of draws\n", flush=True)
            report[f"{bp:g}bp|random null"] = {
                "cost_bp": bp, "arm": "random entries, same regime",
                "sharpe": median, "low": nulls[0], "high": nulls[-1],
                "share_null_at_or_above_dip": beat}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
