"""The volatility-surprise book, specified, sized, and split out of sample.

This assembles the one signal that cleared its drift null into a strategy, and
applies every constraint the programme has measured rather than assumed.

*The signal.*  A session closing beyond a multiple of the move options priced,
followed rather than faded.  The event study located the effect entirely beyond
three implied moves -- between two and three the excess over drift is +0.020,
which is nothing -- and found it in both directions, larger on the down side.
Only the long side is carried here, because the short arm beat its own null and
still landed at zero once drift and cost were paid.

*The conditions.*  Only what replicated.  Cross-sectional relative strength is
the single condition that survived every study run against it, at +0.087,
+0.064, +0.060 and +0.043 with the same sign in all five sub-periods each time.
Market extension and entry chase are carried with it.  Nothing from the option
chain beyond the implied move itself: it forecasts magnitude and the trade
already uses magnitude for its threshold.

*The execution.*  Priced per leg by the order that fills it, because the bracket
study measured the difference between a taker entry and a maker entry at
roughly a full point of Sharpe.  A follow entry at the next open is a taker and
cannot be otherwise, so a patient variant is tested alongside it: rest a limit
slightly below the open and accept that some trades never fill.

*The sizing.*  Fixed fractional, not fixed leverage.  Position risk is size
times stop distance and the stop here spans a factor of twenty-five across
names, so a single notional multiple floats the risk into ruin -- measured at
-99.7% on this very book.  Size is therefore a risk budget divided by the stop,
capped at the venue maximum.  What leverage that produces is an output, and it
is reported per trade rather than chosen.

*The split.*  Every threshold above was picked after looking at the data.  The
first half selects the parameters, the second half is never touched until the
configuration is frozen, and the controls run on the second half only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_vol_stretch_zones as data  # noqa: E402
import run_vol_surprise_study as surprise  # noqa: E402

STRETCHES = (2.5, 3.0, 3.5)
STOPS = (1.5, 2.0, 3.0)
TARGETS = (1.5, 2.0, 3.0)
HOLDS = (3, 5, 10)
RISK_RUNGS = (0.005, 0.01, 0.015, 0.02, 0.03)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--options", type=Path,
                        default=Path("data/options/alphavantage.db"))
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--zone-minutes", type=int, default=30)
    parser.add_argument("--split", default="2022-01-01",
                        help="first out-of-sample session")
    parser.add_argument("--vol-source", choices=("implied", "realized"),
                        default="implied")
    parser.add_argument("--maker-bp", type=float, default=2.5)
    parser.add_argument("--taker-bp", type=float, default=5.0)
    parser.add_argument("--max-leverage", type=float, default=20.0)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--null-draws", type=int, default=25,
                        help="draws of the drift null; the claim rests on this band")
    parser.add_argument("--patient-offset", type=float, default=0.5,
                        help="limit this many implied moves below the open")
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/surprise_strategy.json"))
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# conditions


def condition_panel(book, signal_minutes=None):
    """Relative strength and market extension, both read before the entry.

    Daily bars are the natural frame here: the trigger is a session close, so a
    condition measured on anything finer would be answering a different question
    from the one the signal asks.
    """
    closes, ranks = {}, {}
    for ticker, (_, _, daily) in book.items():
        closes[ticker] = {b.timestamp[:10]: b.close for b in daily}
    pool = defaultdict(list)
    for ticker, (_, _, daily) in book.items():
        for index in range(61, len(daily)):
            base = daily[index - 61].close
            if base > 0:
                pool[daily[index].timestamp[:10]].append(
                    ((daily[index - 1].close - base) / base, ticker))
    for day, rows in pool.items():
        rows.sort()
        size = len(rows)
        for position, (_, ticker) in enumerate(rows):
            ranks[(ticker, day)] = position / (size - 1) if size > 1 else 0.5

    market = {}
    proxy = book.get("SPY")
    if proxy:
        daily = proxy[2]
        for index in range(61, len(daily)):
            window = [daily[i].close for i in range(index - 21, index)]
            spread = statistics.pstdev(window) or 1.0
            market[daily[index].timestamp[:10]] = (
                daily[index - 1].close - statistics.fmean(window)) / spread
    return ranks, market


# ---------------------------------------------------------------------------
# the book


def book_trades(book, rows, args, stretch, stop_mult, target_r, hold, patient,
                ranks, market, null_seed=None):
    """Long follows of an up surprise, priced by the order type that fills."""
    spec = argparse.Namespace(
        stop_mult=stop_mult, target_r=target_r, hold_sessions=hold,
        wait_sessions=1, maker_bp=args.maker_bp, taker_bp=args.taker_bp)

    by_ticker = defaultdict(list)
    if null_seed is None:
        for row in rows:
            if row["direction"] > 0 and row["z_implied"] >= stretch:
                by_ticker[row["ticker"]].append(row)
    else:
        wanted = defaultdict(int)
        for row in rows:
            if row["direction"] > 0 and row["z_implied"] >= stretch:
                wanted[row["ticker"]] += 1
        quiet = defaultdict(list)
        for row in rows:
            if row["z_implied"] < 1.0:
                quiet[row["ticker"]].append(row)
        for ticker, count in wanted.items():
            pool = quiet.get(ticker, [])
            if not pool:
                continue
            rng = random.Random(null_seed + (hash(ticker) % 10_000))
            by_ticker[ticker] = sorted(rng.sample(pool, min(count, len(pool))),
                                       key=lambda r: r["day"])

    trades = []
    for ticker, events_here in by_ticker.items():
        five = book[ticker][0]
        spans = surprise.session_index(five)
        day_at = {day: i for i, (day, _, _) in enumerate(spans)}
        open_until = ""
        for row in events_here:
            if row["day"] < open_until:
                continue
            if patient:
                position = day_at.get(row["day"])
                if position is None or position + 1 >= len(spans):
                    continue
                opening = five[spans[position + 1][1]].open
                outcome = surprise.fade_trade(
                    five, spans, day_at, row["day"], 1,
                    opening + args.patient_offset * row["implied"],
                    row["implied"], spec)
            else:
                outcome = surprise.follow_trade(
                    five, spans, day_at, row["day"], 1, row["implied"], spec)
            if outcome is None:
                continue
            fill, price, reason, stamp, risk = outcome
            open_until = stamp[:10]
            entry_leg = args.maker_bp if patient else args.taker_bp
            exit_leg = args.maker_bp if reason == "target" else args.taker_bp
            trades.append({
                "ticker": ticker, "entry": row["day"], "exit": stamp[:10],
                "r": (price - fill) / risk
                     - (entry_leg + exit_leg) / 10_000 * fill / risk,
                "reason": reason, "stop_pct": risk / fill,
                "rs_rank": ranks.get((ticker, row["day"]), math.nan),
                "mkt_ext": market.get(row["day"], math.nan),
            })
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


# ---------------------------------------------------------------------------
# sizing


def sized_path(taken, risk_fraction, args):
    """Fixed-fractional sizing, capped at the venue's per-position maximum.

    The cap binds only where the stop is tighter than ``risk / max_leverage``.
    Reporting how often that happens is the honest answer to "can I use 20x":
    the strategy uses whatever its stops allow, and no more.
    """
    per_day = defaultdict(float)
    levers, capped = [], 0
    for trade in taken:
        wanted = risk_fraction / trade["stop_pct"]
        lever = min(wanted, args.max_leverage)
        if wanted > args.max_leverage:
            capped += 1
        levers.append(lever)
        per_day[trade["exit"]] += trade["r"] * lever * trade["stop_pct"]
    days = sorted(per_day)
    if not days:
        return None
    nav = peak = 1000.0
    worst = 0.0
    for day in days:
        nav = max(0.0, nav + per_day[day] * nav)
        peak = max(peak, nav)
        worst = min(worst, nav / peak - 1.0)
        if nav <= 0.0:
            return {"ruined": True, "max_drawdown": -1.0, "cagr": -1.0,
                    "median_leverage": statistics.median(levers),
                    "max_leverage_used": max(levers),
                    "share_capped": capped / len(levers)}
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25
    stream = [per_day[d] for d in days]
    deviation = statistics.pstdev(stream)
    return {"ruined": False, "max_drawdown": worst,
            "cagr": (nav / 1000.0) ** (1 / years) - 1,
            "sharpe": (statistics.fmean(stream) / deviation * math.sqrt(252)
                       if deviation > 0 else float("nan")),
            "median_leverage": statistics.median(levers),
            "p95_leverage": sorted(levers)[int(0.95 * len(levers))],
            "max_leverage_used": max(levers),
            "share_capped": capped / len(levers)}


def main(argv=None) -> int:
    args = parse_args(argv)
    book = data.load(args)
    implied = data.implied_moves(args.options, sorted(book), args.start)
    realized = data.realized_moves(book)
    rows = surprise.events(book, implied, realized,
                           argparse.Namespace(**vars(args)))
    if args.vol_source == "realized":
        for row in rows:
            row["z_implied"] = row["z_realized"]
    ranks, market = condition_panel(book)
    print(f"{len(book)} names, {len(rows):,d} sessions with both vol measures, "
          f"{args.vol_source} threshold")
    print(f"in sample {args.start} to {args.split}, out of sample from "
          f"{args.split}\n")
    report = {"in_sample": {}, "out_of_sample": {}, "sizing": {}}

    # ---- parameter selection, first half only ---------------------------
    print("########## selection on the first half ##########")
    print(f"  {'stretch':>8s} {'stop':>6s} {'target':>7s} {'hold':>5s} "
          f"{'trades':>7s} {'meanR':>8s} {'Sharpe':>7s}")
    best, best_score = None, -99.0
    cache = {}
    for stretch in STRETCHES:
        for stop_mult in STOPS:
            for target_r in TARGETS:
                for hold in HOLDS:
                    trades = book_trades(book, rows, args, stretch, stop_mult,
                                         target_r, hold, False, ranks, market)
                    inside = window(trades, end=args.split)
                    if len(inside) < 120:
                        continue
                    result = data.assess(inside, args)
                    if not result:
                        continue
                    cache[(stretch, stop_mult, target_r, hold)] = trades
                    if result["sharpe"] > best_score:
                        best_score = result["sharpe"]
                        best = (stretch, stop_mult, target_r, hold)
    if best is None:
        print("  no configuration cleared the minimum trade count")
        return 1
    for key in sorted(cache, key=lambda k: -data.assess(
            window(cache[k], end=args.split), args)["sharpe"])[:6]:
        inside = window(cache[key], end=args.split)
        stats = describe(inside)
        result = data.assess(inside, args)
        marker = " <- chosen" if key == best else ""
        print(f"  {key[0]:>8.1f} {key[1]:>6.1f} {key[2]:>7.1f} {key[3]:>5d} "
              f"{stats['trades']:>7,d} {stats['mean_r']:>+8.3f} "
              f"{result['sharpe']:>7.2f}{marker}")
    report["in_sample"]["chosen"] = {
        "stretch": best[0], "stop_mult": best[1], "target_r": best[2],
        "hold_sessions": best[3], "sharpe": best_score}

    stretch, stop_mult, target_r, hold = best
    print(f"\n  frozen: {stretch:g} implied moves, stop {stop_mult:g} moves, "
          f"target {target_r:g}R, held {hold} sessions")

    # ---- out of sample, with controls -----------------------------------
    trades = cache[best]
    outside = window(trades, start=args.split)
    print(f"\n########## out of sample, from {args.split} ##########")
    print(f"  {'arm':34s} {'trades':>7s} {'target':>7s} {'stop':>7s} "
          f"{'meanR':>8s} {'Sharpe':>7s} {'[5-95%]':>15s}")

    def show(label, pooled):
        if len(pooled) < 60:
            print(f"  {label:34s} {len(pooled):>7,d}   too few")
            return None
        stats, result = describe(pooled), data.assess(pooled, args)
        if not result:
            print(f"  {label:34s} {len(pooled):>7,d}   not assessable")
            return None
        report["out_of_sample"][label] = {**stats, **result}
        print(f"  {label:34s} {stats['trades']:>7,d} {stats['target_rate']:>7.1%} "
              f"{stats['stop_rate']:>7.1%} {stats['mean_r']:>+8.3f} "
              f"{result['sharpe']:>7.2f} "
              f"{('[%.2f-%.2f]' % (result['sharpe_p05'], result['sharpe_p95'])):>15s}")
        return result

    show("the book (taker entry)", outside)

    patient = book_trades(book, rows, args, stretch, stop_mult, target_r, hold,
                          True, ranks, market)
    show("patient entry (maker limit)", window(patient, start=args.split))

    strong = [t for t in outside
              if not math.isnan(t["rs_rank"]) and t["rs_rank"] >= 0.5]
    show("+ relative strength filter", strong)

    nulls = []
    for draw in range(args.null_draws):
        drawn = book_trades(book, rows, args, stretch, stop_mult, target_r, hold,
                            False, ranks, market, null_seed=9100 + 211 * draw)
        pooled = window(drawn, start=args.split)
        if len(pooled) < 60:
            continue
        result = data.assess(pooled, args)
        if result:
            nulls.append(result["sharpe"])
    if nulls:
        nulls.sort()
        print(f"  {'ordinary sessions (drift null)':34s} {'':>7s} {'':>7s} "
              f"{'':>7s} {'':>8s} {statistics.median(nulls):>7.2f} "
              f"{('[%.2f-%.2f]' % (nulls[0], nulls[-1])):>15s}")
        report["out_of_sample"]["drift_null"] = {
            "sharpe": statistics.median(nulls), "low": nulls[0], "high": nulls[-1]}

    # ---- sizing ---------------------------------------------------------
    taken = data.cap(outside, args.max_positions, random.Random(0))
    print(f"\n########## sizing the out-of-sample book "
          f"({len(taken):,d} trades taken) ##########")
    print(f"  Fixed fractional, capped at {args.max_leverage:g}x per position.")
    print(f"  {'risk':>6s} {'max DD':>9s} {'CAGR':>9s} {'Sharpe':>7s} "
          f"{'median lev':>11s} {'95th lev':>9s} {'max lev':>8s} {'capped':>8s}")
    for fraction in RISK_RUNGS:
        sized = sized_path(taken, fraction, args)
        if not sized:
            continue
        report["sizing"][f"{fraction:.3f}"] = sized
        if sized["ruined"]:
            print(f"  {fraction:>5.1%} {'RUINED':>9s}")
            continue
        print(f"  {fraction:>5.1%} {sized['max_drawdown']:>9.1%} "
              f"{sized['cagr']:>9.1%} {sized['sharpe']:>7.2f} "
              f"{sized['median_leverage']:>10.2f}x {sized['p95_leverage']:>8.2f}x "
              f"{sized['max_leverage_used']:>7.2f}x {sized['share_capped']:>8.1%}")

    stops = sorted(t["stop_pct"] for t in taken)
    needed = args.max_leverage * statistics.median(stops)
    print(f"\n  median stop is {statistics.median(stops):.2%} of price, so reaching "
          f"{args.max_leverage:g}x on a median\n  trade would mean risking "
          f"{needed:.1%} of the account on it.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
