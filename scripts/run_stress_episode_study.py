"""How the book behaves when the market breaks.

Everything measured so far is an average over 11 years, and an average is the
wrong statistic for the question an investor actually asks: what happens in the
bad weeks. This sample contains the episodes that matter -- the February 2018
volatility spike, the fourth quarter of 2018, the March 2020 crash and the 2022
bear market -- so they can be measured rather than assumed.

The option study could not do this: one year of chains, no crisis in it. Price
history can, and does.

Episodes are found rather than named: every peak-to-trough decline in SPY of at
least the threshold, dated from the high to the low. That keeps the definition
free of hindsight about which selloffs turned out to be famous.

Three things are reported for each, and the third is the one that decides
whether this is a diversifier or a leveraged long.

*What the strategy did* against what SPY did, over the identical window.

*What it did in the recovery*, because a system that sidesteps the fall and also
misses the rebound has protected nothing over a round trip.

*How invested it was.* Six positions times two units is not a fixed exposure --
notional is `equity x risk x price/N`, so a volatility spike mechanically shrinks
position size. Some of any crisis protection is that arithmetic rather than a
decision, and it is worth separating.

A VIX-quintile table follows, because episodes are few and a conditional split
across all 2,900 sessions has more to say about the ordinary weeks that make up
most of a decade.
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
    parser.add_argument("--macro", type=Path,
                        default=Path("data/data/macro/macro.db"))
    parser.add_argument("--minutes", type=int, default=240)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--cost-bp", type=float, default=10.0)
    parser.add_argument("--max-units", type=int, default=2)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--episode-threshold", type=float, default=0.10)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/stress_episodes.json"))
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
    book, spy = {}, {}
    for ticker in tickers:
        raw = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='5min' AND ts>=? ORDER BY ts", (ticker, args.start)).fetchall()
        if len(raw) < 4000:
            continue
        bars = resample_regular_session([Bar(*r) for r in raw], minutes=args.minutes)
        if len(bars) >= 400:
            book[ticker] = bars
        if ticker == "SPY":
            for ts, _o, _h, _l, close, _v in raw:
                spy[ts[:10]] = float(close)
    connection.close()
    return book, spy


def vix_series(path):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT obs_date, value FROM observations WHERE series_id='VIXCLS' "
        "AND value IS NOT NULL ORDER BY obs_date").fetchall()
    connection.close()
    return {d: float(v) for d, v in rows}


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


def streams(taken, closes_by_ticker):
    r_by_day = defaultdict(float)
    basis_by_day = defaultdict(float)
    open_by_day = defaultdict(int)
    for trade in taken:
        closes = closes_by_ticker[trade["ticker"]]
        entry_day, exit_day = trade["entry"][:10], trade["exit"][:10]
        previous = 0.0
        for day in (d for d in closes if entry_day <= d <= exit_day):
            live = [u for u in trade["units"] if u.timestamp[:10] <= day]
            if not live:
                continue
            if day < exit_day:
                open_r = sum(trade["dir"] * (closes[day] - u.price) / u.n
                             for u in live)
                r_by_day[day] += open_r - previous
                previous = open_r
            basis_by_day[day] += sum(closes[day] / u.n for u in live)
            open_by_day[day] += 1
        r_by_day[exit_day] += trade["r"] - previous
    return r_by_day, basis_by_day, open_by_day


def solve_risk(series, target, lo=1e-6, hi=0.40):
    def walk(values, risk):
        nav = peak = 1000.0
        worst = 0.0
        for value in values:
            nav = max(0.0, nav + value * risk * nav)
            peak = max(peak, nav)
            worst = min(worst, nav / peak - 1.0)
        return worst

    def dd(risk):
        return statistics.median(abs(walk(v, risk)) for v in series)
    if dd(hi) < target:
        return hi
    for _ in range(28):
        mid = math.sqrt(lo * hi)
        if dd(mid) < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def find_episodes(spy, threshold):
    """Every peak-to-trough decline of at least ``threshold``, plus its recovery."""
    days = sorted(spy)
    episodes = []
    peak_day, peak = days[0], spy[days[0]]
    trough_day, trough = peak_day, peak
    for day in days:
        price = spy[day]
        if price > peak:
            if trough / peak - 1.0 <= -threshold:
                episodes.append({"peak": peak_day, "trough": trough_day,
                                 "recovered": day,
                                 "spy_decline": trough / peak - 1.0})
            peak_day, peak = day, price
            trough_day, trough = day, price
        elif price < trough:
            trough_day, trough = day, price
    if trough / peak - 1.0 <= -threshold:
        episodes.append({"peak": peak_day, "trough": trough_day,
                         "recovered": None, "spy_decline": trough / peak - 1.0})
    return episodes


def window_return(ret_by_day, days, start, end):
    total = 1.0
    for day in days:
        if start <= day <= end:
            total *= 1.0 + ret_by_day.get(day, 0.0)
    return total - 1.0


def main(argv=None) -> int:
    args = parse_args(argv)
    book, spy = load(args)
    closes_by_ticker = {t: {b.timestamp[:10]: b.close for b in bars}
                        for t, bars in book.items()}
    config = TurtleConfig(**BASE, directions=(1,), max_units=args.max_units,
                          round_trip_cost=args.cost_bp / 10_000)
    pooled = []
    for ticker, bars in book.items():
        trades, _ = run_turtle(bars, config=config)
        pooled.extend({"ticker": ticker, "entry": t.entry_timestamp,
                       "exit": t.exit_timestamp, "r": t.net_r, "dir": t.direction,
                       "units": t.unit_entries} for t in trades)
    caps = [cap(pooled, args.max_positions, random.Random(s))
            for s in range(args.trials)]
    triples = [streams(t, closes_by_ticker) for t in caps]
    calendar = sorted({d for c in closes_by_ticker.values() for d in c})
    series = [[r.get(d, 0.0) for d in calendar] for r, _, _ in triples]
    risk = solve_risk(series, args.target_dd)
    returns = [{d: r.get(d, 0.0) * risk for d in calendar} for r, _, _ in triples]
    exposure = [{d: b.get(d, 0.0) * risk for d in calendar} for _, b, _ in triples]
    positions = [o for _, _, o in triples]
    print(f"{len(book)} instruments, {args.minutes}-minute bars, "
          f"{args.max_units} units, {args.cost_bp:g}bp, risk {risk*10_000:.1f}bp/R\n")

    spy_days = sorted(spy)
    spy_ret = {spy_days[i]: spy[spy_days[i]] / spy[spy_days[i - 1]] - 1
               for i in range(1, len(spy_days))}
    episodes = find_episodes(spy, args.episode_threshold)
    print(f"=== every SPY decline of {args.episode_threshold:.0%} or more ===")
    print(f"  {'peak':>10s} {'trough':>10s} {'days':>5s} {'SPY':>8s} "
          f"{'strategy':>9s} {'positions':>10s} {'exposure':>9s} | recovery: "
          f"{'SPY':>7s} {'strat':>7s}")
    report = {"episodes": [], "risk_per_r": risk}
    for ep in episodes:
        length = (date.fromisoformat(ep["trough"])
                  - date.fromisoformat(ep["peak"])).days
        strat = statistics.median(
            window_return(r, calendar, ep["peak"], ep["trough"]) for r in returns)
        held = statistics.median(statistics.fmean(
            [p.get(d, 0) for d in calendar if ep["peak"] <= d <= ep["trough"]] or [0])
            for p in positions)
        expo = statistics.median(statistics.fmean(
            [e.get(d, 0.0) for d in calendar if ep["peak"] <= d <= ep["trough"]] or [0])
            for e in exposure)
        if ep["recovered"]:
            spy_rec = spy[ep["recovered"]] / spy[ep["trough"]] - 1.0
            strat_rec = statistics.median(
                window_return(r, calendar, ep["trough"], ep["recovered"])
                for r in returns)
        else:
            spy_rec = strat_rec = float("nan")
        report["episodes"].append({**ep, "days": length, "strategy": strat,
                                   "positions": held, "exposure": expo,
                                   "spy_recovery": spy_rec,
                                   "strategy_recovery": strat_rec})
        print(f"  {ep['peak']:>10s} {ep['trough']:>10s} {length:>5d} "
              f"{ep['spy_decline']:>8.1%} {strat:>9.1%} {held:>10.1f} "
              f"{expo:>9.0%} | {spy_rec:>7.1%} {strat_rec:>7.1%}")

    vix = vix_series(args.macro)
    paired = [(vix[d], d) for d in calendar if d in vix]
    paired.sort()
    size = len(paired) // 5
    print(f"\n=== conditional on VIX quintile ({len(paired):,} sessions) ===")
    print(f"  {'quintile':>9s} {'VIX range':>14s} {'strategy p.a.':>14s} "
          f"{'SPY p.a.':>10s} {'positions':>10s} {'exposure':>9s}")
    quintiles = []
    for q in range(5):
        block = paired[q * size:(q + 1) * size if q < 4 else len(paired)]
        days = {d for _, d in block}
        strat = statistics.median(
            statistics.fmean([r.get(d, 0.0) for d in days]) for r in returns) * 252
        base = statistics.fmean([spy_ret.get(d, 0.0) for d in days]) * 252
        held = statistics.median(
            statistics.fmean([p.get(d, 0) for d in days]) for p in positions)
        expo = statistics.median(
            statistics.fmean([e.get(d, 0.0) for d in days]) for e in exposure)
        quintiles.append({"quintile": q + 1, "vix_low": block[0][0],
                          "vix_high": block[-1][0], "strategy_annual": strat,
                          "spy_annual": base, "positions": held, "exposure": expo})
        print(f"  {q + 1:>9d} {('%.1f-%.1f' % (block[0][0], block[-1][0])):>14s} "
              f"{strat:>13.1%} {base:>10.1%} {held:>10.1f} {expo:>9.0%}")
    report["vix_quintiles"] = quintiles

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
