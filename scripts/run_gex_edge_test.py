"""Does the SPY/QQQ gamma overlay have a real edge, or is it noise?

The overlay halves risk on new entries when index dealer gamma is low.  Three
nulls are needed to tell a signal from a coincidence, because each one can
produce a false positive on its own:

* **circular shift** -- the same regime rotated to different dates.  Preserves
  every run length and all autocorrelation, so it answers "does *this* regime
  matter, or would any equally persistent one do?"
* **independent noise** -- a coin flipped at the same rate, no persistence.
  Answers "does the clustering matter at all?"
* **always half risk** -- no regime, just less exposure.  Answers "is this
  anything more than trading smaller?"

And one control that is not a null at all but decides how to read the others:
**the sign of the baseline**.  Cutting risk mechanically improves a losing
strategy, so an overlay validated on a window where the base rules lose money
will look good no matter what it is measuring.  That is reported first.

Leakage: gamma for a date is an end-of-day snapshot of that date, verified
against the same session's close.  It is lagged one session everywhere, so the
regime is fixed before the next open and a gap cannot contaminate the decision.
A deliberately leaky same-day variant is included to price what the look-ahead
would have been worth.
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

ETFS = {"SPY", "QQQ", "IWM", "GLD", "EWJ", "EWT", "EWY", "KWEB",
        "SLV", "TBT", "TMF", "UVXY", "XLE"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--options", type=Path,
                        default=Path("data/options/marketdata.db"))
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--risk", type=float, default=0.001)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/gex_edge_test.json"))
    return parser.parse_args(argv)


def load_gamma(path: Path):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT symbol,observation_date,net_gex,underlying_price FROM daily_gex"
    ).fetchall()
    connection.close()
    out = defaultdict(dict)
    spots = defaultdict(dict)
    for symbol, day, net, spot in rows:
        out[symbol][day[:10]] = net
        spots[symbol][day[:10]] = spot
    return out, spots


def verify_timing(spots, bars_path):
    """Confirm the snapshot is end-of-day for its own date, not intraday."""
    splits = load_splits(bars_path)
    connection = sqlite3.connect(f"file:{bars_path}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker='SPY' "
        "AND frequency='5min' ORDER BY ts").fetchall()
    connection.close()
    daily = resample_regular_session(
        adjust_bars([Bar(*r) for r in rows], splits.get("SPY", [])), minutes=390)
    close = {b.timestamp[:10]: b.close for b in daily}
    same = other = 0
    for day, spot in spots["SPY"].items():
        if day not in close or not spot:
            continue
        same += abs(spot - close[day]) / close[day] < 0.002
        other += 1
    return same, other


def lag(flags, sessions):
    ordered = sorted(sessions)
    return {d: flags.get(ordered[i - 1]) if i else None
            for i, d in enumerate(ordered)}


def build_trades(args, window):
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    names = [r[0] for r in connection.execute(
        "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker")]
    config = TurtleConfig(entry_window=55, exit_window=20, atr_window=20,
                          skip_after_winner=False, directions=(1,))
    trades = []
    for ticker in names:
        if ticker in ETFS:
            continue
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='5min' ORDER BY ts", (ticker,)).fetchall()
        if not rows:
            continue
        series = resample_regular_session(
            adjust_bars([Bar(*r) for r in rows], splits.get(ticker, [])),
            minutes=args.minutes)
        for trade in run_turtle(series, config=config)[0]:
            day = trade.entry_timestamp[:10]
            if window[0] <= day <= window[1]:
                trades.append({"day": day, "entry": trade.entry_timestamp,
                               "exit": trade.exit_timestamp, "r": trade.net_r})
    connection.close()
    return trades


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


def walk(taken, days, risk, flag, always=None):
    nav, peak, worst = 1000.0, 1000.0, 0.0
    by_day = defaultdict(list)
    for trade in taken:
        by_day[trade["exit"][:10]].append(trade)
    path = []
    for day in days:
        for trade in by_day.get(day, ()):
            if always is not None:
                size = always
            else:
                size = 0.5 if flag.get(trade["day"]) is True else 1.0
            nav = max(0.0, nav + trade["r"] * risk * size * nav)
        peak = max(peak, nav)
        worst = min(worst, nav / peak - 1.0)
        path.append(nav)
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25
    return {"final": nav, "maxdd": worst,
            "cagr": (nav / 1000.0) ** (1 / years) - 1 if years > 0 and nav > 0 else -1.0}


def main(argv=None):
    args = parse_args(argv)
    gamma, spots = load_gamma(args.options)
    same, total = verify_timing(spots, args.bars)
    print(f"LEAKAGE AUDIT: snapshot underlying matches the SAME-day close on "
          f"{same}/{total} sessions")
    print("  -> observation_date is an end-of-day print; every regime below is "
          "lagged one session\n")

    sessions = sorted(set(gamma["SPY"]) & set(gamma["QQQ"]))
    window = (sessions[0], sessions[-1])
    trades = build_trades(args, window)
    days = sorted({t["exit"][:10] for t in trades})
    days = [d for d in days if window[0] <= d <= window[1]]
    print(f"{len(trades):,} long {args.minutes}-minute stock trades, "
          f"{window[0]} -> {window[1]} ({len(days)} sessions)\n")

    spy = gamma["SPY"]
    qqq = gamma["QQQ"]
    ranked = sorted(spy.values())
    median_spy = ranked[len(ranked) // 2]
    raw = {
        "SPY gamma below its median": {d: spy[d] < median_spy for d in sessions},
        "SPY gamma negative": {d: spy[d] < 0 for d in sessions},
        "SPY and QQQ both negative": {d: spy[d] < 0 and qqq[d] < 0 for d in sessions},
    }
    flags = {k: lag(v, sessions) for k, v in raw.items()}
    flags["LEAKY: SPY negative, same day"] = raw["SPY gamma negative"]

    base_flag = flags["SPY gamma below its median"]
    rate = sum(1 for d in days if base_flag.get(d) is True) / max(len(days), 1)

    results = {}
    def run(name, flag=None, always=None, shift=False, iid=False):
        got = defaultdict(list)
        keys = sorted(base_flag)
        for seed in range(args.trials):
            rng = random.Random(seed)
            taken = cap(trades, args.max_positions, rng)
            use = flag
            if shift:
                offset = random.Random(3000 + seed).randrange(20, len(keys) - 20)
                use = {keys[i]: base_flag[keys[(i + offset) % len(keys)]]
                       for i in range(len(keys))}
            if iid:
                coin = random.Random(6000 + seed)
                use = {d: coin.random() < rate for d in keys}
            for k, v in walk(taken, days, args.risk, use, always).items():
                got[k].append(v)
        results[name] = got

    run("baseline (no overlay)", flag={d: False for d in days})
    for name, flag in flags.items():
        run(name, flag=flag)
    run("NULL: circular-shifted regime", shift=True)
    run("NULL: independent coin, same rate", iid=True)
    run("NULL: always half risk", always=0.5)

    base = results["baseline (no overlay)"]
    pick = lambda xs, f: sorted(xs)[int(f * len(xs))]
    base_cagr = pick(base["cagr"], .5)
    print(f"BASELINE SIGN CONTROL: median CAGR {base_cagr:+.1%} "
          f"-> {'POSITIVE, overlays must earn their keep' if base_cagr > 0 else 'NEGATIVE: any risk cut will look good'}\n")

    print(f"  {'policy':36s} {'median $':>10s} {'CAGR':>8s} {'maxDD':>8s} "
          f"| {'wins':>9s} {'dCAGR':>8s}")
    report = {"window": window, "trades": len(trades), "sessions": len(days),
              "baseline_cagr": base_cagr, "regime_rate": rate,
              "leakage_same_day_match": [same, total]}
    for name, got in results.items():
        wins = sum(1 for a, b in zip(got["final"], base["final"]) if a > b)
        dc = statistics.median([a - b for a, b in zip(got["cagr"], base["cagr"])])
        print(f"  {name:36s} ${pick(got['final'], .5):>9,.0f} "
              f"{pick(got['cagr'], .5):>8.1%} {pick(got['maxdd'], .5):>8.1%} "
              f"| {wins:>4d}/{args.trials:<4d} {dc:>+8.1%}")
        report[name] = {"median_final": pick(got["final"], .5),
                        "median_cagr": pick(got["cagr"], .5),
                        "median_maxdd": pick(got["maxdd"], .5),
                        "wins": wins, "d_cagr": dc}

    real = report["SPY gamma below its median"]
    for null in ("NULL: circular-shifted regime", "NULL: independent coin, same rate",
                 "NULL: always half risk"):
        gap = real["d_cagr"] - report[null]["d_cagr"]
        print(f"\n  real overlay minus {null[6:]}: {gap:+.1%} CAGR")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
