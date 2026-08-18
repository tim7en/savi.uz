"""Industry-grade controls for the GEX risk overlay.

The reported GEX result clears two controls already: half-risk on every trade,
and a date-shuffled GEX series.  Both are necessary and neither is sufficient.

* **Persistence.** SPY/QQQ gamma sign flips roughly every 4-5 sessions, so a
  250-day sample holds about 58 independent regime episodes, not 250.  A
  date-shuffle breaks that persistence entirely, reducing the null to white
  noise -- a far weaker opponent than the real series.  The honest null is a
  *circularly shifted* GEX series: rotating the sign sequence by a random offset
  preserves every run length and all autocorrelation, and destroys only the
  alignment with actual dates.

* **Redundancy.** Negative gamma regimes coincide with volatile, falling tape.
  For a long-only stock book, "cut risk when the market is stressed" would help
  in any sample where the market fell during those windows.  So GEX must be
  compared against cheap stress proxies that carry decades of history rather
  than one year: index below its own trailing mean, and elevated realised
  volatility.

If GEX cannot beat a persistence-matched shuffle, or cannot beat a moving
average, then it is a proxy for market stress and not an independent signal.
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

ETFS = {"SPY", "QQQ", "IWM", "GLD", "EWJ", "EWT", "EWY", "KWEB"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--options", type=Path,
                        default=Path("data/options/marketdata.db"))
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--risk", type=float, default=0.01)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/gex_control_analysis.json"))
    return parser.parse_args(argv)


def gex_flags(path: Path):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = defaultdict(dict)
    for symbol, day, net in connection.execute(
        "SELECT symbol,observation_date,net_gex FROM daily_gex ORDER BY observation_date"
    ):
        rows[symbol][day[:10]] = net
    connection.close()
    return rows


def build_trades(args, window):
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    names = [r[0] for r in connection.execute(
        "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker")]
    config = TurtleConfig(entry_window=55, exit_window=20, atr_window=20,
                          skip_after_winner=False, directions=(1,))
    trades, marks = [], {}
    for ticker in names:
        if ticker in ETFS:
            continue
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='5min' ORDER BY ts", (ticker,)).fetchall()
        if not rows:
            continue
        five = adjust_bars([Bar(*r) for r in rows], splits.get(ticker, []))
        series = resample_regular_session(five, minutes=args.minutes)
        for trade in run_turtle(series, config=config)[0]:
            day = trade.entry_timestamp[:10]
            if window[0] <= day <= window[1]:
                trades.append({"ticker": ticker, "entry": trade.entry_timestamp,
                               "exit": trade.exit_timestamp, "day": day,
                               "net_r": trade.net_r})
    # SPY daily closes for the moving-average and volatility proxies.
    rows = connection.execute(
        "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker='SPY' AND "
        "frequency='5min' ORDER BY ts").fetchall()
    connection.close()
    daily = resample_regular_session(
        adjust_bars([Bar(*r) for r in rows], splits.get("SPY", [])), minutes=390)
    marks = {bar.timestamp[:10]: bar.close for bar in daily}
    return trades, marks


def proxy_flags(marks, days, ma_window=20, vol_window=20):
    ordered = sorted(marks)
    close = [marks[d] for d in ordered]
    index = {d: i for i, d in enumerate(ordered)}
    below_ma, high_vol = {}, {}
    rets = [0.0] + [close[i] / close[i - 1] - 1.0 for i in range(1, len(close))]
    vols = []
    for i in range(len(close)):
        if i >= vol_window:
            vols.append(statistics.stdev(rets[i - vol_window + 1:i + 1]))
        else:
            vols.append(None)
    finite = sorted(v for v in vols if v is not None)
    median_vol = finite[len(finite) // 2] if finite else 0.0
    for day in days:
        i = index.get(day)
        if i is None or i < max(ma_window, vol_window) + 1:
            below_ma[day] = None
            high_vol[day] = None
            continue
        # Both read the previous session only.
        mean = sum(close[i - ma_window:i]) / ma_window
        below_ma[day] = close[i - 1] < mean
        high_vol[day] = (vols[i - 1] or 0.0) > median_vol
    return below_ma, high_vol


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


def walk(taken, days, risk, flag):
    nav = 1000.0
    path = []
    by_day = defaultdict(list)
    for trade in taken:
        by_day[trade["exit"][:10]].append(trade)
    for day in days:
        for trade in by_day.get(day, ()):
            size = 0.5 if flag.get(trade["day"]) is True else 1.0
            nav = max(0.0, nav + trade["net_r"] * risk * size * nav)
        path.append(nav)
    return path


def stats(path, days):
    peak, maxdd = path[0], 0.0
    for value in path:
        peak = max(peak, value)
        maxdd = min(maxdd, value / peak - 1.0)
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25
    cagr = (path[-1] / path[0]) ** (1 / years) - 1 if years > 0 and path[-1] > 0 else -1.0
    rets = [path[i] / path[i - 1] - 1.0 for i in range(1, len(path)) if path[i - 1] > 0]
    sd = statistics.stdev(rets) if len(rets) > 1 else 0.0
    return {"final": path[-1], "cagr": cagr, "maxdd": maxdd,
            "sharpe": (statistics.mean(rets) / sd * math.sqrt(252.0)) if sd else 0.0}


def lag_one_session(flag, sessions):
    """Map each session to the PREVIOUS session's flag.

    ``daily_gex.observation_date`` is an end-of-day snapshot of that date, so
    using it to size a trade entered during the same session reads the close
    before the session has happened. Lagging by one session means the regime is
    known before the next open, which also makes gap risk irrelevant to the
    decision: the size was already fixed when the gap occurred.
    """
    ordered = sorted(sessions)
    return {
        day: flag.get(ordered[i - 1]) if i > 0 else None
        for i, day in enumerate(ordered)
    }


def circular_shift(flag, days, offset):
    """Rotate the regime, preserving every run length and all autocorrelation."""
    ordered = [flag.get(d) for d in days]
    n = len(ordered)
    return {days[i]: ordered[(i + offset) % n] for i in range(n)}


def main(argv=None):
    args = parse_args(argv)
    gex = gex_flags(args.options)
    common = sorted(set(gex.get("SPY", {})) & set(gex.get("QQQ", {})))
    window = (common[0], common[-1])
    trades, marks = build_trades(args, window)
    days = sorted({t["exit"][:10] for t in trades} | set(common))
    days = [d for d in days if window[0] <= d <= window[1]]
    print(f"{len(trades):,} long 30-minute stock trades, {len(days)} sessions, "
          f"{window[0]} -> {window[1]}\n")

    both_raw = {d: (gex["SPY"].get(d, 0) < 0 and gex["QQQ"].get(d, 0) < 0)
                for d in common}
    spy_raw = {d: gex["SPY"].get(d, 0) < 0 for d in common}
    # Everything downstream uses the lagged regime; the same-day version is kept
    # only to quantify how much of the reported edge was look-ahead.
    both_neg = lag_one_session(both_raw, common)
    spy_neg = lag_one_session(spy_raw, common)
    below_ma, high_vol = proxy_flags(marks, days)
    never = {d: False for d in days}

    policies = [
        ("baseline (no overlay)", never, False),
        ("GEX: both negative (lagged 1 session)", both_neg, False),
        ("GEX: SPY negative (lagged 1 session)", spy_neg, False),
        ("LOOK-AHEAD: SPY negative, same day", spy_raw, False),
        ("PROXY: SPY below its 20d mean", below_ma, False),
        ("PROXY: realised vol above median", high_vol, False),
        ("CONTROL: circular-shifted GEX", both_neg, True),
    ]

    results = {}
    for label, flag, shifted in policies:
        got = defaultdict(list)
        finals = []
        for seed in range(args.trials):
            rng = random.Random(seed)
            taken = cap(trades, args.max_positions, rng)
            use = flag
            if shifted:
                # A fresh rotation per trial, never the identity.
                offset = random.Random(10_000 + seed).randrange(20, len(days) - 20)
                use = circular_shift(flag, days, offset)
            s = stats(walk(taken, days, args.risk, use), days)
            finals.append(s["final"])
            for key, value in s.items():
                got[key].append(value)
        results[label] = {"finals": finals, **{k: v for k, v in got.items()}}

    base = results["baseline (no overlay)"]
    pick = lambda xs, f: sorted(xs)[int(f * len(xs))]
    print(f"  {'policy':34s} {'median $':>10s} {'CAGR':>8s} {'Sharpe':>7s} "
          f"{'maxDD':>8s} | {'NAV wins':>9s} {'dCAGR':>8s} {'dSharpe':>8s}")
    report = {"window": window, "trades": len(trades), "sessions": len(days)}
    for label in results:
        r = results[label]
        wins = sum(1 for a, b in zip(r["final"], base["final"]) if a > b)
        dc = statistics.median([a - b for a, b in zip(r["cagr"], base["cagr"])])
        ds = statistics.median([a - b for a, b in zip(r["sharpe"], base["sharpe"])])
        print(f"  {label:34s} ${pick(r['final'], .5):>9,.0f} "
              f"{pick(r['cagr'], .5):>8.1%} {pick(r['sharpe'], .5):>7.2f} "
              f"{pick(r['maxdd'], .5):>8.1%} | "
              f"{wins:>4d}/{args.trials:<4d} {dc:>+8.1%} {ds:>+8.2f}")
        report[label] = {"median_final": pick(r["final"], .5),
                         "median_cagr": pick(r["cagr"], .5),
                         "median_sharpe": pick(r["sharpe"], .5),
                         "median_maxdd": pick(r["maxdd"], .5),
                         "nav_wins": wins, "d_cagr": dc, "d_sharpe": ds}

    seq = [both_raw[d] for d in common]
    runs = 1 + sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    print(f"\n  independent regime episodes in the sample: ~{runs} "
          f"(mean run {len(common) / runs:.1f} sessions)")
    print("  Treating 822 trades as independent observations overstates the "
          "evidence by roughly an order of magnitude.")
    report["regime_runs"] = runs
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
