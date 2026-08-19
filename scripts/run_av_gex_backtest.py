"""Backtest the gamma overlay across nine years of Alpha Vantage chains.

The overlay halves risk on new entries when index dealer gamma is low.  Every
previous test of it was limited by data: 250 sessions with no crisis, or a
fifteen-year proxy that carries a third of the signal.  This runs the same rule
over 2017-2026 on vendor greeks, full strike coverage and every expiration.

Three defences against fooling ourselves, in order of importance:

* **Leakage, checked against the data rather than a field.**  Put-call parity
  implies the underlying the options were struck against.  If that matches the
  *same* session's close and not the next one, the chain is genuinely
  end-of-day, and the one-session lag applied everywhere is sufficient.  A
  deliberately leaky same-day variant prices what the look-ahead would be worth.

* **Three nulls.**  A circular shift preserves regime persistence and destroys
  only the dates; an independent coin preserves the rate and destroys the
  clustering; always-half-risk removes the regime entirely.  A real signal must
  beat all three.

* **The baseline's sign, per period.**  Cutting risk mechanically improves a
  losing strategy, so an overlay that only helps where the base rules lose money
  has shown nothing.  The sign is printed beside every sub-period.

The regime threshold is a *trailing* percentile.  Ranking against the whole
sample would decide what counts as low gamma today using tomorrow's readings.
"""

from __future__ import annotations

import argparse
import bisect
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
                        default=Path("data/options/alphavantage.db"))
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--percentile", type=float, default=0.5)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--risk", type=float, default=0.0005)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/av_gex_backtest.json"))
    return parser.parse_args(argv)


def parity_check(options: Path, bars: Path, symbol: str, samples: int = 60):
    """Does put-call parity point at the same session's close, or the next one?"""
    splits = load_splits(bars)
    connection = sqlite3.connect(f"file:{bars}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
        "frequency='5min' ORDER BY ts", (symbol,)).fetchall()
    connection.close()
    daily = resample_regular_session(
        adjust_bars([Bar(*r) for r in rows], splits.get(symbol, [])), minutes=390)
    close = {b.timestamp[:10]: b.close for b in daily}
    sessions = sorted(close)

    store = sqlite3.connect(f"file:{options}?mode=ro", uri=True)
    days = [d for (d,) in store.execute(
        "SELECT DISTINCT observation_date FROM av_contracts WHERE symbol=? "
        "ORDER BY observation_date", (symbol,))]
    step = max(len(days) // samples, 1)
    same, nxt, used = [], [], 0
    for day in days[::step][:samples]:
        spot = close.get(day)
        if spot is None:
            continue
        pairs = defaultdict(dict)
        for side, strike, dte, bid, ask in store.execute(
            "SELECT side,strike,dte,bid,ask FROM av_contracts WHERE symbol=? AND "
            "observation_date=? AND dte BETWEEN 5 AND 40", (symbol, day)):
            if bid and ask:
                pairs[(strike, dte)][side] = (bid + ask) / 2.0
        best = None
        for (strike, dte), quote in pairs.items():
            if "call" in quote and "put" in quote:
                gap = abs(strike - spot)
                if best is None or gap < best[0]:
                    best = (gap, strike, quote["call"], quote["put"])
        if best is None:
            continue
        _, strike, call, put = best
        implied = call - put + strike        # zero rate, short maturity
        index = bisect.bisect_left(sessions, day)
        following = sessions[index + 1] if index + 1 < len(sessions) else None
        same.append(abs(implied / spot - 1.0))
        if following:
            nxt.append(abs(implied / close[following] - 1.0))
        used += 1
    store.close()
    return used, (statistics.median(same) if same else None), (
        statistics.median(nxt) if nxt else None)


def load_gamma(options: Path, symbol: str):
    store = sqlite3.connect(f"file:{options}?mode=ro", uri=True)
    rows = dict(store.execute(
        "SELECT observation_date,net_gex FROM av_daily WHERE symbol=? AND "
        "net_gex IS NOT NULL ORDER BY observation_date", (symbol,)))
    store.close()
    return rows


def trailing_flag(series, window, percentile):
    """Below its own trailing percentile, using strictly earlier sessions."""
    days = sorted(series)
    values = [series[d] for d in days]
    out = {}
    for i in range(window, len(days)):
        history = sorted(values[i - window:i])
        rank = bisect.bisect_left(history, values[i]) / len(history)
        out[days[i]] = rank < percentile
    return out


def lag_one(flags, sessions):
    ordered = sorted(sessions)
    return {d: flags.get(ordered[i - 1]) if i else None
            for i, d in enumerate(ordered)}


def build_trades(args):
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
            trades.append({"day": trade.entry_timestamp[:10],
                           "entry": trade.entry_timestamp,
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
    for day in days:
        for trade in by_day.get(day, ()):
            size = always if always is not None else (
                0.5 if flag and flag.get(trade["day"]) is True else 1.0)
            nav = max(0.0, nav + trade["r"] * risk * size * nav)
        peak = max(peak, nav)
        worst = min(worst, nav / peak - 1.0)
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25
    return {"final": nav, "maxdd": worst,
            "cagr": (nav / 1000.0) ** (1 / years) - 1 if years > 0 and nav > 0 else -1.0}


def evaluate(trades, days, args, flag, always=None, mode=None, base_flag=None):
    got = defaultdict(list)
    keys = sorted(base_flag) if base_flag else []
    rate = (sum(1 for d in days if base_flag.get(d) is True) / max(len(days), 1)
            if base_flag else 0.0)
    for seed in range(args.trials):
        rng = random.Random(seed)
        taken = cap(trades, args.max_positions, rng)
        use = flag
        if mode == "shift":
            offset = random.Random(4000 + seed).randrange(60, len(keys) - 60)
            use = {keys[i]: base_flag[keys[(i + offset) % len(keys)]]
                   for i in range(len(keys))}
        elif mode == "coin":
            coin = random.Random(8000 + seed)
            use = {d: coin.random() < rate for d in keys}
        for key, value in walk(taken, days, args.risk, use, always).items():
            got[key].append(value)
    return got


def main(argv=None):
    args = parse_args(argv)
    used, same, nxt = parity_check(args.options, args.bars, args.symbol)
    print("LEAKAGE CHECK — put-call parity implies which session's price?")
    print(f"  {used} sampled sessions | median error vs SAME-day close "
          f"{same:.4%} | vs NEXT-day close {nxt:.4%}")
    print(f"  -> chain is priced off the {'SAME' if same < nxt else 'NEXT'} "
          f"session; a one-session lag is "
          f"{'sufficient' if same < nxt else 'NOT sufficient'}\n")

    gamma = load_gamma(args.options, args.symbol)
    sessions = sorted(gamma)
    raw_flag = trailing_flag(gamma, args.window, args.percentile)
    flag = lag_one(raw_flag, sessions)
    trades = build_trades(args)
    days = sorted({t["exit"][:10] for t in trades})
    days = [d for d in days if sessions[0] <= d <= sessions[-1]]
    trades = [t for t in trades if sessions[0] <= t["day"] <= sessions[-1]]
    seq = [raw_flag[d] for d in sorted(raw_flag)]
    runs = 1 + sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    print(f"{len(trades):,} long {args.minutes}-minute trades, {len(days):,} sessions, "
          f"{days[0]} -> {days[-1]}")
    print(f"regime on {sum(seq) / len(seq):.0%} of sessions, "
          f"~{runs:,} independent episodes\n")

    pick = lambda xs, f: sorted(xs)[int(f * len(xs))]
    report = {"symbol": args.symbol, "trades": len(trades), "sessions": len(days),
              "episodes": runs, "parity": {"same": same, "next": nxt}}

    schemes = [
        ("baseline (no overlay)", dict(flag={d: False for d in days})),
        ("gamma low, lagged 1 session", dict(flag=flag)),
        ("LEAKY: gamma low, same day", dict(flag=raw_flag)),
        ("NULL: circular-shifted", dict(flag=None, mode="shift")),
        ("NULL: independent coin", dict(flag=None, mode="coin")),
        ("NULL: always half risk", dict(flag=None, always=0.5)),
    ]
    base = evaluate(trades, days, args, **schemes[0][1], base_flag=raw_flag)
    base_cagr = pick(base["cagr"], .5)
    print(f"BASELINE SIGN: median CAGR {base_cagr:+.1%} -> "
          f"{'positive, overlays must earn their keep' if base_cagr > 0 else 'NEGATIVE: any risk cut flatters'}\n")
    print(f"  {'scheme':32s} {'CAGR':>8s} {'maxDD':>8s} {'wins':>9s} {'dCAGR':>8s}")
    results = {"baseline (no overlay)": base}
    for name, kwargs in schemes:
        got = base if name.startswith("baseline") else evaluate(
            trades, days, args, base_flag=raw_flag, **kwargs)
        results[name] = got
        wins = sum(1 for a, b in zip(got["final"], base["final"]) if a > b)
        dc = statistics.median([a - b for a, b in zip(got["cagr"], base["cagr"])])
        print(f"  {name:32s} {pick(got['cagr'], .5):>8.1%} "
              f"{pick(got['maxdd'], .5):>8.1%} {wins:>4d}/{args.trials:<4d} "
              f"{dc:>+8.1%}")
        report[name] = {"cagr": pick(got["cagr"], .5),
                        "maxdd": pick(got["maxdd"], .5), "wins": wins, "d_cagr": dc}

    print(f"\n  {'period':16s} {'baseline':>10s} {'overlay d':>10s} {'wins':>9s}")
    periods = [("2017-2019", "2017-01-01", "2019-01-01"),
               ("2019-2021", "2019-01-01", "2021-01-01"),
               ("2021-2023", "2021-01-01", "2023-01-01"),
               ("2023-2025", "2023-01-01", "2025-01-01"),
               ("2025-2026", "2025-01-01", "2027-01-01")]
    report["periods"] = {}
    for label, lo, hi in periods:
        sub_days = [d for d in days if lo <= d < hi]
        sub = [t for t in trades if lo <= t["day"] < hi]
        if len(sub) < 50 or len(sub_days) < 60:
            continue
        b = evaluate(sub, sub_days, args, flag={d: False for d in sub_days},
                     base_flag=raw_flag)
        o = evaluate(sub, sub_days, args, flag=flag, base_flag=raw_flag)
        wins = sum(1 for x, y in zip(o["final"], b["final"]) if x > y)
        dc = statistics.median([x - y for x, y in zip(o["cagr"], b["cagr"])])
        bc = pick(b["cagr"], .5)
        print(f"  {label:16s} {bc:>10.1%} {dc:>+10.1%} {wins:>4d}/{args.trials:<4d}")
        report["periods"][label] = {"baseline": bc, "d_cagr": dc, "wins": wins}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
