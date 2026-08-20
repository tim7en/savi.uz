"""How large a fund could run this, using consolidated rather than intraday volume.

A first attempt took dollar volume from the intraday bars and produced answers
around a million dollars, which is wrong by roughly two orders of magnitude: the
intraday endpoint reports only part of the tape, so SPY came out at $384M a day
against a real figure near thirty billion, and the two leveraged bond funds
reported a median bar volume of exactly zero.  Capacity computed on that is
meaningless.

The daily endpoint carries consolidated volume, so it is fetched here instead --
thirty-nine calls, which is cheap next to being wrong.

Two windows are reported.  The full-sample median describes the history the
backtest ran on; the last two hundred and fifty sessions describe the market a
fund would be entering now, and they differ substantially for names whose
liquidity has drained (TBT) or exploded (ASTS).  The recent window is the one to
plan against.

The participation cap is the honest weak point.  One per cent of a day's volume
is a conventional figure and it assumes the position can be spread across the
session.  A breakout entry cannot: it fires at a specific level, in the minutes
when everyone else's breakout is firing at the same level, so realisable capacity
is below this and the gap is widest exactly on the trades that matter most.  Read
these as an upper bound.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sqlite3
import statistics
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import TurtleConfig, run_turtle  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

BASE = dict(entry_window=55, exit_window=20, atr_window=20, skip_after_winner=False,
            use_channel_exit=False, chandelier_atr=3.0, directions=(1,))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=pathlib.Path,
                        default=pathlib.Path("data/intraday/bars.db"))
    parser.add_argument("--funding", type=pathlib.Path,
                        default=pathlib.Path("out/strategy/binance_funding.json"))
    parser.add_argument("--leverage", type=pathlib.Path,
                        default=pathlib.Path("out/strategy/venue_leverage.json"))
    parser.add_argument("--cache", type=pathlib.Path,
                        default=pathlib.Path("out/strategy/daily_volume.json"))
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--participation", type=float, default=0.01)
    parser.add_argument("--cost", type=float, default=0.0002)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("out/strategy/capacity.json"))
    return parser.parse_args(argv)


def api_key() -> str:
    for line in pathlib.Path(".env").read_text(encoding="utf-8").splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "ALPHAVANTAGE_API_KEY":
            return value.strip().strip('"').strip("'")
    raise SystemExit("error: ALPHAVANTAGE_API_KEY missing from .env")


def fetch_volume(symbol, key, attempts=4):
    query = urllib.parse.urlencode({"function": "TIME_SERIES_DAILY_ADJUSTED",
                                    "symbol": symbol, "outputsize": "full",
                                    "apikey": key})
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                f"https://www.alphavantage.co/query?{query}",
                headers={"User-Agent": "research/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode())
        except Exception:
            time.sleep(2 ** attempt)
            continue
        series = payload.get("Time Series (Daily)")
        if series:
            # Dollar volume needs no split adjustment: a split multiplies shares
            # and divides price by the same factor, leaving traded value alone.
            return {day: float(v["6. volume"]) * float(v["4. close"])
                    for day, v in series.items()
                    if v.get("6. volume") and v.get("4. close")}
        time.sleep(15)
    return None


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


def main(argv=None):
    args = parse_args(argv)
    names = sorted(json.loads(args.funding.read_text(encoding="utf-8")))
    base = json.loads(args.leverage.read_text(encoding="utf-8"))[
        str(args.cost)]["base_risk"]

    cache = {}
    if args.cache.exists():
        cache = json.loads(args.cache.read_text(encoding="utf-8"))
    missing = [n for n in names if n not in cache]
    if missing:
        key = api_key()
        print(f"fetching consolidated daily volume for {len(missing)} symbols...",
              flush=True)

        def work(symbol):
            time.sleep(0.85 * missing.index(symbol) % 3)
            return symbol, fetch_volume(symbol, key)

        with ThreadPoolExecutor(max_workers=3) as pool:
            for symbol, series in pool.map(work, missing):
                if series:
                    cache[symbol] = series
                else:
                    print(f"  {symbol}: failed", flush=True)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_text(json.dumps(cache), encoding="utf-8")

    # Position notional per unit of equity, per instrument, from the real book.
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    pooled = []
    config = TurtleConfig(**{**BASE, "round_trip_cost": args.cost})
    for ticker in names:
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='5min' ORDER BY ts", (ticker,)).fetchall()
        if not rows:
            continue
        five = adjust_bars([Bar(*r) for r in rows], splits.get(ticker, []))
        bars = resample_regular_session(five, minutes=args.minutes)
        for trade in run_turtle(bars, config=config)[0]:
            pooled.append({"ticker": ticker, "entry": trade.entry_timestamp,
                           "exit": trade.exit_timestamp,
                           "basis": trade.cost_basis_r})
    connection.close()
    taken = cap(pooled, args.max_positions, random.Random(0))

    by_ticker = defaultdict(list)
    for trade in taken:
        by_ticker[trade["ticker"]].append(trade["basis"])

    print(f"\nposition notional and capacity at {args.participation:.0%} of daily "
          f"volume, at 1x risk ({base:.4%} per 1N)\n")
    print(f"  {'ticker':<7s} {'$vol full':>11s} {'$vol 250d':>11s} "
          f"{'position':>9s} {'cap full':>11s} {'cap recent':>11s}")
    report = {}
    for ticker in sorted(by_ticker):
        series = cache.get(ticker)
        if not series:
            continue
        days = sorted(series)
        full = statistics.median(series[d] for d in days)
        recent = statistics.median(series[d] for d in days[-250:])
        notional = statistics.median(by_ticker[ticker]) * base
        report[ticker] = {"adv_full": full, "adv_recent": recent,
                          "position_fraction": notional,
                          "capacity_full": args.participation * full / notional,
                          "capacity_recent": args.participation * recent / notional}
        print(f"  {ticker:<7s} {full / 1e6:>10,.0f}M {recent / 1e6:>10,.0f}M "
              f"{notional:>8.1%} {report[ticker]['capacity_full'] / 1e6:>10,.0f}M "
              f"{report[ticker]['capacity_recent'] / 1e6:>10,.0f}M")

    recents = sorted(v["capacity_recent"] for v in report.values())
    tight = sorted(report.items(), key=lambda kv: kv[1]["capacity_recent"])[:5]
    print(f"\n  across {len(report)} instruments, on recent volume:")
    print(f"    median   ${recents[len(recents) // 2] / 1e6:,.0f}M")
    print(f"    tightest ${recents[0] / 1e6:,.0f}M  "
          + ", ".join(f"{t} ${v['capacity_recent'] / 1e6:,.0f}M" for t, v in tight))
    print(f"\n  the book must clear the tightest name it actually trades, so the "
          f"binding\n  figure is nearer the low end than the median; at 3x risk, "
          f"divide by three.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
