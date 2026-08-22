"""Earnings as a veto on the breakout book, which is the form the evidence takes.

Every attempt today to make the earnings effect carry a book of its own failed:
the conditional ordering is real and replicates, and it is still too small to
pay for a stop, a round trip and a six-slot cap by itself.  What an effect of
that size can do is decide which of two otherwise-identical trades to take.  So
this stops trying to trade earnings and instead lets earnings decline trades the
breakout book was going to make anyway.

The rule: a 55-bar Donchian breakout is refused if, within the last ``window``
sessions, that name reported a miss the market also sold.  Nothing else changes
-- same entries, same 2N stop, same half-N pyramid, same 3N chandelier, same
cap.  Declining a trade frees its slot for the next name rather than leaving it
idle, which is the only honest way to score a filter on a capacity-bound book.

Three controls, and the second is the one that decides it.

*The unfiltered book*, so the comparison is against the incumbent rather than
against nothing.

*The exact reversal.*  Veto the beats instead -- refuse a breakout in a name
that just beat and rallied.  If declining the good news performs as well as
declining the bad, the veto carries no information and the first result was the
tie-break landing well.

*A random veto of the same size.*  Refusing trades changes which names occupy
the slots, and the capacity ordering alone moves Sharpe by more than most
overlays claim to.  A veto must beat a coin that declines the same number.

The window is not tuned to the answer: three values spanning the drift horizon
measured in the event study are all reported, in sample and out, so a result
that exists at one window and not its neighbours is visible as such.
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
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.turtle import TurtleConfig, run_turtle  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

import run_vol_stretch_zones as shared  # noqa: E402

FIXED = dict(entry_window=55, exit_window=20, atr_window=20,
             skip_after_winner=False, use_channel_exit=False, chandelier_atr=3.0)
WINDOWS = (20, 40, 60)


def ticker_seed(t):
    return zlib.crc32(t.encode("utf-8")) % 10_000


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path,
                        default=Path("data/13f/alphavantage_daily.db"))
    parser.add_argument("--earnings", type=Path, default=Path("data/sp500_data"))
    parser.add_argument("--split", default="2013-01-01")
    parser.add_argument("--cost-bp", type=float, default=10.0)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--null-draws", type=int, default=40)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/earnings_veto.json"))
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


def flagged_days(book, folder, args):
    """Per name, the sessions after a miss the market sold, and after a beat it bought."""
    misses, beats = defaultdict(list), defaultdict(list)
    for ticker, bars in book.items():
        path = folder / f"{ticker}_earnings.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))["data"]
        except Exception:
            continue
        days = [b.timestamp for b in bars]
        closes = [b.close for b in bars]
        for row in payload.get("quarterlyEarnings", []):
            try:
                pct = float(row["surprisePercentage"])
            except (TypeError, ValueError, KeyError):
                continue
            day = str(row.get("reportedDate", ""))[:10]
            if len(day) != 10:
                continue
            position = bisect_left(days, day)
            reaction = position + 1 if str(
                row.get("reportTime", "")).startswith("post") else position
            if reaction < 2 or reaction >= len(bars) - 1:
                continue
            prior = closes[reaction - 1]
            if prior <= 0:
                continue
            move = (closes[reaction] - prior) / prior
            if pct <= 0 and move <= 0:
                misses[ticker].append(reaction)
            elif pct > 0 and move > 0:
                beats[ticker].append(reaction)
    return misses, beats


def book_trades(book, args):
    config = TurtleConfig(**FIXED, directions=(1,),
                          round_trip_cost=args.cost_bp / 10_000)
    trades = []
    for ticker, bars in book.items():
        index_of = {b.timestamp: i for i, b in enumerate(bars)}
        raw, _ = run_turtle(bars, config=config)
        for t in raw:
            trades.append({"ticker": ticker, "entry": t.entry_timestamp[:10],
                           "exit": t.exit_timestamp[:10], "r": t.net_r,
                           "dir": t.direction, "units": t.unit_entries,
                           "index": index_of.get(t.entry_timestamp[:10], -1)})
    trades.sort(key=lambda t: t["entry"])
    return trades


def vetoed(trades, flags, window):
    """True where a flagged report sits within ``window`` sessions before entry."""
    marks = []
    for t in trades:
        days = flags.get(t["ticker"], [])
        if not days or t["index"] < 0:
            marks.append(False)
            continue
        lo = bisect_right(days, t["index"] - window - 1)
        hi = bisect_right(days, t["index"])
        marks.append(hi > lo)
    return marks


def main(argv=None) -> int:
    args = parse_args(argv)
    book = load_book(args)
    misses, beats = flagged_days(book, args.earnings, args)
    trades = book_trades(book, args)
    closes = {t: {b.timestamp: b.close for b in bars} for t, bars in book.items()}
    print(f"{len(book)} names, {len(trades):,d} breakouts, "
          f"{sum(len(v) for v in misses.values()):,d} miss-and-sold reports, "
          f"{sum(len(v) for v in beats.values()):,d} beat-and-bought reports")
    print(f"in sample to {args.split}, out of sample after\n")
    report = {"names": len(book), "breakouts": len(trades), "windows": {}}

    for window in WINDOWS:
        veto_miss = vetoed(trades, misses, window)
        veto_beat = vetoed(trades, beats, window)
        kept_miss = [t for t, v in zip(trades, veto_miss) if not v]
        kept_beat = [t for t, v in zip(trades, veto_beat) if not v]
        declined = sum(veto_miss)
        print(f"########## veto window {window} sessions "
              f"({declined:,d} of {len(trades):,d} breakouts declined, "
              f"{declined/len(trades):.1%}) ##########")
        print(f"  {'arm':34s} {'period':>10s} {'offered':>8s} {'taken':>7s} "
              f"{'Sharpe':>7s} {'[5-95%]':>15s} {'CAGR':>8s}")
        cell = {"declined": declined, "share": declined / len(trades)}

        for period, lo, hi in (("in sample", None, args.split),
                               ("out of sample", args.split, None)):
            def cut(rows):
                return [t for t in rows
                        if (lo is None or t["entry"] >= lo)
                        and (hi is None or t["entry"] < hi)]
            for label, pooled in (("no veto (incumbent)", cut(trades)),
                                  ("veto the misses", cut(kept_miss)),
                                  ("veto the beats (reversal)", cut(kept_beat))):
                result = shared.assess(pooled, args)
                if not result:
                    continue
                cell.setdefault(period, {})[label] = result
                print(f"  {label:34s} {period:>10s} {result['offered']:>8,d} "
                      f"{result['taken']:>7,d} {result['sharpe']:>7.2f} "
                      f"{('[%.2f-%.2f]' % (result['sharpe_p05'], result['sharpe_p95'])):>15s} "
                      f"{result['cagr']:>8.1%}", flush=True)

            if period != "out of sample":
                continue
            nulls = []
            want = len(cut(trades)) - len(cut(kept_miss))
            for draw in range(args.null_draws):
                rng = random.Random(77_000 + 131 * draw)
                pool = cut(trades)
                if want <= 0 or want >= len(pool):
                    break
                drop = set(rng.sample(range(len(pool)), want))
                outcome = shared.assess(
                    [t for i, t in enumerate(pool) if i not in drop], args)
                if outcome:
                    nulls.append(outcome["sharpe"])
            if nulls:
                nulls.sort()
                edge = cell["out of sample"]["veto the misses"]["sharpe"]
                above = sum(1 for x in nulls if x >= edge) / len(nulls)
                cell["null"] = {"median": statistics.median(nulls),
                                "low": nulls[0], "high": nulls[-1], "p": above}
                print(f"  {'random veto, same size (null)':34s} "
                      f"{'out of sample':>10s} {'':>8s} {'':>7s} "
                      f"{statistics.median(nulls):>7.2f} "
                      f"{('[%.2f-%.2f]' % (nulls[0], nulls[-1])):>15s}")
                print(f"  -> the veto beats the coin in {1 - above:.0%} of draws")
        report["windows"][str(window)] = cell
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
