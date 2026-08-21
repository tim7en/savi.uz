"""Conditions knowable *before* the breakout bar, which are the free ones.

The filter test settled something structural. Acting on the breakout bar's own
close, high or volume requires close-confirmed entry, and that costs 0.73 Sharpe
against plain stop entry -- more than the +2.343R selection effect it buys. You
pay up at the next open and the fixed 2N stop sits proportionally tighter, so
the cost compounds rather than behaving like slippage.

Anything computed from bars *strictly before* the breakout is free by contrast.
The stop order still rests at the channel edge and fills exactly as it always
did; only the size changes. That is why the moving-average rule worked where the
higher-timeframe gate had failed.

So this measures pre-breakout conditions, all evaluated at the bar before the
breach:

* **leading volume** -- mean volume over the last 5 bars against the last 20.
  Volume rising *into* the level, as distinct from volume *on* the breakout bar,
  which turned out to mark failure avoidance rather than trend sustainability:
  its right tail moved 1.9 points against close location's 14.2.
* **range compression** -- N against its own 50-bar mean. A channel approached
  through a tightening range is the classic coil, and unlike the breakout bar it
  is fully formed before the decision.
* **distance travelled** -- how far price has already come, in N, over the
  lookback. A breakout arriving after an extended run is a different event from
  one arriving out of a base.

Reported by quintile with the right tail beside the mean, because a trend book
lives on the tail: chapter six's 13F result had a mean t-statistic of -0.57 and
a tail that moved 4.8 points, and only the second one mattered. A label-shuffling
null holds the outcome distribution fixed and destroys only the association.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import (  # noqa: E402
    TurtleConfig, rolling_extremes, run_turtle, wilder_atr,
)
from savi_uz.volume_profile import Bar  # noqa: E402

FIXED = dict(entry_window=55, exit_window=20, atr_window=20,
             skip_after_winner=False, use_channel_exit=False, chandelier_atr=3.0)

LEVERED_MARKERS = ("2X", "3X", "1.5X", "BULL", "BEAR", "LEVERAGED", "INVERSE",
                   "ULTRA", "DAILY ", "SHORT ")

QUINTILES = 5


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/intraday/bars_av.db"))
    parser.add_argument("--source-frequency", default="5min")
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--cost-bp", type=float, default=5.0)
    parser.add_argument("--fast-volume", type=int, default=5)
    parser.add_argument("--slow-volume", type=int, default=20)
    parser.add_argument("--compression-window", type=int, default=50)
    parser.add_argument("--travel-window", type=int, default=55)
    parser.add_argument("--nulls", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/pre_breakout.json"))
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
        "SELECT DISTINCT ticker FROM bars WHERE frequency=? ORDER BY ticker",
        (args.source_frequency,)) if r[0] not in drop]
    if args.limit:
        tickers = tickers[:args.limit]
    book = {}
    for ticker in tickers:
        raw = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency=? AND ts>=? ORDER BY ts",
            (ticker, args.source_frequency, args.start)).fetchall()
        if len(raw) < 4000:
            continue
        bars = [Bar(*r) for r in raw]
        if args.minutes != 5:
            bars = resample_regular_session(bars, minutes=args.minutes)
        if len(bars) >= 800:
            book[ticker] = bars
    connection.close()
    print(f"excluded {len(drop)} levered or inverse wrappers")
    return book


def mean_of(values, end, window):
    """Mean of ``window`` values ending at index ``end`` inclusive."""
    start = end - window + 1
    if start < 0:
        return float("nan")
    block = values[start:end + 1]
    return sum(block) / len(block) if block else float("nan")


def features(bars: list[Bar], args) -> dict[str, dict[str, float]]:
    highs = rolling_extremes([b.high for b in bars], FIXED["entry_window"], True)
    atr = wilder_atr(bars, FIXED["atr_window"])
    volumes = [float(b.volume or 0.0) for b in bars]
    closes = [b.close for b in bars]
    out: dict[str, dict[str, float]] = {}
    for index, bar in enumerate(bars):
        level = highs[index]
        if math.isnan(level) or bar.high <= level:
            continue
        prior = index - 1          # everything is read at the bar before entry
        if prior < max(args.slow_volume, args.compression_window,
                       args.travel_window):
            continue
        n = atr[prior]
        if not n or math.isnan(n) or n <= 0:
            continue
        fast = mean_of(volumes, prior, args.fast_volume)
        slow = mean_of(volumes, prior, args.slow_volume)
        atr_mean = mean_of([a for a in atr], prior, args.compression_window)
        travelled = (closes[prior] - closes[prior - args.travel_window]) / n
        out[bar.timestamp] = {
            "leading_volume": fast / slow if slow > 0 else float("nan"),
            "range_compression": n / atr_mean if atr_mean and atr_mean > 0
            else float("nan"),
            "distance_travelled_n": travelled,
        }
    return out


def quintile_table(pairs):
    ordered = sorted(pairs)
    size = len(ordered) // QUINTILES
    rows = []
    for q in range(QUINTILES):
        lo = q * size
        hi = (q + 1) * size if q < QUINTILES - 1 else len(ordered)
        block = ordered[lo:hi]
        values = [r for _, r in block]
        rows.append({
            "quintile": q + 1,
            "feature_low": block[0][0], "feature_high": block[-1][0],
            "trades": len(values),
            "mean_r": statistics.fmean(values),
            "share_above_3r": sum(1 for v in values if v > 3.0) / len(values),
            "share_stopped": sum(1 for v in values if v <= -0.99) / len(values),
        })
    return rows


def main(argv=None) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)
    book = load(args)
    config = TurtleConfig(**FIXED, directions=(1,),
                          round_trip_cost=args.cost_bp / 10_000)
    records = []
    for ticker, bars in book.items():
        table = features(bars, args)
        trades, _ = run_turtle(bars, config=config)
        for trade in trades:
            row = table.get(trade.entry_timestamp)
            if row:
                records.append({**row, "net_r": trade.net_r})
    print(f"{len(book)} instruments, {len(records):,} breakouts with "
          f"pre-breakout features, {args.cost_bp:g}bp\n")

    report = {}
    for feature in ("leading_volume", "range_compression", "distance_travelled_n"):
        pairs = [(r[feature], r["net_r"]) for r in records
                 if not math.isnan(r[feature])]
        if len(pairs) < 500:
            continue
        rows = quintile_table(pairs)
        print(f"=== {feature} ({len(pairs):,} breakouts) ===")
        print(f"  {'quintile':>8s} {'range':>18s} {'trades':>8s} {'mean R':>8s} "
              f"{'>+3R':>7s} {'stopped':>8s}")
        for row in rows:
            print(f"  {row['quintile']:>8d} "
                  f"{('%.2f-%.2f' % (row['feature_low'], row['feature_high'])):>18s} "
                  f"{row['trades']:>8,d} {row['mean_r']:>8.3f} "
                  f"{row['share_above_3r']:>7.1%} {row['share_stopped']:>8.1%}")
        observed_mean = rows[-1]["mean_r"] - rows[0]["mean_r"]
        observed_tail = rows[-1]["share_above_3r"] - rows[0]["share_above_3r"]
        outcomes = [r for _, r in pairs]
        keys = [f for f, _ in pairs]
        dm, dt = [], []
        for _ in range(args.nulls):
            shuffled = outcomes[:]
            rng.shuffle(shuffled)
            nr = quintile_table(list(zip(keys, shuffled)))
            dm.append(nr[-1]["mean_r"] - nr[0]["mean_r"])
            dt.append(nr[-1]["share_above_3r"] - nr[0]["share_above_3r"])
        p_mean = sum(1 for d in dm if abs(d) >= abs(observed_mean)) / len(dm)
        p_tail = sum(1 for d in dt if abs(d) >= abs(observed_tail)) / len(dt)
        print(f"  top-minus-bottom mean R  {observed_mean:+.3f}  p={p_mean:.3f}")
        print(f"  top-minus-bottom >+3R    {observed_tail:+.1%}  p={p_tail:.3f}")
        print(f"  -> {'tail moves' if p_tail < 0.05 else 'TAIL DOES NOT MOVE'}"
              f" -- the tail is what a trend book lives on\n", flush=True)
        report[feature] = {"quintiles": rows, "spread_mean_r": observed_mean,
                           "p_mean": p_mean, "spread_tail": observed_tail,
                           "p_tail": p_tail, "trades": len(pairs)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
