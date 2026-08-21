"""Which breakouts are worth the slot: thrust, close location, and volume.

Seventy-two per cent of breakouts are refused because the book is full, so the
question of which ones deserve a scarce slot is now the largest single design
choice in the system. Before proposing a rule, measure the conditional
distribution -- and measure the right tail rather than the mean.

That last point is chapter six's lesson and it is not a stylistic preference. A
breakout book stops out of the left tail at a fixed multiple of volatility and
rides the right tail as far as it goes, so a population with an unchanged centre
and a fatter right tail is worth a great deal to it. Chapter six found exactly
that shape in 13F-accumulated names: the difference in means carried a
t-statistic of -0.57, nothing at all, while the share exceeding +50% went from
10.4% to 15.2%. An analyst reading means would have reported no effect.

Three features, each computed from the breakout bar and knowable at entry:

* **thrust** -- how far the bar pushed beyond the channel, in units of N. A
  decisive break and a marginal one are different events wearing one label.
* **close location** -- where the bar closed within its own range. Closing on
  the high is a different bar from closing on the low having spiked through.
* **relative volume** -- the bar's volume against its trailing mean, now that
  volume is consolidated tape rather than the IEX sample that ranked bars only
  0.625 correlated with the truth.

Each is bucketed into quintiles and reported by mean R, median R, and the share
of trades exceeding +3R and stopping at -1R. The top-minus-bottom quintile
spread is then tested against a null that shuffles the feature labels across the
same trades, which holds the outcome distribution fixed and destroys only the
association -- the same construction chapter six used on the calendar.
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
    TurtleConfig, relative_volume, rolling_extremes, run_turtle, wilder_atr,
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
    parser.add_argument("--volume-window", type=int, default=20)
    parser.add_argument("--nulls", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/breakout_quality.json"))
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


def features(bars: list[Bar], args) -> dict[str, dict[str, float]]:
    """Breakout-bar characteristics, keyed by timestamp."""
    highs = rolling_extremes([b.high for b in bars], FIXED["entry_window"], True)
    atr = wilder_atr(bars, FIXED["atr_window"])
    rel = relative_volume(bars, args.volume_window)
    out: dict[str, dict[str, float]] = {}
    for index, bar in enumerate(bars):
        level = highs[index]
        if math.isnan(level) or bar.high <= level:
            continue
        n = atr[index]
        if not n or math.isnan(n) or n <= 0:
            continue
        span = bar.high - bar.low
        out[bar.timestamp] = {
            "thrust": (bar.high - level) / n,
            "close_location": (bar.close - bar.low) / span if span > 0 else 0.5,
            "relative_volume": rel[index] if not math.isnan(rel[index]) else float("nan"),
        }
    return out


def quintile_table(pairs: list[tuple[float, float]]) -> list[dict]:
    """pairs: (feature, net_r). Sorted into equal-count buckets."""
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
            "feature_low": block[0][0],
            "feature_high": block[-1][0],
            "trades": len(values),
            "mean_r": statistics.fmean(values),
            "median_r": statistics.median(values),
            "share_above_3r": sum(1 for v in values if v > 3.0) / len(values),
            "share_stopped": sum(1 for v in values if v <= -0.99) / len(values),
        })
    return rows


def spread(pairs, key="mean_r"):
    rows = quintile_table(pairs)
    return rows[-1][key] - rows[0][key]


def main(argv=None) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)
    book = load(args)
    config = TurtleConfig(**FIXED, directions=(1,),
                          round_trip_cost=args.cost_bp / 10_000)

    records: list[dict] = []
    for ticker, bars in book.items():
        table = features(bars, args)
        trades, _ = run_turtle(bars, config=config)
        for trade in trades:
            row = table.get(trade.entry_timestamp)
            if row:
                records.append({**row, "net_r": trade.net_r})
    print(f"{len(book)} instruments, {len(records):,} breakouts with features, "
          f"{args.cost_bp:g}bp\n")

    report = {}
    for feature in ("thrust", "close_location", "relative_volume"):
        pairs = [(r[feature], r["net_r"]) for r in records
                 if not math.isnan(r[feature])]
        if len(pairs) < 500:
            continue
        rows = quintile_table(pairs)
        print(f"=== {feature} ({len(pairs):,} breakouts) ===")
        print(f"  {'quintile':>8s} {'range':>18s} {'trades':>8s} {'mean R':>8s} "
              f"{'median R':>9s} {'>+3R':>7s} {'stopped':>8s}")
        for row in rows:
            print(f"  {row['quintile']:>8d} "
                  f"{('%.2f-%.2f' % (row['feature_low'], row['feature_high'])):>18s} "
                  f"{row['trades']:>8,d} {row['mean_r']:>8.3f} "
                  f"{row['median_r']:>9.3f} {row['share_above_3r']:>7.1%} "
                  f"{row['share_stopped']:>8.1%}")

        # Null: keep the outcome distribution, destroy the association.
        observed_mean = rows[-1]["mean_r"] - rows[0]["mean_r"]
        observed_tail = rows[-1]["share_above_3r"] - rows[0]["share_above_3r"]
        outcomes = [r for _, r in pairs]
        keys = [f for f, _ in pairs]
        draws_mean, draws_tail = [], []
        for _ in range(args.nulls):
            shuffled = outcomes[:]
            rng.shuffle(shuffled)
            null_rows = quintile_table(list(zip(keys, shuffled)))
            draws_mean.append(null_rows[-1]["mean_r"] - null_rows[0]["mean_r"])
            draws_tail.append(
                null_rows[-1]["share_above_3r"] - null_rows[0]["share_above_3r"])
        p_mean = sum(1 for d in draws_mean if abs(d) >= abs(observed_mean)) / len(draws_mean)
        p_tail = sum(1 for d in draws_tail if abs(d) >= abs(observed_tail)) / len(draws_tail)
        print(f"  top-minus-bottom mean R  {observed_mean:+.3f}  p={p_mean:.3f}")
        print(f"  top-minus-bottom >+3R    {observed_tail:+.1%}  p={p_tail:.3f}\n",
              flush=True)
        report[feature] = {"quintiles": rows, "spread_mean_r": observed_mean,
                           "p_mean": p_mean, "spread_tail": observed_tail,
                           "p_tail": p_tail, "trades": len(pairs)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
