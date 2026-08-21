"""Pilot a 13F watchlist gate on daily long-only Turtle breakouts.

This is deliberately a narrow test.  A position becomes eligible only on the
trading day after its public 13F filing, then remains eligible for one year.
The 13F does not create an entry: the same 55/20 Turtle rules decide that.

The script compares the gated book with every long-only 55-day breakout from
the same tickers, under the same six-position cap.  It uses only high-conviction
new positions (default: at least 5% of a manager's disclosed book).  Exact
filing dates come from the raw holdings export; the matched holdings pickle
maps those holdings to tickers.
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
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import TurtleConfig, TurtleTrade, rolling_extremes, run_turtle  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

DATA_DIR = Path("data/13f")


@dataclass(frozen=True)
class TaggedTrade:
    ticker: str
    trade: TurtleTrade
    conviction_weight: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=DATA_DIR / "alphavantage_daily.db")
    parser.add_argument("--frequency", choices=("5min", "daily"), default="daily",
                        help="Stored bar frequency; daily Alpha Vantage bars are already split-consistent")
    parser.add_argument("--holdings", type=Path, default=DATA_DIR / "holdings_major.json",
                        help="Raw holdings_major.json with exact filing dates")
    parser.add_argument("--matched-holdings", type=Path, default=DATA_DIR / "h13f.pkl",
                        help="h13f.pkl, retaining raw row indexes and mapped tickers")
    parser.add_argument("--book", type=Path, default=DATA_DIR / "book13f.pkl",
                        help="book13f.pkl with is_new and position weights")
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--split", default="2023-01-01",
                        help="First entry date in the validation period")
    parser.add_argument("--end", default="2026-08-19")
    parser.add_argument("--minimum-weight", type=float, default=0.05)
    parser.add_argument("--eligibility-days", type=int, default=365)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--risk-per-r", type=float, default=0.0020)
    parser.add_argument("--tilt-caps", type=float, nargs="+", default=(1.25, 1.50, 2.00),
                        help="Maximum risk multipliers for 10%+ conviction")
    parser.add_argument("--trials", type=int, default=200,
                        help="Randomized same-timestamp capacity orderings")
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/13f_turtle_pilot.json"))
    return parser.parse_args(argv)


def watchlist(args: argparse.Namespace, available: set[str]) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp, float]]]:
    raw = pd.DataFrame(json.loads(args.holdings.read_text(encoding="utf-8")))
    matched = pd.read_pickle(args.matched_holdings).copy()
    book = pd.read_pickle(args.book).copy()

    # h13f retains its original raw-row index after unmatched holdings are removed.
    matched["filed_date"] = pd.to_datetime(raw.loc[matched.index, "filed"].to_numpy())
    matched["period_key"] = matched["period"].astype(str)
    book["period_key"] = book["period"].astype(str)

    filed = (matched.groupby(["mgr", "ticker", "period_key"], as_index=False)
             .filed_date.min())
    signals = book.loc[book.is_new & (book.w >= args.minimum_weight)].merge(
        filed, on=["mgr", "ticker", "period_key"], how="left", validate="many_to_one"
    )
    if signals.filed_date.isna().any():
        raise ValueError("high-conviction positions are missing an exact filing date")

    result: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, float]]] = defaultdict(list)
    for row in signals.itertuples():
        if row.ticker in available:
            filed_date = pd.Timestamp(row.filed_date).normalize()
            result[row.ticker].append(
                (filed_date, filed_date + pd.Timedelta(days=args.eligibility_days), row.w)
            )
    return dict(result)


def load_daily_bars(path: Path, ticker: str, start: str, end: str, frequency: str,
                    splits: dict[str, list[tuple[str, float]]]) -> list[Bar]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars "
            "WHERE ticker=? AND frequency=? AND ts>=? AND ts<? ORDER BY ts",
            (ticker, frequency, start, end),
        ).fetchall()
    finally:
        connection.close()
    if frequency == "daily":
        return [Bar(*row) for row in rows]
    five_minute = adjust_bars([Bar(*row) for row in rows], splits.get(ticker, []))
    return resample_regular_session(five_minute, minutes=390)


def conviction_weight(day: pd.Timestamp,
                      eligible: list[tuple[pd.Timestamp, pd.Timestamp, float]]) -> float:
    """Largest currently-public manager weight; multiple managers reinforce it."""
    return max((weight for filed, expiry, weight in eligible if filed < day <= expiry),
               default=0.0)


def breakout_entries(bars: list[Bar], eligible: list[tuple[pd.Timestamp, pd.Timestamp, float]]) -> tuple[dict[int, int], dict[int, float], dict[str, float]]:
    highs = rolling_extremes([bar.high for bar in bars], 55, True)
    entries: dict[int, int] = {}
    prices: dict[int, float] = {}
    weights: dict[str, float] = {}
    for index, (bar, level) in enumerate(zip(bars, highs)):
        if math.isnan(level) or bar.high <= level:
            continue
        day = pd.Timestamp(bar.timestamp[:10])
        entries[index] = 1
        # Match the engine's long stop-order convention: channel edge or a gap-up open.
        prices[index] = max(level, bar.open)
        weights[bar.timestamp] = conviction_weight(day, eligible)
    return entries, prices, weights


def collect_trades(args: argparse.Namespace, watch: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, float]]]) -> tuple[dict[str, list[TaggedTrade]], dict[str, int]]:
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    try:
        available = {
            row[0] for row in connection.execute(
                "SELECT DISTINCT ticker FROM bars WHERE frequency=?", (args.frequency,)
            )
        }
    finally:
        connection.close()
    splits = load_splits(args.bars) if args.frequency == "5min" else {}
    config = TurtleConfig(
        entry_window=55, exit_window=20, atr_window=20, directions=(1,),
        skip_after_winner=False,
    )
    books: dict[str, list[TaggedTrade]] = {"baseline": [], "13f_watchlist": []}
    candidates = {"baseline": 0, "13f_watchlist": 0}
    for ticker in sorted(watch.keys() & available):
        bars = load_daily_bars(args.bars, ticker, args.start, args.end, args.frequency, splits)
        baseline_entries, baseline_prices, weights = breakout_entries(bars, watch[ticker])
        watch_entries = {index: side for index, side in baseline_entries.items()
                 if weights.get(bars[index].timestamp, 0.0) > 0}
        watch_prices = {index: baseline_prices[index] for index in watch_entries}
        candidates["baseline"] += len(baseline_entries)
        candidates["13f_watchlist"] += len(watch_entries)
        baseline = run_turtle(
            bars, config=config, entries=baseline_entries, entry_prices=baseline_prices
        )[0]
        gated = run_turtle(
            bars, config=config, entries=watch_entries, entry_prices=watch_prices
        )[0]
        books["baseline"].extend(
            TaggedTrade(ticker, trade, weights.get(trade.entry_timestamp, 0.0))
            for trade in baseline
        )
        books["13f_watchlist"].extend(
            TaggedTrade(ticker, trade, weights.get(trade.entry_timestamp, 0.0))
            for trade in gated
        )
    return books, candidates


def apply_cap(trades: list[TaggedTrade], maximum: int, seed: int,
              prioritize_conviction: bool = False) -> tuple[list[TaggedTrade], int]:
    shuffled = list(trades)
    random.Random(seed).shuffle(shuffled)
    live: list[TaggedTrade] = []
    selected: list[TaggedTrade] = []
    refused = 0
    ordered = sorted(
        shuffled,
        key=(lambda item: (item.trade.entry_timestamp, -item.conviction_weight)
             if prioritize_conviction else item.trade.entry_timestamp),
    )
    for trade in ordered:
        live = [item for item in live if item.trade.exit_timestamp > trade.trade.entry_timestamp]
        if len(live) >= maximum:
            refused += 1
            continue
        live.append(trade)
        selected.append(trade)
    return selected, refused


def metrics(trades: list[TaggedTrade], risk_per_r: float, tilt_cap: float) -> dict[str, float | int | None]:
    if not trades:
        return {"trades": 0}
    def multiplier(item: TaggedTrade) -> float:
        # A 5% disclosed position gets half the available tilt; 10%+ gets the cap.
        return 1.0 + (tilt_cap - 1.0) * min(item.conviction_weight / 0.10, 1.0)

    values = [item.trade.net_r * multiplier(item) for item in trades]
    wins = [value for value in values if value > 0]
    losses = [value for value in values if value <= 0]
    by_exit: dict[str, float] = defaultdict(float)
    for item in trades:
        by_exit[item.trade.exit_timestamp[:10]] += item.trade.net_r * multiplier(item)
    equity = peak = 1.0
    maximum_drawdown = 0.0
    for _, total_r in sorted(by_exit.items()):
        equity *= max(0.0, 1.0 + risk_per_r * total_r)
        peak = max(peak, equity)
        maximum_drawdown = min(maximum_drawdown, equity / peak - 1.0)
    first = date.fromisoformat(min(item.trade.entry_timestamp[:10] for item in trades))
    last = date.fromisoformat(max(item.trade.exit_timestamp[:10] for item in trades))
    years = max((last - first).days / 365.25, 1 / 365.25)
    return {
        "trades": len(trades),
        "total_r": sum(values),
        "mean_r": statistics.mean(values),
        "median_r": statistics.median(values),
        "win_rate": len(wins) / len(values),
        "profit_factor": sum(wins) / -sum(losses) if losses and sum(losses) else None,
        "mean_risk_multiplier": statistics.mean(multiplier(item) for item in trades),
        "ending_multiple": equity,
        "cagr": equity ** (1 / years) - 1,
        "max_drawdown_exit_marked": maximum_drawdown,
    }


def summary(trades: list[TaggedTrade], args: argparse.Namespace, start: str,
            tilt_cap: float = 1.0, prioritize_conviction: bool = False) -> dict[str, object]:
    candidates = [item for item in trades if item.trade.entry_timestamp[:10] >= start]
    trials = []
    for seed in range(args.trials):
        selected, refused = apply_cap(
            candidates, args.max_positions, seed,
            prioritize_conviction=prioritize_conviction,
        )
        trials.append((metrics(selected, args.risk_per_r, tilt_cap), refused))
    metric_keys = sorted({key for result, _ in trials for key in result})
    result: dict[str, object] = {
        "signals_before_cap": len(candidates),
        "period_start": start,
        "capacity_trials": args.trials,
    }
    for key in metric_keys:
        values = sorted(item[key] for item, _ in trials if item.get(key) is not None)
        if values:
            result[key] = {
                "p05": values[int(.05 * len(values))],
                "median": values[int(.50 * len(values))],
                "p95": values[int(.95 * len(values))],
            }
    refusals = sorted(item for _, item in trials)
    result["rejected_book_full"] = {
        "p05": refusals[int(.05 * len(refusals))],
        "median": refusals[int(.50 * len(refusals))],
        "p95": refusals[int(.95 * len(refusals))],
    }
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    try:
        available = {
            row[0] for row in connection.execute(
                "SELECT DISTINCT ticker FROM bars WHERE frequency=?", (args.frequency,)
            )
        }
    finally:
        connection.close()
    watch = watchlist(args, available)
    if not watch:
        raise SystemExit("no high-conviction 13F tickers overlap the price store")
    books, candidates = collect_trades(args, watch)
    output = {
        "scope": {
            "tickers": sorted(watch),
            "ticker_count": len(watch),
            "minimum_weight": args.minimum_weight,
            "eligibility_days": args.eligibility_days,
            "entry": "long-only daily 55-day channel breakout after the public filing",
            "bar_source_frequency": args.frequency,
            "exit": "20-day channel, 2N stop, and half-N pyramiding to four units",
            "position_cap": args.max_positions,
            "risk_per_r": args.risk_per_r,
            "warning": "Narrow overlap pilot only; it does not validate the full 13F strategy.",
        },
        "breakout_candidates_before_trade_lifecycle": candidates,
        "full": {name: summary(trades, args, args.start) for name, trades in books.items()},
        "validation_2023_plus": {name: summary(trades, args, args.split) for name, trades in books.items()},
        "conviction_tilt": {
            f"max_{cap:.2f}x": {
                "full": summary(books["baseline"], args, args.start, cap),
                "validation_2023_plus": summary(books["baseline"], args, args.split, cap),
            }
            for cap in args.tilt_caps
        },
        "conviction_priority": {
            "full": summary(
                books["baseline"], args, args.start, prioritize_conviction=True,
            ),
            "validation_2023_plus": summary(
                books["baseline"], args, args.split, prioritize_conviction=True,
            ),
        },
        "watchlist_windows": {
            ticker: [{"filed": filed.date().isoformat(), "expires": expiry.date().isoformat(),
                      "weight": weight}
                     for filed, expiry, weight in windows]
            for ticker, windows in sorted(watch.items())
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())