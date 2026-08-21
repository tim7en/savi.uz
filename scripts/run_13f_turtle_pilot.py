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
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import TurtleConfig, TurtleTrade, rolling_extremes, run_turtle  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--holdings", type=Path, required=True,
                        help="Raw holdings_major.json with exact filing dates")
    parser.add_argument("--matched-holdings", type=Path, required=True,
                        help="h13f.pkl, retaining raw row indexes and mapped tickers")
    parser.add_argument("--book", type=Path, required=True,
                        help="book13f.pkl with is_new and position weights")
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--split", default="2023-01-01",
                        help="First entry date in the validation period")
    parser.add_argument("--end", default="2026-08-19")
    parser.add_argument("--minimum-weight", type=float, default=0.05)
    parser.add_argument("--eligibility-days", type=int, default=365)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--risk-per-r", type=float, default=0.0020)
    parser.add_argument("--trials", type=int, default=200,
                        help="Randomized same-timestamp capacity orderings")
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/13f_turtle_pilot.json"))
    return parser.parse_args(argv)


def watchlist(args: argparse.Namespace, available: set[str]) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
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

    result: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = defaultdict(list)
    for row in signals.itertuples():
        if row.ticker in available:
            filed_date = pd.Timestamp(row.filed_date).normalize()
            result[row.ticker].append(
                (filed_date, filed_date + pd.Timedelta(days=args.eligibility_days))
            )
    return dict(result)


def load_daily_bars(path: Path, ticker: str, start: str, end: str,
                    splits: dict[str, list[tuple[str, float]]]) -> list[Bar]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars "
            "WHERE ticker=? AND frequency='5min' AND ts>=? AND ts<? ORDER BY ts",
            (ticker, start, end),
        ).fetchall()
    finally:
        connection.close()
    five_minute = adjust_bars([Bar(*row) for row in rows], splits.get(ticker, []))
    return resample_regular_session(five_minute, minutes=390)


def breakout_entries(bars: list[Bar], eligible: list[tuple[pd.Timestamp, pd.Timestamp]] | None) -> tuple[dict[int, int], dict[int, float]]:
    highs = rolling_extremes([bar.high for bar in bars], 55, True)
    entries: dict[int, int] = {}
    prices: dict[int, float] = {}
    for index, (bar, level) in enumerate(zip(bars, highs)):
        if math.isnan(level) or bar.high <= level:
            continue
        day = pd.Timestamp(bar.timestamp[:10])
        if eligible is not None and not any(filed < day <= expiry for filed, expiry in eligible):
            continue
        entries[index] = 1
        # Match the engine's long stop-order convention: channel edge or a gap-up open.
        prices[index] = max(level, bar.open)
    return entries, prices


def collect_trades(args: argparse.Namespace, watch: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]) -> tuple[dict[str, list[TurtleTrade]], dict[str, int]]:
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    try:
        available = {
            row[0] for row in connection.execute(
                "SELECT DISTINCT ticker FROM bars WHERE frequency='5min'"
            )
        }
    finally:
        connection.close()
    splits = load_splits(args.bars)
    config = TurtleConfig(
        entry_window=55, exit_window=20, atr_window=20, directions=(1,),
        skip_after_winner=False,
    )
    books: dict[str, list[TurtleTrade]] = {"baseline": [], "13f_watchlist": []}
    candidates = {"baseline": 0, "13f_watchlist": 0}
    for ticker in sorted(watch.keys() & available):
        bars = load_daily_bars(args.bars, ticker, args.start, args.end, splits)
        baseline_entries, baseline_prices = breakout_entries(bars, None)
        watch_entries, watch_prices = breakout_entries(bars, watch[ticker])
        candidates["baseline"] += len(baseline_entries)
        candidates["13f_watchlist"] += len(watch_entries)
        books["baseline"].extend(run_turtle(
            bars, config=config, entries=baseline_entries, entry_prices=baseline_prices
        )[0])
        books["13f_watchlist"].extend(run_turtle(
            bars, config=config, entries=watch_entries, entry_prices=watch_prices
        )[0])
    return books, candidates


def apply_cap(trades: list[TurtleTrade], maximum: int, seed: int) -> tuple[list[TurtleTrade], int]:
    shuffled = list(trades)
    random.Random(seed).shuffle(shuffled)
    live: list[TurtleTrade] = []
    selected: list[TurtleTrade] = []
    refused = 0
    for trade in sorted(shuffled, key=lambda item: item.entry_timestamp):
        live = [item for item in live if item.exit_timestamp > trade.entry_timestamp]
        if len(live) >= maximum:
            refused += 1
            continue
        live.append(trade)
        selected.append(trade)
    return selected, refused


def metrics(trades: list[TurtleTrade], risk_per_r: float) -> dict[str, float | int | None]:
    if not trades:
        return {"trades": 0}
    values = [trade.net_r for trade in trades]
    wins = [value for value in values if value > 0]
    losses = [value for value in values if value <= 0]
    by_exit: dict[str, float] = defaultdict(float)
    for trade in trades:
        by_exit[trade.exit_timestamp[:10]] += trade.net_r
    equity = peak = 1.0
    maximum_drawdown = 0.0
    for _, total_r in sorted(by_exit.items()):
        equity *= max(0.0, 1.0 + risk_per_r * total_r)
        peak = max(peak, equity)
        maximum_drawdown = min(maximum_drawdown, equity / peak - 1.0)
    first = date.fromisoformat(min(trade.entry_timestamp[:10] for trade in trades))
    last = date.fromisoformat(max(trade.exit_timestamp[:10] for trade in trades))
    years = max((last - first).days / 365.25, 1 / 365.25)
    return {
        "trades": len(trades),
        "total_r": sum(values),
        "mean_r": statistics.mean(values),
        "median_r": statistics.median(values),
        "win_rate": len(wins) / len(values),
        "profit_factor": sum(wins) / -sum(losses) if losses and sum(losses) else None,
        "ending_multiple": equity,
        "cagr": equity ** (1 / years) - 1,
        "max_drawdown_exit_marked": maximum_drawdown,
    }


def summary(trades: list[TurtleTrade], args: argparse.Namespace, start: str) -> dict[str, object]:
    candidates = [trade for trade in trades if trade.entry_timestamp[:10] >= start]
    trials = []
    for seed in range(args.trials):
        selected, refused = apply_cap(candidates, args.max_positions, seed)
        trials.append((metrics(selected, args.risk_per_r), refused))
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
                "SELECT DISTINCT ticker FROM bars WHERE frequency='5min'"
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
            "exit": "20-day channel, 2N stop, and half-N pyramiding to four units",
            "position_cap": args.max_positions,
            "risk_per_r": args.risk_per_r,
            "warning": "Narrow overlap pilot only; it does not validate the full 13F strategy.",
        },
        "breakout_candidates_before_trade_lifecycle": candidates,
        "full": {name: summary(trades, args, args.start) for name, trades in books.items()},
        "validation_2023_plus": {name: summary(trades, args, args.split) for name, trades in books.items()},
        "watchlist_windows": {
            ticker: [{"filed": filed.date().isoformat(), "expires": expiry.date().isoformat()}
                     for filed, expiry in windows]
            for ticker, windows in sorted(watch.items())
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())