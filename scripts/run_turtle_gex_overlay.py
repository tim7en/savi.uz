"""Paired prior-day GEX overlays on SPY/QQQ 30-minute Turtle trades.

Daily option features are computed after the observation-day close. A feature
dated D is therefore mapped only to the next valid trading session; the GEX
percentile is a trailing, not full-sample, rank. Turtle signals and fills keep
the existing five-minute path-aware replay and portfolio accounting.
"""

from __future__ import annotations

import argparse
import csv
import math
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_turtle_30m_path_projection import (  # noqa: E402
    median_summary, path_trade, raw_blocks, replay,
)
from run_turtle_leverage_stability import funding_map, load_bars  # noqa: E402
from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import TurtleConfig, run_turtle  # noqa: E402


VARIANTS = (
    "No overlay",
    "High GEX percentile half-risk",
    "Positive net GEX half-risk",
    "Negative net GEX half-risk",
    "Gamma wall overhead half-risk",
    "High GEX or wall overhead half-risk",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--gex", type=Path, default=Path("data/options/marketdata.db"))
    parser.add_argument("--macro", type=Path, default=Path("data/macro/macro.db"))
    parser.add_argument("--start", default="2025-08-18")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--broker-spread", type=float, default=0.015)
    parser.add_argument("--percentile-window", type=int, default=60)
    parser.add_argument("--min-percentile-history", type=int, default=20)
    parser.add_argument("--high-percentile", type=float, default=80.0)
    parser.add_argument("--outdir", type=Path, default=Path("out/strategy"))
    return parser.parse_args(argv)


def trailing_percentile(values: list[float], index: int, window: int,
                        minimum: int) -> float | None:
    sample = values[max(0, index - window + 1):index + 1]
    if len(sample) < minimum:
        return None
    current = values[index]
    below = sum(value < current for value in sample)
    equal = sum(value == current for value in sample)
    return 100.0 * (below + 0.5 * equal) / len(sample)


def load_gex_features(path: Path, start: str, end: str, *, window: int,
                      minimum: int):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    columns = (
        "symbol", "observation_date", "contracts", "usable_contracts",
        "underlying_price", "call_gex", "put_gex", "net_gex", "absolute_gex",
        "gamma_wall_strike", "distance_to_gamma_wall", "gamma_flip_proxy",
        "oi_weighted_iv", "put_call_oi",
    )
    rows = [dict(zip(columns, row)) for row in connection.execute(
        "SELECT " + ",".join(columns) + " FROM daily_gex "
        "WHERE observation_date>=? AND observation_date<=? "
        "ORDER BY symbol,observation_date", (start, end)
    )]
    put_by_day: dict[tuple[str, str], tuple[float, float]] = {}
    for symbol, day, strike, exposure in connection.execute(
        "SELECT symbol,observation_date,strike,"
        "SUM(gamma*open_interest*100.0*underlying_price*underlying_price*0.01) "
        "FROM option_contracts WHERE observation_date>=? AND observation_date<=? "
        "AND side='put' AND gamma IS NOT NULL AND open_interest IS NOT NULL "
        "GROUP BY symbol,observation_date,strike", (start, end)
    ):
        key = (symbol, day)
        if key not in put_by_day or exposure > put_by_day[key][1]:
            put_by_day[key] = (strike, exposure)
    connection.close()

    by_symbol = defaultdict(list)
    for row in rows:
        by_symbol[row["symbol"]].append(row)
    available = {}
    enriched = []
    for symbol, symbol_rows in sorted(by_symbol.items()):
        values = [float(row["net_gex"]) for row in symbol_rows]
        for index, row in enumerate(symbol_rows):
            row["put_wall_strike"] = put_by_day.get(
                (symbol, row["observation_date"]), (None, None)
            )[0]
            row["gex_percentile"] = trailing_percentile(
                values, index, window, minimum
            )
            row["percentile_window"] = min(index + 1, window)
            row["available_on_session"] = (
                symbol_rows[index + 1]["observation_date"]
                if index + 1 < len(symbol_rows) else None
            )
            enriched.append(row)
            if index + 1 < len(symbol_rows):
                available[(symbol, symbol_rows[index + 1]["observation_date"])] = row
    return enriched, available, {
        symbol: {row["observation_date"] for row in symbol_rows}
        for symbol, symbol_rows in by_symbol.items()
    }


def write_features(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def size_function(features, high_percentile):
    def size(variant, trade):
        feature = features.get((trade.ticker, trade.entry[:10]))
        if variant == "No overlay" or feature is None:
            return 1.0
        percentile = feature["gex_percentile"]
        high = percentile is not None and percentile >= high_percentile
        positive = feature["net_gex"] > 0
        negative = feature["net_gex"] < 0
        wall_overhead = feature["distance_to_gamma_wall"] < 0
        flags = {
            "High GEX percentile half-risk": high,
            "Positive net GEX half-risk": positive,
            "Negative net GEX half-risk": negative,
            "Gamma wall overhead half-risk": wall_overhead,
            "High GEX or wall overhead half-risk": high or wall_overhead,
        }
        return 0.5 if flags[variant] else 1.0
    return size


def safe_median(values):
    finite = [value for value in values if math.isfinite(value)]
    return statistics.median(finite) if finite else math.nan


def main(argv=None):
    args = parse_args(argv)
    enriched, available, valid_dates = load_gex_features(
        args.gex, args.start, args.end, window=args.percentile_window,
        minimum=args.min_percentile_history,
    )
    if set(valid_dates) != {"SPY", "QQQ"}:
        raise RuntimeError("GEX database must contain SPY and QQQ")
    features_csv = args.outdir.parent / "options" / "gex_daily_enriched.csv"
    write_features(features_csv, enriched)

    splits = load_splits(args.bars)
    closes = defaultdict(dict)
    all_timestamps = set()
    trades = []
    ticker_counts = {}
    config = TurtleConfig(
        entry_window=55, exit_window=20, atr_window=20,
        skip_after_winner=False, directions=(1,), use_channel_exit=False,
        chandelier_atr=5.0,
    )
    for ticker in ("SPY", "QQQ"):
        raw = adjust_bars(
            load_bars(args.bars, ticker, args.start, args.end), splits.get(ticker, [])
        )
        raw = [bar for bar in raw if bar.timestamp[:10] in valid_dates[ticker]]
        blocks = raw_blocks(raw)
        regular = [bar for rows in blocks.values() for bar in rows]
        regular_index = {bar.timestamp: index for index, bar in enumerate(regular)}
        thirty = resample_regular_session(raw, minutes=30)
        ticker_trades, _audit = run_turtle(thirty, config=config)
        converted = []
        for trade in ticker_trades:
            if trade.unit_entries and trade.n_at_entry > 0:
                converted.append(path_trade(
                    ticker, trade, blocks, regular, regular_index,
                    config.round_trip_cost,
                ))
        trades.extend(converted)
        ticker_counts[ticker] = len(converted)
        for rows in blocks.values():
            for bar in rows:
                closes[bar.timestamp][ticker] = bar.close
                all_timestamps.add(bar.timestamp)
        print(f"{ticker}: {len(converted)} candidate trades", flush=True)

    calendar = sorted(valid_dates["SPY"] & valid_dates["QQQ"])
    timeline = sorted(all_timestamps)
    timeline_index = {stamp: index for index, stamp in enumerate(timeline)}
    prepared = {
        trade: (
            np.asarray([timeline_index[stamp] for stamp in trade.mark_timestamps], dtype=int),
            np.asarray(trade.mark_rs, dtype=float),
        ) for trade in trades
    }
    daily_indices = np.asarray([
        index for index, stamp in enumerate(timeline)
        if index + 1 == len(timeline) or timeline[index + 1][:10] != stamp[:10]
    ], dtype=int)
    macro = sqlite3.connect(f"file:{args.macro}?mode=ro", uri=True)
    funding = funding_map(calendar, macro)
    macro.close()
    risk_size = size_function(available, args.high_percentile)
    joined = [
        (trade, available.get((trade.ticker, trade.entry[:10]))) for trade in trades
    ]
    print(
        f"prior-GEX join: {sum(feature is not None for _, feature in joined)}/"
        f"{len(joined)} candidate trades",
        flush=True,
    )
    for variant in VARIANTS[1:]:
        flagged = sum(
            feature is not None and risk_size(variant, trade) < 1.0
            for trade, feature in joined
        )
        print(f"{variant}: {flagged} candidates flagged", flush=True)
    sign_diagnostics = {}
    for label, predicate in (
        ("Prior net GEX positive", lambda feature: feature["net_gex"] > 0),
        ("Prior net GEX negative", lambda feature: feature["net_gex"] < 0),
    ):
        subset = [trade.net_r for trade, feature in joined
                  if feature is not None and predicate(feature)]
        sign_diagnostics[label] = {
            "trades": len(subset),
            "win_rate": sum(value > 0 for value in subset) / len(subset),
            "mean_r": statistics.mean(subset),
            "median_r": statistics.median(subset),
            "total_r": sum(subset),
        }

    results = {}
    summaries = {}
    for variant in VARIANTS:
        draws = [replay(
            trades, variant, {}, funding, timeline, closes, prepared, daily_indices,
            seed=seed, max_positions=args.max_positions,
            broker_spread=args.broker_spread, size_fn=risk_size,
            post_capacity_size=True,
        ) for seed in range(args.draws)]
        results[variant] = draws
        summaries[variant] = median_summary(draws)
        print(
            f"{variant}: median NAV {summaries[variant]['ending']:.2f}, "
            f"mean size {summaries[variant]['mean_size']:.3f}",
            flush=True,
        )

    baseline = results["No overlay"]
    paired_rows = []
    paired_summary = {}
    for variant in VARIANTS[1:]:
        rows = []
        for seed, (base, over) in enumerate(zip(baseline, results[variant])):
            row = {
                "variant": variant, "seed": seed,
                "delta_nav": over["ending"] - base["ending"],
                "delta_cagr_pp": 100.0 * (over["cagr"] - base["cagr"]),
                "delta_sharpe": over["sharpe"] - base["sharpe"],
            }
            rows.append(row)
            paired_rows.append(row)
        paired_summary[variant] = {
            "nav_wins": sum(row["delta_nav"] > 0 for row in rows),
            "delta_nav": statistics.median(row["delta_nav"] for row in rows),
            "delta_cagr_pp": statistics.median(row["delta_cagr_pp"] for row in rows),
            "delta_sharpe": safe_median([row["delta_sharpe"] for row in rows]),
        }

    args.outdir.mkdir(parents=True, exist_ok=True)
    paired_csv = args.outdir / "turtle_gex_paired.csv"
    with paired_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)

    report = args.outdir / "turtle_gex_overlay.md"
    lines = [
        "# Prior-day GEX overlay on 30-minute Turtle", "",
        f"SPY and QQQ, {args.start} through {args.end}; {len(calendar)} paired sessions, "
        f"{len(trades)} candidate trades ({ticker_counts['SPY']} SPY, "
        f"{ticker_counts['QQQ']} QQQ). Starting NAV $1,000 ($700 trading + $300 sleeve), "
        "1% risk per N, four-unit reservation, six-position capacity, 2x gross cap.", "",
        "All GEX inputs are prior-session EOD values. The percentile is a trailing "
        f"{args.percentile_window}-session within-symbol rank with at least "
        f"{args.min_percentile_history} observations. Synthetic holiday price bars are excluded.", "",
        "| Variant | Median end NAV | CAGR | Sharpe | Trades | NAV wins | Delta CAGR | Delta Sharpe |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        item = summaries[variant]
        if variant == "No overlay":
            wins, dcagr, dsharpe = "--", "--", "--"
        else:
            paired = paired_summary[variant]
            wins = f"{paired['nav_wins']}/{args.draws}"
            dcagr = f"{paired['delta_cagr_pp']:+.2f} pp"
            dsharpe = f"{paired['delta_sharpe']:+.3f}"
        lines.append(
            f"| {variant} | ${item['ending']:,.2f} | {item['cagr']:+.2%} | "
            f"{item['sharpe']:.3f} | {item['trades']:.0f} | {wins} | {dcagr} | {dsharpe} |"
        )
    lines += ["", "## Feature definitions", "",
              "- Call/put GEX = model gamma x open interest x 100 x spot squared x 1%; net GEX is call minus put.",
              "- Gamma wall is the strike with greatest absolute signed GEX; put wall is the strike with greatest put GEX.",
              "- Distance to gamma wall is (spot - wall) / spot; a negative value means the wall is overhead.",
              "- IV is open-interest-weighted midpoint-implied volatility. Calls are positive and puts negative, so this is a proxy rather than observed dealer inventory.",
              "- An overlay halves risk only for new entries; it never changes an already-open position.",
              "", "## Candidate trades by prior-day GEX sign", "",
              "| Prior-day regime | Candidate trades | Win rate | Mean R | Median R | Total R |",
              "|---|---:|---:|---:|---:|---:|"]
    for label, item in sign_diagnostics.items():
        lines.append(
            f"| {label} | {item['trades']} | {item['win_rate']:.1%} | "
            f"{item['mean_r']:+.3f} | {item['median_r']:+.3f} | {item['total_r']:+.2f} |"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {report}, {paired_csv}, {features_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
