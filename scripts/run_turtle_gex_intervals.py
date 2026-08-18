"""Leakage-audited prior-day GEX overlays across Turtle bar intervals."""

from __future__ import annotations

import argparse
import csv
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
from run_turtle_gex_macro import ETF_TICKERS  # noqa: E402
from run_turtle_gex_overlay import load_gex_features, safe_median  # noqa: E402
from run_turtle_leverage_stability import funding_map, load_bars, tickers  # noqa: E402
from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import TurtleConfig, run_turtle  # noqa: E402


INTERVALS = (("15m", 15), ("30m", 30), ("1h", 60), ("2h", 120), ("4h", 240))


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
    parser.add_argument("--outdir", type=Path, default=Path("out/strategy"))
    return parser.parse_args(argv)


def gex_size_function(available):
    def size(variant, trade):
        if variant == "No overlay":
            return 1.0
        day = trade.entry[:10]
        spy = available.get(("SPY", day))
        qqq = available.get(("QQQ", day))
        if spy is None or qqq is None:
            return 1.0
        # This assertion is the study's leakage tripwire. Same-day GEX must
        # never approve or size a trade from that same session.
        assert spy["observation_date"] < day, (spy["observation_date"], day)
        assert qqq["observation_date"] < day, (qqq["observation_date"], day)
        return 0.5 if spy["net_gex"] < 0 and qqq["net_gex"] < 0 else 1.0
    return size


def main(argv=None):
    args = parse_args(argv)
    _rows, available, valid_dates = load_gex_features(
        args.gex, args.start, args.end, window=60, minimum=20,
    )
    calendar = sorted(valid_dates["SPY"] & valid_dates["QQQ"])
    # Independently verify every feature map key: the value must be from an
    # earlier observation date, including across weekends and holidays.
    for (symbol, usable_day), feature in available.items():
        if not feature["observation_date"] < usable_day:
            raise RuntimeError(
                f"look-ahead join rejected: {symbol} {feature['observation_date']} -> {usable_day}"
            )

    splits = load_splits(args.bars)
    requested = [ticker for ticker in tickers(args.bars) if ticker not in ETF_TICKERS]
    raw_by_ticker = {}
    closes = defaultdict(dict)
    all_timestamps = set()
    for ticker in requested:
        raw = adjust_bars(
            load_bars(args.bars, ticker, args.start, args.end), splits.get(ticker, [])
        )
        raw = [bar for bar in raw if bar.timestamp[:10] in valid_dates["SPY"]]
        if not raw:
            continue
        raw_by_ticker[ticker] = raw
        for bar in raw:
            closes[bar.timestamp][ticker] = bar.close
            all_timestamps.add(bar.timestamp)

    timeline = sorted(all_timestamps)
    timeline_index = {stamp: index for index, stamp in enumerate(timeline)}
    daily_indices = np.asarray([
        index for index, stamp in enumerate(timeline)
        if index + 1 == len(timeline) or timeline[index + 1][:10] != stamp[:10]
    ], dtype=int)
    macro = sqlite3.connect(f"file:{args.macro}?mode=ro", uri=True)
    funding = funding_map(calendar, macro)
    macro.close()
    risk_size = gex_size_function(available)

    config = TurtleConfig(
        entry_window=55, exit_window=20, atr_window=20,
        skip_after_winner=False, directions=(1,), use_channel_exit=False,
        chandelier_atr=5.0,
    )
    summaries = {}
    paired_rows = []
    audit_rows = []
    for label, minutes in INTERVALS:
        trades = []
        ticker_counts = {}
        for ticker, raw in raw_by_ticker.items():
            blocks = raw_blocks(raw, minutes=minutes)
            regular = [bar for rows in blocks.values() for bar in rows]
            regular_index = {bar.timestamp: index for index, bar in enumerate(regular)}
            interval_bars = resample_regular_session(raw, minutes=minutes)
            ticker_trades, _audit = run_turtle(interval_bars, config=config)
            converted = [path_trade(
                ticker, trade, blocks, regular, regular_index, config.round_trip_cost,
            ) for trade in ticker_trades if trade.unit_entries and trade.n_at_entry > 0]
            trades.extend(converted)
            ticker_counts[ticker] = len(converted)
        prepared = {
            trade: (
                np.asarray([timeline_index[stamp] for stamp in trade.mark_timestamps], dtype=int),
                np.asarray(trade.mark_rs, dtype=float),
            ) for trade in trades
        }
        joined = flagged = 0
        for trade in trades:
            day = trade.entry[:10]
            spy = available.get(("SPY", day))
            qqq = available.get(("QQQ", day))
            if spy is not None and qqq is not None:
                joined += 1
                assert spy["observation_date"] < day
                assert qqq["observation_date"] < day
                warning = spy["net_gex"] < 0 and qqq["net_gex"] < 0
                flagged += warning
                audit_rows.append({
                    "interval": label, "ticker": trade.ticker,
                    "trade_entry": trade.entry, "trade_date": day,
                    "spy_gex_date": spy["observation_date"],
                    "qqq_gex_date": qqq["observation_date"],
                    "spy_net_gex": spy["net_gex"], "qqq_net_gex": qqq["net_gex"],
                    "both_negative": int(warning),
                })
        print(
            f"{label}: {len(trades)} candidates, prior-GEX join {joined}/{len(trades)}, "
            f"{flagged} flagged", flush=True,
        )
        results = {}
        for variant in ("No overlay", "Both-negative GEX half-risk"):
            results[variant] = [replay(
                trades, variant, {}, funding, timeline, closes, prepared, daily_indices,
                seed=seed, max_positions=args.max_positions,
                broker_spread=args.broker_spread, size_fn=risk_size,
                post_capacity_size=True,
            ) for seed in range(args.draws)]
        base_summary = median_summary(results["No overlay"])
        gex_summary = median_summary(results["Both-negative GEX half-risk"])
        rows = []
        for seed, (base, gex) in enumerate(zip(
            results["No overlay"], results["Both-negative GEX half-risk"]
        )):
            row = {
                "interval": label, "seed": seed,
                "delta_nav": gex["ending"] - base["ending"],
                "delta_cagr_pp": 100.0 * (gex["cagr"] - base["cagr"]),
                "delta_dd_pp": 100.0 * (gex["maxdd"] - base["maxdd"]),
                "delta_sharpe": gex["sharpe"] - base["sharpe"],
            }
            rows.append(row)
            paired_rows.append(row)
        summaries[label] = {
            "candidates": len(trades), "joined": joined, "flagged": flagged,
            "base": base_summary, "gex": gex_summary,
            "nav_wins": sum(row["delta_nav"] > 0 for row in rows),
            "dd_wins": sum(row["delta_dd_pp"] > 0 for row in rows),
            "delta_nav": statistics.median(row["delta_nav"] for row in rows),
            "delta_cagr_pp": statistics.median(row["delta_cagr_pp"] for row in rows),
            "delta_dd_pp": statistics.median(row["delta_dd_pp"] for row in rows),
            "delta_sharpe": safe_median([row["delta_sharpe"] for row in rows]),
        }
        print(
            f"{label}: baseline ${base_summary['ending']:.2f}, "
            f"GEX ${gex_summary['ending']:.2f}", flush=True,
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    paired_csv = args.outdir / "turtle_gex_intervals_paired.csv"
    with paired_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    audit_csv = args.outdir / "turtle_gex_intervals_leakage_audit.csv"
    with audit_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    report = args.outdir / "turtle_gex_intervals.md"
    lines = [
        "# Prior-day GEX across Turtle bar intervals", "",
        f"{len(raw_by_ticker)} stocks, {len(calendar)} sessions, {args.draws} paired capacity "
        "orderings. Starting NAV $1,000 ($700 trading + $300 sleeve), six positions and "
        "2x gross cap. All intervals use the same 55-bar long breakout and 5N chandelier.", "",
        "The overlay halves actual post-cap risk for a new entry only when both SPY and QQQ "
        "net GEX from the previous valid session are negative. The 4h series uses regular-"
        "session 240-minute blocks; the final block is the remaining 150 minutes.", "",
        "| Interval | Candidates | Flagged | Baseline NAV | GEX NAV | NAV wins | Baseline CAGR | GEX CAGR | Delta CAGR | Baseline DD | GEX DD | Delta DD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, _minutes in INTERVALS:
        item = summaries[label]
        base, gex = item["base"], item["gex"]
        lines.append(
            f"| {label} | {item['candidates']} | {item['flagged']} | "
            f"${base['ending']:,.2f} | ${gex['ending']:,.2f} | "
            f"{item['nav_wins']}/{args.draws} | {base['cagr']:+.2%} | "
            f"{gex['cagr']:+.2%} | {item['delta_cagr_pp']:+.2f} pp | "
            f"{base['maxdd']:.1%} | {gex['maxdd']:.1%} | "
            f"{item['delta_dd_pp']:+.2f} pp |"
        )
    lines += ["", "## Leakage audit", "",
              f"- {len(audit_rows):,} trade-feature joins were written to the audit file.",
              "- Runtime assertions require `gex_observation_date < trade_date` for SPY and QQQ.",
              "- Same-day GEX is structurally absent from the feature map; weekends and holidays map to the next valid session.",
              "- Positive delta DD means the overlay reduced the magnitude of drawdown.",
              "- The one-year GEX history is exploratory; the 100 seeds vary capacity ordering, not market history."]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {report}, {paired_csv}, {audit_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
