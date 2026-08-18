"""Test a $100 leveraged index short during prior-day negative-GEX regimes."""

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
    median_summary, metrics, path_trade, raw_blocks, replay,
)
from run_turtle_gex_macro import ETF_TICKERS  # noqa: E402
from run_turtle_gex_overlay import load_gex_features, safe_median  # noqa: E402
from run_turtle_leverage_stability import funding_map, load_bars, tickers  # noqa: E402
from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import TurtleConfig, run_turtle  # noqa: E402


LEVERAGES = (1, 2, 5, 10, 20)


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
    parser.add_argument("--round-trip-cost", type=float, default=0.0002)
    parser.add_argument("--outdir", type=Path, default=Path("out/strategy"))
    return parser.parse_args(argv)


def hedge_path(timeline, bars, signal, funding, *, leverage, initial=100.0,
               round_trip_cost=0.0002, broker_spread=0.015):
    equity = initial
    units = 0.0
    last_price = None
    prior_day = None
    liquidated = False
    entries = 0
    path = []
    for stamp in timeline:
        bar = bars[stamp]
        day = stamp[:10]
        new_day = day != prior_day
        desired = signal.get(day, False)
        if new_day:
            if units:
                equity += units * (last_price - bar.open)
                last_price = bar.open
                rate = (funding.get(day) or 0.0) / 100.0 + broker_spread
                equity -= abs(units * bar.open) * rate / 252.0
                if equity <= 0:
                    equity, units, liquidated = 0.0, 0.0, True
            if units and not desired:
                equity -= abs(units * bar.open) * round_trip_cost / 2.0
                equity = max(equity, 0.0)
                units = 0.0
            if not units and desired and equity > 0:
                units = equity * leverage / bar.open
                equity -= abs(units * bar.open) * round_trip_cost / 2.0
                equity = max(equity, 0.0)
                last_price = bar.open
                entries += 1
            prior_day = day
        if units:
            adverse_equity = equity + units * (last_price - bar.high)
            if adverse_equity <= 0:
                equity, units, liquidated = 0.0, 0.0, True
            else:
                equity += units * (last_price - bar.close)
                last_price = bar.close
                if equity <= 0:
                    equity, units, liquidated = 0.0, 0.0, True
        path.append(equity)
    return np.asarray(path, dtype=float), {"liquidated": liquidated, "entries": entries}


def standalone_metrics(path, daily_indices):
    daily = path[daily_indices]
    peak = np.maximum.accumulate(daily)
    maxdd = float(np.min(np.divide(
        daily, peak, out=np.ones_like(daily), where=peak > 0
    ) - 1.0))
    returns = [daily[index] / daily[index - 1] - 1.0
               for index in range(1, len(daily)) if daily[index - 1] > 0]
    vol = statistics.stdev(returns) if len(returns) > 1 else 0.0
    sharpe = statistics.mean(returns) / vol * math.sqrt(252.0) if vol else math.nan
    years = max((len(daily) - 1) / 252.0, 1 / 252.0)
    cagr = -1.0 if daily[-1] <= 0 else (daily[-1] / daily[0]) ** (1 / years) - 1.0
    return {"ending": daily[-1], "cagr": cagr, "maxdd": maxdd, "sharpe": sharpe}


def main(argv=None):
    args = parse_args(argv)
    _rows, gex_available, valid_dates = load_gex_features(
        args.gex, args.start, args.end, window=60, minimum=20,
    )
    calendar = sorted(valid_dates["SPY"] & valid_dates["QQQ"])
    signal = {}
    for day in calendar:
        spy = gex_available.get(("SPY", day))
        qqq = gex_available.get(("QQQ", day))
        signal[day] = (
            spy is not None and qqq is not None
            and spy["net_gex"] < 0 and qqq["net_gex"] < 0
        )

    splits = load_splits(args.bars)
    universe = [ticker for ticker in tickers(args.bars) if ticker not in ETF_TICKERS]
    config = TurtleConfig(
        entry_window=55, exit_window=20, atr_window=20,
        skip_after_winner=False, directions=(1,), use_channel_exit=False,
        chandelier_atr=5.0,
    )
    all_timestamps = set()
    closes = defaultdict(dict)
    trades = []
    active_universe = []
    for ticker in universe:
        raw = adjust_bars(
            load_bars(args.bars, ticker, args.start, args.end), splits.get(ticker, [])
        )
        raw = [bar for bar in raw if bar.timestamp[:10] in valid_dates["SPY"]]
        blocks = raw_blocks(raw)
        if not blocks:
            continue
        active_universe.append(ticker)
        regular = [bar for rows in blocks.values() for bar in rows]
        regular_index = {bar.timestamp: index for index, bar in enumerate(regular)}
        thirty = resample_regular_session(raw, minutes=30)
        ticker_trades, _audit = run_turtle(thirty, config=config)
        trades.extend(path_trade(
            ticker, trade, blocks, regular, regular_index, config.round_trip_cost,
        ) for trade in ticker_trades if trade.unit_entries and trade.n_at_entry > 0)
        for rows in blocks.values():
            for bar in rows:
                closes[bar.timestamp][ticker] = bar.close
                all_timestamps.add(bar.timestamp)
        print(f"{ticker}: {len(ticker_trades)} trades", flush=True)

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

    baseline = [replay(
        trades, "No overlay", {}, funding, timeline, closes, prepared, daily_indices,
        seed=seed, max_positions=args.max_positions,
        broker_spread=args.broker_spread,
        size_fn=lambda _variant, _trade: 1.0, return_full_path=True,
    ) for seed in range(args.draws)]
    baseline_summary = median_summary(baseline)

    index_bars = {}
    for ticker in ("SPY", "QQQ"):
        raw = adjust_bars(
            load_bars(args.bars, ticker, args.start, args.end), splits.get(ticker, [])
        )
        lookup = {bar.timestamp: bar for bar in raw if bar.timestamp in timeline_index}
        if len(lookup) != len(timeline):
            raise RuntimeError(f"{ticker} hedge path does not cover the stock timeline")
        index_bars[ticker] = lookup

    output_rows = []
    summaries = {}
    for ticker in ("SPY", "QQQ"):
        for leverage in LEVERAGES:
            hedge, audit = hedge_path(
                timeline, index_bars[ticker], signal, funding, leverage=leverage,
                round_trip_cost=args.round_trip_cost,
                broker_spread=args.broker_spread,
            )
            hedge_stats = standalone_metrics(hedge, daily_indices)
            draws = []
            paired = []
            for seed, base in enumerate(baseline):
                combined = np.asarray(base["full_path"]) - 100.0 + hedge
                item = metrics(combined, combined, daily_indices)
                draws.append(item)
                paired.append({
                    "seed": seed,
                    "delta_nav": item["ending"] - base["ending"],
                    "delta_cagr_pp": 100.0 * (item["cagr"] - base["cagr"]),
                    "delta_dd_pp": 100.0 * (item["maxdd"] - base["maxdd"]),
                    "delta_sharpe": item["sharpe"] - base["sharpe"],
                })
            summary = {
                key: statistics.median(draw[key] for draw in draws)
                for key in ("ending", "cagr", "maxdd", "sharpe")
            }
            key = (ticker, leverage)
            summaries[key] = {
                **summary,
                "nav_wins": sum(row["delta_nav"] > 0 for row in paired),
                "dd_wins": sum(row["delta_dd_pp"] > 0 for row in paired),
                "delta_nav": statistics.median(row["delta_nav"] for row in paired),
                "delta_cagr_pp": statistics.median(row["delta_cagr_pp"] for row in paired),
                "delta_dd_pp": statistics.median(row["delta_dd_pp"] for row in paired),
                "delta_sharpe": safe_median([row["delta_sharpe"] for row in paired]),
                "hedge_ending": hedge_stats["ending"],
                "hedge_maxdd": hedge_stats["maxdd"],
                **audit,
            }
            output_rows.extend({"ticker": ticker, "leverage": leverage, **row}
                               for row in paired)
            print(
                f"{ticker} {leverage}x: hedge ${hedge_stats['ending']:.2f}, "
                f"portfolio DD {summary['maxdd']:.1%}", flush=True,
            )

    args.outdir.mkdir(parents=True, exist_ok=True)
    paired_csv = args.outdir / "turtle_gex_hedge_paired.csv"
    with paired_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    report = args.outdir / "turtle_gex_hedge.md"
    lines = [
        "# $100 negative-GEX short hedge on the 30-minute Turtle portfolio", "",
        f"{len(active_universe)} stocks, {len(calendar)} sessions, {len(trades)} candidate "
        f"trades and {args.draws} paired capacity orderings. $100 is moved from the fixed sleeve into "
        "a limited-loss hedge account; $700 Turtle capital and the remaining $200 sleeve are "
        "unchanged.", "",
        "The hedge shorts SPY or QQQ at the next open when the prior EOD net GEX of both "
        "indexes is negative, and exits at the next open after the condition clears. Units "
        "stay fixed within a regime. Five-minute highs trigger liquidation at zero; the hedge "
        "is not replenished. Costs and overnight financing are included.", "",
        f"Baseline: end NAV ${baseline_summary['ending']:,.2f}, CAGR "
        f"{baseline_summary['cagr']:+.2%}, max DD {baseline_summary['maxdd']:.1%}, "
        f"Sharpe {baseline_summary['sharpe']:.3f}.", "",
        "| Hedge | Lev | Hedge end | Liquidated | Portfolio end | Max DD | NAV wins | DD wins | Delta CAGR | Delta DD | Delta Sharpe |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ticker in ("SPY", "QQQ"):
        for leverage in LEVERAGES:
            item = summaries[(ticker, leverage)]
            lines.append(
                f"| {ticker} | {leverage}x | ${item['hedge_ending']:,.2f} | "
                f"{'yes' if item['liquidated'] else 'no'} | ${item['ending']:,.2f} | "
                f"{item['maxdd']:.1%} | {item['nav_wins']}/{args.draws} | "
                f"{item['dd_wins']}/{args.draws} | {item['delta_cagr_pp']:+.2f} pp | "
                f"{item['delta_dd_pp']:+.2f} pp | {item['delta_sharpe']:+.3f} |"
            )
    lines += ["", "Positive delta DD means the hedge reduced drawdown. At 20x, a roughly 5% "
              "adverse index move can consume the entire hedge account before broker maintenance "
              "margin, so the modeled zero-loss boundary is optimistic."]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {report}, {paired_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
