"""Use prior-day SPY/QQQ GEX as a macro risk driver for stock Turtle trades."""

from __future__ import annotations

import argparse
import csv
import math
import random
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
from run_turtle_gex_overlay import load_gex_features, safe_median  # noqa: E402
from run_turtle_leverage_stability import (  # noqa: E402
    build_regimes, funding_map, load_bars, tickers,
)
from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import TurtleConfig, run_turtle  # noqa: E402


ETF_TICKERS = {"SPY", "QQQ", "IWM", "GLD", "EWJ", "EWT", "EWY", "KWEB"}
GEX_VARIANTS = (
    "No overlay",
    "All entries half-risk control",
    "Date-shuffled mapped GEX control",
    "SPY negative GEX half-risk",
    "QQQ negative GEX half-risk",
    "Either index negative half-risk",
    "Both indexes negative half-risk",
    "Correlation-mapped negative GEX half-risk",
)
EARNINGS_VARIANTS = (
    "No overlay",
    "Earnings deterioration half-risk",
    "Both indexes negative GEX half-risk",
    "Earnings OR both-negative GEX half-risk",
    "Earnings AND both-negative GEX half-risk",
    "Earnings x GEX compounded diagnostic",
    "Earnings OR shuffled-GEX control",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--gex", type=Path, default=Path("data/options/marketdata.db"))
    parser.add_argument("--macro", type=Path, default=Path("data/macro/macro.db"))
    parser.add_argument("--equity", type=Path, default=Path("data/equity/equity.db"))
    parser.add_argument("--start", default="2025-08-18")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--broker-spread", type=float, default=0.015)
    parser.add_argument("--correlation-window", type=int, default=60)
    parser.add_argument("--min-correlation-history", type=int, default=20)
    parser.add_argument("--correlation-floor", type=float, default=0.30)
    parser.add_argument("--outdir", type=Path, default=Path("out/strategy"))
    parser.add_argument("--include-earnings", action="store_true")
    return parser.parse_args(argv)


def daily_returns(closes: dict[str, float]) -> dict[str, float]:
    days = sorted(closes)
    return {
        days[index]: closes[days[index]] / closes[days[index - 1]] - 1.0
        for index in range(1, len(days)) if closes[days[index - 1]] > 0
    }


def correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or statistics.pstdev(left) == 0 or statistics.pstdev(right) == 0:
        return None
    return statistics.correlation(left, right)


def build_driver_map(daily_closes: dict[str, dict[str, float]], calendar: list[str],
                     *, window: int, minimum: int, floor: float):
    returns = {ticker: daily_returns(values) for ticker, values in daily_closes.items()}
    mapping = {}
    details = {}
    for ticker in sorted(set(daily_closes) - {"SPY", "QQQ"}):
        for entry_day in calendar:
            common = sorted(
                day for day in returns[ticker]
                if day < entry_day and day in returns["SPY"] and day in returns["QQQ"]
            )[-window:]
            if len(common) < minimum:
                continue
            stock = [returns[ticker][day] for day in common]
            spy = correlation(stock, [returns["SPY"][day] for day in common])
            qqq = correlation(stock, [returns["QQQ"][day] for day in common])
            candidates = [("SPY", spy), ("QQQ", qqq)]
            benchmark, value = max(
                ((name, value) for name, value in candidates if value is not None),
                key=lambda item: item[1], default=(None, None),
            )
            if value is not None and value >= floor:
                mapping[(ticker, entry_day)] = benchmark
                details[(ticker, entry_day)] = (benchmark, value, spy, qqq, len(common))
    return mapping, details


def shuffled_regimes(gex_available, calendar, seed=7419):
    rng = random.Random(seed)
    result = {}
    for symbol in ("SPY", "QQQ"):
        flags = [
            (gex_available.get((symbol, day)) or {}).get("net_gex", 0) < 0
            for day in calendar
        ]
        rng.shuffle(flags)
        result.update(((symbol, day), flag) for day, flag in zip(calendar, flags))
    return result


def size_function(gex_available, driver_map, shuffled, earnings):
    def negative(symbol, day):
        feature = gex_available.get((symbol, day))
        return feature is not None and feature["net_gex"] < 0

    def size(variant, trade):
        if variant == "No overlay":
            return 1.0
        day = trade.entry[:10]
        earnings_warning = earnings.get(day) is True
        if variant == "Earnings deterioration half-risk":
            return 0.5 if earnings_warning else 1.0
        if variant == "All entries half-risk control":
            return 0.5
        spy_negative = negative("SPY", day)
        qqq_negative = negative("QQQ", day)
        both_negative = spy_negative and qqq_negative
        shuffled_both = (
            shuffled.get(("SPY", day), False)
            and shuffled.get(("QQQ", day), False)
        )
        if variant == "Earnings OR both-negative GEX half-risk":
            return 0.5 if earnings_warning or both_negative else 1.0
        if variant == "Earnings AND both-negative GEX half-risk":
            return 0.5 if earnings_warning and both_negative else 1.0
        if variant == "Earnings x GEX compounded diagnostic":
            return ((0.5 if earnings_warning else 1.0)
                    * (0.5 if both_negative else 1.0))
        if variant == "Earnings OR shuffled-GEX control":
            return 0.5 if earnings_warning or shuffled_both else 1.0
        if variant == "Date-shuffled mapped GEX control":
            driver = driver_map.get((trade.ticker, day))
            flag = shuffled.get((driver, day), False)
        elif variant == "SPY negative GEX half-risk":
            flag = spy_negative
        elif variant == "QQQ negative GEX half-risk":
            flag = qqq_negative
        elif variant == "Either index negative half-risk":
            flag = spy_negative or qqq_negative
        elif variant in {"Both indexes negative half-risk",
                         "Both indexes negative GEX half-risk"}:
            flag = spy_negative and qqq_negative
        else:
            driver = driver_map.get((trade.ticker, day))
            flag = negative(driver, day) if driver is not None else False
        return 0.5 if flag else 1.0
    return size


def main(argv=None):
    args = parse_args(argv)
    variants = EARNINGS_VARIANTS if args.include_earnings else GEX_VARIANTS
    _gex_rows, gex_available, valid_dates = load_gex_features(
        args.gex, args.start, args.end, window=60, minimum=20,
    )
    calendar = sorted(valid_dates["SPY"] & valid_dates["QQQ"])
    universe = [ticker for ticker in tickers(args.bars) if ticker not in ETF_TICKERS]
    splits = load_splits(args.bars)
    config = TurtleConfig(
        entry_window=55, exit_window=20, atr_window=20,
        skip_after_winner=False, directions=(1,), use_channel_exit=False,
        chandelier_atr=5.0,
    )
    all_timestamps = set()
    closes = defaultdict(dict)
    daily_closes = defaultdict(dict)
    trades = []
    ticker_counts = {}
    active_universe = []
    for ticker in universe + ["SPY", "QQQ"]:
        raw = adjust_bars(
            load_bars(args.bars, ticker, args.start, args.end), splits.get(ticker, [])
        )
        raw = [bar for bar in raw if bar.timestamp[:10] in valid_dates["SPY"]]
        blocks = raw_blocks(raw)
        if not blocks:
            if ticker not in {"SPY", "QQQ"}:
                print(f"{ticker}: excluded (no bars in period)", flush=True)
            continue
        for rows in blocks.values():
            day = rows[0].timestamp[:10]
            daily_closes[ticker][day] = rows[-1].close
        if ticker in {"SPY", "QQQ"}:
            continue
        active_universe.append(ticker)
        regular = [bar for rows in blocks.values() for bar in rows]
        regular_index = {bar.timestamp: index for index, bar in enumerate(regular)}
        thirty = resample_regular_session(raw, minutes=30)
        ticker_trades, _audit = run_turtle(thirty, config=config)
        converted = [path_trade(
            ticker, trade, blocks, regular, regular_index, config.round_trip_cost,
        ) for trade in ticker_trades if trade.unit_entries and trade.n_at_entry > 0]
        trades.extend(converted)
        ticker_counts[ticker] = len(converted)
        for rows in blocks.values():
            for bar in rows:
                closes[bar.timestamp][ticker] = bar.close
                all_timestamps.add(bar.timestamp)
        print(f"{ticker}: {len(converted)} candidate trades", flush=True)

    driver_map, driver_details = build_driver_map(
        daily_closes, calendar, window=args.correlation_window,
        minimum=args.min_correlation_history, floor=args.correlation_floor,
    )
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
    earnings = {}
    if args.include_earnings:
        equity = sqlite3.connect(f"file:{args.equity}?mode=ro", uri=True)
        earnings = build_regimes(calendar, macro, equity)["earnings"]
        equity.close()
    macro.close()
    risk_size = size_function(
        gex_available, driver_map, shuffled_regimes(gex_available, calendar), earnings
    )
    joined = sum((trade.ticker, trade.entry[:10]) in driver_map for trade in trades)
    print(f"correlation driver available: {joined}/{len(trades)} candidate trades")
    for variant in variants[1:]:
        flagged = sum(risk_size(variant, trade) < 1.0 for trade in trades)
        print(f"{variant}: {flagged} candidates flagged", flush=True)

    results, summaries = {}, {}
    for variant in variants:
        draws = [replay(
            trades, variant, {}, funding, timeline, closes, prepared, daily_indices,
            seed=seed, max_positions=args.max_positions,
            broker_spread=args.broker_spread, size_fn=risk_size,
            post_capacity_size=True,
        ) for seed in range(args.draws)]
        results[variant] = draws
        summaries[variant] = median_summary(draws)
        print(
            f"{variant}: NAV {summaries[variant]['ending']:.2f}, "
            f"Sharpe {summaries[variant]['sharpe']:.3f}", flush=True,
        )

    baseline = results["No overlay"]
    paired_rows, paired_summary = [], {}
    for variant in variants[1:]:
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

    incremental = None
    state_summary = {}
    if args.include_earnings:
        without = results["Earnings deterioration half-risk"]
        with_gex = results["Earnings OR both-negative GEX half-risk"]
        rows = [{
            "delta_nav": combined["ending"] - earnings_only["ending"],
            "delta_cagr_pp": 100.0 * (combined["cagr"] - earnings_only["cagr"]),
            "delta_sharpe": combined["sharpe"] - earnings_only["sharpe"],
        } for earnings_only, combined in zip(without, with_gex)]
        incremental = {
            "nav_wins": sum(row["delta_nav"] > 0 for row in rows),
            "delta_nav": statistics.median(row["delta_nav"] for row in rows),
            "delta_cagr_pp": statistics.median(row["delta_cagr_pp"] for row in rows),
            "delta_sharpe": safe_median([row["delta_sharpe"] for row in rows]),
        }
        states = defaultdict(list)
        for trade in trades:
            day = trade.entry[:10]
            earnings_value = earnings.get(day)
            spy = gex_available.get(("SPY", day))
            qqq = gex_available.get(("QQQ", day))
            gex_warning = (
                spy is not None and qqq is not None
                and spy["net_gex"] < 0 and qqq["net_gex"] < 0
            )
            if earnings_value is None:
                label = "Unknown earnings / Negative" if gex_warning else "Unknown earnings / Positive"
            else:
                label = (
                    ("Deteriorating" if earnings_value else "Good")
                    + (" / Negative" if gex_warning else " / Positive")
                )
            states[label].append(trade.net_r)
        for label, values in states.items():
            state_summary[label] = {
                "trades": len(values),
                "win_rate": sum(value > 0 for value in values) / len(values),
                "mean_r": statistics.mean(values),
                "median_r": statistics.median(values),
                "total_r": sum(values),
            }

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = "turtle_gex_earnings" if args.include_earnings else "turtle_gex_macro"
    paired_csv = args.outdir / f"{stem}_paired.csv"
    with paired_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    mapping_csv = args.outdir / f"{stem}_drivers.csv"
    with mapping_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("ticker", "entry_date", "driver", "selected_corr",
                         "spy_corr", "qqq_corr", "history"))
        for (ticker, day), values in sorted(driver_details.items()):
            writer.writerow((ticker, day, *values))

    report = args.outdir / f"{stem}.md"
    title = (
        "# Earnings and SPY/QQQ GEX on 30-minute stock Turtle trades"
        if args.include_earnings else
        "# SPY/QQQ GEX as a macro driver for 30-minute stock Turtle trades"
    )
    lines = [
        title, "",
        f"{len(active_universe)} individual stocks, {args.start} through {args.end}; "
        f"{len(calendar)} sessions and {len(trades)} candidate trades. ETFs and index products "
        "are excluded. Portfolio settings match the prior path-aware 30-minute test.", "",
        ("Earnings uses the existing two-trading-day publication lag; GEX uses prior-session EOD. "
         "The primary combined rule caps risk at 0.5x when either warning exists and never "
         "multiplies the warnings."
         if args.include_earnings else
         "Every rule uses prior-session EOD index GEX. The adaptive rule selects SPY or QQQ "
         f"from the prior {args.correlation_window} stock-return sessions, requires at least "
         f"{args.min_correlation_history} observations and correlation >= "
         f"{args.correlation_floor:.2f}. Flagged entries receive half actual post-cap risk."), "",
        "| Variant | Median end NAV | CAGR | Sharpe | Trades | NAV wins | Delta NAV | Delta CAGR | Delta Sharpe |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in variants:
        item = summaries[variant]
        if variant == "No overlay":
            wins = delta_nav = delta_cagr = delta_sharpe = "--"
        else:
            paired = paired_summary[variant]
            wins = f"{paired['nav_wins']}/{args.draws}"
            delta_nav = f"${paired['delta_nav']:+,.2f}"
            delta_cagr = f"{paired['delta_cagr_pp']:+.2f} pp"
            delta_sharpe = f"{paired['delta_sharpe']:+.3f}"
        lines.append(
            f"| {variant} | ${item['ending']:,.2f} | {item['cagr']:+.2%} | "
            f"{item['sharpe']:.3f} | {item['trades']:.0f} | {wins} | {delta_nav} | "
            f"{delta_cagr} | {delta_sharpe} |"
        )
    if args.include_earnings and incremental is not None:
        lines += ["", "## Incremental GEX value versus earnings-only", "",
                  "Primary comparison: earnings deterioration half-risk versus a 0.5x cap when "
                  "either earnings deteriorates or both SPY and QQQ net GEX are negative.", "",
                  "| Comparison | NAV wins | Median delta NAV | Delta CAGR | Delta Sharpe |",
                  "|---|---:|---:|---:|---:|",
                  f"| Add GEX to earnings | {incremental['nav_wins']}/{args.draws} | "
                  f"${incremental['delta_nav']:+,.2f} | "
                  f"{incremental['delta_cagr_pp']:+.2f} pp | "
                  f"{incremental['delta_sharpe']:+.3f} |", "",
                  "## Four-state candidate-trade diagnostic", "",
                  "| Earnings / GEX state | Trades | Win rate | Mean R | Median R | Total R |",
                  "|---|---:|---:|---:|---:|---:|"]
        for label in ("Good / Positive", "Good / Negative",
                      "Deteriorating / Positive", "Deteriorating / Negative",
                      "Unknown earnings / Positive", "Unknown earnings / Negative"):
            if label not in state_summary:
                continue
            item = state_summary[label]
            lines.append(
                f"| {label} | {item['trades']} | {item['win_rate']:.1%} | "
                f"{item['mean_r']:+.3f} | {item['median_r']:+.3f} | "
                f"{item['total_r']:+.2f} |"
            )
    lines += ["", "## Coverage", "",
              f"- Correlation driver available for {joined}/{len(trades)} candidate trades.",
              f"- Candidates by ticker: " + ", ".join(
                  f"{ticker} {count}" for ticker, count in sorted(ticker_counts.items())
              ) + ".",
              "- NAV wins are paired capacity-order wins, not independent historical samples.",
              "- GEX is the locally modeled call-positive/put-negative proxy; it is not observed dealer inventory."]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {report}, {paired_csv}, {mapping_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
