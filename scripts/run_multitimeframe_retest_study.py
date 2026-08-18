"""Run the leakage-free 4h Sweep/Engulf -> 15m rejection strategy."""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.multitimeframe_retest import (  # noqa: E402
    RetestConfig,
    run_retest_strategy,
    summarise_retests,
)
from savi_uz.sweep_engulf import SweepConfig, resample_regular_session, run_strategy, summarise  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--ticker", default="GLD")
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--split", default="2023-01-01")
    parser.add_argument("--outdir", type=Path, default=Path("out/strategy"))
    return parser.parse_args(argv)


def load_bars(path: Path, ticker: str) -> list[Bar]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars "
            "WHERE ticker=? AND frequency='5min' ORDER BY ts", (ticker,)
        ).fetchall()
    finally:
        connection.close()
    return [Bar(*row) for row in rows]


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def summary_row(label: str, period: str, trades) -> str:
    result = summarise_retests(trades)
    return (
        f"| {label} | {period} | {result.count:,} | {result.longs}/{result.shorts} | "
        f"{pct(result.win_rate)} | {result.profit_factor:.2f} | {result.mean_r:+.3f} | "
        f"${100 * result.ending_equity:.2f} | {pct(result.cagr)} | "
        f"{pct(result.max_drawdown)} | {pct(result.stop_rate)} | "
        f"{pct(result.target_rate)} | {pct(result.overnight_rate)} |"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ticker = args.ticker.upper()
    source = load_bars(args.db, ticker)
    if not source:
        raise SystemExit(f"error: no {ticker} five-minute bars")
    four_hour = resample_regular_session(source, minutes=240)
    fifteen = resample_regular_session(source, minutes=15)
    pine = SweepConfig()
    trend_ema = replace(pine, invert_trades=False, use_ema=True)
    primary = RetestConfig()
    variants = [
        ("Locked: 1h regression, 0.1 ATR buffer", pine, primary),
        ("Wider 0.5 ATR buffer", pine, replace(primary, stop_buffer_atr=0.5)),
        ("No slope-alignment filter", pine, replace(primary, require_aligned_slope=False)),
        ("Observe 30 minutes", pine, replace(primary, observation_bars=2)),
        ("Observe 90 minutes", pine, replace(primary, observation_bars=6)),
        ("Confirmed-pivot trendline", pine, replace(primary, trendline_method="confirmed pivots")),
        ("Trend-following 4h EMA200", trend_ema, primary),
        ("Trend-following EMA200, 0.5 ATR buffer", trend_ema, replace(primary, stop_buffer_atr=0.5)),
    ]

    results = {}
    audits = {}
    lines = [
        f"# Four-hour bias, 15-minute retest/rejection - {ticker}", "",
        f"Source: **{len(source):,}** Tiingo five-minute bars, resampled into "
        f"**{len(four_hour):,}** cash-session four-hour bars and **{len(fifteen):,}** "
        f"15-minute bars. Split: **{args.split}**.", "",
        "## Locked rules", "",
        "1. A completed four-hour Pine-default sweep supplies direction. Its original ATR(14), "
        "1.5 ATR stop geometry defines a fixed 2R target, but no trade is entered yet.",
        "2. The signal becomes usable only at the next four-hour bar's open. The next four completed "
        "15-minute bars are observation-only.",
        "3. Freeze a least-squares trendline through the observation bars' lows for a long or highs "
        "for a short. Its slope must agree with the four-hour direction.",
        "4. During the rest of that regular session, require a candle to pierce the projected line "
        "and close back through it in the trade direction.",
        "5. Enter at the next 15-minute open. Stop beyond the rejection extreme by 0.1 ATR(14). "
        "The fixed four-hour target must offer at least 2R from the actual fill.",
        "6. Exit at stop, target, or the fifth regular-session close. A gap through a resting order "
        "fills at the observed open. One position at a time; 2 bp round-trip cost.", "",
        "Every level is frozen from completed bars. A rejection on the final cash-session candle is "
        "not accepted because its next open would be overnight.", "",
        "## Chronological results", "",
        "| Variant | Period | Trades | L/S | Win | PF | Mean R | $100 -> | CAGR | Max DD | Stop | Target | Overnight |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, htf_config, config in variants:
        trades, audit = run_retest_strategy(
            four_hour, fifteen, htf_config=htf_config, config=config, start=args.start
        )
        results[label], audits[label] = trades, audit
        train = [trade for trade in trades if trade.htf_signal_timestamp[:10] < args.split]
        test = [trade for trade in trades if trade.htf_signal_timestamp[:10] >= args.split]
        lines.append(summary_row(label, "train", train))
        lines.append(summary_row(label, "test", test))

    locked = results["Locked: 1h regression, 0.1 ATR buffer"]
    audit = audits["Locked: 1h regression, 0.1 ATR buffer"]
    train_locked = [trade for trade in locked if trade.htf_signal_timestamp[:10] < args.split]
    test_locked = [trade for trade in locked if trade.htf_signal_timestamp[:10] >= args.split]
    lines += [
        "", "## Signal funnel", "",
        f"- Raw four-hour signals: **{audit.htf_signals:,}**",
        f"- Slope rejected: **{audit.misaligned_trendline:,}**",
        f"- Four-hour thesis hit stop/target before entry: **{audit.thesis_expired_before_entry:,}**",
        f"- No same-session trendline rejection: **{audit.no_rejection:,}**",
        f"- Invalid or below 2R at actual entry: **{audit.invalid_or_low_reward:,}**",
        f"- Overlap skipped: **{audit.skipped_overlap:,}**",
        f"- Executed trades: **{audit.trades:,}** ({audit.trades / audit.htf_signals * 100:.1f}% of signals)",
    ]

    baseline = run_strategy(four_hour, pine, start=args.start)
    baseline_train = summarise([trade for trade in baseline if trade.signal_timestamp[:10] < args.split])
    baseline_test = summarise([trade for trade in baseline if trade.signal_timestamp[:10] >= args.split])
    locked_train, locked_test = summarise_retests(train_locked), summarise_retests(test_locked)
    lines += [
        "", "## Did lower-timeframe execution improve the four-hour edge?", "",
        "| Model | Train trades | Train PF | Train $100 -> | Test trades | Test PF | Test $100 -> |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Direct four-hour entry | {baseline_train.count:,} | {baseline_train.profit_factor:.2f} | "
        f"${100 * baseline_train.ending_equity:.2f} | {baseline_test.count:,} | "
        f"{baseline_test.profit_factor:.2f} | ${100 * baseline_test.ending_equity:.2f} |",
        f"| 15-minute retest | {locked_train.count:,} | {locked_train.profit_factor:.2f} | "
        f"${100 * locked_train.ending_equity:.2f} | {locked_test.count:,} | "
        f"{locked_test.profit_factor:.2f} | ${100 * locked_test.ending_equity:.2f} |",
        "", "The tighter entry does not inherit the higher-timeframe win probability. Most retests "
        "are stopped before the distant four-hour target, so nominal planned R is high while realized "
        "expectancy is negative.",
    ]

    lines += ["", "## Two-year stability of the locked rule", "",
              "| Years | Trades | PF | Mean R | $100 -> | Max DD |",
              "|---|---:|---:|---:|---:|---:|"]
    for label, first, last in (
        ("2017-18", "2017-01-01", "2019-01-01"),
        ("2019-20", "2019-01-01", "2021-01-01"),
        ("2021-22", "2021-01-01", "2023-01-01"),
        ("2023-24", "2023-01-01", "2025-01-01"),
        ("2025+", "2025-01-01", "9999-01-01"),
    ):
        summary = summarise_retests([
            trade for trade in locked if first <= trade.htf_signal_timestamp[:10] < last
        ])
        lines.append(
            f"| {label} | {summary.count:,} | {summary.profit_factor:.2f} | "
            f"{summary.mean_r:+.3f} | ${100 * summary.ending_equity:.2f} | "
            f"{pct(summary.max_drawdown)} |"
        )

    lines += [
        "", "## Leakage and execution audit", "",
        "- Four-hour direction becomes available at the next four-hour timestamp, never at the "
        "signal bar's opening timestamp.",
        "- The observation hour is fully closed before fitting the trendline.",
        "- Trendline coefficients are frozen; later bars cannot move its anchors or slope.",
        "- Rejection is known only at its close and entry is the following 15-minute open.",
        "- Stop and target ordering inside one OHLC bar is conservative: the stop wins ties.",
        "- Regular-session data cannot observe after-hours paths; overnight gaps fill at the next open.",
        "- This is a first specification tested on GLD. Variant rows are diagnostics, not permission "
        "to select the best test-period result.",
    ]

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = f"multitimeframe_retest_{ticker}"
    report = args.outdir / f"{stem}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    trade_path = args.outdir / f"{stem}.csv"
    with trade_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(asdict(locked[0]).keys()) if locked else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(asdict(trade) for trade in locked)
    print(f"wrote {report} and {trade_path} ({len(locked):,} locked trades)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
