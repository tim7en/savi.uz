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
    if result.count == 0:
        return f"| {label} | {period} | 0 | 0/0 | — | — | — | — | — | — | — | — | — | — |"
    return (
        f"| {label} | {period} | {result.count:,} | {result.longs}/{result.shorts} | "
        f"{pct(result.win_rate)} | {result.profit_factor:.2f} | {result.mean_r:+.3f} | "
        f"${100 * result.ending_equity:.2f} | {pct(result.cagr)} | "
        f"{pct(result.max_drawdown)} | {pct(result.stop_rate)} | {pct(result.breakeven_rate)} | "
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
    liquidity_sweep = replace(primary, require_external_liquidity_sweep=True)
    liquidity_rejection = replace(primary, entry_trigger="liquidity rejection")
    liquidity_stop = replace(primary, stop_method="resting liquidity")
    liquidity_target = replace(primary, target_method="resting liquidity")
    breakeven = replace(primary, breakeven_trigger_r=1.0)
    variants = [
        ("Locked: 1h regression, 0.1 ATR buffer", pine, primary),
        ("Break-even after completed-bar +1R", pine, breakeven),
        ("External liquidity rejection entry", pine, liquidity_rejection),
        ("External liquidity entry + target", pine, replace(
            liquidity_rejection, target_method="resting liquidity"
        )),
        ("Liquidity sweep/reclaim entry", pine, liquidity_sweep),
        ("Stop beyond resting D/W liquidity", pine, liquidity_stop),
        ("Liquidity stop + break-even at 1R", pine, replace(
            liquidity_stop, breakeven_trigger_r=1.0
        )),
        ("Target nearest resting D/W liquidity", pine, liquidity_target),
        ("Liquidity target + break-even at 1R", pine, replace(
            liquidity_target, breakeven_trigger_r=1.0
        )),
        ("Liquidity sweep entry + target", pine, replace(
            liquidity_sweep, target_method="resting liquidity"
        )),
        ("Liquidity stop + target", pine, replace(
            primary, stop_method="resting liquidity", target_method="resting liquidity"
        )),
        ("All liquidity rules", pine, replace(
            liquidity_sweep, stop_method="resting liquidity", target_method="resting liquidity"
        )),
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
        "The break-even variants activate an entry-price stop only after a completed 15-minute bar "
        "has reached +1R. The new stop is executable from the following bar, and transaction costs "
        "mean a nominal break-even exit is a small net loss.", "",
        "## External liquidity definitions", "",
        "- **PDH/PDL:** the immediately preceding completed regular-session high and low.",
        "- **PWH/PWL:** the preceding completed ISO trading week's high and low.",
        "- A pool is *resting* only if no 15-minute bar has touched it since the level became known.",
        "- A sweep entry pierces a resting adverse-side pool and closes back across it on the same "
        "trendline-rejection candle.",
        "- The external-liquidity entry instead lets that sweep/reclaim be the rejection trigger; "
        "the close must remain on the directionally correct side of the frozen trendline.",
        "- A liquidity stop sits 0.1 ATR beyond the nearest still-resting adverse pool, but is rejected "
        "if it lies beyond the original four-hour invalidation.",
        "- A liquidity target is the nearest still-resting directional pool inside the original "
        "four-hour target; it must still offer at least 2R.",
        "- Because this database contains US regular hours, previous session and previous trading day "
        "are the same feature. Asia and London levels cannot be tested from this source.", "",
        "## Chronological results", "",
        "| Variant | Period | Trades | L/S | Win | PF | Mean R | $100 -> | CAGR | Max DD | Stop | BE exit | Target | Overnight |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
    breakeven_trades = results["Break-even after completed-bar +1R"]
    audit = audits["Locked: 1h regression, 0.1 ATR buffer"]
    train_locked = [trade for trade in locked if trade.htf_signal_timestamp[:10] < args.split]
    test_locked = [trade for trade in locked if trade.htf_signal_timestamp[:10] >= args.split]
    train_breakeven = [
        trade for trade in breakeven_trades if trade.htf_signal_timestamp[:10] < args.split
    ]
    test_breakeven = [
        trade for trade in breakeven_trades if trade.htf_signal_timestamp[:10] >= args.split
    ]
    lines += [
        "", "## Effect of moving the stop to break-even at +1R", "",
        "| Period | Base PF | BE PF | Base $100 -> | BE $100 -> | Base DD | BE DD | "
        "BE activated | BE exit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for period, base_rows, be_rows in (
        ("train", train_locked, train_breakeven),
        ("test", test_locked, test_breakeven),
    ):
        base_summary = summarise_retests(base_rows)
        be_summary = summarise_retests(be_rows)
        activated = sum(trade.breakeven_activated for trade in be_rows) / len(be_rows) if be_rows else 0.0
        lines.append(
            f"| {period} | {base_summary.profit_factor:.2f} | {be_summary.profit_factor:.2f} | "
            f"${100 * base_summary.ending_equity:.2f} | ${100 * be_summary.ending_equity:.2f} | "
            f"{pct(base_summary.max_drawdown)} | {pct(be_summary.max_drawdown)} | "
            f"{pct(activated)} | {pct(be_summary.breakeven_rate)} |"
        )
    lines += [
        "", "The +1R rule is a capital-preservation overlay. It can reduce stop losses and drawdown, "
        "but it also converts some eventual targets into small cost-adjusted losses. Profit factor "
        "must remain above one on unseen data before treating the overlay as a tradable edge.",
    ]
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

    lines += [
        "", "## Liquidity-rule signal counts", "",
        "| Variant | Trades | No swept entry | No stop pool | No target pool | Below 2R/invalid |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    liquidity_labels = [
        "External liquidity rejection entry",
        "External liquidity entry + target",
        "Liquidity sweep/reclaim entry",
        "Stop beyond resting D/W liquidity",
        "Liquidity stop + break-even at 1R",
        "Target nearest resting D/W liquidity",
        "Liquidity target + break-even at 1R",
        "Liquidity sweep entry + target",
        "Liquidity stop + target",
        "All liquidity rules",
    ]
    for label in liquidity_labels:
        item = audits[label]
        lines.append(
            f"| {label} | {item.trades:,} | {item.no_liquidity_entry:,} | "
            f"{item.no_liquidity_stop:,} | {item.no_liquidity_target:,} | "
            f"{item.invalid_or_low_reward:,} |"
        )

    baseline = run_strategy(four_hour, pine, start=args.start)
    baseline_train = summarise([trade for trade in baseline if trade.signal_timestamp[:10] < args.split])
    baseline_test = summarise([trade for trade in baseline if trade.signal_timestamp[:10] >= args.split])
    locked_train, locked_test = summarise_retests(train_locked), summarise_retests(test_locked)
    stop_name = "Stop beyond resting D/W liquidity"
    stop_train = summarise_retests([
        trade for trade in results[stop_name] if trade.htf_signal_timestamp[:10] < args.split
    ])
    stop_test = summarise_retests([
        trade for trade in results[stop_name] if trade.htf_signal_timestamp[:10] >= args.split
    ])
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

    lines += [
        "", "## What the external-liquidity rules changed", "",
        "The exact trendline-plus-liquidity-sweep conjunction generated no entries. The independent "
        "liquidity rejection trigger also generated no entries after the one-hour observation, slope, "
        "candle-direction, and same-session requirements.", "",
        f"The protected-stop version produced **{stop_train.count}** training and **{stop_test.count}** "
        f"holdout trades (PF **{stop_train.profit_factor:.2f}** and **{stop_test.profit_factor:.2f}**). "
        "That sample is too small to establish an edge even when both numbers exceed one.", "",
        "The target-only and combined rules should be read directly from the chronological table. "
        "No variant should be selected from its holdout result; this run has now consumed that holdout.",
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
        "- Daily and weekly liquidity levels are calculated only from sessions completed before they "
        "become available; a future bar cannot revise them.",
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
    liquidity_path = args.outdir / f"{stem}_liquidity_variants.csv"
    liquidity_names = set(liquidity_labels)
    liquidity_rows = [
        {"variant": label, **asdict(trade)}
        for label, trades in results.items() if label in liquidity_names
        for trade in trades
    ]
    with liquidity_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(liquidity_rows[0].keys()) if liquidity_rows else ["variant"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(liquidity_rows)
    breakeven_path = args.outdir / f"{stem}_breakeven_variants.csv"
    breakeven_names = {
        "Break-even after completed-bar +1R",
        "Liquidity stop + break-even at 1R",
        "Liquidity target + break-even at 1R",
    }
    breakeven_rows = [
        {"variant": label, **asdict(trade)}
        for label, trades in results.items() if label in breakeven_names
        for trade in trades
    ]
    with breakeven_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(breakeven_rows[0].keys()) if breakeven_rows else ["variant"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(breakeven_rows)
    print(
        f"wrote {report}, {trade_path}, {liquidity_path}, and {breakeven_path} "
        f"({len(locked):,} locked trades)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
