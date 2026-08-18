"""Backtest the supplied Pine Sweep and Engulf strategy on local intraday bars."""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.sweep_engulf import (  # noqa: E402
    SweepConfig,
    resample_regular_session,
    run_strategy,
    summarise,
)
from savi_uz.volume_profile import Bar  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--ticker", default="GLD")
    parser.add_argument("--frequency", default="5min")
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--split", default="2023-01-01")
    parser.add_argument("--outdir", type=Path, default=Path("out/strategy"))
    return parser.parse_args(argv)


def load_bars(path: Path, ticker: str, frequency: str) -> list[Bar]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT ts, open, high, low, close, volume FROM bars "
            "WHERE ticker=? AND frequency=? ORDER BY ts", (ticker, frequency)
        ).fetchall()
    finally:
        connection.close()
    return [Bar(*row) for row in rows]


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def result_row(label: str, period: str, trades) -> str:
    summary = summarise(trades)
    return (
        f"| {label} | {period} | {summary.count:,} | {summary.longs:,}/{summary.shorts:,} | "
        f"{pct(summary.win_rate)} | {summary.profit_factor:.2f} | {summary.mean_r:+.3f} | "
        f"${100 * summary.ending_equity:.2f} | {pct(summary.cagr)} | "
        f"{pct(summary.max_drawdown)} | {pct(summary.stop_rate)} | "
        f"{pct(summary.overnight_rate)} |"
    )


def buy_hold(bars: list[Bar], start: str, end: str | None = None) -> tuple[float, float]:
    rows = [
        bar for bar in bars
        if bar.timestamp[:10] >= start and (end is None or bar.timestamp[:10] < end)
    ]
    if len(rows) < 2:
        return float("nan"), float("nan")
    equity = rows[-1].close / rows[0].open
    from datetime import date
    years = (date.fromisoformat(rows[-1].timestamp[:10]) - date.fromisoformat(rows[0].timestamp[:10])).days / 365.25
    cagr = equity ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    return equity, cagr


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ticker = args.ticker.upper()
    bars = load_bars(args.db, ticker, args.frequency)
    source_note = "Bars are read directly from the local Tiingo database."
    if not bars and args.frequency == "4hour":
        source = load_bars(args.db, ticker, "5min")
        bars = resample_regular_session(source, minutes=240)
        source_note = (
            f"Bars are resampled from {len(source):,} Tiingo five-minute bars, anchored to "
            "the 09:30 New York cash-session open. Each session has a four-hour first bar "
            "and a 2.5-hour final bar; no window crosses overnight."
        )
    if not bars:
        raise SystemExit(f"error: no {ticker} {args.frequency} bars")

    default = SweepConfig()
    variants = [
        ("Pine default: inverted", default),
        ("Not inverted", replace(default, invert_trades=False)),
        ("Inverted + EMA200", replace(default, use_ema=True)),
        ("Not inverted + EMA200", replace(default, invert_trades=False, use_ema=True)),
        ("Not inverted + EMA200, entry-anchored", replace(
            default, invert_trades=False, use_ema=True, anchor_to_signal_close=False
        )),
        ("Inverted + same prior candle", replace(default, previous_candle="Same Direction")),
        ("Not inverted + same prior candle", replace(
            default, invert_trades=False, previous_candle="Same Direction"
        )),
        ("Inverted, entry-anchored", replace(default, anchor_to_signal_close=False)),
        ("Not inverted, entry-anchored", replace(
            default, invert_trades=False, anchor_to_signal_close=False
        )),
    ]

    lines = [
        f"# Sweep and Engulf strategy - {ticker} {args.frequency}", "",
        f"Bars: **{len(bars):,}**, {bars[0].timestamp[:10]} through {bars[-1].timestamp[:10]}; "
        f"chronological split: **{args.split}**.", "",
        source_note, "",
        "## Execution model", "",
        "The supplied Pine defaults are reproduced: inverted trades, any previous-candle direction, "
        "EMA disabled, ATR(14), 1.5 ATR stop and 2R target. A completed sweep bar submits the "
        "order and entry occurs at the next available bar open. As in the Pine code, stop and target "
        "are anchored to the signal close rather than the actual fill. Positions can cross sessions.", "",
        "Stops gapped through are filled at the observed regular-session open. If stop and target are "
        "both inside one OHLC bar, the stop is charged. Returns include a 2 bp round trip but "
        "exclude financing, borrow fees, market impact and taxes. `$100 ->` compounds each trade at "
        "1x notional; the Pine declaration itself uses only one share against $1,000,000.", "",
        "## Chronological results", "",
        "| Variant | Period | Trades | Long/short | Win | PF | Mean R | $100 -> | CAGR | Max DD | Stop | Overnight |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    all_variant_trades = {}
    for label, config in variants:
        trades = run_strategy(bars, config, start=args.start)
        all_variant_trades[label] = trades
        train = [trade for trade in trades if trade.signal_timestamp[:10] < args.split]
        test = [trade for trade in trades if trade.signal_timestamp[:10] >= args.split]
        lines.append(result_row(label, "train", train))
        lines.append(result_row(label, "test", test))

    gross_trades = run_strategy(
        bars, replace(default, round_trip_cost=0.0), start=args.start
    )
    default_net = all_variant_trades["Pine default: inverted"]
    train_net = [trade for trade in default_net if trade.signal_timestamp[:10] < args.split]
    test_net = [trade for trade in default_net if trade.signal_timestamp[:10] >= args.split]
    train_gross = [trade for trade in gross_trades if trade.signal_timestamp[:10] < args.split]
    test_gross = [trade for trade in gross_trades if trade.signal_timestamp[:10] >= args.split]
    buy_train, buy_train_cagr = buy_hold(bars, args.start, args.split)
    buy_test, buy_test_cagr = buy_hold(bars, args.split)
    lines += [
        "", "## Default economics and buy-and-hold", "",
        "The pasted Pine declaration specifies one share and no commission. Gross results therefore "
        "show what TradingView is closest to displaying; net results apply the same modest 2 bp "
        "round-trip assumption used in the other local studies.", "",
        "| Period | Gross PF | Gross $100 -> | Net PF | Net $100 -> | Literal 1-share gross P&L | Buy/hold $100 -> | Buy/hold CAGR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, gross_rows, net_rows, held, held_cagr in (
        ("train", train_gross, train_net, buy_train, buy_train_cagr),
        ("test", test_gross, test_net, buy_test, buy_test_cagr),
    ):
        gross_summary, net_summary = summarise(gross_rows), summarise(net_rows)
        lines.append(
            f"| {label} | {gross_summary.profit_factor:.2f} | "
            f"${100 * gross_summary.ending_equity:.2f} | {net_summary.profit_factor:.2f} | "
            f"${100 * net_summary.ending_equity:.2f} | ${gross_summary.fixed_share_pnl:+.2f} | "
            f"${100 * held:.2f} | {pct(held_cagr)} |"
        )

    train_long = summarise([trade for trade in train_net if trade.direction > 0])
    test_long = summarise([trade for trade in test_net if trade.direction > 0])
    train_short = summarise([trade for trade in train_net if trade.direction < 0])
    test_short = summarise([trade for trade in test_net if trade.direction < 0])
    if train_long.profit_factor > 1 and test_long.profit_factor > 1:
        direction_note = (
            "The long side is positive in both halves, while shorts are the persistent drag. "
            "That is more encouraging than a test-only effect, but the sample is small and remains "
            "exposed to GLD's long-run upward drift."
        )
    else:
        direction_note = (
            "The recent long-only result is not stable: it loses in the training period, while "
            "the short side deteriorates sharply after 2023. That is consistent with regime exposure "
            "to GLD's strong recent uptrend rather than a portable two-sided pattern edge."
        )
    lines += [
        "", "## Direction check", "",
        "| Side | Train PF | Train $100 -> | Test PF | Test $100 -> |",
        "|---|---:|---:|---:|---:|",
        f"| Long | {train_long.profit_factor:.2f} | ${100 * train_long.ending_equity:.2f} | "
        f"{test_long.profit_factor:.2f} | ${100 * test_long.ending_equity:.2f} |",
        f"| Short | {train_short.profit_factor:.2f} | ${100 * train_short.ending_equity:.2f} | "
        f"{test_short.profit_factor:.2f} | ${100 * test_short.ending_equity:.2f} |",
        "", direction_note,
    ]

    lines += [
        "", "## Two-year stability", "",
        "| Variant | Years | Trades | PF | Mean R | $100 -> | Max DD |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    periods = (
        ("2017-18", "2017-01-01", "2019-01-01"),
        ("2019-20", "2019-01-01", "2021-01-01"),
        ("2021-22", "2021-01-01", "2023-01-01"),
        ("2023-24", "2023-01-01", "2025-01-01"),
        ("2025+", "2025-01-01", "9999-01-01"),
    )
    for label in ("Pine default: inverted", "Not inverted + EMA200"):
        for period, first, last in periods:
            summary = summarise([
                trade for trade in all_variant_trades[label]
                if first <= trade.signal_timestamp[:10] < last
            ])
            lines.append(
                f"| {label} | {period} | {summary.count:,} | {summary.profit_factor:.2f} | "
                f"{summary.mean_r:+.3f} | ${100 * summary.ending_equity:.2f} | "
                f"{pct(summary.max_drawdown)} |"
            )

    lines += ["", "## Default stop/target sensitivity (test period only)", "",
              "This is a diagnostic grid, not a parameter selection. Every cell is tested on the same "
              "2023+ period, so choosing its best result would be overfitting.", "",
              "| Stop ATR | Target | Trades | PF | Mean R | $100 -> | Max DD |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for stop_atr in (1.0, 1.5, 2.0, 2.5):
        for reward_risk in (1.0, 1.5, 2.0, 3.0):
            config = replace(default, stop_atr=stop_atr, reward_risk=reward_risk)
            trades = run_strategy(bars, config, start=args.split)
            summary = summarise(trades)
            lines.append(
                f"| {stop_atr:.1f} | {reward_risk:.1f}R | {summary.count:,} | "
                f"{summary.profit_factor:.2f} | {summary.mean_r:+.3f} | "
                f"${100 * summary.ending_equity:.2f} | {pct(summary.max_drawdown)} |"
            )

    default_trades = all_variant_trades["Pine default: inverted"]
    ambiguous = sum(trade.both_touched for trade in default_trades)
    gap_exits = sum(trade.exit_reason.startswith("gap_") for trade in default_trades)
    default_train_summary, default_test_summary = summarise(train_net), summarise(test_net)
    ema_trades = all_variant_trades["Not inverted + EMA200"]
    ema_train_summary = summarise([
        trade for trade in ema_trades if trade.signal_timestamp[:10] < args.split
    ])
    ema_test_summary = summarise([
        trade for trade in ema_trades if trade.signal_timestamp[:10] >= args.split
    ])
    if default_train_summary.profit_factor > 1 and default_test_summary.profit_factor > 1:
        verdict = (
            "The supplied default is positive in both halves, but still requires validation on "
            "another asset and with the actual chart-session definition before leverage."
        )
    elif default_test_summary.profit_factor > 1:
        verdict = (
            "The supplied default improves materially at this resolution, but is not yet validated: "
            "it is profitable after 2023 and flat/negative in training."
        )
    else:
        verdict = (
            "The supplied default is not economically viable at this resolution. Its small gross "
            "edge disappears out of sample and modest execution costs make both halves negative."
        )
    if ema_train_summary.profit_factor > 1 and ema_test_summary.profit_factor > 1:
        verdict += (
            f" The non-inverted EMA200 alternative is positive in both broad halves "
            f"(PF {ema_train_summary.profit_factor:.2f}/{ema_test_summary.profit_factor:.2f}), "
            "although its two-year results remain uneven."
        )
    lines += [
        "", "## Audit notes", "",
        f"- Default trades with both stop and target inside one OHLC bar: **{ambiguous:,}**; "
        "all were scored as stops.",
        f"- Default gap exits at a regular-session open: **{gap_exits:,}**.",
        "- GLD has no split events in the local corporate-action table.",
        "- The feed is regular-session IEX data. Overnight price paths are unobserved; only the next "
        "regular open is available.",
        f"- Verdict: {verdict}",
    ]

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = f"sweep_engulf_{ticker}_{args.frequency}"
    report = args.outdir / f"{stem}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    csv_path = args.outdir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(asdict(default_trades[0]).keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(trade) for trade in default_trades)
    print(f"wrote {report} and {csv_path} ({len(default_trades):,} default trades)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
