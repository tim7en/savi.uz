"""Run the leakage-free daily-bias, hourly SFP, 15-minute execution study."""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.swing_failure_strategy import (  # noqa: E402
    SfpConfig,
    run_sfp_strategy,
    summarise_sfp,
)
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
    result = summarise_sfp(trades)
    if not result.count:
        return f"| {label} | {period} | 0 | 0/0 | — | — | — | — | — | — | — | — | — | — |"
    return (
        f"| {label} | {period} | {result.count} | {result.longs}/{result.shorts} | "
        f"{pct(result.win_rate)} | {result.profit_factor:.2f} | {result.mean_r:+.3f} | "
        f"${100 * result.ending_equity:.2f} | {pct(result.cagr)} | {pct(result.max_drawdown)} | "
        f"{pct(result.stop_rate)} | {pct(result.breakeven_rate)} | "
        f"{pct(result.target_rate)} | {pct(result.time_rate)} |"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ticker = args.ticker.upper()
    source = load_bars(args.db, ticker)
    if not source:
        raise SystemExit(f"error: no {ticker} five-minute bars")
    daily = resample_regular_session(source, minutes=390)
    hourly = resample_regular_session(source, minutes=60)
    fifteen = resample_regular_session(source, minutes=15)

    locked = SfpConfig()
    variants = [
        ("Locked A-grade PDH/PDL SFP", locked),
        ("No break-even move", replace(locked, breakeven_trigger_r=None)),
        ("One-leg daily HH/HL or LH/LL", replace(locked, daily_structure_legs=1)),
        ("Require strong prior daily close", replace(locked, require_strong_daily_close=True)),
        ("Relaxed hourly outside close", replace(locked, hourly_confirmation="outside")),
        ("Directional hourly SFP", replace(locked, hourly_confirmation="directional SFP")),
        ("Directional SFP, one-leg bias", replace(
            locked, hourly_confirmation="directional SFP", daily_structure_legs=1
        )),
        ("Directional SFP at PD or weekly level", replace(
            locked, hourly_confirmation="directional SFP", location_mode="previous day or week"
        )),
        ("Close-back SFP, strict daily bias", replace(
            locked, hourly_confirmation="close-back SFP"
        )),
        ("Close-back SFP, one-leg bias", replace(
            locked, hourly_confirmation="close-back SFP", daily_structure_legs=1
        )),
        ("No rapid-rejection filter", replace(locked, require_fast_rejection=False)),
        ("PD or prior-week SFP location", replace(locked, location_mode="previous day or week")),
        ("Hold up to three sessions", replace(locked, max_hold_sessions=3)),
        ("Minimum 1R instead of 2R", replace(locked, minimum_reward_risk=1.0)),
    ]

    results = {}
    audits = {}
    lines = [
        f"# Daily-bias swing-failure strategy - {ticker}", "",
        f"Source: **{len(source):,}** Tiingo five-minute regular-session bars, resampled to "
        f"**{len(daily):,}** daily, **{len(hourly):,}** hourly, and **{len(fifteen):,}** "
        f"15-minute bars. Chronological split: **{args.split}**.", "",
        "## Locked A-grade rules", "",
        "1. At the session open, use only the three preceding completed daily candles. Long bias "
        "requires two consecutive higher-high/higher-low transitions; short bias requires two "
        "lower-high/lower-low transitions.",
        "2. The latest completed daily candle must close in the bias direction. An opposing candle "
        "is classified as caution/retracement and skipped; an inside/neutral structure has no bias.",
        "3. Mark untouched PDH and PDL before the session. A long setup can only sweep PDL in a bull "
        "bias; a short can only sweep PDH in a bear bias. Any earlier touch consumes the pool and "
        "invalidates that location.",
        "4. The completed hourly failure candle must trade through the level, close back inside, "
        "fully exceed the preceding hour's high and low, and close beyond the preceding hour's "
        "opposite extreme in the bias direction.",
        "5. At most one constituent 15-minute close may remain outside the swept level. This encodes "
        "a quick rejection rather than an hour of acceptance beyond it.",
        "6. Enter at the following 15-minute open. Stop at the failure wick. Target the untouched "
        "opposite prior-day extreme. The actual fill must offer at least 2R.",
        "7. After a completed 15-minute candle reaches +1R, move the stop to entry for the following "
        "bar. Exit any remainder at the regular-session close. Charge 2 bp round-trip cost.", "",
        "The primary test cannot use Asia or London highs/lows because the ETF source contains only "
        "US regular hours. Prior-week locations, relaxed confirmation, and overnight holding are "
        "reported as diagnostics rather than silently folded into the locked rule.", "",
        "## Chronological results", "",
        "| Variant | Period | Trades | L/S | Win | PF | Mean R | $100 -> | CAGR | Max DD | Stop | BE | Target | Time |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, config in variants:
        trades, audit = run_sfp_strategy(
            daily, hourly, fifteen, config=config, start=args.start
        )
        results[label], audits[label] = trades, audit
        train = [trade for trade in trades if trade.session < args.split]
        test = [trade for trade in trades if trade.session >= args.split]
        lines.append(summary_row(label, "train", train))
        lines.append(summary_row(label, "test", test))

    primary_name = "Locked A-grade PDH/PDL SFP"
    primary = results[primary_name]
    audit = audits[primary_name]
    lines += [
        "", "## Locked-rule funnel", "",
        f"- Hourly bars: **{audit.hourly_bars:,}**",
        f"- Hours carrying a strict daily trend bias: **{audit.biased_hours:,}**",
        f"- Rejected by opposing prior-day candle: **{audit.no_daily_alignment:,}**",
        f"- No still-untouched bias-aligned PDH/PDL: **{audit.no_untouched_location:,}**",
        f"- No sweep and reclaim: **{audit.no_swing_failure:,}**",
        f"- Sweep occurred but hourly confirmation was weak: **{audit.weak_hourly_confirmation:,}**",
        f"- Rejection was too slow: **{audit.slow_rejection:,}**",
        f"- Opposite daily target was no longer resting: **{audit.target_not_resting:,}**",
        f"- Invalid or below 2R: **{audit.invalid_or_low_reward:,}**",
        f"- Executed trades: **{audit.trades:,}**",
    ]

    lower_name = "Close-back SFP, strict daily bias"
    lower_train = summarise_sfp([
        trade for trade in results[lower_name] if trade.session < args.split
    ])
    lower_test = summarise_sfp([
        trade for trade in results[lower_name] if trade.session >= args.split
    ])
    lines += [
        "", "## Interpretation", "",
        f"The full A-grade conjunction produced **{len(primary)}** trades. A zero-trade result means "
        "the written rules are not presently a strategy on this instrument; it must not be reported "
        "as a high-win-rate setup.", "",
        f"The separately labelled close-back tier produced **{lower_train.count}** development trades "
        f"(PF **{lower_train.profit_factor:.2f}**) and **{lower_test.count}** holdout trades "
        f"(PF **{lower_test.profit_factor:.2f}**). It omits the outside-bar requirement and therefore "
        "is lower conviction, not a substitute definition selected after the result.", "",
        "Any attractive result based on only a handful of trades is hypothesis-generating. Cross-asset "
        "agreement and a new untouched time period are required before compounding or leverage tests.",
    ]

    lines += [
        "", "## Two-year stability of the locked rule", "",
        "| Years | Trades | L/S | PF | Mean R | $100 -> | Max DD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, first, last in (
        ("2017-18", "2017-01-01", "2019-01-01"),
        ("2019-20", "2019-01-01", "2021-01-01"),
        ("2021-22", "2021-01-01", "2023-01-01"),
        ("2023-24", "2023-01-01", "2025-01-01"),
        ("2025+", "2025-01-01", "9999-01-01"),
    ):
        item = summarise_sfp([trade for trade in primary if first <= trade.session < last])
        if not item.count:
            lines.append(f"| {label} | 0 | 0/0 | — | — | — | — |")
        else:
            lines.append(
                f"| {label} | {item.count} | {item.longs}/{item.shorts} | "
                f"{item.profit_factor:.2f} | {item.mean_r:+.3f} | "
                f"${100 * item.ending_equity:.2f} | {pct(item.max_drawdown)} |"
            )

    lines += [
        "", "## Leakage and execution audit", "",
        "- The daily bias for session D reads only daily candles ending before D.",
        "- PDH/PDL and PWH/PWL are fixed before the session; a touch is evaluated only from bars "
        "that had already completed before the signal hour.",
        "- The hourly SFP becomes actionable at its close, and entry is the next 15-minute open.",
        "- Break-even activates only for the bar after a completed bar reaches +1R.",
        "- If stop and target are both inside one OHLC bar, the stop wins. Gaps fill at the observed open.",
        "- The 2023+ period has now been observed for these variants and cannot remain a pristine "
        "holdout for the next specification.",
    ]

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = f"swing_failure_{ticker}"
    report = args.outdir / f"{stem}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    csv_path = args.outdir / f"{stem}_variants.csv"
    rows = [
        {"variant": label, **asdict(trade)}
        for label, trades in results.items()
        for trade in trades
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0].keys()) if rows else ["variant"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {report} and {csv_path} ({len(primary)} locked trades)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
