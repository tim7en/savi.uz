"""Run the leakage-free daily-bias, hourly SFP, 15-minute execution study."""

from __future__ import annotations

import argparse
import collections
import csv
import sqlite3
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.swing_failure_strategy import (  # noqa: E402
    SfpConfig,
    build_daily_biases,
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
        ("Core SFP (locked)", locked),
        ("Core SFP, no break-even", replace(locked, breakeven_trigger_r=None)),
        ("Core SFP, trend sessions only", replace(locked, trade_neutral_sessions=False)),
        ("Core SFP, strong daily states only", replace(locked, require_strong_daily_close=True)),
        ("Core SFP, opening hour excluded", replace(locked, allow_opening_hour=False)),
        ("Core SFP, hold up to three sessions", replace(locked, max_hold_sessions=3)),
        ("Core SFP at PD or weekly level", replace(locked, location_mode="previous day or week")),
        ("Core SFP, no rapid-rejection filter", replace(locked, require_fast_rejection=False)),
        ("Core SFP, minimum 1R instead of 2R", replace(locked, minimum_reward_risk=1.0)),
        ("Core SFP, no minimum stop distance",
         replace(locked, minimum_stop_cost_multiple=0.0)),
        ("Confluence: + directional body", replace(locked, hourly_confirmation="directional")),
        ("Confluence: + close through prior open",
         replace(locked, hourly_confirmation="close through open")),
        ("Confluence: + outside bar", replace(locked, hourly_confirmation="outside")),
        ("Superseded: strong-outside conjunction",
         replace(locked, hourly_confirmation="strong outside")),
        ("Superseded: previous locked rule", replace(
            locked, bias_mode="three-candle legs", daily_structure_legs=2,
            hourly_confirmation="strong outside", trade_neutral_sessions=False,
            allow_opening_hour=False,
        )),
    ]

    results = {}
    audits = {}
    lines = [
        f"# Daily-bias swing-failure strategy - {ticker}", "",
        f"Source: **{len(source):,}** Tiingo five-minute regular-session bars, resampled to "
        f"**{len(daily):,}** daily, **{len(hourly):,}** hourly, and **{len(fifteen):,}** "
        f"15-minute bars. Chronological split: **{args.split}**.", "",
        "## Locked rules", "",
        "1. Before the session, classify the two completed daily candles that precede it. A close "
        "beyond the prior candle's extreme after a higher-high/higher-low or lower-high/lower-low "
        "transition is a strong state; the transition without that close is a weak state; a "
        "transition that closes against itself is caution and is not traded; an inside candle is "
        "neutral. An outside bar is resolved only by its close.",
        "2. Strong and weak states supply the lean. A neutral session has no lean, so whichever "
        "pool is raided first sets the side. Caution sessions are skipped.",
        "3. Mark untouched PDH and PDL before the session. A long sweeps PDL, a short sweeps PDH. "
        "Any earlier touch consumes the pool and invalidates that location.",
        "4. The failure is defined against the level, not the candle body: the completed hourly bar "
        "must trade through the pool and close back on the origin side of it. Hourly two-candle "
        "patterns are recorded as confluence and reported as separate tiers.",
        "5. At most one constituent 15-minute close may remain outside the swept level, which "
        "encodes a quick rejection rather than an hour of acceptance beyond it.",
        "6. Enter at the following 15-minute open. Stop at the failure wick. Target the untouched "
        "opposite prior-day extreme. The actual fill must offer at least 2R.",
        "7. After a completed 15-minute candle reaches +1R, move the stop to entry for the following "
        "bar. Exit any remainder at the regular-session close. Charge 2 bp round-trip cost.",
        "8. Reject any setup whose stop sits closer than five round trips (10 bp) from the entry "
        "fill. Such a stop cannot pay for its own costs, and the R multiples it produces are "
        "arithmetic artefacts rather than trade outcomes.", "",
        "The opening hour is a valid signal hour: it carries the majority of first raids on PDH and "
        "PDL, and the level-defined failure needs no preceding same-session bar. Confluence tiers "
        "that compare against a prior hour cannot be evaluated there and skip it.", "",
        "This study cannot use Asia or London highs and lows because the ETF source contains only US "
        "regular hours. That matters most in strongly trending sessions, where the prior-day extreme "
        "on the far side is the least likely pool to be raided.", "",
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

    primary_name = "Core SFP (locked)"
    primary = results[primary_name]
    audit = audits[primary_name]
    lines += [
        "", "## Locked-rule funnel", "",
        f"- Hourly bars: **{audit.hourly_bars:,}**",
        f"- Candidate signal hours in range: **{audit.candidate_hours:,}**",
        f"- No completed two-candle daily record yet: **{audit.no_bias_record:,}**",
        f"- Caution session, not traded: **{audit.caution_session:,}**",
        f"- Neutral session skipped by configuration: **{audit.neutral_skipped:,}**",
        f"- Rejected by daily candle alignment: **{audit.no_daily_alignment:,}**",
        f"- Rejected by daily state strength: **{audit.no_daily_strength:,}**",
        f"- Opening hour not usable by this tier: **{audit.opening_hour_skipped:,}**",
        f"- Last hour of session, no entry bar follows: **{audit.no_entry_bar:,}**",
        f"- Hourly bar not aligned to the 15-minute grid: **{audit.unaligned_15m_grid:,}**",
        f"- No still-untouched bias-aligned PDH/PDL: **{audit.no_untouched_location:,}**",
        f"- No sweep and reclaim: **{audit.no_swing_failure:,}**",
        f"- Sweep occurred but hourly confirmation was weak: **{audit.weak_hourly_confirmation:,}**",
        f"- Rejection was too slow: **{audit.slow_rejection:,}**",
        f"- Opposite daily target was no longer resting: **{audit.target_not_resting:,}**",
        f"- Stop too tight to be tradeable: **{audit.stop_too_tight:,}**",
        f"- Invalid or below 2R: **{audit.invalid_or_low_reward:,}**",
        f"- Overlapped an open position: **{audit.overlap_skipped:,}**",
        f"- Executed trades: **{audit.trades:,}**",
        "",
        f"Buckets sum to **{audit.accounted:,}** against **{audit.candidate_hours:,}** candidate "
        f"hours: **{'reconciles' if audit.reconciles() else 'DOES NOT RECONCILE'}**.",
    ]

    biases = build_daily_biases(daily, locked)
    in_range = {
        session: bias for session, bias in biases.items() if session >= args.start
    }
    distribution = collections.Counter(bias.state for bias in in_range.values())
    traded = collections.Counter(trade.bias_state for trade in primary)
    lines += [
        "", "## Daily state distribution and where the trades come from", "",
        "| Daily state | Sessions | Share | Locked trades |",
        "|---|---:|---:|---:|",
    ]
    for state, count in distribution.most_common():
        share = count / len(in_range) if in_range else 0.0
        lines.append(f"| {state} | {count:,} | {pct(share)} | {traded.get(state, 0)} |")

    confluence = collections.Counter()
    for trade in primary:
        for tag in ("directional", "close through open", "outside", "close beyond extreme"):
            if tag in trade.confluence:
                confluence[tag] += 1
    lines += [
        "", "## Confluence carried by the locked trades", "",
        "| Hourly confluence on the failure candle | Trades | Share |",
        "|---|---:|---:|",
    ]
    for tag in ("directional", "close through open", "outside", "close beyond extreme"):
        count = confluence.get(tag, 0)
        share = count / len(primary) if primary else 0.0
        lines.append(f"| {tag} | {count} | {pct(share)} |")
    hour_counts = collections.Counter(trade.signal_session_hour for trade in primary)
    lines += [
        "",
        "Signal hour within the session: "
        + ", ".join(f"H{hour}={hour_counts[hour]}" for hour in sorted(hour_counts))
        + ".",
    ]

    train_summary = summarise_sfp([t for t in primary if t.session < args.split])
    test_summary = summarise_sfp([t for t in primary if t.session >= args.split])
    superseded = results["Superseded: previous locked rule"]
    lines += [
        "", "## Interpretation", "",
        f"The locked rule produced **{train_summary.count}** development trades "
        f"(PF **{train_summary.profit_factor:.2f}**, mean **{train_summary.mean_r:+.3f}R**) and "
        f"**{test_summary.count}** holdout trades (PF **{test_summary.profit_factor:.2f}**, mean "
        f"**{test_summary.mean_r:+.3f}R**).", "",
        f"The previous specification produced **{len(superseded)}** trades. It required the hourly "
        "failure candle to be an outside bar that also closed beyond the preceding hour's opposite "
        "extreme, and it excluded both neutral sessions and the opening hour. The close-beyond-extreme "
        "term belongs to the daily two-candle bias read, not to the entry trigger, and stacking it on "
        "a level sweep made the conjunction unsatisfiable.", "",
        "Confluence tiers are reported so the cost of each added hourly condition is visible. They "
        "are diagnostics: the tier with the best statistics must not be relabelled as the locked rule "
        "afterwards.", "",
        "Any result on a few dozen trades is hypothesis-generating. Cross-asset agreement and a "
        "genuinely untouched period are required before compounding or leverage tests.",
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
        "- The daily state for session D reads only the two daily candles completed before D.",
        "- PDH/PDL and PWH/PWL are fixed before the session; a touch is evaluated only from bars "
        "that had already completed before the signal hour.",
        "- On a neutral session the side is set by the pool the market actually raids, which is "
        "known at the close of the failure candle, not in advance.",
        "- The hourly SFP becomes actionable at its close, and entry is the next 15-minute open.",
        "- Break-even activates only for the bar after a completed bar reaches +1R.",
        "- If stop and target are both inside one OHLC bar, the stop wins. Gaps fill at the observed open.",
        "- Every candidate hour is bucketed exactly once, so the funnel above sums to the candidate count.",
        "- The 2023+ period has been observed for earlier specifications of this study and is no "
        "longer a pristine holdout. Treat the split above as development-versus-review, and reserve "
        "a later period or a further instrument for a genuine out-of-sample test.",
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
