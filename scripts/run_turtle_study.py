"""Turtle breakout system at daily, 30-minute, 15-minute and five-minute bars.

Daily is the reference case the rules were written for.  The intraday runs keep
every rule identical and change only the bar interval, so any difference is
attributable to resolution rather than to a re-tuned system.

Results are pooled across instruments, because a per-instrument ranking on this
sample size has already been shown to be noise.  Uncertainty comes from a
bootstrap that resamples whole sessions, since trades taken on the same day
across correlated instruments are not independent.
"""

from __future__ import annotations

import argparse
import collections
import csv
import random
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import TurtleConfig, run_turtle, summarise_turtle  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

#: Label, resample minutes, and roughly how many bars make up one session.
INTERVALS = (
    ("daily", 390, 1),
    ("30-minute", 30, 13),
    ("15-minute", 15, 26),
    ("5-minute", 5, 78),
)

SYSTEMS = (
    ("System 1 (20/10, filtered)", dict(entry_window=20, exit_window=10, atr_window=20,
                                        skip_after_winner=True)),
    ("System 1 unfiltered (20/10)", dict(entry_window=20, exit_window=10, atr_window=20,
                                         skip_after_winner=False)),
    ("System 2 (55/20)", dict(entry_window=55, exit_window=20, atr_window=20,
                              skip_after_winner=False)),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--split", default="2023-01-01")
    # Pinned: the bar database is appended to by a live download job.
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--min-sessions", type=int, default=500)
    parser.add_argument("--outdir", type=Path, default=Path("out/strategy"))
    return parser.parse_args(argv)


def tickers(path: Path) -> list[str]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return [row[0] for row in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker"
        )]
    finally:
        connection.close()


def load_bars(path: Path, ticker: str, start: str, end: str) -> list[Bar]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars "
            "WHERE ticker=? AND frequency='5min' AND ts>=? AND ts<? ORDER BY ts",
            (ticker, start, end),
        ).fetchall()
    finally:
        connection.close()
    return [Bar(*row) for row in rows]


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def profit_factor(values: list[float]) -> float:
    gains = sum(v for v in values if v > 0)
    losses = -sum(v for v in values if v < 0)
    return gains / losses if losses else float("inf")


def session_block_bootstrap(trades, draws: int = 1000, seed: int = 20240817):
    by_session: dict[str, list[float]] = collections.defaultdict(list)
    for trade in trades:
        by_session[trade.exit_timestamp[:10]].append(trade.net_r)
    sessions = list(by_session)
    if len(sessions) < 10:
        return None
    rng = random.Random(seed)
    factors = []
    positives = 0
    for _ in range(draws):
        pick: list[float] = []
        for _ in range(len(sessions)):
            pick.extend(by_session[sessions[rng.randrange(len(sessions))]])
        value = profit_factor(pick)
        positives += value > 1.0
        if value != float("inf"):
            factors.append(value)
    if not factors:
        return None
    factors.sort()
    return (
        factors[int(0.05 * len(factors))],
        factors[min(int(0.95 * len(factors)), len(factors) - 1)],
        positives / draws,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # The bar table stores prices as printed, so an unadjusted split reads as a
    # 95% overnight collapse and corrupts every volatility and breakout measure.
    splits = load_splits(args.db)
    universe = []
    for ticker in tickers(args.db):
        source = adjust_bars(load_bars(args.db, ticker, args.start, args.end), splits.get(ticker, []))
        if not source:
            continue
        sessions = len({row.timestamp[:10] for row in source})
        if sessions < args.min_sessions:
            continue
        universe.append((ticker, source))
    print(f"{len(universe)} instruments", flush=True)

    # Resampling dominates the runtime, so every interval is built once and
    # reused by the grid, the cost sweep and the overnight comparison.
    series: dict[str, list[list[Bar]]] = {}
    for label, minutes, _ in INTERVALS:
        series[label] = [
            resample_regular_session(source, minutes=minutes) for _, source in universe
        ]
        total = sum(len(bars) for bars in series[label])
        print(f"  resampled {label:10s} {total:9,d} bars", flush=True)

    results: dict[tuple[str, str], list] = {}
    audits: dict[tuple[str, str], tuple[int, int, int]] = {}
    for label, minutes, _ in INTERVALS:
        for system, overrides in SYSTEMS:
            config = TurtleConfig(**overrides)
            pooled = []
            breakouts = small_n = filtered = 0
            for bars in series[label]:
                trades, audit = run_turtle(bars, config=config)
                pooled.extend(trades)
                breakouts += audit.breakouts
                small_n += audit.skipped_small_n
                filtered += audit.skipped_after_winner
            results[(label, system)] = pooled
            audits[(label, system)] = (breakouts, filtered, small_n)
            print(f"  {label:10s} {system:28s} {len(pooled):6d} trades", flush=True)

    lines = [
        "# Turtle breakout system across bar intervals", "",
        f"The published rules, unchanged, over **{len(universe)}** instruments from "
        f"**{args.start}** to **{args.end}**. Only the bar interval differs between rows: "
        "the same 20/10 and 55/20 channels, the same Wilder N, the same 2N stop, the same "
        "half-N pyramid to four units, and the same 2 bp round trip charged per unit.", "",
        "## What changes when the bars get shorter", "",
        "A 20-bar channel means twenty *days* on the daily chart and one hundred *minutes* on "
        "five-minute bars. The rules are scale free but the costs are not: N shrinks roughly with "
        "the square root of the interval while the round trip stays fixed, so the same 2 bp is a "
        "far larger share of a 1N move at five minutes than at one day. The cost column below is "
        "expressed in R so that this is directly visible.", "",
        "| Interval | System | Trades | Units | Win | PF | Mean R | Total R | Cost R/trade | "
        "Cost share of gross | Mean bars held |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, minutes, _ in INTERVALS:
        for system, _overrides in SYSTEMS:
            trades = results[(label, system)]
            item = summarise_turtle(trades)
            if not item.trades:
                lines.append(f"| {label} | {system} | 0 | | | | | | | | |")
                continue
            cost = sum(t.cost_r for t in trades)
            gross = sum(abs(t.gross_r) for t in trades)
            lines.append(
                f"| {label} | {system} | {item.trades:,} | {item.units:,} | "
                f"{pct(item.win_rate)} | {item.profit_factor:.2f} | {item.mean_r:+.3f} | "
                f"{item.total_r:+,.0f} | {cost / item.trades:.3f} | "
                f"{pct(cost / gross) if gross else '—'} | {item.mean_bars_held:.1f} |"
            )

    lines += [
        "", "## Is the daily reference case actually positive?", "",
        "The interval comparison is only meaningful if the system works where it is supposed to. "
        "This is the daily case with a session-block bootstrap.", "",
        "| System | Period | Trades | PF | PF 5-95% | P(PF > 1) | Mean R | Total R |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for system, _overrides in SYSTEMS:
        trades = results[("daily", system)]
        for period, subset in (
            ("train", [t for t in trades if t.exit_timestamp < args.split]),
            ("test", [t for t in trades if t.exit_timestamp >= args.split]),
            ("all", trades),
        ):
            item = summarise_turtle(subset)
            if not item.trades:
                lines.append(f"| {system} | {period} | 0 | — | — | — | — | — |")
                continue
            interval = session_block_bootstrap(subset)
            span = f"{interval[0]:.2f} - {interval[1]:.2f}" if interval else "—"
            share = f"{interval[2]:.0%}" if interval else "—"
            lines.append(
                f"| {system} | {period} | {item.trades:,} | {item.profit_factor:.2f} | "
                f"{span} | {share} | {item.mean_r:+.3f} | {item.total_r:+,.0f} |"
            )

    lines += [
        "", "## Breakouts rejected because N had collapsed", "",
        "Wilder's N decays geometrically through bars with no range, so a long dead stretch drives "
        "it towards zero. Because R is measured per N, such a trade reports an arbitrarily large "
        "multiple that is an artefact of the divisor rather than a result. A breakout is therefore "
        "only taken when a 1N move can pay for at least five round trips.", "",
        "| Interval | Breakouts | Skipped after a winner | Skipped for collapsed N | Taken |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, minutes, _ in INTERVALS:
        breakouts, filtered, small_n = audits[(label, "System 1 (20/10, filtered)")]
        taken = len(results[(label, "System 1 (20/10, filtered)")])
        lines.append(
            f"| {label} | {breakouts:,} | {filtered:,} | {small_n:,} "
            f"({small_n / max(breakouts, 1):.1%}) | {taken:,} |"
        )

    lines += [
        "", "## Cost sensitivity", "",
        "Total R for System 2 at each interval, as the round trip is varied. Zero cost is not a "
        "tradeable assumption; it is there to separate the signal from the friction.", "",
        "| Interval | 0 bp | 1 bp | 2 bp | 5 bp | 10 bp |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, minutes, _ in INTERVALS:
        # One pass with the N floor pinned to an absolute fraction, so varying
        # the commission re-prices the same trades instead of also reselecting
        # them. Cost is linear in the rate, so it is applied afterwards.
        pooled = []
        for bars in series[label]:
            trades, _ = run_turtle(bars, config=TurtleConfig(
                entry_window=55, exit_window=20, atr_window=20,
                skip_after_winner=False, round_trip_cost=0.0,
                minimum_n_cost_multiple=0.0, minimum_n_fraction=0.001,
            ))
            pooled.extend(trades)
        gross = sum(t.gross_r for t in pooled)
        basis = sum(t.cost_basis_r for t in pooled)
        row = [f"| {label} "]
        for cost in (0.0, 0.0001, 0.0002, 0.0005, 0.001):
            row.append(f"| {gross - cost * basis:+,.0f} ")
        lines.append("".join(row) + "|")

    lines += [
        "", "## Holding an intraday position overnight", "",
        "The daily rules hold until the exit channel is breached, however long that takes. On "
        "intraday bars that means carrying risk through a close the data cannot see, since this "
        "source is US regular hours only. The alternative is to flatten at the close and give up "
        "the overnight portion of every trend.", "",
        "| Interval | Overnight held | Trades | PF | Total R | Flat at close | Trades | PF | Total R |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for label, minutes, _ in INTERVALS:
        if label == "daily":
            continue
        row = [f"| {label} "]
        for allow in (True, False):
            pooled = []
            for bars in series[label]:
                trades, _ = run_turtle(bars, config=TurtleConfig(
                    entry_window=55, exit_window=20, atr_window=20,
                    skip_after_winner=False, allow_overnight=allow,
                ))
                pooled.extend(trades)
            item = summarise_turtle(pooled)
            tag = "yes" if allow else "no"
            row.append(
                f"| {tag} | {item.trades:,} | {item.profit_factor:.2f} | {item.total_r:+,.0f} "
            )
        lines.append("".join(row) + "|")

    lines += [
        "", "## Reading this", "",
        "- Costs are charged per unit, so a four-unit pyramid pays four round trips.",
        "- Entries and exits are stop orders filled at the channel edge, or at the open when the "
        "bar gapped past it. No fill is ever better than the level that triggered it.",
        "- Inside a single bar the stop is assumed to trade before the favourable exit.",
        "- The System 1 filter resolves each prior breakout bar by bar, so a skip decision never "
        "reads a bar that had not yet happened.",
        "- Positions still open when the data ends are closed at the final close and marked "
        "`end of data` rather than dropped.",
        "- R is measured against N at entry, so it is comparable across intervals even though the "
        "absolute size of a 1N move is not.",
    ]

    args.outdir.mkdir(parents=True, exist_ok=True)
    report = args.outdir / "turtle_intervals.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    csv_path = args.outdir / "turtle_trades.csv"
    rows = [
        {"interval": label, "system": system, **asdict(trade)}
        for (label, system), trades in results.items()
        for trade in trades
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0].keys()) if rows else ["interval"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {report} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
