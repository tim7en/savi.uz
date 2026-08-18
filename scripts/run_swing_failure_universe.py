"""Run the locked swing-failure rule across every intraday ticker at once.

Testing one rule on many assets is a multiple-comparison problem, so the primary
result here is the pooled sample, not the best-looking ticker.  Uncertainty is
estimated with a bootstrap that resamples whole *sessions* rather than
individual trades, because trades taken on the same day across correlated
instruments are not independent observations.
"""

from __future__ import annotations

import argparse
import collections
import csv
import random
import sqlite3
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
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
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--split", default="2023-01-01")
    # Pinned so a download job appending today's bars cannot move the result.
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--min-sessions", type=int, default=500)
    parser.add_argument("--outdir", type=Path, default=Path("out/strategy"))
    return parser.parse_args(argv)


def tickers(path: Path) -> list[str]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return [
            row[0] for row in connection.execute(
                "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker"
            )
        ]
    finally:
        connection.close()


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


def profit_factor(returns: list[float]) -> float:
    gains = sum(value for value in returns if value > 0)
    losses = -sum(value for value in returns if value < 0)
    return gains / losses if losses else float("inf")


def session_block_bootstrap(trades, draws: int = 2000, seed: int = 20240817):
    """Resample whole sessions, so same-day cross-asset trades move together."""
    by_session: dict[str, list[float]] = collections.defaultdict(list)
    for trade in trades:
        by_session[trade.session].append(trade.net_return)
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
    factors.sort()
    if not factors:
        return None
    return (
        factors[int(0.05 * len(factors))],
        factors[min(int(0.95 * len(factors)), len(factors) - 1)],
        positives / draws,
    )


def binomial_tails(successes: int, trials: int) -> tuple[float, float]:
    """Exact (P at least this many, P at most this many) under a fair coin."""
    from math import comb
    upper = sum(comb(trials, k) for k in range(successes, trials + 1)) / 2 ** trials
    lower = sum(comb(trials, k) for k in range(0, successes + 1)) / 2 ** trials
    return upper, lower


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = SfpConfig()
    splits = load_splits(args.db)
    per_ticker = {}
    skipped = []
    all_trades = []

    for ticker in tickers(args.db):
        source = adjust_bars(load_bars(args.db, ticker), splits.get(ticker, []))
        if not source:
            continue
        daily = resample_regular_session(source, minutes=390)
        if len(daily) < args.min_sessions:
            skipped.append((ticker, len(daily)))
            continue
        hourly = resample_regular_session(source, minutes=60)
        fifteen = resample_regular_session(source, minutes=15)
        trades, _ = run_sfp_strategy(
            daily, hourly, fifteen, config=config, start=args.start, end=args.end
        )
        per_ticker[ticker] = trades
        for trade in trades:
            all_trades.append((ticker, trade))
        print(f"  {ticker:6s} {len(daily):5d} sessions -> {len(trades):4d} trades", flush=True)

    flat = [trade for _, trade in all_trades]
    train = [t for t in flat if t.session < args.split]
    test = [t for t in flat if t.session >= args.split]

    lines = [
        "# Swing-failure rule across the intraday universe", "",
        f"The locked rule, unchanged, applied to **{len(per_ticker)}** instruments with at least "
        f"**{args.min_sessions}** sessions, over **{args.start}** to **{args.end}**. "
        f"Chronological split: **{args.split}**.", "",
        "The end date is pinned. The bar database is appended to by a live download job, so an "
        "unpinned run is not reproducible: two runs minutes apart returned different trade "
        "counts before this was fixed.", "",
        "One rule tested on many instruments is a multiple-comparison problem. The primary result "
        "is therefore the pooled sample and the cross-sectional hit rate, not the best ticker. "
        "Bootstrap intervals resample whole sessions rather than individual trades, because trades "
        "taken on the same day across correlated instruments are not independent.", "",
        "## Pooled result", "",
        "| Period | Instruments | Trades | Win | PF | PF 5-95% | P(PF > 1) | Mean R | Sum R |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for period, subset in (("train", train), ("test", test), ("all", flat)):
        item = summarise_sfp(subset)
        if not item.count:
            lines.append(f"| {period} | 0 | 0 | — | — | — | — | — | — |")
            continue
        interval = session_block_bootstrap(subset)
        span = f"{interval[0]:.2f} - {interval[1]:.2f}" if interval else "—"
        share = f"{interval[2]:.0%}" if interval else "—"
        lines.append(
            f"| {period} | {len(per_ticker)} | {item.count} | {pct(item.win_rate)} | "
            f"{item.profit_factor:.2f} | {span} | {share} | {item.mean_r:+.3f} | "
            f"{sum(t.net_r for t in subset):+.1f} |"
        )

    winners = sum(
        1 for trades in per_ticker.values()
        if trades and profit_factor([t.net_return for t in trades]) > 1.0
    )
    tested = sum(1 for trades in per_ticker.values() if trades)
    lines += [
        "", "## Cross-sectional hit rate", "",
        f"Instruments with a full-sample profit factor above 1.0: **{winners} of {tested}**, "
        f"against the **{tested / 2:.1f}** a coin flip would produce.", "",
        f"- Evidence the rule beats chance, P(at least {winners} winners): "
        f"**{binomial_tails(winners, tested)[0]:.2f}**. Nothing above 0.05 here would be "
        "surprising under the null.",
        f"- Evidence the rule is worse than chance, P(at most {winners} winners): "
        f"**{binomial_tails(winners, tested)[1]:.3f}**.", "",
        "A single instrument clearing 1.0 is not evidence. The question this table answers is "
        "whether the *distribution* sits above a coin flip.", "",
        "## Per-instrument detail", "",
        "| Ticker | Trades | Train n | Train PF | Test n | Test PF | All PF | Win | Mean R | Sum R |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = sorted(
        per_ticker.items(),
        key=lambda row: profit_factor([t.net_return for t in row[1]]) if row[1] else -1.0,
        reverse=True,
    )
    for ticker, trades in ordered:
        if not trades:
            lines.append(f"| {ticker} | 0 | — | — | — | — | — | — | — | — |")
            continue
        tr = [t for t in trades if t.session < args.split]
        te = [t for t in trades if t.session >= args.split]
        item = summarise_sfp(trades)
        lines.append(
            f"| {ticker} | {len(trades)} | {len(tr)} | "
            f"{profit_factor([t.net_return for t in tr]):.2f} | {len(te)} | "
            f"{profit_factor([t.net_return for t in te]):.2f} | "
            f"{item.profit_factor:.2f} | {pct(item.win_rate)} | {item.mean_r:+.3f} | "
            f"{sum(t.net_r for t in trades):+.1f} |"
        )

    lines += [
        "", "### Does a good instrument stay good?", "",
        "If the rule had a real per-instrument edge, train-period ranking would carry information "
        "about test-period ranking. This is the correlation between the two.", "",
    ]
    paired = [
        (profit_factor([t.net_return for t in trades if t.session < args.split]),
         profit_factor([t.net_return for t in trades if t.session >= args.split]))
        for trades in per_ticker.values()
        if len([t for t in trades if t.session < args.split]) >= 10
        and len([t for t in trades if t.session >= args.split]) >= 10
    ]
    paired = [(a, b) for a, b in paired if a != float("inf") and b != float("inf")]
    if len(paired) >= 5:
        correlation = statistics.correlation([a for a, _ in paired], [b for _, b in paired])
        lines.append(
            f"Across **{len(paired)}** instruments with at least ten trades in each period, the "
            f"correlation between train and test profit factor is **{correlation:+.2f}**. A value "
            "near zero means last period's winners tell you nothing about next period's."
        )
    else:
        lines.append("Too few instruments carry ten trades in both periods to correlate.")

    for label, key in (("side", lambda t: "long" if t.bias > 0 else "short"),
                       ("daily state", lambda t: t.bias_state)):
        groups = collections.defaultdict(list)
        for trade in flat:
            groups[key(trade)].append(trade)
        lines += [
            "", f"### Pooled by {label}", "",
            f"| {label.title()} | Train n | Train PF | Test n | Test PF | All PF | Sum R |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name, subset in sorted(groups.items(), key=lambda row: -len(row[1])):
            tr = [t for t in subset if t.session < args.split]
            te = [t for t in subset if t.session >= args.split]
            lines.append(
                f"| {name} | {len(tr)} | {profit_factor([t.net_return for t in tr]):.2f} | "
                f"{len(te)} | {profit_factor([t.net_return for t in te]):.2f} | "
                f"{profit_factor([t.net_return for t in subset]):.2f} | "
                f"{sum(t.net_r for t in subset):+.1f} |"
            )

    if skipped:
        lines += [
            "", "## Excluded for short history", "",
            ", ".join(f"{ticker} ({count} sessions)" for ticker, count in skipped) + ".",
        ]

    args.outdir.mkdir(parents=True, exist_ok=True)
    report = args.outdir / "swing_failure_universe.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    csv_path = args.outdir / "swing_failure_universe_trades.csv"
    rows = [{"ticker": ticker, **asdict(trade)} for ticker, trade in all_trades]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0].keys()) if rows else ["ticker"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {report} and {csv_path} ({len(flat)} pooled trades)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
