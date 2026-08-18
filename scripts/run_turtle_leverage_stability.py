"""Build leveraged Turtle P&L curves with macro risk overlays.

The account starts with $1,000: $300 is an untouched sleeve and $700 is the
trading NAV. Positions use Turtle inverse-volatility sizing (1% of trading NAV
per 1N unit), subject to six-position capacity and a 2x aggregate gross-notional
cap. Regime overlays halve a new position's risk. Curves mark P&L when trades
exit; this is reproducible from the trade ledger but understates intratrade
drawdown because TurtleTrade does not store every unit's daily mark.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sqlite3
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_macro_regime_study import (  # noqa: E402
    combine, factset_rows, lagged_align, policy_target_rows, recent_increase, series,
)
from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import TurtleConfig, run_turtle  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402


INTERVALS = (
    ("30-minute", 30, dict(use_channel_exit=False, chandelier_atr=5.0)),
    ("1-hour", 60, dict(use_channel_exit=False, chandelier_atr=5.0)),
    ("2-hour", 120, dict(use_channel_exit=False, chandelier_atr=5.0)),
    ("daily", 390, dict(exit_window=50)),
)
VARIANTS = (
    "No overlay",
    "Hike + VIX stress half-risk",
    "Earnings deterioration half-risk",
    "Forward P/E premium half-risk",
    "Any preferred overlay half-risk",
)
COLORS = {
    "No overlay": "#111827",
    "Hike + VIX stress half-risk": "#dc2626",
    "Earnings deterioration half-risk": "#2563eb",
    "Forward P/E premium half-risk": "#7c3aed",
    "Any preferred overlay half-risk": "#059669",
}


@dataclass(frozen=True)
class TaggedTrade:
    ticker: str
    entry: str
    exit: str
    sessions: int
    net_r: float
    initial_basis_r: float


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--macro", type=Path, default=Path("data/macro/macro.db"))
    parser.add_argument("--equity", type=Path, default=Path("data/equity/equity.db"))
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--split", default="2023-01-01")
    parser.add_argument("--min-sessions", type=int, default=500)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--broker-spread", type=float, default=0.015)
    parser.add_argument("--outdir", type=Path, default=Path("out/strategy"))
    return parser.parse_args(argv)


def tickers(path):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return [row[0] for row in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker"
        )]
    finally:
        connection.close()


def load_bars(path, ticker, start, end):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? "
            "AND frequency='5min' AND ts>=? AND ts<? ORDER BY ts",
            (ticker, start, end),
        ).fetchall()
    finally:
        connection.close()
    return [Bar(*row) for row in rows]


def build_regimes(days, macro, equity):
    target, _ = lagged_align(days, policy_target_rows(macro))
    vix, _ = lagged_align(days, series(macro, "VIXCLS"), max_age_days=10)
    stress = combine(
        recent_increase(target, 63),
        [None if value is None else value >= 20 for value in vix],
    )
    facts = factset_rows(equity)
    aligned, _ = lagged_align(days, facts, max_age_days=21)
    earnings = [
        None if item is None else (
            item["revision"] is not None and item["revision"] <= -2
        ) for item in aligned
    ]
    valuation = [
        None if item is None else (
            item["pe_premium"] is not None and item["pe_premium"] >= 2
        ) for item in aligned
    ]
    return {
        "stress": dict(zip(days, stress)),
        "earnings": dict(zip(days, earnings)),
        "valuation": dict(zip(days, valuation)),
    }


def multiplier(variant, day, regimes):
    flags = {
        "Hike + VIX stress half-risk": regimes["stress"].get(day) is True,
        "Earnings deterioration half-risk": regimes["earnings"].get(day) is True,
        "Forward P/E premium half-risk": regimes["valuation"].get(day) is True,
    }
    if variant == "No overlay":
        return 1.0
    if variant == "Any preferred overlay half-risk":
        return 0.5 if any(flags.values()) else 1.0
    return 0.5 if flags[variant] else 1.0


def funding_map(days, macro):
    values, _ = lagged_align(days, series(macro, "DFF"), max_age_days=10)
    return dict(zip(days, values))


def replay(trades, variant, regimes, funding, calendar, *, max_positions, seed,
           broker_spread):
    entries = defaultdict(list)
    exits = defaultdict(list)
    for trade in trades:
        entries[trade.entry].append(trade)
    # Active values are (trade, reserved gross, dollars per R, financing).
    active = {}
    trading_nav = 700.0
    sleeve = 300.0
    accepted = 0
    weighted_multiplier = 0.0
    events = sorted(set(entries) | {trade.exit for trade in trades})
    curve_events = {}
    rng = random.Random(seed)
    for stamp in events:
        # Positions exiting on this timestamp free capacity before new entries.
        closing = [key for key, item in active.items() if item[0].exit <= stamp]
        for key in sorted(closing):
            trade, _reserved_gross, risk_per_r, financing = active.pop(key)
            trading_nav = max(
                0.0, trading_nav + risk_per_r * trade.net_r - financing,
            )
        candidates = list(entries.get(stamp, ()))
        rng.shuffle(candidates)
        for trade in candidates:
            if len(active) >= max_positions or trading_nav <= 0:
                continue
            day = trade.entry[:10]
            size = multiplier(variant, day, regimes)
            # One Turtle unit risks 1% of trading NAV for a 1N adverse move.
            # Reserve gross capacity for the known four-unit maximum using only
            # entry-time price and ATR. Using the trade's eventual unit count or
            # cost basis here would leak the future success of its pyramids.
            requested_risk_per_r = trading_nav * 0.01 * size
            requested = requested_risk_per_r * trade.initial_basis_r * 4
            active_gross = sum(item[1] for item in active.values())
            gross_room = max(0.0, 2.0 * trading_nav - active_gross)
            notional = min(requested, gross_room)
            if notional <= 0:
                continue
            allocation_scale = notional / requested
            risk_per_r = requested_risk_per_r * allocation_scale
            rate = (funding.get(day) or 0.0) / 100.0 + broker_spread
            borrowed_before = max(active_gross - trading_nav, 0.0)
            borrowed_after = max(active_gross + notional - trading_nav, 0.0)
            incremental_borrowing = borrowed_after - borrowed_before
            financing = incremental_borrowing * rate * trade.sessions / 252.0
            key = (trade.ticker, trade.entry, accepted)
            active[key] = (trade, notional, risk_per_r, financing)
            accepted += 1
            weighted_multiplier += size
        curve_events[stamp[:10]] = sleeve + trading_nav
    # All input trades close by end-of-data, but keep the sleeve segregated if not.
    path = []
    value = 1000.0
    for day in calendar:
        if day in curve_events:
            value = curve_events[day]
        path.append(value)
    # NOTE: trading_nav only moves when a trade closes, so `path` is a step
    # function between exits. Open-position value is invisible to it. Every risk
    # metric below therefore describes the realised-P&L path, NOT a mark-to-
    # market equity curve: the drawdown is a lower bound, and the Sharpe is not
    # comparable to one computed on a marked path (exit marking both omits
    # open-position variance and concentrates P&L into fewer, larger jumps, so
    # the direction of the bias is not even predictable). Use the five-minute
    # path replay for risk metrics; these are for return accounting only.
    peak = path[0]
    maxdd = 0.0
    for value in path:
        peak = max(peak, value)
        maxdd = min(maxdd, value / peak - 1.0)
    years = max((len(calendar) - 1) / 252.0, 1 / 252.0)
    cagr = (path[-1] / path[0]) ** (1.0 / years) - 1.0
    daily_returns = [path[index] / path[index - 1] - 1.0
                     for index in range(1, len(path))]
    volatility = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 0.0
    sharpe = (statistics.mean(daily_returns) / volatility * math.sqrt(252.0)
              if volatility else math.nan)
    calmar = cagr / abs(maxdd) if maxdd else math.nan
    return {
        "path": path, "ending": path[-1], "cagr": cagr, "maxdd": maxdd,
        "sharpe": sharpe, "calmar": calmar, "metrics_basis": "exit-marked",
        "trades": accepted,
        "mean_size": weighted_multiplier / accepted if accepted else math.nan,
    }


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(round(q * (len(ordered) - 1)), len(ordered) - 1)]


def period_calendar(calendar, start, end):
    return [day for day in calendar if start <= day <= end]


def period_trades(trades, start, end):
    # Exclude boundary-crossing trades rather than valuing an open position at
    # an unknown period-end mark.
    return [trade for trade in trades
            if start <= trade.entry[:10] and trade.exit[:10] <= end]


def svg_chart(path, calendar, curves):
    width, height = 1320, 900
    margin_x, top = 70, 100
    panel_w, panel_h = 570, 325
    gap_x, gap_y = 80, 70
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="60" y="42" font-family="Arial" font-size="26" font-weight="700">'
        '$1,000 Turtle portfolio: $300 sleeve + $700 trading at up to 2x</text>',
    ]
    legend_x = 60
    for variant in VARIANTS:
        chunks += [
            f'<line x1="{legend_x}" y1="70" x2="{legend_x + 24}" y2="70" '
            f'stroke="{COLORS[variant]}" stroke-width="3"/>',
            f'<text x="{legend_x + 30}" y="75" font-family="Arial" font-size="12">'
            f'{escape(variant)}</text>',
        ]
        legend_x += 245
    for panel, (interval, _minutes, _exit) in enumerate(INTERVALS):
        col, row = panel % 2, panel // 2
        x0 = margin_x + col * (panel_w + gap_x)
        y0 = top + row * (panel_h + gap_y)
        values = [value for variant in VARIANTS for value in curves[(interval, variant)]]
        low, high = min(values), max(values)
        pad = max((high - low) * 0.08, 25)
        low, high = max(0, low - pad), high + pad
        chunks += [
            f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" '
            'fill="#f9fafb" stroke="#d1d5db"/>',
            f'<text x="{x0}" y="{y0 - 12}" font-family="Arial" font-size="18" '
            f'font-weight="700">{escape(interval)}</text>',
            f'<text x="{x0 + 5}" y="{y0 + 18}" font-family="Arial" font-size="11" '
            f'fill="#6b7280">${high:,.0f}</text>',
            f'<text x="{x0 + 5}" y="{y0 + panel_h - 6}" font-family="Arial" font-size="11" '
            f'fill="#6b7280">${low:,.0f}</text>',
        ]
        for variant in VARIANTS:
            values = curves[(interval, variant)]
            points = []
            stride = max(1, len(values) // 500)
            for index in range(0, len(values), stride):
                x = x0 + panel_w * index / max(len(values) - 1, 1)
                y = y0 + panel_h * (high - values[index]) / (high - low)
                points.append(f"{x:.1f},{y:.1f}")
            chunks.append(
                f'<polyline fill="none" stroke="{COLORS[variant]}" stroke-width="2" '
                f'points="{" ".join(points)}"/>'
            )
        chunks += [
            f'<text x="{x0}" y="{y0 + panel_h + 18}" font-family="Arial" font-size="11" '
            f'fill="#6b7280">{calendar[0][:4]}</text>',
            f'<text x="{x0 + panel_w - 28}" y="{y0 + panel_h + 18}" '
            f'font-family="Arial" font-size="11" fill="#6b7280">{calendar[-1][:4]}</text>',
        ]
    chunks.append('</svg>')
    path.write_text("\n".join(chunks), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    splits = load_splits(args.bars)
    universe = []
    for ticker in tickers(args.bars):
        bars = adjust_bars(
            load_bars(args.bars, ticker, args.start, args.end), splits.get(ticker, [])
        )
        if len({bar.timestamp[:10] for bar in bars}) >= args.min_sessions:
            universe.append((ticker, bars))
    equity = sqlite3.connect(f"file:{args.equity}?mode=ro", uri=True)
    macro = sqlite3.connect(f"file:{args.macro}?mode=ro", uri=True)
    calendar = [row[0] for row in equity.execute(
        "SELECT obs_date FROM index_prices WHERE ticker='^SP500TR' AND obs_date>=? "
        "AND obs_date<=? ORDER BY obs_date", (args.start, args.end)
    )]
    regimes = build_regimes(calendar, macro, equity)
    funding = funding_map(calendar, macro)

    books = {}
    base = dict(entry_window=55, exit_window=20, atr_window=20,
                skip_after_winner=False, directions=(1,))
    for label, minutes, exits in INTERVALS:
        pooled = []
        config = TurtleConfig(**{**base, **exits})
        for ticker, source in universe:
            bars = resample_regular_session(source, minutes=minutes)
            trades, _audit = run_turtle(bars, config=config)
            for trade in trades:
                if trade.cost_basis_r <= 0:
                    continue
                pooled.append(TaggedTrade(
                    ticker=ticker, entry=trade.entry_timestamp, exit=trade.exit_timestamp,
                    sessions=trade.sessions_held,
                    net_r=trade.net_r,
                    initial_basis_r=trade.entry / trade.n_at_entry,
                ))
        books[label] = pooled
        print(f"{label:10s}: {len(pooled):,} long trades", flush=True)

    periods = (("2017-2022", args.start, "2022-12-31"),
               ("2023+", args.split, args.end),
               ("full", args.start, args.end))
    summaries = {}
    median_curves = {}
    for interval, _minutes, _exit in INTERVALS:
        for variant in VARIANTS:
            for period, start, end in periods:
                cal = period_calendar(calendar, start, end)
                trades = period_trades(books[interval], start, end)
                draws = [replay(
                    trades, variant, regimes, funding, cal,
                    max_positions=args.max_positions, seed=seed,
                    broker_spread=args.broker_spread,
                ) for seed in range(args.draws)]
                summaries[(interval, variant, period)] = {
                    key: statistics.median(draw[key] for draw in draws)
                    for key in ("ending", "cagr", "maxdd", "sharpe", "calmar",
                                "trades", "mean_size")
                } | {
                    "ending_p05": percentile([draw["ending"] for draw in draws], 0.05),
                    "ending_p95": percentile([draw["ending"] for draw in draws], 0.95),
                }
                if period == "full":
                    median_curves[(interval, variant)] = [
                        statistics.median(draw["path"][index] for draw in draws)
                        for index in range(len(cal))
                    ]
            print(f"  {interval:10s} {variant}", flush=True)

    args.outdir.mkdir(parents=True, exist_ok=True)
    curve_csv = args.outdir / "turtle_leverage_curves.csv"
    with curve_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("date", "interval", "variant", "median_nav"))
        for interval, _minutes, _exit in INTERVALS:
            for variant in VARIANTS:
                for day, value in zip(calendar, median_curves[(interval, variant)]):
                    writer.writerow((day, interval, variant, f"{value:.4f}"))
    chart = args.outdir / "turtle_leverage_curves.svg"
    svg_chart(chart, calendar, median_curves)

    lines = [
        "# Leveraged Turtle stability study", "",
        f"**{len(universe)} instruments**, {args.start} through {args.end}. Starting NAV is "
        "**$1,000**: $300 fixed cash sleeve and $700 trading NAV. Trades risk 1% of trading "
        "NAV per 1N Turtle unit, with six-position capacity and a 2x aggregate gross cap. "
        "Gross capacity is conservatively reserved for the known four-unit maximum using "
        "only entry-time price and ATR. "
        "A half-risk overlay cuts a new trade's requested risk in half. Results are medians "
        f"over {args.draws} randomized same-timestamp capacity orderings.", "",
        "Intraday books use the fixed 55-bar entry and 5N chandelier. Daily uses the fixed "
        "55-bar entry and Channel-50 exit. All are long-only. Trade costs are included; "
        f"borrowed capital costs lagged Fed funds + {args.broker_spread:.1%} annually.", "",
        "## Results", "",
        "| Interval | Overlay | Period | End NAV | 5-95% end | CAGR | Max DD | Sharpe | "
        "Calmar | Trades | "
        "Mean risk multiplier |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for interval, _minutes, _exit in INTERVALS:
        for variant in VARIANTS:
            for period, _start, _end in periods:
                item = summaries[(interval, variant, period)]
                lines.append(
                    f"| {interval} | {variant} | {period} | ${item['ending']:,.0f} | "
                    f"${item['ending_p05']:,.0f}-${item['ending_p95']:,.0f} | "
                    f"{item['cagr']:+.2%} | {item['maxdd']:.1%} | "
                    f"{item['sharpe']:.2f} | {item['calmar']:.2f} | "
                    f"{item['trades']:,.0f} | "
                    f"{item['mean_size']:.1%} |"
                )
    lines += ["", "## Interpretation limits", "",
              "- P&L is marked when a trade exits. Intratrade adverse excursion is absent, so "
              "reported drawdown is a lower bound; the underlying five-minute path-aware replay "
              "is the next production-hardening step.",
              "- The 2x gross cap is enforced at entry using a conservative four-unit capacity "
              "reservation based only on entry-time data. Open positions are not forcibly "
              "rebalanced when NAV changes, and a constrained entry can receive less than its "
              "requested 1%-per-N risk. Financing is charged against the reserved amount for "
              "the whole holding period, even when pyramids are not filled immediately.",
              "- Half-risk states are fixed at entry. They do not liquidate an existing trend.",
              "- The final 30 minutes of a US session form a shorter closing bar in the 1h and "
              "2h series; no synthetic overnight bar is inserted.",
              "- The $300 sleeve earns zero and is never used to meet trading losses.",
              "- An equal-dollar diagnostic (which discards Turtle volatility sizing) ended the "
              "full no-overlay runs near $335/$337/$352 for 30m/1h/2h. The plotted intraday "
              "results therefore must be interpreted as volatility-normalized strategies, not "
              "generic equal-notional breakouts.",
              "- Financing excludes hard-to-borrow fees, taxes, slippage beyond the engine's 2 bp, "
              "and market impact."]
    report = args.outdir / "turtle_leverage_stability.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {report}, {curve_csv}, and {chart}")
    macro.close()
    equity.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
