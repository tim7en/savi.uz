"""Paired 30-minute Turtle overlay study with five-minute path-aware P&L.

The 30-minute signal engine remains unchanged. Its stop-entry, pyramid, and
exit fills are located inside the source five-minute bars, then accepted
portfolio positions are marked on every five-minute close. All overlay
comparisons reuse the same capacity-order seed.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sqlite3
import statistics
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_turtle_leverage_stability import (  # noqa: E402
    COLORS, VARIANTS, build_regimes, funding_map, load_bars, multiplier, tickers,
)
from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import TurtleConfig, run_turtle  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402


@dataclass(frozen=True)
class UnitEvent:
    timestamp: str
    price: float
    n: float


@dataclass(frozen=True)
class PathTrade:
    ticker: str
    entry: str
    exit: str
    sessions: int
    net_r: float
    initial_basis_r: float
    units: tuple[UnitEvent, ...]
    round_trip_cost: float
    mark_timestamps: tuple[str, ...]
    mark_rs: tuple[float, ...]


@dataclass
class Position:
    trade: PathTrade
    reserved_gross: float
    risk_per_r: float
    financing: float
    entry_index: int
    exit_index: int


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
    parser.add_argument("--projection-draws", type=int, default=5000)
    parser.add_argument("--projection-years", type=int, default=5)
    parser.add_argument("--block", type=int, default=20)
    parser.add_argument("--broker-spread", type=float, default=0.015)
    parser.add_argument("--outdir", type=Path, default=Path("out/strategy"))
    return parser.parse_args(argv)


def raw_blocks(bars: list[Bar], minutes: int = 30):
    zone = ZoneInfo("America/New_York")
    groups: OrderedDict[tuple[str, int], list[Bar]] = OrderedDict()
    for bar in sorted(bars, key=lambda item: item.timestamp):
        stamp = __import__("datetime").datetime.fromisoformat(
            bar.timestamp.replace("Z", "+00:00")
        )
        local = stamp.astimezone(zone)
        since_open = local.hour * 60 + local.minute - (9 * 60 + 30)
        if 0 <= since_open < 390:
            groups.setdefault(
                (local.date().isoformat(), since_open // minutes), []
            ).append(bar)
    return {rows[0].timestamp: rows for rows in groups.values()}


def first_up_touch(rows: list[Bar], level: float, start: int = 0) -> int:
    tolerance = max(abs(level) * 1e-10, 1e-10)
    for index in range(start, len(rows)):
        if rows[index].open >= level - tolerance or rows[index].high >= level - tolerance:
            return index
    return len(rows) - 1


def first_down_touch(rows: list[Bar], level: float) -> int:
    tolerance = max(abs(level) * 1e-10, 1e-10)
    for index, bar in enumerate(rows):
        if bar.open <= level + tolerance or bar.low <= level + tolerance:
            return index
    return len(rows) - 1


def path_trade(ticker, trade, blocks, regular, regular_index, round_trip_cost):
    events = []
    prior_block = None
    prior_index = 0
    for unit in trade.unit_entries:
        rows = blocks[unit.timestamp]
        start = prior_index if unit.timestamp == prior_block else 0
        touch = first_up_touch(rows, unit.price, start)
        events.append(UnitEvent(rows[touch].timestamp, unit.price, unit.n))
        prior_block, prior_index = unit.timestamp, touch
    exit_rows = blocks[trade.exit_timestamp]
    exit_index = (
        len(exit_rows) - 1 if trade.exit_reason == "end of data"
        else first_down_touch(exit_rows, trade.exit)
    )
    exit_stamp = exit_rows[exit_index].timestamp
    mark_timestamps = []
    mark_values = []
    for bar in regular[
        regular_index[events[0].timestamp]:regular_index[exit_stamp]
    ]:
        entered = [event for event in events if event.timestamp <= bar.timestamp]
        gross = sum((bar.close - event.price) / event.n for event in entered)
        basis = sum(event.price / event.n for event in entered)
        mark_timestamps.append(bar.timestamp)
        mark_values.append(gross - round_trip_cost * basis)
    return PathTrade(
        ticker=ticker,
        entry=events[0].timestamp,
        exit=exit_stamp,
        sessions=trade.sessions_held,
        net_r=trade.net_r,
        initial_basis_r=trade.entry / trade.n_at_entry,
        units=tuple(events),
        round_trip_cost=round_trip_cost,
        mark_timestamps=tuple(mark_timestamps),
        mark_rs=tuple(mark_values),
    )


def mark_r(position: Position, price: float, stamp: str, current_index: int,
           include_current=True):
    comparator = (lambda event: event.timestamp <= stamp) if include_current else (
        lambda event: event.timestamp < stamp
    )
    units = [event for event in position.trade.units if comparator(event)]
    if not units:
        return 0.0, 0.0
    gross = sum((price - event.price) / event.n for event in units)
    basis = sum(event.price / event.n for event in units)
    net = gross - position.trade.round_trip_cost * basis
    span = max(position.exit_index - position.entry_index + 1, 1)
    progress = min(max(current_index - position.entry_index + 1, 0) / span, 1.0)
    return net, position.financing * progress


def metrics(path, exit_only_path, daily_indices):
    maxdd = float(np.min(path / np.maximum.accumulate(path) - 1.0))
    exit_dd = float(np.min(
        exit_only_path / np.maximum.accumulate(exit_only_path) - 1.0
    ))
    daily = path[daily_indices]
    returns = [daily[index] / daily[index - 1] - 1.0
               for index in range(1, len(daily))]
    volatility = statistics.stdev(returns) if len(returns) > 1 else 0.0
    sharpe = (statistics.mean(returns) / volatility * math.sqrt(252.0)
              if volatility else math.nan)
    years = max((len(daily) - 1) / 252.0, 1 / 252.0)
    cagr = (daily[-1] / daily[0]) ** (1 / years) - 1.0
    calmar = cagr / abs(maxdd) if maxdd else math.nan
    return {
        "ending": path[-1], "cagr": cagr, "maxdd": maxdd,
        "exit_only_maxdd": exit_dd, "sharpe": sharpe, "calmar": calmar,
        "daily_path": daily.tolist(),
    }


def replay(trades, variant, regimes, funding, timeline, closes, prepared,
           daily_indices, *, seed, max_positions, broker_spread, size_fn=None,
           post_capacity_size=False, return_full_path=False):
    entries = defaultdict(list)
    for trade in trades:
        entries[trade.entry].append(trade)
    timeline_index = {stamp: index for index, stamp in enumerate(timeline)}
    active = {}
    cash = 700.0
    sleeve = 300.0
    accepted = 0
    risk_sum = 0.0
    accepted_positions = []
    rng = random.Random(seed)

    def marked_nav(stamp, index):
        unrealized = 0.0
        for position in active.values():
            previous = timeline[max(index - 1, 0)]
            price = closes.get(previous, {}).get(position.trade.ticker)
            if price is None:
                price = position.trade.units[0].price
            value_r, accrued_financing = mark_r(
                position, price, stamp, max(index - 1, 0), include_current=False
            )
            unrealized += position.risk_per_r * value_r - accrued_financing
        return max(0.0, cash + unrealized)

    event_stamps = sorted(set(entries) | {trade.exit for trade in trades})
    for stamp in event_stamps:
        index = timeline_index[stamp]
        closing = [key for key, position in active.items()
                   if position.trade.exit <= stamp]
        for key in sorted(closing):
            position = active.pop(key)
            cash = max(
                0.0,
                cash + position.risk_per_r * position.trade.net_r
                - position.financing,
            )

        candidates = list(entries.get(stamp, ()))
        rng.shuffle(candidates)
        for trade in candidates:
            if len(active) >= max_positions:
                continue
            trading_nav = marked_nav(stamp, index)
            if trading_nav <= 0:
                continue
            size = (size_fn(variant, trade) if size_fn is not None
                    else multiplier(variant, trade.entry[:10], regimes))
            active_gross = sum(item.reserved_gross for item in active.values())
            gross_room = max(0.0, 2.0 * trading_nav - active_gross)
            if post_capacity_size:
                full_risk = trading_nav * 0.01
                full_gross = full_risk * trade.initial_basis_r * 4
                full_reserved = min(full_gross, gross_room)
                reserved = full_reserved * size
                risk_per_r = full_risk * full_reserved / full_gross * size
            else:
                requested_risk = trading_nav * 0.01 * size
                requested_gross = requested_risk * trade.initial_basis_r * 4
                reserved = min(requested_gross, gross_room)
                risk_per_r = requested_risk * reserved / requested_gross
            if reserved <= 0:
                continue
            rate = (funding.get(trade.entry[:10]) or 0.0) / 100.0 + broker_spread
            borrowed_before = max(active_gross - trading_nav, 0.0)
            borrowed_after = max(active_gross + reserved - trading_nav, 0.0)
            financing = (
                (borrowed_after - borrowed_before) * rate * trade.sessions / 252.0
            )
            key = (trade.ticker, trade.entry, accepted)
            active[key] = Position(
                trade=trade, reserved_gross=reserved, risk_per_r=risk_per_r,
                financing=financing, entry_index=index,
                exit_index=timeline_index[trade.exit],
            )
            accepted_positions.append(active[key])
            accepted += 1
            risk_sum += size

    unrealized = np.zeros(len(timeline), dtype=float)
    realized_delta = np.zeros(len(timeline), dtype=float)
    for position in accepted_positions:
        indices, values = prepared[position.trade]
        progress = np.arange(1, len(indices) + 1, dtype=float) / max(len(indices) + 1, 1)
        unrealized[indices] += (
            position.risk_per_r * values - position.financing * progress
        )
        realized_delta[position.exit_index] += (
            position.risk_per_r * position.trade.net_r - position.financing
        )
    realized = np.cumsum(realized_delta)
    # Flooring at the sleeve caps any reported drawdown at -(1 - sleeve/1000),
    # i.e. -70% here, so a deeper excursion silently reads as exactly -70%.
    # Keep an unfloored series for the risk metrics and floor only what is
    # plotted, where the segregated sleeve genuinely is a floor.
    exit_only_raw = 1000.0 + realized
    raw = 1000.0 + realized + unrealized
    exit_only_path = np.maximum(sleeve, exit_only_raw)
    path = np.maximum(sleeve, raw)
    censored = bool(np.min(raw) < sleeve)

    result = metrics(raw, exit_only_raw, daily_indices)
    result.update({
        "nav_hit_sleeve_floor": censored,
        "timeline": timeline, "trades": accepted,
        "mean_size": risk_sum / accepted if accepted else math.nan,
    })
    if return_full_path:
        result["full_path"] = raw.tolist()
    return result


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def median_summary(draws):
    return {key: statistics.median(draw[key] for draw in draws)
            for key in ("ending", "cagr", "maxdd", "exit_only_maxdd", "sharpe",
                        "calmar", "trades", "mean_size")}


def daily_median_path(draws):
    length = min(len(draw["daily_path"]) for draw in draws)
    return [statistics.median(draw["daily_path"][index] for draw in draws)
            for index in range(length)]


def projection(draws, *, simulations, years, block, seed):
    sources = []
    for draw in draws:
        trading = [max(value - 300.0, 1e-9) for value in draw["daily_path"]]
        sources.append([trading[index] / trading[index - 1] - 1.0
                        for index in range(1, len(trading))])
    horizon = years * 252
    rng = random.Random(seed)
    checkpoints = sorted(set(range(0, horizon + 1, 5)) | {252, 756, 1260, horizon})
    distributions = {day: [] for day in checkpoints}
    terminals = {year: [] for year in (1, 3, 5) if year <= years}
    for _ in range(simulations):
        source = sources[rng.randrange(len(sources))]
        generated = []
        while len(generated) < horizon:
            start = rng.randrange(max(len(source) - block + 1, 1))
            generated.extend(source[start:start + block])
        trading_nav = 700.0
        checkpoint_set = set(checkpoints)
        distributions[0].append(1000.0)
        for day, value in enumerate(generated[:horizon], 1):
            trading_nav = max(0.0, trading_nav * (1.0 + value))
            nav = 300.0 + trading_nav
            if day in checkpoint_set:
                distributions[day].append(nav)
            if day % 252 == 0 and day // 252 in terminals:
                terminals[day // 252].append(nav)
    quantiles = {
        day: {q: percentile(values, q) for q in (0.05, 0.10, 0.50, 0.90, 0.95)}
        for day, values in distributions.items()
    }
    terminal = {
        year: {
            "p05": percentile(values, 0.05), "p10": percentile(values, 0.10),
            "median": percentile(values, 0.50), "p90": percentile(values, 0.90),
            "p95": percentile(values, 0.95),
            "loss_probability": sum(value < 1000.0 for value in values) / len(values),
        } for year, values in terminals.items()
    }
    return quantiles, terminal


def historical_svg(path, dates, curves):
    width, height = 1280, 650
    all_values = [value for curve in curves.values() for value in curve]
    low, high = min(all_values), max(all_values)
    low = max(250.0, low * 0.90)
    high *= 1.08
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="65" y="42" font-family="Arial" font-size="25" font-weight="700">'
        '30-minute Turtle: five-minute marked P&amp;L</text>',
        '<rect x="70" y="90" width="1140" height="490" fill="#f9fafb" stroke="#d1d5db"/>',
    ]
    legend_x = 70
    for variant in VARIANTS:
        chunks += [
            f'<line x1="{legend_x}" y1="70" x2="{legend_x + 24}" y2="70" '
            f'stroke="{COLORS[variant]}" stroke-width="3"/>',
            f'<text x="{legend_x + 30}" y="75" font-family="Arial" font-size="12">'
            f'{escape(variant)}</text>',
        ]
        legend_x += 235
    for variant, values in curves.items():
        points = []
        for index, value in enumerate(values):
            x = 70 + 1140 * index / max(len(values) - 1, 1)
            y = 90 + 490 * (math.log(high) - math.log(max(value, low))) / (
                math.log(high) - math.log(low)
            )
            points.append(f"{x:.1f},{y:.1f}")
        chunks.append(
            f'<polyline fill="none" stroke="{COLORS[variant]}" stroke-width="2" '
            f'points="{" ".join(points)}"/>'
        )
    chunks += [
        f'<text x="75" y="108" font-family="Arial" font-size="12">${high:,.0f}</text>',
        f'<text x="75" y="570" font-family="Arial" font-size="12">${low:,.0f}</text>',
        f'<text x="70" y="605" font-family="Arial" font-size="12">{dates[0][:4]}</text>',
        f'<text x="1175" y="605" font-family="Arial" font-size="12">{dates[-1][:4]}</text>',
        '<text x="500" y="630" font-family="Arial" font-size="12" fill="#6b7280">'
        'Logarithmic NAV scale; medians across 100 paired capacity orderings</text>',
        '</svg>',
    ]
    path.write_text("\n".join(chunks), encoding="utf-8")


def projection_svg(path, projections, years):
    width, height = 1280, 700
    colors = {"No overlay": "#111827", "Any preferred overlay half-risk": "#059669"}
    values = [value for data in projections.values() for row in data.values()
              for value in row.values()]
    low, high = max(250.0, min(values) * 0.9), max(values) * 1.08
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="65" y="42" font-family="Arial" font-size="25" font-weight="700">'
        '30-minute forward P&amp;L scenarios from a fresh $1,000</text>',
        '<text x="65" y="67" font-family="Arial" font-size="13" fill="#6b7280">'
        '20-session block bootstrap of 2023+ trading-sleeve returns; $300 remains fixed</text>',
        '<rect x="70" y="105" width="1140" height="500" fill="#f9fafb" stroke="#d1d5db"/>',
    ]
    for variant, data in projections.items():
        days = sorted(data)
        color = colors[variant]
        polygon = []
        for day in days:
            x = 70 + 1140 * day / (years * 252)
            y = 105 + 500 * (math.log(high) - math.log(data[day][0.90])) / (
                math.log(high) - math.log(low)
            )
            polygon.append(f"{x:.1f},{y:.1f}")
        for day in reversed(days):
            x = 70 + 1140 * day / (years * 252)
            y = 105 + 500 * (math.log(high) - math.log(data[day][0.10])) / (
                math.log(high) - math.log(low)
            )
            polygon.append(f"{x:.1f},{y:.1f}")
        chunks.append(
            f'<polygon points="{" ".join(polygon)}" fill="{color}" opacity="0.12"/>'
        )
        points = []
        for day in days:
            x = 70 + 1140 * day / (years * 252)
            y = 105 + 500 * (math.log(high) - math.log(data[day][0.50])) / (
                math.log(high) - math.log(low)
            )
            points.append(f"{x:.1f},{y:.1f}")
        chunks.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" '
            f'points="{" ".join(points)}"/>'
        )
    chunks += [
        '<line x1="70" y1="90" x2="95" y2="90" stroke="#111827" stroke-width="3"/>',
        '<text x="102" y="95" font-family="Arial" font-size="13">No overlay</text>',
        '<line x1="220" y1="90" x2="245" y2="90" stroke="#059669" stroke-width="3"/>',
        '<text x="252" y="95" font-family="Arial" font-size="13">Any preferred overlay</text>',
        f'<text x="75" y="124" font-family="Arial" font-size="12">${high:,.0f}</text>',
        f'<text x="75" y="595" font-family="Arial" font-size="12">${low:,.0f}</text>',
        '<text x="70" y="630" font-family="Arial" font-size="12">Now</text>',
        f'<text x="1180" y="630" font-family="Arial" font-size="12">Year {years}</text>',
        '<text x="415" y="670" font-family="Arial" font-size="12" fill="#6b7280">'
        'Solid: median · shaded: 10th–90th percentile · logarithmic NAV scale</text>',
        '</svg>',
    ]
    path.write_text("\n".join(chunks), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    splits = load_splits(args.bars)
    universe = []
    closes = defaultdict(dict)
    all_timestamps = set()
    config = TurtleConfig(
        entry_window=55, exit_window=20, atr_window=20,
        skip_after_winner=False, directions=(1,), use_channel_exit=False,
        chandelier_atr=5.0,
    )
    trades = []
    for ticker in tickers(args.bars):
        raw = adjust_bars(
            load_bars(args.bars, ticker, args.start, args.end), splits.get(ticker, [])
        )
        if len({bar.timestamp[:10] for bar in raw}) < args.min_sessions:
            continue
        universe.append(ticker)
        blocks = raw_blocks(raw)
        regular = [bar for rows in blocks.values() for bar in rows]
        regular_index = {bar.timestamp: index for index, bar in enumerate(regular)}
        thirty = resample_regular_session(raw, minutes=30)
        ticker_trades, _audit = run_turtle(thirty, config=config)
        for trade in ticker_trades:
            if trade.unit_entries and trade.n_at_entry > 0:
                trades.append(path_trade(
                    ticker, trade, blocks, regular, regular_index,
                    config.round_trip_cost,
                ))
        for rows in blocks.values():
            for bar in rows:
                closes[bar.timestamp][ticker] = bar.close
                all_timestamps.add(bar.timestamp)
        print(f"{ticker}: {len(ticker_trades):,} trades", flush=True)

    equity = sqlite3.connect(f"file:{args.equity}?mode=ro", uri=True)
    macro = sqlite3.connect(f"file:{args.macro}?mode=ro", uri=True)
    calendar = [row[0] for row in equity.execute(
        "SELECT obs_date FROM index_prices WHERE ticker='^SP500TR' AND obs_date>=? "
        "AND obs_date<=? ORDER BY obs_date", (args.start, args.end)
    )]
    regimes = build_regimes(calendar, macro, equity)
    funding = funding_map(calendar, macro)
    periods = (("2017-2022", args.start, "2022-12-31"),
               ("2023+", args.split, args.end),
               ("full", args.start, args.end))
    results = {}
    summaries = {}
    for period, start, end in periods:
        selected = [trade for trade in trades
                    if start <= trade.entry[:10] and trade.exit[:10] <= end]
        timeline = sorted(stamp for stamp in all_timestamps if start <= stamp[:10] <= end)
        timeline_index = {stamp: index for index, stamp in enumerate(timeline)}
        prepared = {
            trade: (
                np.asarray([timeline_index[stamp] for stamp in trade.mark_timestamps],
                           dtype=int),
                np.asarray(trade.mark_rs, dtype=float),
            ) for trade in selected
        }
        daily_indices = np.asarray([
            index for index, stamp in enumerate(timeline)
            if index + 1 == len(timeline) or timeline[index + 1][:10] != stamp[:10]
        ], dtype=int)
        for variant in VARIANTS:
            draws = [replay(
                selected, variant, regimes, funding, timeline, closes, prepared,
                daily_indices,
                seed=seed, max_positions=args.max_positions,
                broker_spread=args.broker_spread,
            ) for seed in range(args.draws)]
            results[(period, variant)] = draws
            summaries[(period, variant)] = median_summary(draws)
            print(f"{period:9s} {variant}", flush=True)

    paired_rows = []
    paired_summary = {}
    for period, _start, _end in periods:
        baseline = results[(period, "No overlay")]
        for variant in VARIANTS[1:]:
            overlay = results[(period, variant)]
            rows = []
            for seed, (base, over) in enumerate(zip(baseline, overlay)):
                row = {
                    "period": period, "variant": variant, "seed": seed,
                    "delta_nav": over["ending"] - base["ending"],
                    "delta_cagr_pp": 100 * (over["cagr"] - base["cagr"]),
                    "delta_dd_pp": 100 * (over["maxdd"] - base["maxdd"]),
                    "delta_sharpe": over["sharpe"] - base["sharpe"],
                    "delta_calmar": over["calmar"] - base["calmar"],
                }
                rows.append(row)
                paired_rows.append(row)
            paired_summary[(period, variant)] = {
                "nav_wins": sum(row["delta_nav"] > 0 for row in rows),
                "dd_wins": sum(row["delta_dd_pp"] > 0 for row in rows),
                **{key: statistics.median(row[key] for row in rows)
                   for key in ("delta_nav", "delta_cagr_pp", "delta_dd_pp",
                               "delta_sharpe", "delta_calmar")},
            }

    projections = {}
    terminal = {}
    for index, variant in enumerate(VARIANTS):
        projections[variant], terminal[variant] = projection(
            results[("2023+", variant)], simulations=args.projection_draws,
            years=args.projection_years, block=args.block, seed=9100 + index,
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    paired_csv = args.outdir / "turtle_30m_paired_overlays.csv"
    with paired_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    projection_csv = args.outdir / "turtle_30m_projection.csv"
    with projection_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("variant", "day", "p05", "p10", "median", "p90", "p95"))
        for variant, data in projections.items():
            for day, row in sorted(data.items()):
                writer.writerow((variant, day, row[0.05], row[0.10], row[0.50],
                                 row[0.90], row[0.95]))

    full_dates = []
    seen = set()
    for stamp in results[("full", "No overlay")][0]["timeline"]:
        if stamp[:10] not in seen:
            full_dates.append(stamp[:10])
            seen.add(stamp[:10])
    historical_curves = {
        variant: daily_median_path(results[("full", variant)]) for variant in VARIANTS
    }
    historical_chart = args.outdir / "turtle_30m_path_curves.svg"
    historical_svg(historical_chart, full_dates, historical_curves)
    projection_chart = args.outdir / "turtle_30m_projection.svg"
    projection_svg(
        projection_chart,
        {variant: projections[variant] for variant in
         ("No overlay", "Any preferred overlay half-risk")},
        args.projection_years,
    )

    lines = [
        "# 30-minute Turtle: paired path-aware overlay study", "",
        f"**{len(universe)} instruments**, **{len(trades):,} candidate trades**, "
        f"{args.start} through {args.end}. The account starts at $1,000: $300 fixed sleeve "
        "plus $700 trading NAV. Each unit requests 1% of trading NAV per N, with a "
        "conservative four-unit reservation, six-position capacity, and 2x gross cap.", "",
        "Signals and fills come from the fixed 30-minute 55-bar long breakout with a 5N "
        "chandelier. Unit and exit fills are located inside the source five-minute bars, and "
        "open positions are marked on every five-minute close. Macro observations retain the "
        "D-2 availability lag. Costs and conservative financing are included.", "",
        "## Path-aware results", "",
        "| Variant | Period | End NAV | CAGR | 5m Max DD | Exit-only DD | Sharpe | Calmar | Trades |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        for period, _start, _end in periods:
            item = summaries[(period, variant)]
            lines.append(
                f"| {variant} | {period} | ${item['ending']:,.0f} | {item['cagr']:+.2%} | "
                f"{item['maxdd']:.1%} | {item['exit_only_maxdd']:.1%} | "
                f"{item['sharpe']:.2f} | {item['calmar']:.2f} | {item['trades']:,.0f} |"
            )
    lines += ["", "## Paired overlay deltas versus no overlay", "",
              "Positive delta DD means the overlay's drawdown was less severe. Every row compares "
              "the identical capacity-order seed.", "",
              "| Overlay | Period | NAV wins | DD wins | Median delta NAV | Median delta CAGR | "
              "Median delta DD | Median delta Sharpe | Median delta Calmar |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS[1:]:
        for period in ("2017-2022", "2023+", "full"):
            item = paired_summary[(period, variant)]
            lines.append(
                f"| {variant} | {period} | {item['nav_wins']}/{args.draws} | "
                f"{item['dd_wins']}/{args.draws} | ${item['delta_nav']:+,.0f} | "
                f"{item['delta_cagr_pp']:+.2f} pp | {item['delta_dd_pp']:+.2f} pp | "
                f"{item['delta_sharpe']:+.2f} | {item['delta_calmar']:+.2f} |"
            )
    lines += ["", "## Forward scenario distribution", "",
              f"Each distribution contains {args.projection_draws:,} paths starting from a fresh "
              "$1,000. It resamples 20-session blocks from 2023+ trading-sleeve returns across "
              "the 100 capacity orderings; $300 remains fixed. This is an empirical scenario "
              "range, not a forecast of expected returns.", "",
              "| Variant | Horizon | 5-95% NAV | 10-90% NAV | Median NAV | P(NAV < $1,000) |",
              "|---|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        for year in sorted(terminal[variant]):
            item = terminal[variant][year]
            lines.append(
                f"| {variant} | {year}y | ${item['p05']:,.0f}-${item['p95']:,.0f} | "
                f"${item['p10']:,.0f}-${item['p90']:,.0f} | ${item['median']:,.0f} | "
                f"{item['loss_probability']:.1%} |"
            )
    lines += ["", "## Limits", "",
              "- The five-minute sequence locates 30-minute fills but does not invent tick-level "
              "ordering inside a five-minute candle. The original 30-minute engine's adverse-"
              "first fill convention remains authoritative.",
              "- Same-timestamp randomization measures capacity competition, not universe or "
              "parameter-selection uncertainty.",
              "- Projection blocks assume the 2023+ return-generating process can recur. They do "
              "not model delistings, new instruments, market impact, or structural regime shifts.",
              "- The current downloaded universe is selected and is not a survivorship-free "
              "historical constituent universe."]
    report = args.outdir / "turtle_30m_path_projection.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {report}, {paired_csv}, {projection_csv}, {historical_chart}, {projection_chart}")
    macro.close()
    equity.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
