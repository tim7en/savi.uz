"""Leakage-safe multi-session volume-profile breakout research.

The profile and price range for a signal on session ``t`` use only completed
sessions ``t-window .. t-1``.  A crossing is observed at a five-minute close
and entered at the following bar's open.  The regular-hours feed has no
after-hours path, so an overnight stop that is crossed by a gap is filled at
the next regular-session open rather than at the stop price.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from statistics import median

from savi_uz.volume_profile import Bar, build_profile


@dataclass(frozen=True)
class CompositeEvent:
    session: str
    timestamp: str
    window: int
    boundary: str
    volume_floor: float
    compression_quantile: float | None
    direction: int
    signal_bar: int
    lower: float
    upper: float
    entry: float
    atr: float
    volume_ratio: float
    range_atr_ratio: float

    close_return: float
    next_open_return: float | None
    next_close_return: float | None
    close_3_return: float | None
    close_5_return: float | None


@dataclass(frozen=True)
class TradeResult:
    event: CompositeEvent
    exit_session: str
    exit_timestamp: str
    exit_price: float
    reason: str
    holding_sessions: int
    gross_return: float
    net_return: float


@dataclass(frozen=True)
class TradeSummary:
    count: int
    mean_return: float
    median_return: float
    win_rate: float
    profit_factor: float
    ending_equity: float
    max_drawdown: float
    cagr: float
    calmar: float
    stop_rate: float
    gap_stop_rate: float
    mean_holding_sessions: float


def group_sessions(bars: list[Bar]) -> list[tuple[str, list[Bar]]]:
    """Return complete date groups in timestamp order."""
    grouped: dict[str, list[Bar]] = defaultdict(list)
    for bar in sorted(bars, key=lambda row: row.timestamp):
        grouped[bar.timestamp[:10]].append(bar)
    return [(day, rows) for day, rows in sorted(grouped.items())]


def complete_sessions(
    bars: list[Bar], *, start: str = "2019-01-01", expected_bars: int = 78
) -> list[tuple[str, list[Bar]]]:
    return [
        (day, rows)
        for day, rows in group_sessions(bars)
        if day >= start and len(rows) == expected_bars
    ]


def _coverage(rows: list[Bar]) -> float:
    return sum(row.volume is not None and row.volume > 0 for row in rows) / len(rows)


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _daily_true_ranges(sessions: list[tuple[str, list[Bar]]]) -> list[float]:
    out: list[float] = []
    prior_close: float | None = None
    for _, rows in sessions:
        high = max(row.high for row in rows)
        low = min(row.low for row in rows)
        value = high - low
        if prior_close is not None:
            value = max(value, abs(high - prior_close), abs(low - prior_close))
        out.append(value)
        prior_close = rows[-1].close
    return out


def _intraday_atrs(
    sessions: list[tuple[str, list[Bar]]], lookback: int
) -> dict[tuple[str, int], float]:
    values: list[float] = []
    out: dict[tuple[str, int], float] = {}
    prior_close: float | None = None
    for day, rows in sessions:
        for position, row in enumerate(rows):
            value = row.high - row.low
            if prior_close is not None:
                value = max(value, abs(row.high - prior_close), abs(row.low - prior_close))
            values.append(value)
            out[(day, position)] = sum(values[-lookback:]) / min(lookback, len(values))
            prior_close = row.close
    return out


def _directional(direction: int, exit_price: float, entry: float) -> float:
    return direction * (exit_price / entry - 1.0)


def build_events(
    bars: list[Bar],
    window: int,
    boundary: str,
    volume_floor: float,
    *,
    start: str = "2019-01-01",
    expected_bars: int = 78,
    bins: int = 30,
    min_profile_coverage: float = 0.90,
    volume_lookback: int = 20,
    min_volume_observations: int = 15,
    atr_lookback: int = 20,
    compression_quantile: float | None = None,
    compression_lookback: int = 60,
) -> list[CompositeEvent]:
    """Build the first qualifying composite-profile crossing per session.

    ``boundary`` is ``"value"`` for the composite 70% value area or ``"range"``
    for the full high-low of the prior sessions.  If ``compression_quantile``
    is supplied, the prior range/ATR ratio must be in that trailing quantile;
    its threshold is formed only from older observations.
    """
    if window < 1 or volume_lookback < 1 or atr_lookback < 1:
        raise ValueError("lookbacks and window must be positive")
    if boundary not in {"value", "range"}:
        raise ValueError("boundary must be 'value' or 'range'")
    if compression_quantile is not None and not 0 < compression_quantile < 1:
        raise ValueError("compression quantile must be between zero and one")

    sessions = complete_sessions(bars, start=start, expected_bars=expected_bars)
    daily_tr = _daily_true_ranges(sessions)
    intraday_atr = _intraday_atrs(sessions, atr_lookback)

    ratios: list[float | None] = [None] * len(sessions)
    for index in range(window, len(sessions)):
        prior = [row for _, rows in sessions[index - window:index] for row in rows]
        span = max(row.high for row in prior) - min(row.low for row in prior)
        atr_values = daily_tr[max(0, index - 20):index]
        typical = median(atr_values) if atr_values else 0.0
        ratios[index] = span / typical if typical > 0 else None

    events: list[CompositeEvent] = []
    first = max(window, volume_lookback, compression_lookback if compression_quantile else 0)
    for index in range(first, len(sessions)):
        day, current = sessions[index]
        prior_sessions = sessions[index - window:index]
        if any(_coverage(rows) < min_profile_coverage for _, rows in prior_sessions):
            continue

        ratio = ratios[index]
        if ratio is None:
            continue
        if compression_quantile is not None:
            history = [
                value for value in ratios[max(window, index - compression_lookback):index]
                if value is not None
            ]
            if len(history) < compression_lookback // 2:
                continue
            if ratio > _quantile(history, compression_quantile):
                continue

        prior_bars = [row for _, rows in prior_sessions for row in rows]
        if boundary == "value":
            profile = build_profile(prior_bars, bins=bins)
            if profile is None:
                continue
            lower, upper = profile.value_low, profile.value_high
        else:
            lower = min(row.low for row in prior_bars)
            upper = max(row.high for row in prior_bars)

        for position in range(0, len(current) - 1):
            row = current[position]
            before = current[position - 1].close if position else sessions[index - 1][1][-1].close
            long_break = before <= upper and row.close > upper
            short_break = before >= lower and row.close < lower
            if not (long_break or short_break):
                continue

            comparison = [
                sessions[past][1][position].volume
                for past in range(index - volume_lookback, index)
                if sessions[past][1][position].volume is not None
                and sessions[past][1][position].volume > 0
            ]
            if len(comparison) < min_volume_observations or not row.volume or row.volume <= 0:
                continue
            typical_volume = median(comparison)
            volume_ratio = row.volume / typical_volume if typical_volume > 0 else math.nan
            if not math.isfinite(volume_ratio) or volume_ratio < volume_floor:
                continue

            direction = 1 if long_break else -1
            entry = current[position + 1].open

            def future_return(offset: int, at_open: bool = False) -> float | None:
                future = index + offset
                if future >= len(sessions):
                    return None
                price = sessions[future][1][0].open if at_open else sessions[future][1][-1].close
                return _directional(direction, price, entry)

            events.append(
                CompositeEvent(
                    session=day,
                    timestamp=row.timestamp,
                    window=window,
                    boundary=boundary,
                    volume_floor=volume_floor,
                    compression_quantile=compression_quantile,
                    direction=direction,
                    signal_bar=position,
                    lower=lower,
                    upper=upper,
                    entry=entry,
                    atr=intraday_atr[(day, position)],
                    volume_ratio=volume_ratio,
                    range_atr_ratio=ratio,
                    close_return=_directional(direction, current[-1].close, entry),
                    next_open_return=future_return(1, at_open=True),
                    next_close_return=future_return(1),
                    close_3_return=future_return(3),
                    close_5_return=future_return(5),
                )
            )
            break
    return events


def simulate_trade(
    event: CompositeEvent,
    sessions: list[tuple[str, list[Bar]]],
    *,
    stop_atr: float = 2.5,
    trail_atr: float | None = None,
    activation_atr: float = 2.0,
    max_hold_sessions: int = 1,
    round_trip_cost: float = 0.0002,
) -> TradeResult | None:
    """Simulate a stop/trail through regular sessions, charging gaps at the open.

    A newly raised trailing stop becomes active on the following bar.  This
    avoids inventing the high/low ordering inside a five-minute OHLC bar.
    """
    if stop_atr <= 0 or max_hold_sessions < 0:
        raise ValueError("stop must be positive and holding period non-negative")
    if trail_atr is not None and trail_atr <= 0:
        raise ValueError("trail must be positive")

    index_by_day = {day: index for index, (day, _) in enumerate(sessions)}
    start_index = index_by_day.get(event.session)
    if start_index is None or start_index + max_hold_sessions >= len(sessions):
        return None

    direction = event.direction
    risk = stop_atr * event.atr
    stop = event.entry - direction * risk
    best = event.entry
    last_index = start_index + max_hold_sessions

    for session_index in range(start_index, last_index + 1):
        day, rows = sessions[session_index]
        first_bar = event.signal_bar + 1 if session_index == start_index else 0
        for position in range(first_bar, len(rows)):
            row = rows[position]
            if session_index > start_index and position == 0:
                crossed_at_open = row.open <= stop if direction > 0 else row.open >= stop
                if crossed_at_open:
                    gross = _directional(direction, row.open, event.entry)
                    return TradeResult(
                        event, day, row.timestamp, row.open, "gap_stop",
                        session_index - start_index, gross, gross - round_trip_cost,
                    )

            touched = row.low <= stop if direction > 0 else row.high >= stop
            if touched:
                gross = _directional(direction, stop, event.entry)
                return TradeResult(
                    event, day, row.timestamp, stop, "stop",
                    session_index - start_index, gross, gross - round_trip_cost,
                )

            favorable = row.high if direction > 0 else row.low
            best = max(best, favorable) if direction > 0 else min(best, favorable)
            excursion = direction * (best - event.entry)
            if trail_atr is not None and excursion >= activation_atr * event.atr:
                candidate = best - direction * trail_atr * event.atr
                stop = max(stop, candidate) if direction > 0 else min(stop, candidate)

    day, rows = sessions[last_index]
    exit_bar = rows[-1]
    gross = _directional(direction, exit_bar.close, event.entry)
    return TradeResult(
        event, day, exit_bar.timestamp, exit_bar.close, "time",
        max_hold_sessions, gross, gross - round_trip_cost,
    )


def non_overlapping_results(
    events: list[CompositeEvent],
    sessions: list[tuple[str, list[Bar]]],
    **simulation_args,
) -> list[TradeResult]:
    """Run one position at a time; signals while a trade is open are ignored."""
    results: list[TradeResult] = []
    available_after = ""
    for event in sorted(events, key=lambda row: row.timestamp):
        if event.timestamp <= available_after:
            continue
        result = simulate_trade(event, sessions, **simulation_args)
        if result is None:
            continue
        results.append(result)
        available_after = result.exit_timestamp
    return results


def summarise_trades(results: list[TradeResult]) -> TradeSummary:
    if not results:
        return TradeSummary(0, *(math.nan for _ in range(11)))
    returns = [row.net_return for row in results]
    gains = sum(value for value in returns if value > 0)
    losses = -sum(value for value in returns if value < 0)
    equity = peak = 1.0
    max_drawdown = 0.0
    for value in returns:
        equity *= max(0.0, 1.0 + value)
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity / peak - 1.0)
    elapsed = (
        date.fromisoformat(results[-1].exit_session)
        - date.fromisoformat(results[0].event.session)
    ).days / 365.25
    cagr = equity ** (1 / elapsed) - 1 if elapsed > 0 and equity > 0 else math.nan
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else math.inf
    return TradeSummary(
        count=len(results),
        mean_return=sum(returns) / len(returns),
        median_return=median(returns),
        win_rate=sum(value > 0 for value in returns) / len(returns),
        profit_factor=gains / losses if losses else math.inf,
        ending_equity=equity,
        max_drawdown=max_drawdown,
        cagr=cagr,
        calmar=calmar,
        stop_rate=sum(row.reason in {"stop", "gap_stop"} for row in results) / len(results),
        gap_stop_rate=sum(row.reason == "gap_stop" for row in results) / len(results),
        mean_holding_sessions=sum(row.holding_sessions for row in results) / len(results),
    )
