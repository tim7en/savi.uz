"""Close-confirmed Turtle System 2 with leakage-safe volume/profile filters.

The published Turtle enters intrabar at the channel stop.  Full-bar volume is
not known at that instant, so this research variant confirms the breakout at
the close and enters the following bar's open.  Every filter is compared with
that delayed, close-confirmed control rather than with the original fill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median

from savi_uz.turtle import (
    TurtleTrade,
    _breached,
    _stop_fill,
    rolling_extremes,
    wilder_atr,
)
from savi_uz.volume_profile import Bar, build_profile


@dataclass(frozen=True)
class BreakoutVolume:
    session: str
    volume: float
    typical_volume: float
    volume_ratio: float
    rising_volume: bool
    poc: float
    value_low: float
    value_high: float
    profile_low: float
    profile_high: float


@dataclass(frozen=True)
class VolumeFilter:
    volume_floor: float | None = None
    require_rising_volume: bool = False
    require_outside_value: bool = False
    minimum_poc_distance_n: float | None = None

    def __post_init__(self) -> None:
        if self.volume_floor is not None and self.volume_floor < 0:
            raise ValueError("volume_floor cannot be negative")
        if self.minimum_poc_distance_n is not None and self.minimum_poc_distance_n < 0:
            raise ValueError("minimum_poc_distance_n cannot be negative")


@dataclass(frozen=True)
class ConfirmedTurtleConfig:
    entry_window: int = 55
    exit_window: int = 20
    atr_window: int = 20
    stop_atr: float = 2.0
    add_atr: float = 0.5
    max_units: int = 4
    risk_fraction: float = 0.01
    minimum_n_cost_multiple: float = 5.0
    minimum_n_fraction: float = 0.0
    round_trip_cost: float = 0.0002
    directions: tuple[int, ...] = (1, -1)

    def __post_init__(self) -> None:
        if min(self.entry_window, self.exit_window, self.atr_window) < 2:
            raise ValueError("windows must span at least two bars")
        if self.exit_window >= self.entry_window:
            raise ValueError("exit window must be shorter than entry window")
        if self.stop_atr <= 0 or self.add_atr <= 0 or self.max_units < 1:
            raise ValueError("invalid stop or pyramiding settings")
        if not 0 < self.risk_fraction < 1 or self.round_trip_cost < 0:
            raise ValueError("invalid risk or cost")
        if not self.directions or set(self.directions) - {1, -1}:
            raise ValueError("directions must be a subset of (1, -1)")


@dataclass(frozen=True)
class VolumeTurtleTrade:
    ticker: str
    signal_timestamp: str
    direction: int
    signal_close: float
    channel_level: float
    volume_ratio: float
    rising_volume: bool
    poc: float
    value_low: float
    value_high: float
    outside_value: bool
    poc_distance_n: float
    trade: TurtleTrade


@dataclass(frozen=True)
class VolumeTurtleAudit:
    bars: int
    confirmed_breakouts: int
    missing_volume_profile: int
    rejected_by_filter: int
    skipped_small_n: int
    trades: int


@dataclass(frozen=True)
class _Unit:
    timestamp: str
    price: float
    n: float


def _sessions(bars: list[Bar]) -> list[tuple[str, list[Bar]]]:
    grouped: dict[str, list[Bar]] = {}
    for row in sorted(bars, key=lambda item: item.timestamp):
        grouped.setdefault(row.timestamp[:10], []).append(row)
    return list(grouped.items())


def _coverage(rows: list[Bar]) -> float:
    return sum(row.volume is not None and row.volume > 0 for row in rows) / len(rows)


def build_breakout_volumes(
    intraday_bars: list[Bar], *, profile_sessions: int = 5,
    volume_lookback: int = 20, min_volume_observations: int = 15,
    min_coverage: float = 0.90, bins: int = 30,
) -> dict[str, BreakoutVolume]:
    """Features for D use D's completed volume and profiles from D-5..D-1."""
    if profile_sessions < 1 or volume_lookback < 1:
        raise ValueError("lookbacks must be positive")
    sessions = _sessions(intraday_bars)
    totals: list[float | None] = []
    for _, rows in sessions:
        totals.append(
            sum(float(row.volume or 0.0) for row in rows)
            if rows and _coverage(rows) >= min_coverage else None
        )
    result: dict[str, BreakoutVolume] = {}
    first = max(profile_sessions, volume_lookback)
    for index in range(first, len(sessions)):
        day, current = sessions[index]
        current_volume = totals[index]
        comparison = [
            value for value in totals[index - volume_lookback:index]
            if value is not None and value > 0
        ]
        prior_profile_sessions = sessions[index - profile_sessions:index]
        if (
            current_volume is None
            or len(comparison) < min_volume_observations
            or any(_coverage(rows) < min_coverage for _, rows in prior_profile_sessions)
        ):
            continue
        profile = build_profile(
            [row for _, rows in prior_profile_sessions for row in rows], bins=bins
        )
        if profile is None:
            continue
        typical = median(comparison)
        previous = totals[index - 1]
        if typical <= 0 or previous is None:
            continue
        result[day] = BreakoutVolume(
            session=day,
            volume=current_volume,
            typical_volume=typical,
            volume_ratio=current_volume / typical,
            rising_volume=current_volume > previous,
            poc=profile.poc,
            value_low=profile.value_low,
            value_high=profile.value_high,
            profile_low=profile.low,
            profile_high=profile.high,
        )
    return result


def build_intraday_breakout_volumes(
    intraday_bars: list[Bar], *, profile_sessions: int = 5,
    volume_lookback: int = 20, min_volume_observations: int = 15,
    min_coverage: float = 0.90, expected_bars: int = 78, bins: int = 30,
) -> dict[str, BreakoutVolume]:
    """Five-minute features keyed by timestamp, using same-slot prior volume.

    Each session's profile is frozen from earlier complete sessions.  A bar's
    own completed volume may enter its RVOL numerator because execution occurs
    at the following bar, never inside the signal bar.
    """
    complete = [
        (day, rows) for day, rows in _sessions(intraday_bars)
        if len(rows) == expected_bars and _coverage(rows) >= min_coverage
    ]
    result: dict[str, BreakoutVolume] = {}
    first = max(profile_sessions, volume_lookback)
    for index in range(first, len(complete)):
        day, current = complete[index]
        prior_profile = complete[index - profile_sessions:index]
        profile = build_profile(
            [row for _, rows in prior_profile for row in rows], bins=bins
        )
        if profile is None:
            continue
        ratios: list[float | None] = []
        for position, row in enumerate(current):
            comparison = [
                complete[past][1][position].volume
                for past in range(index - volume_lookback, index)
                if complete[past][1][position].volume is not None
                and complete[past][1][position].volume > 0
            ]
            if (
                len(comparison) < min_volume_observations
                or row.volume is None or row.volume <= 0
            ):
                ratios.append(None)
                continue
            typical = median(comparison)
            ratio = row.volume / typical if typical > 0 else math.nan
            ratios.append(ratio)
            if not math.isfinite(ratio):
                continue
            previous_ratio = ratios[position - 1] if position else None
            result[row.timestamp] = BreakoutVolume(
                session=day,
                volume=row.volume,
                typical_volume=typical,
                volume_ratio=ratio,
                rising_volume=previous_ratio is not None and ratio > previous_ratio,
                poc=profile.poc,
                value_low=profile.value_low,
                value_high=profile.value_high,
                profile_low=profile.low,
                profile_high=profile.high,
            )
    return result


def _passes(
    feature: BreakoutVolume, direction: int, close: float, n: float,
    rule: VolumeFilter,
) -> tuple[bool, bool, float]:
    outside = close > feature.value_high if direction > 0 else close < feature.value_low
    distance = direction * (close - feature.poc) / n
    passed = (
        (rule.volume_floor is None or feature.volume_ratio >= rule.volume_floor)
        and (not rule.require_rising_volume or feature.rising_volume)
        and (not rule.require_outside_value or outside)
        and (rule.minimum_poc_distance_n is None or distance >= rule.minimum_poc_distance_n)
    )
    return passed, outside, distance


def run_volume_turtle(
    ticker: str, daily_bars: list[Bar], features: dict[str, BreakoutVolume], *,
    rule: VolumeFilter = VolumeFilter(),
    config: ConfirmedTurtleConfig = ConfirmedTurtleConfig(),
) -> tuple[list[VolumeTurtleTrade], VolumeTurtleAudit]:
    """Run close-confirmed System 2 and enter each accepted signal next open."""
    rows = sorted(daily_bars, key=lambda row: row.timestamp)
    atr = wilder_atr(rows, config.atr_window)
    highs = rolling_extremes([row.high for row in rows], config.entry_window, True)
    lows = rolling_extremes([row.low for row in rows], config.entry_window, False)
    exit_highs = rolling_extremes([row.high for row in rows], config.exit_window, True)
    exit_lows = rolling_extremes([row.low for row in rows], config.exit_window, False)
    warmup = max(config.entry_window, config.atr_window)
    trades: list[VolumeTurtleTrade] = []
    confirmed = missing = rejected = small_n = 0
    direction = 0
    units: list[_Unit] = []
    entry_index = 0
    stop = n_at_entry = 0.0
    signal_data: tuple[int, float, BreakoutVolume, bool, float, str] | None = None
    pending: tuple[int, float, BreakoutVolume, bool, float, str, float] | None = None

    def close_trade(index: int, price: float, reason: str) -> None:
        nonlocal direction, units, signal_data
        assert signal_data is not None
        signal_direction, channel, feature, outside, distance, signal_timestamp = signal_data
        gross = sum(direction * (price - unit.price) / unit.n for unit in units)
        basis = sum(unit.price / unit.n for unit in units)
        cost = config.round_trip_cost * basis
        sessions = len({row.timestamp[:10] for row in rows[entry_index:index + 1]})
        base = TurtleTrade(
            entry_timestamp=units[0].timestamp,
            exit_timestamp=rows[index].timestamp,
            direction=direction,
            units=len(units),
            entry=units[0].price,
            average_entry=sum(unit.price for unit in units) / len(units),
            exit=price,
            n_at_entry=n_at_entry,
            stop_at_exit=stop,
            exit_reason=reason,
            bars_held=index - entry_index,
            sessions_held=sessions,
            gross_r=gross,
            cost_r=cost,
            net_r=gross - cost,
            equity_return=(gross - cost) * config.risk_fraction,
            cost_basis_r=basis,
        )
        trades.append(VolumeTurtleTrade(
            ticker=ticker,
            signal_timestamp=signal_timestamp,
            direction=signal_direction,
            signal_close=rows[entry_index - 1].close,
            channel_level=channel,
            volume_ratio=feature.volume_ratio,
            rising_volume=feature.rising_volume,
            poc=feature.poc,
            value_low=feature.value_low,
            value_high=feature.value_high,
            outside_value=outside,
            poc_distance_n=distance,
            trade=base,
        ))
        direction = 0
        units = []
        signal_data = None

    index = warmup
    while index < len(rows):
        bar = rows[index]
        if pending is not None and direction == 0:
            candidate, channel, feature, outside, distance, stamp, signal_n = pending
            direction = candidate
            n_at_entry = signal_n
            units = [_Unit(bar.timestamp, bar.open, signal_n)]
            stop = bar.open - candidate * config.stop_atr * signal_n
            entry_index = index
            signal_data = (candidate, channel, feature, outside, distance, stamp)
            pending = None

        if direction:
            channel = exit_lows[index] if direction > 0 else exit_highs[index]
            if _breached(bar, stop, -direction):
                close_trade(index, _stop_fill(stop, bar, -direction), "stop")
                index += 1
                continue
            if not math.isnan(channel) and _breached(bar, channel, -direction):
                close_trade(index, _stop_fill(channel, bar, -direction), "channel")
                index += 1
                continue
            while len(units) < config.max_units:
                add = units[-1].price + direction * config.add_atr * n_at_entry
                if not _breached(bar, add, direction):
                    break
                fill = _stop_fill(add, bar, direction)
                units.append(_Unit(bar.timestamp, fill, n_at_entry))
                stop = fill - direction * config.stop_atr * n_at_entry
            index += 1
            continue

        if index + 1 >= len(rows) or math.isnan(atr[index]):
            index += 1
            continue
        candidate = 0
        channel = math.nan
        if 1 in config.directions and rows[index].close > highs[index]:
            candidate, channel = 1, highs[index]
        elif -1 in config.directions and rows[index].close < lows[index]:
            candidate, channel = -1, lows[index]
        if not candidate:
            index += 1
            continue
        confirmed += 1
        feature = features.get(bar.timestamp) or features.get(bar.timestamp[:10])
        if feature is None:
            missing += 1
            index += 1
            continue
        signal_n = atr[index]
        floor = max(
            config.minimum_n_fraction,
            config.minimum_n_cost_multiple * config.round_trip_cost,
        )
        if signal_n < floor * rows[index].close:
            small_n += 1
            index += 1
            continue
        passed, outside, distance = _passes(feature, candidate, bar.close, signal_n, rule)
        if not passed:
            rejected += 1
            index += 1
            continue
        pending = (
            candidate, channel, feature, outside, distance,
            bar.timestamp, signal_n,
        )
        index += 1

    if direction:
        close_trade(len(rows) - 1, rows[-1].close, "end of data")
    return trades, VolumeTurtleAudit(
        bars=len(rows), confirmed_breakouts=confirmed,
        missing_volume_profile=missing, rejected_by_filter=rejected,
        skipped_small_n=small_n, trades=len(trades),
    )
