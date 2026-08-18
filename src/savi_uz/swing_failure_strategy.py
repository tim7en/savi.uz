"""Leakage-free daily-bias swing-failure strategy.

All structural and liquidity inputs are fixed from completed sessions.  The
primary setup trades a bias-aligned failure at an untouched previous-day level,
confirmed by a completed hourly outside candle, at the next 15-minute open.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from statistics import median

from savi_uz.multitimeframe_retest import (
    LiquidityLevel,
    _is_resting,
    execute_retest_exit,
    prior_liquidity_levels,
)
from savi_uz.volume_profile import Bar


@dataclass(frozen=True)
class SfpConfig:
    daily_structure_legs: int = 2
    require_daily_candle_alignment: bool = True
    require_strong_daily_close: bool = False
    hourly_confirmation: str = "strong outside"
    location_mode: str = "previous day"
    require_fast_rejection: bool = True
    max_outside_15m_closes: int = 1
    minimum_reward_risk: float = 2.0
    breakeven_trigger_r: float | None = 1.0
    max_hold_sessions: int = 1
    round_trip_cost: float = 0.0002

    def __post_init__(self) -> None:
        if self.daily_structure_legs < 1:
            raise ValueError("daily_structure_legs must be positive")
        if self.hourly_confirmation not in {
            "strong outside", "outside", "directional SFP", "close-back SFP"
        }:
            raise ValueError("unknown hourly_confirmation")
        if self.location_mode not in {"previous day", "previous day or week"}:
            raise ValueError("unknown location_mode")
        if self.max_outside_15m_closes < 0:
            raise ValueError("max_outside_15m_closes cannot be negative")
        if self.minimum_reward_risk <= 0 or self.max_hold_sessions < 1:
            raise ValueError("reward/risk and holding period must be positive")
        if self.breakeven_trigger_r is not None and self.breakeven_trigger_r <= 0:
            raise ValueError("breakeven_trigger_r must be positive")


@dataclass(frozen=True)
class DailyBias:
    session: str
    direction: int
    source_first: str
    source_last: str
    last_candle_aligned: bool
    last_candle_strong: bool


@dataclass(frozen=True)
class SfpTrade:
    session: str
    bias: int
    bias_source_first: str
    bias_source_last: str
    location_kind: str
    location: float
    target_kind: str
    signal_timestamp: str
    available_timestamp: str
    previous_hour_high: float
    previous_hour_low: float
    signal_high: float
    signal_low: float
    signal_close: float
    outside_closes: int
    entry_timestamp: str
    entry: float
    stop: float
    target: float
    planned_reward_risk: float
    breakeven_activated: bool
    breakeven_activation_timestamp: str
    exit_timestamp: str
    exit: float
    exit_reason: str
    held_overnight: bool
    both_touched: bool
    gross_return: float
    net_return: float
    net_r: float


@dataclass(frozen=True)
class SfpAudit:
    hourly_bars: int
    biased_hours: int
    no_daily_alignment: int
    no_daily_strength: int
    no_untouched_location: int
    no_swing_failure: int
    weak_hourly_confirmation: int
    slow_rejection: int
    target_not_resting: int
    invalid_or_low_reward: int
    overlap_skipped: int
    trades: int


@dataclass(frozen=True)
class SfpSummary:
    count: int
    longs: int
    shorts: int
    win_rate: float
    profit_factor: float
    mean_r: float
    median_r: float
    stop_rate: float
    breakeven_rate: float
    target_rate: float
    time_rate: float
    overnight_rate: float
    ending_equity: float
    cagr: float
    max_drawdown: float


def build_daily_biases(daily_bars: list[Bar], config: SfpConfig) -> dict[str, DailyBias]:
    """Bias for session D uses only daily bars strictly before D."""
    rows = sorted(daily_bars, key=lambda bar: bar.timestamp)
    width = config.daily_structure_legs + 1
    result: dict[str, DailyBias] = {}
    for index in range(width, len(rows)):
        history = rows[index - width:index]
        highs = [bar.high for bar in history]
        lows = [bar.low for bar in history]
        bullish = all(highs[i] > highs[i - 1] and lows[i] > lows[i - 1]
                      for i in range(1, len(history)))
        bearish = all(highs[i] < highs[i - 1] and lows[i] < lows[i - 1]
                      for i in range(1, len(history)))
        if not bullish and not bearish:
            continue
        direction = 1 if bullish else -1
        last, previous = history[-1], history[-2]
        aligned = last.close > last.open if direction > 0 else last.close < last.open
        outside = last.high > previous.high and last.low < previous.low
        strong = (
            (last.close > previous.high or (outside and last.close > last.open))
            if direction > 0
            else (last.close < previous.low or (outside and last.close < last.open))
        )
        result[rows[index].timestamp[:10]] = DailyBias(
            session=rows[index].timestamp[:10],
            direction=direction,
            source_first=history[0].timestamp,
            source_last=history[-1].timestamp,
            last_candle_aligned=aligned,
            last_candle_strong=strong,
        )
    return result


def _last_allowed_index(bars: list[Bar], entry_index: int, sessions: int) -> int:
    days: list[str] = []
    last = entry_index
    for index in range(entry_index, len(bars)):
        day = bars[index].timestamp[:10]
        if not days or day != days[-1]:
            if len(days) >= sessions:
                break
            days.append(day)
        last = index
    return last


def _candidate_locations(
    levels: tuple[LiquidityLevel, ...], direction: int, mode: str,
) -> list[LiquidityLevel]:
    kinds = {"PDL"} if direction > 0 else {"PDH"}
    if mode == "previous day or week":
        kinds.add("PWL" if direction > 0 else "PWH")
    return [level for level in levels if level.kind in kinds]


def _select_swept_location(
    levels: list[LiquidityLevel], bar: Bar, direction: int,
) -> LiquidityLevel | None:
    swept = [
        level for level in levels
        if ((direction > 0 and bar.low < level.price < bar.close)
            or (direction < 0 and bar.high > level.price > bar.close))
    ]
    if not swept:
        return None
    return max(swept, key=lambda level: level.price) if direction > 0 else min(
        swept, key=lambda level: level.price
    )


def run_sfp_strategy(
    daily_bars: list[Bar], hourly_bars: list[Bar], fifteen_bars: list[Bar],
    *, config: SfpConfig = SfpConfig(), start: str | None = None,
    end: str | None = None,
) -> tuple[list[SfpTrade], SfpAudit]:
    hourly = sorted(hourly_bars, key=lambda bar: bar.timestamp)
    fifteen = sorted(fifteen_bars, key=lambda bar: bar.timestamp)
    positions = {bar.timestamp: index for index, bar in enumerate(fifteen)}
    biases = build_daily_biases(daily_bars, config)
    levels_by_day = prior_liquidity_levels(fifteen)
    trades: list[SfpTrade] = []
    counters = {
        "biased_hours": 0,
        "no_daily_alignment": 0,
        "no_daily_strength": 0,
        "no_untouched_location": 0,
        "no_swing_failure": 0,
        "weak_hourly_confirmation": 0,
        "slow_rejection": 0,
        "target_not_resting": 0,
        "invalid_or_low_reward": 0,
        "overlap_skipped": 0,
    }
    unavailable_until = ""

    for index in range(1, len(hourly) - 1):
        signal, previous = hourly[index], hourly[index - 1]
        session = signal.timestamp[:10]
        if (start and session < start) or (end and session > end):
            continue
        bias = biases.get(session)
        if bias is None or previous.timestamp[:10] != session:
            continue
        counters["biased_hours"] += 1
        if config.require_daily_candle_alignment and not bias.last_candle_aligned:
            counters["no_daily_alignment"] += 1
            continue
        if config.require_strong_daily_close and not bias.last_candle_strong:
            counters["no_daily_strength"] += 1
            continue

        available = hourly[index + 1].timestamp
        if available[:10] != session or available not in positions:
            continue
        entry_index = positions[available]
        first_15 = positions.get(signal.timestamp)
        if first_15 is None or first_15 >= entry_index:
            continue
        levels = levels_by_day.get(session, ())
        locations = [
            level for level in _candidate_locations(levels, bias.direction, config.location_mode)
            if _is_resting(level, fifteen, first_15 - 1)
        ]
        if not locations:
            counters["no_untouched_location"] += 1
            continue
        location = _select_swept_location(locations, signal, bias.direction)
        if location is None:
            counters["no_swing_failure"] += 1
            continue

        outside = signal.high > previous.high and signal.low < previous.low
        directional = signal.close > signal.open if bias.direction > 0 else signal.close < signal.open
        strong_close = signal.close > previous.high if bias.direction > 0 else signal.close < previous.low
        if config.hourly_confirmation == "strong outside":
            confirmed = outside and directional and strong_close
        elif config.hourly_confirmation == "outside":
            confirmed = outside and directional
        elif config.hourly_confirmation == "directional SFP":
            confirmed = directional
        else:
            confirmed = True
        if not confirmed:
            counters["weak_hourly_confirmation"] += 1
            continue

        component = fifteen[first_15:entry_index]
        outside_closes = sum(
            (bar.close < location.price if bias.direction > 0 else bar.close > location.price)
            for bar in component
        )
        if config.require_fast_rejection and outside_closes > config.max_outside_15m_closes:
            counters["slow_rejection"] += 1
            continue

        target_kind = "PDH" if bias.direction > 0 else "PDL"
        target_level = next((level for level in levels if level.kind == target_kind), None)
        if target_level is None or not _is_resting(target_level, fifteen, entry_index - 1):
            counters["target_not_resting"] += 1
            continue
        entry = fifteen[entry_index].open
        stop = signal.low if bias.direction > 0 else signal.high
        target = target_level.price
        risk = bias.direction * (entry - stop)
        reward = bias.direction * (target - entry)
        planned_rr = reward / risk if risk > 0 else -math.inf
        if risk <= 0 or reward <= 0 or planned_rr < config.minimum_reward_risk:
            counters["invalid_or_low_reward"] += 1
            continue
        if available <= unavailable_until:
            counters["overlap_skipped"] += 1
            continue

        last_index = _last_allowed_index(fifteen, entry_index, config.max_hold_sessions)
        (
            exit_index, exit_price, reason, both_touched,
            breakeven_activated, activation_timestamp,
        ) = execute_retest_exit(
            fifteen,
            entry_index=entry_index,
            last_index=last_index,
            direction=bias.direction,
            entry=entry,
            stop=stop,
            target=target,
            breakeven_trigger_r=config.breakeven_trigger_r,
        )
        gross = bias.direction * (exit_price / entry - 1.0)
        net = gross - config.round_trip_cost
        net_r = net * entry / risk
        trades.append(SfpTrade(
            session=session,
            bias=bias.direction,
            bias_source_first=bias.source_first,
            bias_source_last=bias.source_last,
            location_kind=location.kind,
            location=location.price,
            target_kind=target_kind,
            signal_timestamp=signal.timestamp,
            available_timestamp=available,
            previous_hour_high=previous.high,
            previous_hour_low=previous.low,
            signal_high=signal.high,
            signal_low=signal.low,
            signal_close=signal.close,
            outside_closes=outside_closes,
            entry_timestamp=fifteen[entry_index].timestamp,
            entry=entry,
            stop=stop,
            target=target,
            planned_reward_risk=planned_rr,
            breakeven_activated=breakeven_activated,
            breakeven_activation_timestamp=activation_timestamp,
            exit_timestamp=fifteen[exit_index].timestamp,
            exit=exit_price,
            exit_reason=reason,
            held_overnight=session != fifteen[exit_index].timestamp[:10],
            both_touched=both_touched,
            gross_return=gross,
            net_return=net,
            net_r=net_r,
        ))
        unavailable_until = fifteen[exit_index].timestamp

    return trades, SfpAudit(hourly_bars=len(hourly), trades=len(trades), **counters)


def summarise_sfp(trades: list[SfpTrade]) -> SfpSummary:
    if not trades:
        return SfpSummary(0, 0, 0, *(math.nan for _ in range(12)))
    returns = [trade.net_return for trade in trades]
    rs = [trade.net_r for trade in trades if math.isfinite(trade.net_r)]
    gains = sum(value for value in returns if value > 0)
    losses = -sum(value for value in returns if value < 0)
    equity = peak = 1.0
    drawdown = 0.0
    for value in returns:
        equity *= max(0.0, 1.0 + value)
        peak = max(peak, equity)
        drawdown = min(drawdown, equity / peak - 1.0)
    elapsed = (
        date.fromisoformat(trades[-1].exit_timestamp[:10])
        - date.fromisoformat(trades[0].entry_timestamp[:10])
    ).days / 365.25
    return SfpSummary(
        count=len(trades),
        longs=sum(trade.bias > 0 for trade in trades),
        shorts=sum(trade.bias < 0 for trade in trades),
        win_rate=sum(value > 0 for value in returns) / len(returns),
        profit_factor=gains / losses if losses else math.inf,
        mean_r=sum(rs) / len(rs),
        median_r=median(rs),
        stop_rate=sum(trade.exit_reason == "stop" for trade in trades) / len(trades),
        breakeven_rate=sum(trade.exit_reason == "breakeven" for trade in trades) / len(trades),
        target_rate=sum(trade.exit_reason == "target" for trade in trades) / len(trades),
        time_rate=sum(trade.exit_reason == "time" for trade in trades) / len(trades),
        overnight_rate=sum(trade.held_overnight for trade in trades) / len(trades),
        ending_equity=equity,
        cagr=equity ** (1 / elapsed) - 1 if elapsed > 0 and equity > 0 else math.nan,
        max_drawdown=drawdown,
    )
