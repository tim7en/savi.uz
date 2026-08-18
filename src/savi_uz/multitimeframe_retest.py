"""Leakage-free four-hour bias with a frozen 15-minute retest entry.

The higher-timeframe signal and target come from the supplied Sweep and Engulf
rule.  The lower-timeframe strategy waits for one full hour, freezes a trendline
through pivots that were already confirmed at that point, and then requires a
sweep/reclaim of the latest still-resting pivot.  Entry is the following
15-minute open.  No pivot, trendline, target, or rejection candle is allowed to
use a bar that had not closed at the decision time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from statistics import median

from savi_uz.sweep_engulf import (
    SweepConfig,
    SweepSignal,
    _fill_on_bar,
    _true_ranges,
    _wilder,
    build_signals,
)
from savi_uz.volume_profile import Bar


@dataclass(frozen=True)
class RetestConfig:
    observation_bars: int = 4
    pivot_span: int = 2
    pivot_lookback: int = 52
    atr_length: int = 14
    stop_buffer_atr: float = 0.10
    minimum_reward_risk: float = 2.0
    max_hold_sessions: int = 5
    require_aligned_slope: bool = True
    trendline_method: str = "first-hour regression"
    entry_trigger: str = "trendline rejection"
    require_external_liquidity_sweep: bool = False
    stop_method: str = "rejection"
    target_method: str = "four-hour"
    round_trip_cost: float = 0.0002

    def __post_init__(self) -> None:
        if self.observation_bars < 1 or self.pivot_span < 1 or self.pivot_lookback < 5:
            raise ValueError("observation and pivot settings must be positive")
        if self.atr_length < 1 or self.stop_buffer_atr < 0:
            raise ValueError("ATR settings are invalid")
        if self.minimum_reward_risk <= 0 or self.max_hold_sessions < 1:
            raise ValueError("reward/risk and holding period must be positive")
        if self.trendline_method not in {"first-hour regression", "confirmed pivots"}:
            raise ValueError("unknown trendline_method")
        if self.entry_trigger not in {"trendline rejection", "liquidity rejection"}:
            raise ValueError("unknown entry_trigger")
        if self.stop_method not in {"rejection", "resting liquidity"}:
            raise ValueError("unknown stop_method")
        if self.target_method not in {"four-hour", "resting liquidity"}:
            raise ValueError("unknown target_method")


@dataclass(frozen=True)
class LiquidityLevel:
    kind: str
    side: str
    price: float
    resting_from: int


@dataclass(frozen=True)
class RetestTrade:
    htf_signal_timestamp: str
    htf_available_timestamp: str
    pattern: str
    direction: int
    htf_signal_close: float
    htf_atr: float
    htf_base_stop: float
    htf_target: float
    observation_end: str
    pivot_one_timestamp: str
    pivot_two_timestamp: str
    liquidity_level: float
    trendline_slope_per_bar: float
    rejection_timestamp: str
    rejection_level: float
    entry_liquidity_kind: str
    entry_liquidity_level: float
    entry_timestamp: str
    entry: float
    stop: float
    stop_anchor_kind: str
    stop_anchor_level: float
    target: float
    target_anchor_kind: str
    target_anchor_level: float
    planned_reward_risk: float
    exit_timestamp: str
    exit: float
    exit_reason: str
    bars_held: int
    holding_sessions: int
    held_overnight: bool
    both_touched: bool
    gross_return: float
    net_return: float
    net_r: float


@dataclass(frozen=True)
class RetestAudit:
    htf_signals: int
    skipped_overlap: int
    insufficient_observation: int
    no_confirmed_trendline: int
    misaligned_trendline: int
    liquidity_not_resting: int
    thesis_expired_before_entry: int
    no_rejection: int
    no_liquidity_entry: int
    no_liquidity_stop: int
    no_liquidity_target: int
    invalid_or_low_reward: int
    trades: int


@dataclass(frozen=True)
class RetestSummary:
    count: int
    longs: int
    shorts: int
    trades_per_year: float
    win_rate: float
    mean_return: float
    median_return: float
    mean_r: float
    profit_factor: float
    target_rate: float
    stop_rate: float
    time_exit_rate: float
    overnight_rate: float
    ending_equity: float
    cagr: float
    max_drawdown: float


def confirmed_pivots(
    bars: list[Bar], *, end_index: int, span: int, lookback: int, direction: int
) -> list[int]:
    """Pivots fully confirmed by ``end_index``; future bars cannot revise them."""
    first = max(span, end_index - lookback + 1)
    last = end_index - span
    pivots: list[int] = []
    for index in range(first, last + 1):
        if direction > 0:
            value = bars[index].low
            left = [bars[i].low for i in range(index - span, index)]
            right = [bars[i].low for i in range(index + 1, index + span + 1)]
            if value < min(left) and value <= min(right):
                pivots.append(index)
        else:
            value = bars[index].high
            left = [bars[i].high for i in range(index - span, index)]
            right = [bars[i].high for i in range(index + 1, index + span + 1)]
            if value > max(left) and value >= max(right):
                pivots.append(index)
    return pivots


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


def _thesis_touched(signal: SweepSignal, bars: list[Bar], first: int, last: int) -> bool:
    for bar in bars[first:last + 1]:
        if signal.direction > 0:
            if bar.low <= signal.base_stop or bar.high >= signal.target:
                return True
        elif bar.high >= signal.base_stop or bar.low <= signal.target:
            return True
    return False


def prior_liquidity_levels(bars: list[Bar]) -> dict[str, tuple[LiquidityLevel, ...]]:
    """Map each session to levels fully known before that session began.

    Previous-day levels come from the immediately preceding trading session.
    Previous-week levels come from the preceding observed ISO trading week.  A
    level's ``resting_from`` index lets the caller reject a pool already traded
    through since it became actionable.
    """
    sessions: list[tuple[str, int, int, float, float]] = []
    first = 0
    while first < len(bars):
        day = bars[first].timestamp[:10]
        last = first
        while last + 1 < len(bars) and bars[last + 1].timestamp[:10] == day:
            last += 1
        sessions.append((day, first, last, max(row.high for row in bars[first:last + 1]),
                         min(row.low for row in bars[first:last + 1])))
        first = last + 1

    weeks: list[tuple[tuple[int, int], int, float, float]] = []
    for day, session_first, _, high, low in sessions:
        week = date.fromisoformat(day).isocalendar()[:2]
        if not weeks or weeks[-1][0] != week:
            weeks.append((week, session_first, high, low))
        else:
            key, week_first, week_high, week_low = weeks[-1]
            weeks[-1] = (key, week_first, max(week_high, high), min(week_low, low))
    prior_week = {
        weeks[index][0]: weeks[index - 1]
        for index in range(1, len(weeks))
    }
    week_starts = {week: week_first for week, week_first, _, _ in weeks}

    result: dict[str, tuple[LiquidityLevel, ...]] = {}
    for index, (day, session_first, _, _, _) in enumerate(sessions):
        levels: list[LiquidityLevel] = []
        if index:
            _, _, _, prior_high, prior_low = sessions[index - 1]
            levels.extend((
                LiquidityLevel("PDH", "high", prior_high, session_first),
                LiquidityLevel("PDL", "low", prior_low, session_first),
            ))
        week_key = date.fromisoformat(day).isocalendar()[:2]
        if week_key in prior_week:
            _, _, week_high, week_low = prior_week[week_key]
            week_first = week_starts[week_key]
            levels.extend((
                LiquidityLevel("PWH", "high", week_high, week_first),
                LiquidityLevel("PWL", "low", week_low, week_first),
            ))
        result[day] = tuple(levels)
    return result


def _is_resting(level: LiquidityLevel, bars: list[Bar], through: int) -> bool:
    """Whether a pre-known level remained untouched through ``through`` inclusive."""
    if through < level.resting_from:
        return True
    if level.side == "high":
        return all(bar.high < level.price for bar in bars[level.resting_from:through + 1])
    return all(bar.low > level.price for bar in bars[level.resting_from:through + 1])


def _swept_and_reclaimed(
    levels: tuple[LiquidityLevel, ...], bars: list[Bar], index: int, direction: int
) -> LiquidityLevel | None:
    bar = bars[index]
    candidates = []
    for level in levels:
        if not _is_resting(level, bars, index - 1):
            continue
        if direction > 0 and level.side == "low" and bar.low < level.price < bar.close:
            candidates.append(level)
        elif direction < 0 and level.side == "high" and bar.high > level.price > bar.close:
            candidates.append(level)
    if not candidates:
        return None
    return max(candidates, key=lambda level: level.price) if direction > 0 else min(
        candidates, key=lambda level: level.price
    )


def _resting_stop_anchor(
    levels: tuple[LiquidityLevel, ...], bars: list[Bar], through: int,
    direction: int, rejection: Bar,
) -> LiquidityLevel | None:
    candidates = [
        level for level in levels
        if _is_resting(level, bars, through)
        and ((direction > 0 and level.side == "low" and level.price < rejection.low)
             or (direction < 0 and level.side == "high" and level.price > rejection.high))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda level: level.price) if direction > 0 else min(
        candidates, key=lambda level: level.price
    )


def _resting_target_anchor(
    levels: tuple[LiquidityLevel, ...], bars: list[Bar], through: int,
    direction: int, entry: float, htf_target: float,
) -> LiquidityLevel | None:
    candidates = [
        level for level in levels
        if _is_resting(level, bars, through)
        and ((direction > 0 and level.side == "high" and entry < level.price <= htf_target)
             or (direction < 0 and level.side == "low" and entry > level.price >= htf_target))
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda level: level.price) if direction > 0 else max(
        candidates, key=lambda level: level.price
    )


def run_retest_strategy(
    htf_bars: list[Bar],
    entry_bars: list[Bar],
    *,
    htf_config: SweepConfig = SweepConfig(),
    config: RetestConfig = RetestConfig(),
    start: str | None = None,
    end: str | None = None,
) -> tuple[list[RetestTrade], RetestAudit]:
    """Build and execute the nested strategy, returning trades and rejection audit."""
    lower = sorted(entry_bars, key=lambda row: row.timestamp)
    signals = build_signals(htf_bars, htf_config, start=start, end=end)
    positions = {bar.timestamp: index for index, bar in enumerate(lower)}
    atrs = _wilder(_true_ranges(lower), config.atr_length)
    liquidity_by_day = prior_liquidity_levels(lower)
    trades: list[RetestTrade] = []
    counters = {
        "skipped_overlap": 0,
        "insufficient_observation": 0,
        "no_confirmed_trendline": 0,
        "misaligned_trendline": 0,
        "liquidity_not_resting": 0,
        "thesis_expired_before_entry": 0,
        "no_rejection": 0,
        "no_liquidity_entry": 0,
        "no_liquidity_stop": 0,
        "no_liquidity_target": 0,
        "invalid_or_low_reward": 0,
    }
    available_after = ""

    for signal in signals:
        if signal.available_timestamp <= available_after:
            counters["skipped_overlap"] += 1
            continue
        first = positions.get(signal.available_timestamp)
        if first is None:
            counters["insufficient_observation"] += 1
            continue
        observation_end = first + config.observation_bars - 1
        if observation_end >= len(lower) or lower[observation_end].timestamp[:10] != lower[first].timestamp[:10]:
            counters["insufficient_observation"] += 1
            continue
        if _thesis_touched(signal, lower, first, observation_end):
            counters["thesis_expired_before_entry"] += 1
            continue

        if config.trendline_method == "first-hour regression":
            pivot_one, pivot_two = first, observation_end
            values = [
                bar.low if signal.direction > 0 else bar.high
                for bar in lower[first:observation_end + 1]
            ]
            centre_x = (len(values) - 1) / 2
            centre_y = sum(values) / len(values)
            denominator = sum((index - centre_x) ** 2 for index in range(len(values)))
            slope = (
                sum((index - centre_x) * (value - centre_y) for index, value in enumerate(values))
                / denominator
                if denominator else 0.0
            )
            intercept = centre_y - slope * centre_x
            price_one = intercept
            price_two = intercept + slope * (len(values) - 1)
        else:
            pivots = confirmed_pivots(
                lower,
                end_index=observation_end,
                span=config.pivot_span,
                lookback=config.pivot_lookback,
                direction=signal.direction,
            )
            if len(pivots) < 2:
                counters["no_confirmed_trendline"] += 1
                continue
            pivot_one, pivot_two = pivots[-2:]
            price_one = lower[pivot_one].low if signal.direction > 0 else lower[pivot_one].high
            price_two = lower[pivot_two].low if signal.direction > 0 else lower[pivot_two].high
            slope = (price_two - price_one) / (pivot_two - pivot_one)
        if config.require_aligned_slope and signal.direction * slope < 0:
            counters["misaligned_trendline"] += 1
            continue
        if config.trendline_method == "confirmed pivots":
            if signal.direction > 0:
                resting = all(lower[i].low >= price_two for i in range(pivot_two + 1, observation_end + 1))
            else:
                resting = all(lower[i].high <= price_two for i in range(pivot_two + 1, observation_end + 1))
            if not resting:
                counters["liquidity_not_resting"] += 1
                continue

        session_day = lower[first].timestamp[:10]
        session_end = observation_end
        while session_end + 1 < len(lower) and lower[session_end + 1].timestamp[:10] == session_day:
            session_end += 1
        rejection_index: int | None = None
        rejection_line = math.nan
        entry_liquidity: LiquidityLevel | None = None
        saw_plain_rejection = False
        expired = False
        session_levels = liquidity_by_day.get(session_day, ())
        # The final session bar cannot trigger: its next available open is overnight.
        for index in range(observation_end + 1, session_end):
            if _thesis_touched(signal, lower, index, index):
                expired = True
                break
            atr = atrs[index]
            if atr is None:
                continue
            if config.trendline_method == "first-hour regression":
                line = intercept + slope * (index - first)
            else:
                line = price_two + slope * (index - pivot_two)
            bar = lower[index]
            tolerance = config.stop_buffer_atr * atr
            if config.trendline_method == "first-hour regression" and signal.direction > 0:
                trigger = bar.low < line and bar.close > line and bar.close > bar.open
            elif config.trendline_method == "first-hour regression":
                trigger = bar.high > line and bar.close < line and bar.close < bar.open
            elif signal.direction > 0:
                trigger = (
                    bar.low < price_two and bar.low <= line + tolerance
                    and bar.close > price_two and bar.close > line and bar.close > bar.open
                )
            else:
                trigger = (
                    bar.high > price_two and bar.high >= line - tolerance
                    and bar.close < price_two and bar.close < line and bar.close < bar.open
                )
            swept = _swept_and_reclaimed(session_levels, lower, index, signal.direction)
            if config.entry_trigger == "liquidity rejection":
                directional_close = bar.close > bar.open if signal.direction > 0 else bar.close < bar.open
                correct_side_of_line = bar.close > line if signal.direction > 0 else bar.close < line
                trigger = swept is not None and directional_close and correct_side_of_line
            if trigger:
                saw_plain_rejection = True
                if config.require_external_liquidity_sweep and swept is None:
                    continue
                rejection_index, rejection_line = index, line
                entry_liquidity = swept
                break
        if expired:
            counters["thesis_expired_before_entry"] += 1
            continue
        if rejection_index is None:
            liquidity_missing = config.entry_trigger == "liquidity rejection" or (
                config.require_external_liquidity_sweep and saw_plain_rejection
            )
            counters["no_liquidity_entry" if liquidity_missing else "no_rejection"] += 1
            continue

        entry_index = rejection_index + 1
        rejection = lower[rejection_index]
        entry = lower[entry_index].open
        atr = atrs[rejection_index]
        assert atr is not None
        stop_anchor: LiquidityLevel | None = None
        if config.stop_method == "resting liquidity":
            stop_anchor = _resting_stop_anchor(
                session_levels, lower, rejection_index, signal.direction, rejection
            )
            if stop_anchor is None:
                counters["no_liquidity_stop"] += 1
                continue
            stop = stop_anchor.price - signal.direction * config.stop_buffer_atr * atr
            if signal.direction * (stop - signal.base_stop) < 0:
                counters["no_liquidity_stop"] += 1
                continue
        else:
            stop = (
                rejection.low - config.stop_buffer_atr * atr
                if signal.direction > 0
                else rejection.high + config.stop_buffer_atr * atr
            )
        target_anchor: LiquidityLevel | None = None
        target = signal.target
        if config.target_method == "resting liquidity":
            target_anchor = _resting_target_anchor(
                session_levels, lower, rejection_index, signal.direction, entry, signal.target
            )
            if target_anchor is None:
                counters["no_liquidity_target"] += 1
                continue
            target = target_anchor.price
        risk = signal.direction * (entry - stop)
        reward = signal.direction * (target - entry)
        planned_rr = reward / risk if risk > 0 else -math.inf
        if risk <= 0 or reward <= 0 or planned_rr < config.minimum_reward_risk:
            counters["invalid_or_low_reward"] += 1
            continue

        last_index = _last_allowed_index(lower, entry_index, config.max_hold_sessions)
        fill: tuple[float, str, bool] | None = None
        exit_index = last_index
        for index in range(entry_index, last_index + 1):
            fill = _fill_on_bar(lower[index], signal.direction, stop, target)
            if fill is not None:
                exit_index = index
                break
        if fill is None:
            fill = (lower[last_index].close, "time", False)
        exit_price, reason, both_touched = fill
        gross = signal.direction * (exit_price / entry - 1.0)
        net = gross - config.round_trip_cost
        net_r = net * entry / risk
        entry_day, exit_day = lower[entry_index].timestamp[:10], lower[exit_index].timestamp[:10]
        holding_sessions = len({
            lower[index].timestamp[:10] for index in range(entry_index, exit_index + 1)
        })
        trades.append(
            RetestTrade(
                htf_signal_timestamp=signal.timestamp,
                htf_available_timestamp=signal.available_timestamp,
                pattern=signal.pattern,
                direction=signal.direction,
                htf_signal_close=signal.signal_close,
                htf_atr=signal.atr,
                htf_base_stop=signal.base_stop,
                htf_target=signal.target,
                observation_end=lower[observation_end].timestamp,
                pivot_one_timestamp=lower[pivot_one].timestamp,
                pivot_two_timestamp=lower[pivot_two].timestamp,
                liquidity_level=rejection_line if config.trendline_method == "first-hour regression" else price_two,
                trendline_slope_per_bar=slope,
                rejection_timestamp=rejection.timestamp,
                rejection_level=rejection_line,
                entry_liquidity_kind=entry_liquidity.kind if entry_liquidity else "",
                entry_liquidity_level=entry_liquidity.price if entry_liquidity else math.nan,
                entry_timestamp=lower[entry_index].timestamp,
                entry=entry,
                stop=stop,
                stop_anchor_kind=stop_anchor.kind if stop_anchor else "rejection",
                stop_anchor_level=stop_anchor.price if stop_anchor else (
                    rejection.low if signal.direction > 0 else rejection.high
                ),
                target=target,
                target_anchor_kind=target_anchor.kind if target_anchor else "4h",
                target_anchor_level=target_anchor.price if target_anchor else signal.target,
                planned_reward_risk=planned_rr,
                exit_timestamp=lower[exit_index].timestamp,
                exit=exit_price,
                exit_reason=reason,
                bars_held=exit_index - entry_index + 1,
                holding_sessions=holding_sessions,
                held_overnight=entry_day != exit_day,
                both_touched=both_touched,
                gross_return=gross,
                net_return=net,
                net_r=net_r,
            )
        )
        available_after = lower[exit_index].timestamp

    return trades, RetestAudit(htf_signals=len(signals), trades=len(trades), **counters)


def summarise_retests(trades: list[RetestTrade]) -> RetestSummary:
    if not trades:
        return RetestSummary(0, 0, 0, *(math.nan for _ in range(13)))
    returns = [trade.net_return for trade in trades]
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
    finite_r = [trade.net_r for trade in trades if math.isfinite(trade.net_r)]
    return RetestSummary(
        count=len(trades),
        longs=sum(trade.direction > 0 for trade in trades),
        shorts=sum(trade.direction < 0 for trade in trades),
        trades_per_year=len(trades) / elapsed if elapsed > 0 else math.nan,
        win_rate=sum(value > 0 for value in returns) / len(returns),
        mean_return=sum(returns) / len(returns),
        median_return=median(returns),
        mean_r=sum(finite_r) / len(finite_r),
        profit_factor=gains / losses if losses else math.inf,
        target_rate=sum("target" in trade.exit_reason for trade in trades) / len(trades),
        stop_rate=sum("stop" in trade.exit_reason for trade in trades) / len(trades),
        time_exit_rate=sum(trade.exit_reason == "time" for trade in trades) / len(trades),
        overnight_rate=sum(trade.held_overnight for trade in trades) / len(trades),
        ending_equity=equity,
        cagr=equity ** (1.0 / elapsed) - 1.0 if elapsed > 0 and equity > 0 else math.nan,
        max_drawdown=drawdown,
    )
