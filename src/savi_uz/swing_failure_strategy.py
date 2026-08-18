"""Leakage-free daily-bias swing-failure strategy.

All structural and liquidity inputs are fixed from completed sessions.  The
primary setup trades a bias-aligned failure at an untouched previous-day level
and enters at the next 15-minute open.

The swing failure is defined against the *level*: price raids a resting pool and
closes back through it.  Hourly two-candle patterns (directional body, close
through the prior open, outside bar) are confluence tiers reported separately,
not part of the primary definition.
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
from savi_uz.volume_profile import Bar, build_profile

#: Confirmation tiers, from the bare level-defined failure up to the strictest
#: hourly candle conjunction.  The last two names are legacy aliases kept so
#: earlier reports stay reproducible.
CONFIRMATIONS = {
    "core",
    "directional",
    "close through open",
    "outside",
    "strong outside",
    "directional SFP",
    "close-back SFP",
}

#: Tiers that can be evaluated without a same-session preceding hourly bar.
OPENING_HOUR_TIERS = {"core", "close-back SFP", "directional", "directional SFP"}

BIAS_MODES = {"two-candle", "three-candle legs"}

#: States that carry a directional lean and may be traded with that lean.
DIRECTIONAL_STATES = {"strong bull": 1, "weak bull": 1, "strong bear": -1, "weak bear": -1}


@dataclass(frozen=True)
class SfpConfig:
    bias_mode: str = "two-candle"
    daily_structure_legs: int = 2
    trade_neutral_sessions: bool = True
    require_daily_candle_alignment: bool = True
    require_strong_daily_close: bool = False
    hourly_confirmation: str = "core"
    allow_opening_hour: bool = True
    location_mode: str = "previous day"
    target_mode: str = "opposite extreme"
    profile_sessions: int = 5
    require_outside_value: bool = False
    trend_lookback: int = 20
    require_trend_alignment: bool = False
    require_fast_rejection: bool = True
    max_outside_15m_closes: int = 1
    minimum_reward_risk: float = 2.0
    minimum_stop_cost_multiple: float = 5.0
    breakeven_trigger_r: float | None = 1.0
    max_hold_sessions: int = 1
    round_trip_cost: float = 0.0002

    def __post_init__(self) -> None:
        if self.daily_structure_legs < 1:
            raise ValueError("daily_structure_legs must be positive")
        if self.bias_mode not in BIAS_MODES:
            raise ValueError("unknown bias_mode")
        if self.hourly_confirmation not in CONFIRMATIONS:
            raise ValueError("unknown hourly_confirmation")
        if self.location_mode not in {"previous day", "previous day or week",
                                      "previous day or value edge"}:
            raise ValueError("unknown location_mode")
        if self.target_mode not in {"opposite extreme", "profile poc"}:
            raise ValueError("unknown target_mode")
        if self.profile_sessions < 1:
            raise ValueError("profile_sessions must be positive")
        if self.trend_lookback < 2:
            raise ValueError("trend_lookback must be at least two sessions")
        if self.max_outside_15m_closes < 0:
            raise ValueError("max_outside_15m_closes cannot be negative")
        if self.minimum_reward_risk <= 0 or self.max_hold_sessions < 1:
            raise ValueError("reward/risk and holding period must be positive")
        if self.minimum_stop_cost_multiple < 0:
            raise ValueError("minimum_stop_cost_multiple cannot be negative")
        if self.breakeven_trigger_r is not None and self.breakeven_trigger_r <= 0:
            raise ValueError("breakeven_trigger_r must be positive")


@dataclass(frozen=True)
class DailyBias:
    session: str
    direction: int
    state: str
    source_first: str
    source_last: str
    last_candle_aligned: bool
    last_candle_strong: bool


@dataclass(frozen=True)
class SfpTrade:
    session: str
    bias: int
    bias_state: str
    bias_source_first: str
    bias_source_last: str
    location_kind: str
    location: float
    target_kind: str
    value_high: float
    value_low: float
    poc: float
    trend: int
    signal_timestamp: str
    available_timestamp: str
    signal_session_hour: int
    previous_hour_high: float
    previous_hour_low: float
    signal_high: float
    signal_low: float
    signal_close: float
    confluence: str
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
    candidate_hours: int
    no_bias_record: int
    caution_session: int
    neutral_skipped: int
    no_daily_alignment: int
    no_daily_strength: int
    opening_hour_skipped: int
    no_entry_bar: int
    unaligned_15m_grid: int
    no_untouched_location: int
    no_swing_failure: int
    weak_hourly_confirmation: int
    slow_rejection: int
    target_not_resting: int
    no_profile: int
    against_trend: int
    inside_value_area: int
    stop_too_tight: int
    invalid_or_low_reward: int
    overlap_skipped: int
    trades: int

    @property
    def accounted(self) -> int:
        """Every candidate hour lands in exactly one bucket."""
        return (
            self.no_bias_record + self.caution_session + self.neutral_skipped
            + self.no_daily_alignment + self.no_daily_strength
            + self.opening_hour_skipped + self.no_entry_bar + self.unaligned_15m_grid
            + self.no_untouched_location + self.no_swing_failure
            + self.weak_hourly_confirmation + self.slow_rejection
            + self.target_not_resting + self.no_profile + self.against_trend
            + self.inside_value_area + self.stop_too_tight + self.invalid_or_low_reward
            + self.overlap_skipped + self.trades
        )

    def reconciles(self) -> bool:
        return self.accounted == self.candidate_hours


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


def daily_state(previous: Bar, last: Bar) -> str:
    """Classify the two-candle higher-timeframe read.

    A strong state closes beyond the prior candle's extreme, a weak state makes
    the structural transition without that close, a caution state makes the
    transition but closes against it, and an inside candle stays neutral until
    one of its extremes is raided.
    """
    higher = last.high > previous.high
    lower = last.low < previous.low
    inside = last.high <= previous.high and last.low >= previous.low
    bullish = last.close > last.open
    bearish = last.close < last.open
    if inside:
        return "neutral"
    if higher and lower:  # outside bar: only the close resolves it
        if bullish and last.close > previous.high:
            return "strong bull"
        if bearish and last.close < previous.low:
            return "strong bear"
        return "caution"
    if higher:  # higher high and higher low
        if not bullish:
            return "caution"
        return "strong bull" if last.close > previous.high else "weak bull"
    if lower:  # lower high and lower low
        if not bearish:
            return "caution"
        return "strong bear" if last.close < previous.low else "weak bear"
    return "neutral"


def build_daily_biases(daily_bars: list[Bar], config: SfpConfig) -> dict[str, DailyBias]:
    """Bias for session D uses only daily bars strictly before D."""
    rows = sorted(daily_bars, key=lambda bar: bar.timestamp)
    if config.bias_mode == "two-candle":
        return _two_candle_biases(rows)
    return _leg_biases(rows, config.daily_structure_legs)


def _two_candle_biases(rows: list[Bar]) -> dict[str, DailyBias]:
    result: dict[str, DailyBias] = {}
    for index in range(2, len(rows)):
        previous, last = rows[index - 2], rows[index - 1]
        state = daily_state(previous, last)
        result[rows[index].timestamp[:10]] = DailyBias(
            session=rows[index].timestamp[:10],
            direction=DIRECTIONAL_STATES.get(state, 0),
            state=state,
            source_first=previous.timestamp,
            source_last=last.timestamp,
            last_candle_aligned=state != "caution",
            last_candle_strong=state.startswith("strong"),
        )
    return result


def _leg_biases(rows: list[Bar], legs: int) -> dict[str, DailyBias]:
    """Legacy multi-leg trend read, retained so earlier reports reproduce."""
    width = legs + 1
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
            state=("strong " if strong else "weak ") + ("bull" if direction > 0 else "bear"),
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


def _confluence(signal: Bar, previous: Bar | None, direction: int) -> tuple[str, ...]:
    """Optional hourly two-candle confluence carried by the failure candle."""
    tags: list[str] = []
    if (signal.close > signal.open) if direction > 0 else (signal.close < signal.open):
        tags.append("directional")
    if previous is None:
        return tuple(tags)
    if signal.high > previous.high and signal.low < previous.low:
        tags.append("outside")
    if direction > 0:
        if signal.close > previous.open:
            tags.append("close through open")
        if signal.close > previous.high:
            tags.append("close beyond extreme")
    else:
        if signal.close < previous.open:
            tags.append("close through open")
        if signal.close < previous.low:
            tags.append("close beyond extreme")
    return tuple(tags)


def _confirmed(tags: tuple[str, ...], mode: str) -> bool:
    if mode in {"core", "close-back SFP"}:
        return True
    if mode in {"directional", "directional SFP"}:
        return "directional" in tags
    if mode == "close through open":
        return "directional" in tags and "close through open" in tags
    if mode == "outside":
        return "directional" in tags and "outside" in tags
    return "directional" in tags and "outside" in tags and "close beyond extreme" in tags


def _session_hours(hourly: list[Bar]) -> list[int]:
    """Index of each hourly bar within its own session."""
    result: list[int] = []
    current = ""
    offset = 0
    for bar in hourly:
        day = bar.timestamp[:10]
        offset = 0 if day != current else offset + 1
        current = day
        result.append(offset)
    return result


def session_trend(daily_bars: list[Bar], lookback: int) -> dict[str, int]:
    """Longer-horizon lean for each session, from strictly earlier closes only.

    +1 when the last completed close sits above the mean of the ``lookback``
    closes before it, -1 below, 0 when there is not yet enough history.  This is
    deliberately coarser than the two-candle state: it answers "which way has
    this market been going" rather than "what did yesterday do".
    """
    rows = sorted(daily_bars, key=lambda bar: bar.timestamp)
    result: dict[str, int] = {}
    for index in range(len(rows)):
        session = rows[index].timestamp[:10]
        if index < lookback + 1:
            result[session] = 0
            continue
        window = [bar.close for bar in rows[index - lookback - 1:index - 1]]
        last = rows[index - 1].close
        average = sum(window) / len(window)
        result[session] = 1 if last > average else (-1 if last < average else 0)
    return result


@dataclass(frozen=True)
class SessionProfile:
    """Composite value area formed from sessions completed before the session."""

    value_high: float
    value_low: float
    poc: float


def composite_profiles(
    bars: list[Bar], sessions_back: int,
) -> dict[str, SessionProfile]:
    """Composite volume profile of the ``sessions_back`` sessions before each one."""
    grouped: list[tuple[str, list[Bar]]] = []
    for bar in bars:
        day = bar.timestamp[:10]
        if not grouped or grouped[-1][0] != day:
            grouped.append((day, [bar]))
        else:
            grouped[-1][1].append(bar)
    result: dict[str, SessionProfile] = {}
    for index in range(sessions_back, len(grouped)):
        window: list[Bar] = []
        for _, rows in grouped[index - sessions_back:index]:
            window.extend(rows)
        profile = build_profile(window)
        if profile is None:
            continue
        result[grouped[index][0]] = SessionProfile(
            value_high=profile.value_high,
            value_low=profile.value_low,
            poc=profile.poc,
        )
    return result


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
    session_hours = _session_hours(hourly)
    trends = session_trend(daily_bars, config.trend_lookback)
    profiles = composite_profiles(fifteen, config.profile_sessions)
    trades: list[SfpTrade] = []
    counters = dict.fromkeys((
        "no_bias_record", "caution_session", "neutral_skipped", "no_daily_alignment",
        "no_daily_strength", "opening_hour_skipped", "no_entry_bar", "unaligned_15m_grid",
        "no_untouched_location", "no_swing_failure", "weak_hourly_confirmation",
        "slow_rejection", "target_not_resting", "no_profile", "against_trend",
        "inside_value_area", "stop_too_tight", "invalid_or_low_reward", "overlap_skipped",
    ), 0)
    candidate_hours = 0
    unavailable_until = ""

    for index in range(len(hourly) - 1):
        signal = hourly[index]
        session = signal.timestamp[:10]
        if (start and session < start) or (end and session > end):
            continue
        candidate_hours += 1

        bias = biases.get(session)
        if bias is None:
            counters["no_bias_record"] += 1
            continue
        if bias.state == "caution":
            counters["caution_session"] += 1
            continue
        if bias.direction == 0 and not config.trade_neutral_sessions:
            counters["neutral_skipped"] += 1
            continue
        if config.require_daily_candle_alignment and not bias.last_candle_aligned:
            counters["no_daily_alignment"] += 1
            continue
        if config.require_strong_daily_close and not bias.last_candle_strong:
            counters["no_daily_strength"] += 1
            continue

        opening_hour = session_hours[index] == 0
        previous = None if opening_hour else hourly[index - 1]
        needs_prior_hour = config.hourly_confirmation not in OPENING_HOUR_TIERS
        if opening_hour and (not config.allow_opening_hour or needs_prior_hour):
            counters["opening_hour_skipped"] += 1
            continue

        available = hourly[index + 1].timestamp
        if available[:10] != session:
            counters["no_entry_bar"] += 1
            continue
        if available not in positions:
            counters["unaligned_15m_grid"] += 1
            continue
        entry_index = positions[available]
        first_15 = positions.get(signal.timestamp)
        if first_15 is None or first_15 >= entry_index:
            counters["unaligned_15m_grid"] += 1
            continue

        profile = profiles.get(session)
        needs_profile = (
            config.require_outside_value
            or config.target_mode == "profile poc"
            or config.location_mode == "previous day or value edge"
        )
        if needs_profile and profile is None:
            counters["no_profile"] += 1
            continue
        trend = trends.get(session, 0)

        levels = levels_by_day.get(session, ())
        if config.location_mode == "previous day or value edge" and profile is not None:
            levels = levels + (
                LiquidityLevel("VAH", "high", profile.value_high, first_15),
                LiquidityLevel("VAL", "low", profile.value_low, first_15),
            )
        # A neutral session has no lean; whichever pool is raided sets the side.
        directions = (bias.direction,) if bias.direction else (1, -1)
        outcome = None
        reason = "no_untouched_location"
        for direction in directions:
            if config.require_trend_alignment and trend != direction:
                reason = "against_trend"
                continue
            locations = [
                level
                for level in _candidate_locations(levels, direction, config.location_mode)
                if _is_resting(level, fifteen, first_15 - 1)
            ]
            if not locations:
                continue
            reason = "no_swing_failure"
            location = _select_swept_location(locations, signal, direction)
            if location is None:
                continue
            reason = "weak_hourly_confirmation"
            tags = _confluence(signal, previous, direction)
            if not _confirmed(tags, config.hourly_confirmation):
                continue
            reason = "slow_rejection"
            component = fifteen[first_15:entry_index]
            outside_closes = sum(
                (bar.close < location.price if direction > 0 else bar.close > location.price)
                for bar in component
            )
            if config.require_fast_rejection and outside_closes > config.max_outside_15m_closes:
                continue
            reason = "inside_value_area"
            if config.require_outside_value and profile is not None:
                # The raid must have reached beyond accepted value, so the
                # failure rejects an out-of-value excursion rather than noise
                # inside the balance area.
                beyond = (
                    location.price < profile.value_low if direction > 0
                    else location.price > profile.value_high
                )
                if not beyond:
                    continue
            reason = "target_not_resting"
            if config.target_mode == "profile poc" and profile is not None:
                target_kind = "POC"
                target_price = profile.poc
            else:
                target_kind = "PDH" if direction > 0 else "PDL"
                target_level = next(
                    (level for level in levels if level.kind == target_kind), None
                )
                if target_level is None or not _is_resting(
                    target_level, fifteen, entry_index - 1
                ):
                    continue
                target_price = target_level.price
            reason = "stop_too_tight"
            entry = fifteen[entry_index].open
            stop = signal.low if direction > 0 else signal.high
            risk = direction * (entry - stop)
            reward = direction * (target_price - entry)
            # A stop closer than a few multiples of the round trip is not a
            # tradeable stop: costs alone would exceed 1R and the R multiples
            # reported downstream stop meaning anything.
            floor = config.minimum_stop_cost_multiple * config.round_trip_cost * entry
            if risk <= 0 or risk < floor:
                continue
            reason = "invalid_or_low_reward"
            if reward <= 0:
                continue
            planned_rr = reward / risk
            if planned_rr < config.minimum_reward_risk:
                continue
            outcome = (direction, location, target_kind, target_price, entry, stop,
                       planned_rr, outside_closes, tags)
            break

        if outcome is None:
            counters[reason] += 1
            continue
        (direction, location, target_kind, target_price, entry, stop,
         planned_rr, outside_closes, tags) = outcome
        if available <= unavailable_until:
            counters["overlap_skipped"] += 1
            continue

        risk = direction * (entry - stop)
        last_index = _last_allowed_index(fifteen, entry_index, config.max_hold_sessions)
        (
            exit_index, exit_price, exit_reason, both_touched,
            breakeven_activated, activation_timestamp,
        ) = execute_retest_exit(
            fifteen,
            entry_index=entry_index,
            last_index=last_index,
            direction=direction,
            entry=entry,
            stop=stop,
            target=target_price,
            breakeven_trigger_r=config.breakeven_trigger_r,
        )
        gross = direction * (exit_price / entry - 1.0)
        net = gross - config.round_trip_cost
        trades.append(SfpTrade(
            session=session,
            bias=direction,
            bias_state=bias.state,
            bias_source_first=bias.source_first,
            bias_source_last=bias.source_last,
            location_kind=location.kind,
            location=location.price,
            target_kind=target_kind,
            value_high=profile.value_high if profile else math.nan,
            value_low=profile.value_low if profile else math.nan,
            poc=profile.poc if profile else math.nan,
            trend=trend,
            signal_timestamp=signal.timestamp,
            available_timestamp=available,
            signal_session_hour=session_hours[index],
            previous_hour_high=previous.high if previous else math.nan,
            previous_hour_low=previous.low if previous else math.nan,
            signal_high=signal.high,
            signal_low=signal.low,
            signal_close=signal.close,
            confluence="+".join(tags) if tags else "none",
            outside_closes=outside_closes,
            entry_timestamp=fifteen[entry_index].timestamp,
            entry=entry,
            stop=stop,
            target=target_price,
            planned_reward_risk=planned_rr,
            breakeven_activated=breakeven_activated,
            breakeven_activation_timestamp=activation_timestamp,
            exit_timestamp=fifteen[exit_index].timestamp,
            exit=exit_price,
            exit_reason=exit_reason,
            held_overnight=session != fifteen[exit_index].timestamp[:10],
            both_touched=both_touched,
            gross_return=gross,
            net_return=net,
            net_r=net * entry / risk,
        ))
        unavailable_until = fifteen[exit_index].timestamp

    return trades, SfpAudit(
        hourly_bars=len(hourly), candidate_hours=candidate_hours,
        trades=len(trades), **counters,
    )


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
        mean_r=sum(rs) / len(rs) if rs else math.nan,
        median_r=median(rs) if rs else math.nan,
        stop_rate=sum(trade.exit_reason == "stop" for trade in trades) / len(trades),
        breakeven_rate=sum(trade.exit_reason == "breakeven" for trade in trades) / len(trades),
        target_rate=sum(trade.exit_reason == "target" for trade in trades) / len(trades),
        time_rate=sum(trade.exit_reason == "time" for trade in trades) / len(trades),
        overnight_rate=sum(trade.held_overnight for trade in trades) / len(trades),
        ending_equity=equity,
        cagr=equity ** (1 / elapsed) - 1 if elapsed > 0 and equity > 0 else math.nan,
        max_drawdown=drawdown,
    )
