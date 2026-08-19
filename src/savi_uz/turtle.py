"""Turtle breakout system, faithful to the original rules and interval agnostic.

The engine takes any bar series -- daily, 30-minute, 15-minute, five-minute --
and applies the same mechanics, so the question "does this survive at intraday
resolution" is answered by changing the input rather than the rules.

What is implemented, following the original published rules:

* ``N`` is Wilder's average true range over ``atr_window`` bars, updated as
  ``N = ((window - 1) * previous + true_range) / window``.
* Entry on a close-independent breach of the ``entry_window`` channel formed by
  the bars strictly before the current one, filled with a stop order at the
  channel edge, or at the open when the bar gaps through it.
* Units are sized so a 1N move equals ``risk_fraction`` of equity. Up to
  ``max_units`` are pyramided at ``add_atr`` intervals in favour.
* The protective stop sits ``stop_atr`` from the most recent unit, so adding a
  unit pulls every stop up behind it.
* Exit on the opposite ``exit_window`` channel, again stop-filled and gap aware.
* System 1's filter: a breakout is skipped when the previous breakout in that
  market would have been a winner. Every breakout is tracked as a phantom trade
  whether or not it was taken, exactly as the rule requires.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median

from savi_uz.volume_profile import Bar


@dataclass(frozen=True)
class TurtleConfig:
    entry_window: int = 20
    exit_window: int = 10
    atr_window: int = 20
    stop_atr: float = 2.0
    add_atr: float = 0.5
    max_units: int = 4
    risk_fraction: float = 0.01
    #: The N floor is the larger of an absolute fraction of price and a
    #: multiple of the round trip. Holding the absolute one fixed lets a cost
    #: sweep vary pricing without also changing which trades are taken.
    minimum_n_cost_multiple: float = 5.0
    minimum_n_fraction: float = 0.0
    skip_after_winner: bool = True
    round_trip_cost: float = 0.0002
    allow_overnight: bool = True
    directions: tuple[int, ...] = (1, -1)
    #: Higher-timeframe filter. "sma" takes a breakout only when it points the
    #: same way as price sits relative to its ``trend_window`` mean.
    trend_filter: str = "none"
    trend_window: int = 200
    #: "stop" fills intrabar at the channel edge, which is the original rule
    #: but leaves the breakout bar's volume unknowable at entry. "close
    #: confirm" waits for the bar to close beyond the channel and enters at
    #: the next open, giving up entry price to make that volume observable.
    entry_mode: str = "stop"
    min_relative_volume: float = 0.0
    volume_window: int = 20
    #: Exit extensions. A chandelier is based on the most favourable completed
    #: bar since entry and the N observed before entry. Any tightened stop only
    #: becomes active on the following bar.
    use_channel_exit: bool = True
    chandelier_atr: float | None = None
    breakeven_trigger_n: float | None = None

    def __post_init__(self) -> None:
        if min(self.entry_window, self.exit_window, self.atr_window) < 2:
            raise ValueError("windows must span at least two bars")
        if self.exit_window >= self.entry_window:
            raise ValueError("the exit channel must be shorter than the entry channel")
        if self.stop_atr <= 0 or self.add_atr <= 0:
            raise ValueError("stop and add distances must be positive")
        if self.max_units < 1:
            raise ValueError("max_units must be positive")
        if not 0 < self.risk_fraction < 1:
            raise ValueError("risk_fraction must be a fraction of equity")
        if self.round_trip_cost < 0:
            raise ValueError("round_trip_cost cannot be negative")
        if self.minimum_n_cost_multiple < 0 or self.minimum_n_fraction < 0:
            raise ValueError("the N floor cannot be negative")
        if not self.directions or set(self.directions) - {1, -1}:
            raise ValueError("directions must be a non-empty subset of (1, -1)")
        if self.trend_filter not in {"none", "sma"}:
            raise ValueError("unknown trend_filter")
        if self.trend_window < 2:
            raise ValueError("trend_window must span at least two bars")
        if self.entry_mode not in {"stop", "close confirm"}:
            raise ValueError("unknown entry_mode")
        if self.min_relative_volume < 0:
            raise ValueError("min_relative_volume cannot be negative")
        if self.min_relative_volume and self.entry_mode != "close confirm":
            raise ValueError(
                "a volume filter needs close-confirmed entry: a stop order fills "
                "before the breakout bar's volume is known"
            )
        if self.volume_window < 2:
            raise ValueError("volume_window must span at least two bars")
        if self.chandelier_atr is not None and self.chandelier_atr <= 0:
            raise ValueError("chandelier_atr must be positive")
        if self.breakeven_trigger_n is not None and self.breakeven_trigger_n <= 0:
            raise ValueError("breakeven_trigger_n must be positive")
        if not self.use_channel_exit and self.chandelier_atr is None:
            raise ValueError("at least one trailing exit must be enabled")


@dataclass(frozen=True)
class TurtleUnit:
    timestamp: str
    price: float
    n: float


@dataclass(frozen=True)
class TurtleTrade:
    entry_timestamp: str
    exit_timestamp: str
    direction: int
    units: int
    entry: float
    average_entry: float
    exit: float
    n_at_entry: float
    stop_at_exit: float
    exit_reason: str
    bars_held: int
    sessions_held: int
    gross_r: float
    cost_r: float
    net_r: float
    equity_return: float
    cost_basis_r: float = 0.0
    unit_entries: tuple[TurtleUnit, ...] = ()


@dataclass(frozen=True)
class TurtleAudit:
    """Where every breakout went, so the trade list can be reconciled."""

    bars: int
    breakouts: int
    skipped_after_winner: int
    skipped_small_n: int
    skipped_against_trend: int
    skipped_thin_volume: int
    trades: int


@dataclass(frozen=True)
class TurtleSummary:
    trades: int
    units: int
    longs: int
    shorts: int
    win_rate: float
    profit_factor: float
    mean_r: float
    median_r: float
    total_r: float
    largest_win_r: float
    largest_loss_r: float
    stop_rate: float
    channel_exit_rate: float
    mean_bars_held: float
    ending_equity: float
    max_drawdown: float
    skipped: int


def true_ranges(bars: list[Bar]) -> list[float]:
    """True range per bar; the first bar has no previous close to reach back to."""
    result = [bars[0].high - bars[0].low] if bars else []
    for index in range(1, len(bars)):
        current, previous = bars[index], bars[index - 1]
        result.append(max(
            current.high - current.low,
            abs(current.high - previous.close),
            abs(current.low - previous.close),
        ))
    return result


def wilder_atr(bars: list[Bar], window: int) -> list[float]:
    """``N`` at each bar, using only that bar and the ones before it.

    Index ``i`` holds the value known once bar ``i`` has closed, so a decision
    taken on bar ``i + 1`` may read ``atr[i]`` without looking ahead.
    """
    ranges = true_ranges(bars)
    result: list[float] = [math.nan] * len(bars)
    if len(bars) < window:
        return result
    seed = sum(ranges[:window]) / window
    result[window - 1] = seed
    running = seed
    for index in range(window, len(bars)):
        running = ((window - 1) * running + ranges[index]) / window
        result[index] = running
    return result


def rolling_extremes(
    values: list[float], window: int, largest: bool,
) -> list[float]:
    """Sliding max (or min) of the ``window`` values *before* each index.

    Monotonic-deque scan, so a five-minute series costs one pass rather than one
    pass per bar.  Index ``i`` holds the extreme over ``values[i - window:i]``
    and is NaN until a full window exists.
    """
    result: list[float] = [math.nan] * len(values)
    queue: list[int] = []
    for index in range(len(values)):
        if index >= window:
            result[index] = values[queue[0]]
        while queue and (
            values[queue[-1]] <= values[index] if largest
            else values[queue[-1]] >= values[index]
        ):
            queue.pop()
        queue.append(index)
        if queue[0] <= index - window:
            queue.pop(0)
    return result


def relative_volume(bars: list[Bar], window: int) -> list[float]:
    """Each bar's volume against the mean of the ``window`` bars before it."""
    result: list[float] = [math.nan] * len(bars)
    running = 0.0
    for index, bar in enumerate(bars):
        if index >= window:
            mean = running / window
            if mean > 0 and bar.volume is not None:
                result[index] = bar.volume / mean
            running -= bars[index - window].volume or 0.0
        running += bar.volume or 0.0
    return result


def trailing_mean(values: list[float], window: int) -> list[float]:
    """Mean of the ``window`` values ending at each index, NaN until filled.

    Index ``i`` includes ``values[i]``, so a decision taken on bar ``i + 1`` may
    read index ``i`` without seeing its own future.
    """
    result: list[float] = [math.nan] * len(values)
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        if index >= window - 1:
            result[index] = running / window
    return result


def _channel(bars: list[Bar], index: int, window: int) -> tuple[float, float]:
    """Highest high and lowest low over the ``window`` bars before ``index``."""
    window_bars = bars[index - window:index]
    return (
        max(bar.high for bar in window_bars),
        min(bar.low for bar in window_bars),
    )


def _stop_fill(level: float, bar: Bar, direction: int) -> float:
    """A stop order at ``level`` fills there, or at the open when price gaps past."""
    return max(level, bar.open) if direction > 0 else min(level, bar.open)


def _breached(bar: Bar, level: float, direction: int) -> bool:
    return bar.high > level if direction > 0 else bar.low < level


@dataclass
class _Phantom:
    """A breakout replayed as a single unit, whether or not it was taken.

    System 1 needs the outcome of the *previous* breakout.  Resolving it eagerly
    by scanning forward would let a decision read bars that had not happened
    yet, so it is advanced one bar at a time alongside the real position and
    only consulted once it has closed.
    """

    direction: int
    entry: float
    stop: float

    def step(self, bar: Bar, exit_level: float) -> bool | None:
        """Advance through one bar; returns the outcome once it closes."""
        if _breached(bar, self.stop, -self.direction):
            return False
        if math.isnan(exit_level):
            return None
        if _breached(bar, exit_level, -self.direction):
            fill = _stop_fill(exit_level, bar, -self.direction)
            return self.direction * (fill - self.entry) > 0
        return None


def run_turtle(
    bars: list[Bar], *, config: TurtleConfig = TurtleConfig(),
    entries: dict[int, int] | None = None,
) -> tuple[list[TurtleTrade], TurtleAudit]:
    """Replay the system over ``bars``; returns the trades and an audit.

    ``entries`` replaces breakout detection with an explicit ``{bar index:
    direction}`` map, leaving every other rule untouched.  A random-entry null
    then runs through exactly the same stop, pyramid and exit machinery as the
    real signal, so a difference between them cannot come from the exit.
    """
    rows = sorted(bars, key=lambda bar: bar.timestamp)
    atr = wilder_atr(rows, config.atr_window)
    highs = [bar.high for bar in rows]
    lows = [bar.low for bar in rows]
    entry_highs = rolling_extremes(highs, config.entry_window, True)
    entry_lows = rolling_extremes(lows, config.entry_window, False)
    exit_highs = rolling_extremes(highs, config.exit_window, True)
    exit_lows = rolling_extremes(lows, config.exit_window, False)
    warmup = max(config.entry_window, config.atr_window)
    volumes = (
        relative_volume(rows, config.volume_window)
        if config.min_relative_volume else None
    )
    if volumes is not None:
        warmup = max(warmup, config.volume_window + 1)
    trend = (
        trailing_mean([bar.close for bar in rows], config.trend_window)
        if config.trend_filter == "sma" else None
    )
    if trend is not None:
        warmup = max(warmup, config.trend_window)
    trades: list[TurtleTrade] = []
    skipped = 0
    skipped_small_n = 0
    skipped_against_trend = 0
    skipped_thin_volume = 0
    breakouts = 0
    last_breakout_won: dict[int, bool | None] = {1: None, -1: None}

    direction = 0
    units: list[TurtleUnit] = []
    entry_index = 0
    stop = 0.0
    stop_reason = "stop"
    n_at_entry = 0.0
    favourable_extreme = 0.0
    pending: dict[int, _Phantom | None] = {1: None, -1: None}

    index = warmup
    while index < len(rows):
        bar = rows[index]
        previous_n = atr[index - 1]
        if math.isnan(previous_n) or previous_n <= 0:
            index += 1
            continue

        # Advance any unresolved phantom before this bar can be used for a
        # decision, so the filter only ever reads breakouts that have closed.
        for side, phantom in pending.items():
            if phantom is None:
                continue
            level = exit_lows[index] if side > 0 else exit_highs[index]
            outcome = phantom.step(bar, level)
            if outcome is not None:
                last_breakout_won[side] = outcome
                pending[side] = None

        if direction:
            channel_level = (
                exit_lows[index] if direction > 0 else exit_highs[index]
            ) if config.use_channel_exit else math.nan
            session_end = (
                not config.allow_overnight
                and (index + 1 >= len(rows)
                     or rows[index + 1].timestamp[:10] != bar.timestamp[:10])
            )
            reason = ""
            price = 0.0
            # The stop is checked first: inside one bar the adverse level is
            # assumed to trade before the favourable one.
            if _breached(bar, stop, -direction):
                reason, price = stop_reason, _stop_fill(stop, bar, -direction)
            elif not math.isnan(channel_level) and _breached(
                bar, channel_level, -direction
            ):
                reason = "channel"
                price = _stop_fill(channel_level, bar, -direction)
            elif session_end:
                reason, price = "session close", bar.close

            if reason:
                gross = sum(direction * (price - unit.price) / unit.n for unit in units)
                basis = sum(unit.price / unit.n for unit in units)
                cost = config.round_trip_cost * basis
                net = gross - cost
                average = sum(unit.price for unit in units) / len(units)
                sessions = len({
                    row.timestamp[:10] for row in rows[entry_index:index + 1]
                })
                trades.append(TurtleTrade(
                    entry_timestamp=units[0].timestamp,
                    exit_timestamp=bar.timestamp,
                    direction=direction,
                    units=len(units),
                    entry=units[0].price,
                    average_entry=average,
                    exit=price,
                    n_at_entry=n_at_entry,
                    stop_at_exit=stop,
                    exit_reason=reason,
                    bars_held=index - entry_index,
                    sessions_held=sessions,
                    gross_r=gross,
                    cost_r=cost,
                    net_r=net,
                    equity_return=net * config.risk_fraction,
                    cost_basis_r=basis,
                    unit_entries=tuple(units),
                ))
                direction = 0
                units = []
                stop_reason = "stop"
                index += 1
                continue

            # Pyramid: each additional unit sits add_atr further in favour, and
            # pulls the whole position's stop up behind the newest fill.
            while len(units) < config.max_units:
                target = units[-1].price + direction * config.add_atr * n_at_entry
                if not _breached(bar, target, direction):
                    break
                fill = _stop_fill(target, bar, direction)
                units.append(TurtleUnit(bar.timestamp, fill, n_at_entry))
                stop = fill - direction * config.stop_atr * n_at_entry
                stop_reason = "stop"

            # These levels are calculated only after this bar has completed.
            # They therefore become executable on the next bar, avoiding the
            # common error of using a bar's high to create a stop and then
            # claiming that same bar also filled it.
            favourable_extreme = (
                max(favourable_extreme, bar.high) if direction > 0
                else min(favourable_extreme, bar.low)
            )
            if config.chandelier_atr is not None:
                candidate_stop = (
                    favourable_extreme
                    - direction * config.chandelier_atr * n_at_entry
                )
                tighter = (
                    candidate_stop > stop if direction > 0 else candidate_stop < stop
                )
                if tighter:
                    stop = candidate_stop
                    stop_reason = "chandelier"
            if config.breakeven_trigger_n is not None:
                first_entry = units[0].price
                progress_n = direction * (favourable_extreme - first_entry) / n_at_entry
                average = sum(unit.price for unit in units) / len(units)
                tighter = average > stop if direction > 0 else average < stop
                if progress_n >= config.breakeven_trigger_n and tighter:
                    stop = average
                    stop_reason = "breakeven"
            index += 1
            continue

        if entries is not None:
            candidate = entries.get(index)
            if candidate is None:
                index += 1
                continue
            fill = bar.open
            breakouts += 1
            floor = max(
                config.minimum_n_fraction,
                config.minimum_n_cost_multiple * config.round_trip_cost,
            )
            if previous_n < floor * fill:
                skipped_small_n += 1
                index += 1
                continue
            direction = candidate
            n_at_entry = previous_n
            units = [TurtleUnit(bar.timestamp, fill, previous_n)]
            stop = fill - candidate * config.stop_atr * previous_n
            entry_index = index
            index += 1
            continue

        for candidate in config.directions:
            if config.entry_mode == "close confirm":
                # The signal bar is the one that already closed; entry is here,
                # at this bar's open, so nothing is read before it happens.
                signal = index - 1
                level = entry_highs[signal] if candidate > 0 else entry_lows[signal]
                if math.isnan(level):
                    continue
                closed_beyond = (rows[signal].close > level if candidate > 0
                                 else rows[signal].close < level)
                if not closed_beyond:
                    continue
                fill = bar.open
            else:
                level = entry_highs[index] if candidate > 0 else entry_lows[index]
                if math.isnan(level) or not _breached(bar, level, candidate):
                    continue
                fill = _stop_fill(level, bar, candidate)
            breakouts += 1
            if volumes is not None:
                observed = volumes[index - 1]
                if math.isnan(observed) or observed < config.min_relative_volume:
                    skipped_thin_volume += 1
                    break
            if trend is not None:
                # The filter reads the mean as at the previous close, so it is
                # known before this bar opens.
                reference = trend[index - 1]
                if math.isnan(reference):
                    skipped_against_trend += 1
                    break
                if (rows[index - 1].close > reference) != (candidate > 0):
                    skipped_against_trend += 1
                    break
            # A 1N move must be able to pay for the round trips it takes to
            # capture it. Wilder's N decays geometrically through flat bars, so
            # at fine intervals it can collapse towards zero and make every R
            # multiple derived from it meaningless.
            floor = max(
                config.minimum_n_fraction,
                config.minimum_n_cost_multiple * config.round_trip_cost,
            )
            if previous_n < floor * fill:
                skipped_small_n += 1
                break
            won = last_breakout_won[candidate]
            take = not (config.skip_after_winner and won is True)
            pending[candidate] = _Phantom(
                direction=candidate, entry=fill,
                stop=fill - candidate * config.stop_atr * previous_n,
            )
            last_breakout_won[candidate] = None
            if not take:
                skipped += 1
                break
            direction = candidate
            n_at_entry = previous_n
            units = [TurtleUnit(bar.timestamp, fill, previous_n)]
            stop = fill - candidate * config.stop_atr * previous_n
            stop_reason = "stop"
            favourable_extreme = bar.high if candidate > 0 else bar.low
            entry_index = index
            break
        index += 1

    if direction:
        final = rows[-1]
        gross = sum(direction * (final.close - unit.price) / unit.n for unit in units)
        basis = sum(unit.price / unit.n for unit in units)
        cost = config.round_trip_cost * basis
        trades.append(TurtleTrade(
            entry_timestamp=units[0].timestamp,
            exit_timestamp=final.timestamp,
            direction=direction,
            units=len(units),
            entry=units[0].price,
            average_entry=sum(unit.price for unit in units) / len(units),
            exit=final.close,
            n_at_entry=n_at_entry,
            stop_at_exit=stop,
            exit_reason="end of data",
            bars_held=len(rows) - 1 - entry_index,
            sessions_held=len({row.timestamp[:10] for row in rows[entry_index:]}),
            gross_r=gross,
            cost_r=cost,
            net_r=gross - cost,
            equity_return=(gross - cost) * config.risk_fraction,
            cost_basis_r=basis,
            unit_entries=tuple(units),
        ))

    return trades, TurtleAudit(
        bars=len(rows), breakouts=breakouts, skipped_after_winner=skipped,
        skipped_small_n=skipped_small_n,
        skipped_against_trend=skipped_against_trend,
        skipped_thin_volume=skipped_thin_volume, trades=len(trades),
    )


def summarise_turtle(
    trades: list[TurtleTrade], skipped: int = 0,
) -> TurtleSummary:
    if not trades:
        return TurtleSummary(0, 0, 0, 0, *(math.nan for _ in range(12)), skipped)
    rs = [trade.net_r for trade in trades]
    gains = sum(value for value in rs if value > 0)
    losses = -sum(value for value in rs if value < 0)
    equity = peak = 1.0
    drawdown = 0.0
    for trade in trades:
        equity *= max(0.0, 1.0 + trade.equity_return)
        peak = max(peak, equity)
        drawdown = min(drawdown, equity / peak - 1.0)
    return TurtleSummary(
        trades=len(trades),
        units=sum(trade.units for trade in trades),
        longs=sum(trade.direction > 0 for trade in trades),
        shorts=sum(trade.direction < 0 for trade in trades),
        win_rate=sum(value > 0 for value in rs) / len(rs),
        profit_factor=gains / losses if losses else math.inf,
        mean_r=sum(rs) / len(rs),
        median_r=median(rs),
        total_r=sum(rs),
        largest_win_r=max(rs),
        largest_loss_r=min(rs),
        stop_rate=sum(t.exit_reason == "stop" for t in trades) / len(trades),
        channel_exit_rate=sum(t.exit_reason == "channel" for t in trades) / len(trades),
        mean_bars_held=sum(t.bars_held for t in trades) / len(trades),
        ending_equity=equity,
        max_drawdown=drawdown,
        skipped=skipped,
    )
