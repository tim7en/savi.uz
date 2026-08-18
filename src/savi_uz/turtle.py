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
    skip_after_winner: bool = True
    round_trip_cost: float = 0.0002
    allow_overnight: bool = True
    directions: tuple[int, ...] = (1, -1)

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
        if not self.directions or set(self.directions) - {1, -1}:
            raise ValueError("directions must be a non-empty subset of (1, -1)")


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
    skipped_by_filter: bool = False


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
) -> tuple[list[TurtleTrade], int]:
    """Replay the system over ``bars``; returns the trades and the skip count."""
    rows = sorted(bars, key=lambda bar: bar.timestamp)
    atr = wilder_atr(rows, config.atr_window)
    highs = [bar.high for bar in rows]
    lows = [bar.low for bar in rows]
    entry_highs = rolling_extremes(highs, config.entry_window, True)
    entry_lows = rolling_extremes(lows, config.entry_window, False)
    exit_highs = rolling_extremes(highs, config.exit_window, True)
    exit_lows = rolling_extremes(lows, config.exit_window, False)
    warmup = max(config.entry_window, config.atr_window)
    trades: list[TurtleTrade] = []
    skipped = 0
    last_breakout_won: dict[int, bool | None] = {1: None, -1: None}

    direction = 0
    units: list[TurtleUnit] = []
    entry_index = 0
    stop = 0.0
    n_at_entry = 0.0
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
            channel_level = exit_lows[index] if direction > 0 else exit_highs[index]
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
                reason, price = "stop", _stop_fill(stop, bar, -direction)
            elif not math.isnan(channel_level) and _breached(
                bar, channel_level, -direction
            ):
                reason = "channel"
                price = _stop_fill(channel_level, bar, -direction)
            elif session_end:
                reason, price = "session close", bar.close

            if reason:
                gross = sum(direction * (price - unit.price) / unit.n for unit in units)
                cost = sum(
                    config.round_trip_cost * unit.price / unit.n for unit in units
                )
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
                ))
                direction = 0
                units = []
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
            index += 1
            continue

        for candidate in config.directions:
            level = entry_highs[index] if candidate > 0 else entry_lows[index]
            if math.isnan(level) or not _breached(bar, level, candidate):
                continue
            fill = _stop_fill(level, bar, candidate)
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
            entry_index = index
            break
        index += 1

    if direction:
        final = rows[-1]
        gross = sum(direction * (final.close - unit.price) / unit.n for unit in units)
        cost = sum(config.round_trip_cost * unit.price / unit.n for unit in units)
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
        ))

    return trades, skipped


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
