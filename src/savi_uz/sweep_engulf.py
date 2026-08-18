"""Leakage-safe backtest of the Pine ``Sweep and Engulf Strategy``.

The source Pine script calculates a signal at a completed bar close, submits a
market order, and therefore fills at the following bar open.  Its stop and
target are nevertheless anchored to the signal close.  That slightly unusual
behaviour is preserved here by default so the result matches the supplied
strategy rather than an idealised rewrite.

Only OHLC bars are available.  When stop and target are both touched inside one
bar their ordering is unknowable, so the stop is charged conservatively.  A
gap through either resting order fills at the observed open.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime
from statistics import median
from zoneinfo import ZoneInfo

from savi_uz.volume_profile import Bar


def resample_regular_session(
    bars: list[Bar], *, minutes: int = 240, timezone: str = "America/New_York"
) -> list[Bar]:
    """Aggregate intraday bars into exchange-session-anchored windows.

    A US 09:30-16:00 session becomes a four-hour 09:30-13:30 bar followed by
    a 2.5-hour 13:30-16:00 bar.  This matches chart aggregation anchored to the
    cash-session open; windows never cross an overnight boundary.
    """
    if minutes < 1:
        raise ValueError("minutes must be positive")
    zone = ZoneInfo(timezone)
    groups: OrderedDict[tuple[str, int], list[Bar]] = OrderedDict()
    for bar in sorted(bars, key=lambda row: row.timestamp):
        stamp = datetime.fromisoformat(bar.timestamp.replace("Z", "+00:00"))
        local = stamp.astimezone(zone)
        since_open = (local.hour * 60 + local.minute) - (9 * 60 + 30)
        if since_open < 0 or since_open >= 390:
            continue
        groups.setdefault((local.date().isoformat(), since_open // minutes), []).append(bar)

    result: list[Bar] = []
    for rows in groups.values():
        volumes = [row.volume for row in rows if row.volume is not None]
        result.append(
            Bar(
                timestamp=rows[0].timestamp,
                open=rows[0].open,
                high=max(row.high for row in rows),
                low=min(row.low for row in rows),
                close=rows[-1].close,
                volume=sum(volumes) if volumes else None,
            )
        )
    return result


@dataclass(frozen=True)
class SweepConfig:
    invert_trades: bool = True
    previous_candle: str = "Any"
    use_ema: bool = False
    ema_length: int = 200
    stop_type: str = "ATR"
    atr_length: int = 14
    stop_atr: float = 1.5
    reward_risk: float = 2.0
    round_trip_cost: float = 0.0002
    anchor_to_signal_close: bool = True

    def __post_init__(self) -> None:
        if self.previous_candle not in {"Any", "Same Direction"}:
            raise ValueError("previous_candle must be 'Any' or 'Same Direction'")
        if self.stop_type not in {"ATR", "Candle High/Low"}:
            raise ValueError("stop_type must be 'ATR' or 'Candle High/Low'")
        if self.ema_length < 1 or self.atr_length < 1:
            raise ValueError("indicator lengths must be positive")
        if self.stop_atr <= 0 or self.reward_risk <= 0:
            raise ValueError("stop_atr and reward_risk must be positive")
        if self.round_trip_cost < 0:
            raise ValueError("round_trip_cost cannot be negative")


@dataclass(frozen=True)
class SweepTrade:
    signal_timestamp: str
    entry_timestamp: str
    exit_timestamp: str
    pattern: str
    direction: int
    signal_close: float
    atr: float
    entry: float
    stop: float
    target: float
    exit: float
    exit_reason: str
    bars_held: int
    held_overnight: bool
    both_touched: bool
    gross_return: float
    net_return: float
    net_r: float
    fixed_share_pnl: float


@dataclass(frozen=True)
class SweepSignal:
    bar_index: int
    timestamp: str
    available_timestamp: str
    pattern: str
    direction: int
    signal_close: float
    atr: float
    base_stop: float
    target: float


@dataclass(frozen=True)
class SweepSummary:
    count: int
    longs: int
    shorts: int
    years: float
    trades_per_year: float
    win_rate: float
    mean_return: float
    median_return: float
    mean_r: float
    profit_factor: float
    target_rate: float
    stop_rate: float
    gap_exit_rate: float
    overnight_rate: float
    both_touched_rate: float
    ending_equity: float
    cagr: float
    max_drawdown: float
    fixed_share_pnl: float


def _true_ranges(bars: list[Bar]) -> list[float]:
    values: list[float] = []
    previous_close: float | None = None
    for bar in bars:
        value = bar.high - bar.low
        if previous_close is not None:
            value = max(value, abs(bar.high - previous_close), abs(bar.low - previous_close))
        values.append(value)
        previous_close = bar.close
    return values


def _wilder(values: list[float], length: int) -> list[float | None]:
    """TradingView-style RMA used by ``ta.atr``."""
    out: list[float | None] = [None] * len(values)
    if len(values) < length:
        return out
    average = sum(values[:length]) / length
    out[length - 1] = average
    for index in range(length, len(values)):
        average = (average * (length - 1) + values[index]) / length
        out[index] = average
    return out


def _ema(values: list[float], length: int) -> list[float]:
    alpha = 2.0 / (length + 1.0)
    result = [values[0]]
    for value in values[1:]:
        result.append(alpha * value + (1.0 - alpha) * result[-1])
    return result


def _directional_return(direction: int, exit_price: float, entry: float) -> float:
    return direction * (exit_price / entry - 1.0)


def _fill_on_bar(
    bar: Bar, direction: int, stop: float, target: float
) -> tuple[float, str, bool] | None:
    """Return an exit fill, charging the stop when intrabar order is unknown."""
    if direction > 0:
        if bar.open <= stop:
            return bar.open, "gap_stop", False
        if bar.open >= target:
            return bar.open, "gap_target", False
        hit_stop, hit_target = bar.low <= stop, bar.high >= target
    else:
        if bar.open >= stop:
            return bar.open, "gap_stop", False
        if bar.open <= target:
            return bar.open, "gap_target", False
        hit_stop, hit_target = bar.high >= stop, bar.low <= target
    if hit_stop:
        return stop, "stop", hit_target
    if hit_target:
        return target, "target", False
    return None


def build_signals(
    bars: list[Bar],
    config: SweepConfig = SweepConfig(),
    *,
    start: str | None = None,
    end: str | None = None,
) -> list[SweepSignal]:
    """Return every raw completed-bar sweep signal, without position filtering.

    ``available_timestamp`` is the next bar's start.  It is the first instant
    at which the completed higher-timeframe bar can safely influence a lower-
    timeframe strategy.
    """
    rows = sorted(bars, key=lambda bar: bar.timestamp)
    if len(rows) < max(config.atr_length, 2) + 1:
        return []
    atrs = _wilder(_true_ranges(rows), config.atr_length)
    emas = _ema([bar.close for bar in rows], config.ema_length)
    signals: list[SweepSignal] = []
    for index in range(1, len(rows) - 1):
        current, previous = rows[index], rows[index - 1]
        day = current.timestamp[:10]
        if (start and day < start) or (end and day > end) or atrs[index] is None:
            continue
        previous_bullish = previous.close > previous.open
        previous_bearish = previous.close < previous.open
        bullish = (
            current.low < previous.low
            and current.close > previous.high
            and (config.previous_candle == "Any" or previous_bullish)
            and (not config.use_ema or current.close > emas[index])
        )
        bearish = (
            current.high > previous.high
            and current.close < previous.low
            and (config.previous_candle == "Any" or previous_bearish)
            and (not config.use_ema or current.close < emas[index])
        )
        if not (bullish or bearish):
            continue
        pattern = "bullish" if bullish else "bearish"
        pattern_direction = 1 if bullish else -1
        direction = -pattern_direction if config.invert_trades else pattern_direction
        atr = float(atrs[index])
        if config.stop_type == "ATR":
            stop = current.close - direction * atr * config.stop_atr
        else:
            stop = current.low if direction > 0 else current.high
        risk = abs(current.close - stop)
        signals.append(
            SweepSignal(
                bar_index=index,
                timestamp=current.timestamp,
                available_timestamp=rows[index + 1].timestamp,
                pattern=pattern,
                direction=direction,
                signal_close=current.close,
                atr=atr,
                base_stop=stop,
                target=current.close + direction * risk * config.reward_risk,
            )
        )
    return signals


def run_strategy(
    bars: list[Bar],
    config: SweepConfig = SweepConfig(),
    *,
    start: str | None = None,
    end: str | None = None,
) -> list[SweepTrade]:
    """Run one position at a time on bars sorted by timestamp."""
    rows = sorted(bars, key=lambda bar: bar.timestamp)
    if len(rows) < max(config.atr_length, 2) + 1:
        return []
    atrs = _wilder(_true_ranges(rows), config.atr_length)
    emas = _ema([bar.close for bar in rows], config.ema_length)
    trades: list[SweepTrade] = []
    index = 1
    while index < len(rows) - 1:
        current, previous = rows[index], rows[index - 1]
        day = current.timestamp[:10]
        if (start and day < start) or (end and day > end) or atrs[index] is None:
            index += 1
            continue

        previous_bullish = previous.close > previous.open
        previous_bearish = previous.close < previous.open
        require_bullish = config.previous_candle == "Any" or previous_bullish
        require_bearish = config.previous_candle == "Any" or previous_bearish
        bullish_sweep = (
            current.low < previous.low
            and current.close > previous.high
            and require_bullish
            and (not config.use_ema or current.close > emas[index])
        )
        bearish_sweep = (
            current.high > previous.high
            and current.close < previous.low
            and require_bearish
            and (not config.use_ema or current.close < emas[index])
        )
        if not (bullish_sweep or bearish_sweep):
            index += 1
            continue

        pattern = "bullish" if bullish_sweep else "bearish"
        pattern_direction = 1 if bullish_sweep else -1
        direction = -pattern_direction if config.invert_trades else pattern_direction
        entry_index = index + 1
        entry_bar = rows[entry_index]
        entry = entry_bar.open
        anchor = current.close if config.anchor_to_signal_close else entry
        atr = float(atrs[index])
        if config.stop_type == "ATR":
            stop = anchor - direction * atr * config.stop_atr
        else:
            stop = current.low if direction > 0 else current.high
        signal_risk = abs(anchor - stop)
        target = anchor + direction * signal_risk * config.reward_risk

        exit_fill: tuple[float, str, bool] | None = None
        exit_index = entry_index
        for candidate in range(entry_index, len(rows)):
            exit_fill = _fill_on_bar(rows[candidate], direction, stop, target)
            if exit_fill is not None:
                exit_index = candidate
                break
        if exit_fill is None:
            exit_index = len(rows) - 1
            exit_fill = (rows[-1].close, "end_of_data", False)

        exit_price, reason, both_touched = exit_fill
        gross = _directional_return(direction, exit_price, entry)
        net = gross - config.round_trip_cost
        risk_from_fill = direction * (entry - stop)
        net_r = net * entry / risk_from_fill if risk_from_fill > 0 else math.nan
        fixed_pnl = direction * (exit_price - entry) - config.round_trip_cost * entry
        trades.append(
            SweepTrade(
                signal_timestamp=current.timestamp,
                entry_timestamp=entry_bar.timestamp,
                exit_timestamp=rows[exit_index].timestamp,
                pattern=pattern,
                direction=direction,
                signal_close=current.close,
                atr=atr,
                entry=entry,
                stop=stop,
                target=target,
                exit=exit_price,
                exit_reason=reason,
                bars_held=exit_index - entry_index + 1,
                held_overnight=entry_bar.timestamp[:10] != rows[exit_index].timestamp[:10],
                both_touched=both_touched,
                gross_return=gross,
                net_return=net,
                net_r=net_r,
                fixed_share_pnl=fixed_pnl,
            )
        )
        # Pine clears its state before evaluating the exit bar's new signal.
        index = max(index + 1, exit_index)
    return trades


def summarise(trades: list[SweepTrade]) -> SweepSummary:
    if not trades:
        return SweepSummary(0, 0, 0, *(math.nan for _ in range(16)))
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
    cagr = equity ** (1.0 / elapsed) - 1.0 if elapsed > 0 and equity > 0 else math.nan
    finite_r = [trade.net_r for trade in trades if math.isfinite(trade.net_r)]
    return SweepSummary(
        count=len(trades),
        longs=sum(trade.direction > 0 for trade in trades),
        shorts=sum(trade.direction < 0 for trade in trades),
        years=elapsed,
        trades_per_year=len(trades) / elapsed if elapsed > 0 else math.nan,
        win_rate=sum(value > 0 for value in returns) / len(returns),
        mean_return=sum(returns) / len(returns),
        median_return=median(returns),
        mean_r=sum(finite_r) / len(finite_r) if finite_r else math.nan,
        profit_factor=gains / losses if losses else math.inf,
        target_rate=sum("target" in trade.exit_reason for trade in trades) / len(trades),
        stop_rate=sum("stop" in trade.exit_reason for trade in trades) / len(trades),
        gap_exit_rate=sum(trade.exit_reason.startswith("gap_") for trade in trades) / len(trades),
        overnight_rate=sum(trade.held_overnight for trade in trades) / len(trades),
        both_touched_rate=sum(trade.both_touched for trade in trades) / len(trades),
        ending_equity=equity,
        cagr=cagr,
        max_drawdown=drawdown,
        fixed_share_pnl=sum(trade.fixed_share_pnl for trade in trades),
    )
