"""Leakage-safe multi-session Donchian breakout event study.

The channel for session ``t`` is formed exclusively from completed sessions
``t-n .. t-1``.  A signal is known only after a five-minute bar closes outside
that channel, and execution is placed at the next bar's open.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import median

from savi_uz.volume_profile import Bar


@dataclass(frozen=True)
class DonchianEvent:
    session: str
    timestamp: str
    window: int
    volume_floor: float
    direction: int
    signal_bar: int
    channel_high: float
    channel_low: float
    breakout_close: float
    entry: float
    atr: float
    volume_ratio: float

    accepted_30m: bool | None
    accepted_60m: bool | None
    accepted_close: bool
    reentered_30m: bool
    whipsaw_30m: bool
    sustainable: bool

    mfe_r: float
    mae_r: float
    fixed_r: float
    target_before_stop: bool


@dataclass(frozen=True)
class DonchianSummary:
    count: int
    mean_r: float
    profit_factor: float
    target_rate: float
    sustainable_rate: float
    accepted_30m_rate: float
    accepted_60m_rate: float
    accepted_close_rate: float
    reentry_30m_rate: float
    whipsaw_30m_rate: float
    median_mfe_r: float
    median_mae_r: float


def group_sessions(bars: list[Bar]) -> dict[str, list[Bar]]:
    sessions: dict[str, list[Bar]] = defaultdict(list)
    for bar in sorted(bars, key=lambda row: row.timestamp):
        sessions[bar.timestamp[:10]].append(bar)
    return dict(sessions)


def clean_sessions(
    bars: list[Bar], start: str = "2019-01-01", expected_bars: int | None = 78
) -> list[tuple[str, list[Bar]]]:
    """Keep complete regular sessions whose every bar has real volume."""
    out = []
    for session, rows in sorted(group_sessions(bars).items()):
        if session < start:
            continue
        if expected_bars is not None and len(rows) != expected_bars:
            continue
        if not rows or any(row.volume is None or row.volume <= 0 for row in rows):
            continue
        out.append((session, rows))
    return out


def _accepted(close: float, direction: int, high: float, low: float) -> bool:
    return close > high if direction > 0 else close < low


def _atr_map(
    sessions: list[tuple[str, list[Bar]]], lookback: int
) -> dict[tuple[str, int], float]:
    values: list[float] = []
    out: dict[tuple[str, int], float] = {}
    for session, bars in sessions:
        prior_close = None
        for position, bar in enumerate(bars):
            true_range = bar.high - bar.low
            if prior_close is not None:
                true_range = max(
                    true_range, abs(bar.high - prior_close), abs(bar.low - prior_close)
                )
            values.append(true_range)
            window = values[-lookback:]
            out[(session, position)] = sum(window) / len(window)
            prior_close = bar.close
    return out


def build_events(
    bars: list[Bar],
    window: int,
    volume_floor: float,
    *,
    start: str = "2019-01-01",
    expected_bars: int | None = 78,
    volume_lookback: int = 20,
    atr_lookback: int = 20,
    stop_atr: float = 2.0,
    target_r: float = 2.0,
    max_signal_bar: int | None = None,
) -> list[DonchianEvent]:
    """Build the first volume-confirmed channel break in each eligible session.

    Volume is compared with the median of the same bar position over preceding
    clean sessions.  A rejected low-volume crossing does not suppress a later
    qualifying crossing in the same session.
    """
    if window < 1 or volume_lookback < 1 or atr_lookback < 1:
        raise ValueError("lookbacks must be positive")
    sessions = clean_sessions(bars, start=start, expected_bars=expected_bars)
    atrs = _atr_map(sessions, atr_lookback)
    events: list[DonchianEvent] = []

    first_session = max(window, volume_lookback)
    for index in range(first_session, len(sessions)):
        session, current = sessions[index]
        history = [bar for _, rows in sessions[index - window:index] for bar in rows]
        channel_high = max(bar.high for bar in history)
        channel_low = min(bar.low for bar in history)
        prior_close = sessions[index - 1][1][-1].close
        last_signal = len(current) - 2
        if max_signal_bar is not None:
            last_signal = min(last_signal, max_signal_bar)

        for position in range(last_signal + 1):
            bar = current[position]
            before = prior_close if position == 0 else current[position - 1].close
            long_break = before <= channel_high and bar.close > channel_high
            short_break = before >= channel_low and bar.close < channel_low
            if not (long_break or short_break):
                continue

            comparison = [
                sessions[past][1][position].volume
                for past in range(index - volume_lookback, index)
            ]
            typical = median(comparison)
            ratio = bar.volume / typical if typical > 0 else math.nan
            if not math.isfinite(ratio) or ratio < volume_floor:
                continue

            direction = 1 if long_break else -1
            atr = atrs[(session, position)]
            entry = current[position + 1].open
            risk = stop_atr * atr
            stop = entry - direction * risk
            target = entry + direction * target_r * risk
            future = current[position + 1:]

            mfe_r = mae_r = 0.0
            fixed_r: float | None = None
            target_before_stop = False
            whipsaw = False
            for offset, nxt in enumerate(future, start=1):
                favorable = (
                    (nxt.high - entry) / risk
                    if direction > 0 else (entry - nxt.low) / risk
                )
                adverse = (
                    (entry - nxt.low) / risk
                    if direction > 0 else (nxt.high - entry) / risk
                )
                mfe_r = max(mfe_r, favorable)
                mae_r = max(mae_r, adverse)
                hit_stop = nxt.low <= stop if direction > 0 else nxt.high >= stop
                hit_target = nxt.high >= target if direction > 0 else nxt.low <= target
                if fixed_r is None and (hit_stop or hit_target):
                    # With OHLC bars the order is unknowable; charge the stop
                    # whenever both levels occur in one bar.
                    fixed_r = -1.0 if hit_stop else target_r
                    target_before_stop = hit_target and not hit_stop
                if offset <= 6 and hit_stop:
                    whipsaw = True

            if fixed_r is None:
                fixed_r = direction * (future[-1].close - entry) / risk

            next_30 = future[:6]
            next_60 = future[:12]
            accepted_30 = (
                _accepted(next_30[-1].close, direction, channel_high, channel_low)
                if len(next_30) == 6 else None
            )
            accepted_60 = (
                _accepted(next_60[-1].close, direction, channel_high, channel_low)
                if len(next_60) == 12 else None
            )
            reentered = any(
                not _accepted(row.close, direction, channel_high, channel_low)
                for row in next_30
            )
            sustainable = accepted_60 is True and not reentered

            events.append(
                DonchianEvent(
                    session=session,
                    timestamp=bar.timestamp,
                    window=window,
                    volume_floor=volume_floor,
                    direction=direction,
                    signal_bar=position,
                    channel_high=channel_high,
                    channel_low=channel_low,
                    breakout_close=bar.close,
                    entry=entry,
                    atr=atr,
                    volume_ratio=ratio,
                    accepted_30m=accepted_30,
                    accepted_60m=accepted_60,
                    accepted_close=_accepted(
                        current[-1].close, direction, channel_high, channel_low
                    ),
                    reentered_30m=reentered,
                    whipsaw_30m=whipsaw,
                    sustainable=sustainable,
                    mfe_r=mfe_r,
                    mae_r=mae_r,
                    fixed_r=fixed_r,
                    target_before_stop=target_before_stop,
                )
            )
            break
    return events


def summarise(events: list[DonchianEvent]) -> DonchianSummary:
    if not events:
        return DonchianSummary(0, *(math.nan for _ in range(11)))
    returns = [event.fixed_r for event in events]
    gains = sum(value for value in returns if value > 0)
    losses = -sum(value for value in returns if value < 0)

    def rate(name: str) -> float:
        values = [getattr(event, name) for event in events]
        known = [value for value in values if value is not None]
        return sum(known) / len(known) if known else math.nan

    return DonchianSummary(
        count=len(events),
        mean_r=sum(returns) / len(returns),
        profit_factor=gains / losses if losses else math.inf,
        target_rate=rate("target_before_stop"),
        sustainable_rate=rate("sustainable"),
        accepted_30m_rate=rate("accepted_30m"),
        accepted_60m_rate=rate("accepted_60m"),
        accepted_close_rate=rate("accepted_close"),
        reentry_30m_rate=rate("reentered_30m"),
        whipsaw_30m_rate=rate("whipsaw_30m"),
        median_mfe_r=median(event.mfe_r for event in events),
        median_mae_r=median(event.mae_r for event in events),
    )

