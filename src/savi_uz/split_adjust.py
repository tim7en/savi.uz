"""Back-adjust intraday bars for stock splits.

The intraday bar table stores prices as they printed.  A 20:1 split therefore
appears as a 95% overnight collapse, which any breakout, gap or volatility model
reads as a real move: it corrupts ATR, invents enormous losses for longs and
equally fictitious profits for shorts.

The split factors are recorded alongside the bars, so the fix is the standard
back adjustment: every bar *before* a split is divided by the cumulative factor
that applies to it, and its volume multiplied by the same number.  Bars on and
after the split date already print in post-split terms and are left alone.

Adjusting backwards rather than forwards keeps the most recent prices equal to
the prices actually quoted today, which is what position sizing and cost models
in this project assume.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from savi_uz.volume_profile import Bar


def load_splits(path: Path) -> dict[str, list[tuple[str, float]]]:
    """Split dates and factors per ticker, oldest first."""
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT ticker, obs_date, split_factor FROM corporate_actions "
            "WHERE split_factor IS NOT NULL AND split_factor != 1.0 "
            "ORDER BY ticker, obs_date"
        ).fetchall()
    except sqlite3.OperationalError:  # table absent in older databases
        return {}
    finally:
        connection.close()
    result: dict[str, list[tuple[str, float]]] = {}
    for ticker, obs_date, factor in rows:
        result.setdefault(ticker, []).append((obs_date[:10], float(factor)))
    return result


def cumulative_factors(splits: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """For each split date, the factor applying to every bar before it.

    Walking from the newest split backwards, a bar older than two 2:1 splits has
    to be divided by four, not two.
    """
    result: list[tuple[str, float]] = []
    running = 1.0
    for obs_date, factor in reversed(splits):
        running *= factor
        result.append((obs_date, running))
    result.reverse()
    return result


def adjust_bars(bars: list[Bar], splits: list[tuple[str, float]]) -> list[Bar]:
    """Return ``bars`` with pre-split prices restated in post-split terms."""
    if not splits:
        return bars
    schedule = cumulative_factors(sorted(splits))
    adjusted: list[Bar] = []
    for bar in bars:
        session = bar.timestamp[:10]
        factor = next(
            (value for obs_date, value in schedule if session < obs_date), 1.0
        )
        if factor == 1.0:
            adjusted.append(bar)
            continue
        adjusted.append(Bar(
            timestamp=bar.timestamp,
            open=bar.open / factor,
            high=bar.high / factor,
            low=bar.low / factor,
            close=bar.close / factor,
            volume=None if bar.volume is None else bar.volume * factor,
        ))
    return adjusted


def largest_overnight_gap(bars: list[Bar]) -> tuple[str, float]:
    """Biggest session-to-session open-versus-close jump, for sanity checks."""
    worst = ("", 0.0)
    for previous, current in zip(bars, bars[1:]):
        if previous.close <= 0:
            continue
        gap = current.open / previous.close - 1.0
        if abs(gap) > abs(worst[1]):
            worst = (current.timestamp[:10], gap)
    return worst
