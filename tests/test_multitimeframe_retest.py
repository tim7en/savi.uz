from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from savi_uz.multitimeframe_retest import (
    LiquidityLevel,
    RetestConfig,
    _is_resting,
    _swept_and_reclaimed,
    confirmed_pivots,
    prior_liquidity_levels,
    run_retest_strategy,
    summarise_retests,
)
from savi_uz.sweep_engulf import SweepConfig, build_signals
from savi_uz.volume_profile import Bar


def at(stamp: datetime, open_: float, high: float, low: float, close: float) -> Bar:
    return Bar(stamp.isoformat(), open_, high, low, close, 100.0)


class SignalTimingTests(unittest.TestCase):
    def test_htf_signal_is_available_only_at_next_bar(self):
        first = datetime(2024, 1, 1, 14, 30, tzinfo=timezone.utc)
        bars = [at(first + timedelta(days=i), 100, 101, 99, 100) for i in range(16)]
        bars[14] = at(first + timedelta(days=14), 100, 102, 97, 98)
        signals = build_signals(bars, SweepConfig(atr_length=14))
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].available_timestamp, bars[15].timestamp)
        self.assertGreater(signals[0].available_timestamp, signals[0].timestamp)

    def test_confirmed_pivots_ignore_every_bar_after_decision_time(self):
        first = datetime(2024, 1, 1, 14, 30, tzinfo=timezone.utc)
        lows = [10, 9, 8, 9, 10, 9, 7, 9, 10, 8, 9, 10]
        bars = [at(first + timedelta(minutes=15 * i), 10, 11, low, 10) for i, low in enumerate(lows)]
        original = confirmed_pivots(bars, end_index=9, span=1, lookback=20, direction=1)
        changed = bars[:10] + [at(first + timedelta(minutes=15 * i), 1, 100, 0, 50) for i in range(10, 12)]
        self.assertEqual(
            original,
            confirmed_pivots(changed, end_index=9, span=1, lookback=20, direction=1),
        )

    def test_rejection_entry_occurs_after_closed_observation_hour(self):
        htf_first = datetime(2024, 1, 1, 14, 30, tzinfo=timezone.utc)
        htf = [at(htf_first + timedelta(days=i), 100, 101, 99, 100) for i in range(16)]
        htf[14] = at(htf_first + timedelta(days=14), 100, 102, 97, 98)
        available = datetime.fromisoformat(htf[15].timestamp)
        history_first = available - timedelta(days=1)
        lower = [
            at(history_first + timedelta(minutes=15 * i), 98, 98.4, 97.8, 98.2)
            for i in range(14)
        ]
        observation = [
            at(available + timedelta(minutes=15 * i), 98.1 + 0.1 * i,
               98.5 + 0.1 * i, 98.0 + 0.1 * i, 98.3 + 0.1 * i)
            for i in range(4)
        ]
        rejection = at(available + timedelta(minutes=60), 98.55, 98.8, 98.25, 98.75)
        entry = at(available + timedelta(minutes=75), 98.75, 106, 98.7, 105)
        final = at(available + timedelta(minutes=90), 105, 105.2, 104.8, 105)
        lower += observation + [rejection, entry, final]
        trades, _ = run_retest_strategy(
            htf, lower,
            htf_config=SweepConfig(atr_length=14),
            config=RetestConfig(minimum_reward_risk=1.0),
        )
        self.assertEqual(len(trades), 1)
        self.assertGreater(trades[0].rejection_timestamp, trades[0].observation_end)
        self.assertGreater(trades[0].entry_timestamp, trades[0].rejection_timestamp)

    def test_empty_summary_is_valid(self):
        self.assertEqual(summarise_retests([]).count, 0)


class LiquidityTimingTests(unittest.TestCase):
    def test_previous_day_levels_exclude_every_current_day_bar(self):
        first = datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)
        bars = [
            at(first, 100, 101, 99, 100),
            at(first + timedelta(minutes=15), 100, 103, 98, 102),
            at(first + timedelta(days=1), 102, 104, 101, 103),
            at(first + timedelta(days=1, minutes=15), 103, 200, 1, 150),
        ]
        levels = {level.kind: level for level in prior_liquidity_levels(bars)["2024-01-03"]}
        self.assertEqual(levels["PDH"].price, 103)
        self.assertEqual(levels["PDL"].price, 98)

    def test_previous_week_level_only_starts_resting_in_the_new_week(self):
        friday = datetime(2024, 1, 5, 14, 30, tzinfo=timezone.utc)
        monday = datetime(2024, 1, 8, 14, 30, tzinfo=timezone.utc)
        bars = [at(friday, 100, 105, 95, 101), at(monday, 101, 104, 96, 102)]
        levels = {level.kind: level for level in prior_liquidity_levels(bars)["2024-01-08"]}
        self.assertEqual(levels["PWH"].price, 105)
        self.assertEqual(levels["PWL"].price, 95)
        self.assertEqual(levels["PWH"].resting_from, 1)
        self.assertTrue(_is_resting(levels["PWH"], bars, 0))

    def test_liquidity_sweep_requires_level_to_be_untouched_before_rejection(self):
        first = datetime(2024, 1, 8, 14, 30, tzinfo=timezone.utc)
        bars = [
            at(first, 101, 102, 99.5, 101),
            at(first + timedelta(minutes=15), 101, 102, 99, 101.5),
        ]
        level = LiquidityLevel("PDL", "low", 100, 0)
        self.assertIsNone(_swept_and_reclaimed((level,), bars, 1, 1))
        fresh = LiquidityLevel("PDL", "low", 100, 1)
        self.assertEqual(_swept_and_reclaimed((fresh,), bars, 1, 1), fresh)


if __name__ == "__main__":
    unittest.main()
