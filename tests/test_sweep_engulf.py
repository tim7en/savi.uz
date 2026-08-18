from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from savi_uz.sweep_engulf import (
    SweepConfig,
    resample_regular_session,
    run_strategy,
    summarise,
)
from savi_uz.volume_profile import Bar


def bar(index: int, open_: float, high: float, low: float, close: float) -> Bar:
    return Bar(f"2024-01-02T{14 + index // 12:02d}:{(30 + index * 5) % 60:02d}:00Z",
               open_, high, low, close, 100.0)


class SweepEngulfTests(unittest.TestCase):
    def test_four_hour_resampling_is_anchored_to_each_session(self):
        first = datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)
        rows = [
            Bar((first + timedelta(minutes=5 * index)).isoformat(),
                100.0 + index, 101.0 + index, 99.0 + index, 100.5 + index, 10.0)
            for index in range(78)
        ]
        result = resample_regular_session(rows)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].open, rows[0].open)
        self.assertEqual(result[0].close, rows[47].close)
        self.assertEqual(result[1].open, rows[48].open)
        self.assertEqual(result[1].close, rows[77].close)
        self.assertEqual(result[0].volume, 480.0)

    def test_empty_summary_is_well_formed(self):
        self.assertEqual(summarise([]).count, 0)

    def bullish_setup(self) -> list[Bar]:
        return [
            bar(0, 100.0, 101.0, 99.0, 100.0),
            bar(1, 100.0, 102.0, 98.0, 101.5),
            bar(2, 102.0, 106.0, 101.0, 105.0),
            bar(3, 105.0, 105.2, 104.8, 105.0),
        ]

    def test_signal_fills_at_next_open_but_levels_use_signal_close(self):
        config = SweepConfig(invert_trades=False, atr_length=1, stop_atr=1.0,
                             reward_risk=1.0, round_trip_cost=0.0)
        trade = run_strategy(self.bullish_setup(), config)[0]
        self.assertEqual(trade.entry, 102.0)
        self.assertEqual(trade.stop, 97.5)
        self.assertEqual(trade.target, 105.5)
        self.assertEqual(trade.exit_reason, "target")

    def test_default_inversion_turns_bullish_pattern_into_short(self):
        rows = self.bullish_setup()
        rows[2] = bar(2, 101.0, 101.2, 95.0, 96.0)
        config = SweepConfig(atr_length=1, stop_atr=1.0, reward_risk=1.0,
                             round_trip_cost=0.0)
        trade = run_strategy(rows, config)[0]
        self.assertEqual(trade.pattern, "bullish")
        self.assertEqual(trade.direction, -1)
        self.assertEqual(trade.exit_reason, "target")

    def test_same_direction_filter_can_reject_pattern(self):
        rows = self.bullish_setup()
        rows[0] = bar(0, 101.0, 101.0, 99.0, 100.0)
        config = SweepConfig(invert_trades=False, previous_candle="Same Direction",
                             atr_length=1)
        self.assertEqual(run_strategy(rows, config), [])

    def test_both_touched_is_scored_as_stop(self):
        rows = self.bullish_setup()
        rows[2] = bar(2, 101.5, 106.0, 97.0, 102.0)
        config = SweepConfig(invert_trades=False, atr_length=1, stop_atr=1.0,
                             reward_risk=1.0, round_trip_cost=0.0)
        trade = run_strategy(rows, config)[0]
        self.assertEqual(trade.exit_reason, "stop")
        self.assertTrue(trade.both_touched)

    def test_gap_through_stop_fills_at_open(self):
        rows = self.bullish_setup()
        rows[2] = bar(2, 95.0, 96.0, 94.0, 95.0)
        config = SweepConfig(invert_trades=False, atr_length=1, stop_atr=1.0,
                             reward_risk=1.0, round_trip_cost=0.0)
        trade = run_strategy(rows, config)[0]
        self.assertEqual(trade.exit_reason, "gap_stop")
        self.assertEqual(trade.exit, 95.0)


if __name__ == "__main__":
    unittest.main()
