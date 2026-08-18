from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from savi_uz.swing_failure_strategy import (
    SfpConfig,
    build_daily_biases,
    run_sfp_strategy,
    summarise_sfp,
)
from savi_uz.volume_profile import Bar


def bar(stamp: datetime, open_: float, high: float, low: float, close: float) -> Bar:
    return Bar(stamp.isoformat(), open_, high, low, close, 100.0)


class DailyBiasTests(unittest.TestCase):
    def test_current_and_future_daily_bars_cannot_change_current_bias(self):
        first = datetime(2024, 1, 1, 14, 30, tzinfo=timezone.utc)
        rows = [
            bar(first, 7, 10, 5, 8),
            bar(first + timedelta(days=1), 8, 11, 6, 9),
            bar(first + timedelta(days=2), 9, 12, 7, 10),
            bar(first + timedelta(days=3), 1, 100, 0, 50),
            bar(first + timedelta(days=4), 1, 200, -10, 100),
        ]
        original = build_daily_biases(rows, SfpConfig())["2024-01-04"]
        changed = rows[:3] + [
            bar(first + timedelta(days=3), 100, 1000, -100, 500),
            bar(first + timedelta(days=4), 100, 2000, -200, 600),
        ]
        self.assertEqual(
            original,
            build_daily_biases(changed, SfpConfig())["2024-01-04"],
        )
        self.assertEqual(original.direction, 1)
        self.assertLess(original.source_last, rows[3].timestamp)


class StrongSetupTests(unittest.TestCase):
    def fixture(self, *, touch_level_early: bool = False):
        first = datetime(2024, 1, 1, 14, 30, tzinfo=timezone.utc)
        daily = [
            bar(first, 7, 10, 5, 8),
            bar(first + timedelta(days=1), 8, 11, 6, 9),
            bar(first + timedelta(days=2), 9, 12, 7, 10),
            bar(first + timedelta(days=3), 8, 12, 6.8, 10),
        ]
        day3 = first + timedelta(days=2)
        day4 = first + timedelta(days=3)
        fifteen = [bar(day3, 9, 12, 7, 10)]
        for offset in range(4):
            low = 6.9 if touch_level_early and offset == 0 else 7.2
            fifteen.append(bar(day4 + timedelta(minutes=15 * offset), 7.6, 8, low, 7.8))
        fifteen.extend([
            bar(day4 + timedelta(minutes=60), 7.8, 8.1, 6.8, 6.9),
            bar(day4 + timedelta(minutes=75), 6.9, 7.7, 6.9, 7.5),
            bar(day4 + timedelta(minutes=90), 7.5, 8.3, 7.4, 8.0),
            bar(day4 + timedelta(minutes=105), 8.0, 8.5, 7.9, 8.2),
            bar(day4 + timedelta(minutes=120), 8.2, 12.1, 8.1, 12),
        ])
        hourly = [
            bar(day4, 7.6, 8, 7.2 if not touch_level_early else 6.9, 7.8),
            bar(day4 + timedelta(minutes=60), 7.8, 8.5, 6.8, 8.2),
            bar(day4 + timedelta(minutes=120), 8.2, 12.1, 8.1, 12),
        ]
        return daily, hourly, fifteen

    def test_strong_sfp_enters_at_next_fifteen_minute_open(self):
        daily, hourly, fifteen = self.fixture()
        trades, _ = run_sfp_strategy(daily, hourly, fifteen)
        self.assertEqual(len(trades), 1)
        trade = trades[0]
        self.assertEqual(trade.location_kind, "PDL")
        self.assertEqual(trade.location, 7)
        self.assertEqual(trade.entry_timestamp, hourly[2].timestamp)
        self.assertEqual(trade.stop, hourly[1].low)
        self.assertEqual(trade.target, 12)
        self.assertEqual(trade.exit_reason, "target")

    def test_an_earlier_touch_means_the_liquidity_is_no_longer_resting(self):
        daily, hourly, fifteen = self.fixture(touch_level_early=True)
        trades, _ = run_sfp_strategy(daily, hourly, fifteen)
        self.assertEqual(trades, [])

    def test_empty_summary_is_valid(self):
        self.assertEqual(summarise_sfp([]).count, 0)


if __name__ == "__main__":
    unittest.main()
