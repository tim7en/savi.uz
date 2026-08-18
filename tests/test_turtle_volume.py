from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from savi_uz.turtle_volume import (
    BreakoutVolume,
    ConfirmedTurtleConfig,
    VolumeFilter,
    build_breakout_volumes,
    build_intraday_breakout_volumes,
    run_volume_turtle,
)
from savi_uz.volume_profile import Bar


def make_bar(stamp, open_, high, low, close, volume=100.0):
    return Bar(stamp.isoformat(), open_, high, low, close, volume)


class VolumeFeatureTimingTests(unittest.TestCase):
    def test_future_session_cannot_change_current_profile_or_rvol(self):
        first = datetime(2024, 1, 1, 14, 30, tzinfo=timezone.utc)
        rows = [
            make_bar(first + timedelta(days=i), 100 + i, 101 + i, 99 + i, 100 + i,
                     200 if i == 20 else 100)
            for i in range(22)
        ]
        day = rows[20].timestamp[:10]
        original = build_breakout_volumes(rows)[day]
        changed = rows[:21] + [
            make_bar(first + timedelta(days=21), 1, 1000, 0, 500, 100000)
        ]
        self.assertEqual(original, build_breakout_volumes(changed)[day])
        self.assertEqual(original.volume_ratio, 2.0)
        self.assertLess(original.profile_high, rows[20].high)

    def test_intraday_rvol_uses_the_same_slot_and_ignores_future_sessions(self):
        first = datetime(2024, 1, 1, 14, 30, tzinfo=timezone.utc)
        rows = []
        for day in range(22):
            stamp = first + timedelta(days=day)
            rows.extend([
                make_bar(stamp, 100, 101, 99, 100, 200 if day == 20 else 100),
                make_bar(stamp + timedelta(minutes=5), 100, 101, 99, 100, 300 if day == 20 else 100),
            ])
        stamp = rows[40].timestamp
        original = build_intraday_breakout_volumes(rows, expected_bars=2)[stamp]
        changed = rows[:42] + [
            make_bar(first + timedelta(days=21), 1, 1000, 0, 500, 100000),
            make_bar(first + timedelta(days=21, minutes=5), 1, 1000, 0, 500, 100000),
        ]
        self.assertEqual(
            original,
            build_intraday_breakout_volumes(changed, expected_bars=2)[stamp],
        )
        self.assertEqual(original.volume_ratio, 2.0)


class ConfirmedEntryTests(unittest.TestCase):
    def fixture(self, *, signal_close=11.0):
        first = datetime(2024, 1, 1, 14, 30, tzinfo=timezone.utc)
        rows = [make_bar(first + timedelta(days=i), 9.5, 10, 9, 9.5) for i in range(55)]
        rows.extend([
            make_bar(first + timedelta(days=55), 9.7, 11.2, 9.5, signal_close, 200),
            make_bar(first + timedelta(days=56), 11.2, 12, 10.8, 11.8, 100),
            make_bar(first + timedelta(days=57), 11.8, 12, 8, 8.5, 100),
        ])
        feature = BreakoutVolume(
            session=rows[55].timestamp[:10], volume=200, typical_volume=100,
            volume_ratio=2.0, rising_volume=True, poc=9.5,
            value_low=9.2, value_high=9.8, profile_low=9, profile_high=10,
        )
        return rows, {feature.session: feature}

    def test_confirmed_breakout_enters_only_at_the_next_open(self):
        rows, features = self.fixture()
        trades, _ = run_volume_turtle(
            "TEST", rows, features,
            config=ConfirmedTurtleConfig(max_units=1, directions=(1,)),
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].signal_timestamp, rows[55].timestamp)
        self.assertEqual(trades[0].trade.entry_timestamp, rows[56].timestamp)
        self.assertEqual(trades[0].trade.entry, rows[56].open)
        self.assertEqual(trades[0].volume_ratio, 2.0)

    def test_full_bar_volume_filter_cannot_fill_on_the_signal_bar(self):
        rows, features = self.fixture()
        trades, _ = run_volume_turtle(
            "TEST", rows, features,
            rule=VolumeFilter(volume_floor=1.5),
            config=ConfirmedTurtleConfig(max_units=1, directions=(1,)),
        )
        self.assertTrue(trades)
        self.assertGreater(trades[0].trade.entry_timestamp, trades[0].signal_timestamp)

    def test_an_intrabar_high_without_a_close_break_is_not_confirmed(self):
        rows, features = self.fixture(signal_close=9.8)
        rows[56] = make_bar(
            datetime.fromisoformat(rows[56].timestamp), 9.8, 10, 9.4, 9.8
        )
        rows[57] = make_bar(
            datetime.fromisoformat(rows[57].timestamp), 9.8, 10, 9.4, 9.8
        )
        trades, audit = run_volume_turtle(
            "TEST", rows, features,
            config=ConfirmedTurtleConfig(max_units=1, directions=(1,)),
        )
        self.assertEqual(trades, [])
        self.assertEqual(audit.confirmed_breakouts, 0)

    def test_low_relative_volume_is_rejected(self):
        rows, features = self.fixture()
        feature = features[rows[55].timestamp[:10]]
        features[feature.session] = BreakoutVolume(
            **{**feature.__dict__, "volume_ratio": 0.8}
        )
        trades, audit = run_volume_turtle(
            "TEST", rows, features,
            rule=VolumeFilter(volume_floor=1.0),
            config=ConfirmedTurtleConfig(max_units=1, directions=(1,)),
        )
        self.assertEqual(trades, [])
        self.assertGreaterEqual(audit.rejected_by_filter, 1)


if __name__ == "__main__":
    unittest.main()
