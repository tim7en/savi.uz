from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from savi_uz.swing_failure_strategy import (
    SfpConfig,
    build_daily_biases,
    daily_state,
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
            # The early touch must not reclaim, or the opening hour is itself a
            # valid failure candle rather than a pool that has been consumed.
            close = 6.95 if touch_level_early and offset == 3 else 7.8
            fifteen.append(bar(day4 + timedelta(minutes=15 * offset), 7.6, 8, low, close))
        fifteen.extend([
            bar(day4 + timedelta(minutes=60), 7.8, 8.1, 6.8, 6.9),
            bar(day4 + timedelta(minutes=75), 6.9, 7.7, 6.9, 7.5),
            bar(day4 + timedelta(minutes=90), 7.5, 8.3, 7.4, 8.0),
            bar(day4 + timedelta(minutes=105), 8.0, 8.5, 7.9, 8.2),
            bar(day4 + timedelta(minutes=120), 8.2, 12.1, 8.1, 12),
        ])
        hourly = [
            bar(day4, 7.6, 8, 7.2 if not touch_level_early else 6.9,
                6.95 if touch_level_early else 7.8),
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


class DailyStateTests(unittest.TestCase):
    """The two-candle cheat sheet this study is specified from."""

    def state(self, previous, last):
        first = datetime(2024, 1, 1, 14, 30, tzinfo=timezone.utc)
        return daily_state(
            bar(first, *previous), bar(first + timedelta(days=1), *last)
        )

    def test_close_beyond_the_prior_high_is_a_strong_bull(self):
        self.assertEqual(self.state((10, 12, 8, 11), (11, 14, 9, 13)), "strong bull")

    def test_higher_structure_without_that_close_is_only_a_weak_bull(self):
        self.assertEqual(self.state((10, 12, 8, 11), (11, 12.5, 9, 11.5)), "weak bull")

    def test_higher_structure_closing_bearish_is_caution_not_a_lean(self):
        self.assertEqual(self.state((10, 12, 8, 11), (12, 13, 9, 11.5)), "caution")

    def test_close_below_the_prior_low_is_a_strong_bear(self):
        self.assertEqual(self.state((10, 12, 8, 11), (10, 11, 6, 7)), "strong bear")

    def test_an_inside_candle_is_neutral_until_an_extreme_is_raided(self):
        self.assertEqual(self.state((10, 12, 8, 11), (10.5, 11.5, 9, 11)), "neutral")

    def test_an_outside_bar_is_resolved_only_by_its_close(self):
        self.assertEqual(self.state((10, 12, 8, 11), (10, 13, 7, 12.5)), "strong bull")
        self.assertEqual(self.state((10, 12, 8, 11), (10, 13, 7, 7.5)), "strong bear")
        self.assertEqual(self.state((10, 12, 8, 11), (10, 13, 7, 10)), "caution")


class NeutralSessionTests(unittest.TestCase):
    """An inside daily candle has no lean, so the raided pool picks the side."""

    def fixture(self):
        first = datetime(2024, 1, 1, 14, 30, tzinfo=timezone.utc)
        daily = [
            bar(first, 9, 14, 6, 10),
            bar(first + timedelta(days=1), 9, 13, 7, 10),   # inside -> neutral
            bar(first + timedelta(days=2), 9, 13, 7, 10),
        ]
        day3 = first + timedelta(days=2)
        fifteen = [bar(first + timedelta(days=1), 9, 13, 7, 10)]
        for offset in range(4):
            fifteen.append(bar(day3 + timedelta(minutes=15 * offset), 12, 12.5, 11.5, 12))
        fifteen += [
            bar(day3 + timedelta(minutes=60), 12, 13.4, 11.9, 12.2),  # raids PDH, closes back
            bar(day3 + timedelta(minutes=75), 12.2, 12.4, 11.8, 12.0),
            bar(day3 + timedelta(minutes=90), 12.0, 12.2, 11.5, 11.7),
            bar(day3 + timedelta(minutes=105), 11.7, 11.8, 11.2, 11.3),
            bar(day3 + timedelta(minutes=120), 11.3, 11.4, 6.9, 7.0),  # reaches PDL
        ]
        hourly = [
            bar(day3, 12, 12.5, 11.5, 12),
            bar(day3 + timedelta(minutes=60), 12, 13.4, 11.5, 11.3),
            bar(day3 + timedelta(minutes=120), 11.3, 11.4, 6.9, 7.0),
        ]
        return daily, hourly, fifteen

    def test_a_raid_above_the_prior_high_shorts_a_neutral_session(self):
        daily, hourly, fifteen = self.fixture()
        trades, _ = run_sfp_strategy(daily, hourly, fifteen)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].bias_state, "neutral")
        self.assertEqual(trades[0].bias, -1)
        self.assertEqual(trades[0].location_kind, "PDH")
        self.assertEqual(trades[0].target_kind, "PDL")

    def test_neutral_sessions_can_be_switched_off(self):
        daily, hourly, fifteen = self.fixture()
        trades, audit = run_sfp_strategy(
            daily, hourly, fifteen,
            config=SfpConfig(trade_neutral_sessions=False),
        )
        self.assertEqual(trades, [])
        self.assertTrue(audit.neutral_skipped)


class AuditTests(unittest.TestCase):
    def test_every_candidate_hour_is_accounted_for(self):
        daily, hourly, fifteen = StrongSetupTests().fixture()
        for confirmation in ("core", "directional", "outside", "strong outside"):
            with self.subTest(confirmation=confirmation):
                _, audit = run_sfp_strategy(
                    daily, hourly, fifteen,
                    config=SfpConfig(hourly_confirmation=confirmation),
                )
                self.assertTrue(
                    audit.reconciles(),
                    f"{audit.accounted} bucketed vs {audit.candidate_hours} candidates",
                )


if __name__ == "__main__":
    unittest.main()
