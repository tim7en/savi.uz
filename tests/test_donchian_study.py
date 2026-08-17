from __future__ import annotations

import unittest

from savi_uz.donchian_study import build_events, summarise
from savi_uz.volume_profile import Bar


def bar(day: str, slot: int, close: float, volume: float = 100.0,
        high: float | None = None, low: float | None = None) -> Bar:
    high = close + 0.2 if high is None else high
    low = close - 0.2 if low is None else low
    return Bar(f"{day}T{14 + slot // 12:02d}:{(slot % 12) * 5:02d}:00.000Z",
               close, high, low, close, volume)


def session(day: str, closes: list[float], volume: float = 100.0) -> list[Bar]:
    return [bar(day, slot, close, volume) for slot, close in enumerate(closes)]


class DonchianStudyTests(unittest.TestCase):
    def history(self) -> list[Bar]:
        return (
            session("2024-01-02", [99.0, 99.5, 100.0, 99.5])
            + session("2024-01-03", [99.5, 100.0, 100.5, 100.0])
        )

    def build(self, bars: list[Bar], floor: float = 0.0):
        return build_events(
            bars, 2, floor, start="2024-01-01", expected_bars=None,
            volume_lookback=2, atr_lookback=2, stop_atr=1.0, target_r=2.0,
        )

    def test_channel_uses_prior_sessions_and_entry_uses_next_open(self):
        today = session("2024-01-04", [100.0, 101.0, 101.5, 102.0])
        events = self.build(self.history() + today)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertAlmostEqual(event.channel_high, 100.7)
        self.assertEqual(event.breakout_close, 101.0)
        self.assertEqual(event.entry, 101.5)

    def test_future_mutation_cannot_change_signal_features(self):
        today = session("2024-01-04", [100.0, 101.0, 101.5, 102.0])
        before = self.build(self.history() + today)[0]
        today[-1] = bar("2024-01-04", 3, 500.0, 999999.0)
        after = self.build(self.history() + today)[0]
        self.assertEqual(
            (before.timestamp, before.channel_high, before.entry, before.volume_ratio),
            (after.timestamp, after.channel_high, after.entry, after.volume_ratio),
        )

    def test_volume_floor_skips_a_low_volume_crossing(self):
        today = session("2024-01-04", [100.0, 101.0, 100.0, 101.2, 101.4], volume=100.0)
        today[1] = bar("2024-01-04", 1, 101.0, volume=50.0)
        events = self.build(self.history() + today, floor=0.75)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].signal_bar, 3)
        self.assertAlmostEqual(events[0].volume_ratio, 1.0)

    def test_sustainable_requires_no_30m_reentry_and_60m_acceptance(self):
        prior = self.history()
        today = session("2024-01-04", [101.0] + [101.2] * 13)
        event = self.build(prior + today)[0]
        self.assertTrue(event.accepted_30m)
        self.assertTrue(event.accepted_60m)
        self.assertFalse(event.reentered_30m)
        self.assertTrue(event.sustainable)

    def test_same_bar_stop_and_target_is_charged_as_stop(self):
        prior = self.history()
        today = session("2024-01-04", [100.0, 101.0, 101.0, 101.0])
        # The entry bar spans both sides of a deliberately small one-ATR risk.
        today[2] = bar("2024-01-04", 2, 101.0, high=105.0, low=95.0)
        event = self.build(prior + today)[0]
        self.assertEqual(event.fixed_r, -1.0)
        self.assertFalse(event.target_before_stop)

    def test_summary_reports_rates(self):
        today = session("2024-01-04", [101.0] + [101.2] * 13)
        event = self.build(self.history() + today)[0]
        result = summarise([event])
        self.assertEqual(result.count, 1)
        self.assertEqual(result.sustainable_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
