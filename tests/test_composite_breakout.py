from __future__ import annotations

import unittest

from savi_uz.composite_breakout import (
    CompositeEvent,
    build_events,
    non_overlapping_results,
    simulate_trade,
)
from savi_uz.volume_profile import Bar


def session(day: str, closes: list[float], volume: float = 100.0) -> list[Bar]:
    return [
        Bar(
            f"{day}T{14 + position // 12:02d}:{(30 + position * 5) % 60:02d}:00.000Z",
            close,
            close + 0.05,
            close - 0.05,
            close,
            volume,
        )
        for position, close in enumerate(closes)
    ]


def event(day: str = "2024-01-03", signal_bar: int = 0) -> CompositeEvent:
    return CompositeEvent(
        session=day,
        timestamp=f"{day}T14:30:00.000Z",
        window=2,
        boundary="range",
        volume_floor=1.0,
        compression_quantile=None,
        direction=1,
        signal_bar=signal_bar,
        lower=99.0,
        upper=101.0,
        entry=100.0,
        atr=1.0,
        daily_atr=4.0,
        volume_ratio=2.0,
        range_atr_ratio=1.0,
        close_return=0.0,
        next_open_return=0.0,
        next_close_return=0.0,
        close_3_return=None,
        close_5_return=None,
    )


class EventConstructionTests(unittest.TestCase):
    def bars(self) -> list[Bar]:
        history = []
        for day in range(1, 5):
            history += session(f"2024-01-0{day}", [100.0] * 6)
        history += session("2024-01-05", [100.0, 101.0, 101.2, 101.3, 101.4, 101.5], 200.0)
        history += session("2024-01-06", [102.0] * 6)
        return history

    def build(self, bars: list[Bar], floor: float = 1.0):
        return build_events(
            bars,
            2,
            "range",
            floor,
            start="2024-01-01",
            expected_bars=6,
            volume_lookback=2,
            min_volume_observations=2,
            atr_lookback=2,
        )

    def test_profile_uses_prior_sessions_and_entry_uses_next_bar(self):
        result = self.build(self.bars())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].session, "2024-01-05")
        self.assertEqual(result[0].signal_bar, 1)
        self.assertAlmostEqual(result[0].entry, 101.2)
        self.assertLess(result[0].upper, result[0].entry)

    def test_future_session_mutation_cannot_change_signal_features(self):
        bars = self.bars()
        original = self.build(bars)[0]
        changed = [
            Bar(row.timestamp, row.open * 10, row.high * 10, row.low * 10,
                row.close * 10, row.volume)
            if row.timestamp.startswith("2024-01-06") else row
            for row in bars
        ]
        mutated = self.build(changed)[0]
        self.assertEqual(original.timestamp, mutated.timestamp)
        self.assertEqual(original.upper, mutated.upper)
        self.assertEqual(original.entry, mutated.entry)

    def test_relative_volume_floor_can_reject_the_crossing(self):
        self.assertEqual(self.build(self.bars(), floor=3.0), [])


class OvernightExecutionTests(unittest.TestCase):
    def test_gap_through_stop_fills_at_next_open(self):
        sessions = [
            ("2024-01-03", session("2024-01-03", [100.0, 100.0, 100.0])),
            ("2024-01-04", session("2024-01-04", [95.0, 96.0, 97.0])),
        ]
        result = simulate_trade(event(), sessions, stop_atr=2.0, max_hold_sessions=1)
        self.assertEqual(result.reason, "gap_stop")
        self.assertAlmostEqual(result.exit_price, 95.0)
        self.assertLess(result.gross_return, -0.02)

    def test_trailing_stop_activates_for_the_following_bar(self):
        rows = session("2024-01-03", [100.0, 100.0, 101.0])
        rows[1] = Bar(rows[1].timestamp, 100.0, 103.0, 99.0, 102.5, 100.0)
        rows[2] = Bar(rows[2].timestamp, 102.5, 102.8, 101.0, 101.5, 100.0)
        result = simulate_trade(
            event(), [("2024-01-03", rows)], stop_atr=2.5,
            trail_atr=1.0, activation_atr=2.0, max_hold_sessions=0,
        )
        self.assertEqual(result.reason, "stop")
        self.assertAlmostEqual(result.exit_price, 102.0)

    def test_overlapping_signals_are_skipped_while_trade_is_open(self):
        sessions = [
            ("2024-01-03", session("2024-01-03", [100.0, 100.0, 100.0])),
            ("2024-01-04", session("2024-01-04", [100.0, 100.0, 100.0])),
            ("2024-01-05", session("2024-01-05", [100.0, 100.0, 100.0])),
        ]
        events = [event("2024-01-03"), event("2024-01-04")]
        results = non_overlapping_results(events, sessions, max_hold_sessions=1)
        self.assertEqual(len(results), 1)

    def test_no_hard_stop_holds_until_the_time_exit(self):
        sessions = [
            ("2024-01-03", session("2024-01-03", [100.0, 90.0, 91.0])),
            ("2024-01-04", session("2024-01-04", [80.0, 85.0, 90.0])),
        ]
        result = simulate_trade(event(), sessions, stop_atr=None, max_hold_sessions=1)
        self.assertEqual(result.reason, "time")
        self.assertAlmostEqual(result.exit_price, 90.0)

    def test_daily_atr_stop_uses_the_daily_scale(self):
        sessions = [
            ("2024-01-03", session("2024-01-03", [100.0, 97.0, 97.0])),
        ]
        result = simulate_trade(
            event(), sessions, stop_atr=None, stop_daily_atr=0.5,
            max_hold_sessions=0,
        )
        self.assertEqual(result.reason, "stop")
        self.assertAlmostEqual(result.exit_price, 98.0)


if __name__ == "__main__":
    unittest.main()
