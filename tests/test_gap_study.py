from __future__ import annotations

import unittest

from savi_uz.gap_study import (
    bucket_by_size,
    build_gaps,
    group_sessions,
    median,
    summarise,
)
from savi_uz.volume_profile import Bar


def _session(day: str, prices: list[float], volume: float | None = 1000.0) -> list[Bar]:
    """One 5-minute bar per price, each spanning +/-0.05 around it."""
    return [
        Bar(f"{day}T{13 + i // 12:02d}:{(30 + 5 * i) % 60:02d}:00.000Z",
            p, p + 0.05, p - 0.05, p, volume)
        for i, p in enumerate(prices)
    ]


def _pair(prior: list[float], today: list[float]) -> dict[str, list[Bar]]:
    return {
        "2024-01-02": _session("2024-01-02", prior),
        "2024-01-03": _session("2024-01-03", today),
    }


class GapConstructionTests(unittest.TestCase):
    def test_the_gap_is_prior_close_to_next_open(self):
        # prior closes at 100, next opens at 101 -> +100bp
        sessions = _pair([100.0] * 12, [101.0] * 12)
        gap = build_gaps(sessions, bins=10)[0]
        self.assertAlmostEqual(gap.gap_bp, 100.0, places=6)
        self.assertEqual(gap.direction, 1)
        self.assertAlmostEqual(gap.prior_close, 100.0)
        self.assertAlmostEqual(gap.open, 101.0)

    def test_a_down_gap_has_negative_direction(self):
        gap = build_gaps(_pair([100.0] * 12, [99.0] * 12), bins=10)[0]
        self.assertEqual(gap.direction, -1)
        self.assertLess(gap.gap_bp, 0)

    def test_gaps_below_the_floor_are_dropped(self):
        sessions = _pair([100.0] * 12, [100.01] * 12)   # 1bp
        self.assertEqual(build_gaps(sessions, bins=10, min_gap_bp=5.0), [])

    def test_a_session_with_thin_volume_coverage_is_skipped(self):
        sessions = _pair([100.0] * 12, [101.0] * 12)
        sessions["2024-01-03"] = _session("2024-01-03", [101.0] * 12, volume=None)
        self.assertEqual(build_gaps(sessions, bins=10), [])

    def test_only_consecutive_available_sessions_are_paired(self):
        """Three sessions give two gaps, each against the one before it."""
        sessions = {
            "2024-01-02": _session("2024-01-02", [100.0] * 12),
            "2024-01-03": _session("2024-01-03", [101.0] * 12),
            "2024-01-04": _session("2024-01-04", [102.0] * 12),
        }
        gaps = build_gaps(sessions, bins=10)
        self.assertEqual([g.session for g in gaps], ["2024-01-03", "2024-01-04"])


class RetentionTests(unittest.TestCase):
    """Retention is the number that answers 'did the move hold'."""

    def test_holding_the_whole_gap_retains_one(self):
        # opens at 101 after a 100 close, and closes at 101
        gap = build_gaps(_pair([100.0] * 12, [101.0] * 12), bins=10)[0]
        self.assertAlmostEqual(gap.retained, 1.0, places=6)
        self.assertFalse(gap.extended)
        self.assertFalse(gap.reversed_through)

    def test_giving_the_gap_back_exactly_retains_zero(self):
        # opens at 101, closes back at 100
        today = [101.0] * 6 + [100.0] * 6
        gap = build_gaps(_pair([100.0] * 12, today), bins=10)[0]
        self.assertAlmostEqual(gap.retained, 0.0, places=6)
        self.assertTrue(gap.filled)

    def test_extending_beyond_the_gap_retains_above_one(self):
        today = [101.0] * 6 + [102.0] * 6
        gap = build_gaps(_pair([100.0] * 12, today), bins=10)[0]
        self.assertAlmostEqual(gap.retained, 2.0, places=6)
        self.assertTrue(gap.extended)

    def test_crossing_back_through_the_prior_close_retains_below_zero(self):
        today = [101.0] * 6 + [99.0] * 6
        gap = build_gaps(_pair([100.0] * 12, today), bins=10)[0]
        self.assertLess(gap.retained, 0.0)
        self.assertTrue(gap.reversed_through)

    def test_retention_is_scale_free(self):
        """A 20bp gap held in full and a 200bp gap held in full both retain 1."""
        small = build_gaps(_pair([100.0] * 12, [100.2] * 12), bins=10)[0]
        large = build_gaps(_pair([100.0] * 12, [102.0] * 12), bins=10)[0]
        self.assertAlmostEqual(small.retained, large.retained, places=6)


class FillTests(unittest.TestCase):
    def test_a_gap_that_never_returns_is_unfilled(self):
        gap = build_gaps(_pair([100.0] * 12, [101.0] * 12), bins=10)[0]
        self.assertFalse(gap.filled)
        self.assertIsNone(gap.bars_to_fill)

    def test_fill_records_the_bar_it_happened_on(self):
        today = [101.0, 101.0, 100.0] + [100.5] * 9
        gap = build_gaps(_pair([100.0] * 12, today), bins=10)[0]
        self.assertTrue(gap.filled)
        self.assertEqual(gap.bars_to_fill, 3)

    def test_a_down_gap_fills_upward(self):
        today = [99.0, 99.0, 100.0] + [99.5] * 9
        gap = build_gaps(_pair([100.0] * 12, today), bins=10)[0]
        self.assertTrue(gap.filled)
        self.assertEqual(gap.bars_to_fill, 3)

    def test_fill_and_retention_are_different_claims(self):
        """Price can touch the prior close and still close having held the gap."""
        today = [101.0, 100.0] + [101.0] * 10
        gap = build_gaps(_pair([100.0] * 12, today), bins=10)[0]
        self.assertTrue(gap.filled)
        self.assertAlmostEqual(gap.retained, 1.0, places=6)


class AcceptanceTests(unittest.TestCase):
    def test_trading_back_into_yesterdays_range_overlaps(self):
        prior = [100.0 + (i % 5) * 0.1 for i in range(24)]
        gap = build_gaps(_pair(prior, prior), bins=10, min_gap_bp=0.0)
        # identical ranges: if a gap is recorded at all, overlap is total
        if gap:
            self.assertGreater(gap[0].value_overlap, 0.9)

    def test_a_move_to_a_new_level_does_not_overlap(self):
        prior = [100.0 + (i % 5) * 0.02 for i in range(24)]
        today = [110.0 + (i % 5) * 0.02 for i in range(24)]
        gap = build_gaps({"2024-01-02": _session("2024-01-02", prior),
                          "2024-01-03": _session("2024-01-03", today)}, bins=10)[0]
        self.assertEqual(gap.value_overlap, 0.0)
        self.assertGreater(gap.poc_shift_bp, 900)

    def test_opening_volume_ratio_reflects_a_busy_open(self):
        today_bars = _session("2024-01-03", [101.0] * 24)
        loud = [Bar(b.timestamp, b.open, b.high, b.low, b.close,
                    5000.0 if i < 6 else 1000.0) for i, b in enumerate(today_bars)]
        sessions = {"2024-01-02": _session("2024-01-02", [100.0] * 24),
                    "2024-01-03": loud}
        gap = build_gaps(sessions, bins=10)[0]
        self.assertGreater(gap.opening_volume_ratio, 2.0)


class SummaryTests(unittest.TestCase):
    def _gap(self, size_bp: float, retained: float):
        prior = [100.0] * 12
        opening = 100.0 * (1 + size_bp / 10_000)
        closing = 100.0 + (opening - 100.0) * retained
        today = [opening] * 6 + [closing] * 6
        return build_gaps(_pair(prior, today), bins=10, min_gap_bp=0.0)[0]

    def test_median_retention_is_not_dragged_by_outliers(self):
        """A few sessions extending many times over would pull a mean above 1."""
        gaps = [self._gap(50, 0.5) for _ in range(9)] + [self._gap(50, 20.0)]
        summary = summarise(gaps, "x")
        self.assertLess(summary.median_retained, 1.0)

    def test_buckets_partition_by_absolute_size(self):
        gaps = [self._gap(8, 1.0), self._gap(30, 1.0), self._gap(200, 1.0)]
        buckets = {b.label: b for b in bucket_by_size(gaps)}
        self.assertEqual(buckets["0-10bp"].count, 1)
        self.assertEqual(buckets["25-50bp"].count, 1)
        self.assertEqual(buckets["over 100bp"].count, 1)

    def test_down_gaps_bucket_by_magnitude_not_sign(self):
        prior = [100.0] * 12
        gap = build_gaps(_pair(prior, [98.0] * 12), bins=10)[0]
        buckets = {b.label: b for b in bucket_by_size([gap])}
        self.assertIn("over 100bp", buckets)

    def test_median_of_an_empty_list_is_nan(self):
        value = median([])
        self.assertNotEqual(value, value)

    def test_grouping_splits_on_calendar_date(self):
        bars = _session("2024-01-02", [100.0] * 3) + _session("2024-01-03", [101.0] * 3)
        self.assertEqual(sorted(group_sessions(bars)), ["2024-01-02", "2024-01-03"])


if __name__ == "__main__":
    unittest.main()
