from __future__ import annotations

import unittest

from savi_uz.breakout_study import (
    Sample,
    build_samples,
    bucket_by,
    group_sessions,
    quantile_edges,
    quantile_labeller,
    split_by_date,
    stratified_buckets,
    welch_t,
)
from savi_uz.volume_profile import Bar, build_profile, classify_shape, count_peaks


def _bar(ts: str, o: float, h: float, low: float, c: float, v: float | None = 1000.0) -> Bar:
    return Bar(ts, o, h, low, c, v)


def _session(day: str, prices: list[float], volumes: list[float] | None = None) -> list[Bar]:
    """One bar per hour; each bar spans +/-0.5 around its close."""
    volumes = volumes or [1000.0] * len(prices)
    return [
        _bar(f"{day}T{14 + i:02d}:00:00.000Z", p, p + 0.5, p - 0.5, p, volumes[i])
        for i, p in enumerate(prices)
    ]


class ProfileTests(unittest.TestCase):
    def test_volume_concentrates_at_the_price_that_traded_most(self):
        bars = [_bar("d", 100, 101, 99, 100, 100.0), _bar("d", 100, 100.2, 99.8, 100, 900.0)]
        p = build_profile(bars, bins=10)
        self.assertAlmostEqual(p.poc, 100.0, delta=0.3)

    def test_value_area_holds_about_seventy_percent(self):
        bars = _session("2024-01-02", [100, 100.1, 100.2, 100.1, 100, 103])
        p = build_profile(bars, bins=20)
        inside = sum(v for e, v in zip(p.edges, p.volume) if p.value_low <= e < p.value_high)
        self.assertGreaterEqual(inside / p.total_volume, 0.60)

    def test_a_flat_session_does_not_divide_by_zero(self):
        bars = [_bar("d", 100, 100, 100, 100, 500.0)]
        p = build_profile(bars, bins=10)
        self.assertEqual(p.shape, "flat")
        self.assertEqual(p.poc, 100)

    def test_bars_without_volume_are_excluded(self):
        self.assertIsNone(build_profile([_bar("d", 100, 101, 99, 100, None)], bins=10))

    def test_poc_position_is_scale_free(self):
        bars = _session("2024-01-02", [100, 100, 100, 105])
        p = build_profile(bars, bins=20)
        self.assertLess(p.poc_position, 0.35)

    def test_value_width_is_narrow_when_volume_is_stacked(self):
        tight = build_profile(_session("2024-01-02", [100, 100, 100, 100, 106]), bins=20)
        loose = build_profile(_session("2024-01-02", [100, 102, 104, 106, 108]), bins=20)
        self.assertLess(tight.value_width, loose.value_width)


class ShapeTests(unittest.TestCase):
    def test_a_single_middle_peak_is_balanced(self):
        self.assertEqual(classify_shape([1, 2, 9, 2, 1], 0.5)[0], "D")

    def test_volume_high_in_the_range_is_a_p(self):
        self.assertEqual(classify_shape([1, 1, 2, 9], 0.9)[0], "P")

    def test_volume_low_in_the_range_is_a_b(self):
        self.assertEqual(classify_shape([9, 2, 1, 1], 0.1)[0], "b")

    def test_two_separated_modes_are_a_double_distribution(self):
        shape, peaks = classify_shape([9, 1, 1, 1, 9], 0.5)
        self.assertEqual(shape, "B")
        self.assertEqual(peaks, 2)

    def test_a_jagged_single_peak_is_not_counted_as_two(self):
        """Without a trough test almost every histogram reads as multi-modal."""
        self.assertEqual(count_peaks([1, 8, 9, 8, 1]), 1)

    def test_a_shallow_dip_does_not_split_a_peak(self):
        self.assertEqual(count_peaks([10, 9, 10]), 1)

    def test_an_empty_histogram_has_no_peaks(self):
        self.assertEqual(count_peaks([]), 0)
        self.assertEqual(count_peaks([0, 0, 0]), 0)


class LookAheadTests(unittest.TestCase):
    """The load-bearing tests: a feature must not move when the future does."""

    def setUp(self):
        self.bars = _session("2024-01-02", [100, 101, 102, 103, 104, 105])

    def test_features_are_unchanged_when_later_bars_are_rewritten(self):
        base = build_samples(self.bars, bins=12)
        mutated = list(self.bars)
        # Replace the final bar with something wild. Every sample except the
        # one whose target reads it must be byte-identical.
        mutated[-1] = _bar("2024-01-02T19:00:00.000Z", 105, 400, 5, 300, 99999.0)
        after = build_samples(mutated, bins=12)

        self.assertEqual(len(base), len(after))
        for b, a in zip(base, after):
            if b.timestamp == self.bars[-2].timestamp:
                continue  # this row's target legitimately reads the mutated bar
            self.assertEqual(
                (b.shape, b.poc_position, b.value_width, b.concentration,
                 b.close_position, b.range_pct, b.volume_ratio, b.forward_return),
                (a.shape, a.poc_position, a.value_width, a.concentration,
                 a.close_position, a.range_pct, a.volume_ratio, a.forward_return),
            )

    def test_the_profile_at_bar_t_uses_only_bars_up_to_t(self):
        samples = build_samples(self.bars, bins=12)
        first = samples[0]
        # The prefix through bar 3 tops out at 102.5; a profile that had seen
        # the whole rising session would put the close far lower in the range.
        prefix = build_profile(self.bars[:3], bins=12)
        self.assertAlmostEqual(first.close_position, (102 - prefix.low) / prefix.price_range, places=9)
        self.assertGreater(first.close_position, 0.8)

    def test_the_target_is_the_next_bar_and_nothing_else(self):
        samples = build_samples(self.bars, bins=12)
        for s in samples:
            hour = int(s.timestamp[11:13])
            nxt = next(b for b in self.bars if int(b.timestamp[11:13]) == hour + 1)
            self.assertAlmostEqual(s.forward_return, nxt.close / s.close - 1.0, places=12)

    def test_the_last_bar_of_a_session_is_dropped(self):
        """Its next bar is tomorrow morning: an overnight gap, not an intraday move."""
        samples = build_samples(self.bars, bins=12)
        self.assertNotIn(self.bars[-1].timestamp, [s.timestamp for s in samples])

    def test_sessions_do_not_bleed_into_each_other(self):
        two = _session("2024-01-02", [100, 101, 102, 103, 104, 105]) + \
              _session("2024-01-03", [200, 201, 202, 203, 204, 205])
        samples = build_samples(two, bins=12)
        for s in samples:
            self.assertTrue(s.timestamp.startswith(s.session))
            # A cross-session target would show up as a ~100% jump.
            self.assertLess(abs(s.forward_return), 0.5)

    def test_a_session_too_short_to_profile_is_skipped(self):
        self.assertEqual(build_samples(_session("2024-01-02", [100, 101]), bins=12), [])


class SampleShapeTests(unittest.TestCase):
    def test_one_sample_per_eligible_bar(self):
        bars = _session("2024-01-02", [100, 101, 102, 103, 104, 105])
        samples = build_samples(bars, bins=12, min_prefix=3)
        # bars 3,4,5 of six: the last is dropped, the first two are too early.
        self.assertEqual([s.bars_elapsed for s in samples], [3, 4, 5])

    def test_bars_without_volume_do_not_produce_samples(self):
        bars = _session("2024-01-02", [100, 101, 102, 103, 104, 105],
                        volumes=[1000, None, None, None, None, 1000])
        self.assertEqual(build_samples(bars, bins=12), [])

    def test_grouping_orders_bars_within_a_session(self):
        shuffled = list(reversed(_session("2024-01-02", [100, 101, 102])))
        grouped = group_sessions(shuffled)["2024-01-02"]
        self.assertEqual([b.timestamp for b in grouped], sorted(b.timestamp for b in grouped))


class StatsTests(unittest.TestCase):
    def _sample(self, session: str, shape: str, fwd: float) -> Sample:
        return Sample(session, session + "T14:00:00.000Z", 3, 100.0, shape, 0.5, 0.0, 0.5,
                      0.4, 0.5, 0.01, 1.0, fwd, abs(fwd), abs(fwd))

    def test_lift_is_one_when_a_bucket_matches_the_rest(self):
        rows = [self._sample("2024-01-0%d" % i, "D" if i % 2 else "P", 0.01) for i in range(1, 9)]
        buckets = bucket_by(rows, lambda s: s.shape, ["D", "P"])
        for b in buckets:
            self.assertAlmostEqual(b.lift, 1.0, places=9)

    def test_lift_exceeds_one_for_a_genuinely_bigger_bucket(self):
        rows = [self._sample("2024-01-01", "B", 0.03) for _ in range(10)]
        rows += [self._sample("2024-01-02", "D", 0.01) for _ in range(10)]
        b = {x.label: x for x in bucket_by(rows, lambda s: s.shape, ["B", "D"])}
        self.assertGreater(b["B"].lift, 2.5)
        self.assertLess(b["D"].lift, 0.5)

    def test_welch_handles_degenerate_input(self):
        self.assertTrue(math_isnan(welch_t([1.0], [2.0])))
        self.assertTrue(math_isnan(welch_t([1.0, 1.0], [1.0, 1.0])))

    def test_quantile_edges_partition(self):
        self.assertEqual(len(quantile_edges(list(range(100)), 5)), 4)
        self.assertEqual(quantile_edges([], 5), [])

    def test_the_split_is_chronological_not_random(self):
        rows = [self._sample("2024-01-01", "D", 0.01), self._sample("2024-06-01", "D", 0.02)]
        train, test = split_by_date(rows, "2024-03-01")
        self.assertEqual([s.session for s in train], ["2024-01-01"])
        self.assertEqual([s.session for s in test], ["2024-06-01"])


def math_isnan(value: float) -> bool:
    return value != value


if __name__ == "__main__":
    unittest.main()


class StratifiedTests(unittest.TestCase):
    """A feature must not win just by correlating with the control."""

    def _s(self, session: str, shape: str, elapsed: int, fwd: float) -> Sample:
        return Sample(session, session + "T14:00:00.000Z", elapsed, 100.0, shape,
                      0.5, 0.0, 0.5, 0.4, 0.5, 0.01, 1.0, fwd, abs(fwd), abs(fwd))

    def test_a_feature_that_only_proxies_the_control_loses_its_lift(self):
        """Shape 'P' appears only in the loud stratum. Raw lift is large;
        controlled lift must be ~1 because inside each stratum it is average."""
        rows = []
        for i in range(60):
            rows.append(self._s("2024-01-01", "P", 5, 0.04))   # loud bar-5 stratum
            rows.append(self._s("2024-01-01", "D", 5, 0.04))
            rows.append(self._s("2024-01-02", "D", 1, 0.01))   # quiet bar-1 stratum
            rows.append(self._s("2024-01-02", "D", 1, 0.01))

        raw = {b.label: b for b in bucket_by(rows, lambda s: s.shape, ["P", "D"])}
        self.assertGreater(raw["P"].lift, 1.5)

        ctrl = {b.label: b for b in stratified_buckets(
            rows, lambda s: s.shape, ["P", "D"], [lambda s: s.bars_elapsed])}
        self.assertAlmostEqual(ctrl["P"].lift, 1.0, places=6)

    def test_a_genuine_within_stratum_effect_survives(self):
        rows = []
        for i in range(60):
            rows.append(self._s("2024-01-01", "P", 5, 0.06))
            rows.append(self._s("2024-01-01", "D", 5, 0.02))
            rows.append(self._s("2024-01-02", "P", 1, 0.03))
            rows.append(self._s("2024-01-02", "D", 1, 0.01))
        ctrl = {b.label: b for b in stratified_buckets(
            rows, lambda s: s.shape, ["P", "D"], [lambda s: s.bars_elapsed])}
        self.assertGreater(ctrl["P"].lift, 1.4)
        self.assertLess(ctrl["D"].lift, 0.7)

    def test_thin_strata_are_dropped_rather_than_trusted(self):
        rows = [self._s("2024-01-01", "P", 5, 0.05)] * 3
        self.assertEqual(
            stratified_buckets(rows, lambda s: s.shape, ["P"], [lambda s: s.bars_elapsed]), []
        )

    def test_the_t_statistic_tests_the_same_quantity_as_the_lift(self):
        """A t-stat computed on raw moves while lift is stratified would point
        the other way whenever the strata differ in size."""
        rows = []
        for i in range(60):
            rows.append(self._s("2024-01-01", "P", 5, 0.06))
            rows.append(self._s("2024-01-01", "D", 5, 0.02))
            rows.append(self._s("2024-01-02", "D", 1, 0.01))
        ctrl = {b.label: b for b in stratified_buckets(
            rows, lambda s: s.shape, ["P", "D"], [lambda s: s.bars_elapsed])}
        self.assertGreater(ctrl["P"].lift, 1.0)
        self.assertGreater(ctrl["P"].t_stat, 0)
        self.assertLess(ctrl["D"].lift, 1.0)
        self.assertLess(ctrl["D"].t_stat, 0)

    def test_quantile_labeller_splits_into_the_requested_parts(self):
        rows = [self._s("2024-01-01", "D", 1, 0.01 * i) for i in range(1, 101)]
        lab = quantile_labeller(rows, lambda s: s.forward_abs, 5)
        self.assertEqual(sorted({lab(r) for r in rows}), [0, 1, 2, 3, 4])
