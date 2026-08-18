from __future__ import annotations

import unittest

from savi_uz.split_adjust import adjust_bars, cumulative_factors, largest_overnight_gap
from savi_uz.volume_profile import Bar


def bar(day: str, price: float, volume: float = 100.0) -> Bar:
    return Bar(f"{day}T14:30:00+00:00", price, price * 1.01, price * 0.99, price, volume)


class CumulativeFactorTests(unittest.TestCase):
    def test_a_single_split_applies_its_own_factor(self):
        self.assertEqual(cumulative_factors([("2022-06-06", 20.0)]),
                         [("2022-06-06", 20.0)])

    def test_bars_older_than_two_splits_carry_the_product(self):
        # Before the first split a bar has to absorb both; between them, only
        # the later one still applies.
        self.assertEqual(
            cumulative_factors([("2020-01-02", 2.0), ("2022-06-06", 5.0)]),
            [("2020-01-02", 10.0), ("2022-06-06", 5.0)],
        )

    def test_no_splits_means_no_schedule(self):
        self.assertEqual(cumulative_factors([]), [])


class AdjustBarsTests(unittest.TestCase):
    def test_prices_before_the_split_are_divided_and_volume_multiplied(self):
        rows = [bar("2022-06-03", 2400.0, 50.0), bar("2022-06-06", 122.0, 1000.0)]
        out = adjust_bars(rows, [("2022-06-06", 20.0)])
        self.assertAlmostEqual(out[0].open, 120.0)
        self.assertAlmostEqual(out[0].high, 121.2)
        self.assertAlmostEqual(out[0].volume, 1000.0)
        # The split-day bar already prints post-split and must not move.
        self.assertEqual(out[1], rows[1])

    def test_the_fake_overnight_collapse_disappears(self):
        rows = [bar("2022-06-03", 2400.0), bar("2022-06-06", 120.0)]
        before = largest_overnight_gap(rows)[1]
        after = largest_overnight_gap(adjust_bars(rows, [("2022-06-06", 20.0)]))[1]
        self.assertLess(before, -0.9)
        self.assertAlmostEqual(after, 0.0, places=6)

    def test_bars_with_no_split_are_returned_untouched(self):
        rows = [bar("2024-01-02", 100.0), bar("2024-01-03", 101.0)]
        self.assertEqual(adjust_bars(rows, []), rows)

    def test_a_bar_with_no_volume_stays_none(self):
        rows = [Bar("2022-06-03T14:30:00+00:00", 20.0, 21.0, 19.0, 20.0, None)]
        self.assertIsNone(adjust_bars(rows, [("2022-06-06", 2.0)])[0].volume)

    def test_two_splits_step_the_history_in_two_stages(self):
        rows = [bar("2019-01-02", 400.0), bar("2021-01-04", 300.0), bar("2023-01-03", 60.0)]
        out = adjust_bars(rows, [("2020-01-02", 2.0), ("2022-06-06", 5.0)])
        self.assertAlmostEqual(out[0].open, 40.0)   # divided by 10
        self.assertAlmostEqual(out[1].open, 60.0)   # divided by 5
        self.assertAlmostEqual(out[2].open, 60.0)   # untouched


class GapHelperTests(unittest.TestCase):
    def test_reports_the_largest_move_and_its_date(self):
        rows = [bar("2024-01-02", 100.0), bar("2024-01-03", 110.0), bar("2024-01-04", 60.0)]
        day, gap = largest_overnight_gap(rows)
        self.assertEqual(day, "2024-01-04")
        self.assertLess(gap, -0.4)

    def test_a_single_bar_has_no_gap(self):
        self.assertEqual(largest_overnight_gap([bar("2024-01-02", 100.0)]), ("", 0.0))


if __name__ == "__main__":
    unittest.main()
