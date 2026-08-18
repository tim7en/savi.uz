from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta, timezone

from savi_uz.turtle import (
    TurtleConfig,
    rolling_extremes,
    run_turtle,
    summarise_turtle,
    true_ranges,
    wilder_atr,
)
from savi_uz.volume_profile import Bar

START = datetime(2024, 1, 1, 14, 30, tzinfo=timezone.utc)


def bar(step: int, open_: float, high: float, low: float, close: float,
        *, minutes: int = 0) -> Bar:
    stamp = START + (timedelta(minutes=minutes * step) if minutes
                     else timedelta(days=step))
    return Bar(stamp.isoformat(), open_, high, low, close, 1000.0)


def flat_then_breakout(quiet: int = 30, drive: int = 40) -> list[Bar]:
    """A quiet channel, then a sustained one-way move that lets units stack."""
    bars = [bar(i, 100, 101, 99, 100) for i in range(quiet)]
    price = 100.0
    for i in range(drive):
        price += 2.0
        bars.append(bar(quiet + i, price - 1, price + 1, price - 1.5, price))
    return bars


class TrueRangeTests(unittest.TestCase):
    def test_first_bar_uses_its_own_range(self):
        rows = [bar(0, 10, 12, 9, 11)]
        self.assertEqual(true_ranges(rows), [3])

    def test_a_gap_beyond_the_bar_widens_the_range(self):
        rows = [bar(0, 10, 12, 9, 11), bar(1, 20, 21, 19, 20)]
        # High 21 against a previous close of 11 dominates the 2-point bar range.
        self.assertEqual(true_ranges(rows)[1], 10)


class WilderAtrTests(unittest.TestCase):
    def test_seed_is_the_simple_mean_of_the_first_window(self):
        rows = [bar(i, 10, 12, 9, 11) for i in range(10)]
        atr = wilder_atr(rows, window=5)
        self.assertTrue(all(math.isnan(value) for value in atr[:4]))
        self.assertAlmostEqual(atr[4], sum(true_ranges(rows)[:5]) / 5)

    def test_later_values_smooth_towards_new_ranges(self):
        rows = [bar(i, 10, 11, 9, 10) for i in range(6)]
        rows.append(bar(6, 10, 30, 10, 30))
        atr = wilder_atr(rows, window=5)
        self.assertGreater(atr[6], atr[5])
        self.assertLess(atr[6], 20)

    def test_a_short_series_has_no_value_at_all(self):
        self.assertTrue(all(math.isnan(v) for v in wilder_atr([bar(0, 1, 2, 0, 1)], 5)))

    def test_atr_at_a_bar_ignores_every_later_bar(self):
        rows = [bar(i, 10, 11, 9, 10) for i in range(20)]
        extended = rows + [bar(20, 10, 500, 1, 400)]
        self.assertEqual(wilder_atr(rows, 10)[19], wilder_atr(extended, 10)[19])


class EntryAndExitTests(unittest.TestCase):
    def test_a_breakout_enters_at_the_channel_edge(self):
        rows = flat_then_breakout()
        trades, _ = run_turtle(rows, config=TurtleConfig(directions=(1,)))
        self.assertTrue(trades)
        # The channel high over the quiet stretch is 101, so the stop order fills
        # there rather than at the breakout bar's close.
        self.assertAlmostEqual(trades[0].entry, 101.0)
        self.assertEqual(trades[0].direction, 1)

    def test_a_gap_through_the_level_fills_at_the_open(self):
        rows = [bar(i, 100, 101, 99, 100) for i in range(30)]
        rows.append(bar(30, 140, 145, 139, 144))
        trades, _ = run_turtle(rows, config=TurtleConfig(directions=(1,)))
        self.assertAlmostEqual(trades[0].entry, 140.0)

    def test_the_position_leaves_on_the_opposite_channel(self):
        rows = flat_then_breakout()
        price = rows[-1].close
        for i in range(15):
            price -= 4.0
            rows.append(bar(70 + i, price + 2, price + 2.5, price - 1, price))
        trades, _ = run_turtle(rows, config=TurtleConfig(directions=(1,)))
        self.assertIn(trades[0].exit_reason, {"channel", "stop"})
        self.assertLess(trades[0].exit, rows[69].close)


class PyramidTests(unittest.TestCase):
    def test_units_stack_up_to_the_configured_maximum(self):
        rows = flat_then_breakout()
        config = TurtleConfig(directions=(1,), max_units=4)
        trades, _ = run_turtle(rows, config=config)
        self.assertLessEqual(trades[0].units, 4)
        self.assertGreater(trades[0].units, 1)

    def test_a_single_unit_system_never_adds(self):
        rows = flat_then_breakout()
        trades, _ = run_turtle(rows, config=TurtleConfig(directions=(1,), max_units=1))
        self.assertTrue(all(trade.units == 1 for trade in trades))

    def test_each_added_unit_pulls_the_stop_up_behind_it(self):
        rows = flat_then_breakout()
        one, _ = run_turtle(rows, config=TurtleConfig(directions=(1,), max_units=1))
        four, _ = run_turtle(rows, config=TurtleConfig(directions=(1,), max_units=4))
        self.assertGreater(four[0].stop_at_exit, one[0].stop_at_exit)


class SkipFilterTests(unittest.TestCase):
    def test_the_filter_can_only_reduce_the_number_of_trades(self):
        rows = flat_then_breakout()
        price = rows[-1].close
        for cycle in range(6):  # alternate drops and rallies to make many signals
            for i in range(12):
                price += -3.0 if cycle % 2 == 0 else 3.0
                rows.append(bar(len(rows), price, price + 1.5, price - 1.5, price))
        filtered, skipped = run_turtle(rows, config=TurtleConfig(skip_after_winner=True))
        unfiltered, none = run_turtle(rows, config=TurtleConfig(skip_after_winner=False))
        self.assertEqual(none, 0)
        self.assertLessEqual(len(filtered), len(unfiltered))
        self.assertGreaterEqual(skipped, 0)


class OvernightTests(unittest.TestCase):
    def test_an_intraday_only_system_is_flat_by_the_session_close(self):
        rows = []
        for session in range(40):
            for slot in range(8):
                stamp = (START + timedelta(days=session, minutes=30 * slot)).isoformat()
                price = 100 + (session if slot > 3 else 0)
                rows.append(Bar(stamp, price, price + 1, price - 1, price, 1000.0))
        trades, _ = run_turtle(
            rows, config=TurtleConfig(allow_overnight=False, entry_window=10,
                                      exit_window=5, atr_window=10),
        )
        for trade in trades:
            self.assertEqual(trade.sessions_held, 1)
            self.assertEqual(trade.entry_timestamp[:10], trade.exit_timestamp[:10])


class LeakageTests(unittest.TestCase):
    def test_later_bars_cannot_change_earlier_trades(self):
        rows = flat_then_breakout()
        early, _ = run_turtle(rows, config=TurtleConfig(directions=(1,)))
        extended = rows + [bar(len(rows) + i, 500, 900, 100, 200) for i in range(20)]
        late, _ = run_turtle(extended, config=TurtleConfig(directions=(1,)))
        self.assertTrue(early)
        # Every trade that had already closed must survive the extension intact.
        closed = [t for t in early if t.exit_timestamp < rows[-1].timestamp]
        for original in closed:
            self.assertIn(original, late)


class CostTests(unittest.TestCase):
    def test_costs_are_charged_per_unit_and_reduce_the_result(self):
        rows = flat_then_breakout()
        free, _ = run_turtle(rows, config=TurtleConfig(directions=(1,), round_trip_cost=0.0))
        paid, _ = run_turtle(rows, config=TurtleConfig(directions=(1,), round_trip_cost=0.001))
        self.assertEqual(free[0].cost_r, 0.0)
        self.assertGreater(paid[0].cost_r, 0.0)
        self.assertLess(paid[0].net_r, free[0].net_r)


class SummaryTests(unittest.TestCase):
    def test_an_empty_run_summarises_without_dividing_by_zero(self):
        result = summarise_turtle([])
        self.assertEqual(result.trades, 0)
        self.assertTrue(math.isnan(result.mean_r))

    def test_totals_agree_with_the_trade_list(self):
        rows = flat_then_breakout()
        trades, skipped = run_turtle(rows, config=TurtleConfig(directions=(1,)))
        result = summarise_turtle(trades, skipped)
        self.assertEqual(result.trades, len(trades))
        self.assertAlmostEqual(result.total_r, sum(t.net_r for t in trades))
        self.assertEqual(result.units, sum(t.units for t in trades))


class ConfigTests(unittest.TestCase):
    def test_the_exit_channel_must_be_shorter_than_the_entry_channel(self):
        with self.assertRaises(ValueError):
            TurtleConfig(entry_window=20, exit_window=20)

    def test_rejects_nonsense_risk_and_windows(self):
        for kwargs in (
            {"risk_fraction": 0.0}, {"risk_fraction": 1.0}, {"max_units": 0},
            {"stop_atr": 0.0}, {"entry_window": 1}, {"directions": ()},
            {"round_trip_cost": -0.1},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    TurtleConfig(**kwargs)


class RollingExtremeTests(unittest.TestCase):
    """The sliding-window scan replaced a per-bar rescan, so it must agree."""

    def brute(self, values, window, largest):
        picker = max if largest else min
        return [
            math.nan if i < window else picker(values[i - window:i])
            for i in range(len(values))
        ]

    def test_matches_a_brute_force_scan_on_noisy_data(self):
        import random
        rng = random.Random(11)
        values = [rng.uniform(-50, 50) for _ in range(400)]
        for window in (2, 3, 10, 55):
            for largest in (True, False):
                with self.subTest(window=window, largest=largest):
                    got = rolling_extremes(values, window, largest)
                    want = self.brute(values, window, largest)
                    for a, b in zip(got, want):
                        if math.isnan(b):
                            self.assertTrue(math.isnan(a))
                        else:
                            self.assertAlmostEqual(a, b)

    def test_matches_on_a_monotonic_series_where_the_deque_drains(self):
        rising = [float(i) for i in range(50)]
        falling = [float(-i) for i in range(50)]
        for values in (rising, falling):
            for largest in (True, False):
                got = rolling_extremes(values, 5, largest)
                want = self.brute(values, 5, largest)
                for a, b in zip(got, want):
                    if math.isnan(b):
                        self.assertTrue(math.isnan(a))
                    else:
                        self.assertAlmostEqual(a, b)

    def test_a_window_longer_than_the_series_yields_nothing(self):
        self.assertTrue(all(math.isnan(v) for v in rolling_extremes([1.0, 2.0], 5, True)))


class FilterLeakageTests(unittest.TestCase):
    def test_the_skip_decision_cannot_read_bars_that_have_not_happened(self):
        rows = flat_then_breakout()
        price = rows[-1].close
        for cycle in range(8):
            for _ in range(10):
                price += -3.0 if cycle % 2 == 0 else 3.0
                rows.append(bar(len(rows), price, price + 1.5, price - 1.5, price))
        for cut in (100, 120, 140):
            with self.subTest(cut=cut):
                short, _ = run_turtle(rows[:cut], config=TurtleConfig())
                full, _ = run_turtle(rows, config=TurtleConfig())
                # Trades that closed before the cut must be identical either way.
                boundary = rows[cut - 1].timestamp
                settled = [t for t in short if t.exit_timestamp < boundary]
                self.assertTrue(settled)
                for trade in settled:
                    self.assertIn(trade, full)


if __name__ == "__main__":
    unittest.main()
