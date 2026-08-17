from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from savi_uz.intraday_store import IntradayStore
from savi_uz.tiingo_sources import Bar, SymbolMeta


def _bar(ticker: str, ts: str, close: float, frequency: str = "1hour") -> Bar:
    return Bar(ticker, frequency, ts, 1.0, 2.0, 0.5, close, 100.0)


class IntradayStoreTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.store = IntradayStore(Path(self._dir.name) / "bars.db")

    def tearDown(self):
        self.store.close()
        self._dir.cleanup()

    def test_bars_round_trip(self):
        written = self.store.write_bars(
            [_bar("SPY", "2024-01-02T15:00:00.000Z", 470.1),
             _bar("SPY", "2024-01-02T16:00:00.000Z", 471.2)]
        )
        self.assertEqual(written, 2)
        rows = self.store.connection.execute(
            "SELECT ticker, ts, close FROM bars ORDER BY ts"
        ).fetchall()
        self.assertEqual(rows[0], ("SPY", "2024-01-02T15:00:00.000Z", 470.1))

    def test_rewriting_a_bar_updates_in_place(self):
        self.store.write_bars([_bar("SPY", "2024-01-02T15:00:00.000Z", 1.0)])
        self.store.write_bars([_bar("SPY", "2024-01-02T15:00:00.000Z", 2.0)])
        self.assertEqual(
            self.store.connection.execute("SELECT COUNT(*), MAX(close) FROM bars").fetchone(),
            (1, 2.0),
        )

    def test_the_same_timestamp_at_two_frequencies_is_not_a_collision(self):
        self.store.write_bars(
            [_bar("SPY", "2024-01-02T15:00:00.000Z", 1.0, "1hour"),
             _bar("SPY", "2024-01-02T15:00:00.000Z", 2.0, "daily")]
        )
        self.assertEqual(
            self.store.connection.execute("SELECT COUNT(*) FROM bars").fetchone()[0], 2
        )

    def test_symbol_metadata_records_intraday_availability(self):
        meta = SymbolMeta("TCEHY", "Tencent", "PINK", date(2008, 11, 3), date(2026, 8, 14), "adr")
        self.store.upsert_symbol(meta, "China internet", "t")
        row = self.store.connection.execute(
            "SELECT exchange, has_intraday, themes, history_start FROM symbols"
        ).fetchone()
        self.assertEqual(row, ("PINK", 0, "China internet", "2008-11-03"))

    def test_completed_windows_drive_the_resume(self):
        self.store.mark_window("SPY", "1hour", date(2024, 3, 1), date(2024, 3, 31),
                               [_bar("SPY", "2024-03-01T15:00:00.000Z", 1.0)], False, "t")
        self.store.mark_window("SPY", "1hour", date(2025, 3, 1), date(2025, 3, 31),
                               [_bar("SPY", "2025-03-01T15:00:00.000Z", 1.0)], False, "t")
        self.assertEqual(
            self.store.completed_windows("1hour"), {("SPY", 2024, 3), ("SPY", 2025, 3)}
        )
        self.assertEqual(self.store.completed_windows("daily"), set())

    def test_an_empty_window_is_still_marked_so_it_is_not_refetched(self):
        """A month with no bars is a fact about the source, not a failure to retry."""
        self.store.mark_window("RKLB", "1hour", date(2017, 1, 1), date(2017, 1, 31), [], False, "t")
        self.assertIn(("RKLB", 2017, 1), self.store.completed_windows("1hour"))
        row = self.store.connection.execute(
            "SELECT rows, first_ts FROM windows WHERE ticker = 'RKLB'"
        ).fetchone()
        self.assertEqual(row, (0, None))

    def test_truncated_windows_are_reported(self):
        self.store.mark_window("SPY", "1hour", date(2020, 3, 1), date(2020, 3, 31),
                               [_bar("SPY", "2020-03-01T15:00:00.000Z", 1.0)], True, "t")
        self.store.mark_window("SPY", "1hour", date(2021, 3, 1), date(2021, 3, 31),
                               [_bar("SPY", "2021-03-01T15:00:00.000Z", 1.0)], False, "t")
        truncated = self.store.truncated_windows()
        self.assertEqual(len(truncated), 1)
        self.assertEqual(truncated[0][0], "SPY")
        self.assertEqual(truncated[0][2], "2020-03")

    def test_window_bounds_come_from_the_bars(self):
        self.store.mark_window(
            "SPY", "1hour", date(2024, 1, 1), date(2024, 1, 31),
            [_bar("SPY", "2024-01-31T15:00:00.000Z", 1.0), _bar("SPY", "2024-01-02T15:00:00.000Z", 1.0)],
            False, "t",
        )
        row = self.store.connection.execute(
            "SELECT first_ts, last_ts, rows FROM windows"
        ).fetchone()
        self.assertEqual(row, ("2024-01-02T15:00:00.000Z", "2024-01-31T15:00:00.000Z", 2))

    def test_known_symbols_reports_intraday_capability(self):
        self.store.upsert_symbol(
            SymbolMeta("SPY", "SPDR", "NYSE ARCA", date(1993, 1, 29), None), "Broad indices", "t"
        )
        self.assertEqual(self.store.known_symbols()["SPY"], ("NYSE ARCA", 1))

    def test_coverage_groups_by_ticker_and_frequency(self):
        self.store.write_bars(
            [_bar("SPY", "2024-01-02T15:00:00.000Z", 1.0),
             _bar("SPY", "2024-01-03T15:00:00.000Z", 2.0),
             _bar("GLD", "2024-01-02T15:00:00.000Z", 3.0, "daily")]
        )
        coverage = {(row[0], row[1]): row[4] for row in self.store.coverage()}
        self.assertEqual(coverage[("SPY", "1hour")], 2)
        self.assertEqual(coverage[("GLD", "daily")], 1)

    def test_empty_write_is_a_no_op(self):
        self.assertEqual(self.store.write_bars([]), 0)

    def test_a_multi_month_chunk_is_recorded_per_month(self):
        """Per-month state is what lets a sub-year chunk size resume correctly."""
        bars = [
            _bar("SPY", "2020-01-15T15:00:00.000Z", 1.0),
            _bar("SPY", "2020-02-15T15:00:00.000Z", 2.0),
            _bar("SPY", "2020-03-15T15:00:00.000Z", 3.0),
        ]
        self.store.mark_window("SPY", "5min", date(2020, 1, 1), date(2020, 3, 31), bars, False, "t")
        self.assertEqual(
            self.store.completed_windows("5min"),
            {("SPY", 2020, 1), ("SPY", 2020, 2), ("SPY", 2020, 3)},
        )
        rows = self.store.connection.execute(
            "SELECT month, rows FROM windows ORDER BY month"
        ).fetchall()
        self.assertEqual(rows, [(1, 1), (2, 1), (3, 1)])

    def test_a_month_inside_a_chunk_with_no_bars_is_still_marked(self):
        """Otherwise a quiet month is re-requested on every resumed run."""
        self.store.mark_window(
            "GEV", "5min", date(2020, 1, 1), date(2020, 3, 31),
            [_bar("GEV", "2020-03-02T15:00:00.000Z", 1.0)], False, "t",
        )
        self.assertEqual(
            self.store.completed_windows("5min"),
            {("GEV", 2020, 1), ("GEV", 2020, 2), ("GEV", 2020, 3)},
        )

    def test_a_chunk_spanning_a_year_boundary_marks_both_years(self):
        self.store.mark_window("SPY", "5min", date(2020, 11, 1), date(2021, 1, 31), [], False, "t")
        self.assertEqual(
            self.store.completed_windows("5min"),
            {("SPY", 2020, 11), ("SPY", 2020, 12), ("SPY", 2021, 1)},
        )

    def test_an_unparseable_timestamp_does_not_crash_the_run(self):
        self.store.mark_window("SPY", "1hour", date(2024, 1, 1), date(2024, 1, 31),
                               [_bar("SPY", "bad-stamp", 1.0)], False, "t")
        self.assertIn(("SPY", 2024, 1), self.store.completed_windows("1hour"))

    def test_csv_export_writes_every_table(self):
        self.store.write_bars([_bar("SPY", "2024-01-02T15:00:00.000Z", 1.0)])
        outdir = Path(self._dir.name) / "csv"
        written = self.store.export_csv(outdir)
        self.assertEqual(written["bars"], 1)
        self.assertTrue((outdir / "symbols.csv").is_file())


if __name__ == "__main__":
    unittest.main()


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.store = IntradayStore(Path(self._dir.name) / "bars.db")

    def tearDown(self):
        self.store.close()
        self._dir.cleanup()

    def test_a_close_outside_the_bar_range_is_counted_not_repaired(self):
        """IEX resampling puts the close a few cents outside high/low ~0.2% of the time."""
        good = Bar("SPY", "1hour", "2024-01-02T15:00:00.000Z", 1.0, 2.0, 0.5, 1.5, 10.0)
        bad = Bar("SPY", "1hour", "2024-01-02T16:00:00.000Z", 1.0, 2.0, 1.5, 2.5, 10.0)
        self.store.write_bars([good, bad])

        self.assertEqual(self.store.ohlc_violations(), [("SPY", "1hour", 1)])
        # The row is still there, exactly as published.
        self.assertEqual(
            self.store.connection.execute(
                "SELECT close FROM bars WHERE ts = '2024-01-02T16:00:00.000Z'"
            ).fetchone()[0],
            2.5,
        )

    def test_an_open_outside_the_range_also_counts(self):
        self.store.write_bars(
            [Bar("X", "1hour", "2024-01-02T15:00:00.000Z", 5.0, 2.0, 0.5, 1.5, 10.0)]
        )
        self.assertEqual(self.store.ohlc_violations()[0][2], 1)

    def test_bars_with_null_prices_are_not_counted_as_violations(self):
        self.store.write_bars([Bar("X", "1hour", "2024-01-02T15:00:00.000Z", None, None, None, None, None)])
        self.assertEqual(self.store.ohlc_violations(), [])

    def test_missing_volume_is_reported_per_series(self):
        self.store.write_bars([
            Bar("SPY", "1hour", "2024-01-02T15:00:00.000Z", 1.0, 2.0, 0.5, 1.5, 100.0),
            Bar("SPY", "1hour", "2024-01-02T16:00:00.000Z", 1.0, 2.0, 0.5, 1.5, None),
            Bar("SPY", "1hour", "2024-01-02T17:00:00.000Z", 1.0, 2.0, 0.5, 1.5, 0.0),
        ])
        self.assertEqual(self.store.missing_volume(), [("SPY", "1hour", 2, 3)])

    def test_splits_are_listed_and_non_splits_ignored(self):
        from savi_uz.tiingo_sources import Adjustment
        self.store.write_adjustments([
            Adjustment("NVDA", date(2024, 6, 7), 1208.88, 120.68, 1.0, 0.0),
            Adjustment("NVDA", date(2024, 6, 10), 121.79, 121.58, 10.0, 0.0),
            Adjustment("NVDA", date(2024, 6, 11), 120.91, 120.71, 1.0, 0.01),
        ])
        self.assertEqual(self.store.splits(), [("NVDA", "2024-06-10", 10.0)])
        self.assertEqual(self.store.splits("SPY"), [])

    def test_adjustments_update_in_place(self):
        from savi_uz.tiingo_sources import Adjustment
        self.store.write_adjustments([Adjustment("SPY", date(2024, 6, 7), 1.0, 1.0, 1.0, 0.0)])
        self.store.write_adjustments([Adjustment("SPY", date(2024, 6, 7), 2.0, 2.0, 1.0, 0.0)])
        self.assertEqual(
            self.store.connection.execute(
                "SELECT COUNT(*), MAX(close) FROM corporate_actions"
            ).fetchone(),
            (1, 2.0),
        )
