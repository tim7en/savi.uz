from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from savi_uz.equity_sources import IndexBar, ReportedEarnings, SecFact, ShillerRow
from savi_uz.equity_store import SHILLER_FIELDS, EquityStore


def _fact(cik: int, concept: str, frame: str, value: float, unit: str = "USD") -> SecFact:
    return SecFact(
        cik=cik, entity_name=f"COMPANY {cik}", concept=concept, frame=frame, unit=unit,
        period_start=date(2023, 1, 1), period_end=date(2023, 3, 31), value=value,
        accession="0001104659-24-037408", location="US-IL",
    )


class EquityStoreTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.store = EquityStore(Path(self._dir.name) / "equity.db")

    def tearDown(self):
        self.store.close()
        self._dir.cleanup()

    def test_shiller_columns_match_the_catalogue(self):
        columns = {
            row[1]
            for row in self.store.connection.execute("PRAGMA table_info(shiller_monthly)")
        }
        self.assertTrue(set(SHILLER_FIELDS).issubset(columns))
        self.assertIn("cape", columns)
        self.assertIn("sp500_price", columns)

    def test_shiller_rows_with_missing_earnings_store_nulls(self):
        """Recent months carry a price but no earnings yet; that must not drop the row."""
        rows = [
            ShillerRow(date(2024, 8, 1), {"sp500_price": 5500.0, "earnings": 210.0, "cape": 35.1}),
            ShillerRow(date(2024, 9, 1), {"sp500_price": 5700.0}),
        ]
        self.assertEqual(self.store.write_shiller(rows, "https://example/ie.xls"), 2)
        stored = self.store.connection.execute(
            "SELECT obs_date, sp500_price, earnings, cape, source_url FROM shiller_monthly ORDER BY obs_date"
        ).fetchall()
        self.assertEqual(stored[0], ("2024-08-01", 5500.0, 210.0, 35.1, "https://example/ie.xls"))
        self.assertEqual(stored[1], ("2024-09-01", 5700.0, None, None, "https://example/ie.xls"))

    def test_shiller_earnings_gap_reports_the_lag(self):
        self.store.write_shiller(
            [
                ShillerRow(date(2024, 6, 1), {"sp500_price": 5400.0, "earnings": 208.0}),
                ShillerRow(date(2024, 9, 1), {"sp500_price": 5700.0}),
            ],
            "u",
        )
        self.assertEqual(self.store.shiller_earnings_gap(), ("2024-09-01", "2024-06-01"))

    def test_rerunning_shiller_updates_in_place(self):
        self.store.write_shiller([ShillerRow(date(2024, 9, 1), {"sp500_price": 1.0})], "u")
        self.store.write_shiller([ShillerRow(date(2024, 9, 1), {"sp500_price": 2.0})], "u")
        self.assertEqual(
            self.store.connection.execute(
                "SELECT COUNT(*), MAX(sp500_price) FROM shiller_monthly"
            ).fetchone(),
            (1, 2.0),
        )

    def test_sec_facts_are_keyed_by_company_concept_frame_and_unit(self):
        self.store.write_sec_facts(
            [
                _fact(1750, "EarningsPerShareDiluted", "CY2023Q1", 0.62, "USD/shares"),
                _fact(320193, "EarningsPerShareDiluted", "CY2023Q1", 1.52, "USD/shares"),
                _fact(1750, "EarningsPerShareDiluted", "CY2023Q2", 0.71, "USD/shares"),
                _fact(1750, "NetIncomeLoss", "CY2023Q1", 21_000_000.0),
            ]
        )
        self.assertEqual(
            self.store.connection.execute("SELECT COUNT(*) FROM sec_facts").fetchone()[0], 4
        )

    def test_restating_the_same_fact_overwrites_rather_than_duplicating(self):
        """A later filing revises a value for the same company, concept and frame."""
        self.store.write_sec_facts([_fact(1750, "NetIncomeLoss", "CY2023Q1", 1.0)])
        self.store.write_sec_facts([_fact(1750, "NetIncomeLoss", "CY2023Q1", 2.0)])
        self.assertEqual(
            self.store.connection.execute("SELECT COUNT(*), MAX(value) FROM sec_facts").fetchone(),
            (1, 2.0),
        )

    def test_the_same_concept_in_two_units_is_not_a_collision(self):
        self.store.write_sec_facts(
            [
                _fact(1750, "EarningsPerShareDiluted", "CY2023Q1", 0.62, "USD/shares"),
                _fact(1750, "EarningsPerShareDiluted", "CY2023Q1", 0.60, "EUR/shares"),
            ]
        )
        self.assertEqual(
            self.store.connection.execute("SELECT COUNT(*) FROM sec_facts").fetchone()[0], 2
        )

    def test_frame_inventory_records_how_thin_each_quarter_was(self):
        self.store.write_sec_frame("EarningsPerShareDiluted", "CY2009Q1", "USD/shares", 475, "t")
        self.store.write_sec_frame("EarningsPerShareDiluted", "CY2023Q1", "USD/shares", 4994, "t")
        counts = self.store.connection.execute(
            "SELECT frame, facts FROM sec_frames ORDER BY frame"
        ).fetchall()
        self.assertEqual(counts, [("CY2009Q1", 475), ("CY2023Q1", 4994)])

    def test_index_prices_round_trip(self):
        self.store.write_index_prices(
            [
                IndexBar("^GSPC", date(2024, 1, 2), 4742.83, 3.9e9),
                IndexBar("^GSPC", date(2024, 1, 3), 4704.81, None),
            ]
        )
        stored = self.store.connection.execute(
            "SELECT obs_date, close, volume FROM index_prices ORDER BY obs_date"
        ).fetchall()
        self.assertEqual(stored[0], ("2024-01-02", 4742.83, 3.9e9))
        self.assertIsNone(stored[1][2])

    def test_analyst_earnings_round_trip(self):
        self.store.write_analyst_earnings(
            [
                ReportedEarnings("AAPL", date(2024, 6, 30), date(2024, 8, 1), 1.4, 1.35, 0.05, 3.7),
                ReportedEarnings("AAPL", date(2024, 3, 31), None, 1.53, None, None, None),
            ]
        )
        stored = self.store.connection.execute(
            "SELECT fiscal_ending, reported_eps, estimated_eps FROM analyst_earnings ORDER BY fiscal_ending"
        ).fetchall()
        self.assertEqual(stored, [("2024-03-31", 1.53, None), ("2024-06-30", 1.4, 1.35)])

    def test_coverage_lists_every_source_even_when_empty(self):
        keys = [row[0] for row in self.store.coverage()]
        self.assertEqual(keys, ["shiller", "index", "sec", "estimates"])
        self.assertTrue(all(row[3] == 0 for row in self.store.coverage()))

    def test_csv_export_writes_every_table(self):
        self.store.write_index_prices([IndexBar("^GSPC", date(2024, 1, 2), 4742.83, None)])
        outdir = Path(self._dir.name) / "csv"
        written = self.store.export_csv(outdir)
        self.assertEqual(written["index_prices"], 1)
        self.assertTrue((outdir / "shiller_monthly.csv").is_file())

    def test_empty_write_is_a_no_op(self):
        self.assertEqual(self.store.write_sec_facts([]), 0)
        self.assertEqual(self.store.write_index_prices([]), 0)


if __name__ == "__main__":
    unittest.main()
