from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from savi_uz.equity_sources import IndexBar, ReportedEarnings, SecFact, ShillerRow
from savi_uz.equity_store import SHILLER_FIELDS, EquityStore
from savi_uz.factset_sources import KeyMetrics


def _metrics(day: date, values: dict, missing: tuple = (), page_text: str = "Key Metrics ...") -> KeyMetrics:
    return KeyMetrics(
        report_date=day,
        source_url="https://example/EarningsInsight.pdf",
        document_date=day,
        values=values,
        missing=missing,
        page_text=page_text,
    )


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
        self.assertEqual(keys, ["shiller", "index", "sec", "estimates", "factset"])
        self.assertTrue(all(row[3] == 0 for row in self.store.coverage()))

    def test_factset_report_round_trip_keeps_values_and_page_text(self):
        metrics = _metrics(
            date(2026, 8, 7),
            {"quarter": "Q2 2026", "forward_12m_pe": 20.0, "blended_earnings_growth": 50.4},
        )
        self.assertEqual(self.store.write_factset_report(metrics, "t"), 1)
        stored = self.store.connection.execute(
            "SELECT quarter, forward_12m_pe, blended_earnings_growth, estimated_earnings_growth, "
            "missing_core_fields, page_text FROM factset_reports"
        ).fetchone()
        self.assertEqual(stored[:4], ("Q2 2026", 20.0, 50.4, None))
        self.assertEqual(stored[4], "")
        self.assertEqual(stored[5], "Key Metrics ...")

    def test_missing_fields_are_recorded_as_a_list(self):
        metrics = _metrics(date(2026, 6, 18), {"quarter": "Q2 2026"}, missing=("forward_12m_pe",))
        self.store.write_factset_report(metrics, "t")
        stored = self.store.connection.execute(
            "SELECT missing_fields, missing_core_fields, forward_12m_pe FROM factset_reports"
        ).fetchone()
        self.assertEqual(stored, ("forward_12m_pe", "forward_12m_pe", None))

    def test_rewriting_an_edition_updates_in_place(self):
        """--reparse rewrites every stored edition and must not duplicate them."""
        self.store.write_factset_report(_metrics(date(2026, 8, 7), {"forward_12m_pe": 19.0}), "t")
        self.store.write_factset_report(_metrics(date(2026, 8, 7), {"forward_12m_pe": 20.0}), "t")
        self.assertEqual(
            self.store.connection.execute(
                "SELECT COUNT(*), MAX(forward_12m_pe) FROM factset_reports"
            ).fetchone(),
            (1, 20.0),
        )

    def test_page_texts_are_retrievable_for_offline_reparsing(self):
        self.store.write_factset_report(_metrics(date(2026, 8, 7), {}), "t")
        self.store.write_factset_report(_metrics(date(2026, 7, 31), {}, page_text=""), "t")
        texts = self.store.factset_page_texts()
        self.assertEqual([row[0] for row in texts], ["2026-08-07"])

    def test_field_coverage_counts_editions_carrying_each_field(self):
        self.store.write_factset_report(
            _metrics(date(2026, 8, 7), {"forward_12m_pe": 20.0, "blended_earnings_growth": 50.4}), "t"
        )
        self.store.write_factset_report(
            _metrics(date(2026, 6, 26), {"forward_12m_pe": 20.1, "estimated_earnings_growth": 23.1}), "t"
        )
        coverage = dict(self.store.factset_field_coverage())
        self.assertEqual(coverage["forward_12m_pe"], 2)
        self.assertEqual(coverage["blended_earnings_growth"], 1)
        self.assertEqual(coverage["estimated_earnings_growth"], 1)

    def test_field_coverage_is_empty_before_anything_is_stored(self):
        self.assertEqual(self.store.factset_field_coverage(), [])

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
