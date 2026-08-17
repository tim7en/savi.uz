from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path

from savi_uz.cftc_catalog import REPORTS_BY_KEY
from savi_uz.cftc_sources import ReportChunk
from savi_uz.cftc_store import CftcStore

#: A real catalogue entry -- the store keys its export and coverage off the
#: catalogue -- narrowed to a handful of columns so the fixtures stay readable.
SPEC = replace(REPORTS_BY_KEY["legacy_futures"], columns=5, notes="synthetic")

COLUMNS = (
    "contract_code",
    "report_date",
    "market_and_exchange_names",
    "as_of_date_in_form_yyyy_mm_dd",
    "cftc_contract_market_code",
    "cftc_commodity_code",
    "open_interest_all",
)


def _row(code: str, day: str, name: str, open_interest: int) -> tuple:
    return (code, day, name, day, code, "001", open_interest)


def _chunk(rows: list[tuple], columns: tuple[str, ...] = COLUMNS) -> ReportChunk:
    return ReportChunk(
        spec=SPEC,
        source="probe.zip:annual.txt",
        columns=columns,
        rows=rows,
        first_date=date(2024, 1, 2),
        last_date=date(2024, 12, 31),
    )


class CftcStoreTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.store = CftcStore(Path(self._dir.name) / "cot.db")

    def tearDown(self):
        self.store.close()
        self._dir.cleanup()

    def test_table_is_built_from_the_files_own_header(self):
        self.store.ensure_table(SPEC.table, COLUMNS)
        self.assertEqual(self.store.existing_columns(SPEC.table), COLUMNS)

    def test_code_columns_are_text_so_leading_zeros_survive_the_roundtrip(self):
        self.store.write_chunk(_chunk([_row("001602", "2024-12-31", "WHEAT-SRW", 460417)]))
        stored = self.store.connection.execute(
            "SELECT contract_code, cftc_commodity_code, open_interest_all FROM cot_legacy_futures"
        ).fetchone()
        self.assertEqual(stored, ("001602", "001", 460417))

    def test_rewriting_the_same_week_updates_in_place(self):
        self.store.write_chunk(_chunk([_row("001602", "2024-12-31", "WHEAT-SRW", 1)]))
        self.store.write_chunk(_chunk([_row("001602", "2024-12-31", "WHEAT-SRW", 2)]))
        rows = self.store.connection.execute(
            "SELECT COUNT(*), MAX(open_interest_all) FROM cot_legacy_futures"
        ).fetchone()
        self.assertEqual(rows, (1, 2))

    def test_a_new_upstream_column_widens_the_table_rather_than_failing(self):
        self.store.write_chunk(_chunk([_row("001602", "2024-12-31", "WHEAT-SRW", 1)]))
        widened = COLUMNS + ("brand_new_cftc_column",)
        added = self.store.ensure_table(SPEC.table, widened)

        self.assertEqual(added, ("brand_new_cftc_column",))
        self.store.write_rows(
            SPEC.table, widened, [_row("001602", "2025-01-07", "WHEAT-SRW", 3) + (42,)]
        )
        stored = self.store.connection.execute(
            "SELECT brand_new_cftc_column FROM cot_legacy_futures ORDER BY report_date"
        ).fetchall()
        self.assertEqual(stored, [(None,), (42,)])

    def test_duplicate_normalized_columns_are_rejected(self):
        with self.assertRaises(ValueError):
            self.store.ensure_table(SPEC.table, COLUMNS + ("open_interest_all",))

    def test_unsafe_identifiers_are_refused(self):
        with self.assertRaises(ValueError):
            self.store.ensure_table("cot_legacy_futures; DROP TABLE cot_legacy_futures", COLUMNS)

    def test_empty_write_is_a_no_op(self):
        self.assertEqual(self.store.write_rows(SPEC.table, COLUMNS, []), 0)
        self.assertEqual(self.store.existing_columns(SPEC.table), ())

    def test_contract_directory_takes_the_most_recent_market_name(self):
        self.store.write_chunk(
            _chunk(
                [
                    _row("001602", "2024-06-25", "WHEAT-SRW - OLD NAME", 1),
                    _row("001602", "2024-12-31", "WHEAT-SRW - CHICAGO BOARD OF TRADE", 2),
                    _row("088691", "2024-12-31", "GOLD - COMMODITY EXCHANGE INC.", 3),
                ]
            )
        )
        self.assertEqual(self.store.rebuild_contracts(SPEC), 2)
        wheat = self.store.connection.execute(
            "SELECT market_name, commodity_code, first_date, last_date, observations "
            "FROM cot_contracts WHERE contract_code = '001602'"
        ).fetchone()
        self.assertEqual(
            wheat, ("WHEAT-SRW - CHICAGO BOARD OF TRADE", "001", "2024-06-25", "2024-12-31", 2)
        )

    def test_rebuilding_the_directory_drops_contracts_that_are_gone(self):
        self.store.write_chunk(_chunk([_row("001602", "2024-12-31", "WHEAT-SRW", 1)]))
        self.store.rebuild_contracts(SPEC)
        self.store.connection.execute("DELETE FROM cot_legacy_futures")
        self.assertEqual(self.store.rebuild_contracts(SPEC), 0)

    def test_report_summary_records_the_span_actually_stored(self):
        self.store.write_chunk(
            _chunk(
                [
                    _row("001602", "2024-01-02", "WHEAT-SRW", 1),
                    _row("001602", "2024-12-31", "WHEAT-SRW", 2),
                ]
            )
        )
        self.store.upsert_report(SPEC, "2026-08-17T00:00:00+00:00")
        summary = self.store.connection.execute(
            "SELECT table_name, first_date, last_date, rows, notes FROM cot_reports"
        ).fetchone()
        self.assertEqual(summary, ("cot_legacy_futures", "2024-01-02", "2024-12-31", 2, "synthetic"))

    def test_summarising_a_report_that_was_never_downloaded_is_a_no_op(self):
        self.store.upsert_report(SPEC, "2026-08-17T00:00:00+00:00")
        self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM cot_reports").fetchone(), (0,))
        self.assertEqual(self.store.rebuild_contracts(SPEC), 0)

    def test_coverage_reports_span_and_distinct_contracts(self):
        self.store.write_chunk(
            _chunk(
                [
                    _row("001602", "2024-01-02", "WHEAT-SRW", 1),
                    _row("001602", "2024-12-31", "WHEAT-SRW", 2),
                    _row("088691", "2024-12-31", "GOLD", 3),
                ]
            )
        )
        self.assertEqual(
            self.store.coverage(), [("legacy_futures", "2024-01-02", "2024-12-31", 3, 2)]
        )

    def test_coverage_skips_reports_with_no_table_yet(self):
        self.assertEqual(self.store.coverage(), [])

    def test_fetch_log_keeps_every_attempt(self):
        self.store.log("run1", "2026-08-17T00:00:00+00:00", "CFTC", "deacot2024.zip", 16764, "ok")
        self.store.log("run1", "2026-08-17T00:00:01+00:00", "CFTC", "deacot2027.zip", 0, "error", "HTTP 404")
        statuses = self.store.connection.execute(
            "SELECT status, rows FROM fetch_log ORDER BY logged_at"
        ).fetchall()
        self.assertEqual(statuses, [("ok", 16764), ("error", 0)])

    def test_csv_export_writes_a_header_and_every_row(self):
        self.store.write_chunk(
            _chunk(
                [
                    _row("001602", "2024-12-31", "WHEAT-SRW", 1),
                    _row("088691", "2024-12-31", "GOLD", 2),
                ]
            )
        )
        outdir = Path(self._dir.name) / "csv"
        written = self.store.export_csv(outdir)

        self.assertEqual(written["cot_legacy_futures"], 2)
        with (outdir / "cot_legacy_futures.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["contract_code"] for row in rows], ["001602", "088691"])
        self.assertEqual(rows[0]["cftc_commodity_code"], "001")


if __name__ == "__main__":
    unittest.main()
