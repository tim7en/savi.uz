from __future__ import annotations

import csv
import os
import tempfile
import unittest
from datetime import date
from pathlib import Path

from savi_uz.config import load_dotenv
from savi_uz.macro_sources import Observation, ReferenceRate, SvenssonParameters
from savi_uz.macro_store import MacroStore


def _series_record(series_id: str = "DGS10", group: str = "market_implied_path") -> dict:
    return {
        "series_id": series_id,
        "source": "FRED",
        "group_name": group,
        "label": "Ten year",
        "title": "Market Yield on 10-Year",
        "units": "Percent",
        "frequency": "Daily",
        "seasonal_adjustment": "NSA",
        "observation_start": date(1962, 1, 2),
        "observation_end": date(2026, 8, 13),
        "last_updated": "2026-08-14 15:17:23-05",
        "release_id": 18,
        "release_name": "H.15",
        "vintage_policy": "latest",
        "notes": "",
        "fetched_at": "2026-08-16T00:00:00+00:00",
    }


class MacroStoreTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.store = MacroStore(Path(self._dir.name) / "macro.db")

    def tearDown(self):
        self.store.close()
        self._dir.cleanup()

    def test_schema_creates_every_expected_table(self):
        self.assertEqual(set(self.store.table_counts()), {
            "series", "observations", "first_release", "vintages", "release_dates",
            "reference_rates", "gsw_params", "gsw_rates", "fed_path", "fetch_log",
        })

    def test_series_upsert_is_idempotent(self):
        self.store.upsert_series(_series_record())
        self.store.upsert_series(_series_record())
        self.assertEqual(self.store.table_counts()["series"], 1)

    def test_reruns_update_values_rather_than_duplicating_rows(self):
        self.store.write_observations("DGS10", [Observation(date(2024, 1, 2), 4.0)])
        self.store.write_observations("DGS10", [Observation(date(2024, 1, 2), 4.25)])
        rows = self.store.connection.execute("SELECT obs_date, value FROM observations").fetchall()
        self.assertEqual(rows, [("2024-01-02", 4.25)])

    def test_first_release_keeps_the_release_date_separate_from_the_value(self):
        self.store.write_first_releases("PAYEMS", [
            Observation(date(2024, 1, 1), 157700.0, date(2024, 2, 2), date(2024, 3, 7)),
        ])
        row = self.store.connection.execute(
            "SELECT obs_date, value, release_date, superseded_on FROM first_release"
        ).fetchone()
        self.assertEqual(row, ("2024-01-01", 157700.0, "2024-02-02", "2024-03-07"))

    def test_vintages_without_a_vintage_date_are_dropped(self):
        written = self.store.write_vintages("FEDTARMD", [
            Observation(date(2028, 1, 1), 3.1, date(2025, 9, 17)),
            Observation(date(2028, 1, 1), 3.4, None),
        ])
        self.assertEqual(written, 1)

    def test_same_observation_can_hold_many_vintages(self):
        self.store.write_vintages("FEDTARMD", [
            Observation(date(2028, 1, 1), 3.1, date(2025, 9, 17)),
            Observation(date(2028, 1, 1), 3.4, date(2026, 6, 17)),
        ])
        self.assertEqual(self.store.table_counts()["vintages"], 2)

    def test_reference_rates_round_trip(self):
        self.store.write_reference_rates([
            ReferenceRate("EFFR", date(2026, 8, 13), 3.63, 3.60, None, None, 3.65, 98.0, 3.5, None, "R"),
        ])
        row = self.store.connection.execute(
            "SELECT rate_type, percent_rate, revision_indicator FROM reference_rates"
        ).fetchone()
        self.assertEqual(row, ("EFFR", 3.63, "R"))

    def test_curve_and_path_tables_round_trip(self):
        self.store.write_gsw_params([SvenssonParameters(date(2024, 1, 2), 4.5, -1.2, 0.8, 0.5, 1.5, 10.0)])
        self.store.write_gsw_rates([(date(2024, 1, 2), "SVENY01", 4.75)])
        self.store.write_fed_path([(date(2024, 1, 2), 12, 4.12)])
        counts = self.store.table_counts()
        self.assertEqual((counts["gsw_params"], counts["gsw_rates"], counts["fed_path"]), (1, 1, 1))

    def test_coverage_reports_span_and_first_release_count(self):
        self.store.upsert_series(_series_record())
        self.store.write_observations("DGS10", [
            Observation(date(2024, 1, 2), 4.0), Observation(date(2024, 1, 3), 4.1),
        ])
        self.store.write_first_releases("DGS10", [Observation(date(2024, 1, 2), 4.0, date(2024, 1, 3))])
        series_id, first, last, count, first_prints = self.store.coverage()[0]
        self.assertEqual((series_id, first, last, count, first_prints), ("DGS10", "2024-01-02", "2024-01-03", 2, 1))

    def test_empty_writes_are_a_no_op(self):
        self.assertEqual(self.store.write_observations("DGS10", []), 0)

    def test_csv_export_writes_one_file_per_table_with_headers(self):
        self.store.upsert_series(_series_record())
        self.store.write_observations("DGS10", [Observation(date(2024, 1, 2), 4.0)])
        with tempfile.TemporaryDirectory() as outdir:
            written = self.store.export_csv(outdir)
            self.assertEqual(written["observations"], 1)
            exported = Path(outdir) / "observations.csv"
            with exported.open(encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ["series_id", "obs_date", "value"])
            self.assertEqual(rows[1], ["DGS10", "2024-01-02", "4.0"])

    def test_store_works_as_a_context_manager(self):
        with tempfile.TemporaryDirectory() as tmp:
            with MacroStore(Path(tmp) / "nested" / "macro.db") as store:
                store.write_observations("X", [Observation(date(2024, 1, 1), 1.0)])
            self.assertTrue((Path(tmp) / "nested" / "macro.db").exists())


class DotenvTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.path = Path(self._dir.name) / ".env"
        self._saved = {k: os.environ.get(k) for k in ("SAVI_TEST_A", "SAVI_TEST_B", "SAVI_TEST_C")}

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._dir.cleanup()

    def test_quotes_comments_and_export_prefixes_are_handled(self):
        self.path.write_text(
            '# a comment\n'
            'SAVI_TEST_A = "quoted-value"\n'
            "export SAVI_TEST_B='single'\n"
            "\n"
            "not_a_pair\n",
            encoding="utf-8",
        )
        loaded = load_dotenv(self.path)
        self.assertEqual(loaded["SAVI_TEST_A"], "quoted-value")
        self.assertEqual(loaded["SAVI_TEST_B"], "single")
        self.assertEqual(os.environ["SAVI_TEST_A"], "quoted-value")

    def test_existing_environment_wins_unless_overridden(self):
        os.environ["SAVI_TEST_C"] = "from-shell"
        self.path.write_text("SAVI_TEST_C=from-file\n", encoding="utf-8")
        load_dotenv(self.path)
        self.assertEqual(os.environ["SAVI_TEST_C"], "from-shell")
        load_dotenv(self.path, override=True)
        self.assertEqual(os.environ["SAVI_TEST_C"], "from-file")

    def test_missing_file_is_not_an_error(self):
        self.assertEqual(load_dotenv(Path(self._dir.name) / "nope.env"), {})


if __name__ == "__main__":
    unittest.main()
