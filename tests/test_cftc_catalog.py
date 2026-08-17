from __future__ import annotations

import unittest

from savi_uz.cftc_catalog import (
    REPORT_KEYS,
    REPORTS,
    REPORTS_BY_KEY,
    TEXT_COLUMNS,
    archive_files,
    reports_for_keys,
)


class CatalogShapeTests(unittest.TestCase):
    def test_report_keys_and_tables_are_unique(self):
        self.assertEqual(len(REPORT_KEYS), len(set(REPORT_KEYS)))
        tables = [spec.table for spec in REPORTS]
        self.assertEqual(len(tables), len(set(tables)))

    def test_every_report_declares_its_key_columns_as_text(self):
        for spec in REPORTS:
            with self.subTest(spec.key):
                self.assertIn(spec.date_column, TEXT_COLUMNS)
                self.assertIn(spec.contract_column, TEXT_COLUMNS)

    def test_bundle_years_are_consistent(self):
        for spec in REPORTS:
            with self.subTest(spec.key):
                if spec.bundle_template is None:
                    self.assertIsNone(spec.bundle_first_year)
                    continue
                self.assertEqual(spec.first_year, spec.bundle_first_year)
                self.assertLess(spec.bundle_last_year, spec.first_annual_year)

    def test_urls_are_https(self):
        for spec in REPORTS:
            with self.subTest(spec.key):
                self.assertTrue(spec.annual_url(2024).startswith("https://"))
                bundle = spec.bundle_url()
                if bundle is not None:
                    self.assertTrue(bundle.startswith("https://"))

    def test_reports_for_keys_filters_and_validates(self):
        self.assertEqual(reports_for_keys(None), REPORTS)
        self.assertEqual(
            [spec.key for spec in reports_for_keys(("tff_futures",))], ["tff_futures"]
        )
        with self.assertRaises(ValueError):
            reports_for_keys(("no_such_report",))


class ArchivePlanTests(unittest.TestCase):
    def test_legacy_futures_uses_annual_files_only(self):
        files = archive_files(REPORTS_BY_KEY["legacy_futures"], 2000, 2002)
        self.assertEqual(
            [f.filename for f in files], ["deacot2000.zip", "deacot2001.zip", "deacot2002.zip"]
        )

    def test_bundle_covers_back_history_then_annual_takes_over(self):
        files = archive_files(REPORTS_BY_KEY["tff_futures"], 2006, 2018)
        self.assertEqual(
            [f.filename for f in files],
            ["fin_fut_txt_2006_2016.zip", "fut_fin_txt_2017.zip", "fut_fin_txt_2018.zip"],
        )

    def test_range_after_the_bundle_skips_it(self):
        files = archive_files(REPORTS_BY_KEY["disagg_futures"], 2020, 2021)
        self.assertEqual(
            [f.filename for f in files], ["fut_disagg_txt_2020.zip", "fut_disagg_txt_2021.zip"]
        )

    def test_years_before_the_archive_are_dropped_not_requested(self):
        """Asking for 2000 disaggregated data yields the 2006 bundle, not a 404."""
        files = archive_files(REPORTS_BY_KEY["disagg_futures"], 2000, 2010)
        self.assertEqual(files[0].filename, "fut_disagg_txt_hist_2006_2016.zip")
        self.assertEqual(len(files), 1)

    def test_range_entirely_before_the_archive_is_empty(self):
        self.assertEqual(archive_files(REPORTS_BY_KEY["disagg_futures"], 1990, 1999), ())

    def test_inverted_range_is_rejected(self):
        with self.assertRaises(ValueError):
            archive_files(REPORTS_BY_KEY["legacy_futures"], 2010, 2005)


if __name__ == "__main__":
    unittest.main()
