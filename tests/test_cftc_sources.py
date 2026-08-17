from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from datetime import date
from pathlib import Path

from savi_uz.cftc_catalog import ArchiveFile, ReportSpec
from savi_uz.cftc_sources import (
    CftcArchiveClient,
    CftcDownloadError,
    coerce_value,
    normalize_column,
    parse_report_date,
)

HEADER = (
    '"Market and Exchange Names","As of Date in Form YYYY-MM-DD",'
    '"CFTC Contract Market Code","CFTC Commodity Code",'
    '"Open Interest (All)","Noncommercial Positions-Long (All)"'
)

SPEC = ReportSpec(
    key="probe",
    table="cot_probe",
    label="Probe",
    date_column="as_of_date_in_form_yyyy_mm_dd",
    contract_column="cftc_contract_market_code",
    columns=6,
    annual_template="probe{year}.zip",
    first_annual_year=2000,
    first_year=2000,
)


def _zip_bytes(text: str, member: str = "annual.txt") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, text)
    return buffer.getvalue()


def _rows(*records: str) -> str:
    return "\r\n".join([HEADER, *records]) + "\r\n"


class NormalizeColumnTests(unittest.TestCase):
    def test_legacy_and_modern_spellings_of_the_same_field_agree(self):
        self.assertEqual(
            normalize_column("CFTC Contract Market Code"),
            normalize_column("CFTC_Contract_Market_Code"),
        )

    def test_parentheses_and_hyphens_become_underscores(self):
        self.assertEqual(
            normalize_column("Noncommercial Positions-Long (All)"),
            "noncommercial_positions_long_all",
        )

    def test_double_underscore_in_the_source_header_is_collapsed(self):
        self.assertEqual(normalize_column("Swap__Positions_Short_All"), "swap_positions_short_all")

    def test_percent_is_spelled_out_rather_than_dropped(self):
        self.assertEqual(
            normalize_column("Conc-Gross LE 4 TDR-Long (All)"), "conc_gross_le_4_tdr_long_all"
        )
        self.assertEqual(normalize_column("Pct of OI-Long (All)"), "pct_of_oi_long_all")
        self.assertEqual(normalize_column("% of OI (All)"), "pct_of_oi_all")

    def test_iso_date_header_normalizes_predictably(self):
        self.assertEqual(
            normalize_column("Report_Date_as_YYYY-MM-DD"), "report_date_as_yyyy_mm_dd"
        )


class ParseReportDateTests(unittest.TestCase):
    def test_iso_dates(self):
        self.assertEqual(parse_report_date("2024-12-31"), date(2024, 12, 31))
        self.assertEqual(parse_report_date("  2024-12-31  "), date(2024, 12, 31))

    def test_us_order_with_midnight_timestamp(self):
        """The 2006-2016 TFF bundles ship dates in this shape, not ISO."""
        self.assertEqual(parse_report_date("9/9/2014 12:00:00 AM"), date(2014, 9, 9))
        self.assertEqual(parse_report_date("1/10/2012 12:00:00 AM"), date(2012, 1, 10))
        self.assertEqual(parse_report_date("12/30/2008 12:00:00 AM"), date(2008, 12, 30))

    def test_day_and_month_are_not_transposed(self):
        """9/1 is 1 September; reading it as 9 January would shift a whole file."""
        self.assertEqual(parse_report_date("9/1/2014 12:00:00 AM"), date(2014, 9, 1))

    def test_unreadable_values_are_none(self):
        for text in (None, "", "   ", ".", "not a date", "13/45/2014 12:00:00 AM"):
            with self.subTest(text):
                self.assertIsNone(parse_report_date(text))


class CoerceValueTests(unittest.TestCase):
    def test_padded_integers_become_ints(self):
        self.assertEqual(coerce_value("open_interest_all", "  460417"), 460417)

    def test_comma_grouped_numbers_are_parsed(self):
        self.assertEqual(coerce_value("open_interest_all", " 1,234,567 "), 1234567)

    def test_decimals_become_floats(self):
        self.assertEqual(coerce_value("pct_of_oi_all", " 12.4 "), 12.4)

    def test_negative_change_columns_survive(self):
        self.assertEqual(coerce_value("change_in_open_interest_all", " -1234"), -1234)

    def test_missing_markers_become_none(self):
        self.assertIsNone(coerce_value("open_interest_all", "."))
        self.assertIsNone(coerce_value("open_interest_all", "   "))

    def test_code_columns_keep_their_leading_zeros(self):
        self.assertEqual(coerce_value("cftc_contract_market_code", " 001602 "), "001602")
        self.assertEqual(coerce_value("cftc_commodity_code", "001 "), "001")

    def test_unparseable_value_in_a_numeric_column_is_kept_as_text(self):
        self.assertEqual(coerce_value("open_interest_all", "n.a."), "n.a.")


class LoadArchiveTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.cache = Path(self._dir.name)
        self.client = CftcArchiveClient(cache_dir=self.cache)
        self.archive = ArchiveFile("https://www.cftc.gov/files/dea/history/probe.zip", 2024, 2024)

    def tearDown(self):
        self._dir.cleanup()

    def _seed(self, payload: bytes) -> None:
        (self.cache / "probe.zip").write_bytes(payload)

    def test_cached_archive_is_parsed_without_network_access(self):
        self._seed(
            _zip_bytes(
                _rows(
                    '"WHEAT-SRW - CBOT","2024-12-31","001602","001 ","  460417","  136258"',
                    '"GOLD - COMEX","2024-12-31","088691","088 ","  512345","  201122"',
                )
            )
        )
        chunk = self.client.load(SPEC, self.archive)

        self.assertEqual(len(chunk.rows), 2)
        self.assertEqual(chunk.columns[:2], ("contract_code", "report_date"))
        self.assertEqual(chunk.first_date, date(2024, 12, 31))
        self.assertEqual(chunk.last_date, date(2024, 12, 31))
        # Canonical key first, then the file's own columns in order.
        self.assertEqual(chunk.rows[0][:2], ("001602", "2024-12-31"))
        self.assertEqual(chunk.rows[0][2], "WHEAT-SRW - CBOT")
        self.assertEqual(chunk.rows[0][-2:], (460417, 136258))

    def test_rows_outside_the_requested_window_are_filtered(self):
        self._seed(
            _zip_bytes(
                _rows(
                    '"WHEAT-SRW - CBOT","2023-06-27","001602","001 ","1","2"',
                    '"WHEAT-SRW - CBOT","2024-06-25","001602","001 ","3","4"',
                    '"WHEAT-SRW - CBOT","2025-06-24","001602","001 ","5","6"',
                )
            )
        )
        chunk = self.client.load(SPEC, self.archive, start=date(2024, 1, 1), end=date(2024, 12, 31))
        self.assertEqual([row[1] for row in chunk.rows], ["2024-06-25"])

    def test_short_and_undated_rows_are_skipped(self):
        self._seed(
            _zip_bytes(
                _rows(
                    '"WHEAT-SRW - CBOT","2024-12-31","001602","001 ","1","2"',
                    '"TRUNCATED","2024-12-31","001602"',
                    '"NO DATE","","001602","001 ","1","2"',
                )
            )
        )
        chunk = self.client.load(SPEC, self.archive)
        self.assertEqual(len(chunk.rows), 1)
        self.assertEqual(chunk.unparsed, 2)
        self.assertEqual(chunk.filtered, 0)

    def test_a_file_nothing_could_be_read_from_is_an_error_not_an_empty_result(self):
        """Guards the failure mode the TFF date format caused: silent zero rows."""
        self._seed(
            _zip_bytes(
                _rows(
                    '"WHEAT-SRW - CBOT","31.12.2024","001602","001 ","1","2"',
                    '"GOLD - COMEX","31.12.2024","088691","088 ","3","4"',
                )
            )
        )
        with self.assertRaises(CftcDownloadError) as caught:
            self.client.load(SPEC, self.archive)
        self.assertIn("no usable rows", str(caught.exception))

    def test_a_file_filtered_down_to_nothing_is_allowed(self):
        """An archive genuinely outside the window is empty, not broken."""
        self._seed(_zip_bytes(_rows('"WHEAT-SRW - CBOT","2019-12-31","001602","001 ","1","2"')))
        chunk = self.client.load(SPEC, self.archive, start=date(2024, 1, 1))
        self.assertEqual(chunk.rows, [])
        self.assertEqual(chunk.filtered, 1)

    def test_us_order_dates_are_stored_as_iso(self):
        self._seed(
            _zip_bytes(_rows('"5-YEAR NOTE - CBOT","9/9/2014 12:00:00 AM","044601","044 ","1","2"'))
        )
        chunk = self.client.load(SPEC, self.archive)
        self.assertEqual(chunk.rows[0][1], "2014-09-09")
        self.assertEqual(chunk.first_date, date(2014, 9, 9))

    def test_unexpected_column_count_is_rejected(self):
        self._seed(_zip_bytes('"A","B"\r\n"1","2"\r\n'))
        with self.assertRaises(CftcDownloadError) as caught:
            self.client.load(SPEC, self.archive)
        self.assertIn("expected 6", str(caught.exception))

    def test_missing_date_column_is_rejected(self):
        header = HEADER.replace("As of Date in Form YYYY-MM-DD", "Some Other Column")
        self._seed(_zip_bytes(header + "\r\n"))
        with self.assertRaises(CftcDownloadError) as caught:
            self.client.load(SPEC, self.archive)
        self.assertIn("as_of_date_in_form_yyyy_mm_dd", str(caught.exception))

    def test_corrupt_cache_entry_reports_a_clear_error(self):
        self._seed(b"<html>404</html>")
        with self.assertRaises(CftcDownloadError) as caught:
            self.client.load(SPEC, self.archive)
        self.assertIn("ZIP", str(caught.exception))

    def test_archive_without_a_text_member_is_rejected(self):
        self._seed(_zip_bytes("nothing here", member="readme.pdf"))
        with self.assertRaises(CftcDownloadError):
            self.client.load(SPEC, self.archive)

    def test_non_https_url_is_refused(self):
        with self.assertRaises(CftcDownloadError):
            self.client.download(ArchiveFile("http://www.cftc.gov/probe.zip", 2024, 2024))

    def test_legacy_bytes_that_are_not_utf8_do_not_break_the_parse(self):
        record = '"CAF\xc9 - NYBOT","2024-12-31","001602","001 ","1","2"'
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr("annual.txt", _rows(record).encode("latin-1"))
        self._seed(payload.getvalue())
        chunk = self.client.load(SPEC, self.archive)
        self.assertEqual(chunk.rows[0][2], "CAF\xc9 - NYBOT")


if __name__ == "__main__":
    unittest.main()
