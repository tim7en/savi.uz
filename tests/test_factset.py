from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from savi_uz.factset_catalog import (
    ARCHIVE_FIRST_DATE,
    CORE_FIELDS,
    candidate_urls,
    fridays,
)
from savi_uz.factset_sources import (
    FactSetClient,
    normalise,
    parse_document_date,
    parse_key_metrics,
)

# Wording as it actually appears in the two eras of the report.
PAGE_2017 = """
John Butters, Senior Earnings Analyst Media Questions/Requests
February 3, 2017
Key Metrics
- Earnings Scorecard: As of today (with 55% of the companies in the S&P 500 reporting actual
results for Q4 2016), 65% of S&P 500 companies have beat the mean EPS estimate and 52% of S&P 500
companies have beat the mean sales estimate.
- Earnings Growth: For Q4 2016, the blended earnings growth rate for the S&P 500 is 4.6%.
- Earnings Revisions: On December 31, the estimated earnings growth rate for Q4 2016 was 3.1%.
- Earnings Guidance: For Q1 2017, 44 S&P 500 companies have issued negative EPS guidance and 21
S&P 500 companies have issued positive EPS guidance.
- Valuation: The forward 12-month P/E ratio for the S&P 500 is 17.1. This P/E ratio is based on
Thursday's closing price (2280.85) and forward 12-month EPS estimate ($133.54).
"""

PAGE_2026 = """
EARNINGS INSIGHT
August 7, 2026
Key Metrics
- Earnings Scorecard: For Q2 2026 (with 88% of S&P 500 companies reporting actual results), 86% of
S&P 500 companies have reported a positive EPS surprise and 76% of S&P 500 companies has reported
a positive revenue surprise.
- Earnings Growth: For Q2 2026, the blended (year-over-year) earnings growth rate for the S&P 500
is 50.4%.
- Earnings Revisions: On June 30, the estimated (year-over-year) earnings growth rate for the
S&P 500 for Q2 2026 was 23.1%.
- Valuation: The forward 12-month P/E ratio for the S&P 500 is 20.0. This P/E ratio is above the
5-year average (19.9) and above the 10-year average (19.0).
"""

# Down quarters retitle the section and swap "growth rate" for "decline".
PAGE_DECLINE = """
August 2, 2019
Key Metrics
- Earnings Scorecard: For Q2 2019 (with 77% of the companies in the S&P 500 reporting actual
results), 76% of S&P 500 companies have reported a positive EPS surprise and 59% of companies have
reported a positive revenue surprise.
- Earnings Growth: For Q2 2019, the blended earnings decline for the S&P 500 is -1.0%.
- Earnings Revisions: On June 30, the estimated earnings decline for Q2 2019 was -2.7%.
- Valuation: The forward 12-month P/E ratio for the S&P 500 is 16.8. This P/E ratio is above the
5-year average (16.5) and above the 10-year average (14.8).
"""

# Between reporting seasons there is no scorecard, and growth is "estimated".
PAGE_OFF_SEASON = """
June 26, 2026
Key Metrics
- Earnings Growth: For Q2 2026, the estimated (year-over-year) earnings growth rate for the
S&P 500 is 23.1%.
- Earnings Revisions: On March 31, the estimated (year-over-year) earnings growth rate for the
S&P 500 for Q2 2026 was 18.8%.
- Earnings Guidance: For Q2 2026, 48 S&P 500 companies have issued negative EPS guidance and 63
S&P 500 companies have issued positive EPS guidance.
- Valuation: The forward 12-month P/E ratio for the S&P 500 is 20.1. This P/E ratio is above the
5-year average (19.9) and above the 10-year average (19.0).
"""


def _parse(text: str, day: date = date(2026, 8, 7)):
    return parse_key_metrics(text, day, "https://example/EarningsInsight.pdf")


class WeekPlanTests(unittest.TestCase):
    def test_fridays_are_clamped_to_the_first_hosted_edition(self):
        """Asking for 2000 must not generate 17 years of guaranteed 404s."""
        weeks = fridays(date(2000, 1, 1), date(2017, 3, 1))
        self.assertEqual(weeks[0], ARCHIVE_FIRST_DATE)
        self.assertTrue(all(week >= ARCHIVE_FIRST_DATE for week in weeks))

    def test_every_generated_day_is_a_friday(self):
        self.assertTrue(all(week.weekday() == 4 for week in fridays(date(2024, 1, 1), date(2024, 12, 31))))

    def test_a_full_year_has_the_expected_number_of_weeks(self):
        self.assertEqual(len(fridays(date(2024, 1, 1), date(2024, 12, 31))), 52)

    def test_range_entirely_before_the_archive_is_empty(self):
        self.assertEqual(fridays(date(2000, 1, 1), date(2010, 1, 1)), ())

    def test_inverted_range_is_rejected(self):
        with self.assertRaises(ValueError):
            fridays(date(2024, 12, 31), date(2024, 1, 1))

    def test_candidates_cover_the_thursday_and_suffixed_editions(self):
        """Without these fallbacks a quarter of published weeks look missing."""
        names = [c.url.rsplit("/", 1)[-1] for c in candidate_urls(date(2024, 3, 22))]
        self.assertEqual(
            names,
            [
                "EarningsInsight_032224.pdf",
                "EarningsInsight_032224A.pdf",
                "EarningsInsight_032124.pdf",
                "EarningsInsight_032124A.pdf",
            ],
        )

    def test_candidate_dates_match_their_filenames(self):
        candidates = candidate_urls(date(2024, 3, 22))
        self.assertEqual(candidates[0].published, date(2024, 3, 22))
        self.assertEqual(candidates[2].published, date(2024, 3, 21))


class NormaliseTests(unittest.TestCase):
    def test_line_wrapping_is_collapsed_so_patterns_can_span_breaks(self):
        self.assertEqual(normalise("the blended\nearnings   growth\nrate"), "the blended earnings growth rate")

    def test_typographic_punctuation_is_folded(self):
        self.assertEqual(normalise("Author’s Note"), "Author's Note")
        self.assertEqual(normalise("“Why FactSet?”"), '"Why FactSet?"')


class DocumentDateTests(unittest.TestCase):
    def test_dateline_is_read(self):
        self.assertEqual(parse_document_date("EARNINGS INSIGHT August 7, 2026 Key Metrics"), date(2026, 8, 7))
        self.assertEqual(parse_document_date("February 3, 2017"), date(2017, 2, 3))

    def test_absent_dateline_is_none(self):
        self.assertIsNone(parse_document_date("no date here"))


class KeyMetricTests(unittest.TestCase):
    def test_modern_wording(self):
        metrics = _parse(PAGE_2026)
        self.assertEqual(metrics.values["quarter"], "Q2 2026")
        self.assertEqual(metrics.values["pct_reported"], 88.0)
        self.assertEqual(metrics.values["pct_positive_eps"], 86.0)
        self.assertEqual(metrics.values["pct_positive_revenue"], 76.0)
        self.assertEqual(metrics.values["blended_earnings_growth"], 50.4)
        self.assertEqual(metrics.values["estimated_growth_at_quarter_start"], 23.1)
        self.assertEqual(metrics.values["forward_12m_pe"], 20.0)
        self.assertEqual(metrics.values["pe_5y_average"], 19.9)
        self.assertEqual(metrics.values["pe_10y_average"], 19.0)
        self.assertEqual(metrics.missing_core, ())

    def test_2017_wording_is_read_by_the_same_parser(self):
        metrics = _parse(PAGE_2017, date(2017, 2, 3))
        self.assertEqual(metrics.values["quarter"], "Q4 2016")
        self.assertEqual(metrics.values["pct_reported"], 55.0)
        self.assertEqual(metrics.values["pct_positive_eps"], 65.0)
        self.assertEqual(metrics.values["pct_positive_revenue"], 52.0)
        self.assertEqual(metrics.values["blended_earnings_growth"], 4.6)
        self.assertEqual(metrics.values["forward_12m_pe"], 17.1)
        self.assertEqual(metrics.missing_core, ())

    def test_forward_pe_does_not_swallow_the_sentence_period(self):
        self.assertEqual(_parse(PAGE_2017, date(2017, 2, 3)).values["forward_12m_pe"], 17.1)

    def test_only_the_early_era_carries_price_and_forward_eps(self):
        early = _parse(PAGE_2017, date(2017, 2, 3))
        self.assertEqual(early.values["index_price"], 2280.85)
        self.assertEqual(early.values["forward_12m_eps"], 133.54)
        self.assertNotIn("index_price", _parse(PAGE_2026).values)

    def test_guidance_counts_are_integers(self):
        metrics = _parse(PAGE_2017, date(2017, 2, 3))
        self.assertEqual(metrics.values["negative_guidance_count"], 44)
        self.assertEqual(metrics.values["positive_guidance_count"], 21)

    def test_decline_wording_keeps_the_negative_sign(self):
        """Down quarters are retitled 'decline'; the value must stay negative."""
        metrics = _parse(PAGE_DECLINE, date(2019, 8, 2))
        self.assertEqual(metrics.values["blended_earnings_growth"], -1.0)
        self.assertEqual(metrics.values["estimated_growth_at_quarter_start"], -2.7)
        self.assertEqual(metrics.values["pct_positive_revenue"], 59.0)

    def test_off_season_growth_is_estimated_not_blended(self):
        """Blended and estimated are different quantities and must not merge."""
        metrics = _parse(PAGE_OFF_SEASON, date(2026, 6, 26))
        self.assertEqual(metrics.values["estimated_earnings_growth"], 23.1)
        self.assertNotIn("blended_earnings_growth", metrics.values)
        self.assertEqual(metrics.values["quarter"], "Q2 2026")

    def test_in_season_reports_do_not_populate_the_estimated_field(self):
        self.assertNotIn("estimated_earnings_growth", _parse(PAGE_2026).values)

    def test_a_missing_line_is_recorded_rather_than_guessed(self):
        text = PAGE_2026.replace("- Valuation: The forward 12-month P/E ratio for the S&P 500 is 20.0.", "")
        metrics = _parse(text)
        self.assertIn("forward_12m_pe", metrics.missing)
        self.assertIn("forward_12m_pe", metrics.missing_core)
        self.assertNotIn("forward_12m_pe", metrics.values)

    def test_core_fields_are_the_ones_present_in_every_era(self):
        self.assertEqual(set(CORE_FIELDS), {"quarter", "forward_12m_pe"})

    def test_page_text_is_retained_for_reparsing(self):
        metrics = _parse(PAGE_2026)
        self.assertIn("Key Metrics", metrics.page_text)
        self.assertNotIn("\n", metrics.page_text)


class FactSetClientTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.client = FactSetClient(cache_dir=Path(self._dir.name))

    def tearDown(self):
        self._dir.cleanup()

    def test_a_cached_pdf_is_used_without_a_request(self):
        candidate = candidate_urls(date(2026, 8, 7))[0]
        (Path(self._dir.name) / candidate.url.rsplit("/", 1)[-1]).write_bytes(b"%PDF-1.7" + b"x" * 20000)
        with patch("savi_uz.factset_sources.urlopen", side_effect=AssertionError("no request expected")):
            self.assertIsNotNone(self.client.download(candidate))

    def test_an_html_stub_served_as_a_pdf_is_treated_as_unpublished(self):
        """FactSet answers some retired names with a 200 and an HTML page."""
        candidate = candidate_urls(date(2026, 8, 7))[0]

        class _Response:
            def read(self):
                return b"<html>not found</html>"

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        with patch("savi_uz.factset_sources.urlopen", return_value=_Response()):
            self.assertIsNone(self.client.download(candidate))

    def test_a_truncated_pdf_is_treated_as_unpublished(self):
        candidate = candidate_urls(date(2026, 8, 7))[0]

        class _Response:
            def read(self):
                return b"%PDF-1.7 tiny"

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        with patch("savi_uz.factset_sources.urlopen", return_value=_Response()):
            self.assertIsNone(self.client.download(candidate))

    def test_a_week_where_no_candidate_resolves_is_a_non_publication_week(self):
        with patch.object(FactSetClient, "download", return_value=None):
            self.assertIsNone(self.client.fetch_week(date(2024, 3, 29)))

    def test_the_first_resolving_candidate_wins(self):
        calls = []

        def download(candidate):
            calls.append(candidate.url)
            return b"%PDF" if candidate.url.endswith("032124.pdf") else None

        with patch.object(FactSetClient, "download", side_effect=download), \
             patch("savi_uz.factset_sources.extract_first_page", return_value=PAGE_2026):
            metrics = self.client.fetch_week(date(2024, 3, 22))

        self.assertEqual(len(calls), 3)
        self.assertEqual(metrics.report_date, date(2024, 3, 21))


if __name__ == "__main__":
    unittest.main()
