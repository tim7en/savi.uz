from __future__ import annotations

import json
import unittest
from datetime import date
from unittest.mock import patch

from savi_uz.equity_catalog import SEC_CONCEPTS, ConceptSpec, quarters
from savi_uz.equity_sources import (
    AlphaVantageEarningsClient,
    SecFramesClient,
    ShillerClient,
    SourceError,
    parse_shiller_month,
)


class ShillerMonthTests(unittest.TestCase):
    def test_january_is_read_from_the_two_decimal_form(self):
        self.assertEqual(parse_shiller_month(1871.01), date(1871, 1, 1))

    def test_october_is_october_not_january(self):
        """Shiller stores 1871.10 as the float 1871.1; naive parsing gives January."""
        self.assertEqual(parse_shiller_month(1871.1), date(1871, 10, 1))
        self.assertEqual(parse_shiller_month(2024.1), date(2024, 10, 1))

    def test_november_and_december(self):
        self.assertEqual(parse_shiller_month(1871.11), date(1871, 11, 1))
        self.assertEqual(parse_shiller_month(2023.12), date(2023, 12, 1))

    def test_string_dates_are_accepted(self):
        self.assertEqual(parse_shiller_month("2023.09"), date(2023, 9, 1))
        self.assertEqual(parse_shiller_month(" 2023.09 "), date(2023, 9, 1))

    def test_footnote_rows_are_none(self):
        for value in (None, "", "   ", "Sept price is average", 2023.13, 2023.0):
            with self.subTest(value):
                self.assertIsNone(parse_shiller_month(value))


class ShillerMirrorTests(unittest.TestCase):
    """The mirrors drift apart by up to a year, so the fresher one must win."""

    class _Row:
        def __init__(self, obs_date):
            self.obs_date = obs_date
            self.values = {"sp500_price": 1.0}

    def _client(self):
        return ShillerClient(urls=("https://stale.example/ie.xls", "https://fresh.example/ie.xls"))

    def test_the_mirror_with_the_later_last_month_wins(self):
        parsed = {
            "https://stale.example/ie.xls": [self._Row(date(2023, 9, 1))],
            "https://fresh.example/ie.xls": [self._Row(date(2024, 9, 1))],
        }
        client = self._client()
        with patch.object(ShillerClient, "download", side_effect=lambda url: url.encode()), \
             patch.object(ShillerClient, "parse", side_effect=lambda raw: parsed[raw.decode()]):
            url, rows = client.fetch()
        self.assertEqual(url, "https://fresh.example/ie.xls")
        self.assertEqual(rows[-1].obs_date, date(2024, 9, 1))

    def test_a_dead_mirror_is_tolerated_if_another_works(self):
        def parse(raw):
            if b"stale" in raw:
                raise SourceError("404")
            return [self._Row(date(2024, 9, 1))]

        client = self._client()
        with patch.object(ShillerClient, "download", side_effect=lambda url: url.encode()), \
             patch.object(ShillerClient, "parse", side_effect=parse):
            url, rows = client.fetch()
        self.assertEqual(url, "https://fresh.example/ie.xls")

    def test_every_mirror_failing_raises(self):
        client = self._client()
        with patch.object(ShillerClient, "download", side_effect=SourceError("unreachable")):
            with self.assertRaises(SourceError):
                client.fetch()


class SecConceptTests(unittest.TestCase):
    def test_balance_sheet_concepts_use_the_instant_frame_suffix(self):
        assets = next(c for c in SEC_CONCEPTS if c.tag == "Assets")
        self.assertEqual(assets.frame("CY2023Q1"), "CY2023Q1I")

    def test_income_statement_concepts_do_not(self):
        eps = next(c for c in SEC_CONCEPTS if c.tag == "EarningsPerShareDiluted")
        self.assertEqual(eps.frame("CY2023Q1"), "CY2023Q1")

    def test_eps_url_uses_the_hyphenated_unit(self):
        eps = next(c for c in SEC_CONCEPTS if c.tag == "EarningsPerShareDiluted")
        self.assertEqual(
            eps.url("https://data.sec.gov", "CY2023Q1"),
            "https://data.sec.gov/api/xbrl/frames/us-gaap/EarningsPerShareDiluted/"
            "USD-per-shares/CY2023Q1.json",
        )

    def test_both_revenue_tags_are_present(self):
        """Filers moved to the ASC 606 tag in 2018; neither tag spans the period."""
        tags = {concept.tag for concept in SEC_CONCEPTS}
        self.assertIn("Revenues", tags)
        self.assertIn("RevenueFromContractWithCustomerExcludingAssessedTax", tags)

    def test_quarters_are_inclusive_and_ordered(self):
        self.assertEqual(quarters(2009, 2009), ("CY2009Q1", "CY2009Q2", "CY2009Q3", "CY2009Q4"))
        self.assertEqual(len(quarters(2009, 2026)), 72)
        with self.assertRaises(ValueError):
            quarters(2020, 2019)


FRAME_PAYLOAD = json.dumps(
    {
        "taxonomy": "us-gaap",
        "tag": "EarningsPerShareDiluted",
        "ccp": "CY2023Q1",
        "uom": "USD/shares",
        "data": [
            {
                "accn": "0001104659-24-037408", "cik": 1750, "entityName": "AAR CORP",
                "loc": "US-IL", "start": "2022-12-01", "end": "2023-02-28", "val": 0.62,
            },
            {"accn": "x", "cik": 320193, "entityName": "Apple Inc.", "end": "2023-04-01", "val": 1.52},
            {"accn": "y", "cik": None, "entityName": "broken", "val": 1.0},
            {"accn": "z", "cik": 99, "entityName": "no value", "end": "2023-04-01", "val": None},
        ],
    }
).encode()


class SecFramesClientTests(unittest.TestCase):
    def setUp(self):
        self.concept = ConceptSpec("EarningsPerShareDiluted", "USD-per-shares", False, "EPS")
        self.client = SecFramesClient("savi-uz-tests/1.0")

    def test_a_blank_user_agent_is_refused(self):
        for agent in ("", "   "):
            with self.subTest(agent):
                with self.assertRaises(ValueError):
                    SecFramesClient(agent)

    def test_facts_are_parsed_and_incomplete_records_dropped(self):
        with patch("savi_uz.equity_sources._get", return_value=FRAME_PAYLOAD):
            facts = self.client.fetch_frame(self.concept, "CY2023Q1")

        self.assertEqual(len(facts), 2)
        first = facts[0]
        self.assertEqual(first.cik, 1750)
        self.assertEqual(first.entity_name, "AAR CORP")
        self.assertEqual(first.unit, "USD/shares")
        self.assertEqual(first.period_start, date(2022, 12, 1))
        self.assertEqual(first.period_end, date(2023, 2, 28))
        self.assertEqual(first.value, 0.62)
        self.assertIsNone(facts[1].period_start)

    def test_a_missing_frame_is_empty_not_an_error(self):
        """Unfiled quarters and pre-taxonomy concepts 404; sparse years are normal."""
        with patch("savi_uz.equity_sources._get", side_effect=SourceError("HTTP 404 for ...")):
            self.assertEqual(self.client.fetch_frame(self.concept, "CY2009Q1"), [])

    def test_other_http_errors_still_raise(self):
        with patch("savi_uz.equity_sources._get", side_effect=SourceError("HTTP 500 for ...")):
            with self.assertRaises(SourceError):
                self.client.fetch_frame(self.concept, "CY2023Q1")

    def test_non_json_response_is_reported_clearly(self):
        with patch("savi_uz.equity_sources._get", return_value=b"<html>oops</html>"):
            with self.assertRaises(SourceError) as caught:
                self.client.fetch_frame(self.concept, "CY2023Q1")
        self.assertIn("did not return JSON", str(caught.exception))


class AlphaVantageTests(unittest.TestCase):
    def setUp(self):
        self.client = AlphaVantageEarningsClient("key")

    def test_quarterly_earnings_are_parsed(self):
        payload = json.dumps(
            {
                "quarterlyEarnings": [
                    {
                        "fiscalDateEnding": "2024-06-30", "reportedDate": "2024-08-01",
                        "reportedEPS": "1.4", "estimatedEPS": "1.35",
                        "surprise": "0.05", "surprisePercentage": "3.7",
                    },
                    {"fiscalDateEnding": "2024-03-31", "reportedEPS": "1.53", "estimatedEPS": "None"},
                    {"reportedEPS": "9.9"},
                ]
            }
        ).encode()
        with patch("savi_uz.equity_sources._get", return_value=payload):
            rows = self.client.fetch_earnings("AAPL")

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].fiscal_ending, date(2024, 6, 30))
        self.assertEqual(rows[0].estimated_eps, 1.35)
        self.assertIsNone(rows[1].estimated_eps)
        self.assertIsNone(rows[1].reported_date)

    def test_quota_message_returned_with_http_200_is_raised(self):
        """AlphaVantage reports exhaustion in the body, not the status code."""
        for key in ("Note", "Information", "Error Message"):
            with self.subTest(key):
                payload = json.dumps({key: "rate limit is 25 requests per day"}).encode()
                with patch("savi_uz.equity_sources._get", return_value=payload):
                    with self.assertRaises(SourceError):
                        self.client.fetch_earnings("AAPL")

    def test_empty_key_is_refused(self):
        with self.assertRaises(ValueError):
            AlphaVantageEarningsClient("")


if __name__ == "__main__":
    unittest.main()
