from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from savi_uz.tiingo_sources import (
    MAX_ROWS_PER_REQUEST,
    NO_INTRADAY_EXCHANGES,
    HourlyRateLimiter,
    SymbolMeta,
    TiingoClient,
    TiingoError,
    TiingoRateLimitError,
    year_windows,
)

BARS = [
    {"date": "2024-01-02T15:00:00.000Z", "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5,
     "volume": 100},
    {"date": "2024-01-02T16:00:00.000Z", "open": 1.5, "high": 2.5, "low": 1.0, "close": 2.0,
     "volume": 200},
]

META = {
    "name": "Apple Inc", "exchangeCode": "NASDAQ",
    "startDate": "1980-12-12", "endDate": "2026-08-14", "description": "phones",
}


class _Response:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code: int) -> HTTPError:
    return HTTPError("https://api.tiingo.com/x", code, "err", {}, None)


class RateLimiterTests(unittest.TestCase):
    def test_hourly_budget_becomes_an_even_interval(self):
        """The per-minute limiter floors at 60/hour, already over the free tier."""
        self.assertAlmostEqual(HourlyRateLimiter(45).min_interval, 80.0, places=6)
        self.assertAlmostEqual(HourlyRateLimiter(3600).min_interval, 1.0, places=6)

    def test_the_first_request_is_not_delayed(self):
        limiter = HourlyRateLimiter(1)
        with patch("savi_uz.tiingo_sources.time.sleep") as sleep:
            limiter.acquire()
        sleep.assert_not_called()

    def test_the_second_request_waits_a_full_interval(self):
        limiter = HourlyRateLimiter(60)
        limiter.acquire()
        with patch("savi_uz.tiingo_sources.time.sleep") as sleep:
            limiter.acquire()
        self.assertTrue(sleep.called)
        self.assertGreater(sleep.call_args[0][0], 0)

    def test_a_nonsense_rate_is_rejected(self):
        with self.assertRaises(ValueError):
            HourlyRateLimiter(0)


class YearWindowTests(unittest.TestCase):
    def test_a_range_splits_on_calendar_years(self):
        windows = year_windows(date(2017, 3, 1), date(2019, 6, 30))
        self.assertEqual(
            windows,
            [
                (date(2017, 3, 1), date(2017, 12, 31)),
                (date(2018, 1, 1), date(2018, 12, 31)),
                (date(2019, 1, 1), date(2019, 6, 30)),
            ],
        )

    def test_a_single_year_is_one_window(self):
        self.assertEqual(len(year_windows(date(2024, 1, 1), date(2024, 12, 31))), 1)

    def test_an_inverted_range_is_empty(self):
        self.assertEqual(year_windows(date(2024, 1, 1), date(2023, 1, 1)), [])


class SymbolMetaTests(unittest.TestCase):
    def test_otc_exchanges_have_no_intraday(self):
        """IEX is exchange-listed only; the PINK ADRs return zero bars always."""
        for exchange in NO_INTRADAY_EXCHANGES:
            with self.subTest(exchange):
                meta = SymbolMeta("TCEHY", "Tencent", exchange, None, None)
                self.assertFalse(meta.has_intraday)

    def test_listed_exchanges_do(self):
        for exchange in ("NASDAQ", "NYSE", "NYSE ARCA", "BATS"):
            with self.subTest(exchange):
                self.assertTrue(SymbolMeta("AAPL", "Apple", exchange, None, None).has_intraday)

    def test_the_check_is_case_insensitive(self):
        self.assertFalse(SymbolMeta("X", "", "pink", None, None).has_intraday)


class TiingoClientTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.client = TiingoClient(
            "token", cache_dir=Path(self._dir.name), requests_per_hour=3600
        )

    def tearDown(self):
        self._dir.cleanup()

    def test_an_empty_key_is_refused(self):
        with self.assertRaises(ValueError):
            TiingoClient("", cache_dir=Path(self._dir.name))

    def test_metadata_is_parsed(self):
        with patch("savi_uz.tiingo_sources.urlopen", return_value=_Response(META)):
            meta = self.client.fetch_metadata("AAPL")
        self.assertEqual(meta.exchange, "NASDAQ")
        self.assertEqual(meta.start_date, date(1980, 12, 12))
        self.assertTrue(meta.has_intraday)

    def test_intraday_bars_are_parsed(self):
        with patch("savi_uz.tiingo_sources.urlopen", return_value=_Response(BARS)):
            bars, truncated = self.client.fetch_intraday("AAPL", date(2024, 1, 1), date(2024, 12, 31))
        self.assertEqual(len(bars), 2)
        self.assertFalse(truncated)
        self.assertEqual(bars[0].frequency, "1hour")
        self.assertEqual(bars[0].close, 1.5)
        self.assertEqual(bars[0].timestamp, "2024-01-02T15:00:00.000Z")

    def test_a_response_at_the_row_cap_is_flagged(self):
        """Tiingo truncates silently and returns the recent end of the range."""
        payload = [dict(BARS[0], date=f"2024-01-02T{i:05d}") for i in range(MAX_ROWS_PER_REQUEST)]
        with patch("savi_uz.tiingo_sources.urlopen", return_value=_Response(payload)):
            bars, truncated = self.client.fetch_intraday("AAPL", date(2017, 1, 1), date(2026, 1, 1))
        self.assertTrue(truncated)
        self.assertEqual(len(bars), MAX_ROWS_PER_REQUEST)

    def test_responses_are_cached_and_not_refetched(self):
        with patch("savi_uz.tiingo_sources.urlopen", return_value=_Response(BARS)) as opener:
            self.client.fetch_intraday("AAPL", date(2024, 1, 1), date(2024, 12, 31))
            self.client.fetch_intraday("AAPL", date(2024, 1, 1), date(2024, 12, 31))
        self.assertEqual(opener.call_count, 1)
        self.assertEqual(self.client.requests_made, 1)
        self.assertEqual(self.client.cache_hits, 1)

    def test_different_windows_cache_separately(self):
        with patch("savi_uz.tiingo_sources.urlopen", return_value=_Response(BARS)) as opener:
            self.client.fetch_intraday("AAPL", date(2024, 1, 1), date(2024, 12, 31))
            self.client.fetch_intraday("AAPL", date(2025, 1, 1), date(2025, 12, 31))
        self.assertEqual(opener.call_count, 2)

    def test_a_429_raises_the_rate_limit_error_and_does_not_retry(self):
        with patch("savi_uz.tiingo_sources.urlopen", side_effect=_http_error(429)) as opener:
            with self.assertRaises(TiingoRateLimitError):
                self.client.fetch_metadata("AAPL")
        self.assertEqual(opener.call_count, 1)

    def test_an_auth_failure_is_reported_clearly(self):
        for code in (401, 403):
            with self.subTest(code):
                with patch("savi_uz.tiingo_sources.urlopen", side_effect=_http_error(code)):
                    with self.assertRaises(TiingoError) as caught:
                        self.client.fetch_metadata("AAPL")
                self.assertIn("rejected the key", str(caught.exception))

    def test_the_request_budget_stops_further_calls(self):
        client = TiingoClient(
            "token", cache_dir=Path(self._dir.name), requests_per_hour=3600, max_requests=1
        )
        with patch("savi_uz.tiingo_sources.urlopen", return_value=_Response(BARS)):
            client.fetch_intraday("AAPL", date(2024, 1, 1), date(2024, 12, 31))
            self.assertTrue(client.budget_exhausted())
            with self.assertRaises(TiingoError):
                client.fetch_intraday("AAPL", date(2025, 1, 1), date(2025, 12, 31))

    def test_a_cached_window_does_not_count_against_the_budget(self):
        """This is what makes a resumed run free for work already done."""
        with patch("savi_uz.tiingo_sources.urlopen", return_value=_Response(BARS)):
            self.client.fetch_intraday("AAPL", date(2024, 1, 1), date(2024, 12, 31))
        client = TiingoClient(
            "token", cache_dir=Path(self._dir.name), requests_per_hour=3600, max_requests=0
        )
        bars, _ = client.fetch_intraday("AAPL", date(2024, 1, 1), date(2024, 12, 31))
        self.assertEqual(len(bars), 2)
        self.assertEqual(client.requests_made, 0)

    def test_refresh_bypasses_the_cache(self):
        with patch("savi_uz.tiingo_sources.urlopen", return_value=_Response(BARS)):
            self.client.fetch_intraday("AAPL", date(2024, 1, 1), date(2024, 12, 31))
        fresh = TiingoClient(
            "token", cache_dir=Path(self._dir.name), requests_per_hour=3600, refresh=True
        )
        with patch("savi_uz.tiingo_sources.urlopen", return_value=_Response(BARS)) as opener:
            fresh.fetch_intraday("AAPL", date(2024, 1, 1), date(2024, 12, 31))
        self.assertEqual(opener.call_count, 1)

    def test_an_unsupported_frequency_is_rejected_before_any_request(self):
        with patch("savi_uz.tiingo_sources.urlopen") as opener:
            with self.assertRaises(ValueError):
                self.client.fetch_intraday("AAPL", date(2024, 1, 1), date(2024, 12, 31), "3hour")
        opener.assert_not_called()

    def test_daily_bars_carry_the_daily_frequency(self):
        with patch("savi_uz.tiingo_sources.urlopen", return_value=_Response(BARS)):
            bars, _ = self.client.fetch_daily("TCEHY", date(2024, 1, 1), date(2024, 12, 31))
        self.assertEqual(bars[0].frequency, "daily")

    def test_malformed_records_are_skipped(self):
        payload = [BARS[0], {"open": 1.0}, {"date": None}]
        with patch("savi_uz.tiingo_sources.urlopen", return_value=_Response(payload)):
            bars, _ = self.client.fetch_intraday("AAPL", date(2024, 1, 1), date(2024, 12, 31))
        self.assertEqual(len(bars), 1)


if __name__ == "__main__":
    unittest.main()


class AdjustmentTests(unittest.TestCase):
    """Intraday bars are raw, so split and dividend factors have to come along."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.client = TiingoClient("token", cache_dir=Path(self._dir.name), requests_per_hour=3600)

    def tearDown(self):
        self._dir.cleanup()

    def test_split_and_dividend_factors_are_parsed(self):
        payload = [
            {"date": "2024-06-07T00:00:00.000Z", "close": 1208.88, "adjClose": 120.68,
             "splitFactor": 1.0, "divCash": 0.0},
            {"date": "2024-06-10T00:00:00.000Z", "close": 121.79, "adjClose": 121.58,
             "splitFactor": 10.0, "divCash": 0.0},
            {"date": "2024-06-11T00:00:00.000Z", "close": 120.91, "adjClose": 120.71,
             "splitFactor": 1.0, "divCash": 0.01},
        ]
        with patch("savi_uz.tiingo_sources.urlopen", return_value=_Response(payload)):
            rows = self.client.fetch_adjustments("NVDA", date(2024, 6, 1), date(2024, 6, 30))

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1].split_factor, 10.0)
        self.assertTrue(rows[1].is_split)
        self.assertFalse(rows[0].is_split)
        self.assertEqual(rows[2].div_cash, 0.01)

    def test_absent_factors_default_to_no_action(self):
        payload = [{"date": "2024-06-07T00:00:00.000Z", "close": 1.0}]
        with patch("savi_uz.tiingo_sources.urlopen", return_value=_Response(payload)):
            rows = self.client.fetch_adjustments("X", date(2024, 6, 1), date(2024, 6, 30))
        self.assertEqual(rows[0].split_factor, 1.0)
        self.assertEqual(rows[0].div_cash, 0.0)
        self.assertFalse(rows[0].is_split)

    def test_undated_records_are_skipped(self):
        payload = [{"close": 1.0}, {"date": "2024-06-07T00:00:00.000Z", "close": 2.0}]
        with patch("savi_uz.tiingo_sources.urlopen", return_value=_Response(payload)):
            rows = self.client.fetch_adjustments("X", date(2024, 6, 1), date(2024, 6, 30))
        self.assertEqual(len(rows), 1)


class IntradayColumnTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.client = TiingoClient("token", cache_dir=Path(self._dir.name), requests_per_hour=3600)

    def tearDown(self):
        self._dir.cleanup()

    def test_volume_is_requested_explicitly(self):
        """The IEX default projection omits volume without saying so."""
        captured = {}

        def fake(request, **kwargs):
            captured["url"] = request.full_url
            return _Response(BARS)

        with patch("savi_uz.tiingo_sources.urlopen", side_effect=fake):
            self.client.fetch_intraday("SPY", date(2024, 1, 1), date(2024, 12, 31))
        self.assertIn("volume", captured["url"])

    def test_volume_reaches_the_bar(self):
        payload = [dict(BARS[0], volume=150780.0)]
        with patch("savi_uz.tiingo_sources.urlopen", return_value=_Response(payload)):
            bars, _ = self.client.fetch_intraday("SPY", date(2024, 1, 1), date(2024, 12, 31))
        self.assertEqual(bars[0].volume, 150780.0)
