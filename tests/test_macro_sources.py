from __future__ import annotations

import math
import time
import unittest
from datetime import date
from unittest.mock import patch

from savi_uz.macro_catalog import (
    FED_SEP,
    FRED_CATALOG,
    GROUPS,
    VintagePolicy,
    catalog_for_groups,
)
from savi_uz.macro_sources import (
    MAX_VINTAGE_DATES_PER_REQUEST,
    FredClient,
    GswCurveClient,
    NyFedRatesClient,
    RateLimiter,
    SvenssonParameters,
    VintageLimitError,
    implied_policy_path,
)

GSW_CSV = """\
"Note: This is not an official Federal Reserve statistical release."

Series,Compounding Convention,Mnemonic(s)
Zero-coupon yield,Continuously Compounded,SVENYXX

Date,SVENY01,SVENY02,SVEN1F01,BETA0,BETA1,BETA2,BETA3,TAU1,TAU2
2024-01-02,4.75,4.30,4.10,4.5,-1.2,0.8,0.5,1.5,10.0
2024-01-03,4.70,4.28,4.05,4.4,-1.1,0.7,0.4,1.6,9.5
2024-01-04,NA,NA,NA,NA,NA,NA,NA,NA,NA
"""


class SvenssonTests(unittest.TestCase):
    def setUp(self):
        self.fit = SvenssonParameters(date(2024, 1, 2), 4.5, -1.2, 0.8, 0.5, 1.5, 10.0)

    def test_forward_at_zero_horizon_is_beta0_plus_beta1(self):
        """The Svensson curve pins the front end at beta0 + beta1."""
        self.assertAlmostEqual(self.fit.instantaneous_forward(0.0), 4.5 - 1.2, places=4)

    def test_forward_at_long_horizon_converges_to_beta0(self):
        self.assertAlmostEqual(self.fit.instantaneous_forward(200.0), 4.5, places=3)

    def test_forward_matches_the_closed_form(self):
        n = 2.0
        expected = (
            4.5
            + -1.2 * math.exp(-n / 1.5)
            + 0.8 * (n / 1.5) * math.exp(-n / 1.5)
            + 0.5 * (n / 10.0) * math.exp(-n / 10.0)
        )
        self.assertAlmostEqual(self.fit.instantaneous_forward(n), expected, places=9)

    def test_policy_path_emits_one_row_per_date_and_horizon(self):
        rows = list(implied_policy_path([self.fit, self.fit], (3, 12, 24)))
        self.assertEqual(len(rows), 6)
        self.assertEqual({months for _, months, _ in rows}, {3, 12, 24})


class GswParseTests(unittest.TestCase):
    def test_parse_skips_the_disclaimer_preamble(self):
        parameters, rates = GswCurveClient.parse(GSW_CSV)
        self.assertEqual([p.curve_date for p in parameters], [date(2024, 1, 2), date(2024, 1, 3)])
        self.assertEqual(parameters[0].beta0, 4.5)

    def test_parse_collects_rate_mnemonics_in_long_form(self):
        _, rates = GswCurveClient.parse(GSW_CSV)
        first_day = {mnemonic: value for day, mnemonic, value in rates if day == date(2024, 1, 2)}
        self.assertEqual(first_day, {"SVENY01": 4.75, "SVENY02": 4.30, "SVEN1F01": 4.10})

    def test_rows_with_unparseable_parameters_are_dropped(self):
        parameters, rates = GswCurveClient.parse(GSW_CSV)
        self.assertNotIn(date(2024, 1, 4), [p.curve_date for p in parameters])
        self.assertNotIn(date(2024, 1, 4), [day for day, _, _ in rates])

    def test_missing_header_is_an_error(self):
        with self.assertRaises(ValueError):
            GswCurveClient.parse("no header here\njust,text\n")

    def test_plain_http_url_is_refused(self):
        with self.assertRaises(ValueError):
            GswCurveClient(url="http://www.federalreserve.gov/x.csv")


class FredClientTests(unittest.TestCase):
    def setUp(self):
        self.client = FredClient(api_key="testkey")

    def test_missing_api_key_is_refused(self):
        with self.assertRaises(ValueError):
            FredClient(api_key="")

    def test_observations_parse_missing_values_as_none(self):
        payload = {"observations": [
            {"date": "2024-01-01", "value": "1.5"},
            {"date": "2024-01-02", "value": "."},
        ]}
        with patch.object(FredClient, "_get", return_value=payload):
            observations = self.client.fetch_observations("DGS10")
        self.assertEqual(observations[0].value, 1.5)
        self.assertIsNone(observations[1].value)

    def test_first_releases_keep_the_release_date(self):
        payload = {"observations": [
            {"date": "2024-01-01", "value": "157700", "realtime_start": "2024-02-02",
             "realtime_end": "2024-03-07"},
        ]}
        with patch.object(FredClient, "_get", return_value=payload):
            first = self.client.fetch_first_releases("PAYEMS")
        self.assertEqual(first[0].realtime_start, date(2024, 2, 2))
        self.assertEqual(first[0].realtime_end, date(2024, 3, 7))

    def test_all_vintages_melts_the_wide_response(self):
        """ALFRED returns one column per vintage; each becomes its own row."""
        payload = {"observations": [
            {"date": "2028-01-01", "FEDTARMD_20250917": "3.1", "FEDTARMD_20260617": "3.4",
             "FEDTARMD_20240101": "."},
        ]}
        with patch.object(FredClient, "_get", return_value=payload):
            vintages = self.client.fetch_all_vintages("FEDTARMD")
        self.assertEqual(
            {(v.realtime_start, v.value) for v in vintages},
            {(date(2025, 9, 17), 3.1), (date(2026, 6, 17), 3.4)},
        )
        self.assertTrue(all(v.obs_date == date(2028, 1, 1) for v in vintages))

    def test_vintage_limit_triggers_windowed_retry(self):
        """Over ALFRED's 2000-vintage cap the request is split, not abandoned."""
        vintage_dates = [date(2000, 1, 1).replace(year=2000 + i // 300) for i in range(4000)]
        calls: list[dict] = []

        def fake_get(path, **params):
            calls.append(params)
            if path == "series/vintagedates":
                return {"vintage_dates": [d.isoformat() for d in vintage_dates]}
            if params.get("realtime_start") == "1776-07-04":
                raise VintageLimitError("There are 3103 vintage dates ... maximum number of vintage dates")
            return {"observations": [{"date": "2024-01-01", "value": "1", "realtime_start": "2024-01-02"}]}

        with patch.object(FredClient, "_get", side_effect=fake_get):
            first = self.client.fetch_first_releases("T5YIE")

        windowed = [c for c in calls if c.get("realtime_start") not in (None, "1776-07-04")]
        self.assertEqual(len(windowed), math.ceil(len(vintage_dates) / MAX_VINTAGE_DATES_PER_REQUEST))
        self.assertEqual(len(first), len(windowed))

    def test_metadata_is_parsed_into_typed_dates(self):
        payload = {"seriess": [{
            "title": "Ten Year", "units": "Percent", "frequency": "Daily",
            "seasonal_adjustment": "NSA", "observation_start": "1962-01-02",
            "observation_end": "2026-08-13", "last_updated": "2026-08-14 15:17:23-05",
            "popularity": "90", "notes": "n",
        }]}
        with patch.object(FredClient, "_get", return_value=payload):
            metadata = self.client.fetch_metadata("DGS10")
        self.assertEqual(metadata.observation_start, date(1962, 1, 2))
        self.assertEqual(metadata.popularity, 90)


class NyFedClientTests(unittest.TestCase):
    def test_reference_rates_keep_the_revision_flag(self):
        payload = {"refRates": [{
            "effectiveDate": "2026-08-13", "type": "EFFR", "percentRate": 3.63,
            "percentPercentile1": 3.60, "percentPercentile99": 3.65,
            "volumeInBillions": 98.0, "targetRateFrom": 3.50, "revisionIndicator": "R",
        }]}
        client = NyFedRatesClient()
        with patch("savi_uz.macro_sources._http_json", return_value=payload):
            rates = client.fetch_rates("effr", "unsecured", date(2026, 8, 1), date(2026, 8, 14))
        self.assertEqual(rates[0].revision_indicator, "R")
        self.assertEqual(rates[0].percent_rate, 3.63)
        self.assertIsNone(rates[0].percentile_25)

    def test_plain_http_base_url_is_refused(self):
        with self.assertRaises(ValueError):
            NyFedRatesClient(base_url="http://markets.newyorkfed.org/api")


class RateLimiterTests(unittest.TestCase):
    def test_requests_are_spaced_by_the_configured_interval(self):
        limiter = RateLimiter(max_per_minute=600)  # 100ms apart
        start = time.monotonic()
        for _ in range(3):
            limiter.acquire()
        self.assertGreaterEqual(time.monotonic() - start, 0.18)


class CatalogTests(unittest.TestCase):
    def test_series_ids_are_unique(self):
        ids = [spec.series_id for spec in FRED_CATALOG]
        self.assertEqual(len(ids), len(set(ids)))

    def test_sep_series_always_collect_every_vintage(self):
        """The dot plot only exists as a vintage sequence."""
        self.assertTrue(all(spec.vintages is VintagePolicy.ALL for spec in FED_SEP))

    def test_every_spec_belongs_to_a_declared_group(self):
        self.assertTrue(all(spec.group in GROUPS for spec in FRED_CATALOG))

    def test_filtering_by_group_returns_only_that_group(self):
        specs = catalog_for_groups(("credit",))
        self.assertTrue(specs)
        self.assertEqual({spec.group for spec in specs}, {"credit"})

    def test_no_filter_returns_the_whole_catalog(self):
        self.assertEqual(catalog_for_groups(None), FRED_CATALOG)

    def test_unknown_group_is_rejected(self):
        with self.assertRaises(ValueError):
            catalog_for_groups(("not_a_group",))


if __name__ == "__main__":
    unittest.main()
