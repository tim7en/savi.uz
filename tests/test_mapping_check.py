from __future__ import annotations

import unittest
from datetime import date, timedelta

from savi_uz.mapping_check import (
    ASSUMED,
    NO_DATA,
    UNVERIFIED,
    VERIFIED,
    check_mapping,
    pick_best_mapping,
)


def _series(values: list[float], start: date = date(2026, 1, 1)) -> dict[date, float]:
    return {start + timedelta(days=offset): value for offset, value in enumerate(values)}


def _walk(steps: list[float], first: float = 100.0) -> list[float]:
    prices = [first]
    for step in steps:
        prices.append(prices[-1] * (1.0 + step))
    return prices


STEPS = [0.01, -0.02, 0.015, 0.004, -0.011, 0.022, -0.006, 0.013, -0.017, 0.008, 0.019, -0.009]


class MappingCheckTests(unittest.TestCase):
    def test_identical_series_verifies(self):
        prices = _walk(STEPS)
        check = check_mapping("AAPLUSDT", "AAPL", "derived", _series(prices), _series(prices))
        self.assertEqual(check.status, VERIFIED)
        self.assertAlmostEqual(check.scale_median, 1.0, places=6)

    def test_local_currency_underlying_verifies_on_stable_ratio(self):
        """A KRW-quoted underlying tracks at a constant FX scale, not 1:1."""
        prices = _walk(STEPS)
        krw = [price * 1390.0 for price in prices]
        check = check_mapping(
            "SAMSUNGUSDT", "005930.KS", "curated", _series(prices), _series(krw), expect_unit_scale=False
        )
        self.assertEqual(check.status, VERIFIED)
        self.assertAlmostEqual(check.scale_median, 1.0 / 1390.0, places=9)

    def test_unit_scale_check_rejects_a_same_named_but_unrelated_asset(self):
        """Yahoo's ``PENG-USD`` is a memecoin, not Penguin Solutions."""
        prices = _walk(STEPS)
        impostor = [price * 40_000.0 for price in _walk(list(reversed(STEPS)))]
        check = check_mapping("PENGUSDT", "PENG-USD", "venue-mirror", _series(prices), _series(impostor))
        self.assertEqual(check.status, UNVERIFIED)

    def test_mid_window_split_does_not_break_a_correct_mapping(self):
        """Yahoo back-adjusts for splits; only the recent ratio window is scored."""
        prices = _walk(STEPS + STEPS)
        adjusted = [price / 20.0 for price in prices[:8]] + prices[8:]
        check = check_mapping("KORUUSDT", "KORU", "derived", _series(prices), _series(adjusted))
        self.assertEqual(check.status, VERIFIED)

    def test_contract_listed_days_ago_is_assumed_only_when_curated(self):
        prices = _walk(STEPS[:1])
        curated = check_mapping("NAVERUSDT", "035420.KS", "curated", _series(prices), _series(prices))
        derived = check_mapping("NAVERUSDT", "NAVER", "derived", _series(prices), _series(prices))
        self.assertEqual(curated.status, ASSUMED)
        self.assertEqual(derived.status, NO_DATA)

    def test_rank_correlation_survives_a_single_outlier(self):
        prices = _walk(STEPS * 2)
        noisy = list(prices)
        noisy[5] *= 1.6  # one bad print must not sink an otherwise sound mapping
        check = check_mapping("XUSDT", "X", "derived", _series(prices), _series(noisy))
        self.assertGreater(check.rank_corr, 0.5)

    def test_best_mapping_prefers_a_real_underlying_over_the_binance_mirror(self):
        prices = _walk(STEPS)
        mirror = check_mapping("CRWDUSDT", "CRWD-USD", "venue-mirror", _series(prices), _series(prices))
        underlying = check_mapping(
            "CRWDUSDT", "CRWD", "derived", _series(prices), _series(_walk([s * 0.9 for s in STEPS]))
        )
        best = pick_best_mapping([mirror, underlying])
        self.assertEqual(best.yahoo_ticker, "CRWD")
        self.assertTrue(best.is_independent)

    def test_no_overlap_yields_no_data(self):
        check = check_mapping(
            "XUSDT", "X", "derived", _series(_walk(STEPS)), _series(_walk(STEPS), start=date(2030, 1, 1))
        )
        self.assertEqual(check.status, NO_DATA)
        self.assertEqual(check.overlap_days, 0)


if __name__ == "__main__":
    unittest.main()
