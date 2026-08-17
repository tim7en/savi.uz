from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from savi_uz.proxy_tracking import (
    SPECIFIC_CORRELATION,
    best_lagged_correlation,
    classify,
    measure,
    rank_candidates,
)
from savi_uz.us_proxy_map import (
    MARKET_FACTOR,
    PROXY_KINDS,
    US_PROXY_CANDIDATES,
    candidates_for,
    proxy_tickers,
)

BUSINESS_DAYS = pd.bdate_range("2024-01-01", periods=400)


def _prices(returns: np.ndarray, start: float = 100.0) -> pd.Series:
    return pd.Series(start * np.exp(np.cumsum(returns)), index=BUSINESS_DAYS[: len(returns)])


class ProxyCatalogTests(unittest.TestCase):
    def test_every_candidate_declares_a_known_kind(self):
        for base, candidates in US_PROXY_CANDIDATES.items():
            for candidate in candidates:
                with self.subTest(base=base, ticker=candidate.ticker):
                    self.assertIn(candidate.kind, PROXY_KINDS)

    def test_every_candidate_carries_a_rationale(self):
        for base, candidates in US_PROXY_CANDIDATES.items():
            for candidate in candidates:
                with self.subTest(base=base, ticker=candidate.ticker):
                    self.assertTrue(candidate.rationale.strip())

    def test_no_duplicate_tickers_within_one_contract(self):
        for base, candidates in US_PROXY_CANDIDATES.items():
            tickers = [candidate.ticker for candidate in candidates]
            with self.subTest(base):
                self.assertEqual(len(tickers), len(set(tickers)))

    def test_the_market_factor_is_always_downloaded(self):
        self.assertIn(MARKET_FACTOR, proxy_tickers())

    def test_pre_ipo_names_declare_no_candidates(self):
        """Anthropic and OpenAI have no listing, and saying so beats a bad proxy."""
        self.assertEqual(candidates_for("ANTHROPIC"), ())
        self.assertEqual(candidates_for("OPENAI"), ())

    def test_unknown_base_returns_no_candidates(self):
        self.assertEqual(candidates_for("NOT_A_CONTRACT"), ())

    def test_us_bases_are_absent_because_they_proxy_to_themselves(self):
        for base in ("AAPL", "MSFT", "NVDA"):
            with self.subTest(base):
                self.assertNotIn(base, US_PROXY_CANDIDATES)


class LaggedCorrelationTests(unittest.TestCase):
    def test_a_leading_proxy_is_detected_at_lag_plus_one(self):
        """Asia closes before the US, so a real link shows up as the US leading."""
        rng = np.random.default_rng(0)
        signal = rng.normal(0, 0.01, 300)
        proxy = pd.Series(signal, index=BUSINESS_DAYS[:300])
        # The underlying only absorbs the proxy's move on the following day.
        underlying = pd.Series(np.concatenate([[0.0], signal[:-1]]), index=BUSINESS_DAYS[:300])

        best, lag, same_day, _ = best_lagged_correlation(underlying, proxy)
        self.assertEqual(lag, 1)
        self.assertGreater(best, 0.95)
        self.assertLess(abs(same_day), 0.2)

    def test_a_synchronous_proxy_wins_at_lag_zero(self):
        rng = np.random.default_rng(1)
        signal = rng.normal(0, 0.01, 300)
        series = pd.Series(signal, index=BUSINESS_DAYS[:300])
        best, lag, same_day, _ = best_lagged_correlation(series, series)
        self.assertEqual(lag, 0)
        self.assertAlmostEqual(best, 1.0, places=6)
        self.assertAlmostEqual(same_day, 1.0, places=6)

    def test_unrelated_series_correlate_near_zero(self):
        rng = np.random.default_rng(2)
        left = pd.Series(rng.normal(0, 0.01, 300), index=BUSINESS_DAYS[:300])
        right = pd.Series(rng.normal(0, 0.01, 300), index=BUSINESS_DAYS[:300])
        best, _, _, _ = best_lagged_correlation(left, right)
        self.assertLess(best, 0.3)

    def test_a_constant_series_does_not_raise(self):
        flat = pd.Series(np.zeros(300), index=BUSINESS_DAYS[:300])
        rng = np.random.default_rng(3)
        moving = pd.Series(rng.normal(0, 0.01, 300), index=BUSINESS_DAYS[:300])
        best, lag, _, _ = best_lagged_correlation(flat, moving)
        self.assertTrue(np.isnan(best))
        self.assertEqual(lag, 0)


class ClassifyTests(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(classify(0.95, 0.90, 80)[0], "strong")
        self.assertEqual(classify(0.65, 0.60, 80)[0], "usable")
        self.assertEqual(classify(0.40, 0.35, 80)[0], "weak")
        self.assertEqual(classify(0.10, 0.05, 80)[0], "poor")

    def test_a_high_correlation_that_is_all_market_beta_is_demoted(self):
        """A proxy that only co-moves via SPY does not hedge the name."""
        verdict, note = classify(0.85, SPECIFIC_CORRELATION - 0.1, 80)
        self.assertEqual(verdict, "market-beta")
        self.assertIn("market beta", note)

    def test_a_weak_correlation_is_not_promoted_by_the_beta_check(self):
        verdict, _ = classify(0.35, 0.05, 80)
        self.assertEqual(verdict, "weak")

    def test_thin_history_is_refused_a_verdict(self):
        verdict, note = classify(0.99, 0.99, 5)
        self.assertEqual(verdict, "insufficient")
        self.assertIn("5 overlapping weeks", note)

    def test_nan_correlation_is_insufficient(self):
        self.assertEqual(classify(float("nan"), 0.5, 80)[0], "insufficient")


class MeasureTests(unittest.TestCase):
    def _market(self, n: int = 400) -> pd.Series:
        rng = np.random.default_rng(7)
        return _prices(rng.normal(0.0003, 0.008, n))

    def test_a_perfect_tracker_scores_strong_with_beta_one(self):
        rng = np.random.default_rng(11)
        returns = rng.normal(0.0002, 0.012, 400)
        prices = _prices(returns)
        result = measure("HK0700", "0700.HK", prices, "TCEHY", prices, self._market())

        self.assertEqual(result.verdict, "strong")
        self.assertAlmostEqual(result.weekly_corr, 1.0, places=6)
        self.assertAlmostEqual(result.beta, 1.0, places=6)
        self.assertAlmostEqual(result.r_squared, 1.0, places=6)

    def test_a_proxy_that_is_only_market_beta_is_labelled_as_such(self):
        rng = np.random.default_rng(13)
        market_returns = rng.normal(0.0003, 0.010, 400)
        market = _prices(market_returns)
        # Both sides are the market plus their own independent noise, so they
        # correlate through SPY and share nothing specific.
        left = _prices(market_returns + rng.normal(0, 0.004, 400))
        right = _prices(market_returns + rng.normal(0, 0.004, 400))
        result = measure("X", "X.KS", left, "EWY", right, market)

        self.assertGreater(result.weekly_corr, 0.6)
        self.assertLess(result.residual_corr, SPECIFIC_CORRELATION)
        self.assertEqual(result.verdict, "market-beta")

    def test_beta_is_recovered_from_a_scaled_proxy(self):
        rng = np.random.default_rng(17)
        base = rng.normal(0.0002, 0.010, 400)
        proxy = _prices(base)
        underlying = _prices(base * 2.0)
        result = measure("LEV", "LEV.HK", underlying, "P", proxy, self._market())
        self.assertAlmostEqual(result.beta, 2.0, places=4)

    def test_mismatched_calendars_still_produce_weekly_overlap(self):
        """HK and US keep different holidays; weekly resampling must survive it."""
        rng = np.random.default_rng(19)
        prices = _prices(rng.normal(0.0002, 0.012, 400))
        thinned = prices.iloc[::2]  # underlying trades every other day
        result = measure("HK", "0700.HK", thinned, "TCEHY", prices, self._market())
        self.assertGreater(result.overlap_weeks, 50)
        self.assertGreater(result.weekly_corr, 0.9)

    def test_empty_history_is_insufficient_rather_than_an_error(self):
        empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        result = measure("X", "X", empty, "P", empty, self._market())
        self.assertEqual(result.verdict, "insufficient")


class RankTests(unittest.TestCase):
    def _tracking(self, proxy: str, kind: str, weekly: float):
        market = _prices(np.random.default_rng(5).normal(0, 0.008, 200))
        result = measure("B", "U", market, proxy, market, market, kind=kind)
        return type(result)(**{**result.__dict__, "weekly_corr": weekly})

    def test_higher_weekly_correlation_ranks_first(self):
        ranked = rank_candidates(
            [self._tracking("EWY", "country", 0.60), self._tracking("TCEHY", "adr", 0.92)]
        )
        self.assertEqual(ranked[0].proxy, "TCEHY")

    def test_ties_break_toward_the_closer_structural_match(self):
        """An ADR of the same company beats a country ETF that measures the same."""
        ranked = rank_candidates(
            [self._tracking("EWY", "country", 0.90), self._tracking("TCEHY", "adr", 0.90)]
        )
        self.assertEqual(ranked[0].kind, "adr")

    def test_nan_correlations_sort_last(self):
        ranked = rank_candidates(
            [self._tracking("A", "adr", float("nan")), self._tracking("B", "country", 0.4)]
        )
        self.assertEqual(ranked[0].proxy, "B")


if __name__ == "__main__":
    unittest.main()
