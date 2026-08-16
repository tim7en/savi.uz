from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from savi_uz.risk_clustering import (
    average_linkage,
    cluster_resolution_curve,
    components_for_variance,
    correlation_matrix,
    distance_for_correlation,
    effective_number_of_bets,
    factor_betas,
    log_returns,
    max_correlation_to_others,
    residual_returns,
    select_diversified_basket,
)


def _factor_panel(seed: int = 7, days: int = 400) -> pd.DataFrame:
    """Two independent factors, two names loading on each, plus one loner."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2024-01-01", periods=days)
    tech = rng.normal(0, 0.01, days)
    metal = rng.normal(0, 0.01, days)
    columns = {
        "TECH_A": tech + rng.normal(0, 0.004, days),
        "TECH_B": tech + rng.normal(0, 0.004, days),
        "METAL_A": metal + rng.normal(0, 0.004, days),
        "METAL_B": metal + rng.normal(0, 0.004, days),
        "LONER": rng.normal(0, 0.01, days),
    }
    returns = pd.DataFrame(columns, index=index)
    return 100.0 * np.exp(returns.cumsum())


class LogReturnTests(unittest.TestCase):
    def test_returns_span_days_the_asset_did_not_trade(self):
        """A 7-day instrument on the index must not blank out Monday for a 5-day one."""
        index = pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"])
        prices = pd.DataFrame({"WEEKDAY": [100.0, np.nan, np.nan, 110.0]}, index=index)
        returns = log_returns(prices)
        self.assertEqual(returns["WEEKDAY"].notna().sum(), 1)
        self.assertAlmostEqual(returns.loc[index[3], "WEEKDAY"], np.log(1.1), places=9)

    def test_non_positive_prices_are_dropped(self):
        index = pd.bdate_range("2026-01-01", periods=3)
        returns = log_returns(pd.DataFrame({"A": [100.0, 0.0, 121.0]}, index=index))
        self.assertTrue(np.isnan(returns.loc[index[1], "A"]))


class CorrelationTests(unittest.TestCase):
    def test_matrix_is_positive_semidefinite_and_unit_diagonal(self):
        returns = log_returns(_factor_panel())
        corr, counts = correlation_matrix(returns, min_periods=30, shrinkage=0.1)
        np.testing.assert_allclose(np.diag(corr.to_numpy()), 1.0, atol=1e-9)
        self.assertGreaterEqual(np.linalg.eigvalsh(corr.to_numpy()).min(), -1e-9)
        self.assertEqual(counts.at["TECH_A", "TECH_B"], returns[["TECH_A", "TECH_B"]].dropna().shape[0] + 0.0)

    def test_shrinkage_pulls_correlations_toward_the_average(self):
        returns = log_returns(_factor_panel())
        plain, _ = correlation_matrix(returns, min_periods=30, shrinkage=0.0)
        shrunk, _ = correlation_matrix(returns, min_periods=30, shrinkage=0.5)
        self.assertLess(shrunk.at["TECH_A", "TECH_B"], plain.at["TECH_A", "TECH_B"])

    def test_rejects_out_of_range_shrinkage(self):
        returns = log_returns(_factor_panel())
        with self.assertRaises(ValueError):
            correlation_matrix(returns, shrinkage=1.0)


class ClusteringTests(unittest.TestCase):
    def setUp(self):
        self.returns = log_returns(_factor_panel())
        self.corr, _ = correlation_matrix(self.returns, min_periods=30, shrinkage=0.0)

    def test_linkage_recovers_the_generating_factors(self):
        clusters = average_linkage(self.corr).cut(distance_for_correlation(0.5))
        grouped = {frozenset(cluster) for cluster in clusters}
        self.assertIn(frozenset({"TECH_A", "TECH_B"}), grouped)
        self.assertIn(frozenset({"METAL_A", "METAL_B"}), grouped)
        self.assertIn(frozenset({"LONER"}), grouped)

    def test_every_label_lands_in_exactly_one_cluster(self):
        clusters = average_linkage(self.corr).cut(distance_for_correlation(0.2))
        flattened = [label for cluster in clusters for label in cluster]
        self.assertCountEqual(flattened, list(self.corr.columns))

    def test_leaf_order_is_a_permutation_of_the_labels(self):
        self.assertCountEqual(average_linkage(self.corr).ordered_labels(), list(self.corr.columns))

    def test_relaxing_the_threshold_never_increases_the_cluster_count(self):
        curve = cluster_resolution_curve(self.corr, (0.7, 0.5, 0.3, 0.1))
        counts = curve["clusters"].tolist()
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_single_asset_and_empty_inputs_are_handled(self):
        self.assertEqual(average_linkage(pd.DataFrame()).cut(0.5), [])
        lone = pd.DataFrame([[1.0]], index=["A"], columns=["A"])
        self.assertEqual(average_linkage(lone).cut(0.5), [["A"]])


class DiversificationTests(unittest.TestCase):
    def setUp(self):
        self.returns = log_returns(_factor_panel())
        self.corr, _ = correlation_matrix(self.returns, min_periods=30, shrinkage=0.0)

    def test_effective_bets_sits_between_one_and_asset_count(self):
        bets = effective_number_of_bets(self.corr)
        self.assertLess(bets, len(self.corr))
        self.assertGreater(bets, 1.0)

    def test_effective_bets_equals_n_for_an_identity_matrix(self):
        identity = pd.DataFrame(np.eye(4), index=list("ABCD"), columns=list("ABCD"))
        self.assertAlmostEqual(effective_number_of_bets(identity), 4.0, places=6)

    def test_components_for_variance_is_monotone_in_the_target(self):
        self.assertLessEqual(components_for_variance(self.corr, 0.5), components_for_variance(self.corr, 0.95))

    def test_basket_respects_the_pairwise_correlation_cap(self):
        scores = pd.Series({name: 1.0 for name in self.corr.columns})
        scores["TECH_A"] = 5.0
        basket = select_diversified_basket(self.corr, scores, max_abs_correlation=0.3)
        self.assertIn("TECH_A", basket)
        self.assertNotIn("TECH_B", basket)

    def test_basket_takes_at_most_one_name_per_group(self):
        scores = pd.Series({name: float(index) for index, name in enumerate(self.corr.columns)})
        groups = pd.Series({"TECH_A": 0, "TECH_B": 0, "METAL_A": 1, "METAL_B": 1, "LONER": 2})
        basket = select_diversified_basket(self.corr, scores, max_abs_correlation=1.0, groups=groups)
        self.assertEqual(len(basket), 3)
        self.assertCountEqual([groups[name] for name in basket], [0, 1, 2])

    def test_basket_honours_the_limit(self):
        scores = pd.Series({name: 1.0 for name in self.corr.columns})
        self.assertEqual(len(select_diversified_basket(self.corr, scores, 1.0, limit=2)), 2)

    def test_max_correlation_excludes_the_self_pair(self):
        worst = max_correlation_to_others(self.corr)
        self.assertTrue((worst < 1.0).all())
        self.assertGreater(worst["TECH_A"], 0.5)


class FactorTests(unittest.TestCase):
    def test_residualising_against_a_factor_removes_the_shared_move(self):
        returns = log_returns(_factor_panel())
        market = returns[["TECH_A"]].rename(columns={"TECH_A": "MKT"})
        before, _ = correlation_matrix(returns, min_periods=30, shrinkage=0.0)
        residual = residual_returns(returns, market)
        after, _ = correlation_matrix(residual, min_periods=30, shrinkage=0.0)
        self.assertGreater(before.at["TECH_A", "TECH_B"], 0.5)
        self.assertLess(abs(after.at["TECH_A", "TECH_B"]), 0.2)

    def test_beta_against_itself_is_one(self):
        returns = log_returns(_factor_panel())
        market = returns[["TECH_A"]].rename(columns={"TECH_A": "MKT"})
        self.assertAlmostEqual(factor_betas(returns, market).at["TECH_A", "MKT"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
