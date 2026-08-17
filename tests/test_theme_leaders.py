from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from savi_uz.theme_leaders import (
    COHERENT_FACTOR_STRENGTH,
    classify_theme,
    factor_strength,
    principal_component,
    select_leaders,
    summarise_theme,
)


def _equicorrelated(names: list[str], rho: float) -> pd.DataFrame:
    matrix = np.full((len(names), len(names)), rho)
    np.fill_diagonal(matrix, 1.0)
    return pd.DataFrame(matrix, index=names, columns=names)


class PrincipalComponentTests(unittest.TestCase):
    def test_identical_members_load_equally(self):
        corr = _equicorrelated(["A", "B", "C"], 0.9)
        loadings, explained = principal_component(corr)
        self.assertAlmostEqual(abs(loadings["A"]), abs(loadings["C"]), places=6)
        self.assertGreater(explained, 0.9)

    def test_sign_is_normalised_so_the_largest_loading_is_positive(self):
        corr = _equicorrelated(["A", "B"], 0.8)
        loadings, _ = principal_component(corr)
        self.assertGreater(loadings[loadings.abs().idxmax()], 0)

    def test_an_inverse_member_gets_a_negative_loading(self):
        """TMF and TBT are the same rates factor with opposite signs."""
        corr = pd.DataFrame(
            [[1.0, -0.95, 0.4], [-0.95, 1.0, -0.35], [0.4, -0.35, 1.0]],
            index=["TBT", "TMF", "UVXY"], columns=["TBT", "TMF", "UVXY"],
        )
        loadings, _ = principal_component(corr)
        self.assertLess(loadings["TBT"] * loadings["TMF"], 0)

    def test_an_empty_submatrix_does_not_raise(self):
        loadings, explained = principal_component(pd.DataFrame())
        self.assertTrue(loadings.empty)
        self.assertTrue(np.isnan(explained))


class FactorStrengthTests(unittest.TestCase):
    def test_it_recovers_the_common_correlation(self):
        """Under equicorrelation the adjusted statistic is exactly rho."""
        for rho in (0.2, 0.5, 0.8):
            for size in (2, 3, 8):
                names = [f"A{i}" for i in range(size)]
                _, explained = principal_component(_equicorrelated(names, rho))
                with self.subTest(rho=rho, size=size):
                    self.assertAlmostEqual(factor_strength(explained, size), rho, places=6)

    def test_uncorrelated_members_score_zero_at_any_size(self):
        """The raw variance share cannot; that is the whole point of rescaling."""
        for size in (2, 3, 8):
            names = [f"A{i}" for i in range(size)]
            _, explained = principal_component(_equicorrelated(names, 0.0))
            with self.subTest(size=size):
                self.assertAlmostEqual(explained, 1.0 / size, places=6)
                self.assertAlmostEqual(factor_strength(explained, size), 0.0, places=6)

    def test_a_two_member_theme_is_not_flattered_by_its_size(self):
        """Raw share says 0.50 for two unrelated names; strength says 0.00."""
        _, explained = principal_component(_equicorrelated(["A", "B"], 0.0))
        self.assertAlmostEqual(explained, 0.5, places=6)
        self.assertEqual(classify_theme(factor_strength(explained, 2), 2), "not-a-theme")

    def test_a_single_member_has_no_strength(self):
        self.assertTrue(np.isnan(factor_strength(1.0, 1)))


class ClassifyTests(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(classify_theme(0.70, 5), "coherent")
        self.assertEqual(classify_theme(0.35, 5), "loose")
        self.assertEqual(classify_theme(0.10, 5), "not-a-theme")

    def test_a_single_member_theme_is_not_scored(self):
        self.assertEqual(classify_theme(0.9, 1), "too-few-members")

    def test_nan_is_unmeasured(self):
        self.assertEqual(classify_theme(float("nan"), 4), "unmeasured")

    def test_an_inverse_pair_scores_as_a_strong_factor(self):
        """A signed average would call this -0.95; it is a tight factor."""
        corr = pd.DataFrame([[1.0, -0.95], [-0.95, 1.0]], index=["TBT", "TMF"], columns=["TBT", "TMF"])
        _, explained = principal_component(corr)
        strength = factor_strength(explained, 2)
        self.assertAlmostEqual(strength, 0.95, places=6)
        self.assertEqual(classify_theme(strength, 2), "coherent")
        self.assertGreater(strength, COHERENT_FACTOR_STRENGTH)


class SelectLeadersTests(unittest.TestCase):
    def setUp(self):
        self.loadings = pd.Series({"A": 0.9, "B": -0.7, "C": 0.3, "D": 0.05})
        self.liquidity = pd.Series({"A": 8.0, "B": 7.0, "C": 6.0, "D": 5.0})

    def test_ranking_is_by_absolute_loading(self):
        """An inverse member still represents the factor."""
        self.assertEqual(select_leaders(self.loadings, self.liquidity, count=3), ("A", "B", "C"))

    def test_count_is_respected(self):
        self.assertEqual(select_leaders(self.loadings, self.liquidity, count=2), ("A", "B"))

    def test_liquidity_floor_excludes_thin_names(self):
        picks = select_leaders(self.loadings, self.liquidity, count=3, min_liquidity=6.5)
        self.assertEqual(picks, ("A", "B"))

    def test_a_floor_that_would_empty_the_theme_is_relaxed(self):
        picks = select_leaders(self.loadings, self.liquidity, count=2, min_liquidity=99.0)
        self.assertEqual(picks, ("A", "B"))

    def test_empty_loadings_return_nothing(self):
        self.assertEqual(select_leaders(pd.Series(dtype=float), self.liquidity), ())


class SummariseThemeTests(unittest.TestCase):
    def _frames(self, names: list[str], rho: float):
        corr = _equicorrelated(names, rho)
        liquidity = pd.Series({name: 6.0 for name in names})
        metadata = pd.DataFrame(
            {"base_asset": [n.replace("USDT", "") for n in names], "region": ["US"] * len(names)},
            index=names,
        )
        return corr, liquidity, metadata

    def test_a_coherent_theme_returns_ranked_leaders(self):
        names = ["AUSDT", "BUSDT", "CUSDT", "DUSDT"]
        corr, liquidity, metadata = self._frames(names, 0.7)
        summary = summarise_theme("Test", names, corr, liquidity, metadata, {}, count=3)

        self.assertEqual(summary.verdict, "coherent")
        self.assertAlmostEqual(summary.factor_strength, 0.7, places=6)
        self.assertEqual(len(summary.leaders), 3)
        self.assertTrue(summary.is_real)

    def test_members_absent_from_the_panel_are_dropped(self):
        names = ["AUSDT", "BUSDT"]
        corr, liquidity, metadata = self._frames(names, 0.6)
        summary = summarise_theme(
            "Test", names + ["MISSINGUSDT"], corr, liquidity, metadata, {}, count=3
        )
        self.assertEqual(summary.members, ("AUSDT", "BUSDT"))
        self.assertEqual(len(summary.leaders), 2)

    def test_a_theme_with_no_surviving_members_is_flagged(self):
        corr, liquidity, metadata = self._frames(["AUSDT"], 1.0)
        summary = summarise_theme("Test", ["GONEUSDT"], corr, liquidity, metadata, {}, count=3)
        self.assertEqual(summary.verdict, "no-members")
        self.assertEqual(summary.leaders, ())

    def test_proxy_details_are_attached_for_non_us_members(self):
        names = ["AUSDT", "BUSDT"]
        corr, liquidity, metadata = self._frames(names, 0.8)
        metadata.loc["AUSDT", "region"] = "HK"
        summary = summarise_theme(
            "Test", names, corr, liquidity, metadata, {"A": ("TCEHY", "strong")}, count=2
        )
        leader = next(x for x in summary.leaders if x.base_asset == "A")
        self.assertEqual(leader.us_proxy, "TCEHY")
        self.assertEqual(leader.proxy_verdict, "strong")


if __name__ == "__main__":
    unittest.main()
