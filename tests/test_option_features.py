from __future__ import annotations

import math
import unittest

from savi_uz.option_features import (
    Contract,
    aggregate_gamma,
    black_scholes_greeks,
    snapshot_features,
)


def chain(spot=100.0, dte=7, base_iv=0.20, smirk=0.0, call_oi=100.0, put_oi=100.0):
    """A symmetric chain, optionally tilted so puts carry more volatility."""
    rows = []
    for strike in range(90, 111, 2):
        moneyness = strike / spot
        for side in ("call", "put"):
            iv = base_iv + (smirk * (1.0 - moneyness) if side == "put" else 0.0)
            rows.append(Contract(side=side, strike=float(strike), dte=dte, iv=iv,
                                 gamma=None,
                                 open_interest=call_oi if side == "call" else put_oi,
                                 volume=10.0))
    return rows


class GreekTests(unittest.TestCase):
    def test_at_the_money_call_delta_is_about_a_half(self):
        delta, _, _ = black_scholes_greeks(100, 100, 0.25, 0.2, "call")
        self.assertAlmostEqual(delta, 0.52, places=1)

    def test_put_and_call_delta_differ_by_one(self):
        c = black_scholes_greeks(100, 105, 0.25, 0.2, "call")[0]
        p = black_scholes_greeks(100, 105, 0.25, 0.2, "put")[0]
        self.assertAlmostEqual(c - p, 1.0, places=9)

    def test_gamma_is_identical_for_a_call_and_a_put(self):
        c = black_scholes_greeks(100, 105, 0.25, 0.2, "call")[1]
        p = black_scholes_greeks(100, 105, 0.25, 0.2, "put")[1]
        self.assertAlmostEqual(c, p, places=12)

    def test_gamma_peaks_near_the_money(self):
        atm = black_scholes_greeks(100, 100, 0.25, 0.2, "call")[1]
        wing = black_scholes_greeks(100, 130, 0.25, 0.2, "call")[1]
        self.assertGreater(atm, wing)

    def test_degenerate_inputs_return_nothing(self):
        for args in ((0, 100, .25, .2), (100, 0, .25, .2),
                     (100, 100, 0, .2), (100, 100, .25, 0)):
            self.assertIsNone(black_scholes_greeks(*args, "call"))


class SnapshotTests(unittest.TestCase):
    def test_a_flat_chain_has_no_skew(self):
        out = snapshot_features(chain(), 100.0)
        self.assertAlmostEqual(out["skew_moneyness"], 0.0, places=9)

    def test_richer_downside_volatility_produces_positive_skew(self):
        out = snapshot_features(chain(smirk=0.5), 100.0)
        self.assertGreater(out["skew_moneyness"], 0.0)
        self.assertGreater(out["skew_25delta"], 0.0)

    def test_atm_volatility_recovers_the_level_it_was_built_with(self):
        out = snapshot_features(chain(base_iv=0.27), 100.0)
        self.assertAlmostEqual(out["atm_iv"], 0.27, places=6)

    def test_gamma_balance_is_positive_when_calls_dominate_open_interest(self):
        heavy_calls = snapshot_features(chain(call_oi=500, put_oi=50), 100.0)
        heavy_puts = snapshot_features(chain(call_oi=50, put_oi=500), 100.0)
        self.assertGreater(heavy_calls["gamma_balance"], 0)
        self.assertLess(heavy_puts["gamma_balance"], 0)

    def test_gamma_balance_is_bounded_by_one(self):
        for oi in ((500, 50), (50, 500), (100, 100)):
            out = snapshot_features(chain(call_oi=oi[0], put_oi=oi[1]), 100.0)
            self.assertLessEqual(abs(out["gamma_balance"]), 1.0 + 1e-9)

    def test_put_call_ratios_follow_the_open_interest(self):
        out = snapshot_features(chain(call_oi=100, put_oi=250), 100.0)
        self.assertAlmostEqual(out["put_call_oi"], 2.5, places=6)

    def test_a_balanced_book_has_no_gamma_flip_nearby(self):
        # Calls and puts carry equal open interest at every strike, so net gamma
        # never changes sign across the grid.
        out = snapshot_features(chain(call_oi=100, put_oi=100), 100.0)
        self.assertIsNone(out["gamma_flip_distance"])

    def test_zero_dte_share_is_zero_when_nothing_expires_today(self):
        out = snapshot_features(chain(dte=7), 100.0)
        self.assertAlmostEqual(out["zero_dte_share"], 0.0, places=9)

    def test_zero_dte_share_is_one_when_everything_expires_today(self):
        out = snapshot_features(chain(dte=0), 100.0)
        self.assertAlmostEqual(out["zero_dte_share"], 1.0, places=6)

    def test_an_empty_chain_yields_no_features(self):
        self.assertEqual(snapshot_features([], 100.0), {})
        self.assertEqual(snapshot_features(chain(), 0.0), {})

    def test_missing_volatility_does_not_crash_the_snapshot(self):
        rows = chain()
        rows.append(Contract("call", 100.0, 7, None, None, 50.0, 1.0))
        out = snapshot_features(rows, 100.0)
        self.assertIsNotNone(out["atm_iv"])


class AggregateGammaTests(unittest.TestCase):
    def rows(self, call_gamma=0.01, put_gamma=0.01, call_oi=100.0, put_oi=100.0):
        return [
            Contract("call", 100.0, 7, 0.2, call_gamma, call_oi, 0.0),
            Contract("put", 100.0, 7, 0.2, put_gamma, put_oi, 0.0),
        ]

    def test_calls_add_and_puts_subtract(self):
        out = aggregate_gamma(self.rows(call_oi=200.0, put_oi=100.0), 100.0)
        self.assertGreater(out["net_gex"], 0)
        out = aggregate_gamma(self.rows(call_oi=100.0, put_oi=200.0), 100.0)
        self.assertLess(out["net_gex"], 0)

    def test_a_matched_book_nets_to_zero_but_is_not_absolutely_zero(self):
        out = aggregate_gamma(self.rows(), 100.0)
        self.assertAlmostEqual(out["net_gex"], 0.0, places=9)
        self.assertGreater(out["absolute_gex"], 0.0)

    def test_the_dollar_figure_matches_the_stated_convention(self):
        # one call: gamma 0.01, OI 100, x100 multiplier, spot 100, per 1% move
        out = aggregate_gamma([Contract("call", 100.0, 7, 0.2, 0.01, 100.0, 0.0)],
                              100.0)
        self.assertAlmostEqual(out["call_gex"], 0.01 * 100 * 100 * 100 * 100 * 0.01)

    def test_exposure_scales_with_the_square_of_spot(self):
        one = aggregate_gamma(self.rows(put_oi=0.0), 100.0)["net_gex"]
        two = aggregate_gamma(self.rows(put_oi=0.0), 200.0)["net_gex"]
        self.assertAlmostEqual(two / one, 4.0, places=6)

    def test_missing_gamma_or_interest_counts_as_unusable_not_zero(self):
        rows = [Contract("call", 100.0, 7, 0.2, None, 100.0, 0.0),
                Contract("put", 100.0, 7, 0.2, 0.01, 0.0, 0.0),
                Contract("call", 100.0, 7, 0.2, 0.01, 50.0, 0.0)]
        out = aggregate_gamma(rows, 100.0)
        self.assertEqual(out["contracts"], 3)
        self.assertEqual(out["usable"], 1)

    def test_a_nonsense_spot_produces_no_exposure(self):
        out = aggregate_gamma(self.rows(), 0.0)
        self.assertEqual(out["usable"], 0)
        self.assertEqual(out["net_gex"], 0.0)


if __name__ == "__main__":
    unittest.main()
