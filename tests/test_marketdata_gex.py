import tempfile
import unittest
from datetime import date
from pathlib import Path

from savi_uz.marketdata_gex import (
    ChainResponse, CreditUsage, GexStore, calculate_gex, enrich_contracts,
    model_gamma, normalise_contract, option_price,
)


def chain():
    return (
        {
            "optionSymbol": "SPY260918C00095000", "expiration": 1789761600,
            "side": "call", "strike": 95, "dte": 30, "volume": 2,
            "openInterest": 10, "underlyingPrice": 100, "updated": 1787356800,
            "iv": 0.20, "delta": 0.6, "gamma": 0.01,
        },
        {
            "optionSymbol": "SPY260918P00105000", "expiration": 1789761600,
            "side": "put", "strike": 105, "dte": 30, "volume": 3,
            "openInterest": 5, "underlyingPrice": 100, "updated": 1787356800,
            "iv": 0.30, "delta": -0.6, "gamma": 0.02,
        },
    )


class GexCalculationTests(unittest.TestCase):
    def test_iv_and_gamma_can_be_reconstructed_from_mid(self):
        years = 30 / 365.25
        mid = option_price("call", 100, 100, years, 0.04, 0.01, 0.25)
        source = ({
            "side": "call", "mid": mid, "underlyingPrice": 100,
            "strike": 100, "dte": 30, "openInterest": 10,
        },)
        enriched = enrich_contracts(source, risk_free=0.04, dividend_yield=0.01)
        self.assertEqual(enriched[0]["gammaSource"], "black_scholes_mid")
        self.assertAlmostEqual(enriched[0]["iv"], 0.25, places=6)
        expected = model_gamma(100, 100, years, 0.04, 0.01, 0.25)
        self.assertAlmostEqual(enriched[0]["gamma"], expected, places=9)

    def test_call_positive_put_negative_proxy(self):
        result = calculate_gex("SPY", "2026-08-18", chain())
        self.assertEqual(result[:4], ("SPY", "2026-08-18", 2, 2))
        self.assertEqual(result[5], 1_000)
        self.assertEqual(result[6], 1_000)
        self.assertEqual(result[7], 0)
        self.assertEqual(result[8], 2_000)
        self.assertEqual(result[9], 95)
        self.assertEqual(result[11], 105)
        self.assertAlmostEqual(result[12], 0.20 * 2 / 3 + 0.30 / 3)
        self.assertEqual(result[13], 0.5)

    def test_contract_normalisation(self):
        row = normalise_contract("spy", "2026-08-18", chain()[0])
        self.assertEqual(row[0:3], ("SPY", "2026-08-18", "SPY260918C00095000"))
        self.assertEqual(row[4], "call")
        self.assertEqual(row[5], 95.0)


class GexStoreTests(unittest.TestCase):
    def test_write_is_resumable_and_builds_daily_feature(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "options.db"
            response = ChainResponse("ok", chain(), CreditUsage(1, 99, 100, None))
            with GexStore(path) as store:
                store.write("SPY", date(2026, 8, 18), response)
                store.write("SPY", date(2026, 8, 18), response)
                self.assertIn(("SPY", "2026-08-18"), store.completed())
                self.assertEqual(store.connection.execute(
                    "SELECT COUNT(*) FROM option_contracts"
                ).fetchone()[0], 2)
                self.assertEqual(store.connection.execute(
                    "SELECT net_gex FROM daily_gex"
                ).fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
