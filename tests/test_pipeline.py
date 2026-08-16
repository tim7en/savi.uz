from __future__ import annotations

import unittest

from savi_uz.data_sources import AlphaVantageClient
from savi_uz.pipeline import MarketDataPipeline, build_uncorrelated_clusters


class FakeAlphaVantageClient(AlphaVantageClient):
    def __init__(self):
        super().__init__(api_key="demo")

    def _get(self, **params):
        if params["function"] == "TIME_SERIES_INTRADAY":
            return {
                "Time Series (5min)": {
                    "2026-01-01 09:30:00": {
                        "1. open": "100",
                        "2. high": "102",
                        "3. low": "99",
                        "4. close": "101",
                        "5. volume": "1000",
                    }
                }
            }
        if params["function"] == "HISTORICAL_OPTIONS":
            return {"data": [{"contractID": "SPY_CALL_1"}]}
        if params["function"] == "FEDERAL_FUNDS_RATE":
            return {"data": [{"date": "2026-01-01", "value": "5.25"}]}
        if params["function"] == "TREASURY_YIELD":
            return {"data": [{"date": "2026-01-01", "value": "5.0"}]}
        return {"data": [{"date": "2026-01-01", "value": "1"}]}


class FakeBinanceClient:
    def fetch_tradfi_reference_symbols(self):
        return ["BTCUSDT", "ETHUSDT"]


class PipelineTests(unittest.TestCase):
    def test_fetch_intraday_ohlc_parses_numeric_values(self):
        rows = FakeAlphaVantageClient().fetch_intraday_ohlc(symbol="SPY")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["open"], 100.0)
        self.assertEqual(rows[0]["close"], 101.0)

    def test_uncorrelated_clusters_separates_high_correlation_assets(self):
        clusters = build_uncorrelated_clusters(
            {
                "A": [100, 101, 102, 103],
                "B": [200, 202, 204, 206],  # highly correlated with A
                "C": [100, 99, 100, 99],  # weakly correlated pattern
            },
            correlation_threshold=0.3,
        )
        cluster_by_symbol = {}
        for cluster_index, cluster in enumerate(clusters):
            for symbol in cluster:
                cluster_by_symbol[symbol] = cluster_index
        self.assertNotEqual(cluster_by_symbol["A"], cluster_by_symbol["B"])

    def test_pipeline_downloads_required_dataset_groups(self):
        pipeline = MarketDataPipeline(alpha_vantage=FakeAlphaVantageClient(), binance=FakeBinanceClient())
        result = pipeline.download_core_datasets()
        self.assertEqual(sorted(result.keys()), ["binance_reference_symbols", "equities_5min_ohlc", "macro", "options"])
        self.assertIn("SPY", result["equities_5min_ohlc"])
        self.assertIn("fed_funds_rate", result["macro"])


if __name__ == "__main__":
    unittest.main()
