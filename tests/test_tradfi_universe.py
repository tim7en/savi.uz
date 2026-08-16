from __future__ import annotations

import json
import unittest
from datetime import date
from unittest.mock import patch

from savi_uz.seed_groups import SEED_RISK_GROUPS, seed_base_assets, seed_group_by_base
from savi_uz.tradfi_universe import (
    CURATED_YAHOO_TICKERS,
    BinanceTradFiClient,
    TradFiInstrument,
    candidate_yahoo_tickers,
)

EXCHANGE_INFO = {
    "symbols": [
        {
            "symbol": "AAPLUSDT",
            "contractType": "TRADIFI_PERPETUAL",
            "status": "TRADING",
            "baseAsset": "AAPL",
            "quoteAsset": "USDT",
            "underlyingType": "EQUITY",
            "underlyingSubType": ["TradFi"],
            "onboardDate": 1775483400000,
        },
        {
            "symbol": "HK0700USDT",
            "contractType": "TRADIFI_PERPETUAL",
            "status": "TRADING",
            "baseAsset": "HK0700",
            "quoteAsset": "USDT",
            "underlyingType": "HK_EQUITY",
            "underlyingSubType": ["TradFi"],
            "onboardDate": 1784689200000,
        },
        {
            "symbol": "DELISTEDUSDT",
            "contractType": "TRADIFI_PERPETUAL",
            "status": "SETTLING",
            "baseAsset": "DELISTED",
            "quoteAsset": "USDT",
            "underlyingType": "EQUITY",
            "underlyingSubType": ["TradFi"],
            "onboardDate": 0,
        },
        {
            "symbol": "BTCUSDT",
            "contractType": "PERPETUAL",
            "status": "TRADING",
            "baseAsset": "BTC",
            "quoteAsset": "USDT",
            "underlyingType": "COIN",
            "underlyingSubType": ["Layer-1"],
            "onboardDate": 1569398400000,
        },
    ]
}


def _instrument(base: str, underlying_type: str = "EQUITY") -> TradFiInstrument:
    return TradFiInstrument(
        binance_symbol=f"{base}USDT",
        base_asset=base,
        quote_asset="USDT",
        underlying_type=underlying_type,
        sub_types=("TradFi",),
        status="TRADING",
        onboard_date=date(2026, 4, 6),
    )


class ClientTests(unittest.TestCase):
    def test_only_trading_tradfi_perpetuals_are_returned(self):
        client = BinanceTradFiClient()
        with patch.object(BinanceTradFiClient, "_get", return_value=EXCHANGE_INFO):
            instruments = client.fetch_tradfi_instruments()
        self.assertEqual([i.binance_symbol for i in instruments], ["AAPLUSDT", "HK0700USDT"])
        self.assertEqual(instruments[0].onboard_date, date(2026, 4, 6))
        self.assertEqual(instruments[1].region, "HK")

    def test_halted_contracts_can_be_included_on_request(self):
        client = BinanceTradFiClient()
        with patch.object(BinanceTradFiClient, "_get", return_value=EXCHANGE_INFO):
            instruments = client.fetch_tradfi_instruments(include_halted=True)
        self.assertIn("DELISTEDUSDT", [i.binance_symbol for i in instruments])

    def test_daily_bars_parse_close_and_quote_volume(self):
        row = [1767225600000, "1", "2", "0.5", "1.5", "10", 1767311999999, "2500", 7, "5", "1250", "0"]
        client = BinanceTradFiClient()
        with patch.object(BinanceTradFiClient, "_get", return_value=[row]):
            bars = client.fetch_daily_bars("AAPLUSDT")
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars.closes[date(2026, 1, 1)], 1.5)
        self.assertEqual(bars.quote_volumes[date(2026, 1, 1)], 2500.0)

    def test_liquidity_snapshot_is_keyed_by_symbol(self):
        payload = [{"symbol": "AAPLUSDT", "lastPrice": "306.3", "quoteVolume": "2500", "count": "5",
                    "priceChangePercent": "1.2"}]
        client = BinanceTradFiClient()
        with patch.object(BinanceTradFiClient, "_get", return_value=payload):
            liquidity = client.fetch_24h_liquidity()
        self.assertEqual(liquidity["AAPLUSDT"].quote_volume_24h, 2500.0)
        self.assertEqual(liquidity["AAPLUSDT"].avg_trade_size, 500.0)

    def test_plain_http_base_url_is_refused(self):
        with self.assertRaises(ValueError):
            BinanceTradFiClient(base_url="http://fapi.binance.com/fapi/v1")

    def test_binance_error_payload_becomes_an_exception(self):
        client = BinanceTradFiClient()
        with patch("savi_uz.tradfi_universe.urlopen") as opener:
            opener.return_value.__enter__.return_value.read.return_value = json.dumps(
                {"code": -1121, "msg": "Invalid symbol."}
            ).encode()
            with self.assertRaises(ValueError) as caught:
                client.fetch_tradfi_instruments()
        self.assertIn("-1121", str(caught.exception))


class CandidateTickerTests(unittest.TestCase):
    def test_us_equity_derives_its_ticker_and_keeps_the_mirror_last(self):
        candidates = candidate_yahoo_tickers(_instrument("AAPL"))
        self.assertEqual(candidates[0], ("AAPL", "derived"))
        self.assertEqual(candidates[-1], ("AAPL-USD", "venue-mirror"))

    def test_curated_mapping_outranks_the_derived_guess(self):
        candidates = candidate_yahoo_tickers(_instrument("BRKB"))
        self.assertEqual(candidates[0], ("BRK-B", "curated"))

    def test_non_us_listing_does_not_derive_a_bare_ticker(self):
        candidates = candidate_yahoo_tickers(_instrument("HK0700", "HK_EQUITY"))
        self.assertEqual([ticker for ticker, _ in candidates], ["0700.HK", "HK0700-USD"])

    def test_search_results_are_inserted_before_the_mirror(self):
        candidates = candidate_yahoo_tickers(_instrument("MINIMAX", "HK_EQUITY"), ["2571.HK"])
        self.assertEqual(candidates, [("2571.HK", "search"), ("MINIMAX-USD", "venue-mirror")])

    def test_pre_ipo_contracts_have_no_candidates(self):
        self.assertEqual(candidate_yahoo_tickers(_instrument("OPENAI", "PREMARKET")), [])

    def test_candidates_are_deduplicated(self):
        candidates = candidate_yahoo_tickers(_instrument("TENCENT", "HK_EQUITY"), ["0700.HK", "0700.HK"])
        self.assertEqual(len(candidates), len({ticker for ticker, _ in candidates}))


class SeedGroupTests(unittest.TestCase):
    def test_every_base_maps_back_to_exactly_one_group(self):
        reverse = seed_group_by_base()
        self.assertEqual(len(reverse), len(seed_base_assets()))
        for label, bases in SEED_RISK_GROUPS.items():
            for base in bases:
                self.assertEqual(reverse[base], label)

    def test_curated_hong_kong_and_korea_tickers_carry_an_exchange_suffix(self):
        for base, ticker in CURATED_YAHOO_TICKERS.items():
            if base in {"BRKB"} or ticker.endswith("=F"):
                continue
            self.assertRegex(ticker, r"\.(HK|KS|SZ|SS)$", msg=f"{base} -> {ticker}")


if __name__ == "__main__":
    unittest.main()
