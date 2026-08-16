"""External data source clients used by the ingestion pipeline."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen


class AlphaVantageClient:
    """Thin AlphaVantage API client for market and macro data collection."""

    def __init__(self, api_key: str, base_url: str = "https://www.alphavantage.co/query"):
        if not base_url.startswith("https://"):
            raise ValueError("AlphaVantage base_url must use https://")
        self.api_key = api_key
        self.base_url = base_url

    def _get(self, **params: str) -> dict[str, Any]:
        query = urlencode({**params, "apikey": self.api_key})
        with urlopen(f"{self.base_url}?{query}") as response:  # nosec B310
            data = json.loads(response.read().decode("utf-8"))
        if "Error Message" in data:
            raise ValueError(data["Error Message"])
        return data

    def fetch_intraday_ohlc(self, symbol: str, interval: str = "5min", outputsize: str = "full") -> list[dict[str, Any]]:
        payload = self._get(
            function="TIME_SERIES_INTRADAY",
            symbol=symbol,
            interval=interval,
            outputsize=outputsize,
            adjusted="true",
        )
        series_key = f"Time Series ({interval})"
        series = payload.get(series_key, {})
        records: list[dict[str, Any]] = []
        for timestamp, values in sorted(series.items()):
            records.append(
                {
                    "timestamp": timestamp,
                    "open": float(values["1. open"]),
                    "high": float(values["2. high"]),
                    "low": float(values["3. low"]),
                    "close": float(values["4. close"]),
                    "volume": float(values["5. volume"]),
                }
            )
        return records

    def fetch_options_chain(self, symbol: str, date: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, str] = {"function": "HISTORICAL_OPTIONS", "symbol": symbol}
        if date:
            params["date"] = date
        payload = self._get(**params)
        return list(payload.get("data", []))

    def fetch_fed_funds_rate(self, interval: str = "monthly") -> list[dict[str, Any]]:
        payload = self._get(function="FEDERAL_FUNDS_RATE", interval=interval)
        return list(payload.get("data", []))

    def fetch_treasury_yield(self, interval: str = "monthly", maturity: str = "3month") -> list[dict[str, Any]]:
        payload = self._get(
            function="TREASURY_YIELD",
            interval=interval,
            maturity=maturity,
        )
        return list(payload.get("data", []))

    def fetch_macro_series(self, function_name: str, interval: str = "monthly") -> list[dict[str, Any]]:
        payload = self._get(function=function_name, interval=interval)
        return list(payload.get("data", []))


class BinanceClient:
    """Client for Binance exchange metadata used as trad-FI reference universe."""

    def __init__(self, base_url: str = "https://api.binance.com/api/v3"):
        if not base_url.startswith("https://"):
            raise ValueError("Binance base_url must use https://")
        self.base_url = base_url

    def fetch_tradfi_reference_symbols(self) -> list[str]:
        with urlopen(f"{self.base_url}/exchangeInfo") as response:  # nosec B310
            payload = json.loads(response.read().decode("utf-8"))
        if "code" in payload and "msg" in payload:
            raise ValueError(f"Binance API error {payload['code']}: {payload['msg']}")
        symbols = []
        for item in payload.get("symbols", []):
            if item.get("status") != "TRADING":
                continue
            if item.get("quoteAsset") not in {"USDT", "USDC", "FDUSD", "TUSD"}:
                continue
            symbols.append(item["symbol"])
        return sorted(set(symbols))
