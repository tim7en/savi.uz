"""Dataset assembly and clustering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

from .data_sources import AlphaVantageClient, BinanceClient


def _returns(series: list[float]) -> list[float]:
    returns: list[float] = []
    for idx in range(1, len(series)):
        prev = series[idx - 1]
        current = series[idx]
        if prev == 0:
            continue
        returns.append((current - prev) / prev)
    return returns


def _corr(x: list[float], y: list[float]) -> float:
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    x = x[:n]
    y = y[:n]
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    cov = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
    x_var = sum((i - x_mean) ** 2 for i in x)
    y_var = sum((i - y_mean) ** 2 for i in y)
    if x_var == 0 or y_var == 0:
        return 0.0
    return cov / sqrt(x_var * y_var)


def build_uncorrelated_clusters(
    close_prices_by_symbol: dict[str, list[float]],
    correlation_threshold: float = 0.35,
) -> list[list[str]]:
    """Greedy clustering where cluster members stay below absolute correlation threshold."""
    symbols = sorted(close_prices_by_symbol.keys())
    symbol_returns = {s: _returns(close_prices_by_symbol[s]) for s in symbols}

    clusters: list[list[str]] = []
    for symbol in symbols:
        assigned = False
        for cluster in clusters:
            if all(abs(_corr(symbol_returns[symbol], symbol_returns[other])) <= correlation_threshold for other in cluster):
                cluster.append(symbol)
                assigned = True
                break
        if not assigned:
            clusters.append([symbol])
    return clusters


@dataclass
class MarketDataPipeline:
    alpha_vantage: AlphaVantageClient
    binance: BinanceClient

    def download_core_datasets(
        self,
        equity_symbols: tuple[str, ...] = ("SPY", "QQQ"),
    ) -> dict[str, Any]:
        equities = {
            symbol: self.alpha_vantage.fetch_intraday_ohlc(symbol=symbol, interval="5min", outputsize="full")
            for symbol in equity_symbols
        }
        options = {symbol: self.alpha_vantage.fetch_options_chain(symbol=symbol) for symbol in equity_symbols}
        macro = {
            "fed_funds_rate": self.alpha_vantage.fetch_fed_funds_rate(interval="monthly"),
            "predicted_rates_proxy": self.alpha_vantage.fetch_predicted_rates_proxy(interval="monthly", maturity="3month"),
            "cpi": self.alpha_vantage.fetch_macro_series("CPI", interval="monthly"),
            "unemployment": self.alpha_vantage.fetch_macro_series("UNEMPLOYMENT", interval="monthly"),
        }
        binance_reference_symbols = self.binance.fetch_tradfi_reference_symbols()
        return {
            "equities_5min_ohlc": equities,
            "options": options,
            "macro": macro,
            "binance_reference_symbols": binance_reference_symbols,
        }
