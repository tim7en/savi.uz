"""savi.uz data ingestion package."""

from .config import get_alphavantage_api_key, get_fred_api_key, load_dotenv
from .data_sources import AlphaVantageClient, BinanceClient
from .macro_catalog import FRED_CATALOG, SeriesSpec, VintagePolicy
from .macro_sources import FredClient, GswCurveClient, NyFedRatesClient
from .macro_store import MacroStore
from .mapping_check import MappingCheck, check_mapping, pick_best_mapping
from .pipeline import MarketDataPipeline, build_uncorrelated_clusters
from .seed_groups import SEED_RISK_GROUPS
from .tradfi_universe import BinanceTradFiClient, TradFiInstrument, candidate_yahoo_tickers

__all__ = [
    "AlphaVantageClient",
    "BinanceClient",
    "BinanceTradFiClient",
    "FRED_CATALOG",
    "FredClient",
    "GswCurveClient",
    "MacroStore",
    "MappingCheck",
    "MarketDataPipeline",
    "NyFedRatesClient",
    "SEED_RISK_GROUPS",
    "SeriesSpec",
    "TradFiInstrument",
    "VintagePolicy",
    "build_uncorrelated_clusters",
    "candidate_yahoo_tickers",
    "check_mapping",
    "get_alphavantage_api_key",
    "get_fred_api_key",
    "load_dotenv",
    "pick_best_mapping",
]
