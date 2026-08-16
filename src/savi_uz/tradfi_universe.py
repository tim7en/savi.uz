"""Binance trad-FI perpetual universe discovery and Yahoo Finance symbol mapping."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Iterable, Sequence
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

FUTURES_BASE_URL = "https://fapi.binance.com/fapi/v1"
TRADFI_CONTRACT_TYPE = "TRADIFI_PERPETUAL"

REGION_BY_UNDERLYING_TYPE = {
    "EQUITY": "US",
    "HK_EQUITY": "HK",
    "KR_EQUITY": "KR",
    "COMMODITY": "COMMODITY",
    "INDEX": "INDEX",
    "PREMARKET": "PRE_IPO",
}

#: Bases with no public listing to map onto; the perp is the only tradable venue.
UNLISTED_BASES = frozenset({"ANTHROPIC", "OPENAI"})

#: Bases whose Yahoo ticker cannot be derived by stripping the quote asset.
CURATED_YAHOO_TICKERS = {
    # Commodities -> CME/COMEX/NYMEX continuous front-month futures.
    "XAU": "GC=F",
    "XAG": "SI=F",
    "XPT": "PL=F",
    "XPD": "PA=F",
    "CL": "CL=F",
    "BZ": "BZ=F",
    "NATGAS": "NG=F",
    "COPPER": "HG=F",
    # US equities with share-class punctuation.
    "BRKB": "BRK-B",
    # Hong Kong listings.
    "HK0700": "0700.HK",
    "HK1810": "1810.HK",
    "TENCENT": "0700.HK",
    "MEITUAN": "3690.HK",
    "KUAISHOU": "1024.HK",
    "POPMART": "9992.HK",
    "GIGADEV": "3986.HK",
    "ZHONGJI": "300308.SZ",
    # Korean listings.
    "SAMSUNG": "005930.KS",
    "SAMSUNGEM": "009150.KS",
    "SKHYNIX": "000660.KS",
    "HYUNDAI": "005380.KS",
    "NAVER": "035420.KS",
    "LGELECTRONICS": "066570.KS",
    "HANMI": "042700.KS",
    "KODEX200": "069500.KS",
}

#: Free-text queries used to resolve a base through Yahoo search when the
#: derived ticker fails validation. Keyed by Binance base asset.
YAHOO_SEARCH_HINTS = {
    "CSOPSAMSUNG2L": "CSOP Samsung Electronics Daily 2x Long",
    "CSOPSKHYNIX2L": "CSOP SK Hynix Daily 2x Long",
    "GIGADEV": "GigaDevice Semiconductor",
    "MINIMAX": "MiniMax AI",
    "ZHIPU": "Zhipu Knowledge Atlas",
    "ZHONGJI": "Zhongji Innolight",
}


@dataclass(frozen=True)
class TradFiInstrument:
    """A single Binance trad-FI perpetual contract."""

    binance_symbol: str
    base_asset: str
    quote_asset: str
    underlying_type: str
    sub_types: tuple[str, ...]
    status: str
    onboard_date: date | None

    @property
    def region(self) -> str:
        return REGION_BY_UNDERLYING_TYPE.get(self.underlying_type, "OTHER")

    @property
    def is_pre_ipo(self) -> bool:
        return "Pre-IPO" in self.sub_types or self.underlying_type == "PREMARKET"

    @property
    def is_etf(self) -> bool:
        return "ETF" in self.sub_types


@dataclass(frozen=True)
class Liquidity:
    """Rolling 24h trading activity for one contract, in quote-asset units."""

    binance_symbol: str
    last_price: float
    quote_volume_24h: float
    trade_count_24h: int
    price_change_pct_24h: float

    @property
    def avg_trade_size(self) -> float:
        if self.trade_count_24h <= 0:
            return 0.0
        return self.quote_volume_24h / self.trade_count_24h


@dataclass
class DailyBars:
    """Daily OHLCV history for one contract, keyed by UTC open date."""

    binance_symbol: str
    closes: dict[date, float] = field(default_factory=dict)
    quote_volumes: dict[date, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.closes)


class BinanceTradFiClient:
    """Read-only client for the Binance USDⓈ-M futures trad-FI universe."""

    def __init__(self, base_url: str = FUTURES_BASE_URL, timeout: float = 30.0):
        if not base_url.startswith("https://"):
            raise ValueError("Binance futures base_url must use https://")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, **params: Any) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        if params:
            url = f"{url}?{urlencode(params)}"
        try:
            with urlopen(url, timeout=self.timeout) as response:  # nosec B310
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise ValueError(f"Binance API request failed: HTTP {exc.code} for {path}") from exc
        if isinstance(payload, dict) and "code" in payload and "msg" in payload:
            raise ValueError(f"Binance API error {payload['code']}: {payload['msg']}")
        return payload

    def fetch_tradfi_instruments(self, include_halted: bool = False) -> list[TradFiInstrument]:
        payload = self._get("exchangeInfo")
        instruments: list[TradFiInstrument] = []
        for item in payload.get("symbols", []):
            if item.get("contractType") != TRADFI_CONTRACT_TYPE:
                continue
            if not include_halted and item.get("status") != "TRADING":
                continue
            instruments.append(
                TradFiInstrument(
                    binance_symbol=item["symbol"],
                    base_asset=item["baseAsset"],
                    quote_asset=item["quoteAsset"],
                    underlying_type=item.get("underlyingType", "UNKNOWN"),
                    sub_types=tuple(item.get("underlyingSubType") or ()),
                    status=item.get("status", "UNKNOWN"),
                    onboard_date=_epoch_ms_to_date(item.get("onboardDate")),
                )
            )
        return sorted(instruments, key=lambda instrument: instrument.binance_symbol)

    def fetch_24h_liquidity(self) -> dict[str, Liquidity]:
        payload = self._get("ticker/24hr")
        liquidity: dict[str, Liquidity] = {}
        for item in payload:
            symbol = item["symbol"]
            liquidity[symbol] = Liquidity(
                binance_symbol=symbol,
                last_price=float(item.get("lastPrice", 0.0)),
                quote_volume_24h=float(item.get("quoteVolume", 0.0)),
                trade_count_24h=int(item.get("count", 0)),
                price_change_pct_24h=float(item.get("priceChangePercent", 0.0)),
            )
        return liquidity

    def fetch_daily_bars(self, symbol: str, limit: int = 500) -> DailyBars:
        rows = self._get("klines", symbol=symbol, interval="1d", limit=min(limit, 1500))
        bars = DailyBars(binance_symbol=symbol)
        for row in rows:
            bar_date = _epoch_ms_to_date(row[0])
            if bar_date is None:
                continue
            bars.closes[bar_date] = float(row[4])
            bars.quote_volumes[bar_date] = float(row[7])
        return bars

    def fetch_daily_bars_bulk(
        self,
        symbols: Sequence[str],
        limit: int = 500,
        max_workers: int = 6,
    ) -> dict[str, DailyBars]:
        """Fetch daily bars concurrently, staying well inside the futures weight budget.

        One contract that refuses to serve klines must not sink the whole universe,
        so failures come back as empty histories.
        """

        def fetch(symbol: str) -> DailyBars:
            try:
                return self.fetch_daily_bars(symbol, limit=limit)
            except (ValueError, OSError):
                return DailyBars(binance_symbol=symbol)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return {bars.binance_symbol: bars for bars in pool.map(fetch, symbols)}


def _epoch_ms_to_date(value: Any) -> date | None:
    if value in (None, 0):
        return None
    return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc).date()


def derived_yahoo_ticker(instrument: TradFiInstrument) -> str:
    """Yahoo ticker implied by the Binance base asset, before any curation."""
    return instrument.base_asset.upper()


def venue_mirror_ticker(instrument: TradFiInstrument) -> str:
    """Yahoo's mirror of the Binance derivative itself (e.g. ``ZHIPU-USD``).

    This is the same price feed we are trying to validate, so it is only a
    last-resort source: it gives a usable history but no independent check.
    """
    return f"{instrument.base_asset.upper()}-USD"


def candidate_yahoo_tickers(
    instrument: TradFiInstrument,
    search_results: Iterable[str] = (),
) -> list[tuple[str, str]]:
    """Ordered ``(ticker, source)`` candidates to try for one instrument."""
    base = instrument.base_asset.upper()
    if base in UNLISTED_BASES:
        return []

    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(ticker: str, source: str) -> None:
        if ticker and ticker not in seen:
            seen.add(ticker)
            candidates.append((ticker, source))

    if base in CURATED_YAHOO_TICKERS:
        add(CURATED_YAHOO_TICKERS[base], "curated")
    if instrument.region == "US":
        add(derived_yahoo_ticker(instrument), "derived")
    for ticker in search_results:
        add(ticker.upper(), "search")
    add(venue_mirror_ticker(instrument), "venue-mirror")
    return candidates
