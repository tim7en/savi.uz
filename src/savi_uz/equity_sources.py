"""Clients for earnings and valuation data: Shiller, SEC XBRL, Yahoo, Alpha Vantage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from savi_uz.equity_catalog import (
    SEC_BASE_URL,
    SEC_MAX_REQUESTS_PER_SECOND,
    SHILLER_COLUMNS,
    SHILLER_FIRST_DATA_ROW,
    SHILLER_SHEET,
    SHILLER_URLS,
    ConceptSpec,
)
from savi_uz.macro_sources import RateLimiter

USER_AGENT = "savi-uz-equity/1.0 (research)"


class SourceError(RuntimeError):
    """A source could not be fetched or did not parse."""


def _get(url: str, timeout: float, user_agent: str, limiter: RateLimiter | None = None) -> bytes:
    if not url.startswith(("https://", "http://")):
        raise SourceError(f"unsupported url scheme: {url}")
    if limiter is not None:
        limiter.acquire()
    request = Request(url, headers={"User-Agent": user_agent})
    try:
        with urlopen(request, timeout=timeout) as response:  # nosec B310
            return response.read()
    except HTTPError as exc:
        raise SourceError(f"HTTP {exc.code} for {url}") from exc
    except URLError as exc:
        raise SourceError(f"{url} unreachable: {exc.reason}") from exc


# ---------------------------------------------------------------- Shiller ----


@dataclass(frozen=True)
class ShillerRow:
    """One month of Shiller's series. Trailing months carry price but not yet
    earnings, so most fields are optional."""

    obs_date: date
    values: dict[str, float]


def parse_shiller_month(raw: Any) -> date | None:
    """Turn Shiller's decimal date into the first of that month.

    The encoding is ``YYYY.MM`` held as a float, which makes October read as
    ``1871.1`` rather than ``1871.10`` -- formatting to two decimals is what
    keeps October from being parsed as January.
    """
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        try:
            raw = float(raw)
        except ValueError:
            return None
    if not isinstance(raw, (int, float)):
        return None
    stamp = f"{float(raw):.2f}"
    try:
        year, month = int(stamp[:4]), int(stamp[5:7])
        return date(year, month, 1)
    except ValueError:
        return None


class ShillerClient:
    """Downloads and parses Shiller's ``ie_data.xls``.

    Every mirror is tried and the one with the latest observation wins, because
    the copies drift apart by as much as a year.
    """

    def __init__(self, urls: tuple[str, ...] = SHILLER_URLS, timeout: float = 180.0):
        self.urls = urls
        self.timeout = timeout

    def download(self, url: str) -> bytes:
        return _get(url, self.timeout, USER_AGENT)

    @staticmethod
    def parse(payload: bytes) -> list[ShillerRow]:
        try:
            import xlrd
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise SourceError(
                "reading Shiller's .xls needs xlrd; pip install -r requirements.txt"
            ) from exc

        try:
            sheet = xlrd.open_workbook(file_contents=payload).sheet_by_name(SHILLER_SHEET)
        except Exception as exc:
            raise SourceError(f"could not open Shiller workbook: {exc}") from exc

        rows: list[ShillerRow] = []
        for index in range(SHILLER_FIRST_DATA_ROW, sheet.nrows):
            obs_date = parse_shiller_month(sheet.cell_value(index, 0))
            if obs_date is None:
                continue  # footnote rows at the foot of the sheet
            values: dict[str, float] = {}
            for column, name in SHILLER_COLUMNS.items():
                if column >= sheet.ncols:
                    continue
                cell = sheet.cell_value(index, column)
                if isinstance(cell, (int, float)) and not isinstance(cell, bool):
                    values[name] = float(cell)
            if values:
                rows.append(ShillerRow(obs_date, values))
        if not rows:
            raise SourceError("Shiller workbook parsed to zero rows")
        return rows

    def fetch(self) -> tuple[str, list[ShillerRow]]:
        """Return the freshest mirror's url and rows."""
        best_url = ""
        best: list[ShillerRow] = []
        errors = []
        for url in self.urls:
            try:
                rows = self.parse(self.download(url))
            except SourceError as exc:
                errors.append(f"{url}: {exc}")
                continue
            if not best or rows[-1].obs_date > best[-1].obs_date:
                best_url, best = url, rows
        if not best:
            raise SourceError("no Shiller mirror could be read: " + "; ".join(errors))
        return best_url, best


# ------------------------------------------------------------------- SEC ----


@dataclass(frozen=True)
class SecFact:
    """One company's reported value for one concept in one period."""

    cik: int
    entity_name: str
    concept: str
    unit: str
    frame: str
    period_start: date | None
    period_end: date | None
    value: float
    accession: str
    location: str


class SecFramesClient:
    """SEC XBRL ``frames`` client: one request returns every filer's value.

    Pulling frames rather than per-company ``companyfacts`` is what makes market
    -wide history affordable -- one request per concept per quarter against one
    request per company.

    The SEC rejects default library user agents outright, so a descriptive one
    is mandatory; their fair-access policy asks that it identify you.
    """

    def __init__(
        self,
        user_agent: str,
        base_url: str = SEC_BASE_URL,
        timeout: float = 120.0,
        max_per_second: int = SEC_MAX_REQUESTS_PER_SECOND,
    ):
        if not user_agent or not user_agent.strip():
            raise ValueError("SEC requires a descriptive User-Agent")
        if not base_url.startswith("https://"):
            raise ValueError("SEC base_url must use https://")
        self.user_agent = user_agent.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._limiter = RateLimiter(max_per_second * 60)

    def fetch_frame(self, concept: ConceptSpec, period: str) -> list[SecFact]:
        """Every filer's value for one concept in one period.

        A frame that does not exist -- an unfiled quarter, or a concept the
        taxonomy did not carry yet -- is a 404 and returns empty rather than
        raising, because sparse early years are expected.
        """
        url = concept.url(self.base_url, period)
        try:
            payload = _get(url, self.timeout, self.user_agent, self._limiter)
        except SourceError as exc:
            if "HTTP 404" in str(exc):
                return []
            raise

        try:
            document = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SourceError(f"{url} did not return JSON: {exc}") from exc

        unit = document.get("uom", concept.unit)
        frame = document.get("ccp", concept.frame(period))
        facts = []
        for record in document.get("data", []):
            value = record.get("val")
            cik = record.get("cik")
            if value is None or cik is None:
                continue
            facts.append(
                SecFact(
                    cik=int(cik),
                    entity_name=(record.get("entityName") or "").strip(),
                    concept=concept.tag,
                    unit=unit,
                    frame=frame,
                    period_start=_iso_or_none(record.get("start")),
                    period_end=_iso_or_none(record.get("end")),
                    value=float(value),
                    accession=record.get("accn") or "",
                    location=record.get("loc") or "",
                )
            )
        return facts


def _iso_or_none(text: str | None) -> date | None:
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


# ----------------------------------------------------------------- Yahoo ----


@dataclass(frozen=True)
class IndexBar:
    ticker: str
    obs_date: date
    close: float
    volume: float | None


class YahooIndexClient:
    """Daily index closes, used to bring the S&P history up to today.

    FRED's ``SP500`` only reaches back ten years under a licence window, and
    Shiller's workbook is monthly and lags; Yahoo covers 1927 to today daily.
    """

    def __init__(self, cache_dir: str | Path = ".cache/equity", refresh: bool = False):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.refresh = refresh

    def fetch(self, ticker: str, start: date) -> list[IndexBar]:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise SourceError("index history needs yfinance; pip install -r requirements.txt") from exc

        try:
            frame = yf.download(
                ticker, start=start.isoformat(), interval="1d",
                auto_adjust=False, progress=False, threads=False,
            )
        except Exception as exc:
            raise SourceError(f"yahoo download failed for {ticker}: {exc}") from exc
        if frame is None or frame.empty:
            return []

        if hasattr(frame.columns, "levels"):
            frame = frame.xs(ticker, axis=1, level=1, drop_level=True)

        bars = []
        for stamp, row in frame.iterrows():
            close = row.get("Close")
            if close is None or close != close:  # NaN
                continue
            volume = row.get("Volume")
            bars.append(
                IndexBar(
                    ticker=ticker,
                    obs_date=stamp.date(),
                    close=float(close),
                    volume=None if volume is None or volume != volume else float(volume),
                )
            )
        return bars


# --------------------------------------------------------- Alpha Vantage ----


@dataclass(frozen=True)
class ReportedEarnings:
    ticker: str
    fiscal_ending: date
    reported_date: date | None
    reported_eps: float | None
    estimated_eps: float | None
    surprise: float | None
    surprise_percent: float | None


class AlphaVantageEarningsClient:
    """Reported-versus-expected EPS per company.

    The free tier is roughly 25 requests a day, so this is a per-ticker call
    over a short list, not a market-wide sweep.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str, timeout: float = 60.0, max_per_minute: int = 5):
        if not api_key:
            raise ValueError("AlphaVantage api_key is required")
        self._api_key = api_key
        self.timeout = timeout
        self._limiter = RateLimiter(max_per_minute)

    def fetch_earnings(self, ticker: str) -> list[ReportedEarnings]:
        query = urlencode({"function": "EARNINGS", "symbol": ticker, "apikey": self._api_key})
        payload = _get(f"{self.BASE_URL}?{query}", self.timeout, USER_AGENT, self._limiter)
        document = json.loads(payload.decode("utf-8"))

        # AlphaVantage answers quota exhaustion and bad symbols with HTTP 200
        # and an explanatory body, so the failure has to be read out of the JSON.
        for key in ("Note", "Information", "Error Message"):
            if key in document:
                raise SourceError(f"AlphaVantage: {document[key]}")

        rows = []
        for record in document.get("quarterlyEarnings", []):
            ending = _iso_or_none(record.get("fiscalDateEnding"))
            if ending is None:
                continue
            rows.append(
                ReportedEarnings(
                    ticker=ticker,
                    fiscal_ending=ending,
                    reported_date=_iso_or_none(record.get("reportedDate")),
                    reported_eps=_float_or_none(record.get("reportedEPS")),
                    estimated_eps=_float_or_none(record.get("estimatedEPS")),
                    surprise=_float_or_none(record.get("surprise")),
                    surprise_percent=_float_or_none(record.get("surprisePercentage")),
                )
            )
        return rows


def _float_or_none(text: Any) -> float | None:
    if text in (None, "", "None", "-"):
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
