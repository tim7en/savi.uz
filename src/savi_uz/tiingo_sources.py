"""Tiingo client for intraday and daily bars, built to stay inside the quota.

Three things about this API drive the design.

**The row cap is silent.** A request spanning more years than fit in one
response returns exactly 10,000 bars and no error, and the window it returns is
the *recent* end -- so asking for 2017 onward hands back 2020 onward and looks
like the history simply starts later. Requests are therefore chunked by year
(~1,550 hourly bars) and a response that comes back at the cap is reported
rather than trusted.

**IEX is exchange-listed only.** The OTC ADRs in this universe (`TCEHY`,
`XIACY`, `MPNGY`, `PMRTY`, all on `PINK`) return zero intraday bars for any
window, while having years of daily history. Intraday is not available for them
at any price, so they fall back to daily.

**The quota is small and shared across the day.** Every response is cached on
disk, the pacing is configurable and defaults below the free tier's documented
ceiling, and a 429 stops the run immediately instead of retrying into a block.
"""

from __future__ import annotations

import json
import ssl
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = "https://api.tiingo.com"

#: Tiingo returns at most this many rows per request, without saying so.
MAX_ROWS_PER_REQUEST = 10_000

#: Documented free-tier ceilings. The defaults sit under them deliberately; a
#: paid plan can raise them with --requests-per-hour.
FREE_TIER_REQUESTS_PER_HOUR = 50
FREE_TIER_REQUESTS_PER_DAY = 1000
DEFAULT_REQUESTS_PER_HOUR = 45

#: Exchanges with no IEX intraday feed; these can only be had as daily bars.
NO_INTRADAY_EXCHANGES = frozenset({"PINK", "OTC", "OTCMKTS", "GREY", "OTCBB"})

SUPPORTED_FREQUENCIES = ("1min", "5min", "15min", "30min", "1hour", "4hour", "daily")

#: Must be requested explicitly. The IEX endpoint's default projection is
#: date/open/high/low/close and drops volume without saying so.
INTRADAY_COLUMNS = "open,high,low,close,volume"

#: IEX is one venue with a low single-digit share of consolidated volume, so
#: these counts are a relative activity measure, not tradable share volume.
VOLUME_IS_IEX_ONLY = True


class HourlyRateLimiter:
    """Spaces requests evenly across the hour.

    The existing per-minute limiter cannot express this: it floors at one
    request per minute, which is 60 an hour and already over the free tier.
    Spacing evenly rather than bursting also means a run interrupted at any
    point has never exceeded the rolling budget.
    """

    def __init__(self, requests_per_hour: int):
        if requests_per_hour < 1:
            raise ValueError("requests_per_hour must be at least 1")
        self.min_interval = 3600.0 / requests_per_hour
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = max(self._next_slot - now, 0.0)
            self._next_slot = max(now, self._next_slot) + self.min_interval
        if wait > 0:
            time.sleep(wait)


class TiingoError(RuntimeError):
    """A Tiingo request failed."""


class TiingoRateLimitError(TiingoError):
    """The quota is exhausted. Stop; do not retry into a block."""


def _ssl_context() -> ssl.SSLContext:
    """Trust certifi rather than the OS store.

    On Windows the system store still carries an expired root that Python picks
    for api.tiingo.com, so verification fails there while curl succeeds.
    Pinning certifi fixes it without weakening verification.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:  # pragma: no cover - certifi ships with the stack
        return ssl.create_default_context()


@dataclass(frozen=True)
class SymbolMeta:
    """What Tiingo knows about a ticker, including where its history starts."""

    ticker: str
    name: str
    exchange: str
    start_date: date | None
    end_date: date | None
    description: str = ""

    @property
    def has_intraday(self) -> bool:
        return self.exchange.upper() not in NO_INTRADAY_EXCHANGES


@dataclass(frozen=True)
class Adjustment:
    """One day's split and dividend factors, for adjusting raw intraday bars."""

    ticker: str
    obs_date: date
    close: float | None
    adj_close: float | None
    split_factor: float
    div_cash: float

    @property
    def is_split(self) -> bool:
        return self.split_factor not in (1.0, 0.0)


@dataclass(frozen=True)
class Bar:
    ticker: str
    frequency: str
    timestamp: str
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: float | None


def _parse_date(text: str | None) -> date | None:
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class TiingoClient:
    """Rate-limited, cached Tiingo client with a hard request budget."""

    def __init__(
        self,
        api_key: str,
        cache_dir: str | Path = ".cache/tiingo",
        requests_per_hour: int = DEFAULT_REQUESTS_PER_HOUR,
        max_requests: int | None = None,
        timeout: float = 90.0,
        refresh: bool = False,
    ):
        if not api_key:
            raise ValueError("Tiingo api_key is required")
        if requests_per_hour < 1:
            raise ValueError("requests_per_hour must be at least 1")
        self._api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.refresh = refresh
        self.max_requests = max_requests
        self.requests_made = 0
        self.cache_hits = 0
        self._context = _ssl_context()
        self.requests_per_hour = requests_per_hour
        self._limiter = HourlyRateLimiter(requests_per_hour)

    # -- plumbing ---------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def budget_exhausted(self) -> bool:
        return self.max_requests is not None and self.requests_made >= self.max_requests

    def _get(self, path: str, params: dict[str, Any], cache_key: str) -> Any:
        cached = self._cache_path(cache_key)
        if cached.is_file() and not self.refresh:
            self.cache_hits += 1
            return json.loads(cached.read_text(encoding="utf-8"))

        if self.budget_exhausted():
            raise TiingoError(
                f"request budget of {self.max_requests} reached; rerun to continue"
            )

        self._limiter.acquire()
        query = urlencode(params)
        url = f"{BASE_URL}{path}?{query}" if query else f"{BASE_URL}{path}"
        request = Request(
            url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Token {self._api_key}",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout, context=self._context) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            if exc.code == 429:
                raise TiingoRateLimitError(f"Tiingo quota exhausted: {detail}") from exc
            if exc.code in (401, 403):
                raise TiingoError(f"Tiingo rejected the key (HTTP {exc.code}): {detail}") from exc
            raise TiingoError(f"HTTP {exc.code} for {path}: {detail}") from exc
        except URLError as exc:
            raise TiingoError(f"{path} unreachable: {exc.reason}") from exc

        self.requests_made += 1
        cached.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    # -- endpoints --------------------------------------------------------

    def fetch_metadata(self, ticker: str) -> SymbolMeta:
        """Ticker metadata, chiefly the first date any history exists.

        Worth one request per symbol: it stops the downloader asking for years
        before a company listed, which is most of the wasted budget otherwise.
        """
        payload = self._get(f"/tiingo/daily/{ticker}", {}, f"meta_{ticker}")
        return SymbolMeta(
            ticker=ticker,
            name=payload.get("name", "") or "",
            exchange=(payload.get("exchangeCode") or "").strip(),
            start_date=_parse_date(payload.get("startDate")),
            end_date=_parse_date(payload.get("endDate")),
            description=(payload.get("description") or "")[:500],
        )

    def _bars(self, payload: Any, ticker: str, frequency: str) -> list[Bar]:
        bars = []
        for record in payload or []:
            stamp = record.get("date")
            if not stamp:
                continue
            bars.append(
                Bar(
                    ticker=ticker,
                    frequency=frequency,
                    timestamp=str(stamp),
                    open=_parse_float(record.get("open")),
                    high=_parse_float(record.get("high")),
                    low=_parse_float(record.get("low")),
                    close=_parse_float(record.get("close")),
                    volume=_parse_float(record.get("volume")),
                )
            )
        return bars

    def fetch_intraday(
        self, ticker: str, start: date, end: date, frequency: str = "1hour"
    ) -> tuple[list[Bar], bool]:
        """Intraday bars for one window. Returns ``(bars, hit_row_cap)``.

        ``columns`` has to be spelled out: the IEX endpoint returns only
        date/open/high/low/close by default and silently omits volume, so a
        download that does not ask for it ends up with none at all.

        The cap flag matters: at exactly ``MAX_ROWS_PER_REQUEST`` the window was
        truncated and the caller must narrow it, or silently lose the early part
        of the range.
        """
        if frequency not in SUPPORTED_FREQUENCIES:
            raise ValueError(f"unsupported frequency {frequency!r}")
        payload = self._get(
            f"/iex/{ticker}/prices",
            {
                "startDate": start.isoformat(),
                "endDate": end.isoformat(),
                "resampleFreq": frequency,
                "columns": INTRADAY_COLUMNS,
            },
            f"iex_{ticker}_{frequency}_{start.isoformat()}_{end.isoformat()}_ohlcv",
        )
        bars = self._bars(payload, ticker, frequency)
        return bars, len(bars) >= MAX_ROWS_PER_REQUEST

    def fetch_adjustments(self, ticker: str, start: date, end: date) -> list[Adjustment]:
        """Daily split and dividend factors for rebuilding an adjusted series.

        Intraday bars come back raw: NVDA's close goes from 1,203 to 121 across
        its 10:1 split, which a backtest reads as a 90% loss in a day. The daily
        endpoint carries ``splitFactor`` and ``divCash``, so storing them lets a
        backtest adjust intraday prices itself. One request covers 20 years.
        """
        payload = self._get(
            f"/tiingo/daily/{ticker}/prices",
            {"startDate": start.isoformat(), "endDate": end.isoformat()},
            f"adj_{ticker}_{start.isoformat()}_{end.isoformat()}",
        )
        rows = []
        for record in payload or []:
            day = _parse_date(record.get("date"))
            if day is None:
                continue
            rows.append(
                Adjustment(
                    ticker=ticker,
                    obs_date=day,
                    close=_parse_float(record.get("close")),
                    adj_close=_parse_float(record.get("adjClose")),
                    split_factor=_parse_float(record.get("splitFactor")) or 1.0,
                    div_cash=_parse_float(record.get("divCash")) or 0.0,
                )
            )
        return rows

    def fetch_daily(self, ticker: str, start: date, end: date) -> tuple[list[Bar], bool]:
        """End-of-day bars; the only route for OTC tickers."""
        payload = self._get(
            f"/tiingo/daily/{ticker}/prices",
            {"startDate": start.isoformat(), "endDate": end.isoformat()},
            f"daily_{ticker}_{start.isoformat()}_{end.isoformat()}",
        )
        bars = self._bars(payload, ticker, "daily")
        return bars, len(bars) >= MAX_ROWS_PER_REQUEST


#: Measured: a full year of hourly bars is about 1,566 rows.
HOURLY_BARS_PER_YEAR = 1_566

#: Three years is ~4,700 rows, comfortably under half the cap, and costs a third
#: of the requests that single-year chunks do -- which is what decides whether a
#: free-tier pull takes four hours or eleven.
DEFAULT_YEARS_PER_REQUEST = 3


#: Bars a full year produces, per frequency. Used to size a request so it never
#: reaches the silent row cap.
BARS_PER_YEAR = {
    "1hour": HOURLY_BARS_PER_YEAR,
    "4hour": HOURLY_BARS_PER_YEAR // 4,
    "30min": HOURLY_BARS_PER_YEAR * 2,
    "15min": HOURLY_BARS_PER_YEAR * 4,
    "5min": HOURLY_BARS_PER_YEAR * 12,
    "1min": HOURLY_BARS_PER_YEAR * 60,
    "daily": 252,
}


def max_safe_months(frequency: str = "1hour", safety: float = 0.5) -> int:
    """Largest chunk in months that stays under ``safety`` of the row cap.

    Sub-hourly frequencies overflow a single calendar year -- 5-minute bars run
    to roughly 19,700 a year against a 10,000 cap -- so the chunker has to work
    in months, not years, or every request silently loses its early months.
    """
    per_year = BARS_PER_YEAR.get(frequency, HOURLY_BARS_PER_YEAR)
    months = int(MAX_ROWS_PER_REQUEST * safety / (per_year / 12.0))
    return max(min(months, 240), 1)


def max_safe_years(frequency: str = "1hour", safety: float = 0.5) -> int:
    """Largest whole-year chunk that stays under ``safety`` of the row cap."""
    return max(max_safe_months(frequency, safety) // 12, 0)


def month_windows(start: date, end: date, months_per_chunk: int) -> list[tuple[date, date]]:
    """Split a range into chunks of whole calendar months.

    Chunks are anchored to the calendar (January, then every ``months_per_chunk``
    after it) rather than to ``start``, so two runs over different ranges ask for
    the same windows and share the cache.
    """
    if start > end:
        return []
    if months_per_chunk < 1:
        raise ValueError("months_per_chunk must be at least 1")

    windows = []
    index = start.year * 12 + (start.month - 1)
    last_index = end.year * 12 + (end.month - 1)
    index -= index % months_per_chunk
    while index <= last_index:
        chunk_end = index + months_per_chunk - 1
        first = date(index // 12, index % 12 + 1, 1)
        end_year, end_month = chunk_end // 12, chunk_end % 12 + 1
        if end_month == 12:
            last_day = date(end_year, 12, 31)
        else:
            last_day = date(end_year, end_month + 1, 1) - timedelta(days=1)
        first = max(first, start)
        last_day = min(last_day, end)
        if first <= last_day:
            windows.append((first, last_day))
        index = chunk_end + 1
    return windows


def year_windows(
    start: date, end: date, years_per_chunk: int = 1
) -> list[tuple[date, date]]:
    """Split a range into chunks of whole calendar years."""
    if years_per_chunk < 1:
        raise ValueError("years_per_chunk must be at least 1")
    return month_windows(start, end, years_per_chunk * 12)


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
