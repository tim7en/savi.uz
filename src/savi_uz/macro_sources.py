"""Clients for macro data: FRED/ALFRED, the NY Fed reference rates API, and the
Federal Reserve's Gurkaynak-Sack-Wright fitted yield curve.
"""

from __future__ import annotations

import csv
import io
import json
import math
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Iterator
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

FRED_BASE_URL = "https://api.stlouisfed.org/fred"
NYFED_BASE_URL = "https://markets.newyorkfed.org/api"
GSW_CURVE_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"

#: ALFRED needs an explicit real-time window this wide to hand back every vintage.
ALL_TIME_REALTIME = {"realtime_start": "1776-07-04", "realtime_end": "9999-12-31"}

USER_AGENT = "savi-uz-macro/1.0 (research)"

FRED_OUTPUT_LATEST = 1
FRED_OUTPUT_ALL_VINTAGES = 2
FRED_OUTPUT_INITIAL_RELEASE = 4

#: ALFRED's own ceiling on vintage dates per observations request.
MAX_VINTAGE_DATES_PER_REQUEST = 1900


class RateLimiter:
    """Token bucket keeping request bursts inside a documented per-minute cap."""

    def __init__(self, max_per_minute: int):
        self._min_interval = 60.0 / max(max_per_minute, 1)
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = max(self._next_slot - now, 0.0)
            self._next_slot = max(now, self._next_slot) + self._min_interval
        if wait > 0:
            time.sleep(wait)


class VintageLimitError(ValueError):
    """ALFRED refuses a real-time window holding more than 2000 vintage dates."""


def _http_json(url: str, timeout: float, limiter: RateLimiter | None = None) -> Any:
    if limiter is not None:
        limiter.acquire()
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:  # nosec B310
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        if "maximum number of vintage dates" in detail:
            raise VintageLimitError(detail) from exc
        raise ValueError(f"HTTP {exc.code} for {url.split('?')[0]}: {detail}") from exc


@dataclass(frozen=True)
class Observation:
    """One value of a series, with the real-time window it was current for."""

    obs_date: date
    value: float | None
    realtime_start: date | None = None
    realtime_end: date | None = None


@dataclass(frozen=True)
class SeriesMetadata:
    series_id: str
    title: str
    units: str
    frequency: str
    seasonal_adjustment: str
    observation_start: date | None
    observation_end: date | None
    last_updated: str
    popularity: int
    notes: str


def _parse_date(text: str | None) -> date | None:
    if not text or text in (".", ""):
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _parse_value(text: Any) -> float | None:
    if text in (None, ".", ""):
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


class FredClient:
    """FRED/ALFRED client covering current values, first prints and full vintages."""

    def __init__(self, api_key: str, base_url: str = FRED_BASE_URL, timeout: float = 60.0,
                 max_per_minute: int = 100):
        if not api_key:
            raise ValueError("FRED api_key is required")
        if not base_url.startswith("https://"):
            raise ValueError("FRED base_url must use https://")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._limiter = RateLimiter(max_per_minute)

    def _get(self, path: str, **params: Any) -> dict[str, Any]:
        query = urlencode({**params, "api_key": self._api_key, "file_type": "json"})
        return _http_json(f"{self.base_url}/{path}?{query}", self.timeout, self._limiter)

    def fetch_metadata(self, series_id: str) -> SeriesMetadata:
        payload = self._get("series", series_id=series_id)
        record = payload["seriess"][0]
        return SeriesMetadata(
            series_id=series_id,
            title=record.get("title", ""),
            units=record.get("units", ""),
            frequency=record.get("frequency", ""),
            seasonal_adjustment=record.get("seasonal_adjustment", ""),
            observation_start=_parse_date(record.get("observation_start")),
            observation_end=_parse_date(record.get("observation_end")),
            last_updated=record.get("last_updated", ""),
            popularity=int(record.get("popularity", 0) or 0),
            notes=(record.get("notes") or "")[:2000],
        )

    def fetch_release(self, series_id: str) -> tuple[int | None, str]:
        payload = self._get("series/release", series_id=series_id)
        releases = payload.get("releases") or []
        if not releases:
            return None, ""
        return int(releases[0]["id"]), releases[0].get("name", "")

    def fetch_release_dates(self, release_id: int, start: date | None = None) -> list[date]:
        params: dict[str, Any] = {"release_id": release_id, "limit": 10000, "sort_order": "asc"}
        if start:
            params["realtime_start"] = start.isoformat()
        payload = self._get("release/dates", **params)
        return [d for d in (_parse_date(row.get("date")) for row in payload.get("release_dates", [])) if d]

    def fetch_observations(self, series_id: str, start: date | None = None) -> list[Observation]:
        """Current values: the latest revision of every observation."""
        params: dict[str, Any] = {"series_id": series_id, "output_type": FRED_OUTPUT_LATEST}
        if start:
            params["observation_start"] = start.isoformat()
        payload = self._get("series/observations", **params)
        return [
            Observation(obs_date=day, value=_parse_value(row.get("value")))
            for row in payload.get("observations", [])
            if (day := _parse_date(row.get("date"))) is not None
        ]

    def fetch_vintage_dates(self, series_id: str) -> list[date]:
        payload = self._get("series/vintagedates", series_id=series_id, limit=10000)
        return [d for d in (_parse_date(text) for text in payload.get("vintage_dates", [])) if d]

    def _realtime_windows(self, series_id: str) -> list[dict[str, str]]:
        """Split a series' vintage dates into windows ALFRED will accept.

        Windows are disjoint and cover every vintage, so per-observation results
        concatenate without duplicates or gaps.
        """
        vintage_dates = self.fetch_vintage_dates(series_id)
        if not vintage_dates:
            return [dict(ALL_TIME_REALTIME)]
        windows = []
        for offset in range(0, len(vintage_dates), MAX_VINTAGE_DATES_PER_REQUEST):
            chunk = vintage_dates[offset : offset + MAX_VINTAGE_DATES_PER_REQUEST]
            windows.append({"realtime_start": chunk[0].isoformat(), "realtime_end": chunk[-1].isoformat()})
        return windows

    def _observations_over_windows(
        self, series_id: str, output_type: int, start: date | None
    ) -> list[dict[str, Any]]:
        """Run one observations query, splitting the real-time window if ALFRED balks."""
        params: dict[str, Any] = {"series_id": series_id, "output_type": output_type}
        if start:
            params["observation_start"] = start.isoformat()
        try:
            return self._get("series/observations", **params, **ALL_TIME_REALTIME).get("observations", [])
        except VintageLimitError:
            rows: list[dict[str, Any]] = []
            for window in self._realtime_windows(series_id):
                rows.extend(self._get("series/observations", **params, **window).get("observations", []))
            return rows

    def fetch_first_releases(self, series_id: str, start: date | None = None) -> list[Observation]:
        """Initial print of every observation, stamped with its release date."""
        observations = []
        for row in self._observations_over_windows(series_id, FRED_OUTPUT_INITIAL_RELEASE, start):
            day = _parse_date(row.get("date"))
            if day is None:
                continue
            observations.append(
                Observation(
                    obs_date=day,
                    value=_parse_value(row.get("value")),
                    realtime_start=_parse_date(row.get("realtime_start")),
                    realtime_end=_parse_date(row.get("realtime_end")),
                )
            )
        return observations

    def fetch_all_vintages(self, series_id: str, start: date | None = None) -> list[Observation]:
        """Every vintage of every observation, flattened to one row per pair.

        ALFRED returns this wide, one column per vintage date named
        ``<SERIES_ID>_<YYYYMMDD>``; it is melted here into long form.
        """
        observations: list[Observation] = []
        for row in self._observations_over_windows(series_id, FRED_OUTPUT_ALL_VINTAGES, start):
            obs_date = _parse_date(row.get("date"))
            if obs_date is None:
                continue
            for column, raw in row.items():
                if column == "date" or not column.startswith(f"{series_id}_"):
                    continue
                value = _parse_value(raw)
                if value is None:
                    continue
                stamp = column.rsplit("_", 1)[-1]
                vintage = _parse_date(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}") if len(stamp) == 8 else None
                observations.append(Observation(obs_date=obs_date, value=value, realtime_start=vintage))
        return observations


@dataclass(frozen=True)
class ReferenceRate:
    """A NY Fed published reference rate, including its revision flag."""

    rate_type: str
    effective_date: date
    percent_rate: float | None
    percentile_1: float | None
    percentile_25: float | None
    percentile_75: float | None
    percentile_99: float | None
    volume_billions: float | None
    target_rate_from: float | None
    target_rate_to: float | None
    revision_indicator: str


class NyFedRatesClient:
    """Reference rates straight from the publisher, with revision flags intact."""

    def __init__(self, base_url: str = NYFED_BASE_URL, timeout: float = 60.0, max_per_minute: int = 120):
        if not base_url.startswith("https://"):
            raise ValueError("NY Fed base_url must use https://")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._limiter = RateLimiter(max_per_minute)

    def fetch_rates(self, rate_type: str, secured: str, start: date, end: date) -> list[ReferenceRate]:
        query = urlencode({"startDate": start.isoformat(), "endDate": end.isoformat(), "type": rate_type.upper()})
        url = f"{self.base_url}/rates/{secured}/{rate_type.lower()}/search.json?{query}"
        payload = _http_json(url, self.timeout, self._limiter)
        rates = []
        for row in payload.get("refRates", []):
            effective = _parse_date(row.get("effectiveDate"))
            if effective is None:
                continue
            rates.append(
                ReferenceRate(
                    rate_type=row.get("type", rate_type.upper()),
                    effective_date=effective,
                    percent_rate=_parse_value(row.get("percentRate")),
                    percentile_1=_parse_value(row.get("percentPercentile1")),
                    percentile_25=_parse_value(row.get("percentPercentile25")),
                    percentile_75=_parse_value(row.get("percentPercentile75")),
                    percentile_99=_parse_value(row.get("percentPercentile99")),
                    volume_billions=_parse_value(row.get("volumeInBillions")),
                    target_rate_from=_parse_value(row.get("targetRateFrom")),
                    target_rate_to=_parse_value(row.get("targetRateTo")),
                    revision_indicator=row.get("revisionIndicator") or "",
                )
            )
        return rates


@dataclass(frozen=True)
class SvenssonParameters:
    """Daily fit of the Svensson forward-rate curve published by the Board."""

    curve_date: date
    beta0: float
    beta1: float
    beta2: float
    beta3: float
    tau1: float
    tau2: float

    def instantaneous_forward(self, horizon_years: float) -> float:
        """Forward rate ``horizon_years`` ahead, in percent.

        This is a Treasury forward and therefore carries a term premium; it is a
        proxy for the market-implied policy path, not an OIS-implied path.
        """
        n = max(horizon_years, 1e-6)
        first = n / self.tau1
        second = n / self.tau2
        return (
            self.beta0
            + self.beta1 * math.exp(-first)
            + self.beta2 * first * math.exp(-first)
            + self.beta3 * second * math.exp(-second)
        )


class GswCurveClient:
    """Downloader/parser for the Board's fitted Treasury yield curve table."""

    def __init__(self, url: str = GSW_CURVE_URL, timeout: float = 180.0):
        if not url.startswith("https://"):
            raise ValueError("GSW url must use https://")
        self.url = url
        self.timeout = timeout

    def download(self) -> str:
        request = Request(self.url, headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=self.timeout) as response:  # nosec B310
            return response.read().decode("utf-8", "replace")

    @staticmethod
    def parse(text: str) -> tuple[list[SvenssonParameters], list[tuple[date, str, float]]]:
        """Return Svensson parameters and long-form (date, mnemonic, value) rates.

        The file carries a multi-line research disclaimer before the header row.
        """
        lines = text.splitlines()
        header_index = next((i for i, line in enumerate(lines) if line.startswith("Date,")), None)
        if header_index is None:
            raise ValueError("GSW table has no 'Date,' header row")

        reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
        parameters: list[SvenssonParameters] = []
        rates: list[tuple[date, str, float]] = []
        rate_prefixes = ("SVENY", "SVENPY", "SVENF", "SVEN1F")

        for row in reader:
            curve_date = _parse_date(row.get("Date"))
            if curve_date is None:
                continue
            betas = [_parse_value(row.get(name)) for name in ("BETA0", "BETA1", "BETA2", "BETA3")]
            taus = [_parse_value(row.get(name)) for name in ("TAU1", "TAU2")]
            if all(value is not None for value in betas + taus) and taus[0] and taus[1]:
                parameters.append(
                    SvenssonParameters(curve_date, betas[0], betas[1], betas[2], betas[3], taus[0], taus[1])
                )
            for column, raw in row.items():
                if not column or not column.startswith(rate_prefixes):
                    continue
                value = _parse_value(raw)
                if value is not None:
                    rates.append((curve_date, column, value))
        return parameters, rates

    def fetch(self) -> tuple[list[SvenssonParameters], list[tuple[date, str, float]]]:
        return self.parse(self.download())


def implied_policy_path(
    parameters: list[SvenssonParameters],
    horizon_months: tuple[int, ...],
) -> Iterator[tuple[date, int, float]]:
    """Market-implied forward path at each horizon, one row per (date, horizon)."""
    for fit in parameters:
        for months in horizon_months:
            yield fit.curve_date, months, fit.instantaneous_forward(months / 12.0)


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
