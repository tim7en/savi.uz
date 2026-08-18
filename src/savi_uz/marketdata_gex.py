"""MarketData.app historical option chains and leakage-safe daily GEX features."""

from __future__ import annotations

import json
import math
import sqlite3
import ssl
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_URL = "https://api.marketdata.app/v1/options/chain/{symbol}/"


class MarketDataError(RuntimeError):
    """A MarketData.app request or response failed validation."""


class MarketDataRateLimitError(MarketDataError):
    """The API credit window is exhausted; callers must stop."""


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:  # pragma: no cover
        return ssl.create_default_context()


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _date_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, timezone.utc).date().isoformat()
    text = str(value)
    if text.isdigit():
        return datetime.fromtimestamp(int(text), timezone.utc).date().isoformat()
    return text[:10]


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def option_price(side: str, spot: float, strike: float, years: float,
                 rate: float, dividend_yield: float, volatility: float) -> float:
    if years <= 0 or volatility <= 0 or spot <= 0 or strike <= 0:
        return max(spot - strike, 0.0) if side == "call" else max(strike - spot, 0.0)
    root = math.sqrt(years)
    d1 = (math.log(spot / strike) + (
        rate - dividend_yield + 0.5 * volatility * volatility
    ) * years) / (volatility * root)
    d2 = d1 - volatility * root
    if side == "call":
        return (spot * math.exp(-dividend_yield * years) * _normal_cdf(d1)
                - strike * math.exp(-rate * years) * _normal_cdf(d2))
    return (strike * math.exp(-rate * years) * _normal_cdf(-d2)
            - spot * math.exp(-dividend_yield * years) * _normal_cdf(-d1))


def implied_volatility(side: str, price: float, spot: float, strike: float,
                       years: float, rate: float, dividend_yield: float) -> float | None:
    if side not in {"call", "put"} or min(price, spot, strike, years) <= 0:
        return None
    lower = option_price(side, spot, strike, years, rate, dividend_yield, 1e-6)
    upper = (spot * math.exp(-dividend_yield * years) if side == "call"
             else strike * math.exp(-rate * years))
    tolerance = max(spot * 1e-8, 1e-6)
    if price < lower - tolerance or price > upper + tolerance:
        return None
    low, high = 1e-4, 5.0
    if option_price(side, spot, strike, years, rate, dividend_yield, high) < price:
        return None
    for _ in range(70):
        middle = (low + high) / 2.0
        if option_price(side, spot, strike, years, rate, dividend_yield, middle) < price:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def model_gamma(spot: float, strike: float, years: float, rate: float,
                dividend_yield: float, volatility: float) -> float | None:
    if min(spot, strike, years, volatility) <= 0:
        return None
    root = math.sqrt(years)
    d1 = (math.log(spot / strike) + (
        rate - dividend_yield + 0.5 * volatility * volatility
    ) * years) / (volatility * root)
    density = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    return math.exp(-dividend_yield * years) * density / (spot * volatility * root)


def enrich_contracts(contracts: tuple[dict[str, Any], ...], *, risk_free: float,
                     dividend_yield: float) -> tuple[dict[str, Any], ...]:
    enriched = []
    for original in contracts:
        row = dict(original)
        gamma = _number(row.get("gamma"))
        iv = _number(row.get("iv"))
        source = "vendor" if gamma is not None else ""
        if gamma is None:
            mid = _number(row.get("mid"))
            spot = _number(row.get("underlyingPrice"))
            strike = _number(row.get("strike"))
            dte = _number(row.get("dte"))
            side = str(row.get("side", "")).lower()
            years = dte / 365.25 if dte is not None else 0.0
            if None not in (mid, spot, strike) and years > 0:
                if iv is None:
                    iv = implied_volatility(
                        side, mid, spot, strike, years, risk_free, dividend_yield
                    )
                if iv is not None:
                    gamma = model_gamma(
                        spot, strike, years, risk_free, dividend_yield, iv
                    )
                    source = "black_scholes_mid"
        row["iv"] = iv
        row["gamma"] = gamma
        row["gammaSource"] = source
        enriched.append(row)
    return tuple(enriched)


@dataclass(frozen=True)
class CreditUsage:
    consumed: int | None
    remaining: int | None
    limit: int | None
    reset: int | None


@dataclass(frozen=True)
class ChainResponse:
    status: str
    contracts: tuple[dict[str, Any], ...]
    credits: CreditUsage
    message: str = ""


def _credit_usage(headers: Any) -> CreditUsage:
    return CreditUsage(*(
        _integer(headers.get(name)) for name in (
            "X-Api-Ratelimit-Consumed", "X-Api-Ratelimit-Remaining",
            "X-Api-Ratelimit-Limit", "X-Api-Ratelimit-Reset",
        )
    ))


class MarketDataClient:
    def __init__(self, token: str, timeout: float = 60.0):
        if not token.strip():
            raise ValueError("MarketData.app token is empty")
        self.token = token.strip()
        self.timeout = timeout

    def fetch_chain(
        self, symbol: str, observation_date: date, *, max_dte_days: int = 60,
        strike_limit: int = 60,
    ) -> ChainResponse:
        if max_dte_days < 0 or strike_limit < 1:
            raise ValueError("max_dte_days and strike_limit must be positive")
        # The API's `to` expiration bound is exclusive.
        from datetime import timedelta

        params = {
            "date": observation_date.isoformat(),
            "from": observation_date.isoformat(),
            "to": (observation_date + timedelta(days=max_dte_days + 1)).isoformat(),
            "strikeLimit": strike_limit,
        }
        url = BASE_URL.format(symbol=symbol.upper()) + "?" + urlencode(params)
        request = Request(
            url, headers={"Authorization": f"Bearer {self.token}",
                          "Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout, context=_ssl_context()) as response:
                payload = json.loads(response.read().decode("utf-8"))
                headers = response.headers
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            try:
                error_payload = json.loads(body)
            except json.JSONDecodeError:
                error_payload = {}
            if exc.code == 404 and error_payload.get("s") == "no_data":
                message = error_payload.get("errmsg") or error_payload.get("message") or ""
                return ChainResponse("no_data", (), _credit_usage(exc.headers), str(message))
            if exc.code == 429:
                raise MarketDataRateLimitError(f"HTTP 429 credit limit: {body}") from exc
            raise MarketDataError(f"HTTP {exc.code}: {body}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise MarketDataError(f"MarketData.app request failed: {exc}") from exc

        credits = _credit_usage(headers)
        status = str(payload.get("s", "error"))
        if status == "no_data":
            return ChainResponse(status, (), credits)
        if status != "ok":
            message = (payload.get("errmsg") or payload.get("message")
                       or payload.get("error") or json.dumps(payload)[:500])
            return ChainResponse(status, (), credits, str(message))
        symbols = payload.get("optionSymbol")
        if not isinstance(symbols, list):
            raise MarketDataError("successful response omitted optionSymbol array")
        arrays = {key: value for key, value in payload.items() if isinstance(value, list)}
        if any(len(value) != len(symbols) for value in arrays.values()):
            raise MarketDataError("response arrays have inconsistent lengths")
        contracts = []
        for index, option_symbol in enumerate(symbols):
            row = {key: value[index] for key, value in arrays.items()}
            row["optionSymbol"] = option_symbol
            contracts.append(row)
        return ChainResponse(status, tuple(contracts), credits)


SCHEMA = """
CREATE TABLE IF NOT EXISTS option_contracts (
    symbol TEXT NOT NULL,
    observation_date TEXT NOT NULL,
    option_symbol TEXT NOT NULL,
    expiration TEXT,
    side TEXT,
    strike REAL,
    dte INTEGER,
    volume REAL,
    open_interest REAL,
    underlying_price REAL,
    updated INTEGER,
    iv REAL,
    delta REAL,
    gamma REAL,
    bid REAL,
    ask REAL,
    mid REAL,
    last REAL,
    gamma_source TEXT,
    PRIMARY KEY (symbol, observation_date, option_symbol)
);
CREATE INDEX IF NOT EXISTS idx_option_contracts_day
    ON option_contracts(symbol, observation_date);

CREATE TABLE IF NOT EXISTS daily_gex (
    symbol TEXT NOT NULL,
    observation_date TEXT NOT NULL,
    contracts INTEGER NOT NULL,
    usable_contracts INTEGER NOT NULL,
    underlying_price REAL,
    call_gex REAL,
    put_gex REAL,
    net_gex REAL,
    absolute_gex REAL,
    gamma_wall_strike REAL,
    distance_to_gamma_wall REAL,
    gamma_flip_proxy REAL,
    oi_weighted_iv REAL,
    put_call_oi REAL,
    PRIMARY KEY (symbol, observation_date)
);

CREATE TABLE IF NOT EXISTS fetch_log (
    symbol TEXT NOT NULL,
    observation_date TEXT NOT NULL,
    status TEXT NOT NULL,
    rows INTEGER NOT NULL,
    credits_consumed INTEGER,
    credits_remaining INTEGER,
    message TEXT,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (symbol, observation_date)
);
"""


class GexStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript(SCHEMA)
        existing = {row[1] for row in self.connection.execute(
            "PRAGMA table_info(option_contracts)"
        )}
        for name, kind in (("bid", "REAL"), ("ask", "REAL"), ("mid", "REAL"),
                           ("last", "REAL"), ("gamma_source", "TEXT")):
            if name not in existing:
                self.connection.execute(
                    f"ALTER TABLE option_contracts ADD COLUMN {name} {kind}"
                )
        self.connection.commit()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def close(self):
        self.connection.commit()
        self.connection.close()

    def completed(self) -> set[tuple[str, str]]:
        return set(self.connection.execute(
            "SELECT f.symbol, f.observation_date FROM fetch_log f "
            "LEFT JOIN daily_gex g USING(symbol, observation_date) "
            "WHERE f.status='no_data' OR (f.status='ok' AND "
            "g.usable_contracts >= g.contracts * 0.5)"
        ))

    def write(self, symbol: str, observation_date: date, response: ChainResponse,
              *, risk_free: float = 0.04, dividend_yield: float = 0.0):
        day = observation_date.isoformat()
        contracts = enrich_contracts(
            response.contracts, risk_free=risk_free, dividend_yield=dividend_yield
        )
        rows = [normalise_contract(symbol, day, row) for row in contracts]
        feature = None
        if rows:
            self.connection.executemany(
                "INSERT OR REPLACE INTO option_contracts "
                "(symbol,observation_date,option_symbol,expiration,side,strike,dte,volume,"
                "open_interest,underlying_price,updated,iv,delta,gamma,bid,ask,mid,last,"
                "gamma_source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows,
            )
            feature = calculate_gex(symbol, day, contracts)
            self.connection.execute(
                "INSERT OR REPLACE INTO daily_gex VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", feature,
            )
        status = response.status
        message = response.message
        if feature is not None and feature[3] < feature[2] * 0.5:
            status = "incomplete"
            message = f"only {feature[3]}/{feature[2]} contracts have usable gamma"
        self.log(symbol, day, status, len(rows), response.credits, message)
        return feature

    def log(self, symbol: str, day: str, status: str, rows: int,
            credits: CreditUsage = CreditUsage(None, None, None, None), message: str = ""):
        self.connection.execute(
            "INSERT OR REPLACE INTO fetch_log VALUES (?,?,?,?,?,?,?,?)",
            (symbol, day, status, rows, credits.consumed, credits.remaining, message,
             datetime.now(timezone.utc).replace(microsecond=0).isoformat()),
        )
        self.connection.commit()


def normalise_contract(symbol: str, day: str, row: dict[str, Any]):
    return (
        symbol.upper(), day, str(row.get("optionSymbol", "")),
        _date_value(row.get("expiration")), str(row.get("side", "")).lower(),
        _number(row.get("strike")), _integer(row.get("dte")),
        _number(row.get("volume")), _number(row.get("openInterest")),
        _number(row.get("underlyingPrice")), _integer(row.get("updated")),
        _number(row.get("iv")), _number(row.get("delta")), _number(row.get("gamma")),
        _number(row.get("bid")), _number(row.get("ask")), _number(row.get("mid")),
        _number(row.get("last")), str(row.get("gammaSource", "")),
    )


def calculate_gex(symbol: str, day: str, contracts: tuple[dict[str, Any], ...]):
    usable = []
    for row in contracts:
        gamma = _number(row.get("gamma"))
        oi = _number(row.get("openInterest"))
        spot = _number(row.get("underlyingPrice"))
        strike = _number(row.get("strike"))
        side = str(row.get("side", "")).lower()
        if None in (gamma, oi, spot, strike) or side not in {"call", "put"}:
            continue
        # Dollar gamma exposure for a 1% underlying move. Omitting 0.01 is a
        # valid raw S^2 scaling, but is 100x the convention used in GEX charts.
        absolute = gamma * oi * 100.0 * spot * spot * 0.01
        signed = absolute if side == "call" else -absolute
        usable.append((side, strike, oi, spot, absolute, signed, _number(row.get("iv"))))
    if not usable:
        return (symbol.upper(), day, len(contracts), 0, *(None for _ in range(10)))
    spots = [row[3] for row in usable]
    spot = statistics_median(spots)
    call_gex = sum(row[4] for row in usable if row[0] == "call")
    put_gex = sum(row[4] for row in usable if row[0] == "put")
    net = call_gex - put_gex
    by_strike: dict[float, float] = {}
    for row in usable:
        by_strike[row[1]] = by_strike.get(row[1], 0.0) + row[5]
    wall = max(by_strike, key=lambda strike: abs(by_strike[strike]))
    ordered = sorted(by_strike.items())
    running = 0.0
    flip = ordered[0][0]
    smallest = math.inf
    for strike, value in ordered:
        running += value
        if abs(running) < smallest:
            smallest, flip = abs(running), strike
    call_oi = sum(row[2] for row in usable if row[0] == "call")
    put_oi = sum(row[2] for row in usable if row[0] == "put")
    iv_rows = [(row[6], row[2]) for row in usable if row[6] is not None and row[2] > 0]
    weighted_iv = (
        sum(iv * weight for iv, weight in iv_rows) / sum(weight for _, weight in iv_rows)
        if iv_rows else None
    )
    return (
        symbol.upper(), day, len(contracts), len(usable), spot, call_gex, put_gex,
        net, call_gex + put_gex, wall, (spot - wall) / spot if spot else None,
        flip, weighted_iv, put_oi / call_oi if call_oi else None,
    )


def statistics_median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return (ordered[middle] if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2.0)
