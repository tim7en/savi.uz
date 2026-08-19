"""S&P-universe fundamentals snapshots and Alpha Vantage refresh jobs.

The existing ``data/sp500_data`` folder is the durable store.  Each source JSON
is wrapped with its fetch timestamp, symbol and endpoint name; a refresh writes
the same shape atomically so the historical research files and the dashboard
never diverge into two copies of the truth.
"""

from __future__ import annotations

import csv
import io
import json
import os
import threading
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import get_alphavantage_api_key
from .macro_sources import RateLimiter


DEFAULT_FOLDER = Path("data/sp500_data")
DEFAULT_REQUESTS_PER_MINUTE = 72
PLAN_REQUESTS_PER_MINUTE = 75
USER_AGENT = "savi-uz-fundamentals/1.0 (research)"
BASE_URL = "https://www.alphavantage.co/query"

# Function, existing filename suffix, dashboard label.
FUNDAMENTAL_ENDPOINTS: tuple[tuple[str, str, str], ...] = (
    ("OVERVIEW", "overview", "Company overview"),
    ("EARNINGS", "earnings", "Earnings history"),
    ("INCOME_STATEMENT", "income_statement", "Income statement"),
    ("BALANCE_SHEET", "balance_sheet", "Balance sheet"),
    ("CASH_FLOW", "cash_flow", "Cash flow"),
)
EARNINGS_ESTIMATES_ENDPOINTS: tuple[tuple[str, str, str], ...] = (
    ("EARNINGS_ESTIMATES", "earnings_estimates", "Forward earnings estimates"),
)
EARNINGS_ANALYSIS_ENDPOINTS: tuple[tuple[str, str, str], ...] = (
    ("EARNINGS", "earnings", "Reported earnings history"),
    *EARNINGS_ESTIMATES_ENDPOINTS,
)


class FundamentalsSourceError(RuntimeError):
    """Alpha Vantage returned an unavailable, throttled, or malformed payload."""


def load_universe(folder: str | Path = DEFAULT_FOLDER) -> tuple[list[str], dict[str, Any]]:
    path = Path(folder) / "sp500_symbols.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    symbols = sorted({str(symbol).strip().upper() for symbol in document.get("symbols", []) if symbol})
    return symbols, {
        "date": document.get("date"),
        "source": document.get("source"),
        "path": str(path),
    }


def _read_payload(folder: Path, ticker: str, suffix: str) -> tuple[dict[str, Any], str | None]:
    path = folder / f"{ticker}_{suffix}.json"
    if not path.is_file():
        return {}, None
    try:
        wrapper = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, None
    if isinstance(wrapper, dict) and isinstance(wrapper.get("data"), dict):
        return wrapper["data"], wrapper.get("timestamp")
    # Accept raw Alpha Vantage JSON too; this makes imports less brittle.
    return (wrapper if isinstance(wrapper, dict) else {}), None


def _float(value: Any) -> float | None:
    if value in (None, "", "None", "-", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_report(document: dict[str, Any], key: str = "quarterlyReports") -> dict[str, Any]:
    rows = [row for row in document.get(key, []) if isinstance(row, dict)]
    return max(rows, key=lambda row: str(row.get("fiscalDateEnding") or ""), default={})


def _yoy_report(document: dict[str, Any], latest: dict[str, Any]) -> dict[str, Any]:
    rows = sorted(
        (row for row in document.get("quarterlyReports", []) if isinstance(row, dict)),
        key=lambda row: str(row.get("fiscalDateEnding") or ""), reverse=True,
    )
    latest_date = str(latest.get("fiscalDateEnding") or "")
    if not latest_date:
        return {}
    try:
        target_year = str(int(latest_date[:4]) - 1)
        target_month = latest_date[5:7]
    except (ValueError, IndexError):
        return rows[4] if len(rows) > 4 else {}
    same_period = [row for row in rows[1:] if str(row.get("fiscalDateEnding") or "").startswith(target_year + "-" + target_month)]
    return same_period[0] if same_period else (rows[4] if len(rows) > 4 else {})


def _growth(current: float | None, prior: float | None) -> float | None:
    if current is None or prior in (None, 0.0):
        return None
    return round((current / prior - 1.0) * 100.0, 4)


def fundamentals_snapshot(folder: str | Path = DEFAULT_FOLDER, today: date | None = None) -> dict[str, Any]:
    """Summarise the latest stored report for every universe member."""

    root = Path(folder)
    today = today or date.today()
    symbols, universe = load_universe(root)
    companies: list[dict[str, Any]] = []
    endpoint_counts = {suffix: 0 for _, suffix, _ in FUNDAMENTAL_ENDPOINTS}

    for ticker in symbols:
        overview, overview_at = _read_payload(root, ticker, "overview")
        earnings, earnings_at = _read_payload(root, ticker, "earnings")
        income, income_at = _read_payload(root, ticker, "income_statement")
        balance, balance_at = _read_payload(root, ticker, "balance_sheet")
        cashflow, cashflow_at = _read_payload(root, ticker, "cash_flow")
        timestamps = [overview_at, earnings_at, income_at, balance_at, cashflow_at]
        for suffix, stamp in zip(endpoint_counts, timestamps):
            if stamp:
                endpoint_counts[suffix] += 1

        latest_earnings = _latest_report(earnings, "quarterlyEarnings")
        latest_income = _latest_report(income)
        prior_income = _yoy_report(income, latest_income)
        latest_balance = _latest_report(balance)
        latest_cash = _latest_report(cashflow)

        revenue = _float(latest_income.get("totalRevenue"))
        prior_revenue = _float(prior_income.get("totalRevenue"))
        net_income = _float(latest_income.get("netIncome"))
        operating_cash = _float(latest_cash.get("operatingCashflow"))
        capex = _float(latest_cash.get("capitalExpenditures"))
        free_cash_flow = (
            operating_cash - abs(capex)
            if operating_cash is not None and capex is not None else None
        )
        cash = _float(latest_balance.get("cashAndShortTermInvestments"))
        if cash is None:
            cash = _float(latest_balance.get("cashAndCashEquivalentsAtCarryingValue"))
        debt = _float(latest_balance.get("shortLongTermDebtTotal"))
        if debt is None:
            current_debt = _float(latest_balance.get("shortTermDebt")) or 0.0
            long_debt = _float(latest_balance.get("longTermDebt")) or 0.0
            debt = current_debt + long_debt if current_debt or long_debt else None

        valid_timestamps = [stamp for stamp in timestamps if stamp]
        oldest_update = min(valid_timestamps) if valid_timestamps else None
        latest_quarter = max(
            (str(value) for value in (
                overview.get("LatestQuarter"), latest_earnings.get("fiscalDateEnding"),
                latest_income.get("fiscalDateEnding"), latest_balance.get("fiscalDateEnding"),
                latest_cash.get("fiscalDateEnding"),
            ) if value and value != "None"),
            default=None,
        )
        endpoints_present = len(valid_timestamps)
        is_current = endpoints_present == len(FUNDAMENTAL_ENDPOINTS) and all(
            str(stamp)[:10] == today.isoformat() for stamp in valid_timestamps
        )

        companies.append({
            "ticker": ticker,
            "name": overview.get("Name") or ticker,
            "sector": overview.get("Sector") or "Unclassified",
            "industry": overview.get("Industry") or "",
            "currency": overview.get("Currency") or "USD",
            "latest_quarter": latest_quarter,
            "reported_date": latest_earnings.get("reportedDate"),
            "report_time": latest_earnings.get("reportTime"),
            "reported_eps": _float(latest_earnings.get("reportedEPS")),
            "estimated_eps": _float(latest_earnings.get("estimatedEPS")),
            "surprise_pct": _float(latest_earnings.get("surprisePercentage")),
            "revenue": revenue,
            "revenue_yoy_pct": _growth(revenue, prior_revenue),
            "net_income": net_income,
            "net_margin_pct": round(net_income / revenue * 100.0, 4) if net_income is not None and revenue else None,
            "operating_cashflow": operating_cash,
            "free_cash_flow": free_cash_flow,
            "cash": cash,
            "total_debt": debt,
            "market_cap": _float(overview.get("MarketCapitalization")),
            "pe_ratio": _float(overview.get("PERatio")),
            "forward_pe": _float(overview.get("ForwardPE")),
            "return_on_equity_pct": (
                _float(overview.get("ReturnOnEquityTTM")) * 100.0
                if _float(overview.get("ReturnOnEquityTTM")) is not None else None
            ),
            "updated_at": oldest_update,
            "endpoints_present": endpoints_present,
            "status": "current" if is_current else ("partial" if endpoints_present < 5 else "stale"),
        })

    latest_quarter = max((row["latest_quarter"] for row in companies if row["latest_quarter"]), default=None)
    surprises = [row["surprise_pct"] for row in companies if row["surprise_pct"] is not None]
    revenue_growth = [row["revenue_yoy_pct"] for row in companies if row["revenue_yoy_pct"] is not None]
    sectors = sorted({row["sector"] for row in companies})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "today": today.isoformat(),
        "universe": universe,
        "count": len(companies),
        "current_count": sum(row["status"] == "current" for row in companies),
        "complete_count": sum(row["endpoints_present"] == 5 for row in companies),
        "latest_quarter": latest_quarter,
        "positive_surprise_count": sum(value > 0 for value in surprises),
        "surprise_count": len(surprises),
        "median_revenue_growth_pct": _median(revenue_growth),
        "sectors": sectors,
        "endpoint_counts": endpoint_counts,
        "companies": companies,
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    rows = sorted(values)
    middle = len(rows) // 2
    return rows[middle] if len(rows) % 2 else (rows[middle - 1] + rows[middle]) / 2.0


class AlphaVantageFundamentalsClient:
    """Five normalized fundamental endpoints, paced below the premium cap."""

    def __init__(self, api_key: str, requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
                 timeout: float = 60.0, retries: int = 2) -> None:
        if not api_key:
            raise ValueError("Alpha Vantage API key is required")
        if not 1 <= requests_per_minute <= PLAN_REQUESTS_PER_MINUTE:
            raise ValueError(f"requests_per_minute must be 1..{PLAN_REQUESTS_PER_MINUTE}")
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.limiter = RateLimiter(requests_per_minute)

    def fetch(self, function: str, ticker: str) -> dict[str, Any]:
        allowed = {row[0] for row in FUNDAMENTAL_ENDPOINTS + EARNINGS_ESTIMATES_ENDPOINTS}
        if function not in allowed:
            raise ValueError(f"unsupported fundamental function {function}")
        params = {"function": function, "symbol": ticker, "apikey": self.api_key}
        request = Request(f"{BASE_URL}?{urlencode(params)}", headers={"User-Agent": USER_AGENT})
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            self.limiter.acquire()
            try:
                with urlopen(request, timeout=self.timeout) as response:  # nosec B310
                    document = json.loads(response.read().decode("utf-8"))
                if not isinstance(document, dict):
                    raise FundamentalsSourceError("Alpha Vantage returned a non-object payload")
                for key in ("Note", "Information", "Error Message"):
                    if key in document:
                        raise FundamentalsSourceError(f"Alpha Vantage: {document[key]}")
                if not document and function == "EARNINGS_ESTIMATES":
                    # The endpoint returns an empty object for unsupported or
                    # delisted legacy-universe symbols. Cache that as checked
                    # coverage so every dashboard refresh does not retry it.
                    return {"symbol": ticker, "estimates": []}
                if not document:
                    raise FundamentalsSourceError(f"Alpha Vantage returned no {function} data for {ticker}")
                return document
            except (HTTPError, URLError, json.JSONDecodeError, FundamentalsSourceError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(2 ** attempt)
        # Never include the request URL here: it contains the API key.
        raise FundamentalsSourceError(f"{function} for {ticker} failed: {last_error}") from last_error


def _wrapper_is_current(path: Path, today: date) -> bool:
    if not path.is_file():
        return False
    try:
        stamp = json.loads(path.read_text(encoding="utf-8")).get("timestamp")
        return bool(stamp and str(stamp)[:10] == today.isoformat())
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


def _write_wrapper(path: Path, ticker: str, suffix: str, document: dict[str, Any]) -> None:
    wrapper = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "symbol": ticker,
        "data_type": suffix,
        "data": document,
    }
    temporary = path.with_suffix(path.suffix + ".tmp-" + uuid.uuid4().hex[:8])
    temporary.write_text(json.dumps(wrapper, indent=2), encoding="utf-8")
    os.replace(temporary, path)


class FundamentalsRefreshManager:
    """Resumable 465-company refresh with browser-friendly progress."""

    def __init__(
        self,
        folder: str | Path = DEFAULT_FOLDER,
        requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
        client_factory: Callable[..., Any] = AlphaVantageFundamentalsClient,
        api_key_factory: Callable[[], str] = get_alphavantage_api_key,
        today_factory: Callable[[], date] = date.today,
    ) -> None:
        self.folder = Path(folder)
        self.requests_per_minute = requests_per_minute
        self.client_factory = client_factory
        self.api_key_factory = api_key_factory
        self.today_factory = today_factory
        self._lock = threading.Lock()
        self._state = self._idle_state()
        self._thread: threading.Thread | None = None
        self.endpoints = FUNDAMENTAL_ENDPOINTS

    def _idle_state(self) -> dict[str, Any]:
        return {
            "state": "idle", "running": False, "completed": 0, "total": 0,
            "current_symbol": None, "current_dataset": None, "files_updated": 0,
            "skipped_current": 0, "errors": [], "requests_per_minute": self.requests_per_minute,
            "started_at": None, "finished_at": None,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            state = dict(self._state)
            state["errors"] = [dict(row) for row in self._state["errors"]]
        total = state["total"]
        state["percent"] = round(state["completed"] / total * 100.0, 1) if total else 0.0
        remaining = max(total - state["completed"], 0)
        state["estimated_seconds_remaining"] = round(remaining / self.requests_per_minute * 60)
        return state

    def _set(self, **changes: Any) -> None:
        with self._lock:
            self._state.update(changes)

    def _error(self, ticker: str, dataset: str, message: str) -> None:
        with self._lock:
            self._state["errors"].append({
                "ticker": ticker, "dataset": dataset, "message": message[:240],
            })

    def start(self, force: bool = False) -> bool:
        api_key = self.api_key_factory()
        with self._lock:
            if self._state["running"]:
                return False
            self._state = self._idle_state() | {
                "state": "starting", "running": True,
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            self._thread = threading.Thread(
                target=self._run, args=(api_key, force), name="dashboard-fundamentals-refresh",
                daemon=True,
            )
            self._thread.start()
        return True

    def _run(self, api_key: str, force: bool) -> None:
        try:
            today = self.today_factory()
            symbols, _ = load_universe(self.folder)
            full_queue = [
                (ticker, function, suffix, label)
                for ticker in symbols for function, suffix, label in self.endpoints
            ]
            queue = [
                item for item in full_queue
                if force or not _wrapper_is_current(self.folder / f"{item[0]}_{item[2]}.json", today)
            ]
            self._set(total=len(queue), skipped_current=len(full_queue) - len(queue), state="running")
            if not queue:
                self._set(
                    state="complete", running=False,
                    finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                )
                return

            client = self.client_factory(api_key, requests_per_minute=self.requests_per_minute)
            for index, (ticker, function, suffix, label) in enumerate(queue, start=1):
                self._set(current_symbol=ticker, current_dataset=label)
                try:
                    document = client.fetch(function, ticker)
                    _write_wrapper(self.folder / f"{ticker}_{suffix}.json", ticker, suffix, document)
                    with self._lock:
                        self._state["files_updated"] += 1
                except Exception as exc:
                    self._error(ticker, label, str(exc))
                self._set(completed=index)

            errors = len(self.status()["errors"])
            self._set(
                state="complete_with_errors" if errors else "complete", running=False,
                current_symbol=None, current_dataset=None,
                finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
        except Exception as exc:
            self._error("refresh", "setup", str(exc))
            self._set(
                state="failed", running=False, current_symbol=None, current_dataset=None,
                finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )


class EarningsEstimatesRefreshManager(FundamentalsRefreshManager):
    """Refresh only forward EPS/revenue estimates, separately from statements."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.endpoints = EARNINGS_ESTIMATES_ENDPOINTS


def refresh_earnings_calendar(
    folder: str | Path,
    api_key: str,
    horizon: str = "3month",
) -> int:
    """Download the official upcoming calendar CSV and cache universe rows."""

    root = Path(folder)
    symbols, _ = load_universe(root)
    query = urlencode({
        "function": "EARNINGS_CALENDAR", "horizon": horizon, "apikey": api_key,
    })
    request = Request(f"{BASE_URL}?{query}", headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:  # nosec B310
        body = response.read().decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(body)))
    if not rows or "symbol" not in rows[0]:
        raise FundamentalsSourceError("Alpha Vantage returned no earnings calendar data")
    allowed = set(symbols)
    selected = [row for row in rows if row.get("symbol") in allowed]
    wrapper = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "horizon": horizon,
        "rows": selected,
    }
    path = root / "earnings_calendar.json"
    temporary = path.with_suffix(path.suffix + ".tmp-" + uuid.uuid4().hex[:8])
    temporary.write_text(json.dumps(wrapper, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return len(selected)


class EarningsAnalysisRefreshManager(FundamentalsRefreshManager):
    """Refresh actual/consensus history, forward estimates, and calendar."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.endpoints = EARNINGS_ANALYSIS_ENDPOINTS

    def _run(self, api_key: str, force: bool) -> None:
        self._set(current_symbol="CALENDAR", current_dataset="Upcoming earnings calendar")
        try:
            refresh_earnings_calendar(self.folder, api_key)
        except Exception as exc:
            self._error("CALENDAR", "Upcoming earnings calendar", str(exc))
        super()._run(api_key, force)
