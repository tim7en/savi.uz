"""Resumable Alpha Vantage intraday OHLCV for the cross-asset ETF book.

Alpha Vantage returns at most one historical month per intraday request.  This
downloader therefore records every completed ``(ticker, month)`` window in the
same SQLite schema used by the rest of the project, so a stopped run resumes
without spending requests on data already stored.

Only regular-session bars are requested.  Timestamps arrive in US/Eastern and
are converted to UTC before storage.  ``adjusted=true`` is intentional: the
book contains split-heavy products such as VXX, and an unadjusted reverse split
would otherwise look like a strategy windfall or collapse.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.config import get_alphavantage_api_key  # noqa: E402
from savi_uz.intraday_store import IntradayStore  # noqa: E402


FREQUENCY = "30min"
SOURCE = "ALPHAVANTAGE"
BASE_URL = "https://www.alphavantage.co/query"
DEFAULT_START = "2017-01"

# Kept in sync with ``download_cross_asset_etfs.py``.  The 30-minute run is a
# resolution test of the same book, not a fresh universe-selection exercise.
UNIVERSE = {
    "GLD": ("metals", "PAXGUSDT / XAUUSDT"),
    "SLV": ("metals", "XAGUSDT"),
    "GDX": ("metals", None),
    "PPLT": ("metals", None),
    "USO": ("energy", None),
    "UNG": ("energy", None),
    "BNO": ("energy", None),
    "DBC": ("commodity", None),
    "DBA": ("commodity", None),
    "GSG": ("commodity", None),
    "TLT": ("duration", None),
    "IEF": ("duration", None),
    "SHY": ("duration", None),
    "TIP": ("duration", None),
    "LQD": ("credit", None),
    "HYG": ("credit", None),
    "UUP": ("fx", "USDT complex"),
    "FXE": ("fx", "EURUSDT"),
    "FXY": ("fx", "JPY pairs, thin"),
    "FXB": ("fx", "GBPUSDT"),
    "VXX": ("volatility", None),
}

VENUE_SCHEMA = """
CREATE TABLE IF NOT EXISTS venue_map (
    ticker    TEXT PRIMARY KEY,
    sleeve    TEXT,
    binance   TEXT
);
"""


@dataclass(frozen=True)
class IntradayBar:
    ticker: str
    frequency: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class MonthResult:
    ticker: str
    month: str
    bars: tuple[IntradayBar, ...]
    status: str
    message: str = ""


class RequestPacer:
    """Thread-safe spacing between request starts, not response completions."""

    def __init__(self, requests_per_minute: int):
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be positive")
        self.interval = 60.0 / requests_per_minute
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_at - now)
            self.next_at = max(now, self.next_at) + self.interval
        if delay:
            time.sleep(delay)


def previous_month(today: date | None = None) -> str:
    current = today or date.today()
    if current.month == 1:
        return f"{current.year - 1}-12"
    return f"{current.year}-{current.month - 1:02d}"


def month_range(first: str, last: str) -> list[str]:
    try:
        cursor = date.fromisoformat(first + "-01")
        stop = date.fromisoformat(last + "-01")
    except ValueError as exc:
        raise ValueError("months must use YYYY-MM") from exc
    if cursor > stop:
        raise ValueError("start month is after end month")
    months = []
    while cursor <= stop:
        months.append(cursor.strftime("%Y-%m"))
        cursor = date(cursor.year + (cursor.month // 12), cursor.month % 12 + 1, 1)
    return months


def utc_timestamp(value: str) -> str:
    eastern = ZoneInfo("America/New_York")
    parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=eastern)
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def parse_series(ticker: str, payload: dict) -> tuple[IntradayBar, ...]:
    series = payload.get(f"Time Series ({FREQUENCY})") or {}
    bars = []
    for stamp, values in series.items():
        try:
            bars.append(IntradayBar(
                ticker=ticker,
                frequency=FREQUENCY,
                timestamp=utc_timestamp(stamp),
                open=float(values["1. open"]),
                high=float(values["2. high"]),
                low=float(values["3. low"]),
                close=float(values["4. close"]),
                volume=float(values["5. volume"]),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(sorted(bars, key=lambda row: row.timestamp))


def response_message(payload: dict) -> str:
    return str(
        payload.get("Information") or payload.get("Note")
        or payload.get("Error Message") or "empty response"
    )


def fetch_month(
    ticker: str, month: str, api_key: str, pacer: RequestPacer, attempts: int = 5,
) -> MonthResult:
    query = urllib.parse.urlencode({
        "function": "TIME_SERIES_INTRADAY",
        "symbol": ticker,
        "interval": FREQUENCY,
        "month": month,
        "outputsize": "full",
        "adjusted": "true",
        "extended_hours": "false",
        "apikey": api_key,
    })
    for attempt in range(attempts):
        pacer.wait()
        try:
            request = urllib.request.Request(
                f"{BASE_URL}?{query}", headers={"User-Agent": "savi-uz-research/1.0"}
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == attempts - 1:
                return MonthResult(ticker, month, (), "error", f"{type(exc).__name__}: {exc}")
            time.sleep(2 ** attempt)
            continue

        bars = parse_series(ticker, payload)
        if bars:
            return MonthResult(ticker, month, bars, "ok")
        message = response_message(payload)
        lowered = message.lower()
        if "rate limit" in lowered or "call frequency" in lowered:
            if attempt == attempts - 1:
                return MonthResult(ticker, month, (), "rate_limited", message)
            time.sleep(65)
            continue
        # Months before a fund listed legitimately have no bars.  Alpha Vantage
        # normally returns an empty series; explicit API errors remain errors.
        if message == "empty response":
            return MonthResult(ticker, month, (), "ok", "no bars")
        return MonthResult(ticker, month, (), "error", message)
    return MonthResult(ticker, month, (), "error", "retry budget exhausted")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path,
                        default=Path("data/cross_assets/etf_30min.db"))
    parser.add_argument("--symbols", nargs="+", default=sorted(UNIVERSE))
    parser.add_argument("--start", default=DEFAULT_START, help="first month, YYYY-MM")
    parser.add_argument("--end", default=previous_month(), help="last month, YYYY-MM")
    parser.add_argument("--requests-per-minute", type=int, default=14)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-requests", type=int, default=0,
                        help="live request cap; zero means no cap")
    parser.add_argument("--force", action="store_true",
                        help="refetch months already marked complete")
    parser.add_argument("--plan", action="store_true")
    return parser.parse_args(argv)


def store_result(store: IntradayStore, result: MonthResult, run_id: str) -> None:
    first = date.fromisoformat(result.month + "-01")
    next_month = date(first.year + (first.month // 12), first.month % 12 + 1, 1)
    last = next_month - timedelta(days=1)
    store.write_bars(result.bars)
    store.mark_window(
        result.ticker, FREQUENCY, first, last, list(result.bars), False,
        datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    )
    store.log(
        run_id, datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        SOURCE, f"{result.ticker}/{result.month}", len(result.bars), result.status,
        result.message,
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    months = month_range(args.start, args.end)
    symbols = tuple(dict.fromkeys(symbol.upper() for symbol in args.symbols))
    if args.workers < 1:
        raise SystemExit("error: --workers must be positive")

    with IntradayStore(args.db) as store:
        store.connection.executescript(VENUE_SCHEMA)
        for ticker, (sleeve, binance) in UNIVERSE.items():
            store.connection.execute(
                "INSERT OR REPLACE INTO venue_map VALUES (?,?,?)", (ticker, sleeve, binance)
            )
        store.connection.commit()

        done = store.completed_windows(FREQUENCY)
        jobs = [
            (ticker, month) for ticker in symbols for month in months
            if args.force or (ticker, int(month[:4]), int(month[5:])) not in done
        ]
        if args.max_requests:
            jobs = jobs[:args.max_requests]
        print(f"symbols: {len(symbols)}; months: {args.start}..{args.end}; "
              f"pending requests: {len(jobs):,}; completed windows: {len(done):,}")
        estimate = len(jobs) / max(args.requests_per_minute, 1)
        print(f"pace: {args.requests_per_minute}/minute with {args.workers} workers; "
              f"minimum wall time ~{estimate:.1f} minutes")
        if args.plan or not jobs:
            return 0

        api_key = get_alphavantage_api_key()
        pacer = RequestPacer(args.requests_per_minute)
        run_id = uuid.uuid4().hex[:12]
        pending: dict[Future, tuple[str, str]] = {}
        completed = stored = failures = 0

        def consume(future: Future) -> None:
            nonlocal completed, stored, failures
            result = future.result()
            completed += 1
            if result.status == "ok":
                store_result(store, result, run_id)
                stored += len(result.bars)
            else:
                failures += 1
                store.log(
                    run_id, datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                    SOURCE, f"{result.ticker}/{result.month}", 0, result.status,
                    result.message,
                )
                print(f"  {result.ticker}/{result.month} {result.status}: "
                      f"{result.message[:120]}", flush=True)
            if completed % 25 == 0 or completed == len(jobs):
                print(f"  progress {completed:,}/{len(jobs):,}; "
                      f"{stored:,} bars; {failures} failures", flush=True)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for ticker, month in jobs:
                while len(pending) >= args.workers:
                    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        pending.pop(future, None)
                        consume(future)
                future = pool.submit(fetch_month, ticker, month, api_key, pacer)
                pending[future] = (ticker, month)
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    pending.pop(future, None)
                    consume(future)

        coverage = store.coverage()
        print(f"\nstored {stored:,} bars this run; {failures} failed requests")
        print(f"database: {args.db}")
        for ticker, frequency, first, last, rows in coverage:
            if frequency == FREQUENCY:
                print(f"  {ticker:5s} {rows:>7,d}  {first[:10]} -> {last[:10]}")
        return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
