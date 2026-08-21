"""Alpha Vantage intraday bars for the Binance trad-FI equity universe.

Two problems this fixes at once.

*The volume is wrong.*  The existing 5-minute book comes from Tiingo, whose US
intraday feed is IEX only.  IEX prints a low single-digit share of consolidated
volume, and 16.9% of the stored bars carry no volume at all.  Every volume-based
result in the programme was measured on that, including the two volume overlays
that were rejected.  Alpha Vantage serves consolidated tape, so this rebuilds the
book on volume that means what it says.

*The universe is stale.*  Binance now lists 137 US trad-FI perpetuals against the
42 names held here.  The perpetual list is the universe definition only -- bars
still come from Alpha Vantage for the underlying US listing.

Written to a separate database from the IEX book on purpose.  The old bars stay
addressable so the two volume series can be compared rather than silently
replaced; nothing is overwritten until that comparison has been made.

Base assets are not tickers.  Binance carries ``PAYP`` beside ``PYPL`` and
``MVLL`` beside ``MRVL``, and several bases (``SNXX``, ``QNTX``, ``DRAM``) do not
correspond to any US listing.  ``--validate`` spends one request per candidate to
find out which resolve before the full history is committed, which is the
difference between 140 wasted requests and 19,600.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.config import get_alphavantage_api_key  # noqa: E402
from savi_uz.intraday_store import IntradayStore  # noqa: E402
from savi_uz.tradfi_universe import (  # noqa: E402
    CURATED_YAHOO_TICKERS,
    BinanceTradFiClient,
)

SOURCE = "ALPHAVANTAGE"
BASE_URL = "https://www.alphavantage.co/query"

#: Names held in the IEX book that Binance does not list. Kept regardless.
ALWAYS_INCLUDE = ("GLD", "KWEB", "SLV")

VALIDATION_SCHEMA = """
CREATE TABLE IF NOT EXISTS symbol_validation (
    ticker      TEXT PRIMARY KEY,
    resolved    INTEGER NOT NULL,
    first_bar   TEXT,
    bars_seen   INTEGER,
    message     TEXT,
    checked_at  TEXT
);
"""


class RequestPacer:
    """Even spacing between request starts, shared across worker threads."""

    def __init__(self, requests_per_minute: float):
        if requests_per_minute <= 0:
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


@dataclass(frozen=True)
class Bar:
    ticker: str
    frequency: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Fetch:
    ticker: str
    month: str
    bars: tuple[Bar, ...]
    status: str
    message: str = ""


def us_equity_universe(cache: Path, refresh: bool = False) -> list[str]:
    """Binance trad-FI US equity bases mapped to their US listing.

    Cached, because Binance's ``exchangeInfo`` is a single large response that
    times out often enough to kill a long run at its first step, and the answer
    changes on the timescale of new listings rather than of a download.
    """
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))["tickers"]

    instruments = None
    for attempt in range(3):
        try:
            instruments = BinanceTradFiClient().fetch_tradfi_instruments()
            break
        except (OSError, TimeoutError, ValueError) as exc:
            if attempt == 2:
                raise SystemExit(
                    f"error: Binance universe unavailable ({exc}). Retry, or pass "
                    f"--symbols explicitly."
                ) from exc
            time.sleep(3 * (attempt + 1))

    tickers = set(ALWAYS_INCLUDE)
    for instrument in instruments:
        if instrument.region != "US" or instrument.is_pre_ipo:
            continue
        mapped = CURATED_YAHOO_TICKERS.get(instrument.base_asset,
                                           instrument.base_asset)
        # Curated entries may point off-exchange; only US listings apply here.
        if "." in mapped:
            continue
        tickers.add(mapped)

    resolved = sorted(tickers)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "fetched_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "instruments": len(instruments), "tickers": resolved}, indent=1),
        encoding="utf-8")
    return resolved


def month_range(first: str, last: str) -> list[str]:
    cursor = date.fromisoformat(first + "-01")
    stop = date.fromisoformat(last + "-01")
    if cursor > stop:
        raise ValueError("start month is after end month")
    months = []
    while cursor <= stop:
        months.append(cursor.strftime("%Y-%m"))
        cursor = date(cursor.year + (cursor.month // 12), cursor.month % 12 + 1, 1)
    return months


def previous_month(today: date | None = None) -> str:
    current = today or date.today()
    if current.month == 1:
        return f"{current.year - 1}-12"
    return f"{current.year}-{current.month - 1:02d}"


def utc_timestamp(value: str) -> str:
    eastern = ZoneInfo("America/New_York")
    parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=eastern)
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def call(params: dict, pacer: RequestPacer, attempts: int = 4):
    """One Alpha Vantage call, returning (payload, error)."""
    query = urllib.parse.urlencode(params)
    for attempt in range(attempts):
        pacer.wait()
        try:
            request = urllib.request.Request(
                f"{BASE_URL}?{query}",
                headers={"User-Agent": "savi-uz-research/1.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8")), None
        except (OSError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == attempts - 1:
                return None, f"{type(exc).__name__}: {exc}"
            time.sleep(2 ** attempt)
    return None, "retry budget exhausted"


def message_of(payload: dict) -> str:
    return str(payload.get("Information") or payload.get("Note")
               or payload.get("Error Message") or "empty response")


def parse_bars(ticker: str, frequency: str, payload: dict) -> tuple[Bar, ...]:
    series = payload.get(f"Time Series ({frequency})") or {}
    bars = []
    for stamp, values in series.items():
        try:
            bars.append(Bar(ticker, frequency, utc_timestamp(stamp),
                            float(values["1. open"]), float(values["2. high"]),
                            float(values["3. low"]), float(values["4. close"]),
                            float(values["5. volume"])))
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(sorted(bars, key=lambda b: b.timestamp))


def fetch_month(ticker: str, month: str, frequency: str, key: str,
                pacer: RequestPacer) -> Fetch:
    payload, error = call({
        "function": "TIME_SERIES_INTRADAY", "symbol": ticker,
        "interval": frequency, "month": month, "outputsize": "full",
        "adjusted": "true", "extended_hours": "false", "apikey": key,
    }, pacer)
    if payload is None:
        return Fetch(ticker, month, (), "error", error or "unknown")
    bars = parse_bars(ticker, frequency, payload)
    if bars:
        return Fetch(ticker, month, bars, "ok")
    text = message_of(payload)
    lowered = text.lower()
    if "rate limit" in lowered or "call frequency" in lowered or "burst" in lowered:
        return Fetch(ticker, month, (), "throttled", text)
    # A month before the listing legitimately has no bars.
    if text == "empty response":
        return Fetch(ticker, month, (), "ok", "no bars")
    return Fetch(ticker, month, (), "error", text)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path,
                        default=Path("data/intraday/bars_av.db"))
    parser.add_argument("--interval", default="5min",
                        choices=["1min", "5min", "15min", "30min", "60min"])
    parser.add_argument("--start", default="2015-01", help="first month, YYYY-MM")
    parser.add_argument("--end", default=previous_month(), help="last month, YYYY-MM")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="override the Binance-derived universe")
    parser.add_argument("--universe-cache", type=Path,
                        default=Path("data/intraday/tradfi_universe.json"))
    parser.add_argument("--refresh-universe", action="store_true",
                        help="re-query Binance instead of using the cache")
    parser.add_argument("--requests-per-minute", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--validate", action="store_true",
                        help="one request per symbol to see which tickers resolve")
    parser.add_argument("--resolved-only", action="store_true",
                        help="restrict the download to validated tickers")
    parser.add_argument("--ignore-listing-dates", action="store_true",
                        help="request every month even before a ticker listed")
    parser.add_argument("--plan", action="store_true",
                        help="report the request budget, spend nothing")
    return parser.parse_args(argv)


def validate(store: IntradayStore, tickers: list[str], args, key: str) -> None:
    """Spend one recent month per ticker to learn which bases are real."""
    store.connection.executescript(VALIDATION_SCHEMA)
    store.connection.commit()
    done = {r[0] for r in store.connection.execute(
        "SELECT ticker FROM symbol_validation")}
    pending = [t for t in tickers if t not in done]
    print(f"validating {len(pending)} of {len(tickers)} tickers "
          f"({len(done)} already checked)\n", flush=True)
    pacer = RequestPacer(args.requests_per_minute)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    resolved = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_month, t, args.end, args.interval, key, pacer): t
                   for t in pending}
        for future in list(futures):
            result = future.result()
            ok = bool(result.bars)
            resolved += ok
            failed += not ok
            store.connection.execute(
                "INSERT OR REPLACE INTO symbol_validation VALUES (?,?,?,?,?,?)",
                (result.ticker, int(ok),
                 result.bars[0].timestamp[:10] if result.bars else None,
                 len(result.bars), result.message or None, now))
            store.connection.commit()
            if not ok:
                print(f"  {result.ticker:6s} unresolved: "
                      f"{(result.message or result.status)[:90]}", flush=True)
    print(f"\n  resolved {resolved}, unresolved {failed}")


def main(argv=None) -> int:
    args = parse_args(argv)
    tickers = args.symbols or us_equity_universe(args.universe_cache,
                                                 args.refresh_universe)
    tickers = [t.upper() for t in dict.fromkeys(tickers)]
    months = month_range(args.start, args.end)

    args.db.parent.mkdir(parents=True, exist_ok=True)
    with IntradayStore(args.db) as store:
        if args.resolved_only or args.validate:
            store.connection.executescript(VALIDATION_SCHEMA)
            store.connection.commit()
        if args.validate:
            return validate(store, tickers, args, get_alphavantage_api_key())
        if args.resolved_only:
            good = {r[0] for r in store.connection.execute(
                "SELECT ticker FROM symbol_validation WHERE resolved=1")}
            skipped = [t for t in tickers if t not in good]
            tickers = [t for t in tickers if t in good]
            if skipped:
                print(f"skipping {len(skipped)} unvalidated: "
                      f"{' '.join(skipped[:20])}"
                      f"{' ...' if len(skipped) > 20 else ''}")

        done = store.completed_windows(args.interval)
        # Months before a ticker listed return an empty series, so requesting
        # them costs budget and yields nothing. Forty-odd of these names listed
        # after 2015, which is a quarter of the naive request count.
        listed: dict[str, str] = {}
        if not args.ignore_listing_dates:
            listed = {r[0]: r[1][:7] for r in store.connection.execute(
                "SELECT ticker, history_start FROM symbols "
                "WHERE history_start IS NOT NULL") if r[1]}
        jobs = [(t, m) for t in tickers for m in months
                if (t, int(m[:4]), int(m[5:])) not in done
                and m >= listed.get(t, "0000-00")]
        if listed:
            naive = len(tickers) * len(months) - len(done)
            print(f"listing dates known for {len(listed)} tickers; "
                  f"skipping {naive - len(jobs):,} pre-listing months")
        if args.max_requests:
            jobs = jobs[:args.max_requests]

        hours = len(jobs) / max(args.requests_per_minute, 0.1) / 60
        print(f"universe: {len(tickers)} tickers; interval: {args.interval}; "
              f"months: {args.start}..{args.end} ({len(months)})")
        print(f"pending requests: {len(jobs):,}; completed windows: {len(done):,}")
        print(f"pace: {args.requests_per_minute}/min with {args.workers} workers "
              f"-> {hours:.1f} hours minimum ({hours/24:.1f} days)")
        if args.plan or not jobs:
            return 0

        key = get_alphavantage_api_key()
        pacer = RequestPacer(args.requests_per_minute)
        run_id = uuid.uuid4().hex[:12]
        completed = stored = failures = throttled = 0

        def consume(future) -> None:
            nonlocal completed, stored, failures, throttled
            result = future.result()
            completed += 1
            now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            if result.status == "ok":
                first = date.fromisoformat(result.month + "-01")
                nxt = date(first.year + (first.month // 12), first.month % 12 + 1, 1)
                store.write_bars(result.bars)
                store.mark_window(result.ticker, args.interval, first,
                                  nxt - timedelta(days=1), list(result.bars),
                                  False, now)
                stored += len(result.bars)
            else:
                failures += 1
                throttled += result.status == "throttled"
                print(f"  {result.ticker}/{result.month} {result.status}: "
                      f"{result.message[:100]}", flush=True)
            store.log(run_id, now, SOURCE, f"{result.ticker}/{result.month}",
                      len(result.bars), result.status, result.message)
            if completed % 50 == 0 or completed == len(jobs):
                print(f"  progress {completed:,}/{len(jobs):,}; {stored:,} bars; "
                      f"{failures} failed ({throttled} throttled)", flush=True)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending: dict = {}
            for ticker, month in jobs:
                while len(pending) >= args.workers:
                    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        pending.pop(future, None)
                        consume(future)
                pending[pool.submit(fetch_month, ticker, month,
                                    args.interval, key, pacer)] = (ticker, month)
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    pending.pop(future, None)
                    consume(future)

        print(f"\nstored {stored:,} bars; {failures} failures "
              f"({throttled} throttled)")
        print(f"database: {args.db}")
        return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
