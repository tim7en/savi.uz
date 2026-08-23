"""Download unadjusted daily closes, the correct spot reference for option strikes.

Option strikes print as they were quoted: a pre-split AAPL chain is struck around
$500, not around the $120 that the same session shows in a back-adjusted bar
series.  Feeding an adjusted close into the feature layer therefore compares
prices in two different units, and every moneyness-dependent measurement -- ATM
volatility, skew, the gamma flip -- is computed against the wrong strikes.

Splits are the visible half of the problem and dividends are the quiet half: a
steady payer drifts away from its adjusted series a few percent a year with no
split anywhere, which is large enough to move an ATM interpolation without ever
looking obviously wrong.

``TIME_SERIES_DAILY_ADJUSTED`` returns both series in one request per symbol, so
the raw close is stored alongside the adjusted one and the ratio between them
doubles as a per-session audit of how far off the old spot was.

Note the vendor is not self-consistent about ticker format: this endpoint wants
``BRK-B`` while the option endpoint wants ``BRKB``, so no symbol translation is
applied here.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ENDPOINT = "https://www.alphavantage.co/query"
SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_closes (
  symbol TEXT, observation_date TEXT,
  raw_close REAL, adj_close REAL,
  split_coefficient REAL, dividend REAL,
  PRIMARY KEY (symbol, observation_date));
"""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=pathlib.Path,
                        default=pathlib.Path("data/options/options/alphavantage.db"))
    parser.add_argument("--universe", type=pathlib.Path,
                        default=pathlib.Path(
                            "data/intraday/intraday/tradfi_universe.json"))
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--refresh", action="store_true",
                        help="re-fetch symbols already stored")
    return parser.parse_args(argv)


def api_key() -> str:
    for line in pathlib.Path(".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            name, value = line.split("=", 1)
            if name.strip() == "ALPHAVANTAGE_API_KEY":
                return value.strip().strip('"').strip("'")
    raise SystemExit("error: ALPHAVANTAGE_API_KEY missing from .env")


class _Pacer:
    def __init__(self, requests_per_minute: float):
        self.interval = 60.0 / max(requests_per_minute, 1e-9)
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_at - now)
            self.next_at = max(now, self.next_at) + self.interval
        if delay:
            time.sleep(delay)


PACER = _Pacer(70.0)


def fetch(symbol: str, key: str, attempts: int = 4):
    params = {"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": symbol,
              "outputsize": "full", "apikey": key}
    url = f"{ENDPOINT}?{urllib.parse.urlencode(params)}"
    for attempt in range(attempts):
        PACER.wait()
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                payload = json.loads(response.read().decode())
        except Exception as error:
            time.sleep(2 * (attempt + 1))
            if attempt == attempts - 1:
                return None, f"transport: {error}"
            continue
        for gate in ("Note", "Information"):
            if gate in payload:
                time.sleep(15 * (attempt + 1))
                if attempt == attempts - 1:
                    return None, f"{gate}: {str(payload[gate])[:120]}"
                break
        else:
            if "Error Message" in payload:
                return None, f"error: {str(payload['Error Message'])[:120]}"
            series = payload.get("Time Series (Daily)")
            if not series:
                return None, "empty series"
            return series, "ok"
    return None, "exhausted retries"


def main(argv=None):
    args = parse_args(argv)
    key = api_key()
    symbols = args.symbols or json.loads(
        args.universe.read_text(encoding="utf-8"))["tickers"]

    args.db.parent.mkdir(parents=True, exist_ok=True)
    store = sqlite3.connect(args.db)
    store.executescript(SCHEMA)
    store.commit()

    if not args.refresh:
        have = {s for (s,) in store.execute(
            "SELECT DISTINCT symbol FROM raw_closes")}
        symbols = [s for s in symbols if s not in have]
    print(f"{len(symbols)} symbols to fetch", flush=True)
    if not symbols:
        return 0

    def work(symbol):
        series, status = fetch(symbol, key)
        return symbol, series, status

    stored = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for symbol, series, status in pool.map(work, symbols):
            if series is None:
                print(f"  {symbol:6s} FAIL {status}", flush=True)
                failed += 1
                continue
            rows = []
            for day, fields in series.items():
                try:
                    rows.append((
                        symbol, day,
                        float(fields["4. close"]),
                        float(fields["5. adjusted close"]),
                        float(fields.get("8. split coefficient", 1.0)),
                        float(fields.get("7. dividend amount", 0.0)),
                    ))
                except (KeyError, TypeError, ValueError):
                    continue
            store.executemany(
                "INSERT OR REPLACE INTO raw_closes VALUES (?,?,?,?,?,?)", rows)
            store.commit()
            stored += 1
            # The ratio is the whole point: 1.0 means the old adjusted spot was
            # already correct for this name, anything else means it was not.
            worst = max((r[2] / r[3] for r in rows if r[3]), default=1.0)
            print(f"  {symbol:6s} {len(rows):5d} sessions  max raw/adj {worst:6.2f}",
                  flush=True)

    print(f"\ndone: {stored} symbols stored, {failed} failed")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
