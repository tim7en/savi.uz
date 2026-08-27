"""Download hourly intraday bars from Alpha Vantage into the bar database.

The 5-minute history in this project starts in 2015, which is where the vendor's
5-minute archive begins.  The 60-minute series reaches back to 2000, so an hourly
download roughly doubles the usable intraday span -- it covers the dot-com unwind
and 2008, neither of which exists at 5-minute resolution anywhere in this data.

Alpha Vantage serves intraday one calendar month per request, so the work is one
request per symbol-month and the run is resumable at that granularity: months
already stored are skipped.

**These bars are adjusted**, matching the 5-minute rows already in this table.
That is correct for returns and volatility and wrong for anything compared
against option strikes, which print unadjusted -- the mistake that corrupted the
option feature set before ``raw_closes`` existed.  Use raw_closes for moneyness,
never these.
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
from datetime import date

ENDPOINT = "https://www.alphavantage.co/query"
SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
  ticker TEXT, frequency TEXT, ts TEXT,
  open REAL, high REAL, low REAL, close REAL, volume REAL,
  PRIMARY KEY (ticker, frequency, ts));
CREATE INDEX IF NOT EXISTS idx_bars_ticker_ts ON bars(ticker, ts);
"""


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=pathlib.Path,
                   default=pathlib.Path("data/intraday/intraday/bars_av.db"))
    p.add_argument("--symbols", nargs="+", default=["SPY", "QQQ"])
    p.add_argument("--interval", default="60min",
                   choices=["1min", "5min", "15min", "30min", "60min"])
    p.add_argument("--start", default="2000-01", help="first month, YYYY-MM")
    p.add_argument("--end", default=None, help="last month, YYYY-MM (default: this month)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--extended-hours", action="store_true",
                   help="include pre/post market; off by default so the bars match "
                        "the regular-session 5-minute rows already stored")
    return p.parse_args(argv)


def api_key() -> str:
    for line in pathlib.Path(".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            name, value = line.split("=", 1)
            if name.strip() == "ALPHAVANTAGE_API_KEY":
                return value.strip().strip('"').strip("'")
    raise SystemExit("error: ALPHAVANTAGE_API_KEY missing from .env")


class _Pacer:
    """Even spacing between request starts, shared across worker threads."""

    def __init__(self, per_minute: float):
        self.interval = 60.0 / max(per_minute, 1e-9)
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


def months(start: str, end: str):
    y0, m0 = (int(x) for x in start.split("-"))
    y1, m1 = (int(x) for x in end.split("-"))
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m > 12:
            y, m = y + 1, 1


def fetch(symbol: str, month: str, interval: str, key: str,
          extended: bool, attempts: int = 4):
    params = {"function": "TIME_SERIES_INTRADAY", "symbol": symbol,
              "interval": interval, "month": month, "outputsize": "full",
              "extended_hours": "true" if extended else "false", "apikey": key}
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
                    return None, f"{gate}: {str(payload[gate])[:100]}"
                break
        else:
            if "Error Message" in payload:
                return None, f"error: {str(payload['Error Message'])[:100]}"
            series = payload.get(f"Time Series ({interval})")
            # An empty month is a normal answer for a symbol that did not trade
            # yet, not a failure -- it is recorded so the month is not retried.
            return (series or {}), "ok"
    return None, "exhausted retries"


def main(argv=None):
    args = parse_args(argv)
    key = api_key()
    end = args.end or date.today().strftime("%Y-%m")
    args.db.parent.mkdir(parents=True, exist_ok=True)
    store = sqlite3.connect(args.db)
    store.execute("PRAGMA journal_mode=WAL")
    store.execute("PRAGMA synchronous=NORMAL")
    store.executescript(SCHEMA)
    store.commit()

    have = {}
    for sym in args.symbols:
        rows = store.execute(
            "SELECT DISTINCT substr(ts,1,7) FROM bars WHERE ticker=? AND frequency=?",
            (sym, args.interval)).fetchall()
        have[sym] = {r[0] for r in rows}

    todo = [(s, m) for s in args.symbols for m in months(args.start, end)
            if m not in have[s]]
    print(f"{len(todo)} symbol-months to fetch "
          f"({', '.join(args.symbols)} at {args.interval}, {args.start} to {end})",
          flush=True)
    if not todo:
        print("nothing to do")
        return 0

    def work(job):
        sym, month = job
        series, status = fetch(sym, month, args.interval, key, args.extended_hours)
        return sym, month, series, status

    stored = empty = failed = bars = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (sym, month, series, status) in enumerate(pool.map(work, todo), 1):
            if series is None:
                failed += 1
                print(f"  {sym} {month} FAIL {status}", flush=True)
                continue
            if not series:
                empty += 1
            rows = []
            for ts, ohlcv in series.items():
                try:
                    rows.append((sym, args.interval, ts.replace(" ", "T") + "Z",
                                 float(ohlcv["1. open"]), float(ohlcv["2. high"]),
                                 float(ohlcv["3. low"]), float(ohlcv["4. close"]),
                                 float(ohlcv["5. volume"])))
                except (KeyError, TypeError, ValueError):
                    continue
            if rows:
                store.executemany(
                    "INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?,?)", rows)
                store.commit()
                bars += len(rows)
                stored += 1
            if i % 25 == 0:
                rate = i / max(time.time() - started, 1e-9) * 60
                print(f"  {i}/{len(todo)}  {bars:,} bars  {rate:.0f}/min  "
                      f"empty {empty}  fails {failed}", flush=True)

    print(f"\ndone: {stored} months stored ({bars:,} bars), "
          f"{empty} empty, {failed} failed")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
