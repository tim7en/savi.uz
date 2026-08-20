"""Five-minute history for the non-equity ETFs, from inception.

The cross-asset case rests on two measured facts that point in opposite
directions: six uncorrelated instruments bought as much Sharpe as thirty-two
correlated ones, and the same rules score 0.13 on daily bars where they score
2.64 at thirty minutes.  Testing that case needs intraday history for the
diversifying names, which is what this fetches.

It cannot start in 2000.  None of the twenty-one ETFs existed then -- the bond
funds list in July 2002 and most of the rest in 2006 and 2007 -- so each symbol
is pulled from its own inception month instead, which is as far back as the
instrument goes rather than as far back as the vendor does.

Five minutes rather than thirty, because Alpha Vantage bills by the month slice
regardless of interval, so the finer bar is free and every existing script
resamples from five-minute input anyway.

Splits looked like the trap here and are not.  These funds have reverse split
repeatedly -- USO one for eight in 2020, UNG four times, VXX eight -- so an
unadjusted series would read those as overnight collapses the breakout logic
would happily trade.  Alpha Vantage's intraday endpoint, however, already
returns split-adjusted prices throughout its history: checked against the daily
store, VXX in November 2010 agrees to a ratio of 1.000 despite a cumulative
factor near 65,000, and USO's 2020 split shows no discontinuity.

Applying an adjustment here would therefore be the actual bug, and was: a first
version divided already-adjusted 2006 prices by eight.  Split events are still
fetched and stored, because a silent change in vendor behaviour would otherwise
be invisible, and because the check that caught this is worth being able to
repeat.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
import sys
import time
import urllib.parse
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    ticker     TEXT NOT NULL,
    frequency  TEXT NOT NULL,
    ts         TEXT NOT NULL,
    open       REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (ticker, frequency, ts)
);
CREATE INDEX IF NOT EXISTS idx_bars_ticker_ts ON bars (ticker, ts);
CREATE TABLE IF NOT EXISTS splits (
    ticker TEXT NOT NULL, split_date TEXT NOT NULL, factor REAL NOT NULL,
    PRIMARY KEY (ticker, split_date)
);
CREATE TABLE IF NOT EXISTS slice_log (
    ticker TEXT NOT NULL, month TEXT NOT NULL, status TEXT, rows INTEGER,
    note TEXT, logged_at TEXT, PRIMARY KEY (ticker, month)
);
"""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=pathlib.Path,
                        default=pathlib.Path("data/cross_assets/etf_intraday.db"))
    parser.add_argument("--daily", type=pathlib.Path,
                        default=pathlib.Path("data/cross_assets/etf_daily.db"))
    parser.add_argument("--interval", default="5min")
    parser.add_argument("--end", default="2026-08")
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--pace", type=float, default=0.82,
                        help="minimum seconds between request starts, across all "
                             "workers. The plan allows 75 calls a minute, which is "
                             "0.80s; the default sits a shade under that so network "
                             "jitter cannot push a burst over the line")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args(argv)


def api_key() -> str:
    for line in pathlib.Path(".env").read_text(encoding="utf-8").splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "ALPHAVANTAGE_API_KEY":
            return value.strip().strip('"').strip("'")
    raise SystemExit("error: ALPHAVANTAGE_API_KEY missing from .env")


#: Requests are paced to a fixed spacing rather than fired as fast as the pool
#: allows.  The plan permits 75 calls a minute, so the spacing is 0.80s and the
#: default leaves a little under that; the pool size then only decides how much
#: latency is hidden, never the request rate.
#:
#: Worth knowing before tuning this: the "Burst pattern detected" rejections that
#: prompted the pacer were not the vendor's doing.  Detached downloads survived
#: the shutdown of the shell that started them, so five copies were running at
#: once against one key.  Check for orphans before assuming a quota problem --
#: the giveaway is that restarting makes it worse rather than better.
_PACE = threading.Lock()
_LAST = [0.0]
_MIN_INTERVAL = [0.7]


def pace():
    with _PACE:
        wait = _MIN_INTERVAL[0] - (time.monotonic() - _LAST[0])
        if wait > 0:
            time.sleep(wait)
        _LAST[0] = time.monotonic()


def call(params, key, attempts=5):
    query = urllib.parse.urlencode({**params, "apikey": key})
    for attempt in range(attempts):
        pace()
        try:
            request = urllib.request.Request(
                f"https://www.alphavantage.co/query?{query}",
                headers={"User-Agent": "research/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode())
        except Exception as error:
            if attempt == attempts - 1:
                return None, f"{type(error).__name__}: {error}"
            time.sleep(2 ** attempt)
            continue
        series = next((payload[k] for k in payload if "Time Series" in k), None)
        if series is not None:
            return series, "ok"
        note = (payload.get("Note") or payload.get("Information")
                or payload.get("Error Message") or str(payload)[:150])
        # "Burst pattern detected" matches neither "limit" nor "frequency", so an
        # earlier version treated throttling as a hard failure and burned the
        # slice.  Any of these means wait and retry, not give up.
        lowered = note.lower()
        if any(word in lowered for word in ("limit", "frequency", "burst", "thank you")):
            time.sleep(20 + 10 * attempt)
            continue
        return None, note
    return None, "exhausted"


def fetch_splits(symbol, key):
    """Split events from the daily endpoint, oldest first."""
    series, status = call({"function": "TIME_SERIES_DAILY_ADJUSTED",
                           "symbol": symbol, "outputsize": "full"}, key)
    if series is None:
        return [], status
    events = []
    for day, values in series.items():
        try:
            coefficient = float(values.get("8. split coefficient", 1.0))
        except ValueError:
            continue
        if abs(coefficient - 1.0) > 1e-9:
            events.append((day, coefficient))
    return sorted(events), "ok"


def months_between(start: str, end: str):
    year, month = int(start[:4]), int(start[5:7])
    last_year, last_month = int(end[:4]), int(end[5:7])
    out = []
    while (year, month) <= (last_year, last_month):
        out.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return out


def splits_check(store, daily_path, symbol):
    """Largest disagreement between an intraday close and the daily store.

    Both series should already be split-adjusted, so this should sit at 1.0.  A
    ratio near a split factor means the vendor changed behaviour and the bars
    need adjusting after all.
    """
    daily = sqlite3.connect(f"file:{daily_path}?mode=ro", uri=True)
    reference = dict(daily.execute(
        "SELECT substr(ts,1,10), close FROM bars WHERE ticker=? AND frequency='daily'",
        (symbol,)))
    daily.close()
    worst = 1.0
    for day, close in store.execute(
        "SELECT substr(ts,1,10), close FROM bars WHERE ticker=? "
        "GROUP BY substr(ts,1,10) HAVING ts=MAX(ts)", (symbol,)
    ):
        other = reference.get(day)
        if other:
            ratio = close / other
            if abs(ratio - 1.0) > abs(worst - 1.0):
                worst = ratio
    return worst


def main(argv=None):
    args = parse_args(argv)
    key = api_key()
    _MIN_INTERVAL[0] = args.pace
    args.db.parent.mkdir(parents=True, exist_ok=True)
    store = sqlite3.connect(args.db)
    store.executescript(SCHEMA)
    store.commit()

    daily = sqlite3.connect(f"file:{args.daily}?mode=ro", uri=True)
    inception = dict(daily.execute(
        "SELECT ticker, MIN(ts) FROM bars WHERE frequency='daily' GROUP BY ticker"))
    daily.close()
    symbols = args.symbols or sorted(inception)

    known = {t for (t,) in store.execute("SELECT DISTINCT ticker FROM splits")}
    for symbol in symbols:
        if symbol in known:
            continue
        events, status = fetch_splits(symbol, key)
        for split_date, value in events:
            store.execute("INSERT OR REPLACE INTO splits VALUES (?,?,?)",
                          (symbol, split_date, value))
        store.execute("INSERT OR REPLACE INTO splits VALUES (?,?,?)",
                      (symbol, "0000-00-00", 1.0))  # marks the symbol as checked
        store.commit()
        if events:
            print(f"  {symbol:5s} {len(events)} split(s): "
                  + ", ".join(f"{d} x{v:g}" for d, v in events), flush=True)
    print(flush=True)

    done = {(t, m) for t, m in store.execute(
        "SELECT ticker, month FROM slice_log WHERE status='ok'")}
    todo = []
    for symbol in symbols:
        start = inception[symbol][:7]
        for month in months_between(start, args.end):
            if (symbol, month) not in done:
                todo.append((symbol, month))
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo):,} (symbol, month) slices to fetch across {len(symbols)} "
          f"symbols at {args.interval}\n", flush=True)
    if not todo:
        return 0

    def work(item):
        symbol, month = item
        series, status = call({"function": "TIME_SERIES_INTRADAY", "symbol": symbol,
                               "interval": args.interval, "month": month,
                               "outputsize": "full", "extended_hours": "false"}, key)
        return symbol, month, series, status

    started, stored, failed, bars = time.time(), 0, 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for symbol, month, series, status in pool.map(work, todo):
            stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            if series is None:
                store.execute("INSERT OR REPLACE INTO slice_log VALUES (?,?,?,?,?,?)",
                              (symbol, month, "fail", 0, status[:200], stamp))
                store.commit()
                failed += 1
                continue
            rows = []
            for timestamp, values in series.items():
                # No scaling: the vendor has already applied it. See the module
                # docstring for the check, and splits_check() to repeat it.
                try:
                    rows.append((symbol, args.interval,
                                 timestamp.replace(" ", "T") + ".000Z",
                                 float(values["1. open"]),
                                 float(values["2. high"]),
                                 float(values["3. low"]),
                                 float(values["4. close"]),
                                 float(values["5. volume"])))
                except (KeyError, ValueError):
                    continue
            store.executemany(
                "INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?,?)", rows)
            store.execute("INSERT OR REPLACE INTO slice_log VALUES (?,?,?,?,?,?)",
                          (symbol, month, "ok", len(rows), "", stamp))
            store.commit()
            stored += 1
            bars += len(rows)
            if stored % 100 == 0:
                rate = stored / max(time.time() - started, 1e-9)
                left = (len(todo) - stored - failed) / max(rate, 1e-9)
                print(f"  {stored + failed:,}/{len(todo):,}  {bars:,} bars  "
                      f"{rate * 60:.0f}/min  ~{left / 60:.0f} min left  "
                      f"fails {failed}", flush=True)

    total = store.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
    names = store.execute("SELECT COUNT(DISTINCT ticker) FROM bars").fetchone()[0]
    span = store.execute("SELECT MIN(ts), MAX(ts) FROM bars").fetchone()
    store.close()
    print(f"\ndone: {stored:,} slices stored, {failed} failed")
    print(f"  {names} symbols, {total:,} bars, {span[0][:10]} -> {span[1][:10]}")
    print(f"  wrote {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
