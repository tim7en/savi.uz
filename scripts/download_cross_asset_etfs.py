"""Daily OHLC for non-equity ETFs, from Alpha Vantage.

The equity book already holds six non-equity instruments, and their daily R
stream correlates +0.03 with the equity sleeve -- +0.004 on the equity sleeve's
down days.  Six such names lifted Sharpe as much as thirty-two more equities did,
so the cheapest remaining improvement is more of them rather than more stocks.

Every symbol here is tagged with what, if anything, is tradeable on Binance.  The
honest picture is that Binance carries gold and silver and the major FX crosses
and nothing else in this list -- no oil, no bonds, no broad commodity, no
volatility -- so a Binance-only book cannot hold most of what makes this sleeve
work.  The tags are recorded rather than used as a filter: knowing that duration
is the strongest diversifier and also untradeable there is more useful than
quietly dropping it.

Written to its own database rather than into ``bars.db``, because the options
download is holding that file for hours at a time and a writer competing with it
buys nothing.  The schema matches, so ``--bars`` on any existing script points
here unchanged.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    ticker     TEXT NOT NULL,
    frequency  TEXT NOT NULL,
    ts         TEXT NOT NULL,
    open       REAL,
    high       REAL,
    low        REAL,
    close      REAL,
    volume     REAL,
    PRIMARY KEY (ticker, frequency, ts)
);
CREATE INDEX IF NOT EXISTS idx_bars_ticker_ts ON bars (ticker, ts);
CREATE TABLE IF NOT EXISTS venue_map (
    ticker    TEXT PRIMARY KEY,
    sleeve    TEXT,
    binance   TEXT
);
"""

#: ticker -> (sleeve, what trades on Binance for the same exposure, or None)
UNIVERSE = {
    # metals -- the one place Binance has real coverage
    "GLD":  ("metals", "PAXGUSDT / XAUUSDT"),
    "SLV":  ("metals", "XAGUSDT"),
    "GDX":  ("metals", None),
    "PPLT": ("metals", None),
    # energy -- nothing on Binance
    "USO":  ("energy", None),
    "UNG":  ("energy", None),
    "BNO":  ("energy", None),
    # broad commodity and agriculture -- nothing on Binance
    "DBC":  ("commodity", None),
    "DBA":  ("commodity", None),
    "GSG":  ("commodity", None),
    # duration and credit -- nothing on Binance, and the strongest diversifier
    "TLT":  ("duration", None),
    "IEF":  ("duration", None),
    "SHY":  ("duration", None),
    "TIP":  ("duration", None),
    "LQD":  ("credit", None),
    "HYG":  ("credit", None),
    # dollar and FX -- Binance lists the major crosses
    "UUP":  ("fx", "USDT complex"),
    "FXE":  ("fx", "EURUSDT"),
    "FXY":  ("fx", "JPY pairs, thin"),
    "FXB":  ("fx", "GBPUSDT"),
    # volatility -- nothing on Binance
    "VXX":  ("volatility", None),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=pathlib.Path,
                        default=pathlib.Path("data/cross_assets/etf_daily.db"))
    parser.add_argument("--symbols", nargs="+", default=sorted(UNIVERSE))
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="pause between calls; the options download shares this key")
    parser.add_argument("--force", action="store_true",
                        help="refetch symbols already stored")
    return parser.parse_args(argv)


def api_key() -> str:
    for line in pathlib.Path(".env").read_text(encoding="utf-8").splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "ALPHAVANTAGE_API_KEY":
            return value.strip().strip('"').strip("'")
    raise SystemExit("error: ALPHAVANTAGE_API_KEY missing from .env")


def fetch(symbol: str, key: str, attempts: int = 4):
    """Daily OHLC, split-adjusted, full history.

    The adjusted close is deliberately ignored: the engine needs a consistent
    OHLC bar, and mixing a dividend-adjusted close with unadjusted highs and lows
    produces true ranges that never happened.  Splits are handled by taking the
    adjusted series' ratio, which keeps the bar internally consistent.
    """
    query = urllib.parse.urlencode({
        "function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": symbol,
        "outputsize": "full", "apikey": key})
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                f"https://www.alphavantage.co/query?{query}",
                headers={"User-Agent": "research/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode())
        except Exception as error:
            if attempt == attempts - 1:
                return None, f"{type(error).__name__}: {error}"
            time.sleep(2 ** attempt)
            continue
        series = payload.get("Time Series (Daily)")
        if series:
            return series, "ok"
        note = (payload.get("Note") or payload.get("Information")
                or payload.get("Error Message") or str(payload)[:160])
        if "limit" in note.lower() or "frequency" in note.lower():
            time.sleep(15)
            continue
        return None, note
    return None, "exhausted"


def rows_for(symbol: str, series: dict):
    """OHLC bars with splits removed, dividends left alone."""
    out = []
    for day, values in series.items():
        try:
            close = float(values["4. close"])
            adjusted = float(values["5. adjusted close"])
            factor = adjusted / close if close else 1.0
            # Dividend adjustment would shift the close relative to the high and
            # low; only the split component is wanted, and it is the part that
            # moves in discrete jumps far from 1.
            if not 0.2 < factor < 5.0:
                scale = factor
            else:
                scale = 1.0
            out.append((symbol, "daily", f"{day}T00:00:00.000Z",
                        float(values["1. open"]) * scale,
                        float(values["2. high"]) * scale,
                        float(values["3. low"]) * scale,
                        close * scale,
                        float(values["6. volume"])))
        except (KeyError, ValueError):
            continue
    return sorted(out, key=lambda r: r[2])


def main(argv=None):
    args = parse_args(argv)
    key = api_key()
    args.db.parent.mkdir(parents=True, exist_ok=True)
    store = sqlite3.connect(args.db)
    store.executescript(SCHEMA)
    for ticker, (sleeve, binance) in UNIVERSE.items():
        store.execute("INSERT OR REPLACE INTO venue_map VALUES (?,?,?)",
                      (ticker, sleeve, binance))
    store.commit()

    have = {t for (t,) in store.execute(
        "SELECT DISTINCT ticker FROM bars WHERE frequency='daily'")}
    todo = [s for s in args.symbols if args.force or s not in have]
    print(f"{len(todo)} of {len(args.symbols)} symbols to fetch"
          f"{'' if not have else f'; {len(have)} already stored'}\n", flush=True)

    ok = fail = 0
    for symbol in todo:
        series, status = fetch(symbol, key)
        if series is None:
            print(f"  {symbol:5s} FAILED  {status[:90]}", flush=True)
            fail += 1
            time.sleep(args.sleep)
            continue
        rows = rows_for(symbol, series)
        store.executemany("INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?,?)", rows)
        store.commit()
        sleeve, binance = UNIVERSE.get(symbol, ("?", None))
        print(f"  {symbol:5s} {len(rows):>6,d} bars  {rows[0][2][:10]} -> "
              f"{rows[-1][2][:10]}  {sleeve:10s} "
              f"binance: {binance or '--'}", flush=True)
        ok += 1
        time.sleep(args.sleep)

    total = store.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
    names = store.execute(
        "SELECT COUNT(DISTINCT ticker) FROM bars WHERE frequency='daily'").fetchone()[0]
    store.close()
    print(f"\n  {ok} stored, {fail} failed; {names} symbols, {total:,} daily bars")
    print(f"  wrote {args.db}")
    return 1 if fail and not ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
