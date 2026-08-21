"""Populate ticker metadata for the Binance trad-FI universe.

The bar downloader stores prices and nothing about what it is storing. That gap
matters more than usual for this universe, because Binance base assets are not
tickers and several resolve to instruments that are not the company they appear
to name -- ``MVLL`` beside ``MRVL``, ``PAYP`` beside ``PYPL``. A ticker's real
name settles whether the book holds two views of one underlying, which a
six-position cap filled by random tie-break would otherwise size twice.

Metadata comes from Tiingo rather than Alpha Vantage on purpose: a separate
quota, so this can run while the bar download has the Alpha Vantage budget
saturated. Tiingo responses are cached on disk, so re-runs cost nothing.

Also stored is the Binance side -- the perpetual symbol, its declared underlying
type and its onboard date -- so the universe's provenance travels with it.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.config import get_tiingo_api_key  # noqa: E402
from savi_uz.tiingo_sources import TiingoClient, TiingoError  # noqa: E402
from savi_uz.tradfi_universe import (  # noqa: E402
    CURATED_YAHOO_TICKERS,
    BinanceTradFiClient,
)

BINANCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS binance_map (
    ticker          TEXT PRIMARY KEY,
    binance_symbol  TEXT,
    base_asset      TEXT,
    underlying_type TEXT,
    sub_types       TEXT,
    status          TEXT,
    onboard_date    TEXT
);
"""

#: Name fragments that mark a levered or inverse wrapper rather than the equity.
DERIVATIVE_MARKERS = (
    "2X", "3X", "1.5X", "BULL", "BEAR", "LEVERAGED", "INVERSE", "ULTRA",
    "ULTRASHORT", "DAILY", "-1X", "SHORT ",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path,
                        default=Path("data/intraday/bars_av.db"))
    parser.add_argument("--universe-cache", type=Path,
                        default=Path("data/intraday/tradfi_universe.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/tiingo"))
    parser.add_argument("--requests-per-hour", type=int, default=400)
    parser.add_argument("--refresh", action="store_true",
                        help="ignore the on-disk Tiingo cache")
    return parser.parse_args(argv)


def resolved_tickers(connection: sqlite3.Connection) -> list[str]:
    tables = {r[0] for r in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    if "symbol_validation" not in tables:
        raise SystemExit("error: run the downloader with --validate first")
    return [r[0] for r in connection.execute(
        "SELECT ticker FROM symbol_validation WHERE resolved=1 ORDER BY ticker")]


def store_binance_map(connection: sqlite3.Connection) -> dict[str, str]:
    """Binance provenance per ticker; returns ticker -> binance symbol."""
    connection.executescript(BINANCE_SCHEMA)
    try:
        instruments = BinanceTradFiClient().fetch_tradfi_instruments()
    except (OSError, TimeoutError, ValueError) as exc:
        print(f"  warning: Binance unavailable ({exc}); provenance not refreshed")
        return {}
    mapping = {}
    for instrument in instruments:
        if instrument.region != "US" or instrument.is_pre_ipo:
            continue
        ticker = CURATED_YAHOO_TICKERS.get(instrument.base_asset,
                                           instrument.base_asset)
        if "." in ticker:
            continue
        mapping[ticker] = instrument.binance_symbol
        connection.execute(
            "INSERT OR REPLACE INTO binance_map VALUES (?,?,?,?,?,?,?)",
            (ticker, instrument.binance_symbol, instrument.base_asset,
             instrument.underlying_type, ",".join(instrument.sub_types),
             instrument.status,
             instrument.onboard_date.isoformat() if instrument.onboard_date else None))
    return mapping


def looks_derivative(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in DERIVATIVE_MARKERS)


def main(argv=None) -> int:
    args = parse_args(argv)
    # A bar download may hold the write lock. Wait for it rather than failing,
    # and keep every transaction here to a single row so this never becomes the
    # writer something else is waiting on.
    connection = sqlite3.connect(args.db, isolation_level=None, timeout=60.0)
    connection.execute("PRAGMA busy_timeout=60000")
    try:
        tickers = resolved_tickers(connection)
        print(f"{len(tickers)} resolved tickers\n")
        binance = store_binance_map(connection)

        client = TiingoClient(get_tiingo_api_key(), cache_dir=args.cache_dir,
                              requests_per_hour=args.requests_per_hour,
                              refresh=args.refresh)
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        stored = failed = 0
        flagged: list[tuple[str, str]] = []

        have = {r[0] for r in connection.execute(
            "SELECT ticker FROM symbols WHERE name IS NOT NULL AND name != ''")}
        if have:
            print(f"  {len(have)} already stored, fetching the rest\n")

        for ticker in tickers:
            if ticker in have and not args.refresh:
                continue
            meta = None
            for attempt in range(3):
                try:
                    meta = client.fetch_metadata(ticker)
                    break
                except (TiingoError, OSError, TimeoutError) as exc:
                    if attempt == 2:
                        failed += 1
                        print(f"  {ticker:6s} metadata failed: {str(exc)[:80]}",
                              flush=True)
                    else:
                        time.sleep(2 ** attempt)
            if meta is None:
                continue
            connection.execute(
                "INSERT OR REPLACE INTO symbols VALUES (?,?,?,?,?,?,?,?,?)",
                (ticker, meta.name, meta.exchange,
                 meta.start_date.isoformat() if meta.start_date else None,
                 meta.end_date.isoformat() if meta.end_date else None,
                 int(meta.has_intraday),
                 f"binance:{binance.get(ticker, '')}" if binance else None,
                 meta.description, now))
            stored += 1
            if looks_derivative(meta.name):
                flagged.append((ticker, meta.name))

        print(f"\n  stored {stored}, failed {failed}; "
              f"{client.requests_made} Tiingo requests, "
              f"{client.cache_hits} cache hits")

        if flagged:
            print(f"\n  {len(flagged)} tickers look like levered or inverse "
                  f"wrappers, not the underlying equity:")
            for ticker, name in sorted(flagged):
                print(f"    {ticker:6s} {name[:66]}")
            print("\n  Treat each as correlated with its underlying for capacity "
                  "purposes. Holding both is one position taken twice.")
        return 0
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
