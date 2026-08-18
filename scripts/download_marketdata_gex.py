"""Download resumable SPY/QQQ historical option chains and calculate daily GEX.

Defaults deliberately request only 0-60 DTE and the 60 strikes nearest spot.
The job stops with ten daily credits in reserve and records every completed day
in SQLite, so rerunning the same command resumes without duplicate requests.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.config import get_marketdata_token  # noqa: E402
from savi_uz.marketdata_gex import (  # noqa: E402
    GexStore, MarketDataClient, MarketDataError, MarketDataRateLimitError,
)


def parse_args(argv=None):
    today = date.today()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/options/marketdata.db"))
    parser.add_argument("--csv", type=Path, default=Path("out/options/gex_daily.csv"))
    parser.add_argument("--macro-db", type=Path, default=Path("data/macro/macro.db"))
    parser.add_argument("--symbols", nargs="+", default=("SPY", "QQQ"))
    parser.add_argument(
        "--token-env", default="MARKETDATA_TOKEN",
        help="environment/.env variable containing the bearer token (never logged)",
    )
    parser.add_argument("--start", type=date.fromisoformat,
                        default=today - timedelta(days=365))
    parser.add_argument("--end", type=date.fromisoformat, default=today - timedelta(days=1))
    parser.add_argument("--max-dte", type=int, default=60)
    parser.add_argument("--strike-limit", type=int, default=60)
    parser.add_argument("--max-credits", type=int, default=90)
    parser.add_argument("--credit-reserve", type=int, default=10)
    parser.add_argument("--max-requests", type=int, default=200)
    parser.add_argument("--oldest-first", action="store_true")
    parser.add_argument("--pause", type=float, default=0.25)
    parser.add_argument("--plan", action="store_true")
    return parser.parse_args(argv)


def export_daily_gex(connection: sqlite3.Connection, path: Path) -> int:
    """Export quality-controlled EOD features with an explicit lag contract."""
    columns = [row[1] for row in connection.execute("PRAGMA table_info(daily_gex)")]
    rows = connection.execute(
        "SELECT * FROM daily_gex WHERE usable_contracts >= contracts * 0.5 "
        "ORDER BY observation_date, symbol"
    ).fetchall()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns + ["signal_lag_sessions"])
        writer.writerows(tuple(row) + (1,) for row in rows)
    return len(rows)


def weekdays(start: date, end: date):
    if start > end:
        raise ValueError("start date is after end date")
    days = []
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def risk_free_rate(path: Path, day: date) -> float:
    if not path.is_file():
        return 0.04
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT value FROM observations WHERE series_id='DFF' AND obs_date<=? "
            "AND value IS NOT NULL ORDER BY obs_date DESC LIMIT 1", (day.isoformat(),)
        ).fetchone()
    finally:
        connection.close()
    return float(row[0]) / 100.0 if row else 0.04


def main(argv=None):
    args = parse_args(argv)
    symbols = tuple(dict.fromkeys(symbol.upper() for symbol in args.symbols))
    days = weekdays(args.start, args.end)
    if not args.oldest_first:
        days.reverse()
    with GexStore(args.db) as store:
        done = store.completed()
        jobs = [(symbol, day) for day in days for symbol in symbols
                if (symbol, day.isoformat()) not in done]
        print(f"symbols: {', '.join(symbols)}")
        print(f"history: {args.start} through {args.end}")
        print(f"filter : 0-{args.max_dte} DTE, {args.strike_limit} closest strikes")
        print(f"pending: {len(jobs)} symbol-days; completed: {len(done)}")
        if args.plan:
            return 0
        try:
            client = MarketDataClient(get_marketdata_token(args.token_env))
        except ValueError as exc:
            print(f"error: {exc}")
            return 2

        used = requests = 0
        stopped = ""
        for symbol, day in jobs:
            if requests >= args.max_requests or used >= args.max_credits:
                stopped = "local request/credit budget reached"
                break
            try:
                response = client.fetch_chain(
                    symbol, day, max_dte_days=args.max_dte,
                    strike_limit=args.strike_limit,
                )
            except MarketDataRateLimitError as exc:
                store.log(symbol, day.isoformat(), "rate_limited", 0,
                          message=str(exc))
                stopped = str(exc)
                break
            except MarketDataError as exc:
                store.log(symbol, day.isoformat(), "error", 0, message=str(exc))
                print(f"{symbol} {day}: ERROR {exc}", flush=True)
                requests += 1
                time.sleep(args.pause)
                continue
            requests += 1
            used += response.credits.consumed or 0
            if response.status not in {"ok", "no_data"}:
                store.log(symbol, day.isoformat(), "error", 0,
                          response.credits, response.message)
                print(f"{symbol} {day}: API error {response.message}", flush=True)
                continue
            dividend_yield = {"SPY": 0.012, "QQQ": 0.006}.get(symbol, 0.0)
            feature = store.write(
                symbol, day, response, risk_free=risk_free_rate(args.macro_db, day),
                dividend_yield=dividend_yield,
            )
            usable = feature[3] if feature is not None else 0
            print(
                f"{symbol} {day}: {response.status}, {len(response.contracts):,} contracts, "
                f"{usable:,} gamma, credits {response.credits.consumed}, "
                f"remaining {response.credits.remaining}",
                flush=True,
            )
            remaining = response.credits.remaining
            if remaining is not None and remaining <= args.credit_reserve:
                stopped = f"daily credits at reserve ({remaining} remaining)"
                break
            time.sleep(args.pause)

        counts = store.connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT symbol||'/'||observation_date) "
            "FROM option_contracts"
        ).fetchone()
        features = store.connection.execute("SELECT COUNT(*) FROM daily_gex").fetchone()[0]
        exported = export_daily_gex(store.connection, args.csv)
        print(f"stored: {counts[0]:,} contracts across {counts[1]:,} snapshots; "
              f"{features:,} GEX rows")
        print(f"exported: {exported:,} quality-controlled rows to {args.csv}")
        if stopped:
            print(f"stopped: {stopped}; rerun the same command to resume")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
