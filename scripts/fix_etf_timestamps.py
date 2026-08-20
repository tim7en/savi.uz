"""Restate the ETF intraday timestamps from Eastern local time to UTC.

Alpha Vantage returns intraday stamps in US Eastern wall-clock time.  The
downloader appended a ``Z`` to them, which asserts UTC and is wrong by four or
five hours depending on daylight saving.  Nothing downstream would have raised:
``resample_regular_session`` converts UTC to New York and keeps only 09:30 to
16:00, so a 09:30 Eastern bar labelled 09:30Z becomes 04:30 local and is silently
dropped.  The failure mode is an empty book rather than a wrong one, which is
kinder than the alternative but still needs fixing at the source.

The equity store in ``bars.db`` is genuinely UTC -- its sessions run 14:30Z to
20:55Z -- so this brings the two into agreement and lets every existing script
read either with the same loader.

Conversion goes through ``America/New_York`` rather than a fixed offset, because
the offset changes twice a year and half the sample sits on each side.
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, nargs="+",
                        default=[Path("data/cross_assets/etf_30min.db")])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def to_utc(stamp: str) -> str:
    """'2024-03-05T09:30:00.000Z' read as Eastern -> the true UTC stamp."""
    naive = datetime.strptime(stamp[:19], "%Y-%m-%dT%H:%M:%S")
    return (naive.replace(tzinfo=EASTERN).astimezone(UTC)
            .strftime("%Y-%m-%dT%H:%M:%S.000Z"))


def already_utc(connection) -> bool:
    """A session whose first bar is before 12:00 is still on local time."""
    row = connection.execute(
        "SELECT MIN(substr(ts,12,5)) FROM bars WHERE ts LIKE '2024-03-05%'"
    ).fetchone()
    return bool(row and row[0] and row[0] >= "12:00")


def main(argv=None):
    args = parse_args(argv)
    for path in args.db:
        if not path.exists():
            print(f"  {path}: missing, skipped")
            continue
        connection = sqlite3.connect(path)
        total = connection.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
        if already_utc(connection):
            print(f"  {path}: already UTC, nothing to do ({total:,} bars)")
            connection.close()
            continue
        sample = connection.execute(
            "SELECT ts FROM bars ORDER BY ts LIMIT 1").fetchone()[0]
        print(f"  {path}: {total:,} bars, first {sample} -> {to_utc(sample)}")
        if args.dry_run:
            connection.close()
            continue

        rows = connection.execute(
            "SELECT ticker, frequency, ts, open, high, low, close, volume "
            "FROM bars").fetchall()
        converted = [(t, f, to_utc(ts), o, h, l, c, v)
                     for t, f, ts, o, h, l, c, v in rows]
        connection.execute("DROP TABLE IF EXISTS bars_utc")
        connection.execute(
            "CREATE TABLE bars_utc (ticker TEXT NOT NULL, frequency TEXT NOT NULL, "
            "ts TEXT NOT NULL, open REAL, high REAL, low REAL, close REAL, "
            "volume REAL, PRIMARY KEY (ticker, frequency, ts))")
        connection.executemany(
            "INSERT OR REPLACE INTO bars_utc VALUES (?,?,?,?,?,?,?,?)", converted)
        moved = connection.execute("SELECT COUNT(*) FROM bars_utc").fetchone()[0]
        if moved != total:
            connection.rollback()
            connection.close()
            raise SystemExit(f"error: {path} would lose {total - moved:,} bars")
        connection.execute("DROP TABLE bars")
        connection.execute("ALTER TABLE bars_utc RENAME TO bars")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_bars_ticker_ts ON bars (ticker, ts)")
        connection.commit()
        check = connection.execute(
            "SELECT MIN(substr(ts,12,5)), MAX(substr(ts,12,5)) FROM bars "
            "WHERE ts LIKE '2024-03-05%'").fetchone()
        print(f"     converted {moved:,} bars; that session now runs "
              f"{check[0]}Z to {check[1]}Z")
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
