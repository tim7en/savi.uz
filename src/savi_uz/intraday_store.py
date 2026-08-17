"""SQLite store for intraday and daily bars, designed to be resumed.

The quota makes a full download a multi-session job, so the store records not
just the bars but which (ticker, frequency, year) windows are already complete.
A resumed run reads that table and asks only for what is missing, which is what
keeps a restart from spending requests re-fetching work already done.
"""

from __future__ import annotations

import csv
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

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

CREATE TABLE IF NOT EXISTS symbols (
    ticker        TEXT PRIMARY KEY,
    name          TEXT,
    exchange      TEXT,
    history_start TEXT,
    history_end   TEXT,
    has_intraday  INTEGER,
    themes        TEXT,
    description   TEXT,
    fetched_at    TEXT
);

-- One row per (ticker, frequency, year) that has been fetched to completion.
-- This is what makes a resumed run cheap.
CREATE TABLE IF NOT EXISTS windows (
    ticker      TEXT NOT NULL,
    frequency   TEXT NOT NULL,
    year        INTEGER NOT NULL,
    first_ts    TEXT,
    last_ts     TEXT,
    rows        INTEGER,
    truncated   INTEGER DEFAULT 0,
    fetched_at  TEXT,
    PRIMARY KEY (ticker, frequency, year)
);

CREATE TABLE IF NOT EXISTS fetch_log (
    run_id     TEXT NOT NULL,
    logged_at  TEXT NOT NULL,
    source     TEXT NOT NULL,
    target     TEXT NOT NULL,
    rows       INTEGER,
    status     TEXT NOT NULL,
    message    TEXT
);

CREATE INDEX IF NOT EXISTS idx_bars_ticker_ts ON bars (ticker, ts);
CREATE INDEX IF NOT EXISTS idx_fetch_log_run ON fetch_log (run_id);
"""

EXPORT_TABLES = ("symbols", "windows", "bars", "fetch_log")


def _iso(value: date | str | None) -> str | None:
    if value is None:
        return None
    return value.isoformat() if isinstance(value, date) else str(value)


class IntradayStore:
    """Idempotent bar store with per-window resume state."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript(SCHEMA)
        self.connection.commit()

    def __enter__(self) -> IntradayStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def _write(self, table: str, columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> int:
        payload = list(rows)
        if not payload:
            return 0
        placeholders = ", ".join("?" * len(columns))
        self.connection.executemany(
            f"INSERT OR REPLACE INTO {table} ({', '.join(columns)}) VALUES ({placeholders})", payload
        )
        self.connection.commit()
        return len(payload)

    def upsert_symbol(self, meta: Any, themes: str, fetched_at: str) -> None:
        self._write(
            "symbols",
            ("ticker", "name", "exchange", "history_start", "history_end",
             "has_intraday", "themes", "description", "fetched_at"),
            [(
                meta.ticker, meta.name, meta.exchange, _iso(meta.start_date), _iso(meta.end_date),
                int(meta.has_intraday), themes, meta.description, fetched_at,
            )],
        )

    def write_bars(self, bars: Iterable[Any]) -> int:
        return self._write(
            "bars",
            ("ticker", "frequency", "ts", "open", "high", "low", "close", "volume"),
            [(b.ticker, b.frequency, b.timestamp, b.open, b.high, b.low, b.close, b.volume)
             for b in bars],
        )

    def mark_window(
        self, ticker: str, frequency: str, year: int, bars: list[Any],
        truncated: bool, fetched_at: str,
    ) -> None:
        stamps = [bar.timestamp for bar in bars]
        self._write(
            "windows",
            ("ticker", "frequency", "year", "first_ts", "last_ts", "rows", "truncated", "fetched_at"),
            [(ticker, frequency, year, min(stamps) if stamps else None,
              max(stamps) if stamps else None, len(bars), int(truncated), fetched_at)],
        )

    def completed_windows(self, frequency: str) -> set[tuple[str, int]]:
        """(ticker, year) pairs already fetched, so a rerun can skip them."""
        return {
            (row[0], row[1])
            for row in self.connection.execute(
                "SELECT ticker, year FROM windows WHERE frequency = ?", (frequency,)
            )
        }

    def known_symbols(self) -> dict[str, tuple[str, int]]:
        """ticker -> (exchange, has_intraday) for symbols already described."""
        return {
            row[0]: (row[1] or "", int(row[2] or 0))
            for row in self.connection.execute("SELECT ticker, exchange, has_intraday FROM symbols")
        }

    def truncated_windows(self) -> list[tuple[str, str, int, int]]:
        """Windows that came back at the row cap and so lost their early part."""
        return self.connection.execute(
            "SELECT ticker, frequency, year, rows FROM windows WHERE truncated = 1 "
            "ORDER BY ticker, year"
        ).fetchall()

    def log(self, run_id: str, logged_at: str, source: str, target: str, rows: int,
            status: str, message: str = "") -> None:
        self._write(
            "fetch_log",
            ("run_id", "logged_at", "source", "target", "rows", "status", "message"),
            [(run_id, logged_at, source, target, rows, status, message)],
        )

    def table_counts(self) -> dict[str, int]:
        return {
            table: self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in EXPORT_TABLES
        }

    def coverage(self) -> list[tuple[str, str, str, str, int]]:
        """Per ticker and frequency: span and bar count."""
        return self.connection.execute(
            "SELECT ticker, frequency, MIN(ts), MAX(ts), COUNT(*) FROM bars "
            "GROUP BY ticker, frequency ORDER BY ticker, frequency"
        ).fetchall()

    def export_csv(self, outdir: str | Path) -> dict[str, int]:
        target = Path(outdir)
        target.mkdir(parents=True, exist_ok=True)
        written = {}
        for table in EXPORT_TABLES:
            cursor = self.connection.execute(f"SELECT * FROM {table}")
            columns = [description[0] for description in cursor.description]
            with (target / f"{table}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
                count = 0
                for row in cursor:
                    writer.writerow(row)
                    count += 1
            written[table] = count
        return written
