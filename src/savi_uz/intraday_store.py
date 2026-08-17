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

-- Intraday bars are raw. These factors are what lets a backtest rebuild an
-- adjusted series instead of reading a 10:1 split as a 90% loss.
CREATE TABLE IF NOT EXISTS corporate_actions (
    ticker       TEXT NOT NULL,
    obs_date     TEXT NOT NULL,
    close        REAL,
    adj_close    REAL,
    split_factor REAL,
    div_cash     REAL,
    PRIMARY KEY (ticker, obs_date)
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

EXPORT_TABLES = ("symbols", "windows", "bars", "corporate_actions", "fetch_log")


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
        self, ticker: str, frequency: str, first_year: int, bars: list[Any],
        truncated: bool, fetched_at: str, last_year: int | None = None,
    ) -> None:
        """Record a fetched chunk as one row per calendar year it covered.

        Resume state is kept per year even when a request spans several, so
        changing ``--years-per-request`` between runs does not invalidate work
        already done: the downloader re-checks years, not request windows.
        """
        span_end = first_year if last_year is None else last_year
        by_year: dict[int, list[str]] = {year: [] for year in range(first_year, span_end + 1)}
        for bar in bars:
            stamp = str(bar.timestamp)
            # A timestamp that does not start with a year is attributed to the
            # window's first year rather than crashing the run: losing the split
            # is recoverable, losing the whole download is not.
            head = stamp[:4]
            year = int(head) if head.isdigit() else first_year
            by_year.setdefault(year, []).append(stamp)

        self._write(
            "windows",
            ("ticker", "frequency", "year", "first_ts", "last_ts", "rows", "truncated", "fetched_at"),
            [
                (ticker, frequency, year, min(stamps) if stamps else None,
                 max(stamps) if stamps else None, len(stamps), int(truncated), fetched_at)
                for year, stamps in sorted(by_year.items())
            ],
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

    def write_adjustments(self, rows: Iterable[Any]) -> int:
        return self._write(
            "corporate_actions",
            ("ticker", "obs_date", "close", "adj_close", "split_factor", "div_cash"),
            [(r.ticker, _iso(r.obs_date), r.close, r.adj_close, r.split_factor, r.div_cash)
             for r in rows],
        )

    def splits(self, ticker: str | None = None) -> list[tuple[str, str, float]]:
        """Split events, which are what silently break an unadjusted backtest."""
        clause = "AND ticker = ?" if ticker else ""
        params = (ticker,) if ticker else ()
        return self.connection.execute(
            f"SELECT ticker, obs_date, split_factor FROM corporate_actions "
            f"WHERE split_factor IS NOT NULL AND split_factor != 1.0 {clause} "
            f"ORDER BY obs_date",
            params,
        ).fetchall()

    def ohlc_violations(self, ticker: str | None = None) -> list[tuple[str, str, int]]:
        """Bars whose close or open falls outside the bar's own high/low range.

        IEX resampling takes the close from the last trade while the high and
        low come from the interval's own aggregation, and the two occasionally
        disagree by a few cents. The rows are left exactly as published -- a
        backtest that assumes ``low <= close <= high`` should know the count
        rather than have the data quietly rewritten under it.
        """
        clause = "AND ticker = ?" if ticker else ""
        params = (ticker,) if ticker else ()
        return self.connection.execute(
            f"""
            SELECT ticker, frequency, COUNT(*) FROM bars
            WHERE open IS NOT NULL AND high IS NOT NULL
              AND low IS NOT NULL AND close IS NOT NULL
              AND (close > high OR close < low OR open > high OR open < low)
              {clause}
            GROUP BY ticker, frequency ORDER BY ticker
            """,
            params,
        ).fetchall()

    def missing_volume(self) -> list[tuple[str, str, int, int]]:
        """Per series: how many bars carry no volume, against the total."""
        return self.connection.execute(
            "SELECT ticker, frequency, SUM(CASE WHEN volume IS NULL OR volume = 0 THEN 1 ELSE 0 END), "
            "COUNT(*) FROM bars GROUP BY ticker, frequency ORDER BY ticker"
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
