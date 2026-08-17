"""SQLite store for CFTC Commitments of Traders history.

One wide table per reporting regime, mirroring the published file rather than
reshaping it: COT is already one row per contract per Tuesday, and analysts
reach for named columns like ``managed_money_positions_long_all``, so long-form
storage would multiply 1.9M rows into 200M for no gain.

Tables are built from the header of the first file ingested and widened with
``ALTER TABLE`` if the CFTC ever adds a column, so a schema change upstream
shows up as new columns rather than a crash. Every table carries a canonical
``contract_code`` / ``report_date`` pair as its primary key, which is the one
thing the three header conventions do not share.
"""

from __future__ import annotations

import csv
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from savi_uz.cftc_catalog import REPORTS, TEXT_COLUMNS, ReportSpec

BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS cot_reports (
    report        TEXT PRIMARY KEY,
    table_name    TEXT NOT NULL,
    label         TEXT,
    columns       INTEGER,
    first_date    TEXT,
    last_date     TEXT,
    rows          INTEGER,
    notes         TEXT,
    fetched_at    TEXT
);

CREATE TABLE IF NOT EXISTS cot_contracts (
    report            TEXT NOT NULL,
    contract_code     TEXT NOT NULL,
    market_name       TEXT,
    exchange          TEXT,
    commodity_code    TEXT,
    first_date        TEXT,
    last_date         TEXT,
    observations      INTEGER,
    PRIMARY KEY (report, contract_code)
);

CREATE TABLE IF NOT EXISTS fetch_log (
    run_id      TEXT NOT NULL,
    logged_at   TEXT NOT NULL,
    source      TEXT NOT NULL,
    target      TEXT NOT NULL,
    rows        INTEGER,
    status      TEXT NOT NULL,
    message     TEXT
);

CREATE INDEX IF NOT EXISTS idx_fetch_log_run ON fetch_log (run_id);
CREATE INDEX IF NOT EXISTS idx_cot_contracts_name ON cot_contracts (market_name);
"""

#: Written into every report table so the canonical key is never ambiguous.
KEY_COLUMNS = ("contract_code", "report_date")

METADATA_TABLES = ("cot_reports", "cot_contracts", "fetch_log")


def _quote(identifier: str) -> str:
    if not identifier.replace("_", "").isalnum():
        raise ValueError(f"unsafe SQL identifier: {identifier!r}")
    return f'"{identifier}"'


def _column_type(column: str) -> str:
    return "TEXT" if column in TEXT_COLUMNS else "NUMERIC"


class CftcStore:
    """Idempotent SQLite store; re-running a download updates rows in place."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript(BASE_SCHEMA)
        self.connection.commit()

    def __enter__(self) -> CftcStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    # -- schema -----------------------------------------------------------

    def existing_columns(self, table: str) -> tuple[str, ...]:
        rows = self.connection.execute(f"PRAGMA table_info({_quote(table)})").fetchall()
        return tuple(row[1] for row in rows)

    def ensure_table(self, table: str, columns: Sequence[str]) -> tuple[str, ...]:
        """Create or widen ``table`` so it holds every name in ``columns``.

        Returns the columns that were added, which is empty on the common path.
        """
        payload = [column for column in columns if column not in KEY_COLUMNS]
        duplicates = {column for column in payload if payload.count(column) > 1}
        if duplicates:
            raise ValueError(f"{table}: duplicate columns after normalisation: {sorted(duplicates)}")

        current = self.existing_columns(table)
        if not current:
            definitions = ", ".join(
                [
                    "contract_code TEXT NOT NULL",
                    "report_date TEXT NOT NULL",
                    *(f"{_quote(column)} {_column_type(column)}" for column in payload),
                    "PRIMARY KEY (contract_code, report_date)",
                ]
            )
            self.connection.execute(f"CREATE TABLE {_quote(table)} ({definitions})")
            self.connection.execute(
                f"CREATE INDEX IF NOT EXISTS {_quote(f'idx_{table}_date')} "
                f"ON {_quote(table)} (report_date)"
            )
            self.connection.commit()
            return ()

        added = tuple(column for column in payload if column not in current)
        for column in added:
            self.connection.execute(
                f"ALTER TABLE {_quote(table)} ADD COLUMN {_quote(column)} {_column_type(column)}"
            )
        if added:
            self.connection.commit()
        return added

    # -- writes -----------------------------------------------------------

    def write_rows(self, table: str, columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> int:
        payload = list(rows)
        if not payload:
            return 0
        self.ensure_table(table, columns)
        names = ", ".join(_quote(column) for column in columns)
        placeholders = ", ".join("?" * len(columns))
        self.connection.executemany(
            f"INSERT OR REPLACE INTO {_quote(table)} ({names}) VALUES ({placeholders})", payload
        )
        self.connection.commit()
        return len(payload)

    def write_chunk(self, chunk: Any) -> int:
        return self.write_rows(chunk.spec.table, chunk.columns, chunk.rows)

    def upsert_report(self, spec: ReportSpec, fetched_at: str) -> None:
        """Record what a report table actually ended up holding."""
        if not self.existing_columns(spec.table):
            return
        first, last, rows = self.connection.execute(
            f"SELECT MIN(report_date), MAX(report_date), COUNT(*) FROM {_quote(spec.table)}"
        ).fetchone()
        self.connection.execute(
            "INSERT OR REPLACE INTO cot_reports "
            "(report, table_name, label, columns, first_date, last_date, rows, notes, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                spec.key,
                spec.table,
                spec.label,
                len(self.existing_columns(spec.table)),
                first,
                last,
                rows,
                spec.notes,
                fetched_at,
            ),
        )
        self.connection.commit()

    def rebuild_contracts(self, spec: ReportSpec) -> int:
        """Refresh the contract directory: what each code is and when it traded.

        The market name is taken from the most recent report date, because
        exchanges rename contracts and the current name is the searchable one.
        """
        columns = self.existing_columns(spec.table)
        if not columns:
            return 0
        exchange = next(
            (c for c in ("cftc_market_code", "cftc_market_code_in_initials") if c in columns), None
        )
        exchange_expression = _quote(exchange) if exchange else "NULL"
        commodity = "cftc_commodity_code" if "cftc_commodity_code" in columns else None
        commodity_expression = _quote(commodity) if commodity else "NULL"

        self.connection.execute("DELETE FROM cot_contracts WHERE report = ?", (spec.key,))
        self.connection.execute(
            f"""
            INSERT OR REPLACE INTO cot_contracts
            SELECT ?,
                   t.contract_code,
                   (SELECT market_and_exchange_names FROM {_quote(spec.table)} n
                     WHERE n.contract_code = t.contract_code
                     ORDER BY n.report_date DESC LIMIT 1),
                   (SELECT {exchange_expression} FROM {_quote(spec.table)} n
                     WHERE n.contract_code = t.contract_code
                     ORDER BY n.report_date DESC LIMIT 1),
                   (SELECT {commodity_expression} FROM {_quote(spec.table)} n
                     WHERE n.contract_code = t.contract_code
                     ORDER BY n.report_date DESC LIMIT 1),
                   MIN(t.report_date),
                   MAX(t.report_date),
                   COUNT(*)
            FROM {_quote(spec.table)} t
            GROUP BY t.contract_code
            """,
            (spec.key,),
        )
        self.connection.commit()
        return self.connection.execute(
            "SELECT COUNT(*) FROM cot_contracts WHERE report = ?", (spec.key,)
        ).fetchone()[0]

    def log(self, run_id: str, logged_at: str, source: str, target: str, rows: int,
            status: str, message: str = "") -> None:
        self.connection.execute(
            "INSERT INTO fetch_log (run_id, logged_at, source, target, rows, status, message) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, logged_at, source, target, rows, status, message),
        )
        self.connection.commit()

    # -- reads ------------------------------------------------------------

    def tables(self) -> tuple[str, ...]:
        present = {
            row[0]
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        report_tables = tuple(spec.table for spec in REPORTS if spec.table in present)
        return report_tables + tuple(t for t in METADATA_TABLES if t in present)

    def table_counts(self) -> dict[str, int]:
        return {
            table: self.connection.execute(f"SELECT COUNT(*) FROM {_quote(table)}").fetchone()[0]
            for table in self.tables()
        }

    def coverage(self) -> list[tuple[str, str | None, str | None, int, int]]:
        """Per report: date span, row count and distinct contracts."""
        summary = []
        for spec in REPORTS:
            if not self.existing_columns(spec.table):
                continue
            first, last, rows, contracts = self.connection.execute(
                f"SELECT MIN(report_date), MAX(report_date), COUNT(*), "
                f"COUNT(DISTINCT contract_code) FROM {_quote(spec.table)}"
            ).fetchone()
            summary.append((spec.key, first, last, rows, contracts))
        return summary

    def export_csv(self, outdir: str | Path) -> dict[str, int]:
        target = Path(outdir)
        target.mkdir(parents=True, exist_ok=True)
        written = {}
        for table in self.tables():
            cursor = self.connection.execute(f"SELECT * FROM {_quote(table)}")
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
