"""SQLite store for earnings and valuation history.

Each source keeps its own table rather than being merged into one series, since
they disagree on purpose: Shiller's earnings are index-level and smoothed across
a long history, SEC facts are per-company as-filed, and Alpha Vantage's are
per-company as-expected. Reconciling them is analysis, not ingestion.
"""

from __future__ import annotations

import csv
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from savi_uz.equity_catalog import SHILLER_COLUMNS

#: Shiller's numeric columns, in a stable order, so the wide table matches the
#: catalogue without hand-maintaining a second list.
SHILLER_FIELDS: tuple[str, ...] = tuple(SHILLER_COLUMNS[key] for key in sorted(SHILLER_COLUMNS))

SHILLER_TABLE_SQL = "CREATE TABLE IF NOT EXISTS shiller_monthly (\n    obs_date TEXT PRIMARY KEY,\n" + "".join(
    f"    {field} REAL,\n" for field in SHILLER_FIELDS
) + "    source_url TEXT\n);"

SCHEMA = f"""
{SHILLER_TABLE_SQL}

CREATE TABLE IF NOT EXISTS sec_facts (
    cik           INTEGER NOT NULL,
    concept       TEXT NOT NULL,
    frame         TEXT NOT NULL,
    unit          TEXT NOT NULL,
    entity_name   TEXT,
    period_start  TEXT,
    period_end    TEXT,
    value         REAL,
    accession     TEXT,
    location      TEXT,
    PRIMARY KEY (cik, concept, frame, unit)
);

CREATE TABLE IF NOT EXISTS sec_frames (
    concept     TEXT NOT NULL,
    frame       TEXT NOT NULL,
    unit        TEXT NOT NULL,
    facts       INTEGER,
    fetched_at  TEXT,
    PRIMARY KEY (concept, frame, unit)
);

CREATE TABLE IF NOT EXISTS index_prices (
    ticker    TEXT NOT NULL,
    obs_date  TEXT NOT NULL,
    close     REAL,
    volume    REAL,
    PRIMARY KEY (ticker, obs_date)
);

CREATE TABLE IF NOT EXISTS analyst_earnings (
    ticker            TEXT NOT NULL,
    fiscal_ending     TEXT NOT NULL,
    reported_date     TEXT,
    reported_eps      REAL,
    estimated_eps     REAL,
    surprise          REAL,
    surprise_percent  REAL,
    PRIMARY KEY (ticker, fiscal_ending)
);

CREATE TABLE IF NOT EXISTS factset_reports (
    report_date                        TEXT PRIMARY KEY,
    document_date                      TEXT,
    source_url                          TEXT,
    quarter                             TEXT,
    pct_reported                        REAL,
    pct_positive_eps                    REAL,
    pct_positive_revenue                REAL,
    blended_earnings_growth             REAL,
    estimated_earnings_growth           REAL,
    estimated_growth_at_quarter_start   REAL,
    forward_12m_pe                      REAL,
    pe_5y_average                       REAL,
    pe_10y_average                      REAL,
    negative_guidance_count             INTEGER,
    positive_guidance_count             INTEGER,
    index_price                         REAL,
    forward_12m_eps                     REAL,
    missing_fields                      TEXT,
    missing_core_fields                 TEXT,
    page_text                           TEXT,
    fetched_at                          TEXT
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

CREATE INDEX IF NOT EXISTS idx_sec_facts_concept ON sec_facts (concept, frame);
CREATE INDEX IF NOT EXISTS idx_sec_facts_entity ON sec_facts (entity_name);
CREATE INDEX IF NOT EXISTS idx_index_prices_date ON index_prices (obs_date);
CREATE INDEX IF NOT EXISTS idx_fetch_log_run ON fetch_log (run_id);
"""

EXPORT_TABLES = (
    "shiller_monthly",
    "sec_facts",
    "sec_frames",
    "index_prices",
    "analyst_earnings",
    "factset_reports",
    "fetch_log",
)

#: Parsed numbers from page 1, in the order the table declares them.
FACTSET_FIELDS: tuple[str, ...] = (
    "quarter",
    "pct_reported",
    "pct_positive_eps",
    "pct_positive_revenue",
    "blended_earnings_growth",
    "estimated_earnings_growth",
    "estimated_growth_at_quarter_start",
    "forward_12m_pe",
    "pe_5y_average",
    "pe_10y_average",
    "negative_guidance_count",
    "positive_guidance_count",
    "index_price",
    "forward_12m_eps",
)


def _iso(value: date | str | None) -> str | None:
    if value is None:
        return None
    return value.isoformat() if isinstance(value, date) else str(value)


class EquityStore:
    """Idempotent SQLite store; re-running a download updates rows in place."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript(SCHEMA)
        self.connection.commit()

    def __enter__(self) -> EquityStore:
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

    def write_shiller(self, rows: Iterable[Any], source_url: str) -> int:
        columns = ("obs_date", *SHILLER_FIELDS, "source_url")
        return self._write(
            "shiller_monthly",
            columns,
            [
                (_iso(row.obs_date), *(row.values.get(field) for field in SHILLER_FIELDS), source_url)
                for row in rows
            ],
        )

    def write_sec_facts(self, facts: Iterable[Any]) -> int:
        return self._write(
            "sec_facts",
            (
                "cik", "concept", "frame", "unit", "entity_name",
                "period_start", "period_end", "value", "accession", "location",
            ),
            [
                (
                    fact.cik, fact.concept, fact.frame, fact.unit, fact.entity_name,
                    _iso(fact.period_start), _iso(fact.period_end), fact.value,
                    fact.accession, fact.location,
                )
                for fact in facts
            ],
        )

    def write_sec_frame(self, concept: str, frame: str, unit: str, facts: int, fetched_at: str) -> int:
        return self._write(
            "sec_frames", ("concept", "frame", "unit", "facts", "fetched_at"),
            [(concept, frame, unit, facts, fetched_at)],
        )

    def write_index_prices(self, bars: Iterable[Any]) -> int:
        return self._write(
            "index_prices",
            ("ticker", "obs_date", "close", "volume"),
            [(bar.ticker, _iso(bar.obs_date), bar.close, bar.volume) for bar in bars],
        )

    def write_analyst_earnings(self, rows: Iterable[Any]) -> int:
        return self._write(
            "analyst_earnings",
            (
                "ticker", "fiscal_ending", "reported_date", "reported_eps",
                "estimated_eps", "surprise", "surprise_percent",
            ),
            [
                (
                    row.ticker, _iso(row.fiscal_ending), _iso(row.reported_date),
                    row.reported_eps, row.estimated_eps, row.surprise, row.surprise_percent,
                )
                for row in rows
            ],
        )

    def write_factset_report(self, metrics: Any, fetched_at: str) -> int:
        """Store one weekly edition, parsed numbers and page text together.

        The page text is kept so that an improved pattern can be replayed over
        the whole history without re-downloading nine years of PDFs.
        """
        columns = (
            "report_date", "document_date", "source_url", *FACTSET_FIELDS,
            "missing_fields", "missing_core_fields", "page_text", "fetched_at",
        )
        row = (
            _iso(metrics.report_date),
            _iso(metrics.document_date),
            metrics.source_url,
            *(metrics.values.get(name) for name in FACTSET_FIELDS),
            ",".join(metrics.missing),
            ",".join(metrics.missing_core),
            metrics.page_text,
            fetched_at,
        )
        return self._write("factset_reports", columns, [row])

    def factset_page_texts(self) -> list[tuple[str, str, str]]:
        """(report_date, source_url, page_text) for offline re-parsing."""
        return self.connection.execute(
            "SELECT report_date, source_url, page_text FROM factset_reports "
            "WHERE page_text IS NOT NULL AND page_text != '' ORDER BY report_date"
        ).fetchall()

    def factset_field_coverage(self) -> list[tuple[str, int]]:
        """How many editions carry each field, to show where parsing thins out."""
        total = self.connection.execute("SELECT COUNT(*) FROM factset_reports").fetchone()[0]
        if not total:
            return []
        return [
            (
                name,
                self.connection.execute(
                    f"SELECT COUNT({name}) FROM factset_reports"
                ).fetchone()[0],
            )
            for name in FACTSET_FIELDS
        ]

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

    def coverage(self) -> list[tuple[str, str | None, str | None, int]]:
        """Date span and row count per source, for the run summary."""
        spans = [
            ("shiller", "SELECT MIN(obs_date), MAX(obs_date), COUNT(*) FROM shiller_monthly"),
            ("index", "SELECT MIN(obs_date), MAX(obs_date), COUNT(*) FROM index_prices"),
            ("sec", "SELECT MIN(period_end), MAX(period_end), COUNT(*) FROM sec_facts"),
            (
                "estimates",
                "SELECT MIN(fiscal_ending), MAX(fiscal_ending), COUNT(*) FROM analyst_earnings",
            ),
            ("factset", "SELECT MIN(report_date), MAX(report_date), COUNT(*) FROM factset_reports"),
        ]
        summary = []
        for key, query in spans:
            first, last, rows = self.connection.execute(query).fetchone()
            summary.append((key, first, last, rows))
        return summary

    def shiller_earnings_gap(self) -> tuple[str | None, str | None]:
        """Latest month with a price, and the latest with an earnings figure.

        Shiller fills price before earnings, so these differ by a few months
        even in a fresh copy; the gap is what a CAPE calculation has to bridge.
        """
        return self.connection.execute(
            "SELECT MAX(obs_date), (SELECT MAX(obs_date) FROM shiller_monthly WHERE earnings IS NOT NULL) "
            "FROM shiller_monthly WHERE sp500_price IS NOT NULL"
        ).fetchone()

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
