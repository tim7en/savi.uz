"""SQLite store for macro history, keeping release and vintage timestamps.

A plain wide CSV cannot express "this value, as it was known on that date", so
the store is long-form and keyed by the real-time dimension: ``observations``
holds current values, ``first_release`` holds the print the market traded on,
and ``vintages`` holds the full revision matrix where it was collected.
"""

from __future__ import annotations

import csv
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS series (
    series_id            TEXT PRIMARY KEY,
    source               TEXT NOT NULL,
    group_name           TEXT NOT NULL,
    label                TEXT,
    title                TEXT,
    units                TEXT,
    frequency            TEXT,
    seasonal_adjustment  TEXT,
    observation_start    TEXT,
    observation_end      TEXT,
    last_updated         TEXT,
    release_id           INTEGER,
    release_name         TEXT,
    vintage_policy       TEXT,
    notes                TEXT,
    fetched_at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS observations (
    series_id  TEXT NOT NULL,
    obs_date   TEXT NOT NULL,
    value      REAL,
    PRIMARY KEY (series_id, obs_date)
);

CREATE TABLE IF NOT EXISTS first_release (
    series_id     TEXT NOT NULL,
    obs_date      TEXT NOT NULL,
    value         REAL,
    release_date  TEXT,
    superseded_on TEXT,
    PRIMARY KEY (series_id, obs_date)
);

CREATE TABLE IF NOT EXISTS vintages (
    series_id     TEXT NOT NULL,
    obs_date      TEXT NOT NULL,
    vintage_date  TEXT NOT NULL,
    value         REAL,
    PRIMARY KEY (series_id, obs_date, vintage_date)
);

CREATE TABLE IF NOT EXISTS release_dates (
    release_id    INTEGER NOT NULL,
    release_date  TEXT NOT NULL,
    PRIMARY KEY (release_id, release_date)
);

CREATE TABLE IF NOT EXISTS reference_rates (
    rate_type           TEXT NOT NULL,
    effective_date      TEXT NOT NULL,
    percent_rate        REAL,
    percentile_1        REAL,
    percentile_25       REAL,
    percentile_75       REAL,
    percentile_99       REAL,
    volume_billions     REAL,
    target_rate_from    REAL,
    target_rate_to      REAL,
    revision_indicator  TEXT,
    PRIMARY KEY (rate_type, effective_date)
);

CREATE TABLE IF NOT EXISTS gsw_params (
    curve_date TEXT PRIMARY KEY,
    beta0 REAL, beta1 REAL, beta2 REAL, beta3 REAL, tau1 REAL, tau2 REAL
);

CREATE TABLE IF NOT EXISTS gsw_rates (
    curve_date TEXT NOT NULL,
    mnemonic   TEXT NOT NULL,
    value      REAL,
    PRIMARY KEY (curve_date, mnemonic)
);

CREATE TABLE IF NOT EXISTS fed_path (
    curve_date      TEXT NOT NULL,
    horizon_months  INTEGER NOT NULL,
    forward_rate    REAL,
    PRIMARY KEY (curve_date, horizon_months)
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

CREATE INDEX IF NOT EXISTS idx_observations_date ON observations (obs_date);
CREATE INDEX IF NOT EXISTS idx_first_release_date ON first_release (release_date);
CREATE INDEX IF NOT EXISTS idx_vintages_vintage ON vintages (vintage_date);
CREATE INDEX IF NOT EXISTS idx_series_group ON series (group_name);
CREATE INDEX IF NOT EXISTS idx_fetch_log_run ON fetch_log (run_id);
"""

#: Tables exported to CSV, in dependency-free order.
EXPORT_TABLES = (
    "series",
    "observations",
    "first_release",
    "vintages",
    "release_dates",
    "reference_rates",
    "gsw_params",
    "gsw_rates",
    "fed_path",
    "fetch_log",
)


def _iso(value: date | str | None) -> str | None:
    if value is None:
        return None
    return value.isoformat() if isinstance(value, date) else str(value)


class MacroStore:
    """Idempotent SQLite store; re-running a download updates rows in place."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript(SCHEMA)
        self.connection.commit()

    def __enter__(self) -> MacroStore:
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
        statement = f"INSERT OR REPLACE INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
        self.connection.executemany(statement, payload)
        self.connection.commit()
        return len(payload)

    def upsert_series(self, record: dict[str, Any]) -> None:
        columns = (
            "series_id", "source", "group_name", "label", "title", "units", "frequency",
            "seasonal_adjustment", "observation_start", "observation_end", "last_updated",
            "release_id", "release_name", "vintage_policy", "notes", "fetched_at",
        )
        self._write(
            "series", columns, [tuple(_iso(record.get(column)) if column.endswith(("_start", "_end"))
                                     else record.get(column) for column in columns)]
        )

    def write_observations(self, series_id: str, observations: Iterable[Any]) -> int:
        return self._write(
            "observations",
            ("series_id", "obs_date", "value"),
            [(series_id, _iso(o.obs_date), o.value) for o in observations],
        )

    def write_first_releases(self, series_id: str, observations: Iterable[Any]) -> int:
        return self._write(
            "first_release",
            ("series_id", "obs_date", "value", "release_date", "superseded_on"),
            [
                (series_id, _iso(o.obs_date), o.value, _iso(o.realtime_start), _iso(o.realtime_end))
                for o in observations
            ],
        )

    def write_vintages(self, series_id: str, observations: Iterable[Any]) -> int:
        return self._write(
            "vintages",
            ("series_id", "obs_date", "vintage_date", "value"),
            [
                (series_id, _iso(o.obs_date), _iso(o.realtime_start), o.value)
                for o in observations
                if o.realtime_start is not None
            ],
        )

    def write_release_dates(self, release_id: int, dates: Iterable[date]) -> int:
        return self._write(
            "release_dates", ("release_id", "release_date"), [(release_id, _iso(day)) for day in dates]
        )

    def write_reference_rates(self, rates: Iterable[Any]) -> int:
        return self._write(
            "reference_rates",
            (
                "rate_type", "effective_date", "percent_rate", "percentile_1", "percentile_25",
                "percentile_75", "percentile_99", "volume_billions", "target_rate_from",
                "target_rate_to", "revision_indicator",
            ),
            [
                (
                    rate.rate_type, _iso(rate.effective_date), rate.percent_rate, rate.percentile_1,
                    rate.percentile_25, rate.percentile_75, rate.percentile_99, rate.volume_billions,
                    rate.target_rate_from, rate.target_rate_to, rate.revision_indicator,
                )
                for rate in rates
            ],
        )

    def write_gsw_params(self, parameters: Iterable[Any]) -> int:
        return self._write(
            "gsw_params",
            ("curve_date", "beta0", "beta1", "beta2", "beta3", "tau1", "tau2"),
            [
                (_iso(p.curve_date), p.beta0, p.beta1, p.beta2, p.beta3, p.tau1, p.tau2)
                for p in parameters
            ],
        )

    def write_gsw_rates(self, rates: Iterable[tuple[date, str, float]]) -> int:
        return self._write(
            "gsw_rates",
            ("curve_date", "mnemonic", "value"),
            [(_iso(curve_date), mnemonic, value) for curve_date, mnemonic, value in rates],
        )

    def write_fed_path(self, rows: Iterable[tuple[date, int, float]]) -> int:
        return self._write(
            "fed_path",
            ("curve_date", "horizon_months", "forward_rate"),
            [(_iso(curve_date), months, rate) for curve_date, months, rate in rows],
        )

    def log(self, run_id: str, logged_at: str, source: str, target: str, rows: int,
            status: str, message: str = "") -> None:
        self._write(
            "fetch_log",
            ("run_id", "logged_at", "source", "target", "rows", "status", "message"),
            [(run_id, logged_at, source, target, rows, status, message)],
        )

    def table_counts(self) -> dict[str, int]:
        counts = {}
        for table in EXPORT_TABLES:
            counts[table] = self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return counts

    def coverage(self) -> list[tuple[str, str, str, int, int]]:
        """Per series: observation span, row count and how many first prints exist."""
        query = """
            SELECT s.series_id,
                   MIN(o.obs_date),
                   MAX(o.obs_date),
                   COUNT(o.obs_date),
                   (SELECT COUNT(*) FROM first_release f WHERE f.series_id = s.series_id)
            FROM series s LEFT JOIN observations o ON o.series_id = s.series_id
            GROUP BY s.series_id ORDER BY s.series_id
        """
        return self.connection.execute(query).fetchall()

    def export_csv(self, outdir: str | Path) -> dict[str, int]:
        """Dump every table to CSV for tools that will not open SQLite."""
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
