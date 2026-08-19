from __future__ import annotations

import importlib.util
import sqlite3
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_cross_asset_intraday.py"
SPEC = importlib.util.spec_from_file_location("download_cross_asset_intraday", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

RUNNER_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_cross_asset_daily.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("run_cross_asset_daily", RUNNER_SCRIPT)
assert RUNNER_SPEC and RUNNER_SPEC.loader
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = RUNNER
RUNNER_SPEC.loader.exec_module(RUNNER)


class CrossAssetIntradayTests(unittest.TestCase):
    def test_month_range_is_inclusive(self):
        self.assertEqual(
            MODULE.month_range("2025-11", "2026-02"),
            ["2025-11", "2025-12", "2026-01", "2026-02"],
        )

    def test_previous_month_crosses_year_boundary(self):
        self.assertEqual(MODULE.previous_month(date(2026, 1, 4)), "2025-12")

    def test_parse_series_converts_new_york_time_to_utc(self):
        payload = {
            "Time Series (30min)": {
                "2026-07-01 09:30:00": {
                    "1. open": "100",
                    "2. high": "102",
                    "3. low": "99",
                    "4. close": "101",
                    "5. volume": "1234",
                }
            }
        }
        rows = MODULE.parse_series("SPY", payload)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].timestamp, "2026-07-01T13:30:00.000Z")
        self.assertEqual(rows[0].frequency, "30min")
        self.assertEqual(rows[0].close, 101.0)

    def test_backtest_loader_selects_only_requested_frequency(self):
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "bars.db"
            connection = sqlite3.connect(database)
            connection.executescript("""
                CREATE TABLE venue_map (ticker TEXT PRIMARY KEY, sleeve TEXT);
                CREATE TABLE bars (
                    ticker TEXT, frequency TEXT, ts TEXT, open REAL, high REAL,
                    low REAL, close REAL, volume REAL
                );
            """)
            connection.executemany(
                "INSERT INTO venue_map VALUES (?, ?)",
                (("DAILY", "test"), ("INTRA", "test")),
            )
            rows = []
            for index in range(400):
                stamp = f"2025-01-01T00:{index:03d}:00Z"
                rows.append(("DAILY", "daily", stamp, 1, 2, 0.5, 1.5, 10))
                rows.append(("INTRA", "30min", stamp, 1, 2, 0.5, 1.5, 10))
            connection.executemany("INSERT INTO bars VALUES (?,?,?,?,?,?,?,?)", rows)
            connection.commit()
            connection.close()

            book, _ = RUNNER.load(SimpleNamespace(
                db=database, frequency="30min", start="2000-01-01",
            ))
            self.assertEqual(list(book), ["INTRA"])
            self.assertEqual(len(book["INTRA"]), 400)


if __name__ == "__main__":
    unittest.main()
