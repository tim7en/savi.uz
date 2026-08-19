from __future__ import annotations

import tempfile
import time
import unittest
from datetime import date
from pathlib import Path

from savi_uz.dashboard import RefreshManager, tracked_snapshot
from savi_uz.intraday_store import IntradayStore
from savi_uz.tiingo_sources import Bar, SymbolMeta


def bar(ticker: str, day: str, stamp: str, close: float, volume: float,
        frequency: str = "5min") -> Bar:
    return Bar(ticker, frequency, f"{day}T{stamp}.000Z", close, close, close, close, volume)


class DashboardSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.db = Path(self.directory.name) / "bars.db"
        with IntradayStore(self.db) as store:
            store.upsert_symbol(
                SymbolMeta("SPY", "SPDR S&P 500", "NYSE ARCA", date(1993, 1, 29), None),
                "Broad index", "now",
            )
            store.write_bars([
                bar("SPY", "2026-08-17", "13:30:00", 100.0, 40.0),
                bar("SPY", "2026-08-17", "19:55:00", 101.0, 60.0),
                bar("SPY", "2026-08-18", "13:30:00", 102.0, 90.0),
                bar("SPY", "2026-08-18", "19:55:00", 103.0, 110.0),
            ])

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_snapshot_reports_latest_price_change_and_session_volume(self) -> None:
        snapshot = tracked_snapshot(self.db, today=date(2026, 8, 19))
        row = snapshot["assets"][0]
        self.assertEqual(row["price"], 103.0)
        self.assertAlmostEqual(row["change_pct"], 1.98, places=2)
        self.assertEqual(row["volume"], 200.0)
        self.assertEqual(row["average_volume_20d"], 100.0)
        self.assertEqual(row["volume_ratio"], 2.0)
        self.assertEqual(row["session_date"], "2026-08-18")
        self.assertEqual(row["status"], "current")

    def test_old_latest_session_is_marked_stale(self) -> None:
        row = tracked_snapshot(self.db, today=date(2026, 9, 1))["assets"][0]
        self.assertEqual(row["status"], "stale")


class FakeClient:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def fetch_intraday(self, ticker: str, start: date, end: date, frequency: str):
        return [bar(ticker, end.isoformat(), "19:55:00", 105.0, 123.0)], False

    def fetch_daily(self, ticker: str, start: date, end: date):
        return [bar(ticker, end.isoformat(), "00:00:00", 10.0, 50.0, "daily")], False


class RefreshManagerTests(unittest.TestCase):
    def test_refresh_writes_recent_bars_and_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "bars.db"
            with IntradayStore(db) as store:
                store.upsert_symbol(
                    SymbolMeta("SPY", "SPDR", "NYSE ARCA", date(1993, 1, 29), None),
                    "Index", "now",
                )
            manager = RefreshManager(
                db, client_factory=FakeClient, api_key_factory=lambda: "test",
                today_factory=lambda: date(2026, 8, 19),
            )
            self.assertTrue(manager.start())
            for _ in range(100):
                if not manager.status()["running"]:
                    break
                time.sleep(0.01)
            status = manager.status()
            self.assertEqual(status["state"], "complete")
            self.assertEqual(status["completed"], 1)
            self.assertEqual(status["percent"], 100.0)
            self.assertEqual(status["bars_received"], 1)
            self.assertEqual(
                tracked_snapshot(db, today=date(2026, 8, 19))["assets"][0]["price"], 105.0
            )


if __name__ == "__main__":
    unittest.main()
