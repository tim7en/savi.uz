from __future__ import annotations

import json
import tempfile
import time
import unittest
from datetime import date
from pathlib import Path

from savi_uz.fundamentals import (
    EarningsEstimatesRefreshManager,
    FundamentalsRefreshManager,
    fundamentals_snapshot,
)


def write_payload(folder: Path, ticker: str, suffix: str, data: dict, stamp: str = "2026-05-12T09:00:00") -> None:
    (folder / f"{ticker}_{suffix}.json").write_text(json.dumps({
        "timestamp": stamp, "symbol": ticker, "data_type": suffix, "data": data,
    }), encoding="utf-8")


class FundamentalsSnapshotTests(unittest.TestCase):
    def test_latest_quarter_and_growth_metrics_are_derived(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "sp500_symbols.json").write_text(json.dumps({
                "date": "2026-01-01", "source": "test", "symbols": ["AAA"],
            }), encoding="utf-8")
            stamp = "2026-08-19T10:00:00"
            write_payload(folder, "AAA", "overview", {
                "Name": "Alpha Co", "Sector": "TECHNOLOGY", "LatestQuarter": "2026-06-30",
                "MarketCapitalization": "1000000", "PERatio": "20", "ForwardPE": "18",
                "ReturnOnEquityTTM": "0.25",
            }, stamp)
            write_payload(folder, "AAA", "earnings", {"quarterlyEarnings": [{
                "fiscalDateEnding": "2026-06-30", "reportedDate": "2026-07-20",
                "reportedEPS": "1.2", "estimatedEPS": "1.0", "surprisePercentage": "20",
            }]}, stamp)
            write_payload(folder, "AAA", "income_statement", {"quarterlyReports": [
                {"fiscalDateEnding": "2026-06-30", "totalRevenue": "120", "netIncome": "24"},
                {"fiscalDateEnding": "2025-06-30", "totalRevenue": "100", "netIncome": "10"},
            ]}, stamp)
            write_payload(folder, "AAA", "balance_sheet", {"quarterlyReports": [{
                "fiscalDateEnding": "2026-06-30", "cashAndShortTermInvestments": "50",
                "shortLongTermDebtTotal": "30",
            }]}, stamp)
            write_payload(folder, "AAA", "cash_flow", {"quarterlyReports": [{
                "fiscalDateEnding": "2026-06-30", "operatingCashflow": "40",
                "capitalExpenditures": "10",
            }]}, stamp)

            snapshot = fundamentals_snapshot(folder, today=date(2026, 8, 19))
            row = snapshot["companies"][0]
            self.assertEqual(row["latest_quarter"], "2026-06-30")
            self.assertEqual(row["surprise_pct"], 20.0)
            self.assertEqual(row["revenue_yoy_pct"], 20.0)
            self.assertEqual(row["net_margin_pct"], 20.0)
            self.assertEqual(row["free_cash_flow"], 30.0)
            self.assertEqual(row["return_on_equity_pct"], 25.0)
            self.assertEqual(row["status"], "current")


class FakeClient:
    calls: list[tuple[str, str]] = []

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def fetch(self, function: str, ticker: str) -> dict:
        self.calls.append((function, ticker))
        if function == "EARNINGS_ESTIMATES":
            return {"symbol": ticker, "estimates": []}
        if function == "OVERVIEW":
            return {"Symbol": ticker, "Name": "Test company"}
        if function == "EARNINGS":
            return {"symbol": ticker, "quarterlyEarnings": []}
        return {"symbol": ticker, "quarterlyReports": []}


class FundamentalsRefreshManagerTests(unittest.TestCase):
    def test_refresh_writes_five_files_and_a_same_day_resume_skips_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "sp500_symbols.json").write_text(
                json.dumps({"date": "2026-01-01", "source": "test", "symbols": ["AAA"]}),
                encoding="utf-8",
            )
            FakeClient.calls = []
            manager = FundamentalsRefreshManager(
                folder, client_factory=FakeClient, api_key_factory=lambda: "key",
                today_factory=lambda: date.today(),
            )
            self.assertTrue(manager.start())
            for _ in range(100):
                if not manager.status()["running"]:
                    break
                time.sleep(0.01)
            status = manager.status()
            self.assertEqual(status["state"], "complete")
            self.assertEqual(status["files_updated"], 5)
            self.assertEqual(len(FakeClient.calls), 5)

            resumed = FundamentalsRefreshManager(
                folder, client_factory=FakeClient, api_key_factory=lambda: "key",
                today_factory=lambda: date.today(),
            )
            resumed.start()
            for _ in range(100):
                if not resumed.status()["running"]:
                    break
                time.sleep(0.01)
            self.assertEqual(resumed.status()["total"], 0)
            self.assertEqual(resumed.status()["skipped_current"], 5)
            self.assertEqual(len(FakeClient.calls), 5)

    def test_forward_estimate_refresh_is_a_separate_one_call_per_symbol_queue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "sp500_symbols.json").write_text(
                json.dumps({"date": "2026-01-01", "source": "test", "symbols": ["AAA", "BBB"]}),
                encoding="utf-8",
            )
            FakeClient.calls = []
            manager = EarningsEstimatesRefreshManager(
                folder, client_factory=FakeClient, api_key_factory=lambda: "key",
            )
            self.assertTrue(manager.start())
            for _ in range(100):
                if not manager.status()["running"]:
                    break
                time.sleep(0.01)
            self.assertEqual(manager.status()["state"], "complete")
            self.assertEqual(manager.status()["files_updated"], 2)
            self.assertEqual(FakeClient.calls, [
                ("EARNINGS_ESTIMATES", "AAA"), ("EARNINGS_ESTIMATES", "BBB"),
            ])
            self.assertTrue((folder / "AAA_earnings_estimates.json").is_file())


if __name__ == "__main__":
    unittest.main()
