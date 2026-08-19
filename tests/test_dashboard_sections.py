from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path

from savi_uz.dashboard_sections import earnings_analysis_snapshot


class EarningsAnalysisSnapshotTests(unittest.TestCase):
    def test_uses_forward_quarter_and_prior_30_day_consensus_without_lookahead(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "equity.db"
            connection = sqlite3.connect(database)
            connection.execute(
                """CREATE TABLE factset_reports (
                    report_date TEXT, quarter TEXT, pct_reported REAL,
                    pct_positive_eps REAL, pct_positive_revenue REAL,
                    blended_earnings_growth REAL, estimated_earnings_growth REAL,
                    estimated_growth_at_quarter_start REAL, forward_12m_pe REAL,
                    pe_5y_average REAL, pe_10y_average REAL,
                    negative_guidance_count REAL, positive_guidance_count REAL,
                    index_price REAL, forward_12m_eps REAL
                )"""
            )
            connection.execute(
                "INSERT INTO factset_reports VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ("2026-08-14", "Q2 2026", 90, 80, 70, 12, 9, 7, 21, 20, 19, 4, 3, 6400, 300),
            )
            connection.commit()
            connection.close()

            (root / "sp500_symbols.json").write_text(json.dumps({
                "date": "2026-08-01", "source": "test", "symbols": ["AAA"],
            }), encoding="utf-8")
            wrapper = lambda suffix, data: {
                "timestamp": "2026-08-19T09:00:00+05:00", "symbol": "AAA",
                "data_type": suffix, "data": data,
            }
            (root / "AAA_overview.json").write_text(json.dumps(wrapper("overview", {
                "Name": "Alpha Co", "Sector": "Technology",
            })), encoding="utf-8")
            (root / "AAA_earnings_estimates.json").write_text(json.dumps(wrapper(
                "earnings_estimates", {"symbol": "AAA", "estimates": [{
                    "date": "2026-09-30", "horizon": "fiscal quarter",
                    "eps_estimate_average": "2.20",
                    "eps_estimate_average_30_days_ago": "2.00",
                    "eps_estimate_revision_up_trailing_30_days": "5",
                    "eps_estimate_revision_down_trailing_30_days": "2",
                    "eps_estimate_analyst_count": "20",
                }]},
            )), encoding="utf-8")

            payload = earnings_analysis_snapshot(database, root, today=date(2026, 8, 19))
            self.assertEqual(payload["factset_latest"]["report_date"], "2026-08-14")
            self.assertEqual(payload["estimate_covered"], 1)
            self.assertEqual(payload["estimates"][0]["period"], "2026-09-30")
            self.assertEqual(payload["estimates"][0]["eps_revision_pct"], 10.0)
            self.assertEqual(payload["net_revision_breadth"], 3.0)


if __name__ == "__main__":
    unittest.main()
