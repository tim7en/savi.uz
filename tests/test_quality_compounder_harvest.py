import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_spy_quality_rotation.py"
SPEC = importlib.util.spec_from_file_location("spy_quality_rotation_compounder", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def args():
    return SimpleNamespace(
        initial=100_000.0,
        spy_share=0.80,
        spread=0.0,
        trade_bp=0.0,
        min_history_years=0.0,
        risk_per_stock=0.01,
        stock_tail_loss=0.79,
        max_quality_hold_years=5.0,
        trend_exit_days=200,
        quality_harvest_share=0.05,
        quality_harvest_step=0.20,
        compounder_cagr=0.05,
    )


class QualityCompounderHarvestTests(unittest.TestCase):
    def test_earnings_guardrail_ignores_reports_after_signal_date(self):
        dates = pd.date_range("2018-01-15", periods=20, freq="QS")
        history = pd.DataFrame(
            {"reported_eps": [1.0] * 16 + [-2.0] * 4}, index=dates
        )

        before_future_reports = MODULE.earnings_quality_as_of(
            history, dates[15]
        )
        after_future_reports = MODULE.earnings_quality_as_of(
            history, dates[-1]
        )

        self.assertFalse(before_future_reports["broken"])
        self.assertEqual(before_future_reports["reports_used"], 16)
        self.assertTrue(after_future_reports["broken"])
        self.assertEqual(after_future_reports["reason"], "non_positive_ttm")

    def test_rolling_cagr_does_not_read_future_price(self):
        dates = pd.date_range("2015-01-02", periods=7, freq="YS")
        base = pd.Series([100, 105, 110, 115, 120, 125, 10_000], index=dates)
        changed_future = base.copy()
        changed_future.iloc[-1] = 1.0

        first = MODULE.rolling_total_return_cagr_as_of(base, 5, 5.0)
        second = MODULE.rolling_total_return_cagr_as_of(
            changed_future, 5, 5.0
        )

        self.assertAlmostEqual(first, second)
        self.assertGreater(first, 0.04)

    def test_profit_harvest_uses_prior_close_and_five_percent_current_shares(self):
        dates = pd.date_range("2020-01-02", periods=6, freq="B")
        prices = pd.DataFrame(
            {
                "SPY": [100.0, 80.0, 80.0, 80.0, 80.0, 80.0],
                "QUALITY": [10.0, 10.0, 10.0, 12.0, 12.0, 12.0],
            },
            index=dates,
        )
        rates = pd.Series(0.0, index=dates)

        _, events = MODULE.simulate(
            prices, rates, args(), "harvest_test", "step_3_2_1",
            staging=True, quality_at_40=True, harvest_share=0.0,
            signal_source="portfolio", exit_policy="compounder_guardrail",
        )
        purchase = next(event for event in events if event.kind == "deploy_quality")
        harvests = [
            event for event in events if event.kind == "quality_profit_harvest"
        ]

        self.assertEqual(len(harvests), 1)
        self.assertEqual(harvests[0].date, dates[4].date().isoformat())
        self.assertAlmostEqual(harvests[0].amount, purchase.amount * 0.05 * 1.20)

    def test_post_five_year_exit_requires_low_cagr_and_broken_earnings(self):
        dates = pd.to_datetime([
            "2020-01-02", "2020-01-03", "2020-01-06",
            "2025-01-07", "2025-01-08",
        ])
        prices = pd.DataFrame(
            {"SPY": [100.0, 80.0, 80.0, 80.0, 80.0],
             "QUALITY": [10.0, 10.0, 10.0, 8.0, 8.0]},
            index=dates,
        )
        rates = pd.Series(0.0, index=dates)
        report_dates = pd.date_range("2020-03-01", periods=16, freq="QS")
        history = pd.DataFrame(
            {"reported_eps": [1.0] * 12 + [-1.0] * 4},
            index=report_dates,
        )

        _, events = MODULE.simulate(
            prices, rates, args(), "exit_test", "step_3_2_1",
            staging=True, quality_at_40=True, harvest_share=0.0,
            signal_source="portfolio", exit_policy="compounder_guardrail",
            earnings_histories={"QUALITY": history},
        )
        exits = [
            event for event in events if event.kind == "quality_compounder_exit"
        ]

        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0].date, dates[-1].date().isoformat())
        self.assertIn("fundamentals non_positive_ttm", exits[0].detail)


if __name__ == "__main__":
    unittest.main()
