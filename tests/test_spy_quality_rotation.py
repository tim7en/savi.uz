import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_spy_quality_rotation.py"
SPEC = importlib.util.spec_from_file_location("spy_quality_rotation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SpyQualityRotationTests(unittest.TestCase):
    def test_leverage_policies_follow_drawdown_bands(self):
        self.assertEqual(MODULE.leverage_for(-0.10, "constant_3x"), 3.0)
        self.assertEqual(MODULE.leverage_for(-0.39, "late_3_to_1"), 3.0)
        self.assertEqual(MODULE.leverage_for(-0.40, "late_3_to_1"), 1.0)
        self.assertEqual(MODULE.leverage_for(-0.19, "step_3_2_1"), 3.0)
        self.assertEqual(MODULE.leverage_for(-0.20, "step_3_2_1"), 2.0)
        self.assertEqual(MODULE.leverage_for(-0.40, "step_3_2_1"), 1.0)

    def test_reserve_ladder_uses_next_close_and_exhausts_four_equal_tranches(self):
        dates = pd.date_range("2020-01-02", periods=6, freq="B")
        prices = pd.DataFrame(
            {"SPY": [100.0, 89.0, 79.0, 69.0, 59.0, 59.0]}, index=dates
        )
        rates = pd.Series(0.0, index=dates)
        args = SimpleNamespace(
            initial=100_000.0,
            spy_share=0.80,
            spread=0.0,
            trade_bp=0.0,
            min_history_years=3.0,
            risk_per_stock=0.01,
            stock_tail_loss=0.79,
        )

        path, events = MODULE.simulate(
            prices, rates, args, "ladder_test", "step_3_2_1",
            staging=True, quality_at_40=False, harvest_share=0.0,
        )
        deployments = [event for event in events if event.kind == "deploy_spy"]

        self.assertEqual([event.date for event in deployments], [
            dates[2].date().isoformat(), dates[3].date().isoformat(),
            dates[4].date().isoformat(), dates[5].date().isoformat(),
        ])
        self.assertEqual([round(event.amount, 2) for event in deployments], [
            5_000.0, 5_000.0, 5_000.0, 5_000.0,
        ])
        self.assertAlmostEqual(path["reserve"].iloc[-1], 0.0)

    def test_portfolio_signal_reacts_to_levered_nav_not_spy_drawdown(self):
        dates = pd.date_range("2020-01-02", periods=3, freq="B")
        prices = pd.DataFrame({"SPY": [100.0, 95.0, 95.0]}, index=dates)
        rates = pd.Series(0.0, index=dates)
        args = SimpleNamespace(
            initial=100_000.0,
            spy_share=0.80,
            spread=0.0,
            trade_bp=0.0,
            min_history_years=3.0,
            risk_per_stock=0.01,
            stock_tail_loss=0.79,
        )

        _, spy_events = MODULE.simulate(
            prices, rates, args, "spy_signal", "step_3_2_1",
            staging=True, quality_at_40=False, harvest_share=0.0,
            signal_source="spy",
        )
        path, portfolio_events = MODULE.simulate(
            prices, rates, args, "portfolio_signal", "step_3_2_1",
            staging=True, quality_at_40=False, harvest_share=0.0,
            signal_source="portfolio",
        )

        self.assertFalse(any(event.kind == "deploy_spy" for event in spy_events))
        deployments = [
            event for event in portfolio_events if event.kind == "deploy_spy"
        ]
        self.assertEqual(len(deployments), 1)
        self.assertEqual(deployments[0].date, dates[2].date().isoformat())
        self.assertAlmostEqual(deployments[0].signal_drawdown, -0.12)
        self.assertAlmostEqual(path["reserve"].iloc[-1], 15_000.0)

    def test_spy_recovery_exit_waits_for_prior_high_and_sells_one_tenth(self):
        dates = pd.date_range("2020-01-02", periods=5, freq="B")
        prices = pd.DataFrame(
            {"SPY": [100.0, 80.0, 80.0, 100.0, 100.0],
             "QUALITY": [10.0] * 5},
            index=dates,
        )
        rates = pd.Series(0.0, index=dates)
        args = SimpleNamespace(
            initial=100_000.0,
            spy_share=0.80,
            spread=0.0,
            trade_bp=0.0,
            min_history_years=0.0,
            risk_per_stock=0.01,
            stock_tail_loss=0.79,
            max_quality_hold_years=5.0,
            trend_exit_days=200,
        )

        path, events = MODULE.simulate(
            prices, rates, args, "recovery_test", "step_3_2_1",
            staging=True, quality_at_40=True, harvest_share=0.0,
            signal_source="portfolio", exit_policy="spy_recovery_ladder",
        )
        purchases = [event for event in events if event.kind == "deploy_quality"]
        sales = [event for event in events if event.kind == "quality_sale"]

        self.assertEqual(len(purchases), 1)
        self.assertEqual(len(sales), 1)
        self.assertEqual(sales[0].date, dates[4].date().isoformat())
        self.assertAlmostEqual(sales[0].amount, purchases[0].amount * 0.10)
        self.assertGreater(path["quality_sleeve"].iloc[-1], 0.0)


if __name__ == "__main__":
    unittest.main()
