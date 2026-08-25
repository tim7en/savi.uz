import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_weekly_covered_call_study.py"
SPEC = importlib.util.spec_from_file_location("weekly_covered_calls", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(SCRIPT.parent))
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def trade(**changes):
    values = dict(
        symbol="SPY", issue_date="2020-01-03", expiration="2020-01-10",
        dte=7, target_delta=0.05, quoted_delta=0.05, spot=100.0,
        strike=105.0, bid=0.10, ask=0.12, expiry_spot=104.0,
        premium_yield=0.001, payoff_yield=0.0, underlying_return=0.04,
    )
    values.update(changes)
    return MODULE.CallTrade(**values)


class WeeklyCoveredCallTests(unittest.TestCase):
    def test_worthless_rate_and_upside_paid_are_separate(self):
        rows = [
            trade(),
            trade(issue_date="2020-01-10", expiration="2020-01-17",
                  expiry_spot=108.0, payoff_yield=0.03, underlying_return=0.08),
        ]
        summary = MODULE.summarize_trades(rows, assignment_cost_bp=0.0)
        self.assertEqual(summary["worthless"], 1)
        self.assertEqual(summary["assigned_or_itm"], 1)
        self.assertEqual(summary["worthless_rate"], 0.5)
        self.assertAlmostEqual(summary["premium_yield_sum"], 0.002)
        self.assertAlmostEqual(summary["payoff_yield_sum"], 0.03)

    def test_leveraged_injection_is_removed_from_eligible_notional(self):
        index = pd.to_datetime(["2020-01-03"])
        frame = pd.DataFrame({
            "test_wealth": [100.0], "test_performance_index": [1.0],
            "test_contribution": [0.0], "test_spy_weight": [0.9],
            "test_injection_core_spy": [30.0],
            "test_injection_weighted_leverage": [2.0],
        }, index=index)
        frame.index.name = "date"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "daily.csv"
            frame.to_csv(path)
            result = MODULE.strategy_frame(path, "test")
        self.assertAlmostEqual(result["eligible_fraction"].iloc[0], 0.60)

    def test_standard_contract_rounding_skips_a_ten_thousand_dollar_account(self):
        index = pd.to_datetime(["2020-01-03", "2020-01-10"])
        base = pd.DataFrame({
            "wealth": [10_000.0, 10_000.0], "performance_index": [1.0, 1.0],
            "contribution": [0.0, 0.0], "eligible_fraction": [1.0, 1.0],
        }, index=index)
        row = trade(spot=200.0, strike=210.0, bid=0.20,
                    premium_yield=0.001, payoff_yield=0.0)
        _, metrics = MODULE.apply_overlay(
            base, [row], pd.DataFrame(index=index), 2.0, round_contracts=True
        )
        self.assertEqual(metrics["calls_written"], 0)
        self.assertEqual(metrics["terminal_wealth"], 10_000.0)


if __name__ == "__main__":
    unittest.main()
