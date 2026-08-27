import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_contribution_quality_strategy.py"
SPEC = importlib.util.spec_from_file_location("contribution_quality", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(SCRIPT.parent))
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ContributionQualityTests(unittest.TestCase):
    def test_contribution_schedule_adds_thirty_thousand_every_third_year(self):
        dates = pd.to_datetime([f"{year}-01-02" for year in range(2020, 2027)])
        schedule = MODULE.contribution_schedule(dates, 10_000, 30_000)
        self.assertEqual(schedule.tolist(), [0, 10_000, 10_000, 40_000, 10_000, 10_000, 40_000])

    def test_leverage_hysteresis(self):
        self.assertEqual(MODULE.leverage_state(3.0, -0.16), 2.0)
        self.assertEqual(MODULE.leverage_state(2.0, -0.31), 1.0)
        self.assertEqual(MODULE.leverage_state(1.0, -0.11), 1.0)
        self.assertEqual(MODULE.leverage_state(1.0, -0.09), 2.0)
        self.assertEqual(MODULE.leverage_state(2.0, -0.01), 2.0)
        self.assertEqual(MODULE.leverage_state(2.0, 0.0), 3.0)

    def test_spy_cash_flow_unitisation_ignores_contribution_jump(self):
        dates = pd.to_datetime(["2020-01-02", "2021-01-04", "2022-01-03"])
        prices = pd.DataFrame({"SPY": [100.0, 100.0, 100.0]}, index=dates)
        args = SimpleNamespace(
            initial=10_000.0, annual_contribution=10_000.0,
            triennial_contribution=30_000.0,
        )
        path = MODULE.simulate_spy(prices, pd.Series(0.0, index=dates), args)
        self.assertEqual(path["wealth"].tolist(), [10_000, 20_000, 30_000])
        self.assertEqual(path["performance_index"].tolist(), [1.0, 1.0, 1.0])

    def test_xirr_of_one_year_double_is_one_hundred_percent(self):
        flows = [(pd.Timestamp("2020-01-01"), -100.0),
                 (pd.Timestamp("2021-01-01"), 200.0)]
        # 2020 has 366 actual days under the ACT/365.2425 convention.
        self.assertAlmostEqual(MODULE.xirr(flows), 0.997133, places=5)


if __name__ == "__main__":
    unittest.main()
