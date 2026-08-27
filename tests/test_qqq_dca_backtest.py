import importlib.util
from pathlib import Path
import sys
import unittest

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_qqq_dca_backtest.py"
SPEC = importlib.util.spec_from_file_location("qqq_dca_backtest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(SCRIPT.parent))
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class QqqDcaBacktestTests(unittest.TestCase):
    def test_monthly_schedule_preserves_full_year_and_triennial_cash(self):
        index = pd.date_range("2020-01-02", "2023-12-29", freq="B")
        schedule = MODULE.monthly_equivalent_schedule(index, 12_000.0, 30_000.0)
        self.assertEqual(schedule.loc["2020"].sum(), 0.0)
        self.assertEqual(schedule.loc["2021"].sum(), 12_000.0)
        self.assertEqual(schedule.loc["2022"].sum(), 12_000.0)
        self.assertEqual(schedule.loc["2023"].sum(), 42_000.0)

    def test_dotcom_episode_uses_flow_adjusted_performance_not_wealth(self):
        index = pd.to_datetime([
            "2000-01-03", "2000-03-27", "2001-01-02", "2002-10-09", "2004-01-02"
        ])
        path = pd.DataFrame({
            "performance_index": [1.0, 1.2, 0.8, 0.2, 1.2],
            "wealth": [100.0, 120.0, 1_000.0, 900.0, 2_000.0],
        }, index=index)
        episode = MODULE.drawdown_episode(path)
        self.assertEqual(episode["peak_date"], "2000-03-27")
        self.assertEqual(episode["trough_date"], "2002-10-09")
        self.assertAlmostEqual(episode["max_drawdown"], 0.2 / 1.2 - 1.0)
        self.assertEqual(episode["recovery_date"], "2004-01-02")


if __name__ == "__main__":
    unittest.main()
