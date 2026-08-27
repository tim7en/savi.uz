import importlib.util
from pathlib import Path
import sys
import unittest

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_simple_investor_guide.py"
SPEC = importlib.util.spec_from_file_location("simple_investor_guide", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(SCRIPT.parent))
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DrawdownRecoveryGuideTests(unittest.TestCase):
    def test_only_first_threshold_crossing_per_episode_is_counted(self):
        dates = pd.date_range("2020-01-01", periods=9, freq="D")
        values = [100, 90, 85, 92, 88, 100, 101, 89, 102]
        events = MODULE.drawdown_events(dates, values, thresholds=(0.10, 0.20))
        ten = [row for row in events if row["threshold"] == 0.10]
        twenty = [row for row in events if row["threshold"] == 0.20]
        self.assertEqual(len(ten), 2)
        self.assertEqual(len(twenty), 0)
        self.assertEqual(ten[0]["entry_date"], "2020-01-02")
        self.assertEqual(ten[0]["old_high_recovery_date"], "2020-01-06")

    def test_fresh_capital_clock_is_separate_from_old_high(self):
        dates = pd.date_range("2020-01-01", periods=7, freq="D")
        values = [100, 80, 75, 88, 90, 99, 100]
        event = MODULE.drawdown_events(dates, values, thresholds=(0.20,))[0]
        self.assertEqual(event["plus_10_date"], "2020-01-04")
        self.assertEqual(event["days_to_plus_10"], 2)
        self.assertEqual(event["old_high_recovery_date"], "2020-01-07")
        self.assertEqual(event["days_to_old_high"], 5)
        self.assertAlmostEqual(event["further_loss_after_entry"], -0.0625)


if __name__ == "__main__":
    unittest.main()
