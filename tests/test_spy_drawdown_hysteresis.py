import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_spy_drawdown_hysteresis.py"
SPEC = importlib.util.spec_from_file_location("spy_drawdown_hysteresis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DrawdownHysteresisTests(unittest.TestCase):
    def test_rungs_stay_engaged_until_old_high(self):
        dates = pd.date_range("2020-01-01", periods=9, freq="D")
        signal = pd.Series([100, 89, 95, 100, 89, 49, 60, 100, 101], index=dates)

        rungs, _, episodes = MODULE.drawdown_rungs(signal)

        self.assertEqual(rungs.tolist(), [0, 1, 1, 0, 1, 2, 2, 0, 0])
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0].entry, "2020-01-02")
        self.assertEqual(episodes[0].recovery, "2020-01-04")
        self.assertEqual(episodes[1].deepest_rung_date, "2020-01-06")
        self.assertEqual(episodes[1].recovery, "2020-01-08")

    def test_close_signal_applies_on_following_session(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        returns = pd.Series([0.0, -0.10, 0.10], index=dates)
        targets = pd.Series([1.0, 2.0, 2.0], index=dates)
        rates = pd.Series(0.0, index=dates)

        path = MODULE.backtest(returns, targets, rates, spread=0.0)

        self.assertEqual(path["applied_leverage"].tolist(), [1.0, 1.0, 2.0])
        self.assertAlmostEqual(path["wealth"].iloc[-1], 108.0)

    def test_financing_accrues_over_weekend(self):
        dates = pd.to_datetime(["2020-01-03", "2020-01-06"])
        returns = pd.Series([0.0, 0.0], index=dates)
        targets = pd.Series(2.0, index=dates)
        rates = pd.Series(0.365, index=dates)

        path = MODULE.backtest(returns, targets, rates, spread=0.0)

        self.assertAlmostEqual(path["financing_return"].iloc[-1], 0.003)
        self.assertAlmostEqual(path["wealth"].iloc[-1], 99.7)


if __name__ == "__main__":
    unittest.main()
