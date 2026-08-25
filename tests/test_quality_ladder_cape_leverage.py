import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_quality_ladder_cape_leverage.py"
SPEC = importlib.util.spec_from_file_location("quality_ladder_cape_leverage", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(SCRIPT.parent))
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class QualityLadderCapeLeverageTests(unittest.TestCase):
    def test_vix_brake_and_reversal_are_directionally_opposite(self):
        cap = MODULE.ladder.vix_leverage_cap
        self.assertEqual([cap(3.0, rank, "brake") for rank in (0.50, 0.80, 0.95)], [3.0, 2.0, 1.0])
        self.assertEqual([cap(3.0, rank, "reverse") for rank in (0.50, 0.80, 0.95)], [1.0, 2.0, 3.0])
        self.assertEqual(cap(2.0, 0.50, "brake"), 2.0)

    def test_cape_leverage_boundaries(self):
        cape = pd.Series([24.99, 25.0, 35.0, 35.01])
        self.assertEqual(MODULE.cape_leverage(cape).tolist(), [3.0, 2.0, 2.0, 1.0])

    def test_contribution_schedule_matches_annual_plus_triennial_rule(self):
        index = pd.to_datetime([
            "2020-01-02", "2021-01-04", "2022-01-03", "2023-01-03", "2024-01-02"
        ])
        schedule = MODULE.contribution_schedule(index, 10_000, 30_000)
        self.assertEqual(schedule.tolist(), [0.0, 10_000.0, 10_000.0, 40_000.0, 10_000.0])

    def test_only_the_core_receives_cape_leverage(self):
        index = pd.to_datetime(["2020-01-02", "2020-01-03"])
        prices = pd.DataFrame({"SPY": [100.0, 110.0]}, index=index)
        args = SimpleNamespace(
            initial=100.0, spy_share=0.80, trade_bp=0.0, spread=0.0,
            relative_step=0.20, harvest_share=0.05, cape_excessive=35.0,
        )
        path, _, _ = MODULE.ladder.simulate(
            prices, pd.DataFrame(index=index), {},
            pd.Series(0.0, index=index), pd.Series(20.0, index=index), args,
            name="test", rungs_enabled=False, quality_enabled=False,
            harvest_enabled=False, cape_enabled=False,
            contribution_series=pd.Series(0.0, index=index),
            core_leverage=pd.Series(3.0, index=index),
        )
        # 80 of core equity earns 3 x 10%; the 20 Treasury dollars are unlevered.
        self.assertAlmostEqual(path["wealth"].iloc[-1], 124.0)
        self.assertAlmostEqual(path["gross_exposure"].iloc[-1], 312.0 / 124.0)

    def test_nav_brake_uses_prior_close_and_holds_one_x(self):
        index = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
        prices = pd.DataFrame({"SPY": [100.0, 95.0, 95.0]}, index=index)
        args = SimpleNamespace(
            initial=100.0, spy_share=0.80, trade_bp=0.0, spread=0.0,
            relative_step=0.20, harvest_share=0.05, cape_excessive=35.0,
        )
        path, events, _ = MODULE.ladder.simulate(
            prices, pd.DataFrame(index=index), {},
            pd.Series(0.0, index=index), pd.Series(20.0, index=index), args,
            name="test", rungs_enabled=False, quality_enabled=False,
            harvest_enabled=False, cape_enabled=False,
            contribution_series=pd.Series(0.0, index=index),
            core_leverage=pd.Series(3.0, index=index),
            nav_deleverage_at=0.10,
        )
        self.assertEqual(path["core_leverage"].tolist(), [3.0, 3.0, 1.0])
        self.assertEqual(
            [event.kind for event in events if event.kind == "nav_deleverage"],
            ["nav_deleverage"],
        )

    def test_fresh_contribution_uses_cape_leverage_during_nav_brake(self):
        index = pd.to_datetime([
            "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"
        ])
        prices = pd.DataFrame({"SPY": [100.0, 95.0, 95.0, 104.5]}, index=index)
        args = SimpleNamespace(
            initial=100.0, spy_share=0.80, trade_bp=0.0, spread=0.0,
            relative_step=0.20, harvest_share=0.05, cape_excessive=35.0,
        )
        contributions = pd.Series([0.0, 0.0, 100.0, 0.0], index=index)
        path, _, _ = MODULE.ladder.simulate(
            prices, pd.DataFrame(index=index), {},
            pd.Series(0.0, index=index), pd.Series(20.0, index=index), args,
            name="test", rungs_enabled=False, quality_enabled=False,
            harvest_enabled=False, cape_enabled=False,
            contribution_series=contributions,
            core_leverage=pd.Series(3.0, index=index),
            nav_deleverage_at=0.10,
            fresh_capital_cape_leverage=True,
        )
        self.assertAlmostEqual(path["fresh_core_spy"].iloc[2], 80.0)
        self.assertEqual(path["legacy_core_leverage"].iloc[2], 1.0)
        self.assertEqual(path["fresh_core_leverage"].iloc[2], 3.0)
        # On the next 10% SPY rise, legacy equity earns 10% and fresh equity 30%.
        self.assertAlmostEqual(path["legacy_core_spy"].iloc[3], 74.8)
        self.assertAlmostEqual(path["fresh_core_spy"].iloc[3], 104.0)

    def test_dual_guard_leverages_only_drawdown_injection_and_resets_at_recovery(self):
        index = pd.to_datetime([
            "2020-01-02", "2020-01-03", "2020-01-06",
            "2020-01-07", "2020-01-08"
        ])
        prices = pd.DataFrame(
            {"SPY": [100.0, 85.0, 85.0, 100.0, 100.0]}, index=index
        )
        args = SimpleNamespace(
            initial=100.0, spy_share=0.80, trade_bp=0.0, spread=0.0,
            relative_step=0.20, harvest_share=0.05, cape_excessive=35.0,
        )
        contributions = pd.Series([0.0, 0.0, 100.0, 0.0, 0.0], index=index)
        path, events, _ = MODULE.ladder.simulate(
            prices, pd.DataFrame(index=index), {},
            pd.Series(0.0, index=index), pd.Series(20.0, index=index), args,
            name="test", rungs_enabled=False, quality_enabled=False,
            harvest_enabled=False, cape_enabled=False,
            contribution_series=contributions,
            core_leverage=pd.Series(1.0, index=index),
            injection_leverage=pd.Series(3.0, index=index),
            injection_nav_drawdown=0.10,
        )
        self.assertAlmostEqual(path["injection_core_spy"].iloc[2], 80.0)
        self.assertAlmostEqual(path["injection_gross_exposure"].iloc[2], 240.0)
        self.assertEqual(path["base_core_leverage"].tolist(), [1.0] * 5)
        self.assertEqual(
            [event.kind for event in events if event.kind == "injection_leverage_reset"],
            ["injection_leverage_reset"],
        )
        self.assertAlmostEqual(path["injection_core_spy"].iloc[-1], 0.0)

    def test_vix_brake_delays_and_then_restores_cape_leverage(self):
        index = pd.to_datetime([
            "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"
        ])
        prices = pd.DataFrame({"SPY": [100.0, 85.0, 85.0, 85.0]}, index=index)
        args = SimpleNamespace(
            initial=100.0, spy_share=0.80, trade_bp=0.0, spread=0.0,
            relative_step=0.20, harvest_share=0.05, cape_excessive=35.0,
        )
        path, _, _ = MODULE.ladder.simulate(
            prices, pd.DataFrame(index=index), {},
            pd.Series(0.0, index=index), pd.Series(20.0, index=index), args,
            name="test", rungs_enabled=False, quality_enabled=False,
            harvest_enabled=False, cape_enabled=False,
            contribution_series=pd.Series([0.0, 0.0, 100.0, 0.0], index=index),
            core_leverage=pd.Series(1.0, index=index),
            injection_leverage=pd.Series(3.0, index=index),
            injection_nav_drawdown=0.10,
            injection_vix_percentile=pd.Series([0.95, 0.95, 0.95, 0.50], index=index),
            injection_vix_mode="brake",
        )
        self.assertEqual(path["injection_weighted_leverage"].iloc[2], 1.0)
        self.assertEqual(path["injection_weighted_leverage"].iloc[3], 3.0)


if __name__ == "__main__":
    unittest.main()
