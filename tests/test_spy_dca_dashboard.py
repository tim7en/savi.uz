import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_spy_dca_dashboard.py"
SPEC = importlib.util.spec_from_file_location("spy_dca_dashboard", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(SCRIPT.parent))
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DcaDecisionTests(unittest.TestCase):
    def test_new_account_at_high_is_one_x_even_when_other_gates_allow_three_x(self):
        leverage, gates = MODULE.dca_leverage(0.0, 20.0, 0.50)
        self.assertEqual(leverage, 1.0)
        self.assertEqual(gates["nav"].ceiling, 1.0)

    def test_moderate_drawdown_allows_two_x_only_if_both_risk_caps_allow_it(self):
        self.assertEqual(MODULE.dca_leverage(-0.15, 30.0, 0.50)[0], 2.0)
        self.assertEqual(MODULE.dca_leverage(-0.15, 38.0, 0.50)[0], 1.0)
        self.assertEqual(MODULE.dca_leverage(-0.15, 20.0, 0.95)[0], 1.0)

    def test_deep_drawdown_does_not_override_valuation_or_volatility(self):
        self.assertEqual(MODULE.dca_leverage(-0.25, 20.0, 0.50)[0], 3.0)
        self.assertEqual(MODULE.dca_leverage(-0.25, 30.0, 0.50)[0], 2.0)
        self.assertEqual(MODULE.dca_leverage(-0.25, 20.0, 0.95)[0], 1.0)

    def test_threshold_boundaries_are_conservative(self):
        self.assertEqual(MODULE.nav_dca_signal(-0.10).ceiling, 2.0)
        self.assertEqual(MODULE.nav_dca_signal(-0.20).ceiling, 3.0)
        self.assertEqual(MODULE.cape_signal(35.0).ceiling, 2.0)
        self.assertEqual(MODULE.vix_signal(0.90).ceiling, 1.0)


if __name__ == "__main__":
    unittest.main()
