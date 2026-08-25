import importlib.util
from pathlib import Path
import sys
import unittest

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_cape_leverage_study.py"
SPEC = importlib.util.spec_from_file_location("cape_leverage", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(SCRIPT.parent))
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CapeLeverageStudyTests(unittest.TestCase):
    def test_cape_is_available_only_next_month(self):
        cape = pd.Series([25.0, 30.0], index=pd.to_datetime(["2020-01-01", "2020-02-01"]))
        days = pd.to_datetime(["2020-01-31", "2020-02-03", "2020-03-02"])
        known = MODULE.known_cape_daily(cape, days)
        self.assertTrue(pd.isna(known.iloc[0]))
        self.assertEqual(known.iloc[1], 25.0)
        self.assertEqual(known.iloc[2], 30.0)

    def test_fixed_cap_has_three_bands(self):
        cape = pd.Series([24.9, 25.0, 34.9, 35.0])
        self.assertEqual(MODULE.fixed_cap(cape, 25, 35).tolist(), [3, 2, 2, 1])


if __name__ == "__main__":
    unittest.main()
