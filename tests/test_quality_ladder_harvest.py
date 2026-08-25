import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_quality_ladder_harvest.py"
SPEC = importlib.util.spec_from_file_location("quality_ladder_harvest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(SCRIPT.parent))
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class QualityLadderHarvestTests(unittest.TestCase):
    def test_rungs_consume_the_entire_episode_reserve(self):
        self.assertAlmostEqual(sum(row[1] for row in MODULE.RUNGS), 1.0)
        self.assertEqual(
            [row[2] for row in MODULE.RUNGS],
            ["quality", "spy", "quality", "spy"],
        )

    def test_relative_excess_uses_wealth_not_return_ratio(self):
        result = MODULE.relative_excess(132, 100, 110, 100)
        self.assertAlmostEqual(result, 0.20)
        self.assertAlmostEqual(
            MODULE.relative_excess(112, 100, 110, 100),
            1.12 / 1.10 - 1.0,
        )

    def test_each_new_twenty_percent_band_harvests_once(self):
        self.assertEqual(MODULE.new_harvest_bands(0.19, 0), 0)
        self.assertEqual(MODULE.new_harvest_bands(0.20, 0), 1)
        self.assertEqual(MODULE.new_harvest_bands(0.45, 0), 2)
        self.assertEqual(MODULE.new_harvest_bands(0.45, 1), 1)
        self.assertEqual(MODULE.new_harvest_bands(0.45, 2), 0)


if __name__ == "__main__":
    unittest.main()
