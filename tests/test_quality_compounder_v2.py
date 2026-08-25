import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_quality_compounder_v2.py"
SPEC = importlib.util.spec_from_file_location("quality_compounder_v2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(SCRIPT.parent))
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class QualityCompounderV2Tests(unittest.TestCase):
    def test_monthly_schedule_skips_initial_month(self):
        dates = pd.to_datetime(["2020-01-02", "2020-01-31", "2020-02-03", "2020-03-02"])
        result = MODULE.monthly_schedule(dates, 10_000)
        self.assertEqual(result.tolist(), [0, 0, 10_000, 10_000])

    def test_share_counts_are_not_available_until_ninety_days_after_period(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ABC_balance_sheet.json"
            path.write_text('{"data":{"quarterlyReports":[{"fiscalDateEnding":"2020-03-31","commonStockSharesOutstanding":"100"}]}}')
            history = MODULE.shares_history("ABC", Path(directory))
        self.assertEqual(history.index[0], pd.Timestamp("2020-06-29"))
        self.assertEqual(history.iloc[0], 100)

    def test_mega_seven_uses_market_cap_weights_not_price_rank(self):
        date = pd.Timestamp("2020-01-02")
        tickers = list("ABCDEFGH")
        adjusted = pd.DataFrame({ticker: [1.0] for ticker in tickers}, index=[date])
        raw = pd.DataFrame({ticker: [float(index + 1)] for index, ticker in enumerate(tickers)}, index=[date])
        shares = {ticker: pd.Series([1.0], index=[date]) for ticker in tickers}
        selected = MODULE.mega_seven(date, 0, adjusted, raw, shares)
        self.assertEqual([row["ticker"] for row in selected], list("HGFEDCB"))


if __name__ == "__main__":
    unittest.main()
