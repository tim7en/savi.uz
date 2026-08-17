"""Catalogue of earnings and valuation sources.

Four sources with genuinely different reach, which is the thing to keep straight
when combining them:

- **Shiller** is the only free series with S&P 500 earnings, dividends and CAPE
  back to 1871, but it is updated by hand and runs a year or two behind.
- **SEC XBRL** is authoritative per company, but XBRL was phased in from 2009
  and has no coverage before then -- there is no route to company fundamentals
  back to 2000 through this API.
- **Yahoo** carries daily index prices back to 1927 and is current, so it
  bridges the recent end that Shiller has not filled and that FRED's ``SP500``
  cannot reach (a rolling ten-year licence window).
- **Alpha Vantage** has per-company analyst estimates and reported-vs-expected
  EPS, on a free tier tight enough that the universe has to be a hand-picked
  list rather than the whole market.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Shiller's workbook moved from Yale to shillerdata.com and the two copies
#: drift apart by up to a year. Both are tried and the fresher one wins, so a
#: stale mirror degrades the end date rather than the whole download.
SHILLER_URLS: tuple[str, ...] = (
    "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/ie_data.xls",
    "http://www.econ.yale.edu/~shiller/data/ie_data.xls",
)

SHILLER_SHEET = "Data"

#: The sheet carries four stacked header rows; data starts here (0-indexed).
SHILLER_FIRST_DATA_ROW = 8

#: Column index in the Data sheet -> stored name. Indices 13 and 15 are spacer
#: columns and index 5 is Shiller's own decimal date, all dropped.
SHILLER_COLUMNS: dict[int, str] = {
    1: "sp500_price",
    2: "dividend",
    3: "earnings",
    4: "cpi",
    6: "long_rate_gs10",
    7: "real_price",
    8: "real_dividend",
    9: "real_total_return_price",
    10: "real_earnings",
    11: "real_tr_scaled_earnings",
    12: "cape",
    14: "tr_cape",
    16: "excess_cape_yield",
    17: "monthly_total_bond_return",
    18: "real_total_bond_return",
    19: "ten_year_annualized_stock_real_return",
    20: "ten_year_annualized_bond_real_return",
    21: "real_ten_year_excess_annualized_return",
}


@dataclass(frozen=True)
class ConceptSpec:
    """One us-gaap concept to pull from the SEC's ``frames`` endpoint.

    ``instant`` marks balance-sheet concepts, whose frame identifiers carry a
    trailing ``I`` (``CY2023Q1I``) because they are point-in-time rather than
    measured over the period.
    """

    tag: str
    #: Unit as it appears in the URL path; the API echoes it back with a slash
    #: (``USD-per-shares`` in, ``USD/shares`` out).
    unit: str
    instant: bool
    label: str
    taxonomy: str = "us-gaap"

    def frame(self, period: str) -> str:
        return f"{period}I" if self.instant else period

    def url(self, base_url: str, period: str) -> str:
        return f"{base_url}/api/xbrl/frames/{self.taxonomy}/{self.tag}/{self.unit}/{self.frame(period)}.json"


#: Revenue is split across two tags: filers moved from ``Revenues`` to the
#: ASC 606 tag from 2018, and neither covers the whole period on its own.
SEC_CONCEPTS: tuple[ConceptSpec, ...] = (
    ConceptSpec("Revenues", "USD", False, "Revenues (pre-ASC 606 tag)"),
    ConceptSpec(
        "RevenueFromContractWithCustomerExcludingAssessedTax", "USD", False, "Revenue (ASC 606 tag)"
    ),
    ConceptSpec("NetIncomeLoss", "USD", False, "Net income (loss)"),
    ConceptSpec("OperatingIncomeLoss", "USD", False, "Operating income (loss)"),
    ConceptSpec("EarningsPerShareBasic", "USD-per-shares", False, "EPS, basic"),
    ConceptSpec("EarningsPerShareDiluted", "USD-per-shares", False, "EPS, diluted"),
    ConceptSpec(
        "WeightedAverageNumberOfDilutedSharesOutstanding", "shares", False, "Diluted share count"
    ),
    ConceptSpec("Assets", "USD", True, "Total assets"),
    ConceptSpec("Liabilities", "USD", True, "Total liabilities"),
    ConceptSpec("StockholdersEquity", "USD", True, "Stockholders' equity"),
)

#: XBRL was phased in over 2009-2011: the 2009 frames hold a few hundred filers
#: against four thousand from 2012. Earlier years exist as frame identifiers but
#: return nothing usable, so the download starts here regardless of --start.
SEC_FIRST_YEAR = 2009

#: Below this many facts a frame is a phase-in artefact rather than a quarter of
#: market data, and is recorded as sparse in the fetch log.
SEC_SPARSE_FRAME_FACTS = 100

SEC_BASE_URL = "https://data.sec.gov"

#: The SEC's published fair-access ceiling.
SEC_MAX_REQUESTS_PER_SECOND = 10

#: Yahoo tickers for the index history that bridges Shiller's stale tail.
INDEX_TICKERS: tuple[tuple[str, str], ...] = (
    ("^GSPC", "S&P 500"),
    ("^SP500TR", "S&P 500 total return"),
    ("^NDX", "Nasdaq 100"),
    ("^RUT", "Russell 2000"),
)

#: Alpha Vantage's free tier allows ~25 requests a day, so the estimate universe
#: is deliberately small and overridable with --tickers.
DEFAULT_ESTIMATE_TICKERS: tuple[str, ...] = (
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "AVGO", "TSLA", "JPM",
)

SOURCE_KEYS: tuple[str, ...] = ("shiller", "sec", "index", "estimates")


def quarters(start_year: int, end_year: int) -> tuple[str, ...]:
    """``CY2009Q1``-style frame identifiers, inclusive of both years."""
    if start_year > end_year:
        raise ValueError(f"start_year {start_year} is after end_year {end_year}")
    return tuple(
        f"CY{year}Q{quarter}"
        for year in range(start_year, end_year + 1)
        for quarter in (1, 2, 3, 4)
    )
