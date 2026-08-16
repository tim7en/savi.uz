"""Catalogue of macro series behind the rates/inflation/labour/liquidity dashboard.

Every entry names a FRED series, the theme it belongs to, and how much of its
revision history to keep. Vintage policy matters because these series are not
all the same kind of object:

* ``LATEST`` -- market prices that are never meaningfully restated.
* ``FIRST``  -- released once, then revised. The first print is what the market
  actually traded on, so it is kept alongside the current value.
* ``ALL``    -- the revision history *is* the data. The SEP dot plot is a single
  FRED series whose vintages are the successive FOMC meetings.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class VintagePolicy(str, Enum):
    LATEST = "latest"
    FIRST = "first"
    ALL = "all"


@dataclass(frozen=True)
class SeriesSpec:
    series_id: str
    group: str
    label: str
    vintages: VintagePolicy = VintagePolicy.FIRST


def _specs(group: str, entries: tuple[tuple[str, str], ...], vintages: VintagePolicy) -> tuple[SeriesSpec, ...]:
    return tuple(SeriesSpec(series_id, group, label, vintages) for series_id, label in entries)


POLICY_RATE = _specs(
    "policy_rate",
    (
        ("DFF", "Federal funds effective rate (daily, 1954-)"),
        ("EFFR", "Effective federal funds rate (NY Fed methodology, 2000-)"),
        ("DFEDTARU", "Fed funds target range, upper limit"),
        ("DFEDTARL", "Fed funds target range, lower limit"),
        ("DFEDTAR", "Fed funds target rate (discontinued 2008)"),
        ("IORB", "Interest on reserve balances"),
        ("SOFR", "Secured overnight financing rate"),
        ("OBFR", "Overnight bank funding rate"),
    ),
    VintagePolicy.LATEST,
)

MARKET_IMPLIED_PATH = _specs(
    "market_implied_path",
    (
        ("DGS1MO", "Treasury constant maturity, 1 month"),
        ("DGS3MO", "Treasury constant maturity, 3 month"),
        ("DGS6MO", "Treasury constant maturity, 6 month"),
        ("DGS1", "Treasury constant maturity, 1 year"),
        ("DGS2", "Treasury constant maturity, 2 year"),
        ("DGS3", "Treasury constant maturity, 3 year"),
        ("DGS5", "Treasury constant maturity, 5 year"),
        ("DGS7", "Treasury constant maturity, 7 year"),
        ("DGS10", "Treasury constant maturity, 10 year"),
        ("DGS20", "Treasury constant maturity, 20 year"),
        ("DGS30", "Treasury constant maturity, 30 year"),
        ("DFII5", "TIPS real yield, 5 year"),
        ("DFII10", "TIPS real yield, 10 year"),
        ("T10Y2Y", "10y minus 2y term spread"),
        ("T10Y3M", "10y minus 3m term spread"),
        ("THREEFF1", "Fitted instantaneous forward, 1 year ahead"),
        ("THREEFF2", "Fitted instantaneous forward, 2 years ahead"),
        ("THREEFF5", "Fitted instantaneous forward, 5 years ahead"),
        ("THREEFYTP10", "Kim-Wright 10y term premium"),
    ),
    VintagePolicy.LATEST,
)

#: SEP projections. The observation date is the projection's target year and the
#: vintage date is the FOMC meeting that published it, so ALL is mandatory here.
FED_SEP = _specs(
    "fed_sep",
    (
        ("FEDTARMD", "SEP fed funds rate, median"),
        ("FEDTARMDLR", "SEP fed funds rate, median longer run"),
        ("FEDTARCTM", "SEP fed funds rate, central tendency midpoint"),
        ("FEDTARCTH", "SEP fed funds rate, central tendency high"),
        ("FEDTARCTL", "SEP fed funds rate, central tendency low"),
        ("FEDTARRM", "SEP fed funds rate, range midpoint"),
        ("FEDTARRH", "SEP fed funds rate, range high"),
        ("FEDTARRL", "SEP fed funds rate, range low"),
        ("GDPC1MD", "SEP real GDP growth, median"),
        ("PCECTPIMD", "SEP PCE inflation, median"),
        ("JCXFEMD", "SEP core PCE inflation, median"),
        ("UNRATEMD", "SEP unemployment rate, median"),
    ),
    VintagePolicy.ALL,
)

INFLATION = _specs(
    "inflation",
    (
        ("CPIAUCSL", "CPI, all items, SA"),
        ("CPILFESL", "CPI, core, SA"),
        ("PCEPI", "PCE price index"),
        ("PCEPILFE", "PCE price index, core"),
        ("T5YIE", "5y breakeven inflation"),
        ("T10YIE", "10y breakeven inflation"),
        ("T5YIFR", "5y5y forward breakeven inflation"),
        ("MICH", "University of Michigan 1y inflation expectations"),
        ("EXPINF1YR", "Cleveland Fed 1y expected inflation"),
    ),
    VintagePolicy.FIRST,
)

LABOR = _specs(
    "labor",
    (
        ("PAYEMS", "Nonfarm payrolls"),
        ("UNRATE", "Unemployment rate"),
        ("U6RATE", "U-6 underemployment rate"),
        ("ICSA", "Initial jobless claims"),
        ("CCSA", "Continuing claims"),
        ("CIVPART", "Labour force participation rate"),
        ("JTSJOL", "JOLTS job openings"),
        ("JTSQUR", "JOLTS quits rate"),
        ("CES0500000003", "Average hourly earnings, private"),
        ("AWHAETP", "Average weekly hours, private"),
    ),
    VintagePolicy.FIRST,
)

CREDIT = _specs(
    "credit",
    (
        ("BAMLH0A0HYM2", "ICE BofA US high yield OAS"),
        ("BAMLH0A3HYC", "ICE BofA CCC and lower OAS"),
        ("BAMLC0A0CM", "ICE BofA US corporate OAS"),
        ("BAMLC0A4CBBB", "ICE BofA BBB corporate OAS"),
        ("BAMLH0A0HYM2EY", "ICE BofA US high yield effective yield"),
    ),
    VintagePolicy.LATEST,
)

BALANCE_SHEET = _specs(
    "balance_sheet",
    (
        ("WALCL", "Fed total assets"),
        ("WSHOSHO", "Securities held outright"),
        ("WRESBAL", "Reserve balances with Federal Reserve Banks"),
        ("WTREGEN", "Treasury General Account"),
        ("WDTGAL", "Deposits with Fed Reserve Banks, Treasury general account"),
        ("RRPONTSYD", "Overnight reverse repo, Treasury securities sold"),
        ("RRPONTSYAWARD", "Overnight reverse repo award rate"),
        ("H41RESPPALDKNWW", "Discount window primary credit"),
    ),
    VintagePolicy.FIRST,
)

FRED_CATALOG: tuple[SeriesSpec, ...] = (
    POLICY_RATE + MARKET_IMPLIED_PATH + FED_SEP + INFLATION + LABOR + CREDIT + BALANCE_SHEET
)

GROUPS: tuple[str, ...] = tuple(dict.fromkeys(spec.group for spec in FRED_CATALOG))

#: NY Fed publishes these with a revision flag and intraday distribution that
#: FRED flattens away.
NYFED_RATE_TYPES: tuple[tuple[str, str], ...] = (
    ("effr", "unsecured"),
    ("obfr", "unsecured"),
    ("sofr", "secured"),
    ("tgcr", "secured"),
    ("bgcr", "secured"),
)

#: Horizons for the market-implied policy path derived from the GSW curve.
FED_PATH_HORIZON_MONTHS: tuple[int, ...] = (3, 6, 9, 12, 18, 24, 36, 48, 60)


def catalog_for_groups(groups: tuple[str, ...] | None) -> tuple[SeriesSpec, ...]:
    if not groups:
        return FRED_CATALOG
    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise ValueError(f"Unknown group(s): {', '.join(sorted(unknown))}. Known: {', '.join(GROUPS)}")
    return tuple(spec for spec in FRED_CATALOG if spec.group in groups)
