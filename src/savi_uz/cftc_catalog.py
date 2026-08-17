"""Catalogue of CFTC Commitments of Traders archives.

The CFTC publishes COT under three reporting regimes, each in a futures-only and
a futures-plus-options ("combined") flavour, so six archives in all. They do not
share a start date: the legacy breakdown runs back to 1986, the disaggregated
and financial-trader breakdowns only to 2006, because they did not exist before
then. Asking for 2000 therefore returns legacy data alone for the first six
years, which is a property of the source, not a gap in the download.

Each archive is served as one ZIP per year, plus -- for everything except the
legacy futures-only file -- a single multi-year bundle covering the back
history. Fetching the bundle for the early years and annual files thereafter
keeps a full pull to a few dozen requests instead of a few hundred.
"""

from __future__ import annotations

from dataclasses import dataclass

ARCHIVE_BASE_URL = "https://www.cftc.gov/files/dea/history"

#: Column counts are stable across every era of each archive; a file that
#: arrives with a different width has changed shape and is rejected rather than
#: silently loaded into a mismatched table.
LEGACY_COLUMNS = 129
DISAGGREGATED_COLUMNS = 191
FINANCIAL_COLUMNS = 87


@dataclass(frozen=True)
class ArchiveFile:
    """One ZIP to fetch, and the years it is expected to contain."""

    url: str
    first_year: int
    last_year: int

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class ReportSpec:
    """One COT reporting regime and where its history lives."""

    key: str
    table: str
    label: str
    #: Normalised name of the column holding the ISO report date. The legacy
    #: files call it "As of Date in Form YYYY-MM-DD", the newer ones
    #: "Report_Date_as_YYYY-MM-DD".
    date_column: str
    contract_column: str
    columns: int
    annual_template: str
    first_annual_year: int
    first_year: int
    bundle_template: str | None = None
    bundle_first_year: int | None = None
    bundle_last_year: int | None = None
    notes: str = ""

    def annual_url(self, year: int) -> str:
        return f"{ARCHIVE_BASE_URL}/{self.annual_template.format(year=year)}"

    def bundle_url(self) -> str | None:
        if self.bundle_template is None:
            return None
        return f"{ARCHIVE_BASE_URL}/{self.bundle_template}"


LEGACY_DATE_COLUMN = "as_of_date_in_form_yyyy_mm_dd"
MODERN_DATE_COLUMN = "report_date_as_yyyy_mm_dd"
CONTRACT_COLUMN = "cftc_contract_market_code"

# The current year's annual file is rewritten by the CFTC every Friday, so
# ``--refresh`` over that one year is the whole incremental update; there is no
# hardcoded newest year, and a year the CFTC has not published yet simply 404s
# and is reported as a skipped archive.

REPORTS: tuple[ReportSpec, ...] = (
    ReportSpec(
        key="legacy_futures",
        table="cot_legacy_futures",
        label="Legacy, futures only",
        date_column=LEGACY_DATE_COLUMN,
        contract_column=CONTRACT_COLUMN,
        columns=LEGACY_COLUMNS,
        annual_template="deacot{year}.zip",
        first_annual_year=1986,
        first_year=1986,
        notes="Non-commercial / commercial / non-reportable. Monthly then biweekly "
              "before 1992 and only weekly from 2000, so early years hold far fewer dates.",
    ),
    ReportSpec(
        key="legacy_combined",
        table="cot_legacy_combined",
        label="Legacy, futures and options combined",
        date_column=LEGACY_DATE_COLUMN,
        contract_column=CONTRACT_COLUMN,
        columns=LEGACY_COLUMNS,
        annual_template="deahistfo{year}.zip",
        first_annual_year=2017,
        first_year=1995,
        bundle_template="deahistfo_1995_2016.zip",
        bundle_first_year=1995,
        bundle_last_year=2016,
        notes="Options are delta-weighted into futures equivalents; starts 1995-03-21.",
    ),
    ReportSpec(
        key="disagg_futures",
        table="cot_disagg_futures",
        label="Disaggregated, futures only",
        date_column=MODERN_DATE_COLUMN,
        contract_column=CONTRACT_COLUMN,
        columns=DISAGGREGATED_COLUMNS,
        annual_template="fut_disagg_txt_{year}.zip",
        first_annual_year=2017,
        first_year=2006,
        bundle_template="fut_disagg_txt_hist_2006_2016.zip",
        bundle_first_year=2006,
        bundle_last_year=2016,
        notes="Producer/merchant, swap dealer, managed money, other reportable. "
              "Physical commodities only; starts 2006-06-13.",
    ),
    ReportSpec(
        key="disagg_combined",
        table="cot_disagg_combined",
        label="Disaggregated, futures and options combined",
        date_column=MODERN_DATE_COLUMN,
        contract_column=CONTRACT_COLUMN,
        columns=DISAGGREGATED_COLUMNS,
        annual_template="com_disagg_txt_{year}.zip",
        first_annual_year=2017,
        first_year=2006,
        bundle_template="com_disagg_txt_hist_2006_2016.zip",
        bundle_first_year=2006,
        bundle_last_year=2016,
        notes="Physical commodities only; starts 2006-06-13.",
    ),
    ReportSpec(
        key="tff_futures",
        table="cot_tff_futures",
        label="Traders in Financial Futures, futures only",
        date_column=MODERN_DATE_COLUMN,
        contract_column=CONTRACT_COLUMN,
        columns=FINANCIAL_COLUMNS,
        annual_template="fut_fin_txt_{year}.zip",
        first_annual_year=2017,
        first_year=2006,
        bundle_template="fin_fut_txt_2006_2016.zip",
        bundle_first_year=2006,
        bundle_last_year=2016,
        notes="Dealer/intermediary, asset manager, leveraged funds, other reportable. "
              "This is the file that carries positioning in rates, equity index and FX.",
    ),
    ReportSpec(
        key="tff_combined",
        table="cot_tff_combined",
        label="Traders in Financial Futures, futures and options combined",
        date_column=MODERN_DATE_COLUMN,
        contract_column=CONTRACT_COLUMN,
        columns=FINANCIAL_COLUMNS,
        annual_template="com_fin_txt_{year}.zip",
        first_annual_year=2017,
        first_year=2006,
        bundle_template="fin_com_txt_2006_2016.zip",
        bundle_first_year=2006,
        bundle_last_year=2016,
        notes="Financial contracts, options folded in.",
    ),
)

REPORT_KEYS: tuple[str, ...] = tuple(spec.key for spec in REPORTS)

REPORTS_BY_KEY: dict[str, ReportSpec] = {spec.key: spec for spec in REPORTS}

#: Identity and code columns, kept TEXT so that leading zeros survive --
#: contract market code ``001602`` and commodity code ``001`` are labels, not
#: quantities, and ``int()`` would quietly destroy them.
TEXT_COLUMNS: frozenset[str] = frozenset(
    {
        "market_and_exchange_names",
        "as_of_date_in_form_yymmdd",
        LEGACY_DATE_COLUMN,
        MODERN_DATE_COLUMN,
        "cftc_contract_market_code",
        "cftc_contract_market_code_quotes",
        "cftc_market_code",
        "cftc_market_code_quotes",
        "cftc_market_code_in_initials",
        "cftc_market_code_in_initials_quotes",
        "cftc_region_code",
        "cftc_region_code_quotes",
        "cftc_commodity_code",
        "cftc_commodity_code_quotes",
        "cftc_subgroup_code",
        "cftc_commodity_subgroup_name",
        "cftc_industry_group_name",
        "commodity_name",
        "contract_units",
        "futonly_or_combined",
    }
)


def reports_for_keys(keys: tuple[str, ...] | None) -> tuple[ReportSpec, ...]:
    if not keys:
        return REPORTS
    unknown = set(keys) - set(REPORT_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown report(s): {', '.join(sorted(unknown))}. Known: {', '.join(REPORT_KEYS)}"
        )
    return tuple(spec for spec in REPORTS if spec.key in keys)


def archive_files(spec: ReportSpec, start_year: int, end_year: int) -> tuple[ArchiveFile, ...]:
    """The smallest set of ZIPs covering ``start_year``..``end_year``.

    The multi-year bundle is preferred for the back history because it is one
    request instead of eleven or twenty; annual files take over from
    ``first_annual_year``. Requested years outside the archive's own coverage
    are dropped rather than fetched and 404'd.
    """
    if start_year > end_year:
        raise ValueError(f"start_year {start_year} is after end_year {end_year}")

    first = max(start_year, spec.first_year)
    last = end_year
    if first > last:
        return ()

    files: list[ArchiveFile] = []
    bundle_url = spec.bundle_url()
    if (
        bundle_url is not None
        and spec.bundle_first_year is not None
        and spec.bundle_last_year is not None
        and first <= spec.bundle_last_year
    ):
        files.append(ArchiveFile(bundle_url, spec.bundle_first_year, spec.bundle_last_year))

    annual_start = max(first, spec.first_annual_year)
    for year in range(annual_start, last + 1):
        files.append(ArchiveFile(spec.annual_url(year), year, year))
    return tuple(files)
