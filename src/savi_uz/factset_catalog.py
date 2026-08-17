"""Catalogue for the FactSet Earnings Insight weekly report.

This is the one source in the stack with no API: John Butters' report is a PDF
published most Fridays, and the numbers on its first page -- forward 12-month
P/E, blended earnings growth, surprise rates -- are the free market's reference
for forward S&P 500 earnings expectations.

Two constraints shape everything here:

- **The archive starts 2017-02-03.** Nothing earlier is hosted, so this source
  cannot reach 2000 whatever start year is asked for.
- **The wording drifts.** Over nine years FactSet has renamed the section from
  "Earnings Growth" to "Earnings Decline" in down quarters, moved between "beat
  the mean EPS estimate" and "reported a positive EPS surprise", and dropped
  "S&P 500" before "companies". Every field therefore carries alternative
  patterns, and a field that no pattern matches is recorded as missing rather
  than guessed -- some editions genuinely omit a line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

BASE_URL = (
    "https://advantage.factset.com/hubfs/Website/Resources%20Section"
    "/Research%20Desk/Earnings%20Insight"
)

#: Oldest edition still hosted, confirmed by probing quarterly back to 2010.
ARCHIVE_FIRST_DATE = date(2017, 2, 3)

#: The current edition is also served from a stable alias, useful for a quick
#: "is there anything newer" check without guessing a date.
LATEST_URL = "https://www.factset.com/earningsinsight"

#: Most editions are the Friday file. Some weeks publish on Thursday, and some
#: carry an ``A`` suffix (a re-issue); without these fallbacks roughly a quarter
#: of the weeks that do exist look like non-publication.
FILENAME_TEMPLATE = "EarningsInsight_{stamp}{suffix}.pdf"


@dataclass(frozen=True)
class Candidate:
    """One URL to try for a given week, with the date it would represent."""

    published: date
    url: str


def candidate_urls(friday: date) -> tuple[Candidate, ...]:
    """URLs to try for the week ending ``friday``, best guess first."""
    thursday = friday - timedelta(days=1)
    return tuple(
        Candidate(day, f"{BASE_URL}/{FILENAME_TEMPLATE.format(stamp=day.strftime('%m%d%y'), suffix=suffix)}")
        for day, suffix in ((friday, ""), (friday, "A"), (thursday, ""), (thursday, "A"))
    )


def fridays(start: date, end: date) -> tuple[date, ...]:
    """Every Friday in the range, clamped to the archive's first edition."""
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    cursor = max(start, ARCHIVE_FIRST_DATE)
    cursor += timedelta(days=(4 - cursor.weekday()) % 7)
    weeks = []
    while cursor <= end:
        weeks.append(cursor)
        cursor += timedelta(days=7)
    return tuple(weeks)


@dataclass(frozen=True)
class FieldSpec:
    """One number to lift off page 1, and every wording it has appeared under."""

    name: str
    patterns: tuple[str, ...]
    kind: str = "float"
    #: Core fields exist in every era; a miss means the parser has drifted.
    #: Non-core fields are genuinely absent from some editions.
    core: bool = False

    def search(self, text: str) -> str | None:
        for pattern in self.patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None


_NUM = r"(-?\d+(?:\.\d+)?)"
_PCT = r"([\d.]+)"

FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        "quarter",
        (
            r"For (Q\d 20\d{2}) \(with",
            r"reporting actual results for (Q\d 20\d{2})\)",
            # Out of reporting season there is no scorecard line, but the
            # earnings section still names the quarter it is forecasting.
            r"Earnings (?:Growth|Decline): For (Q\d 20\d{2})",
        ),
        kind="text",
        core=True,
    ),
    FieldSpec(
        "pct_reported",
        (r"with " + _PCT + r"% of (?:the companies in the S&P 500|S&P 500 companies) report",),
    ),
    FieldSpec(
        "pct_positive_eps",
        (
            _PCT + r"% of S&P 500 companies (?:have|has) "
            r"(?:reported a positive EPS surprise|beat the mean EPS estimate)",
        ),
    ),
    FieldSpec(
        "pct_positive_revenue",
        (
            _PCT + r"% of (?:S&P 500 )?companies (?:have|has) "
            r"(?:reported a positive revenue surprise|beat the mean sales estimate)",
        ),
    ),
    # "growth rate ... is" flips to "decline ... is" in negative quarters; the
    # sign is already in the number, so both map to one field.
    #
    # Blended and estimated are deliberately separate columns, not one merged
    # "growth" field. Blended mixes actuals reported so far with estimates for
    # the rest and only exists once a quarter is under way; estimated is the
    # pure forecast the report carries between seasons. Averaging across the two
    # would splice two different quantities into one series.
    FieldSpec(
        "blended_earnings_growth",
        (
            r"blended (?:\(year-over-year\) )?earnings (?:growth rate|decline) "
            r"for the S&P 500 is " + _NUM + r"%",
        ),
    ),
    FieldSpec(
        "estimated_earnings_growth",
        (
            r"estimated (?:\(year-over-year\) )?earnings (?:growth rate|decline) "
            r"for the S&P 500 is " + _NUM + r"%",
        ),
    ),
    FieldSpec(
        "forward_12m_pe",
        (r"forward 12-month P/E ratio for the S&P 500 is (\d+(?:\.\d+)?)",),
        core=True,
    ),
    FieldSpec(
        "estimated_growth_at_quarter_start",
        (
            r"estimated (?:\(year-over-year\) )?earnings (?:growth rate|decline) "
            r"for (?:the S&P 500 for )?Q\d 20\d{2} was " + _NUM + r"%",
        ),
    ),
    FieldSpec("pe_5y_average", (r"5-year average \((\d+(?:\.\d+)?)\)",)),
    FieldSpec("pe_10y_average", (r"10-year average \((\d+(?:\.\d+)?)\)",)),
    FieldSpec(
        "negative_guidance_count",
        (r"(\d+) S&P 500 companies have issued negative EPS guidance",),
        kind="int",
    ),
    FieldSpec(
        "positive_guidance_count",
        (r"(\d+) S&P 500 companies have issued positive EPS guidance",),
        kind="int",
    ),
    # Only the pre-2018 editions spell out the price and forward EPS behind the
    # P/E; later ones give 5- and 10-year averages instead.
    FieldSpec("index_price", (r"closing price \(([\d,]+\.?\d*)\)",)),
    FieldSpec("forward_12m_eps", (r"forward 12-month EPS estimate \(\$([\d.]+)\)",)),
)

FIELD_NAMES: tuple[str, ...] = tuple(field.name for field in FIELDS)

CORE_FIELDS: tuple[str, ...] = tuple(field.name for field in FIELDS if field.core)

#: Matches the standalone dateline under the masthead, e.g. "August 7, 2026".
DOCUMENT_DATE_PATTERN = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2}),\s+(20\d{2})\b"
)

MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}
