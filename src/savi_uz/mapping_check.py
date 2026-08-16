"""Validation of Binance trad-FI perps against their assumed Yahoo Finance underlying.

Roughly a third of the Binance trad-FI tickers are non-obvious (``HK0700``,
``KODEX200``, ``CSOPSAMSUNG2L``, ...), so a hand-written mapping table cannot be
trusted on its own. Every candidate mapping is scored against the contract's own
Binance price history before it is used for risk work.

Two independent tests are applied, because neither is sufficient alone:

* **Price-ratio stability.** Unrelated securities do not hold a constant price
  ratio for weeks. This is the stronger test, and the only one that survives the
  session-time mismatch on HK/KR names, where a Binance UTC daily bar straddles
  two local sessions and wrecks day-over-day return correlation.
* **Rank correlation of returns.** Outlier-resistant, and the test that catches
  a stable-ratio coincidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import sqrt
from statistics import fmean, median

VERIFIED = "verified"
WEAK = "weak"
ASSUMED = "assumed"
UNVERIFIED = "unverified"
NO_DATA = "no-data"
UNLISTED = "unlisted"

USABLE_STATUSES = frozenset({VERIFIED, WEAK, ASSUMED})

#: Overlapping bars needed before either test says anything at all.
MIN_TESTABLE_DAYS = 3
#: Overlapping bars needed before the ratio test alone can verify a mapping.
VERIFIED_MIN_DAYS = 6
#: Overlapping bars needed before rank correlation alone can verify a mapping.
VERIFIED_CORR_MIN_DAYS = 20

VERIFIED_DISPERSION = 0.05
WEAK_DISPERSION = 0.10
VERIFIED_RANK_CORR = 0.70
WEAK_RANK_CORR = 0.45

#: Ratio dispersion is measured on the most recent overlapping bars only, so a
#: mid-window share split cannot masquerade as a broken mapping.
SCALE_WINDOW = 10

#: US equities, commodities and Yahoo's own mirror of a Binance contract all
#: quote the same number as the perp. HK/KR perps are USD-quoted against a
#: local-currency underlying, so their ratio is an FX rate and is exempt.
UNIT_SCALE_BAND = (0.5, 2.0)


@dataclass(frozen=True)
class MappingCheck:
    """Outcome of comparing one Binance contract to one Yahoo candidate."""

    binance_symbol: str
    yahoo_ticker: str | None
    source: str
    status: str
    overlap_days: int
    rank_corr: float
    best_lag: int
    scale_dispersion: float
    scale_median: float

    @property
    def is_usable(self) -> bool:
        return self.status in USABLE_STATUSES

    @property
    def is_independent(self) -> bool:
        """True when the Yahoo series is a real underlying, not Binance's own feed."""
        return self.is_usable and self.source != "venue-mirror"


def _pearson(x: list[float], y: list[float]) -> float:
    n = min(len(x), len(y))
    if n < 3:
        return 0.0
    x, y = x[:n], y[:n]
    x_mean, y_mean = fmean(x), fmean(y)
    cov = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    x_var = sum((xi - x_mean) ** 2 for xi in x)
    y_var = sum((yi - y_mean) ** 2 for yi in y)
    if x_var <= 0 or y_var <= 0:
        return 0.0
    return cov / sqrt(x_var * y_var)


def _ranks(values: list[float]) -> list[float]:
    """Average ranks, so ties do not bias the rank correlation."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        shared = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            ranks[order[index]] = shared
        position = end + 1
    return ranks


def _spearman(x: list[float], y: list[float]) -> float:
    n = min(len(x), len(y))
    if n < 3:
        return 0.0
    return _pearson(_ranks(x[:n]), _ranks(y[:n]))


def _simple_returns(values: list[float]) -> list[float]:
    return [0.0 if values[i - 1] <= 0 else (values[i] - values[i - 1]) / values[i - 1] for i in range(1, len(values))]


def _best_lagged_rank_corr(left: list[float], right: list[float]) -> tuple[float, int]:
    """Best rank correlation over lags -1/0/+1, absorbing non-overlapping sessions."""
    best_corr, best_lag = 0.0, 0
    for lag in (-1, 0, 1):
        if lag == 0:
            a, b = left, right
        elif lag > 0:
            a, b = left[lag:], right[:-lag]
        else:
            a, b = left[:lag], right[-lag:]
        corr = _spearman(a, b)
        if abs(corr) > abs(best_corr):
            best_corr, best_lag = corr, lag
    return best_corr, best_lag


def _quantile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    position = fraction * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def _ratio_dispersion(ratios: list[float]) -> tuple[float, float]:
    """Robust (IQR / median) spread of the price ratio, plus the median itself."""
    if not ratios:
        return 0.0, 0.0
    window = sorted(ratios[-SCALE_WINDOW:])
    centre = median(window)
    if centre <= 0:
        return float("inf"), 0.0
    return (_quantile(window, 0.75) - _quantile(window, 0.25)) / centre, centre


def check_mapping(
    binance_symbol: str,
    yahoo_ticker: str,
    source: str,
    binance_closes: dict[date, float],
    yahoo_closes: dict[date, float],
    expect_unit_scale: bool = True,
) -> MappingCheck:
    """Score one candidate mapping on overlapping daily bars."""
    common = sorted(set(binance_closes) & set(yahoo_closes))
    binance_series = [binance_closes[day] for day in common]
    yahoo_series = [yahoo_closes[day] for day in common]
    ratios = [b / y for b, y in zip(binance_series, yahoo_series) if y > 0]
    dispersion, scale_median = _ratio_dispersion(ratios)
    corr, lag = _best_lagged_rank_corr(_simple_returns(binance_series), _simple_returns(yahoo_series))

    status = _classify(
        overlap=len(common),
        source=source,
        dispersion=dispersion,
        scale_median=scale_median,
        rank_corr=corr,
        expect_unit_scale=expect_unit_scale,
    )
    return MappingCheck(
        binance_symbol=binance_symbol,
        yahoo_ticker=yahoo_ticker,
        source=source,
        status=status,
        overlap_days=len(common),
        rank_corr=corr,
        best_lag=lag,
        scale_dispersion=dispersion,
        scale_median=scale_median,
    )


def _classify(
    overlap: int,
    source: str,
    dispersion: float,
    scale_median: float,
    rank_corr: float,
    expect_unit_scale: bool,
) -> str:
    if overlap < MIN_TESTABLE_DAYS:
        # Contracts listed days ago cannot be tested; trust curation, nothing else.
        return ASSUMED if source == "curated" else NO_DATA

    low, high = UNIT_SCALE_BAND
    if expect_unit_scale and not low <= scale_median <= high:
        return UNVERIFIED

    if overlap >= VERIFIED_MIN_DAYS and dispersion <= VERIFIED_DISPERSION:
        return VERIFIED
    if overlap >= VERIFIED_CORR_MIN_DAYS and rank_corr >= VERIFIED_RANK_CORR:
        return VERIFIED
    if dispersion <= WEAK_DISPERSION or rank_corr >= WEAK_RANK_CORR:
        return WEAK
    return UNVERIFIED


def unlisted_check(binance_symbol: str) -> MappingCheck:
    """Marker for pre-IPO contracts that have no public underlying to map to."""
    return MappingCheck(binance_symbol, None, "none", UNLISTED, 0, 0.0, 0, 0.0, 0.0)


def pick_best_mapping(checks: list[MappingCheck]) -> MappingCheck | None:
    """Prefer a usable independent underlying over Binance's own mirrored feed.

    A mirror always validates near-perfectly -- it is the same price series -- so
    it must never outrank a real underlying that merely validates less cleanly.
    """
    if not checks:
        return None
    rank = {VERIFIED: 4, WEAK: 3, ASSUMED: 2, UNVERIFIED: 1, NO_DATA: 0, UNLISTED: 0}
    return max(
        checks,
        key=lambda check: (
            check.is_independent,
            rank[check.status],
            check.overlap_days,
            abs(check.rank_corr),
        ),
    )
