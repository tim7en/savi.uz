"""Validation of Binance trad-FI perps against their assumed Yahoo Finance underlying.

Roughly a third of the Binance trad-FI tickers are non-obvious (``HK0700``,
``KODEX200``, ``CSOPSAMSUNG2L``, ...), so a hand-written mapping table cannot be
trusted on its own. Every candidate mapping is therefore checked against the
contract's own Binance price history before it is used for risk work.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import sqrt
from statistics import fmean, pstdev

VERIFIED = "verified"
WEAK = "weak"
UNVERIFIED = "unverified"
NO_DATA = "no-data"
UNLISTED = "unlisted"

#: A mapping is accepted when overlapping daily returns line up this closely.
MIN_OVERLAP_DAYS = 20
VERIFIED_CORR = 0.70
WEAK_CORR = 0.45
#: Perps on HK/KR names are USD-quoted against a local-currency underlying, so
#: the price ratio is an FX rate: allowed to drift, not to jump.
MAX_SCALE_CV = 0.15


@dataclass(frozen=True)
class MappingCheck:
    """Outcome of comparing one Binance contract to one Yahoo candidate."""

    binance_symbol: str
    yahoo_ticker: str | None
    source: str
    status: str
    overlap_days: int
    return_corr: float
    best_lag: int
    scale_cv: float
    scale_median: float

    @property
    def is_usable(self) -> bool:
        return self.status in (VERIFIED, WEAK)

    @property
    def is_independent(self) -> bool:
        """True when the Yahoo series is a real underlying, not Binance's own feed."""
        return self.is_usable and self.source != "venue-mirror"


def _pearson(x: list[float], y: list[float]) -> float:
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    x, y = x[:n], y[:n]
    x_mean, y_mean = fmean(x), fmean(y)
    cov = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    x_var = sum((xi - x_mean) ** 2 for xi in x)
    y_var = sum((yi - y_mean) ** 2 for yi in y)
    if x_var <= 0 or y_var <= 0:
        return 0.0
    return cov / sqrt(x_var * y_var)


def _simple_returns(values: list[float]) -> list[float]:
    returns: list[float] = []
    for idx in range(1, len(values)):
        prev = values[idx - 1]
        returns.append(0.0 if prev <= 0 else (values[idx] - prev) / prev)
    return returns


def _best_lagged_corr(left: list[float], right: list[float]) -> tuple[float, int]:
    """Best correlation over lags -1/0/+1 to absorb non-overlapping session times."""
    best_corr, best_lag = 0.0, 0
    for lag in (-1, 0, 1):
        if lag == 0:
            a, b = left, right
        elif lag > 0:
            a, b = left[lag:], right[:-lag]
        else:
            a, b = left[:lag], right[-lag:]
        corr = _pearson(a, b)
        if abs(corr) > abs(best_corr):
            best_corr, best_lag = corr, lag
    return best_corr, best_lag


def check_mapping(
    binance_symbol: str,
    yahoo_ticker: str,
    source: str,
    binance_closes: dict[date, float],
    yahoo_closes: dict[date, float],
) -> MappingCheck:
    """Score one candidate mapping on overlapping daily bars."""
    common = sorted(set(binance_closes) & set(yahoo_closes))
    binance_series = [binance_closes[day] for day in common]
    yahoo_series = [yahoo_closes[day] for day in common]

    if len(common) < 2:
        return MappingCheck(binance_symbol, yahoo_ticker, source, NO_DATA, len(common), 0.0, 0, 0.0, 0.0)

    corr, lag = _best_lagged_corr(_simple_returns(binance_series), _simple_returns(yahoo_series))
    ratios = [b / y for b, y in zip(binance_series, yahoo_series) if y > 0]
    scale_mean = fmean(ratios) if ratios else 0.0
    scale_cv = (pstdev(ratios) / scale_mean) if len(ratios) > 1 and scale_mean > 0 else 0.0
    scale_median = sorted(ratios)[len(ratios) // 2] if ratios else 0.0

    if len(common) < MIN_OVERLAP_DAYS:
        status = NO_DATA
    elif corr >= VERIFIED_CORR and scale_cv <= MAX_SCALE_CV:
        status = VERIFIED
    elif corr >= WEAK_CORR:
        status = WEAK
    else:
        status = UNVERIFIED

    return MappingCheck(
        binance_symbol=binance_symbol,
        yahoo_ticker=yahoo_ticker,
        source=source,
        status=status,
        overlap_days=len(common),
        return_corr=corr,
        best_lag=lag,
        scale_cv=scale_cv,
        scale_median=scale_median,
    )


def unlisted_check(binance_symbol: str) -> MappingCheck:
    """Marker for pre-IPO contracts that have no public underlying to map to."""
    return MappingCheck(binance_symbol, None, "none", UNLISTED, 0, 0.0, 0, 0.0, 0.0)


def pick_best_mapping(checks: list[MappingCheck]) -> MappingCheck | None:
    """Prefer an independent verified underlying, then correlation strength."""
    if not checks:
        return None
    rank = {VERIFIED: 3, WEAK: 2, UNVERIFIED: 1, NO_DATA: 0, UNLISTED: 0}
    return max(
        checks,
        key=lambda check: (
            rank[check.status],
            check.source != "venue-mirror",
            abs(check.return_corr),
            check.overlap_days,
        ),
    )
