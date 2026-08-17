"""Measure how well a US-listed proxy tracks a non-US underlying.

Three things make the naive answer wrong, and each is handled explicitly.

**Sessions do not overlap.** Hong Kong closes at 08:00 UTC and Korea at 06:00,
while the US closes at 20:00 or 21:00. A US instrument's close-to-close return
for day T therefore contains half a day of news that the Asian market cannot
reflect until day T+1. Same-day correlation understates the relationship, and a
genuine link usually shows up as the US series *leading* by one day. Both the
lag structure and a weekly measure -- which mostly washes the offset out -- are
reported.

**Market beta is not tracking.** Two instruments that both rise with the S&P
will correlate whether or not they share any specific risk. Correlation of
SPY-neutral residuals separates "this proxy follows the name" from "both follow
the market", and it is the number that decides whether a country ETF is really
standing in for a single stock.

**A high correlation with the wrong slope is still not a hedge.** Beta and R^2
are reported so a position can be sized, not just signed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Positive lag means the proxy leads: the underlying's move on day t is
#: compared with the proxy's move on day t-1.
DEFAULT_LAGS: tuple[int, ...] = (-1, 0, 1)

#: Below this many overlapping observations the estimate is not worth a verdict.
MIN_DAILY_OVERLAP = 60
MIN_WEEKLY_OVERLAP = 20

STRONG_CORRELATION = 0.80
USABLE_CORRELATION = 0.55
WEAK_CORRELATION = 0.30

#: A proxy that only tracks once the market factor is left in is carrying index
#: beta, not the name.
SPECIFIC_CORRELATION = 0.30


@dataclass(frozen=True)
class Tracking:
    """One (underlying, proxy) pair, scored."""

    base_asset: str
    underlying: str
    proxy: str
    kind: str
    rationale: str
    overlap_days: int
    overlap_weeks: int
    daily_corr: float
    daily_lag: int
    daily_corr_same_day: float
    weekly_corr: float
    residual_corr: float
    beta: float
    r_squared: float
    verdict: str
    note: str = ""

    @property
    def is_usable(self) -> bool:
        return self.verdict in ("direct", "strong", "usable")


def _log_returns(prices: pd.Series) -> pd.Series:
    clean = prices.where(prices > 0).dropna()
    return np.log(clean).diff().dropna()


def _weekly_log_returns(prices: pd.Series) -> pd.Series:
    """Friday-to-Friday returns.

    Resampling before differencing is what makes this robust to two markets
    keeping different holiday calendars: whatever each traded last in the week
    is compared, instead of dropping the week entirely.
    """
    weekly = prices.where(prices > 0).resample("W-FRI").last().dropna()
    return np.log(weekly).diff().dropna()


def _correlation(left: pd.Series, right: pd.Series) -> tuple[float, int]:
    aligned = pd.concat([left, right], axis=1).dropna()
    if len(aligned) < 3:
        return float("nan"), len(aligned)
    if aligned.iloc[:, 0].std() == 0 or aligned.iloc[:, 1].std() == 0:
        return float("nan"), len(aligned)
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1])), len(aligned)


def best_lagged_correlation(
    underlying: pd.Series, proxy: pd.Series, lags: tuple[int, ...] = DEFAULT_LAGS
) -> tuple[float, int, float, int]:
    """Highest correlation over ``lags`` and the lag that produced it.

    Returns ``(best_corr, best_lag, same_day_corr, overlap_at_best)``. A best lag
    of +1 means the US proxy moved first, which is what the session offset
    predicts for an Asian underlying rather than evidence of anything odd.
    """
    same_day, _ = _correlation(underlying, proxy)
    best_corr, best_lag, best_overlap = float("-inf"), 0, 0
    for lag in lags:
        corr, overlap = _correlation(underlying, proxy.shift(lag))
        if np.isnan(corr):
            continue
        if corr > best_corr:
            best_corr, best_lag, best_overlap = corr, lag, overlap
    if best_corr == float("-inf"):
        return float("nan"), 0, same_day, 0
    return best_corr, best_lag, same_day, best_overlap


def _residualise(returns: pd.Series, market: pd.Series) -> pd.Series:
    """Returns with market beta regressed out."""
    aligned = pd.concat([returns, market], axis=1).dropna()
    if len(aligned) < 10:
        return pd.Series(dtype=float)
    y = aligned.iloc[:, 0].to_numpy()
    design = np.column_stack([np.ones(len(aligned)), aligned.iloc[:, 1].to_numpy()])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return pd.Series(y - design @ coefficients, index=aligned.index)


def _beta_and_r2(underlying: pd.Series, proxy: pd.Series) -> tuple[float, float]:
    """OLS of the underlying on the proxy, for sizing a hedge."""
    aligned = pd.concat([underlying, proxy], axis=1).dropna()
    if len(aligned) < 10 or aligned.iloc[:, 1].std() == 0:
        return float("nan"), float("nan")
    y = aligned.iloc[:, 0].to_numpy()
    x = aligned.iloc[:, 1].to_numpy()
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    total = float(np.sum((y - y.mean()) ** 2))
    residual = float(np.sum((y - fitted) ** 2))
    r_squared = 1.0 - residual / total if total > 0 else float("nan")
    return float(coefficients[1]), r_squared


def classify(weekly_corr: float, residual_corr: float, overlap_weeks: int) -> tuple[str, str]:
    """Verdict and a note explaining anything the headline number hides."""
    if overlap_weeks < MIN_WEEKLY_OVERLAP or np.isnan(weekly_corr):
        return "insufficient", f"only {overlap_weeks} overlapping weeks"

    if weekly_corr >= STRONG_CORRELATION:
        verdict = "strong"
    elif weekly_corr >= USABLE_CORRELATION:
        verdict = "usable"
    elif weekly_corr >= WEAK_CORRELATION:
        verdict = "weak"
    else:
        verdict = "poor"

    note = ""
    if not np.isnan(residual_corr) and residual_corr < SPECIFIC_CORRELATION:
        note = "market beta only: little co-movement left once SPY is removed"
        if verdict in ("strong", "usable"):
            # The headline correlation is real but generic; a strategy trading
            # the name's own risk would not be hedged by this.
            verdict = "market-beta"
    return verdict, note


def measure(
    base_asset: str,
    underlying_name: str,
    underlying_prices: pd.Series,
    proxy_name: str,
    proxy_prices: pd.Series,
    market_prices: pd.Series,
    kind: str = "",
    rationale: str = "",
) -> Tracking:
    """Score one candidate proxy against its underlying."""
    underlying_daily = _log_returns(underlying_prices)
    proxy_daily = _log_returns(proxy_prices)
    daily_corr, daily_lag, same_day, daily_overlap = best_lagged_correlation(
        underlying_daily, proxy_daily
    )

    underlying_weekly = _weekly_log_returns(underlying_prices)
    proxy_weekly = _weekly_log_returns(proxy_prices)
    market_weekly = _weekly_log_returns(market_prices)
    weekly_corr, weekly_overlap = _correlation(underlying_weekly, proxy_weekly)

    residual_corr, _ = _correlation(
        _residualise(underlying_weekly, market_weekly),
        _residualise(proxy_weekly, market_weekly),
    )
    beta, r_squared = _beta_and_r2(underlying_weekly, proxy_weekly)
    verdict, note = classify(weekly_corr, residual_corr, weekly_overlap)

    return Tracking(
        base_asset=base_asset,
        underlying=underlying_name,
        proxy=proxy_name,
        kind=kind,
        rationale=rationale,
        overlap_days=daily_overlap,
        overlap_weeks=weekly_overlap,
        daily_corr=daily_corr,
        daily_lag=daily_lag,
        daily_corr_same_day=same_day,
        weekly_corr=weekly_corr,
        residual_corr=residual_corr,
        beta=beta,
        r_squared=r_squared,
        verdict=verdict,
        note=note,
    )


def rank_candidates(results: list[Tracking]) -> list[Tracking]:
    """Best proxy first: usable ones by weekly correlation, then the rest."""
    from savi_uz.us_proxy_map import PROXY_KINDS

    def key(result: Tracking) -> tuple:
        corr = result.weekly_corr if not np.isnan(result.weekly_corr) else -1.0
        kind_rank = PROXY_KINDS.index(result.kind) if result.kind in PROXY_KINDS else len(PROXY_KINDS)
        return (-round(corr, 3), kind_rank)

    return sorted(results, key=key)
