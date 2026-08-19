"""Daily features from an option chain snapshot.

Every feature is built from a single end-of-day snapshot and is meant to predict
the *following* session, so nothing here may read the bar it is forecasting.

The snapshots carry implied volatility and gamma but no delta, so delta, vanna
and charm are re-derived from the stored volatility with Black-Scholes.  A zero
rate is used: over the 0-49 day maturities present, the carry term shifts d1 by
well under a volatility point and cannot change the ranking of any feature.

Features fall into three families:

* **level** -- at-the-money volatility and its term structure, which is the
  standard forward estimate of realised volatility
* **shape** -- skew between downside and upside strikes, the classic directional
  measure from the option-pricing literature
* **positioning** -- dealer gamma, its concentration, where the gamma flip sits
  relative to spot, and how much of it is same-day expiry
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_greeks(spot, strike, years, vol, side):
    """Delta, gamma and vanna at a zero rate; None when the inputs are degenerate."""
    if spot <= 0 or strike <= 0 or years <= 0 or vol <= 0:
        return None
    d1 = (math.log(spot / strike) + 0.5 * vol * vol * years) / (vol * math.sqrt(years))
    d2 = d1 - vol * math.sqrt(years)
    pdf = _norm_pdf(d1)
    delta = _norm_cdf(d1) if side == "call" else _norm_cdf(d1) - 1.0
    gamma = pdf / (spot * vol * math.sqrt(years))
    vanna = -pdf * d2 / vol
    return delta, gamma, vanna


@dataclass(frozen=True)
class Contract:
    side: str
    strike: float
    dte: int
    iv: float | None
    gamma: float | None
    open_interest: float
    volume: float


def _interpolate(points, target):
    """Linear interpolation over (x, y) pairs, clamped at both ends."""
    usable = sorted((x, y) for x, y in points if y is not None)
    if not usable:
        return None
    if target <= usable[0][0]:
        return usable[0][1]
    if target >= usable[-1][0]:
        return usable[-1][1]
    for (x0, y0), (x1, y1) in zip(usable, usable[1:]):
        if x0 <= target <= x1:
            if x1 == x0:
                return y0
            weight = (target - x0) / (x1 - x0)
            return y0 + weight * (y1 - y0)
    return usable[-1][1]


def _atm_iv(rows, spot):
    """Volatility interpolated to the money, averaging the call and put wings."""
    out = {}
    for side in ("call", "put"):
        points = [(c.strike, c.iv) for c in rows if c.side == side and c.iv]
        out[side] = _interpolate(points, spot)
    both = [v for v in out.values() if v is not None]
    return sum(both) / len(both) if both else None


def _skew(rows, spot, low=0.95, high=1.05):
    """Downside minus upside volatility, a moneyness-anchored smirk measure."""
    puts = [(c.strike, c.iv) for c in rows if c.side == "put" and c.iv]
    calls = [(c.strike, c.iv) for c in rows if c.side == "call" and c.iv]
    down = _interpolate(puts, spot * low)
    up = _interpolate(calls, spot * high)
    if down is None or up is None:
        return None
    return down - up


def _delta_skew(rows, spot, target=0.25):
    """25-delta put volatility minus 25-delta call volatility."""
    picked = {}
    for side, want in (("put", -target), ("call", target)):
        best = None
        for c in rows:
            if c.side != side or not c.iv:
                continue
            greeks = black_scholes_greeks(spot, c.strike, max(c.dte, 1) / 365.0,
                                          c.iv, side)
            if greeks is None:
                continue
            distance = abs(greeks[0] - want)
            if best is None or distance < best[0]:
                best = (distance, c.iv)
        picked[side] = best[1] if best else None
    if picked["put"] is None or picked["call"] is None:
        return None
    return picked["put"] - picked["call"]


def _gamma_flip(rows, spot, span=0.06, steps=25):
    """Spot level where dealer gamma changes sign, as a share of spot.

    Net gamma is re-evaluated across a grid of hypothetical spots; the crossing
    is where dealers stop damping moves and start amplifying them.
    """
    def net_at(price):
        total = 0.0
        for c in rows:
            if not c.iv or not c.open_interest:
                continue
            greeks = black_scholes_greeks(price, c.strike, max(c.dte, 1) / 365.0,
                                          c.iv, c.side)
            if greeks is None:
                continue
            sign = 1.0 if c.side == "call" else -1.0
            total += sign * greeks[1] * c.open_interest * price * price
        return total

    grid = [spot * (1.0 - span + 2 * span * i / (steps - 1)) for i in range(steps)]
    values = [(price, net_at(price)) for price in grid]
    # A book with matched call and put open interest nets to zero at every
    # price. That is not a flip point, it is the absence of one, so a strict
    # sign change is required rather than merely touching zero.
    for (p0, v0), (p1, v1) in zip(values, values[1:]):
        if (v0 < 0) == (v1 < 0) or v1 == v0:
            continue
        crossing = p0 + (p1 - p0) * (-v0) / (v1 - v0)
        return (crossing - spot) / spot
    return None


def snapshot_features(rows: list[Contract], spot: float) -> dict:
    """All features for one symbol on one session."""
    if not rows or spot <= 0:
        return {}
    front = [c for c in rows if 5 <= c.dte <= 12]
    mid = [c for c in rows if 20 <= c.dte <= 40]
    near = front or [c for c in rows if c.dte <= 12]

    atm_front = _atm_iv(near, spot)
    atm_mid = _atm_iv(mid, spot) if mid else None
    calls = [c for c in rows if c.side == "call"]
    puts = [c for c in rows if c.side == "put"]
    call_oi = sum(c.open_interest for c in calls)
    put_oi = sum(c.open_interest for c in puts)
    call_vol = sum(c.volume for c in calls)
    put_vol = sum(c.volume for c in puts)

    gamma_dollars = 0.0
    zero_dte_gamma = 0.0
    vanna_total = 0.0
    for c in rows:
        if not c.iv or not c.open_interest:
            continue
        greeks = black_scholes_greeks(spot, c.strike, max(c.dte, 1) / 365.0,
                                      c.iv, c.side)
        if greeks is None:
            continue
        sign = 1.0 if c.side == "call" else -1.0
        contribution = sign * greeks[1] * c.open_interest * spot * spot
        gamma_dollars += contribution
        if c.dte == 0:
            zero_dte_gamma += abs(contribution)
        vanna_total += sign * greeks[2] * c.open_interest

    absolute = sum(
        abs(black_scholes_greeks(spot, c.strike, max(c.dte, 1) / 365.0, c.iv, c.side)[1]
            * c.open_interest * spot * spot)
        for c in rows
        if c.iv and c.open_interest
        and black_scholes_greeks(spot, c.strike, max(c.dte, 1) / 365.0, c.iv, c.side)
    )

    return {
        "atm_iv": atm_front,
        "iv_term_slope": (atm_front - atm_mid) if (atm_front and atm_mid) else None,
        "skew_moneyness": _skew(near, spot),
        "skew_25delta": _delta_skew(near, spot),
        "put_call_oi": (put_oi / call_oi) if call_oi else None,
        "put_call_volume": (put_vol / call_vol) if call_vol else None,
        "net_gamma": gamma_dollars,
        "gamma_balance": (gamma_dollars / absolute) if absolute else None,
        "gamma_flip_distance": _gamma_flip(near, spot),
        "zero_dte_share": (zero_dte_gamma / absolute) if absolute else None,
        "vanna": vanna_total,
        "total_oi": call_oi + put_oi,
        "total_volume": call_vol + put_vol,
    }


def aggregate_gamma(rows: list[Contract], spot: float,
                    multiplier: float = 100.0) -> dict:
    """Dealer gamma exposure in dollars per 1% move.

    The standard convention has dealers long calls and short puts, so calls add
    and puts subtract.  Contracts without a gamma or without open interest carry
    no exposure and are counted as unusable rather than as zero, which keeps the
    coverage figure honest when a vendor leaves fields blank.
    """
    call_gex = put_gex = absolute = 0.0
    usable = 0
    for contract in rows:
        if not contract.gamma or not contract.open_interest or spot <= 0:
            continue
        usable += 1
        dollars = (contract.gamma * contract.open_interest * multiplier
                   * spot * spot * 0.01)
        absolute += abs(dollars)
        if contract.side == "call":
            call_gex += dollars
        else:
            put_gex += dollars
    return {"call_gex": call_gex, "put_gex": put_gex,
            "net_gex": call_gex - put_gex, "absolute_gex": absolute,
            "usable": usable, "contracts": len(rows)}


FEATURE_NAMES = (
    "atm_iv", "iv_term_slope", "skew_moneyness", "skew_25delta",
    "put_call_oi", "put_call_volume", "net_gamma", "gamma_balance",
    "gamma_flip_distance", "zero_dte_share", "vanna", "total_oi", "total_volume",
)
