"""Backtest a macro-equilibrium regime overlay on SPY and QQQ.

Five point-in-time macro pillars (labor, inflation, policy, financial
conditions, volatility) are each flagged "stressed" or "balanced" using only
data that would have been publicly known on the trading date (publication
lags below). The number of simultaneously stressed pillars maps to a target
equity weight; the rest sits in cash earning the prior-known effective fed
funds rate (DFF). This is a diversification/de-risking overlay, not a market
call: it holds less when several macro dimensions are stretched at once and
full exposure when the regime looks balanced.

Nasdaq-100 (^NDX) is a price index (no dividend reinvestment) and is used as
the QQQ proxy; S&P 500 total return (^SP500TR) is used as the SPY proxy.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro-db", type=Path, default=Path("data/macro/macro.db"))
    parser.add_argument("--equity-db", type=Path, default=Path("data/equity/equity.db"))
    parser.add_argument("--out", type=Path, default=Path("out/strategy/macro_equilibrium"))
    parser.add_argument("--initial", type=float, default=100_000.0)
    parser.add_argument("--switch-cost-bp", type=float, default=2.0)
    parser.add_argument("--episode-threshold", type=float, default=-0.15)
    return parser.parse_args(argv)


# Publication lag in calendar days: the soonest a series value could have
# been acted on, given typical release schedules.
PUBLICATION_LAG_DAYS = {
    "UNRATE": 40,     # BLS employment situation, ~5 weeks after month start
    "CPIAUCSL": 45,   # CPI report, ~6-7 weeks after month start
    "DFF": 2,
    "T10Y2Y": 2,
    "NFCI": 9,        # weekly, released the following Friday
    "BAA10Y": 2,
    "VIXCLS": 2,
}

EXPOSURE_BY_STRESS = {0: 1.00, 1: 1.00, 2: 0.70, 3: 0.45, 4: 0.20, 5: 0.00}


def load_macro_series(connection, series_id):
    rows = connection.execute(
        "SELECT obs_date, value FROM observations WHERE series_id=? AND value IS NOT NULL "
        "ORDER BY obs_date", (series_id,),
    ).fetchall()
    if not rows:
        raise ValueError(f"no observations for {series_id}")
    dates = pd.to_datetime([row[0] for row in rows])
    values = np.array([row[1] for row in rows], dtype=float)
    return pd.Series(values, index=dates)


def pit_align(trading_dates: pd.DatetimeIndex, obs: pd.Series, lag_days: int) -> np.ndarray:
    """Value knowable on each trading date, respecting a publication lag."""
    obs = obs.sort_index()
    obs_dates = obs.index.values.astype("datetime64[D]")
    cutoffs = (trading_dates - pd.Timedelta(days=lag_days)).values.astype("datetime64[D]")
    positions = np.searchsorted(obs_dates, cutoffs, side="right") - 1
    values = np.full(len(trading_dates), np.nan)
    valid = positions >= 0
    values[valid] = obs.values[positions[valid]]
    return values


def load_price(connection, ticker, start, end):
    rows = connection.execute(
        "SELECT obs_date, close FROM index_prices WHERE ticker=? AND obs_date BETWEEN ? AND ? "
        "AND close IS NOT NULL ORDER BY obs_date", (ticker, start, end),
    ).fetchall()
    dates = pd.to_datetime([row[0] for row in rows])
    values = np.array([row[1] for row in rows], dtype=float)
    return pd.Series(values, index=dates, name=ticker)


def _masked(condition: pd.Series, *required: pd.Series) -> pd.Series:
    """Boolean-as-float flag that is NaN wherever an input is not yet known."""
    invalid = required[0].isna()
    for series in required[1:]:
        invalid = invalid | series.isna()
    flag = condition.astype(float)
    flag[invalid] = np.nan
    return flag


def build_pillars(trading_dates: pd.DatetimeIndex, macro: dict) -> pd.DataFrame:
    unrate = pd.Series(pit_align(trading_dates, macro["UNRATE"], PUBLICATION_LAG_DAYS["UNRATE"]), index=trading_dates).ffill()
    unrate_3mo = unrate.rolling(63, min_periods=21).mean()
    unrate_floor = unrate_3mo.rolling(252, min_periods=21).min()
    labor_stress = _masked((unrate_3mo - unrate_floor) >= 0.50, unrate_3mo, unrate_floor)

    cpi = pd.Series(pit_align(trading_dates, macro["CPIAUCSL"], PUBLICATION_LAG_DAYS["CPIAUCSL"]), index=trading_dates).ffill()
    cpi_yoy = cpi / cpi.shift(252) - 1.0
    inflation_stress = _masked(cpi_yoy >= 0.04, cpi_yoy)

    dff = pd.Series(pit_align(trading_dates, macro["DFF"], PUBLICATION_LAG_DAYS["DFF"]), index=trading_dates).ffill()
    t10y2y = pd.Series(pit_align(trading_dates, macro["T10Y2Y"], PUBLICATION_LAG_DAYS["T10Y2Y"]), index=trading_dates).ffill()
    real_rate = dff - cpi_yoy * 100.0
    policy_stress = _masked((real_rate >= 1.0) | (t10y2y < 0), real_rate, t10y2y)

    nfci = pd.Series(pit_align(trading_dates, macro["NFCI"], PUBLICATION_LAG_DAYS["NFCI"]), index=trading_dates).ffill()
    baa10y = pd.Series(pit_align(trading_dates, macro["BAA10Y"], PUBLICATION_LAG_DAYS["BAA10Y"]), index=trading_dates).ffill()
    baa_mean = baa10y.rolling(756, min_periods=60).mean()
    baa_std = baa10y.rolling(756, min_periods=60).std()
    baa_z = (baa10y - baa_mean) / baa_std
    fc_stress = _masked((nfci > 0) | (baa_z > 1.0), nfci, baa_z)

    vix = pd.Series(pit_align(trading_dates, macro["VIXCLS"], PUBLICATION_LAG_DAYS["VIXCLS"]), index=trading_dates).ffill()
    vix_20d = vix.rolling(20, min_periods=10).mean()
    vix_pctile = vix_20d.rolling(252, min_periods=60).apply(
        lambda window: (window <= window[-1]).mean(), raw=True
    )
    vol_stress = _masked(vix_pctile >= 0.70, vix_pctile)

    frame = pd.DataFrame({
        "labor_stress": labor_stress,
        "inflation_stress": inflation_stress,
        "policy_stress": policy_stress,
        "fc_stress": fc_stress,
        "vol_stress": vol_stress,
        "unrate_3mo": unrate_3mo,
        "cpi_yoy": cpi_yoy,
        "real_rate": real_rate,
        "t10y2y": t10y2y,
        "nfci": nfci,
        "baa_z": baa_z,
        "vix_pctile": vix_pctile,
        "dff": dff,
    }, index=trading_dates)
    return frame


def simulate(price: pd.Series, target_exposure: pd.Series, cash_rate: pd.Series,
             initial: float, switch_cost_bp: float) -> pd.Series:
    dates = price.index
    wealth = np.empty(len(dates))
    equity_value = initial * target_exposure.iloc[0]
    cash = initial - equity_value
    wealth[0] = initial
    prev_target = target_exposure.iloc[0]
    for i in range(1, len(dates)):
        ret = price.iloc[i] / price.iloc[i - 1] - 1.0
        equity_value *= (1.0 + ret)
        days = (dates[i] - dates[i - 1]).days
        cash *= (1.0 + (cash_rate.iloc[i - 1] / 100.0) * days / 365.0)
        total = equity_value + cash
        target = target_exposure.iloc[i]
        if target != prev_target:
            desired_equity = total * target
            turnover = abs(desired_equity - equity_value)
            cost = turnover * switch_cost_bp / 10_000.0
            total -= cost
            equity_value = total * target
            cash = total - equity_value
            prev_target = target
        wealth[i] = equity_value + cash
    return pd.Series(wealth, index=dates)


def metrics(wealth: pd.Series, cash_rate: pd.Series) -> dict:
    returns = wealth.pct_change().dropna()
    days = (wealth.index[-1] - wealth.index[0]).days
    years = days / 365.25
    cagr = (wealth.iloc[-1] / wealth.iloc[0]) ** (1.0 / years) - 1.0
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    vol = returns.std() * np.sqrt(252)
    rf_daily = (cash_rate.reindex(returns.index).ffill() / 100.0) / 252.0
    excess = returns - rf_daily
    sharpe = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else float("nan")
    return {
        "terminal_wealth": round(float(wealth.iloc[-1]), 2),
        "cagr": float(cagr),
        "max_drawdown": float(drawdown.min()),
        "annual_volatility": float(vol),
        "sharpe": float(sharpe),
    }


def find_episodes(price: pd.Series, threshold: float) -> list[dict]:
    dates = price.index
    values = price.values
    episodes = []
    peak_idx = 0
    peak_val = values[0]
    trough_idx = None
    trough_val = None
    breached = False
    for i in range(1, len(values)):
        if values[i] >= peak_val:
            if breached:
                episodes.append({
                    "peak_idx": peak_idx, "trough_idx": trough_idx,
                    "recovery_idx": i, "depth": trough_val / peak_val - 1.0,
                })
            peak_idx, peak_val = i, values[i]
            trough_idx, trough_val, breached = None, None, False
            continue
        if trough_val is None or values[i] < trough_val:
            trough_idx, trough_val = i, values[i]
        if values[i] / peak_val - 1.0 <= threshold:
            breached = True
    if breached:
        episodes.append({
            "peak_idx": peak_idx, "trough_idx": trough_idx,
            "recovery_idx": None, "depth": trough_val / peak_val - 1.0,
        })
    out = []
    for episode in episodes:
        peak_date = dates[episode["peak_idx"]]
        trough_date = dates[episode["trough_idx"]]
        recovery_date = dates[episode["recovery_idx"]] if episode["recovery_idx"] is not None else None
        out.append({
            "peak_date": peak_date.strftime("%Y-%m-%d"),
            "trough_date": trough_date.strftime("%Y-%m-%d"),
            "recovery_date": recovery_date.strftime("%Y-%m-%d") if recovery_date is not None else None,
            "depth": float(episode["depth"]),
            "days_to_trough": int((trough_date - peak_date).days),
            "days_to_recover": int((recovery_date - peak_date).days) if recovery_date is not None else None,
            "peak_idx": episode["peak_idx"],
        })
    return out


def find_local_episode(price: pd.Series, window_start: str, window_end: str) -> dict:
    """Peak-to-trough-to-recovery within a bounded window, nested inside a
    longer drawdown that had not yet regained its own prior high."""
    window = price.loc[window_start:window_end]
    running_max = window.cummax()
    drawdown = window / running_max - 1.0
    trough_date = drawdown.idxmin()
    depth = float(drawdown.min())
    peak_val = running_max.loc[trough_date]
    peak_date = window[window == peak_val].index[0]
    after = price.loc[trough_date:]
    recovered = after[after >= peak_val]
    recovery_date = recovered.index[0] if not recovered.empty else None
    return {
        "peak_date": peak_date.strftime("%Y-%m-%d"),
        "trough_date": trough_date.strftime("%Y-%m-%d"),
        "recovery_date": recovery_date.strftime("%Y-%m-%d") if recovery_date is not None else None,
        "depth": depth,
        "days_to_trough": int((trough_date - peak_date).days),
        "days_to_recover": int((recovery_date - peak_date).days) if recovery_date is not None else None,
        "peak_idx": int(price.index.get_loc(peak_date)),
    }


def annotate_episodes(episodes: list[dict], stress_count: pd.Series, target_exposure: pd.Series,
                       pillars: pd.DataFrame, lookback: int = 250) -> list[dict]:
    dates = stress_count.index
    pillar_names = ["labor_stress", "inflation_stress", "policy_stress", "fc_stress", "vol_stress"]
    exposure_values = target_exposure.values
    for episode in episodes:
        peak_idx = episode.pop("peak_idx")
        exposure_at_peak = float(exposure_values[peak_idx])
        lead_days = None
        if exposure_at_peak < 1.0:
            # Consecutive de-risked run ending at (and including) the peak day.
            i = peak_idx
            floor = max(0, peak_idx - lookback)
            while i >= floor and exposure_values[i] < 1.0:
                i -= 1
            run_start_idx = i + 1
            lead_days = int((dates[peak_idx] - dates[run_start_idx]).days)
        count_at_peak = stress_count.iloc[peak_idx]
        stressed_at_peak = [
            name.replace("_stress", "") for name in pillar_names
            if pillars[name].iloc[peak_idx] == 1.0
        ]
        episode["stress_count_at_peak"] = None if pd.isna(count_at_peak) else int(count_at_peak)
        episode["exposure_at_peak"] = exposure_at_peak
        episode["stressed_pillars_at_peak"] = stressed_at_peak
        episode["lead_days_below_full_exposure"] = lead_days
    return episodes


def main(argv=None):
    args = parse_args(argv)
    macro_con = sqlite3.connect(args.macro_db)
    equity_con = sqlite3.connect(args.equity_db)

    macro_ids = ["UNRATE", "CPIAUCSL", "DFF", "T10Y2Y", "NFCI", "BAA10Y", "VIXCLS"]
    macro = {series_id: load_macro_series(macro_con, series_id) for series_id in macro_ids}

    spy = load_price(equity_con, "^SP500TR", "2000-01-01", "2026-12-31")
    ndx = load_price(equity_con, "^NDX", "2000-01-01", "2026-12-31")
    trading_dates = spy.index.intersection(ndx.index)
    spy, ndx = spy.reindex(trading_dates), ndx.reindex(trading_dates)

    pillars = build_pillars(trading_dates, macro)
    pillar_cols = ["labor_stress", "inflation_stress", "policy_stress", "fc_stress", "vol_stress"]
    all_valid = pillars[pillar_cols].notna().all(axis=1)
    start_date = trading_dates[all_valid.values.argmax()]

    stress_count = pillars[pillar_cols].sum(axis=1, skipna=False)
    target_exposure = stress_count.map(lambda x: EXPOSURE_BY_STRESS.get(int(x)) if pd.notna(x) else 1.0)
    target_exposure = pd.Series(target_exposure.values, index=trading_dates, dtype=float)
    stress_count = pd.Series(stress_count.values, index=trading_dates)

    cash_rate = pillars["dff"]

    spy_regime = simulate(spy, target_exposure, cash_rate, args.initial, args.switch_cost_bp)
    ndx_regime = simulate(ndx, target_exposure, cash_rate, args.initial, args.switch_cost_bp)
    spy_buyhold = args.initial * spy / spy.iloc[0]
    ndx_buyhold = args.initial * ndx / ndx.iloc[0]

    episodes = find_episodes(ndx, args.episode_threshold)
    episodes = annotate_episodes(episodes, stress_count, target_exposure, pillars)
    episodes.sort(key=lambda e: e["depth"])

    nested_gfc = find_local_episode(ndx, "2002-10-08", "2009-12-31")
    nested_gfc = annotate_episodes([nested_gfc], stress_count, target_exposure, pillars)[0]

    exposure_valid = target_exposure[trading_dates >= start_date]
    time_at_exposure = (
        exposure_valid.value_counts(normalize=True).sort_index(ascending=False).to_dict()
    )

    results = {
        "sample": {
            "trading_start": trading_dates[0].strftime("%Y-%m-%d"),
            "trading_end": trading_dates[-1].strftime("%Y-%m-%d"),
            "signal_start": start_date.strftime("%Y-%m-%d"),
            "sessions": int(len(trading_dates)),
        },
        "pillars": {
            "labor": "3-month average UNRATE at least 0.50pp above its trailing 12-month low (Sahm-style proxy); ~40-day publication lag.",
            "inflation": "Headline CPI year-over-year at or above 4%; ~45-day publication lag.",
            "policy": "Effective fed funds minus CPI YoY at or above 1.0pp (restrictive real rate) OR 2s10s curve (T10Y2Y) inverted.",
            "financial_conditions": "Chicago Fed NFCI positive (tighter than average) OR Baa-10Y spread more than 1 std above its trailing 3-year mean.",
            "volatility": "20-day average VIX at or above the 70th percentile of its trailing 1-year range.",
        },
        "exposure_ladder": EXPOSURE_BY_STRESS,
        "switch_cost_bp": args.switch_cost_bp,
        "time_at_exposure": {f"{k:.2f}": v for k, v in time_at_exposure.items()},
        "variants": {
            "ndx_regime": metrics(ndx_regime, cash_rate),
            "ndx_buyhold": metrics(ndx_buyhold, cash_rate),
            "spy_regime": metrics(spy_regime, cash_rate),
            "spy_buyhold": metrics(spy_buyhold, cash_rate),
        },
        "episodes": episodes,
        "nested_2008_episode": nested_gfc,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    daily = pd.DataFrame({
        "date": trading_dates,
        "ndx_price": ndx.values,
        "spy_price": spy.values,
        "stress_count": stress_count.values,
        "target_exposure": target_exposure.values,
        "ndx_regime_wealth": ndx_regime.values,
        "ndx_buyhold_wealth": ndx_buyhold.values,
        "spy_regime_wealth": spy_regime.values,
        "spy_buyhold_wealth": spy_buyhold.values,
        "labor_stress": pillars["labor_stress"].values,
        "inflation_stress": pillars["inflation_stress"].values,
        "policy_stress": pillars["policy_stress"].values,
        "fc_stress": pillars["fc_stress"].values,
        "vol_stress": pillars["vol_stress"].values,
    })
    daily.to_csv(args.out / "daily.csv", index=False)

    print(json.dumps(results["sample"], indent=2))
    print(json.dumps(results["variants"], indent=2))
    print(f"episodes found: {len(episodes)}")


if __name__ == "__main__":
    main()
