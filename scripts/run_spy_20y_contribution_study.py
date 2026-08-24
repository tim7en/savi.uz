"""Study rolling 20-year outcomes for $10,000/year in the staged SPY strategy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_spy_staged_funding import simulate, xirr  # noqa: E402


ANNUAL_CONTRIBUTION = 10_000.0
YEARS = 20
SOURCE = ROOT / "out/strategy/spy_reverse_vault/daily.csv"
OUTPUT = ROOT / "out/strategy/spy_20y_contributions/results.json"


def underwater_stats(nav: pd.Series) -> dict:
    """Measure peak-to-recovery spells on a flow-adjusted NAV."""
    peak_value = float(nav.iloc[0])
    peak_date = nav.index[0]
    episode_start = None
    recovered_days: list[int] = []

    for stamp, value in nav.iloc[1:].items():
        value = float(value)
        if value >= peak_value * (1.0 - 1e-12):
            if episode_start is not None:
                recovered_days.append((stamp - episode_start).days)
                episode_start = None
            if value > peak_value:
                peak_value = value
                peak_date = stamp
        elif episode_start is None:
            episode_start = peak_date

    open_days = ((nav.index[-1] - episode_start).days
                 if episode_start is not None else 0)
    drawdown = nav / nav.cummax() - 1.0
    all_spells = recovered_days + ([open_days] if open_days else [])
    return {
        "longest_underwater_years": max(all_spells, default=0) / 365.2425,
        "longest_recovered_spell_years": max(recovered_days, default=0) / 365.2425,
        "time_underwater_share": float((drawdown < -1e-12).mean()),
        "time_below_10pct_share": float((drawdown <= -0.10).mean()),
        "time_below_20pct_share": float((drawdown <= -0.20).mean()),
        "time_below_50pct_share": float((drawdown <= -0.50).mean()),
        "ends_underwater": episode_start is not None,
        "ending_underwater_years": open_days / 365.2425,
    }


def exact_monthly_schedule(index: pd.DatetimeIndex) -> pd.Series:
    """Make 239 deposits after the opening payment, one month apart."""
    schedule = pd.Series(0.0, index=index)
    for month in range(1, YEARS * 12):
        due = index[0] + pd.DateOffset(months=month)
        position = index.searchsorted(due)
        if position >= len(index):
            raise ValueError(f"20-year window ends before payment {month + 1}")
        schedule.iloc[position] += ANNUAL_CONTRIBUTION / 12.0
    return schedule


def direct_dca(asset_return: pd.Series, contributions: pd.Series) -> tuple[pd.Series, dict]:
    """Invest every payment directly in the asset; dividends are in its return."""
    wealth = ANNUAL_CONTRIBUTION / 12.0
    values = []
    cash_flows = [(asset_return.index[0], -wealth)]
    for position, (stamp, daily_return) in enumerate(asset_return.items()):
        if position:
            wealth *= 1.0 + float(daily_return)
        contribution = float(contributions.loc[stamp])
        if contribution:
            wealth += contribution
            cash_flows.append((stamp, -contribution))
        values.append(wealth)
    cash_flows.append((asset_return.index[-1], wealth))
    nav = (1.0 + asset_return).cumprod()
    nav.iloc[0] = 1.0
    return nav, {"terminal": wealth, "xirr": xirr(cash_flows)}


def quantiles(values: pd.Series) -> dict:
    return {
        "min": float(values.min()),
        "p10": float(values.quantile(0.10)),
        "median": float(values.median()),
        "p90": float(values.quantile(0.90)),
        "max": float(values.max()),
    }


def fixed_return_fv(annual_rate: float) -> float:
    monthly_rate = (1.0 + annual_rate) ** (1.0 / 12.0) - 1.0
    payment = ANNUAL_CONTRIBUTION / 12.0
    months = YEARS * 12
    return payment * ((1.0 + monthly_rate) ** months - 1.0) / monthly_rate


def main() -> int:
    source = pd.read_csv(SOURCE, parse_dates=["date"], index_col="date")
    strategy_return = source["reverse_no_vault"].pct_change().fillna(0.0)
    spy_return = source["spy_hold"].pct_change().fillna(0.0)
    rates = source["treasury_rate"]

    first_sessions = source.index.to_series().groupby(source.index.to_period("M")).first()
    cohorts = []
    for start in first_sessions:
        target = start + pd.DateOffset(years=YEARS)
        if target > source.index[-1]:
            continue
        end_pos = source.index.searchsorted(target, side="right") - 1
        if end_pos < 0 or end_pos >= len(source.index):
            continue
        end = source.index[end_pos]
        window_return = strategy_return.loc[start:end].copy()
        window_return.iloc[0] = 0.0
        window_rates = rates.loc[start:end]
        contributions = exact_monthly_schedule(window_return.index)
        path, _, stats = simulate(
            window_return, window_rates, contributions,
            initial=ANNUAL_CONTRIBUTION / 12.0, harvest_share=0.10)
        water = underwater_stats(path["combined_flow_adjusted_nav"])
        window_spy_return = spy_return.loc[start:end].copy()
        window_spy_return.iloc[0] = 0.0
        spy_nav, spy_stats = direct_dca(window_spy_return, contributions)
        spy_water = underwater_stats(spy_nav)
        cohorts.append({
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
            "terminal_wealth": stats["terminal"],
            "xirr": stats["xirr"],
            "max_drawdown": stats["max_combined_drawdown_flow_adjusted"],
            "total_contributed": stats["total_cash_including_initial"],
            "spy_terminal_wealth": spy_stats["terminal"],
            "spy_xirr": spy_stats["xirr"],
            "spy_max_drawdown": float((spy_nav / spy_nav.cummax() - 1.0).min()),
            "spy_longest_underwater_years": spy_water["longest_underwater_years"],
            **water,
        })

    frame = pd.DataFrame(cohorts)
    results = {
        "method": {
            "data_start": source.index[0].date().isoformat(),
            "data_end": source.index[-1].date().isoformat(),
            "cohorts": len(frame),
            "cohort_frequency": "monthly starts; overlapping",
            "horizon_years": YEARS,
            "contribution": "$833.33 monthly ($10,000/year; 240 payments)",
            "total_contributed": ANNUAL_CONTRIBUTION * YEARS,
            "strategy": "reverse leverage plus 10% annual-profit Treasury reserve and 20/30/50/80% staged deployment",
            "underwater_definition": "flow-adjusted combined NAV below its prior high; contributions cannot create a recovery",
        },
        "rolling_20y": {
            "terminal_wealth": quantiles(frame["terminal_wealth"]),
            "xirr": quantiles(frame["xirr"]),
            "max_drawdown": quantiles(frame["max_drawdown"]),
            "longest_underwater_years": quantiles(frame["longest_underwater_years"]),
            "time_underwater_share": quantiles(frame["time_underwater_share"]),
            "time_below_10pct_share": quantiles(frame["time_below_10pct_share"]),
            "time_below_20pct_share": quantiles(frame["time_below_20pct_share"]),
            "time_below_50pct_share": quantiles(frame["time_below_50pct_share"]),
            "ending_underwater_share": float(frame["ends_underwater"].mean()),
            "beats_spy_terminal_share": float(
                (frame["terminal_wealth"] > frame["spy_terminal_wealth"]).mean()),
        },
        "spy_x1_rolling_20y": {
            "terminal_wealth": quantiles(frame["spy_terminal_wealth"]),
            "xirr": quantiles(frame["spy_xirr"]),
            "max_drawdown": quantiles(frame["spy_max_drawdown"]),
            "longest_underwater_years": quantiles(
                frame["spy_longest_underwater_years"]),
        },
        "fixed_return_scenarios": {
            f"{rate:.2%}": fixed_return_fv(rate)
            for rate in (0.05, 0.08, 0.10, 0.12, 0.142614, 0.149688)
        },
        "cohorts": cohorts,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in results.items() if key != "cohorts"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
