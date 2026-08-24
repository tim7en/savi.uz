"""Test an unlevered rescue tranche at a configurable strategy drawdown."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_spy_20y_contribution_study import (  # noqa: E402
    ANNUAL_CONTRIBUTION,
    YEARS,
    exact_monthly_schedule,
    quantiles,
    underwater_stats,
)
from run_spy_staged_funding import (  # noqa: E402
    DEPLOY_FRACTIONS,
    THRESHOLDS,
    xirr,
)


SOURCE = ROOT / "out/strategy/spy_reverse_vault/daily.csv"
OUTPUT = ROOT / "out/strategy/spy_rescue_capital/results.json"
OUTPUT_DAILY = ROOT / "out/strategy/spy_rescue_capital/matched_30y.csv"


def simulate(asset_return: pd.Series, rates: pd.Series,
             contributions: pd.Series,
             levels: tuple[float, ...] = (5.0, 3.0, 3.0, 1.0),
             leverage_thresholds: tuple[float, ...] = (0.10, 0.30, 0.50),
             reserve_thresholds: tuple[float, ...] = THRESHOLDS,
             reserve_fractions: tuple[float, ...] = DEPLOY_FRACTIONS,
             rescue_multiple: float = 1.0,
             profit_sweep_frequency: str = "annual",
             rescue_threshold: float = 0.60,
             ) -> tuple[pd.DataFrame, dict]:
    index = asset_return.index
    stamps = index.to_series()
    days = stamps.diff().dt.days.fillna(0.0)
    known_rate = rates.shift(1).ffill().bfill()
    year_end = stamps.dt.year.ne(stamps.shift(-1).dt.year) & stamps.dt.month.eq(12)
    quarter_end = stamps.dt.quarter.ne(stamps.shift(-1).dt.quarter)
    if profit_sweep_frequency not in {"annual", "quarterly"}:
        raise ValueError("profit_sweep_frequency must be annual or quarterly")
    harvest_boundary = quarter_end if profit_sweep_frequency == "quarterly" else year_end

    opening = ANNUAL_CONTRIBUTION / 12.0
    main = opening
    regular_reserve = 0.0
    rescue_savings = 0.0
    rescue_active = 0.0
    rescue_basis = 0.0
    strategy_nav = strategy_peak = 1.0
    combined_nav = combined_peak = 1.0
    leverage = levels[0]
    leverage_rung = 0
    reserve_fired: set[float] = set()
    rescue_fired = False
    period_profit = 0.0
    rescue_calls = rescue_exits = 0
    total_rescue_external = 0.0
    cash_flows = [(index[0], -opening)]
    rows = []

    for position, stamp in enumerate(index):
        prior_combined = main + regular_reserve + rescue_savings + rescue_active
        if position:
            day_count = float(days.iloc[position])
            rate = float(known_rate.iloc[position])
            regular_reserve *= 1.0 + rate * day_count / 365.0
            rescue_savings *= 1.0 + rate * day_count / 365.0
            rescue_active *= 1.0 + float(asset_return.iloc[position])

            financing = max(leverage - 1.0, 0.0) * (rate + 0.01) * day_count / 365.0
            trading_return = leverage * float(asset_return.iloc[position]) - financing
            profit = main * trading_return
            main = max(main + profit, 0.0)
            period_profit += profit
            strategy_nav *= 1.0 + trading_return

            combined_before_flow = main + regular_reserve + rescue_savings + rescue_active
            if prior_combined > 0.0:
                combined_nav *= combined_before_flow / prior_combined

        # The rescue tranche leaves market risk as soon as its own total return
        # reaches +10%; proceeds remain protected for a future fear episode.
        if rescue_active > 0.0 and rescue_active >= rescue_basis * 1.10:
            rescue_savings += rescue_active
            rescue_active = 0.0
            rescue_basis = 0.0
            rescue_exits += 1

        contribution = float(contributions.iloc[position])
        if contribution:
            main += contribution
            cash_flows.append((stamp, -contribution))

        recovered = strategy_nav >= strategy_peak * (1.0 - 1e-12)
        if recovered:
            strategy_peak = max(strategy_peak, strategy_nav)
            leverage_rung = 0
            reserve_fired.clear()
            rescue_fired = False
        signal_drawdown = strategy_nav / strategy_peak - 1.0

        if not recovered:
            for threshold_position in range(len(leverage_thresholds) - 1, -1, -1):
                if signal_drawdown <= -leverage_thresholds[threshold_position]:
                    leverage_rung = max(leverage_rung, threshold_position + 1)
                    break
        target_leverage = levels[leverage_rung]

        for threshold, fraction in zip(reserve_thresholds, reserve_fractions):
            if threshold in reserve_fired or signal_drawdown > -threshold:
                continue
            reserve_fired.add(threshold)
            amount = regular_reserve * fraction
            regular_reserve -= amount
            main += amount

        if signal_drawdown <= -rescue_threshold and not rescue_fired:
            rescue_fired = True
            # Match the capital currently left in the complete account.  Reuse
            # protected rescue savings first; only the shortfall is new capital.
            required = ((main + regular_reserve + rescue_savings + rescue_active)
                        * rescue_multiple)
            reused_protected = min(rescue_savings, required)
            rescue_savings -= reused_protected
            remaining = required - reused_protected
            reused_regular = min(regular_reserve, remaining)
            regular_reserve -= reused_regular
            external = remaining - reused_regular
            if external > 0.0:
                cash_flows.append((stamp, -external))
                total_rescue_external += external
            rescue_active += required
            rescue_basis += required
            rescue_calls += 1

        if bool(harvest_boundary.iloc[position]):
            harvest = min(max(period_profit, 0.0) * 0.10, max(main, 0.0))
            if harvest:
                main -= harvest
                regular_reserve += harvest
            period_profit = 0.0

        combined_peak = max(combined_peak, combined_nav)
        combined_drawdown = combined_nav / combined_peak - 1.0
        rows.append({
            "combined_wealth": main + regular_reserve + rescue_savings + rescue_active,
            "main": main,
            "regular_reserve": regular_reserve,
            "rescue_savings": rescue_savings,
            "rescue_active": rescue_active,
            "strategy_drawdown": signal_drawdown,
            "combined_flow_adjusted_nav": combined_nav,
            "combined_drawdown": combined_drawdown,
            "applied_leverage": leverage,
            "target_leverage": target_leverage,
        })
        leverage = target_leverage

    path = pd.DataFrame(rows, index=index)
    terminal = float(path["combined_wealth"].iloc[-1])
    cash_flows.append((index[-1], terminal))
    water = underwater_stats(path["combined_flow_adjusted_nav"])
    stats = {
        "terminal_wealth": terminal,
        "xirr": xirr(cash_flows),
        "max_drawdown": float(path["combined_drawdown"].min()),
        "total_scheduled_contributions": float(opening + contributions.sum()),
        "total_rescue_external": total_rescue_external,
        "total_external_capital": float(opening + contributions.sum() + total_rescue_external),
        "rescue_calls": rescue_calls,
        "rescue_exits": rescue_exits,
        "ending_active_rescue": float(path["rescue_active"].iloc[-1]),
        "ending_rescue_savings": float(path["rescue_savings"].iloc[-1]),
        "mean_applied_leverage": float(path["applied_leverage"].mean()),
        "time_at_3x": float(path["applied_leverage"].eq(3.0).mean()),
        "time_at_2x": float(path["applied_leverage"].eq(2.0).mean()),
        "time_at_1x": float(path["applied_leverage"].eq(1.0).mean()),
        **water,
    }
    return path, stats


def summarize(frame: pd.DataFrame) -> dict:
    return {
        "terminal_wealth": quantiles(frame["terminal_wealth"]),
        "xirr": quantiles(frame["xirr"]),
        "max_drawdown": quantiles(frame["max_drawdown"]),
        "longest_underwater_years": quantiles(frame["longest_underwater_years"]),
        "total_rescue_external": quantiles(frame["total_rescue_external"]),
        "rescue_calls": quantiles(frame["rescue_calls"]),
        "mean_applied_leverage": quantiles(frame["mean_applied_leverage"]),
        "rescue_exit_share": float(
            (frame["rescue_exits"] >= frame["rescue_calls"]).mean()),
    }


def schedule_30y(index: pd.DatetimeIndex) -> pd.Series:
    schedule = pd.Series(0.0, index=index)
    for month in range(1, 360):
        position = index.searchsorted(index[0] + pd.DateOffset(months=month))
        schedule.iloc[position] += ANNUAL_CONTRIBUTION / 12.0
    return schedule


def add_drawdown_dates(path: pd.DataFrame, stats: dict) -> None:
    trough = path["combined_drawdown"].idxmin()
    peak = path.loc[:trough, "combined_flow_adjusted_nav"].idxmax()
    stats["max_drawdown_peak"] = peak.date().isoformat()
    stats["max_drawdown_trough"] = trough.date().isoformat()
    stats["wealth_at_peak"] = float(path.loc[peak, "combined_wealth"])
    stats["wealth_at_trough"] = float(path.loc[trough, "combined_wealth"])


def simulate_spy(asset_return: pd.Series,
                 contributions: pd.Series) -> pd.DataFrame:
    """Build the same-contribution unlevered SPY path for plotting."""
    opening = ANNUAL_CONTRIBUTION / 12.0
    wealth = opening
    nav = peak = 1.0
    rows = []
    for position, stamp in enumerate(asset_return.index):
        if position:
            prior = wealth
            wealth *= 1.0 + float(asset_return.iloc[position])
            if prior > 0.0:
                nav *= wealth / prior
        contribution = float(contributions.iloc[position])
        if contribution:
            wealth += contribution
        peak = max(peak, nav)
        rows.append({
            "spy_wealth": wealth,
            "spy_flow_adjusted_nav": nav,
            "spy_drawdown": nav / peak - 1.0,
        })
    return pd.DataFrame(rows, index=asset_return.index)


def main() -> int:
    source = pd.read_csv(SOURCE, parse_dates=["date"], index_col="date")
    returns = source["spy_hold"].pct_change().fillna(0.0)
    rates = source["treasury_rate"]

    records = []
    records_5_2_1 = []
    records_corrected_half = []
    records_corrected_equal = []
    starts = source.index.to_series().groupby(source.index.to_period("M")).first()
    for start in starts:
        target = start + pd.DateOffset(years=YEARS)
        if target > source.index[-1]:
            continue
        end = source.index[source.index.searchsorted(target, side="right") - 1]
        window_return = returns.loc[start:end].copy()
        window_return.iloc[0] = 0.0
        contribution_schedule = exact_monthly_schedule(window_return.index)
        _, stats = simulate(window_return, rates.loc[start:end], contribution_schedule)
        _, stats_5_2_1 = simulate(
            window_return, rates.loc[start:end], contribution_schedule,
            (5.0, 2.0, 1.0), (0.20, 0.50))
        _, stats_corrected_half = simulate(
            window_return, rates.loc[start:end], contribution_schedule,
            (3.0, 2.0, 1.0), (0.30, 0.50),
            (0.10, 0.20, 0.30, 0.40), (1 / 4, 1 / 3, 1 / 2, 1.0),
            0.50, "annual", 0.50)
        _, stats_corrected_equal = simulate(
            window_return, rates.loc[start:end], contribution_schedule,
            (3.0, 2.0, 1.0), (0.30, 0.50),
            (0.10, 0.20, 0.30, 0.40), (1 / 4, 1 / 3, 1 / 2, 1.0),
            1.0, "annual", 0.50)
        records.append({"start": start.date().isoformat(),
                        "end": end.date().isoformat(), **stats})
        records_5_2_1.append({"start": start.date().isoformat(),
                              "end": end.date().isoformat(), **stats_5_2_1})
        records_corrected_half.append({"start": start.date().isoformat(),
                                       "end": end.date().isoformat(),
                                       **stats_corrected_half})
        records_corrected_equal.append({"start": start.date().isoformat(),
                                        "end": end.date().isoformat(),
                                        **stats_corrected_equal})

    start_30, end_30 = pd.Timestamp("1996-08-21"), pd.Timestamp("2026-08-21")
    return_30 = returns.loc[start_30:end_30].copy()
    return_30.iloc[0] = 0.0
    path_30, stats_30 = simulate(
        return_30, rates.loc[start_30:end_30], schedule_30y(return_30.index))
    add_drawdown_dates(path_30, stats_30)

    path_30_5_2_1, stats_30_5_2_1 = simulate(
        return_30, rates.loc[start_30:end_30], schedule_30y(return_30.index),
        (5.0, 2.0, 1.0), (0.20, 0.50))
    add_drawdown_dates(path_30_5_2_1, stats_30_5_2_1)

    schedule = schedule_30y(return_30.index)
    path_30_corrected_half, stats_30_corrected_half = simulate(
        return_30, rates.loc[start_30:end_30], schedule,
        (3.0, 2.0, 1.0), (0.30, 0.50),
        (0.10, 0.20, 0.30, 0.40), (1 / 4, 1 / 3, 1 / 2, 1.0),
        0.50, "annual", 0.50)
    add_drawdown_dates(path_30_corrected_half, stats_30_corrected_half)
    path_30_corrected_equal, stats_30_corrected_equal = simulate(
        return_30, rates.loc[start_30:end_30], schedule,
        (3.0, 2.0, 1.0), (0.30, 0.50),
        (0.10, 0.20, 0.30, 0.40), (1 / 4, 1 / 3, 1 / 2, 1.0),
        1.0, "annual", 0.50)
    add_drawdown_dates(path_30_corrected_equal, stats_30_corrected_equal)

    result = {
        "method": {
            "base_rule": "5x at high, 3x at -10/-30%, 1x at -50%; reset only at prior high",
            "new_rule": "5x at high, 2x at -20%, 1x at -50%; reset only at prior high",
            "corrected_rule": "3x at high, 2x at -30%, 1x at -50%; reset only at prior high",
            "profit_sweep": "10% of positive calendar-year trading P&L at year-end only",
            "rescue": "at -50%, buy 1x SPY with outside capital equal to total account then remaining; exit entire tranche at +10% total return",
            "rescue_funding": "reuse protected prior rescue proceeds, then record external top-up as a cash flow",
            "regular_savings": "corrected variants sweep at year-end and deploy 25%/33%/50%/100% of remaining reserve at -10/-20/-30/-40%",
            "funding": "prior-known DGS3MO + 1% on borrowed base exposure",
        },
        "rolling_20y": summarize(pd.DataFrame(records)),
        "matched_30y": stats_30,
        "new_5_2_1_rolling_20y": summarize(pd.DataFrame(records_5_2_1)),
        "new_5_2_1_matched_30y": stats_30_5_2_1,
        "corrected_3_2_1_half_balance_rolling_20y": summarize(
            pd.DataFrame(records_corrected_half)),
        "corrected_3_2_1_half_balance_matched_30y": stats_30_corrected_half,
        "corrected_3_2_1_equal_balance_rolling_20y": summarize(
            pd.DataFrame(records_corrected_equal)),
        "corrected_3_2_1_equal_balance_matched_30y": stats_30_corrected_equal,
        "cohorts": records,
        "new_5_2_1_cohorts": records_5_2_1,
        "corrected_3_2_1_half_balance_cohorts": records_corrected_half,
        "corrected_3_2_1_equal_balance_cohorts": records_corrected_equal,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    spy_path = simulate_spy(return_30, schedule)
    comparison = pd.DataFrame(index=return_30.index)
    comparison.index.name = "date"
    comparison["spy_wealth"] = spy_path["spy_wealth"]
    comparison["strategy_wealth"] = path_30_corrected_equal["combined_wealth"]
    comparison["spy_flow_adjusted_nav"] = spy_path["spy_flow_adjusted_nav"]
    comparison["strategy_flow_adjusted_nav"] = path_30_corrected_equal["combined_flow_adjusted_nav"]
    comparison["spy_drawdown"] = spy_path["spy_drawdown"]
    comparison["strategy_drawdown"] = path_30_corrected_equal["combined_drawdown"]
    comparison["strategy_leverage"] = path_30_corrected_equal["applied_leverage"]
    comparison["strategy_reserve"] = path_30_corrected_equal["regular_reserve"]
    comparison["strategy_rescue"] = path_30_corrected_equal["rescue_active"]
    comparison.to_csv(OUTPUT_DAILY)
    print(json.dumps({key: value for key, value in result.items()
                      if not key.endswith("cohorts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
