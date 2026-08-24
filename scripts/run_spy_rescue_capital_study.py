"""Test an unlevered rescue tranche at a 60% strategy drawdown."""

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


def simulate(asset_return: pd.Series, rates: pd.Series,
             contributions: pd.Series) -> tuple[pd.DataFrame, dict]:
    index = asset_return.index
    stamps = index.to_series()
    days = stamps.diff().dt.days.fillna(0.0)
    known_rate = rates.shift(1).ffill().bfill()
    year_end = stamps.dt.year.ne(stamps.shift(-1).dt.year) & stamps.dt.month.eq(12)

    opening = ANNUAL_CONTRIBUTION / 12.0
    main = opening
    regular_reserve = 0.0
    rescue_savings = 0.0
    rescue_active = 0.0
    rescue_basis = 0.0
    strategy_nav = strategy_peak = 1.0
    combined_nav = combined_peak = 1.0
    leverage = 5.0
    leverage_rung = 0
    reserve_fired: set[float] = set()
    rescue_fired = False
    annual_profit = 0.0
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
            annual_profit += profit
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
            if signal_drawdown <= -0.50:
                leverage_rung = max(leverage_rung, 3)
            elif signal_drawdown <= -0.30:
                leverage_rung = max(leverage_rung, 2)
            elif signal_drawdown <= -0.10:
                leverage_rung = max(leverage_rung, 1)
        target_leverage = (5.0, 3.0, 3.0, 1.0)[leverage_rung]

        for threshold, fraction in zip(THRESHOLDS, DEPLOY_FRACTIONS):
            if threshold in reserve_fired or signal_drawdown > -threshold:
                continue
            reserve_fired.add(threshold)
            amount = regular_reserve * fraction
            regular_reserve -= amount
            main += amount

        if signal_drawdown <= -0.60 and not rescue_fired:
            rescue_fired = True
            # Match the capital currently left in the complete account.  Reuse
            # protected rescue savings first; only the shortfall is new capital.
            required = main + regular_reserve + rescue_savings + rescue_active
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

        if bool(year_end.iloc[position]):
            harvest = min(max(annual_profit, 0.0) * 0.10, max(main, 0.0))
            if harvest:
                main -= harvest
                regular_reserve += harvest
            annual_profit = 0.0

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
        "rescue_exit_share": float(
            (frame["rescue_exits"] >= frame["rescue_calls"]).mean()),
    }


def schedule_30y(index: pd.DatetimeIndex) -> pd.Series:
    schedule = pd.Series(0.0, index=index)
    for month in range(1, 360):
        position = index.searchsorted(index[0] + pd.DateOffset(months=month))
        schedule.iloc[position] += ANNUAL_CONTRIBUTION / 12.0
    return schedule


def main() -> int:
    source = pd.read_csv(SOURCE, parse_dates=["date"], index_col="date")
    returns = source["spy_hold"].pct_change().fillna(0.0)
    rates = source["treasury_rate"]

    records = []
    starts = source.index.to_series().groupby(source.index.to_period("M")).first()
    for start in starts:
        target = start + pd.DateOffset(years=YEARS)
        if target > source.index[-1]:
            continue
        end = source.index[source.index.searchsorted(target, side="right") - 1]
        window_return = returns.loc[start:end].copy()
        window_return.iloc[0] = 0.0
        _, stats = simulate(
            window_return, rates.loc[start:end], exact_monthly_schedule(window_return.index))
        records.append({"start": start.date().isoformat(),
                        "end": end.date().isoformat(), **stats})

    start_30, end_30 = pd.Timestamp("1996-08-21"), pd.Timestamp("2026-08-21")
    return_30 = returns.loc[start_30:end_30].copy()
    return_30.iloc[0] = 0.0
    path_30, stats_30 = simulate(
        return_30, rates.loc[start_30:end_30], schedule_30y(return_30.index))
    trough = path_30["combined_drawdown"].idxmin()
    peak = path_30.loc[:trough, "combined_flow_adjusted_nav"].idxmax()
    stats_30["max_drawdown_peak"] = peak.date().isoformat()
    stats_30["max_drawdown_trough"] = trough.date().isoformat()
    stats_30["wealth_at_peak"] = float(path_30.loc[peak, "combined_wealth"])
    stats_30["wealth_at_trough"] = float(path_30.loc[trough, "combined_wealth"])

    result = {
        "method": {
            "base_rule": "5x at high, 3x at -10/-30%, 1x at -50%; reset only at prior high",
            "rescue": "at -60%, buy 1x SPY with capital equal to total account then remaining; exit entire tranche at +10% total return",
            "rescue_funding": "reuse protected prior rescue proceeds, then record external top-up as a cash flow",
            "regular_savings": "10% positive calendar-year trading P&L; regular 20/30/50/80 deployment ladder",
            "funding": "prior-known DGS3MO + 1% on borrowed base exposure",
        },
        "rolling_20y": summarize(pd.DataFrame(records)),
        "matched_30y": stats_30,
        "cohorts": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cohorts"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
