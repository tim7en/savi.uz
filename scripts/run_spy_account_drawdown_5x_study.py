"""Rolling 20-year study of account-drawdown-controlled 5x SPY leverage."""

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
    direct_dca,
    exact_monthly_schedule,
    quantiles,
    underwater_stats,
)
from run_spy_staged_funding import THRESHOLDS, DEPLOY_FRACTIONS, xirr  # noqa: E402


SOURCE = ROOT / "out/strategy/spy_reverse_vault/daily.csv"
OUTPUT = ROOT / "out/strategy/spy_account_drawdown_5x/results.json"


def simulate_account_rule(asset_return: pd.Series, rates: pd.Series,
                          contributions: pd.Series,
                          levels: tuple[float, ...],
                          drawdown_driver: str,
                          contribution_destination: str,
                          leverage_thresholds: tuple[float, ...] = (0.10, 0.30, 0.50),
                          ) -> tuple[pd.DataFrame, dict]:
    """Run leverage and reserve rungs from flow-adjusted combined-account DD."""
    index = asset_return.index
    stamps = index.to_series()
    days = stamps.diff().dt.days.fillna(0.0)
    known_rate = rates.shift(1).ffill().bfill()
    year_end = stamps.dt.year.ne(stamps.shift(-1).dt.year) & stamps.dt.month.eq(12)

    opening = ANNUAL_CONTRIBUTION / 12.0
    main, reserve = opening, 0.0
    performance_nav, performance_peak = 1.0, 1.0
    trading_nav, signal_peak = 1.0, 1.0
    applied_leverage = levels[0]
    rung = 0
    fired: set[float] = set()
    annual_profit = 0.0
    liquidated = False
    deployments = 0
    cash_flows = [(index[0], -opening)]
    rows = []

    for position, stamp in enumerate(index):
        prior_combined = main + reserve
        if position:
            reserve *= 1.0 + float(known_rate.iloc[position]) * float(days.iloc[position]) / 365.0
            financing = max(applied_leverage - 1.0, 0.0) * (
                float(known_rate.iloc[position]) + 0.01
            ) * float(days.iloc[position]) / 365.0
            trading_return = applied_leverage * float(asset_return.iloc[position]) - financing
            trading_nav *= 1.0 + trading_return
            profit = main * trading_return
            main += profit
            annual_profit += profit
            if main <= 0.0:
                main = 0.0
                liquidated = True

            combined_before_flow = main + reserve
            if prior_combined > 0.0:
                performance_nav *= combined_before_flow / prior_combined
            else:
                performance_nav = 0.0

        contribution = float(contributions.iloc[position])
        if contribution:
            if contribution_destination == "trading_sleeve":
                main += contribution
            else:
                reserve += contribution
            cash_flows.append((stamp, -contribution))

        performance_peak = max(performance_peak, performance_nav)
        account_drawdown = performance_nav / performance_peak - 1.0
        signal_nav = (performance_nav if drawdown_driver == "combined_account"
                      else trading_nav)
        recovered = signal_nav >= signal_peak * (1.0 - 1e-12)
        if recovered:
            signal_peak = max(signal_peak, signal_nav)
            rung = 0
            fired.clear()
        signal_drawdown = signal_nav / signal_peak - 1.0

        if not recovered:
            for threshold_position in range(len(leverage_thresholds) - 1, -1, -1):
                if signal_drawdown <= -leverage_thresholds[threshold_position]:
                    rung = max(rung, threshold_position + 1)
                    break
        target_leverage = levels[rung]

        for threshold, fraction in zip(THRESHOLDS, DEPLOY_FRACTIONS):
            if threshold in fired or signal_drawdown > -threshold:
                continue
            fired.add(threshold)
            amount = reserve * fraction
            if amount > 0.0:
                reserve -= amount
                main += amount
                deployments += 1

        if bool(year_end.iloc[position]):
            harvest = min(max(annual_profit, 0.0) * 0.10, max(main, 0.0))
            if harvest:
                main -= harvest
                reserve += harvest
            annual_profit = 0.0

        rows.append({
            "combined_wealth": main + reserve,
            "main": main,
            "reserve": reserve,
            "performance_nav": performance_nav,
            "account_drawdown": account_drawdown,
            "signal_drawdown": signal_drawdown,
            "applied_leverage": applied_leverage,
            "target_leverage": target_leverage,
        })
        applied_leverage = target_leverage if main > 0.0 else 0.0

    path = pd.DataFrame(rows, index=index)
    terminal = float(path["combined_wealth"].iloc[-1])
    cash_flows.append((index[-1], terminal))
    water = underwater_stats(path["performance_nav"])
    stats = {
        "terminal_wealth": terminal,
        "xirr": xirr(cash_flows),
        "max_drawdown": float(path["account_drawdown"].min()),
        "mean_applied_leverage": float(path["applied_leverage"].mean()),
        "liquidated": liquidated,
        "deployments": deployments,
        **water,
    }
    return path, stats


def summarize(frame: pd.DataFrame) -> dict:
    return {
        "terminal_wealth": quantiles(frame["terminal_wealth"]),
        "xirr": quantiles(frame["xirr"]),
        "max_drawdown": quantiles(frame["max_drawdown"]),
        "longest_underwater_years": quantiles(frame["longest_underwater_years"]),
        "time_underwater_share": quantiles(frame["time_underwater_share"]),
        "time_below_10pct_share": quantiles(frame["time_below_10pct_share"]),
        "time_below_20pct_share": quantiles(frame["time_below_20pct_share"]),
        "time_below_50pct_share": quantiles(frame["time_below_50pct_share"]),
        "mean_applied_leverage": quantiles(frame["mean_applied_leverage"]),
        "liquidation_share": float(frame["liquidated"].mean()),
        "beats_spy_terminal_share": float(
            (frame["terminal_wealth"] > frame["spy_terminal_wealth"]).mean()),
    }


def main() -> int:
    source = pd.read_csv(SOURCE, parse_dates=["date"], index_col="date")
    spy_return = source["spy_hold"].pct_change().fillna(0.0)
    rates = source["treasury_rate"]
    variants = {
        "literal_profit_reserve_5_3_3_1": {
            "levels": (5.0, 3.0, 3.0, 1.0), "driver": "trading_sleeve",
            "contribution_destination": "trading_sleeve"},
        "sensitivity_profit_reserve_5_3_2_1": {
            "levels": (5.0, 3.0, 2.0, 1.0), "driver": "trading_sleeve",
            "contribution_destination": "trading_sleeve",
            "thresholds": (0.10, 0.30, 0.50)},
        "fear_relever_profit_reserve_5_3_3_1_3": {
            "levels": (5.0, 3.0, 3.0, 1.0, 3.0), "driver": "trading_sleeve",
            "contribution_destination": "trading_sleeve",
            "thresholds": (0.10, 0.30, 0.50, 0.60)},
    }
    variants["literal_profit_reserve_5_3_3_1"]["thresholds"] = (0.10, 0.30, 0.50)
    records = {name: [] for name in variants}
    spy_records = []

    starts = source.index.to_series().groupby(source.index.to_period("M")).first()
    for start in starts:
        target = start + pd.DateOffset(years=YEARS)
        if target > source.index[-1]:
            continue
        end = source.index[source.index.searchsorted(target, side="right") - 1]
        window_return = spy_return.loc[start:end].copy()
        window_return.iloc[0] = 0.0
        window_rates = rates.loc[start:end]
        contributions = exact_monthly_schedule(window_return.index)

        spy_nav, spy_stats = direct_dca(window_return, contributions)
        spy_water = underwater_stats(spy_nav)
        spy_record = {
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
            "terminal_wealth": spy_stats["terminal"],
            "xirr": spy_stats["xirr"],
            "max_drawdown": float((spy_nav / spy_nav.cummax() - 1.0).min()),
            **spy_water,
        }
        spy_records.append(spy_record)

        for name, specification in variants.items():
            _, stats = simulate_account_rule(
                window_return, window_rates, contributions,
                specification["levels"], specification["driver"],
                specification["contribution_destination"],
                specification["thresholds"])
            records[name].append({
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
                "spy_terminal_wealth": spy_stats["terminal"],
                **stats,
            })

    spy_frame = pd.DataFrame(spy_records)
    results = {
        "method": {
            "data_start": source.index[0].date().isoformat(),
            "data_end": source.index[-1].date().isoformat(),
            "cohorts": len(spy_frame),
            "horizon_years": YEARS,
            "total_contributed": ANNUAL_CONTRIBUTION * YEARS,
            "contribution_timing": "$833.33 monthly into the trading sleeve; opening payment plus 239 deposits",
            "leverage_signal": "selected flow-adjusted account NAV, close-known and applied next session",
            "reset": "all drawdown rungs remain latched until the account regains its prior performance high",
            "reserve": "only 10% of positive annual trading P&L enters savings; Treasury yield; deploy 25%/one-third/half/all at account -20/-30/-50/-80%",
            "funding": "prior-known DGS3MO + 1% on borrowed exposure",
        },
        "spy_x1": {
            "terminal_wealth": quantiles(spy_frame["terminal_wealth"]),
            "xirr": quantiles(spy_frame["xirr"]),
            "max_drawdown": quantiles(spy_frame["max_drawdown"]),
            "longest_underwater_years": quantiles(
                spy_frame["longest_underwater_years"]),
        },
        "variants": {
            name: {"levels": list(variants[name]["levels"]),
                   "driver": variants[name]["driver"],
                   "contribution_destination": variants[name]["contribution_destination"],
                   "thresholds": list(variants[name]["thresholds"]),
                   **summarize(pd.DataFrame(items))}
            for name, items in records.items()
        },
        "cohorts": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in results.items() if key != "cohorts"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
