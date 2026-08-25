"""Test weekly far-OTM covered calls on unleveraged SPY and QQQ sleeves.

The primary test uses historical Alpha Vantage option-chain quotes from 2017.
At the last quoted session of each week it sells the next-week call whose delta
is closest to 5%, 10%, or 20%, using the quoted bid.  The option is held to
expiration.  Premium is attributed to Treasury; intrinsic value is the upside
given away.  Strategy signals are held fixed, so this is an overlay attribution
rather than a fully recursive re-optimization of the underlying strategy.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import build_spy_dca_dashboard as prices_io
from run_contribution_quality_strategy import performance_metrics


@dataclass(frozen=True)
class CallTrade:
    symbol: str
    issue_date: str
    expiration: str
    dte: int
    target_delta: float
    quoted_delta: float
    spot: float
    strike: float
    bid: float
    ask: float
    expiry_spot: float
    premium_yield: float
    payoff_yield: float
    underlying_return: float

    @property
    def expired_worthless(self) -> bool:
        return self.payoff_yield <= 1e-12

    @property
    def net_overlay_yield(self) -> float:
        return self.premium_yield - self.payoff_yield


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2017-01-03")
    parser.add_argument("--end", default="2024-09-30")
    parser.add_argument("--deltas", default="0.05,0.10,0.20")
    parser.add_argument("--options-db", type=Path, default=Path("data/options/alphavantage.db"))
    parser.add_argument("--price-cache", type=Path, default=Path(".cache/yahoo_daily"))
    parser.add_argument("--spy-daily", type=Path, default=Path("out/strategy/quality_ladder_cape_leverage/daily.csv"))
    parser.add_argument("--qqq-daily", type=Path, default=Path("out/strategy/qqq_dca_backtest/daily.csv"))
    parser.add_argument("--out", type=Path, default=Path("out/strategy/weekly_covered_calls"))
    parser.add_argument("--assignment-cost-bp", type=float, default=2.0)
    return parser.parse_args(argv)


def _asof(series: pd.Series, stamp: pd.Timestamp) -> float | None:
    values = series.loc[:stamp].dropna()
    return float(values.iloc[-1]) if not values.empty else None


def load_price_frame(symbol: str, cache: Path) -> pd.DataFrame:
    return prices_io.load_spy(cache / f"{symbol}.json")


def select_weekly_calls(db: Path, symbol: str, target_delta: float,
                        prices: pd.DataFrame, start: str, end: str) -> list[CallTrade]:
    connection = sqlite3.connect(f"file:{db.resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        dates = connection.execute(
            "SELECT observation_date,spot FROM av_daily WHERE symbol=? "
            "AND observation_date BETWEEN ? AND ? AND spot IS NOT NULL ORDER BY observation_date",
            (symbol, start, end),
        ).fetchall()
        daily = pd.DataFrame([dict(row) for row in dates])
        if daily.empty:
            return []
        daily["observation_date"] = pd.to_datetime(daily["observation_date"])
        issues = daily.groupby(daily["observation_date"].dt.to_period("W-FRI"))["observation_date"].max()
        trades = []
        previous_expiry = pd.Timestamp.min
        for issue in issues:
            if issue < previous_expiry:
                continue
            rows = connection.execute(
                "SELECT expiration,dte,strike,delta,bid,ask FROM av_contracts "
                "WHERE symbol=? AND observation_date=? AND side='call' "
                "AND dte BETWEEN 5 AND 9 AND delta BETWEEN 0.005 AND 0.35 "
                "AND bid>0 AND ask>=bid",
                (symbol, issue.date().isoformat()),
            ).fetchall()
            if not rows:
                continue
            candidates = pd.DataFrame([dict(row) for row in rows])
            candidates["expiration"] = pd.to_datetime(candidates["expiration"])
            candidates["expiry_distance"] = (candidates["expiration"] - issue).dt.days
            expiry = (
                candidates[["expiration", "expiry_distance"]]
                .drop_duplicates()
                .assign(distance=lambda x: (x["expiry_distance"] - 7).abs())
                .sort_values(["distance", "expiration"])
                .iloc[0]["expiration"]
            )
            candidates = candidates[candidates["expiration"] == expiry].copy()
            spot_row = daily.loc[daily["observation_date"] == issue, "spot"]
            if spot_row.empty:
                continue
            spot = float(spot_row.iloc[-1])
            candidates = candidates[candidates["strike"] >= spot]
            if candidates.empty:
                continue
            candidates["delta_distance"] = (candidates["delta"] - target_delta).abs()
            chosen = candidates.sort_values(["delta_distance", "strike"]).iloc[0]
            expiry_spot = _asof(prices["close"], expiry)
            issue_adjusted = _asof(prices["adjusted"], issue)
            expiry_adjusted = _asof(prices["adjusted"], expiry)
            if expiry_spot is None or issue_adjusted is None or expiry_adjusted is None:
                continue
            premium_yield = float(chosen["bid"]) / spot
            payoff_yield = max(expiry_spot - float(chosen["strike"]), 0.0) / spot
            trades.append(CallTrade(
                symbol=symbol, issue_date=issue.date().isoformat(),
                expiration=expiry.date().isoformat(), dte=int(chosen["dte"]),
                target_delta=target_delta, quoted_delta=float(chosen["delta"]),
                spot=spot, strike=float(chosen["strike"]), bid=float(chosen["bid"]),
                ask=float(chosen["ask"]), expiry_spot=expiry_spot,
                premium_yield=premium_yield, payoff_yield=payoff_yield,
                underlying_return=expiry_adjusted / issue_adjusted - 1.0,
            ))
            previous_expiry = expiry
        return trades
    finally:
        connection.close()


def summarize_trades(trades: list[CallTrade], assignment_cost_bp: float = 2.0) -> dict:
    if not trades:
        return {}
    premium = np.array([row.premium_yield for row in trades])
    payoff = np.array([row.payoff_yield for row in trades])
    assigned = payoff > 1e-12
    costs = assigned.astype(float) * assignment_cost_bp / 10_000.0
    net = premium - payoff - costs
    underlying = np.array([1.0 + row.underlying_return for row in trades])
    covered = underlying + premium - payoff - costs
    years = (
        pd.Timestamp(trades[-1].expiration) - pd.Timestamp(trades[0].issue_date)
    ).days / 365.2425
    return {
        "weeks": len(trades),
        "worthless": int((~assigned).sum()),
        "assigned_or_itm": int(assigned.sum()),
        "worthless_rate": float((~assigned).mean()),
        "premium_yield_sum": float(premium.sum()),
        "payoff_yield_sum": float(payoff.sum()),
        "assignment_cost_yield_sum": float(costs.sum()),
        "net_overlay_yield_sum": float(net.sum()),
        "median_premium_bp": float(np.median(premium) * 10_000.0),
        "mean_premium_bp": float(premium.mean() * 10_000.0),
        "mean_missed_upside_assigned": float(payoff[assigned].mean()) if assigned.any() else 0.0,
        "worst_missed_upside": float(payoff.max()),
        "profitable_option_weeks": int((net >= 0.0).sum()),
        "profitable_option_week_rate": float((net >= 0.0).mean()),
        "underlying_compound_return": float(underlying.prod() - 1.0),
        "covered_call_compound_return": float(covered.prod() - 1.0),
        "underlying_cagr": float(underlying.prod() ** (1.0 / years) - 1.0),
        "covered_call_cagr": float(covered.prod() ** (1.0 / years) - 1.0),
        "cagr_difference": float(covered.prod() ** (1.0 / years) - underlying.prod() ** (1.0 / years)),
        "average_quoted_delta": float(np.mean([row.quoted_delta for row in trades])),
        "average_dte": float(np.mean([row.dte for row in trades])),
        "average_bid_ask_spread_pct_of_premium": float(np.mean([
            (row.ask - row.bid) / row.bid for row in trades if row.bid > 0
        ])),
    }


def strategy_frame(path: Path, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    wealth = frame[f"{prefix}_wealth"].astype(float)
    performance = frame[f"{prefix}_performance_index"].astype(float)
    contribution_name = f"{prefix}_contribution"
    contribution = (
        frame[contribution_name].astype(float)
        if contribution_name in frame else frame.get("contribution", pd.Series(0.0, index=frame.index)).astype(float)
    )
    spy_value = frame[f"{prefix}_spy_weight"].astype(float) * wealth
    injection = frame[f"{prefix}_injection_core_spy"].astype(float)
    injection_level = frame[f"{prefix}_injection_weighted_leverage"].astype(float)
    eligible = spy_value - injection.where(injection_level > 1.0 + 1e-12, 0.0)
    return pd.DataFrame({
        "wealth": wealth,
        "performance_index": performance,
        "contribution": contribution,
        "eligible_fraction": (eligible / wealth).clip(0.0, 1.0),
    })


def load_regimes(path: Path, volatility_column: str) -> pd.DataFrame:
    frame = pd.read_csv(
        path, usecols=["date", "cape_known", volatility_column],
        parse_dates=["date"],
    ).set_index("date").sort_index()
    return frame.rename(columns={volatility_column: "volatility_percentile"}).astype(float)


def regime_is_eligible(cape: float | None, volatility_percentile: float | None,
                       policy: str) -> bool:
    cape_high = cape is not None and not pd.isna(cape) and cape >= 25.0
    cape_extreme = cape is not None and not pd.isna(cape) and cape > 35.0
    volatility_calm = (
        volatility_percentile is not None
        and not pd.isna(volatility_percentile)
        and volatility_percentile < 0.70
    )
    choices = {
        "always": True,
        "cape_high": cape_high,
        "cape_extreme": cape_extreme,
        "vix_calm": volatility_calm,
        "cape_high_or_vix_calm": cape_high or volatility_calm,
        "cape_extreme_or_vix_calm": cape_extreme or volatility_calm,
        "cape_high_and_vix_calm": cape_high and volatility_calm,
        "cape_extreme_and_vix_calm": cape_extreme and volatility_calm,
    }
    if policy not in choices:
        raise ValueError(f"unknown covered-call regime policy: {policy}")
    return choices[policy]


def filter_trades_by_regime(trades: list[CallTrade], regimes: pd.DataFrame,
                            policy: str) -> list[CallTrade]:
    selected = []
    for trade in trades:
        known = regimes.loc[:pd.Timestamp(trade.issue_date)].dropna(how="all")
        if known.empty:
            cape = volatility = None
        else:
            cape = known["cape_known"].iloc[-1]
            volatility = known["volatility_percentile"].iloc[-1]
        if regime_is_eligible(cape, volatility, policy):
            selected.append(trade)
    return selected


def apply_overlay(base: pd.DataFrame, trades: list[CallTrade],
                  prices: pd.DataFrame, assignment_cost_bp: float,
                  round_contracts: bool, *,
                  period_start: str | pd.Timestamp | None = None,
                  period_end: str | pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    if not trades and (period_start is None or period_end is None):
        raise ValueError("trades or an explicit period are required")
    start = pd.Timestamp(period_start) if period_start is not None else pd.Timestamp(trades[0].issue_date)
    requested_end = pd.Timestamp(period_end) if period_end is not None else pd.Timestamp(trades[-1].expiration)
    end = min(requested_end, base.index[-1])
    base = base.loc[start:end].copy()
    trades_by_issue = {pd.Timestamp(row.issue_date): row for row in trades}
    active: dict[pd.Timestamp, tuple[CallTrade, float]] = {}
    wealth = float(base["wealth"].iloc[0])
    initial = wealth
    perf = 1.0
    high = 1.0
    previous = wealth
    rows = []
    gross_premium = gross_payoff = assignment_cost = 0.0
    calls_written = assigned_calls = 0
    for position, (stamp, row) in enumerate(base.iterrows()):
        contribution = 0.0 if position == 0 else float(row["contribution"])
        if position:
            base_previous = float(base["wealth"].iloc[position - 1])
            base_factor = (
                (float(row["wealth"]) - contribution) / base_previous
                if base_previous > 0 else 1.0
            )
            wealth *= base_factor
        option_pnl = 0.0
        expiring = active.pop(stamp, None)
        if expiring is not None:
            trade, notional = expiring
            premium_cash = notional * trade.premium_yield
            payoff_cash = notional * trade.payoff_yield
            cost_cash = (
                notional * assignment_cost_bp / 10_000.0
                if trade.payoff_yield > 1e-12 else 0.0
            )
            option_pnl = premium_cash - payoff_cash - cost_cash
            wealth += option_pnl
            gross_premium += premium_cash
            gross_payoff += payoff_cash
            assignment_cost += cost_cash
            assigned_calls += int(trade.payoff_yield > 1e-12)
        pre_flow = wealth
        if position:
            perf *= pre_flow / previous
        wealth += contribution
        previous = wealth
        high = max(high, perf)
        issue = trades_by_issue.get(stamp)
        if issue is not None and pd.Timestamp(issue.expiration) <= end:
            notional = wealth * float(row["eligible_fraction"])
            if round_contracts:
                contract_notional = issue.spot * 100.0
                notional = math.floor(notional / contract_notional) * contract_notional
            if notional > 0:
                active[pd.Timestamp(issue.expiration)] = (issue, notional)
                calls_written += 1
        rows.append({
            "wealth": wealth, "performance_index": perf,
            "flow_adjusted_drawdown": perf / high - 1.0,
            "contribution": contribution, "option_pnl": option_pnl,
        })
    path = pd.DataFrame(rows, index=base.index)
    rates = pd.Series(0.0, index=path.index)
    metrics = performance_metrics(path, rates, initial)
    metrics.update({
        "calls_written": calls_written,
        "assigned_calls": assigned_calls,
        "gross_premium_to_treasury": gross_premium,
        "upside_paid_away": gross_payoff,
        "assignment_cost": assignment_cost,
        "net_option_pnl": gross_premium - gross_payoff - assignment_cost,
        "start_wealth": initial,
        "round_contracts": round_contracts,
    })
    return path, metrics


def main(argv=None) -> int:
    args = parse_args(argv)
    deltas = [float(value) for value in args.deltas.split(",")]
    strategy_specs = {
        "SPY": (args.spy_daily, "quality_dual_guard_vix_sma60", "vix_sma60_percentile"),
        "QQQ": (args.qqq_daily, "dashboard_quality_annual", "vxn_sma60_percentile"),
    }
    regime_policies = [
        "always", "cape_high", "cape_extreme", "vix_calm",
        "cape_high_or_vix_calm", "cape_extreme_or_vix_calm",
        "cape_high_and_vix_calm", "cape_extreme_and_vix_calm",
    ]
    result = {
        "sample": {"start": args.start, "end": args.end},
        "rules": {
            "schedule": "sell at the last quoted session of each week; choose the closest next-week expiration (5-9 DTE)",
            "strike": "closest quoted call delta to 5%, 10%, or 20%; strike must be at or above spot",
            "execution": "sell at quoted bid; hold to expiration; intrinsic value is paid away",
            "eligibility": "cover only the unleveraged index sleeve; leveraged injection lots and individual quality stocks are excluded",
            "premium_routing": "gross option premium attributed to Treasury",
            "assignment_cost": f"{args.assignment_cost_bp:g} bp of covered notional when the call expires ITM",
            "conditional_gate": "sell only when the selected CAPE/VIX regime is true at the issue date",
            "cape_high": "CAPE >= 25; extreme means CAPE > 35",
            "volatility_calm": "prior-known 60-session VIX/VXN percentile < 70%",
        },
        "chains": {}, "strategy_overlays": {}, "conditional": {}, "warnings": [
            "A standard ETF option contract represents 100 shares; fractional-notional results are not executable in a $10,000 SPY or QQQ account.",
            "ETF options are American-style; early assignment and ex-dividend exercise are omitted.",
            "Using the historical bid captures the entry spread, but commissions, taxes and stock repurchase slippage beyond the fixed assignment cost are omitted.",
            "The strategy overlay holds the underlying strategy signals and eligible weights fixed; option P&L does not recursively change future NAV gates or Treasury deployments.",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    overlay_daily = []
    conditional_daily = []
    all_trade_rows = []
    for symbol, (daily_path, prefix, volatility_column) in strategy_specs.items():
        price_frame = load_price_frame(symbol, args.price_cache)
        base = strategy_frame(daily_path, prefix)
        regimes = load_regimes(daily_path, volatility_column)
        result["chains"][symbol] = {}
        result["strategy_overlays"][symbol] = {}
        result["conditional"][symbol] = {}
        for delta in deltas:
            key = f"delta_{int(round(delta * 100)):02d}"
            trades = select_weekly_calls(
                args.options_db, symbol, delta, price_frame, args.start, args.end
            )
            summary = summarize_trades(trades, args.assignment_cost_bp)
            result["chains"][symbol][key] = summary
            for row in trades:
                all_trade_rows.append(asdict(row))
            fractional_path, fractional = apply_overlay(
                base, trades, price_frame, args.assignment_cost_bp, False
            )
            rounded_path, rounded = apply_overlay(
                base, trades, price_frame, args.assignment_cost_bp, True
            )
            baseline = base.loc[fractional_path.index]
            baseline_terminal = float(baseline["wealth"].iloc[-1])
            baseline_path = pd.DataFrame({
                "wealth": baseline["wealth"],
                "performance_index": baseline["performance_index"] / float(baseline["performance_index"].iloc[0]),
                "contribution": baseline["contribution"],
            })
            baseline_metrics = performance_metrics(
                baseline_path, pd.Series(0.0, index=baseline_path.index),
                float(baseline_path["wealth"].iloc[0]),
            )
            result["strategy_overlays"][symbol][key] = {
                "baseline_terminal": baseline_terminal,
                "baseline": baseline_metrics,
                "fractional": fractional,
                "whole_contracts": rounded,
            }
            comparison_start = pd.Timestamp(trades[0].issue_date)
            comparison_end = pd.Timestamp(trades[-1].expiration)
            result["conditional"][symbol][key] = {}
            for policy in regime_policies:
                conditional_trades = filter_trades_by_regime(trades, regimes, policy)
                _, conditional_fractional = apply_overlay(
                    base, conditional_trades, price_frame, args.assignment_cost_bp, False,
                    period_start=comparison_start, period_end=comparison_end,
                )
                conditional_path, conditional_rounded = apply_overlay(
                    base, conditional_trades, price_frame, args.assignment_cost_bp, True,
                    period_start=comparison_start, period_end=comparison_end,
                )
                result["conditional"][symbol][key][policy] = {
                    "active_weeks": len(conditional_trades),
                    "share_of_available_weeks": len(conditional_trades) / len(trades),
                    "chain": summarize_trades(conditional_trades, args.assignment_cost_bp),
                    "fractional": conditional_fractional,
                    "whole_contracts": conditional_rounded,
                }
                if abs(delta - 0.05) < 1e-12:
                    conditional_daily.append(pd.DataFrame({
                        "date": conditional_path.index,
                        "symbol": symbol, "delta": delta, "policy": policy,
                        "whole_contract_wealth": conditional_path["wealth"].to_numpy(),
                        "baseline_wealth": base.loc[conditional_path.index, "wealth"].to_numpy(),
                    }))
            overlay_daily.append(pd.DataFrame({
                "date": fractional_path.index,
                "symbol": symbol, "delta": delta,
                "fractional_wealth": fractional_path["wealth"].to_numpy(),
                "fractional_drawdown": fractional_path["flow_adjusted_drawdown"].to_numpy(),
                "whole_contract_wealth": rounded_path["wealth"].to_numpy(),
                "baseline_wealth": baseline["wealth"].to_numpy(),
            }))
    (args.out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(all_trade_rows).drop_duplicates(
        ["symbol", "issue_date", "target_delta"]
    ).to_csv(args.out / "trades.csv", index=False)
    pd.concat(overlay_daily, ignore_index=True).to_csv(args.out / "daily.csv", index=False)
    pd.concat(conditional_daily, ignore_index=True).to_csv(
        args.out / "conditional_daily.csv", index=False
    )
    print(json.dumps({"chains": result["chains"], "overlays": result["strategy_overlays"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
