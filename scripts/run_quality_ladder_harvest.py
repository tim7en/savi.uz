"""Backtest an unlevered SPY/Treasury ladder with quality-stock harvesting.

The strategy begins 80% SPY / 20% Treasury.  Fractions of the Treasury
snapshot at the preceding SPY total-return high are deployed once at -10%,
-20%, -30%, and -50% SPY drawdowns.  The -10% and -30% rungs buy the seven
largest date-ranked companies; the other rungs buy SPY.  Quality lots are
benchmarked against SPY from their own entry dates and sell 5% of original
shares for every new 20 percentage points of relative wealth outperformance.

The historical top-seven universe is date-ranked but survivor-biased because
the local data do not contain a complete point-in-time listed/delisted universe.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import run_cape_leverage_study as cape_data
import run_quality_compounder_v2 as market
import run_spy_quality_rotation as core
from run_contribution_quality_strategy import Event, performance_metrics


RUNGS = (
    (0.10, 0.20, "quality"),
    (0.20, 0.30, "spy"),
    (0.30, 0.30, "quality"),
    (0.50, 0.20, "spy"),
)


@dataclass
class QualityLot:
    ticker: str
    shares: float
    original_shares: float
    cost: float
    entry_date: pd.Timestamp
    entry_price: float
    entry_spy: float
    harvest_bands: int = 0
    harvested_shares: float = 0.0


@dataclass
class InjectionLot:
    """Equity in a contribution tranche held until account NAV recovery."""

    equity: float
    leverage: float
    desired_leverage: float
    entry_date: pd.Timestamp
    entry_drawdown: float
    entry_cape: float


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="1993-01-29")
    parser.add_argument("--end", default=None)
    parser.add_argument("--initial", type=float, default=10_000.0)
    parser.add_argument("--monthly-contribution", type=float, default=10_000.0)
    parser.add_argument("--spy-share", type=float, default=0.80)
    parser.add_argument("--cape-excessive", type=float, default=35.0)
    parser.add_argument("--relative-step", type=float, default=0.20)
    parser.add_argument("--harvest-share", type=float, default=0.05)
    parser.add_argument("--trade-bp", type=float, default=5.0)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--fundamentals", type=Path, default=Path("data/sp500_data"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/yahoo_daily"))
    parser.add_argument(
        "--out", type=Path, default=Path("out/strategy/quality_ladder_harvest")
    )
    return parser.parse_args(argv)


def relative_excess(stock_now: float, stock_entry: float,
                    spy_now: float, spy_entry: float) -> float:
    """Relative total-return wealth since a lot's entry date."""
    return (stock_now / stock_entry) / (spy_now / spy_entry) - 1.0


def new_harvest_bands(excess: float, already_harvested: int,
                      step: float = 0.20) -> int:
    """Count newly crossed positive relative-return milestones."""
    if not math.isfinite(excess) or excess < step:
        return 0
    reached = int(math.floor((excess + 1e-12) / step))
    return max(reached - already_harvested, 0)


def summarize_events(events: list[Event]) -> dict:
    counts = Counter(event.kind for event in events)
    amounts = defaultdict(float)
    for event in events:
        amounts[event.kind] += event.amount
    return {
        kind: {"count": counts[kind], "amount": amounts[kind]}
        for kind in sorted(counts)
    }


def quality_value(lots: list[QualityLot], row: pd.Series) -> float:
    return float(sum(
        lot.shares * float(row.get(lot.ticker, np.nan))
        for lot in lots
        if lot.shares > 1e-12 and pd.notna(row.get(lot.ticker, np.nan))
    ))


def _buy_spy(amount: float, price: float, cost_rate: float) -> float:
    return amount * (1.0 - cost_rate) / price if amount > 0 else 0.0


def vix_leverage_cap(desired: float, percentile: float,
                     mode: str | None) -> float:
    """Apply the pre-registered VIX brake or its directional reversal."""
    if mode is None or not math.isfinite(percentile):
        return desired
    if mode == "brake":
        cap = 1.0 if percentile >= 0.90 else 2.0 if percentile >= 0.70 else 3.0
    elif mode == "reverse":
        cap = 3.0 if percentile >= 0.90 else 2.0 if percentile >= 0.70 else 1.0
    else:
        raise ValueError(f"unknown VIX leverage mode: {mode}")
    return min(desired, cap)


def simulate(
    prices: pd.DataFrame,
    raw: pd.DataFrame,
    shares: dict[str, pd.Series],
    rates: pd.Series,
    cape: pd.Series,
    args,
    *,
    name: str,
    rungs_enabled: bool,
    quality_enabled: bool,
    harvest_enabled: bool,
    cape_enabled: bool,
    all_spy_rungs: bool = False,
    contribution_series: pd.Series | None = None,
    core_leverage: pd.Series | None = None,
    nav_deleverage_at: float | None = None,
    nav_restore_drawdown: float = 0.0,
    fresh_capital_cape_leverage: bool = False,
    injection_leverage: pd.Series | None = None,
    injection_nav_drawdown: float | None = None,
    injection_vix_percentile: pd.Series | None = None,
    injection_vix_mode: str | None = None,
    deferred_contribution_series: pd.Series | None = None,
    deferred_deployment_percentile: pd.Series | None = None,
    deferred_deployment_thresholds: tuple[float, ...] = (0.70, 0.80, 0.90, 0.95),
    deferred_reset_below: float = 0.50,
    nav_leverage_ladder: bool = False,
    nav_ladder_restore: str = "hysteresis",
    nav_ladder_down_2x: float = 0.10,
    nav_ladder_down_1x: float = 0.20,
    spy_dividend_yield: pd.Series | None = None,
    spy_dividends_to_treasury: bool = False,
    treasury_interest_to_spy_annual: bool = False,
) -> tuple[pd.DataFrame, list[Event], list[dict]]:
    spy = prices["SPY"].dropna()
    prices = prices.reindex(spy.index)
    raw = raw.reindex(spy.index)
    rates = rates.reindex(spy.index).ffill().bfill()
    cape = cape.reindex(spy.index).ffill()
    contributions = (
        market.monthly_schedule(spy.index, args.monthly_contribution)
        if contribution_series is None
        else contribution_series.reindex(spy.index).fillna(0.0)
    )
    deferred_contributions = (
        pd.Series(0.0, index=spy.index)
        if deferred_contribution_series is None
        else deferred_contribution_series.reindex(spy.index).fillna(0.0)
    )
    deferred_signal = (
        pd.Series(np.nan, index=spy.index)
        if deferred_deployment_percentile is None
        else deferred_deployment_percentile.reindex(spy.index).ffill().astype(float)
    )
    dividend_yields = (
        pd.Series(0.0, index=spy.index)
        if spy_dividend_yield is None
        else spy_dividend_yield.reindex(spy.index).fillna(0.0).astype(float)
    )
    leverage = (
        pd.Series(1.0, index=spy.index)
        if core_leverage is None
        else core_leverage.reindex(spy.index).ffill().bfill().astype(float)
    )
    injection_levels = (
        leverage
        if injection_leverage is None
        else injection_leverage.reindex(spy.index).ffill().bfill().astype(float)
    )
    vix_percentiles = (
        pd.Series(np.nan, index=spy.index)
        if injection_vix_percentile is None
        else injection_vix_percentile.reindex(spy.index).ffill().astype(float)
    )
    spy_returns = spy.pct_change().fillna(0.0)
    days = spy.index.to_series().diff().dt.days.fillna(0.0)
    spy_high = spy.cummax()
    spy_drawdown = spy / spy_high - 1.0
    cost_rate = args.trade_bp / 10_000.0

    core_spy = args.initial * args.spy_share
    # Contributions made while the legacy NAV brake is active can optionally
    # live in a separate sleeve.  That sleeve follows CAPE leverage until NAV
    # recovers, when it merges into the legacy core for the next drawdown cycle.
    fresh_core_spy = 0.0
    rescue_spy_shares = 0.0
    reserve = args.initial * (1.0 - args.spy_share)
    pending_annual_cash = 0.0
    vix_deployment_fired: set[float] = set()
    vix_episode_base = 0.0
    lots: list[QualityLot] = []
    injection_lots: list[InjectionLot] = []
    events: list[Event] = []
    rows = []
    fired: set[float] = set()
    episode_reserve = reserve
    reserve_target = reserve
    performance_index = 1.0
    performance_high = 1.0
    previous_flow_drawdown = 0.0
    nav_brake_active = False
    previous_total = args.initial
    max_quality_weight = 0.0
    max_spy_exposure = args.spy_share * float(leverage.iloc[0])
    harvest_to_reserve_total = 0.0
    harvest_to_spy_total = 0.0
    cape_incremental_reserve_total = 0.0
    financing_cost_total = 0.0
    nav_ladder_level = 3.0
    treasury_interest_this_year = 0.0
    dividend_to_treasury_total = 0.0
    treasury_interest_to_spy_total = 0.0

    for position, stamp in enumerate(spy.index):
        signal_position = max(position - 1, 0)
        signal_date = spy.index[signal_position]
        signal_dd = float(spy_drawdown.iloc[signal_position])
        signal_cape = float(cape.iloc[signal_position]) if pd.notna(cape.iloc[signal_position]) else np.nan
        current_row = prices.iloc[position]
        new_quarter = bool(
            position and stamp.to_period("Q") != signal_date.to_period("Q")
        )
        current_vix_percentile = float(vix_percentiles.iloc[position])

        if (treasury_interest_to_spy_annual and position
                and stamp.year != signal_date.year):
            sweep = min(treasury_interest_this_year, reserve)
            if sweep > 0:
                reserve -= sweep
                core_spy += sweep
                treasury_interest_to_spy_total += sweep
                events.append(Event(
                    stamp.date().isoformat(), "treasury_interest_to_spy",
                    sweep, previous_flow_drawdown,
                    "Prior-year Treasury interest reinvested into unlevered SPY core",
                ))
            treasury_interest_this_year = 0.0

        # Leveraged injection tranches become ordinary 1x core capital once
        # the account regains its previous flow-adjusted high.
        if position and injection_lots and previous_flow_drawdown >= -1e-12:
            merged = sum(lot.equity for lot in injection_lots)
            core_spy += merged
            injection_lots.clear()
            events.append(Event(
                stamp.date().isoformat(), "injection_leverage_reset", merged,
                previous_flow_drawdown,
                "Account NAV recovered; injection tranches converted to 1x core",
            ))

        # The VIX layer can reduce or restore tranche leverage using only the
        # prior-close percentile supplied for this execution day. CAPE fixes
        # the desired ceiling at entry; VIX controls the path to that ceiling.
        for lot in injection_lots:
            old_level = lot.leverage
            lot.leverage = vix_leverage_cap(
                lot.desired_leverage, current_vix_percentile,
                injection_vix_mode,
            )
            if abs(lot.leverage - old_level) > 1e-12:
                events.append(Event(
                    stamp.date().isoformat(), "vix_injection_leverage_change",
                    lot.equity, previous_flow_drawdown,
                    f"{old_level:.0f}x to {lot.leverage:.0f}x; "
                    f"VIX trailing percentile {current_vix_percentile:.1%}",
                ))

        if nav_deleverage_at is not None:
            was_active = nav_brake_active
            if not nav_brake_active and previous_flow_drawdown <= -nav_deleverage_at + 1e-12:
                nav_brake_active = True
            elif nav_brake_active and previous_flow_drawdown >= -nav_restore_drawdown - 1e-12:
                nav_brake_active = False
            if nav_brake_active != was_active:
                kind = "nav_deleverage" if nav_brake_active else "nav_leverage_restore"
                detail = (
                    f"NAV drawdown {previous_flow_drawdown:.1%}; core set to 1x"
                    if nav_brake_active
                    else "NAV recovered its high; current CAPE leverage restored"
                )
                events.append(Event(
                    stamp.date().isoformat(), kind, 0.0,
                    previous_flow_drawdown, detail,
                ))
                if not nav_brake_active and fresh_core_spy > 0:
                    merged = fresh_core_spy
                    core_spy += fresh_core_spy
                    fresh_core_spy = 0.0
                    events.append(Event(
                        stamp.date().isoformat(), "fresh_capital_merge", merged,
                        previous_flow_drawdown,
                        "NAV recovered; CAPE-levered contribution sleeve merged into legacy core",
                    ))

        base_position = max(position - 1, 0)
        base_level = float(leverage.iloc[base_position])
        if nav_leverage_ladder:
            old_nav_ladder_level = nav_ladder_level
            if nav_ladder_restore == "symmetric":
                if previous_flow_drawdown <= -nav_ladder_down_1x + 1e-12:
                    nav_ladder_level = 1.0
                elif previous_flow_drawdown <= -nav_ladder_down_2x + 1e-12:
                    nav_ladder_level = 2.0
                else:
                    nav_ladder_level = 3.0
            elif nav_ladder_restore == "hysteresis":
                if previous_flow_drawdown <= -nav_ladder_down_1x + 1e-12:
                    nav_ladder_level = 1.0
                elif nav_ladder_level >= 3.0 and previous_flow_drawdown <= -nav_ladder_down_2x + 1e-12:
                    nav_ladder_level = 2.0
                elif nav_ladder_level <= 1.0 and previous_flow_drawdown > -nav_ladder_down_2x + 1e-12:
                    nav_ladder_level = 2.0
                elif nav_ladder_level == 2.0 and previous_flow_drawdown >= -1e-12:
                    nav_ladder_level = 3.0
            else:
                raise ValueError(f"unknown NAV ladder restore mode: {nav_ladder_restore}")
            if nav_ladder_level != old_nav_ladder_level:
                events.append(Event(
                    stamp.date().isoformat(), "nav_leverage_ladder_change", 0.0,
                    previous_flow_drawdown,
                    f"{old_nav_ladder_level:.0f}x to {nav_ladder_level:.0f}x; "
                    f"{nav_ladder_restore} reversal",
                ))
            applied_level = min(base_level, nav_ladder_level)
        else:
            applied_level = 1.0 if nav_brake_active else base_level

        if position:
            elapsed = float(days.iloc[position]) / 365.0
            known_rate = float(rates.iloc[position - 1])
            total_spy_return = float(spy_returns.iloc[position])
            dividend_yield = float(dividend_yields.iloc[position])
            routed_dividend = 0.0
            applied_spy_return = (
                total_spy_return - dividend_yield
                if spy_dividends_to_treasury else total_spy_return
            )
            legacy_financing_cost = (
                core_spy * max(applied_level - 1.0, 0.0)
                * (known_rate + float(getattr(args, "spread", 0.01)))
                * elapsed
            )
            fresh_financing_cost = (
                fresh_core_spy * max(base_level - 1.0, 0.0)
                * (known_rate + float(getattr(args, "spread", 0.01)))
                * elapsed
            )
            injection_financing_cost = sum(
                lot.equity * max(lot.leverage - 1.0, 0.0)
                * (known_rate + float(getattr(args, "spread", 0.01)))
                * elapsed
                for lot in injection_lots
            )
            financing_cost = (
                legacy_financing_cost + fresh_financing_cost
                + injection_financing_cost
            )
            financing_cost_total += financing_cost
            if spy_dividends_to_treasury and dividend_yield > 0:
                routed_dividend = (
                    core_spy * applied_level
                    + fresh_core_spy * base_level
                    + sum(lot.equity * lot.leverage for lot in injection_lots)
                ) * dividend_yield
            core_spy = max(
                core_spy * (
                    1.0
                    + applied_level * applied_spy_return
                    - max(applied_level - 1.0, 0.0)
                    * (known_rate + float(getattr(args, "spread", 0.01)))
                    * elapsed
                ),
                0.0,
            )
            fresh_core_spy = max(
                fresh_core_spy * (
                    1.0
                    + base_level * applied_spy_return
                    - max(base_level - 1.0, 0.0)
                    * (known_rate + float(getattr(args, "spread", 0.01)))
                    * elapsed
                ),
                0.0,
            )
            for lot in injection_lots:
                lot.equity = max(
                    lot.equity * (
                        1.0
                        + lot.leverage * applied_spy_return
                        - max(lot.leverage - 1.0, 0.0)
                        * (known_rate + float(getattr(args, "spread", 0.01)))
                        * elapsed
                    ),
                    0.0,
                )
            treasury_interest = reserve * known_rate * elapsed
            reserve += treasury_interest + routed_dividend
            treasury_interest_this_year += treasury_interest
            dividend_to_treasury_total += routed_dividend
            pending_annual_cash *= 1.0 + known_rate * elapsed
        else:
            financing_cost = 0.0

        pre_flow_total = (
            core_spy
            + fresh_core_spy
            + sum(lot.equity for lot in injection_lots)
            + rescue_spy_shares * float(spy.iloc[position])
            + reserve
            + pending_annual_cash
            + quality_value(lots, current_row)
        )
        if position and previous_total > 0:
            performance_index *= pre_flow_total / previous_total

        contribution = float(contributions.iloc[position])
        deferred_contribution = float(deferred_contributions.iloc[position])
        total_contribution = contribution + deferred_contribution
        if contribution:
            spy_cash = contribution * args.spy_share
            reserve_cash = contribution - spy_cash
            injection_gate = (
                injection_nav_drawdown is not None
                and previous_flow_drawdown <= -injection_nav_drawdown + 1e-12
            )
            if injection_gate:
                desired_injection_level = float(injection_levels.iloc[signal_position])
                injection_level = vix_leverage_cap(
                    desired_injection_level, current_vix_percentile,
                    injection_vix_mode,
                )
                if desired_injection_level > 1.0:
                    injection_lots.append(InjectionLot(
                        equity=spy_cash,
                        leverage=injection_level,
                        desired_leverage=desired_injection_level,
                        entry_date=stamp,
                        entry_drawdown=previous_flow_drawdown,
                        entry_cape=signal_cape,
                    ))
                    contribution_detail = (
                        f"{args.spy_share:.0%} SPY injection at {injection_level:.0f}x "
                        f"(CAPE ceiling {desired_injection_level:.0f}x); "
                        f"NAV DD {previous_flow_drawdown:.1%}; CAPE {signal_cape:.1f}; "
                        f"VIX percentile {current_vix_percentile:.1%}; "
                        f"{1.0 - args.spy_share:.0%} Treasury"
                    )
                else:
                    core_spy += spy_cash
                    contribution_detail = (
                        f"{args.spy_share:.0%} SPY injection at 1x; expensive CAPE "
                        f"{signal_cape:.1f}; {1.0 - args.spy_share:.0%} Treasury"
                    )
            elif fresh_capital_cape_leverage and nav_brake_active:
                fresh_core_spy += spy_cash
                contribution_detail = (
                    f"{args.spy_share:.0%} CAPE-levered fresh core / "
                    f"{1.0 - args.spy_share:.0%} Treasury"
                )
            else:
                core_spy += spy_cash
                contribution_detail = (
                    f"{args.spy_share:.0%} SPY / "
                    f"{1.0 - args.spy_share:.0%} Treasury"
                )
            reserve += reserve_cash
            reserve_target += reserve_cash
            events.append(Event(
                stamp.date().isoformat(), "contribution", contribution, signal_dd,
                contribution_detail,
            ))

        if deferred_contribution:
            pending_annual_cash += deferred_contribution
            events.append(Event(
                stamp.date().isoformat(), "deferred_annual_contribution",
                deferred_contribution, signal_dd,
                "Annual cash entered the waiting pool and earns DGS3MO",
            ))

        deployment_rank = float(deferred_signal.iloc[position])
        if math.isfinite(deployment_rank) and deployment_rank < deferred_reset_below:
            vix_deployment_fired.clear()
            vix_episode_base = 0.0
        if math.isfinite(deployment_rank) and pending_annual_cash > 0:
            triggered_vix_levels = [
                threshold for threshold in deferred_deployment_thresholds
                if threshold not in vix_deployment_fired
                and deployment_rank >= threshold
            ]
            if triggered_vix_levels and not vix_deployment_fired:
                vix_episode_base = pending_annual_cash
            rung_cash = (
                vix_episode_base / len(deferred_deployment_thresholds)
                if vix_episode_base > 0 else 0.0
            )
            for threshold in triggered_vix_levels:
                cash = (
                    pending_annual_cash
                    if threshold == deferred_deployment_thresholds[-1]
                    else min(rung_cash, pending_annual_cash)
                )
                if cash <= 0:
                    break
                pending_annual_cash -= cash
                invested = cash * (1.0 - cost_rate)
                core_spy += invested
                vix_deployment_fired.add(threshold)
                events.append(Event(
                    stamp.date().isoformat(), "vix_annual_spy_deployment",
                    cash, signal_dd,
                    f"Unlevered SPY at smoothed VIX percentile "
                    f"{deployment_rank:.1%}; -{threshold:.0%} rung",
                ))

        # A recovered/new SPY high re-arms the ladder.  It does not force a sale
        # of SPY to refill Treasury; the actual reserve becomes the next budget.
        if signal_dd >= -1e-12:
            if fired:
                events.append(Event(
                    stamp.date().isoformat(), "episode_reset", 0.0, signal_dd,
                    "SPY total-return high recovered; ladder re-armed",
                ))
            fired.clear()
            episode_reserve = reserve
            reserve_target = max(reserve_target, reserve)

        triggered = []
        if rungs_enabled:
            triggered = [
                (threshold, fraction, destination)
                for threshold, fraction, destination in RUNGS
                if threshold not in fired and signal_dd <= -threshold + 1e-12
            ]

        # Check existing lots at quarter boundaries and every new ladder signal.
        if harvest_enabled and (new_quarter or triggered):
            for lot in lots:
                if lot.shares <= 1e-12:
                    continue
                signal_stock = prices[lot.ticker].iloc[signal_position]
                execution_stock = current_row.get(lot.ticker, np.nan)
                if pd.isna(signal_stock) or pd.isna(execution_stock):
                    continue
                excess = relative_excess(
                    float(signal_stock), lot.entry_price,
                    float(spy.iloc[signal_position]), lot.entry_spy,
                )
                bands = new_harvest_bands(
                    excess, lot.harvest_bands, args.relative_step
                )
                if bands <= 0:
                    continue
                shares_to_sell = min(
                    lot.shares,
                    lot.original_shares * args.harvest_share * bands,
                )
                gross = shares_to_sell * float(execution_stock)
                proceeds = gross * (1.0 - cost_rate)
                lot.shares -= shares_to_sell
                lot.harvested_shares += shares_to_sell
                lot.harvest_bands += bands

                total_before_routing = (
                    core_spy
                    + fresh_core_spy
                    + sum(lot.equity for lot in injection_lots)
                    + rescue_spy_shares * float(spy.iloc[position])
                    + reserve + proceeds
                    + quality_value(lots, current_row)
                )
                base_target = reserve_target
                target = base_target
                if cape_enabled and pd.notna(signal_cape) and signal_cape >= args.cape_excessive:
                    target = max(target, 0.20 * total_before_routing)
                to_reserve = min(proceeds, max(target - reserve, 0.0))
                base_to_reserve = min(
                    proceeds, max(base_target - reserve, 0.0)
                )
                to_spy = proceeds - to_reserve
                harvest_to_reserve_total += to_reserve
                harvest_to_spy_total += to_spy
                cape_incremental_reserve_total += max(
                    to_reserve - base_to_reserve, 0.0
                )
                reserve += to_reserve
                rescue_spy_shares += _buy_spy(
                    to_spy, float(spy.iloc[position]), cost_rate
                )
                events.append(Event(
                    stamp.date().isoformat(), "quality_relative_harvest", proceeds,
                    signal_dd,
                    f"{lot.ticker}; relative excess {excess:.1%}; "
                    f"{bands} new band(s); Treasury {to_reserve:.2f}; SPY {to_spy:.2f}",
                ))

        for threshold, fraction, destination in triggered:
            fired.add(threshold)
            planned = episode_reserve * fraction
            amount = min(planned, reserve)
            if amount <= 0:
                events.append(Event(
                    stamp.date().isoformat(), "rung_unfunded", 0.0, signal_dd,
                    f"-{threshold:.0%}; planned {planned:.2f}",
                ))
                continue
            actual_destination = "spy" if all_spy_rungs else destination
            if actual_destination == "spy" or not quality_enabled:
                reserve -= amount
                rescue_spy_shares += _buy_spy(
                    amount, float(spy.iloc[position]), cost_rate
                )
                events.append(Event(
                    stamp.date().isoformat(), "deploy_treasury_spy", amount,
                    signal_dd, f"-{threshold:.0%}; {fraction:.0%} of episode reserve",
                ))
                continue

            selected = market.mega_seven(
                signal_date, signal_position, prices, raw, shares
            )
            if len(selected) < 7:
                events.append(Event(
                    stamp.date().isoformat(), "quality_rung_skipped", 0.0,
                    signal_dd,
                    f"-{threshold:.0%}; only {len(selected)} market caps available",
                ))
                continue
            total_cap = sum(item["market_cap"] for item in selected)
            deployed = 0.0
            names = []
            for item in selected:
                cash = min(
                    amount * item["market_cap"] / total_cap,
                    reserve,
                )
                execution_price = current_row.get(item["ticker"], np.nan)
                if cash <= 0 or pd.isna(execution_price):
                    continue
                bought = cash * (1.0 - cost_rate) / float(execution_price)
                reserve -= cash
                deployed += cash
                lots.append(QualityLot(
                    ticker=item["ticker"], shares=bought,
                    original_shares=bought, cost=cash, entry_date=stamp,
                    entry_price=float(execution_price),
                    entry_spy=float(spy.iloc[position]),
                ))
                names.append(f"{item['ticker']} {item['market_cap'] / total_cap:.1%}")
            events.append(Event(
                stamp.date().isoformat(), "deploy_quality_seven", deployed,
                signal_dd,
                f"-{threshold:.0%}; " + "; ".join(names),
            ))

        quality = quality_value(lots, current_row)
        rescue_spy = rescue_spy_shares * float(spy.iloc[position])
        injection_core_spy = sum(lot.equity for lot in injection_lots)
        injection_exposure = sum(
            lot.equity * lot.leverage for lot in injection_lots
        )
        injection_weighted_leverage = (
            injection_exposure / injection_core_spy
            if injection_core_spy > 0 else 0.0
        )
        core_value = core_spy + fresh_core_spy + injection_core_spy
        spy_value = core_value + rescue_spy
        total = spy_value + reserve + pending_annual_cash + quality
        investable_after_flow = total - total_contribution
        if pre_flow_total > 0 and investable_after_flow > 0:
            performance_index *= investable_after_flow / pre_flow_total
        performance_high = max(performance_high, performance_index)
        flow_dd = performance_index / performance_high - 1.0
        quality_weight = quality / total if total > 0 else 0.0
        spy_exposure = spy_value / total if total > 0 else 0.0
        max_quality_weight = max(max_quality_weight, quality_weight)
        gross_exposure = (
            core_spy * applied_level
            + fresh_core_spy * base_level
            + injection_exposure
            + rescue_spy + quality
        ) / total if total > 0 else 0.0
        effective_core_leverage = (
            (
                core_spy * applied_level
                + fresh_core_spy * base_level
                + injection_exposure
            ) / core_value
            if core_value > 0 else 0.0
        )
        max_spy_exposure = max(max_spy_exposure, gross_exposure)
        previous_total = total
        previous_flow_drawdown = flow_dd
        rows.append({
            "wealth": total,
            "performance_index": performance_index,
            "contribution": total_contribution,
            "flow_adjusted_drawdown": flow_dd,
            "spy": spy_value,
            "core_spy": core_value,
            "legacy_core_spy": core_spy,
            "fresh_core_spy": fresh_core_spy,
            "injection_core_spy": injection_core_spy,
            "injection_gross_exposure": injection_exposure,
            "injection_lot_count": len(injection_lots),
            "injection_weighted_leverage": injection_weighted_leverage,
            "vix_percentile": current_vix_percentile,
            "rescue_spy": rescue_spy,
            "treasury": reserve + pending_annual_cash,
            "available_treasury": reserve,
            "pending_annual_cash": pending_annual_cash,
            "quality": quality,
            "quality_weight": quality_weight,
            "spy_weight": spy_exposure,
            "spy_drawdown": float(spy_drawdown.iloc[position]),
            "episode_reserve": episode_reserve,
            "reserve_target": reserve_target,
            "cape_known": signal_cape,
            "core_leverage": effective_core_leverage,
            "legacy_core_leverage": applied_level,
            "fresh_core_leverage": base_level if fresh_core_spy > 0 else 0.0,
            "base_core_leverage": base_level,
            "nav_brake_active": float(nav_brake_active),
            "gross_exposure": gross_exposure,
            "financing_cost": financing_cost,
        })

    path = pd.DataFrame(rows, index=spy.index)
    holdings = []
    for ticker in sorted({lot.ticker for lot in lots if lot.shares > 1e-12}):
        ticker_lots = [
            lot for lot in lots if lot.ticker == ticker and lot.shares > 1e-12
        ]
        value = sum(lot.shares for lot in ticker_lots) * float(prices[ticker].iloc[-1])
        holdings.append({
            "ticker": ticker,
            "market_value": float(value),
            "portfolio_weight": float(value / path["wealth"].iloc[-1]),
            "lots": len(ticker_lots),
            "harvest_bands": sum(lot.harvest_bands for lot in ticker_lots),
            "harvested_original_share": float(
                sum(lot.harvested_shares for lot in ticker_lots)
                / sum(lot.original_shares for lot in ticker_lots)
            ),
            "entry_dates": sorted({
                lot.entry_date.date().isoformat() for lot in ticker_lots
            }),
        })
    holdings.sort(key=lambda row: row["market_value"], reverse=True)
    path.attrs["max_quality_weight"] = max_quality_weight
    path.attrs["max_spy_weight"] = max_spy_exposure
    path.attrs["harvest_to_reserve"] = harvest_to_reserve_total
    path.attrs["harvest_to_spy"] = harvest_to_spy_total
    path.attrs["cape_incremental_reserve"] = cape_incremental_reserve_total
    path.attrs["financing_cost"] = financing_cost_total
    path.attrs["dividend_to_treasury"] = dividend_to_treasury_total
    path.attrs["treasury_interest_to_spy"] = treasury_interest_to_spy_total
    return path, events, holdings


def main(argv=None):
    args = parse_args(argv)
    load_args = market.parse_args([])
    for field in (
        "start", "end", "initial", "monthly_contribution", "trade_bp",
        "refresh", "fundamentals", "cache",
    ):
        setattr(load_args, field, getattr(args, field))
    prices, raw, shares, coverage = market.load_market_data(load_args)
    cape_monthly = cape_data.load_cape()
    cape_end = cape_monthly.index[-1] + pd.offsets.MonthEnd(1)
    if args.end:
        cape_end = min(cape_end, pd.Timestamp(args.end))
    prices = prices.loc[:cape_end]
    raw = raw.loc[prices.index]
    rates = core.load_rates(prices.index)
    known_cape = cape_data.known_cape_daily(cape_monthly, prices.index)

    variants = {
        "quality_ladder": dict(
            rungs_enabled=True, quality_enabled=True, harvest_enabled=True,
            cape_enabled=True,
        ),
        "quality_ladder_no_cape": dict(
            rungs_enabled=True, quality_enabled=True, harvest_enabled=True,
            cape_enabled=False,
        ),
        "quality_ladder_no_harvest": dict(
            rungs_enabled=True, quality_enabled=True, harvest_enabled=False,
            cape_enabled=False,
        ),
        "spy_ladder": dict(
            rungs_enabled=True, quality_enabled=False, harvest_enabled=False,
            cape_enabled=False, all_spy_rungs=True,
        ),
        "static_80_20": dict(
            rungs_enabled=False, quality_enabled=False, harvest_enabled=False,
            cape_enabled=False,
        ),
    }
    paths, event_sets, holdings, metrics = {}, {}, {}, {}
    for name, settings in variants.items():
        path, events, open_holdings = simulate(
            prices, raw, shares, rates, known_cape, args,
            name=name, **settings,
        )
        paths[name] = path
        event_sets[name] = events
        holdings[name] = open_holdings
        summary = performance_metrics(path, rates, args.initial)
        summary.update({
            "ending_treasury": float(path["treasury"].iloc[-1]),
            "ending_treasury_weight": float(
                path["treasury"].iloc[-1] / path["wealth"].iloc[-1]
            ),
            "ending_quality_weight": float(path["quality_weight"].iloc[-1]),
            "max_quality_weight": float(path["quality_weight"].max()),
            "ending_reserve_target": float(path["reserve_target"].iloc[-1]),
            "ending_reserve_shortfall": float(max(
                path["reserve_target"].iloc[-1] - path["treasury"].iloc[-1], 0.0
            )),
            "harvest_to_reserve": float(path.attrs["harvest_to_reserve"]),
            "harvest_to_spy": float(path.attrs["harvest_to_spy"]),
            "cape_incremental_reserve": float(
                path.attrs["cape_incremental_reserve"]
            ),
            "events": summarize_events(events),
        })
        metrics[name] = summary

    spy_path = market.simulate_spy(prices, load_args)
    metrics["spy_1x"] = performance_metrics(spy_path, rates, args.initial)
    result = {
        "sample": {
            "start": prices.index[0].date().isoformat(),
            "end": prices.index[-1].date().isoformat(),
            "sessions": len(prices),
            "cape_source_end": cape_monthly.index[-1].date().isoformat(),
        },
        "cash_flows": {
            "initial": args.initial,
            "monthly_contribution": args.monthly_contribution,
        },
        "rules": {
            "starting_allocation": f"{args.spy_share:.0%} SPY / {1-args.spy_share:.0%} Treasury; no leverage",
            "rungs": "20% Treasury to quality at -10% SPY DD; 30% to SPY at -20%; 30% to quality at -30%; 20% to SPY at -50%",
            "drawdown": "SPY adjusted-total-return drawdown; first crossing per peak-to-recovery episode; prior-close signal, next-close trade",
            "quality_selection": "top seven date-ranked market-cap estimates from a fixed historical-leader union; market-cap weighted",
            "harvest": f"quarter-end and rung checks; sell {args.harvest_share:.0%} of original lot shares for every new {args.relative_step:.0%} relative-wealth band versus SPY",
            "routing": f"rebuild pre-fall Treasury target, then SPY; when known CAPE >= {args.cape_excessive:g}, rebuild to at least 20% current NAV first",
            "dividends": "reinvested through adjusted total returns",
            "treasury": "prior-known DGS3MO",
            "trade_cost": f"{args.trade_bp:g} bp on drawdown and harvest trades",
        },
        "variants": metrics,
        "holdings": holdings,
        "events": {
            name: [asdict(event) for event in events]
            for name, events in event_sets.items()
        },
        "coverage": coverage,
        "warnings": [
            "The point-in-time market-cap union contains current survivors and omits delisted historical leaders.",
            "Share-count history is unavailable for the early sample, so affected quality rungs are skipped rather than backfilled with hindsight.",
            "CAPE is lagged one month and the common test ends with the local September 2024 Shiller observation.",
            "Taxes, bid/ask spread beyond the fixed trade cost, market impact, and stock-specific delisting gaps are omitted.",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    daily = pd.DataFrame(index=prices.index)
    daily["spy_1x_wealth"] = spy_path["wealth"]
    daily["spy_1x_performance"] = spy_path["performance_index"]
    for name, path in paths.items():
        for column in (
            "wealth", "performance_index", "treasury", "quality",
            "quality_weight", "spy_weight", "spy_drawdown",
            "episode_reserve", "reserve_target",
        ):
            daily[f"{name}_{column}"] = path[column]
    daily.to_csv(args.out / "daily.csv", index_label="date")
    print(json.dumps({
        "sample": result["sample"],
        "cash_flows": result["cash_flows"],
        "variants": metrics,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
