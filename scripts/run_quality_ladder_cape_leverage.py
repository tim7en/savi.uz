"""Backtest the quality drawdown ladder with CAPE-conditioned SPY leverage.

The standing 80% SPY core uses 3x when lagged CAPE is below 25, 2x from
25 through 35, and 1x above 35.  Drawdown-funded SPY and quality positions
remain unlevered.  Cash flows are $10,000 annually plus an additional $30,000
every third contribution year.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

import pandas as pd

import run_cape_leverage_study as cape_data
import run_quality_compounder_v2 as market
import run_quality_ladder_harvest as ladder
import run_spy_quality_rotation as core
from run_contribution_quality_strategy import (
    contribution_schedule,
    performance_metrics,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="1993-01-29")
    parser.add_argument("--end", default=None)
    parser.add_argument("--initial", type=float, default=10_000.0)
    parser.add_argument("--annual-contribution", type=float, default=10_000.0)
    parser.add_argument("--triennial-contribution", type=float, default=30_000.0)
    parser.add_argument("--spy-share", type=float, default=0.80)
    parser.add_argument("--spread", type=float, default=0.01)
    parser.add_argument("--cape-excessive", type=float, default=35.0)
    parser.add_argument("--nav-deleverage-at", type=float, default=0.10)
    parser.add_argument("--relative-step", type=float, default=0.20)
    parser.add_argument("--harvest-share", type=float, default=0.05)
    parser.add_argument("--trade-bp", type=float, default=5.0)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--fundamentals", type=Path, default=Path("data/sp500_data"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/yahoo_daily"))
    parser.add_argument(
        "--macro-db", type=Path, default=Path("data/macro/macro.db")
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("out/strategy/quality_ladder_cape_leverage"),
    )
    return parser.parse_args(argv)


def cape_leverage(cape: pd.Series) -> pd.Series:
    """Map prior-known CAPE to 3x below 25, 2x through 35, 1x above 35."""
    result = pd.Series(3.0, index=cape.index, dtype=float)
    result.loc[cape >= 25.0] = 2.0
    result.loc[cape > 35.0] = 1.0
    return result


def trailing_percentile(signal: pd.Series, index: pd.DatetimeIndex,
                        window: int = 252) -> pd.Series:
    """Prior-close trailing rank of a daily signal, aligned without look-ahead."""
    signal = signal.sort_index().astype(float)

    def rank(values) -> float:
        history, current = values[:-1], values[-1]
        return float((history < current).sum() / len(history))

    close_rank = signal.rolling(window + 1, min_periods=window + 1).apply(
        rank, raw=True
    )
    # A close dated D controls exposure no earlier than the following SPY
    # session. Reindex first so holidays are handled before the trading-day lag.
    return close_rank.reindex(index).ffill().shift(1)


def trailing_vix_percentile(index: pd.DatetimeIndex, macro_db: Path,
                            window: int = 252,
                            smooth_sessions: int = 1) -> pd.Series:
    """Prior-close VIX rank, optionally after a trailing simple moving average."""
    connection = sqlite3.connect(f"file:{macro_db}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT obs_date, value FROM observations "
            "WHERE series_id='VIXCLS' AND value IS NOT NULL ORDER BY obs_date"
        ).fetchall()
    finally:
        connection.close()
    raw = pd.Series(
        [float(value) for _, value in rows],
        index=pd.to_datetime([stamp for stamp, _ in rows]),
        dtype=float,
    )

    if smooth_sessions < 1:
        raise ValueError("smooth_sessions must be at least one")
    signal = raw.rolling(
        smooth_sessions, min_periods=smooth_sessions
    ).mean() if smooth_sessions > 1 else raw
    return trailing_percentile(signal, index, window)


def vix_leverage_ceiling(percentile: pd.Series) -> pd.Series:
    """Map the prior-known VIX percentile to a 3x/2x/1x ceiling."""
    result = pd.Series(3.0, index=percentile.index, dtype=float)
    result.loc[percentile >= 0.70] = 2.0
    result.loc[percentile >= 0.90] = 1.0
    return result


def spy_dividend_yield(index: pd.DatetimeIndex, cache: Path) -> pd.Series:
    """Cash dividend per prior raw close on each SPY ex-dividend session."""
    path = cache / "SPY.json"
    payload = json.loads(path.read_text(encoding="utf-8"))["chart"]["result"][0]
    dividends = payload.get("events", {}).get("dividends", {})
    cash = pd.Series(0.0, index=index)
    for event in dividends.values():
        stamp = pd.to_datetime(event["date"], unit="s", utc=True).tz_localize(None).normalize()
        if stamp in cash.index:
            cash.loc[stamp] += float(event["amount"])
    raw_close = market.raw_close_from_cache("SPY", cache, index)
    return (cash / raw_close.shift(1)).replace([float("inf"), -float("inf")], 0.0).fillna(0.0)


def simulate_spy(spy: pd.Series, contributions: pd.Series,
                 initial: float = 10_000.0) -> pd.DataFrame:
    wealth = initial
    perf = 1.0
    high = 1.0
    previous = wealth
    rows = []
    for position, stamp in enumerate(spy.index):
        if position:
            wealth *= float(spy.iloc[position] / spy.iloc[position - 1])
            perf *= wealth / previous
        contribution = float(contributions.iloc[position])
        wealth += contribution
        previous = wealth
        high = max(high, perf)
        rows.append({
            "wealth": wealth,
            "performance_index": perf,
            "contribution": contribution,
            "flow_adjusted_drawdown": perf / high - 1.0,
        })
    return pd.DataFrame(rows, index=spy.index)


def simulate_treasury(rates: pd.Series, contributions: pd.Series,
                      initial: float = 10_000.0) -> pd.DataFrame:
    """Compound at the prior-known DGS3MO yield with matched cash flows."""
    wealth = initial
    perf = 1.0
    previous = wealth
    rows = []
    days = rates.index.to_series().diff().dt.days.fillna(0.0)
    for position, stamp in enumerate(rates.index):
        if position:
            known_rate = float(rates.iloc[position - 1])
            wealth *= 1.0 + known_rate * float(days.iloc[position]) / 365.0
            perf *= wealth / previous
        contribution = float(contributions.iloc[position])
        wealth += contribution
        previous = wealth
        rows.append({
            "wealth": wealth,
            "performance_index": perf,
            "contribution": contribution,
            "flow_adjusted_drawdown": 0.0,
        })
    return pd.DataFrame(rows, index=rates.index)


def main(argv=None):
    args = parse_args(argv)
    load_args = market.parse_args([])
    for field in ("start", "end", "initial", "refresh", "fundamentals", "cache"):
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
    known_vix_percentile = trailing_vix_percentile(prices.index, args.macro_db)
    known_vix_sma5 = trailing_vix_percentile(
        prices.index, args.macro_db, smooth_sessions=5
    )
    known_vix_sma20 = trailing_vix_percentile(
        prices.index, args.macro_db, smooth_sessions=20
    )
    known_vix_sma60 = trailing_vix_percentile(
        prices.index, args.macro_db, smooth_sessions=60
    )
    known_vix_monthly = known_vix_percentile.groupby(
        prices.index.to_period("M")
    ).transform("first")
    leverage = cape_leverage(known_cape)
    vix_ceiling = vix_leverage_ceiling(known_vix_sma60)
    cape_vix_ceiling = pd.concat([leverage, vix_ceiling], axis=1).min(axis=1)
    dividend_yield = spy_dividend_yield(prices.index, args.cache)
    one = pd.Series(1.0, index=prices.index)
    three = pd.Series(3.0, index=prices.index)
    contributions = contribution_schedule(
        prices.index, args.annual_contribution, args.triennial_contribution
    )
    annual_contributions = contribution_schedule(
        prices.index, args.annual_contribution, 0.0
    )
    triennial_contributions = contribution_schedule(
        prices.index, 0.0, args.triennial_contribution
    )

    variants = {
        "quality_cape": dict(
            rungs_enabled=True, quality_enabled=True, harvest_enabled=True,
            cape_enabled=True, core_leverage=leverage,
            nav_deleverage_at=args.nav_deleverage_at,
        ),
        "quality_cape_fresh_cape": dict(
            rungs_enabled=True, quality_enabled=True, harvest_enabled=True,
            cape_enabled=True, core_leverage=leverage,
            nav_deleverage_at=args.nav_deleverage_at,
            fresh_capital_cape_leverage=True,
        ),
        "quality_dual_guard_injections": dict(
            rungs_enabled=True, quality_enabled=True, harvest_enabled=True,
            cape_enabled=True, core_leverage=one,
            injection_leverage=leverage,
            injection_nav_drawdown=args.nav_deleverage_at,
        ),
        "quality_dual_guard_vix_brake": dict(
            rungs_enabled=True, quality_enabled=True, harvest_enabled=True,
            cape_enabled=True, core_leverage=one,
            injection_leverage=leverage,
            injection_nav_drawdown=args.nav_deleverage_at,
            injection_vix_percentile=known_vix_percentile,
            injection_vix_mode="brake",
        ),
        "quality_dual_guard_vix_reverse": dict(
            rungs_enabled=True, quality_enabled=True, harvest_enabled=True,
            cape_enabled=True, core_leverage=one,
            injection_leverage=leverage,
            injection_nav_drawdown=args.nav_deleverage_at,
            injection_vix_percentile=known_vix_percentile,
            injection_vix_mode="reverse",
        ),
        "quality_dual_guard_vix_sma5": dict(
            rungs_enabled=True, quality_enabled=True, harvest_enabled=True,
            cape_enabled=True, core_leverage=one,
            injection_leverage=leverage,
            injection_nav_drawdown=args.nav_deleverage_at,
            injection_vix_percentile=known_vix_sma5,
            injection_vix_mode="brake",
        ),
        "quality_dual_guard_vix_sma20": dict(
            rungs_enabled=True, quality_enabled=True, harvest_enabled=True,
            cape_enabled=True, core_leverage=one,
            injection_leverage=leverage,
            injection_nav_drawdown=args.nav_deleverage_at,
            injection_vix_percentile=known_vix_sma20,
            injection_vix_mode="brake",
        ),
        "quality_dual_guard_vix_sma60": dict(
            rungs_enabled=True, quality_enabled=True, harvest_enabled=True,
            cape_enabled=True, core_leverage=one,
            injection_leverage=leverage,
            injection_nav_drawdown=args.nav_deleverage_at,
            injection_vix_percentile=known_vix_sma60,
            injection_vix_mode="brake",
        ),
        "quality_dual_guard_vix_monthly": dict(
            rungs_enabled=True, quality_enabled=True, harvest_enabled=True,
            cape_enabled=True, core_leverage=one,
            injection_leverage=leverage,
            injection_nav_drawdown=args.nav_deleverage_at,
            injection_vix_percentile=known_vix_monthly,
            injection_vix_mode="brake",
        ),
        "quality_dual_guard_vix_sma60_deferred_annual": dict(
            rungs_enabled=True, quality_enabled=True, harvest_enabled=True,
            cape_enabled=True, core_leverage=one,
            injection_leverage=leverage,
            injection_nav_drawdown=args.nav_deleverage_at,
            injection_vix_percentile=known_vix_sma60,
            injection_vix_mode="brake",
            contribution_series=triennial_contributions,
            deferred_contribution_series=annual_contributions,
            deferred_deployment_percentile=known_vix_sma60,
        ),
        "quality_nav3_symmetric": dict(
            rungs_enabled=True, quality_enabled=True, harvest_enabled=True,
            cape_enabled=True, core_leverage=three,
            nav_leverage_ladder=True, nav_ladder_restore="symmetric",
        ),
        "quality_nav3_hysteresis": dict(
            rungs_enabled=True, quality_enabled=True, harvest_enabled=True,
            cape_enabled=True, core_leverage=three,
            nav_leverage_ladder=True, nav_ladder_restore="hysteresis",
        ),
        "quality_nav3_hysteresis_cape": dict(
            rungs_enabled=True, quality_enabled=True, harvest_enabled=True,
            cape_enabled=True, core_leverage=leverage,
            nav_leverage_ladder=True, nav_ladder_restore="hysteresis",
        ),
        "quality_nav3_hysteresis_vix": dict(
            rungs_enabled=True, quality_enabled=True, harvest_enabled=True,
            cape_enabled=True, core_leverage=vix_ceiling,
            nav_leverage_ladder=True, nav_ladder_restore="hysteresis",
        ),
        "quality_nav3_hysteresis_cape_vix": dict(
            rungs_enabled=True, quality_enabled=True, harvest_enabled=True,
            cape_enabled=True, core_leverage=cape_vix_ceiling,
            nav_leverage_ladder=True, nav_ladder_restore="hysteresis",
        ),
        "quality_nav3_cape_vix_dividends_treasury": dict(
            rungs_enabled=True, quality_enabled=True, harvest_enabled=True,
            cape_enabled=True, core_leverage=cape_vix_ceiling,
            nav_leverage_ladder=True, nav_ladder_restore="hysteresis",
            spy_dividend_yield=dividend_yield,
            spy_dividends_to_treasury=True,
        ),
        "quality_nav3_cape_vix_interest_to_spy": dict(
            rungs_enabled=True, quality_enabled=True, harvest_enabled=True,
            cape_enabled=True, core_leverage=cape_vix_ceiling,
            nav_leverage_ladder=True, nav_ladder_restore="hysteresis",
            treasury_interest_to_spy_annual=True,
        ),
        "quality_nav3_cape_vix_cash_routing": dict(
            rungs_enabled=True, quality_enabled=True, harvest_enabled=True,
            cape_enabled=True, core_leverage=cape_vix_ceiling,
            nav_leverage_ladder=True, nav_ladder_restore="hysteresis",
            spy_dividend_yield=dividend_yield,
            spy_dividends_to_treasury=True,
            treasury_interest_to_spy_annual=True,
        ),
        "quality_cape_no_brake": dict(
            rungs_enabled=True, quality_enabled=True, harvest_enabled=True,
            cape_enabled=True, core_leverage=leverage,
        ),
        "quality_cape_no_harvest": dict(
            rungs_enabled=True, quality_enabled=True, harvest_enabled=False,
            cape_enabled=True, core_leverage=leverage,
            nav_deleverage_at=args.nav_deleverage_at,
        ),
        "spy_ladder_cape": dict(
            rungs_enabled=True, quality_enabled=False, harvest_enabled=False,
            cape_enabled=False, all_spy_rungs=True, core_leverage=leverage,
            nav_deleverage_at=args.nav_deleverage_at,
        ),
        "spy_dual_guard_injections": dict(
            rungs_enabled=True, quality_enabled=False, harvest_enabled=False,
            cape_enabled=False, all_spy_rungs=True, core_leverage=one,
            injection_leverage=leverage,
            injection_nav_drawdown=args.nav_deleverage_at,
        ),
        "spy_dual_guard_vix_brake": dict(
            rungs_enabled=True, quality_enabled=False, harvest_enabled=False,
            cape_enabled=False, all_spy_rungs=True, core_leverage=one,
            injection_leverage=leverage,
            injection_nav_drawdown=args.nav_deleverage_at,
            injection_vix_percentile=known_vix_percentile,
            injection_vix_mode="brake",
        ),
        "cape_core_80_20": dict(
            rungs_enabled=False, quality_enabled=False, harvest_enabled=False,
            cape_enabled=False, core_leverage=leverage,
            nav_deleverage_at=args.nav_deleverage_at,
        ),
        "cape_core_80_20_fresh_cape": dict(
            rungs_enabled=False, quality_enabled=False, harvest_enabled=False,
            cape_enabled=False, core_leverage=leverage,
            nav_deleverage_at=args.nav_deleverage_at,
            fresh_capital_cape_leverage=True,
        ),
        "quality_1x": dict(
            rungs_enabled=True, quality_enabled=True, harvest_enabled=True,
            cape_enabled=True, core_leverage=one,
        ),
        "static_80_20": dict(
            rungs_enabled=False, quality_enabled=False, harvest_enabled=False,
            cape_enabled=False, core_leverage=one,
        ),
    }
    paths, event_sets, holdings, metrics = {}, {}, {}, {}
    for name, settings in variants.items():
        settings = dict(settings)
        variant_contributions = settings.pop(
            "contribution_series", contributions
        )
        path, events, open_holdings = ladder.simulate(
            prices, raw, shares, rates, known_cape, args,
            name=name, contribution_series=variant_contributions, **settings,
        )
        paths[name] = path
        event_sets[name] = events
        holdings[name] = open_holdings
        summary = performance_metrics(path, rates, args.initial)
        summary.update({
            "ending_treasury": float(path["treasury"].iloc[-1]),
            "ending_treasury_weight": float(path["treasury"].iloc[-1] / path["wealth"].iloc[-1]),
            "ending_quality_weight": float(path["quality_weight"].iloc[-1]),
            "max_quality_weight": float(path["quality_weight"].max()),
            "ending_reserve_target": float(path["reserve_target"].iloc[-1]),
            "ending_reserve_shortfall": float(max(
                path["reserve_target"].iloc[-1] - path["treasury"].iloc[-1], 0.0
            )),
            "mean_core_leverage": float(path["core_leverage"].mean()),
            "time_at_1x": float((path["core_leverage"] == 1.0).mean()),
            "time_at_2x": float((path["core_leverage"] == 2.0).mean()),
            "time_at_3x": float((path["core_leverage"] == 3.0).mean()),
            "mean_gross_exposure": float(path["gross_exposure"].mean()),
            "max_gross_exposure": float(path["gross_exposure"].max()),
            "nav_brake_share": float(path["nav_brake_active"].mean()),
            "financing_cost": float(path["financing_cost"].sum()),
            "fresh_capital_active_share": float((path["fresh_core_spy"] > 0).mean()),
            "mean_fresh_capital_weight": float(
                (path["fresh_core_spy"] / path["wealth"]).mean()
            ),
            "injection_active_share": float((path["injection_lot_count"] > 0).mean()),
            "mean_injection_weight": float(
                (path["injection_core_spy"] / path["wealth"]).mean()
            ),
            "max_injection_weight": float(
                (path["injection_core_spy"] / path["wealth"]).max()
            ),
            "mean_injection_leverage_when_active": float(
                path.loc[
                    path["injection_lot_count"] > 0,
                    "injection_weighted_leverage",
                ].mean()
            ) if (path["injection_lot_count"] > 0).any() else 0.0,
            "ending_pending_annual_cash": float(
                path["pending_annual_cash"].iloc[-1]
            ),
            "dividend_to_treasury": float(
                path.attrs["dividend_to_treasury"]
            ),
            "treasury_interest_to_spy": float(
                path.attrs["treasury_interest_to_spy"]
            ),
            "harvest_to_reserve": float(path.attrs["harvest_to_reserve"]),
            "harvest_to_spy": float(path.attrs["harvest_to_spy"]),
            "cape_incremental_reserve": float(path.attrs["cape_incremental_reserve"]),
            "events": ladder.summarize_events(events),
        })
        metrics[name] = summary

    spy = simulate_spy(prices["SPY"], contributions, args.initial)
    metrics["spy_1x"] = performance_metrics(spy, rates, args.initial)
    treasury = simulate_treasury(rates, contributions, args.initial)
    metrics["treasury_100"] = performance_metrics(treasury, rates, args.initial)
    result = {
        "sample": {
            "start": prices.index[0].date().isoformat(),
            "end": prices.index[-1].date().isoformat(),
            "sessions": len(prices),
            "cape_source_end": cape_monthly.index[-1].date().isoformat(),
        },
        "cash_flows": {
            "initial": args.initial,
            "annual": args.annual_contribution,
            "additional_every_third_year": args.triennial_contribution,
            "contribution_events": int((contributions > 0).sum()),
            "total_contributed": float(args.initial + contributions.sum()),
        },
        "rules": {
            "core": "80% SPY core / 20% Treasury at inception; core alone uses CAPE leverage",
            "cape_leverage": "3x below 25; 2x from 25 through 35; 1x above 35; monthly CAPE usable next month",
            "nav_brake": f"core drops to 1x at -{args.nav_deleverage_at:.0%} flow-adjusted NAV drawdown and restores current CAPE leverage only at a new NAV high",
            "fresh_capital": "in the fresh-capital variants, the 80% core portion of a contribution made while the NAV brake is active follows CAPE leverage; it merges into the legacy core at NAV recovery",
            "dual_guard_injections": f"permanent core stays 1x; when prior-close account NAV DD is at least {args.nav_deleverage_at:.0%}, the SPY portion of a new contribution enters at the CAPE tier and resets to 1x when account NAV recovers",
            "vix_brake": "prior-close VIX trailing 252-session percentile dynamically caps injection leverage: CAPE ceiling below the 70th percentile, 2x from the 70th through 90th, and 1x at or above the 90th; smoothing tests use 5-, 20-, or 60-session trailing averages before ranking, plus a monthly-held raw signal",
            "vix_timed_annual_contributions": "the annual $10,000 enters a DGS3MO waiting pool and deploys unlevered to SPY in four equal episode-base rungs at the 70th, 80th, 90th, and 95th percentiles of 60-session-smoothed VIX; re-arm below the 50th percentile; triennial $30,000 follows the ordinary rule",
            "standing_nav_leverage": "80% SPY core starts at 3x; prior-close flow-adjusted NAV reduces it to 2x at -10% and 1x at -20%; symmetric and hysteretic recovery controls are compared",
            "standing_signal_caps": "CAPE and 60-session-smoothed VIX can independently cap the standing NAV leverage; the combined variant uses the lower ceiling",
            "cash_routing": "adjusted total returns reinvest dividends by default; controls route SPY core dividends to Treasury and/or sweep accrued Treasury interest to unlevered SPY at the next year start",
            "financing": f"prior-known DGS3MO + {args.spread:.2%} on core exposure above 1x",
            "rungs": "20% Treasury to quality at -10% SPY DD; 30% to unlevered SPY at -20%; 30% to quality at -30%; 20% to unlevered SPY at -50%",
            "harvest": f"sell {args.harvest_share:.0%} original lot shares at every new {args.relative_step:.0%} relative-wealth band versus SPY",
            "cash_flows": f"{args.annual_contribution:,.0f} annually plus {args.triennial_contribution:,.0f} additional every third contribution year",
        },
        "variants": metrics,
        "holdings": holdings,
        "events": {
            name: [asdict(event) for event in events]
            for name, events in event_sets.items()
        },
        "coverage": coverage,
        "warnings": [
            "CAPE is lagged one month and the common sample ends with the local September 2024 observation.",
            "The core is modeled as daily rebalanced leverage with financing; ETF tracking error, margin calls, and tax are omitted.",
            "Fresh-capital variants isolate deposits made during an active NAV brake; those deposits follow CAPE leverage until the account recovers its previous high.",
            "Drawdown-funded SPY and quality positions are unlevered.",
            "The market-cap candidate union is date-ranked but survivor-biased and omits delisted historical leaders.",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    daily = pd.DataFrame(index=prices.index)
    daily["cape_known"] = known_cape
    daily["cape_leverage"] = leverage
    daily["vix_percentile"] = known_vix_percentile
    daily["vix_sma5_percentile"] = known_vix_sma5
    daily["vix_sma20_percentile"] = known_vix_sma20
    daily["vix_sma60_percentile"] = known_vix_sma60
    daily["vix_monthly_percentile"] = known_vix_monthly
    daily["spy_dividend_yield"] = dividend_yield
    daily["contribution"] = contributions
    daily["cumulative_contribution"] = args.initial + contributions.cumsum()
    daily["spy_1x_wealth"] = spy["wealth"]
    daily["spy_1x_performance"] = spy["performance_index"]
    daily["spy_1x_pnl"] = spy["wealth"] - daily["cumulative_contribution"]
    daily["treasury_100_wealth"] = treasury["wealth"]
    daily["treasury_100_performance"] = treasury["performance_index"]
    daily["treasury_100_pnl"] = treasury["wealth"] - daily["cumulative_contribution"]
    for name, path in paths.items():
        for column in (
            "wealth", "performance_index", "treasury", "quality",
            "quality_weight", "spy_weight", "spy_drawdown", "core_leverage",
            "base_core_leverage", "nav_brake_active", "gross_exposure",
            "legacy_core_leverage", "fresh_core_leverage", "fresh_core_spy",
            "injection_core_spy", "injection_gross_exposure", "injection_lot_count",
            "injection_weighted_leverage", "vix_percentile",
            "pending_annual_cash", "available_treasury", "financing_cost",
        ):
            daily[f"{name}_{column}"] = path[column]
        daily[f"{name}_pnl"] = path["wealth"] - daily["cumulative_contribution"]
    daily.to_csv(args.out / "daily.csv", index_label="date")
    print(json.dumps({
        "sample": result["sample"],
        "cash_flows": result["cash_flows"],
        "variants": metrics,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
