"""Backtest the contribution-funded SPY / Treasury / quality rescue policy.

This is the cash-flow-aware version of the policy developed in the accompanying
research page.  Deposits are kept separate from returns through unitisation, and
the SPY comparison receives the exact same deposits on the exact same dates.
Signals use the prior close and actions occur at the current close.
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

import run_spy_quality_rotation as core


LEVERAGE_DOWN = (0.15, 0.30)
RESERVE_RUNGS = ((0.10, 0.10), (0.20, 0.20), (0.30, 0.30), (0.40, None))


@dataclass
class Event:
    date: str
    kind: str
    amount: float
    drawdown: float
    detail: str


@dataclass
class QualityLot:
    ticker: str
    shares: float
    original_cost: float
    remaining_cost: float
    entry_price: float
    entry_date: pd.Timestamp
    recovery_spy: float
    last_harvest_price: float
    harvests: int = 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="1993-01-29")
    parser.add_argument("--end", default=None)
    parser.add_argument("--initial", type=float, default=10_000.0)
    parser.add_argument("--annual-contribution", type=float, default=10_000.0)
    parser.add_argument("--triennial-contribution", type=float, default=30_000.0)
    parser.add_argument("--spy-share", type=float, default=0.80)
    parser.add_argument("--spread", type=float, default=0.01)
    parser.add_argument("--quarterly-profit-sweep", type=float, default=0.10)
    parser.add_argument("--quality-quarterly-harvest", type=float, default=0.01)
    parser.add_argument("--trade-bp", type=float, default=5.0)
    parser.add_argument("--quality-loss-budget", type=float, default=0.25)
    parser.add_argument("--risk-per-stock", type=float, default=0.01)
    parser.add_argument("--max-quality-names", type=int, default=7)
    parser.add_argument("--min-quality-names", type=int, default=5)
    parser.add_argument("--min-history-years", type=float, default=5.0)
    parser.add_argument("--discount-gap", type=float, default=0.05)
    parser.add_argument("--review-years", type=float, default=5.0)
    parser.add_argument("--rolling-years", type=int, default=20)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--fundamentals", type=Path,
                        default=Path("data/sp500_data"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/yahoo_daily"))
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/contribution_quality"))
    return parser.parse_args(argv)


def contribution_schedule(index: pd.DatetimeIndex, annual: float,
                          triennial: float) -> pd.Series:
    """Contribute at the first close of each anniversary calendar year.

    The initial deposit is separate.  Contribution years 3, 6, 9, ... receive
    the regular amount plus the additional triennial amount.
    """
    result = pd.Series(0.0, index=index)
    years = sorted(set(index.year))
    for number, year in enumerate(years[1:], start=1):
        day = index[index.year == year][0]
        result.loc[day] = annual + (triennial if number % 3 == 0 else 0.0)
    return result


def leverage_state(previous: float, drawdown: float) -> float:
    """Apply the 3x/2x/1x ladder with asymmetric recovery thresholds."""
    if drawdown >= -1e-12:
        return 3.0
    if previous == 3.0:
        if drawdown <= -LEVERAGE_DOWN[1]:
            return 1.0
        if drawdown <= -LEVERAGE_DOWN[0]:
            return 2.0
        return 3.0
    if previous == 2.0:
        return 1.0 if drawdown <= -LEVERAGE_DOWN[1] else 2.0
    return 2.0 if drawdown > -0.10 else 1.0


def lot_value(lots: list[QualityLot], row: pd.Series) -> float:
    return float(sum(
        lot.shares * float(row.get(lot.ticker, np.nan))
        for lot in lots
        if lot.shares > 0 and pd.notna(row.get(lot.ticker, np.nan))
    ))


def trailing_cagr(series: pd.Series, position: int,
                  years: float = 5.0) -> float | None:
    return core.rolling_total_return_cagr_as_of(series, position, years)


def quality_candidates(prices: pd.DataFrame, earnings: dict[str, pd.DataFrame],
                       position: int, args) -> list[dict]:
    """Rank only information available at ``position`` (the signal close)."""
    signal_date = prices.index[position]
    spy = prices["SPY"]
    spy_cagr = trailing_cagr(spy, position, args.min_history_years)
    if spy_cagr is None:
        return []
    spy_history = spy.iloc[:position + 1].dropna()
    spy_dd = float(spy_history.iloc[-1] / spy_history.max() - 1.0)
    ranked = []
    for ticker in prices.columns:
        if ticker == "SPY":
            continue
        if not core.point_in_time_earnings_eligible(
                earnings.get(ticker, pd.DataFrame()), signal_date):
            continue
        stock_cagr = trailing_cagr(
            prices[ticker], position, args.min_history_years
        )
        history = prices[ticker].iloc[:position + 1].dropna()
        if stock_cagr is None or history.empty or stock_cagr <= spy_cagr:
            continue
        stock_dd = float(history.iloc[-1] / history.max() - 1.0)
        discount = spy_dd - stock_dd
        if discount < args.discount_gap:
            continue
        ranked.append({
            "ticker": ticker,
            "five_year_cagr": stock_cagr,
            "spy_five_year_cagr": spy_cagr,
            "drawdown": stock_dd,
            "spy_drawdown": spy_dd,
            "discount_gap": discount,
            "score": (stock_cagr - spy_cagr) + discount,
        })
    return sorted(ranked, key=lambda row: row["score"], reverse=True)[
        :args.max_quality_names
    ]


def xirr(cash_flows: list[tuple[pd.Timestamp, float]]) -> float | None:
    """Money-weighted annual return, solved in log-rate space."""
    if not cash_flows or not any(value < 0 for _, value in cash_flows):
        return None
    first = cash_flows[0][0]
    times = np.array([(day - first).days / 365.2425 for day, _ in cash_flows])
    values = np.array([value for _, value in cash_flows], dtype=float)

    def npv(log_growth: float) -> float:
        return float(np.sum(values * np.exp(-log_growth * times)))

    low, high = math.log(0.001), math.log(101.0)
    f_low, f_high = npv(low), npv(high)
    if f_low * f_high > 0:
        return None
    for _ in range(200):
        mid = (low + high) / 2.0
        value = npv(mid)
        if abs(value) < 1e-7:
            break
        if f_low * value <= 0:
            high, f_high = mid, value
        else:
            low, f_low = mid, value
    return math.exp((low + high) / 2.0) - 1.0


def performance_metrics(path: pd.DataFrame, rates: pd.Series,
                        initial: float) -> dict:
    wealth = path["wealth"]
    perf = path["performance_index"]
    years = (wealth.index[-1] - wealth.index[0]).days / 365.2425
    returns = perf.pct_change().fillna(0.0)
    drawdown = perf / perf.cummax() - 1.0
    cash_flows = [(wealth.index[0], -initial)]
    cash_flows.extend(
        (stamp, -float(value))
        for stamp, value in path["contribution"].items() if value > 0
    )
    cash_flows.append((wealth.index[-1], float(wealth.iloc[-1])))
    underwater = drawdown < -1e-12
    longest = 0
    run = 0
    for value in underwater:
        run = run + 1 if value else 0
        longest = max(longest, run)
    volatility = float(returns.iloc[1:].std(ddof=1) * math.sqrt(252.0))
    excess = returns.iloc[1:] - rates.reindex(returns.index).iloc[1:] / 252.0
    std = float(returns.iloc[1:].std(ddof=1))
    total_paid = initial + float(path["contribution"].sum())
    return {
        "terminal_wealth": float(wealth.iloc[-1]),
        "total_contributed": total_paid,
        "net_gain": float(wealth.iloc[-1] - total_paid),
        "time_weighted_cagr": float(perf.iloc[-1] ** (1.0 / years) - 1.0),
        "xirr": xirr(cash_flows),
        "max_flow_adjusted_drawdown": float(drawdown.min()),
        "max_drawdown_date": drawdown.idxmin().date().isoformat(),
        "annual_volatility": volatility,
        "sharpe_vs_treasury": (
            float(excess.mean() / std * math.sqrt(252.0)) if std > 0 else None
        ),
        "time_underwater_share": float(underwater.mean()),
        "longest_underwater_sessions": int(longest),
        "current_drawdown": float(drawdown.iloc[-1]),
    }


def simulate_strategy(prices: pd.DataFrame, rates: pd.Series, earnings, args,
                      name: str = "strategy") -> tuple[pd.DataFrame, list[Event]]:
    spy = prices["SPY"].dropna()
    prices = prices.reindex(spy.index)
    rates = rates.reindex(spy.index).ffill().bfill()
    returns = spy.pct_change().fillna(0.0)
    days = spy.index.to_series().diff().dt.days.fillna(0.0)
    contributions = contribution_schedule(
        spy.index, args.annual_contribution, args.triennial_contribution
    )

    main = args.initial * args.spy_share
    reserve = args.initial * (1.0 - args.spy_share)
    rescue_shares = 0.0
    lots: list[QualityLot] = []
    events: list[Event] = []
    rows = []
    leverage = 3.0
    quarter_profit = 0.0
    performance_index = 1.0
    previous_total = args.initial
    previous_drawdown = 0.0
    performance_high = 1.0
    episode_reserve = reserve
    fired: set[float] = set()
    cost_rate = args.trade_bp / 10_000.0

    for position, stamp in enumerate(spy.index):
        signal_position = max(position - 1, 0)
        signal_dd = previous_drawdown
        current_row = prices.iloc[position]

        # The first close of a new quarter executes the prior quarter's sweep.
        if position and stamp.to_period("Q") != spy.index[position - 1].to_period("Q"):
            amount = min(max(quarter_profit, 0.0) * args.quarterly_profit_sweep,
                         max(main, 0.0))
            if amount > 0:
                main -= amount
                reserve += amount
                events.append(Event(
                    stamp.date().isoformat(), "quarterly_profit_sweep", amount,
                    signal_dd, "10% of positive financed SPY-sleeve quarter P&L",
                ))
            quarter_profit = 0.0

        leverage = leverage_state(leverage, signal_dd)
        if position:
            elapsed = float(days.iloc[position]) / 365.0
            known_rate = float(rates.iloc[position - 1])
            reserve *= 1.0 + known_rate * elapsed
            spy_r = float(returns.iloc[position])
            if main > 0:
                financing = max(leverage - 1.0, 0.0) * (
                    known_rate + args.spread
                ) * elapsed
                pnl = main * (leverage * spy_r - financing)
                main = max(main + pnl, 0.0)
                quarter_profit += pnl
            rescue_shares *= 1.0  # explicit: adjusted-close shares are unchanged

        # Exits and quarterly 1%-of-current-share harvesting use prior-close data.
        is_new_quarter = (
            position and stamp.to_period("Q") != spy.index[position - 1].to_period("Q")
        )
        for lot in lots:
            if lot.shares <= 0:
                continue
            signal_price = prices[lot.ticker].iloc[signal_position]
            execution_price = current_row.get(lot.ticker, np.nan)
            if pd.isna(signal_price) or pd.isna(execution_price):
                continue
            held_years = (spy.index[signal_position] - lot.entry_date).days / 365.2425
            stock_cagr = trailing_cagr(
                prices[lot.ticker], signal_position, args.review_years
            )
            spy_cagr = trailing_cagr(spy, signal_position, args.review_years)
            fundamental = core.earnings_quality_as_of(
                earnings.get(lot.ticker, pd.DataFrame()),
                spy.index[signal_position],
            )
            if (held_years >= args.review_years and stock_cagr is not None
                    and spy_cagr is not None and stock_cagr < spy_cagr
                    and fundamental["broken"] is True):
                proceeds = lot.shares * float(execution_price) * (1.0 - cost_rate)
                reserve += proceeds
                events.append(Event(
                    stamp.date().isoformat(), "quality_guardrail_exit", proceeds,
                    signal_dd, f"{lot.ticker}; 5y CAGR below SPY and earnings broken",
                ))
                lot.shares = 0.0
                lot.remaining_cost = 0.0
                continue
            if (is_new_quarter and float(spy.iloc[signal_position]) >= lot.recovery_spy
                    and float(signal_price) > lot.entry_price
                    and float(signal_price) > lot.last_harvest_price):
                shares = lot.shares * args.quality_quarterly_harvest
                proceeds = shares * float(execution_price) * (1.0 - cost_rate)
                cost_removed = lot.remaining_cost * args.quality_quarterly_harvest
                lot.shares -= shares
                lot.remaining_cost -= cost_removed
                lot.last_harvest_price = float(signal_price)
                lot.harvests += 1
                reserve += proceeds
                events.append(Event(
                    stamp.date().isoformat(), "quality_quarterly_harvest", proceeds,
                    signal_dd, f"{lot.ticker}; 1% of current shares after SPY recovery",
                ))

        quality_value = lot_value(lots, current_row)
        rescue_value = rescue_shares * float(spy.iloc[position])
        pre_flow_total = main + reserve + rescue_value + quality_value
        if position:
            performance_index *= pre_flow_total / previous_total
        performance_high = max(performance_high, performance_index)
        current_drawdown = performance_index / performance_high - 1.0

        contribution = float(contributions.iloc[position])
        if contribution > 0:
            main_add = contribution * args.spy_share
            reserve_add = contribution - main_add
            main += main_add
            reserve += reserve_add
            events.append(Event(
                stamp.date().isoformat(), "contribution", contribution,
                current_drawdown, f"{args.spy_share:.0%} leveraged SPY / "
                f"{1 - args.spy_share:.0%} Treasury",
            ))

        # A new unitised NAV high starts a new reserve episode.
        if current_drawdown >= -1e-12:
            if fired:
                events.append(Event(
                    stamp.date().isoformat(), "episode_reset", 0.0,
                    current_drawdown, "flow-adjusted NAV recovered its high",
                ))
            fired.clear()
            episode_reserve = reserve

        # Rungs use the prior close and execute at today's close.
        for threshold, fraction in RESERVE_RUNGS:
            if threshold in fired or signal_dd > -threshold:
                continue
            if threshold < 0.40:
                amount = min(episode_reserve * float(fraction), reserve)
                fired.add(threshold)
                if amount > 0:
                    reserve -= amount
                    rescue_shares += amount * (1.0 - cost_rate) / float(spy.iloc[position])
                    events.append(Event(
                        stamp.date().isoformat(), "deploy_reserve_spy", amount,
                        signal_dd, f"-{threshold:.0%} rung; unlevered SPY",
                    ))
                continue

            ranked = quality_candidates(prices, earnings, signal_position, args)
            fired.add(threshold)
            if len(ranked) < args.min_quality_names or reserve <= 0:
                events.append(Event(
                    stamp.date().isoformat(), "quality_rung_skipped", 0.0,
                    signal_dd, f"-{threshold:.0%} rung; only {len(ranked)} "
                    f"of {args.min_quality_names} required names qualified",
                ))
                continue
            available_cash = reserve
            account = main + reserve + rescue_shares * float(spy.iloc[position]) \
                + lot_value(lots, current_row)
            cap = args.risk_per_stock * account / args.quality_loss_budget
            per_name = min(available_cash / len(ranked), cap)
            deployed = 0.0
            tickers = []
            recovery_spy = float(spy.iloc[:signal_position + 1].max())
            for item in ranked:
                ticker = item["ticker"]
                price = current_row.get(ticker, np.nan)
                cash = min(per_name, reserve)
                if cash <= 0 or pd.isna(price):
                    continue
                shares = cash * (1.0 - cost_rate) / float(price)
                reserve -= cash
                deployed += cash
                tickers.append(ticker)
                lots.append(QualityLot(
                    ticker=ticker, shares=shares, original_cost=cash,
                    remaining_cost=cash, entry_price=float(price), entry_date=stamp,
                    recovery_spy=recovery_spy, last_harvest_price=float(price),
                ))
            events.append(Event(
                stamp.date().isoformat(), "deploy_reserve_quality", deployed,
                signal_dd, f"-{threshold:.0%} rung; {', '.join(tickers)}",
            ))

        quality_value = lot_value(lots, current_row)
        rescue_value = rescue_shares * float(spy.iloc[position])
        total = main + reserve + rescue_value + quality_value
        # Capture same-close stock transaction costs without treating the
        # contribution as performance.
        investable_after_flow = total - contribution
        if pre_flow_total > 0 and investable_after_flow > 0:
            performance_index *= investable_after_flow / pre_flow_total
            performance_high = max(performance_high, performance_index)
            current_drawdown = performance_index / performance_high - 1.0
        previous_total = total
        previous_drawdown = current_drawdown
        rows.append({
            "wealth": total,
            "performance_index": performance_index,
            "contribution": contribution,
            "leveraged_spy_sleeve": main,
            "treasury": reserve,
            "unlevered_rescue_spy": rescue_value,
            "quality_stocks": quality_value,
            "flow_adjusted_drawdown": current_drawdown,
            "applied_leverage": leverage,
            "gross_spy_exposure": (
                (main * leverage + rescue_value) / total if total > 0 else 0.0
            ),
        })

    path = pd.DataFrame(rows, index=spy.index)
    positions = []
    for ticker in sorted({lot.ticker for lot in lots if lot.shares > 0}):
        ticker_lots = [lot for lot in lots if lot.ticker == ticker and lot.shares > 0]
        value = sum(lot.shares for lot in ticker_lots) * float(prices[ticker].iloc[-1])
        cost = sum(lot.remaining_cost for lot in ticker_lots)
        positions.append({
            "ticker": ticker,
            "market_value": float(value),
            "remaining_cost": float(cost),
            "unrealized_multiple": float(value / cost) if cost > 0 else None,
            "portfolio_weight": float(value / path["wealth"].iloc[-1]),
            "lots": len(ticker_lots),
            "harvests": sum(lot.harvests for lot in ticker_lots),
            "entry_dates": sorted({lot.entry_date.date().isoformat() for lot in ticker_lots}),
        })
    path.attrs["positions"] = sorted(
        positions, key=lambda row: row["market_value"], reverse=True
    )
    return path, events


def simulate_spy(prices: pd.DataFrame, rates: pd.Series, args) -> pd.DataFrame:
    spy = prices["SPY"].dropna()
    contributions = contribution_schedule(
        spy.index, args.annual_contribution, args.triennial_contribution
    )
    wealth = args.initial
    perf = 1.0
    rows = []
    previous = wealth
    performance_high = 1.0
    for position, stamp in enumerate(spy.index):
        if position:
            wealth *= float(spy.iloc[position] / spy.iloc[position - 1])
            perf *= wealth / previous
        contribution = float(contributions.iloc[position])
        wealth += contribution
        previous = wealth
        performance_high = max(performance_high, perf)
        rows.append({
            "wealth": wealth,
            "performance_index": perf,
            "contribution": contribution,
            "flow_adjusted_drawdown": perf / performance_high - 1.0,
        })
    return pd.DataFrame(rows, index=spy.index)


def event_summary(events: list[Event]) -> dict:
    counts = Counter(event.kind for event in events)
    amounts = defaultdict(float)
    for event in events:
        amounts[event.kind] += event.amount
    return {
        kind: {"count": counts[kind], "amount": amounts[kind]}
        for kind in sorted(counts)
    }


def rolling_study(prices, rates, earnings, args):
    first_sessions = prices.index.to_series().groupby(prices.index.year).first()
    records = []
    for start in first_sessions:
        target = start + pd.DateOffset(years=args.rolling_years)
        if target > prices.index[-1]:
            continue
        end_pos = prices.index.searchsorted(target, side="right") - 1
        end = prices.index[end_pos]
        window_prices = prices.loc[start:end]
        window_rates = rates.loc[start:end]
        strategy, _ = simulate_strategy(
            window_prices, window_rates, earnings, args, "rolling"
        )
        spy = simulate_spy(window_prices, window_rates, args)
        strategy_stats = performance_metrics(strategy, window_rates, args.initial)
        spy_stats = performance_metrics(spy, window_rates, args.initial)
        records.append({
            "start": start.date().isoformat(), "end": end.date().isoformat(),
            "strategy_terminal": strategy_stats["terminal_wealth"],
            "strategy_xirr": strategy_stats["xirr"],
            "strategy_twr_cagr": strategy_stats["time_weighted_cagr"],
            "strategy_max_drawdown": strategy_stats["max_flow_adjusted_drawdown"],
            "spy_terminal": spy_stats["terminal_wealth"],
            "spy_xirr": spy_stats["xirr"],
            "spy_twr_cagr": spy_stats["time_weighted_cagr"],
            "spy_max_drawdown": spy_stats["max_flow_adjusted_drawdown"],
        })
    frame = pd.DataFrame(records)
    summary = {"years": args.rolling_years, "cohorts": len(frame)}
    for column in frame.columns[2:]:
        summary[column] = {
            "p10": float(frame[column].quantile(0.10)),
            "median": float(frame[column].median()),
            "p90": float(frame[column].quantile(0.90)),
        }
    summary["strategy_beats_spy_terminal_share"] = float(
        (frame["strategy_terminal"] > frame["spy_terminal"]).mean()
    )
    return frame, summary


def main(argv=None) -> int:
    args = parse_args(argv)
    if not 0 < args.spy_share < 1:
        raise ValueError("--spy-share must be between zero and one")
    if not 0 < args.quality_loss_budget <= 1:
        raise ValueError("--quality-loss-budget must be in (0, 1]")
    core_args = core.parse_args([])
    for name in ("start", "end", "initial", "spy_share", "spread", "trade_bp",
                 "min_history_years", "refresh", "fundamentals", "cache", "out"):
        setattr(core_args, name, getattr(args, name))
    prices, coverage = core.load_prices(core_args)
    rates = core.load_rates(prices.index)
    earnings = core.load_earnings_histories(
        args.fundamentals, [ticker for ticker in prices.columns if ticker != "SPY"]
    )

    strategy, events = simulate_strategy(prices, rates, earnings, args)
    spy = simulate_spy(prices, rates, args)
    strategy_stats = performance_metrics(strategy, rates, args.initial)
    spy_stats = performance_metrics(spy, rates, args.initial)
    rolling_frame, rolling_summary = rolling_study(
        prices, rates, earnings, args
    )

    result = {
        "sample": {
            "start": prices.index[0].date().isoformat(),
            "end": prices.index[-1].date().isoformat(),
            "sessions": len(prices),
        },
        "cash_flows": {
            "initial": args.initial,
            "annual": args.annual_contribution,
            "additional_every_third_year": args.triennial_contribution,
            "allocation": [args.spy_share, 1.0 - args.spy_share],
        },
        "rules": {
            "leverage": "3x; reduce to 2x at 15% NAV DD and 1x at 30%; 1x->2x above -10%; 3x only at recovery high",
            "reserve": "10%/20%/30% of episode-start Treasury at 10%/20%/30% flow-adjusted NAV DD into unlevered SPY; all remaining Treasury at 40% into quality",
            "quarterly_sweep": args.quarterly_profit_sweep,
            "quality_harvest": args.quality_quarterly_harvest,
            "quality_selection": "point-in-time positive earnings, >=5y history, 5y CAGR above SPY, drawdown >=5pp deeper than SPY; top seven fixed-basket survivors",
            "quality_sizing": f"1% NAV risk budget divided by {args.quality_loss_budget:.0%} loss budget; maximum {args.risk_per_stock / args.quality_loss_budget:.0%} NAV per name",
            "financing": f"prior-known DGS3MO + {args.spread:.2%}",
            "reserve_yield": "prior-known DGS3MO",
            "execution": "prior-close signal, next-close action",
            "dividends": "reinvested through adjusted-close total returns",
        },
        "strategy": strategy_stats,
        "spy_1x": spy_stats,
        "event_summary": event_summary(events),
        "events": [asdict(event) for event in events],
        "open_quality_positions": strategy.attrs["positions"],
        "rolling_20y": rolling_summary,
        "coverage": coverage,
        "bias_warning": "Quality candidates are a present-day fixed basket with no delisted names; quality-stock alpha is survivor-biased.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    daily = pd.DataFrame(index=prices.index)
    for column in strategy.columns:
        daily[f"strategy_{column}"] = strategy[column]
    for column in spy.columns:
        daily[f"spy_{column}"] = spy[column]
    daily.to_csv(args.out / "daily.csv", index_label="date")
    pd.DataFrame([asdict(event) for event in events]).to_csv(
        args.out / "events.csv", index=False
    )
    rolling_frame.to_csv(args.out / "rolling_20y.csv", index=False)
    print(json.dumps({
        "sample": result["sample"], "strategy": strategy_stats,
        "spy_1x": spy_stats, "rolling_20y": rolling_summary,
        "event_summary": result["event_summary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
