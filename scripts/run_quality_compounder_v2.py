"""Backtest the revised monthly-funded Quality Compounder Harvest rules.

The historical "mega seven" is approximated from a fixed union of companies
that have been mega-cap leaders in the local dataset.  At each -40% trigger the
script ranks that union using raw close times the latest balance-sheet share
count conservatively made available 90 days after period end.  This makes the
ranking date-aware, but not survivor-free: delisted firms and true historical
index membership are not available locally.
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

import run_spy_quality_rotation as data
from run_contribution_quality_strategy import (
    Event, leverage_state, performance_metrics,
)


MEGA_CAP_UNION = (
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B",
    "XOM", "GE", "WMT", "CSCO", "IBM", "INTC", "PFE", "JNJ", "PG",
    "JPM", "BAC", "C", "V", "HD", "UNH", "KO", "PEP", "COST", "MCD",
)
RUNGS = ((0.10, 0.10), (0.20, 0.20), (0.30, 0.30), (0.40, 0.40))


@dataclass
class Lot:
    ticker: str
    shares: float
    cost: float
    entry_date: pd.Timestamp
    entry_price: float
    recovery_spy: float
    recovery_harvested: bool = False
    harvests: int = 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="1993-01-29")
    parser.add_argument("--end", default=None)
    parser.add_argument("--initial", type=float, default=10_000.0)
    parser.add_argument("--monthly-contribution", type=float, default=10_000.0)
    parser.add_argument("--spread", type=float, default=0.01)
    parser.add_argument("--trade-bp", type=float, default=5.0)
    parser.add_argument("--rolling-years", type=int, default=20)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--fundamentals", type=Path, default=Path("data/sp500_data"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/yahoo_daily"))
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/quality_compounder_v2"))
    return parser.parse_args(argv)


def monthly_schedule(index: pd.DatetimeIndex, amount: float) -> pd.Series:
    schedule = pd.Series(0.0, index=index)
    first_sessions = index.to_series().groupby(index.to_period("M")).first()
    for stamp in first_sessions.iloc[1:]:
        schedule.loc[stamp] = amount
    return schedule


def raw_close_from_cache(ticker: str, cache: Path, index: pd.DatetimeIndex) -> pd.Series:
    path = cache / f"{ticker.replace('/', '-')}.json"
    chart = json.loads(path.read_text(encoding="utf-8"))["chart"]["result"][0]
    stamps = pd.to_datetime(chart["timestamp"], unit="s", utc=True).tz_localize(None).normalize()
    closes = chart["indicators"]["quote"][0]["close"]
    values = {
        stamp: float(value) for stamp, value in zip(stamps, closes)
        if value is not None and math.isfinite(float(value)) and float(value) > 0
    }
    return pd.Series(values, dtype=float, name=ticker).sort_index().reindex(index).ffill()


def shares_history(ticker: str, fundamentals: Path) -> pd.Series:
    path = fundamentals / f"{ticker}_balance_sheet.json"
    try:
        reports = json.loads(path.read_text(encoding="utf-8"))["data"]["quarterlyReports"]
    except (OSError, KeyError, json.JSONDecodeError):
        return pd.Series(dtype=float)
    values = {}
    for row in reports:
        try:
            period_end = pd.Timestamp(str(row["fiscalDateEnding"])[:10])
            shares = float(row["commonStockSharesOutstanding"])
        except (KeyError, TypeError, ValueError):
            continue
        if shares > 0 and math.isfinite(shares):
            values[period_end + pd.Timedelta(days=90)] = shares
    return pd.Series(values, dtype=float).sort_index()


def load_market_data(args):
    adjusted = {"SPY": data.yahoo_series(
        "SPY", args.start, args.end, args.cache, args.refresh
    )}
    coverage = {}
    for ticker in MEGA_CAP_UNION:
        try:
            adjusted[ticker] = data.yahoo_series(
                ticker, args.start, args.end, args.cache, args.refresh
            )
            coverage[ticker] = True
        except Exception as exc:
            coverage[ticker] = str(exc)
    spy_index = adjusted["SPY"].index
    prices = pd.DataFrame(adjusted).reindex(spy_index)
    for ticker in prices.columns:
        if ticker != "SPY":
            prices[ticker] = prices[ticker].ffill()
    raw = pd.DataFrame(index=spy_index)
    shares = {}
    for ticker in prices.columns:
        if ticker == "SPY":
            continue
        raw[ticker] = raw_close_from_cache(ticker, args.cache, spy_index)
        shares[ticker] = shares_history(ticker, args.fundamentals)
    return prices, raw, shares, coverage


def mega_seven(signal_date: pd.Timestamp, signal_position: int,
               adjusted: pd.DataFrame, raw: pd.DataFrame,
               shares: dict[str, pd.Series]) -> list[dict]:
    ranked = []
    for ticker in raw.columns:
        price = raw[ticker].iloc[signal_position]
        history = shares.get(ticker, pd.Series(dtype=float)).loc[:signal_date]
        adjusted_price = adjusted[ticker].iloc[signal_position]
        if pd.isna(price) or pd.isna(adjusted_price) or history.empty:
            continue
        market_cap = float(price) * float(history.iloc[-1])
        if market_cap > 0:
            ranked.append({"ticker": ticker, "market_cap": market_cap})
    return sorted(ranked, key=lambda row: row["market_cap"], reverse=True)[:7]


def lot_value(lots: list[Lot], row: pd.Series) -> float:
    return float(sum(
        lot.shares * float(row.get(lot.ticker, np.nan))
        for lot in lots if lot.shares > 0 and pd.notna(row.get(lot.ticker, np.nan))
    ))


def trailing_cagr(series: pd.Series, position: int, years: float = 5.0):
    return data.rolling_total_return_cagr_as_of(series, position, years)


def simulate(prices: pd.DataFrame, raw: pd.DataFrame, shares, rates,
             args, *, reserve_share: float, contribution_mode: str,
             name: str) -> tuple[pd.DataFrame, list[Event], list[dict]]:
    if contribution_mode not in {"immediate", "treasury_first"}:
        raise ValueError("unknown contribution mode")
    spy = prices["SPY"].dropna()
    prices = prices.reindex(spy.index)
    raw = raw.reindex(spy.index)
    rates = rates.reindex(spy.index).ffill().bfill()
    spy_returns = spy.pct_change().fillna(0.0)
    days = spy.index.to_series().diff().dt.days.fillna(0.0)
    contributions = monthly_schedule(spy.index, args.monthly_contribution)

    main = args.initial * (1.0 - reserve_share)
    reserve = args.initial * reserve_share
    rescue_shares = 0.0
    parked_contributions = 0.0
    lots: list[Lot] = []
    events: list[Event] = []
    rows = []
    leverage = 3.0
    quarter_profit = 0.0
    quarter_factor = 1.0
    performance_index = 1.0
    performance_high = 1.0
    previous_total = args.initial
    previous_drawdown = 0.0
    episode_reserve = reserve
    fired: set[float] = set()
    cost_rate = args.trade_bp / 10_000.0

    for position, stamp in enumerate(spy.index):
        signal_position = max(position - 1, 0)
        signal_date = spy.index[signal_position]
        signal_dd = previous_drawdown
        current_row = prices.iloc[position]
        new_quarter = position and stamp.to_period("Q") != signal_date.to_period("Q")
        new_year = position and stamp.year != signal_date.year

        # Prior period signals execute at the current close.
        if new_quarter:
            quarter_return = quarter_factor - 1.0
            if quarter_return >= 0.20 and quarter_profit > 0:
                amount = min(quarter_profit * 0.10, max(main, 0.0))
                main -= amount
                reserve += amount
                events.append(Event(
                    stamp.date().isoformat(), "quarterly_spy_sweep", amount,
                    signal_dd, f"sleeve quarter {quarter_return:.1%}; 10% of P&L",
                ))
            quarter_profit, quarter_factor = 0.0, 1.0
        if new_year:
            year_prices = spy.loc[str(signal_date.year)]
            year_return = float(year_prices.iloc[-1] / year_prices.iloc[0] - 1.0)
            if year_return >= 0.10 and main > 0:
                amount = main * 0.01
                main -= amount
                reserve += amount
                events.append(Event(
                    stamp.date().isoformat(), "annual_spy_sweep", amount,
                    signal_dd, f"SPY prior calendar year {year_return:.1%}; 1% of sleeve",
                ))

        leverage = leverage_state(leverage, signal_dd)
        if position:
            elapsed = float(days.iloc[position]) / 365.0
            known_rate = float(rates.iloc[position - 1])
            reserve *= 1.0 + known_rate * elapsed
            daily_net = leverage * float(spy_returns.iloc[position]) - max(
                leverage - 1.0, 0.0
            ) * (known_rate + args.spread) * elapsed
            if main > 0:
                pnl = main * daily_net
                main = max(main + pnl, 0.0)
                quarter_profit += pnl
                quarter_factor *= max(1.0 + daily_net, 1e-12)

        # Quality rules: 5% at a recovery event or after a +20% stock quarter.
        for lot in lots:
            if lot.shares <= 0:
                continue
            signal_price = prices[lot.ticker].iloc[signal_position]
            execution_price = current_row.get(lot.ticker, np.nan)
            if pd.isna(signal_price) or pd.isna(execution_price):
                continue
            held_years = (signal_date - lot.entry_date).days / 365.2425
            stock_cagr = trailing_cagr(prices[lot.ticker], signal_position)
            spy_cagr = trailing_cagr(spy, signal_position)
            if (held_years >= 5.0 and stock_cagr is not None and spy_cagr is not None
                    and stock_cagr < spy_cagr):
                proceeds = lot.shares * float(execution_price) * (1.0 - cost_rate)
                reserve += proceeds
                events.append(Event(
                    stamp.date().isoformat(), "quality_five_year_exit", proceeds,
                    signal_dd, f"{lot.ticker}; trailing 5y CAGR {stock_cagr:.1%} < SPY {spy_cagr:.1%}",
                ))
                lot.shares = 0.0
                continue
            recovered = (
                not lot.recovery_harvested
                and float(spy.iloc[signal_position]) >= lot.recovery_spy
            )
            quarter_up = False
            if new_quarter:
                prior_quarter = signal_date.to_period("Q")
                quarter_prices = prices.loc[
                    prices.index.to_period("Q") == prior_quarter, lot.ticker
                ].dropna()
                quarter_up = (
                    len(quarter_prices) >= 2
                    and float(quarter_prices.iloc[-1] / quarter_prices.iloc[0] - 1.0) >= 0.20
                )
            profitable = float(signal_price) > lot.entry_price
            if profitable and (recovered or quarter_up):
                shares_sold = lot.shares * 0.05
                proceeds = shares_sold * float(execution_price) * (1.0 - cost_rate)
                lot.shares -= shares_sold
                lot.harvests += 1
                if recovered:
                    lot.recovery_harvested = True
                reserve += proceeds
                reason = "SPY recovered prior high" if recovered else "stock quarter >=20%"
                events.append(Event(
                    stamp.date().isoformat(), "quality_harvest", proceeds,
                    signal_dd, f"{lot.ticker}; 5% current shares; {reason}",
                ))

        quality = lot_value(lots, current_row)
        rescue = rescue_shares * float(spy.iloc[position])
        pre_flow_total = main + reserve + rescue + quality
        if position:
            performance_index *= pre_flow_total / previous_total
        performance_high = max(performance_high, performance_index)
        current_dd = performance_index / performance_high - 1.0

        contribution = float(contributions.iloc[position])
        if contribution:
            if contribution_mode == "immediate":
                reserve_add = contribution * reserve_share
                main += contribution - reserve_add
                reserve += reserve_add
            else:
                reserve += contribution
                parked_contributions += contribution
            events.append(Event(
                stamp.date().isoformat(), "contribution", contribution, current_dd,
                contribution_mode.replace("_", " "),
            ))

        # Deploy fixed fractions of the episode-start reserve.
        for threshold, fraction in RUNGS:
            if threshold in fired or signal_dd > -threshold:
                continue
            fired.add(threshold)
            amount = min(episode_reserve * fraction, reserve)
            if amount <= 0:
                continue
            if threshold < 0.40:
                reserve -= amount
                rescue_shares += amount * (1.0 - cost_rate) / float(spy.iloc[position])
                events.append(Event(
                    stamp.date().isoformat(), "deploy_treasury_spy", amount,
                    signal_dd, f"-{threshold:.0%}; unlevered SPY",
                ))
                continue
            selected = mega_seven(signal_date, signal_position, prices, raw, shares)
            if len(selected) < 7:
                events.append(Event(
                    stamp.date().isoformat(), "mega_seven_skipped", 0.0,
                    signal_dd, f"only {len(selected)} market caps available",
                ))
                continue
            total_cap = sum(item["market_cap"] for item in selected)
            deployed, names = 0.0, []
            recovery_spy = float(spy.iloc[:signal_position + 1].max())
            for item in selected:
                ticker = item["ticker"]
                cash = min(amount * item["market_cap"] / total_cap, reserve)
                price = current_row.get(ticker, np.nan)
                if cash <= 0 or pd.isna(price):
                    continue
                reserve -= cash
                deployed += cash
                names.append(f"{ticker} {item['market_cap'] / total_cap:.1%}")
                lots.append(Lot(
                    ticker=ticker, shares=cash * (1.0 - cost_rate) / float(price),
                    cost=cash, entry_date=stamp, entry_price=float(price),
                    recovery_spy=recovery_spy,
                ))
            events.append(Event(
                stamp.date().isoformat(), "deploy_mega_seven", deployed,
                signal_dd, "; ".join(names),
            ))

        quality = lot_value(lots, current_row)
        rescue = rescue_shares * float(spy.iloc[position])
        total = main + reserve + rescue + quality
        investable_after_flow = total - contribution
        if pre_flow_total > 0 and investable_after_flow > 0:
            performance_index *= investable_after_flow / pre_flow_total
            performance_high = max(performance_high, performance_index)
            current_dd = performance_index / performance_high - 1.0

        # At a high, reset rungs/leverage and release only parked contributions.
        if current_dd >= -1e-12:
            leverage = 3.0
            if fired:
                events.append(Event(
                    stamp.date().isoformat(), "episode_reset", 0.0, current_dd,
                    "NAV recovered; rungs reset and 3x restored",
                ))
            fired.clear()
            if rescue_shares > 0:
                reset_value = rescue_shares * float(spy.iloc[position])
                main += reset_value
                rescue_shares = 0.0
                events.append(Event(
                    stamp.date().isoformat(), "reset_rescue_spy", reset_value,
                    current_dd, "unlevered rescue SPY returned to the 3x core sleeve",
                ))
            if contribution_mode == "treasury_first" and parked_contributions > 0:
                release = min(
                    parked_contributions * (1.0 - reserve_share), reserve
                )
                reserve -= release
                main += release
                events.append(Event(
                    stamp.date().isoformat(), "release_parked_contributions",
                    release, current_dd, "released to leveraged SPY at NAV high",
                ))
                parked_contributions = 0.0
            episode_reserve = reserve

        total = main + reserve + rescue_shares * float(spy.iloc[position]) \
            + lot_value(lots, current_row)
        previous_total = total
        previous_drawdown = current_dd
        rows.append({
            "wealth": total, "performance_index": performance_index,
            "contribution": contribution, "flow_adjusted_drawdown": current_dd,
            "leveraged_spy": main, "treasury": reserve,
            "rescue_spy": rescue_shares * float(spy.iloc[position]),
            "quality": lot_value(lots, current_row), "leverage": leverage,
            "parked_contributions": parked_contributions,
        })

    path = pd.DataFrame(rows, index=spy.index)
    holdings = []
    for ticker in sorted({lot.ticker for lot in lots if lot.shares > 0}):
        ticker_lots = [lot for lot in lots if lot.ticker == ticker and lot.shares > 0]
        value = sum(lot.shares for lot in ticker_lots) * float(prices[ticker].iloc[-1])
        holdings.append({
            "ticker": ticker, "market_value": float(value),
            "portfolio_weight": float(value / path["wealth"].iloc[-1]),
            "lots": len(ticker_lots), "harvests": sum(lot.harvests for lot in ticker_lots),
            "entry_dates": sorted({lot.entry_date.date().isoformat() for lot in ticker_lots}),
        })
    holdings.sort(key=lambda row: row["market_value"], reverse=True)
    return path, events, holdings


def simulate_spy(prices: pd.DataFrame, args) -> pd.DataFrame:
    spy = prices["SPY"].dropna()
    schedule = monthly_schedule(spy.index, args.monthly_contribution)
    wealth, perf, high, previous = args.initial, 1.0, 1.0, args.initial
    rows = []
    for position, stamp in enumerate(spy.index):
        if position:
            wealth *= float(spy.iloc[position] / spy.iloc[position - 1])
            perf *= wealth / previous
        contribution = float(schedule.iloc[position])
        wealth += contribution
        previous = wealth
        high = max(high, perf)
        rows.append({
            "wealth": wealth, "performance_index": perf,
            "contribution": contribution,
            "flow_adjusted_drawdown": perf / high - 1.0,
        })
    return pd.DataFrame(rows, index=spy.index)


def summarize_events(events):
    counts = Counter(event.kind for event in events)
    amounts = defaultdict(float)
    for event in events:
        amounts[event.kind] += event.amount
    return {kind: {"count": counts[kind], "amount": amounts[kind]}
            for kind in sorted(counts)}


def rolling(prices, raw, shares, rates, args):
    starts = prices.index.to_series().groupby(prices.index.year).first()
    records = []
    for start in starts:
        target = start + pd.DateOffset(years=args.rolling_years)
        if target > prices.index[-1]:
            continue
        end = prices.index[prices.index.searchsorted(target, side="right") - 1]
        p, r = prices.loc[start:end], rates.loc[start:end]
        raw_window = raw.loc[start:end]
        strategy, _, _ = simulate(
            p, raw_window, shares, r, args, reserve_share=0.20,
            contribution_mode="treasury_first", name="rolling",
        )
        spy = simulate_spy(p, args)
        sm = performance_metrics(strategy, r, args.initial)
        bm = performance_metrics(spy, r, args.initial)
        records.append({
            "start": start.date().isoformat(), "end": end.date().isoformat(),
            "strategy_terminal": sm["terminal_wealth"], "strategy_xirr": sm["xirr"],
            "strategy_cagr": sm["time_weighted_cagr"],
            "strategy_drawdown": sm["max_flow_adjusted_drawdown"],
            "spy_terminal": bm["terminal_wealth"], "spy_xirr": bm["xirr"],
            "spy_cagr": bm["time_weighted_cagr"],
            "spy_drawdown": bm["max_flow_adjusted_drawdown"],
        })
    frame = pd.DataFrame(records)
    summary = {"years": args.rolling_years, "cohorts": len(frame)}
    for column in frame.columns[2:]:
        summary[column] = {
            "p10": float(frame[column].quantile(0.10)),
            "median": float(frame[column].median()),
            "p90": float(frame[column].quantile(0.90)),
        }
    summary["beats_spy_share"] = float(
        (frame["strategy_terminal"] > frame["spy_terminal"]).mean()
    )
    return frame, summary


def main(argv=None):
    args = parse_args(argv)
    prices, raw, shares, coverage = load_market_data(args)
    rates = data.load_rates(prices.index)
    variants = {
        "treasury_first_10": (0.10, "treasury_first"),
        "treasury_first_20": (0.20, "treasury_first"),
        "treasury_first_30": (0.30, "treasury_first"),
        "immediate_10": (0.10, "immediate"),
        "immediate_20": (0.20, "immediate"),
        "immediate_30": (0.30, "immediate"),
    }
    paths, event_sets, holdings, metrics = {}, {}, {}, {}
    for name, (reserve_share, mode) in variants.items():
        path, events, positions = simulate(
            prices, raw, shares, rates, args, reserve_share=reserve_share,
            contribution_mode=mode, name=name,
        )
        paths[name], event_sets[name], holdings[name] = path, events, positions
        metrics[name] = performance_metrics(path, rates, args.initial)
        metrics[name]["ending_treasury"] = float(path["treasury"].iloc[-1])
        metrics[name]["events"] = summarize_events(events)
    spy = simulate_spy(prices, args)
    metrics["spy_1x"] = performance_metrics(spy, rates, args.initial)
    rolling_frame, rolling_summary = rolling(prices, raw, shares, rates, args)

    result = {
        "sample": {"start": prices.index[0].date().isoformat(),
                   "end": prices.index[-1].date().isoformat(), "sessions": len(prices)},
        "cash_flows": {"initial": args.initial,
                       "monthly_contribution": args.monthly_contribution},
        "rules": {
            "leverage": "3x/2x/1x at 0%/-15%/-30% flow-adjusted NAV DD; 1x->2x above -10%; 3x at new NAV high",
            "quarterly_spy_sweep": "if financed sleeve quarter return >=20%, move 10% of dollar P&L to Treasury",
            "annual_spy_sweep": "if SPY calendar-year total return >=10%, move 1% of core sleeve to Treasury",
            "reserve_rungs": "10%/20%/30%/40% of episode-start reserve at -10%/-20%/-30%/-40% NAV DD",
            "mega_seven": "top seven point-in-time market-cap estimates from the fixed historical-leader union; market-cap weighted",
            "quality_harvest": "5% current shares when SPY recovers its episode high, or after a stock quarter >=20%; profitable positions only",
            "quality_exit": "after five years, exit immediately when trailing 5y CAGR is below SPY",
            "financing": f"prior-known DGS3MO + {args.spread:.2%}",
            "dividends": "reinvested through adjusted total returns",
        },
        "variants": metrics, "holdings": holdings,
        "events": {name: [asdict(event) for event in values]
                   for name, values in event_sets.items()},
        "rolling_20y_treasury_first_20": rolling_summary,
        "coverage": coverage,
        "bias_warning": "The market-cap union contains current survivors and omits delisted historical leaders; rankings are date-aware but not survivor-free.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    daily = pd.DataFrame(index=prices.index)
    daily["spy_1x_wealth"] = spy["wealth"]
    daily["spy_1x_performance"] = spy["performance_index"]
    for name, path in paths.items():
        daily[f"{name}_wealth"] = path["wealth"]
        daily[f"{name}_performance"] = path["performance_index"]
        daily[f"{name}_treasury"] = path["treasury"]
        daily[f"{name}_quality"] = path["quality"]
        daily[f"{name}_leverage"] = path["leverage"]
    daily.to_csv(args.out / "daily.csv", index_label="date")
    rolling_frame.to_csv(args.out / "rolling_20y.csv", index=False)
    print(json.dumps({
        "sample": result["sample"], "cash_flows": result["cash_flows"],
        "variants": metrics, "rolling": rolling_summary,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
