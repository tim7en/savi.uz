"""Backtest a leveraged SPY / Treasury reserve / quality-stock rotation policy.

The requested idea is encoded as an auditable first pass:

* start with 80% of capital in a daily-reset SPY sleeve at 3x and 20% in a
  Treasury reserve;
* transfer 10% of positive calendar-year SPY-sleeve trading profit to reserve;
* deploy 25%, one third, one half, and all remaining reserve after SPY closes
  10%, 20%, 30%, and 40% below its adjusted-close high;
* the first three tranches recapitalise the SPY sleeve; the final tranche buys
  a static illustrative basket of mega-cap compounders and dividend growers;
* stock positions are capped so a 79% stress loss costs at most 1% of account
  equity per name; no stop is invented;
* after the stock purchase, sell 10% of the original shares whenever the chosen
  control signal has risen another 10% from its purchase level, and move the
  proceeds to reserve;
* drawdown signals are known at one close and executed at the next close.

Two leverage interpretations are run: 3x -> 1x at a 40% drawdown, and the more
defensive 3x -> 2x at 20% -> 1x at 40%.  The main comparison runs the defensive
policy using either SPY drawdown or total-portfolio drawdown as the control
signal.  The quality basket is a current, survivor-selected illustration, not
point-in-time membership.  Results before the underlying companies' listings
use only the names then available.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = (0.10, 0.20, 0.30, 0.40)
DEPLOY_FRACTIONS = (0.25, 1.0 / 3.0, 0.50, 1.0)

QUALITY_GROUPS = {
    "mega_cap_compounders": (
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    ),
    "durable_compounders": ("BRK-B", "COST", "MCD"),
    "dividend_growers": ("WMT", "KO", "JNJ", "PG", "PEP", "ADP", "LOW", "CL"),
}


@dataclass
class Event:
    date: str
    strategy: str
    kind: str
    amount: float
    signal_drawdown: float
    spy_drawdown: float
    detail: str


@dataclass
class Lot:
    ticker: str
    original_shares: float
    remaining_shares: float
    entry_price: float
    entry_reference: float
    next_sale_rung: int
    cost: float


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="1993-01-29")
    parser.add_argument("--end", default=None)
    parser.add_argument("--initial", type=float, default=100_000.0)
    parser.add_argument("--spy-share", type=float, default=0.80)
    parser.add_argument("--harvest", type=float, default=0.10)
    parser.add_argument("--spread", type=float, default=0.01)
    parser.add_argument("--risk-per-stock", type=float, default=0.01)
    parser.add_argument("--stock-tail-loss", type=float, default=0.79)
    parser.add_argument("--min-history-years", type=float, default=3.0)
    parser.add_argument("--trade-bp", type=float, default=5.0)
    parser.add_argument("--rolling-years", type=int, default=20)
    parser.add_argument(
        "--cohort-frequency", choices=("annual", "quarterly", "monthly"),
        default="annual",
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--fundamentals", type=Path,
        default=Path("data/data/sp500_data"),
    )
    parser.add_argument(
        "--cache", type=Path,
        default=Path(".cache/yahoo_daily"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("out/strategy/spy_quality_rotation"),
    )
    return parser.parse_args(argv)


def unix_timestamp(day: str) -> int:
    return int(pd.Timestamp(day, tz="UTC").timestamp())


def yahoo_series(ticker: str, start: str, end: str | None,
                 cache: Path, refresh: bool) -> pd.Series:
    cache.mkdir(parents=True, exist_ok=True)
    safe = ticker.replace("/", "-")
    path = cache / f"{safe}.json"
    if refresh or not path.exists():
        end_day = end or "2030-01-01"
        quoted = urllib.parse.quote(ticker, safe="")
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{quoted}"
            f"?period1={unix_timestamp(start)}&period2={unix_timestamp(end_day)}"
            "&interval=1d&events=div%2Csplits&includeAdjustedClose=true"
        )
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read().decode("utf-8")
        path.write_text(raw, encoding="utf-8")

    chart = json.loads(path.read_text(encoding="utf-8"))["chart"]
    if chart.get("error") or not chart.get("result"):
        raise RuntimeError(f"Yahoo returned no data for {ticker}: {chart.get('error')}")
    result = chart["result"][0]
    stamps = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_localize(None)
    indicators = result["indicators"]
    adjusted = indicators.get("adjclose", [{}])[0].get("adjclose")
    if adjusted is None:
        adjusted = indicators["quote"][0]["close"]
    values = {}
    for stamp, value in zip(stamps, adjusted):
        if value is None:
            continue
        price = float(value)
        if math.isfinite(price) and price > 0.0:
            values[stamp.normalize()] = price
    series = pd.Series(values, dtype=float, name=ticker).sort_index()
    return series.loc[start:end]


def load_rates(index: pd.DatetimeIndex) -> pd.Series:
    local = ROOT / "data/data/macro/macro.db"
    rows = []
    if local.exists():
        connection = sqlite3.connect(f"file:{local}?mode=ro", uri=True)
        rows = connection.execute(
            "SELECT obs_date,value FROM observations WHERE series_id='DGS3MO' "
            "ORDER BY obs_date"
        ).fetchall()
        connection.close()
    local_rates = pd.Series(
        {pd.Timestamp(day): float(value) / 100.0 for day, value in rows
         if value is not None}, dtype=float
    ).sort_index()

    if local_rates.empty or local_rates.index.min() > index.min():
        url = (
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"
            f"&cosd={index.min().date().isoformat()}"
        )
        frame = pd.read_csv(url, parse_dates=["observation_date"])
        remote = pd.to_numeric(
            frame.set_index("observation_date")["DGS3MO"], errors="coerce"
        ) / 100.0
        rates = pd.concat([remote, local_rates]).sort_index()
        rates = rates[~rates.index.duplicated(keep="last")]
    else:
        rates = local_rates
    return rates.reindex(index).ffill().bfill().rename("treasury_rate")


def usable_earnings(path: Path) -> int:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))["data"]
    except (OSError, KeyError, json.JSONDecodeError):
        return 0
    count = 0
    for row in payload.get("quarterlyEarnings", []):
        eps = str(row.get("reportedEPS", "")).strip().lower()
        day = str(row.get("reportedDate", ""))[:10]
        if eps not in {"", "-", "none", "null"} and len(day) == 10:
            count += 1
    return count


def load_prices(args) -> tuple[pd.DataFrame, dict[str, dict]]:
    basket = tuple(dict.fromkeys(
        ticker for group in QUALITY_GROUPS.values() for ticker in group
    ))
    series = {"SPY": yahoo_series(
        "SPY", args.start, args.end, args.cache, args.refresh
    )}
    coverage = {}
    for ticker in basket:
        earnings = usable_earnings(args.fundamentals / f"{ticker}_earnings.json")
        if earnings < 20:
            coverage[ticker] = {"earnings_quarters": earnings, "included": False}
            continue
        try:
            values = yahoo_series(
                ticker, args.start, args.end, args.cache, args.refresh
            )
        except Exception as exc:
            coverage[ticker] = {
                "earnings_quarters": earnings, "included": False,
                "error": str(exc),
            }
            continue
        series[ticker] = values
        coverage[ticker] = {
            "earnings_quarters": earnings,
            "included": True,
            "price_start": values.index.min().date().isoformat(),
            "price_end": values.index.max().date().isoformat(),
            "group": next(name for name, members in QUALITY_GROUPS.items()
                          if ticker in members),
        }
    frame = pd.DataFrame(series).sort_index()
    spy_index = series["SPY"].index
    frame = frame.reindex(spy_index)
    for ticker in frame.columns:
        if ticker != "SPY":
            frame[ticker] = frame[ticker].ffill()
    return frame, coverage


def leverage_for(drawdown: float, policy: str) -> float:
    if policy == "constant_3x":
        return 3.0
    if policy == "late_3_to_1":
        return 1.0 if drawdown <= -0.40 else 3.0
    if policy == "step_3_2_1":
        if drawdown <= -0.40:
            return 1.0
        return 2.0 if drawdown <= -0.20 else 3.0
    raise ValueError(f"unknown leverage policy: {policy}")


def lot_value(lots: list[Lot], row: pd.Series) -> float:
    return sum(
        lot.remaining_shares * float(row.get(lot.ticker, np.nan))
        for lot in lots
        if lot.remaining_shares > 0.0 and pd.notna(row.get(lot.ticker, np.nan))
    )


def metrics(wealth: pd.Series, rates: pd.Series) -> dict:
    returns = wealth.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    years = (wealth.index[-1] - wealth.index[0]).days / 365.2425
    drawdown = wealth / wealth.cummax() - 1.0
    volatility = returns.iloc[1:].std(ddof=1) * math.sqrt(252.0)
    excess = returns.iloc[1:] - rates.reindex(returns.index).iloc[1:] / 252.0
    std = returns.iloc[1:].std(ddof=1)
    return {
        "terminal": float(wealth.iloc[-1]),
        "multiple": float(wealth.iloc[-1] / wealth.iloc[0]),
        "cagr": float((wealth.iloc[-1] / wealth.iloc[0]) ** (1.0 / years) - 1.0)
        if wealth.iloc[-1] > 0.0 else -1.0,
        "max_drawdown": float(drawdown.min()),
        "max_drawdown_date": drawdown.idxmin().date().isoformat(),
        "annual_volatility": float(volatility),
        "sharpe_vs_treasury": float(excess.mean() / std * math.sqrt(252.0))
        if std > 0.0 else None,
        "worst_day": float(returns.min()),
        "time_below_20": float((drawdown <= -0.20).mean()),
    }


def simulate(prices: pd.DataFrame, rates: pd.Series, args, name: str,
             leverage_policy: str, staging: bool,
             quality_at_40: bool, harvest_share: float,
             signal_source: str = "spy") -> tuple[pd.DataFrame, list[Event]]:
    if signal_source not in {"spy", "portfolio"}:
        raise ValueError(f"unknown signal source: {signal_source}")
    spy = prices["SPY"].dropna()
    prices = prices.reindex(spy.index)
    rates = rates.reindex(spy.index).ffill().bfill()
    spy_return = spy.pct_change().fillna(0.0)
    spy_drawdown = spy / spy.cummax() - 1.0
    days = spy.index.to_series().diff().dt.days.fillna(0.0)
    year_end = spy.index.to_series().dt.year.ne(
        spy.index.to_series().shift(-1).dt.year
    ) & spy.index.to_series().dt.month.eq(12)

    main = args.initial * args.spy_share
    reserve = args.initial * (1.0 - args.spy_share)
    annual_spy_profit = 0.0
    lots: list[Lot] = []
    fired: set[float] = set()
    events: list[Event] = []
    rows = []
    ruined = False
    cost_rate = args.trade_bp / 10_000.0
    portfolio_high = args.initial
    previous_portfolio_drawdown = 0.0

    for position, stamp in enumerate(spy.index):
        signal_position = max(0, position - 1)
        signal_drawdown = (
            float(spy_drawdown.iloc[signal_position])
            if signal_source == "spy"
            else previous_portfolio_drawdown
        )
        applied_leverage = leverage_for(signal_drawdown, leverage_policy)

        if position:
            reserve *= 1.0 + float(rates.iloc[position - 1]) * float(days.iloc[position]) / 365.0
            if main > 0.0:
                financing = ((applied_leverage - 1.0)
                             * (float(rates.iloc[position - 1]) + args.spread)
                             * float(days.iloc[position]) / 365.0)
                daily_return = applied_leverage * float(spy_return.iloc[position]) - financing
                profit = main * daily_return
                main += profit
                annual_spy_profit += profit
                if main <= 0.0:
                    main = 0.0
                    ruined = True

        current_row = prices.iloc[position]
        current_quality = lot_value(lots, current_row)
        pre_action_total = main + reserve + current_quality
        recovery_reference = (
            float(spy.iloc[position])
            if signal_source == "spy"
            else pre_action_total
        )

        # Sell one tenth at each 10% recovery in the selected control signal.
        for lot in lots:
            stock_price = current_row.get(lot.ticker, np.nan)
            if pd.isna(stock_price) or lot.remaining_shares <= 0.0:
                continue
            while (lot.next_sale_rung <= 10
                   and recovery_reference / lot.entry_reference
                   >= 1.0 + lot.next_sale_rung * 0.10 - 1e-12):
                shares = min(lot.original_shares * 0.10, lot.remaining_shares)
                proceeds = shares * float(stock_price) * (1.0 - cost_rate)
                lot.remaining_shares -= shares
                reserve += proceeds
                events.append(Event(
                    stamp.date().isoformat(), name, "quality_sale", proceeds,
                    signal_drawdown,
                    float(spy_drawdown.iloc[position]),
                    f"{lot.ticker}; {signal_source} "
                    f"+{lot.next_sale_rung * 10}% from entry",
                ))
                lot.next_sale_rung += 1

        if staging:
            if signal_drawdown >= -1e-12:
                fired.clear()
            for threshold, fraction in zip(THRESHOLDS, DEPLOY_FRACTIONS):
                if threshold in fired or signal_drawdown > -threshold:
                    continue
                fired.add(threshold)
                amount = reserve * fraction
                if amount <= 0.0:
                    continue
                if threshold < 0.40 or not quality_at_40:
                    reserve -= amount
                    main += amount
                    events.append(Event(
                        stamp.date().isoformat(), name, "deploy_spy", amount,
                        signal_drawdown,
                        float(spy_drawdown.iloc[position]),
                        f"-{threshold:.0%} reserve rung",
                    ))
                    continue

                history_cutoff = stamp - pd.Timedelta(days=365.2425 * args.min_history_years)
                available = [
                    ticker for ticker in prices.columns if ticker != "SPY"
                    and pd.notna(current_row.get(ticker, np.nan))
                    and prices[ticker].first_valid_index() is not None
                    and prices[ticker].first_valid_index() <= history_cutoff
                ]
                if not available:
                    continue
                account = main + reserve + lot_value(lots, current_row)
                per_name_cap = args.risk_per_stock * account / args.stock_tail_loss
                per_name = min(amount / len(available), per_name_cap)
                deployed = 0.0
                for ticker in available:
                    stock_price = float(current_row[ticker])
                    cash = min(per_name, reserve)
                    if cash <= 0.0:
                        break
                    shares = cash * (1.0 - cost_rate) / stock_price
                    reserve -= cash
                    deployed += cash
                    lots.append(Lot(
                        ticker=ticker,
                        original_shares=shares,
                        remaining_shares=shares,
                        entry_price=stock_price,
                        entry_reference=recovery_reference,
                        next_sale_rung=1,
                        cost=cash,
                    ))
                events.append(Event(
                    stamp.date().isoformat(), name, "deploy_quality", deployed,
                    signal_drawdown,
                    float(spy_drawdown.iloc[position]),
                    f"-{threshold:.0%} rung; {len(available)} names; "
                    f"${per_name:,.0f} per name cap",
                ))

        if bool(year_end.iloc[position]) and harvest_share > 0.0:
            amount = min(max(annual_spy_profit, 0.0) * harvest_share, main)
            if amount > 0.0:
                main -= amount
                reserve += amount
                events.append(Event(
                    stamp.date().isoformat(), name, "annual_harvest", amount,
                    signal_drawdown,
                    float(spy_drawdown.iloc[position]),
                    "10% of positive calendar-year SPY-sleeve profit",
                ))
            annual_spy_profit = 0.0

        current_quality = lot_value(lots, current_row)
        total = main + reserve + current_quality
        portfolio_high = max(portfolio_high, total)
        portfolio_drawdown = total / portfolio_high - 1.0
        previous_portfolio_drawdown = portfolio_drawdown
        rows.append({
            "wealth": total,
            "spy_sleeve": main,
            "reserve": reserve,
            "quality_sleeve": current_quality,
            "spy_drawdown": float(spy_drawdown.iloc[position]),
            "portfolio_drawdown": portfolio_drawdown,
            "action_drawdown": (
                float(spy_drawdown.iloc[position])
                if signal_source == "spy"
                else portfolio_drawdown
            ),
            "applied_leverage": applied_leverage,
            "spy_gross_exposure": (main * applied_leverage / total) if total > 0 else 0.0,
            "quality_exposure": (current_quality / total) if total > 0 else 0.0,
            "ruined": ruined,
        })

    return pd.DataFrame(rows, index=spy.index), events


def quantiles(values: pd.Series) -> dict:
    return {
        "min": float(values.min()),
        "p10": float(values.quantile(0.10)),
        "median": float(values.median()),
        "p90": float(values.quantile(0.90)),
        "max": float(values.max()),
    }


def rolling_study(prices: pd.DataFrame, rates: pd.Series, args) -> dict:
    starts = prices.index.to_series().groupby(prices.index.to_period("M")).first()
    stride = {"monthly": 1, "quarterly": 3, "annual": 12}[args.cohort_frequency]
    starts = starts.iloc[::stride]
    records = []
    for start in starts:
        target = start + pd.DateOffset(years=args.rolling_years)
        if target > prices.index[-1]:
            continue
        end_position = prices.index.searchsorted(target, side="right") - 1
        end = prices.index[end_position]
        window_prices = prices.loc[start:end]
        window_rates = rates.loc[start:end]
        step, events = simulate(
            window_prices, window_rates, args, "quality_step_rolling",
            "step_3_2_1", True, True, args.harvest,
        )
        portfolio_step, portfolio_events = simulate(
            window_prices, window_rates, args, "quality_portfolio_step_rolling",
            "step_3_2_1", True, True, args.harvest,
            signal_source="portfolio",
        )
        all_spy, _ = simulate(
            window_prices, window_rates, args, "all_spy_step_rolling",
            "step_3_2_1", True, False, args.harvest,
        )
        spy = args.initial * window_prices["SPY"] / window_prices["SPY"].iloc[0]
        step_stats = metrics(step["wealth"], window_rates)
        portfolio_stats = metrics(portfolio_step["wealth"], window_rates)
        all_spy_stats = metrics(all_spy["wealth"], window_rates)
        spy_stats = metrics(spy, window_rates)
        records.append({
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
            "quality_cagr": step_stats["cagr"],
            "quality_max_drawdown": step_stats["max_drawdown"],
            "quality_terminal": step_stats["terminal"],
            "portfolio_signal_cagr": portfolio_stats["cagr"],
            "portfolio_signal_max_drawdown": portfolio_stats["max_drawdown"],
            "portfolio_signal_terminal": portfolio_stats["terminal"],
            "all_spy_cagr": all_spy_stats["cagr"],
            "all_spy_max_drawdown": all_spy_stats["max_drawdown"],
            "all_spy_terminal": all_spy_stats["terminal"],
            "spy_cagr": spy_stats["cagr"],
            "spy_max_drawdown": spy_stats["max_drawdown"],
            "spy_terminal": spy_stats["terminal"],
            "quality_deployments": sum(e.kind == "deploy_quality" for e in events),
            "portfolio_signal_quality_deployments": sum(
                e.kind == "deploy_quality" for e in portfolio_events
            ),
        })
    frame = pd.DataFrame(records)
    return {
        "years": args.rolling_years,
        "cohort_frequency": args.cohort_frequency,
        "cohorts": len(frame),
        "quality_cagr": quantiles(frame["quality_cagr"]),
        "quality_max_drawdown": quantiles(frame["quality_max_drawdown"]),
        "portfolio_signal_cagr": quantiles(frame["portfolio_signal_cagr"]),
        "portfolio_signal_max_drawdown": quantiles(
            frame["portfolio_signal_max_drawdown"]
        ),
        "spy_cagr": quantiles(frame["spy_cagr"]),
        "spy_max_drawdown": quantiles(frame["spy_max_drawdown"]),
        "quality_beats_spy_share": float(
            (frame["quality_terminal"] > frame["spy_terminal"]).mean()
        ),
        "quality_beats_all_spy_share": float(
            (frame["quality_terminal"] > frame["all_spy_terminal"]).mean()
        ),
        "portfolio_signal_beats_spy_share": float(
            (frame["portfolio_signal_terminal"] > frame["spy_terminal"]).mean()
        ),
        "portfolio_signal_beats_spy_signal_share": float(
            (frame["portfolio_signal_terminal"] > frame["quality_terminal"]).mean()
        ),
        "cohorts_with_quality_deployment": int(
            (frame["quality_deployments"] > 0).sum()
        ),
        "cohorts_with_portfolio_signal_quality_deployment": int(
            (frame["portfolio_signal_quality_deployments"] > 0).sum()
        ),
        "records": records,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    if not 0.0 < args.spy_share < 1.0:
        raise ValueError("--spy-share must be between zero and one")
    prices, coverage = load_prices(args)
    rates = load_rates(prices.index)

    variants = {
        "static_3x_80_20": ("constant_3x", False, False, 0.0, "spy"),
        "all_spy_step": ("step_3_2_1", True, False, args.harvest, "spy"),
        "quality_step": ("step_3_2_1", True, True, args.harvest, "spy"),
        "quality_portfolio_step": (
            "step_3_2_1", True, True, args.harvest, "portfolio"
        ),
        "quality_late": ("late_3_to_1", True, True, args.harvest, "spy"),
    }
    paths, event_sets, summaries = {}, {}, {}
    for name, (policy, staging, quality, harvest, signal_source) in variants.items():
        path, events = simulate(
            prices, rates, args, name, policy, staging, quality, harvest,
            signal_source=signal_source,
        )
        paths[name], event_sets[name] = path, events
        summaries[name] = metrics(path["wealth"], rates)
        summaries[name].update({
            "signal_source": signal_source,
            "ending_reserve": float(path["reserve"].iloc[-1]),
            "ending_quality": float(path["quality_sleeve"].iloc[-1]),
            "annual_harvested": float(sum(
                event.amount for event in events if event.kind == "annual_harvest"
            )),
            "quality_deployed": float(sum(
                event.amount for event in events if event.kind == "deploy_quality"
            )),
            "quality_sale_proceeds": float(sum(
                event.amount for event in events if event.kind == "quality_sale"
            )),
        })

    spy_wealth = args.initial * prices["SPY"] / prices["SPY"].iloc[0]
    summaries["spy_1x"] = metrics(spy_wealth, rates)
    rolling = rolling_study(prices, rates, args)

    result = {
        "assumptions": {
            "initial": args.initial,
            "starting_spy_equity_share": args.spy_share,
            "starting_reserve_share": 1.0 - args.spy_share,
            "annual_spy_profit_harvest": args.harvest,
            "reserve_rungs": list(THRESHOLDS),
            "reserve_fractions": list(DEPLOY_FRACTIONS),
            "quality_trigger": -0.40,
            "action_signal_variants": {
                "quality_step": "prior-close SPY drawdown from its high",
                "quality_portfolio_step": (
                    "prior-close total-portfolio drawdown from its NAV high"
                ),
            },
            "quality_sale_rule": (
                "10% of original shares at every +10% rebound in the selected "
                "SPY or total-portfolio control signal"
            ),
            "risk_per_stock": args.risk_per_stock,
            "stock_tail_loss_assumption": args.stock_tail_loss,
            "minimum_price_history_years_at_purchase": args.min_history_years,
            "financing": f"prior-known DGS3MO + {args.spread:.2%}",
            "reserve_yield": "prior-known DGS3MO",
            "stock_trade_cost_each_side": args.trade_bp / 10_000.0,
            "signal_execution": "next close",
            "dividends": "reinvested through adjusted-close total returns",
            "taxes": "omitted",
            "quality_bias_warning": (
                "Static present-day quality basket and present-day earnings "
                "coverage create survivorship and look-ahead bias."
            ),
        },
        "sample": {
            "start": prices.index[0].date().isoformat(),
            "end": prices.index[-1].date().isoformat(),
            "sessions": len(prices),
        },
        "quality_groups": QUALITY_GROUPS,
        "coverage": coverage,
        "full_period": summaries,
        "rolling": {key: value for key, value in rolling.items() if key != "records"},
        "events": {
            name: [asdict(event) for event in events]
            for name, events in event_sets.items()
        },
    }

    args.out.mkdir(parents=True, exist_ok=True)
    result_path = args.out / "results.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(rolling["records"]).to_csv(
        args.out / "rolling_20y.csv", index=False
    )
    all_events = [asdict(event) for events in event_sets.values() for event in events]
    pd.DataFrame(all_events).to_csv(args.out / "events.csv", index=False)

    daily = pd.DataFrame(index=prices.index)
    daily["spy_adjusted_close"] = prices["SPY"]
    daily["spy_drawdown"] = prices["SPY"] / prices["SPY"].cummax() - 1.0
    daily["spy_1x"] = spy_wealth
    for name, path in paths.items():
        daily[name] = path["wealth"]
        daily[f"{name}_reserve"] = path["reserve"]
        daily[f"{name}_quality"] = path["quality_sleeve"]
        daily[f"{name}_spy_gross"] = path["spy_gross_exposure"]
        daily[f"{name}_action_drawdown"] = path["action_drawdown"]
    daily.to_csv(args.out / "daily.csv", index_label="date")

    print(json.dumps({
        "sample": result["sample"],
        "full_period": result["full_period"],
        "rolling": result["rolling"],
        "quality_events": result["events"]["quality_step"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
