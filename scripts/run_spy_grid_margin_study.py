"""Thirty-year SPY grid proxy with leverage, savings, and cash contributions.

Daily OHLC cannot identify the order of the high and low.  Each grid scenario is
therefore run twice: Open-Low-High-Close and Open-High-Low-Close.  The model is
a transparent proxy, not an execution-grade reconstruction of intraday fills.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RATES = ROOT / "out/strategy/spy_reverse_vault/daily.csv"
OUTPUT = ROOT / "out/strategy/spy_grid_margin/results.json"
OHLC_OUTPUT = ROOT / "out/strategy/spy_grid_margin/spy_adjusted_ohlc.csv"
ANNUAL_CONTRIBUTION = 10_000.0
MONTHLY_CONTRIBUTION = ANNUAL_CONTRIBUTION / 12.0
START = pd.Timestamp("1996-08-21")
END = pd.Timestamp("2026-08-21")


@dataclass(frozen=True)
class GridSpec:
    name: str
    spacing: float
    levels: int
    fee_bps: float
    bias: str = "long"


def load_prices() -> pd.DataFrame:
    raw = yf.download("SPY", start="1995-01-01", end="2026-08-25",
                      auto_adjust=True, actions=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs("SPY", axis=1, level="Ticker")
    frame = raw[["Open", "High", "Low", "Close"]].dropna().sort_index()
    OHLC_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OHLC_OUTPUT, index_label="date")
    return frame


def monthly_schedule(index: pd.DatetimeIndex) -> pd.Series:
    schedule = pd.Series(0.0, index=index)
    for month in range(1, 30 * 12):
        due = index[0] + pd.DateOffset(months=month)
        position = index.searchsorted(due)
        if position >= len(index):
            raise ValueError(f"window ends before contribution {month + 1}")
        schedule.iloc[position] += MONTHLY_CONTRIBUTION
    return schedule


def xirr(cash_flows: list[tuple[pd.Timestamp, float]]) -> float:
    origin = cash_flows[0][0]

    def value(rate: float) -> float:
        return sum(amount / (1 + rate) ** ((stamp - origin).days / 365.2425)
                   for stamp, amount in cash_flows)

    low, high = -0.9999, 1.0
    while value(high) > 0 and high < 1_000:
        high *= 2
    for _ in range(100):
        middle = (low + high) / 2
        if value(middle) > 0:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def exact_bucket(price: float, center: float, spacing: float, levels: int) -> int:
    raw = int(np.floor(np.log(price / center) / np.log1p(spacing) + 1e-12))
    return int(np.clip(raw, -levels, levels))


def target_leverage(bucket: int, cap: float, levels: int, bias: str) -> float:
    if bias == "long":
        return cap * (levels - bucket) / (2.0 * levels)
    if bias == "long_core":
        # Retain a long SPY core: 100% of the cap at the grid bottom,
        # 75% at its center, and 50% at the top.
        return cap * (0.75 - bucket / (4.0 * levels))
    if bias == "neutral":
        return -cap * bucket / levels
    raise ValueError(f"unknown grid bias: {bias}")


def rebalance(quantity: float, cash: float, price: float, target: float,
              fee_rate: float) -> tuple[float, float, float, bool]:
    equity = quantity * price + cash
    if equity <= 0:
        return 0.0, equity, 0.0, True
    desired = target * equity / price
    trade_value = (desired - quantity) * price
    fee = abs(trade_value) * fee_rate
    return desired, cash - trade_value - fee, abs(trade_value), False


def segment(quantity: float, cash: float, start_price: float, end_price: float,
            center: float, cap: float, spec: GridSpec) -> tuple[float, float, float, bool]:
    if end_price == start_price:
        return quantity, cash, 0.0, False
    log_step = np.log1p(spec.spacing)
    low_k = max(-spec.levels, int(np.ceil(np.log(min(start_price, end_price) / center) / log_step)))
    high_k = min(spec.levels, int(np.floor(np.log(max(start_price, end_price) / center) / log_step)))
    levels = [(k, center * (1.0 + spec.spacing) ** k)
              for k in range(low_k, high_k + 1)]
    if end_price < start_price:
        levels.reverse()
    turnover = 0.0
    for k, price in levels:
        if end_price > start_price and not (start_price < price <= end_price):
            continue
        if end_price < start_price and not (end_price <= price < start_price):
            continue
        bucket = k if end_price > start_price else k - 1
        bucket = int(np.clip(bucket, -spec.levels, spec.levels))
        target = target_leverage(bucket, cap, spec.levels, spec.bias)
        quantity, cash, traded, ruined = rebalance(
            quantity, cash, price, target, spec.fee_bps / 10_000.0)
        turnover += traded
        if ruined:
            return quantity, cash, turnover, True
    return quantity, cash, turnover, False


def withdraw_proportionally(quantity: float, cash: float,
                            amount: float, price: float) -> tuple[float, float]:
    equity = quantity * price + cash
    if equity <= 0 or amount <= 0:
        return quantity, cash
    scale = max((equity - amount) / equity, 0.0)
    return quantity * scale, cash * scale


def simulate(prices: pd.DataFrame, rates: pd.Series, spec: GridSpec,
             path_order: str) -> tuple[pd.DataFrame, dict]:
    frame = prices.loc[START:END].copy()
    ema = prices["Close"].ewm(span=20, adjust=False).mean().shift(1).reindex(frame.index)
    rates = rates.reindex(frame.index).ffill().bfill()
    contributions = monthly_schedule(frame.index)
    days = frame.index.to_series().diff().dt.days.fillna(0.0)
    known_rate = rates.shift(1).ffill().bfill()

    opening = MONTHLY_CONTRIBUTION
    trading_cash = opening * 0.70
    savings = opening * 0.30
    quantity = 0.0
    performance_nav = 1.0
    performance_peak = 1.0
    leverage_cap = 5.0
    fired: set[float] = set()
    annual_profit = 0.0
    cash_flows = [(frame.index[0], -opening)]
    total_turnover = 0.0
    total_financing = 0.0
    margin_events = 0
    rows = []

    for position, (stamp, bar) in enumerate(frame.iterrows()):
        prior_close = float(frame["Close"].iloc[position - 1] if position else bar["Open"])
        prior_trading_equity = quantity * prior_close + trading_cash
        prior_combined = prior_trading_equity + savings

        if position:
            savings *= 1.0 + float(known_rate.iloc[position]) * float(days.iloc[position]) / 365.0
            borrowed = max(-trading_cash, 0.0)
            financing = borrowed * (float(known_rate.iloc[position]) + 0.01) * float(days.iloc[position]) / 365.0
            trading_cash -= financing
            total_financing += financing

        center = float(ema.loc[stamp])
        open_price = float(bar["Open"])
        open_bucket = exact_bucket(open_price, center, spec.spacing, spec.levels)
        opening_target = target_leverage(open_bucket, leverage_cap, spec.levels, spec.bias)
        quantity, trading_cash, traded, ruined = rebalance(
            quantity, trading_cash, open_price, opening_target, spec.fee_bps / 10_000.0)
        total_turnover += traded

        intraday = ([open_price, float(bar["Low"]), float(bar["High"]), float(bar["Close"])]
                    if path_order == "OLHC" else
                    [open_price, float(bar["High"]), float(bar["Low"]), float(bar["Close"])])
        for start_price, end_price in zip(intraday, intraday[1:]):
            quantity, trading_cash, traded, segment_ruin = segment(
                quantity, trading_cash, start_price, end_price,
                center, leverage_cap, spec)
            total_turnover += traded
            ruined = ruined or segment_ruin
            if segment_ruin:
                break

        close = float(bar["Close"])
        trading_equity = quantity * close + trading_cash
        if ruined or trading_equity <= 0:
            margin_events += 1
            deficit = max(-trading_equity, 0.0)
            cover = min(savings, deficit)
            savings -= cover
            trading_cash = trading_equity + cover
            quantity = 0.0
            trading_equity = trading_cash

        daily_profit = trading_equity - prior_trading_equity
        annual_profit += daily_profit
        combined_before_flow = trading_equity + savings
        if prior_combined > 0:
            performance_nav *= combined_before_flow / prior_combined
        performance_peak = max(performance_peak, performance_nav)
        drawdown = performance_nav / performance_peak - 1.0

        contribution = float(contributions.iloc[position])
        if contribution:
            trading_cash += contribution * 0.70
            savings += contribution * 0.30
            trading_equity += contribution * 0.70
            cash_flows.append((stamp, -contribution))

        recovered = drawdown >= -1e-12
        if recovered:
            leverage_cap = 5.0
            fired.clear()
        elif drawdown <= -0.60:
            leverage_cap = 1.0
        elif drawdown <= -0.20:
            leverage_cap = min(leverage_cap, 2.0)

        for threshold, fraction in ((0.20, 1 / 3), (0.30, 1 / 2), (0.50, 1.0)):
            if threshold in fired or drawdown > -threshold:
                continue
            fired.add(threshold)
            amount = savings * fraction
            savings -= amount
            trading_cash += amount
            trading_equity += amount

        next_stamp = frame.index[position + 1] if position + 1 < len(frame) else None
        if next_stamp is None or next_stamp.year != stamp.year:
            harvest = min(max(annual_profit, 0.0) * 0.10, max(trading_equity, 0.0))
            if harvest:
                quantity, trading_cash = withdraw_proportionally(
                    quantity, trading_cash, harvest, close)
                savings += harvest
                trading_equity -= harvest
            annual_profit = 0.0

        combined = trading_equity + savings
        rows.append({
            "combined_wealth": combined,
            "trading_equity": trading_equity,
            "savings": savings,
            "performance_nav": performance_nav,
            "drawdown": drawdown,
            "leverage_cap": leverage_cap,
            "effective_leverage": abs(quantity * close) / combined if combined > 0 else np.nan,
        })

    path = pd.DataFrame(rows, index=frame.index)
    terminal = float(path["combined_wealth"].iloc[-1])
    cash_flows.append((frame.index[-1], terminal))
    years = (frame.index[-1] - frame.index[0]).days / 365.2425
    stats = {
        "terminal_wealth": terminal,
        "xirr": xirr(cash_flows),
        "time_weighted_cagr": float(performance_nav ** (1 / years) - 1),
        "max_drawdown": float(path["drawdown"].min()),
        "ending_savings": float(path["savings"].iloc[-1]),
        "median_effective_leverage": float(path["effective_leverage"].median()),
        "turnover_multiple_of_contributions": total_turnover / (ANNUAL_CONTRIBUTION * 30),
        "transaction_cost_paid": total_turnover * spec.fee_bps / 10_000.0,
        "financing_cost_paid": total_financing,
        "time_at_5x_cap": float((path["leverage_cap"] == 5.0).mean()),
        "time_at_2x_cap": float((path["leverage_cap"] == 2.0).mean()),
        "time_at_1x_cap": float((path["leverage_cap"] == 1.0).mean()),
        "margin_events": margin_events,
    }
    return path, stats


def spy_benchmark(prices: pd.DataFrame) -> dict:
    close = prices.loc[START:END, "Close"]
    contributions = monthly_schedule(close.index)
    wealth = MONTHLY_CONTRIBUTION
    cash_flows = [(close.index[0], -MONTHLY_CONTRIBUTION)]
    for position in range(1, len(close)):
        wealth *= float(close.iloc[position] / close.iloc[position - 1])
        contribution = float(contributions.iloc[position])
        if contribution:
            wealth += contribution
            cash_flows.append((close.index[position], -contribution))
    cash_flows.append((close.index[-1], wealth))
    return {"terminal_wealth": wealth, "xirr": xirr(cash_flows),
            "max_drawdown_flow_adjusted": float((close / close.cummax() - 1.0).min())}


def main() -> int:
    prices = load_prices()
    rates_source = pd.read_csv(SOURCE_RATES, parse_dates=["date"], index_col="date")
    rates = rates_source["treasury_rate"]
    specs = [
        GridSpec("wide", 0.005, 10, 2.0),
        GridSpec("base", 0.004, 12, 1.0),
        GridSpec("tight", 0.003, 15, 1.0),
        GridSpec("long_core_base", 0.004, 12, 1.0, bias="long_core"),
        GridSpec("neutral_base", 0.004, 12, 1.0, bias="neutral"),
    ]
    scenarios = {}
    for spec in specs:
        for order in ("OLHC", "OHLC"):
            _, stats = simulate(prices, rates, spec, order)
            scenarios[f"{spec.name}_{order}"] = {
                "spacing": spec.spacing, "levels_each_side": spec.levels,
                "fee_bps": spec.fee_bps, "bias": spec.bias,
                "intraday_path": order, **stats,
            }

    result = {
        "method": {
            "start": START.date().isoformat(), "end": END.date().isoformat(),
            "contributed": ANNUAL_CONTRIBUTION * 30,
            "contributions": "opening $833.33 plus 359 monthly payments; 70% trading / 30% savings",
            "grid_center": "prior-known 20-session EMA",
            "leverage": "cap 5x; latch to 2x at -20% combined flow-adjusted account DD and 1x at -60%; reset at recovery high",
            "profit_sweep": "10% positive calendar-year trading P&L to savings",
            "savings_deployment": "one-third at -20%, half remaining at -30%, all remaining at -50%",
            "funding": "prior-known DGS3MO + 1% on negative trading cash",
            "limitations": "daily OHLC path proxy; no real intraday sequence, queue priority, partial fills, taxes, broker maintenance margin, or market impact",
        },
        "spy_x1": spy_benchmark(prices),
        "scenarios": scenarios,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
