"""Plot a reverse-leverage SPY account with a self-funded savings vault.

The trading sleeve uses the reverse 3x/2x/1x hysteresis schedule from
``run_spy_drawdown_hysteresis.py``.  At each calendar year-end, 10% of positive
trading P&L is transferred to a savings sleeve earning the 3-month Treasury
rate.  The savings sleeve is transferred back into trading on the first 50%
strategy drawdown from a flow-adjusted performance high.  The trigger rearms
only after the strategy NAV recovers its old high.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from run_spy_drawdown_hysteresis import (
    backtest,
    drawdown_rungs,
    load_rates,
    load_spy,
    metrics,
)


@dataclass
class VaultEvent:
    date: str
    kind: str
    amount: float
    strategy_drawdown: float


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="1993-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--initial", type=float, default=10_000.0)
    parser.add_argument("--harvest", type=float, default=0.10,
                        help="share of positive calendar-year trading P&L saved")
    parser.add_argument("--deploy-drawdown", type=float, default=0.50)
    parser.add_argument("--spread", type=float, default=0.01)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/spy_reverse_vault"))
    return parser.parse_args(argv)


def simulate_vault(strategy_return: pd.Series,
                   treasury_rate: pd.Series,
                   initial: float = 10_000.0,
                   harvest_share: float = 0.10,
                   deploy_drawdown: float = 0.50) -> tuple[pd.DataFrame, list[VaultEvent]]:
    """Run the two-sleeve policy; all transfers are internal cash flows."""
    index = strategy_return.index
    stamps = index.to_series()
    # Harvest only completed calendar years.  The final partial year is left in
    # trading rather than pretending the backtest end date is a year-end.
    year_end = stamps.dt.year.ne(stamps.shift(-1).dt.year) & stamps.dt.month.eq(12)
    days = index.to_series().diff().dt.days.fillna(0.0)
    known_rate = treasury_rate.shift(1).ffill().bfill()

    main = initial
    vault = 0.0
    strategy_nav = 1.0
    strategy_peak = 1.0
    annual_trading_profit = 0.0
    deployment_armed = True
    events: list[VaultEvent] = []
    rows = []

    for position, stamp in enumerate(index):
        daily_return = float(strategy_return.iloc[position])
        if position:
            vault *= 1.0 + float(known_rate.iloc[position]) * float(days.iloc[position]) / 365.0
            daily_profit = main * daily_return
            main += daily_profit
            annual_trading_profit += daily_profit
            strategy_nav *= 1.0 + daily_return

        if strategy_nav >= strategy_peak * (1.0 - 1e-12):
            strategy_peak = max(strategy_peak, strategy_nav)
            deployment_armed = True
        drawdown = strategy_nav / strategy_peak - 1.0

        if deployment_armed and drawdown <= -deploy_drawdown and vault > 0.0:
            amount = vault
            main += amount
            vault = 0.0
            deployment_armed = False
            events.append(VaultEvent(stamp.date().isoformat(), "deploy", amount, drawdown))

        if bool(year_end.iloc[position]):
            harvest = max(annual_trading_profit, 0.0) * harvest_share
            harvest = min(harvest, max(main, 0.0))
            if harvest > 0.0:
                main -= harvest
                vault += harvest
                events.append(VaultEvent(stamp.date().isoformat(), "harvest", harvest, drawdown))
            annual_trading_profit = 0.0

        rows.append({
            "main_account": main,
            "savings_vault": vault,
            "combined_wealth": main + vault,
            "strategy_nav": strategy_nav,
            "strategy_drawdown": drawdown,
        })

    return pd.DataFrame(rows, index=index), events


def path_metrics(wealth: pd.Series) -> dict:
    years = (wealth.index[-1] - wealth.index[0]).days / 365.2425
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "terminal": float(wealth.iloc[-1]),
        "cagr": float((wealth.iloc[-1] / wealth.iloc[0]) ** (1.0 / years) - 1.0),
        "max_drawdown": float(drawdown.min()),
        "max_drawdown_date": drawdown.idxmin().date().isoformat(),
    }


def make_plot(daily: pd.DataFrame, events: list[VaultEvent], out: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0]}, constrained_layout=True,
    )
    top.plot(daily.index, daily["reverse_no_vault"], color="#64748b", lw=1.8,
             label="Reverse 3x/2x/1x, no vault")
    top.plot(daily.index, daily["combined_wealth"], color="#0f766e", lw=2.2,
             label="Reverse + 10% profit vault (combined)")
    top.plot(daily.index, daily["main_account"], color="#2563eb", lw=1.25,
             alpha=0.9, label="Trading account")
    top.plot(daily.index, daily["savings_vault"].replace(0.0, float("nan")),
             color="#d97706", lw=1.4, ls="--", label="Savings vault")
    top.plot(daily.index, daily["spy_hold"], color="#111827", lw=1.4,
             alpha=0.75, label="SPY 1x hold")
    top.set_yscale("log")
    top.set_ylabel("Account value, USD (log scale)")
    top.set_title("SPY reverse leverage with a self-funded drawdown reserve")
    top.legend(loc="upper left", ncol=2, frameon=True)

    bottom.fill_between(daily.index, daily["strategy_drawdown"] * 100.0, 0.0,
                        color="#ef4444", alpha=0.20)
    bottom.plot(daily.index, daily["strategy_drawdown"] * 100.0,
                color="#b91c1c", lw=1.3, label="Reverse-strategy drawdown")
    combined_drawdown = daily["combined_wealth"] / daily["combined_wealth"].cummax() - 1.0
    bottom.plot(daily.index, combined_drawdown * 100.0,
                color="#0f766e", lw=1.5, label="Combined account drawdown")
    bottom.axhline(-50.0, color="#7f1d1d", lw=1.1, ls="--",
                   label="Vault deployment trigger")
    for event in events:
        if event.kind != "deploy":
            continue
        stamp = pd.Timestamp(event.date)
        bottom.scatter(stamp, event.strategy_drawdown * 100.0, s=55,
                       color="#f59e0b", edgecolor="#78350f", zorder=5)
        bottom.annotate(
            f"Deploy ${event.amount:,.0f}",
            (stamp, event.strategy_drawdown * 100.0), xytext=(8, -22),
            textcoords="offset points", fontsize=8, color="#78350f",
        )
    bottom.set_ylabel("Drawdown from ATH")
    bottom.set_xlabel("Date")
    bottom.legend(loc="lower right", frameon=True)
    bottom.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")

    figure.savefig(out / "reverse_vault.png", dpi=170)
    figure.savefig(out / "reverse_vault.svg")
    plt.close(figure)


def main(argv=None) -> int:
    args = parse_args(argv)
    prices = load_spy(args.start, args.end)
    rates = load_rates(prices.index)
    asset_return = prices["Adj Close"].pct_change().fillna(0.0)
    rungs, _, _ = drawdown_rungs(prices["Adj Close"])
    reverse_target = rungs.map({0: 3.0, 1: 2.0, 2: 1.0})
    reverse = backtest(asset_return, reverse_target, rates, args.spread,
                       initial=args.initial)
    hold = backtest(asset_return, pd.Series(1.0, index=prices.index), rates,
                    args.spread, initial=args.initial)
    vault_path, events = simulate_vault(
        reverse["strategy_return"], rates, args.initial,
        args.harvest, args.deploy_drawdown,
    )

    daily = pd.concat([vault_path, pd.DataFrame({
        "reverse_no_vault": reverse["wealth"],
        "spy_hold": hold["wealth"],
        "reverse_leverage": reverse["applied_leverage"],
        "treasury_rate": rates,
    })], axis=1)
    harvests = [event for event in events if event.kind == "harvest"]
    deployments = [event for event in events if event.kind == "deploy"]
    report = {
        "assumptions": {
            "initial": args.initial,
            "annual_profit_harvest_share": args.harvest,
            "vault_yield": "DGS3MO",
            "deployment_trigger": -args.deploy_drawdown,
            "financing": f"DGS3MO + {args.spread:.2%}",
            "leverage": "reverse 3x/2x/1x; SPY adjusted-close ATH/-10%/-50%",
        },
        "sample": {"start": prices.index[0].date().isoformat(),
                   "end": prices.index[-1].date().isoformat(),
                   "observations": len(prices)},
        "reverse_no_vault": path_metrics(reverse["wealth"]),
        "spy_hold": path_metrics(hold["wealth"]),
        "vault_policy_combined": path_metrics(vault_path["combined_wealth"]),
        "vault_policy_main": path_metrics(vault_path["main_account"]),
        "ending_savings_vault": float(vault_path["savings_vault"].iloc[-1]),
        "total_harvested": sum(event.amount for event in harvests),
        "total_deployed": sum(event.amount for event in deployments),
        "harvest_count": len(harvests),
        "deployment_count": len(deployments),
        "events": [event.__dict__ for event in events],
    }

    args.out.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.out / "daily.csv", index_label="date")
    (args.out / "results.json").write_text(json.dumps(report, indent=2),
                                            encoding="utf-8")
    make_plot(daily, events, args.out)

    print(json.dumps(report, indent=2))
    print(f"Wrote {args.out / 'reverse_vault.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
