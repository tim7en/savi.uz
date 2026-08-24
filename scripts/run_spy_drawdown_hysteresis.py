"""Backtest a drawdown-triggered SPY leverage ratchet over SPY's full history.

The requested ladder is stateful: hold 1x until SPY closes 10% below its
dividend-adjusted all-time high, hold 2x until either a 50% drawdown is reached
(then hold 3x) or the old high is recovered, and reset directly to 1x only on
that recovery.  A close signal sets the following session's exposure, avoiding
look-ahead.  Exposure is reset daily to the target leverage.

Borrowing is charged on (leverage - 1) times account equity using the prior
available 3-month Treasury yield plus a configurable spread.  Calendar days,
not trading days, accrue interest.  Results omit taxes, bid/ask spread, market
impact, and forced-liquidation rules.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class Episode:
    entry: str
    recovery: str | None
    deepest_rung_date: str | None
    trough_date: str
    trough_drawdown: float
    calendar_days: int | None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="1993-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--spread", type=float, default=0.01,
                        help="annual financing spread over DGS3MO (default 1%%)")
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/spy_drawdown_hysteresis"))
    return parser.parse_args(argv)


def load_spy(start: str, end: str | None) -> pd.DataFrame:
    frame = yf.download(
        "SPY", start=start, end=end, interval="1d", auto_adjust=False,
        actions=True, progress=False, threads=False,
    )
    if frame.empty:
        raise RuntimeError("Yahoo returned no SPY observations")
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.xs("SPY", axis=1, level="Ticker")
    return frame[["Close", "Adj Close"]].dropna().sort_index()


def load_rates(index: pd.DatetimeIndex) -> pd.Series:
    url = ("https://fred.stlouisfed.org/graph/fredgraph.csv?"
           f"id=DGS3MO&cosd={index[0].date().isoformat()}")
    frame = pd.read_csv(url, parse_dates=["observation_date"])
    rates = pd.to_numeric(frame.set_index("observation_date")["DGS3MO"],
                          errors="coerce") / 100.0
    return rates.reindex(index).ffill().bfill().rename("treasury_rate")


def drawdown_rungs(signal: pd.Series,
                   shallow: float = -0.10,
                   deep: float = -0.50) -> tuple[pd.Series, pd.Series, list[Episode]]:
    """Return close-known state (0/1/2), drawdown, and completed/open episodes."""
    peak = float(signal.iloc[0])
    rung = 0
    rungs: list[int] = []
    drawdowns: list[float] = []
    episodes: list[Episode] = []
    active: dict | None = None

    for stamp, value_obj in signal.items():
        value = float(value_obj)
        recovered = value >= peak * (1.0 - 1e-12)
        if recovered:
            if active is not None:
                episodes.append(Episode(
                    entry=active["entry"].date().isoformat(),
                    recovery=stamp.date().isoformat(),
                    deepest_rung_date=(active["deep"].date().isoformat()
                                       if active["deep"] is not None else None),
                    trough_date=active["trough_date"].date().isoformat(),
                    trough_drawdown=active["trough"],
                    calendar_days=(stamp - active["entry"]).days,
                ))
                active = None
            peak = max(peak, value)
            rung = 0

        drawdown = value / peak - 1.0
        if rung == 0 and drawdown <= shallow:
            rung = 1
            active = {"entry": stamp, "deep": None,
                      "trough_date": stamp, "trough": drawdown}
        elif rung == 1 and drawdown <= deep:
            rung = 2
            assert active is not None
            active["deep"] = stamp

        if active is not None and drawdown < active["trough"]:
            active["trough"] = drawdown
            active["trough_date"] = stamp
        rungs.append(rung)
        drawdowns.append(drawdown)

    if active is not None:
        episodes.append(Episode(
            entry=active["entry"].date().isoformat(), recovery=None,
            deepest_rung_date=(active["deep"].date().isoformat()
                               if active["deep"] is not None else None),
            trough_date=active["trough_date"].date().isoformat(),
            trough_drawdown=active["trough"], calendar_days=None,
        ))
    return (pd.Series(rungs, index=signal.index, name="rung"),
            pd.Series(drawdowns, index=signal.index, name="spy_drawdown"),
            episodes)


def backtest(asset_return: pd.Series,
             close_target: pd.Series,
             annual_rate: pd.Series,
             spread: float,
             initial: float = 100.0) -> pd.DataFrame:
    """Daily-reset leverage with prior-close signals and calendar-day carry."""
    applied = close_target.shift(1).fillna(float(close_target.iloc[0]))
    days = asset_return.index.to_series().diff().dt.days.fillna(0.0)
    known_rate = annual_rate.shift(1).ffill().bfill()
    borrowed = (applied - 1.0).clip(lower=0.0)
    financing = borrowed * (known_rate + spread) * days / 365.0
    strategy_return = applied * asset_return - financing
    wealth = initial * (1.0 + strategy_return).cumprod()
    return pd.DataFrame({
        "target_close": close_target,
        "applied_leverage": applied,
        "financing_return": financing,
        "strategy_return": strategy_return,
        "wealth": wealth,
    })


def metrics(path: pd.DataFrame, annual_rate: pd.Series) -> dict:
    wealth = path["wealth"]
    returns = path["strategy_return"].iloc[1:]
    years = (wealth.index[-1] - wealth.index[0]).days / 365.2425
    underwater = wealth / wealth.cummax() - 1.0
    trough = underwater.idxmin()
    peak = wealth.loc[:trough].idxmax()
    after = wealth.loc[trough:]
    recovered = after[after >= wealth.loc[peak]]
    rf_daily = annual_rate.reindex(returns.index).ffill().bfill() / 252.0
    excess = returns - rf_daily
    volatility = returns.std(ddof=1) * math.sqrt(252.0)
    sharpe = excess.mean() / returns.std(ddof=1) * math.sqrt(252.0)
    return {
        "terminal_100": float(wealth.iloc[-1]),
        "cagr": float((wealth.iloc[-1] / wealth.iloc[0]) ** (1.0 / years) - 1.0),
        "max_drawdown": float(underwater.min()),
        "max_drawdown_peak": peak.date().isoformat(),
        "max_drawdown_trough": trough.date().isoformat(),
        "max_drawdown_recovery": (recovered.index[0].date().isoformat()
                                  if len(recovered) else None),
        "annual_volatility": float(volatility),
        "sharpe_vs_treasury": float(sharpe),
        "worst_day": float(returns.min()),
        "worst_day_date": returns.idxmin().date().isoformat(),
        "mean_leverage": float(path["applied_leverage"].mean()),
    }


def pct(value: float) -> str:
    return f"{value:.2%}"


def money(value: float) -> str:
    return f"${value:,.2f}"


def build_report(summary: dict, episodes: list[dict], sensitivity: list[dict],
                 start: str, end: str, observations: int, spread: float,
                 exposure: dict, changes: int, turnover: float,
                 break_even_spread: float, weighted_treasury: float) -> str:
    order = ["hold_1x", "ratchet_1_2_3", "reverse_3_2_1",
             "continuous_2x", "inverse_minus_1x"]
    labels = {
        "hold_1x": "SPY hold 1x",
        "ratchet_1_2_3": "Ratchet 1→2→3",
        "reverse_3_2_1": "Reverse 3→2→1",
        "continuous_2x": "Continuous 2x",
        "inverse_minus_1x": "Daily −1x SPY",
    }
    lines = [
        "# SPY drawdown-leverage hysteresis study", "",
        f"Daily observations: **{observations:,}**, {start} through {end}.",
        f"Base financing: prior-known 3-month Treasury yield + **{spread:.1%}**; "
        "borrowed notional accrues over calendar days.", "",
        "## Full-period comparison", "",
        "| Strategy | $100 became | CAGR | Max drawdown | Ann. vol | Sharpe |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in order:
        item = summary[name]
        lines.append(
            f"| {labels[name]} | {money(item['terminal_100'])} | "
            f"{pct(item['cagr'])} | {pct(item['max_drawdown'])} | "
            f"{pct(item['annual_volatility'])} | {item['sharpe_vs_treasury']:.2f} |"
        )
    ratchet = summary["ratchet_1_2_3"]
    lines += [
        "", "## Frequency and implementation", "",
        f"There were **{len(episodes)}** separate 10% drawdown episodes and "
        f"**{changes}** regime changes. The 50% / 3x rung fired "
        f"**{sum(1 for e in episodes if e['deepest_rung_date'])}** time(s).",
        f"Applied exposure was 1x on {exposure['1x_days']:,} sessions "
        f"({exposure['1x_share']:.1%}), 2x on {exposure['2x_days']:,} "
        f"({exposure['2x_share']:.1%}), and 3x on {exposure['3x_days']:,} "
        f"({exposure['3x_share']:.1%}).",
        f"Although the rung changed only {changes} times, maintaining exact daily "
        f"leverage required trading on {exposure['rebalance_days']:,} sessions; "
        f"estimated gross notional turnover was about {turnover:.1f}x account "
        "equity per year before bid/ask costs.", "",
        f"The borrowed exposure saw a weighted-average Treasury rate of "
        f"{weighted_treasury:.2%}. The ratchet's break-even financing spread "
        f"versus 1x holding was about {break_even_spread:.2%} over Treasury; "
        "above that, it finished with less money than 1x in this sample.", "",
        "## Ratchet drawdown anatomy", "",
        f"The worst account drawdown was {pct(ratchet['max_drawdown'])}, from "
        f"{ratchet['max_drawdown_peak']} to {ratchet['max_drawdown_trough']}; "
        f"the account did not regain that equity high until "
        f"{ratchet['max_drawdown_recovery']}.", "",
        "| 10% entry | Recovery of old SPY high | SPY trough | 3x date | Days |",
        "|---|---|---:|---|---:|",
    ]
    for episode in episodes:
        lines.append(
            f"| {episode['entry']} | {episode['recovery'] or 'open'} | "
            f"{pct(episode['trough_drawdown'])} | "
            f"{episode['deepest_rung_date'] or '—'} | "
            f"{episode['calendar_days'] if episode['calendar_days'] is not None else '—'} |"
        )
    lines += [
        "", "## Financing sensitivity for the ratchet", "",
        "| Borrowing assumption | $100 became | CAGR | Max drawdown |",
        "|---|---:|---:|---:|",
    ]
    for item in sensitivity:
        lines.append(
            f"| {item['label']} | {money(item['terminal_100'])} | "
            f"{pct(item['cagr'])} | {pct(item['max_drawdown'])} |"
        )
    lines += [
        "", "## Model boundaries", "",
        "- The signal and returns use adjusted close, so dividends are reinvested "
        "and ex-dividend dates do not create false drawdowns.",
        "- A close signal changes exposure for the following session. There is no "
        "same-close look-ahead.",
        "- Leverage is reset daily. A fixed-debt margin position behaves very "
        "differently: its leverage rises as SPY falls and can be liquidated before "
        "the 50% rung is reached.",
        "- No tax, slippage, spread, leveraged-product fee, tracking error, or "
        "broker liquidation is modeled. The funding sensitivity isolates the "
        "largest explicit omitted cost.",
        "- Only one SPY episode reached 50%, so the 3x rule has one historical "
        "crisis observation—not a reliable sample.",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    args = parse_args(argv)
    prices = load_spy(args.start, args.end)
    rates = load_rates(prices.index)
    adjusted = prices["Adj Close"]
    asset_return = adjusted.pct_change().fillna(0.0)
    rungs, spy_drawdown, episode_objects = drawdown_rungs(adjusted)

    targets = {
        "hold_1x": pd.Series(1.0, index=prices.index),
        "ratchet_1_2_3": rungs.map({0: 1.0, 1: 2.0, 2: 3.0}),
        "reverse_3_2_1": rungs.map({0: 3.0, 1: 2.0, 2: 1.0}),
        "continuous_2x": pd.Series(2.0, index=prices.index),
        "inverse_minus_1x": pd.Series(-1.0, index=prices.index),
    }
    paths = {name: backtest(asset_return, target, rates, args.spread)
             for name, target in targets.items()}
    summary = {name: metrics(path, rates) for name, path in paths.items()}

    episodes = [asdict(item) for item in episode_objects]
    applied = paths["ratchet_1_2_3"]["applied_leverage"]
    exposure = {
        "1x_days": int((applied == 1.0).sum()),
        "2x_days": int((applied == 2.0).sum()),
        "3x_days": int((applied == 3.0).sum()),
        "1x_share": float((applied == 1.0).mean()),
        "2x_share": float((applied == 2.0).mean()),
        "3x_share": float((applied == 3.0).mean()),
    }
    regime_changes = int((targets["ratchet_1_2_3"] !=
                          targets["ratchet_1_2_3"].shift()).sum() - 1)

    # Estimate the rebalancing implied by maintaining exact target leverage.
    path = paths["ratchet_1_2_3"]
    previous_wealth = path["wealth"].shift().fillna(100.0)
    asset_before_trade = (path["applied_leverage"] * previous_wealth
                          * (1.0 + asset_return))
    desired_asset = path["target_close"] * path["wealth"]
    normalized_turnover = ((desired_asset - asset_before_trade).abs()
                           / path["wealth"]).iloc[1:]
    exposure["rebalance_days"] = int((normalized_turnover > 1e-9).sum())
    years = (prices.index[-1] - prices.index[0]).days / 365.2425
    annual_turnover = float(normalized_turnover.sum() / years)

    sensitivity = []
    cases = [("No financing cost", None),
             ("Treasury only", 0.00),
             ("Treasury + 1%", 0.01),
             ("Treasury + 2%", 0.02),
             ("Treasury + 3%", 0.03),
             ("Treasury + 5%", 0.05)]
    for label, spread in cases:
        use_rates = pd.Series(0.0, index=rates.index) if spread is None else rates
        use_spread = 0.0 if spread is None else spread
        got = metrics(backtest(asset_return, targets["ratchet_1_2_3"],
                               use_rates, use_spread), rates)
        sensitivity.append({"label": label, **got})

    # Solve the spread at which the ratchet's terminal wealth equals 1x SPY.
    hold_terminal = summary["hold_1x"]["terminal_100"]
    low, high = 0.0, 0.10
    for _ in range(60):
        middle = (low + high) / 2.0
        terminal = backtest(asset_return, targets["ratchet_1_2_3"], rates,
                            middle)["wealth"].iloc[-1]
        if terminal > hold_terminal:
            low = middle
        else:
            high = middle
    break_even_spread = (low + high) / 2.0
    calendar_days = prices.index.to_series().diff().dt.days.fillna(0.0)
    borrowed_units = (applied - 1.0).clip(lower=0.0)
    weighted_treasury = float(
        (borrowed_units * rates * calendar_days).sum()
        / (borrowed_units * calendar_days).sum()
    )

    daily = pd.DataFrame({
        "spy_close": prices["Close"], "spy_adjusted_close": adjusted,
        "spy_total_return": asset_return, "spy_drawdown": spy_drawdown,
        "treasury_rate": rates, "ratchet_target_close": targets["ratchet_1_2_3"],
        "ratchet_applied_leverage": applied,
        "ratchet_wealth": paths["ratchet_1_2_3"]["wealth"],
        "hold_wealth": paths["hold_1x"]["wealth"],
        "reverse_wealth": paths["reverse_3_2_1"]["wealth"],
        "inverse_minus_1x_wealth": paths["inverse_minus_1x"]["wealth"],
    })

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).T.to_csv(args.out / "summary.csv")
    pd.DataFrame(episodes).to_csv(args.out / "episodes.csv", index=False)
    daily.to_csv(args.out / "daily.csv", index_label="date")
    payload = {
        "assumptions": {"spread": args.spread, "shallow_drawdown": -0.10,
                        "deep_drawdown": -0.50, "signal": "adjusted close",
                        "execution": "next session", "reset": "daily"},
        "sample": {"start": prices.index[0].date().isoformat(),
                   "end": prices.index[-1].date().isoformat(),
                   "observations": len(prices)},
        "summary": summary, "episodes": episodes, "exposure": exposure,
        "regime_changes": regime_changes,
        "annual_normalized_turnover": annual_turnover,
        "break_even_spread_over_treasury": break_even_spread,
        "borrowed_exposure_weighted_treasury_rate": weighted_treasury,
        "financing_sensitivity": sensitivity,
    }
    (args.out / "results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    report = build_report(
        summary, episodes, sensitivity,
        prices.index[0].date().isoformat(), prices.index[-1].date().isoformat(),
        len(prices), args.spread, exposure, regime_changes, annual_turnover,
        break_even_spread, weighted_treasury,
    )
    (args.out / "report.md").write_text(report, encoding="utf-8")

    # Windows shells in this project may use cp1251; keep the Markdown Unicode
    # intact on disk while making console output best-effort.
    print(report.encode("ascii", errors="replace").decode("ascii"))
    print(f"Wrote {args.out / 'report.md'}")
    print(f"Wrote {args.out / 'results.json'}")
    print(f"Wrote {args.out / 'daily.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
