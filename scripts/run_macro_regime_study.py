"""Test observable macro regimes as long/cash overlays on US equity indices.

Every close-to-close exposure for trading day D is formed from information
dated no later than two trading days earlier, so it was knowable before the
previous close where the exposure changes. FactSet reports are dated publications;
undated analyst estimates are deliberately excluded because their historical
rows do not prove what estimate was known at the time.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import sqlite3
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Signal:
    name: str
    values: list[bool | None]
    family: str = "macro"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro-db", type=Path, default=Path("data/macro/macro.db"))
    parser.add_argument("--equity-db", type=Path, default=Path("data/equity/equity.db"))
    parser.add_argument("--out", type=Path, default=Path("out/strategy/macro_regime_study.md"))
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--switch-cost", type=float, default=0.0002)
    return parser.parse_args(argv)


def series(connection, series_id):
    rows = connection.execute(
        "SELECT obs_date,value FROM observations WHERE series_id=? AND value IS NOT NULL "
        "ORDER BY obs_date", (series_id,),
    ).fetchall()
    return [(day, float(value)) for day, value in rows]


def prices(connection, ticker, end):
    return [(day, float(close)) for day, close in connection.execute(
        "SELECT obs_date,close FROM index_prices WHERE ticker=? AND obs_date<=? "
        "AND close IS NOT NULL ORDER BY obs_date", (ticker, end),
    )]


def lagged_align(days, rows, max_age_days=None, trading_lag=2):
    """Last observation known before the close where exposure is changed."""
    source_days = [row[0] for row in rows]
    values = []
    positions = []
    for index, day in enumerate(days):
        cutoff = days[index - trading_lag] if index >= trading_lag else "0000-00-00"
        position = bisect.bisect_right(source_days, cutoff) - 1
        if position < 0:
            values.append(None)
            positions.append(None)
            continue
        if max_age_days is not None:
            from datetime import date
            age = (date.fromisoformat(day) - date.fromisoformat(source_days[position])).days
            if age > max_age_days:
                values.append(None)
                positions.append(position)
                continue
        values.append(rows[position][1])
        positions.append(position)
    return values, positions


def prior(values, index, lookback):
    target = index - lookback
    return values[target] if target >= 0 else None


def delta_flag(values, lookback, threshold=0.0):
    result = []
    for index, value in enumerate(values):
        old = prior(values, index, lookback)
        result.append(None if value is None or old is None else value - old > threshold)
    return result


def recent_increase(values, window, minimum_step=0.10):
    """True for `window` trading days after an observable positive step."""
    age = window + 1
    result = []
    previous = None
    for value in values:
        if value is None:
            result.append(None)
            continue
        if previous is not None and value - previous >= minimum_step:
            age = 0
        result.append(age < window)
        age += 1
        previous = value
    return result


def combine(*items):
    result = []
    for values in zip(*items):
        known = [value for value in values if value is not None]
        result.append(None if len(known) != len(values) else all(known))
    return result


def overlay_metrics(price_rows, risk_off, start, end, switch_cost, risk_exposure=0.0):
    selected = [i for i, (day, _) in enumerate(price_rows) if start <= day <= end]
    if len(selected) < 2:
        return None
    daily = []
    benchmark = []
    exposure = []
    switches = 0
    previous_exposure = None
    for index in selected:
        if index == 0 or risk_off[index] is None:
            continue
        raw = price_rows[index][1] / price_rows[index - 1][1] - 1.0
        active = risk_exposure if risk_off[index] else 1.0
        cost = 0.0
        if previous_exposure is not None and active != previous_exposure:
            switches += 1
            cost = switch_cost
        daily.append(active * raw - cost)
        benchmark.append(raw)
        exposure.append(active)
        previous_exposure = active
    if len(daily) < 50:
        return None
    return path_metrics(daily) | {
        "benchmark": path_metrics(benchmark),
        "exposure": statistics.mean(exposure), "switches": switches,
        "days": len(daily),
    }


def path_metrics(returns):
    equity = peak = 1.0
    max_drawdown = 0.0
    for value in returns:
        equity *= 1.0 + value
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity / peak - 1.0)
    years = len(returns) / 252.0
    cagr = equity ** (1.0 / years) - 1.0 if years > 0 and equity > 0 else -1.0
    volatility = statistics.stdev(returns) * math.sqrt(252) if len(returns) > 1 else math.nan
    annual_return = statistics.mean(returns) * 252
    return {
        "ending": equity, "cagr": cagr, "vol": volatility,
        "sharpe": annual_return / volatility if volatility else math.nan,
        "maxdd": max_drawdown,
    }


def trade_stats(values):
    if not values:
        return None
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    return {
        "n": len(values), "pf": gains / losses if losses else math.inf,
        "mean": statistics.mean(values),
    }


def association(price_rows, risk_off, start, end, horizon=63):
    buckets = {False: [], True: []}
    for index in range(0, len(price_rows) - horizon, 5):
        day, close = price_rows[index]
        state = risk_off[index]
        if not (start <= day <= end) or state is None:
            continue
        path = [price_rows[j][1] / close - 1.0 for j in range(index + 1, index + horizon + 1)]
        daily_abs = sum(abs(price_rows[j][1] / price_rows[j - 1][1] - 1.0)
                        for j in range(index + 1, index + horizon + 1))
        forward = path[-1]
        efficiency = abs(forward) / daily_abs if daily_abs else 0.0
        buckets[state].append((forward, min(path), efficiency))
    result = {}
    for state, rows in buckets.items():
        if not rows:
            continue
        result[state] = {
            "n": len(rows),
            "return": statistics.mean(row[0] for row in rows),
            "negative": sum(row[0] < 0 for row in rows) / len(rows),
            "drawdown": statistics.median(row[1] for row in rows),
            "efficiency": statistics.median(row[2] for row in rows),
        }
    return result


def factset_rows(connection):
    rows = connection.execute(
        "SELECT report_date,forward_12m_pe,pe_10y_average,estimated_earnings_growth,"
        "blended_earnings_growth,estimated_growth_at_quarter_start,"
        "negative_guidance_count,positive_guidance_count FROM factset_reports "
        "ORDER BY report_date"
    ).fetchall()
    result = []
    for row in rows:
        current_growth = row[4] if row[4] is not None else row[3]
        result.append((row[0], {
            "pe_premium": (row[1] - row[2]) if row[1] is not None and row[2] is not None else None,
            "growth": current_growth,
            "revision": (current_growth - row[5]) if current_growth is not None and row[5] is not None else None,
            "guidance_ratio": (row[6] / max(row[7], 1))
            if row[6] is not None and row[7] is not None else None,
        }))
    return result


def policy_target_rows(connection):
    """Single target-rate history across the pre/post-2008 convention change."""
    old = series(connection, "DFEDTAR")
    upper = series(connection, "DFEDTARU")
    return sorted({day: value for day, value in (*old, *upper)}.items())


def main(argv=None):
    args = parse_args(argv)
    macro = sqlite3.connect(f"file:{args.macro_db}?mode=ro", uri=True)
    equity = sqlite3.connect(f"file:{args.equity_db}?mode=ro", uri=True)
    assets = {ticker: prices(equity, ticker, args.end)
              for ticker in ("^SP500TR", "^NDX", "^RUT")}
    base = assets["^SP500TR"]
    days = [row[0] for row in base]

    target, _ = lagged_align(days, policy_target_rows(macro))
    vix, _ = lagged_align(days, series(macro, "VIXCLS"), max_age_days=10)
    vxv, _ = lagged_align(days, series(macro, "VXVCLS"), max_age_days=10)
    curve, _ = lagged_align(days, series(macro, "T10Y2Y"), max_age_days=10)
    credit, _ = lagged_align(days, series(macro, "BAA10Y"), max_age_days=10)
    forward12_rows = [(day, value) for day, months, value in macro.execute(
        "SELECT curve_date,horizon_months,forward_rate FROM fed_path "
        "WHERE horizon_months=12 ORDER BY curve_date"
    )]
    forward12, _ = lagged_align(days, forward12_rows, max_age_days=10)

    hiking6m = delta_flag(target, 126, 0.01)
    recent21 = recent_increase(target, 21)
    recent63 = recent_increase(target, 63)
    recent126 = recent_increase(target, 126)
    vix20 = [None if value is None else value >= 20 for value in vix]
    vix25 = [None if value is None else value >= 25 for value in vix]
    vol_inverted = [None if a is None or b is None else a > b for a, b in zip(vix, vxv)]
    curve_inverted = [None if value is None else value < 0 for value in curve]
    policy_above_forward = [
        None if a is None or b is None else a - b >= 0.25
        for a, b in zip(target, forward12)
    ]
    credit_widening = []
    for index, value in enumerate(credit):
        old = prior(credit, index, 63)
        credit_widening.append(None if value is None or old is None else value - old >= 0.25)

    signals = [
        Signal("Hike in prior 21 trading days", recent21),
        Signal("Hike in prior 63 trading days", recent63),
        Signal("Hike in prior 126 trading days", recent126),
        Signal("Policy rate higher than 6m ago", hiking6m),
        Signal("2y/10y curve inverted", curve_inverted),
        Signal("Policy >= 12m Treasury forward +25bp", policy_above_forward),
        Signal("VIX >= 20", vix20),
        Signal("VIX >= 25", vix25),
        Signal("VIX term structure inverted", vol_inverted),
        Signal("Baa spread widened >=25bp in 3m", credit_widening),
        Signal("Recent hike AND VIX >=20", combine(recent63, vix20)),
        Signal("Higher policy rate AND inverted curve", combine(hiking6m, curve_inverted)),
    ]

    facts = factset_rows(equity)
    fact_values, _ = lagged_align(days, facts, max_age_days=21)
    def fact_flag(predicate):
        return [None if value is None else predicate(value) for value in fact_values]
    signals += [
        Signal("Forward P/E premium >=2 vs published 10y avg",
               fact_flag(lambda x: x["pe_premium"] is not None and x["pe_premium"] >= 2), "earnings"),
        Signal("Current-quarter earnings growth <0",
               fact_flag(lambda x: x["growth"] is not None and x["growth"] < 0), "earnings"),
        Signal("Earnings growth deteriorated >=2pp from quarter start",
               fact_flag(lambda x: x["revision"] is not None and x["revision"] <= -2), "earnings"),
        Signal("Negative guidance at least 2x positive",
               fact_flag(lambda x: x["guidance_ratio"] is not None and x["guidance_ratio"] >= 2), "earnings"),
    ]

    periods = {"macro": (("train", "2000-01-03", "2016-12-30"),
                           ("test", "2017-01-03", args.end)),
               "earnings": (("train", "2017-02-06", "2022-12-30"),
                              ("test", "2023-01-03", args.end))}
    lines = [
        "# Macro regime and equity-risk overlay study", "",
        "All signals are lagged: trading-day D's close-to-close return uses observations "
        "dated no later than D-2, making them available before the D-1 close where exposure "
        "changes. Cash earns 0%; each exposure change costs 2 bp. Macro/VIX "
        "signals use 2000-2016 as discovery and 2017+ as validation. Dated FactSet reports "
        "use 2017-2022 and 2023+ respectively.", "",
        "## S&P 500 total-return long/cash overlays", "",
        "| Signal (neutral while true) | Period | Exposure | Switches | CAGR | Buy/hold CAGR | "
        "Max DD | Buy/hold DD | Sharpe | Buy/hold Sharpe |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    all_metrics = {}
    for signal in signals:
        for period, start, end in periods[signal.family]:
            item = overlay_metrics(base, signal.values, start, end, args.switch_cost)
            if item is None:
                continue
            all_metrics[(signal.name, period)] = item
            bench = item["benchmark"]
            lines.append(
                f"| {signal.name} | {period} | {item['exposure']:.1%} | {item['switches']} | "
                f"{item['cagr']:+.2%} | {bench['cagr']:+.2%} | {item['maxdd']:.1%} | "
                f"{bench['maxdd']:.1%} | {item['sharpe']:.2f} | {bench['sharpe']:.2f} |"
            )

    lines += ["", "## Forward 3-month market behavior", "",
              "Weekly-sampled observations reduce overlap. Efficiency is absolute net movement "
              "divided by the sum of daily absolute movements; lower values mean a choppier path.", "",
              "| Signal | Period | State | Samples | Mean 3m return | Negative | Median path DD | "
              "Median efficiency |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for signal in signals:
        for period, start, end in periods[signal.family]:
            result = association(base, signal.values, start, end)
            for state in (False, True):
                if state not in result:
                    continue
                item = result[state]
                lines.append(
                    f"| {signal.name} | {period} | {'risk-off' if state else 'normal'} | "
                    f"{item['n']} | {item['return']:+.2%} | {item['negative']:.1%} | "
                    f"{item['drawdown']:.1%} | {item['efficiency']:.3f} |"
                )

    lines += ["", "## Cross-index validation-period overlays", "",
              "| Signal | Asset | Exposure | CAGR | Buy/hold CAGR | Max DD | Buy/hold DD | Sharpe |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for signal in signals:
        period, start, end = periods[signal.family][1]
        # Signals share the S&P trading calendar. These index series use the same dates.
        for ticker, rows in assets.items():
            item = overlay_metrics(rows, signal.values, start, end, args.switch_cost)
            if item is None:
                continue
            bench = item["benchmark"]
            lines.append(
                f"| {signal.name} | {ticker} | {item['exposure']:.1%} | {item['cagr']:+.2%} | "
                f"{bench['cagr']:+.2%} | {item['maxdd']:.1%} | {bench['maxdd']:.1%} | "
                f"{item['sharpe']:.2f} |"
            )

    half_risk_names = {
        "Hike in prior 63 trading days", "VIX >= 20",
        "VIX term structure inverted", "Recent hike AND VIX >=20",
        "Forward P/E premium >=2 vs published 10y avg",
        "Earnings growth deteriorated >=2pp from quarter start",
    }
    lines += ["", "## Half-risk overlays", "",
              "Instead of going to cash, exposure falls from 100% to 50% while the signal is true.", "",
              "| Signal | Period | Exposure | CAGR | Buy/hold CAGR | Max DD | Buy/hold DD | Sharpe |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for signal in signals:
        if signal.name not in half_risk_names:
            continue
        for period, start, end in periods[signal.family]:
            item = overlay_metrics(base, signal.values, start, end, args.switch_cost,
                                   risk_exposure=0.5)
            if item is None:
                continue
            bench = item["benchmark"]
            lines.append(
                f"| {signal.name} | {period} | {item['exposure']:.1%} | "
                f"{item['cagr']:+.2%} | {bench['cagr']:+.2%} | {item['maxdd']:.1%} | "
                f"{bench['maxdd']:.1%} | {item['sharpe']:.2f} |"
            )

    portable_half = {"VIX term structure inverted",
                     "Earnings growth deteriorated >=2pp from quarter start"}
    lines += ["", "## Half-risk cross-index validation", "",
              "| Signal | Asset | CAGR | Buy/hold CAGR | Max DD | Buy/hold DD | Sharpe |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for signal in signals:
        if signal.name not in portable_half:
            continue
        _period, start, end = periods[signal.family][1]
        for ticker, rows in assets.items():
            item = overlay_metrics(rows, signal.values, start, end, args.switch_cost,
                                   risk_exposure=0.5)
            if item is None:
                continue
            bench = item["benchmark"]
            lines.append(
                f"| {signal.name} | {ticker} | {item['cagr']:+.2%} | "
                f"{bench['cagr']:+.2%} | {item['maxdd']:.1%} | {bench['maxdd']:.1%} | "
                f"{item['sharpe']:.2f} |"
            )

    trade_path = Path("out/strategy/turtle_exit_trades.csv")
    if trade_path.exists():
        books = {
            ("daily", "Channel 50"): "Daily Channel-50 long",
            ("30-minute", "Chandelier 5N"): "30m Chandelier-5N long",
            ("30-minute", "Channel 20 + break-even at 1N"): "30m BE-at-1N long",
        }
        trades = {label: [] for label in books.values()}
        with trade_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                label = books.get((row["interval"], row["variant"]))
                if label is None or int(row["direction"]) <= 0:
                    continue
                trades[label].append((row["entry_timestamp"][:10], float(row["net_r"])))
        lines += ["", "## Regimes at actual Turtle long entries", "",
                  "This is a conditional diagnostic on trades the unfiltered engine took, not a "
                  "filtered replay. Skipping a trade changes when that market next becomes "
                  "available, so promising rows must be replayed before deployment.", "",
                  "| Book | Signal | Period | State | Trades | PF | Mean R |",
                  "|---|---|---|---|---:|---:|---:|"]
        trade_periods = (("2017-2022", "2017-02-06", "2022-12-30"),
                         ("2023+", "2023-01-03", args.end))
        for signal in signals:
            state_by_day = dict(zip(days, signal.values))
            for period, start, end in trade_periods:
                for label, rows in trades.items():
                    for state in (False, True):
                        values = [value for day, value in rows
                                  if start <= day <= end and state_by_day.get(day) is state]
                        item = trade_stats(values)
                        if item is None:
                            continue
                        lines.append(
                            f"| {label} | {signal.name} | {period} | "
                            f"{'risk-off' if state else 'normal'} | {item['n']:,} | "
                            f"{item['pf']:.2f} | {item['mean']:+.3f} |"
                        )

    lines += ["", "## Leakage and interpretation limits", "",
              "- FRED market series, policy targets and FactSet reports are shifted two trading "
              "days for close-to-close execution. They are never used on their publication date "
              "or for the first overnight gap after publication.",
              "- FactSet metrics become eligible only after their dated report and expire after "
              "21 calendar days if no newer report is available.",
              "- The `fed_path` field is a fitted Treasury instantaneous forward rate, not a "
              "Fed-funds-futures probability. The report names it accordingly.",
              "- Current revised macro observations such as GDP and payrolls are excluded. A "
              "separate vintage-aware study is required for them.",
              "- These thresholds are now inspected. The validation rows are evidence, but no "
              "longer pristine holdouts for further threshold tuning."]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    macro.close()
    equity.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
