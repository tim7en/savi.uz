"""How much does dealer gamma really forecast, and why does the strategy not benefit?

Rank correlations establish that a relationship exists.  They do not establish
that it is *useful*, because a forecaster is judged against the best available
alternative, out of sample, on a loss function suited to the target.  This runs
the analysis the way the volatility-forecasting literature does:

* **HAR-RV is the benchmark** (Corsi, 2009).  Realised volatility is strongly
  autoregressive across daily, weekly and monthly horizons, so any new variable
  must beat a model that already knows that.  Implied volatility is added as a
  second benchmark, since it is a market forecast of the same quantity.
* **Out-of-sample, expanding window.**  Coefficients are fitted only on data
  before the day being forecast; nothing is estimated on the test point.
* **QLIKE alongside RMSE** (Patton, 2011).  Realised volatility is a noisy proxy
  for the true quantity, and QLIKE is robust to that noise in a way squared
  error is not.  It also penalises under-forecasting asymmetrically, which is
  what a risk manager cares about.
* **Diebold-Mariano** with a Newey-West correction, so "model A beats model B"
  is a test rather than a comparison of two point estimates.

Then the part that matters for trading: gamma forecasts *market* volatility, but
a sizing overlay needs it to forecast the *strategy's* dispersion.  Those are
different questions and the second is tested directly on trade outcomes.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import TurtleConfig, run_turtle  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

ETFS = {"SPY", "QQQ", "IWM", "GLD", "EWJ", "EWT", "EWY", "KWEB",
        "SLV", "TBT", "TMF", "UVXY", "XLE"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--options", type=Path,
                        default=Path("data/options/alphavantage.db"))
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--symbols", nargs="+", default=["SPY", "QQQ"])
    parser.add_argument("--burn", type=int, default=500,
                        help="sessions used to fit before forecasting begins")
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/gex_deep_analysis.json"))
    return parser.parse_args(argv)


def realised(bars_path: Path, symbol: str):
    """Annualised realised volatility per session from five-minute returns."""
    splits = load_splits(bars_path)
    connection = sqlite3.connect(f"file:{bars_path}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
        "frequency='5min' ORDER BY ts", (symbol,)).fetchall()
    connection.close()
    bars = adjust_bars([Bar(*r) for r in rows], splits.get(symbol, []))
    by_day = defaultdict(list)
    for bar in bars:
        by_day[bar.timestamp[:10]].append(bar)
    out, signed = {}, {}
    for day, rows_ in by_day.items():
        rets = [rows_[i].close / rows_[i - 1].close - 1.0
                for i in range(1, len(rows_)) if rows_[i - 1].close > 0]
        if len(rets) < 10:
            continue
        out[day] = math.sqrt(sum(r * r for r in rets) * 252) * 100
        signed[day] = (rows_[-1].close / rows_[0].open - 1.0) * 100
    return out, signed


def load_features(options: Path, symbol: str):
    store = sqlite3.connect(f"file:{options}?mode=ro", uri=True)
    rows = store.execute(
        "SELECT observation_date, atm_iv, net_gex, gamma_balance, absolute_gex "
        "FROM av_daily WHERE symbol=? ORDER BY observation_date", (symbol,)).fetchall()
    store.close()
    return {r[0]: {"atm_iv": r[1], "net_gex": r[2], "gamma_balance": r[3],
                   "absolute_gex": r[4]} for r in rows}


def ols_forecast(design, target, point):
    """Fit by least squares on the history and predict one step ahead."""
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return float(point @ coefficients)


def qlike(actual, forecast):
    """Patton's QLIKE on variances; robust to noise in the volatility proxy."""
    a, f = actual ** 2, max(forecast, 1e-6) ** 2
    return a / f - math.log(a / f) - 1.0


def diebold_mariano(loss_a, loss_b, lag=5):
    """Newey-West corrected test that two forecast losses differ."""
    d = np.array(loss_a) - np.array(loss_b)
    n = len(d)
    mean = d.mean()
    gamma0 = ((d - mean) ** 2).mean()
    variance = gamma0
    for k in range(1, lag + 1):
        cov = ((d[k:] - mean) * (d[:-k] - mean)).mean()
        variance += 2.0 * (1.0 - k / (lag + 1.0)) * cov
    if variance <= 0:
        return None, None
    stat = mean / math.sqrt(variance / n)
    p = math.erfc(abs(stat) / math.sqrt(2.0))
    return stat, p


def run_symbol(args, symbol):
    rv, signed = realised(args.bars, symbol)
    feats = load_features(args.options, symbol)
    days = sorted(set(rv) & set(feats))
    days = [d for d in days if feats[d]["atm_iv"] and feats[d]["gamma_balance"] is not None]
    print(f"\n{'=' * 78}\n{symbol}: {len(days):,} sessions "
          f"{days[0]} -> {days[-1]}")

    log_rv = {d: math.log(max(rv[d], 1e-6)) for d in days}
    index = {d: i for i, d in enumerate(days)}

    # Build every regressor once. Rebuilding them inside the expanding-window
    # loop costs O(n^2) row constructions for no benefit; the rows themselves
    # never change, only how many of them are in the fit.
    series = np.array([log_rv[d] for d in days])
    n = len(days)
    d1 = series
    d5 = np.array([series[max(i - 4, 0):i + 1].mean() for i in range(n)])
    d22 = np.array([series[max(i - 21, 0):i + 1].mean() for i in range(n)])
    iv = np.array([math.log(max(feats[d]["atm_iv"], 1e-6)) for d in days])
    gb = np.array([feats[d]["gamma_balance"] for d in days], dtype=float)
    ones = np.ones(n)

    columns = {
        "HAR-RV": np.column_stack([ones, d1, d5, d22]),
        "HAR-RV + IV": np.column_stack([ones, d1, d5, d22, iv]),
        "HAR-RV + IV + gamma": np.column_stack([ones, d1, d5, d22, iv, gb]),
        "HAR-RV + gamma": np.column_stack([ones, d1, d5, d22, gb]),
    }
    losses = {name: {"rmse": [], "qlike": []} for name in columns}
    actuals = []
    start = max(args.burn, 30)
    for i in range(start, len(days) - 1):
        actual = rv[days[i + 1]]
        actuals.append(actual)
        target = series[23:i + 1]
        for name, matrix in columns.items():
            design = matrix[22:i]
            prediction = math.exp(ols_forecast(design, target, matrix[i]))
            losses[name]["rmse"].append((math.log(max(actual, 1e-6))
                                         - math.log(max(prediction, 1e-6))) ** 2)
            losses[name]["qlike"].append(qlike(actual, prediction))
        if (i - start) % 400 == 0:
            print(f"    forecasting… {i - start:,}/{len(days) - 1 - start:,}",
                  flush=True)

    print(f"  out-of-sample forecasts: {len(actuals):,} "
          f"(fitted on an expanding window, first {start} sessions withheld)")
    print(f"\n  {'model':22s} {'RMSE(log)':>10s} {'QLIKE':>9s} "
          f"{'vs HAR':>9s} {'vs HAR+IV':>11s}")
    summary = {}
    for name in columns:
        rmse = math.sqrt(statistics.mean(losses[name]["rmse"]))
        ql = statistics.mean(losses[name]["qlike"])
        cells = []
        for benchmark in ("HAR-RV", "HAR-RV + IV"):
            if name == benchmark:
                cells.append("     —")
                continue
            stat, p = diebold_mariano(losses[name]["qlike"],
                                      losses[benchmark]["qlike"])
            if stat is None:
                cells.append("     —")
            else:
                mark = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
                cells.append(f"{stat:+.2f}{mark}")
        print(f"  {name:22s} {rmse:>10.4f} {ql:>9.4f} {cells[0]:>9s} {cells[1]:>11s}")
        summary[name] = {"rmse": rmse, "qlike": ql,
                         "dm_vs_har": cells[0], "dm_vs_har_iv": cells[1]}
    print("  (Diebold-Mariano t; negative favours the row. * 10% ** 5% *** 1%)")

    # --- does gamma predict the DOWNSIDE specifically? ---
    print(f"\n  asymmetry: next-session outcome by gamma quintile")
    ordered = sorted(days[:-1], key=lambda d: feats[d]["gamma_balance"])
    size = len(ordered) // 5
    print(f"    {'quintile':>9s} {'gamma bal':>11s} {'next RV':>9s} "
          f"{'mean ret':>9s} {'down days':>10s} {'worst':>8s}")
    asym = []
    for q in range(5):
        chunk = ordered[q * size:(q + 1) * size] if q < 4 else ordered[4 * size:]
        nxt = [days[index[d] + 1] for d in chunk if index[d] + 1 < len(days)]
        vols = [rv[d] for d in nxt]
        rets = [signed[d] for d in nxt]
        down = sum(1 for r in rets if r < 0) / len(rets)
        print(f"    {q + 1:>9d} {statistics.mean(feats[d]['gamma_balance'] for d in chunk):>+11.3f} "
              f"{statistics.mean(vols):>9.2f} {statistics.mean(rets):>+9.3f} "
              f"{down:>10.1%} {min(rets):>+8.2f}")
        asym.append({"quintile": q + 1, "rv": statistics.mean(vols),
                     "ret": statistics.mean(rets), "down_share": down,
                     "worst": min(rets)})
    return {"sessions": len(days), "oos": len(actuals),
            "models": summary, "asymmetry": asym}


def strategy_link(args, feats_by_symbol):
    """Does gamma forecast the STRATEGY's dispersion, or only the market's?"""
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    names = [r[0] for r in connection.execute(
        "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker")]
    config = TurtleConfig(entry_window=55, exit_window=20, atr_window=20,
                          skip_after_winner=False, directions=(1,))
    trades = []
    for ticker in names:
        if ticker in ETFS:
            continue
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='5min' ORDER BY ts", (ticker,)).fetchall()
        if not rows:
            continue
        series = resample_regular_session(
            adjust_bars([Bar(*r) for r in rows], splits.get(ticker, [])), minutes=30)
        for trade in run_turtle(series, config=config)[0]:
            trades.append({"day": trade.entry_timestamp[:10], "r": trade.net_r})
    connection.close()

    feats = feats_by_symbol["SPY"]
    sessions = sorted(feats)
    prior = {sessions[i]: sessions[i - 1] for i in range(1, len(sessions))}
    tagged = []
    for trade in trades:
        earlier = prior.get(trade["day"])
        if earlier and feats[earlier]["gamma_balance"] is not None:
            tagged.append((feats[earlier]["gamma_balance"], trade["r"]))
    tagged.sort()
    size = len(tagged) // 5
    print(f"\n{'=' * 78}\nSTRATEGY LINK: {len(tagged):,} trades tagged with the "
          f"PRIOR session's SPY gamma")
    print(f"  {'quintile':>9s} {'gamma bal':>11s} {'n':>6s} {'mean R':>9s} "
          f"{'sd of R':>9s} {'win':>7s} {'|R| mean':>9s}")
    rows_out = []
    for q in range(5):
        chunk = tagged[q * size:(q + 1) * size] if q < 4 else tagged[4 * size:]
        rs = [r for _, r in chunk]
        print(f"  {q + 1:>9d} {statistics.mean(g for g, _ in chunk):>+11.3f} "
              f"{len(rs):>6d} {statistics.mean(rs):>+9.3f} "
              f"{statistics.stdev(rs):>9.3f} "
              f"{sum(1 for r in rs if r > 0) / len(rs):>7.1%} "
              f"{statistics.mean(abs(r) for r in rs):>9.3f}")
        rows_out.append({"quintile": q + 1, "n": len(rs),
                         "mean_r": statistics.mean(rs), "sd_r": statistics.stdev(rs),
                         "win": sum(1 for r in rs if r > 0) / len(rs)})
    lo, hi = rows_out[0], rows_out[-1]
    print(f"\n  bottom-minus-top gamma quintile: mean R {lo['mean_r'] - hi['mean_r']:+.3f}, "
          f"dispersion ratio {lo['sd_r'] / hi['sd_r']:.2f}x")
    return rows_out


def main(argv=None):
    args = parse_args(argv)
    report, feats_by_symbol = {}, {}
    for symbol in args.symbols:
        feats_by_symbol[symbol] = load_features(args.options, symbol)
        report[symbol] = run_symbol(args, symbol)
    report["strategy"] = strategy_link(args, feats_by_symbol)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
