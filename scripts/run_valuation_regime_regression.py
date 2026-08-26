"""Replicate and extend the CAPE / Excess-CAPE-Yield / profitability regression
on subsequent 10-year real S&P 500 returns, using only data already in
data/equity/equity.db (Shiller's own dataset, source_url column).

The reference result being tested (a user-supplied first pass, not ours):
  CAPE only:            R^2 = 39.0%   (b = -0.471)
  Excess CAPE Yield:     R^2 = 52.3%   (b = +1.180)
  ECY + aggregate ROE:  R^2 = 55.1%   (b_ecy=+1.096, b_roe=-0.259)

We do not have the long-run FRED aggregate-ROE series locally, so the third
test is not replicated as-is. Instead we test the two adaptations the
original analysis proposed testing next: ECY plus the dividend retention
rate, and ECY plus a trailing profitability (real-earnings) cycle z-score.
Both use only Shiller's own dividend/earnings columns, so they carry no new
data-source risk.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equity-db", type=Path, default=Path("data/equity/equity.db"))
    parser.add_argument("--out", type=Path, default=Path("out/strategy/valuation_regression"))
    parser.add_argument("--window-start", default="1946-01-01")
    parser.add_argument("--window-end", default="2013-01-01")
    return parser.parse_args(argv)


def ols(y: np.ndarray, X: np.ndarray) -> dict:
    """Plain least squares with an intercept column prepended to X."""
    design = np.column_stack([np.ones(len(y)), X])
    coefs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefs
    resid = y - fitted
    sse = float((resid ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - sse / sst
    n, k = design.shape
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k)
    rmse = float(np.sqrt(sse / n))
    return {
        "n": int(n), "intercept": float(coefs[0]), "coefficients": [float(c) for c in coefs[1:]],
        "r2": float(r2), "adjusted_r2": float(adj_r2), "rmse": rmse, "fitted": fitted, "resid": resid,
    }


def expanding_oos(y: np.ndarray, X: np.ndarray, min_train: int = 20) -> dict:
    """Refit on data up to t-1 (non-overlapping in fit target), predict t."""
    preds, actuals = [], []
    for t in range(min_train, len(y)):
        train_y, train_X = y[:t], X[:t]
        design = np.column_stack([np.ones(len(train_y)), train_X])
        coefs, _, _, _ = np.linalg.lstsq(design, train_y, rcond=None)
        pred = np.concatenate([[1.0], X[t]]) @ coefs
        preds.append(float(pred))
        actuals.append(float(y[t]))
    preds, actuals = np.array(preds), np.array(actuals)
    sse = float(((actuals - preds) ** 2).sum())
    sst = float(((actuals - actuals.mean()) ** 2).sum())
    oos_r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    rank_corr = float(pd.Series(preds).corr(pd.Series(actuals), method="spearman"))
    return {"oos_r2": oos_r2, "rmse": float(np.sqrt(sse / len(actuals))), "rank_correlation": rank_corr, "n": int(len(actuals))}


def quintile_table(frame: pd.DataFrame, column: str, target: str) -> list[dict]:
    ranked = frame[[column, target]].dropna().copy()
    ranked["quintile"] = pd.qcut(ranked[column].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    out = ranked.groupby("quintile", observed=True)[target].mean()
    return [{"quintile": int(q), "mean_forward_10y_real_return": float(v)} for q, v in out.items()]


def main(argv=None):
    args = parse_args(argv)
    con = sqlite3.connect(args.equity_db)
    df = pd.read_sql(
        "SELECT obs_date, cape, excess_cape_yield, dividend, earnings, real_earnings, "
        "ten_year_annualized_stock_real_return FROM shiller_monthly ORDER BY obs_date",
        con, parse_dates=["obs_date"],
    )
    df["retention_rate"] = 1.0 - (df["dividend"] / df["earnings"])
    trailing_mean = df["real_earnings"].rolling(120, min_periods=60).mean()
    trailing_std = df["real_earnings"].rolling(120, min_periods=60).std()
    df["earnings_cycle_z"] = (df["real_earnings"] - trailing_mean) / trailing_std

    jan = df[df["obs_date"].dt.month == 1].reset_index(drop=True)
    full_sample = jan.dropna(subset=["cape", "excess_cape_yield", "ten_year_annualized_stock_real_return"])
    window = full_sample[
        (full_sample["obs_date"] >= args.window_start) & (full_sample["obs_date"] <= args.window_end)
    ]

    def run_all(sample: pd.DataFrame, label: str) -> dict:
        y = sample["ten_year_annualized_stock_real_return"].values * 100.0
        cape = sample["cape"].values
        ecy = sample["excess_cape_yield"].values * 100.0
        retention = sample["retention_rate"].values
        cycle = sample["earnings_cycle_z"].values

        cape_fit = ols(y, cape.reshape(-1, 1))
        ecy_fit = ols(y, ecy.reshape(-1, 1))

        ext = sample.dropna(subset=["retention_rate", "earnings_cycle_z"])
        y_ext = ext["ten_year_annualized_stock_real_return"].values * 100.0
        ecy_ext = ext["excess_cape_yield"].values * 100.0
        retention_ext = ext["retention_rate"].values
        cycle_ext = ext["earnings_cycle_z"].values
        ecy_retention_fit = ols(y_ext, np.column_stack([ecy_ext, retention_ext]))
        ecy_cycle_fit = ols(y_ext, np.column_stack([ecy_ext, cycle_ext]))

        oos = expanding_oos(y, ecy.reshape(-1, 1))

        for fit in (cape_fit, ecy_fit, ecy_retention_fit, ecy_cycle_fit):
            fit.pop("fitted")
            fit.pop("resid")

        return {
            "label": label,
            "n_observations": int(len(sample)),
            "date_range": [sample["obs_date"].min().strftime("%Y-%m-%d"), sample["obs_date"].max().strftime("%Y-%m-%d")],
            "cape_only": cape_fit,
            "excess_cape_yield": ecy_fit,
            "ecy_plus_retention_rate": ecy_retention_fit,
            "ecy_plus_earnings_cycle": ecy_cycle_fit,
            "ecy_expanding_window_oos": oos,
            "cape_quintiles": quintile_table(sample, "cape", "ten_year_annualized_stock_real_return"),
            "ecy_quintiles": quintile_table(sample, "excess_cape_yield", "ten_year_annualized_stock_real_return"),
        }

    results = {
        "reference_result_pasted_by_user": {
            "note": "First-pass result supplied by the user; not computed by this script.",
            "cape_only_r2": 0.390,
            "excess_cape_yield_r2": 0.523,
            "ecy_plus_aggregate_roe_r2": 0.551,
            "roe_coefficient_sign": "negative",
        },
        "user_window": run_all(window, f"January observations, {args.window_start[:4]}-{args.window_end[:4]} (matches user's FRED-ROE-availability window)"),
        "full_shiller_history": run_all(full_sample, "January observations, full Shiller history"),
        "data_limitation": (
            "No long-run (1946+) FRED aggregate nonfinancial-corporate ROE series is present in "
            "data/macro/macro.db. 'ecy_plus_retention_rate' and 'ecy_plus_earnings_cycle' are adaptations "
            "of the profitability idea using only Shiller's own dividend/earnings columns, not a "
            "replication of the user's third regression."
        ),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in results["user_window"].items() if k not in ("cape_quintiles", "ecy_quintiles")}, indent=2))


if __name__ == "__main__":
    main()
