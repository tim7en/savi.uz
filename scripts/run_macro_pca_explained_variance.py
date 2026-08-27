"""How much of SPY/QQQ return and drawdown variance is explained by a small
set of macro principal components?

This is a descriptive, in-sample variance-decomposition exercise (unlike the
point-in-time regime overlay), so full-sample standardization is used
throughout and is explicitly not a trading signal.
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
    parser.add_argument("--macro-db", type=Path, default=Path("data/macro/macro.db"))
    parser.add_argument("--equity-db", type=Path, default=Path("data/equity/equity.db"))
    parser.add_argument("--out", type=Path, default=Path("out/strategy/macro_pca"))
    parser.add_argument("--n-components", type=int, default=5)
    return parser.parse_args(argv)


MACRO_SERIES = {
    "fed_funds": "DFF",
    "curve_2s10s": "T10Y2Y",
    "curve_3m10y": "T10Y3M",
    "credit_spread_baa10y": "BAA10Y",
    "financial_conditions": "NFCI",
    "vix": "VIXCLS",
    "unemployment": "UNRATE",
    "cpi_level": "CPIAUCSL",
    "initial_claims": "ICSA",
    "ten_year_real_yield": "DFII10",
    "fed_balance_sheet": "WALCL",
}


def month_end_series(connection, series_id: str) -> pd.Series:
    rows = connection.execute(
        "SELECT obs_date, value FROM observations WHERE series_id=? AND value IS NOT NULL ORDER BY obs_date",
        (series_id,),
    ).fetchall()
    dates = pd.to_datetime([row[0] for row in rows])
    values = np.array([row[1] for row in rows], dtype=float)
    series = pd.Series(values, index=dates).sort_index()
    return series.resample("ME").last().ffill()


def load_price_monthly(connection, ticker: str) -> pd.Series:
    rows = connection.execute(
        "SELECT obs_date, close FROM index_prices WHERE ticker=? AND close IS NOT NULL ORDER BY obs_date",
        (ticker,),
    ).fetchall()
    dates = pd.to_datetime([row[0] for row in rows])
    values = np.array([row[1] for row in rows], dtype=float)
    daily = pd.Series(values, index=dates).sort_index()
    return daily.resample("ME").last()


def pca(standardized: np.ndarray, n_components: int) -> dict:
    u, s, vt = np.linalg.svd(standardized, full_matrices=False)
    variance = s ** 2
    explained_ratio = variance / variance.sum()
    scores = u * s
    return {
        "scores": scores[:, :n_components],
        "loadings": vt[:n_components, :],
        "explained_variance_ratio": explained_ratio[:n_components],
        "cumulative_variance_ratio": np.cumsum(explained_ratio)[:n_components],
        "all_explained_variance_ratio": explained_ratio,
    }


def ols_r2(y: np.ndarray, X: np.ndarray) -> dict:
    design = np.column_stack([np.ones(len(y)), X])
    coefs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefs
    resid = y - fitted
    sse = float((resid ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    n, k = design.shape
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k)
    return {"r2": float(r2), "adjusted_r2": float(adj_r2), "n": int(n)}


def drawdown_series(price: pd.Series) -> pd.Series:
    running_max = price.cummax()
    return price / running_max - 1.0


def main(argv=None):
    args = parse_args(argv)
    macro_con = sqlite3.connect(args.macro_db)
    equity_con = sqlite3.connect(args.equity_db)

    macro_frame = pd.DataFrame({name: month_end_series(macro_con, sid) for name, sid in MACRO_SERIES.items()})
    spy = load_price_monthly(equity_con, "^SP500TR")
    ndx = load_price_monthly(equity_con, "^NDX")

    frame = macro_frame.copy()
    frame["spy_price"] = spy
    frame["ndx_price"] = ndx
    frame = frame.dropna()

    frame["spy_return_fwd_1m"] = frame["spy_price"].shift(-1) / frame["spy_price"] - 1.0
    frame["ndx_return_fwd_1m"] = frame["ndx_price"].shift(-1) / frame["ndx_price"] - 1.0
    frame["spy_return_fwd_12m"] = frame["spy_price"].shift(-12) / frame["spy_price"] - 1.0
    frame["ndx_return_fwd_12m"] = frame["ndx_price"].shift(-12) / frame["ndx_price"] - 1.0
    frame["spy_drawdown"] = drawdown_series(frame["spy_price"])
    frame["ndx_drawdown"] = drawdown_series(frame["ndx_price"])

    macro_cols = list(MACRO_SERIES.keys())
    macro_values = frame[macro_cols].values
    standardized = (macro_values - macro_values.mean(axis=0)) / macro_values.std(axis=0)

    components = pca(standardized, args.n_components)
    scores = components["scores"]

    targets = {
        "spy_drawdown_contemporaneous": frame["spy_drawdown"].values,
        "ndx_drawdown_contemporaneous": frame["ndx_drawdown"].values,
        "spy_return_fwd_1m": frame["spy_return_fwd_1m"].values,
        "ndx_return_fwd_1m": frame["ndx_return_fwd_1m"].values,
        "spy_return_fwd_12m": frame["spy_return_fwd_12m"].values,
        "ndx_return_fwd_12m": frame["ndx_return_fwd_12m"].values,
    }

    regressions = {}
    for name, y in targets.items():
        valid = ~np.isnan(y)
        regressions[name] = {
            "all_5_pcs": ols_r2(y[valid], scores[valid]),
            "pc1_only": ols_r2(y[valid], scores[valid, :1]),
        }

    loadings_table = []
    for i in range(args.n_components):
        row = {"component": f"PC{i+1}", "explained_variance": float(components["explained_variance_ratio"][i])}
        for j, name in enumerate(macro_cols):
            row[name] = float(components["loadings"][i, j])
        loadings_table.append(row)

    results = {
        "sample": {
            "start": frame.index.min().strftime("%Y-%m-%d"),
            "end": frame.index.max().strftime("%Y-%m-%d"),
            "months": int(len(frame)),
            "macro_variables": macro_cols,
        },
        "explained_variance_ratio_all_components": [float(v) for v in components["all_explained_variance_ratio"]],
        "explained_variance_ratio_top": [float(v) for v in components["explained_variance_ratio"]],
        "cumulative_variance_ratio_top": [float(v) for v in components["cumulative_variance_ratio"]],
        "loadings": loadings_table,
        "regressions": regressions,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    scores_frame = pd.DataFrame(scores, index=frame.index, columns=[f"pc{i+1}" for i in range(args.n_components)])
    scores_frame["spy_drawdown"] = frame["spy_drawdown"].values
    scores_frame["ndx_drawdown"] = frame["ndx_drawdown"].values
    scores_frame.reset_index().rename(columns={"index": "date"}).to_csv(args.out / "scores.csv", index=False)

    print(json.dumps(results["sample"], indent=2))
    print("explained variance (top 5):", results["explained_variance_ratio_top"])
    print(json.dumps(regressions, indent=2))


if __name__ == "__main__":
    main()
