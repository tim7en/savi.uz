"""What the universe actually is: how many independent bets, and against what.

Every drawdown measured in this programme has been larger than a book of twelve
positions ought to produce, and this is the obvious place to look for the reason.
Twelve slots only buy twelve independent bets if the names in them are
independent, and a universe assembled from institutional filings is not a random
draw from the market -- it is a list of things large managers liked, which is a
description of one exposure wearing 142 tickers.

Three measurements, in the order that answers the question.

*How many independent bets exist at all.*  The entropy of the correlation
matrix's eigenvalues -- N when everything is orthogonal, 1 when everything moves
together.  If this comes back near twelve, the slot cap is the binding
constraint on diversification.  If it comes back near two, the slot cap is
decoration and the book has been running one position in twelve pieces.

*What the clusters are.*  Average-linkage over correlation distance, cut at
several thresholds, with members named.  A cluster is only interesting if you
can look at it and say what it is.

*Beta, and to what.*  Against an equal-weight index of the universe itself,
which is the honest benchmark for a book that trades only these names, and
against the broad market so the two can be told apart.

Weekly returns rather than daily: daily correlation on 142 names is dominated by
non-synchronous trading and microstructure, and the question here is about
shared exposure over the horizons the book actually holds.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from savi_uz.risk_clustering import (  # noqa: E402
    annualized_volatility, average_linkage, cluster_resolution_curve,
    components_for_variance, correlation_matrix, distance_for_correlation,
    effective_number_of_bets, factor_betas, log_returns, max_correlation_to_others,
    resample_weekly,
)
from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

THRESHOLDS = (0.30, 0.45, 0.60, 0.75)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path,
                        default=Path("data/13f/alphavantage_daily.db"))
    parser.add_argument("--market", type=Path,
                        default=Path("data/cross_assets/etf_daily.db"))
    parser.add_argument("--market-ticker", default="DIA")
    parser.add_argument("--start", default="2013-01-01",
                        help="the out-of-sample window every result above used")
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/universe_structure.json"))
    return parser.parse_args(argv)


def closes(path, frequency, start, keep=None):
    splits = load_splits(path)
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    series = {}
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency=? ORDER BY ticker",
            (frequency,)):
        if keep and ticker not in keep:
            continue
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency=? AND ts>=? ORDER BY ts",
            (ticker, frequency, start)).fetchall()
        bars = adjust_bars([Bar(r[0][:10], r[1], r[2], r[3], r[4], r[5])
                            for r in rows], splits.get(ticker, []))
        if len(bars) >= 250:
            series[ticker] = pd.Series(
                {b.timestamp: b.close for b in bars}, dtype=float)
    connection.close()
    frame = pd.DataFrame(series).sort_index()
    # resample_weekly needs real timestamps, not the ISO strings the bars carry
    frame.index = pd.to_datetime(frame.index)
    return frame


def main(argv=None) -> int:
    args = parse_args(argv)
    prices = closes(args.bars, "daily", args.start)
    weekly = resample_weekly(prices)
    returns = log_returns(weekly).dropna(how="all")
    usable = returns.columns[returns.notna().sum() >= 60]
    returns = returns[usable]
    print(f"{len(usable)} names with enough weekly history from {args.start}, "
          f"{len(returns)} weeks\n")

    corr, pairs = correlation_matrix(returns, min_periods=40, shrinkage=0.10)
    bets = effective_number_of_bets(corr)
    off = corr.to_numpy()[~np.eye(len(corr), dtype=bool)]
    report = {"names": int(len(usable)), "weeks": int(len(returns)),
              "average_correlation": float(np.mean(off)),
              "effective_bets": bets,
              "components_for_80pct": int(components_for_variance(corr, 0.80))}

    print("########## how many independent bets ##########")
    print(f"  average pairwise correlation : {np.mean(off):+.3f}")
    print(f"  effective number of bets     : {bets:.2f}   (out of {len(usable)})")
    print(f"  components for 80% of variance: {components_for_variance(corr, 0.80)}")
    print(f"\n  A twelve-slot cap can only buy twelve independent bets if the "
          f"universe\n  contains twelve. It contains {bets:.1f}.")

    print("\n########## how the universe resolves into clusters ##########")
    curve = cluster_resolution_curve(corr, THRESHOLDS)
    print(f"  {'corr cut':>9s} {'clusters':>9s} {'singletons':>11s} {'largest':>8s}")
    for _, row in curve.iterrows():
        print(f"  {row['corr_threshold']:>9.2f} {int(row['clusters']):>9d} "
              f"{int(row['singletons']):>11d} {int(row['largest_cluster']):>8d}")
    report["resolution"] = curve.to_dict(orient="records")

    tree = average_linkage(corr)
    clusters = tree.cut(distance_for_correlation(0.45))
    volatility = annualized_volatility(returns, periods_per_year=52)
    print(f"\n  blocks at a 0.45 correlation cut "
          f"({sum(1 for c in clusters if len(c) > 2)} with three or more members)")
    named = []
    for cluster in clusters:
        if len(cluster) < 3:
            continue
        inner = corr.loc[cluster, cluster].to_numpy()
        mean_inner = float(inner[~np.eye(len(cluster), dtype=bool)].mean())
        named.append({"size": len(cluster), "inner_corr": mean_inner,
                      "members": cluster,
                      "median_vol": float(volatility[cluster].median())})
        print(f"    {len(cluster):>3d} names, inner corr {mean_inner:+.2f}, "
              f"vol {volatility[cluster].median():.0%}  "
              f"{', '.join(cluster[:9])}{' ...' if len(cluster) > 9 else ''}")
    report["clusters"] = named

    print("\n########## beta, and to what ##########")
    # Two univariate fits, not one joint one. An equal-weight index of these
    # same names already contains the market, so regressing on both at once
    # leaves the broad index only the residual and reports a beta near zero that
    # says nothing about market exposure.
    own = returns.mean(axis=1).rename("universe")
    betas = factor_betas(returns, pd.DataFrame({"universe": own})).dropna()
    market = closes(args.market, "daily", args.start, keep={args.market_ticker})
    if not market.empty:
        broad = log_returns(resample_weekly(market))[args.market_ticker]
        broad.name = "market"
        against_market = factor_betas(returns, pd.DataFrame({"market": broad}))
        betas = betas.join(against_market, how="left")
    print(f"  against an equal-weight index of these same names:")
    print(f"    median beta {betas['universe'].median():.2f}, "
          f"quartiles {betas['universe'].quantile(.25):.2f} to "
          f"{betas['universe'].quantile(.75):.2f}")
    if "market" in betas:
        print(f"  against {args.market_ticker}:")
        print(f"    median beta {betas['market'].median():.2f}, "
              f"quartiles {betas['market'].quantile(.25):.2f} to "
              f"{betas['market'].quantile(.75):.2f}")
    report["beta"] = {
        "universe_median": float(betas["universe"].median()),
        "universe_q1": float(betas["universe"].quantile(.25)),
        "universe_q3": float(betas["universe"].quantile(.75))}
    if "market" in betas:
        report["beta"]["market_median"] = float(betas["market"].median())

    high = betas["universe"].sort_values(ascending=False)
    print(f"\n  highest beta to the universe : "
          f"{', '.join(f'{t} {v:.1f}' for t, v in high.head(6).items())}")
    print(f"  lowest                       : "
          f"{', '.join(f'{t} {v:.1f}' for t, v in high.tail(6).items())}")

    lonely = max_correlation_to_others(corr).sort_values()
    print(f"\n  least correlated to anything else: "
          f"{', '.join(f'{t} {v:.2f}' for t, v in lonely.head(6).items())}")
    report["most_independent"] = {t: float(v) for t, v in lonely.head(10).items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
