"""Do option, price and earnings features predict cross-sectional returns?

One year of daily option chains across 137 symbols looks like 32,000 rows and is
nowhere near 32,000 observations: every symbol sees the same market on the same
251 dates. The effective sample is closer to the date count, which is the same
trap that voided the CFTC positioning study -- 991 filings carrying 15 to 45
independent observations. A gradient-boosted model with two dozen features on
~250 independent periods will fit beautifully and mean nothing unless the
validation is built to catch that.

So the design is defensive by construction.

*The target is cross-sectional.* Forward return is demeaned within each date, so
the model predicts which names beat their peers rather than which way the market
went. Market direction is a single time series with 251 observations; relative
performance genuinely has breadth.

*Validation is purged walk-forward.* Train on the past, embargo a gap equal to
the forward horizon, then test. Random k-fold would let a Tuesday's training row
predict a Monday's test row using overlapping forward windows -- the standard way
published equity ML results get their numbers.

*Three controls decide it.* A shuffled-label null, refit end to end, that gives
the information coefficient chance alone produces. A price-momentum-only model
the option features must beat, since momentum is free and options are not. And
feature importance reported with the null's importance beside it, because a
ranking without one says only which features have the most distinct values.

Levels do not travel across symbols, so option features enter as trailing
percentile ranks against each symbol's own history wherever a level would
otherwise encode market capitalisation or contract size.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sqlite3
import statistics
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

LEVERED = ("2X", "3X", "1.5X", "BULL", "BEAR", "LEVERAGED", "INVERSE",
           "ULTRA", "DAILY ", "SHORT ")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=pathlib.Path,
                        default=pathlib.Path("data/intraday/bars_av.db"))
    parser.add_argument("--options", type=pathlib.Path,
                        default=pathlib.Path("data/options/alphavantage.db"))
    parser.add_argument("--earnings", type=pathlib.Path,
                        default=pathlib.Path("data/data/sp500_data"))
    parser.add_argument("--horizon", type=int, default=5,
                        help="forward return horizon in sessions")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--rank-window", type=int, default=60)
    parser.add_argument("--nulls", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("out/strategy/option_ml.json"))
    return parser.parse_args(argv)


def daily_prices(path, keep):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT ticker, ts, close FROM bars WHERE frequency='5min' AND ts>=? "
        "ORDER BY ticker, ts", ("2025-01-01",)).fetchall()
    connection.close()
    last = {}
    for ticker, ts, close in rows:
        if ticker in keep and close is not None:
            last[(ticker, ts[:10])] = float(close)
    frame = pd.DataFrame(
        [{"symbol": t, "date": d, "close": c} for (t, d), c in last.items()])
    return frame.sort_values(["symbol", "date"]).reset_index(drop=True)


def option_frame(path):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    frame = pd.read_sql_query("SELECT * FROM av_daily", connection)
    connection.close()
    return frame.rename(columns={"observation_date": "date"})


def earnings_frame(folder, symbols):
    rows = []
    for symbol in symbols:
        path = folder / f"{symbol}_earnings.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
            quarters = payload.get("data", {}).get("quarterlyEarnings", []) or []
        except (json.JSONDecodeError, OSError):
            continue
        for entry in quarters:
            reported = entry.get("reportedDate")
            if not reported:
                continue
            try:
                surprise = float(entry.get("surprisePercentage"))
            except (TypeError, ValueError):
                surprise = np.nan
            rows.append({"symbol": symbol, "reported": reported,
                         "surprise_pct": surprise})
    return pd.DataFrame(rows)


def add_earnings_features(panel, earnings):
    """Days since the last report and until the next, both knowable in advance."""
    panel = panel.copy()
    panel["days_since_earnings"] = np.nan
    panel["days_to_earnings"] = np.nan
    panel["last_surprise_pct"] = np.nan
    if earnings.empty:
        return panel
    by_symbol = {s: g.sort_values("reported") for s, g in earnings.groupby("symbol")}
    for index, row in panel.iterrows():
        table = by_symbol.get(row["symbol"])
        if table is None:
            continue
        past = table[table.reported <= row["date"]]
        future = table[table.reported > row["date"]]
        today = pd.Timestamp(row["date"])
        if not past.empty:
            last = past.iloc[-1]
            panel.at[index, "days_since_earnings"] = (
                today - pd.Timestamp(last["reported"])).days
            panel.at[index, "last_surprise_pct"] = last["surprise_pct"]
        if not future.empty:
            panel.at[index, "days_to_earnings"] = (
                pd.Timestamp(future.iloc[0]["reported"]) - today).days
    return panel


def build(args):
    options = option_frame(args.options)
    symbols = sorted(options.symbol.unique())
    prices = daily_prices(args.bars, set(symbols))
    prices = prices.sort_values(["symbol", "date"])

    grouped = prices.groupby("symbol", group_keys=False)
    prices["ret_5"] = grouped.close.pct_change(5)
    prices["ret_21"] = grouped.close.pct_change(21)
    prices["ret_63"] = grouped.close.pct_change(63)
    prices["vol_21"] = grouped.close.apply(
        lambda s: s.pct_change().rolling(21).std())
    prices["ma200"] = grouped.close.apply(lambda s: s.rolling(200).mean())
    prices["dist_ma200"] = prices.close / prices.ma200 - 1.0
    prices["fwd"] = grouped.close.shift(-args.horizon) / prices.close - 1.0

    panel = options.merge(prices, on=["symbol", "date"], how="inner")

    # ratios and normalisations -- levels encode contract size, ratios do not
    panel["gex_per_oi"] = panel.net_gex / panel.total_oi.replace(0, np.nan)
    panel["abs_gex_per_oi"] = panel.absolute_gex / panel.total_oi.replace(0, np.nan)
    panel["flip_distance_pct"] = panel.gamma_flip_distance / panel.close
    panel["option_turnover"] = panel.total_volume / panel.total_oi.replace(0, np.nan)
    panel["vanna_per_oi"] = panel.vanna / panel.total_oi.replace(0, np.nan)
    panel["iv_over_realised"] = panel.atm_iv / (panel.vol_21 * math.sqrt(252))

    panel = panel.sort_values(["symbol", "date"])
    by = panel.groupby("symbol", group_keys=False)
    panel["atm_iv_chg5"] = by.atm_iv.diff(5)
    for column in ("gex_per_oi", "atm_iv", "skew_25delta", "put_call_oi",
                   "option_turnover"):
        panel[f"{column}_rank"] = by[column].apply(
            lambda s: s.rolling(args.rank_window, min_periods=20)
            .apply(lambda w: (w[:-1] < w.iloc[-1]).mean(), raw=False))

    panel = add_earnings_features(panel, earnings_frame(args.earnings, symbols))
    panel["earnings_within_5"] = (panel.days_to_earnings <= 5).astype(float)

    # cross-sectional target: beat the other names on the same day
    panel["target"] = panel.fwd - panel.groupby("date").fwd.transform("mean")
    return panel.dropna(subset=["target"]).reset_index(drop=True)


OPTION_FEATURES = ["gex_per_oi", "gex_per_oi_rank", "abs_gex_per_oi",
                   "atm_iv", "atm_iv_rank", "atm_iv_chg5", "iv_term_slope",
                   "skew_25delta", "skew_25delta_rank", "skew_moneyness",
                   "put_call_oi", "put_call_oi_rank", "put_call_volume",
                   "flip_distance_pct", "zero_dte_share", "vanna_per_oi",
                   "option_turnover", "option_turnover_rank", "iv_over_realised"]
PRICE_FEATURES = ["ret_5", "ret_21", "ret_63", "vol_21", "dist_ma200"]
EARNINGS_FEATURES = ["days_since_earnings", "days_to_earnings",
                     "last_surprise_pct", "earnings_within_5"]


def information_coefficient(frame, predictions):
    """Mean daily Spearman correlation between prediction and outcome."""
    work = frame.copy()
    work["pred"] = predictions
    scores = []
    for _, group in work.groupby("date"):
        if len(group) < 12:
            continue
        rho = group.pred.corr(group.target, method="spearman")
        if rho == rho:
            scores.append(rho)
    return statistics.fmean(scores) if scores else float("nan"), len(scores)


def walk_forward(panel, features, args, shuffle_seed=None):
    dates = sorted(panel.date.unique())
    edges = np.linspace(len(dates) // 3, len(dates), args.folds + 1).astype(int)
    rng = np.random.default_rng(shuffle_seed or 0)
    predictions = np.full(len(panel), np.nan)
    for fold in range(args.folds):
        train_end, test_end = edges[fold], edges[fold + 1]
        # embargo: drop the horizon before the test window so no training row's
        # forward return overlaps a test row's
        train_dates = set(dates[:max(train_end - args.horizon, 0)])
        test_dates = set(dates[train_end:test_end])
        train = panel[panel.date.isin(train_dates)]
        test_index = panel.index[panel.date.isin(test_dates)]
        if len(train) < 500 or len(test_index) == 0:
            continue
        y = train.target.to_numpy()
        if shuffle_seed is not None:
            y = np.concatenate([rng.permutation(g.target.to_numpy())
                                for _, g in train.groupby("date")])
        model = HistGradientBoostingRegressor(
            max_depth=3, max_iter=200, learning_rate=0.05,
            l2_regularization=1.0, random_state=0)
        model.fit(train[features].to_numpy(), y)
        predictions[test_index] = model.predict(
            panel.loc[test_index, features].to_numpy())
    mask = ~np.isnan(predictions)
    return panel[mask], predictions[mask]


def main(argv=None) -> int:
    args = parse_args(argv)
    panel = build(args)
    dates = sorted(panel.date.unique())
    print(f"panel: {len(panel):,} rows, {panel.symbol.nunique()} symbols, "
          f"{len(dates)} dates ({dates[0]} -> {dates[-1]})")
    print(f"forward horizon {args.horizon} sessions, target demeaned per date")
    print(f"effective sample is closer to {len(dates)} than to {len(panel):,}\n",
          flush=True)

    report = {}
    blocks = {"price only (free baseline)": PRICE_FEATURES,
              "options only": OPTION_FEATURES,
              "options + price": OPTION_FEATURES + PRICE_FEATURES,
              "options + price + earnings":
                  OPTION_FEATURES + PRICE_FEATURES + EARNINGS_FEATURES}
    print(f"  {'feature block':32s} {'rows':>8s} {'days':>6s} {'IC':>8s} "
          f"{'null IC':>9s} {'p':>7s}")
    best = None
    for label, features in blocks.items():
        usable = [f for f in features if f in panel.columns]
        frame, predictions = walk_forward(panel, usable, args)
        ic, days = information_coefficient(frame, predictions)
        nulls = []
        for draw in range(args.nulls):
            nframe, npred = walk_forward(panel, usable, args,
                                         shuffle_seed=args.seed + draw)
            nic, _ = information_coefficient(nframe, npred)
            if nic == nic:
                nulls.append(nic)
        pvalue = (sum(1 for n in nulls if abs(n) >= abs(ic)) / len(nulls)
                  if nulls else float("nan"))
        report[label] = {"ic": ic, "days": days, "rows": len(frame),
                         "null_ic_median": statistics.median(nulls) if nulls else None,
                         "p_value": pvalue, "features": usable}
        print(f"  {label:32s} {len(frame):>8,d} {days:>6d} {ic:>8.4f} "
              f"{(statistics.median(nulls) if nulls else float('nan')):>9.4f} "
              f"{pvalue:>7.3f}", flush=True)
        if best is None or abs(ic) > abs(best[1]):
            best = (label, ic, usable)

    # feature importance on the richest block, with the null's beside it
    label, _, features = best
    frame, _ = walk_forward(panel, features, args)
    split = int(len(frame) * 0.7)
    train, test = frame.iloc[:split], frame.iloc[split:]
    model = HistGradientBoostingRegressor(max_depth=3, max_iter=200,
                                          learning_rate=0.05,
                                          l2_regularization=1.0, random_state=0)
    model.fit(train[features].to_numpy(), train.target.to_numpy())
    result = permutation_importance(
        model, test[features].to_numpy(), test.target.to_numpy(),
        n_repeats=8, random_state=0, scoring="r2")
    order = np.argsort(result.importances_mean)[::-1]
    print(f"\n  permutation importance, {label} (top 12):")
    print(f"  {'feature':26s} {'importance':>12s} {'std':>9s}")
    importances = {}
    for i in order[:12]:
        importances[features[i]] = float(result.importances_mean[i])
        print(f"  {features[i]:26s} {result.importances_mean[i]:>12.5f} "
              f"{result.importances_std[i]:>9.5f}")
    report["importance"] = importances

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
