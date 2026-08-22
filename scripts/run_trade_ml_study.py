"""What separates the breakout trades that made money from the ones that did not.

The programme has spent eleven overlays asking whether some outside series --
dealer gamma, theme strength, macro regime -- can tell a good breakout from a
bad one, and every one of them failed.  This asks the same question from the
inside: given only what the price series itself knows at the moment the stop
fills, is there any structure that separates a 4R trade from a -1R trade?

Two halves, and they answer different questions.

*The descriptive half* ranks each condition against realised net R and reports
the quintile spread.  This is the honest answer to "what did the winners have in
common" and it is not a strategy.  A condition can separate outcomes beautifully
in-sample and still be untradeable, because the spread may live entirely in one
regime or one instrument.

*The model half* is the one that can be traded, and it is built to fail loudly.
LightGBM is fitted walk-forward -- trained only on trades that had already
closed, predicting a year it has never seen -- and the prediction is then used
to decline trades before the six-slot cap sees them, so a declined trade frees
its slot for the next name rather than sitting idle.

Three controls decide it, and the second is the one that matters.

*The unfiltered book*, so the comparison is against the incumbent.

*The reversal.*  The same model, taking the trades it ranked worst.  If the
bottom half performs as well as the top half, the ranking carries no
information and the top half was a coin that landed well.  Three of the eleven
rejected overlays died exactly here.

*A random filter of the same size.*  Declining half the trades changes the
capacity ordering, and the random tie-break alone spans a wide Sharpe band, so
a filter must beat a coin that declines the same number of trades.

Every feature reads bars strictly before the entry bar.  A stop order fills
intrabar, so the entry bar's own high, low, close and volume are unknowable at
fill time and none of them appear.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import (  # noqa: E402
    TurtleConfig, relative_volume, rolling_extremes, run_turtle, trailing_mean,
    wilder_atr,
)
from savi_uz.volume_profile import Bar  # noqa: E402

FIXED = dict(entry_window=55, exit_window=20, atr_window=20,
             skip_after_winner=False, use_channel_exit=False, chandelier_atr=3.0)

PERIODS = [("2017-18", "2017", "2019"), ("2019-20", "2019", "2021"),
           ("2021-22", "2021", "2023"), ("2023-24", "2023", "2025"),
           ("2025-26", "2025", "2027")]

FEATURES = [
    "trend_50", "trend_200", "mom_20", "mom_60", "vol_ratio", "n_pct",
    "extension", "channel_width", "dd_from_high", "bars_since_high",
    "vwap_dist", "rel_volume", "close_loc", "rs_rank",
    "mkt_trend_200", "mkt_vol_ratio", "mkt_mom_20", "bar_of_session",
]

MARKET_KEYS = ("mkt_trend_200", "mkt_vol_ratio", "mkt_mom_20")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--frequency", default="5min",
                        help="the stored bar frequency to read")
    parser.add_argument("--minutes", type=int, default=240,
                        help="resample target; 0 leaves the stored frequency alone")
    parser.add_argument("--market-ticker", default="SPY",
                        help="broad-market series behind the mkt_* features")
    parser.add_argument("--market-db", type=Path, default=None,
                        help="where to find it, when the book has no index member")
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--cost-bp", type=float, default=10.0)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--null-draws", type=int, default=20)
    parser.add_argument("--oos-from", type=int, default=2021,
                        help="first year predicted out of sample")
    parser.add_argument("--min-sessions", type=int, default=400)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/trade_ml.json"))
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# data


def load_series(path, frequency, start, minutes, keep=None):
    """Bars per ticker, split-adjusted where the database records splits.

    ``minutes`` of zero leaves the series at the frequency it is stored in,
    which is what a daily database wants.  Anything else resamples into
    session-anchored windows, and only an intraday source can supply those.
    """
    splits = load_splits(path)
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    book = {}
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency=? ORDER BY ticker",
            (frequency,)):
        if keep is not None and ticker not in keep:
            continue
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency=? AND ts>=? ORDER BY ts",
            (ticker, frequency, start)).fetchall()
        bars = adjust_bars([Bar(*r) for r in rows], splits.get(ticker, []))
        book[ticker] = bars if minutes == 0 else resample_regular_session(
            bars, minutes=minutes)
    connection.close()
    return book


def load_book(args):
    book = load_series(args.bars, args.frequency, args.start, args.minutes)
    return {ticker: bars for ticker, bars in book.items()
            if len({b.timestamp[:10] for b in bars}) >= args.min_sessions}


def load_market(args, book):
    """The broad-market series behind the mkt_* features.

    Read from a separate database when one is given: the 13F daily universe is
    a list of holdings and contains no index member of its own, so the market
    regime has to be supplied from outside it.
    """
    if args.market_db:
        bars = load_series(args.market_db, args.frequency, args.start,
                           args.minutes, keep={args.market_ticker}
                           ).get(args.market_ticker)
    else:
        bars = book.get(args.market_ticker)
    if not bars:
        print(f"  no {args.market_ticker} series -- mkt_* features will be blank")
        return {}
    columns = build_columns(bars)
    market = {}
    for index, bar in enumerate(bars):
        if index < 202:
            continue
        n = columns["atr"][index - 1]
        close = columns["close"][index - 1]
        market[bar.timestamp] = {
            "mkt_trend_200": ratio(close - columns["ma200"][index - 1], n),
            "mkt_vol_ratio": ratio(n, columns["atr_mean"][index - 1]),
            "mkt_mom_20": ratio(close - columns["close"][index - 21], n),
        }
    return market


def ratio(numerator, denominator):
    if denominator is None or math.isnan(denominator) or denominator <= 0:
        return math.nan
    return numerator / denominator


def build_columns(bars):
    """Every series a feature needs, each aligned so index i-1 is readable at i."""
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    typical = [(b.high + b.low + b.close) / 3 for b in bars]
    volumes = [max(float(b.volume or 0.0), 0.0) for b in bars]
    atr = wilder_atr(bars, FIXED["atr_window"])
    columns = {
        "close": closes,
        "atr": atr,
        "ma50": trailing_mean(closes, 50),
        "ma200": trailing_mean(closes, 200),
        "atr_mean": trailing_mean([0.0 if math.isnan(a) else a for a in atr], 100),
        "hi252": rolling_extremes(highs, 252, True),
        "hi55": rolling_extremes(highs, FIXED["entry_window"], True),
        "lo55": rolling_extremes(lows, FIXED["entry_window"], False),
        "relvol": relative_volume(bars, 20),
    }

    # trailing 13-bar VWAP, the anchor the dip study used
    vwap = [math.nan] * len(bars)
    pv = vv = 0.0
    for index in range(len(bars)):
        pv += typical[index] * volumes[index]
        vv += volumes[index]
        if index >= 13:
            pv -= typical[index - 13] * volumes[index - 13]
            vv -= volumes[index - 13]
        if index >= 12 and vv > 0:
            vwap[index] = pv / vv
    columns["vwap"] = vwap

    # bars since the running high, a staleness measure the extremes do not carry
    since = [0.0] * len(bars)
    best, age = -math.inf, 0
    for index in range(len(bars)):
        if highs[index] >= best:
            best, age = highs[index], 0
        else:
            age += 1
        since[index] = float(age)
    columns["since_high"] = since

    columns["close_loc"] = [ratio(b.close - b.low, b.high - b.low) for b in bars]
    columns["ret60"] = [
        ratio(closes[i] - closes[i - 60], closes[i - 60]) if i >= 60 else math.nan
        for i in range(len(bars))]
    return columns


def instrument_features(columns, index, bar_of_session):
    """Read at entry bar ``index`` using only bars that had already closed."""
    previous = index - 1
    n = columns["atr"][previous]
    if not n or math.isnan(n) or n <= 0:
        return None
    close = columns["close"][previous]
    return {
        "trend_50": ratio(close - columns["ma50"][previous], n),
        "trend_200": ratio(close - columns["ma200"][previous], n),
        "mom_20": ratio(close - columns["close"][index - 21], n) if index >= 21 else math.nan,
        "mom_60": ratio(close - columns["close"][index - 61], n) if index >= 61 else math.nan,
        "vol_ratio": ratio(n, columns["atr_mean"][previous]),
        "n_pct": ratio(n, close),
        "extension": ratio(close - columns["hi55"][previous], n),
        "channel_width": ratio(columns["hi55"][previous] - columns["lo55"][previous], n),
        "dd_from_high": ratio(close - columns["hi252"][previous], n),
        "bars_since_high": columns["since_high"][previous],
        "vwap_dist": ratio(close - columns["vwap"][previous], n),
        "rel_volume": columns["relvol"][previous],
        "close_loc": columns["close_loc"][previous],
        "bar_of_session": float(bar_of_session),
    }


def collect(book, market, args):
    """Run the breakout book and attach a feature row to every trade."""
    columns = {t: build_columns(b) for t, b in book.items()}
    index_of = {t: {b.timestamp: i for i, b in enumerate(bars)}
                for t, bars in book.items()}

    session_slot = {}
    for ticker, bars in book.items():
        seen = defaultdict(int)
        slots = []
        for bar in bars:
            day = bar.timestamp[:10]
            slots.append(seen[day])
            seen[day] += 1
        session_slot[ticker] = slots

    # cross-sectional relative strength: percentile of the 60-bar return, per stamp
    pool_by_stamp = defaultdict(list)
    for ticker, bars in book.items():
        series = columns[ticker]["ret60"]
        for index, bar in enumerate(bars):
            value = series[index]
            if not math.isnan(value):
                pool_by_stamp[bar.timestamp].append((value, ticker))
    rs_rank = {}
    for stamp, rows in pool_by_stamp.items():
        rows.sort()
        size = len(rows)
        for position, (_, ticker) in enumerate(rows):
            rs_rank[(ticker, stamp)] = position / (size - 1) if size > 1 else 0.5

    config = TurtleConfig(**FIXED, directions=(1,),
                          round_trip_cost=args.cost_bp / 10_000)
    blank = {k: math.nan for k in MARKET_KEYS}
    trades = []
    for ticker, bars in book.items():
        raw, _ = run_turtle(bars, config=config)
        for trade in raw:
            index = index_of[ticker].get(trade.entry_timestamp)
            if index is None or index < 262:
                continue
            row = instrument_features(columns[ticker], index,
                                      session_slot[ticker][index])
            if row is None:
                continue
            row["rs_rank"] = rs_rank.get((ticker, trade.entry_timestamp), math.nan)
            row.update(market.get(trade.entry_timestamp, blank))
            trades.append({
                "ticker": ticker,
                "entry": trade.entry_timestamp,
                "exit": trade.exit_timestamp,
                "r": trade.net_r,
                "dir": trade.direction,
                "units": trade.unit_entries,
                "bars_held": trade.bars_held,
                "reason": trade.exit_reason,
                "features": row,
            })
    trades.sort(key=lambda t: t["entry"])
    return trades


# ---------------------------------------------------------------------------
# descriptive


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    position = 0
    while position < len(order):
        stop = position
        while (stop + 1 < len(order)
               and values[order[stop + 1]] == values[order[position]]):
            stop += 1
        mean = (position + stop) / 2
        for k in range(position, stop + 1):
            out[order[k]] = mean
        position = stop + 1
    return out


def spearman(pairs):
    rows = [(x, y) for x, y in pairs
            if x is not None and not math.isnan(x) and not math.isnan(y)]
    if len(rows) < 50:
        return math.nan, len(rows)
    xs = _ranks([r[0] for r in rows])
    ys = _ranks([r[1] for r in rows])
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in xs)
                            * sum((b - my) ** 2 for b in ys))
    return (numerator / denominator if denominator else math.nan), len(rows)


def quintiles(trades, name, buckets=5):
    rows = [(t["features"][name], t["r"]) for t in trades
            if not math.isnan(t["features"].get(name, math.nan))]
    if len(rows) < 20 * buckets:
        return None
    # Ties must not be broken by the outcome.  Sorting (feature, r) pairs orders
    # tied rows by the very thing being predicted, which manufactures a spread
    # out of a feature that takes only two values.  Shuffle first, then sort on
    # the feature alone -- Python's sort is stable, so tied rows stay shuffled.
    if len({value for value, _ in rows}) < buckets:
        return None
    random.Random(11).shuffle(rows)
    rows.sort(key=lambda pair: pair[0])
    size = len(rows)
    out = []
    for bucket in range(buckets):
        chunk = rows[bucket * size // buckets:(bucket + 1) * size // buckets]
        values = [r for _, r in chunk]
        out.append({"n": len(chunk), "lo": chunk[0][0], "hi": chunk[-1][0],
                    "mean_r": statistics.fmean(values),
                    "win_rate": sum(1 for v in values if v > 0) / len(values),
                    "share_over_2r": sum(1 for v in values if v > 2) / len(values)})
    return out


def describe(trades):
    report = {}
    ranked = []
    for name in FEATURES:
        rho, count = spearman([(t["features"].get(name, math.nan), t["r"])
                               for t in trades])
        if math.isnan(rho):
            continue
        agreement = 0
        for _, lo, hi in PERIODS:
            subset = [t for t in trades if lo <= t["entry"][:4] < hi]
            sub_rho, sub_n = spearman([(t["features"].get(name, math.nan), t["r"])
                                       for t in subset])
            if not math.isnan(sub_rho) and sub_n >= 50 and (sub_rho > 0) == (rho > 0):
                agreement += 1
        table = quintiles(trades, name)
        # How many of the four bucket-to-bucket steps move the way rho claims.
        # A spread carried by one non-monotonic bucket is noise wearing a hat.
        steps = 0
        if table:
            direction = 1 if rho > 0 else -1
            steps = sum(1 for a, b in zip(table, table[1:])
                        if (b["mean_r"] - a["mean_r"]) * direction > 0)
        report[name] = {"rho": rho, "n": count, "period_agreement": agreement,
                        "monotone_steps": steps, "quintiles": table}
        if table:
            ranked.append((abs(rho), name, rho, count, table, agreement, steps))

    ranked.sort(reverse=True)
    print(f"\n{'condition at entry':18s} {'rho':>7s} {'n':>6s} {'Q1 meanR':>9s} "
          f"{'Q5 meanR':>9s} {'spread':>7s} {'Q1 >2R':>7s} {'Q5 >2R':>7s} "
          f"{'mono':>5s} {'agree':>6s}")
    print("-" * 95)
    for _, name, rho, count, table, agreement, steps in ranked:
        spread = table[-1]["mean_r"] - table[0]["mean_r"]
        print(f"{name:18s} {rho:>+7.3f} {count:>6,d} {table[0]['mean_r']:>9.2f} "
              f"{table[-1]['mean_r']:>9.2f} {spread:>+7.2f} "
              f"{table[0]['share_over_2r']:>7.1%} {table[-1]['share_over_2r']:>7.1%} "
              f"{steps:>3d}/4 {agreement:>4d}/5")
    return report


# ---------------------------------------------------------------------------
# portfolio assessment, the same machinery the dip study used


def cap(trades, limit, rng, priority=None):
    """Greedy occupancy of ``limit`` slots, oldest signal first.

    ``priority`` orders the signals that arrive on the same bar, which is the
    only place a ranking can act once the book is capacity-bound: an earlier
    trade already holds its slot regardless of how it scores.  Without one the
    tie-break is the shuffle, which is the book's current behaviour.
    """
    shuffled = list(trades)
    rng.shuffle(shuffled)
    order = ((lambda t: (t["entry"], -priority(t))) if priority
             else (lambda t: t["entry"]))
    live, taken = [], []
    for trade in sorted(shuffled, key=order):
        live = [x for x in live if x["exit"] > trade["entry"]]
        if len(live) >= limit:
            continue
        live.append(trade)
        taken.append(trade)
    return taken


def close_index(book):
    """Per ticker, the session closes plus a sorted day list to bisect into.

    Scanning the whole history per trade is quadratic in the panel and the 143
    name daily universe is where that starts to bite, so the marking window is
    sliced rather than filtered.
    """
    out = {}
    for ticker, bars in book.items():
        closes = {bar.timestamp[:10]: bar.close for bar in bars}
        out[ticker] = (sorted(closes), closes)
    return out


def marked_map(taken, closes_by_ticker):
    by_day = defaultdict(float)
    for trade in taken:
        days, closes = closes_by_ticker[trade["ticker"]]
        entry_day, exit_day = trade["entry"][:10], trade["exit"][:10]
        previous = 0.0
        window = days[bisect.bisect_left(days, entry_day):
                      bisect.bisect_left(days, exit_day)]
        for day in window:
            live = [u for u in trade["units"] if u.timestamp[:10] <= day]
            if not live:
                continue
            open_r = sum(trade["dir"] * (closes[day] - u.price) / u.n for u in live)
            by_day[day] += open_r - previous
            previous = open_r
        by_day[exit_day] += trade["r"] - previous
    return by_day


def path(values, risk):
    nav = peak = 1000.0
    worst = 0.0
    for value in values:
        nav = max(0.0, nav + value * risk * nav)
        peak = max(peak, nav)
        worst = min(worst, nav / peak - 1.0)
    return nav, worst


def solve_risk(series, target, lo=1e-6, hi=0.40):
    def drawdown(risk):
        return statistics.median(abs(path(v, risk)[1]) for _, v in series)
    if drawdown(hi) < target:
        return hi
    for _ in range(28):
        mid = math.sqrt(lo * hi)
        if drawdown(mid) < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def sharpe(stream):
    sd = statistics.pstdev(stream)
    return statistics.fmean(stream) / sd * math.sqrt(252) if sd > 0 else float("nan")


def assess(pooled, closes_by_ticker, args, priority=None):
    if len(pooled) < 100:
        return None
    caps = [cap(pooled, args.max_positions, random.Random(s), priority)
            for s in range(args.trials)]
    marks = [marked_map(t, closes_by_ticker) for t in caps]
    series = [(sorted(m), [m[d] for d in sorted(m)]) for m in marks]
    risk = solve_risk(series, args.target_dd)
    cagrs, sharpes, taken = [], [], []
    for (days, values), chosen in zip(series, caps):
        nav, _ = path(values, risk)
        years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25
        cagrs.append((nav / 1000.0) ** (1 / years) - 1 if nav > 0 else -1.0)
        sharpes.append(sharpe([v * risk for v in values]))
        taken.append(len(chosen))
    spread = sorted(sharpes)
    return {"offered": len(pooled), "taken": int(statistics.median(taken)),
            "risk_fraction": risk, "sharpe": statistics.median(spread),
            "sharpe_p05": spread[int(.05 * len(spread))],
            "sharpe_p95": spread[min(int(.95 * len(spread)), len(spread) - 1)],
            "cagr": statistics.median(cagrs)}


# ---------------------------------------------------------------------------
# model


def new_model():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                         min_child_samples=50, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, reg_lambda=1.0, verbose=-1,
                         random_state=7)


def walk_forward(trades, args):
    import numpy as np

    matrix = np.array([[t["features"].get(f, math.nan) for f in FEATURES]
                       for t in trades], dtype=float)
    target = np.array([t["r"] for t in trades], dtype=float)
    year_of = np.array([int(t["entry"][:4]) for t in trades])
    predictions = np.full(len(trades), math.nan)

    per_year = {}
    for year in sorted({int(y) for y in year_of if y >= args.oos_from}):
        train = year_of < year
        test = year_of == year
        if train.sum() < 300 or test.sum() < 20:
            continue
        model = new_model()
        model.fit(matrix[train], target[train])
        predictions[test] = model.predict(matrix[test])
        rho, _ = spearman(list(zip(predictions[test].tolist(), target[test].tolist())))
        per_year[str(year)] = {"n": int(test.sum()), "train": int(train.sum()),
                               "ic": rho}
        print(f"  {year}   train {train.sum():>5,d}   test {test.sum():>5,d}   "
              f"out-of-sample IC {rho:>+.3f}")

    covered = ~np.isnan(predictions)
    pooled_ic, _ = spearman(list(zip(predictions[covered].tolist(),
                                     target[covered].tolist())))
    print(f"  pooled out-of-sample IC {pooled_ic:+.3f} "
          f"over {int(covered.sum()):,d} trades")

    full = new_model()
    full.fit(matrix, target)
    gain = dict(sorted(zip(FEATURES, full.booster_.feature_importance("gain").tolist()),
                       key=lambda kv: -kv[1]))
    total = sum(gain.values()) or 1.0
    share = {k: v / total for k, v in gain.items()}
    print("\n  feature importance, full-sample fit (share of gain)")
    for name in list(share)[:8]:
        print(f"    {name:18s} {share[name]:>6.1%}")
    return predictions, covered, per_year, pooled_ic, share


def main(argv=None) -> int:
    args = parse_args(argv)
    book = load_book(args)
    size = "daily" if args.minutes == 0 else f"{args.minutes}-minute"
    print(f"{len(book)} instruments, {size} bars, "
          f"{args.cost_bp:g}bp round trip, long only, from {args.start}")
    if args.minutes == 0 and "bar_of_session" in FEATURES:
        FEATURES.remove("bar_of_session")  # one bar per session, nothing to say
    market = load_market(args, book)
    closes_by_ticker = close_index(book)
    trades = collect(book, market, args)
    outcomes = [t["r"] for t in trades]
    print(f"{len(trades):,d} breakout trades, mean {statistics.fmean(outcomes):+.3f}R, "
          f"win rate {sum(1 for r in outcomes if r > 0) / len(outcomes):.1%}, "
          f"{sum(1 for r in outcomes if r > 2) / len(outcomes):.1%} over +2R")

    print("\n########## what the winners had in common ##########")
    print("Rank correlation against net R, with the mean R of the bottom and top "
          "fifth.\n'agree' counts how many of five sub-periods keep the full-sample sign.")
    description = describe(trades)

    print("\n########## walk-forward model ##########")
    predictions, covered, per_year, pooled_ic, share = walk_forward(trades, args)

    oos = [t for t, flag in zip(trades, covered) if flag]
    if len(oos) < 200:
        print("too few out-of-sample trades to assess a filter")
        return 1
    scores = [float(p) for p, flag in zip(predictions, covered) if flag]
    order = sorted(range(len(oos)), key=lambda i: scores[i])
    half = len(order) // 2
    bottom = sorted(order[:half])
    top = sorted(order[half:])

    print(f"\n########## does the ranking survive its controls "
          f"({oos[0]['entry'][:4]}-{oos[-1]['entry'][:4]}) ##########")
    print(f"  {'arm':38s} {'offered':>8s} {'taken':>7s} {'Sharpe':>7s} "
          f"{'[5-95%]':>15s} {'CAGR':>8s}")
    # The score as a slot priority rather than a filter.  Once the book is
    # capacity-bound most offers are declined anyway, so reordering the signals
    # that compete on the same bar changes more than declining half of them.
    score_of = {(t["ticker"], t["entry"]): s for t, s in zip(oos, scores)}
    rank = (lambda t: score_of.get((t["ticker"], t["entry"]), 0.0))

    arms = {
        "all trades, random tie-break (incumbent)": (oos, None),
        "model top half (filter)": ([oos[i] for i in top], None),
        "model bottom half (filter reversal)": ([oos[i] for i in bottom], None),
        "model picks the slot (priority)": (oos, rank),
        "model picks worst (priority reversal)": (oos, lambda t: -rank(t)),
    }
    report = {"per_year": per_year, "pooled_ic": pooled_ic, "gain_share": share,
              "features": description, "arms": {}}
    for label, (pooled, priority) in arms.items():
        result = assess(pooled, closes_by_ticker, args, priority)
        if not result:
            continue
        report["arms"][label] = result
        print(f"  {label:38s} {result['offered']:>8,d} {result['taken']:>7,d} "
              f"{result['sharpe']:>7.2f} "
              f"{('[%.2f-%.2f]' % (result['sharpe_p05'], result['sharpe_p95'])):>15s} "
              f"{result['cagr']:>8.1%}")

    nulls = []
    for draw in range(args.null_draws):
        rng = random.Random(4100 + draw)
        outcome = assess(rng.sample(oos, len(top)), closes_by_ticker, args)
        if outcome:
            nulls.append(outcome["sharpe"])
    if nulls:
        nulls.sort()
        model_sharpe = report["arms"].get(
            "model top half (filter)", {}).get("sharpe", math.nan)
        beat = sum(1 for x in nulls if x >= model_sharpe) / len(nulls)
        print(f"  {'random half of the same size (null)':38s} {'':>8s} {'':>7s} "
              f"{statistics.median(nulls):>7.2f} "
              f"{('[%.2f-%.2f]' % (nulls[0], nulls[-1])):>15s}")
        print(f"  -> the model beats the coin in {1 - beat:.0%} of draws")
        report["arms"]["random half (null)"] = {
            "sharpe": statistics.median(nulls), "low": nulls[0], "high": nulls[-1],
            "share_null_at_or_above_model": beat}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
