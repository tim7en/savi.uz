"""Does falling less than your peers predict compounding better than they do?

A different question from the two already answered.  Absolute drawdown failed --
deeper was worse, and depth carried a t-statistic of 0.36.  Accounting quality
worked, but only on the median and only at the extremes of the ranking.  This
asks about neither: it asks whether a stock that *holds up* when the market falls
goes on to compound better than one in the same business that falls harder.

The measure is downside capture -- what share of the market's decline a name
takes on the days the market actually declines.  A capture of 0.6 means the stock
gave up sixty cents for every dollar the index lost.  It is computed over the
trailing year and it is deliberately not beta: beta is symmetric and a name can
have a high beta because it rises hard, which is not what is being asked about
here.

The comparison is made **within correlation clusters**, because the question is
about the same field.  A utility falling less than a semiconductor says nothing
except that it is a utility.  A semiconductor falling less than other
semiconductors is the claim under test.  Clusters are measured on data before the
evaluation window and applied forward.

Two things get separated that are easily confused.

*Resilience against the market* -- capture measured against a broad index.

*Resilience against peers* -- the same name ranked inside its own cluster, which
removes the sector's own defensiveness and leaves only relative standing.

And one cross that decides how useful it is: whether resilience adds anything
once accounting quality is already known, or whether it is the same information
arriving by a different route.  Forward returns are excess over an equal-weight
hold of the same names in the same month, and the median is the headline for the
reasons the quality study set out.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sqlite3
import sys
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from savi_uz.risk_clustering import (  # noqa: E402
    average_linkage, correlation_matrix, distance_for_correlation, log_returns,
    resample_weekly,
)
from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

import run_quality_holding as quality  # noqa: E402
import run_quality_value_drawdown as base  # noqa: E402

HORIZONS = (252, 504, 756)
LABELS = {252: "1y", 504: "2y", 756: "3y"}
CAPTURE_WINDOW = 252


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", type=Path,
                        default=Path("data/cross_assets/etf_daily.db"))
    parser.add_argument("--market-ticker", default="DIA")
    parser.add_argument("--bars", type=Path,
                        default=Path("data/13f/alphavantage_daily.db"))
    parser.add_argument("--cluster-until", default="2013-01-01")
    parser.add_argument("--cluster-cut", type=float, default=0.45,
                        help="a looser cut collapses the universe into one blob")
    parser.add_argument("--drop-illiquid", type=float, default=0.20,
                        help="share of the cross-section dropped each month")
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/relative_resilience.json"))
    known, _ = parser.parse_known_args(argv)
    return known


def load_with_volume(args):
    """Bars including volume. The shared loader drops it, and the illiquidity
    filter below needs dollar turnover to keep stale prices out of the sample."""
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    book = {}
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency='daily' "
            "ORDER BY ticker"):
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='daily' ORDER BY ts", (ticker,)).fetchall()
        bars = adjust_bars([Bar(r[0][:10], r[1], r[2], r[3], r[4], r[5])
                            for r in rows], splits.get(ticker, []))
        if len(bars) >= 500:
            book[ticker] = bars
    connection.close()
    return book


def market_series(args):
    splits = load_splits(args.market)
    connection = sqlite3.connect(f"file:{args.market}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT ts,open,high,low,close FROM bars WHERE ticker=? AND "
        "frequency='daily' ORDER BY ts", (args.market_ticker,)).fetchall()
    connection.close()
    bars = adjust_bars([Bar(r[0][:10], r[1], r[2], r[3], r[4], None) for r in rows],
                       splits.get(args.market_ticker, []))
    return {b.timestamp: b.close for b in bars}


def clusters_before(book, until, cut):
    frame = {}
    for ticker, bars in book.items():
        closes = {b.timestamp: b.close for b in bars if b.timestamp < until}
        if len(closes) >= 250:
            frame[ticker] = pd.Series(closes, dtype=float)
    prices = pd.DataFrame(frame).sort_index()
    prices.index = pd.to_datetime(prices.index)
    returns = log_returns(resample_weekly(prices)).dropna(how="all")
    keep = returns.columns[returns.notna().sum() >= 60]
    corr, _ = correlation_matrix(returns[keep], min_periods=40, shrinkage=0.10)
    groups = average_linkage(corr).cut(distance_for_correlation(cut))
    label = {}
    for index, members in enumerate(groups):
        for name in members:
            label[name] = index
    sizes = sorted((len(m) for m in groups), reverse=True)
    print(f"  {len(groups)} clusters from data before {until}; "
          f"largest {sizes[:5]}")
    return label


def downside_capture(stock_days, stock, market_days, market, spot):
    """Share of the market's fall this name took, over the trailing year."""
    window = stock_days[max(0, spot - CAPTURE_WINDOW):spot + 1]
    down_market, down_stock = 0.0, 0.0
    previous = None
    for day in window:
        if day not in market:
            continue
        if previous is not None and previous in market:
            move = market[day] / market[previous] - 1.0
            if move < 0:
                own = stock[day] / stock[previous] - 1.0
                down_market += move
                down_stock += own
        previous = day
    if down_market >= -0.05:
        return None
    return down_stock / down_market


def summarise(chunk, horizon):
    values = [r[f"xs_{horizon}"] for r in chunk if r.get(f"xs_{horizon}") is not None]
    if len(values) < 40:
        return None
    return {"n": len(values), "median": statistics.median(values),
            "mean": statistics.fmean(values),
            "win": sum(1 for v in values if v > 0) / len(values)}


def band(chunk, label, width=28):
    cells = []
    for horizon in HORIZONS:
        got = summarise(chunk, horizon)
        cells.append(f"{got['median']:>+9.1%}" if got else f"{'-':>9s}")
    got = summarise(chunk, 756) or summarise(chunk, 252)
    print(f"  {label:{width}s} {(got['n'] if got else 0):>7,d} " + " ".join(cells)
          + (f" {got['win']:>7.0%}" if got else f" {'-':>7s}"))
    return got


def main(argv=None) -> int:
    args = parse_args(argv)
    inner = base.parse_args([])
    mapping = json.loads(inner.map.read_text())
    book = load_with_volume(args)
    mapping = {t: c for t, c in mapping.items() if t in book}
    panel = base.fundamentals(inner, mapping)
    market = market_series(args)
    market_days = sorted(market)
    label = clusters_before(book, args.cluster_until, args.cluster_cut)

    prices = {t: {b.timestamp: b.close for b in bars} for t, bars in book.items()}
    dollars = {t: {b.timestamp: b.close * (b.volume or 0.0) for b in bars}
               for t, bars in book.items()}
    ordered = {t: sorted(p) for t, p in prices.items()}
    calendar = sorted({d for days in ordered.values() for d in days})
    months, seen = [], set()
    for day in calendar:
        if day[:7] not in seen and day >= "2011-01-01":
            seen.add(day[:7])
            months.append(day)

    rows = []
    for day in months:
        here = []
        for ticker, days in ordered.items():
            spot = bisect_right(days, day) - 1
            if spot < CAPTURE_WINDOW + 5 or spot + min(HORIZONS) >= len(days):
                continue
            capture = downside_capture(days, prices[ticker], market_days,
                                       market, spot)
            if capture is None:
                continue
            entry = prices[ticker][days[spot]]
            if entry <= 0:
                continue
            window = days[max(0, spot - 252):spot + 1]
            turnover = statistics.median(
                [dollars[ticker][d] for d in window if d in dollars[ticker]] or [0.0])
            own_dd = entry / max(prices[ticker][d] for d in window) - 1.0
            row = {"ticker": ticker, "day": day, "capture": capture,
                   "turnover": turnover, "own_dd": own_dd,
                   "cluster": label.get(ticker)}
            facts = panel.get(ticker)
            if facts:
                usable = [f for f in facts if f[0] <= day]
                if usable:
                    score = quality.quality_score(usable[-1][1])
                    if sum(1 for k in quality.COMPONENTS if k in score) >= 3:
                        row.update(score)
            for horizon in HORIZONS:
                ahead = spot + horizon
                row[f"fwd_{horizon}"] = (prices[ticker][days[ahead]] / entry - 1.0
                                         if ahead < len(days) else None)
            here.append(row)
        if len(here) < 25:
            continue
        # Drop the least-traded names before anything is ranked: a stock that
        # barely trades shows a low downside capture because its price is stale,
        # not because it held up, and that lands straight in the top bucket.
        here.sort(key=lambda r: r["turnover"])
        here = here[int(len(here) * args.drop_illiquid):]
        if len(here) < 20:
            continue
        for horizon in HORIZONS:
            have = [r for r in here if r[f"fwd_{horizon}"] is not None]
            if len(have) < 15:
                for r in here:
                    r[f"xs_{horizon}"] = None
                continue
            average = statistics.fmean(r[f"fwd_{horizon}"] for r in have)
            for r in here:
                r[f"xs_{horizon}"] = (r[f"fwd_{horizon}"] - average
                                      if r[f"fwd_{horizon}"] is not None else None)
        # rank capture inside the whole cross-section and inside each cluster
        values = sorted(r["capture"] for r in here)
        for r in here:
            r["capture_rank"] = ((bisect_right(values, r["capture"]) - 1)
                                 / max(len(values) - 1, 1))
        by_cluster = defaultdict(list)
        for r in here:
            if r["cluster"] is not None:
                by_cluster[r["cluster"]].append(r)
        for group in by_cluster.values():
            if len(group) < 4:
                continue
            inner_values = sorted(g["capture"] for g in group)
            for g in group:
                g["peer_rank"] = ((bisect_right(inner_values, g["capture"]) - 1)
                                  / max(len(inner_values) - 1, 1))
        rows.extend(here)

    print(f"\n{len(rows):,d} name-months, {len({r['day'] for r in rows})} months, "
          f"{len({r['ticker'] for r in rows})} names, "
          f"{rows[0]['day']} to {rows[-1]['day']}")
    peers = [r for r in rows if "peer_rank" in r]
    print(f"{len(peers):,d} have at least three peers in their cluster")
    caps = sorted(r["capture"] for r in rows)
    print(f"downside capture: median {statistics.median(caps):.2f}, "
          f"quartiles {caps[len(caps)//4]:.2f} to {caps[3*len(caps)//4]:.2f}\n")
    report = {"observations": len(rows), "with_peers": len(peers)}

    header = (f"  {'bucket':28s} {'n':>7s} " +
              " ".join(f"{'med ' + LABELS[h]:>9s}" for h in HORIZONS) + f" {'win':>7s}")

    print("########## resilience against the market, whole cross-section ##########")
    print("  Low capture = fell less than the index on its down days.")
    print(header)
    ranked = sorted(rows, key=lambda r: r["capture"])
    fifth = len(ranked) // 5
    report["market_quintiles"] = {}
    for q in range(5):
        chunk = ranked[q * fifth:(q + 1) * fifth]
        name = (f"Q{q+1} capture {statistics.median(c['capture'] for c in chunk):.2f}"
                + (" (most resilient)" if q == 0 else " (falls hardest)" if q == 4 else ""))
        got = band(chunk, name)
        if got:
            report["market_quintiles"][f"Q{q+1}"] = got

    print("\n########## resilience against peers in the same cluster ##########")
    print("  The question as asked: same field, who fell shallower.")
    print(header)
    ranked_peers = sorted(peers, key=lambda r: r["peer_rank"])
    fifth = len(ranked_peers) // 5
    report["peer_quintiles"] = {}
    for q in range(5):
        chunk = ranked_peers[q * fifth:(q + 1) * fifth]
        name = f"P{q+1}" + (" shallowest in field" if q == 0
                            else " deepest in field" if q == 4 else "")
        got = band(chunk, name)
        if got:
            report["peer_quintiles"][f"P{q+1}"] = got

    print("\n########## does resilience add anything to quality? ##########")
    print(header)
    graded = [r for r in rows if "roe" in r and "solvency" in r]
    if graded:
        by_month = defaultdict(list)
        for r in graded:
            by_month[r["day"]].append(r)
        for group in by_month.values():
            if len(group) < 12:
                continue
            for key in ("roe", "solvency"):
                values = sorted(g[key] for g in group if key in g)
                for g in group:
                    if key in g:
                        g[f"{key}_r"] = ((bisect_right(values, g[key]) - 1)
                                         / max(len(values) - 1, 1))
            for g in group:
                if "roe_r" in g and "solvency_r" in g:
                    g["q"] = (g["roe_r"] + g["solvency_r"]) / 2
        scored = [r for r in graded if "q" in r and "peer_rank" in r]
        median_q = statistics.median(r["q"] for r in scored)
        report["cross"] = {}
        for label_text, test in (
                ("high quality + shallow fall",
                 lambda r: r["q"] >= median_q and r["peer_rank"] <= 0.4),
                ("high quality + deep fall",
                 lambda r: r["q"] >= median_q and r["peer_rank"] >= 0.6),
                ("low quality + shallow fall",
                 lambda r: r["q"] < median_q and r["peer_rank"] <= 0.4),
                ("low quality + deep fall",
                 lambda r: r["q"] < median_q and r["peer_rank"] >= 0.6)):
            chunk = [r for r in scored if test(r)]
            got = band(chunk, label_text)
            if got:
                report["cross"][label_text] = got

    print("\n########## the three-way cell ##########")
    print("  A quality business that does not amplify the market, already fallen")
    print("  hard on its own. Capture and own-drawdown measure different things:")
    print("  one is how much of the index's fall you take, the other is how far")
    print("  you have dropped from your own high for your own reasons.")
    print(header)
    graded3 = [r for r in rows if "q" in r and "own_dd" in r]
    if graded3:
        cap_mid = statistics.median(r["capture"] for r in graded3)
        q_mid = statistics.median(r["q"] for r in graded3)
        print(f"  (median capture {cap_mid:.2f}, quality split at its median)")
        report["three_way"] = {}
        for text, test in (
                ("quality + low capture + down 25%+",
                 lambda r: r["q"] >= q_mid and r["capture"] <= cap_mid
                 and r["own_dd"] <= -0.25),
                ("quality + low capture, any entry",
                 lambda r: r["q"] >= q_mid and r["capture"] <= cap_mid),
                ("quality + high capture + down 25%+",
                 lambda r: r["q"] >= q_mid and r["capture"] > cap_mid
                 and r["own_dd"] <= -0.25),
                ("low quality + down 25%+",
                 lambda r: r["q"] < q_mid and r["own_dd"] <= -0.25),
                ("everything", lambda r: True)):
            chunk = [r for r in graded3 if test(r)]
            got = band(chunk, text)
            if got:
                report["three_way"][text] = got

    print("\n########## the most and least resilient names ##########")
    average = defaultdict(list)
    for r in rows:
        average[r["ticker"]].append(r["capture"])
    ranking = sorted(((statistics.median(v), t) for t, v in average.items()
                      if len(v) >= 40))
    print("  fell least: " + ", ".join(f"{t} {c:.2f}" for c, t in ranking[:10]))
    print("  fell most:  " + ", ".join(f"{t} {c:.2f}" for c, t in ranking[-8:]))
    report["capture_by_name"] = {t: c for c, t in ranking}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
