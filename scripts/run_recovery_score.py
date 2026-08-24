"""Does a stock's past recovery record predict its next recovery?

The proposition: quality shows up in price behaviour, not in accounting ratios.
A business with real durability falls and comes back; one without it falls and
stays down.  So rank names by how they have handled their previous drawdowns and
buy the ones with the best record whenever they fall again.

This is the most seductive idea in the whole thread and the easiest to fake.
Rating a stock on how well it recovered and then checking whether it recovers is
circular, and on a universe of survivors it produces a spectacular result that
means nothing.  Three things keep it honest.

*Only resolved episodes count.*  A name is scored at month T using drawdowns that
both began and finished before T.  A drawdown still open at T tells you the
answer to the question being asked and is excluded.

*The score is recomputed every month.*  A name that recovered well through 2010
and stopped recovering in 2015 is scored high in 2012 and low in 2018, which is
what a real user of the rule would have seen.

*The benchmark is the index, not zero.*  Everything is measured against what DIA
did over identical dates, because a stock that recovers when the whole market
recovers has demonstrated nothing about itself.

The score has three parts, all from resolved history: the share of past
drawdowns that regained the prior high within a year, the median time taken, and
the median excess over the index in the year after the fall.  A name needs at
least two resolved episodes before it can be scored at all.

The prediction, registered before the run: forward excess after a new drawdown
rises with the recovery score.  If it does not, past recovery does not persist
and the rating is describing history rather than character.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

HORIZONS = (252, 504, 756)
LABELS = {252: "1y", 504: "2y", 756: "3y"}
LOOKBACK = 252
TRIGGER = 0.15
RESOLVE_WINDOW = 252


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path,
                        default=Path("data/13f/alphavantage_daily.db"))
    parser.add_argument("--market", type=Path,
                        default=Path("data/cross_assets/etf_daily.db"))
    parser.add_argument("--market-ticker", default="DIA")
    parser.add_argument("--min-episodes", type=int, default=2)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/recovery_score.json"))
    known, _ = parser.parse_known_args(argv)
    return known


def load(path, keep=None):
    splits = load_splits(path)
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    book = {}
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency='daily' "
            "ORDER BY ticker"):
        if keep and ticker not in keep:
            continue
        rows = connection.execute(
            "SELECT ts,open,high,low,close FROM bars WHERE ticker=? AND "
            "frequency='daily' ORDER BY ts", (ticker,)).fetchall()
        bars = adjust_bars([Bar(r[0][:10], r[1], r[2], r[3], r[4], None)
                            for r in rows], splits.get(ticker, []))
        if len(bars) >= 1000:
            book[ticker] = bars
    connection.close()
    return book


def episodes(bars, market):
    """Every drawdown crossing, with how it resolved. Resolution is dated."""
    days = [b.timestamp for b in bars]
    closes = [b.close for b in bars]
    peak, running = [], []
    for i in range(len(bars)):
        running.append(closes[i])
        if len(running) > LOOKBACK:
            running.pop(0)
        peak.append(max(running))
    out, armed = [], True
    for i in range(LOOKBACK, len(bars) - 1):
        if peak[i] <= 0:
            continue
        drop = closes[i] / peak[i] - 1.0
        if drop > -TRIGGER * 0.5:
            armed = True
        if not armed or drop > -TRIGGER:
            continue
        armed = False
        target = peak[i]
        end = min(i + RESOLVE_WINDOW, len(bars) - 1)
        recovered, took = False, None
        for j in range(i + 1, end + 1):
            if closes[j] >= target:
                recovered, took = True, j - i
                break
        forward = None
        if i + RESOLVE_WINDOW < len(bars):
            own = closes[i + RESOLVE_WINDOW] / closes[i] - 1.0
            reference = market.get(days[i]), market.get(days[min(
                i + RESOLVE_WINDOW, len(days) - 1)])
            if all(reference) and reference[0] > 0:
                forward = own - (reference[1] / reference[0] - 1.0)
        out.append({"index": i, "day": days[i], "depth": -drop,
                    "recovered": recovered, "took": took, "excess": forward,
                    "resolved_on": days[min(i + RESOLVE_WINDOW,
                                            len(bars) - 1)]})
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    book = load(args.bars)
    market_book = load(args.market, keep={args.market_ticker})
    market = {b.timestamp: b.close
              for b in market_book.get(args.market_ticker, [])}
    print(f"{len(book)} names, market proxy {args.market_ticker} with "
          f"{len(market):,d} sessions")

    history = {t: episodes(bars, market) for t, bars in book.items()}
    total = sum(len(v) for v in history.values())
    print(f"{total:,d} drawdown episodes of {TRIGGER:.0%} or more\n")

    prices = {t: {b.timestamp: b.close for b in bars} for t, bars in book.items()}
    ordered = {t: sorted(p) for t, p in prices.items()}
    market_days = sorted(market)

    rows = []
    for ticker, events in history.items():
        days = ordered[ticker]
        for event in events:
            # score using only episodes that had already resolved by this day
            prior = [e for e in events if e["resolved_on"] < event["day"]
                     and e["excess"] is not None]
            if len(prior) < args.min_episodes:
                continue
            rate = sum(1 for e in prior if e["recovered"]) / len(prior)
            speed = statistics.median([e["took"] for e in prior if e["took"]]
                                      or [RESOLVE_WINDOW])
            edge = statistics.median(e["excess"] for e in prior)
            spot = event["index"]
            entry = prices[ticker][days[spot]]
            row = {"ticker": ticker, "day": event["day"], "depth": event["depth"],
                   "episodes": len(prior), "rate": rate,
                   "speed": -speed, "edge": edge}
            ok = False
            for horizon in HORIZONS:
                ahead = spot + horizon
                if ahead >= len(days):
                    row[f"xs_{horizon}"] = None
                    continue
                own = prices[ticker][days[ahead]] / entry - 1.0
                a = bisect_right(market_days, event["day"]) - 1
                b = a + horizon
                if a < 0 or b >= len(market_days):
                    row[f"xs_{horizon}"] = None
                    continue
                index_move = (market[market_days[b]]
                              / market[market_days[a]] - 1.0)
                row[f"xs_{horizon}"] = own - index_move
                row[f"own_{horizon}"] = own
                row[f"mkt_{horizon}"] = index_move
                ok = True
            if ok:
                rows.append(row)
    rows.sort(key=lambda r: r["day"])
    print(f"{len(rows):,d} scoreable drawdowns "
          f"({len({r['ticker'] for r in rows})} names, "
          f"{rows[0]['day']} to {rows[-1]['day']})")

    # rank the three components within each month, then average
    by_month = defaultdict(list)
    for r in rows:
        by_month[r["day"][:7]].append(r)
    scored = []
    for group in by_month.values():
        if len(group) < 6:
            continue
        for key in ("rate", "speed", "edge"):
            values = sorted(g[key] for g in group)
            for g in group:
                g[f"{key}_r"] = ((bisect_right(values, g[key]) - 1)
                                 / max(len(values) - 1, 1))
        for g in group:
            g["score"] = (g["rate_r"] + g["speed_r"] + g["edge_r"]) / 3
            scored.append(g)
    print(f"{len(scored):,d} ranked against at least five peers the same month\n")
    report = {"episodes": total, "scoreable": len(rows), "ranked": len(scored)}

    def summarise(chunk, horizon):
        values = [r[f"xs_{horizon}"] for r in chunk
                  if r.get(f"xs_{horizon}") is not None]
        if len(values) < 30:
            return None
        return {"n": len(values), "median": statistics.median(values),
                "beat": sum(1 for v in values if v > 0) / len(values)}

    print("########## forward excess over the index, by recovery score ##########")
    print("  Score is built only from drawdowns that had already resolved.")
    print(f"  {'recovery score':22s} {'n':>7s} " +
          " ".join(f"{'med ' + LABELS[h]:>10s}" for h in HORIZONS) +
          f" {'beat 3y':>9s}")
    ranked = sorted(scored, key=lambda r: r["score"])
    fifth = len(ranked) // 5
    medians = []
    report["quintiles"] = {}
    for q in range(5):
        chunk = ranked[q * fifth:(q + 1) * fifth]
        cells, three = [], None
        for horizon in HORIZONS:
            got = summarise(chunk, horizon)
            cells.append(f"{got['median']:>+10.1%}" if got else f"{'-':>10s}")
            if horizon == 756:
                three = got
        name = f"Q{q+1}" + (" worst record" if q == 0
                            else " best record" if q == 4 else "")
        if three:
            medians.append(three["median"])
            report["quintiles"][f"Q{q+1}"] = three
            print(f"  {name:22s} {three['n']:>7,d} " + " ".join(cells) +
                  f" {three['beat']:>9.0%}")
    if len(medians) >= 4:
        steps = sum(1 for a, b in zip(medians, medians[1:]) if b > a)
        print(f"  -> rises across {steps} of {len(medians)-1} steps")
        report["monotone"] = f"{steps}/{len(medians)-1}"

    print()
    print("########## the components separately ##########")
    print(f"  {'component, top vs bottom half':30s} {'n':>7s} {'med 3y':>10s} "
          f"{'beat 3y':>9s}")
    report["components"] = {}
    for key, text in (("rate_r", "past recovery rate"),
                      ("speed_r", "past recovery speed"),
                      ("edge_r", "past excess over index")):
        for half, name in ((lambda r: r[key] >= 0.5, "high"),
                           (lambda r: r[key] < 0.5, "low")):
            chunk = [r for r in scored if half(r)]
            got = summarise(chunk, 756)
            if got:
                report["components"][f"{text} {name}"] = got
                print(f"  {text + ' ' + name:30s} {got['n']:>7,d} "
                      f"{got['median']:>+10.1%} {got['beat']:>9.0%}")

    print()
    print("########## best record, by how far it has fallen ##########")
    best = ranked[3 * fifth:]
    print(f"  {'depth at entry':22s} {'n':>7s} " +
          " ".join(f"{'med ' + LABELS[h]:>10s}" for h in HORIZONS) +
          f" {'beat 3y':>9s}")
    report["best_by_depth"] = {}
    for low, high in ((0.15, 0.25), (0.25, 0.40), (0.40, 1.01)):
        chunk = [r for r in best if low <= r["depth"] < high]
        cells, three = [], None
        for horizon in HORIZONS:
            got = summarise(chunk, horizon)
            cells.append(f"{got['median']:>+10.1%}" if got else f"{'-':>10s}")
            if horizon == 756:
                three = got
        if three:
            label = f"{low:.0%} to {high:.0%}" if high < 1 else f"{low:.0%}+"
            report["best_by_depth"][label] = three
            print(f"  {label:22s} {three['n']:>7,d} " + " ".join(cells) +
                  f" {three['beat']:>9.0%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print()
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
