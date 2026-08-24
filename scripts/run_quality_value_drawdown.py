"""Margin of safety with the gates attached: does quality rescue the drawdown?

The plain version of this failed.  Buying a discount to the rolling high scored
2.64 out of sample against a drift null of 2.38 -- p = 0.37 -- and, worse, the
effect ran the wrong way: entries 15-25% below the high returned -0.121 against
an unconditional +1.215, while shallow 5-15% pullbacks beat the baseline.  The
sweep, handed depths from 10% to 30%, chose the shallowest available.  Deeper was
worse, which is the value trap stated as a measurement.

The proposed fix is that price drawdown is only an opportunity generator, and
what makes a discount safe is the business behind it.  That is a real prediction
and this tests it: the same trigger, with survival, quality and valuation gates
in front of it.

What the gates can and cannot be, honestly.  The SEC company-facts store here
carries ten concepts.  Free cash flow, EBITDA, net debt, interest coverage and
analyst revisions are all absent, so the framework's specific thresholds -- net
debt/EBITDA under 2.5x, coverage over 6x -- cannot be tested.  What can:

* **survival** as total liabilities over assets, plus share-count growth, which
  is the dilution the framework wants to avoid;
* **quality** as return on equity and operating margin, with a hard requirement
  that the company is actually profitable;
* **value** as book and earnings yield against the cross-section, which is a
  relative-value proxy and emphatically not a discount to appraised fair value.

Facts carry a period end but no filing date, so every fact is held unusable
until **90 days after its period ends**.  A December quarter is not readable
until April.  That is conservative and it is the difference between a backtest
and a fantasy.

Four arms, and the third is the one that decides it.

*Drawdown alone* -- the version that already failed, carried forward so the
comparison is paired rather than remembered.

*Drawdown plus gates* -- the proposal.

*Drawdown plus the gates inverted* -- expensive, levered, unprofitable names in
the same drawdowns.  If buying bad businesses cheap does as well as buying good
ones, the gates are decoration.

*Gates without the drawdown* -- quality-value names bought on ordinary days.
This separates the discount from the screen: if it scores the same, the drawdown
trigger contributes nothing and the whole margin-of-safety framing is beside the
point.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import statistics
import sys
import zlib
from bisect import bisect_right
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

import run_vol_stretch_zones as shared  # noqa: E402

REPORT_LAG_DAYS = 90
FLOWS = ("NetIncomeLoss", "Revenues", "OperatingIncomeLoss")
INSTANTS = ("Assets", "Liabilities", "StockholdersEquity",
            "WeightedAverageNumberOfDilutedSharesOutstanding")
# Fixed, not swept. The gates pass roughly 5% of crossings by
# construction -- three independent median cuts -- so a sweep over
# twelve cells would be searching a forty-trade sample.
WINDOWS = (200,)
DEPTHS = (0.10,)
HOLDS = (60,)
RISK_RUNGS = (0.0025, 0.005, 0.01)
LEVERAGE_CAP = 5.0


def ticker_seed(t):
    return zlib.crc32(t.encode("utf-8")) % 10_000


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path,
                        default=Path("data/13f/alphavantage_daily.db"))
    parser.add_argument("--equity", type=Path, default=Path("data/equity/equity.db"))
    parser.add_argument("--map", type=Path,
                        default=Path("out/strategy/ticker_cik.json"))
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--split", default="2016-01-01")
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--stop-mult", type=float, default=3.0)
    parser.add_argument("--taker-bp", type=float, default=5.0)
    parser.add_argument("--max-positions", type=int, default=12)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--null-draws", type=int, default=30)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/quality_value_drawdown.json"))
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------


def load_book(args):
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    book = {}
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency='daily' "
            "ORDER BY ticker"):
        rows = connection.execute(
            "SELECT ts,open,high,low,close FROM bars WHERE ticker=? AND "
            "frequency='daily' AND ts>=? ORDER BY ts",
            (ticker, "2008-01-01")).fetchall()
        bars = adjust_bars([Bar(r[0][:10], r[1], r[2], r[3], r[4], None)
                            for r in rows], splits.get(ticker, []))
        if len(bars) >= 500:
            book[ticker] = bars
    connection.close()
    return book


def fundamentals(args, mapping):
    """Point-in-time facts per ticker: (usable_from, metric dict), oldest first.

    Flows are summed over the trailing four quarters; instants take the latest
    reading.  Nothing becomes readable until ``REPORT_LAG_DAYS`` after its period
    closes, because the store carries no filing date.
    """
    connection = sqlite3.connect(f"file:{args.equity}?mode=ro", uri=True)
    by_cik = defaultdict(lambda: defaultdict(dict))
    wanted = set(FLOWS) | set(INSTANTS)
    ciks = tuple(mapping.values())
    marks = ",".join("?" * len(ciks))
    for cik, concept, start, end, value in connection.execute(
            f"SELECT cik, concept, period_start, period_end, value FROM sec_facts "
            f"WHERE cik IN ({marks}) AND value IS NOT NULL", ciks):
        if concept not in wanted or not end:
            continue
        if concept in FLOWS:
            if not start:
                continue
            span = (date.fromisoformat(end[:10]) - date.fromisoformat(start[:10])).days
            if not 60 <= span <= 100:          # quarterly only, so sums are clean
                continue
        by_cik[cik][concept][end[:10]] = float(value)
    connection.close()

    panel = {}
    for ticker, cik in mapping.items():
        facts = by_cik.get(cik)
        if not facts:
            continue
        ends = sorted({d for series in facts.values() for d in series})
        rows = []
        for end in ends:
            entry = {}
            for concept in FLOWS:
                series = facts.get(concept, {})
                quarters = [series[d] for d in sorted(series) if d <= end][-4:]
                if len(quarters) == 4:
                    entry[concept] = sum(quarters)
            for concept in INSTANTS:
                series = facts.get(concept, {})
                prior = [series[d] for d in sorted(series) if d <= end]
                if prior:
                    entry[concept] = prior[-1]
                older = [series[d] for d in sorted(series)
                         if d <= end and (date.fromisoformat(end)
                                          - date.fromisoformat(d)).days >= 330]
                if concept.startswith("Weighted") and older:
                    entry["shares_year_ago"] = older[-1]
            usable = (date.fromisoformat(end)
                      + timedelta(days=REPORT_LAG_DAYS)).isoformat()
            rows.append((usable, entry))
        if rows:
            panel[ticker] = rows
    return panel


def metrics(entry, price):
    """The gates, from what the ten concepts allow."""
    assets = entry.get("Assets")
    liabilities = entry.get("Liabilities")
    equity = entry.get("StockholdersEquity")
    income = entry.get("NetIncomeLoss")
    revenue = entry.get("Revenues")
    operating = entry.get("OperatingIncomeLoss")
    shares = entry.get("WeightedAverageNumberOfDilutedSharesOutstanding")
    before = entry.get("shares_year_ago")
    out = {}
    if assets and assets > 0 and liabilities is not None:
        out["leverage"] = liabilities / assets
    if equity and equity > 0 and income is not None:
        out["roe"] = income / equity
    if revenue and revenue > 0 and operating is not None:
        out["op_margin"] = operating / revenue
    if shares and before and before > 0:
        out["dilution"] = shares / before - 1.0
    if shares and shares > 0 and price > 0:
        cap = shares * price
        if equity is not None:
            out["book_yield"] = equity / cap
        if income is not None:
            out["earn_yield"] = income / cap
    out["profitable"] = 1.0 if (income is not None and income > 0) else 0.0
    return out


def build_events(book, panel, args):
    """Drawdown crossings and ordinary days, each carrying its readable facts."""
    triggers, ordinary = defaultdict(list), []
    for ticker, bars in book.items():
        rows = panel.get(ticker)
        if not rows:
            continue
        usable_dates = [r[0] for r in rows]
        closes = [b.close for b in bars]
        opens = [b.open for b in bars]
        days = [b.timestamp for b in bars]
        returns, sigma = [], {}
        for i in range(1, len(bars)):
            a, b = closes[i - 1], closes[i]
            if a > 0 and b > 0:
                returns.append(math.log(b / a))
            if len(returns) > args.vol_window:
                returns.pop(0)
            if len(returns) == args.vol_window:
                sigma[days[i]] = statistics.pstdev(returns)

        def facts_at(day, price):
            position = bisect_right(usable_dates, day) - 1
            if position < 0:
                return None
            return metrics(rows[position][1], price)

        for window in WINDOWS:
            running = []
            peak = []
            for i in range(len(bars)):
                running.append(closes[i])
                if len(running) > window:
                    running.pop(0)
                peak.append(max(running))
            for depth in DEPTHS:
                armed = True
                for i in range(max(window, args.vol_window) + 1, len(bars) - 70):
                    if peak[i] <= 0 or days[i] < args.start:
                        continue
                    drop = closes[i] / peak[i] - 1.0
                    if drop > -depth * 0.5:
                        armed = True
                    if not armed or drop > -depth:
                        continue
                    armed = False
                    s, entry = sigma.get(days[i]), opens[i + 1]
                    if not s or s <= 0 or entry <= 0:
                        continue
                    got = facts_at(days[i], closes[i])
                    if not got:
                        continue
                    triggers[(window, depth)].append(
                        {"ticker": ticker, "index": i, "day": days[i],
                         "sigma": s, "depth": -drop, **got})
        for i in range(args.vol_window + 2, len(bars) - 70):
            if (i * 2654435761) % 100 >= 8 or days[i] < args.start:
                continue
            s, entry = sigma.get(days[i]), opens[i + 1]
            if not s or s <= 0 or entry <= 0:
                continue
            got = facts_at(days[i], closes[i])
            if got:
                ordinary.append({"ticker": ticker, "index": i, "day": days[i],
                                 "sigma": s, "depth": 0.0, **got})
    return triggers, ordinary


def universe_medians(book, panel, args):
    """Monthly cross-sectional medians over the WHOLE universe.

    The comparison has to be against every name that existed that month, not
    against the handful that happen to be crossing a drawdown threshold on the
    same day -- there are usually one or two of those, which is far too few to
    take a median of and was silently discarding almost every candidate.
    """
    prices = {t: {b.timestamp: b.close for b in bars} for t, bars in book.items()}
    calendar = sorted({b.timestamp for bars in book.values() for b in bars
                       if b.timestamp >= args.start})
    months, seen = [], set()
    for day in calendar:
        if day[:7] not in seen:
            seen.add(day[:7])
            months.append(day)
    table = {}
    for day in months:
        rows = []
        for ticker, facts in panel.items():
            usable = [f for f in facts if f[0] <= day]
            price = prices.get(ticker, {}).get(day)
            if not usable or not price:
                continue
            got = metrics(usable[-1][1], price)
            if {"leverage", "roe", "book_yield"} <= got.keys():
                rows.append(got)
        if len(rows) >= 20:
            table[day] = {
                key: statistics.median([r[key] for r in rows])
                for key in ("leverage", "roe", "book_yield")}
    print(f"  universe medians on {len(table)} month-ends "
          f"({months[0][:7]} to {months[-1][:7]})")
    return sorted(table), table


def cross_sectional_gates(rows, month_days, medians, invert=False, need=2):
    """Survival, quality and value, each judged against the universe that month."""
    kept = []
    for g in rows:
        if not {"leverage", "roe", "book_yield"} <= g.keys():
            continue
        position = bisect_right(month_days, g["day"]) - 1
        if position < 0:
            continue
        mid = medians[month_days[position]]
        survives = g["leverage"] <= mid["leverage"] and g.get("dilution", 0.0) < 0.10
        quality = g["profitable"] > 0 and g["roe"] >= mid["roe"]
        cheap = g["book_yield"] >= mid["book_yield"]
        score = int(survives) + int(quality) + int(cheap)
        # Requiring all three at once passes ~5% of crossings -- three
        # independent median cuts -- which leaves too few trades to assess.
        # The framework's own candidate model scores rather than vetoes, so
        # ``need`` is the number of gates demanded and both are reported.
        passes = score >= need if not invert else score <= (3 - need)
        if passes:
            kept.append(g)
    kept.sort(key=lambda r: r["day"])
    return kept


def trade(book, rows, hold, args):
    out = []
    for row in rows:
        bars = book[row["ticker"]]
        start = row["index"] + 1
        fill = bars[start].open
        risk = args.stop_mult * row["sigma"] * fill
        if fill <= 0 or risk <= 0:
            continue
        stop = fill - risk
        last = min(start + hold, len(bars) - 1)
        price, reason, when = bars[last].close, "time", bars[last].timestamp
        for i in range(start, last + 1):
            if bars[i].low <= stop:
                price, reason, when = stop, "stop", bars[i].timestamp
                break
        cost = 2 * args.taker_bp / 10_000 * fill / risk
        out.append({"ticker": row["ticker"], "entry": row["day"], "exit": when,
                    "r": (price - fill) / risk - cost, "reason": reason,
                    "stop_pct": risk / fill})
    out.sort(key=lambda t: t["entry"])
    return out


def compound(taken, fraction):
    per_day, levers = defaultdict(float), []
    for t in taken:
        lever = min(fraction / t["stop_pct"], LEVERAGE_CAP)
        levers.append(lever)
        per_day[t["exit"]] += t["r"] * lever * t["stop_pct"]
    days = sorted(per_day)
    if not days:
        return None
    nav, peak, worst = 1000.0, 1000.0, 0.0
    for d in days:
        nav = max(0.0, nav + per_day[d] * nav)
        peak = max(peak, nav)
        worst = min(worst, nav / peak - 1.0)
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25
    return {"cagr": (nav / 1000.0) ** (1 / years) - 1 if nav > 0 else -1.0,
            "max_drawdown": worst, "median_leverage": statistics.median(levers)}


def main(argv=None) -> int:
    args = parse_args(argv)
    mapping = json.loads(args.map.read_text())
    book = load_book(args)
    mapping = {t: c for t, c in mapping.items() if t in book}
    panel = fundamentals(args, mapping)
    print(f"{len(book)} names with bars, {len(mapping)} with a CIK, "
          f"{len(panel)} with usable facts")
    print(f"facts held unreadable for {REPORT_LAG_DAYS} days after period end")
    triggers, ordinary = build_events(book, panel, args)
    month_days, medians = universe_medians(book, panel, args)
    print(f"{sum(len(v) for v in triggers.values()):,d} drawdown crossings, "
          f"{len(ordinary):,d} ordinary-day samples")
    print(f"in sample to {args.split}, out of sample after\n")
    report = {"names": len(panel)}

    print("########## the frozen specification ##########")
    best, best_score, cache = None, -99.0, {}
    for (window, depth), rows in triggers.items():
        gated = cross_sectional_gates(rows, month_days, medians)
        for hold in HOLDS:
            trades = trade(book, gated, hold, args)
            early = [t for t in trades if t["entry"] < args.split]
            if len(early) < 30:
                continue
            result = shared.assess(early, args)
            if result:
                cache[(window, depth, hold)] = (rows, gated, trades)
                if result["sharpe"] > best_score:
                    best_score, best = result["sharpe"], (window, depth, hold)
    if best is None:
        print("  nothing cleared the in-sample minimum")
        return 1
    print(f"  {'window':>7s} {'depth':>6s} {'hold':>5s} {'gated IS':>9s} {'Sharpe':>7s}")
    for key in sorted(cache, key=lambda k: -shared.assess(
            [t for t in cache[k][2] if t["entry"] < args.split], args)["sharpe"])[:6]:
        early = [t for t in cache[key][2] if t["entry"] < args.split]
        mark = " <- chosen" if key == best else ""
        print(f"  {key[0]:>7d} {key[1]:>5.0%} {key[2]:>5d} {len(early):>9,d} "
              f"{shared.assess(early, args)['sharpe']:>7.2f}{mark}")
    window, depth, hold = best
    rows, gated, _ = cache[best]
    report["chosen"] = {"window": window, "depth": depth, "hold": hold,
                        "in_sample_sharpe": best_score}
    print(f"\n  frozen: {depth:.0%} below the {window}-session high, held {hold}\n")

    print("########## out of sample ##########")
    print(f"  {'arm':38s} {'trades':>8s} {'taken':>7s} {'Sharpe':>7s} {'[5-95%]':>14s}")

    def score(label, pooled):
        outside = [t for t in pooled if t["entry"] >= args.split]
        if len(outside) < 100:
            print(f"  {label:38s} {len(outside):>8,d}   too few")
            return None
        result = shared.assess(outside, args)
        if not result:
            return None
        report.setdefault("arms", {})[label] = result
        print(f"  {label:38s} {result['offered']:>8,d} {result['taken']:>7,d} "
              f"{result['sharpe']:>7.2f} "
              f"{('[%.2f-%.2f]' % (result['sharpe_p05'], result['sharpe_p95'])):>14s}",
              flush=True)
        return result

    score("drawdown alone, no gates", trade(book, rows, hold, args))
    gated_result = score("drawdown + gates (2 of 3)",
                         trade(book, gated, hold, args))
    strict = cross_sectional_gates(rows, month_days, medians, need=3)
    score("drawdown + gates (all 3, strict)", trade(book, strict, hold, args))
    score("drawdown + gates INVERTED (reversal)",
          trade(book, cross_sectional_gates(rows, month_days, medians,
                                            invert=True), hold, args))
    score("gates without the drawdown",
          trade(book, cross_sectional_gates(ordinary, month_days, medians),
                hold, args))

    if gated_result:
        nulls = []
        for draw in range(args.null_draws):
            rng = random.Random(41_000 + 137 * draw)
            picked = rng.sample(ordinary, min(len(gated), len(ordinary)))
            drawn = trade(book, sorted(picked, key=lambda r: r["day"]), hold, args)
            pooled = [t for t in drawn if t["entry"] >= args.split]
            if len(pooled) < 100:
                continue
            outcome = shared.assess(pooled, args)
            if outcome:
                nulls.append(outcome["sharpe"])
        if nulls:
            nulls.sort()
            above = sum(1 for x in nulls if x >= gated_result["sharpe"]) / len(nulls)
            report["drift_null"] = {"median": statistics.median(nulls),
                                    "low": nulls[0], "high": nulls[-1], "p": above}
            print(f"  {'random days (drift null)':38s} {'':>8s} {'':>7s} "
                  f"{statistics.median(nulls):>7.2f} "
                  f"{('[%.2f-%.2f]' % (nulls[0], nulls[-1])):>14s}")
            print(f"  -> p = {above:.2f}, "
                  f"{'clears' if above <= 0.05 else 'inside'} its null")

    print(f"\n########## compounding the gated book, capped at {LEVERAGE_CAP:g}x "
          f"##########")
    outside = [t for t in trade(book, gated, hold, args) if t["entry"] >= args.split]
    taken = shared.cap(outside, args.max_positions, random.Random(0))
    print(f"  {'risk':>6s} {'CAGR':>9s} {'max DD':>9s} {'median lev':>11s}")
    for fraction in RISK_RUNGS:
        row = compound(taken, fraction)
        if row:
            report.setdefault("compounding", []).append({"risk": fraction, **row})
            print(f"  {fraction:>5.2%} {row['cagr']:>+9.1%} "
                  f"{row['max_drawdown']:>9.1%} {row['median_leverage']:>10.2f}x")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
