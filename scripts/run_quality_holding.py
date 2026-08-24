"""Quality, bought in drawdowns, held for years -- the test the others were not.

Every earlier test in this thread measured a rebound over twenty to sixty
sessions, and over that horizon the accounting variables said nothing while
volatility and market fear said a great deal.  That is a real finding about
rebounds and a worthless one about investing, because it is not the horizon a
quality-and-value approach operates on.  Nobody buying a good business at a
discount expects to be judged in two months.

So this changes the horizon and the benchmark together.

*The horizon* is one, two and three years.  Fundamentals here begin in 2009, so
a three-year forward return needs an entry no later than 2023 -- which is the
binding constraint on the whole exercise and the reason the sample is what it is.

*The benchmark* is no longer a random-entry drift null.  For a multi-year holding
strategy the honest comparison is the thing you would otherwise have done: own
the whole universe, equally weighted, over exactly the same window.  Every return
below is therefore reported as an **excess over the universe's own return across
the identical dates**, which removes the market and leaves only selection.

*The prediction*, stated before reading the answer: excess return should rise
monotonically across quality quintiles, and the effect should be larger when the
purchase is made during a drawdown or while the market is frightened.  A single
quintile beating the rest is what a search finds in noise; an ordering across
five is a prediction that can fail cleanly.

Quality is scored from what the ten SEC concepts allow -- return on equity,
operating margin, the consistency of profitability, balance-sheet leverage and
share-count growth -- ranked cross-sectionally each month, because a raw ROE
threshold means different things in 2010 and 2021.  Free cash flow, interest
coverage and ROIC are absent from the store and cannot be part of the score.
Facts stay unreadable for ninety days after their period ends.
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
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_quality_value_drawdown as base  # noqa: E402

HORIZONS = (252, 504, 756)
LABELS = {252: "1 year", 504: "2 years", 756: "3 years"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro", type=Path, default=Path("data/macro/macro.db"))
    parser.add_argument("--last-entry", default="2023-08-01")
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/quality_holding.json"))
    known, _ = parser.parse_known_args(argv)
    return known


def quality_score(entry):
    """Higher is better. Built only from concepts the store actually carries."""
    assets = entry.get("Assets")
    liabilities = entry.get("Liabilities")
    equity = entry.get("StockholdersEquity")
    income = entry.get("NetIncomeLoss")
    revenue = entry.get("Revenues")
    operating = entry.get("OperatingIncomeLoss")
    shares = entry.get("WeightedAverageNumberOfDilutedSharesOutstanding")
    before = entry.get("shares_year_ago")
    out = {}
    if equity and equity > 0 and income is not None:
        out["roe"] = income / equity
    if revenue and revenue > 0 and operating is not None:
        out["margin"] = operating / revenue
    if assets and assets > 0 and liabilities is not None:
        out["solvency"] = 1.0 - liabilities / assets
    if shares and before and before > 0:
        out["no_dilution"] = -(shares / before - 1.0)
    out["profitable"] = 1.0 if (income is not None and income > 0) else 0.0
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    inner = base.parse_args([])
    mapping = json.loads(inner.map.read_text())
    book = base.load_book(inner)
    mapping = {t: c for t, c in mapping.items() if t in book}
    panel = base.fundamentals(inner, mapping)

    vix = {}
    if args.macro.exists():
        connection = sqlite3.connect(f"file:{args.macro}?mode=ro", uri=True)
        for day, value in connection.execute(
                "SELECT obs_date, value FROM observations WHERE "
                "series_id='VIXCLS' AND value IS NOT NULL"):
            vix[day[:10]] = float(value)
        connection.close()

    prices = {t: {b.timestamp: b.close for b in bars} for t, bars in book.items()}
    ordered = {t: sorted(p) for t, p in prices.items()}
    calendar = sorted({d for days in ordered.values() for d in days})
    universe = {}
    for day in calendar:
        values = [prices[t][day] for t in prices if day in prices[t]]
        if len(values) >= 40:
            universe[day] = statistics.fmean(values)
    market_days = sorted(universe)

    # one observation per name per month
    months, seen = [], set()
    for day in calendar:
        if day[:7] not in seen and "2010-01" <= day[:7]:
            seen.add(day[:7])
            months.append(day)
    months = [m for m in months if m <= args.last_entry]

    rows = []
    for day in months:
        position = bisect_right(market_days, day) - 1
        if position < 0:
            continue
        peak_window = market_days[max(0, position - 200):position + 1]
        market_dd = universe[day] / max(universe[d] for d in peak_window) - 1.0
        here = []
        for ticker, facts in panel.items():
            days = ordered.get(ticker, [])
            spot = bisect_right(days, day) - 1
            if spot < 0 or spot + max(HORIZONS) >= len(days):
                continue
            usable = [f for f in facts if f[0] <= day]
            if not usable:
                continue
            score = quality_score(usable[-1][1])
            if len(score) < 4:
                continue
            entry_price = prices[ticker][days[spot]]
            if entry_price <= 0:
                continue
            window = days[max(0, spot - 200):spot + 1]
            drawdown = entry_price / max(prices[ticker][d] for d in window) - 1.0
            row = {"ticker": ticker, "day": day, "drawdown": drawdown,
                   "vix": vix.get(day), "market_dd": market_dd, **score}
            ok = True
            for horizon in HORIZONS:
                row[f"fwd_{horizon}"] = prices[ticker][days[spot + horizon]] / entry_price - 1.0
                ahead = bisect_right(market_days, day) - 1 + horizon
                if ahead >= len(market_days):
                    ok = False
                    break
                row[f"mkt_{horizon}"] = (universe[market_days[ahead]]
                                         / universe[day] - 1.0)
            if ok:
                here.append(row)
        if len(here) < 30:
            continue
        # rank each component within the month, then average the ranks
        for key in ("roe", "margin", "solvency", "no_dilution"):
            values = sorted(r[key] for r in here if key in r)
            for r in here:
                if key in r:
                    r[f"{key}_rank"] = (bisect_right(values, r[key]) - 1) / max(len(values) - 1, 1)
        scored = []
        for r in here:
            parts = [r[f"{k}_rank"] for k in ("roe", "margin", "solvency", "no_dilution")
                     if f"{k}_rank" in r]
            if len(parts) == 4:
                r["quality"] = statistics.fmean(parts) * (0.5 + 0.5 * r["profitable"])
                scored.append(r)
        # Excess is measured against an equal-weight hold of the SAME names in the
        # SAME month. Benchmarking against the wider index instead measured which
        # names survived the filters, which is survivorship rather than selection.
        for horizon in HORIZONS:
            average = statistics.fmean(r[f"fwd_{horizon}"] for r in scored)
            for r in scored:
                r[f"xs_{horizon}"] = r[f"fwd_{horizon}"] - average
        rows.extend(scored)
    print(f"{len(rows):,d} name-months, {len({r['day'] for r in rows})} months, "
          f"{len({r['ticker'] for r in rows})} names")
    print(f"entries {rows[0]['day']} to {rows[-1]['day']}, "
          f"forward windows to {HORIZONS[-1]} sessions\n")
    report = {"observations": len(rows)}

    def excess(chunk, horizon):
        return [r[f"xs_{horizon}"] for r in chunk]

    print("########## the prediction: does quality order the excess return? ##########")
    print("  Excess over an equal-weight hold of the same universe, same dates.")
    print(f"  {'quality quintile':20s} {'n':>7s} " +
          " ".join(f"{'mean ' + LABELS[h]:>12s}" for h in HORIZONS) +
          f" {'median 3y':>10s} {'win 3y':>8s}")
    ranked = sorted(rows, key=lambda r: r["quality"])
    fifth = len(ranked) // 5
    table, means_3y = {}, []
    for q in range(5):
        chunk = ranked[q * fifth:(q + 1) * fifth]
        means = [statistics.fmean(excess(chunk, h)) for h in HORIZONS]
        median_3y = statistics.median(excess(chunk, 756))
        wins = sum(1 for x in excess(chunk, 756) if x > 0) / len(chunk)
        means_3y.append(median_3y)
        table[f"Q{q+1}"] = {"n": len(chunk),
                            **{LABELS[h]: m for h, m in zip(HORIZONS, means)},
                            "median_3y": median_3y, "win_3y": wins}
        name = f"Q{q+1}" + (" (worst)" if q == 0 else " (best)" if q == 4 else "")
        print(f"  {name:20s} {len(chunk):>7,d} " +
              " ".join(f"{m:>+12.2%}" for m in means) +
              f" {median_3y:>+10.2%} {wins:>8.1%}")
    steps = sum(1 for a, b in zip(means_3y, means_3y[1:]) if b > a)
    print(f"\n  rises across {steps} of 4 quintile steps at 3 years -> "
          f"{'MONOTONE' if steps == 4 else 'not monotone'}")
    report["quintiles"] = table
    report["monotone_steps"] = f"{steps}/4"

    print("\n########## does buying it cheaper help? ##########")
    print("  Top-quintile quality only, split by how far the name had fallen.")
    best = ranked[4 * fifth:]
    print(f"  {'entry condition':24s} {'n':>7s} " +
          " ".join(f"{LABELS[h]:>12s}" for h in HORIZONS))
    cuts = (("no drawdown (0 to -5%)", lambda r: r["drawdown"] > -0.05),
            ("down 5-15%", lambda r: -0.15 < r["drawdown"] <= -0.05),
            ("down 15-30%", lambda r: -0.30 < r["drawdown"] <= -0.15),
            ("down more than 30%", lambda r: r["drawdown"] <= -0.30))
    report["by_drawdown"] = {}
    for label, test in cuts:
        chunk = [r for r in best if test(r)]
        if len(chunk) < 60:
            continue
        means = [statistics.fmean(excess(chunk, h)) for h in HORIZONS]
        report["by_drawdown"][label] = {"n": len(chunk),
                                        **{LABELS[h]: m for h, m in zip(HORIZONS, means)}}
        print(f"  {label:24s} {len(chunk):>7,d} " +
              " ".join(f"{m:>+12.2%}" for m in means))

    print("\n########## does buying it frightened help? ##########")
    print("  Top-quintile quality only, split by VIX on the entry day.")
    graded = [r for r in best if r["vix"] is not None]
    if graded:
        levels = sorted(r["vix"] for r in graded)
        low, high = levels[len(levels) // 3], levels[2 * len(levels) // 3]
        report["by_vix"] = {}
        print(f"  {'entry condition':24s} {'n':>7s} " +
              " ".join(f"{LABELS[h]:>12s}" for h in HORIZONS))
        for label, test in ((f"calm (VIX <= {low:.0f})", lambda v: v <= low),
                            ("normal", lambda v: low < v <= high),
                            (f"fear (VIX > {high:.0f})", lambda v: v > high)):
            chunk = [r for r in graded if test(r["vix"])]
            if len(chunk) < 60:
                continue
            means = [statistics.fmean(excess(chunk, h)) for h in HORIZONS]
            report["by_vix"][label] = {"n": len(chunk),
                                       **{LABELS[h]: m for h, m in zip(HORIZONS, means)}}
            print(f"  {label:24s} {len(chunk):>7,d} " +
                  " ".join(f"{m:>+12.2%}" for m in means))

    print("\n########## the combination ##########")
    print("  Top-quintile quality bought down more than 15%, against everything else.")
    combo = [r for r in best if r["drawdown"] <= -0.15]
    rest = [r for r in rows if not (r in best and r["drawdown"] <= -0.15)]
    print(f"  {'arm':30s} {'n':>7s} " +
          " ".join(f"{LABELS[h]:>12s}" for h in HORIZONS))
    for label, chunk in (("quality + down 15%+", combo),
                         ("everything else", rest),
                         ("the whole cross-section", rows)):
        if len(chunk) < 60:
            continue
        means = [statistics.fmean(excess(chunk, h)) for h in HORIZONS]
        report.setdefault("combination", {})[label] = {
            "n": len(chunk), **{LABELS[h]: m for h, m in zip(HORIZONS, means)}}
        print(f"  {label:30s} {len(chunk):>7,d} " +
              " ".join(f"{m:>+12.2%}" for m in means))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
