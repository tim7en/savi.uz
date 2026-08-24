"""Does quality compound? Ranked across the whole universe, judged on the median.

An earlier version of this answered the wrong question twice over.  It measured
rebounds over twenty to sixty sessions, which is not the horizon a
quality approach operates on, and it benchmarked against an index containing
names the sample had already filtered out, which measured survivorship rather
than selection.  Both are fixed here.

*Every name, every horizon it can support.*  Requiring three years of forward
prices for every observation cut the universe to fifty names.  Each horizon now
uses whatever names have the data for it, so the one-year column is far wider
than the three-year one and both say so.  The quality score needs three of its
four components rather than all four.

*The benchmark is the same names in the same month.*  Excess is measured against
an equal-weight hold of exactly the cross-section being ranked, so the whole
sample sums to zero by construction and what remains is selection.

*The median is the headline, not the mean.*  Excess returns are violently
right-skewed: a bucket can carry a large positive mean while its typical member
loses badly, because a few multi-baggers drag the average up.  A strategy that
holds nine hundred names collects that mean.  A strategy that holds fifteen
collects something much closer to the median, so the median and the win rate are
the statistics that govern compounding and the mean is reported beside them as a
warning rather than a result.

*The prediction*: the median excess return rises with quality rank, and the win
rate rises with it.  Deciles as well as quintiles, because an ordering that holds
across ten buckets is far harder to produce by accident than one across five.

Quality is ranked cross-sectionally each month from return on equity, operating
margin, balance-sheet solvency and share-count growth, gated on actually being
profitable.  Free cash flow, interest coverage and ROIC are not in the store and
cannot be part of it.  Facts stay unreadable for ninety days after period end.
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
LABELS = {252: "1y", 504: "2y", 756: "3y"}
COMPONENTS = ("roe", "margin", "solvency", "no_dilution")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro", type=Path, default=Path("data/macro/macro.db"))
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/quality_holding.json"))
    known, _ = parser.parse_known_args(argv)
    return known


def quality_score(entry):
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


def summarise(chunk, horizon):
    values = [r[f"xs_{horizon}"] for r in chunk if r.get(f"xs_{horizon}") is not None]
    if len(values) < 40:
        return None
    return {"n": len(values), "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "win": sum(1 for v in values if v > 0) / len(values)}


def band(chunk, label, width=26):
    cells = []
    for horizon in HORIZONS:
        got = summarise(chunk, horizon)
        cells.append(f"{got['median']:>+9.1%}" if got else f"{'-':>9s}")
    got = summarise(chunk, 756) or summarise(chunk, 252)
    n = got["n"] if got else 0
    win = f"{got['win']:>7.0%}" if got else f"{'-':>7s}"
    print(f"  {label:{width}s} {n:>7,d} " + " ".join(cells) + f" {win}")
    return got


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
    months, seen = [], set()
    for day in calendar:
        if day[:7] not in seen and day >= "2010-01-01":
            seen.add(day[:7])
            months.append(day)

    rows = []
    for day in months:
        here = []
        for ticker, facts in panel.items():
            days = ordered.get(ticker, [])
            spot = bisect_right(days, day) - 1
            if spot < 0 or spot + min(HORIZONS) >= len(days):
                continue
            usable = [f for f in facts if f[0] <= day]
            if not usable:
                continue
            score = quality_score(usable[-1][1])
            if sum(1 for k in COMPONENTS if k in score) < 3:
                continue
            entry = prices[ticker][days[spot]]
            if entry <= 0:
                continue
            window = days[max(0, spot - 200):spot + 1]
            row = {"ticker": ticker, "day": day, "vix": vix.get(day),
                   "drawdown": entry / max(prices[ticker][d] for d in window) - 1.0,
                   **score}
            for horizon in HORIZONS:
                ahead = spot + horizon
                row[f"fwd_{horizon}"] = (prices[ticker][days[ahead]] / entry - 1.0
                                         if ahead < len(days) else None)
            here.append(row)
        if len(here) < 25:
            continue
        for key in COMPONENTS:
            values = sorted(r[key] for r in here if key in r)
            if len(values) < 10:
                continue
            for r in here:
                if key in r:
                    r[f"{key}_rank"] = ((bisect_right(values, r[key]) - 1)
                                        / max(len(values) - 1, 1))
        scored = []
        for r in here:
            parts = [r[f"{k}_rank"] for k in COMPONENTS if f"{k}_rank" in r]
            if len(parts) >= 3:
                r["quality"] = statistics.fmean(parts) * (0.5 + 0.5 * r["profitable"])
                scored.append(r)
        # Excess against the same names in the same month, per horizon.
        for horizon in HORIZONS:
            have = [r for r in scored if r[f"fwd_{horizon}"] is not None]
            if len(have) < 15:
                for r in scored:
                    r[f"xs_{horizon}"] = None
                continue
            average = statistics.fmean(r[f"fwd_{horizon}"] for r in have)
            for r in scored:
                r[f"xs_{horizon}"] = (r[f"fwd_{horizon}"] - average
                                      if r[f"fwd_{horizon}"] is not None else None)
        rows.extend(scored)

    names = {r["ticker"] for r in rows}
    print(f"{len(rows):,d} name-months, {len({r['day'] for r in rows})} months, "
          f"{len(names)} names, {rows[0]['day']} to {rows[-1]['day']}")
    for horizon in HORIZONS:
        got = sum(1 for r in rows if r.get(f"xs_{horizon}") is not None)
        print(f"  {LABELS[horizon]} forward window: {got:,d} observations, "
              f"{len({r['ticker'] for r in rows if r.get(f'xs_{horizon}') is not None})} names")
    report = {"observations": len(rows), "names": len(names)}

    header = (f"  {'bucket':26s} {'n':>7s} " +
              " ".join(f"{'med ' + LABELS[h]:>9s}" for h in HORIZONS) + f" {'win':>7s}")
    print("\n########## quality deciles: median excess over same-month peers "
          "##########")
    print(header)
    ranked = sorted(rows, key=lambda r: r["quality"])
    tenth = len(ranked) // 10
    medians = []
    report["deciles"] = {}
    for d in range(10):
        chunk = ranked[d * tenth:(d + 1) * tenth]
        label = f"D{d+1}" + (" lowest quality" if d == 0
                             else " highest quality" if d == 9 else "")
        got = band(chunk, label)
        if got:
            report["deciles"][f"D{d+1}"] = got
            three = summarise(chunk, 756)
            medians.append(three["median"] if three else None)
    clean = [m for m in medians if m is not None]
    steps = sum(1 for a, b in zip(clean, clean[1:]) if b > a)
    print(f"\n  median rises across {steps} of {len(clean)-1} decile steps at 3y")

    print("\n########## quality quintiles ##########")
    print(header)
    fifth = len(ranked) // 5
    report["quintiles"] = {}
    for q in range(5):
        chunk = ranked[q * fifth:(q + 1) * fifth]
        label = f"Q{q+1}" + (" (worst)" if q == 0 else " (best)" if q == 4 else "")
        got = band(chunk, label)
        if got:
            report["quintiles"][f"Q{q+1}"] = got

    best = ranked[4 * fifth:]
    print("\n########## top-quintile quality, by how far it had fallen ##########")
    print(header)
    report["by_drawdown"] = {}
    for label, test in (("no drawdown (0 to -5%)", lambda r: r["drawdown"] > -0.05),
                        ("down 5-15%", lambda r: -0.15 < r["drawdown"] <= -0.05),
                        ("down 15-30%", lambda r: -0.30 < r["drawdown"] <= -0.15),
                        ("down more than 30%", lambda r: r["drawdown"] <= -0.30)):
        chunk = [r for r in best if test(r)]
        got = band(chunk, label)
        if got:
            report["by_drawdown"][label] = got

    print("\n########## the combination ##########")
    print(header)
    report["combination"] = {}
    for label, chunk in (
            ("top quality, down 15%+", [r for r in best if r["drawdown"] <= -0.15]),
            ("top quality, any entry", best),
            ("bottom quality, down 15%+",
             [r for r in ranked[:fifth] if r["drawdown"] <= -0.15]),
            ("the whole cross-section", rows)):
        got = band(chunk, label)
        if got:
            report["combination"][label] = got

    print("\n########## which names sit in the top quintile most often ##########")
    counts = defaultdict(int)
    for r in best:
        counts[r["ticker"]] += 1
    total = defaultdict(int)
    for r in rows:
        total[r["ticker"]] += 1
    share = sorted(((counts[t] / total[t], t, total[t]) for t in total
                    if total[t] >= 40), reverse=True)
    print("  most often high quality: " +
          ", ".join(f"{t} {s:.0%}" for s, t, _ in share[:12]))
    print("  least often:             " +
          ", ".join(f"{t} {s:.0%}" for s, t, _ in share[-8:]))
    report["top_quintile_share"] = {t: s for s, t, _ in share[:25]}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
