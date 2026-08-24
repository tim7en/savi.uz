"""What actually predicts a post-drawdown rebound: fundamentals, or convexity?

The gated-drawdown test produced a result it could not support: names failing
the quality and value screens rebounded slightly better than names passing them,
2.59 against 2.11, with a null spanning [0.78, 3.57].  The difference was inside
one standard error, so nothing was established -- but it posed a sharp question,
and screening harder is not the way to answer it.

The hypothesis worth testing is that the inverted screen was never selecting bad
fundamentals at all.  Book yield and return on equity are correlated with beta
and idiosyncratic volatility, and a levered firm's equity behaves like a call
option on its assets, so it rebounds multiplicatively when the outlook improves.
If that is what happened, the screen was a volatility proxy wearing an accounting
costume, and the honest description of the effect is convexity rather than junk.

Two tests, and the second is the one that decides it.

*A regression on the whole cross-section.*  Forward return after a drawdown
crossing, against the depth itself, market beta, idiosyncratic volatility, the
three accounting variables, and the market state at entry.  Coefficients are
standardised so their magnitudes can be compared directly, and the interactions
the hypothesis predicts -- depth against beta, depth against idiosyncratic vol --
are fitted alongside.

*A matched-pairs comparison.*  Every event that fails the screens is paired with
an event that passes it, in the same month, at a similar drawdown depth, similar
beta and similar volatility.  Then the fundamentals are the only thing left
differing between them.  If the rebound gap survives that, the accounting
variables carry information.  If it collapses, they were proxying for
convexity all along, and the matched comparison says so cleanly.

Forward returns here are raw, not volatility-normalised.  Normalising would
divide out the very quantity under test.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_quality_value_drawdown as base  # noqa: E402

HORIZONS = (20, 60)
FEATURES = ("depth", "beta", "ivol", "book_yield", "roe", "leverage",
            "vix", "market_dd")
# The regime interactions matter as much as the drawdown ones: the claim under
# test is that fundamentals are priced in calm markets and ignored in fearful
# ones, which is a valuation-against-VIX term, not a valuation term.
INTERACTIONS = (("depth", "beta"), ("depth", "ivol"), ("depth", "market_dd"),
                ("book_yield", "vix"), ("roe", "vix"), ("leverage", "vix"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro", type=Path, default=Path("data/macro/macro.db"))
    parser.add_argument("--beta-weeks", type=int, default=52)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/rebound_anatomy.json"))
    known, _ = parser.parse_known_args(argv)
    return known


def weekly_returns(book):
    """Friday-to-Friday log returns per name, plus the equal-weight universe."""
    series = {}
    for ticker, bars in book.items():
        weekly, last = {}, None
        for bar in bars:
            iso = bar.timestamp
            year, week, _ = __import__("datetime").date.fromisoformat(iso).isocalendar()
            weekly[(year, week)] = bar.close
        keys = sorted(weekly)
        out = {}
        for a, b in zip(keys, keys[1:]):
            if weekly[a] > 0 and weekly[b] > 0:
                out[b] = math.log(weekly[b] / weekly[a])
        series[ticker] = out
    every = defaultdict(list)
    for out in series.values():
        for key, value in out.items():
            every[key].append(value)
    universe = {k: statistics.fmean(v) for k, v in every.items() if len(v) >= 20}
    return series, universe


def beta_panel(book, args):
    """Trailing beta and residual volatility, recomputed each week."""
    series, universe = weekly_returns(book)
    weeks = sorted(universe)
    index = {w: i for i, w in enumerate(weeks)}
    out = defaultdict(dict)
    for ticker, own in series.items():
        keys = sorted(k for k in own if k in index)
        for position in range(args.beta_weeks, len(keys)):
            window = keys[position - args.beta_weeks:position]
            x = np.array([universe[k] for k in window])
            y = np.array([own[k] for k in window])
            if len(x) < 30 or x.std() == 0:
                continue
            slope = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
            residual = y - (slope * x + (y.mean() - slope * x.mean()))
            out[ticker][keys[position]] = (slope,
                                           float(residual.std(ddof=1)) * math.sqrt(52))
    return out


def market_context(args, book):
    """VIX and the universe's own drawdown, both readable on the day."""
    vix = {}
    if args.macro.exists():
        connection = sqlite3.connect(f"file:{args.macro}?mode=ro", uri=True)
        for day, value in connection.execute(
                "SELECT obs_date, value FROM observations WHERE series_id='VIXCLS' "
                "AND value IS NOT NULL"):
            vix[day[:10]] = float(value)
        connection.close()
    level = defaultdict(list)
    for bars in book.values():
        for bar in bars:
            level[bar.timestamp].append(bar.close)
    days = sorted(level)
    index, running, peaks = {}, [], {}
    base_value = None
    for day in days:
        value = statistics.fmean(level[day])
        base_value = base_value or value
        index[day] = value / base_value
        running.append(index[day])
        if len(running) > 200:
            running.pop(0)
        peaks[day] = max(running)
    market_dd = {d: index[d] / peaks[d] - 1.0 for d in days}
    return vix, market_dd


def main(argv=None) -> int:
    args = parse_args(argv)
    inner = base.parse_args([])
    mapping = json.loads(inner.map.read_text())
    book = base.load_book(inner)
    mapping = {t: c for t, c in mapping.items() if t in book}
    panel = base.fundamentals(inner, mapping)
    triggers, _ = base.build_events(book, panel, inner)
    month_days, medians = base.universe_medians(book, panel, inner)
    rows = triggers[(200, 0.10)]
    betas = beta_panel(book, args)
    vix, market_dd = market_context(args, book)
    print(f"\n{len(rows):,d} drawdown crossings, beta panel for {len(betas)} names")

    import datetime as dt
    prices = {t: {b.timestamp: b.close for b in bars} for t, bars in book.items()}
    ordered = {t: sorted(p) for t, p in prices.items()}
    samples = []
    for r in rows:
        ticker, day = r["ticker"], r["day"]
        year, week, _ = dt.date.fromisoformat(day).isocalendar()
        got = betas.get(ticker, {})
        key = max((k for k in got if k <= (year, week)), default=None)
        if key is None:
            continue
        beta, ivol = got[key]
        days = ordered[ticker]
        position = bisect_right(days, day) - 1
        if position < 0 or position + max(HORIZONS) >= len(days):
            continue
        entry = prices[ticker][days[position]]
        if entry <= 0:
            continue
        row = {"ticker": ticker, "day": day, "depth": r["depth"],
               "beta": beta, "ivol": ivol,
               "book_yield": r.get("book_yield"), "roe": r.get("roe"),
               "leverage": r.get("leverage"),
               "vix": vix.get(day), "market_dd": market_dd.get(day)}
        for h in HORIZONS:
            row[f"fwd_{h}"] = prices[ticker][days[position + h]] / entry - 1.0
        if all(row[f] is not None for f in FEATURES):
            samples.append(row)
    print(f"{len(samples):,d} events with every feature present\n")

    print("########## regression: what predicts the rebound ##########")
    print("  Standardised coefficients, so magnitudes compare directly.")
    for horizon in HORIZONS:
        names = list(FEATURES) + [f"{a}x{b}" for a, b in INTERACTIONS]
        design, target = [], []
        for s in samples:
            row = [s[f] for f in FEATURES]
            row += [s[a] * s[b] for a, b in INTERACTIONS]
            design.append(row)
            target.append(s[f"fwd_{horizon}"])
        X = np.array(design, dtype=float)
        y = np.array(target, dtype=float)
        X = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
        y_s = (y - y.mean()) / y.std()
        A = np.column_stack([np.ones(len(X)), X])
        coefficients, *_ = np.linalg.lstsq(A, y_s, rcond=None)
        residual = y_s - A @ coefficients
        dof = max(len(X) - A.shape[1], 1)
        se = math.sqrt(residual @ residual / dof) * np.sqrt(
            np.diag(np.linalg.pinv(A.T @ A)))
        print(f"\n  +{horizon} sessions   (n = {len(X):,d}, "
              f"R^2 = {1 - residual.var() / y_s.var():.3f})")
        print(f"    {'feature':16s} {'beta':>8s} {'t':>7s}")
        order = sorted(range(1, len(coefficients)),
                       key=lambda i: -abs(coefficients[i] / se[i]))
        for i in order:
            t = coefficients[i] / se[i]
            flag = "  <-" if abs(t) >= 2 else ""
            print(f"    {names[i-1]:16s} {coefficients[i]:>+8.3f} {t:>+7.2f}{flag}")

    print("\n########## matched pairs: junk vs quality, convexity held equal ##########")
    gated = {(g["ticker"], g["day"]) for g in
             base.cross_sectional_gates(rows, month_days, medians, need=2)}
    failed = {(g["ticker"], g["day"]) for g in
              base.cross_sectional_gates(rows, month_days, medians,
                                         invert=True, need=2)}
    pool_pass = [s for s in samples if (s["ticker"], s["day"]) in gated]
    pool_fail = [s for s in samples if (s["ticker"], s["day"]) in failed]
    print(f"  {len(pool_pass):,d} pass the screens, {len(pool_fail):,d} fail them")

    by_month = defaultdict(list)
    for s in pool_pass:
        by_month[s["day"][:7]].append(s)
    pairs = []
    used = set()
    for s in pool_fail:
        best, gap = None, None
        for c in by_month.get(s["day"][:7], []):
            key = (c["ticker"], c["day"])
            if key in used:
                continue
            if abs(c["beta"] - s["beta"]) > 0.25:
                continue
            if abs(c["ivol"] - s["ivol"]) > 0.10:
                continue
            if abs(c["depth"] - s["depth"]) > 0.05:
                continue
            distance = (abs(c["beta"] - s["beta"]) + abs(c["ivol"] - s["ivol"])
                        + abs(c["depth"] - s["depth"]))
            if gap is None or distance < gap:
                best, gap = c, distance
        if best is not None:
            used.add((best["ticker"], best["day"]))
            pairs.append((s, best))
    print(f"  matched {len(pairs):,d} pairs on month, depth, beta and volatility")
    report = {"events": len(samples), "pairs": len(pairs)}
    if pairs:
        print(f"\n  {'':22s} {'fails screen':>13s} {'passes screen':>14s} "
              f"{'difference':>11s} {'t':>7s}")
        for horizon in HORIZONS:
            a = np.array([p[0][f"fwd_{horizon}"] for p in pairs])
            b = np.array([p[1][f"fwd_{horizon}"] for p in pairs])
            d = a - b
            t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 2 else 0.0
            print(f"  forward +{horizon:<3d} sessions {a.mean():>+12.2%} "
                  f"{b.mean():>+14.2%} {d.mean():>+11.2%} {t:>+7.2f}")
            report[f"pair_{horizon}"] = {
                "fails": float(a.mean()), "passes": float(b.mean()),
                "difference": float(d.mean()), "t": float(t)}
        for label, key in (("beta", "beta"), ("volatility", "ivol"),
                           ("depth", "depth")):
            a = statistics.fmean(p[0][key] for p in pairs)
            b = statistics.fmean(p[1][key] for p in pairs)
            print(f"  matched on {label:12s} {a:>12.2f} {b:>14.2f}")

        # The same pairs, split by how frightened the market was on entry day.
        # If fundamentals are priced in calm and ignored in fear, the screen
        # should earn its keep in the low tercile and nowhere else.
        levels = sorted(p[0]["vix"] for p in pairs)
        low, high = levels[len(levels) // 3], levels[2 * len(levels) // 3]
        print()
        print(f"  split by VIX at entry (terciles at {low:.1f} and {high:.1f})")
        print(f"  {'regime':22s} {'pairs':>6s} {'fails':>9s} {'passes':>9s} "
              f"{'difference':>11s} {'t':>7s}")
        report["by_regime"] = {}
        for label, test in (("calm  (VIX low)", lambda v: v <= low),
                            ("normal", lambda v: low < v <= high),
                            ("fear  (VIX high)", lambda v: v > high)):
            chunk = [p for p in pairs if test(p[0]["vix"])]
            if len(chunk) < 30:
                continue
            for horizon in (20,):
                a = np.array([p[0][f"fwd_{horizon}"] for p in chunk])
                b = np.array([p[1][f"fwd_{horizon}"] for p in chunk])
                d = a - b
                t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
                print(f"  {label:22s} {len(chunk):>6d} {a.mean():>+9.2%} "
                      f"{b.mean():>+9.2%} {d.mean():>+11.2%} {t:>+7.2f}")
                report["by_regime"][label.strip()] = {
                    "pairs": len(chunk), "fails": float(a.mean()),
                    "passes": float(b.mean()), "difference": float(d.mean()),
                    "t": float(t)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
