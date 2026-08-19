"""Which option-chain features predict the next session?

Each end-of-day snapshot produces a feature vector; the target is the *following*
session, so the direction of time is never crossed.  Three targets are scored
separately because they are not equally forecastable:

* **realised volatility** -- the easy one. Implied volatility is already a
  forecast of it, so any new feature has to beat plain ATM IV to be interesting.
* **absolute return** -- magnitude without direction.
* **signed return** -- the hard one, and the one where a spurious result is most
  expensive. Reported alongside the others so it can be judged on the same page.

Scoring is by rank correlation, which is robust to the fat tails these series
have, plus a split-sample check: a feature ranked on the first half must keep
its sign on the second half, or it is noise that happened to fit.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.option_features import (  # noqa: E402
    FEATURE_NAMES,
    Contract,
    snapshot_features,
)
from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--options", type=Path,
                        default=Path("data/options/marketdata.db"))
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/option_features.json"))
    return parser.parse_args(argv)


def spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 20:
        return None, len(pairs)
    a = [x for x, _ in pairs]
    b = [y for _, y in pairs]
    ra, rb = [0] * len(a), [0] * len(b)
    for rank, index in enumerate(sorted(range(len(a)), key=lambda i: a[i])):
        ra[index] = rank
    for rank, index in enumerate(sorted(range(len(b)), key=lambda i: b[i])):
        rb[index] = rank
    if len(set(ra)) < 2 or len(set(rb)) < 2:
        return None, len(pairs)
    return statistics.correlation(ra, rb), len(pairs)


def load_chains(path: Path, symbol: str):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT observation_date,side,strike,dte,iv,gamma,open_interest,volume,"
        "underlying_price FROM option_contracts WHERE symbol=? "
        "ORDER BY observation_date", (symbol,)).fetchall()
    connection.close()
    chains, spots = defaultdict(list), {}
    for day, side, strike, dte, iv, gamma, oi, volume, spot in rows:
        chains[day[:10]].append(Contract(side, float(strike), int(dte),
                                         float(iv) if iv else None,
                                         float(gamma) if gamma else None,
                                         float(oi or 0.0), float(volume or 0.0)))
        # A handful of rows carry no underlying print; the snapshot's spot comes
        # from whichever rows do have one.
        if spot is not None:
            spots[day[:10]] = float(spot)
    return chains, spots


def session_targets(path: Path, symbol: str):
    """Realised volatility, absolute return and signed return, per session."""
    splits = load_splits(path)
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
        "frequency='5min' ORDER BY ts", (symbol,)).fetchall()
    connection.close()
    bars = adjust_bars([Bar(*r) for r in rows], splits.get(symbol, []))
    by_day = defaultdict(list)
    for bar in bars:
        by_day[bar.timestamp[:10]].append(bar)
    out = {}
    for day, rows_ in by_day.items():
        rets = [rows_[i].close / rows_[i - 1].close - 1.0
                for i in range(1, len(rows_)) if rows_[i - 1].close > 0]
        if len(rets) < 10:
            continue
        session = rows_[-1].close / rows_[0].open - 1.0
        out[day] = {
            "realised_vol": math.sqrt(sum(r * r for r in rets) * 252) * 100,
            "abs_return": abs(session) * 100,
            "signed_return": session * 100,
        }
    return out


def main(argv=None):
    args = parse_args(argv)
    chains, spots = load_chains(args.options, args.symbol)
    targets = session_targets(args.bars, args.symbol)
    days = sorted(set(chains) & set(spots))
    print(f"{args.symbol}: {len(days)} chain snapshots, building features...", flush=True)

    table = {}
    for day in days:
        feats = snapshot_features(chains[day], spots[day])
        if feats:
            table[day] = feats
    ordered = sorted(table)
    print(f"  {len(ordered)} feature rows\n")

    # Feature on day D is matched to the outcome of the NEXT session.
    session_days = sorted(targets)
    pairs = []
    for day in ordered:
        later = [d for d in session_days if d > day]
        if not later:
            continue
        pairs.append((day, later[0]))
    print(f"  {len(pairs)} feature/next-session pairs "
          f"({pairs[0][0]} -> {pairs[-1][1]})\n")

    half = len(pairs) // 2
    report = {"symbol": args.symbol, "pairs": len(pairs)}
    for target in ("realised_vol", "abs_return", "signed_return"):
        print(f"  TARGET: next-session {target}")
        print(f"    {'feature':22s} {'rho(all)':>9s} {'rho(1st)':>9s} "
              f"{'rho(2nd)':>9s} {'stable':>7s}")
        scored = []
        for name in FEATURE_NAMES:
            xs = [table[d][name] for d, _ in pairs]
            ys = [targets[n][target] for _, n in pairs]
            rho, n = spearman(xs, ys)
            r1, _ = spearman(xs[:half], ys[:half])
            r2, _ = spearman(xs[half:], ys[half:])
            if rho is None:
                continue
            stable = (r1 is not None and r2 is not None
                      and (r1 > 0) == (r2 > 0) and abs(rho) > 0.10)
            scored.append((abs(rho), name, rho, r1, r2, stable))
        scored.sort(reverse=True)
        for _, name, rho, r1, r2, stable in scored:
            f1 = f"{r1:+.3f}" if r1 is not None else "  n/a"
            f2 = f"{r2:+.3f}" if r2 is not None else "  n/a"
            print(f"    {name:22s} {rho:>+9.3f} {f1:>9s} {f2:>9s} "
                  f"{'yes' if stable else '-':>7s}")
        report[target] = [
            {"feature": n, "rho": r, "first": r1, "second": r2, "stable": s}
            for _, n, r, r1, r2, s in scored
        ]
        best = scored[0]
        iv = next((s for s in scored if s[1] == "atm_iv"), None)
        if iv:
            print(f"    -> best {best[1]} ({best[2]:+.3f}) vs ATM IV benchmark "
                  f"({iv[2]:+.3f})")
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
