"""Does CFTC positioning predict index returns, or is it noise with a story?

The Traders in Financial Futures report splits open interest into dealers, asset
managers, leveraged money and the rest.  The folk claim is that leveraged money
is contrarian at extremes -- crowded longs precede declines.  This tests that
against the only control that matters for a slow, highly autocorrelated series:
the same signal with its alignment to price destroyed and its persistence kept.

Three decisions are fixed before the run, because each is a place where a null
result can be turned into a positive one by choosing again.

*Keying.*  Positioning is scored on ``effective_date`` from ``cot_release_calendar``
-- the first session after the Friday publication -- never on the Tuesday as-of.
The as-of arm is run too, but only to measure the bias it introduces.

*Thresholds do not travel, ranks do.*  Net position is normalised by open
interest and then converted to a percentile against its own trailing three
years.  A raw threshold would encode the growth of the contract rather than the
crowdedness of the position.

*Persistence, not sample size.*  Positioning barely moves week to week, so 1,000
observations are nowhere near 1,000 independent ones.  The null is a circular
shift, which preserves the autocorrelation and the run lengths exactly while
destroying only the alignment to returns; an effective sample size is reported
beside every nominal one.

The kill criterion, pre-registered: an arm survives only if the quintile spread
clears the circular-shift null at p < 0.05 *and* keeps its sign under the
release-date lag.  Anything that only works on as-of keying is look-ahead.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import statistics
import sys
from pathlib import Path

#: Contract code is stable across the 2022 renames; the display name is not.
CONTRACTS = {
    "13874A": ("E-mini S&P 500", "^GSPC"),
    "209742": ("Nasdaq-100 mini", "^NDX"),
    "1170E1": ("VIX futures", "^GSPC"),
}

CATEGORIES = ("lev_money", "asset_mgr", "dealer")

RANK_WINDOW = 156
RANK_MINIMUM = 52
QUINTILE = 5


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cot", type=Path, default=Path("data/data/cftc/cot.db"))
    parser.add_argument("--equity", type=Path,
                        default=Path("data/data/equity/equity.db"))
    parser.add_argument("--horizon", type=int, default=1,
                        help="forward horizon in weekly observations")
    parser.add_argument("--nulls", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/cftc_positioning.json"))
    return parser.parse_args(argv)


def load_prices(path: Path) -> dict[str, dict[str, float]]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    prices: dict[str, dict[str, float]] = {}
    for ticker, day, close in connection.execute(
        "SELECT ticker, obs_date, close FROM index_prices ORDER BY obs_date"
    ):
        prices.setdefault(ticker, {})[day[:10]] = close
    connection.close()
    return prices


def load_positioning(path: Path, code: str, category: str, keying: str):
    """Weekly (date, net-as-share-of-open-interest), keyed as requested."""
    date_column = {"effective": "k.effective_date",
                   "as_of": "t.report_date"}[keying]
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = connection.execute(
        f"""
        SELECT {date_column},
               t."{category}_positions_long_all",
               t."{category}_positions_short_all",
               t.open_interest_all
        FROM cot_tff_futures t
        JOIN cot_release_calendar k ON t.report_date = k.report_date
        WHERE t.contract_code = ? AND {date_column} IS NOT NULL
        ORDER BY t.report_date
        """,
        (code,),
    ).fetchall()
    connection.close()
    series = []
    for day, long, short, interest in rows:
        if None in (long, short, interest) or not interest:
            continue
        series.append((day[:10], (float(long) - float(short)) / float(interest)))
    return series


def percentile_ranks(values: list[float]) -> list[float | None]:
    """Rank against the trailing window only -- never the whole sample."""
    ranks: list[float | None] = []
    for index, value in enumerate(values):
        window = values[max(0, index - RANK_WINDOW):index]
        if len(window) < RANK_MINIMUM:
            ranks.append(None)
            continue
        ranks.append(sum(1 for w in window if w < value) / len(window))
    return ranks


def forward_returns(dates: list[str], prices: dict[str, float], horizon: int):
    """Close-to-close from each observation to ``horizon`` observations later."""
    sessions = sorted(prices)
    out: list[float | None] = []
    for index, day in enumerate(dates):
        target = index + horizon
        if target >= len(dates):
            out.append(None)
            continue
        start, end = nearest(sessions, day), nearest(sessions, dates[target])
        if start is None or end is None or start == end:
            out.append(None)
            continue
        out.append(prices[end] / prices[start] - 1.0)
    return out


def nearest(sessions: list[str], day: str) -> str | None:
    """The session on or immediately before ``day``."""
    import bisect
    index = bisect.bisect_right(sessions, day) - 1
    return sessions[index] if index >= 0 else None


def quintile_spread(ranks: list[float], returns: list[float]) -> float:
    paired = sorted(zip(ranks, returns))
    size = len(paired) // QUINTILE
    if size < 2:
        return float("nan")
    bottom = statistics.fmean(r for _, r in paired[:size])
    top = statistics.fmean(r for _, r in paired[-size:])
    return top - bottom


def autocorrelation(values: list[float]) -> float:
    if len(values) < 3:
        return float("nan")
    mean = statistics.fmean(values)
    num = sum((a - mean) * (b - mean) for a, b in zip(values, values[1:]))
    den = sum((v - mean) ** 2 for v in values)
    return num / den if den else float("nan")


def effective_n(count: int, rho: float) -> float:
    if not (-1 < rho < 1):
        return float(count)
    return count * (1 - rho) / (1 + rho)


def analyse(ranks, returns, nulls, rng):
    real = quintile_spread(ranks, returns)
    drawn = []
    size = len(ranks)
    for _ in range(nulls):
        shift = rng.randrange(1, size)
        rotated = ranks[shift:] + ranks[:shift]
        drawn.append(quintile_spread(rotated, returns))
    drawn = [d for d in drawn if d == d]
    beat = sum(1 for d in drawn if abs(d) >= abs(real))
    return {
        "spread": real,
        "null_median": statistics.median(drawn) if drawn else float("nan"),
        "null_p05": statistics.quantiles(drawn, n=20)[0] if len(drawn) > 20 else float("nan"),
        "null_p95": statistics.quantiles(drawn, n=20)[18] if len(drawn) > 20 else float("nan"),
        "p_value": beat / len(drawn) if drawn else float("nan"),
        "observations": size,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)
    prices = load_prices(args.equity)
    report: dict = {}

    print(f"forward horizon: {args.horizon} weekly observation(s); "
          f"{args.nulls} circular-shift nulls\n")
    header = (f"  {'contract':16s} {'category':10s} {'keying':10s} {'obs':>5s} "
              f"{'effN':>6s} {'spread':>9s} {'null 5-95%':>18s} {'p':>7s}")

    for code, (label, ticker) in CONTRACTS.items():
        print(f"{label}  (vs {ticker})")
        print(header)
        for category in CATEGORIES:
            for keying in ("effective", "as_of"):
                series = load_positioning(args.cot, code, category, keying)
                if len(series) < RANK_MINIMUM + 20:
                    continue
                dates = [d for d, _ in series]
                values = [v for _, v in series]
                ranks = percentile_ranks(values)
                forwards = forward_returns(dates, prices[ticker], args.horizon)
                paired = [(r, f) for r, f in zip(ranks, forwards)
                          if r is not None and f is not None]
                if len(paired) < 100:
                    continue
                pr = [r for r, _ in paired]
                fr = [f for _, f in paired]
                stats = analyse(pr, fr, args.nulls, rng)
                rho = autocorrelation(values)
                stats["autocorrelation"] = rho
                stats["effective_n"] = effective_n(stats["observations"], rho)
                report[f"{code}|{category}|{keying}"] = {
                    "contract": label, "ticker": ticker,
                    "category": category, "keying": keying, **stats}
                print(f"  {label:16s} {category:10s} {keying:10s} "
                      f"{stats['observations']:>5d} {stats['effective_n']:>6.0f} "
                      f"{stats['spread']*100:>8.3f}% "
                      f"{stats['null_p05']*100:>8.3f}%..{stats['null_p95']*100:>6.3f}% "
                      f"{stats['p_value']:>7.3f}")
        print()

    survivors = [k for k, v in report.items()
                 if v["keying"] == "effective" and v["p_value"] < 0.05]
    print(f"  arms clearing p < 0.05 on release-date keying: "
          f"{len(survivors)} of {sum(1 for v in report.values() if v['keying']=='effective')}")
    for key in survivors:
        v = report[key]
        print(f"    {v['contract']} / {v['category']}: "
              f"spread {v['spread']*100:+.3f}%, p={v['p_value']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
