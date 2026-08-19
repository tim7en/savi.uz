"""Which option-chain features actually predict the next session?

The earlier version of this ran on 249 SPY sessions.  This runs on roughly 2,500
per symbol across nine years, with QQQ as an independent replication rather than
a second look at the same data.

A feature is only called *proven* when it clears four hurdles at once:

* a rank correlation of at least 0.10 over the full sample,
* the same sign on **both** SPY and QQQ,
* the same sign in at least four of five non-overlapping sub-periods,
* and, for anything claimed against volatility, it must add something ATM
  implied volatility does not already provide -- IV is the obvious forecast and
  the honest bar to clear.

Thirteen features against three targets is thirty-nine tests, so some will look
good by chance.  Cross-symbol and cross-period agreement is the correction:
a coincidence rarely repeats on a different instrument in a different decade.

Direction of time: the feature is the end-of-day snapshot of session D, the
target is what session D+1 does.
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

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

FEATURES = ("atm_iv", "iv_term_slope", "skew_moneyness", "skew_25delta",
            "put_call_oi", "put_call_volume", "net_gex", "gamma_balance",
            "gamma_flip_distance", "zero_dte_share", "vanna", "total_oi",
            "total_volume")
TARGETS = ("realised_vol", "abs_return", "signed_return")
PERIODS = [("2017-18", "2017-01-01", "2019-01-01"),
           ("2019-20", "2019-01-01", "2021-01-01"),
           ("2021-22", "2021-01-01", "2023-01-01"),
           ("2023-24", "2023-01-01", "2025-01-01"),
           ("2025-26", "2025-01-01", "2027-01-01")]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--options", type=Path,
                        default=Path("data/options/alphavantage.db"))
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--symbols", nargs="+", default=["SPY", "QQQ"])
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/av_feature_power.json"))
    return parser.parse_args(argv)


def spearman(pairs):
    if len(pairs) < 40:
        return None
    a = [x for x, _ in pairs]
    b = [y for _, y in pairs]
    ra, rb = [0] * len(a), [0] * len(b)
    for rank, i in enumerate(sorted(range(len(a)), key=lambda k: a[k])):
        ra[i] = rank
    for rank, i in enumerate(sorted(range(len(b)), key=lambda k: b[k])):
        rb[i] = rank
    if len(set(ra)) < 2 or len(set(rb)) < 2:
        return None
    return statistics.correlation(ra, rb)


def partial_spearman(pairs_x, pairs_control, pairs_y):
    """Rank correlation of x with y after removing what the control explains."""
    n = len(pairs_y)
    if n < 40:
        return None
    def ranks(values):
        out = [0] * len(values)
        for rank, i in enumerate(sorted(range(len(values)), key=lambda k: values[k])):
            out[i] = rank
        return out
    rx, rc, ry = ranks(pairs_x), ranks(pairs_control), ranks(pairs_y)
    try:
        rxy = statistics.correlation(rx, ry)
        rxc = statistics.correlation(rx, rc)
        rcy = statistics.correlation(rc, ry)
    except statistics.StatisticsError:
        return None
    denominator = math.sqrt(max((1 - rxc ** 2) * (1 - rcy ** 2), 1e-12))
    return (rxy - rxc * rcy) / denominator


def session_targets(bars_path: Path, symbol: str):
    splits = load_splits(bars_path)
    connection = sqlite3.connect(f"file:{bars_path}?mode=ro", uri=True)
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
        out[day] = {"realised_vol": math.sqrt(sum(r * r for r in rets) * 252) * 100,
                    "abs_return": abs(session) * 100,
                    "signed_return": session * 100}
    return out


def main(argv=None):
    args = parse_args(argv)
    store = sqlite3.connect(f"file:{args.options}?mode=ro", uri=True)
    table = {}
    for symbol in args.symbols:
        rows = store.execute(
            f"SELECT observation_date,{','.join(FEATURES)} FROM av_daily "
            "WHERE symbol=? ORDER BY observation_date", (symbol,)).fetchall()
        table[symbol] = {r[0]: dict(zip(FEATURES, r[1:])) for r in rows}
    store.close()

    matched = {}
    for symbol in args.symbols:
        targets = session_targets(args.bars, symbol)
        days = sorted(set(table[symbol]) & set(targets))
        later = sorted(targets)
        pairs = []
        for day in days:
            nxt = next((d for d in later if d > day), None)
            if nxt:
                pairs.append((day, nxt))
        matched[symbol] = (pairs, targets)
        print(f"{symbol}: {len(pairs):,} feature/next-session pairs "
              f"({pairs[0][0]} -> {pairs[-1][1]})")
    print()

    report = {}
    for target in TARGETS:
        print(f"{'=' * 78}\nTARGET: next-session {target}")
        header = (f"  {'feature':22s} {'SPY':>7s} {'QQQ':>7s} {'same':>5s} "
                  f"{'periods':>8s} {'ex-IV':>7s} {'verdict':>9s}")
        print(header)
        rows_out = []
        for feature in FEATURES:
            rho, consistent, partial = {}, 0, {}
            for symbol in args.symbols:
                pairs, targets = matched[symbol]
                data = [(table[symbol][d][feature], targets[n][target])
                        for d, n in pairs
                        if table[symbol][d][feature] is not None]
                rho[symbol] = spearman(data)
                if target == "realised_vol" and feature != "atm_iv":
                    trio = [(table[symbol][d][feature], table[symbol][d]["atm_iv"],
                             targets[n][target])
                            for d, n in pairs
                            if table[symbol][d][feature] is not None
                            and table[symbol][d]["atm_iv"] is not None]
                    if len(trio) >= 40:
                        partial[symbol] = partial_spearman(
                            [t[0] for t in trio], [t[1] for t in trio],
                            [t[2] for t in trio])
            values = [v for v in rho.values() if v is not None]
            if len(values) < len(args.symbols):
                continue
            same_sign = all(v > 0 for v in values) or all(v < 0 for v in values)
            for label, lo, hi in PERIODS:
                signs = []
                for symbol in args.symbols:
                    pairs, targets = matched[symbol]
                    data = [(table[symbol][d][feature], targets[n][target])
                            for d, n in pairs if lo <= d < hi
                            and table[symbol][d][feature] is not None]
                    r = spearman(data)
                    if r is not None:
                        signs.append(r > 0)
                if signs and all(s == (values[0] > 0) for s in signs):
                    consistent += 1
            ex_iv = ([v for v in partial.values() if v is not None] or [None])
            ex_iv_val = (sum(ex_iv) / len(ex_iv)) if ex_iv[0] is not None else None
            strong = min(abs(v) for v in values) >= 0.10
            proven = strong and same_sign and consistent >= 4
            if target == "realised_vol" and feature != "atm_iv":
                proven = proven and ex_iv_val is not None and abs(ex_iv_val) >= 0.05
            print(f"  {feature:22s} {values[0]:>+7.3f} {values[1]:>+7.3f} "
                  f"{'yes' if same_sign else 'no':>5s} {consistent:>6d}/5 "
                  f"{(f'{ex_iv_val:+.3f}' if ex_iv_val is not None else '   —'):>7s} "
                  f"{'PROVEN' if proven else '-':>9s}")
            rows_out.append({"feature": feature, "spy": values[0], "qqq": values[1],
                             "same_sign": same_sign, "periods": consistent,
                             "partial_ex_iv": ex_iv_val, "proven": proven})
        report[target] = rows_out
        winners = [r["feature"] for r in rows_out if r["proven"]]
        print(f"  -> proven: {', '.join(winners) if winners else 'NOTHING'}\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
