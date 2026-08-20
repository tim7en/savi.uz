"""Do macro regimes move ETF breakouts, and which way?

Two analyses, deliberately separated, because they answer different questions and
only the first has real power.

**The pre-registered test.**  Sector responses to interest rates have a mechanism
that can be written down before looking: utilities and real estate are bond
proxies whose cash flows discount like long duration, technology is long-duration
growth, staples are defensive and bond-like; financials earn on a steeper curve,
and energy, materials and industrials answer to the cycle rather than to the
discount rate.  So the prediction is signed in advance:

    HURT by rising rates:    XLU, XLRE, XLK, XLP
    HELPED by rising rates:  XLF, XLE, XLB, XLI
    no prior:                XLV, XLY, XLC   (excluded from the statistic)

    statistic = (helped, inside minus outside) - (hurt, inside minus outside)
    predicted POSITIVE for long breakouts, NEGATIVE for short breakouts

That is one comparison, not a search, so it needs no multiplicity correction and
carries far more power than an unsigned table.  The sign flip between long and
short is a free consistency check: if utilities really suffer when rates rise,
shorting them should do correspondingly well, and a result that fails to flip is
not describing a rate mechanism whatever its p-value says.

**The exploratory table.**  Every instrument against every regime in both
directions, reported because it is what was asked for, and scored by comparing
the largest gap anywhere in the table against the largest gap in the same table
under circularly shifted labels.  Taking the maximum on both sides prices the
search.  An earlier version of this on the twenty-one non-sector funds returned
p = 0.796 -- the real table was less striking than a typical shuffled one.

Effective sample size is reported as episodes rather than sessions throughout.
Twenty-six years contains few independent rate cycles however many bars it holds.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import TurtleConfig, run_turtle  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

HURT = ("XLU", "XLRE", "XLK", "XLP")
HELPED = ("XLF", "XLE", "XLB", "XLI")
NO_PRIOR = ("XLV", "XLY", "XLC")
SECTORS = HURT + HELPED + NO_PRIOR


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etf", type=Path,
                        default=Path("data/cross_assets/etf_30min.db"))
    parser.add_argument("--macro", type=Path, default=Path("data/macro/macro.db"))
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--entry-window", type=int, default=55)
    parser.add_argument("--trail", type=float, default=3.0)
    parser.add_argument("--cost", type=float, default=0.0002)
    parser.add_argument("--min-trades", type=int, default=40)
    parser.add_argument("--nulls", type=int, default=200)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/sector_regime.json"))
    return parser.parse_args(argv)


def load(args):
    connection = sqlite3.connect(f"file:{args.etf}?mode=ro", uri=True)
    book = {}
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars ORDER BY ticker"):
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? "
            "ORDER BY ts", (ticker,)).fetchall()
        if len(rows) >= 2000:
            book[ticker] = resample_regular_session([Bar(*r) for r in rows],
                                                    minutes=args.minutes)
    connection.close()
    return book


def regimes_from(args, book):
    connection = sqlite3.connect(f"file:{args.macro}?mode=ro", uri=True)
    horizons = defaultdict(dict)
    for day, horizon, rate in connection.execute(
        "SELECT curve_date, horizon_months, forward_rate FROM fed_path "
        "WHERE horizon_months IN (3, 12, 24)"
    ):
        horizons[day][horizon] = rate
    tenors = defaultdict(dict)
    for day, mnemonic, value in connection.execute(
        "SELECT curve_date, mnemonic, value FROM gsw_rates "
        "WHERE mnemonic IN ('SVENY02','SVENY10')"
    ):
        tenors[day][mnemonic] = value
    connection.close()

    out = {}
    out["tightening priced"] = {
        d: v[12] > v[3] for d, v in horizons.items()
        if v.get(3) is not None and v.get(12) is not None}
    out["curve inverted"] = {
        d: v["SVENY10"] < v["SVENY02"] for d, v in tenors.items()
        if v.get("SVENY02") is not None and v.get("SVENY10") is not None}
    days = sorted(horizons)
    level = {d: horizons[d].get(12) for d in days}
    out["policy path rising"] = {
        d: level[d] > level[days[max(0, i - 63)]] for i, d in enumerate(days)
        if level[d] is not None and level[days[max(0, i - 63)]] is not None}

    # Commodity and dollar trend, from price rather than from a macro vendor:
    # a fund above its own trailing average is the simplest honest statement of
    # "this complex is trending", and it needs no series that can be restated.
    for ticker, name in (("DBC", "commodities trending up"),
                         ("UUP", "dollar trending up")):
        bars = book.get(ticker)
        if not bars:
            continue
        closes = {}
        for bar in bars:
            closes[bar.timestamp[:10]] = bar.close
        ordered = sorted(closes)
        labels, seen = {}, []
        for day in ordered:
            if len(seen) >= 120:
                labels[day] = closes[day] > statistics.fmean(seen[-120:])
            seen.append(closes[day])
        out[name] = labels
    return out


def lag(labels, sessions):
    days = sorted(labels)
    result, position, carried = {}, 0, None
    for day in sessions:
        while position < len(days) and days[position] < day:
            carried = labels[days[position]]
            position += 1
        if carried is not None:
            result[day] = carried
    return result


def episodes(labels):
    days = sorted(labels)
    if not days:
        return 0
    runs, previous = 1, labels[days[0]]
    for day in days[1:]:
        if labels[day] != previous:
            runs += 1
            previous = labels[day]
    return runs


def split(trades, labels):
    inside = [t["r"] for t in trades if labels.get(t["entry"][:10]) is True]
    outside = [t["r"] for t in trades if labels.get(t["entry"][:10]) is False]
    if len(inside) < 20 or len(outside) < 20:
        return None
    return statistics.fmean(inside) - statistics.fmean(outside)


def basket_gap(trades, labels, names, side):
    values = [split(trades[(n, side)], labels) for n in names
              if (n, side) in trades and len(trades[(n, side)]) >= 40]
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def shifted(labels, offset):
    days = sorted(labels)
    values = [labels[d] for d in days]
    return {d: values[(i + offset) % len(days)] for i, d in enumerate(days)}


def main(argv=None):
    args = parse_args(argv)
    book = load(args)
    sessions = sorted({b.timestamp[:10] for bars in book.values() for b in bars})
    print(f"{len(book)} ETFs ({sum(1 for t in book if t in SECTORS)} sectors), "
          f"{len(sessions):,} sessions {sessions[0]} -> {sessions[-1]}", flush=True)

    trades = {}
    for side, directions in (("long", (1,)), ("short", (-1,)), ("both", (1, -1))):
        config = TurtleConfig(entry_window=args.entry_window,
                              exit_window=max(5, args.entry_window // 3),
                              atr_window=20, skip_after_winner=False,
                              directions=directions, use_channel_exit=False,
                              chandelier_atr=args.trail, round_trip_cost=args.cost)
        for ticker, bars in book.items():
            found, _ = run_turtle(bars, config=config)
            trades[(ticker, side)] = [{"entry": t.entry_timestamp, "r": t.net_r}
                                      for t in found]

    regimes = {name: lag(labels, sessions)
               for name, labels in regimes_from(args, book).items()}
    print(f"\n  {'regime':26s} {'episodes':>9s} {'share':>7s}")
    for name, labels in regimes.items():
        print(f"  {name:26s} {episodes(labels):>9d} "
              f"{sum(1 for v in labels.values() if v) / max(len(labels), 1):>7.1%}")

    # ---------------- the pre-registered test ----------------
    print(f"\n{'=' * 78}\nPRE-REGISTERED: helped minus hurt, sign fixed before the run")
    print(f"  helped {HELPED}\n  hurt   {HURT}\n")
    print(f"  {'regime':26s} {'side':>6s} {'helped':>8s} {'hurt':>8s} "
          f"{'stat':>8s} {'null p95':>9s} {'p':>7s}  predicted")
    registered = {}
    rng = random.Random(3)
    for name, labels in regimes.items():
        if name in ("commodities trending up", "dollar trending up"):
            continue
        for side, want in (("long", "+"), ("short", "-")):
            helped = basket_gap(trades, labels, HELPED, side)
            hurt = basket_gap(trades, labels, HURT, side)
            if helped is None or hurt is None:
                continue
            statistic = helped - hurt
            nulls = []
            for _ in range(args.nulls):
                fake = shifted(labels, rng.randrange(200, max(400, len(labels))))
                a = basket_gap(trades, fake, HELPED, side)
                b = basket_gap(trades, fake, HURT, side)
                if a is not None and b is not None:
                    nulls.append(a - b)
            nulls.sort()
            if not nulls:
                continue
            if want == "+":
                beat = sum(1 for v in nulls if v >= statistic)
                p95 = nulls[int(0.95 * len(nulls))]
            else:
                beat = sum(1 for v in nulls if v <= statistic)
                p95 = nulls[int(0.05 * len(nulls))]
            p_value = (beat + 1) / (len(nulls) + 1)
            registered[f"{name}/{side}"] = {
                "helped": helped, "hurt": hurt, "stat": statistic,
                "p": p_value, "null_p95": p95, "predicted": want}
            print(f"  {name:26s} {side:>6s} {helped:>+8.3f} {hurt:>+8.3f} "
                  f"{statistic:>+8.3f} {p95:>+9.3f} {p_value:>7.3f}  {want}",
                  flush=True)

    flips = [(k, v) for k, v in registered.items()]
    passed = [k for k, v in registered.items()
              if v["p"] < 0.05 and ((v["predicted"] == "+" and v["stat"] > 0)
                                    or (v["predicted"] == "-" and v["stat"] < 0))]
    print(f"\n  cells matching the predicted sign at p<0.05: "
          f"{len(passed)}/{len(flips)}"
          + (f"  -> {', '.join(passed)}" if passed else ""))

    # ---------------- the exploratory table ----------------
    print(f"\n{'=' * 78}\nEXPLORATORY: every instrument, every regime, both sides")
    cells = []
    for name, labels in regimes.items():
        for ticker in book:
            for side in ("long", "short"):
                pooled = trades.get((ticker, side), [])
                if len(pooled) < args.min_trades:
                    continue
                value = split(pooled, labels)
                if value is not None:
                    cells.append({"regime": name, "ticker": ticker,
                                  "side": side, "gap": value, "n": len(pooled)})
    observed = max(abs(c["gap"]) for c in cells)
    print(f"  {len(cells)} cells; largest absolute gap {observed:.3f}R", flush=True)
    null_max = []
    for _ in range(args.nulls):
        fake = {name: shifted(labels, rng.randrange(200, max(400, len(labels))))
                for name, labels in regimes.items()}
        best = 0.0
        for cell in cells:
            value = split(trades[(cell["ticker"], cell["side"])],
                          fake[cell["regime"]])
            if value is not None:
                best = max(best, abs(value))
        null_max.append(best)
    null_max.sort()
    pick = lambda f: null_max[min(int(f * len(null_max)), len(null_max) - 1)]
    beat = sum(1 for v in null_max if v >= observed)
    print(f"  shuffled tables: median {pick(.5):.3f}  p95 {pick(.95):.3f}  "
          f"max {null_max[-1]:.3f}")
    print(f"  empirical p = {(beat + 1) / (len(null_max) + 1):.3f}")
    print(f"  cells clearing the corrected bar: "
          f"{sum(1 for c in cells if abs(c['gap']) > pick(.95))}")
    print(f"\n  strongest cells, for description only:")
    for cell in sorted(cells, key=lambda c: -abs(c["gap"]))[:10]:
        print(f"    {cell['regime']:26s} {cell['ticker']:5s} {cell['side']:5s} "
              f"{cell['gap']:>+7.3f}R  ({cell['n']:,} trades)")

    report = {"registered": registered, "cells": cells, "observed_max": observed,
              "exploratory_p": (beat + 1) / (len(null_max) + 1),
              "null_p95": pick(.95),
              "episodes": {n: episodes(l) for n, l in regimes.items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
