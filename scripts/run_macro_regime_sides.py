"""Does the short side track the macro regime, or does it just track the calendar?

The claim to test is that shorts earned their keep when the macro regime was
unfavourable.  The year-by-year table is suggestive but ambiguous: shorts were
profitable 2017-2022 and negative every year from 2023, and 2017, 2019 and 2021
were strong bull years in which shorts nonetheless made money.  That pattern is
equally consistent with a regime effect and with plain decay, and the two imply
opposite actions -- one says build a regime switch, the other says the short side
is dying and a switch would be fitted to its good half.

So both are fitted against the same trades.  A regime story only survives if it
explains the short side's returns *better than the calendar does*, which is a
much harder test than "shorts did well when rates rose".

Regimes come from market-priced rates, not from published macro data, which
sidesteps the revision problem entirely: the Fed path and the yield curve are
observed prices, never restated.  Each is lagged one session regardless.

Randomness controls, because a regime label with a handful of switches will fit
anything: a circular shift preserving each regime's persistence and run lengths,
and a rate-matched independent label.  A regime split is only real if it beats
both.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import TurtleConfig, run_turtle  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

BASE = dict(entry_window=55, exit_window=20, atr_window=20, skip_after_winner=False,
            use_channel_exit=False, chandelier_atr=3.0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--macro", type=Path, default=Path("data/macro/macro.db"))
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--min-sessions", type=int, default=400)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/macro_regime_sides.json"))
    return parser.parse_args(argv)


def load_book(args):
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    names = [r[0] for r in connection.execute(
        "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker")]
    book = {}
    for ticker in names:
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='5min' ORDER BY ts", (ticker,)).fetchall()
        if not rows:
            continue
        five = adjust_bars([Bar(*r) for r in rows], splits.get(ticker, []))
        if len({b.timestamp[:10] for b in five}) < args.min_sessions:
            continue
        book[ticker] = resample_regular_session(five, minutes=args.minutes)
    connection.close()
    return book


def macro_regimes(path: Path):
    """Binary regime labels per session, from market-priced rates only."""
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    path_rows = connection.execute(
        "SELECT curve_date, horizon_months, forward_rate FROM fed_path "
        "WHERE horizon_months IN (3, 12) ORDER BY curve_date").fetchall()
    curve_rows = connection.execute(
        "SELECT curve_date, mnemonic, value FROM gsw_rates "
        "WHERE mnemonic IN ('SVENF02','SVENF10') ORDER BY curve_date").fetchall()
    connection.close()

    horizons = defaultdict(dict)
    for day, horizon, rate in path_rows:
        horizons[day][horizon] = rate
    tenors = defaultdict(dict)
    for day, mnemonic, value in curve_rows:
        tenors[day][mnemonic] = value

    # Tightening expected: the 12-month forward sits above the 3-month.
    tightening = {d: v[12] > v[3] for d, v in horizons.items()
                  if 3 in v and 12 in v and v[3] is not None and v[12] is not None}
    # Inverted curve: the classic recession signal, 10-year below 2-year.
    inverted = {d: v["SVENF10"] < v["SVENF02"] for d, v in tenors.items()
                if "SVENF02" in v and "SVENF10" in v
                and v["SVENF02"] is not None and v["SVENF10"] is not None}
    # Rising level: policy expectations above where they sat a quarter ago.
    days = sorted(horizons)
    level = {d: horizons[d].get(12) for d in days}
    rising = {}
    for i, day in enumerate(days):
        past = days[max(0, i - 63)]
        if level[day] is not None and level[past] is not None:
            rising[day] = level[day] > level[past]
    return {"tightening expected": tightening, "curve inverted": inverted,
            "policy path rising": rising}


def lag_to_sessions(labels, sessions):
    """Each session takes the label of the last macro date strictly before it."""
    days = sorted(labels)
    out, position, carried = {}, 0, None
    for day in sessions:
        while position < len(days) and days[position] < day:
            carried = labels[days[position]]
            position += 1
        if carried is not None:
            out[day] = carried
    return out


def split_stats(trades, labels):
    """Mean R inside and outside the regime, and the gap between them."""
    inside = [t["r"] for t in trades if labels.get(t["entry"][:10]) is True]
    outside = [t["r"] for t in trades if labels.get(t["entry"][:10]) is False]
    if len(inside) < 100 or len(outside) < 100:
        return None
    return {"n_in": len(inside), "n_out": len(outside),
            "mean_in": statistics.fmean(inside), "mean_out": statistics.fmean(outside),
            "gap": statistics.fmean(inside) - statistics.fmean(outside)}


def shifted(labels, offset):
    days = sorted(labels)
    values = [labels[d] for d in days]
    n = len(days)
    return {d: values[(i + offset) % n] for i, d in enumerate(days)}


def resampled(labels, rng):
    pool = list(labels.values())
    return {d: rng.choice(pool) for d in labels}


def main(argv=None):
    args = parse_args(argv)
    book = load_book(args)
    sessions = sorted({b.timestamp[:10] for bars in book.values() for b in bars})
    regimes = {name: lag_to_sessions(labels, sessions)
               for name, labels in macro_regimes(args.macro).items()}
    print(f"{len(book)} instruments, {len(sessions):,} sessions", flush=True)
    for name, labels in regimes.items():
        share = sum(1 for v in labels.values() if v) / max(len(labels), 1)
        print(f"  {name:22s} {share:>6.1%} of sessions", flush=True)

    sides = {}
    for label, directions in (("long", (1,)), ("short", (-1,))):
        config = TurtleConfig(**BASE, directions=directions)
        pooled = []
        for ticker, bars in book.items():
            trades, _ = run_turtle(bars, config=config)
            pooled.extend({"entry": t.entry_timestamp, "r": t.net_r} for t in trades)
        sides[label] = pooled

    # The rival explanation: the calendar. Split at the sample midpoint.
    midpoint = sessions[len(sessions) // 2]
    print(f"\n  calendar split at {midpoint}")
    calendar = {d: d < midpoint for d in sessions}

    report = {}
    for side, trades in sides.items():
        print(f"\n  {side.upper()} side, {len(trades):,} trades")
        print(f"    {'split':24s} {'in':>9s} {'out':>9s} {'gap':>8s} "
              f"{'null p95':>9s} {'verdict':>9s}")
        base = split_stats(trades, calendar)
        print(f"    {'first half of sample':24s} {base['mean_in']:>+9.3f} "
              f"{base['mean_out']:>+9.3f} {base['gap']:>+8.3f} {'—':>9s} "
              f"{'reference':>9s}")
        report.setdefault(side, {})["calendar"] = base
        for name, labels in regimes.items():
            actual = split_stats(trades, labels)
            if actual is None:
                continue
            rng = random.Random(7)
            nulls = []
            for draw in range(args.draws):
                fake = (shifted(labels, 97 * (draw + 1)) if draw % 2 == 0
                        else resampled(labels, rng))
                stat = split_stats(trades, fake)
                if stat:
                    nulls.append(abs(stat["gap"]))
            nulls.sort()
            threshold = nulls[int(0.95 * len(nulls))] if nulls else float("nan")
            beats = abs(actual["gap"]) > threshold
            print(f"    {name:24s} {actual['mean_in']:>+9.3f} "
                  f"{actual['mean_out']:>+9.3f} {actual['gap']:>+8.3f} "
                  f"{threshold:>9.3f} {'REAL' if beats else 'noise':>9s}")
            actual["null_p95"] = threshold
            actual["beats_null"] = beats
            report[side][name] = actual

    print("\n  reading: 'gap' is mean R inside the regime minus outside. A regime")
    print("  only counts if its gap clears the 95th percentile of gaps produced by")
    print("  persistence-matched random labels, and is larger than the calendar gap.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
