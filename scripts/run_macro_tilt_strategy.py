"""The macro tilt, run as an actual strategy and tested against randomness.

The rule, as proposed:

* outside the unfavourable regime -- long only, full size, as banked;
* inside it -- long size halved, and the short sleeve switched on.

Unfavourable means the market prices tightening ahead or the curve is inverted,
lagged one session, from market-priced rates that are never restated.

Long and short run as separate sleeves sharing one capacity cap, which is how it
would actually be traded.  The regime gates *entries*: a position already open
when the regime flips continues under its own rules, because the alternative --
liquidating a healthy trade because a forward rate moved -- is not what was
proposed and would confound the exit result with the regime one.

Two questions have to be answered separately, and only the second one is hard:

1. Does the tilt beat long-only at matched drawdown?  Halving longs cuts risk,
   so an unmatched comparison would flatter it for free.
2. Is the *timing* doing the work, or merely the composition?  Three controls
   separate these -- the same rule with circularly shifted labels (which keeps
   the regime's persistence and run lengths but destroys its alignment to
   price), the same rule with rate-matched random labels, and the same
   composition applied unconditionally at all times.

An empirical p-value falls out of the null draws directly: the share of shuffled
regimes that match or beat the real one.  If that share is large, the rule is a
story told about noise, however sensible the story sounds.
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
from datetime import date
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
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--long-weight-in-regime", type=float, default=0.5)
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--nulls", type=int, default=200)
    parser.add_argument("--null-variant", default="tilt",
                        choices=["tilt", "shorts-only-added"],
                        help="which configuration the null regimes are run through")
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/macro_tilt.json"))
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


def unfavourable(path: Path):
    """Sessions where the market prices tightening ahead or the curve is inverted.

    Inversion uses the zero-coupon yields SVENY02/SVENY10, not the instantaneous
    forwards SVENF02/SVENF10.  The forwards are a different object and give a
    badly wrong answer: they flag two short windows in 2022 and miss the entire
    July 2022 to August 2024 inversion, the longest on record and the only real
    inversion episode this sample contains.  The yields reproduce the known
    history -- 2000, 2006-07 and 2022-24 -- which is the check that caught it.
    """
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    horizons = defaultdict(dict)
    for day, horizon, rate in connection.execute(
        "SELECT curve_date, horizon_months, forward_rate FROM fed_path "
        "WHERE horizon_months IN (3, 12)"
    ):
        horizons[day][horizon] = rate
    tenors = defaultdict(dict)
    for day, mnemonic, value in connection.execute(
        "SELECT curve_date, mnemonic, value FROM gsw_rates "
        "WHERE mnemonic IN ('SVENY02','SVENY10')"
    ):
        tenors[day][mnemonic] = value
    connection.close()

    labels = {}
    for day in set(horizons) | set(tenors):
        path_view = horizons.get(day, {})
        curve = tenors.get(day, {})
        tightening = (path_view.get(3) is not None and path_view.get(12) is not None
                      and path_view[12] > path_view[3])
        inverted = (curve.get("SVENY02") is not None and curve.get("SVENY10") is not None
                    and curve["SVENY10"] < curve["SVENY02"])
        if path_view or curve:
            labels[day] = bool(tightening or inverted)
    return labels


def lag_to_sessions(labels, sessions):
    days = sorted(labels)
    out, position, carried = {}, 0, None
    for day in sessions:
        while position < len(days) and days[position] < day:
            carried = labels[days[position]]
            position += 1
        if carried is not None:
            out[day] = carried
    return out


def session_closes(bars):
    out = {}
    for bar in bars:
        out[bar.timestamp[:10]] = bar.close
    return out


def trade_marks(trade, closes):
    entry_day, exit_day = trade.entry_timestamp[:10], trade.exit_timestamp[:10]
    marks = []
    for day in (d for d in closes if entry_day <= d < exit_day):
        live = [u for u in trade.unit_entries if u.timestamp[:10] <= day]
        if live:
            marks.append((day, sum(trade.direction * (closes[day] - u.price) / u.n
                                   for u in live)))
    return tuple(marks)


def cap(trades, limit, rng):
    shuffled = list(trades)
    rng.shuffle(shuffled)
    live, taken = [], []
    for trade in sorted(shuffled, key=lambda t: t["entry"]):
        live = [x for x in live if x["exit"] > trade["entry"]]
        if len(live) >= limit:
            continue
        live.append(trade)
        taken.append(trade)
    return taken


def marked_series(taken):
    """Daily R, each trade scaled by the size it was actually taken at."""
    by_day = defaultdict(float)
    for trade in taken:
        weight = trade["weight"]
        previous = 0.0
        for day, open_r in trade["marks"]:
            by_day[day] += (open_r - previous) * weight
            previous = open_r
        by_day[trade["exit"][:10]] += (trade["r"] - previous) * weight
    days = sorted(by_day)
    return days, [by_day[d] for d in days]


def path_metrics(days, values, risk):
    nav, peak, worst = 1000.0, 1000.0, 0.0
    for value in values:
        nav = max(0.0, nav + value * risk * nav)
        peak = max(peak, nav)
        worst = min(worst, nav / peak - 1.0)
    if len(days) < 2:
        return nav, worst, 0.0
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25
    cagr = (nav / 1000.0) ** (1 / years) - 1 if years > 0 and nav > 0 else -1.0
    return nav, worst, cagr


def solve_risk(series, target, lo=1e-6, hi=0.08):
    def dd(risk):
        return statistics.median(abs(path_metrics(d, v, risk)[1]) for d, v in series)
    if dd(hi) < target:
        return hi
    for _ in range(32):
        mid = math.sqrt(lo * hi)
        if dd(mid) < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def sharpe(series):
    scores = []
    for days, values in series:
        if len(days) < 30:
            continue
        span = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days
        stream = values + [0.0] * max(0, int(span * 252 / 365.25) - len(values))
        sd = statistics.pstdev(stream)
        if sd > 0:
            scores.append(statistics.fmean(stream) / sd * math.sqrt(252))
    return statistics.median(scores) if scores else math.nan


def assemble(longs, shorts, labels, long_in, long_out, short_in, short_out):
    """Weight every trade by the regime in force on its entry session."""
    out = []
    for trade in longs:
        inside = labels.get(trade["entry"][:10])
        weight = long_in if inside else long_out
        if weight > 0:
            out.append({**trade, "weight": weight})
    for trade in shorts:
        inside = labels.get(trade["entry"][:10])
        weight = short_in if inside else short_out
        if weight > 0:
            out.append({**trade, "weight": weight})
    return out


def score(pooled, args):
    if len(pooled) < 200:
        return None
    series = [marked_series(cap(pooled, args.max_positions, random.Random(s)))
              for s in range(args.trials)]
    risk = solve_risk(series, args.target_dd)
    years = sorted({d[:4] for d in series[0][0]})
    by_year = {}
    for year in years:
        sliced = [([d for d in days if d[:4] == year],
                   [v for d, v in zip(days, values) if d[:4] == year])
                  for days, values in series]
        by_year[year] = sharpe([s for s in sliced if len(s[0]) >= 30])
    return {"trades": len(pooled), "sharpe": sharpe(series), "risk": risk,
            "cagr": statistics.median(path_metrics(d, v, risk)[2] for d, v in series),
            "years": by_year}


def main(argv=None):
    args = parse_args(argv)
    book = load_book(args)
    sessions = sorted({b.timestamp[:10] for bars in book.values() for b in bars})
    labels = lag_to_sessions(unfavourable(args.macro), sessions)
    share = sum(1 for v in labels.values() if v) / len(labels)
    print(f"{len(book)} instruments, {len(sessions):,} sessions; "
          f"unfavourable regime covers {share:.1%}\n", flush=True)

    sleeves = {}
    for name, directions in (("long", (1,)), ("short", (-1,))):
        config = TurtleConfig(**BASE, directions=directions)
        pooled = []
        for ticker, bars in book.items():
            trades, _ = run_turtle(bars, config=config)
            closes = session_closes(bars)
            pooled.extend({"entry": t.entry_timestamp, "exit": t.exit_timestamp,
                           "r": t.net_r, "marks": trade_marks(t, closes)}
                          for t in trades)
        sleeves[name] = pooled
    half = args.long_weight_in_regime

    variants = {
        "long only (banked)": (labels, 1.0, 1.0, 0.0, 0.0),
        "MACRO TILT (proposed)": (labels, half, 1.0, 1.0, 0.0),
        "tilt, regime reversed": (labels, 1.0, half, 0.0, 1.0),
        "always half long + shorts": (labels, half, half, 1.0, 1.0),
        "shorts added, longs unchanged": (labels, 1.0, 1.0, 1.0, 0.0),
    }
    report = {}
    print(f"  {'variant':32s} {'trades':>8s} {'Sharpe':>7s} {'lev':>8s} {'CAGR':>8s}")
    for name, (lab, li, lo, si, so) in variants.items():
        result = score(assemble(sleeves["long"], sleeves["short"], lab, li, lo, si, so),
                       args)
        report[name] = result
        print(f"  {name:32s} {result['trades']:>8,d} {result['sharpe']:>7.2f} "
              f"{result['risk']:>8.4%} {result['cagr']:>8.1%}", flush=True)

    real = report["MACRO TILT (proposed)" if args.null_variant == "tilt"
                  else "shorts added, longs unchanged"]
    days = sorted(labels)
    values = [labels[d] for d in days]
    n = len(days)
    print(f"\n  running {args.nulls} null regimes through the identical rule...",
          flush=True)
    null_sharpes, null_cagrs = [], []
    rng = random.Random(11)
    for draw in range(args.nulls):
        if draw % 2 == 0:
            offset = rng.randrange(1, n)
            fake = {d: values[(i + offset) % n] for i, d in enumerate(days)}
        else:
            fake = {d: rng.random() < share for d in days}
        weights = ((half, 1.0, 1.0, 0.0) if args.null_variant == "tilt"
                   else (1.0, 1.0, 1.0, 0.0))
        result = score(assemble(sleeves["long"], sleeves["short"], fake, *weights),
                       args)
        if result:
            null_sharpes.append(result["sharpe"])
            null_cagrs.append(result["cagr"])
    null_sharpes.sort()
    null_cagrs.sort()
    beat = sum(1 for s in null_sharpes if s >= real["sharpe"])
    p_value = (beat + 1) / (len(null_sharpes) + 1)
    pick = lambda xs, f: xs[min(int(f * len(xs)), len(xs) - 1)]
    print(f"\n  null Sharpe : p05 {pick(null_sharpes, .05):.2f}  "
          f"median {pick(null_sharpes, .5):.2f}  p95 {pick(null_sharpes, .95):.2f}  "
          f"max {null_sharpes[-1]:.2f}")
    print(f"  real Sharpe : {real['sharpe']:.2f}")
    print(f"  null CAGR   : median {pick(null_cagrs, .5):.1%}  "
          f"p95 {pick(null_cagrs, .95):.1%}   real {real['cagr']:.1%}")
    print(f"\n  empirical p = {p_value:.3f}  "
          f"({beat} of {len(null_sharpes)} random regimes matched or beat it)")

    base = report["long only (banked)"]
    years = sorted(real["years"])
    print(f"\n  Sharpe by year:")
    print("    " + "".join(f"{y:>7s}" for y in years))
    for key, tag in (("long only (banked)", "banked"),
                     ("MACRO TILT (proposed)", "tilt"),
                     ("shorts added, longs unchanged", "shorts+")):
        print("    " + "".join(f"{report[key]['years'][y]:>7.2f}" for y in years)
              + f"   {tag}")
    wins = sum(1 for y in years if real["years"][y] > base["years"][y])
    passed = (real["sharpe"] > base["sharpe"] and p_value < 0.05 and wins >= 7)
    print(f"\n  tilt beats banked in {wins}/{len(years)} years (needed 7)")
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}"
          f"   [beats baseline: {real['sharpe'] > base['sharpe']}, "
          f"p<0.05: {p_value < 0.05}, years: {wins >= 7}]")
    report["null"] = {"p_value": p_value, "n": len(null_sharpes),
                      "median_sharpe": pick(null_sharpes, .5),
                      "p95_sharpe": pick(null_sharpes, .95),
                      "max_sharpe": null_sharpes[-1]}
    report["verdict"] = {"passed": passed, "years_won": wins}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
