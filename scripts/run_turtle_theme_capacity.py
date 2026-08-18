"""Does allocating capacity by theme beat a single global position cap?

Raising the cap from six to ten slots also raises gross exposure, so a naive
comparison rewards whichever setting simply carries more risk.  Every policy is
therefore run twice: once at fixed per-position risk, and once with risk scaled
by ``6 / cap`` so that a full book carries the same total risk regardless of how
many slots it is divided into.  Only the matched-risk column answers the
question that was actually asked.

Policies:

* **global cap N**        first come first served across the whole book
* **one per theme**       at most one open position per hand-labelled theme,
                          which forbids six simultaneous semiconductor trades
* **theme sub-accounts**  each theme trades its own fixed pot, so a hot theme
                          cannot borrow capacity from a cold one and its wins
                          compound only within itself
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

from savi_uz.seed_groups import SEED_RISK_GROUPS  # noqa: E402
from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import TurtleConfig, run_turtle  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

#: Names the hand-labelled seed groups miss, kept separate rather than dumped
#: into an existing theme they do not belong to.
EXTRA_THEMES = {"GLD": "Gold", "KWEB": "China internet"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--min-sessions", type=int, default=500)
    parser.add_argument("--risk", type=float, default=0.0005)
    parser.add_argument("--sleeve", type=float, default=0.30)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/turtle_theme_capacity.json"))
    return parser.parse_args(argv)


def theme_of(ticker: str) -> str:
    for name, members in SEED_RISK_GROUPS.items():
        if ticker in members:
            return name
    return EXTRA_THEMES.get(ticker, "Unthemed")


def collect(args):
    splits = load_splits(args.db)
    connection = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    names = [r[0] for r in connection.execute(
        "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker")]
    config = TurtleConfig(entry_window=55, exit_window=20, atr_window=20,
                          skip_after_winner=False, directions=(1,))
    trades = []
    for ticker in names:
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='5min' AND ts>=? AND ts<? ORDER BY ts",
            (ticker, args.start, args.end)).fetchall()
        if not rows:
            continue
        five = adjust_bars([Bar(*r) for r in rows], splits.get(ticker, []))
        if len({b.timestamp[:10] for b in five}) < args.min_sessions:
            continue
        series = resample_regular_session(five, minutes=args.minutes)
        for trade in run_turtle(series, config=config)[0]:
            trades.append({
                "ticker": ticker, "theme": theme_of(ticker),
                "entry": trade.entry_timestamp, "exit": trade.exit_timestamp,
                "net_r": trade.net_r,
            })
    connection.close()
    return trades


def admit(trades, rng, *, cap, per_theme=None, sub_accounts=False):
    """Return accepted trades in exit order, tagged with the pot they used."""
    shuffled = list(trades)
    rng.shuffle(shuffled)
    live = []
    taken = []
    for trade in sorted(shuffled, key=lambda t: t["entry"]):
        live = [x for x in live if x["exit"] > trade["entry"]]
        if sub_accounts:
            if sum(1 for x in live if x["theme"] == trade["theme"]) >= 1:
                continue
        else:
            if len(live) >= cap:
                continue
            if per_theme is not None and sum(
                1 for x in live if x["theme"] == trade["theme"]
            ) >= per_theme:
                continue
        live.append(trade)
        taken.append(trade)
    return taken


def walk(taken, days, risk, *, sub_accounts, themes, sleeve_frac):
    """Compound realised P&L; returns the total-NAV path."""
    trading_total = 1000.0 * (1.0 - sleeve_frac)
    sleeve = 1000.0 * sleeve_frac
    if sub_accounts:
        # Sizing off a 1/len(themes) pot would make every trade risk that much
        # less money, so the comparison would measure position size rather than
        # the allocation structure. Scale risk up so a trade risks the same
        # dollars it would in the single-pot book, isolating the structural
        # effect: capital stranded in cold themes and no cross-theme compounding.
        risk = risk * len(themes)
        pots = {theme: trading_total / len(themes) for theme in themes}
        by_day = defaultdict(list)
        for trade in taken:
            by_day[trade["exit"][:10]].append(trade)
        path = []
        for day in days:
            for trade in by_day.get(day, ()):
                pot = pots[trade["theme"]]
                pots[trade["theme"]] = max(0.0, pot + trade["net_r"] * risk * pot)
            path.append(sleeve + sum(pots.values()))
        return path
    by_day = defaultdict(float)
    for trade in taken:
        by_day[trade["exit"][:10]] += trade["net_r"]
    path = []
    value = trading_total
    for day in days:
        value = max(0.0, value + by_day.get(day, 0.0) * risk * value)
        path.append(sleeve + value)
    return path


def stats(path, days):
    peak, maxdd = path[0], 0.0
    for value in path:
        peak = max(peak, value)
        maxdd = min(maxdd, value / peak - 1.0)
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25
    cagr = (path[-1] / path[0]) ** (1 / years) - 1 if years > 0 and path[-1] > 0 else -1.0
    return {"final": path[-1], "cagr": cagr, "maxdd": maxdd,
            "calmar": cagr / abs(maxdd) if maxdd else math.nan}


def main(argv=None):
    args = parse_args(argv)
    trades = collect(args)
    themes = sorted({t["theme"] for t in trades})
    stamps = sorted({t["exit"][:10] for t in trades})
    start, end = date.fromisoformat(stamps[0]), date.fromisoformat(stamps[-1])
    days, day = [], start
    while day <= end:
        if day.weekday() < 5:
            days.append(day.isoformat())
        day = date.fromordinal(day.toordinal() + 1)
    print(f"{len(trades):,} long-only {args.minutes}-minute signals across "
          f"{len(themes)} themes, {len(days):,} weekdays\n")

    policies = []
    for cap in (6, 8, 10, 12):
        policies.append((f"global cap {cap}", dict(cap=cap)))
    for cap in (8, 10, 12):
        policies.append((f"cap {cap}, max 1 per theme", dict(cap=cap, per_theme=1)))
    policies.append((f"theme sub-accounts ({len(themes)})", dict(cap=99, sub_accounts=True)))

    report = {"themes": themes, "signals": len(trades)}
    print(f"  {'policy':30s} | {'fixed risk':^30s} | {'matched risk':^30s}")
    print(f"  {'':30s} | {'median $':>10s} {'maxDD':>8s} {'Calmar':>8s} "
          f"| {'median $':>10s} {'maxDD':>8s} {'Calmar':>8s}")
    for label, kwargs in policies:
        cap = kwargs.get("cap", 6)
        effective_slots = len(themes) if kwargs.get("sub_accounts") else cap
        row = {}
        for mode in ("fixed", "matched"):
            risk = args.risk if mode == "fixed" else args.risk * 6.0 / effective_slots
            got = defaultdict(list)
            for seed in range(args.trials):
                taken = admit(trades, random.Random(seed), **kwargs)
                path = walk(taken, days, risk, sub_accounts=kwargs.get("sub_accounts", False),
                            themes=themes, sleeve_frac=args.sleeve)
                for key, value in stats(path, days).items():
                    got[key].append(value)
            pick = lambda xs, f: sorted(xs)[int(f * len(xs))]
            row[mode] = {k: {"p05": pick(v, .05), "median": pick(v, .5),
                             "p95": pick(v, .95)} for k, v in got.items()}
            row[mode]["trades"] = len(taken)
        print(f"  {label:30s} | ${row['fixed']['final']['median']:>9,.0f} "
              f"{row['fixed']['maxdd']['median']:>8.1%} "
              f"{row['fixed']['calmar']['median']:>8.2f} "
              f"| ${row['matched']['final']['median']:>9,.0f} "
              f"{row['matched']['maxdd']['median']:>8.1%} "
              f"{row['matched']['calmar']['median']:>8.2f}")
        report[label] = row

    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
