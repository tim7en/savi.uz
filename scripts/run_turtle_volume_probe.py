"""Is a volume burst at a Turtle breakout worth anything?

Rather than building a filter and discovering afterwards that it only reduced
the trade count, this measures the signal inside the trades the system already
takes.  Each breakout is tagged with

* **relative volume**, both a tradeable version using volume through the bar
  *before* entry and a lookahead version using the breakout bar's own volume.
  The system enters intrabar on a stop order, so the breakout bar's total volume
  is not knowable at entry: the lookahead column exists only to put a ceiling on
  what this idea could ever be worth.  If the ceiling is flat there is nothing
  to build.
* **location against a composite volume profile** built from the previous three
  or five completed sessions, so a breakout can be classed as leaving value or
  merely moving inside it.

Outcomes are then bucketed by those tags.  If high-volume breakouts out of value
really do continue, their mean R has to separate from the rest.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.swing_failure_strategy import composite_profiles  # noqa: E402
from savi_uz.turtle import TurtleConfig, relative_volume, run_turtle  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

CONFIG = TurtleConfig(entry_window=55, exit_window=20, atr_window=20,
                      skip_after_winner=False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--min-sessions", type=int, default=500)
    parser.add_argument("--volume-window", type=int, default=20)
    parser.add_argument("--profile-sessions", type=int, default=3)
    parser.add_argument("--outdir", type=Path, default=Path("out/strategy"))
    return parser.parse_args(argv)


def bucket_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    gains = sum(v for v in values if v > 0)
    losses = -sum(v for v in values if v < 0)
    return {
        "n": len(values),
        "mean_r": statistics.mean(values),
        "median_r": statistics.median(values),
        "win_rate": sum(1 for v in values if v > 0) / len(values),
        "pf": (gains / losses) if losses else None,
        "total_r": sum(values),
    }


def main(argv=None):
    args = parse_args(argv)
    splits = load_splits(args.db)
    connection = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    names = [row[0] for row in connection.execute(
        "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker")]

    tagged: dict[str, list[dict]] = {"daily": [], "30-minute": []}
    for ticker in names:
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='5min' AND ts>=? AND ts<? ORDER BY ts", (ticker, args.start, args.end),
        ).fetchall()
        if not rows:
            continue
        five = adjust_bars([Bar(*row) for row in rows], splits.get(ticker, []))
        if len({b.timestamp[:10] for b in five}) < args.min_sessions:
            continue
        profiles = composite_profiles(five, args.profile_sessions)

        for label, minutes in (("daily", 390), ("30-minute", 30)):
            series = resample_regular_session(five, minutes=minutes)
            rel = relative_volume(series, args.volume_window)
            index_of = {bar.timestamp: i for i, bar in enumerate(series)}
            trades, _ = run_turtle(series, config=CONFIG)
            for trade in trades:
                i = index_of.get(trade.entry_timestamp)
                if i is None or i < args.volume_window + 1:
                    continue
                prior = rel[i - 1]
                same = rel[i]
                profile = profiles.get(trade.entry_timestamp[:10])
                if profile is None or prior != prior or same != same:
                    continue
                price = trade.entry
                if price > profile.value_high:
                    zone = "above value"
                elif price < profile.value_low:
                    zone = "below value"
                else:
                    zone = "inside value"
                leaving = (
                    (trade.direction > 0 and zone == "above value")
                    or (trade.direction < 0 and zone == "below value")
                )
                tagged[label].append({
                    "ticker": ticker,
                    "direction": trade.direction,
                    "net_r": trade.net_r,
                    "prior_rel_volume": prior,
                    "entry_rel_volume": same,
                    "zone": zone,
                    "leaving_value": leaving,
                    "poc_distance_n": ((price - profile.poc) / trade.n_at_entry
                                       if trade.n_at_entry else 0.0),
                })
    connection.close()

    report = {}
    for label, rows in tagged.items():
        if not rows:
            continue
        print(f"\n{'=' * 74}\n{label}: {len(rows)} tagged trades")
        block = {"trades": len(rows), "overall": bucket_stats([r["net_r"] for r in rows])}
        base = block["overall"]
        print(f"  baseline: mean {base['mean_r']:+.3f}R  PF {base['pf']:.2f}  "
              f"win {base['win_rate']:.1%}")

        for key, title in (("prior_rel_volume", "TRADEABLE  volume through the prior bar"),
                           ("entry_rel_volume", "LOOKAHEAD  breakout bar's own volume")):
            ordered = sorted(rows, key=lambda r: r[key])
            size = len(ordered) // 5
            print(f"\n  {title}, by quintile")
            print(f"    {'quintile':>9s} {'rel vol':>16s} {'n':>5s} {'mean R':>8s} "
                  f"{'PF':>6s} {'win':>7s}")
            quints = []
            for q in range(5):
                chunk = ordered[q * size:(q + 1) * size] if q < 4 else ordered[4 * size:]
                stats = bucket_stats([r["net_r"] for r in chunk])
                span = f"{chunk[0][key]:.2f}-{chunk[-1][key]:.2f}"
                pf = f"{stats['pf']:.2f}" if stats["pf"] else "inf"
                print(f"    {q + 1:>9d} {span:>16s} {stats['n']:5d} "
                      f"{stats['mean_r']:+8.3f} {pf:>6s} {stats['win_rate']:>7.1%}")
                quints.append({"span": span, **stats})
            block[key] = quints

        print(f"\n  location against the {args.profile_sessions}-session composite profile")
        print(f"    {'zone':>28s} {'n':>5s} {'mean R':>8s} {'PF':>6s} {'win':>7s}")
        loc = {}
        groups = defaultdict(list)
        for r in rows:
            tag = "leaving value" if r["leaving_value"] else (
                "inside value" if r["zone"] == "inside value" else "against value")
            groups[tag].append(r["net_r"])
        for tag in ("leaving value", "inside value", "against value"):
            stats = bucket_stats(groups[tag])
            if not stats["n"]:
                continue
            pf = f"{stats['pf']:.2f}" if stats["pf"] else "inf"
            print(f"    {tag:>28s} {stats['n']:5d} {stats['mean_r']:+8.3f} "
                  f"{pf:>6s} {stats['win_rate']:>7.1%}")
            loc[tag] = stats
        block["location"] = loc

        print("\n  the actual hypothesis: leaving value AND volume expanding")
        print(f"    {'cell':>28s} {'n':>5s} {'mean R':>8s} {'PF':>6s} {'win':>7s}")
        cells = {}
        for leaving in (True, False):
            for hot in (True, False):
                sub = [r["net_r"] for r in rows
                       if r["leaving_value"] == leaving
                       and (r["prior_rel_volume"] >= 1.0) == hot]
                stats = bucket_stats(sub)
                if not stats["n"]:
                    continue
                name = ("leaving" if leaving else "not leaving") + \
                       (" + volume up" if hot else " + volume flat")
                pf = f"{stats['pf']:.2f}" if stats["pf"] else "inf"
                print(f"    {name:>28s} {stats['n']:5d} {stats['mean_r']:+8.3f} "
                      f"{pf:>6s} {stats['win_rate']:>7.1%}")
                cells[name] = stats
        block["hypothesis"] = cells

        print("\n  by side, top volume quintile only (tradeable)")
        ordered = sorted(rows, key=lambda r: r["prior_rel_volume"])
        top = ordered[4 * (len(ordered) // 5):]
        for side, sign in (("long", 1), ("short", -1)):
            allside = bucket_stats([r["net_r"] for r in rows if r["direction"] == sign])
            hotside = bucket_stats([r["net_r"] for r in top if r["direction"] == sign])
            if not allside["n"] or not hotside["n"]:
                continue
            print(f"    {side:>8s}  all: n={allside['n']:4d} mean {allside['mean_r']:+.3f}R"
                  f"   top-quintile volume: n={hotside['n']:4d} "
                  f"mean {hotside['mean_r']:+.3f}R")
            block[f"{side}_all"] = allside
            block[f"{side}_hot"] = hotside
        report[label] = block

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "turtle_volume_probe.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
