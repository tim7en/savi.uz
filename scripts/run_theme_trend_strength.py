"""Do trades in strongly trending themes outperform trades in weak ones?

A per-theme capacity cap already failed: forcing the book to spread across
themes costs more in declined signals than it saves in correlated risk.  The
opposite question is the interesting one.  If some themes trend harder than
others, then leaning *into* the strongest theme is concentration that pays, and
the book should be allowed - or encouraged - to stack there.

This measures the signal before building anything.  Every long 30-minute Turtle
trade is tagged with three descriptions of its theme, each computed from an
equal-weight daily index of the theme's members using only sessions strictly
before the entry date:

* **momentum**    the theme index return over the prior ``lookback`` sessions
* **efficiency**  Kaufman's ratio, net move divided by the sum of absolute daily
                  moves, which separates a clean trend from a choppy one that
                  travelled the same distance
* **breadth**     the share of members trading above their own 50-session mean
* **rank**        where the theme sat among all themes that day on momentum

If leaning into strength works, mean R has to rise with these.
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

from savi_uz.seed_groups import SEED_RISK_GROUPS  # noqa: E402
from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import TurtleConfig, run_turtle  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

EXTRA_THEMES = {"GLD": "Gold", "KWEB": "China internet"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--min-sessions", type=int, default=500)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--breadth-window", type=int, default=50)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/theme_trend_strength.json"))
    return parser.parse_args(argv)


def theme_of(ticker):
    for name, members in SEED_RISK_GROUPS.items():
        if ticker in members:
            return name
    return EXTRA_THEMES.get(ticker, "Unthemed")


def load_universe(args):
    splits = load_splits(args.db)
    connection = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    names = [r[0] for r in connection.execute(
        "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker")]
    daily, intraday = {}, {}
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
        daily[ticker] = resample_regular_session(five, minutes=390)
        intraday[ticker] = resample_regular_session(five, minutes=args.minutes)
    connection.close()
    return daily, intraday


def theme_indices(daily):
    """Equal-weight daily index per theme, as {theme: {session: level}}."""
    members = defaultdict(list)
    for ticker in daily:
        members[theme_of(ticker)].append(ticker)
    result = {}
    for theme, names in members.items():
        # Average of each member's normalised close, so a high-priced name does
        # not dominate the index.
        series = defaultdict(list)
        for ticker in names:
            bars = daily[ticker]
            base = bars[0].close
            for bar in bars:
                if base > 0:
                    series[bar.timestamp[:10]].append(bar.close / base)
        result[theme] = {day: sum(vals) / len(vals)
                         for day, vals in series.items() if vals}
    return result, members


def strength(index, lookback):
    """Momentum and Kaufman efficiency per session, using only earlier data."""
    days = sorted(index)
    levels = [index[d] for d in days]
    momentum, efficiency = {}, {}
    for i in range(lookback + 1, len(days)):
        window = levels[i - lookback - 1:i]          # ends at day i-1
        net = window[-1] - window[0]
        travel = sum(abs(window[j] - window[j - 1]) for j in range(1, len(window)))
        if window[0] > 0:
            momentum[days[i]] = net / window[0]
        efficiency[days[i]] = (abs(net) / travel) if travel > 0 else 0.0
    return momentum, efficiency


def breadth_map(daily, members, window):
    """Share of a theme's members above their own trailing mean, per session."""
    above = defaultdict(dict)
    for ticker, bars in daily.items():
        closes = [b.close for b in bars]
        running = 0.0
        for i, bar in enumerate(bars):
            running += closes[i]
            if i >= window:
                running -= closes[i - window]
            if i >= window:
                # Compare the previous close with the mean ending there.
                above[ticker][bar.timestamp[:10]] = closes[i - 1] > running / window
    result = defaultdict(dict)
    for theme, names in members.items():
        days = set()
        for ticker in names:
            days |= set(above[ticker])
        for day in days:
            flags = [above[t][day] for t in names if day in above[t]]
            if flags:
                result[theme][day] = sum(flags) / len(flags)
    return result


def buckets(rows, key, labels=5):
    ordered = [r for r in rows if r[key] is not None]
    ordered.sort(key=lambda r: r[key])
    size = len(ordered) // labels
    out = []
    for q in range(labels):
        chunk = ordered[q * size:(q + 1) * size] if q < labels - 1 else ordered[(labels - 1) * size:]
        values = [r["net_r"] for r in chunk]
        gains = sum(v for v in values if v > 0)
        losses = -sum(v for v in values if v < 0)
        out.append({
            "span": f"{chunk[0][key]:+.3f} to {chunk[-1][key]:+.3f}",
            "n": len(values),
            "mean_r": statistics.mean(values),
            "win": sum(1 for v in values if v > 0) / len(values),
            "pf": gains / losses if losses else None,
            "total_r": sum(values),
        })
    return out


def main(argv=None):
    args = parse_args(argv)
    daily, intraday = load_universe(args)
    indices, members = theme_indices(daily)
    momentum, efficiency = {}, {}
    for theme, index in indices.items():
        momentum[theme], efficiency[theme] = strength(index, args.lookback)
    breadth = breadth_map(daily, members, args.breadth_window)

    config = TurtleConfig(entry_window=55, exit_window=20, atr_window=20,
                          skip_after_winner=False, directions=(1,))
    rows = []
    for ticker, series in intraday.items():
        theme = theme_of(ticker)
        for trade in run_turtle(series, config=config)[0]:
            day = trade.entry_timestamp[:10]
            rows.append({
                "ticker": ticker, "theme": theme, "day": day,
                "net_r": trade.net_r,
                "momentum": momentum[theme].get(day),
                "efficiency": efficiency[theme].get(day),
                "breadth": breadth[theme].get(day),
            })

    # Rank each trade's theme against the other themes on the same day.
    per_day = defaultdict(dict)
    for theme, values in momentum.items():
        for day, value in values.items():
            per_day[day][theme] = value
    for row in rows:
        ranking = per_day.get(row["day"], {})
        if len(ranking) < 4 or row["momentum"] is None:
            row["rank_pct"] = None
            continue
        ordered = sorted(ranking.values())
        below = sum(1 for v in ordered if v < row["momentum"])
        row["rank_pct"] = below / (len(ordered) - 1)

    tagged = [r for r in rows if r["momentum"] is not None]
    values = [r["net_r"] for r in tagged]
    gains = sum(v for v in values if v > 0)
    losses = -sum(v for v in values if v < 0)
    print(f"{len(tagged):,} tagged long 30-minute trades across "
          f"{len(indices)} themes")
    print(f"baseline: mean {statistics.mean(values):+.3f}R  "
          f"PF {gains / losses:.2f}  win {sum(1 for v in values if v > 0) / len(values):.1%}\n")

    report = {"baseline_mean_r": statistics.mean(values), "trades": len(tagged)}
    for key, title in (
        ("momentum", f"theme momentum over the prior {args.lookback} sessions"),
        ("efficiency", "theme trend efficiency (clean trend vs chop)"),
        ("breadth", f"share of members above their {args.breadth_window}-session mean"),
        ("rank_pct", "theme's momentum rank among all themes that day"),
    ):
        table = buckets(tagged, key)
        print(f"  {title}")
        print(f"    {'quintile':>9s} {'range':>22s} {'n':>6s} {'mean R':>9s} "
              f"{'PF':>6s} {'win':>7s}")
        for i, b in enumerate(table, 1):
            pf = f"{b['pf']:.2f}" if b["pf"] else "inf"
            print(f"    {i:>9d} {b['span']:>22s} {b['n']:6,d} {b['mean_r']:+9.3f} "
                  f"{pf:>6s} {b['win']:>7.1%}")
        spread = table[-1]["mean_r"] - table[0]["mean_r"]
        print(f"    top-minus-bottom quintile: {spread:+.3f}R\n")
        report[key] = {"buckets": table, "spread": spread}

    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
