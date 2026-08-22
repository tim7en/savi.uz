"""Do names with heavier option activity move further, and trend more cleanly?

Two different claims live inside that question and they need separating, because
a name can move a long way while going nowhere.

*Move stronger* is a magnitude claim: the absolute forward move, scaled by the
name's own volatility so a quiet utility and a broken biotech are on one axis.

*Trend stronger* is a path claim, and the efficiency ratio is the standard
measure of it: net change over the sum of the absolute daily changes.  One means
every session pushed the same way; near zero means the same distance was covered
by thrashing.  A rule that buys breakouts cares about this one, because it is
what separates a trend from a range that happened to be wide.

Two cuts, and they answer different things.

*Between names.*  Does MSTR, which carries an enormous option book, trend better
than WMT, which does not?  This is the question as usually asked, and it is the
weaker test: there are 42 names, so 42 points, and any correlation across them
is one draw of a very small sample dressed up as a large one.

*Within a name, over time.*  When a name's own option activity runs hot against
its own recent norm, does what follows trend better than that same name usually
does?  This has the sample size, it removes the between-name confound entirely,
and it is the version a trading rule could act on.

Three activity measures, because "option activity" is ambiguous.  Contract
volume against its own trailing mean; contract volume against *share* volume,
which is the closest thing to a measure of how option-driven the name is; and
volume against open interest, which separates new positioning from churn.

Everything is read from the prior session's close, and the forward window starts
the next session, so nothing reads its own outcome.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_vol_stretch_zones as data  # noqa: E402

HORIZONS = (10, 20)
ACTIVITY = ("optvol_rel", "optvol_per_share", "oi_turnover")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--options", type=Path,
                        default=Path("data/options/alphavantage.db"))
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--zone-minutes", type=int, default=30)
    parser.add_argument("--window", type=int, default=60,
                        help="trailing sessions each activity measure is scaled by")
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/option_activity_trend.json"))
    return parser.parse_args(argv)


def ratio(numerator, denominator):
    if denominator is None or math.isnan(denominator) or denominator <= 0:
        return math.nan
    return numerator / denominator


def trailing(values, window):
    out = [math.nan] * len(values)
    running, live = 0.0, 0
    for index, value in enumerate(values):
        running += value
        live += 1
        if index >= window:
            running -= values[index - window]
            live -= 1
        if live == window:
            out[index] = running / window
    return out


def activity_panel(path, tickers, start, window):
    """Per name and session, three readings of how busy the option book was."""
    connection = sqlite3_connect(path)
    panel = {}
    for ticker in tickers:
        rows = connection.execute(
            "SELECT observation_date, total_volume, total_oi FROM av_daily "
            "WHERE symbol=? AND observation_date>=? ORDER BY observation_date",
            (ticker, start[:10])).fetchall()
        if len(rows) < window + 40:
            continue
        days = [r[0][:10] for r in rows]
        volume = [float(r[1]) if r[1] is not None else math.nan for r in rows]
        interest = [float(r[2]) if r[2] is not None else math.nan for r in rows]
        clean = [0.0 if math.isnan(v) else v for v in volume]
        mean = trailing(clean, window)
        panel[ticker] = {
            day: {"opt_volume": volume[i], "opt_oi": interest[i],
                  "optvol_rel": ratio(volume[i], mean[i]),
                  "oi_turnover": ratio(volume[i], interest[i])}
            for i, day in enumerate(days)}
    connection.close()
    return panel


def sqlite3_connect(path):
    import sqlite3
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def efficiency(closes, start, horizon):
    """Net change over the sum of absolute daily changes: 1 trends, 0 thrashes."""
    travelled = sum(abs(closes[i] - closes[i - 1])
                    for i in range(start + 1, start + horizon + 1))
    if travelled <= 0:
        return math.nan
    return abs(closes[start + horizon] - closes[start]) / travelled


def build(book, panel, realized, args):
    rows = []
    for ticker, (_, _, daily) in book.items():
        table, history = panel.get(ticker), realized.get(ticker)
        if not table or not history:
            continue
        closes = [b.close for b in daily]
        share_volume = [float(b.volume or 0.0) for b in daily]
        share_mean = trailing(share_volume, args.window)
        for index in range(args.window + 1, len(daily) - max(HORIZONS) - 1):
            previous = daily[index - 1].timestamp[:10]
            snapshot = table.get(previous)
            deviation = history.get(previous)
            if not snapshot or not deviation or deviation <= 0:
                continue
            share_rel = ratio(share_volume[index - 1], share_mean[index - 1])
            row = {
                "ticker": ticker, "day": daily[index].timestamp[:10],
                "optvol_rel": snapshot["optvol_rel"],
                "oi_turnover": snapshot["oi_turnover"],
                "optvol_per_share": ratio(snapshot["opt_volume"],
                                          share_volume[index - 1]),
                "share_rel": share_rel,
            }
            base = closes[index - 1]
            for horizon in HORIZONS:
                row[f"eff_{horizon}"] = efficiency(closes, index - 1, horizon)
                row[f"move_{horizon}"] = ratio(
                    abs(closes[index - 1 + horizon] - base),
                    deviation * math.sqrt(horizon) * base)
            rows.append(row)
    return rows


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    position = 0
    while position < len(order):
        stop = position
        while (stop + 1 < len(order)
               and values[order[stop + 1]] == values[order[position]]):
            stop += 1
        mean = (position + stop) / 2
        for k in range(position, stop + 1):
            out[order[k]] = mean
        position = stop + 1
    return out


def spearman(pairs):
    clean = [(x, y) for x, y in pairs
             if not math.isnan(x) and not math.isnan(y)
             and not math.isinf(x) and not math.isinf(y)]
    if len(clean) < 30:
        return math.nan, len(clean)
    xs, ys = _ranks([r[0] for r in clean]), _ranks([r[1] for r in clean])
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in xs)
                            * sum((b - my) ** 2 for b in ys))
    return (numerator / denominator if denominator else math.nan), len(clean)


def within_name(rows, key, target):
    """Rank correlation inside each name, then the median across names.

    Pooling would let the busiest names dominate and quietly turn this back into
    the between-name question.
    """
    per_name = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["ticker"]].append(row)
    for ticker, chunk in grouped.items():
        rho, count = spearman([(r[key], r[target]) for r in chunk])
        if not math.isnan(rho) and count >= 200:
            per_name.append((rho, ticker))
    if not per_name:
        return math.nan, 0, 0.0
    values = [r for r, _ in per_name]
    positive = sum(1 for v in values if v > 0) / len(values)
    return statistics.median(values), len(per_name), positive


def main(argv=None) -> int:
    args = parse_args(argv)
    book = data.load(args)
    realized = data.realized_moves(book)
    panel = activity_panel(args.options, sorted(book), args.start, args.window)
    rows = build(book, panel, realized, args)
    print(f"{len(panel)} names with an option book, {len(rows):,d} sessions\n")
    report = {"between": {}, "within": {}}

    print("########## between names: does a busier option book trend better? "
          "##########")
    print("  42 points.  Treat every number here as one draw of a small sample.")
    per_name = defaultdict(dict)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["ticker"]].append(row)
    for ticker, chunk in grouped.items():
        for key in ACTIVITY:
            values = [r[key] for r in chunk
                      if not math.isnan(r[key]) and not math.isinf(r[key])]
            if values:
                per_name[ticker][key] = statistics.median(values)
        for horizon in HORIZONS:
            for stem in ("eff", "move"):
                values = [r[f"{stem}_{horizon}"] for r in chunk
                          if not math.isnan(r[f"{stem}_{horizon}"])]
                if values:
                    per_name[ticker][f"{stem}_{horizon}"] = statistics.fmean(values)

    print(f"  {'activity measure':20s} " +
          " ".join(f"{'eff ' + str(h):>10s}" for h in HORIZONS) +
          " ".join(f"{'move ' + str(h):>11s}" for h in HORIZONS))
    for key in ACTIVITY:
        cells = []
        for stem in ("eff", "move"):
            for horizon in HORIZONS:
                rho, count = spearman([(v[key], v[f"{stem}_{horizon}"])
                                       for v in per_name.values()
                                       if key in v and f"{stem}_{horizon}" in v])
                cells.append(rho)
                report["between"][f"{key}|{stem}_{horizon}"] = {
                    "rho": rho, "n": count}
        print(f"  {key:20s} {cells[0]:>+10.3f} {cells[1]:>+10.3f} "
              f"{cells[2]:>+11.3f} {cells[3]:>+11.3f}")

    ranked = sorted(per_name.items(),
                    key=lambda kv: -kv[1].get("optvol_per_share", 0.0))
    print("\n  most and least option-driven, by contracts per share of volume")
    print(f"    {'ticker':8s} {'opt/share':>10s} {'eff 20':>8s} {'move 20':>9s}")
    for ticker, values in ranked[:5] + ranked[-5:]:
        print(f"    {ticker:8s} {values.get('optvol_per_share', float('nan')):>10.4f} "
              f"{values.get('eff_20', float('nan')):>8.3f} "
              f"{values.get('move_20', float('nan')):>9.3f}")

    print("\n########## within a name, over time ##########")
    print("  Median rank correlation across names, and the share of names")
    print("  agreeing on the sign.  This is the version with the sample size.")
    print(f"  {'activity measure':20s} {'target':10s} {'median rho':>11s} "
          f"{'names':>7s} {'share +':>9s}")
    for key in ACTIVITY + ("share_rel",):
        for horizon in HORIZONS:
            for stem in ("eff", "move"):
                rho, count, positive = within_name(rows, key, f"{stem}_{horizon}")
                if math.isnan(rho):
                    continue
                report["within"][f"{key}|{stem}_{horizon}"] = {
                    "median_rho": rho, "names": count, "share_positive": positive}
                if horizon == 20:
                    print(f"  {key:20s} {stem + ' 20':10s} {rho:>+11.3f} "
                          f"{count:>7d} {positive:>9.0%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
