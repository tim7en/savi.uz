"""How much of the result is hindsight in the universe?

Forty-two instruments were chosen in 2026 and then backtested from 2017.  Nine of
them had not listed yet, so they could not have been traded at all; the other
thirty-three were picked knowing how the decade went.  Neither problem is visible
in any performance statistic, and both inflate every number this programme has
produced.

Three universes separate the two effects:

* **all 42** -- as banked, with both biases intact;
* **full history** -- the 33 that existed in January 2017, removing the names it
  was impossible to hold but keeping the hindsight in which survivors were named;
* **ETFs only** -- the 13 index, sector, commodity and rates vehicles, which carry
  almost no single-name selection bias: an index fund is picked for what it tracks
  rather than for how its constituents turned out, and its constituents are
  refreshed by the index provider rather than by the person running the backtest.

The ETF book is the honest one.  It is also the interesting one for a different
reason -- it spans equities, gold, silver, energy, rates and foreign markets,
which is the diversification a trend system is supposed to live on, so it doubles
as a first look at whether this works outside US equities.

Everything is scored at matched drawdown per calendar year, as always: a smaller
universe carries less risk and would otherwise be flattered for it.
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
            directions=(1,), use_channel_exit=False, chandelier_atr=3.0)

ETFS = {"SPY", "QQQ", "IWM", "GLD", "SLV", "XLE", "EWJ", "EWT", "EWY", "KWEB",
        "TMF", "TBT", "UVXY"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--min-sessions", type=int, default=400)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/selection_bias.json"))
    return parser.parse_args(argv)


def load_book(args):
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    names = [r[0] for r in connection.execute(
        "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker")]
    book, first = {}, {}
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
        first[ticker] = rows[0][0][:10]
    connection.close()
    return book, first


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
    by_day = defaultdict(float)
    for trade in taken:
        previous = 0.0
        for day, open_r in trade["marks"]:
            by_day[day] += open_r - previous
            previous = open_r
        by_day[trade["exit"][:10]] += trade["r"] - previous
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


def evaluate(book, names, args):
    config = TurtleConfig(**BASE)
    pooled = []
    for ticker in names:
        bars = book[ticker]
        trades, _ = run_turtle(bars, config=config)
        closes = session_closes(bars)
        pooled.extend({"entry": t.entry_timestamp, "exit": t.exit_timestamp,
                       "r": t.net_r, "marks": trade_marks(t, closes)} for t in trades)
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
    live = [v for v in by_year.values() if v == v]
    return {"names": len(names), "trades": len(pooled), "sharpe": sharpe(series),
            "risk": risk, "years": by_year,
            "cagr": statistics.median(path_metrics(d, v, risk)[2] for d, v in series),
            "worst_year": min(live) if live else float("nan"),
            "year_sd": statistics.pstdev(live) if len(live) > 1 else float("nan")}


def main(argv=None):
    args = parse_args(argv)
    book, first = load_book(args)
    full = sorted(t for t in book if first[t] <= "2017-01-10")
    etfs = sorted(t for t in book if t in ETFS)
    universes = [("all 42 (as banked)", sorted(book)),
                 ("full history only", full),
                 ("ETFs only", etfs),
                 ("single names only", sorted(t for t in book if t not in ETFS))]
    print(f"{len(book)} instruments; {len(full)} with history from Jan 2017; "
          f"{len(etfs)} ETFs\n", flush=True)

    report = {}
    print(f"  {'universe':22s} {'names':>6s} {'trades':>8s} {'Sharpe':>7s} "
          f"{'CAGR':>8s} {'worst yr':>9s} {'yr sd':>7s}")
    for label, names in universes:
        result = evaluate(book, names, args)
        report[label] = result
        print(f"  {label:22s} {result['names']:>6d} {result['trades']:>8,d} "
              f"{result['sharpe']:>7.2f} {result['cagr']:>8.1%} "
              f"{result['worst_year']:>9.2f} {result['year_sd']:>7.2f}", flush=True)

    years = sorted(report["all 42 (as banked)"]["years"])
    print(f"\n  Sharpe by year:")
    print("    " + "".join(f"{y:>7s}" for y in years))
    for label, _ in universes:
        cells = "".join(f"{report[label]['years'].get(y, float('nan')):>7.2f}"
                        for y in years)
        print(f"    {cells}   {label}")

    banked, honest = report["all 42 (as banked)"], report["ETFs only"]
    print(f"\n  hindsight premium: {banked['cagr'] - honest['cagr']:+.1f}pp CAGR, "
          f"{banked['sharpe'] - honest['sharpe']:+.2f} Sharpe")
    print(f"  consistency: worst year {banked['worst_year']:.2f} (banked) vs "
          f"{honest['worst_year']:.2f} (ETFs); "
          f"year-to-year sd {banked['year_sd']:.2f} vs {honest['year_sd']:.2f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
