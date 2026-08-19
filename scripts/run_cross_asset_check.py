"""Do the non-equity holdings already in the book actually diversify it?

Before buying futures data to add commodities and bonds, it is worth noticing
that the universe already contains them: GLD and SLV for metals, XLE for energy,
TMF and TBT for long and short duration, UVXY for volatility.  If those already
supply an uncorrelated return stream, the consistency problem is a weighting
problem and no download fixes it.  If they do not, the download is justified and
we learn why.

The number that decides it is the correlation between the equity sleeve's daily
R stream and the non-equity sleeve's.  Diversification is not "holding different
things", it is holding things whose *returns* move apart, and a long-only trend
system can easily turn several asset classes into one bet: in a liquidity-driven
rally everything trends up together, and in a shock everything gaps down.

Both sleeves are also scored alone and combined, at matched drawdown, so the
question "would splitting the risk budget across them have produced steadier
years" gets a direct answer rather than an inference from the correlation.
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

EQUITY_ETF = ["SPY", "QQQ", "IWM", "EWJ", "EWT", "EWY", "KWEB"]
OTHER_ETF = ["GLD", "SLV", "XLE", "TMF", "TBT", "UVXY"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/cross_asset_check.json"))
    return parser.parse_args(argv)


def load(args, names):
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    book = {}
    for ticker in names:
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='5min' ORDER BY ts", (ticker,)).fetchall()
        if not rows:
            continue
        five = adjust_bars([Bar(*r) for r in rows], splits.get(ticker, []))
        book[ticker] = resample_regular_session(five, minutes=args.minutes)
    connection.close()
    return book


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


def marked_map(taken):
    by_day = defaultdict(float)
    for trade in taken:
        previous = 0.0
        for day, open_r in trade["marks"]:
            by_day[day] += open_r - previous
            previous = open_r
        by_day[trade["exit"][:10]] += trade["r"] - previous
    return by_day


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


def sharpe_of(days, values, calendar):
    stream = [values.get(d, 0.0) for d in calendar]
    sd = statistics.pstdev(stream)
    return statistics.fmean(stream) / sd * math.sqrt(252) if sd > 0 else float("nan")


def trades_for(book, names, config):
    pooled = []
    for ticker in names:
        if ticker not in book:
            continue
        bars = book[ticker]
        found, _ = run_turtle(bars, config=config)
        closes = session_closes(bars)
        pooled.extend({"entry": t.entry_timestamp, "exit": t.exit_timestamp,
                       "r": t.net_r, "marks": trade_marks(t, closes)} for t in found)
    return pooled


def main(argv=None):
    args = parse_args(argv)
    book = load(args, EQUITY_ETF + OTHER_ETF)
    config = TurtleConfig(**BASE)
    calendar = sorted({b.timestamp[:10] for bars in book.values() for b in bars})

    sleeves = {"equity ETFs": EQUITY_ETF, "non-equity ETFs": OTHER_ETF,
               "both together": EQUITY_ETF + OTHER_ETF}
    pooled = {name: trades_for(book, names, config) for name, names in sleeves.items()}
    print(f"{len(book)} instruments; " + ", ".join(
        f"{k} {len(v):,} trades" for k, v in pooled.items()) + "\n", flush=True)

    maps, report = {}, {}
    print(f"  {'sleeve':18s} {'names':>6s} {'trades':>7s} {'Sharpe':>7s} "
          f"{'CAGR':>8s} {'worst yr':>9s} {'yr sd':>7s}")
    for name, names in sleeves.items():
        caps = [cap(pooled[name], args.max_positions, random.Random(s))
                for s in range(args.trials)]
        series = []
        for taken in caps:
            by_day = marked_map(taken)
            days = sorted(by_day)
            series.append((days, [by_day[d] for d in days]))
        maps[name] = marked_map(caps[0])
        risk = solve_risk(series, args.target_dd)
        years = sorted({d[:4] for d in calendar})
        by_year = {}
        for year in years:
            window = [d for d in calendar if d[:4] == year]
            scores = [sharpe_of(window, marked_map(t), window) for t in caps[:12]]
            live = [s for s in scores if s == s]
            by_year[year] = statistics.median(live) if live else float("nan")
        live = [v for v in by_year.values() if v == v]
        entry = {"names": len([n for n in names if n in book]),
                 "trades": len(pooled[name]),
                 "sharpe": statistics.median(
                     sharpe_of(calendar, marked_map(t), calendar) for t in caps[:12]),
                 "cagr": statistics.median(
                     path_metrics(d, v, risk)[2] for d, v in series),
                 "years": by_year, "worst_year": min(live),
                 "year_sd": statistics.pstdev(live)}
        report[name] = entry
        print(f"  {name:18s} {entry['names']:>6d} {entry['trades']:>7,d} "
              f"{entry['sharpe']:>7.2f} {entry['cagr']:>8.1%} "
              f"{entry['worst_year']:>9.2f} {entry['year_sd']:>7.2f}", flush=True)

    a = [maps["equity ETFs"].get(d, 0.0) for d in calendar]
    b = [maps["non-equity ETFs"].get(d, 0.0) for d in calendar]
    both = [(x, y) for x, y in zip(a, b) if x or y]
    rho = statistics.correlation([x for x, _ in both], [y for _, y in both])
    down = [(x, y) for x, y in both if x < 0]
    rho_down = (statistics.correlation([x for x, _ in down], [y for _, y in down])
                if len(down) > 30 else float("nan"))

    print(f"\n  daily R correlation, equity vs non-equity sleeve: {rho:+.3f}")
    print(f"  correlation on the equity sleeve's DOWN days:      {rho_down:+.3f}"
          f"   ({len(down):,} days)")
    print("\n  Sharpe by year:")
    years = sorted(report["equity ETFs"]["years"])
    print("    " + "".join(f"{y:>7s}" for y in years))
    for name in sleeves:
        print("    " + "".join(f"{report[name]['years'][y]:>7.2f}" for y in years)
              + f"   {name}")
    report["correlation"] = {"all_days": rho, "equity_down_days": rho_down}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
