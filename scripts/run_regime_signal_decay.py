"""Where does the macro effect go between the asset and the trade?

The published evidence on sector rate-sensitivity is not in doubt: utilities and
real estate carry negative rate beta, financials gain from a steeper curve.  Our
own regime tests found nothing.  Both can be true, because they measure different
things, and this measures them side by side on identical regime labels to find
out which step destroys the signal.

Four measurements of the same underlying claim, in order of how much processing
sits between the market and the number:

1. **Raw daily return.**  What the literature measures.  Helped basket minus hurt
   basket, inside the regime versus outside.
2. **Volatility-normalised return.**  The same daily return divided by the
   instrument's ATR, which is what expressing anything in R units does.  If the
   effect survives step 1 and dies here, the strategy is blind to macro because
   it normalises by the very quantity macro moves.
3. **Return on days a position was open.**  Adds the entry filter: breakouts only
   fire after price has already moved, so the sample is conditioned on movement.
4. **Realised trade R.**  What we measured before, adding the exit rule and the
   holding period.

Each step is scored against the same circular-shift null so the numbers are
comparable, and the drop between consecutive steps is the diagnostic.
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

from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import TurtleConfig, run_turtle, wilder_atr  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

HURT = ("XLU", "XLRE", "XLK", "XLP")
HELPED = ("XLF", "XLE", "XLB", "XLI")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etf", type=Path,
                        default=Path("data/cross_assets/etf_30min.db"))
    parser.add_argument("--macro", type=Path, default=Path("data/macro/macro.db"))
    parser.add_argument("--nulls", type=int, default=300)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/regime_signal_decay.json"))
    return parser.parse_args(argv)


def load(args):
    connection = sqlite3.connect(f"file:{args.etf}?mode=ro", uri=True)
    book = {}
    for ticker in HURT + HELPED:
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? "
            "ORDER BY ts", (ticker,)).fetchall()
        if len(rows) >= 2000:
            book[ticker] = resample_regular_session([Bar(*r) for r in rows],
                                                    minutes=30)
    connection.close()
    return book


def regimes_from(path: Path):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    horizons = defaultdict(dict)
    for day, horizon, rate in connection.execute(
        "SELECT curve_date, horizon_months, forward_rate FROM fed_path "
        "WHERE horizon_months IN (3, 12)"
    ):
        horizons[day][horizon] = rate
    connection.close()
    days = sorted(horizons)
    level = {d: horizons[d].get(12) for d in days}
    return {
        "tightening priced": {
            d: v[12] > v[3] for d, v in horizons.items()
            if v.get(3) is not None and v.get(12) is not None},
        "policy path rising": {
            d: level[d] > level[days[max(0, i - 63)]] for i, d in enumerate(days)
            if level[d] is not None and level[days[max(0, i - 63)]] is not None},
    }


def lag(labels, sessions):
    days = sorted(labels)
    out, position, carried = {}, 0, None
    for day in sessions:
        while position < len(days) and days[position] < day:
            carried = labels[days[position]]
            position += 1
        if carried is not None:
            out[day] = carried
    return out


def per_instrument_series(bars):
    """Daily return, volatility-normalised return, and the ATR level."""
    atr = wilder_atr(bars, 20)
    by_day = defaultdict(list)
    index_of = {}
    for i, bar in enumerate(bars):
        by_day[bar.timestamp[:10]].append(bar.close)
        index_of.setdefault(bar.timestamp[:10], i)
    days = sorted(by_day)
    raw, normalised = {}, {}
    for i in range(1, len(days)):
        previous, current = by_day[days[i - 1]][-1], by_day[days[i]][-1]
        if previous <= 0:
            continue
        move = current / previous - 1.0
        raw[days[i]] = move * 100.0
        n = atr[max(0, index_of[days[i]] - 1)]
        if n == n and n > 0:
            normalised[days[i]] = (current - previous) / n
    return raw, normalised


def basket_gap(series_by_ticker, labels, names, restrict=None):
    """Mean value inside the regime minus outside, averaged across a basket."""
    gaps = []
    for ticker in names:
        series = series_by_ticker.get(ticker)
        if not series:
            continue
        allowed = restrict.get(ticker) if restrict else None
        inside = [v for d, v in series.items()
                  if labels.get(d) is True and (allowed is None or d in allowed)]
        outside = [v for d, v in series.items()
                   if labels.get(d) is False and (allowed is None or d in allowed)]
        if len(inside) >= 50 and len(outside) >= 50:
            gaps.append(statistics.fmean(inside) - statistics.fmean(outside))
    return statistics.fmean(gaps) if gaps else None


def shifted(labels, offset):
    days = sorted(labels)
    values = [labels[d] for d in days]
    return {d: values[(i + offset) % len(days)] for i, d in enumerate(days)}


def score(series_by_ticker, labels, restrict, nulls, rng):
    statistic = ((basket_gap(series_by_ticker, labels, HELPED, restrict) or 0.0)
                 - (basket_gap(series_by_ticker, labels, HURT, restrict) or 0.0))
    draws = []
    for _ in range(nulls):
        fake = shifted(labels, rng.randrange(200, max(400, len(labels))))
        helped = basket_gap(series_by_ticker, fake, HELPED, restrict)
        hurt = basket_gap(series_by_ticker, fake, HURT, restrict)
        if helped is not None and hurt is not None:
            draws.append(helped - hurt)
    if not draws:
        return statistic, float("nan"), float("nan")
    draws.sort()
    spread = statistics.pstdev(draws)
    beat = sum(1 for v in draws if v >= statistic)
    return statistic, (beat + 1) / (len(draws) + 1), (
        statistic / spread if spread > 0 else float("nan"))


def main(argv=None):
    args = parse_args(argv)
    book = load(args)
    sessions = sorted({b.timestamp[:10] for bars in book.values() for b in bars})
    print(f"{len(book)} sector funds, {len(sessions):,} sessions "
          f"{sessions[0]} -> {sessions[-1]}", flush=True)

    raw, normalised, open_days, trade_r = {}, {}, {}, {}
    config = TurtleConfig(entry_window=55, exit_window=18, atr_window=20,
                          skip_after_winner=False, directions=(1,),
                          use_channel_exit=False, chandelier_atr=3.0,
                          round_trip_cost=0.0002)
    for ticker, bars in book.items():
        raw[ticker], normalised[ticker] = per_instrument_series(bars)
        held, entries = set(), []
        for trade in run_turtle(bars, config=config)[0]:
            entries.append({"entry": trade.entry_timestamp[:10], "r": trade.net_r})
            for day in sessions:
                if trade.entry_timestamp[:10] <= day <= trade.exit_timestamp[:10]:
                    held.add(day)
        open_days[ticker] = held
        trade_r[ticker] = {}
        for item in entries:
            trade_r[ticker].setdefault(item["entry"], []).append(item["r"])
        trade_r[ticker] = {d: statistics.fmean(v) for d, v in trade_r[ticker].items()}

    regimes = {name: lag(labels, sessions)
               for name, labels in regimes_from(args.macro).items()}

    report = {}
    for name, labels in regimes.items():
        print(f"\n{'=' * 78}\nREGIME: {name}")
        print(f"  {'measurement':38s} {'helped-hurt':>12s} {'in null sd':>11s} "
              f"{'p':>7s}")
        rng = random.Random(17)
        steps = [
            ("1. raw daily return, % ", raw, None),
            ("2. same return, divided by ATR", normalised, None),
            ("3. ATR return, position-open days only", normalised, open_days),
            ("4. realised trade R", trade_r, None),
        ]
        rows = {}
        for label, series, restrict in steps:
            statistic, p_value, in_sd = score(series, labels, restrict,
                                              args.nulls, rng)
            rows[label] = {"stat": statistic, "p": p_value, "sd": in_sd}
            print(f"  {label:38s} {statistic:>+12.4f} {in_sd:>11.2f} "
                  f"{p_value:>7.3f}", flush=True)
        report[name] = rows
        first, last = rows[steps[0][0]], rows[steps[-1][0]]
        if first["sd"] == first["sd"] and last["sd"] == last["sd"]:
            print(f"\n  effect size falls from {first['sd']:.2f} to "
                  f"{last['sd']:.2f} null standard deviations "
                  f"across the four steps")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
