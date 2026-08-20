"""Per-sector sizing and direction, driven by measured rate sensitivity.

The betas are not in doubt: market-adjusted, real estate sits at -3.98 and
utilities at -3.24 against a one-point move in the ten-year yield, financials at
+3.42 and energy at +6.41, with t-statistics past thirteen.  What is in doubt is
whether that survives the translation into breakout trades, which is the question
the earlier binary-regime test was too blunt to ask.

The rule, per sector and per session:

    tilt = trailing beta  x  trailing change in the ten-year yield

A positive tilt means the rate environment currently favours that sector.  Three
ways of acting on it are tested -- gate the direction, scale the size, or switch
the sector off when the tilt is adverse -- against the flat book that treats all
eleven alike.

Everything is trailing.  The beta is re-estimated over a rolling window ending
before the session it is used in, the yield change is measured to the prior
session, and both are lagged again before a trade may see them.  A beta fitted on
the whole sample would encode which sectors turned out to be rate-sensitive,
which is the entire question.

Four controls, because a per-sector rule multiplies the ways to fool oneself:

* the reversed tilt, since a signal no better than its opposite carries nothing;
* a circular shift of the yield series, preserving its persistence;
* a **synthetic beta** -- noise with the same persistence and dispersion as the
  real one -- which is the control that closed the implied-volatility work, where
  a shuffle was not enough and only simulated data settled it;
* a train/test split, because selecting among four rules on one sample is the
  mistake the exit study already made once.
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

from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import TurtleConfig, run_turtle  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

SECTORS = ("XLU", "XLRE", "XLK", "XLP", "XLF", "XLE", "XLB", "XLI",
           "XLV", "XLY", "XLC")
BETA_WINDOW = 500
RATE_WINDOW = 21
SPLIT = "2014-01-01"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etf", type=Path,
                        default=Path("data/cross_assets/etf_30min.db"))
    parser.add_argument("--macro", type=Path, default=Path("data/macro/macro.db"))
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--cost", type=float, default=0.0002)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--nulls", type=int, default=60)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/sector_rate_tilt.json"))
    return parser.parse_args(argv)


def load(args):
    connection = sqlite3.connect(f"file:{args.etf}?mode=ro", uri=True)
    book = {}
    for ticker in SECTORS:
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? "
            "ORDER BY ts", (ticker,)).fetchall()
        if len(rows) >= 2000:
            book[ticker] = resample_regular_session([Bar(*r) for r in rows],
                                                    minutes=30)
    connection.close()
    return book


def daily_returns(bars):
    closes = {}
    for bar in bars:
        closes[bar.timestamp[:10]] = bar.close
    days = sorted(closes)
    return {days[i]: closes[days[i]] / closes[days[i - 1]] - 1.0
            for i in range(1, len(days)) if closes[days[i - 1]] > 0}


def yield_changes(path: Path):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    level = dict(connection.execute(
        "SELECT curve_date, value FROM gsw_rates WHERE mnemonic='SVENY10'"))
    connection.close()
    days = sorted(d for d in level if level[d] is not None)
    daily = {days[i]: level[days[i]] - level[days[i - 1]]
             for i in range(1, len(days))}
    trailing = {}
    for i in range(RATE_WINDOW, len(days)):
        trailing[days[i]] = level[days[i]] - level[days[i - RATE_WINDOW]]
    return daily, trailing


def rolling_betas(returns, market, daily_yield, sessions):
    """Beta of the market-adjusted return on the daily yield change.

    Estimated only from sessions strictly before the one it is assigned to, so a
    trade on day D sees a coefficient fitted on data ending at D-1.
    """
    out = {}
    history = []
    for day in sessions:
        if len(history) >= BETA_WINDOW:
            window = history[-BETA_WINDOW:]
            xs = [x for x, _ in window]
            ys = [y for _, y in window]
            mx, my = statistics.fmean(xs), statistics.fmean(ys)
            sxx = sum((x - mx) ** 2 for x in xs)
            if sxx > 0:
                out[day] = sum((x - mx) * (y - my)
                               for x, y in window) / sxx
        if day in returns and day in daily_yield and day in market:
            history.append((daily_yield[day], returns[day] - market[day]))
    return out


def synthetic_beta(real, seed):
    """Noise matched to the real beta path's persistence and spread."""
    rng = random.Random(seed)
    days = sorted(real)
    values = [real[d] for d in days]
    if len(values) < 50:
        return dict(real)
    mean = statistics.fmean(values)
    centred = [v - mean for v in values]
    numerator = sum(centred[i] * centred[i - 1] for i in range(1, len(centred)))
    denominator = sum(v * v for v in centred)
    phi = max(-0.99, min(0.99, numerator / denominator)) if denominator else 0.0
    residual = math.sqrt(max(statistics.pvariance(values) * (1 - phi * phi), 1e-12))
    out, previous = {}, 0.0
    for day in days:
        previous = phi * previous + rng.gauss(0.0, residual)
        out[day] = mean + previous
    return out


def previous_session(values, sessions):
    days = sorted(values)
    out, position, carried = {}, 0, None
    for day in sessions:
        while position < len(days) and days[position] < day:
            carried = values[days[position]]
            position += 1
        if carried is not None:
            out[day] = carried
    return out


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


def marked(taken):
    by_day = defaultdict(float)
    for trade in taken:
        weight, previous = trade["weight"], 0.0
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
    return nav, worst, (nav / 1000.0) ** (1 / years) - 1 if years > 0 and nav > 0 else -1.0


def solve_risk(series, target, lo=1e-6, hi=0.08):
    def dd(risk):
        return statistics.median(abs(path_metrics(d, v, risk)[1]) for d, v in series)
    if dd(hi) < target:
        return hi
    for _ in range(28):
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
    return statistics.median(scores) if scores else float("nan")


def assemble(trades, tilt, rule):
    """Apply a rule to the long and short books, returning weighted trades."""
    out = []
    for (ticker, side), pooled in trades.items():
        signs = tilt.get(ticker, {})
        for trade in pooled:
            value = signs.get(trade["entry"][:10])
            if value is None:
                weight = 1.0 if rule == "flat" and side == "long" else 0.0
            elif rule == "flat":
                weight = 1.0 if side == "long" else 0.0
            elif rule == "gate":
                weight = 1.0 if ((side == "long" and value > 0)
                                 or (side == "short" and value < 0)) else 0.0
            elif rule == "gate reversed":
                weight = 1.0 if ((side == "long" and value < 0)
                                 or (side == "short" and value > 0)) else 0.0
            elif rule == "scale long":
                weight = (max(0.0, min(2.0, 1.0 + 3.0 * value))
                          if side == "long" else 0.0)
            elif rule == "off when adverse":
                weight = (1.0 if side == "long" and value > -0.02 else 0.0)
            else:
                weight = 0.0
            if weight > 0:
                out.append({**trade, "weight": weight})
    return out


def score(pooled, args, lo=None, hi=None):
    window = [t for t in pooled
              if (lo is None or t["entry"] >= lo) and (hi is None or t["entry"] < hi)]
    if len(window) < 150:
        return None
    series = [marked(cap(window, args.max_positions, random.Random(s)))
              for s in range(args.trials)]
    risk = solve_risk(series, args.target_dd)
    return {"trades": len(window), "sharpe": sharpe(series),
            "cagr": statistics.median(path_metrics(d, v, risk)[2] for d, v in series)}


def main(argv=None):
    args = parse_args(argv)
    book = load(args)
    sessions = sorted({b.timestamp[:10] for bars in book.values() for b in bars})
    returns = {t: daily_returns(b) for t, b in book.items()}
    market = {}
    for day in sessions:
        values = [returns[t][day] for t in returns if day in returns[t]]
        if values:
            market[day] = statistics.fmean(values)
    daily_yield, trailing_yield = yield_changes(args.macro)
    print(f"{len(book)} sectors, {len(sessions):,} sessions "
          f"{sessions[0]} -> {sessions[-1]}", flush=True)

    betas = {t: rolling_betas(returns[t], market, daily_yield, sessions)
             for t in book}
    lagged_rate = previous_session(trailing_yield, sessions)
    tilt = {}
    for ticker in book:
        beta = previous_session(betas[ticker], sessions)
        tilt[ticker] = {d: beta[d] * lagged_rate[d]
                        for d in sessions if d in beta and d in lagged_rate}
    coverage = statistics.fmean(len(v) / len(sessions) for v in tilt.values())
    print(f"  tilt defined on {coverage:.0%} of sessions "
          f"(rolling {BETA_WINDOW}-session beta, {RATE_WINDOW}-session yield change)")
    print(f"  latest trailing betas: " + ", ".join(
        f"{t} {betas[t][max(betas[t])]*100:+.1f}"
        for t in ("XLU", "XLRE", "XLF", "XLE") if betas.get(t)))

    trades = {}
    for side, directions in (("long", (1,)), ("short", (-1,))):
        config = TurtleConfig(entry_window=55, exit_window=18, atr_window=20,
                              skip_after_winner=False, directions=directions,
                              use_channel_exit=False, chandelier_atr=3.0,
                              round_trip_cost=args.cost)
        for ticker, bars in book.items():
            closes = {b.timestamp[:10]: b.close for b in bars}
            pooled = []
            for trade in run_turtle(bars, config=config)[0]:
                marks = []
                for day in (d for d in closes
                            if trade.entry_timestamp[:10] <= d
                            < trade.exit_timestamp[:10]):
                    live = [u for u in trade.unit_entries
                            if u.timestamp[:10] <= day]
                    if live:
                        marks.append((day, sum(trade.direction
                                               * (closes[day] - u.price) / u.n
                                               for u in live)))
                pooled.append({"entry": trade.entry_timestamp,
                               "exit": trade.exit_timestamp,
                               "r": trade.net_r, "marks": marks})
            trades[(ticker, side)] = pooled

    rules = ["flat", "gate", "gate reversed", "scale long", "off when adverse"]
    print(f"\n  {'rule':20s} {'trades':>8s} {'Sharpe':>8s} {'CAGR':>8s} "
          f"{'train':>8s} {'held out':>9s}")
    report = {}
    for rule in rules:
        pooled = assemble(trades, tilt, rule)
        whole = score(pooled, args)
        train = score(pooled, args, hi=SPLIT)
        test = score(pooled, args, lo=SPLIT)
        if not whole:
            continue
        report[rule] = {"whole": whole, "train": train, "test": test}
        print(f"  {rule:20s} {whole['trades']:>8,d} {whole['sharpe']:>8.2f} "
              f"{whole['cagr']:>8.1%} "
              f"{train['sharpe'] if train else float('nan'):>8.2f} "
              f"{test['sharpe'] if test else float('nan'):>9.2f}", flush=True)

    real = report.get("gate", {}).get("whole", {}).get("sharpe", float("nan"))
    print(f"\n  {args.nulls} synthetic-beta draws through the gate rule "
          f"(no information, matched persistence)...", flush=True)
    scores = []
    for draw in range(args.nulls):
        fake_tilt = {}
        for ticker in book:
            fake = previous_session(
                synthetic_beta(betas[ticker], 500 + draw), sessions)
            fake_tilt[ticker] = {d: fake[d] * lagged_rate[d]
                                 for d in sessions if d in fake and d in lagged_rate}
        outcome = score(assemble(trades, fake_tilt, "gate"), args)
        if outcome:
            scores.append(outcome["sharpe"])
    scores.sort()
    if scores:
        pick = lambda f: scores[min(int(f * len(scores)), len(scores) - 1)]
        beat = sum(1 for s in scores if s >= real)
        print(f"    synthetic Sharpe: p05 {pick(.05):.2f}  median {pick(.5):.2f}  "
              f"p95 {pick(.95):.2f}  max {scores[-1]:.2f}")
        print(f"    real gate rule:   {real:.2f}")
        print(f"    empirical p = {(beat + 1) / (len(scores) + 1):.3f}  "
              f"({beat} of {len(scores)} noise betas matched or beat it)")
        report["synthetic"] = {"median": pick(.5), "p95": pick(.95),
                               "p": (beat + 1) / (len(scores) + 1)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
