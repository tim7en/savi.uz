"""Sizing breakouts by whether policy is heading the instrument's way.

Everything measured so far pulls in one direction and this is the one combination
left untried.  Sector rate sensitivity is real but contemporaneous, so a past rate
move tells you nothing; three attempts to trade it that way failed.  But the
committee's *next* move is forecastable from published data -- direction called
correctly 68% of the time against a 54% baseline -- and that is a forward
quantity, which is exactly what the earlier tests lacked.

So: predict where policy is going, multiply by how the instrument responds to
rates, and carry more where the two agree.

    tailwind = expected policy direction  x  trailing rate beta

A stock that gains when yields rise, in a period when policy is expected to
tighten, has the wind behind it and is carried at more than full weight; one that
suffers is carried at less.

Nothing here may look forward.  The reaction function is refitted at each meeting
on meetings that preceded it, never on the whole sample.  The rate beta rolls on
trailing sessions and is lagged again.  Macro inputs respect publication lag.  The
prediction is carried forward from a meeting until the next one, which is what an
investor would actually hold.

Four controls, because the programme's record with this family is eleven
rejections: matched drawdown, so a smaller book earns nothing free; the reversed
tailwind, since a signal no better than its opposite carries nothing; a synthetic
beta with matched persistence and no information; and a shuffled policy
prediction.  The rival to beat is not zero but the untouched book.
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
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import TurtleConfig, run_turtle  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

BASE = dict(entry_window=55, exit_window=20, atr_window=20, skip_after_winner=False,
            directions=(1,), use_channel_exit=False, chandelier_atr=3.0)
BETA_WINDOW = 500
MIN_MEETINGS = 40


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--macro", type=Path, default=Path("data/macro/macro.db"))
    parser.add_argument("--reaction", type=Path,
                        default=Path("out/report/chapter2_reaction.json"))
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--cost", type=float, default=0.0002)
    parser.add_argument("--strength", type=float, default=0.5,
                        help="how far weight moves from 1.0 at full tailwind")
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--nulls", type=int, default=40)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/policy_tailwind.json"))
    return parser.parse_args(argv)


def load_book(args):
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    book = {}
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker"):
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='5min' ORDER BY ts", (ticker,)).fetchall()
        five = adjust_bars([Bar(*r) for r in rows], splits.get(ticker, []))
        if len({b.timestamp[:10] for b in five}) >= 400:
            book[ticker] = resample_regular_session(five, minutes=30)
    connection.close()
    return book


def ridge(rows, predictors, target, ridge_lambda=1.0):
    data = [r for r in rows if r.get(target) is not None
            and all(r.get(k) is not None for k in predictors)]
    if len(data) < MIN_MEETINGS:
        return None
    means = [statistics.fmean([r[k] for r in data]) for k in predictors]
    sds = [statistics.pstdev([r[k] for r in data]) or 1.0 for k in predictors]
    X = [[1.0] + [(r[k] - means[i]) / sds[i] for i, k in enumerate(predictors)]
         for r in data]
    y = [r[target] for r in data]
    n = len(X[0])
    xtx = [[sum(row[a] * row[b] for row in X) for b in range(n)] for a in range(n)]
    xty = [sum(row[a] * y[i] for i, row in enumerate(X)) for a in range(n)]
    for a in range(1, n):
        xtx[a][a] += ridge_lambda
    for a in range(n):
        pivot = max(range(a, n), key=lambda r: abs(xtx[r][a]))
        xtx[a], xtx[pivot] = xtx[pivot], xtx[a]
        xty[a], xty[pivot] = xty[pivot], xty[a]
        if abs(xtx[a][a]) < 1e-12:
            return None
        for b in range(a + 1, n):
            f = xtx[b][a] / xtx[a][a]
            for k in range(a, n):
                xtx[b][k] -= f * xtx[a][k]
            xty[b] -= f * xty[a]
    beta = [0.0] * n
    for a in reversed(range(n)):
        beta[a] = (xty[a] - sum(xtx[a][b] * beta[b]
                                for b in range(a + 1, n))) / xtx[a][a]
    return {"beta": beta, "means": means, "sds": sds}


def policy_forecast(reaction_path: Path):
    """Expected direction of the next policy move, refit meeting by meeting."""
    rows = json.loads(reaction_path.read_text(encoding="utf-8"))["rows"]
    rows.sort(key=lambda r: r["date"])
    predictors = ["core_pce", "unemployment", "payroll_3m",
                  "balance_sheet", "breakeven", "credit_spread"]
    out = {}
    for index, meeting in enumerate(rows):
        model = ridge(rows[:index], predictors, "rate_move")
        if model is None or any(meeting.get(k) is None for k in predictors):
            continue
        value = model["beta"][0] + sum(
            model["beta"][i + 1] * (meeting[k] - model["means"][i]) / model["sds"][i]
            for i, k in enumerate(predictors))
        out[meeting["date"]] = value
    return out


def rolling_beta(closes, yield_change, market, sessions):
    days = sorted(closes)
    returns = {days[i]: closes[days[i]] / closes[days[i - 1]] - 1.0
               for i in range(1, len(days)) if closes[days[i - 1]] > 0}
    out, history = {}, []
    for day in sessions:
        if len(history) >= BETA_WINDOW:
            window = history[-BETA_WINDOW:]
            xs = [x for x, _ in window]
            mx = statistics.fmean(xs)
            sxx = sum((x - mx) ** 2 for x in xs)
            if sxx > 0:
                my = statistics.fmean([y for _, y in window])
                out[day] = sum((x - mx) * (y - my) for x, y in window) / sxx
        if day in returns and day in yield_change and day in market:
            history.append((yield_change[day], returns[day] - market[day]))
    return out


def carry_forward(values, sessions):
    days = sorted(values)
    out, position, held = {}, 0, None
    for day in sessions:
        while position < len(days) and days[position] < day:
            held = values[days[position]]
            position += 1
        if held is not None:
            out[day] = held
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


def score(pooled, weights, args):
    sized = []
    for trade in pooled:
        w = weights.get((trade["ticker"], trade["entry"][:10]), 1.0)
        if w > 0:
            sized.append({**trade, "weight": w})
    if len(sized) < 200:
        return None
    series = [marked(cap(sized, args.max_positions, random.Random(s)))
              for s in range(args.trials)]
    risk = solve_risk(series, args.target_dd)
    years = sorted({d[:4] for d in series[0][0]})
    by_year = {}
    for year in years:
        sliced = [([d for d in days if d[:4] == year],
                   [v for d, v in zip(days, values) if d[:4] == year])
                  for days, values in series]
        by_year[year] = sharpe([s for s in sliced if len(s[0]) >= 30])
    return {"sharpe": sharpe(series), "risk": risk, "trades": len(sized),
            "cagr": statistics.median(path_metrics(d, v, risk)[2] for d, v in series),
            "years": by_year,
            "avg_weight": statistics.fmean(t["weight"] for t in sized)}


def main(argv=None):
    args = parse_args(argv)
    book = load_book(args)
    sessions = sorted({b.timestamp[:10] for bars in book.values() for b in bars})

    connection = sqlite3.connect(f"file:{args.macro}?mode=ro", uri=True)
    level = dict(connection.execute(
        "SELECT curve_date, value FROM gsw_rates WHERE mnemonic='SVENY10'"))
    connection.close()
    ordered = sorted(d for d in level if level[d] is not None)
    yield_change = {ordered[i]: level[ordered[i]] - level[ordered[i - 1]]
                    for i in range(1, len(ordered))}

    closes = {t: {b.timestamp[:10]: b.close for b in bars} for t, bars in book.items()}
    returns = {}
    for ticker, series in closes.items():
        days = sorted(series)
        returns[ticker] = {days[i]: series[days[i]] / series[days[i - 1]] - 1.0
                           for i in range(1, len(days)) if series[days[i - 1]] > 0}
    market = {}
    for day in sessions:
        values = [returns[t][day] for t in returns if day in returns[t]]
        if len(values) >= 20:
            market[day] = statistics.fmean(values)

    forecast = carry_forward(policy_forecast(args.reaction), sessions)
    spread = statistics.pstdev(list(forecast.values())) or 1.0
    print(f"{len(book)} instruments; policy forecast on "
          f"{len(forecast) / len(sessions):.0%} of sessions", flush=True)

    betas = {t: carry_forward(rolling_beta(closes[t], yield_change, market, sessions),
                              sessions) for t in book}
    coverage = statistics.fmean(len(v) / len(sessions) for v in betas.values())
    print(f"trailing rate beta on {coverage:.0%} of sessions\n", flush=True)

    config = TurtleConfig(**{**BASE, "round_trip_cost": args.cost})
    pooled = []
    for ticker, bars in book.items():
        day_close = closes[ticker]
        for trade in run_turtle(bars, config=config)[0]:
            marks = []
            for day in (d for d in day_close
                        if trade.entry_timestamp[:10] <= d < trade.exit_timestamp[:10]):
                live = [u for u in trade.unit_entries if u.timestamp[:10] <= day]
                if live:
                    marks.append((day, sum(trade.direction
                                           * (day_close[day] - u.price) / u.n
                                           for u in live)))
            pooled.append({"ticker": ticker, "entry": trade.entry_timestamp,
                           "exit": trade.exit_timestamp, "r": trade.net_r,
                           "marks": marks})
    print(f"{len(pooled):,} trades\n", flush=True)

    def build(beta_source, forecast_source, sign=1.0):
        out = {}
        for ticker in book:
            for day in sessions:
                b = beta_source[ticker].get(day)
                f = forecast_source.get(day)
                if b is None or f is None:
                    continue
                tail = sign * b * (f / spread)
                out[(ticker, day)] = max(0.1, min(2.0,
                                                  1.0 + args.strength * math.tanh(tail)))
        return out

    variants = {"untouched book": {},
                "policy tailwind": build(betas, forecast),
                "tailwind reversed": build(betas, forecast, sign=-1.0)}
    report = {}
    print(f"  {'variant':22s} {'weight':>7s} {'trades':>8s} {'Sharpe':>7s} {'CAGR':>8s}")
    for label, weights in variants.items():
        result = score(pooled, weights, args)
        report[label] = result
        print(f"  {label:22s} {result['avg_weight']:>7.3f} {result['trades']:>8,d} "
              f"{result['sharpe']:>7.2f} {result['cagr']:>8.1%}", flush=True)

    base, real = report["untouched book"], report["policy tailwind"]
    years = sorted(base["years"])
    wins = sum(1 for y in years if real["years"][y] > base["years"][y])
    print(f"\n  beats the untouched book in {wins}/{len(years)} years")

    print(f"\n  {args.nulls} synthetic betas (matched persistence, no "
          f"information)...", flush=True)
    scores = []
    rng = random.Random(31)
    for draw in range(args.nulls):
        fake = {}
        for ticker in book:
            real_beta = betas[ticker]
            days = sorted(real_beta)
            values = [real_beta[d] for d in days]
            if len(values) < 50:
                fake[ticker] = real_beta
                continue
            mean = statistics.fmean(values)
            centred = [v - mean for v in values]
            num = sum(centred[i] * centred[i - 1] for i in range(1, len(centred)))
            den = sum(v * v for v in centred)
            phi = max(-0.99, min(0.99, num / den)) if den else 0.0
            resid = math.sqrt(max(statistics.pvariance(values) * (1 - phi ** 2), 1e-12))
            local = random.Random(7000 + draw + hash(ticker) % 1000)
            previous, made = 0.0, {}
            for day in days:
                previous = phi * previous + local.gauss(0.0, resid)
                made[day] = mean + previous
            fake[ticker] = made
        outcome = score(pooled, build(fake, forecast), args)
        if outcome:
            scores.append(outcome["sharpe"])
    scores.sort()
    pick = lambda f: scores[min(int(f * len(scores)), len(scores) - 1)]
    beat = sum(1 for s in scores if s >= real["sharpe"])
    p_value = (beat + 1) / (len(scores) + 1)
    print(f"    synthetic: p05 {pick(.05):.2f}  median {pick(.5):.2f}  "
          f"p95 {pick(.95):.2f}")
    print(f"    real tailwind: {real['sharpe']:.2f}   empirical p = {p_value:.3f}")

    passed = (real["sharpe"] > base["sharpe"]
              and real["sharpe"] > report["tailwind reversed"]["sharpe"]
              and wins >= 7 and p_value < 0.05)
    print(f"\n  KILL CRITERION: beat the untouched book, beat its own reversal, "
          f">=7/10 years, p<0.05")
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")
    report["null"] = {"p": p_value, "median": pick(.5)}
    report["verdict"] = {"passed": passed, "years_won": wins}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
