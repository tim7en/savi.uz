"""Implied volatility as the risk unit, against Wilder's N.

N is the whole risk architecture of this system: position size is ``risk / N``,
the hard stop is ``2N``, pyramids space at ``half N`` and the trail sits at
``3N``.  Replacing it therefore changes entry sizing, stop placement and exit
distance in one move, which is the substitution this tests.

The earlier options work closed a narrower question.  Conditioning the *trail
multiplier* on a volatility state failed, and failed even when handed an
instrument's own realised volatility -- the ceiling on what any option chain
could estimate.  That verdict does not carry here.  It said a volatility state
adds nothing on top of N; this asks whether a better volatility estimate makes a
better N in the first place, which was step two of the original plan and was
never run.

There is a real reason to expect a difference.  Wilder's ATR is a twenty-bar
backward average: it learns that volatility rose only after it has risen.  Implied
volatility is a forward quote, so it can mark a jump on the day it is priced --
before earnings, into a Fed meeting -- which is exactly when a backward average
is most wrong.

Scale is controlled rather than tested.  Implied volatility annualised in percent
and an ATR in dollars per thirty-minute bar are different objects, and swapping
one for the other raw would change the average size of every position, so the
comparison would confound level with timing.  Each implied series is therefore
rescaled to the trailing mean of that instrument's own ATR, leaving average risk
identical and only the *timing* of volatility changes to differ.

Look-ahead: the option snapshot is end-of-day for session D and is read only by
bars in session D+1 or later.  The rescaling window is trailing.
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
from savi_uz.turtle import TurtleConfig, run_turtle, wilder_atr  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

BASE = dict(entry_window=55, exit_window=20, atr_window=20, skip_after_winner=False,
            directions=(1,), use_channel_exit=False, chandelier_atr=3.0)

#: Bars per session at 30 minutes, and sessions per year, for annualisation.
BARS_PER_SESSION = 13
SESSIONS_PER_YEAR = 252
SCALE_WINDOW = 250


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--options", type=Path,
                        default=Path("data/options/alphavantage.db"))
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--min-sessions", type=int, default=400)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--cost", type=float, default=0.0002)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--nulls", type=int, default=0,
                        help="circular-shift draws of the implied series; the "
                             "shift keeps its persistence and destroys only its "
                             "alignment to price")
    parser.add_argument("--noise", type=int, default=0,
                        help="synthetic persistence-matched risk units; the "
                             "decisive control if a shuffled series already "
                             "beats the ATR")
    parser.add_argument("--gamma-sweep", action="store_true",
                        help="sweep the gamma coefficient, reversal included")
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/implied_n.json"))
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


def option_series(path: Path, symbols):
    """Per-symbol daily option features, keyed by session."""
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    out = {}
    for symbol in symbols:
        rows = connection.execute(
            "SELECT observation_date, atm_iv, gamma_balance, net_gex, spot "
            "FROM av_daily WHERE symbol=? ORDER BY observation_date",
            (symbol,)).fetchall()
        out[symbol] = {r[0]: {"atm_iv": r[1], "gamma_balance": r[2],
                              "net_gex": r[3], "spot": r[4]} for r in rows}
    connection.close()
    return out


def realised_session_vol(bars):
    """Close-to-close volatility within each session, in price units per bar."""
    by_day = defaultdict(list)
    for bar in bars:
        by_day[bar.timestamp[:10]].append(bar.close)
    out = {}
    for day, closes in by_day.items():
        moves = [abs(closes[i] - closes[i - 1]) for i in range(1, len(closes))]
        if moves:
            out[day] = statistics.fmean(moves)
    return out


def implied_bar_move(iv_percent, spot):
    """Expected absolute move over one bar, from an annualised implied vol."""
    if iv_percent is None or spot is None or iv_percent <= 0 or spot <= 0:
        return None
    years = 1.0 / (SESSIONS_PER_YEAR * BARS_PER_SESSION)
    return spot * (iv_percent / 100.0) * math.sqrt(years)


def rescaled(raw_by_day, atr_by_day, days):
    """Put a forecast on the ATR's scale using only its own past.

    Without this the comparison would be between two different risk levels
    rather than two forecasts, and the level alone moves drawdown.
    """
    out, seen_raw, seen_atr = {}, [], []
    for day in days:
        raw = raw_by_day.get(day)
        atr_value = atr_by_day.get(day)
        if (len(seen_raw) >= SCALE_WINDOW and raw is not None
                and statistics.fmean(seen_raw[-SCALE_WINDOW:]) > 0):
            factor = (statistics.fmean(seen_atr[-SCALE_WINDOW:])
                      / statistics.fmean(seen_raw[-SCALE_WINDOW:]))
            out[day] = raw * factor
        if raw is not None and atr_value is not None:
            seen_raw.append(raw)
            seen_atr.append(atr_value)
    return out


def previous_session_map(values, sessions):
    """Each session takes the value of the last session strictly before it."""
    days = sorted(values)
    out, position, carried = {}, 0, None
    for day in sessions:
        while position < len(days) and days[position] < day:
            carried = values[days[position]]
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


def evaluate(book, overrides, args, label):
    config = TurtleConfig(**{**BASE, "round_trip_cost": args.cost})
    pooled, covered = [], 0
    for ticker, bars in book.items():
        supplied = overrides.get(ticker) if overrides else None
        if supplied:
            covered += len(supplied)
        trades, _ = run_turtle(bars, config=config, n_by_bar=supplied)
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
    return {"label": label, "trades": len(pooled), "bars_overridden": covered,
            "sharpe": sharpe(series), "risk": risk,
            "cagr": statistics.median(path_metrics(d, v, risk)[2] for d, v in series),
            "years": by_year, "worst_year": min(live) if live else float("nan")}


def build_units(book, features, coefficient=0.35, shift=0):
    """Per-bar N from implied volatility, optionally gamma-scaled or shifted."""
    out = {}
    for ticker, bars in book.items():
        atr = wilder_atr(bars, BASE["atr_window"])
        atr_by_day, index_of_day = {}, defaultdict(list)
        for i, bar in enumerate(bars):
            day = bar.timestamp[:10]
            index_of_day[day].append(i)
            if not math.isnan(atr[i]) and atr[i] > 0:
                atr_by_day[day] = atr[i]
        sessions = sorted(index_of_day)
        rows = features.get(ticker, {})
        raw = {}
        for day in sessions:
            row = rows.get(day)
            if not row:
                continue
            move = implied_bar_move(row["atm_iv"], row["spot"])
            if move is None:
                continue
            balance = row["gamma_balance"]
            if coefficient and balance is not None:
                move *= 1.0 - coefficient * max(-1.0, min(1.0, balance))
            raw[day] = move
        if shift:
            days = sorted(raw)
            values = [raw[d] for d in days]
            raw = {d: values[(i + shift) % len(days)] for i, d in enumerate(days)}
        scaled = rescaled(raw, atr_by_day, sessions)
        lagged = previous_session_map(scaled, sessions)
        out[ticker] = {i: lagged[day] for day in sessions if day in lagged
                       for i in index_of_day[day]}
    return out


def synthetic_units(book, features, seed):
    """A risk unit made of noise, matched to implied volatility's own shape.

    The circular-shift null already beats Wilder's N, which means the gain may
    have nothing to do with forecasting anything -- any smooth, mean-reverting,
    positively-skewed series rescaled onto the ATR's level might do it.  This
    builds exactly that and nothing more: an AR(1) fitted to each instrument's
    log implied volatility, then re-simulated with independent innovations, so
    persistence, dispersion and skew survive while every trace of alignment to
    real volatility is gone.

    If this also beats the ATR, the finding belongs to the ATR being a poor risk
    unit rather than to option data, and implied volatility is closed.
    """
    rng = random.Random(seed)
    out = {}
    for ticker, bars in book.items():
        atr = wilder_atr(bars, BASE["atr_window"])
        atr_by_day, index_of_day = {}, defaultdict(list)
        for i, bar in enumerate(bars):
            day = bar.timestamp[:10]
            index_of_day[day].append(i)
            if not math.isnan(atr[i]) and atr[i] > 0:
                atr_by_day[day] = atr[i]
        sessions = sorted(index_of_day)
        rows = features.get(ticker, {})

        observed = []
        for day in sessions:
            row = rows.get(day)
            if row:
                move = implied_bar_move(row["atm_iv"], row["spot"])
                if move and move > 0:
                    observed.append((day, math.log(move)))
        if len(observed) < 300:
            out[ticker] = {}
            continue

        values = [v for _, v in observed]
        mean = statistics.fmean(values)
        centred = [v - mean for v in values]
        lag_numerator = sum(centred[i] * centred[i - 1] for i in range(1, len(centred)))
        lag_denominator = sum(v * v for v in centred)
        phi = max(-0.99, min(0.99, lag_numerator / lag_denominator))
        residual = math.sqrt(max(statistics.pvariance(values) * (1 - phi * phi), 1e-12))

        simulated, previous = {}, 0.0
        for day, _ in observed:
            previous = phi * previous + rng.gauss(0.0, residual)
            simulated[day] = math.exp(mean + previous)

        scaled = rescaled(simulated, atr_by_day, sessions)
        lagged = previous_session_map(scaled, sessions)
        out[ticker] = {i: lagged[day] for day in sessions if day in lagged
                       for i in index_of_day[day]}
    return out


def main(argv=None):
    args = parse_args(argv)
    book = load_book(args)
    features = option_series(args.options, sorted(book))
    print(f"{len(book)} instruments at {args.minutes}-minute bars", flush=True)

    # Per instrument: the ATR in daily terms, the implied bar move, and a
    # realised-volatility rival built only from price.
    implied, realised, gamma_scaled = {}, {}, {}
    for ticker, bars in book.items():
        atr = wilder_atr(bars, BASE["atr_window"])
        atr_by_day, index_of_day = {}, defaultdict(list)
        for i, bar in enumerate(bars):
            day = bar.timestamp[:10]
            index_of_day[day].append(i)
            if not math.isnan(atr[i]) and atr[i] > 0:
                atr_by_day[day] = atr[i]
        sessions = sorted(index_of_day)
        rows = features.get(ticker, {})

        raw_implied = {}
        raw_gamma = {}
        for day in sessions:
            row = rows.get(day)
            if not row:
                continue
            move = implied_bar_move(row["atm_iv"], row["spot"])
            if move is not None:
                raw_implied[day] = move
                balance = row["gamma_balance"]
                # Negative dealer gamma is the state that precedes the largest
                # moves; widen the unit there and tighten it when gamma is long.
                if balance is not None:
                    raw_gamma[day] = move * (1.0 - 0.35 * max(-1.0, min(1.0, balance)))
        raw_realised = realised_session_vol(bars)

        for source, target in ((raw_implied, implied), (raw_realised, realised),
                               (raw_gamma, gamma_scaled)):
            scaled = rescaled(source, atr_by_day, sessions)
            lagged = previous_session_map(scaled, sessions)
            target[ticker] = {i: lagged[day] for day in sessions if day in lagged
                              for i in index_of_day[day]}

    total_bars = sum(len(b) for b in book.values())
    print(f"  implied N covers {sum(len(v) for v in implied.values()) / total_bars:.0%}"
          f" of bars, realised {sum(len(v) for v in realised.values()) / total_bars:.0%}"
          f"\n", flush=True)

    variants = [("Wilder N (banked)", None),
                ("implied-vol N", implied),
                ("implied N, gamma-scaled", gamma_scaled),
                ("realised-vol N (CONTROL)", realised)]
    report = {}
    print(f"  {'risk unit':28s} {'trades':>8s} {'Sharpe':>7s} {'lev':>8s} "
          f"{'CAGR':>8s} {'worst yr':>9s}")
    for label, overrides in variants:
        result = evaluate(book, overrides, args, label)
        report[label] = result
        print(f"  {label:28s} {result['trades']:>8,d} {result['sharpe']:>7.2f} "
              f"{result['risk']:>8.4%} {result['cagr']:>8.1%} "
              f"{result['worst_year']:>9.2f}", flush=True)

    base = report["Wilder N (banked)"]
    control = report["realised-vol N (CONTROL)"]
    years = sorted(base["years"])
    print(f"\n  Sharpe by year:")
    print("    " + "".join(f"{y:>7s}" for y in years))
    for label, _ in variants:
        print("    " + "".join(f"{report[label]['years'][y]:>7.2f}" for y in years)
              + f"   {label}")
    for label, _ in variants[1:]:
        wins = sum(1 for y in years if report[label]["years"][y] > base["years"][y])
        over = sum(1 for y in years
                   if report[label]["years"][y] > control["years"][y])
        print(f"\n  {label}: beats Wilder in {wins}/{len(years)} years, "
              f"beats the realised-vol control in {over}/{len(years)}")
    winner = max(variants[1:], key=lambda v: report[v[0]]["sharpe"])[0]
    passed = (report[winner]["sharpe"] > base["sharpe"]
              and report[winner]["sharpe"] > control["sharpe"]
              and sum(1 for y in years
                      if report[winner]["years"][y] > base["years"][y]) >= 7)
    print(f"\n  KILL CRITERION: an implied risk unit must beat Wilder N and the "
          f"realised-vol control, in >=7 of {len(years)} years")
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")
    report["verdict"] = {"passed": passed, "winner": winner}

    if args.gamma_sweep:
        print(f"\ngamma coefficient sweep (0 is plain implied N; negative "
              f"reverses the story):")
        for coefficient in (-0.7, -0.35, 0.0, 0.15, 0.35, 0.7):
            units = build_units(book, features, coefficient=coefficient)
            outcome = evaluate(book, units, args, f"gamma {coefficient:+.2f}")
            report[f"gamma {coefficient:+.2f}"] = outcome
            print(f"    coefficient {coefficient:>+5.2f}   Sharpe "
                  f"{outcome['sharpe']:>5.2f}   CAGR {outcome['cagr']:>6.1%}",
                  flush=True)

    if args.nulls:
        print(f"\n{args.nulls} circular-shift nulls through the identical "
              f"machinery...", flush=True)
        scores = []
        rng = random.Random(4)
        for draw in range(args.nulls):
            units = build_units(book, features, coefficient=0.35,
                                shift=rng.randrange(200, 2000))
            scores.append(evaluate(book, units, args, "null")["sharpe"])
        scores.sort()
        real = report["implied N, gamma-scaled"]["sharpe"]
        beat = sum(1 for s in scores if s >= real)
        pick = lambda f: scores[min(int(f * len(scores)), len(scores) - 1)]
        print(f"    null Sharpe  p05 {pick(.05):.2f}  median {pick(.5):.2f}  "
              f"p95 {pick(.95):.2f}  max {scores[-1]:.2f}")
        print(f"    real Sharpe  {real:.2f}")
        print(f"    empirical p = {(beat + 1) / (len(scores) + 1):.3f}  "
              f"({beat} of {len(scores)} shuffles matched or beat it)")
        report["null"] = {"p_value": (beat + 1) / (len(scores) + 1),
                          "median": pick(.5), "p95": pick(.95), "max": scores[-1]}

    if args.noise:
        print(f"{chr(10)}  {args.noise} synthetic persistence-matched risk units "
              f"(no information at all)...", flush=True)
        scores = []
        for draw in range(args.noise):
            units = synthetic_units(book, features, seed=1000 + draw)
            scores.append(evaluate(book, units, args, "noise")["sharpe"])
        scores.sort()
        pick = lambda f: scores[min(int(f * len(scores)), len(scores) - 1)]
        wilder = report["Wilder N (banked)"]["sharpe"]
        real = report["implied N, gamma-scaled"]["sharpe"]
        above = sum(1 for s in scores if s > wilder)
        print(f"    noise Sharpe  p05 {pick(.05):.2f}  median {pick(.5):.2f}  "
              f"p95 {pick(.95):.2f}  max {scores[-1]:.2f}")
        print(f"    Wilder N      {wilder:.2f}      "
              f"noise beats it in {above}/{len(scores)} draws")
        print(f"    implied N     {real:.2f}      "
              f"beaten by {sum(1 for s in scores if s >= real)}/{len(scores)} draws")
        verdict = ("the gain belongs to the ATR being a poor risk unit, not to "
                   "option data" if above > len(scores) * 0.5 else
                   "noise does not reproduce it; the implied series carries "
                   "something the ATR does not")
        print(f"    READING: {verdict}")
        report["noise"] = {"median": pick(.5), "p95": pick(.95),
                           "beats_wilder": above, "draws": len(scores),
                           "reading": verdict}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
