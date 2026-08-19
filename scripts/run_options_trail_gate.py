"""Can the options state set the trail distance? The gate test.

The hypothesis, narrowed to one moving part: options do not predict direction and
do not improve sizing, but they may say whether the next stretch is a large
volatile excursion or a compressed, pinning one -- and that is exactly what a
trailing stop wants to know.  So the trail multiplier alone becomes a function of
the prior session's options state, while sizing, the hard stop and pyramid
spacing all stay on Wilder N.

    k = 5.0  when the state says large excursion   (widen, let it run)
    k = 2.0  when the state says compression       (tighten, harvest)

The control that decides it is not a null -- it is a *rival forecast*.  Dealer
gamma sorts next-session realised volatility 2.5:1, but so does trailing realised
volatility, and the strategy is already ATR-normalised.  If a trail conditioned on
trailing RV does the same job, the options data is redundant and the answer is no.
The RV control is deliberately built from SPY, not from each instrument, so it
carries the same market-wide information scope as the gamma state and wins on
merit rather than on extra data.

Three further controls run alongside:

* a reversed mapping -- if widening on compression works just as well, the effect
  is noise wearing a story;
* a circular shift of the state series, which preserves gamma's strong
  autocorrelation while destroying its alignment to price;
* an independent draw matched to the state's marginal distribution.

Look-ahead: the feature is the end-of-day snapshot of session D and is read only
by entries on session D+1 or later.  Quintile boundaries are trailing, computed
from sessions strictly before the one being classified, never from the full
sample.

Kill criterion, fixed before the run: the options-conditioned trail must beat the
RV-conditioned trail at matched drawdown in at least 7 of 10 calendar years.
"""

from __future__ import annotations

import argparse
import bisect
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

#: Expected-excursion rank 1 (most compressed) to 5 (largest) -> trail multiple.
K_BY_RANK = {1: 2.0, 2: 2.5, 3: 3.0, 4: 4.0, 5: 5.0}
WARMUP = 250


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--options", type=Path,
                        default=Path("data/options/alphavantage.db"))
    parser.add_argument("--state-symbol", default="SPY")
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--min-sessions", type=int, default=400)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--null-draws", type=int, default=8)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/options_trail_gate.json"))
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


def trailing_ranks(pairs):
    """Quintile of each value among the values that preceded it.

    A full-sample percentile would let a 2018 session know the 2025 distribution.
    Each value is instead ranked against its own past only, which is what a live
    system would have had, and the first ``WARMUP`` sessions go unranked.
    """
    ranks, seen = {}, []
    for day, value in pairs:
        if len(seen) >= WARMUP:
            share = bisect.bisect_left(seen, value) / len(seen)
            ranks[day] = min(5, int(share * 5) + 1)
        bisect.insort(seen, value)
    return ranks


def realised_vol(bars):
    """Per-session realised volatility from the intraday bars of that session."""
    by_day = defaultdict(list)
    for bar in bars:
        by_day[bar.timestamp[:10]].append(bar.close)
    out = {}
    for day, closes in by_day.items():
        rets = [closes[i] / closes[i - 1] - 1.0
                for i in range(1, len(closes)) if closes[i - 1] > 0]
        if rets:
            out[day] = math.sqrt(sum(r * r for r in rets))
    return out


def load_states(args, book):
    """Every state series, keyed by session, as expected-excursion ranks 1..5."""
    store = sqlite3.connect(f"file:{args.options}?mode=ro", uri=True)
    rows = store.execute(
        "SELECT observation_date, gamma_balance, atm_iv FROM av_daily WHERE "
        "symbol=? ORDER BY observation_date", (args.state_symbol,)).fetchall()
    store.close()

    gamma = [(d, g) for d, g, _ in rows if g is not None]
    iv = [(d, v) for d, _, v in rows if v is not None]
    spot = realised_vol(book[args.state_symbol])
    rv = sorted(spot.items())

    # Negative gamma predicts the largest next-session moves, so the most
    # negative quintile is the largest expected excursion: invert its rank.
    gamma_rank = {d: 6 - r for d, r in trailing_ranks(gamma).items()}
    return {"gamma": gamma_rank,
            "iv": trailing_ranks(iv),
            "rv": trailing_ranks(rv)}


def lagged(state, sessions):
    """Map each session to the state of the last session strictly before it."""
    days = sorted(state)
    out, position, carried = {}, 0, None
    for day in sessions:
        while position < len(days) and days[position] < day:
            carried = state[days[position]]
            position += 1
        if carried is not None:
            out[day] = carried
    return out


def schedule(bars, ranks, mapping=K_BY_RANK):
    """Trail multiple per bar index, from that bar's session state."""
    return {i: mapping[ranks[bar.timestamp[:10]]]
            for i, bar in enumerate(bars) if bar.timestamp[:10] in ranks}


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


def solve_risk(series, target, lo=1e-6, hi=0.05):
    def dd(risk):
        return statistics.median(abs(path_metrics(d, v, risk)[1]) for d, v in series)
    if dd(hi) < target:
        return hi
    for _ in range(35):
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


def evaluate(book, schedules, args):
    """Pool trades under a per-instrument trail schedule and score at matched DD."""
    config = TurtleConfig(**BASE)
    pooled = []
    for ticker, bars in book.items():
        trades, _ = run_turtle(bars, config=config,
                               chandelier_by_bar=schedules.get(ticker))
        closes = session_closes(bars)
        pooled.extend({"entry": t.entry_timestamp, "exit": t.exit_timestamp,
                       "r": t.net_r, "marks": trade_marks(t, closes)}
                      for t in trades)
    series = [marked_series(cap(pooled, args.max_positions, random.Random(s)))
              for s in range(args.trials)]
    risk = solve_risk(series, args.target_dd)
    years = sorted({d[:4] for d, _ in [(x, 0) for x in series[0][0]]})
    by_year = {}
    for year in years:
        sliced = [([d for d in days if d[:4] == year],
                   [v for d, v in zip(days, values) if d[:4] == year])
                  for days, values in series]
        by_year[year] = sharpe([s for s in sliced if len(s[0]) >= 30])
    return {"trades": len(pooled), "sharpe": sharpe(series), "risk": risk,
            "cagr": statistics.median(path_metrics(d, v, risk)[2] for d, v in series),
            "years": by_year}


def main(argv=None):
    args = parse_args(argv)
    book = load_book(args)
    states = load_states(args, book)
    print(f"{len(book)} instruments; state from {args.state_symbol}: "
          + ", ".join(f"{k} {len(v):,} ranked sessions" for k, v in states.items())
          + "\n", flush=True)

    per_ticker_lag = {name: {k: lagged(v, sorted({b.timestamp[:10] for b in bars}))
                             for k, v in states.items()}
                      for name, bars in book.items()}

    def build(key, mapping=K_BY_RANK, transform=None):
        out = {}
        for name, bars in book.items():
            ranks = per_ticker_lag[name][key]
            if transform is not None:
                ranks = transform(ranks)
            out[name] = schedule(bars, ranks, mapping)
        return out

    reversed_map = {r: K_BY_RANK[6 - r] for r in K_BY_RANK}

    def shift_by(offset):
        def apply(ranks):
            days = sorted(ranks)
            values = [ranks[d] for d in days]
            n = len(days)
            return {d: values[(i + offset) % n] for i, d in enumerate(days)}
        return apply

    def resample(seed):
        def apply(ranks):
            rng = random.Random(seed)
            pool = list(ranks.values())
            return {d: rng.choice(pool) for d in ranks}
        return apply

    # The ceiling on any per-symbol forecast: rank each instrument by its OWN
    # trailing realised volatility.  Options data could at best approximate this
    # quantity, so if conditioning on it does not beat a fixed trail, no
    # per-symbol option chain can either -- the limit is redundancy with N, not
    # the accuracy of the forecast.
    own = {}
    for name, bars in book.items():
        ranks = trailing_ranks(sorted(realised_vol(bars).items()))
        own[name] = schedule(bars, lagged(ranks, sorted({b.timestamp[:10]
                                                         for b in bars})))

    runs = {
        "fixed k=3 (banked)": {},
        "own realised vol (CEILING)": own,
        "gamma state": build("gamma"),
        "IV state": build("iv"),
        "realised-vol state (CONTROL)": build("rv"),
        "gamma reversed": build("gamma", reversed_map),
    }
    report = {}
    print(f"  {'variant':32s} {'trades':>8s} {'Sharpe':>7s} {'lev':>8s} {'CAGR':>8s}")
    for label, schedules in runs.items():
        result = evaluate(book, schedules, args)
        report[label] = result
        print(f"  {label:32s} {result['trades']:>8,d} {result['sharpe']:>7.2f} "
              f"{result['risk']:>8.4%} {result['cagr']:>8.1%}", flush=True)

    for name, transform in (("circular shift", shift_by), ("independent draw", resample)):
        picks = []
        for draw in range(args.null_draws):
            offset = 137 * (draw + 1) if name == "circular shift" else draw
            picks.append(evaluate(book, build("gamma", transform=transform(offset)),
                                  args))
        median = {"sharpe": statistics.median(p["sharpe"] for p in picks),
                  "cagr": statistics.median(p["cagr"] for p in picks),
                  "best": max(p["sharpe"] for p in picks),
                  "trades": statistics.median(p["trades"] for p in picks)}
        report[f"null: {name}"] = median
        print(f"  null: {name:26s} {median['trades']:>8,.0f} "
              f"{median['sharpe']:>7.2f} {'':>8s} {median['cagr']:>8.1%} "
              f"(best of {args.null_draws}: {median['best']:.2f})", flush=True)

    gamma, control = report["gamma state"], report["realised-vol state (CONTROL)"]
    years = sorted(gamma["years"])
    wins = sum(1 for y in years if gamma["years"][y] > control["years"][y])
    print(f"\n  Sharpe by year, gamma vs realised-vol control:")
    print("    " + "".join(f"{y:>8s}" for y in years))
    print("    " + "".join(f"{gamma['years'][y]:>8.2f}" for y in years) + "   gamma")
    print("    " + "".join(f"{control['years'][y]:>8.2f}" for y in years) + "   RV")
    print(f"\n  KILL CRITERION: gamma beats the RV control in {wins}/{len(years)} "
          f"years (needed 7)")
    print(f"  VERDICT: {'PASS' if wins >= 7 else 'FAIL'}")
    report["verdict"] = {"years_won": wins, "years": len(years),
                         "passed": wins >= 7}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
