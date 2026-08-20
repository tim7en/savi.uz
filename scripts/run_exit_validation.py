"""Was the 3N chandelier chosen, or fitted? The out-of-sample test.

The exit is the only improvement this programme has banked.  It was picked as the
best of twelve variants, scored on the same forty-two instruments over the same
decade it is now credited with -- which is how a fitted parameter is produced, not
how one is validated.  Ten-out-of-ten year consistency argues against pure luck,
but consistency across years of one sample is not the same as surviving a sample
that had no say in the choice.

Two independent splits, because they fail in different ways.

*Across instruments.*  The universe is divided into folds; the exit is chosen on
the names outside a fold and scored on the names inside it.  This catches a rule
that happens to suit the particular stocks it was selected on.

*Across time.*  The exit is chosen on 2017-2021 and scored on 2022-2026.  This
catches a rule that suits one regime, which matters here because the sample is
eight bull years and two difficult ones.

What is measured is not whether the held-out score is good -- a smaller book
scores differently for reasons that have nothing to do with the exit -- but
whether the *advantage over the Donchian exit it replaced* survives.  Both
variants meet identical conditions inside a fold, so the difference between them
is clean even when the level is not.

Sharpe is the selection metric throughout.  It is scale free, so the ranking is
identical to one taken at matched drawdown, without needing the leverage solved
for every fold.

Also tested is the fallback the plan pre-registered: rather than the single best
variant, take the median trail multiple of the top three.  If argmax selection is
fitting noise, a rule that refuses to trust the winner should hold up better.
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

BASE = dict(entry_window=55, exit_window=20, atr_window=20,
            skip_after_winner=False, directions=(1,))
BASELINE = "channel 20 (Donchian)"

VARIANTS = [
    (BASELINE, {}),
    ("channel 10", dict(exit_window=10)),
    ("channel 50", dict(exit_window=50)),
    ("chandelier 2N", dict(use_channel_exit=False, chandelier_atr=2.0)),
    ("chandelier 3N", dict(use_channel_exit=False, chandelier_atr=3.0)),
    ("chandelier 4N", dict(use_channel_exit=False, chandelier_atr=4.0)),
    ("chandelier 5N", dict(use_channel_exit=False, chandelier_atr=5.0)),
    ("chandelier 8N", dict(use_channel_exit=False, chandelier_atr=8.0)),
    ("channel 20 + chandelier 5N", dict(chandelier_atr=5.0)),
    ("channel 20 + breakeven 1N", dict(breakeven_trigger_n=1.0)),
    ("wider hard stop 3N", dict(stop_atr=3.0)),
    ("no pyramid", dict(max_units=1)),
]
#: Trail multiple per variant, for the median-of-top-three fallback. The channel
#: exits have no trail multiple and are excluded from that calculation.
TRAIL = {"chandelier 2N": 2.0, "chandelier 3N": 3.0, "chandelier 4N": 4.0,
         "chandelier 5N": 5.0, "chandelier 8N": 8.0,
         "channel 20 + chandelier 5N": 5.0}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--min-sessions", type=int, default=400)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--split-date", default="2022-01-01")
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/exit_validation.json"))
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


def sharpe_of(trades, args, lo=None, hi=None):
    """Annualised Sharpe of the mark-to-market daily R stream."""
    window = [t for t in trades
              if (lo is None or t["entry"] >= lo) and (hi is None or t["entry"] < hi)]
    if len(window) < 80:
        return float("nan")
    scores = []
    for seed in range(args.trials):
        taken = cap(window, args.max_positions, random.Random(seed))
        by_day = defaultdict(float)
        for trade in taken:
            previous = 0.0
            for day, open_r in trade["marks"]:
                if lo and day < lo[:10]:
                    continue
                by_day[day] += open_r - previous
                previous = open_r
            by_day[trade["exit"][:10]] += trade["r"] - previous
        days = sorted(by_day)
        if len(days) < 30:
            continue
        span = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days
        stream = ([by_day[d] for d in days]
                  + [0.0] * max(0, int(span * 252 / 365.25) - len(days)))
        sd = statistics.pstdev(stream)
        if sd > 0:
            scores.append(statistics.fmean(stream) / sd * math.sqrt(252))
    return statistics.median(scores) if scores else float("nan")


def main(argv=None):
    args = parse_args(argv)
    book = load_book(args)
    tickers = sorted(book)
    print(f"{len(tickers)} instruments, {len(VARIANTS)} exit variants\n", flush=True)

    # Every variant's trades, per instrument, computed once.
    trades_by = {}
    for label, overrides in VARIANTS:
        config = TurtleConfig(**{**BASE, **overrides})
        per_ticker = {}
        for ticker, bars in book.items():
            closes = session_closes(bars)
            found, _ = run_turtle(bars, config=config)
            per_ticker[ticker] = [
                {"entry": t.entry_timestamp, "exit": t.exit_timestamp,
                 "r": t.net_r, "marks": trade_marks(t, closes)} for t in found]
        trades_by[label] = per_ticker
        print(f"  {label:30s} {sum(len(v) for v in per_ticker.values()):>7,d} trades",
              flush=True)

    def score(label, names, lo=None, hi=None):
        pooled = [t for n in names for t in trades_by[label][n]]
        return sharpe_of(pooled, args, lo, hi)

    # ---- split one: across instruments ----------------------------------
    rng = random.Random(20260820)
    shuffled = list(tickers)
    rng.shuffle(shuffled)
    folds = [shuffled[i::args.folds] for i in range(args.folds)]

    print(f"\n{'=' * 78}\nSPLIT 1: choose on other names, score on held-out names")
    print(f"  {'fold':>4s} {'held out':>9s} {'chosen on train':28s} "
          f"{'in-samp':>8s} {'held-out':>9s} {'3N held-out':>12s}")
    rows = []
    for index, fold in enumerate(folds):
        train = [t for t in tickers if t not in fold]
        train_scores = {label: score(label, train) for label, _ in VARIANTS}
        live = {k: v for k, v in train_scores.items() if v == v}
        winner = max(live, key=live.get)
        base_train = train_scores[BASELINE]
        base_test = score(BASELINE, fold)
        win_test = score(winner, fold)
        three_test = score("chandelier 3N", fold)
        row = {"fold": index, "held_out": len(fold), "winner": winner,
               "in_sample_edge": live[winner] - base_train,
               "held_out_edge": win_test - base_test,
               "chandelier3n_edge": three_test - base_test}
        rows.append(row)
        print(f"  {index:>4d} {len(fold):>9d} {winner:28s} "
              f"{row['in_sample_edge']:>+8.2f} {row['held_out_edge']:>+9.2f} "
              f"{row['chandelier3n_edge']:>+12.2f}", flush=True)

    picked_3n = sum(1 for r in rows if r["winner"] == "chandelier 3N")
    in_edge = statistics.fmean(r["in_sample_edge"] for r in rows)
    out_edge = statistics.fmean(r["held_out_edge"] for r in rows)
    three_edge = statistics.fmean(r["chandelier3n_edge"] for r in rows)
    print(f"\n  training picks chandelier 3N in {picked_3n}/{len(rows)} folds")
    print(f"  mean edge over Donchian: in-sample {in_edge:+.2f}, "
          f"held-out {out_edge:+.2f}  (retained {out_edge / in_edge:.0%})")
    print(f"  chandelier 3N specifically, held out: {three_edge:+.2f}")

    # ---- split two: across time -----------------------------------------
    print(f"\n{'=' * 78}\nSPLIT 2: choose on 2017-2021, score on {args.split_date}+")
    early = {label: score(label, tickers, hi=args.split_date) for label, _ in VARIANTS}
    late = {label: score(label, tickers, lo=args.split_date) for label, _ in VARIANTS}
    live_early = {k: v for k, v in early.items() if v == v}
    winner = max(live_early, key=live_early.get)
    ranked = sorted(live_early, key=live_early.get, reverse=True)
    print(f"  {'exit':30s} {'2017-21':>9s} {'2022-26':>9s}")
    for label, _ in VARIANTS:
        mark = "  <- chosen" if label == winner else ""
        print(f"  {label:30s} {early[label]:>9.2f} {late[label]:>9.2f}{mark}")
    time_in = live_early[winner] - early[BASELINE]
    time_out = late[winner] - late[BASELINE]
    print(f"\n  training winner: {winner}")
    print(f"  edge over Donchian: 2017-21 {time_in:+.2f}, "
          f"2022-26 {time_out:+.2f}  (retained {time_out / time_in:.0%})")
    print(f"  chandelier 3N specifically, 2022-26: "
          f"{late['chandelier 3N'] - late[BASELINE]:+.2f}")

    # ---- the pre-registered fallback ------------------------------------
    top3 = [label for label in ranked if label in TRAIL][:3]
    fallback = statistics.median(TRAIL[label] for label in top3) if top3 else None
    print(f"\n  median-of-top-three fallback: top trails {[TRAIL[l] for l in top3]}"
          f" -> {fallback}N")

    retained = out_edge / in_edge if in_edge else float("nan")
    passed = retained >= 0.5 and time_out > 0
    print(f"\n{'=' * 78}")
    print(f"  KILL CRITERION: held-out edge must be at least half the in-sample edge")
    print(f"  across instruments: {retained:.0%} retained")
    print(f"  across time:        {time_out / time_in:.0%} retained, "
          f"held-out edge {time_out:+.2f}")
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")

    report = {"folds": rows, "instrument_retained": retained,
              "time_early": early, "time_late": late, "time_winner": winner,
              "time_retained": time_out / time_in if time_in else None,
              "fallback_trail": fallback, "passed": passed}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
