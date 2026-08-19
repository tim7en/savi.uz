"""How much of the strategy's result is the signal, and how much is luck?

Four questions, each with its own resampling scheme:

1. **Does the breakout entry beat a random one?**  The engine accepts explicit
   entries, so a random-entry null runs through the *same* stop, pyramid and
   exit code as the real signal.  Any difference is therefore attributable to
   entry timing alone, not to the exit doing the work.

   Two nulls, because they destroy different information:

   * *uniform* -- the same number of entries per instrument, placed at random
     eligible bars.  Destroys both timing and clustering.
   * *symbol-shuffled* -- the same entry dates, reassigned to randomly chosen
     instruments.  Preserves when trades happen and how they cluster, and
     destroys only the choice of *which* instrument was breaking out.

2. **How wide is the outcome distribution?**  Trades are resampled with
   replacement to give a terminal-wealth range.

3. **How much rests on the instrument list?**  Random subsets of the universe.

4. **How much rests on capacity ordering?**  Same-day entries compete for slots
   and the winner is arbitrary, so the tie-break is randomised throughout.

No future information is used anywhere: random entries are drawn from bars after
the warmup, sized from the volatility known at the previous close, and exited by
the same rules as the real trades.
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--min-sessions", type=int, default=400)
    parser.add_argument("--long-only", action="store_true", default=True)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--risk", type=float, default=0.0005)
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/stability.json"))
    return parser.parse_args(argv)


def load_universe(args):
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    names = [r[0] for r in connection.execute(
        "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker")]
    book = {}
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
        book[ticker] = resample_regular_session(five, minutes=args.minutes)
    connection.close()
    return book


def config_for(args):
    return TurtleConfig(entry_window=55, exit_window=20, atr_window=20,
                        skip_after_winner=False,
                        directions=(1,) if args.long_only else (1, -1))


def collect(book, config, entries_by_symbol=None):
    out = []
    for ticker, bars in book.items():
        kwargs = {}
        if entries_by_symbol is not None:
            kwargs["entries"] = entries_by_symbol.get(ticker, {})
        trades, _ = run_turtle(bars, config=config, **kwargs)
        for trade in trades:
            out.append({"ticker": ticker, "entry": trade.entry_timestamp,
                        "exit": trade.exit_timestamp, "r": trade.net_r})
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


def wealth(taken, risk):
    by_day = defaultdict(float)
    for trade in taken:
        by_day[trade["exit"][:10]] += trade["r"]
    nav, peak, worst = 1000.0, 1000.0, 0.0
    for day in sorted(by_day):
        nav = max(0.0, nav + by_day[day] * risk * nav)
        peak = max(peak, nav)
        worst = min(worst, nav / peak - 1.0)
    return nav, worst, sum(t["r"] for t in taken)


def pick(values, fraction):
    ordered = sorted(values)
    return ordered[min(int(fraction * len(ordered)), len(ordered) - 1)]


def summarise(label, finals, dds, totals, draws):
    return {
        "label": label, "draws": draws,
        "final_p05": pick(finals, .05), "final_median": pick(finals, .5),
        "final_p95": pick(finals, .95),
        "dd_median": pick(dds, .5), "dd_worst": min(dds),
        "total_r_median": pick(totals, .5),
        "total_r_p05": pick(totals, .05), "total_r_p95": pick(totals, .95),
    }


def main(argv=None):
    args = parse_args(argv)
    book = load_universe(args)
    config = config_for(args)
    print(f"{len(book)} instruments at {args.minutes}-minute bars, "
          f"{args.start} -> {args.end}", flush=True)

    real = collect(book, config)
    warmup = max(config.entry_window, config.atr_window) + 1
    index_of = {t: {b.timestamp: i for i, b in enumerate(bars)}
                for t, bars in book.items()}
    per_symbol = defaultdict(int)
    entry_days = []
    for trade in real:
        per_symbol[trade["ticker"]] += 1
        entry_days.append(trade["entry"])
    print(f"real signal: {len(real):,} trades across "
          f"{len(per_symbol)} instruments\n", flush=True)

    report = {"instruments": len(book), "minutes": args.minutes,
              "real_trades": len(real)}

    # ---- 1. real strategy, varying only the capacity tie-break ----
    finals, dds, totals = [], [], []
    for seed in range(args.draws):
        n, d, t = wealth(cap(real, args.max_positions, random.Random(seed)), args.risk)
        finals.append(n); dds.append(d); totals.append(t)
    report["real"] = summarise("real breakout signal", finals, dds, totals, args.draws)

    # ---- 2. uniform random entries, same count per instrument ----
    ufinals, udds, utotals = [], [], []
    for seed in range(args.draws):
        rng = random.Random(50_000 + seed)
        entries = {}
        for ticker, count in per_symbol.items():
            span = len(book[ticker])
            if span <= warmup + 5 or count == 0:
                continue
            slots = rng.sample(range(warmup, span - 1), min(count, span - warmup - 1))
            entries[ticker] = {i: 1 for i in slots}
        sample = collect(book, config, entries)
        n, d, t = wealth(cap(sample, args.max_positions, rng), args.risk)
        ufinals.append(n); udds.append(d); utotals.append(t)
    report["uniform_null"] = summarise("random entries, uniform",
                                       ufinals, udds, utotals, args.draws)

    # ---- 3. same entry dates, shuffled across instruments ----
    sfinals, sdds, stotals = [], [], []
    tickers = list(book)
    for seed in range(args.draws):
        rng = random.Random(90_000 + seed)
        entries = defaultdict(dict)
        for stamp in entry_days:
            for _ in range(6):                      # a few tries to land a bar
                ticker = rng.choice(tickers)
                position = index_of[ticker].get(stamp)
                if position is not None and position >= warmup:
                    entries[ticker][position] = 1
                    break
        sample = collect(book, config, entries)
        n, d, t = wealth(cap(sample, args.max_positions, rng), args.risk)
        sfinals.append(n); sdds.append(d); stotals.append(t)
    report["shuffled_null"] = summarise("random entries, same dates",
                                        sfinals, sdds, stotals, args.draws)

    # ---- 4. bootstrap the real trades ----
    bfinals, bdds, btotals = [], [], []
    for seed in range(args.draws * 5):
        rng = random.Random(20_000 + seed)
        taken = cap(real, args.max_positions, random.Random(seed % args.draws))
        drawn = [taken[rng.randrange(len(taken))] for _ in range(len(taken))]
        by_day = defaultdict(float)
        for i, trade in enumerate(drawn):
            by_day[i] += trade["r"]
        nav, peak, worst = 1000.0, 1000.0, 0.0
        for key in sorted(by_day):
            nav = max(0.0, nav + by_day[key] * args.risk * nav)
            peak = max(peak, nav); worst = min(worst, nav / peak - 1.0)
        bfinals.append(nav); bdds.append(worst)
        btotals.append(sum(t["r"] for t in drawn))
    report["trade_bootstrap"] = summarise("bootstrapped trades", bfinals, bdds,
                                          btotals, args.draws * 5)

    # ---- 5. random subsets of the instrument list ----
    subsets = {}
    for size in (10, 20, 30):
        ffinals, fdds, ftotals = [], [], []
        for seed in range(args.draws):
            rng = random.Random(70_000 + seed)
            chosen = set(rng.sample(tickers, min(size, len(tickers))))
            subset = [t for t in real if t["ticker"] in chosen]
            if not subset:
                continue
            n, d, t = wealth(cap(subset, args.max_positions, rng), args.risk)
            ffinals.append(n); fdds.append(d); ftotals.append(t)
        subsets[str(size)] = summarise(f"{size} random instruments",
                                       ffinals, fdds, ftotals, args.draws)
    report["subsets"] = subsets
    report["curves"] = {
        "real_total_r": totals, "uniform_total_r": utotals,
        "shuffled_total_r": stotals, "bootstrap_final": bfinals,
        "real_final": finals, "uniform_final": ufinals, "shuffled_final": sfinals,
    }

    print(f"  {'scheme':30s} {'median $':>11s} {'5-95% band':>25s} "
          f"{'median R':>10s} {'maxDD':>8s}")
    for key in ("real", "uniform_null", "shuffled_null", "trade_bootstrap"):
        d = report[key]
        print(f"  {d['label']:30s} ${d['final_median']:>10,.0f} "
              f"${d['final_p05']:>10,.0f}-${d['final_p95']:>10,.0f} "
              f"{d['total_r_median']:>+10.0f} {d['dd_median']:>8.1%}")
    for size, d in subsets.items():
        print(f"  {d['label']:30s} ${d['final_median']:>10,.0f} "
              f"${d['final_p05']:>10,.0f}-${d['final_p95']:>10,.0f} "
              f"{d['total_r_median']:>+10.0f} {d['dd_median']:>8.1%}")

    beat_u = sum(1 for x in ufinals if x >= pick(finals, .5)) / len(ufinals)
    beat_s = sum(1 for x in sfinals if x >= pick(finals, .5)) / len(sfinals)
    print(f"\n  random-entry draws reaching the real median:")
    print(f"    uniform          {beat_u:.0%}")
    print(f"    same-date shuffle {beat_s:.0%}")
    report["null_beats_real"] = {"uniform": beat_u, "shuffled": beat_s}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
