"""When does the recovery outrun the borrowing cost?

Every earlier test measured how much a drawdown entry returns.  That is the wrong
quantity for a levered book.  Carry accrues per unit of time and the recovery does
not, so the decisive variable is the *rate*, not the size: +30% over three years
annualises to 9.1% and loses to a 5% borrow once the levered portion is paid for;
the same +30% in six months annualises to 69% and pays for itself many times over.

So the number reported here is the **annualised recovery rate**, which is exactly
the breakeven financing rate.  If an entry recovers at 25% a year you can borrow
at anything under 25% and the levered portion still adds.  Everything else is a
question about which entries have a high one, and whether that is knowable in
advance.

Five conditioners, all readable at the entry bar and none of them using the
future:

* **depth** below the trailing high;
* **speed of the fall**, sessions from the peak to the entry -- a crash and a
  grind are different animals and the distinction is free;
* **realised volatility** over the prior 20 sessions;
* **breadth**, the share of the universe simultaneously 10% or more below its own
  high, which separates a market-wide dislocation from one company's problem;
* **run-up** into the peak that was given back.

Then the arithmetic that decides whether any of it is usable.  At leverage L a
position is closed out when the mark falls 1/L, and the equity path carries the
carry, so survival and speed have to be measured together:

    equity(t) = 1 + L*(P(t)/P(0) - 1) - (L-1) * f * t/252

Wipeout is checked against daily lows, so an intraday spike that would trigger a
margin call counts.  A wiped position is -100% on equity and stays in every
average, because a study that quietly drops its ruined paths is measuring the
survivors and calling it a strategy.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

LOOKBACK = 252
HORIZON = 756
RUNUP = 504
LEVERAGES = (1.0, 2.0, 3.0)
DECAYING = ("USO", "UNG", "GSG", "DBC", "DBA", "BNO", "VXX")
NON_EQUITY = ("FXB", "FXE", "FXY", "UUP", "HYG", "IEF", "LQD", "SHY",
              "TIP", "TLT", "GLD", "SLV", "PPLT")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etfs", type=Path,
                        default=Path("data/cross_assets/etf_daily.db"))
    parser.add_argument("--stocks", type=Path,
                        default=Path("data/13f/alphavantage_daily.db"))
    parser.add_argument("--depth", type=float, default=0.18)
    parser.add_argument("--financing", type=float, default=0.05)
    parser.add_argument("--null-draws", type=int, default=200)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/recovery_vs_carry.json"))
    known, _ = parser.parse_known_args(argv)
    return known


def load(path, drop):
    splits = load_splits(path)
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    book = {}
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency='daily' "
            "ORDER BY ticker"):
        if drop and (ticker in DECAYING or ticker in NON_EQUITY):
            continue
        rows = connection.execute(
            "SELECT ts,open,high,low,close FROM bars WHERE ticker=? AND "
            "frequency='daily' ORDER BY ts", (ticker,)).fetchall()
        bars = adjust_bars([Bar(r[0][:10], r[1], r[2], r[3], r[4], None)
                            for r in rows], splits.get(ticker, []))
        if len(bars) >= 1200:
            book[ticker] = bars
    connection.close()
    return book


def breadth_by_day(book):
    """Share of the universe 10%+ below its own trailing high, per session."""
    down = defaultdict(int)
    alive = defaultdict(int)
    for bars in book.values():
        closes = [b.close for b in bars]
        running = []
        for i, bar in enumerate(bars):
            running.append(closes[i])
            if len(running) > LOOKBACK:
                running.pop(0)
            if i < LOOKBACK:
                continue
            top = max(running)
            alive[bar.timestamp] += 1
            if top > 0 and closes[i] / top - 1.0 <= -0.10:
                down[bar.timestamp] += 1
    return {day: down[day] / n for day, n in alive.items() if n >= 10}


def episodes(book, breadth, args):
    """Fresh crossings with everything knowable at entry and what followed."""
    rows = []
    for ticker, bars in book.items():
        closes = [b.close for b in bars]
        lows = [b.low for b in bars]
        peak, running = [], []
        for i in range(len(bars)):
            running.append((closes[i], i))
            if len(running) > LOOKBACK:
                running.pop(0)
            peak.append(max(running))
        armed = True
        for i in range(LOOKBACK + RUNUP, len(bars) - 60):
            top, top_at = peak[i]
            if top <= 0 or closes[i] <= 0:
                continue
            drop = closes[i] / top - 1.0
            if drop > -args.depth * 0.6:
                armed = True
            if not armed or drop > -args.depth:
                continue
            armed = False
            day = bars[i].timestamp
            if day not in breadth:
                continue

            # --- knowable at entry -------------------------------------
            window = closes[i - 20:i + 1]
            steps = [window[k] / window[k - 1] - 1.0 for k in range(1, len(window))
                     if window[k - 1] > 0]
            vol = (statistics.pstdev(steps) * (252 ** 0.5)) if len(steps) > 5 else None
            before = top_at - RUNUP
            runup = (closes[top_at] / closes[before] - 1.0
                     if before >= 0 and closes[before] > 0 else None)

            # --- what followed -----------------------------------------
            end = min(i + HORIZON, len(bars) - 1)
            recovered_at = None
            for j in range(i + 1, end + 1):
                if closes[j] >= top:
                    recovered_at = j
                    break
            exit_at = recovered_at if recovered_at else end
            held = exit_at - i
            if held < 5:
                continue
            gain = closes[exit_at] / closes[i] - 1.0
            rate = (1.0 + gain) ** (252.0 / held) - 1.0
            floor = min(lows[i + 1:exit_at + 1] or [closes[i]])

            rows.append({
                "ticker": ticker, "day": day, "index": i, "exit": exit_at,
                "depth": -drop, "speed": i - top_at, "vol": vol,
                "breadth": breadth[day], "runup": runup,
                "held": held, "gain": gain, "rate": rate,
                "recovered": recovered_at is not None,
                "further": floor / closes[i] - 1.0})
    return rows


def levered(book, row, leverage, financing):
    """Equity outcome net of carry, wipeout checked against daily lows."""
    bars = book[row["ticker"]]
    entry = bars[row["index"]].close
    equity_min, ruined = 1.0, False
    for j in range(row["index"] + 1, row["exit"] + 1):
        t = (j - row["index"]) / 252.0
        mark = 1.0 + leverage * (bars[j].low / entry - 1.0) \
            - (leverage - 1.0) * financing * t
        equity_min = min(equity_min, mark)
        if mark <= 0:
            ruined = True
            break
    if ruined:
        return {"equity": 0.0, "ruined": True, "rate": -1.0}
    years = row["held"] / 252.0
    equity = 1.0 + leverage * row["gain"] - (leverage - 1.0) * financing * years
    if equity <= 0:
        return {"equity": 0.0, "ruined": True, "rate": -1.0}
    return {"equity": equity, "ruined": False,
            "rate": equity ** (1.0 / years) - 1.0, "worst": equity_min}


def percentile(values, share):
    ordered = sorted(values)
    return ordered[min(int(share * len(ordered)), len(ordered) - 1)]


def bucket_report(rows, key, edges, label, args, report, book):
    print()
    print(f"########## {label} ##########")
    print(f"  {'bucket':16s} {'n':>5s} {'med rate':>9s} {'p25 rate':>9s} "
          f"{'recovered':>10s} {'med days':>9s} {'med gain':>9s} "
          f"{'3x net':>8s} {'3x ruin':>8s}")
    report[key] = {}
    for low, high in zip(edges, edges[1:]):
        chunk = [r for r in rows if r[key] is not None and low <= r[key] < high]
        if len(chunk) < 25:
            continue
        rates = [r["rate"] for r in chunk]
        three = [levered(book, r, 3.0, args.financing) for r in chunk]
        name = (f"{low:.0%}-{high:.0%}" if key in ("depth", "breadth", "vol", "runup")
                else f"{low:.0f}-{high:.0f}d")
        got = {"n": len(chunk), "median_rate": statistics.median(rates),
               "p25_rate": percentile(rates, 0.25),
               "recovered": sum(1 for r in chunk if r["recovered"]) / len(chunk),
               "median_days": statistics.median(r["held"] for r in chunk),
               "median_gain": statistics.median(r["gain"] for r in chunk),
               "lev3_median": statistics.median(t["rate"] for t in three),
               "lev3_ruin": sum(1 for t in three if t["ruined"]) / len(three)}
        report[key][name] = got
        print(f"  {name:16s} {got['n']:>5,d} {got['median_rate']:>+9.1%} "
              f"{got['p25_rate']:>+9.1%} {got['recovered']:>10.0%} "
              f"{got['median_days']:>9.0f} {got['median_gain']:>+9.1%} "
              f"{got['lev3_median']:>+8.1%} {got['lev3_ruin']:>8.0%}")


def run_universe(book, name, args, report):
    breadth = breadth_by_day(book)
    rows = episodes(book, breadth, args)
    print()
    print("=" * 78)
    print(f"{name}: {len(book)} instruments, {len(rows):,d} entries at "
          f"{args.depth:.0%} below the 252-day high")
    print("=" * 78)
    if len(rows) < 50:
        print("  too few entries to bucket")
        return rows
    here = report.setdefault(name, {})
    rates = [r["rate"] for r in rows]
    print()
    print("########## the hurdle: annualised recovery rate ##########")
    print("  This IS the breakeven financing rate. Borrow below it and the")
    print("  levered portion adds; borrow above it and carry eats the recovery.")
    for lab, share in (("5th worst", 0.05), ("25th", 0.25), ("median", 0.50),
                       ("75th", 0.75), ("95th", 0.95)):
        print(f"  {lab:>12s} {percentile(rates, share):>+9.1%}")
    share_above = sum(1 for r in rates if r > args.financing) / len(rates)
    here["overall"] = {"n": len(rows), "median_rate": statistics.median(rates),
                       "share_above_financing": share_above,
                       "median_days": statistics.median(r["held"] for r in rows),
                       "recovered": sum(1 for r in rows if r["recovered"]) / len(rows)}
    print(f"  {share_above:.0%} of entries clear a {args.financing:.0%} borrow; "
          f"median hold {statistics.median(r['held'] for r in rows):.0f} sessions")

    bucket_report(rows, "speed", [0, 40, 80, 160, 320, 100000],
                  "by speed of the fall -- sessions from peak to entry",
                  args, here, book)
    bucket_report(rows, "depth", [0.18, 0.25, 0.35, 0.50, 1.01],
                  "by depth at entry", args, here, book)
    bucket_report(rows, "breadth", [0.0, 0.25, 0.50, 0.75, 1.01],
                  "by breadth -- share of the universe already 10% down",
                  args, here, book)
    bucket_report(rows, "vol", [0.0, 0.20, 0.30, 0.45, 9.0],
                  "by realised volatility at entry", args, here, book)
    bucket_report(rows, "runup", [-1.0, 0.15, 0.35, 0.70, 9.0],
                  "by run-up into the peak", args, here, book)
    return rows


def main(argv=None) -> int:
    args = parse_args(argv)
    report = {"depth": args.depth, "financing": args.financing}
    print(f"Carry {args.financing:.0%} a year on the borrowed portion. "
          f"Wipeout checked against daily lows.")
    print("A ruined position is -100% on equity and stays in every average.")

    etfs = load(args.etfs, drop=True)
    etf_rows = run_universe(etfs, "equity ETFs", args, report)

    stocks = load(args.stocks, drop=False)
    stock_rows = run_universe(stocks, "single stocks", args, report)

    # -------- the joint cell, on whichever universe has the entries --------
    for label, book, rows in (("equity ETFs", etfs, etf_rows),
                              ("single stocks", stocks, stock_rows)):
        if len(rows) < 100:
            continue
        print()
        print(f"########## {label}: fast and broad, against everything else #####")
        print("  Fast is a fall of 80 sessions or less; broad is a quarter of the")
        print("  universe already 10% down. Both are known at the entry bar.")
        fast_broad = [r for r in rows if r["speed"] <= 80 and r["breadth"] >= 0.25]
        rest = [r for r in rows if not (r["speed"] <= 80 and r["breadth"] >= 0.25)]
        if len(fast_broad) < 25:
            print("  too few in the cell")
            continue
        print(f"  {'cell':16s} {'n':>5s} {'med rate':>9s} {'med days':>9s} "
              + " ".join(f"{'net ' + format(L, '.0f') + 'x':>9s}" for L in LEVERAGES)
              + f" {'3x ruin':>8s}")
        cells = {}
        for cell_name, chunk in (("fast and broad", fast_broad), ("everything else", rest)):
            line = [f"  {cell_name:16s} {len(chunk):>5,d} "
                    f"{statistics.median(r['rate'] for r in chunk):>+9.1%} "
                    f"{statistics.median(r['held'] for r in chunk):>9.0f}"]
            entry = {"n": len(chunk),
                     "median_rate": statistics.median(r["rate"] for r in chunk),
                     "median_days": statistics.median(r["held"] for r in chunk)}
            for leverage in LEVERAGES:
                got = [levered(book, r, leverage, args.financing) for r in chunk]
                med = statistics.median(t["rate"] for t in got)
                entry[f"lev{leverage:.0f}"] = med
                entry[f"ruin{leverage:.0f}"] = sum(1 for t in got if t["ruined"]) / len(got)
                line.append(f" {med:>+9.1%}")
            line.append(f" {entry['ruin3']:>8.0%}")
            cells[cell_name] = entry
            print("".join(line))
        report.setdefault(label, {})["joint"] = cells

        # the null: same number of entries drawn at random from the same pool
        rng = random.Random(20260824)
        draws = []
        for _ in range(args.null_draws):
            sample = rng.sample(rows, len(fast_broad))
            draws.append(statistics.median(r["rate"] for r in sample))
        draws.sort()
        actual = cells["fast and broad"]["median_rate"]
        beats = sum(1 for d in draws if d < actual) / len(draws)
        report[label]["joint_null"] = {
            "actual": actual, "null_median": draws[len(draws) // 2],
            "p05": draws[int(0.05 * len(draws))],
            "p95": draws[int(0.95 * len(draws))], "percentile": beats}
        print(f"  null: {args.null_draws} random draws of {len(fast_broad)} "
              f"entries give a median rate of "
              f"{draws[len(draws)//2]:+.1%}, 90% band "
              f"[{draws[int(0.05*len(draws))]:+.1%}, "
              f"{draws[int(0.95*len(draws))]:+.1%}]. "
              f"The cell sits at the {beats:.0%} percentile.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print()
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
