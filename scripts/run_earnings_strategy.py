"""Earnings drift written as a rule: entry, exit, size, and what it compounds to.

Two timescales, because the question spans both.

*The century.*  Per-name earnings surprises only exist back to 1999 here, so the
long view has to be taken at the index.  Shiller's monthly series carries S&P
earnings and real total return from 1871, which is 154 years -- enough to ask
whether a period of rising index earnings is followed by better real returns
than a period of falling ones, across every regime the market has had rather
than the two-and-a-half this programme usually sees.

*The names.*  141 instruments, 1999-2026, one earnings event per name per
quarter.  This is where the rule is specified and sized.

The rule under test, stated once:

* **Entry.**  The session after an earnings reaction, at the open -- the first
  price a person reading the release could obtain.  Long only.  Which reactions
  qualify is the arm being tested and is not chosen after the fact: four entry
  rules run side by side, including the deliberately-worst one.
* **Exit.**  A time exit at the drift horizon, with a volatility stop beneath it.
  The effect measured is a drift in the conditional mean, not a bracket, so the
  clock is the primary exit and the stop is there to bound the left tail rather
  than to harvest anything.
* **Size.**  Risk budget divided by stop distance, so a name whose daily
  deviation is 6% takes a third of the position a 2% name takes.  This is the
  whole risk-adjustment logic: exposure is inversely proportional to the
  volatility being underwritten, which makes leverage an output.

Leveraged and inverse wrappers are excluded.  A 3x fund's daily deviation is a
mechanical multiple of its underlying, so including them would let the sizing
rule believe it had found extra names when it had found the same exposure three
times over.

Horizon, stop and slot count are chosen on the first half and the second half is
untouched until they are frozen.  Every arm is reported against a drift null
drawn from ordinary sessions in the same names.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import statistics
import sys
import zlib
from bisect import bisect_left
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

import run_vol_stretch_zones as shared  # noqa: E402

LEVERED = ("2X", "3X", "1.5X", "BULL", "BEAR", "LEVERAGED", "INVERSE",
           "ULTRA", "PROSHARES", "DIREXION", "-1X")
HOLDS = (10, 20, 40)
STOPS = (2.0, 3.0, 4.0)
SLOTS = (6, 12)
RISK_RUNGS = (0.005, 0.01, 0.02, 0.03)
LEVERAGE_CAP = 20.0

ARMS = (
    ("beat and rose", lambda r: r["surprise_pct"] > 0 and r["reaction"] > 0),
    ("rose, any surprise", lambda r: r["reaction"] > 0),
    ("every announcement", lambda r: True),
    ("miss and fell (worst bucket)",
     lambda r: r["surprise_pct"] <= 0 and r["reaction"] <= 0),
)


def ticker_seed(t):
    return zlib.crc32(t.encode("utf-8")) % 10_000


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path,
                        default=Path("data/13f/alphavantage_daily.db"))
    parser.add_argument("--equity", type=Path, default=Path("data/equity/equity.db"))
    parser.add_argument("--earnings", type=Path, default=Path("data/sp500_data"))
    parser.add_argument("--split", default="2013-01-01")
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--maker-bp", type=float, default=2.5)
    parser.add_argument("--taker-bp", type=float, default=5.0)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--null-draws", type=int, default=40)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/earnings_strategy.json"))
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# the century, at the index


def century(args):
    connection = sqlite3.connect(f"file:{args.equity}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT obs_date, earnings, real_total_return_price FROM shiller_monthly "
        "WHERE earnings IS NOT NULL AND real_total_return_price IS NOT NULL "
        "ORDER BY obs_date").fetchall()
    connection.close()
    if len(rows) < 400:
        return None
    days = [r[0][:10] for r in rows]
    earn = [float(r[1]) for r in rows]
    total = [float(r[2]) for r in rows]
    out = {"span": (days[0], days[-1]), "months": len(rows), "buckets": {}}
    print(f"\n########## the century: S&P earnings and real total return "
          f"##########")
    print(f"  {days[0]} to {days[-1]}, {len(rows):,d} monthly observations\n")
    print(f"  {'trailing 12m earnings':26s} {'n':>6s} " +
          " ".join(f"{h:>10s}" for h in ("+1y", "+3y", "+5y", "+10y")))
    horizons = (12, 36, 60, 120)
    buckets = [("falling more than 20%", lambda g: g <= -0.20),
               ("falling 0 to 20%", lambda g: -0.20 < g <= 0.0),
               ("rising 0 to 20%", lambda g: 0.0 < g <= 0.20),
               ("rising more than 20%", lambda g: g > 0.20)]
    prepared = []
    for i in range(12, len(rows) - max(horizons)):
        if earn[i - 12] <= 0:
            continue
        growth = earn[i] / earn[i - 12] - 1.0
        forwards = [(total[i + h] / total[i]) ** (12 / h) - 1.0 for h in horizons]
        prepared.append((growth, forwards))
    base = [statistics.fmean(col) for col in zip(*[f for _, f in prepared])]
    print(f"  {'all months':26s} {len(prepared):>6,d} " +
          " ".join(f"{b:>+10.2%}" for b in base))
    for name, test in buckets:
        chunk = [f for g, f in prepared if test(g)]
        if len(chunk) < 40:
            continue
        means = [statistics.fmean(col) for col in zip(*chunk)]
        out["buckets"][name] = {"n": len(chunk),
                                "forward": dict(zip(("1y","3y","5y","10y"), means))}
        print(f"  {name:26s} {len(chunk):>6,d} " +
              " ".join(f"{m:>+10.2%}" for m in means))
    out["baseline"] = dict(zip(("1y","3y","5y","10y"), base))
    return out


# ---------------------------------------------------------------------------
# the names


def load_book(args):
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    try:
        names = {t: (n or "").upper() for t, n in connection.execute(
            "SELECT ticker, name FROM symbols")}
    except sqlite3.OperationalError:
        names = {}
    book, dropped = {}, []
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency='daily' "
            "ORDER BY ticker"):
        label = names.get(ticker, "")
        if any(marker in label for marker in LEVERED) or any(
                marker in ticker.upper() for marker in ("3X", "2X")):
            dropped.append(ticker)
            continue
        rows = connection.execute(
            "SELECT ts,open,high,low,close FROM bars WHERE ticker=? AND "
            "frequency='daily' ORDER BY ts", (ticker,)).fetchall()
        bars = adjust_bars([Bar(r[0][:10], r[1], r[2], r[3], r[4], None)
                            for r in rows], splits.get(ticker, []))
        if len(bars) >= 750:
            book[ticker] = bars
    connection.close()
    return book, dropped


def load_earnings(folder, tickers):
    out = {}
    for ticker in tickers:
        path = folder / f"{ticker}_earnings.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))["data"]
        except Exception:
            continue
        rows = []
        for row in payload.get("quarterlyEarnings", []):
            try:
                pct = float(row["surprisePercentage"])
            except (TypeError, ValueError, KeyError):
                continue
            day = str(row.get("reportedDate", ""))[:10]
            if len(day) == 10:
                rows.append({"date": day, "surprise_pct": pct,
                             "post": str(row.get("reportTime", "")).startswith("post")})
        if rows:
            out[ticker] = sorted(rows, key=lambda r: r["date"])
    return out


def deviations(bars, window):
    table, returns = {}, []
    for i in range(1, len(bars)):
        a, b = bars[i - 1].close, bars[i].close
        if a > 0 and b > 0:
            returns.append(math.log(b / a))
        if len(returns) > window:
            returns.pop(0)
        if len(returns) == window:
            table[bars[i].timestamp] = statistics.pstdev(returns)
    return table


def build_events(book, earnings, args):
    events, quiet = [], []
    for ticker, bars in book.items():
        rows = earnings.get(ticker)
        if not rows:
            continue
        days = [b.timestamp for b in bars]
        closes = [b.close for b in bars]
        sigma = deviations(bars, args.vol_window)
        marked = set()
        for row in rows:
            position = bisect_left(days, row["date"])
            reaction = position + 1 if row["post"] else position
            if reaction < args.vol_window + 2 or reaction + 45 >= len(bars):
                continue
            s = sigma.get(days[reaction - 1])
            prior = closes[reaction - 1]
            if not s or s <= 0 or prior <= 0:
                continue
            marked.add(reaction)
            events.append({"ticker": ticker, "index": reaction,
                           "day": days[reaction],
                           "surprise_pct": row["surprise_pct"],
                           "reaction": (closes[reaction] - prior) / (s * prior),
                           "sigma": s})
        for i in range(args.vol_window + 2, len(bars) - 45):
            if i in marked or (i * 2654435761) % 100 >= 20:
                continue
            s = sigma.get(days[i - 1])
            if s and s > 0:
                quiet.append({"ticker": ticker, "index": i, "day": days[i],
                              "sigma": s})
    return events, quiet


def run(book, chosen, hold, stop_mult, args):
    """Enter at the next open, exit on the clock unless the stop goes first."""
    trades = []
    for event in chosen:
        bars = book[event["ticker"]]
        start = event["index"] + 1
        fill = bars[start].open
        risk = stop_mult * event["sigma"] * fill
        if fill <= 0 or risk <= 0:
            continue
        stop = fill - risk
        last = min(start + hold, len(bars) - 1)
        price, reason, when = bars[last].close, "time", bars[last].timestamp
        for i in range(start, last + 1):
            if bars[i].low <= stop:
                price, reason, when = stop, "stop", bars[i].timestamp
                break
        cost = (args.taker_bp + args.taker_bp) / 10_000 * fill / risk
        trades.append({"ticker": event["ticker"], "entry": event["day"],
                       "exit": when, "r": (price - fill) / risk - cost,
                       "reason": reason, "stop_pct": risk / fill})
    trades.sort(key=lambda t: t["entry"])
    return trades


def compound(taken, fraction, cap):
    per_day, levers = defaultdict(float), []
    for t in taken:
        lever = min(fraction / t["stop_pct"], cap)
        levers.append(lever)
        per_day[t["exit"]] += t["r"] * lever * t["stop_pct"]
    days = sorted(per_day)
    nav, peak, worst = 1000.0, 1000.0, 0.0
    for d in days:
        nav = max(0.0, nav + per_day[d] * nav)
        peak = max(peak, nav)
        worst = min(worst, nav / peak - 1.0)
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25
    stream = [per_day[d] for d in days]
    sd = statistics.pstdev(stream)
    return {"cagr": (nav / 1000.0) ** (1 / years) - 1 if nav > 0 else -1.0,
            "max_drawdown": worst, "terminal": nav / 1000.0,
            "sharpe": (statistics.fmean(stream) / sd * math.sqrt(252)) if sd > 0 else float("nan"),
            "median_leverage": statistics.median(levers), "max_leverage": max(levers),
            "share_capped": sum(1 for x in levers if x >= cap) / len(levers)}


def main(argv=None) -> int:
    args = parse_args(argv)
    report = {"century": century(args)}

    book, dropped = load_book(args)
    earnings = load_earnings(args.earnings, sorted(book))
    events, quiet = build_events(book, earnings, args)
    covered = len({e["ticker"] for e in events})
    print(f"\n########## the names ##########")
    print(f"  {len(book)} instruments after dropping {len(dropped)} levered or "
          f"inverse wrappers")
    print(f"  {covered} with earnings, {len(events):,d} announcements, "
          f"{len(quiet):,d} ordinary sessions for the null")
    print(f"  in sample to {args.split}, out of sample after\n")
    report["names"] = {"instruments": len(book), "dropped": dropped,
                       "with_earnings": covered, "events": len(events)}

    for label, rule in ARMS:
        chosen = [e for e in events if rule(e)]
        if len(chosen) < 300:
            continue
        best, best_score = None, -99.0
        for hold in HOLDS:
            for stop_mult in STOPS:
                for slots in SLOTS:
                    trades = run(book, chosen, hold, stop_mult, args)
                    early = [t for t in trades if t["entry"] < args.split]
                    if len(early) < 200:
                        continue
                    args.max_positions = slots
                    result = shared.assess(early, args)
                    if result and result["sharpe"] > best_score:
                        best_score, best = result["sharpe"], (hold, stop_mult, slots)
        if best is None:
            continue
        hold, stop_mult, slots = best
        args.max_positions = slots
        trades = run(book, chosen, hold, stop_mult, args)
        outside = [t for t in trades if t["entry"] >= args.split]
        result = shared.assess(outside, args)
        print(f"### {label}")
        print(f"    chosen in sample: hold {hold} sessions, stop {stop_mult:g} "
              f"deviations, {slots} slots  (IS Sharpe {best_score:.2f})")
        if not result:
            print("    out of sample: not assessable\n")
            continue
        stops = sum(1 for t in outside if t["reason"] == "stop") / len(outside)
        print(f"    out of sample: {len(outside):,d} offered, "
              f"{result['taken']:,d} taken, {stops:.1%} stopped, "
              f"Sharpe {result['sharpe']:.2f} "
              f"[{result['sharpe_p05']:.2f}-{result['sharpe_p95']:.2f}]")

        nulls = []
        for draw in range(args.null_draws):
            rng = random.Random(64_000 + 137 * draw)
            by_ticker = defaultdict(list)
            for q in quiet:
                by_ticker[q["ticker"]].append(q)
            want = defaultdict(int)
            for e in chosen:
                want[e["ticker"]] += 1
            picked = []
            for ticker, count in want.items():
                pool = by_ticker.get(ticker, [])
                if pool:
                    r = random.Random(64_000 + 137 * draw + ticker_seed(ticker))
                    picked.extend(r.sample(pool, min(count, len(pool))))
            drawn = run(book, sorted(picked, key=lambda e: e["day"]),
                        hold, stop_mult, args)
            pooled = [t for t in drawn if t["entry"] >= args.split]
            if len(pooled) < 200:
                continue
            outcome = shared.assess(pooled, args)
            if outcome:
                nulls.append(outcome["sharpe"])
        entry = {"chosen": {"hold": hold, "stop": stop_mult, "slots": slots},
                 "in_sample_sharpe": best_score, "out_of_sample": result,
                 "stop_rate": stops}
        if nulls:
            nulls.sort()
            above = sum(1 for x in nulls if x >= result["sharpe"])
            entry["null"] = {"median": statistics.median(nulls), "low": nulls[0],
                             "high": nulls[-1], "p": above / len(nulls)}
            verdict = "clears" if above / len(nulls) <= 0.05 else "inside"
            print(f"    drift null:    {statistics.median(nulls):.2f} "
                  f"[{nulls[0]:.2f}-{nulls[-1]:.2f}], p = {above/len(nulls):.2f} "
                  f"-> {verdict} its null")

        taken = shared.cap(outside, slots, random.Random(0))
        print(f"    {'risk':>6s} {'CAGR':>8s} {'max DD':>9s} {'x money':>9s} "
              f"{'med lev':>8s} {'max lev':>8s} {'capped':>7s}")
        entry["compounding"] = []
        for fraction in RISK_RUNGS:
            row = compound(taken, fraction, LEVERAGE_CAP)
            entry["compounding"].append({"risk": fraction, **row})
            print(f"    {fraction:>5.1%} {row['cagr']:>+8.1%} "
                  f"{row['max_drawdown']:>9.1%} {row['terminal']:>8.1f}x "
                  f"{row['median_leverage']:>7.2f}x {row['max_leverage']:>7.2f}x "
                  f"{row['share_capped']:>7.1%}")
        print()
        report.setdefault("arms", {})[label] = entry

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
