"""The drawdown rule on the whole company universe, with a price-you-pay gate.

The ETF version worked and could not be deployed.  Eighteen names produced 81
signals in twenty-seven years -- three a year -- so the book sat at 3-10% gross
and returned +2.5% against the index's +6.2%.  Nothing was wrong with the trade;
there was not enough of it.  A hundred and forty names is the obvious fix, and it
is the only change that raises exposure without raising the tail.

Two gates go in front of the trigger.

*Quality*, because the ETF result rests on an index recovering and a single
company need not.  Profitable, return on equity above the universe's, operating
margin above the universe's, liabilities-over-assets below it, dilution under
10%.

*Price*, because a quality screen with no valuation gate buys the compounders
everybody has already bid up, and a 20% drawdown from a bubble peak is not a
discount.  Earnings yield above the universe median -- a relative cut, not an
appraisal of fair value, and worth exactly what a relative cut is worth.

The valuation gate is the claim under test, so it is tested against its own
inverse: the same quality names in the same drawdowns, but the *expensive* half.
If the expensive half does as well, paying attention to price bought nothing.
The discount itself is tested the same way, by running identical gates at a 5%
pullback instead of 18%.

Facts are held unreadable for 90 days after their period ends, because the store
carries no filing date.  Sizing is risk divided by the fifth-percentile further
fall measured on this universe, not the ETF one -- single companies fall further
and the number has to come from the names actually being traded.

Survivorship is the honest caveat and it cannot be repaired here: this is a
present-day membership list, so every name in it survived to be listed.  The
equal-weight benchmark carries the identical bias, which is why the comparison is
against that rather than against an index.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import statistics
import sys
from bisect import bisect_right
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

from run_quality_value_drawdown import metrics  # noqa: E402

REPORT_LAG_DAYS = 90
# Revenue is tagged two ways. Companies moved to the ASC 606 concept during
# 2018 and most never backfilled the old one, so reading only ``Revenues``
# loses a third of the universe and pushes the usable window forward to 2015.
# They are the same line item and are merged, old tag preferred where both
# exist because it is the one with history.
REVENUE_TAGS = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax")
FLOWS = ("NetIncomeLoss", "OperatingIncomeLoss") + REVENUE_TAGS
INSTANTS = ("Assets", "Liabilities", "StockholdersEquity",
            "WeightedAverageNumberOfDilutedSharesOutstanding")

LOOKBACK = 252
RUNUP = 504
MAX_HOLD = 756
RISK_RUNGS = (0.01, 0.02, 0.03)
# Operating margin is not in the required set: it depends on revenue, which is
# the sparsest concept in the store, and demanding it costs five years of window
# for one of four gates. It is applied as a cut wherever it is readable.
NEEDED = ("leverage", "roe", "earn_yield")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path,
                        default=Path("data/13f/alphavantage_daily.db"))
    parser.add_argument("--equity", type=Path, default=Path("data/equity/equity.db"))
    parser.add_argument("--map", type=Path,
                        default=Path("out/strategy/ticker_cik.json"))
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--depth", type=float, default=0.18)
    parser.add_argument("--shallow", type=float, default=0.05)
    parser.add_argument("--max-runup", type=float, default=0.50)
    parser.add_argument("--max-positions", type=int, default=15)
    parser.add_argument("--financing", type=float, default=0.05)
    parser.add_argument("--null-draws", type=int, default=25)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/stock_portfolio.json"))
    known, _ = parser.parse_known_args(argv)
    return known


def fundamentals(args, mapping):
    """Point-in-time facts per ticker: (usable_from, metrics), oldest first.

    Flows are summed over the trailing four quarters, instants take the latest
    reading, and nothing is readable until 90 days after its period closes
    because the store carries no filing date.
    """
    connection = sqlite3.connect(f"file:{args.equity}?mode=ro", uri=True)
    by_cik = defaultdict(lambda: defaultdict(dict))
    wanted = set(FLOWS) | set(INSTANTS)
    ciks = tuple(mapping.values())
    marks = ",".join("?" * len(ciks))
    for cik, concept, start, end, value in connection.execute(
            f"SELECT cik, concept, period_start, period_end, value FROM sec_facts "
            f"WHERE cik IN ({marks}) AND value IS NOT NULL", ciks):
        if concept not in wanted or not end:
            continue
        if concept in FLOWS:
            if not start:
                continue
            span = (date.fromisoformat(end[:10])
                    - date.fromisoformat(start[:10])).days
            if not 60 <= span <= 100:        # quarterly only, so sums are clean
                continue
        by_cik[cik][concept][end[:10]] = float(value)
    connection.close()

    panel = {}
    for ticker, cik in mapping.items():
        facts = by_cik.get(cik)
        if not facts:
            continue
        merged = dict(facts.get(REVENUE_TAGS[1], {}))
        merged.update(facts.get(REVENUE_TAGS[0], {}))
        facts["Revenues"] = merged
        ends = sorted({d for series in facts.values() for d in series})
        rows = []
        for end in ends:
            entry = {}
            for concept in ("NetIncomeLoss", "OperatingIncomeLoss", "Revenues"):
                series = facts.get(concept, {})
                quarters = [series[d] for d in sorted(series) if d <= end][-4:]
                if len(quarters) == 4:
                    entry[concept] = sum(quarters)
            for concept in INSTANTS:
                series = facts.get(concept, {})
                prior = [series[d] for d in sorted(series) if d <= end]
                if prior:
                    entry[concept] = prior[-1]
                older = [series[d] for d in sorted(series)
                         if d <= end and (date.fromisoformat(end)
                                          - date.fromisoformat(d)).days >= 330]
                if concept.startswith("Weighted") and older:
                    entry["shares_year_ago"] = older[-1]
            usable = (date.fromisoformat(end)
                      + timedelta(days=REPORT_LAG_DAYS)).isoformat()
            rows.append((usable, entry))
        if rows:
            panel[ticker] = rows
    return panel


def load_book(args):
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    book = {}
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency='daily' "
            "ORDER BY ticker"):
        rows = connection.execute(
            "SELECT ts,open,high,low,close FROM bars WHERE ticker=? AND "
            "frequency='daily' AND ts>=? ORDER BY ts",
            (ticker, "2006-01-01")).fetchall()
        bars = adjust_bars([Bar(r[0][:10], r[1], r[2], r[3], r[4], None)
                            for r in rows], splits.get(ticker, []))
        if len(bars) >= 1200:
            book[ticker] = bars
    connection.close()
    return book


def monthly_medians(book, panel, args):
    """Cross-sectional medians of every gate, over the whole universe, monthly."""
    prices = {t: {b.timestamp: b.close for b in bars} for t, bars in book.items()}
    calendar = sorted({b.timestamp for bars in book.values() for b in bars
                       if b.timestamp >= args.start})
    months, seen = [], set()
    for day in calendar:
        if day[:7] not in seen:
            seen.add(day[:7])
            months.append(day)
    table = {}
    for day in months:
        rows = []
        for ticker, facts in panel.items():
            usable = [f for f in facts if f[0] <= day]
            price = prices.get(ticker, {}).get(day)
            if not usable or not price:
                continue
            got = metrics(usable[-1][1], price)
            if set(NEEDED) <= got.keys():
                rows.append(got)
        if len(rows) >= 25:
            entry = {k: statistics.median([r[k] for r in rows]) for k in NEEDED}
            margins = [r["op_margin"] for r in rows if "op_margin" in r]
            if len(margins) >= 20:
                entry["op_margin"] = statistics.median(margins)
            table[day] = entry
    return sorted(table), table


def crossings(book, panel, args, depth):
    """Fresh crossings, each carrying the facts readable that day and its run-up."""
    out = []
    for ticker, bars in book.items():
        facts = panel.get(ticker)
        if not facts:
            continue
        usable_days = [f[0] for f in facts]
        closes = [b.close for b in bars]
        lows = [b.low for b in bars]
        peak, running = [], []
        for i in range(len(bars)):
            running.append((closes[i], i))
            if len(running) > LOOKBACK:
                running.pop(0)
            peak.append(max(running))
        armed = True
        for i in range(LOOKBACK + RUNUP, len(bars) - 1):
            day = bars[i].timestamp
            if day < args.start:
                continue
            top, top_at = peak[i]
            if top <= 0 or closes[i] <= 0:
                continue
            drop = closes[i] / top - 1.0
            if drop > -depth * 0.6:
                armed = True
            if not armed or drop > -depth:
                continue
            armed = False
            position = bisect_right(usable_days, day) - 1
            if position < 0:
                continue
            got = metrics(facts[position][1], closes[i])
            if not set(NEEDED) <= got.keys():
                continue
            before = top_at - RUNUP
            runup = (closes[top_at] / closes[before] - 1.0
                     if before >= 0 and closes[before] > 0 else None)
            end = min(i + MAX_HOLD, len(bars) - 1)
            floor = min(lows[i + 1:end + 1] or [closes[i]])
            got.update({"ticker": ticker, "day": day, "target": top,
                        "runup": runup, "further": floor / closes[i] - 1.0})
            out.append(got)
    out.sort(key=lambda r: r["day"])
    return out


def gate(rows, month_days, medians, quality=False, value=None, runup=None):
    """quality: all four business cuts. value: 'cheap', 'rich' or None."""
    kept = []
    for r in rows:
        if runup is not None and (r["runup"] is None or r["runup"] > runup):
            continue
        at = bisect_right(month_days, r["day"]) - 1
        if at < 0:
            continue
        mid = medians[month_days[at]]
        if quality:
            if not (r["profitable"] > 0 and r["roe"] >= mid["roe"]
                    and r["leverage"] <= mid["leverage"]
                    and r.get("dilution", 0.0) < 0.10):
                continue
            # applied only where revenue is readable for both the name and the
            # cross-section; absent, the other three cuts stand alone
            if "op_margin" in r and "op_margin" in mid                     and r["op_margin"] < mid["op_margin"]:
                continue
        if value == "cheap" and r["earn_yield"] < mid["earn_yield"]:
            continue
        if value == "rich" and r["earn_yield"] >= mid["earn_yield"]:
            continue
        kept.append(r)
    return kept


def simulate(book, rows, risk, tail, args, begins):
    """Open on signal, hold to the prior high or three years, size by risk.

    ``begins`` is the first day the gates can be evaluated, not ``--start``.
    Walking NAV from 2010 while no signal can fire until the fundamentals panel
    is dense enough divides the compounding by five dead years and quietly
    understates every arm.
    """
    prices, calendar = args.cache
    by_day = defaultdict(list)
    for r in rows:
        by_day[r["day"]].append(r)
    size = risk / tail

    nav, peak, worst = 1.0, 1.0, 0.0
    live, exposures, counts = [], [], []
    refused, opened = 0, 0
    for a, b in zip(calendar, calendar[1:]):
        gross = sum(p["size"] for p in live)
        move = 0.0
        for position in live:
            p0 = prices[position["ticker"]].get(a)
            p1 = prices[position["ticker"]].get(b)
            if p0 and p1 and p0 > 0:
                move += position["size"] * (p1 / p0 - 1.0)
        carry = max(gross - 1.0, 0.0) * args.financing / 252.0
        nav *= (1.0 + move - carry)
        peak = max(peak, nav)
        worst = min(worst, nav / peak - 1.0)
        exposures.append(gross)
        counts.append(len(live))

        kept = []
        for position in live:
            price = prices[position["ticker"]].get(b)
            position["age"] += 1
            if not ((price is not None and price >= position["target"])
                    or position["age"] >= MAX_HOLD):
                kept.append(position)
        live = kept

        # cheapest first when more signal arrives in a day than the book can hold
        for signal in sorted(by_day.get(b, []), key=lambda r: -r["earn_yield"]):
            if any(p["ticker"] == signal["ticker"] for p in live):
                continue
            if len(live) >= args.max_positions:
                refused += 1
                continue
            live.append({"ticker": signal["ticker"], "target": signal["target"],
                         "size": size, "age": 0})
            opened += 1

    years = (date.fromisoformat(calendar[-1])
             - date.fromisoformat(calendar[0])).days / 365.25
    return {"cagr": nav ** (1 / years) - 1 if nav > 0 else -1.0,
            "terminal": nav, "max_drawdown": worst, "opened": opened,
            "refused": refused,
            "exposure_median": statistics.median(exposures),
            "exposure_p95": sorted(exposures)[int(0.95 * len(exposures))],
            "exposure_max": max(exposures),
            "positions_median": statistics.median(counts),
            "positions_max": max(counts)}


def equal_weight(book, begins):
    """Buy the whole universe on day one and hold. Same survivorship bias."""
    calendar = sorted({b.timestamp for bars in book.values() for b in bars
                       if b.timestamp >= begins})
    prices = {t: {b.timestamp: b.close for b in bars} for t, bars in book.items()}
    nav, peak, worst = 1.0, 1.0, 0.0
    for a, b in zip(calendar, calendar[1:]):
        moves = [prices[t][b] / prices[t][a] - 1.0 for t in book
                 if prices[t].get(a) and prices[t].get(b) and prices[t][a] > 0]
        if moves:
            nav *= 1.0 + sum(moves) / len(moves)
        peak = max(peak, nav)
        worst = min(worst, nav / peak - 1.0)
    years = (date.fromisoformat(calendar[-1])
             - date.fromisoformat(calendar[0])).days / 365.25
    return {"cagr": nav ** (1 / years) - 1, "terminal": nav,
            "max_drawdown": worst}


def percentile(values, share):
    ordered = sorted(values)
    return ordered[min(int(share * len(ordered)), len(ordered) - 1)]


def solve_risk(book, rows, tail, args, begins, target):
    """The risk budget that puts median gross exposure at ``target``.

    Arms cannot be compared at a common risk setting.  A gate that passes 52
    signals holds fewer positions than one passing 951, so it runs a smaller
    book and earns less for that reason alone -- which says nothing about
    whether the gate selects better names.  Matching exposure first is what
    separates selection from activity.
    """
    low, high = 0.0005, 0.40
    got = None
    for _ in range(14):
        mid = (low + high) / 2
        got = simulate(book, rows, mid, tail, args, begins)
        if got["exposure_median"] < target:
            low = mid
        else:
            high = mid
    return mid, got


def main(argv=None) -> int:
    args = parse_args(argv)
    mapping = json.loads(args.map.read_text(encoding="utf-8"))
    book = load_book(args)
    panel = fundamentals(args, mapping)
    book = {t: b for t, b in book.items() if t in panel}
    print(f"{len(book)} companies with price and readable facts, from {args.start}")

    month_days, medians = monthly_medians(book, panel, args)
    begins = month_days[0]
    args.cache = ({t: {b.timestamp: b.close for b in bars}
                   for t, bars in book.items()},
                  sorted({b.timestamp for bars in book.values() for b in bars
                          if b.timestamp >= month_days[0]}))
    print(f"universe medians on {len(month_days)} month-ends "
          f"({month_days[0][:7]} to {month_days[-1][:7]})")
    print(f"the book is walked from {begins}, the first day a gate can be read")

    deep = crossings(book, panel, args, args.depth)
    shallow = crossings(book, panel, args, args.shallow)
    print(f"{len(deep):,d} fresh crossings at {args.depth:.0%} below the "
          f"252-day high, {len(shallow):,d} at {args.shallow:.0%}")

    falls = [r["further"] for r in deep]
    tail = abs(percentile(falls, 0.05))
    print()
    print("########## the tail that sets position size ##########")
    print(f"  {len(deep):,d} entries. Worst mark below entry over three years, "
          "from daily lows.")
    for label, share in (("median", 0.50), ("75th worst", 0.25),
                         ("90th worst", 0.10), ("95th worst", 0.05),
                         ("99th worst", 0.01)):
        print(f"  {label:>12s} {percentile(falls, share):>8.1%}")
    print(f"  sizing denominator (95th worst): {tail:.0%}  "
          f"-- the ETF equivalent was 60%")
    report = {"companies": len(book), "crossings_deep": len(deep),
              "tail": tail, "arms": {}}

    arms = {
        "drawdown only": gate(deep, month_days, medians),
        "+ quality": gate(deep, month_days, medians, quality=True),
        "+ quality + cheap": gate(deep, month_days, medians, quality=True,
                                  value="cheap"),
        "+ quality + expensive": gate(deep, month_days, medians, quality=True,
                                      value="rich"),
        "+ quality + cheap, run-up filtered":
            gate(deep, month_days, medians, quality=True, value="cheap",
                 runup=args.max_runup),
        "5% pullback + quality + cheap":
            gate(shallow, month_days, medians, quality=True, value="cheap"),
    }

    print()
    print("########## what each arm passes ##########")
    for label, rows in arms.items():
        if not rows:
            continue
        base = len(shallow) if label.startswith("5%") else len(deep)
        print(f"  {label:36s} {len(rows):>6,d} signals ({len(rows)/base:>5.1%} "
              f"of crossings)  p05 further fall "
              f"{percentile([r['further'] for r in rows], 0.05):>7.1%}")

    print()
    print("########## returns, exposure and drawdown by risk per trade ##########")
    print(f"  book capped at {args.max_positions} positions; financing "
          f"{args.financing:.0%} on the borrowed portion; sizing risk / {tail:.0%}")
    print()
    print(f"  {'arm':36s} {'risk':>5s} {'CAGR':>8s} {'max DD':>8s} "
          f"{'x money':>8s} {'gross med':>10s} {'gross max':>10s} "
          f"{'held':>5s} {'refused':>8s}")
    for label, rows in arms.items():
        if len(rows) < 10:
            continue
        report["arms"][label] = {"signals": len(rows), "risk": {}}
        for risk in RISK_RUNGS:
            got = simulate(book, rows, risk, tail, args, begins)
            report["arms"][label]["risk"][f"{risk:.0%}"] = got
            print(f"  {label:36s} {risk:>4.0%} {got['cagr']:>+8.1%} "
                  f"{got['max_drawdown']:>8.1%} {got['terminal']:>7.2f}x "
                  f"{got['exposure_median']:>10.0%} {got['exposure_max']:>10.0%} "
                  f"{got['positions_median']:>5.0f} {got['refused']:>8,d}")
        print()

    print("########## the same arms held to a common 25% gross exposure ##########")
    print("  Risk is solved per arm so every book runs the same size. What is")
    print("  left is selection: whether the gate picks better names, not more.")
    print(f"  {'arm':36s} {'signals':>8s} {'risk':>7s} {'CAGR':>8s} "
          f"{'max DD':>8s} {'gross med':>10s}")
    report["matched"] = {}
    for label, rows in arms.items():
        if len(rows) < 10:
            continue
        risk, got = solve_risk(book, rows, tail, args, begins, 0.25)
        report["matched"][label] = {"risk": risk, **got}
        print(f"  {label:36s} {len(rows):>8,d} {risk:>6.2%} "
              f"{got['cagr']:>+8.1%} {got['max_drawdown']:>8.1%} "
              f"{got['exposure_median']:>10.0%}")

    print()
    print("########## the null: the same count, drawn at random ##########")
    print("  Every gate is a subset of the drawdown pool. Drawing the same")
    print("  number of entries at random from that pool, at matched exposure,")
    print("  says what a gate has to beat to have selected anything.")
    pool = arms["drawdown only"]
    rng = random.Random(20260824)
    report["null"] = {}
    for label in ("+ quality", "+ quality + cheap", "+ quality + expensive"):
        rows = arms.get(label)
        if not rows or len(rows) < 10:
            continue
        draws = []
        for _ in range(args.null_draws):
            sample = sorted(rng.sample(pool, len(rows)), key=lambda r: r["day"])
            _, got = solve_risk(book, sample, tail, args, begins, 0.25)
            draws.append(got["cagr"])
        draws.sort()
        actual = report["matched"][label]["cagr"]
        band = (draws[int(0.05 * len(draws))], draws[int(0.95 * len(draws))])
        beats = sum(1 for d in draws if d < actual) / len(draws)
        report["null"][label] = {"actual": actual, "median": draws[len(draws)//2],
                                 "p05": band[0], "p95": band[1], "percentile": beats}
        print(f"  {label:26s} actual {actual:>+7.1%}   null median "
              f"{draws[len(draws)//2]:>+7.1%}   90% band "
              f"[{band[0]:>+6.1%},{band[1]:>+6.1%}]   pct {beats:>4.0%}")

    print()
    bench = equal_weight(book, begins)
    report["benchmark"] = bench
    print("########## the benchmark, carrying the same survivorship bias ##########")
    print(f"  equal-weight buy and hold of all {len(book)} names: "
          f"CAGR {bench['cagr']:+.1%}, max drawdown {bench['max_drawdown']:.1%}, "
          f"{bench['terminal']:.2f}x money")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print()
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
