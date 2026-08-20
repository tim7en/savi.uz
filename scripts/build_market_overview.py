"""The quarter century in numbers, for the report's second chapter.

Everything here is arithmetic on stored data rather than analysis: what a hundred
dollars became in each sector, what the economy and the valuation of the market
were doing while that happened, and where the regime boundaries fell.  The point
is to give the reader the ground truth before any strategy is discussed, so that
later claims can be checked against what the period actually offered.

Two honesty constraints shape the output.  Sector funds are quoted from their own
first day rather than a common one, because XLRE and XLC did not exist in 2000
and pretending otherwise would flatter them by omitting the decade they missed;
both a common-window and an inception-to-date figure are printed.  And prices
here are split-adjusted but not dividend-adjusted, so the sector totals understate
total return -- materially for utilities and staples, least for technology.  The
S&P 500 total-return index is printed beside the price index to show the size of
that gap.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from datetime import date
from pathlib import Path

SECTORS = {
    "XLF": "Financials", "XLK": "Technology", "XLU": "Utilities",
    "XLV": "Health Care", "XLP": "Staples", "XLI": "Industrials",
    "XLY": "Discretionary", "XLB": "Materials", "XLE": "Energy",
    "XLRE": "Real Estate", "XLC": "Communications",
}
COMMODITIES = {"GLD": "Gold", "SLV": "Silver", "USO": "Crude oil",
               "UNG": "Natural gas", "DBA": "Agriculture", "GDX": "Gold miners",
               "TLT": "20y Treasuries", "HYG": "High yield", "UUP": "US dollar"}
EVENTS = [
    ("2000-03-10", "Nasdaq peaks; the dot-com unwind begins"),
    ("2001-09-11", "Attacks close US markets for four sessions"),
    ("2002-10-09", "S&P 500 troughs, 49% below its 2000 high"),
    ("2007-10-09", "Pre-crisis peak"),
    ("2008-09-15", "Lehman Brothers files"),
    ("2009-03-09", "S&P 500 troughs, 57% below peak"),
    ("2011-08-05", "US sovereign downgrade"),
    ("2013-05-22", "Taper tantrum"),
    ("2015-08-24", "China devaluation shock"),
    ("2018-02-05", "Volatility complex breaks"),
    ("2020-03-23", "Pandemic trough, 34% in 23 sessions"),
    ("2022-01-03", "Peak before the tightening bear"),
    ("2022-10-12", "Trough of the 2022 decline"),
    ("2024-08-07", "Longest yield-curve inversion on record ends"),
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etf", type=Path,
                        default=Path("data/cross_assets/etf_daily.db"))
    parser.add_argument("--equity", type=Path, default=Path("data/equity/equity.db"))
    parser.add_argument("--macro", type=Path, default=Path("data/macro/macro.db"))
    parser.add_argument("--common-start", default="2000-01-03")
    parser.add_argument("--out", type=Path,
                        default=Path("out/report/market_overview.json"))
    return parser.parse_args(argv)


def daily_closes(path: Path, ticker: str):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT substr(ts,1,10), close FROM bars WHERE ticker=? AND "
        "frequency='daily' ORDER BY ts", (ticker,)).fetchall()
    connection.close()
    return {d: c for d, c in rows if c}


def growth(closes, start=None):
    days = sorted(closes)
    if start:
        days = [d for d in days if d >= start]
    if len(days) < 200:
        return None
    first, last = closes[days[0]], closes[days[-1]]
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25
    peak, worst = first, 0.0
    for day in days:
        peak = max(peak, closes[day])
        worst = min(worst, closes[day] / peak - 1.0)
    by_year = {}
    for year in sorted({d[:4] for d in days}):
        window = [d for d in days if d[:4] == year]
        if len(window) > 100:
            by_year[year] = closes[window[-1]] / closes[window[0]] - 1.0
    return {"from": days[0], "to": days[-1], "years": years,
            "hundred_becomes": 100.0 * last / first,
            "cagr": (last / first) ** (1 / years) - 1 if years > 0 else None,
            "max_drawdown": worst,
            "best_year": max(by_year.items(), key=lambda kv: kv[1]) if by_year else None,
            "worst_year": min(by_year.items(), key=lambda kv: kv[1]) if by_year else None,
            "positive_years": sum(1 for v in by_year.values() if v > 0),
            "total_years": len(by_year), "by_year": by_year}


def main(argv=None):
    args = parse_args(argv)
    report = {}

    print("=" * 78)
    print("$100 INVESTED, PRICE RETURN ONLY (dividends excluded)\n")
    print(f"  {'sector':16s} {'from':>10s} {'$100 ->':>9s} {'CAGR':>7s} "
          f"{'max DD':>8s} {'up yrs':>7s} {'best':>14s} {'worst':>14s}")
    sector_rows = {}
    for ticker, name in SECTORS.items():
        closes = daily_closes(args.etf, ticker)
        if not closes:
            continue
        whole = growth(closes)
        common = growth(closes, args.common_start)
        if not whole:
            continue
        sector_rows[ticker] = {"name": name, "inception": whole, "common": common}
        best, worst = whole["best_year"], whole["worst_year"]
        print(f"  {name:16s} {whole['from']:>10s} {whole['hundred_becomes']:>9,.0f} "
              f"{whole['cagr']:>7.1%} {whole['max_drawdown']:>8.0%} "
              f"{whole['positive_years']:>3d}/{whole['total_years']:<3d} "
              f"{best[0]} {best[1]:>+6.0%} {worst[0]} {worst[1]:>+6.0%}")
    report["sectors"] = sector_rows

    equity = sqlite3.connect(f"file:{args.equity}?mode=ro", uri=True)
    print(f"\n  {'benchmark':16s} {'from':>10s} {'$100 ->':>9s} {'CAGR':>7s} "
          f"{'max DD':>8s}")
    benchmarks = {}
    for ticker, label in (("^GSPC", "S&P 500 price"), ("^SP500TR", "S&P 500 total"),
                          ("^NDX", "Nasdaq 100"), ("^RUT", "Russell 2000")):
        rows = equity.execute(
            "SELECT obs_date, close FROM index_prices WHERE ticker=? ORDER BY obs_date",
            (ticker,)).fetchall()
        closes = {d[:10]: c for d, c in rows if c}
        result = growth(closes, args.common_start)
        if result:
            benchmarks[label] = result
            print(f"  {label:16s} {result['from']:>10s} "
                  f"{result['hundred_becomes']:>9,.0f} {result['cagr']:>7.1%} "
                  f"{result['max_drawdown']:>8.0%}")
    report["benchmarks"] = benchmarks

    print(f"\n{'=' * 78}\nCOMMODITIES, RATES AND THE DOLLAR\n")
    print(f"  {'instrument':16s} {'from':>10s} {'$100 ->':>9s} {'CAGR':>7s} "
          f"{'max DD':>8s}")
    others = {}
    for ticker, name in COMMODITIES.items():
        closes = daily_closes(args.etf, ticker)
        result = growth(closes) if closes else None
        if result:
            others[ticker] = {"name": name, **result}
            print(f"  {name:16s} {result['from']:>10s} "
                  f"{result['hundred_becomes']:>9,.0f} {result['cagr']:>7.1%} "
                  f"{result['max_drawdown']:>8.0%}")
    report["commodities"] = others

    print(f"\n{'=' * 78}\nVALUATION AND THE ECONOMY\n")
    shiller = equity.execute(
        "SELECT obs_date, sp500_price, earnings, cpi, long_rate_gs10 "
        "FROM shiller_monthly WHERE obs_date >= '1999-01-01' ORDER BY obs_date"
    ).fetchall()
    cape = {}
    for i, (day, price, earn, cpi, rate) in enumerate(shiller):
        window = [e for _, _, e, _, _ in shiller[max(0, i - 119):i + 1] if e]
        if price and len(window) >= 60 and statistics.fmean(window) > 0:
            cape[day[:7]] = price / statistics.fmean(window)
    for label in ("1999-12", "2007-10", "2009-03", "2020-03", "2021-12", "2024-09"):
        if label in cape:
            print(f"  Shiller-style CAPE {label}: {cape[label]:.1f}")
    report["cape"] = cape
    equity.close()

    macro = sqlite3.connect(f"file:{args.macro}?mode=ro", uri=True)
    print()
    for series, name in (("GDPC1", "Real GDP"), ("UNRATE", "Unemployment"),
                         ("CPIAUCSL", "CPI")):
        rows = macro.execute(
            "SELECT obs_date, value FROM observations WHERE series_id=? "
            "AND obs_date >= '2000-01-01' ORDER BY obs_date", (series,)).fetchall()
        if len(rows) < 8:
            continue
        first, last = rows[0], rows[-1]
        report.setdefault("macro", {})[series] = {
            "first": first, "last": last, "n": len(rows)}
        print(f"  {name:14s} {first[0][:7]} {first[1]:>12,.1f}  ->  "
              f"{last[0][:7]} {last[1]:>12,.1f}   ({len(rows):,} observations)")
    macro.close()

    print(f"\n{'=' * 78}\nEVENT MARKERS\n")
    for day, note in EVENTS:
        print(f"  {day}  {note}")
    report["events"] = EVENTS

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
