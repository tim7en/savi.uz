"""Every series the market-overview chapter needs, in one JSON.

Two stores are used deliberately.  The daily store is split-adjusted only, so it
is a price index; the 30-minute store carries the vendor's fully adjusted series
and is therefore total return.  Reporting both is the point rather than an
inconvenience -- the gap between them is the dividend, and for utilities over a
quarter of a century it is larger than the price gain itself.

Charts need shape as well as endpoints, so growth paths are emitted monthly
rather than as a single ratio, and the valuation series is emitted in full.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from datetime import date
from pathlib import Path

SECTORS = {"XLF": "Financials", "XLK": "Technology", "XLU": "Utilities",
           "XLV": "Health care", "XLP": "Staples", "XLI": "Industrials",
           "XLY": "Discretionary", "XLB": "Materials", "XLE": "Energy",
           "XLRE": "Real estate", "XLC": "Communications"}
OTHERS = {"GLD": "Gold", "SLV": "Silver", "USO": "Crude oil", "UNG": "Natural gas",
          "DBA": "Agriculture", "GDX": "Gold miners", "TLT": "20y Treasuries",
          "IEF": "7-10y Treasuries", "LQD": "Investment grade", "HYG": "High yield",
          "UUP": "US dollar", "VXX": "Volatility"}
START = "2000-01-03"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily", type=Path,
                        default=Path("data/cross_assets/etf_daily.db"))
    parser.add_argument("--intraday", type=Path,
                        default=Path("data/cross_assets/etf_30min.db"))
    parser.add_argument("--equity", type=Path, default=Path("data/equity/equity.db"))
    parser.add_argument("--macro", type=Path, default=Path("data/macro/macro.db"))
    parser.add_argument("--out", type=Path,
                        default=Path("out/report/chapter2.json"))
    return parser.parse_args(argv)


def closes(connection, ticker, daily):
    query = ("SELECT substr(ts,1,10), close FROM bars WHERE ticker=? AND "
             "frequency='daily' ORDER BY ts" if daily else
             "SELECT substr(ts,1,10), close FROM bars WHERE ticker=? ORDER BY ts")
    out = {}
    for day, value in connection.execute(query, (ticker,)):
        if value:
            out[day] = value
    return out


def summarise(series, start=START):
    days = [d for d in sorted(series) if d >= start]
    if len(days) < 200:
        return None
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25
    peak, worst, trough_day = series[days[0]], 0.0, days[0]
    for day in days:
        peak = max(peak, series[day])
        drop = series[day] / peak - 1.0
        if drop < worst:
            worst, trough_day = drop, day
    by_year = {}
    for year in sorted({d[:4] for d in days}):
        window = [d for d in days if d[:4] == year]
        if len(window) > 100:
            by_year[year] = series[window[-1]] / series[window[0]] - 1.0
    monthly, seen = [], set()
    base = series[days[0]]
    for day in days:
        if day[:7] not in seen:
            seen.add(day[:7])
            monthly.append([day[:7], round(100.0 * series[day] / base, 1)])
    return {"from": days[0], "to": days[-1], "hundred": 100.0 * series[days[-1]] / base,
            "cagr": (series[days[-1]] / base) ** (1 / years) - 1,
            "max_drawdown": worst, "trough": trough_day, "by_year": by_year,
            "path": monthly,
            "positive_years": sum(1 for v in by_year.values() if v > 0),
            "total_years": len(by_year)}


def main(argv=None):
    args = parse_args(argv)
    daily = sqlite3.connect(f"file:{args.daily}?mode=ro", uri=True)
    intraday = sqlite3.connect(f"file:{args.intraday}?mode=ro", uri=True)
    report = {"sectors": {}, "others": {}}

    for ticker, name in SECTORS.items():
        price = summarise(closes(daily, ticker, True))
        total = summarise(closes(intraday, ticker, False))
        if price and total:
            report["sectors"][ticker] = {
                "name": name, "price": price, "total": total,
                "dividend_gap": total["hundred"] / price["hundred"] - 1.0}
    for ticker, name in OTHERS.items():
        total = summarise(closes(intraday, ticker, False),
                          start=min(closes(intraday, ticker, False) or ["9999"]))
        if total:
            report["others"][ticker] = {"name": name, "total": total}
    daily.close()
    intraday.close()

    equity = sqlite3.connect(f"file:{args.equity}?mode=ro", uri=True)
    for ticker, label in (("^GSPC", "S&P 500 price"), ("^SP500TR", "S&P 500 total"),
                          ("^NDX", "Nasdaq 100"), ("^RUT", "Russell 2000")):
        rows = equity.execute(
            "SELECT obs_date, close FROM index_prices WHERE ticker=? ORDER BY obs_date",
            (ticker,)).fetchall()
        result = summarise({d[:10]: c for d, c in rows if c})
        if result:
            report.setdefault("benchmarks", {})[label] = result

    shiller = equity.execute(
        "SELECT obs_date, sp500_price, earnings FROM shiller_monthly "
        "WHERE obs_date >= '1990-01-01' ORDER BY obs_date").fetchall()
    cape = []
    for i, (day, price, _) in enumerate(shiller):
        window = [e for _, _, e in shiller[max(0, i - 119):i + 1] if e]
        if price and len(window) >= 60 and statistics.fmean(window) > 0:
            cape.append([day[:7], round(price / statistics.fmean(window), 1)])
    report["cape"] = cape
    equity.close()

    macro = sqlite3.connect(f"file:{args.macro}?mode=ro", uri=True)
    for series, name in (("GDPC1", "real_gdp"), ("UNRATE", "unemployment"),
                         ("CPIAUCSL", "cpi")):
        rows = macro.execute(
            "SELECT obs_date, value FROM observations WHERE series_id=? AND "
            "obs_date >= '1999-01-01' ORDER BY obs_date", (series,)).fetchall()
        if rows:
            report.setdefault("macro", {})[name] = [[d[:7], v] for d, v in rows]
    curve = {}
    for day, mnemonic, value in macro.execute(
        "SELECT curve_date, mnemonic, value FROM gsw_rates WHERE mnemonic IN "
        "('SVENY02','SVENY10') AND curve_date >= '1999-01-01'"
    ):
        curve.setdefault(day, {})[mnemonic] = value
    spread, seen = [], set()
    for day in sorted(curve):
        pair = curve[day]
        if pair.get("SVENY02") is not None and pair.get("SVENY10") is not None:
            if day[:7] not in seen:
                seen.add(day[:7])
                spread.append([day[:7], round(pair["SVENY10"] - pair["SVENY02"], 3)])
    report["curve_spread"] = spread
    inverted, runs, start = None, [], None
    for day in sorted(curve):
        pair = curve[day]
        if pair.get("SVENY02") is None or pair.get("SVENY10") is None:
            continue
        state = pair["SVENY10"] < pair["SVENY02"]
        if state and not inverted:
            start = day
        elif not state and inverted and start:
            runs.append([start, day])
        inverted = state
    report["inversions"] = [r for r in runs
                            if (date.fromisoformat(r[1])
                                - date.fromisoformat(r[0])).days >= 30]
    macro.close()

    # ---- how each sector answers to rates, and to the named episodes -------
    intraday = sqlite3.connect(f"file:{args.intraday}?mode=ro", uri=True)
    macro2 = sqlite3.connect(f"file:{args.macro}?mode=ro", uri=True)
    level = dict(macro2.execute(
        "SELECT curve_date, value FROM gsw_rates WHERE mnemonic='SVENY10'"))
    macro2.close()
    ordered = sorted(d for d in level if level[d] is not None)
    change = {ordered[i]: level[ordered[i]] - level[ordered[i - 1]]
              for i in range(1, len(ordered))}

    prices = {t: closes(intraday, t, False) for t in SECTORS}
    intraday.close()
    returns = {}
    for ticker, series in prices.items():
        days = sorted(series)
        returns[ticker] = {days[i]: series[days[i]] / series[days[i - 1]] - 1.0
                           for i in range(1, len(days)) if series[days[i - 1]] > 0}
    common = sorted({d for r in returns.values() for d in r})
    market = {}
    for day in common:
        values = [returns[t][day] for t in returns if day in returns[t]]
        if len(values) >= 8:
            market[day] = statistics.fmean(values)

    betas = {}
    for ticker, own in returns.items():
        pairs = [(change[d], own[d] - market[d]) for d in own
                 if d in change and d in market]
        if len(pairs) < 500:
            continue
        xs = [x for x, _ in pairs]
        ys = [y for _, y in pairs]
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        beta = sum((x - mx) * (y - my) for x, y in pairs) / sxx if sxx else 0.0
        resid = [y - (my + beta * (x - mx)) for x, y in pairs]
        se = ((sum(r * r for r in resid) / (len(pairs) - 2) / sxx) ** 0.5
              if sxx else float("nan"))
        betas[ticker] = {"name": SECTORS[ticker], "beta": beta * 100,
                         "t": beta / se if se else float("nan"), "n": len(pairs)}
    report["rate_betas"] = betas

    EPISODES = [
        ("Dot-com decline", "2000-03-24", "2002-10-09"),
        ("Recovery to the peak", "2002-10-09", "2007-10-09"),
        ("Financial crisis", "2007-10-09", "2009-03-09"),
        ("The long expansion", "2009-03-09", "2020-02-19"),
        ("Pandemic crash", "2020-02-19", "2020-03-23"),
        ("Stimulus rally", "2020-03-23", "2022-01-03"),
        ("Tightening bear", "2022-01-03", "2022-10-12"),
        ("Since the trough", "2022-10-12", "2026-08-19"),
    ]
    episodes = []
    for name, start, end in EPISODES:
        row = {"name": name, "from": start, "to": end, "sectors": {}}
        for ticker, series in prices.items():
            window = [d for d in sorted(series) if start <= d <= end]
            if len(window) > 15:
                row["sectors"][ticker] = series[window[-1]] / series[window[0]] - 1.0
        episodes.append(row)
    report["episodes"] = episodes

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, separators=(",", ":")), encoding="utf-8")
    size = args.out.stat().st_size
    print(f"  sectors {len(report['sectors'])}, others {len(report['others'])}, "
          f"benchmarks {len(report.get('benchmarks', {}))}")
    print(f"  cape points {len(report['cape'])}, curve points {len(report['curve_spread'])}, "
          f"inversions {len(report['inversions'])}")
    for run in report["inversions"]:
        print(f"     inverted {run[0]} -> {run[1]}")
    print(f"  wrote {args.out} ({size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
