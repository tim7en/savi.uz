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
import math
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
    # Calendar-year returns measured close-of-prior-year to close-of-year. The
    # first-to-last-observation-within-the-year form used earlier silently drops
    # the turn-of-year move and counts a partial final year as a whole one.
    ends, by_year = {}, {}
    for year in sorted({d[:4] for d in days}):
        window = [d for d in days if d[:4] == year]
        if window:
            ends[year] = series[window[-1]]
    years_present = sorted(ends)
    final_year = days[-1][:4]
    partial_year = None
    for index in range(1, len(years_present)):
        year = years_present[index]
        previous = years_present[index - 1]
        value = ends[year] / ends[previous] - 1.0
        if year == final_year and days[-1][5:7] != "12":
            partial_year = {"year": year, "return": value, "to": days[-1]}
            continue
        by_year[year] = value
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
            "total_years": len(by_year), "partial_year": partial_year}


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

    # The published cyclically adjusted ratio, not a reconstruction. An earlier
    # version divided a nominal price by a ten-year mean of nominal earnings,
    # which omits the inflation adjustment the measure is defined by and ran 1.6
    # to 6.2 points high; the table carries the correct series already.
    shiller = equity.execute(
        "SELECT obs_date, cape FROM shiller_monthly WHERE obs_date >= '1990-01-01' "
        "AND cape IS NOT NULL ORDER BY obs_date").fetchall()
    cape = [[day[:7], round(value, 1)] for day, value in shiller]
    report["cape"] = cape
    report["cape_ends"] = cape[-1][0] if cape else None
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
    # A curve that un-inverts for two sessions has not ended an episode. Runs
    # separated by less than a fortnight are joined before the length filter is
    # applied, which is what turns the March-to-September 2000 pair into the one
    # episode a reader would recognise.
    merged = []
    for run in runs:
        if merged and (date.fromisoformat(run[0])
                       - date.fromisoformat(merged[-1][1])).days <= 14:
            merged[-1][1] = run[1]
        else:
            merged.append(list(run))
    report["inversions"] = [
        r for r in merged
        if (date.fromisoformat(r[1]) - date.fromisoformat(r[0])).days >= 30]
    report["inversion_note"] = (
        "Runs separated by 14 days or fewer are treated as one episode; episodes "
        "shorter than 30 days are dropped. Both rules are applied with hindsight "
        "and describe the record rather than a rule an investor could have "
        "followed on the first inverted day.")
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

    def newey_west(pairs, beta, intercept, lags=5):
        """Standard error robust to the autocorrelation daily returns carry."""
        n = len(pairs)
        xs = [x for x, _ in pairs]
        mean_x = statistics.fmean(xs)
        sxx = sum((x - mean_x) ** 2 for x in xs)
        if sxx <= 0:
            return float("nan")
        resid = [y - (intercept + beta * (x - mean_x)) for x, y in pairs]
        centred = [(x - mean_x) * r for (x, _), r in zip(pairs, resid)]
        total = sum(v * v for v in centred)
        for lag in range(1, lags + 1):
            weight = 1.0 - lag / (lags + 1)
            total += 2 * weight * sum(centred[i] * centred[i - lag]
                                      for i in range(lag, n))
        return math.sqrt(max(total, 0.0)) / sxx

    def regress(own, driver, exclude):
        """Sector return against a rate change, net of the other sectors.

        The comparison basket leaves the sector itself out. Including it puts a
        fraction of the dependent variable on both sides and pulls every
        coefficient toward zero.
        """
        pairs = []
        for day, value in own.items():
            if day not in driver or driver[day] is None:
                continue
            peers = [returns[o][day] for o in returns
                     if o != exclude and day in returns[o]]
            if len(peers) < 7:
                continue
            pairs.append((driver[day], value - statistics.fmean(peers)))
        if len(pairs) < 500:
            return None
        xs = [x for x, _ in pairs]
        ys = [y for _, y in pairs]
        mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
        sxx = sum((x - mean_x) ** 2 for x in xs)
        if sxx <= 0:
            return None
        beta = sum((x - mean_x) * (y - mean_y) for x, y in pairs) / sxx
        se = newey_west(pairs, beta, mean_y)
        return {"beta": beta * 100, "t": beta / se if se else float("nan"),
                "n": len(pairs)}

    ordered_days = sorted(change)
    lagged = {ordered_days[i]: change[ordered_days[i - 1]]
              for i in range(1, len(ordered_days))}

    betas = {}
    for ticker, own in returns.items():
        same = regress(own, change, ticker)
        after = regress(own, lagged, ticker)
        if not same:
            continue
        days_covered = sorted(own)
        betas[ticker] = {
            "name": SECTORS[ticker], "beta": same["beta"], "t": same["t"],
            "n": same["n"], "from": days_covered[0][:7],
            "lagged_beta": after["beta"] if after else None,
            "lagged_t": after["t"] if after else None}
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
        row = {"name": name, "from": start, "to": end, "sectors": {},
               "partial": []}
        span = (date.fromisoformat(end) - date.fromisoformat(start)).days
        for ticker, series in prices.items():
            window = [d for d in sorted(series) if start <= d <= end]
            if len(window) < 15:
                continue
            # A fund that listed midway through an episode did not live through
            # it, and ranking it against those that did is a comparison of
            # different periods wearing one label.
            covered = (date.fromisoformat(window[-1])
                       - date.fromisoformat(window[0])).days
            row["sectors"][ticker] = series[window[-1]] / series[window[0]] - 1.0
            if covered < span * 0.9:
                row["partial"].append(ticker)
        full = {k: v for k, v in row["sectors"].items()
                if k not in row["partial"]}
        row["best"] = max(full, key=full.get) if full else None
        row["worst"] = min(full, key=full.get) if full else None
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
