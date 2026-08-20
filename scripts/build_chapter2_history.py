"""Century-scale context and a policy-regime taxonomy for chapter two.

Two things the chapter lacked. It opened in 2000 without telling the reader
whether that quarter century was ordinary, and it discussed rate *changes* and
inversions without ever naming the *regimes* -- the zero bound, the liftoff, the
2022 tightening -- which is what an investor actually lived through.

The long record runs on real total return, because over 154 years inflation is
not a detail: a dollar of 1871 buys about three cents today, so a nominal series
would be describing the currency rather than the market.

Regimes are classified from the policy rate itself rather than from a list of
remembered episodes: each month is labelled by where the funds rate sits relative
to twelve months earlier, with a separate label for the zero bound. That is
reproducible and, unlike a hand-drawn timeline, cannot quietly encode what the
author knows came next -- though the thresholds are still chosen with hindsight
and the labels describe the record rather than a rule anyone could have followed
in advance.
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
ZERO_BOUND = 0.5      # per cent; below this the rate is at its floor
MOVE = 0.5            # per cent over twelve months to count as a change


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equity", type=Path, default=Path("data/equity/equity.db"))
    parser.add_argument("--macro", type=Path, default=Path("data/macro/macro.db"))
    parser.add_argument("--intraday", type=Path,
                        default=Path("data/cross_assets/etf_30min.db"))
    parser.add_argument("--out", type=Path,
                        default=Path("out/report/chapter2_history.json"))
    return parser.parse_args(argv)


def century(path: Path):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT obs_date, real_total_return_price, real_price, cape, cpi, "
        "long_rate_gs10 FROM shiller_monthly ORDER BY obs_date").fetchall()
    connection.close()
    months = [(d[:7], tr, rp, cape, cpi, rate) for d, tr, rp, cape, cpi, rate in rows
              if tr]
    first, last = months[0], months[-1]
    years = (int(last[0][:4]) - int(first[0][:4])) + (
        int(last[0][5:7]) - int(first[0][5:7])) / 12

    # Real total return by decade, which is the only fair way to show a series
    # spanning two currencies' worth of inflation.
    by_decade = {}
    for decade in range(1870, 2030, 10):
        window = [m for m in months if decade <= int(m[0][:4]) < decade + 10]
        if len(window) > 60:
            growth = window[-1][1] / window[0][1]
            span = len(window) / 12
            by_decade[f"{decade}s"] = (growth ** (1 / span) - 1) * 100

    peak, worst, trough = months[0][1], 0.0, months[0][0]
    drawdowns = []
    running_peak, peak_month = months[0][1], months[0][0]
    in_draw = False
    for month, tr, *_ in months:
        if tr >= running_peak:
            if in_draw and worst < -0.20:
                drawdowns.append({"from": peak_month, "trough": trough,
                                  "depth": worst * 100})
            running_peak, peak_month, worst, in_draw = tr, month, 0.0, False
        else:
            in_draw = True
            drop = tr / running_peak - 1
            if drop < worst:
                worst, trough = drop, month
    if in_draw and worst < -0.20:
        drawdowns.append({"from": peak_month, "trough": trough, "depth": worst * 100})

    capes = sorted(c for _, _, _, c, _, _ in months if c)
    latest = [c for _, _, _, c, _, _ in months if c][-1]
    rank = sum(1 for c in capes if c < latest) / len(capes)
    rates = [r for *_, r in months if r]
    return {
        "from": first[0], "to": last[0], "years": years,
        "real_growth_of_one": last[1] / first[1],
        "real_cagr": ((last[1] / first[1]) ** (1 / years) - 1) * 100,
        "inflation_multiple": last[4] / first[4],
        "by_decade": by_decade,
        "drawdowns": sorted(drawdowns, key=lambda d: d["depth"])[:8],
        "drawdowns_over_20": len(drawdowns),
        "cape_percentiles": {p: round(capes[int(p / 100 * (len(capes) - 1))], 1)
                             for p in (5, 25, 50, 75, 95)},
        "cape_latest": latest, "cape_rank": rank * 100,
        "long_rate_min": min(rates), "long_rate_max": max(rates),
    }


def policy_regimes(macro: Path, intraday: Path):
    connection = sqlite3.connect(f"file:{macro}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT obs_date, value FROM observations WHERE series_id='DFF' "
        "AND obs_date >= '1998-01-01' ORDER BY obs_date").fetchall()
    connection.close()
    daily = {d[:10]: v for d, v in rows if v is not None}
    days = sorted(daily)

    label = {}
    for index, day in enumerate(days):
        past = days[max(0, index - 365)]
        if index < 365:
            continue
        rate, before = daily[day], daily[past]
        if rate < ZERO_BOUND:
            label[day] = "At the floor"
        elif rate - before >= MOVE:
            label[day] = "Tightening"
        elif before - rate >= MOVE:
            label[day] = "Easing"
        else:
            label[day] = "On hold"

    # Contiguous spells, so the reader sees episodes rather than a scatter.
    spells, current = [], None
    for day in sorted(label):
        if current is None or label[day] != current["label"]:
            if current and (date.fromisoformat(day)
                            - date.fromisoformat(current["from"])).days >= 120:
                current["to"] = day
                spells.append(current)
            current = {"label": label[day], "from": day, "to": day}
        else:
            current["to"] = day
    if current:
        spells.append(current)

    store = sqlite3.connect(f"file:{intraday}?mode=ro", uri=True)
    prices = {}
    for ticker in SECTORS:
        series = {}
        for day, close in store.execute(
            "SELECT substr(ts,1,10), close FROM bars WHERE ticker=? ORDER BY ts",
            (ticker,)
        ):
            if close:
                series[day] = close
        if len(series) > 400:
            prices[ticker] = series
    store.close()

    for spell in spells:
        spell["rate_from"] = daily.get(spell["from"])
        spell["rate_to"] = daily.get(spell["to"])
        spell["months"] = round((date.fromisoformat(spell["to"])
                                 - date.fromisoformat(spell["from"])).days / 30.4)
        returns = {}
        for ticker, series in prices.items():
            window = [d for d in sorted(series) if spell["from"] <= d <= spell["to"]]
            covered = ((date.fromisoformat(window[-1]) - date.fromisoformat(window[0])).days
                       if len(window) > 15 else 0)
            span = (date.fromisoformat(spell["to"])
                    - date.fromisoformat(spell["from"])).days
            if covered >= span * 0.9:
                returns[ticker] = (series[window[-1]] / series[window[0]] - 1) * 100
        spell["sectors"] = returns
        spell["best"] = max(returns, key=returns.get) if returns else None
        spell["worst"] = min(returns, key=returns.get) if returns else None
    return [s for s in spells if s["months"] >= 6]


def main(argv=None):
    args = parse_args(argv)
    report = {"century": century(args.equity),
              "regimes": policy_regimes(args.macro, args.intraday)}
    c = report["century"]
    print(f"CENTURY  {c['from']} -> {c['to']}  ({c['years']:.0f} years)")
    print(f"  $1 real, dividends reinvested -> ${c['real_growth_of_one']:,.0f}"
          f"   ({c['real_cagr']:.2f}% a year after inflation)")
    print(f"  prices rose {c['inflation_multiple']:.0f}-fold over the same span")
    print(f"  real declines worse than 20%: {c['drawdowns_over_20']}")
    for d in c["drawdowns"][:5]:
        print(f"     {d['from']} to {d['trough']}   {d['depth']:.0f}%")
    print(f"  CAPE percentiles {c['cape_percentiles']}")
    print(f"  latest {c['cape_latest']:.1f} sits at the {c['cape_rank']:.0f}th percentile")
    print(f"  long rate ranged {c['long_rate_min']:.2f}% to {c['long_rate_max']:.2f}%")
    print(f"\n  real return by decade:")
    for k, v in c["by_decade"].items():
        print(f"     {k}  {v:>+6.1f}%")
    print(f"\nPOLICY REGIMES since 1999: {len(report['regimes'])} spells")
    names = SECTORS
    for s in report["regimes"]:
        tail = ""
        if s["best"]:
            tail = (f"  best {names[s['best']]} {s['sectors'][s['best']]:+.0f}%"
                    f"  worst {names[s['worst']]} {s['sectors'][s['worst']]:+.0f}%")
        print(f"  {s['label']:12s} {s['from']} -> {s['to']} ({s['months']:>3d}m) "
              f"{s['rate_from']:.2f}% -> {s['rate_to']:.2f}%{tail}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
