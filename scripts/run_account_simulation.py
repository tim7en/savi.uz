"""A real account: $10,000, monthly contributions, two-unit pyramiding.

Sharpe answers a time-weighted question and this is a money-weighted one. When
money arrives monthly, *when* it arrives matters -- contributions made before a
recovery compound differently from the same dollars added at a peak. So this
runs the actual cash flows rather than reporting a ratio.

Three deployment policies on identical returns:

* **monthly** -- $1,000 on the first session of each month, always;
* **dip-only** -- the same $1,000 accrues in cash and is deployed only once the
  equity curve is 15% or more below its high-water mark;
* **lump sum** -- everything on day one, as the ceiling that timing must beat.

Each is run against the strategy and against SPY buy-and-hold with the same
schedule, because the question is not whether the strategy makes money but
whether it beats the passive alternative for the same dollars at the same times.

Gross exposure is reported because it is the thing a backtest silently hides.
Position size is `equity x risk / N`, so notional per unit is
`equity x risk x price / N` -- which means exposure depends on how volatile the
instruments happen to be, not on any exposure target. A book can be at 30% or
300% invested without the risk setting changing at all.

The returns driving this are the 4-hour, two-unit, 10bp configuration: the
Sharpe optimum of the pyramid sweep, not the banked four-unit version.
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

from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import TurtleConfig, run_turtle  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

BASE = dict(entry_window=55, exit_window=20, atr_window=20,
            skip_after_winner=False, use_channel_exit=False, chandelier_atr=3.0)

LEVERED = ("2X", "3X", "1.5X", "BULL", "BEAR", "LEVERAGED", "INVERSE",
           "ULTRA", "DAILY ", "SHORT ")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/intraday/bars_av.db"))
    parser.add_argument("--minutes", type=int, default=240)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--cost-bp", type=float, default=10.0)
    parser.add_argument("--max-units", type=int, default=2)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--opening", type=float, default=10_000.0)
    parser.add_argument("--monthly", type=float, default=1_000.0)
    parser.add_argument("--dip-triggers", type=float, nargs="+",
                        default=(0.03, 0.05, 0.08, 0.10, 0.15))
    parser.add_argument("--cash-apr", type=float, default=0.0,
                        help="annual yield on idle cash, e.g. 0.05")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/account_simulation.json"))
    return parser.parse_args(argv)


def load(args):
    connection = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT ticker, name FROM symbols WHERE name IS NOT NULL").fetchall()
        drop = {t for t, n in rows if any(m in n.upper() for m in LEVERED)}
    except sqlite3.OperationalError:
        drop = set()
    tickers = [r[0] for r in connection.execute(
        "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker")
        if r[0] not in drop]
    if args.limit:
        tickers = tickers[:args.limit]
    book, spy = {}, {}
    for ticker in tickers:
        raw = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='5min' AND ts>=? ORDER BY ts", (ticker, args.start)).fetchall()
        if len(raw) < 4000:
            continue
        bars = resample_regular_session([Bar(*r) for r in raw], minutes=args.minutes)
        if len(bars) >= 400:
            book[ticker] = bars
        if ticker == "SPY":
            for ts, _o, _h, _l, close, _v in raw:
                spy[ts[:10]] = close
    connection.close()
    return book, spy


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


def daily_streams(taken, closes_by_ticker):
    """Daily R, and daily sum of price/N over open units (the exposure basis)."""
    r_by_day = defaultdict(float)
    basis_by_day = defaultdict(float)
    for trade in taken:
        closes = closes_by_ticker[trade["ticker"]]
        entry_day, exit_day = trade["entry"][:10], trade["exit"][:10]
        previous = 0.0
        for day in (d for d in closes if entry_day <= d <= exit_day):
            live = [u for u in trade["units"] if u.timestamp[:10] <= day]
            if not live:
                continue
            if day < exit_day:
                open_r = sum(trade["dir"] * (closes[day] - u.price) / u.n
                             for u in live)
                r_by_day[day] += open_r - previous
                previous = open_r
            basis_by_day[day] += sum(closes[day] / u.n for u in live)
        r_by_day[exit_day] += trade["r"] - previous
    return r_by_day, basis_by_day


def solve_risk(series, target, lo=1e-6, hi=0.40):
    def walk(values, risk):
        nav = peak = 1000.0
        worst = 0.0
        for value in values:
            nav = max(0.0, nav + value * risk * nav)
            peak = max(peak, nav)
            worst = min(worst, nav / peak - 1.0)
        return worst

    def dd(risk):
        return statistics.median(abs(walk(v, risk)) for v in series)
    if dd(hi) < target:
        return hi
    for _ in range(28):
        mid = math.sqrt(lo * hi)
        if dd(mid) < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def simulate(days, ret_by_day, args, policy, basis=None, risk=None):
    """Walk the account day by day under a contribution policy."""
    equity = args.opening
    cash = 0.0
    deployments = 0
    idle_days = 0
    peak = equity
    worst = 0.0
    contributed = args.opening
    flows = [(days[0], -args.opening)]
    exposures = []
    month = days[0][:7]
    if policy == "lump":
        extra = args.monthly * (len(set(d[:7] for d in days)) - 1)
        equity += extra
        contributed += extra
        flows[0] = (days[0], -(args.opening + extra))
    for day in days:
        if policy != "lump" and day[:7] != month:
            month = day[:7]
            contributed += args.monthly
            flows.append((day, -args.monthly))
            if policy == "monthly":
                equity += args.monthly
            else:
                cash += args.monthly
        if policy == "dip" and cash > 0 and peak > 0 and equity / peak - 1 <= -args.dip_trigger:
            equity += cash
            cash = 0.0
            deployments += 1
        if cash > 0:
            idle_days += 1
            cash *= (1.0 + args.cash_apr) ** (1 / 252)
        equity = max(0.0, equity * (1.0 + ret_by_day.get(day, 0.0)))
        total = equity + cash
        peak = max(peak, total)
        worst = min(worst, total / peak - 1.0)
        if basis is not None and risk is not None and total > 0:
            exposures.append(basis.get(day, 0.0) * risk * equity / total)
    final = equity + cash
    flows.append((days[-1], final))
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25
    return {"final": final, "contributed": contributed,
            "deployments": deployments,
            "share_of_days_holding_cash": idle_days / max(len(days), 1),
            "multiple": final / contributed if contributed else 0.0,
            "irr": irr(flows), "max_drawdown": worst, "years": years,
            "exposure_median": statistics.median(exposures) if exposures else None,
            "exposure_p95": (sorted(exposures)[int(.95 * len(exposures))]
                             if exposures else None),
            "exposure_max": max(exposures) if exposures else None}


def irr(flows):
    """Money-weighted annual return; flows are (date, amount) with inflows negative."""
    base = date.fromisoformat(flows[0][0])
    times = [(date.fromisoformat(d) - base).days / 365.25 for d, _ in flows]
    amounts = [a for _, a in flows]

    def npv(rate):
        return sum(a / (1 + rate) ** t for a, t in zip(amounts, times))
    lo, hi = -0.95, 3.0
    if npv(lo) * npv(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def main(argv=None) -> int:
    args = parse_args(argv)
    book, spy = load(args)
    closes_by_ticker = {t: {b.timestamp[:10]: b.close for b in bars}
                        for t, bars in book.items()}
    config = TurtleConfig(**BASE, directions=(1,), max_units=args.max_units,
                          round_trip_cost=args.cost_bp / 10_000)
    pooled = []
    for ticker, bars in book.items():
        trades, _ = run_turtle(bars, config=config)
        pooled.extend({"ticker": ticker, "entry": t.entry_timestamp,
                       "exit": t.exit_timestamp, "r": t.net_r, "dir": t.direction,
                       "units": t.unit_entries} for t in trades)
    caps = [cap(pooled, args.max_positions, random.Random(s))
            for s in range(args.trials)]
    streams = [daily_streams(t, closes_by_ticker) for t in caps]
    calendar = sorted({d for c in closes_by_ticker.values() for d in c})
    series = [[r.get(d, 0.0) for d in calendar] for r, _ in streams]
    risk = solve_risk(series, args.target_dd)
    print(f"{len(book)} instruments, {args.minutes}-minute bars, "
          f"{args.max_units} units, {args.cost_bp:g}bp")
    print(f"{len(pooled):,} trades; risk solved to {risk*10_000:.1f}bp per R "
          f"for a {args.target_dd:.0%} median drawdown\n", flush=True)

    spy_days = sorted(spy)
    spy_ret = {}
    for i in range(1, len(spy_days)):
        spy_ret[spy_days[i]] = spy[spy_days[i]] / spy[spy_days[i - 1]] - 1

    report = {}
    print(f"  {'policy':34s} {'final':>11s} {'IRR':>6s} {'maxDD':>8s} "
          f"{'fired':>6s} {'cash%':>8s}")
    policies = [("lump", "all at once", None), ("monthly", "every month", None)]
    policies += [("dip", f"held, deploy below -{t:.0%}", t) for t in args.dip_triggers]
    for policy, label, trigger in policies:
        args.dip_trigger = trigger if trigger is not None else 1.0
        runs = []
        for (r_by_day, basis) in streams:
            ret = {d: r_by_day.get(d, 0.0) * risk for d in calendar}
            runs.append(simulate(calendar, ret, args, policy, basis, risk))
        med = {k: statistics.median([r[k] for r in runs])
               for k in ("final", "multiple", "irr", "max_drawdown")}
        exposures = [r["exposure_median"] for r in runs if r["exposure_median"]]
        p95 = [r["exposure_p95"] for r in runs if r["exposure_p95"]]
        key = f"strategy|{policy}" + (f"|{trigger}" if trigger else "")
        report[key] = {**med, "contributed": runs[0]["contributed"],
                       "deployments": statistics.median([r["deployments"] for r in runs]),
                       "cash_days": statistics.median([r["share_of_days_holding_cash"] for r in runs]),
                                        "exposure_median": statistics.median(exposures),
                                        "exposure_p95": statistics.median(p95)}
        fired = statistics.median([r["deployments"] for r in runs])
        idle = statistics.median([r["share_of_days_holding_cash"] for r in runs])
        print(f"  {('strategy, ' + label):34s} {med['final']:>11,.0f} "
              f"{med['irr']:>6.1%} {med['max_drawdown']:>8.1%} "
              f"{fired:>6.0f} {idle:>8.0%}")
        base = simulate(spy_days, spy_ret, args, policy)
        report[f"spy|{policy}" + (f"|{trigger}" if trigger else "")] = base
        print(f"  {('   SPY, ' + label):34s} {base['final']:>11,.0f} "
              f"{base['irr']:>6.1%} {base['max_drawdown']:>8.1%} "
              f"{base['deployments']:>6.0f} {base['share_of_days_holding_cash']:>8.0%}")

    med_exp = report["strategy|monthly"]["exposure_median"]
    p95_exp = report["strategy|monthly"]["exposure_p95"]
    print(f"  gross exposure while invested: median {med_exp:.0%} of account, "
          f"95th percentile {p95_exp:.0%}")
    print(f"  (six positions x {args.max_units} units; notional per unit is "
          f"equity x risk x price/N)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
