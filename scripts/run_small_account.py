"""A thousand dollars on Binance: what actually survives contact with the account.

Percentages are size-blind and small accounts are not.  Three things bite at a
thousand dollars that never appear in a matched-drawdown table.

*The order floor.*  Every TradFi contract carries a five dollar minimum notional.
A unit whose notional falls below it cannot be sent at all, so the position is
not merely smaller than intended -- it is absent, and the trade it belonged to is
a different trade.

*Lot granularity.*  Quantity steps in hundredths of a share.  On a four hundred
dollar stock that is four dollars a step, so a twelve dollar order is three steps
and can miss its target size by a third.  The error is unbiased but it is not
small, and it is largest exactly where the notional is smallest.

*The BNB discount is an inventory decision, not a free ten per cent.*  Fees are
paid out of a BNB balance that must be held in advance, and BNB carries something
like sixty per cent annualised volatility.  Holding a hundred dollars of it
against a thousand dollar account is a ten per cent unhedged position in a
volatile asset, and the arithmetic below asks whether the discount it buys is
worth more than the variance it adds.  The answer decides the top-up schedule.

Fees are run at nine basis points -- five a side, less the ten per cent BNB
discount, taken twice -- rather than the ten used in the venue test.
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

BASE = dict(entry_window=55, exit_window=20, atr_window=20, skip_after_winner=False,
            use_channel_exit=False, chandelier_atr=3.0, directions=(1,))

MIN_NOTIONAL = 5.0
LOT_STEP = 0.01
BNB_VOL = 0.60


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--funding", type=Path,
                        default=Path("out/strategy/binance_funding.json"))
    parser.add_argument("--equity", type=float, default=1000.0)
    parser.add_argument("--bnb", type=float, default=100.0)
    parser.add_argument("--cost", type=float, default=0.0009,
                        help="round trip: 5bp a side less the 10%% BNB discount")
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--min-sessions", type=int, default=400)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/small_account.json"))
    return parser.parse_args(argv)


def load_book(args, keep):
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    book = {}
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker"):
        if ticker not in keep:
            continue
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='5min' ORDER BY ts", (ticker,)).fetchall()
        five = adjust_bars([Bar(*r) for r in rows], splits.get(ticker, []))
        if len({b.timestamp[:10] for b in five}) >= args.min_sessions:
            book[ticker] = resample_regular_session(five, minutes=args.minutes)
    connection.close()
    return book


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


def build(book, config):
    """Trades, plus the per-unit notional needed to test them against the floor."""
    pooled = []
    for ticker, bars in book.items():
        closes = {}
        for bar in bars:
            closes[bar.timestamp[:10]] = bar.close
        for trade in run_turtle(bars, config=config)[0]:
            entry_day, exit_day = (trade.entry_timestamp[:10],
                                   trade.exit_timestamp[:10])
            marks = []
            for day in closes:
                if entry_day <= day < exit_day:
                    live = [u for u in trade.unit_entries if u.timestamp[:10] <= day]
                    if live:
                        marks.append((day, sum(trade.direction * (closes[day] - u.price)
                                               / u.n for u in live)))
            pooled.append({
                "ticker": ticker, "entry": trade.entry_timestamp,
                "exit": trade.exit_timestamp, "r": trade.net_r, "marks": marks,
                "basis": trade.cost_basis_r,
                # per unit: notional fraction of equity is (price / N) * risk
                "units": [(u.price / u.n, u.price) for u in trade.unit_entries]})
    return pooled


def daily_r(taken):
    by_day = defaultdict(float)
    for trade in taken:
        previous = 0.0
        for day, open_r in trade["marks"]:
            by_day[day] += open_r - previous
            previous = open_r
        by_day[trade["exit"][:10]] += trade["r"] - previous
    return by_day


def walk(by_day, calendar, risk):
    nav, peak, worst = 1.0, 1.0, 0.0
    for day in calendar:
        nav = max(1e-12, nav * (1.0 + by_day.get(day, 0.0) * risk))
        peak = max(peak, nav)
        worst = min(worst, nav / peak - 1.0)
    years = (date.fromisoformat(calendar[-1])
             - date.fromisoformat(calendar[0])).days / 365.25
    return nav, worst, nav ** (1 / years) - 1


def solve(maps, calendar, target):
    lo, hi = 1e-7, 0.5
    for _ in range(38):
        mid = math.sqrt(lo * hi)
        if statistics.median(abs(walk(m, calendar, mid)[1]) for m in maps) < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def order_quality(taken, equity, risk):
    """What fraction of unit orders clear the floor, and how badly lots round."""
    below, total, errors = 0, 0, []
    for trade in taken:
        for notional_r, price in trade["units"]:
            total += 1
            want = notional_r * risk * equity
            if want < MIN_NOTIONAL:
                below += 1
                continue
            qty = want / price
            rounded = round(qty / LOT_STEP) * LOT_STEP
            if rounded <= 0:
                below += 1
                continue
            errors.append(abs(rounded * price - want) / want)
    return {"orders": total, "below_floor": below,
            "below_share": below / total if total else 0.0,
            "median_rounding": statistics.median(errors) if errors else 0.0,
            "p90_rounding": (sorted(errors)[int(0.9 * len(errors))]
                             if errors else 0.0)}


def main(argv=None):
    args = parse_args(argv)
    keep = set(json.loads(args.funding.read_text(encoding="utf-8")))
    book = load_book(args, keep)
    calendar = sorted({b.timestamp[:10] for bars in book.values() for b in bars})
    years = (date.fromisoformat(calendar[-1])
             - date.fromisoformat(calendar[0])).days / 365.25

    config = TurtleConfig(**{**BASE, "round_trip_cost": args.cost})
    pooled = build(book, config)
    caps = [cap(pooled, args.max_positions, random.Random(s))
            for s in range(args.trials)]
    maps = [daily_r(c) for c in caps]
    base = solve(maps, calendar, args.target_dd)
    taken = caps[0]
    turn = sum(2.0 * t["basis"] for t in taken) / years

    print(f"{len(book)} instruments, {years:.1f} years, long only, "
          f"{args.cost * 1e4:.0f}bp round trip (5bp/side less the BNB discount)")
    print(f"account ${args.equity:,.0f}; 1x risks {base:.4%} per 1N\n")

    print(f"  {'lev':>4s} {'median DD':>10s} {'worst DD':>9s} {'CAGR':>7s} "
          f"{'yr 1 P&L':>9s} {'trough':>9s} {'traded/yr':>11s} {'fees/yr':>8s}")
    report = {}
    for multiple in (1.0, 2.0, 3.0):
        risk = base * multiple
        results = [walk(m, calendar, risk) for m in maps]
        dds = sorted(abs(r[1]) for r in results)
        cagrs = sorted(r[2] for r in results)
        mid = len(dds) // 2
        worst = dds[min(int(0.95 * len(dds)), len(dds) - 1)]
        traded = turn * risk * args.equity
        fees = traded * args.cost
        report[f"{multiple:g}x"] = {
            "risk": risk, "median_dd": dds[mid], "worst_dd": worst,
            "cagr": cagrs[mid], "pnl": cagrs[mid] * args.equity,
            "traded_per_year": traded, "fees_per_year": fees,
            "orders": order_quality(taken, args.equity, risk)}
        print(f"  {multiple:>3g}x {dds[mid]:>10.1%} {worst:>9.1%} "
              f"{cagrs[mid]:>7.1%} {cagrs[mid] * args.equity:>8,.0f}$ "
              f"{-worst * args.equity:>8,.0f}$ {traded:>10,.0f}$ "
              f"{fees:>7,.0f}$")

    print(f"\norder feasibility at ${args.equity:,.0f} "
          f"(${MIN_NOTIONAL:.0f} minimum, {LOT_STEP} lot step)")
    print(f"  {'lev':>4s} {'unit orders':>12s} {'below $5':>10s} "
          f"{'median round err':>17s} {'90th pct':>9s}")
    for multiple in (1.0, 2.0, 3.0):
        q = report[f"{multiple:g}x"]["orders"]
        print(f"  {multiple:>3g}x {q['orders']:>12,d} {q['below_share']:>9.1%} "
              f"{q['median_rounding']:>16.1%} {q['p90_rounding']:>8.1%}")

    print(f"\nBNB: the discount saves 10% of the fee bill, and the balance is "
          f"consumed as fees are paid")
    print(f"  {'lev':>4s} {'fees/yr':>9s} {'saved/yr':>9s} {'BNB burn':>10s} "
          f"{'${:.0f} lasts'.format(args.bnb):>12s} {'top up every':>13s}")
    for multiple in (1.0, 2.0, 3.0):
        item = report[f"{multiple:g}x"]
        fees = item["fees_per_year"]
        # the discount is 10% off the undiscounted bill, so the saving is
        # fees / 0.9 * 0.1 relative to paying in USDT
        saved = fees / 0.9 * 0.1
        months = args.bnb / fees * 12 if fees else float("inf")
        # a one-month buffer keeps inventory low without risking a lapse
        monthly = fees / 12
        item["bnb_saved"] = saved
        item["bnb_months"] = months
        item["bnb_monthly"] = monthly
        print(f"  {multiple:>3g}x {fees:>8,.0f}$ {saved:>8,.0f}$ "
              f"{fees:>9,.0f}$ {months:>11.1f}mo {monthly:>12,.0f}$/mo")

    one_sigma = args.bnb * BNB_VOL
    saved_1x = report["1x"]["bnb_saved"]
    print(f"\n  holding ${args.bnb:.0f} of BNB is a "
          f"{args.bnb / args.equity:.0%} position in an asset at roughly "
          f"{BNB_VOL:.0%} annual volatility:")
    print(f"    one standard deviation on that stake  ${one_sigma:,.0f} a year")
    print(f"    the discount it buys at 1x            ${saved_1x:,.0f} a year")
    print(f"    ratio of risk taken to fees saved     {one_sigma / saved_1x:.1f}x")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"equity": args.equity, "cost": args.cost, "base_risk": base,
         "turnover_per_unit_risk": turn, "levels": report}, indent=1),
        encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
