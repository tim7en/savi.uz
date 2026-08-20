"""Which ETF breakouts care about which macro regime, and is any of it real?

The premise is reasonable: a duration fund and a gold fund should not respond to
a tightening cycle the same way, so one direction and one size cannot suit all
twenty-one.  The difficulty is that testing it means slicing twenty-one
instruments by three or four regimes and two directions, which is well over a
hundred cells, and a hundred cells always contain a striking one.

So the statistic reported is not any individual cell.  It is the **largest gap
found anywhere in the table**, compared against the distribution of the largest
gap found anywhere in the same table when the regime labels are circularly
shifted.  A shift preserves each regime's persistence and run lengths and
destroys only its alignment to price, and taking the maximum on both sides prices
the search itself.  A cell that clears that bar has survived the multiple
comparisons; a cell that merely looks impressive has not.

Effective sample size is reported beside every regime as the number of *episodes*
-- runs of consecutive sessions sharing a label -- because that is what the
evidence actually consists of.  Twenty-four years of daily sessions inside six
tightening cycles is six observations wearing six thousand costumes.

Regime sources, all lagged one session:

* policy, from the market-priced Fed path and the GSW zero-coupon curve, which
  are quotes and are never restated;
* positioning, from CFTC managed-money futures data mapped to the matching
  commodity fund, keyed on the **release** date rather than the as-of date --
  the report describes Tuesday and is published the following Friday, so keying
  on as-of leaks three days.
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
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import TurtleConfig, run_turtle  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

#: Commodity fund -> the COT market whose managed money positioning drives it.
COT_MAP = {
    "GLD": "GOLD - COMMODITY EXCHANGE INC.",
    "SLV": "SILVER - COMMODITY EXCHANGE INC.",
    "USO": "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE",
    "BNO": "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE",
    "UNG": "NATURAL GAS - NEW YORK MERCANTILE EXCHANGE",
    "DBA": "WHEAT-SRW - CHICAGO BOARD OF TRADE",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etf", type=Path,
                        default=Path("data/cross_assets/etf_30min.db"))
    parser.add_argument("--macro", type=Path, default=Path("data/macro/macro.db"))
    parser.add_argument("--cot", type=Path, default=Path("data/cftc/cot.db"))
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--entry-window", type=int, default=55)
    parser.add_argument("--trail", type=float, default=3.0)
    parser.add_argument("--cost", type=float, default=0.0002)
    parser.add_argument("--min-trades", type=int, default=40)
    parser.add_argument("--nulls", type=int, default=200)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/etf_macro_regimes.json"))
    return parser.parse_args(argv)


def load_etf(args):
    connection = sqlite3.connect(f"file:{args.etf}?mode=ro", uri=True)
    book = {}
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars ORDER BY ticker"):
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? "
            "ORDER BY ts", (ticker,)).fetchall()
        if len(rows) >= 2000:
            book[ticker] = resample_regular_session([Bar(*r) for r in rows],
                                                    minutes=args.minutes)
    connection.close()
    return book


def policy_regimes(path: Path):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    horizons = defaultdict(dict)
    for day, horizon, rate in connection.execute(
        "SELECT curve_date, horizon_months, forward_rate FROM fed_path "
        "WHERE horizon_months IN (3, 12)"
    ):
        horizons[day][horizon] = rate
    tenors = defaultdict(dict)
    for day, mnemonic, value in connection.execute(
        "SELECT curve_date, mnemonic, value FROM gsw_rates "
        "WHERE mnemonic IN ('SVENY02','SVENY10')"
    ):
        tenors[day][mnemonic] = value
    connection.close()

    tightening = {d: v[12] > v[3] for d, v in horizons.items()
                  if v.get(3) is not None and v.get(12) is not None}
    inverted = {d: v["SVENY10"] < v["SVENY02"] for d, v in tenors.items()
                if v.get("SVENY02") is not None and v.get("SVENY10") is not None}
    days = sorted(horizons)
    level = {d: horizons[d].get(12) for d in days}
    rising = {}
    for i, day in enumerate(days):
        past = days[max(0, i - 63)]
        if level[day] is not None and level[past] is not None:
            rising[day] = level[day] > level[past]
    return {"tightening priced": tightening, "curve inverted": inverted,
            "policy path rising": rising}


def _cot_date(raw: str) -> str | None:
    """COT stores dates in two formats; normalise, then move to the release day."""
    text = (raw or "").strip()
    stamp = None
    if len(text) >= 10 and text[4] == "-":
        try:
            stamp = datetime.strptime(text[:10], "%Y-%m-%d")
        except ValueError:
            stamp = None
    if stamp is None:
        head = text.split(" ")[0]
        for pattern in ("%m/%d/%Y", "%m/%d/%y"):
            try:
                stamp = datetime.strptime(head, pattern)
                break
            except ValueError:
                continue
    if stamp is None:
        return None
    # The report describes Tuesday and is published the following Friday.
    return (stamp + timedelta(days=3)).strftime("%Y-%m-%d")


def positioning_regimes(path: Path, window: int = 104):
    """Managed money net long, above or below its own trailing median."""
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    out = {}
    for ticker, market in COT_MAP.items():
        rows = connection.execute(
            "SELECT report_date_as_yyyy_mm_dd, m_money_positions_long_all, "
            "m_money_positions_short_all FROM cot_disagg_futures "
            "WHERE market_and_exchange_names=?", (market,)).fetchall()
        series = []
        for raw, long_side, short_side in rows:
            released = _cot_date(raw)
            if released is None or long_side is None or short_side is None:
                continue
            try:
                series.append((released, float(long_side) - float(short_side)))
            except (TypeError, ValueError):
                continue
        series.sort()
        labels, seen = {}, []
        for released, net in series:
            if len(seen) >= window:
                labels[released] = net > statistics.median(seen[-window:])
            seen.append(net)
        if len(labels) >= 200:
            out[ticker] = labels
    connection.close()
    return out


def lag_to_sessions(labels, sessions):
    days = sorted(labels)
    out, position, carried = {}, 0, None
    for day in sessions:
        while position < len(days) and days[position] < day:
            carried = labels[days[position]]
            position += 1
        if carried is not None:
            out[day] = carried
    return out


def episodes(labels):
    """Runs of consecutive sessions sharing a label: the real sample size."""
    days = sorted(labels)
    if not days:
        return 0
    runs, previous = 1, labels[days[0]]
    for day in days[1:]:
        if labels[day] != previous:
            runs += 1
            previous = labels[day]
    return runs


def gap(trades, labels):
    inside = [t["r"] for t in trades if labels.get(t["entry"][:10]) is True]
    outside = [t["r"] for t in trades if labels.get(t["entry"][:10]) is False]
    if len(inside) < 20 or len(outside) < 20:
        return None
    return statistics.fmean(inside) - statistics.fmean(outside)


def shifted(labels, offset):
    days = sorted(labels)
    values = [labels[d] for d in days]
    return {d: values[(i + offset) % len(days)] for i, d in enumerate(days)}


def main(argv=None):
    args = parse_args(argv)
    book = load_etf(args)
    sessions = sorted({b.timestamp[:10] for bars in book.values() for b in bars})
    print(f"{len(book)} ETFs, {len(sessions):,} sessions "
          f"{sessions[0]} -> {sessions[-1]}", flush=True)

    trades = {}
    for name, directions in (("long", (1,)), ("short", (-1,))):
        config = TurtleConfig(entry_window=args.entry_window,
                              exit_window=max(5, args.entry_window // 3),
                              atr_window=20, skip_after_winner=False,
                              directions=directions, use_channel_exit=False,
                              chandelier_atr=args.trail, round_trip_cost=args.cost)
        for ticker, bars in book.items():
            found, _ = run_turtle(bars, config=config)
            trades[(ticker, name)] = [
                {"entry": t.entry_timestamp, "r": t.net_r} for t in found]

    regimes = {name: lag_to_sessions(labels, sessions)
               for name, labels in policy_regimes(args.macro).items()}
    positioning = positioning_regimes(args.cot)
    print(f"\n  regime            episodes   share   applies to")
    for name, labels in regimes.items():
        share = sum(1 for v in labels.values() if v) / max(len(labels), 1)
        print(f"  {name:20s} {episodes(labels):>6d}  {share:>6.1%}   all 21")
    for ticker, labels in positioning.items():
        mapped = lag_to_sessions(labels, sessions)
        regimes[f"managed money long ({ticker})"] = mapped
        print(f"  {'COT net long ' + ticker:20s} {episodes(mapped):>6d}  "
              f"{sum(1 for v in mapped.values() if v) / max(len(mapped), 1):>6.1%}"
              f"   {ticker}")

    # Every cell, then the largest absolute gap anywhere in the table.
    cells = []
    for regime, labels in regimes.items():
        scope = [regime.split("(")[-1].rstrip(")")] if "(" in regime else list(book)
        for ticker in scope:
            for side in ("long", "short"):
                pooled = trades.get((ticker, side), [])
                if len(pooled) < args.min_trades:
                    continue
                value = gap(pooled, labels)
                if value is not None:
                    cells.append({"regime": regime, "ticker": ticker, "side": side,
                                  "gap": value, "n": len(pooled)})
    if not cells:
        print("\n  no cell had enough trades to score")
        return 1
    observed = max(abs(c["gap"]) for c in cells)
    print(f"\n  {len(cells)} scoreable cells; largest absolute gap "
          f"{observed:.3f}R", flush=True)

    print(f"\n  running {args.nulls} shifted-label tables...", flush=True)
    rng = random.Random(9)
    null_max = []
    for _ in range(args.nulls):
        offsets = {name: rng.randrange(200, max(400, len(labels)))
                   for name, labels in regimes.items()}
        fake = {name: shifted(labels, offsets[name])
                for name, labels in regimes.items()}
        best = 0.0
        for cell in cells:
            value = gap(trades[(cell["ticker"], cell["side"])], fake[cell["regime"]])
            if value is not None:
                best = max(best, abs(value))
        null_max.append(best)
    null_max.sort()
    pick = lambda f: null_max[min(int(f * len(null_max)), len(null_max) - 1)]
    beat = sum(1 for v in null_max if v >= observed)
    p_value = (beat + 1) / (len(null_max) + 1)
    print(f"    null largest gap: median {pick(.5):.3f}  p95 {pick(.95):.3f}  "
          f"max {null_max[-1]:.3f}")
    print(f"    observed:         {observed:.3f}")
    print(f"    empirical p = {p_value:.3f}  "
          f"({beat} of {len(null_max)} shuffled tables produced one as large)")

    threshold = pick(.95)
    survivors = [c for c in cells if abs(c["gap"]) > threshold]
    print(f"\n  cells clearing the search-corrected bar ({threshold:.3f}R): "
          f"{len(survivors)}")
    for cell in sorted(survivors, key=lambda c: -abs(c["gap"]))[:12]:
        print(f"    {cell['regime']:28s} {cell['ticker']:5s} {cell['side']:5s} "
              f"{cell['gap']:>+7.3f}R  ({cell['n']:,} trades)")

    print(f"\n  strongest cells regardless of the bar, for reference:")
    for cell in sorted(cells, key=lambda c: -abs(c["gap"]))[:8]:
        print(f"    {cell['regime']:28s} {cell['ticker']:5s} {cell['side']:5s} "
              f"{cell['gap']:>+7.3f}R  ({cell['n']:,} trades)")

    report = {"cells": cells, "observed_max": observed, "p_value": p_value,
              "null_p95": threshold, "survivors": survivors,
              "episodes": {name: episodes(labels) for name, labels in regimes.items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
