"""The leverage ratchet over 154 years, on total return, at real financing rates.

The ETF version of this test carries two biases and both run against leverage.
It uses price returns, so a levered book never collects the dividend it would
earn on the full notional -- worth roughly (L-1) times the yield, which at 3x is
four points a year.  And it charges a flat 5%, which is punitive across
2009-2021 when the actual cost of margin was near zero.

Shiller's monthly series fixes both.  Total return is reconstructed to nominal by
multiplying the real total-return index by CPI, so dividends are in.  Financing
is the effective fed funds rate where it exists and the long rate before 1954,
plus a broker spread.  And the sample runs 1871-2024 rather than 1999-2026, which
means 1929, 1937, 1973-74 and 2000 are in it rather than absent.

One bias runs the other way and it is not small.  **Monthly data cannot see an
intra-month crash.**  At leverage L ruin needs a fall of 1/L, and October 1929
and October 1987 both moved far more inside the month than the month-end print
shows.  So ruin counts here are floors, and the daily ETF test -- which checks
every session's low -- is the conservative one.  Read the two together.

Rolling 30-year windows are reported alongside the single full-history path,
because one 154-year path is one observation and the question is whether a
schedule beats holding across eras rather than in aggregate.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_leverage_ratchet import SCHEDULES, target_for  # noqa: E402

WINDOW_YEARS = 30


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equity", type=Path, default=Path("data/equity/equity.db"))
    parser.add_argument("--macro", type=Path, default=Path("data/macro/macro.db"))
    parser.add_argument("--spread", type=float, default=0.015,
                        help="broker spread over the short rate")
    parser.add_argument("--band", type=float, default=0.10)
    parser.add_argument("--taker-bp", type=float, default=5.0)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/ratchet_long_history.json"))
    known, _ = parser.parse_known_args(argv)
    return known


def load_series(args):
    """Monthly nominal total return, and the short rate to borrow at."""
    connection = sqlite3.connect(f"file:{args.equity}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT obs_date, real_total_return_price, cpi, long_rate_gs10 "
        "FROM shiller_monthly WHERE real_total_return_price IS NOT NULL "
        "AND cpi IS NOT NULL ORDER BY obs_date").fetchall()
    connection.close()

    funds = {}
    try:
        macro = sqlite3.connect(f"file:{args.macro}?mode=ro", uri=True)
        table = macro.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND "
            "name IN ('observations','obs','values')").fetchone()
        if table:
            for day, value in macro.execute(
                    f"SELECT obs_date, value FROM {table[0]} WHERE series_id='DFF' "
                    f"AND value IS NOT NULL ORDER BY obs_date"):
                funds.setdefault(str(day)[:7], []).append(float(value))
        macro.close()
    except Exception as error:                       # noqa: BLE001
        print(f"  (fed funds unavailable: {error}; using the long rate throughout)")
    funds = {k: sum(v) / len(v) for k, v in funds.items()}

    series = []
    for day, real_tr, cpi, long_rate in rows:
        nominal = float(real_tr) * float(cpi)
        short = funds.get(str(day)[:7])
        if short is None:
            short = float(long_rate) if long_rate is not None else 4.0
        series.append({"day": str(day)[:10], "level": nominal,
                       "rate": short / 100.0 + args.spread})
    return series


def run(series, schedule, args, start=0, stop=None):
    """Monthly walk. Equity 1.0; ruin when a month's move takes it through zero."""
    stop = len(series) if stop is None else stop
    fee = args.taker_bp / 10_000.0
    drift = isinstance(schedule, str)
    leverage = float(schedule[-1]) if drift else target_for(schedule, 0.0)

    equity = 1.0
    notional = leverage * equity
    peak_level = series[start]["level"]
    peak_equity, worst, ruined = 1.0, 0.0, None
    peak_lev, turnover = leverage, 0.0

    for i in range(start + 1, stop):
        prev, now = series[i - 1], series[i]
        if prev["level"] <= 0:
            continue
        move = now["level"] / prev["level"] - 1.0
        equity += notional * move
        borrowed = max(notional - equity, 0.0)
        equity -= borrowed * prev["rate"] / 12.0
        if equity <= 0:
            ruined = now["day"]
            equity = 0.0
            break

        notional *= (1.0 + move)
        peak_level = max(peak_level, now["level"])
        peak_equity = max(peak_equity, equity)
        worst = min(worst, equity / peak_equity - 1.0)
        peak_lev = max(peak_lev, notional / equity)

        if not drift:
            drawdown = 1.0 - now["level"] / peak_level
            want = target_for(schedule, drawdown) * equity
            if abs(want - notional) > args.band * max(notional, 1e-9):
                turnover += abs(want - notional)
                equity -= abs(want - notional) * fee
                notional = want

    years = (stop - start) / 12.0
    return {"cagr": (equity ** (1 / years) - 1) if equity > 0 else -1.0,
            "terminal": equity, "max_drawdown": worst,
            "ruined": ruined is not None, "ruined_on": ruined,
            "peak_leverage": peak_lev, "turnover": turnover}


def windows(series, schedule, args):
    """Every overlapping 30-year window, stepped a year at a time."""
    span = WINDOW_YEARS * 12
    out = []
    for start in range(0, len(series) - span, 12):
        out.append(run(series, schedule, args, start, start + span))
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    series = load_series(args)
    rates = [s["rate"] for s in series]
    print(f"{len(series)} months, {series[0]['day']} to {series[-1]['day']}")
    print(f"nominal total return; financing = short rate + {args.spread:.1%} "
          f"(median {statistics.median(rates):.1%}, "
          f"range {min(rates):.1%} to {max(rates):.1%})")
    print(f"monthly marks: an intra-month crash is invisible, so ruin is a FLOOR\n")
    report = {"months": len(series), "spread": args.spread, "full": {}, "windows": {}}

    print("########## the full history, one path ##########")
    print(f"  {'schedule':22s} {'CAGR':>8s} {'max DD':>9s} {'x money':>12s} "
          f"{'peak lev':>9s} {'ruin':>9s}")
    for name, schedule in SCHEDULES:
        got = run(series, schedule, args)
        report["full"][name] = got
        flag = got["ruined_on"][:7] if got["ruined"] else "-"
        money = f"{got['terminal']:,.0f}x" if got["terminal"] > 1 else f"{got['terminal']:.2f}x"
        print(f"  {name:22s} {got['cagr']:>+8.1%} {got['max_drawdown']:>9.1%} "
              f"{money:>12s} {got['peak_leverage']:>8.2f}x {flag:>9s}")

    print()
    print(f"########## every overlapping {WINDOW_YEARS}-year window ##########")
    print("  One 154-year path is one observation. This is the question that")
    print("  matters: does the schedule beat holding, across eras?")
    base = windows(series, SCHEDULES[0][1], args)
    print(f"  {'schedule':22s} {'med CAGR':>9s} {'worst':>8s} {'med maxDD':>10s} "
          f"{'worst DD':>9s} {'ruined':>9s} {'beat 1x':>9s}")
    for name, schedule in SCHEDULES:
        got = windows(series, schedule, args)
        cagrs = [w["cagr"] for w in got]
        dds = [w["max_drawdown"] for w in got]
        ruined = sum(1 for w in got if w["ruined"])
        beat = sum(1 for a, b in zip(got, base) if a["cagr"] > b["cagr"])
        report["windows"][name] = {
            "n": len(got), "median_cagr": statistics.median(cagrs),
            "worst_cagr": min(cagrs), "median_dd": statistics.median(dds),
            "worst_dd": min(dds), "ruined": ruined, "beat_1x": beat}
        print(f"  {name:22s} {statistics.median(cagrs):>+9.1%} {min(cagrs):>+8.1%} "
              f"{statistics.median(dds):>10.1%} {min(dds):>9.1%} "
              f"{ruined:>4d}/{len(got):<4d} {beat:>4d}/{len(got):<4d}")

    print()
    print("########## where the deep rung should sit (1x to 3x) ##########")
    print(f"  {'3x below':>10s} {'med CAGR':>9s} {'worst':>8s} {'med maxDD':>10s} "
          f"{'ruined':>9s} {'beat 1x':>9s}")
    report["sweep"] = {}
    for threshold in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
        got = windows(series, [(0.00, 1.0), (threshold, 3.0)], args)
        cagrs = [w["cagr"] for w in got]
        ruined = sum(1 for w in got if w["ruined"])
        beat = sum(1 for a, b in zip(got, base) if a["cagr"] > b["cagr"])
        report["sweep"][f"{threshold:.0%}"] = {
            "median_cagr": statistics.median(cagrs), "worst_cagr": min(cagrs),
            "median_dd": statistics.median(w["max_drawdown"] for w in got),
            "ruined": ruined, "beat_1x": beat, "n": len(got)}
        print(f"  {threshold:>10.0%} {statistics.median(cagrs):>+9.1%} "
              f"{min(cagrs):>+8.1%} "
              f"{statistics.median(w['max_drawdown'] for w in got):>10.1%} "
              f"{ruined:>4d}/{len(got):<4d} {beat:>4d}/{len(got):<4d}")

    print()
    print("########## how much the deep rung can carry (trigger 30%) ##########")
    print(f"  {'deep lev':>9s} {'med CAGR':>9s} {'worst':>8s} {'med maxDD':>10s} "
          f"{'worst DD':>9s} {'ruined':>9s} {'beat 1x':>9s}")
    report["deep_leverage"] = {}
    for leverage in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        got = windows(series, [(0.00, 1.0), (0.30, leverage)], args)
        cagrs = [w["cagr"] for w in got]
        dds = [w["max_drawdown"] for w in got]
        ruined = sum(1 for w in got if w["ruined"])
        beat = sum(1 for a, b in zip(got, base) if a["cagr"] > b["cagr"])
        report["deep_leverage"][f"{leverage:g}x"] = {
            "median_cagr": statistics.median(cagrs), "worst_cagr": min(cagrs),
            "median_dd": statistics.median(dds), "worst_dd": min(dds),
            "ruined": ruined, "beat_1x": beat, "n": len(got)}
        print(f"  {leverage:>8.1f}x {statistics.median(cagrs):>+9.1%} "
              f"{min(cagrs):>+8.1%} {statistics.median(dds):>10.1%} "
              f"{min(dds):>9.1%} {ruined:>4d}/{len(got):<4d} {beat:>4d}/{len(got):<4d}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print()
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
