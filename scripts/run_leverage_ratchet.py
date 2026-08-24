"""Hold always; add leverage only as the drawdown deepens.

Different in kind from every entry rule tested before.  There is no entry: the
book is invested the whole time and the only decision is how much leverage to
carry, as a function of how far the index sits below its own all-time high.  That
is a martingale -- adding to a loser -- and martingales have a specific failure
mode, so the measurement has to be about survival before it is about return.

Three things this separates that are usually conflated.

*A fixed-notional hold levers itself.*  Buy one unit and never trade: as price
falls, equity shrinks faster than notional and leverage rises mechanically.  Half
of what "ratchet up into the drawdown" means is simply "do not rebalance", and
that arm is free.  It is included so the discretionary part of the ratchet is
measured against it rather than credited with its effect.

*Rebalancing to a constant leverage costs money.*  Daily-rebalanced constant
leverage sells low and buys high by construction, which is the decay every
leveraged ETF carries.  Turnover is charged at ``--taker-bp`` on the notional
traded, so a schedule that trades more is penalised for it.

*The ratchet must beat its own inverse.*  A schedule that levers up into
drawdowns and one that levers down into them carry the same average exposure over
the path; if the first is not clearly better, the result is exposure and not
timing.  That arm runs on every path.

Ruin is checked against the daily low, so an intraday spike that would trigger a
margin call ends the path -- at leverage L a fall of 1/L takes the account, and
it does not matter whether the broker or the trader does the selling.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

DECAYING = ("USO", "UNG", "GSG", "DBC", "DBA", "BNO", "VXX")
NON_EQUITY = ("FXB", "FXE", "FXY", "UUP", "HYG", "IEF", "LQD", "SHY",
              "TIP", "TLT", "GLD", "SLV", "PPLT")

# (name, [(drawdown_at_or_beyond, leverage), ...] shallowest first)
SCHEDULES = [
    ("flat 1x",            [(0.00, 1.0)]),
    ("flat 2x",            [(0.00, 2.0)]),
    ("flat 3x",            [(0.00, 3.0)]),
    ("no rebalance 1x",    "drift1"),
    ("no rebalance 2x",    "drift2"),
    ("gentle 1>1.5>2",     [(0.00, 1.0), (0.15, 1.5), (0.30, 2.0)]),
    ("classic 1>2>3",      [(0.00, 1.0), (0.10, 2.0), (0.25, 3.0)]),
    ("late 1>2>3",         [(0.00, 1.0), (0.20, 2.0), (0.35, 3.0)]),
    ("deep only 1>3",      [(0.00, 1.0), (0.30, 3.0)]),
    ("hot 2>3>4",          [(0.00, 2.0), (0.20, 3.0), (0.40, 4.0)]),
    ("INVERSE 3>2>1",      [(0.00, 3.0), (0.10, 2.0), (0.25, 1.0)]),
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path,
                        default=Path("data/cross_assets/etf_daily.db"))
    parser.add_argument("--financing", type=float, default=0.05)
    parser.add_argument("--taker-bp", type=float, default=5.0)
    parser.add_argument("--band", type=float, default=0.10,
                        help="no-trade band around target leverage")
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/leverage_ratchet.json"))
    known, _ = parser.parse_known_args(argv)
    return known


def load(args):
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    book = {}
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency='daily' "
            "ORDER BY ticker"):
        if ticker in DECAYING or ticker in NON_EQUITY:
            continue
        rows = connection.execute(
            "SELECT ts,open,high,low,close FROM bars WHERE ticker=? AND "
            "frequency='daily' ORDER BY ts", (ticker,)).fetchall()
        bars = adjust_bars([Bar(r[0][:10], r[1], r[2], r[3], r[4], None)
                            for r in rows], splits.get(ticker, []))
        if len(bars) >= 2000:
            book[ticker] = bars
    connection.close()
    return book


def target_for(schedule, drawdown):
    """Deepest rung whose threshold the drawdown has reached."""
    leverage = schedule[0][1]
    for threshold, value in schedule:
        if drawdown >= threshold:
            leverage = value
    return leverage


def run(bars, schedule, args):
    """Walk one path. Equity starts at 1.0; ruin ends it."""
    fee = args.taker_bp / 10_000.0
    drift = isinstance(schedule, str)
    if drift:
        leverage = float(schedule[-1])          # "drift1" / "drift2"

    equity, peak_price, peak_equity = 1.0, bars[0].close, 1.0
    notional = (leverage if drift else target_for(schedule, 0.0)) * equity
    worst_equity, worst_lev, turnover = 0.0, 0.0, 0.0
    ruined_on = None

    for i in range(1, len(bars)):
        prev, bar = bars[i - 1].close, bars[i]
        if prev <= 0:
            continue
        shares = notional / prev            # notional held into the day

        # margin call against the intraday low, before any close-based marking
        if equity + shares * (bar.low - prev) <= 0:
            ruined_on = bar.timestamp
            equity = 0.0
            break

        equity += shares * (bar.close - prev)
        borrowed = max(notional - equity, 0.0)
        equity -= borrowed * args.financing / 252.0
        if equity <= 0:
            ruined_on = bar.timestamp
            equity = 0.0
            break

        notional = shares * bar.close
        peak_price = max(peak_price, bar.close)
        peak_equity = max(peak_equity, equity)
        worst_equity = min(worst_equity, equity / peak_equity - 1.0)
        worst_lev = max(worst_lev, notional / equity if equity > 0 else 0.0)

        if not drift:
            drawdown = 1.0 - bar.close / peak_price
            want = target_for(schedule, drawdown) * equity
            # a band keeps the book from trading every day; constant-leverage
            # rebalancing sells low and buys high by construction and the cost
            # of that is exactly what sinks a leveraged ETF
            if abs(want - notional) > args.band * max(notional, 1e-9):
                turnover += abs(want - notional)
                equity -= abs(want - notional) * fee
                notional = want

    years = (date.fromisoformat(bars[-1].timestamp)
             - date.fromisoformat(bars[0].timestamp)).days / 365.25
    return {"cagr": (equity ** (1 / years) - 1) if equity > 0 else -1.0,
            "terminal": equity, "max_drawdown": worst_equity,
            "ruined": ruined_on is not None, "ruined_on": ruined_on,
            "peak_leverage": worst_lev, "turnover": turnover, "years": years}


def main(argv=None) -> int:
    args = parse_args(argv)
    book = load(args)
    print(f"{len(book)} equity ETFs; financing {args.financing:.0%}, "
          f"turnover {args.taker_bp:g}bp, no-trade band {args.band:.0%}")
    print("Leverage is a function of the drawdown from the running all-time high.")
    print("Ruin is checked against the daily low: at L a fall of 1/L ends it.\n")
    report = {"financing": args.financing, "band": args.band, "paths": {}}

    market = book.get("DIA")
    if market:
        print("########## one path: DIA, the whole history ##########")
        print(f"  {market[0].timestamp} to {market[-1].timestamp}")
        print(f"  {'schedule':22s} {'CAGR':>8s} {'max DD':>9s} {'x money':>9s} "
              f"{'peak lev':>9s} {'turnover':>9s} {'ruin':>6s}")
        for name, schedule in SCHEDULES:
            got = run(market, schedule, args)
            report["paths"].setdefault("DIA", {})[name] = got
            flag = got["ruined_on"][:7] if got["ruined"] else "-"
            print(f"  {name:22s} {got['cagr']:>+8.1%} {got['max_drawdown']:>9.1%} "
                  f"{got['terminal']:>8.2f}x {got['peak_leverage']:>8.2f}x "
                  f"{got['turnover']:>9.1f} {flag:>6s}")

    print()
    print("########## every ETF as its own path ##########")
    print("  Eighteen correlated paths are not eighteen independent trials, but")
    print("  they do say whether a schedule survives anything other than DIA.")
    print(f"  {'schedule':22s} {'med CAGR':>9s} {'worst CAGR':>11s} "
          f"{'med maxDD':>10s} {'worst DD':>9s} {'ruined':>8s} {'beat 1x':>8s}")
    base = {t: run(bars, SCHEDULES[0][1], args) for t, bars in book.items()}
    for name, schedule in SCHEDULES:
        results = {t: run(bars, schedule, args) for t, bars in book.items()}
        cagrs = [r["cagr"] for r in results.values()]
        dds = [r["max_drawdown"] for r in results.values()]
        ruined = sum(1 for r in results.values() if r["ruined"])
        beat = sum(1 for t, r in results.items() if r["cagr"] > base[t]["cagr"])
        report.setdefault("universe", {})[name] = {
            "median_cagr": statistics.median(cagrs), "worst_cagr": min(cagrs),
            "median_dd": statistics.median(dds), "worst_dd": min(dds),
            "ruined": ruined, "n": len(results), "beat_1x": beat}
        print(f"  {name:22s} {statistics.median(cagrs):>+9.1%} {min(cagrs):>+11.1%} "
              f"{statistics.median(dds):>10.1%} {min(dds):>9.1%} "
              f"{ruined:>4d}/{len(results):<3d} {beat:>4d}/{len(results):<3d}")

    print()
    print("########## where the deep rung should sit ##########")
    print("  A 1x-to-3x ratchet, sweeping the drawdown that triggers 3x.")
    print(f"  {'3x below':>10s} {'med CAGR':>9s} {'worst CAGR':>11s} "
          f"{'med maxDD':>10s} {'ruined':>8s} {'beat 1x':>8s} {'DIA CAGR':>9s}")
    report["sweep"] = {}
    for threshold in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
        schedule = [(0.00, 1.0), (threshold, 3.0)]
        results = {t: run(bars, schedule, args) for t, bars in book.items()}
        cagrs = [r["cagr"] for r in results.values()]
        ruined = sum(1 for r in results.values() if r["ruined"])
        beat = sum(1 for t, r in results.items() if r["cagr"] > base[t]["cagr"])
        dia = run(market, schedule, args)["cagr"] if market else float("nan")
        report["sweep"][f"{threshold:.0%}"] = {
            "median_cagr": statistics.median(cagrs), "worst_cagr": min(cagrs),
            "median_dd": statistics.median(r["max_drawdown"] for r in results.values()),
            "ruined": ruined, "beat_1x": beat, "dia_cagr": dia}
        print(f"  {threshold:>10.0%} {statistics.median(cagrs):>+9.1%} "
              f"{min(cagrs):>+11.1%} "
              f"{statistics.median(r['max_drawdown'] for r in results.values()):>10.1%} "
              f"{ruined:>4d}/{len(results):<3d} {beat:>4d}/{len(results):<3d} "
              f"{dia:>+9.1%}")

    print()
    print("########## and how much leverage the deep rung can carry ##########")
    print("  Trigger fixed at 30% down; sweeping the leverage taken there.")
    print(f"  {'deep lev':>9s} {'med CAGR':>9s} {'worst CAGR':>11s} "
          f"{'med maxDD':>10s} {'worst DD':>9s} {'ruined':>8s} {'DIA CAGR':>9s}")
    report["deep_leverage"] = {}
    for leverage in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        schedule = [(0.00, 1.0), (0.30, leverage)]
        results = {t: run(bars, schedule, args) for t, bars in book.items()}
        cagrs = [r["cagr"] for r in results.values()]
        dds = [r["max_drawdown"] for r in results.values()]
        ruined = sum(1 for r in results.values() if r["ruined"])
        dia = run(market, schedule, args)["cagr"] if market else float("nan")
        report["deep_leverage"][f"{leverage:g}x"] = {
            "median_cagr": statistics.median(cagrs), "worst_cagr": min(cagrs),
            "median_dd": statistics.median(dds), "worst_dd": min(dds),
            "ruined": ruined, "dia_cagr": dia}
        print(f"  {leverage:>8.1f}x {statistics.median(cagrs):>+9.1%} "
              f"{min(cagrs):>+11.1%} {statistics.median(dds):>10.1%} "
              f"{min(dds):>9.1%} {ruined:>4d}/{len(results):<3d} {dia:>+9.1%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print()
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
