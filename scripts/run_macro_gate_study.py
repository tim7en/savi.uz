"""Macro regime as a sizing gate, on the configuration that actually survived.

The macro tilt was tested once before and failed: Sharpe 2.24 against 2.63
banked, better in four years of ten, p = 0.453, and beaten by its own reversal
at 2.35. That verdict was reached on the 30-minute book over 2017-2026, a window
holding a single yield-curve inversion which happens to be the historical
exception.

Two things have changed. The sample now starts in 2015 and contains three
inversion episodes rather than one. And the configuration under test is no
longer the 30-minute book, which scores 0.20 at true taker cost against 0.95 for
four-hour bars -- so the earlier test was run on a configuration since shown to
be close to the worst available.

Unfavourable means the curve is inverted or the market prices tightening ahead,
lagged one session, from market-priced rates that are never restated. Half size
against it, full with it: a tilt rather than a gate, because the gate version of
a trend filter was already rejected for removing more winners than losers.

The reversal decides, and the episode count is reported beside the session count
because three episodes is the effective sample however many sessions they span.


The programme closed "regime overlays on volatility" as redundant with N, and
that closure was right for the quantity it tested: an instrument's own realised
volatility is what N already measures, so sizing on it twice adds noise.

VIX is a different object. It is market-wide, and a six-position book's
correlation risk -- every name falling together -- is precisely what per-instrument
N cannot see. So the redundancy argument does not automatically carry over, and
the test is whether VIX adds anything *on top of* the moving-average tilt that
already survived, rather than whether it does something on its own.

Levels do not travel, ranks do -- the lesson from two vendors' gamma series
agreeing at Spearman +0.81 while their sign agreement was 54.8%. VIX is therefore
converted to a trailing percentile against its own prior year, and the label is
lagged one session so nothing is read before it is knowable.

Arms, with the reversal deciding as always: full size; constant same-mean
exposure; half size when VIX is high; half size when VIX is low. If halving into
high volatility performs no better than halving into low, the label carries
nothing.

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

FIXED = dict(entry_window=55, exit_window=20, atr_window=20,
             skip_after_winner=False, use_channel_exit=False, chandelier_atr=3.0)

LEVERED_MARKERS = ("2X", "3X", "1.5X", "BULL", "BEAR", "LEVERAGED", "INVERSE",
                   "ULTRA", "DAILY ", "SHORT ")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/intraday/bars_av.db"))
    parser.add_argument("--source-frequency", default="5min")
    parser.add_argument("--minutes", type=int, default=240)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--ma-sessions", type=int, nargs="+", default=(200,))
    parser.add_argument("--macro-db", type=Path,
                        default=Path("data/data/macro/macro.db"))
    parser.add_argument("--vix-high", type=float, default=0.70,
                        help="percentile above which VIX counts as elevated")
    parser.add_argument("--bars-per-session", type=int, default=13)
    parser.add_argument("--reduced", type=float, default=0.5)
    parser.add_argument("--cost-bp", type=float, default=10.0)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/ma_regime_sizing.json"))
    return parser.parse_args(argv)


def load(args):
    connection = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT ticker, name FROM symbols WHERE name IS NOT NULL").fetchall()
        drop = {t for t, n in rows if any(m in n.upper() for m in LEVERED_MARKERS)}
    except sqlite3.OperationalError:
        drop = set()
    tickers = [r[0] for r in connection.execute(
        "SELECT DISTINCT ticker FROM bars WHERE frequency=? ORDER BY ticker",
        (args.source_frequency,)) if r[0] not in drop]
    if args.limit:
        tickers = tickers[:args.limit]
    book = {}
    for ticker in tickers:
        raw = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency=? AND ts>=? ORDER BY ts",
            (ticker, args.source_frequency, args.start)).fetchall()
        if len(raw) < 4000:
            continue
        bars = [Bar(*r) for r in raw]
        if args.minutes != 5:
            bars = resample_regular_session(bars, minutes=args.minutes)
        if len(bars) >= 800:
            book[ticker] = bars
    connection.close()
    print(f"excluded {len(drop)} levered or inverse wrappers")
    return book


def macro_regime(macro_db: Path):
    """Sessions where the curve is inverted or tightening is priced ahead."""
    connection = sqlite3.connect(f"file:{macro_db}?mode=ro", uri=True)
    curve = {}
    for day, mnemonic, value in connection.execute(
        "SELECT curve_date, mnemonic, value FROM gsw_rates "
        "WHERE mnemonic IN ('SVENY02','SVENY10')"):
        if value is not None:
            curve.setdefault(day, {})[mnemonic] = float(value)
    forward = {}
    for day, horizon, rate in connection.execute(
        "SELECT curve_date, horizon_months, forward_rate FROM fed_path"):
        if rate is not None:
            forward.setdefault(day, {})[int(horizon)] = float(rate)
    connection.close()
    labels = {}
    for day in sorted(set(curve) | set(forward)):
        tenors, path = curve.get(day, {}), forward.get(day, {})
        inverted = (tenors.get("SVENY10") is not None
                    and tenors.get("SVENY02") is not None
                    and tenors["SVENY10"] < tenors["SVENY02"])
        tightening = (path.get(3) is not None and path.get(12) is not None
                      and path[12] > path[3])
        labels[day] = bool(inverted or tightening)
    flagged = sorted(d for d, f in labels.items() if f)
    episodes = 1 if flagged else 0
    for a, b in zip(flagged, flagged[1:]):
        if (date.fromisoformat(b) - date.fromisoformat(a)).days > 30:
            episodes += 1
    return labels, episodes


def vix_percentiles(macro_db: Path, window: int = 252) -> dict[str, float]:
    """VIX close as a trailing percentile of its own prior ``window`` sessions."""
    connection = sqlite3.connect(f"file:{macro_db}?mode=ro", uri=True)
    rows = [(d, v) for d, v in connection.execute(
        "SELECT obs_date, value FROM observations WHERE series_id='VIXCLS' "
        "AND value IS NOT NULL ORDER BY obs_date") ]
    connection.close()
    out: dict[str, float] = {}
    values = [v for _, v in rows]
    for index in range(window, len(rows)):
        past = values[index - window:index]
        # The label attaches to the *next* session, so a day is never sized
        # using its own close.
        if index + 1 < len(rows):
            out[rows[index + 1][0]] = sum(1 for p in past if p < values[index]) / len(past)
    return out


def above_ma(bars: list[Bar], window: int) -> dict[str, bool]:
    """Was the close above its trailing mean at the bar before each timestamp?"""
    closes = [b.close for b in bars]
    out: dict[str, bool] = {}
    running = 0.0
    for index, close in enumerate(closes):
        running += close
        if index >= window:
            running -= closes[index - window]
            # Compare the previous close against the mean ending there, so the
            # label is knowable before the bar it is attached to opens.
            out[bars[index].timestamp] = closes[index - 1] > running / window
    return out


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


def marked_map(taken, closes_by_ticker, weight):
    by_day = defaultdict(float)
    for trade in taken:
        scale = weight(trade)
        if scale == 0.0:
            continue
        closes = closes_by_ticker[trade["ticker"]]
        entry_day, exit_day = trade["entry"][:10], trade["exit"][:10]
        previous = 0.0
        for day in (d for d in closes if entry_day <= d < exit_day):
            live = [u for u in trade["units"] if u.timestamp[:10] <= day]
            if not live:
                continue
            open_r = sum(trade["dir"] * (closes[day] - u.price) / u.n for u in live)
            by_day[day] += (open_r - previous) * scale
            previous = open_r
        by_day[exit_day] += (trade["r"] - previous) * scale
    return by_day


def path(values, risk):
    nav = peak = 1000.0
    worst = 0.0
    for value in values:
        nav = max(0.0, nav + value * risk * nav)
        peak = max(peak, nav)
        worst = min(worst, nav / peak - 1.0)
    return nav, worst


def solve_risk(series, target, lo=1e-6, hi=0.40):
    def dd(risk):
        return statistics.median(abs(path(v, risk)[1]) for _, v in series)
    if dd(hi) < target:
        return hi
    for _ in range(32):
        mid = math.sqrt(lo * hi)
        if dd(mid) < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def sharpe(stream):
    sd = statistics.pstdev(stream)
    return statistics.fmean(stream) / sd * math.sqrt(252) if sd > 0 else float("nan")


def assess(caps, closes_by_ticker, weight, args):
    marks = [marked_map(t, closes_by_ticker, weight) for t in caps]
    series = [(sorted(m), [m[d] for d in sorted(m)]) for m in marks]
    risk = solve_risk(series, args.target_dd)
    cagrs, sharpes = [], []
    for days, values in series:
        nav, _ = path(values, risk)
        years = (date.fromisoformat(days[-1])
                 - date.fromisoformat(days[0])).days / 365.25
        cagrs.append((nav / 1000.0) ** (1 / years) - 1 if nav > 0 else -1.0)
        sharpes.append(sharpe([v * risk for v in values]))
    return {"sharpe": statistics.median(sharpes),
            "cagr": statistics.median(cagrs),
            "risk_per_r": risk}


def main(argv=None) -> int:
    args = parse_args(argv)
    book = load(args)
    closes_by_ticker = {t: {b.timestamp[:10]: b.close for b in bars}
                        for t, bars in book.items()}
    books = {}
    for label, directions in (("long only", (1,)), ("long and short", (1, -1))):
        config = TurtleConfig(**FIXED, directions=directions,
                              round_trip_cost=args.cost_bp / 10_000)
        pooled = []
        for ticker, bars in book.items():
            trades, _ = run_turtle(bars, config=config)
            pooled.extend({"ticker": ticker, "entry": t.entry_timestamp,
                           "exit": t.exit_timestamp, "r": t.net_r,
                           "dir": t.direction, "units": t.unit_entries}
                          for t in trades)
        books[label] = pooled
    caps_by_book = {k: [cap(v, args.max_positions, random.Random(s))
                        for s in range(args.trials)] for k, v in books.items()}
    pooled = books["long only"]
    caps = caps_by_book["long only"]
    print(f"{len(book)} instruments, {len(pooled):,} breakouts, "
          f"{args.cost_bp:g}bp, all arms matched to "
          f"{args.target_dd:.0%} median drawdown\n", flush=True)

    report = {}
    low = args.reduced
    for sessions in args.ma_sessions:
        window = sessions * args.bars_per_session
        labels: dict[tuple[str, str], bool] = {}
        for ticker, bars in book.items():
            for timestamp, flag in above_ma(bars, window).items():
                labels[(ticker, timestamp)] = flag
        share = statistics.fmean(
            1.0 if labels.get((t["ticker"], t["entry"]), True) else 0.0
            for t in pooled)
        average = low + (1.0 - low) * share
        regime, episodes = macro_regime(args.macro_db)
        share = statistics.fmean(
            1.0 if regime.get(t["entry"][:10], False) else 0.0 for t in pooled)
        print(f"  unfavourable regime covers {share:.0%} of entries, "
              f"{episodes} episodes")

        def macro_tilt(trade, reverse=False):
            bad = regime.get(trade["entry"][:10], False)
            if reverse:
                bad = not bad
            if trade["dir"] > 0:
                return low if bad else 1.0
            return 1.0 if bad else low

        vix = vix_percentiles(args.macro_db)

        def calm(trade):
            """Was VIX below the elevated threshold on the prior session?"""
            rank = vix.get(trade["entry"][:10])
            return True if rank is None else rank < args.vix_high

        def vix_tilt(trade, reverse=False):
            quiet = calm(trade)
            if reverse:
                quiet = not quiet
            return 1.0 if quiet else low

        def both(trade):
            return tilt(trade) * (1.0 if calm(trade) else low)

        def tilt(trade, reverse=False):
            above = labels.get((trade["ticker"], trade["entry"]), True)
            if reverse:
                above = not above
            if trade["dir"] > 0:
                return 1.0 if above else low
            return low if above else 1.0

        arms = {
            "full size always": lambda t: 1.0,
            f"MA tilt only ({low:g}x against trend)": tilt,
            "macro tilt (half when inverted/tightening)": macro_tilt,
            "macro tilt REVERSED": lambda t: macro_tilt(t, reverse=True),
            "macro and MA combined": lambda t: macro_tilt(t) * tilt(t),
        }
        print(f"=== {sessions}-session moving average ({window:,} bars); "
              f"{share:.0%} of breakouts fire above it ===")
        print(f"  {'book and arm':58s} {'Sharpe':>7s} {'CAGR':>8s} {'risk/R':>8s}")
        for label, weight in arms.items():
            for side, side_caps in caps_by_book.items():
                if label.startswith("full size") or label.startswith("constant"):
                    pass
                result = assess(side_caps, closes_by_ticker, weight, args)
                shorts = sum(1 for t in books[side] if t["dir"] < 0)
                report[f"{sessions}|{side}|{label}"] = {
                    "ma_sessions": sessions, "side": side, "arm": label,
                    "short_trades": shorts, **result}
                print(f"  {(side + ' — ' + label):58s} {result['sharpe']:>7.2f} "
                      f"{result['cagr']:>8.1%} "
                      f"{result['risk_per_r']*10_000:>6.1f}bp", flush=True)
        print(flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
