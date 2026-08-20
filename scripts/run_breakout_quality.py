"""Do breakouts work better when the instrument leads and the regime agrees?

The claim is about the odds on a signal rather than the size of a bet: a breakout
in a laggard fighting an unfavourable rate environment should convert less often
than one in a leader with the wind behind it.  That is a statement about
conditional expectancy, and it is testable directly.

Each trade is tagged at entry with two trailing, lagged quantities:

    relative strength -- the instrument's own trailing return minus the
                         cross-sectional mean of the book that session
    regime favour    -- the instrument's rolling rate beta times the trailing
                         change in the ten-year yield

Both are known before the entry bar.  The cells are then the four combinations of
their signs, and the pre-registered statistic is the difference between the best
corner (leading, favoured) and the worst (lagging, unfavoured), predicted
positive.

The control that matters is not a null.  **Relative strength is already a
documented predictor**, so a gradient across the 2x2 proves nothing on its own --
it could be momentum alone, arranged in a square.  The test that decides it is
whether adding the regime dimension improves on relative strength by itself,
which is reported as the marginal column.  A synthetic-beta draw sits behind it
for the same reason it closed the implied-volatility work.

Run on both universes: the sectors, where the rate mechanism is cleanest, and the
forty-two single names, where the strategy actually earns anything.  The sector
book scores 0.33 flat against the equity book's 2.64, so a filter that helps only
the sectors is improving something with little to improve.
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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.turtle import TurtleConfig, run_turtle  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

SECTORS = ("XLU", "XLRE", "XLK", "XLP", "XLF", "XLE", "XLB", "XLI",
           "XLV", "XLY", "XLC")
BETA_WINDOW = 500
RATE_WINDOW = 21
STRENGTH_WINDOW = 63


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etf", type=Path,
                        default=Path("data/cross_assets/etf_30min.db"))
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--macro", type=Path, default=Path("data/macro/macro.db"))
    parser.add_argument("--cost", type=float, default=0.0002)
    parser.add_argument("--nulls", type=int, default=300)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/breakout_quality.json"))
    return parser.parse_args(argv)


def load_sectors(args):
    connection = sqlite3.connect(f"file:{args.etf}?mode=ro", uri=True)
    book = {}
    for ticker in SECTORS:
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? "
            "ORDER BY ts", (ticker,)).fetchall()
        if len(rows) >= 2000:
            book[ticker] = resample_regular_session([Bar(*r) for r in rows],
                                                    minutes=30)
    connection.close()
    return book


def load_equity(args):
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    book = {}
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker"):
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='5min' ORDER BY ts", (ticker,)).fetchall()
        five = adjust_bars([Bar(*r) for r in rows], splits.get(ticker, []))
        if len({b.timestamp[:10] for b in five}) >= 400:
            book[ticker] = resample_regular_session(five, minutes=30)
    connection.close()
    return book


def session_closes(bars):
    out = {}
    for bar in bars:
        out[bar.timestamp[:10]] = bar.close
    return out


def yield_series(path: Path):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    level = dict(connection.execute(
        "SELECT curve_date, value FROM gsw_rates WHERE mnemonic='SVENY10'"))
    connection.close()
    days = sorted(d for d in level if level[d] is not None)
    daily = {days[i]: level[days[i]] - level[days[i - 1]] for i in range(1, len(days))}
    trailing = {days[i]: level[days[i]] - level[days[i - RATE_WINDOW]]
                for i in range(RATE_WINDOW, len(days))}
    return daily, trailing


def previous_session(values, sessions):
    days = sorted(values)
    out, position, carried = {}, 0, None
    for day in sessions:
        while position < len(days) and days[position] < day:
            carried = values[days[position]]
            position += 1
        if carried is not None:
            out[day] = carried
    return out


def build_state(book, daily_yield, trailing_yield, seed=None):
    """Relative strength and regime favour per instrument per session, lagged."""
    closes = {t: session_closes(b) for t, b in book.items()}
    sessions = sorted({d for c in closes.values() for d in c})
    returns = {}
    for ticker, series in closes.items():
        days = sorted(series)
        returns[ticker] = {days[i]: series[days[i]] / series[days[i - 1]] - 1.0
                           for i in range(1, len(days)) if series[days[i - 1]] > 0}
    market = {}
    for day in sessions:
        values = [returns[t][day] for t in returns if day in returns[t]]
        if values:
            market[day] = statistics.fmean(values)

    strength, favour = {}, {}
    rng = random.Random(seed) if seed is not None else None
    for ticker in book:
        series, own = closes[ticker], returns[ticker]
        days = sorted(series)
        raw_strength = {}
        for i in range(STRENGTH_WINDOW, len(days)):
            past = series[days[i - STRENGTH_WINDOW]]
            if past > 0:
                raw_strength[days[i]] = series[days[i]] / past - 1.0
        history, raw_beta = [], {}
        for day in days:
            if len(history) >= BETA_WINDOW:
                window = history[-BETA_WINDOW:]
                xs = [x for x, _ in window]
                mx = statistics.fmean(xs)
                sxx = sum((x - mx) ** 2 for x in xs)
                if sxx > 0:
                    my = statistics.fmean([y for _, y in window])
                    raw_beta[day] = sum((x - mx) * (y - my)
                                        for x, y in window) / sxx
            if day in own and day in daily_yield and day in market:
                history.append((daily_yield[day], own[day] - market[day]))
        if rng is not None and raw_beta:
            # Synthetic beta: same persistence and spread, no information.
            ordered = sorted(raw_beta)
            values = [raw_beta[d] for d in ordered]
            mean = statistics.fmean(values)
            centred = [v - mean for v in values]
            num = sum(centred[i] * centred[i - 1] for i in range(1, len(centred)))
            den = sum(v * v for v in centred)
            phi = max(-0.99, min(0.99, num / den)) if den else 0.0
            resid = math.sqrt(max(statistics.pvariance(values) * (1 - phi ** 2), 1e-12))
            previous = 0.0
            raw_beta = {}
            for day in ordered:
                previous = phi * previous + rng.gauss(0.0, resid)
                raw_beta[day] = mean + previous
        strength[ticker] = previous_session(raw_strength, sessions)
        favour[ticker] = {}
        beta = previous_session(raw_beta, sessions)
        rate = previous_session(trailing_yield, sessions)
        for day in sessions:
            if day in beta and day in rate:
                favour[ticker][day] = beta[day] * rate[day]

    # Relative strength is cross-sectional: rank against the book that session.
    relative = {t: {} for t in book}
    for day in sessions:
        values = {t: strength[t][day] for t in book if day in strength[t]}
        if len(values) < 4:
            continue
        mean = statistics.fmean(values.values())
        for ticker, value in values.items():
            relative[ticker][day] = value - mean
    return relative, favour


def collect(book, cost):
    config = TurtleConfig(entry_window=55, exit_window=18, atr_window=20,
                          skip_after_winner=False, directions=(1,),
                          use_channel_exit=False, chandelier_atr=3.0,
                          round_trip_cost=cost)
    out = {}
    for ticker, bars in book.items():
        out[ticker] = [{"entry": t.entry_timestamp[:10], "r": t.net_r}
                       for t in run_turtle(bars, config=config)[0]]
    return out


def grid(trades, relative, favour):
    cells = defaultdict(list)
    for ticker, pooled in trades.items():
        for trade in pooled:
            day = trade["entry"]
            strength = relative.get(ticker, {}).get(day)
            wind = favour.get(ticker, {}).get(day)
            if strength is None or wind is None:
                continue
            cells[(strength > 0, wind > 0)].append(trade["r"])
    return cells


def summarise(cells):
    out = {}
    for (leading, favoured), values in cells.items():
        if len(values) < 30:
            continue
        out[f"{'leading' if leading else 'lagging'}/"
            f"{'favoured' if favoured else 'adverse'}"] = {
            "n": len(values), "mean_r": statistics.fmean(values),
            "win": sum(1 for v in values if v > 0) / len(values)}
    return out


def main(argv=None):
    args = parse_args(argv)
    daily_yield, trailing_yield = yield_series(args.macro)
    report = {}

    for label, loader in (("SECTORS", load_sectors), ("SINGLE NAMES", load_equity)):
        book = loader(args)
        relative, favour = build_state(book, daily_yield, trailing_yield)
        trades = collect(book, args.cost)
        total = sum(len(v) for v in trades.values())
        print(f"\n{'=' * 78}\n{label}: {len(book)} instruments, {total:,} breakouts")

        cells = grid(trades, relative, favour)
        table = summarise(cells)
        print(f"  {'cell':22s} {'n':>7s} {'mean R':>8s} {'win':>7s}")
        for key in ("leading/favoured", "leading/adverse",
                    "lagging/favoured", "lagging/adverse"):
            row = table.get(key)
            if row:
                print(f"  {key:22s} {row['n']:>7,d} {row['mean_r']:>+8.3f} "
                      f"{row['win']:>7.1%}")

        best = table.get("leading/favoured")
        worst = table.get("lagging/adverse")
        corner = (best["mean_r"] - worst["mean_r"]) if best and worst else None

        # Rival: relative strength on its own, ignoring the regime entirely.
        by_strength = defaultdict(list)
        for ticker, pooled in trades.items():
            for trade in pooled:
                value = relative.get(ticker, {}).get(trade["entry"])
                if value is not None:
                    by_strength[value > 0].append(trade["r"])
        strength_only = (statistics.fmean(by_strength[True])
                         - statistics.fmean(by_strength[False])
                         if by_strength[True] and by_strength[False] else None)
        # Within leaders only, does the regime add anything?
        lead_fav = table.get("leading/favoured", {}).get("mean_r")
        lead_adv = table.get("leading/adverse", {}).get("mean_r")
        marginal = (lead_fav - lead_adv) if lead_fav is not None and lead_adv is not None else None

        print(f"\n  corner spread (leading/favoured - lagging/adverse): "
              f"{corner:+.3f}R" if corner is not None else "")
        print(f"  relative strength alone (leading - lagging):         "
              f"{strength_only:+.3f}R" if strength_only is not None else "")
        print(f"  regime's MARGINAL effect, within leaders only:       "
              f"{marginal:+.3f}R" if marginal is not None else "")

        nulls = []
        for draw in range(args.nulls // 10):
            fake_rel, fake_fav = build_state(book, daily_yield, trailing_yield,
                                             seed=900 + draw)
            fake = summarise(grid(trades, fake_rel, fake_fav))
            a = fake.get("leading/favoured", {}).get("mean_r")
            b = fake.get("leading/adverse", {}).get("mean_r")
            if a is not None and b is not None:
                nulls.append(a - b)
        if nulls and marginal is not None:
            nulls.sort()
            beat = sum(1 for v in nulls if v >= marginal)
            print(f"  synthetic-beta marginal: median {statistics.median(nulls):+.3f}R,"
                  f"  p = {(beat + 1) / (len(nulls) + 1):.3f}  "
                  f"({beat}/{len(nulls)} noise betas as good)")
            report[label] = {"table": table, "corner": corner,
                             "strength_only": strength_only, "marginal": marginal,
                             "null_median": statistics.median(nulls),
                             "p": (beat + 1) / (len(nulls) + 1)}
        else:
            report[label] = {"table": table, "corner": corner,
                             "strength_only": strength_only}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
