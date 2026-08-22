"""Earnings, the reaction, and whether either one predicts what follows.

The volatility-surprise study was a proxy for this.  It looked for sessions that
moved further than options priced and asked what happened next, without ever
knowing why the move happened.  Most of those sessions were earnings.  This
identifies the catalyst directly, which lets three things be separated that the
proxy had to leave tangled.

*The earnings surprise itself* -- reported EPS against the consensus estimate.
This is the classic post-earnings-announcement drift variable, and the question
the whole literature turns on is whether a beat keeps paying after the
announcement day or whether the reaction has already priced it.

*The reaction* -- what price actually did on the session that traded the news.
Surprise and reaction disagree more often than intuition suggests: a beat can
sell off.  When they disagree, one of them is the better predictor and it is an
empirical question which.

*What options had priced* -- the implied move going in, and whether the reaction
exceeded it.  This is where the option chain can contribute something the price
series cannot: the chain states in advance how large a move it expects, so a
reaction can be scored against a genuine ex-ante forecast rather than against
the stock's own trailing deviation.  The vol risk premium before the event,
implied over realised, says whether that forecast was expensive or cheap.

Timing is handled explicitly because it decides everything.  A report released
after the close is traded the following session, one released before the open is
traded that morning, and the drift is measured from the open *after* the
reaction session -- the first price a person reading the release could actually
get.  Nothing here is measured from a close that had already absorbed the news.

Two universes.  The 42-name set carries an option chain and answers the options
question.  The 142-name 13F set carries no chain and answers whether the
price-only part survives on names that were not picked with hindsight.  No
strategy is fitted: this is an event study, and every number is reported against
the same names' unconditional drift so the equity tailwind is visible rather
than absorbed.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
import sys
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

HORIZONS = (1, 5, 10, 20)
TRADING_DAYS = 252.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", choices=("intraday", "daily"),
                        default="intraday")
    parser.add_argument("--earnings", type=Path, default=Path("data/sp500_data"))
    parser.add_argument("--options", type=Path,
                        default=Path("data/options/alphavantage.db"))
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    args.out = args.out or Path(
        f"out/strategy/earnings_drift_{args.universe}.json")
    return args


def ratio(a, b):
    if b is None or (isinstance(b, float) and math.isnan(b)) or b <= 0:
        return math.nan
    return a / b


# ---------------------------------------------------------------------------


def daily_from(bars):
    groups = defaultdict(list)
    for bar in bars:
        groups[bar.timestamp[:10]].append(bar)
    return [Bar(day, rows[0].open, max(r.high for r in rows),
                min(r.low for r in rows), rows[-1].close, None)
            for day, rows in sorted(groups.items())]


def load_book(args):
    if args.universe == "intraday":
        path, frequency, floor = Path("data/intraday/bars.db"), "5min", 400
    else:
        path, frequency, floor = (Path("data/13f/alphavantage_daily.db"),
                                  "daily", 750)
    splits = load_splits(path)
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    book = {}
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency=? ORDER BY ticker",
            (frequency,)):
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency=? ORDER BY ts", (ticker, frequency)).fetchall()
        if frequency == "5min":
            bars = daily_from(resample_regular_session(
                adjust_bars([Bar(*r) for r in rows], splits.get(ticker, [])),
                minutes=5))
        else:
            bars = adjust_bars([Bar(r[0][:10], *r[1:]) for r in rows],
                               splits.get(ticker, []))
        if len(bars) >= floor:
            book[ticker] = bars
    connection.close()
    return book


def load_earnings(folder, tickers):
    """Reported date, consensus surprise, and whether it landed after the close."""
    out = {}
    for ticker in tickers:
        path = folder / f"{ticker}_earnings.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))["data"]
        except Exception:
            continue
        rows = []
        for row in payload.get("quarterlyEarnings", []):
            try:
                pct = float(row["surprisePercentage"])
            except (TypeError, ValueError, KeyError):
                continue
            date = str(row.get("reportedDate", ""))[:10]
            if len(date) != 10:
                continue
            rows.append({"date": date, "surprise_pct": pct,
                         "post": str(row.get("reportTime", "")).startswith("post")})
        if rows:
            out[ticker] = sorted(rows, key=lambda r: r["date"])
    return out


def implied_panel(path, tickers):
    if not path.exists():
        return {}
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    panel = {}
    for ticker in tickers:
        table = {d[:10]: float(iv) / math.sqrt(TRADING_DAYS)
                 for d, iv in connection.execute(
                     "SELECT observation_date, atm_iv FROM av_daily WHERE "
                     "symbol=? AND atm_iv IS NOT NULL", (ticker,))
                 if iv and float(iv) > 0}
        if table:
            panel[ticker] = table
    connection.close()
    return panel


def deviations(bars, window):
    table, returns = {}, []
    for index in range(1, len(bars)):
        a, b = bars[index - 1].close, bars[index].close
        if a > 0 and b > 0:
            returns.append(math.log(b / a))
        if len(returns) > window:
            returns.pop(0)
        if len(returns) == window:
            table[bars[index].timestamp[:10]] = statistics.pstdev(returns)
    return table


def build(book, earnings, implied, args):
    rows, baseline = [], []
    for ticker, bars in book.items():
        events = earnings.get(ticker)
        if not events:
            continue
        days = [b.timestamp[:10] for b in bars]
        closes = [b.close for b in bars]
        opens = [b.open for b in bars]
        deviation = deviations(bars, args.vol_window)
        chain = implied.get(ticker, {})

        # unconditional drift on this name, for the same forward windows
        for index in range(args.vol_window + 1, len(bars) - max(HORIZONS) - 2):
            sigma = deviation.get(days[index - 1])
            if not sigma or sigma <= 0:
                continue
            base = opens[index + 1]
            if base <= 0:
                continue
            baseline.append([(closes[index + 1 + h] - base) / (sigma * base)
                             for h in HORIZONS])

        for event in events:
            position = bisect_left(days, event["date"])
            if position >= len(days):
                continue
            # a post-close release is traded the following session
            reaction = position + 1 if event["post"] else position
            if days[position] != event["date"] and not event["post"]:
                reaction = position
            if reaction < args.vol_window + 2 or reaction + max(HORIZONS) + 2 >= len(bars):
                continue
            sigma = deviation.get(days[reaction - 1])
            prior = closes[reaction - 1]
            if not sigma or sigma <= 0 or prior <= 0:
                continue
            move = (closes[reaction] - prior) / prior
            entry = opens[reaction + 1]
            if entry <= 0:
                continue
            iv = chain.get(days[reaction - 1])
            row = {
                "ticker": ticker, "date": days[reaction],
                "surprise_pct": event["surprise_pct"],
                "reaction": move / sigma,
                "reaction_sign": 1 if move > 0 else -1,
                "surprise_sign": 1 if event["surprise_pct"] > 0 else -1,
                "iv_rv": ratio(iv, sigma) if iv else math.nan,
                "vs_implied": ratio(abs(move), iv) if iv else math.nan,
                "gap": (entry - closes[reaction]) / (sigma * prior),
            }
            for h in HORIZONS:
                row[f"fwd_{h}"] = (closes[reaction + 1 + h] - entry) / (sigma * entry)
            rows.append(row)
    base = [statistics.fmean(col) for col in zip(*baseline)] if baseline else None
    return rows, base


# ---------------------------------------------------------------------------


def table(rows, base, label, key, buckets):
    print(f"\n  {label}")
    print(f"    {'bucket':26s} {'n':>7s} " +
          " ".join(f"{'+' + str(h) + 's':>8s}" for h in HORIZONS) +
          f" {'up at +20':>10s}")
    out = {}
    for name, test in buckets:
        chunk = [r for r in rows if test(r)]
        if len(chunk) < 40:
            continue
        means = [statistics.fmean(r[f"fwd_{h}"] for r in chunk) for h in HORIZONS]
        share = sum(1 for r in chunk if r["fwd_20"] > 0) / len(chunk)
        out[name] = {"n": len(chunk),
                     "forward": dict(zip(map(str, HORIZONS), means)),
                     "excess_20": means[-1] - base[-1], "share_up": share}
        print(f"    {name:26s} {len(chunk):>7,d} " +
              " ".join(f"{m:>+8.3f}" for m in means) + f" {share:>10.1%}")
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    book = load_book(args)
    earnings = load_earnings(args.earnings, sorted(book))
    implied = (implied_panel(args.options, sorted(book))
               if args.universe == "intraday" else {})
    rows, base = build(book, earnings, implied, args)
    covered = len({r["ticker"] for r in rows})
    print(f"{args.universe}: {covered} of {len(book)} names have earnings, "
          f"{len(rows):,d} announcements")
    print(f"option chain for {len(implied)} names\n")
    print("Forward move from the open AFTER the reaction session, in the name's")
    print("own daily deviations. Positive is up, not 'continued'.")
    print(f"  {'unconditional drift':26s} {'':>7s} " +
          " ".join(f"{b:>+8.3f}" for b in base))
    report = {"names": covered, "events": len(rows),
              "baseline": dict(zip(map(str, HORIZONS), base))}

    report["surprise"] = table(rows, base, "by earnings surprise", "surprise_pct", [
        ("miss, worse than -10%", lambda r: r["surprise_pct"] <= -10),
        ("miss, -10% to 0", lambda r: -10 < r["surprise_pct"] <= 0),
        ("beat, 0 to +10%", lambda r: 0 < r["surprise_pct"] <= 10),
        ("beat, better than +10%", lambda r: r["surprise_pct"] > 10)])

    report["reaction"] = table(rows, base, "by the reaction the news got", "reaction", [
        ("fell more than 2 sigma", lambda r: r["reaction"] <= -2),
        ("fell 0 to 2 sigma", lambda r: -2 < r["reaction"] <= 0),
        ("rose 0 to 2 sigma", lambda r: 0 < r["reaction"] < 2),
        ("rose more than 2 sigma", lambda r: r["reaction"] >= 2)])

    report["agreement"] = table(rows, base, "when surprise and reaction disagree", "", [
        ("beat, price rose", lambda r: r["surprise_sign"] > 0 and r["reaction"] > 0),
        ("beat, price fell", lambda r: r["surprise_sign"] > 0 and r["reaction"] <= 0),
        ("miss, price rose", lambda r: r["surprise_sign"] < 0 and r["reaction"] > 0),
        ("miss, price fell", lambda r: r["surprise_sign"] < 0 and r["reaction"] <= 0)])

    if implied:
        priced = [r for r in rows if not math.isnan(r["vs_implied"])]
        report["vs_implied"] = table(priced, base,
            "did the reaction exceed what options priced", "vs_implied", [
                ("inside the implied move", lambda r: r["vs_implied"] < 1),
                ("1 to 2x implied", lambda r: 1 <= r["vs_implied"] < 2),
                ("2 to 3x implied", lambda r: 2 <= r["vs_implied"] < 3),
                ("beyond 3x implied", lambda r: r["vs_implied"] >= 3)])
        ranked = sorted(r["iv_rv"] for r in priced if not math.isnan(r["iv_rv"]))
        q1, q3 = ranked[len(ranked)//4], ranked[3*len(ranked)//4]
        report["iv_rv"] = table(priced, base,
            "by how expensive the chain was going in (implied over realised)",
            "iv_rv", [
                (f"cheapest quarter (<{q1:.2f})", lambda r: r["iv_rv"] < q1),
                ("middle half", lambda r: q1 <= r["iv_rv"] <= q3),
                (f"dearest quarter (>{q3:.2f})", lambda r: r["iv_rv"] > q3)])
        report["priced_direction"] = table(priced, base,
            "beyond 2x implied, split by which way it went", "", [
                ("2x+ implied, price rose",
                 lambda r: r["vs_implied"] >= 2 and r["reaction"] > 0),
                ("2x+ implied, price fell",
                 lambda r: r["vs_implied"] >= 2 and r["reaction"] <= 0)])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
