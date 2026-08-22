"""Does a move the options did not price continue, or does it come back?

The vol-stretch study assumed reversion and fitted a fade to it.  That was an
assumption, not a finding, and it is the assumption this replaces.  The
proposition here is the opposite: a session that closes outside what the chain
priced is *information*.  Options had a view on how far the name could travel;
price went further; the move is therefore not noise around a known
distribution, it is news the distribution did not contain.

If that is right, such moves continue and the trade is a follow.  If the
reversion story is right, they come back and the trade is a fade.  Both cannot
be true, so the first half of this is an event study with no strategy attached:
after a move of *z* implied moves, what does price actually do over the next
one, three, five and ten sessions?  A strategy fitted before that question is
answered is a strategy fitted to a coin.

The distinction the study turns on is implied against realised.  A move can be
large two different ways.  Large against its own recent deviation but *inside*
what the chain priced is a move the market expected -- an earnings date, a known
catalyst -- and there is no surprise in it.  Large against the chain is the case
where the pricing was wrong.  Separating those is the whole point, so every
result is cut by the vol risk premium, implied over realised, measured before
the move.

Both directions run.  The programme's finding that the short side is dead was
measured on breakout shorts, where the entry chases strength; fading an outsized
*up* move and following an outsized *down* move are different trades and inherit
nothing from that verdict.  They are reported separately so the short side can
fail on its own evidence rather than by inheritance.

Execution is priced per leg because the two families cannot be executed the same
way.  A follow enters at the next open and is a taker.  A fade rests a limit and
is a maker.  The bracket study measured that difference at roughly one full
point of Sharpe, so charging both a flat rate would hand the follow arm a
subsidy it does not get in the market.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import zlib
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_vol_stretch_zones as data  # noqa: E402

HORIZONS = (1, 3, 5, 10)
Z_BUCKETS = ((1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 99.0))


def ticker_seed(ticker):
    """A stable per-name seed.

    ``hash()`` on a str is salted per interpreter process, so a null drawn with
    it is not reproducible between runs -- and a control whose band moves when
    you run it again cannot settle anything.
    """
    return zlib.crc32(ticker.encode("utf-8")) % 10_000


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--options", type=Path,
                        default=Path("data/options/alphavantage.db"))
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--zone-minutes", type=int, default=30,
                        help="unused here; the shared loader builds the series")
    parser.add_argument("--stretch", type=float, default=2.0,
                        help="implied moves that define a surprise")
    parser.add_argument("--hold-sessions", type=int, default=5)
    parser.add_argument("--stop-mult", type=float, default=2.0)
    parser.add_argument("--target-r", type=float, default=2.0)
    parser.add_argument("--wait-sessions", type=int, default=3)
    parser.add_argument("--maker-bp", type=float, default=2.5)
    parser.add_argument("--taker-bp", type=float, default=5.0)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/vol_surprise.json"))
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# the event study


def events(book, implied, realized, args):
    """Every session whose close moved at least one implied move, either way.

    ``z_implied`` normalises by what the chain priced, ``z_realized`` by the
    name's own recent deviation.  Their ratio is the vol risk premium, and a
    move can be large on one scale and ordinary on the other.
    """
    out = []
    for ticker, (_, _, daily) in book.items():
        chain, history = implied.get(ticker), realized.get(ticker)
        if not chain or not history:
            continue
        closes = [b.close for b in daily]
        opens = [b.open for b in daily]
        for index in range(1, len(daily) - max(HORIZONS) - 1):
            previous = daily[index - 1].timestamp[:10]
            iv, rv = chain.get(previous), history.get(previous)
            if not iv or not rv or iv <= 0 or rv <= 0:
                continue
            base = closes[index - 1]
            if base <= 0:
                continue
            change = (closes[index] - base) / base
            if change == 0:
                continue
            direction = 1 if change > 0 else -1
            row = {
                "ticker": ticker, "index": index,
                "day": daily[index].timestamp[:10],
                "direction": direction,
                "z_implied": abs(change) / iv,
                "z_realized": abs(change) / rv,
                "iv_premium": iv / rv,
                "implied": iv * base,
                "close": closes[index],
            }
            # Signed so that positive always means the move continued.
            for horizon in HORIZONS:
                ahead = index + horizon
                row[f"fwd_{horizon}"] = direction * (
                    closes[ahead] - closes[index]) / (iv * base)
                row[f"open_{horizon}"] = direction * (
                    closes[ahead] - opens[index + 1]) / (iv * base)
            out.append(row)
    return out


def raw_means(chunk):
    """Forward move in its own direction, not the trigger's.

    Equities drift up, so a table signed by the trigger direction flatters every
    up move and penalises every down one by the same amount.  Reading the raw
    number against the unconditional drift is the only way to see what the
    trigger contributed rather than what the asset class did.
    """
    return [statistics.fmean(r["direction"] * r[f"open_{h}"] for r in chunk)
            for h in HORIZONS]


def bucket_table(rows, label, baseline):
    print(f"\n  {label}")
    print(f"    {'z implied':12s} {'n':>7s} " +
          " ".join(f"{'+' + str(h) + 's':>8s}" for h in HORIZONS) +
          f" {'share up':>9s}  vs drift at +5s")
    for low, high in Z_BUCKETS:
        chunk = [r for r in rows if low <= r["z_implied"] < high]
        if len(chunk) < 40:
            continue
        raw = raw_means(chunk)
        share = sum(1 for r in chunk if r["direction"] * r["open_5"] > 0) / len(chunk)
        span = f"{low:g}-{high:g}" if high < 90 else f"{low:g}+"
        print(f"    {span:12s} {len(chunk):>7,d} " +
              " ".join(f"{m:>+8.3f}" for m in raw) +
              f" {share:>9.1%}  {raw[2] - baseline[2]:>+8.3f}")


def premium_table(rows, label, stretch):
    """The same moves, cut by whether the chain was already pricing wide."""
    chunk = [r for r in rows if r["z_implied"] >= stretch]
    if len(chunk) < 100:
        print(f"\n  {label}: only {len(chunk):,d} surprises, too few to cut")
        return {}
    chunk.sort(key=lambda r: r["iv_premium"])
    quarter = len(chunk) // 4
    print(f"\n  {label}, {len(chunk):,d} moves beyond {stretch:g} implied, "
          f"cut by implied/realised before the move")
    print(f"    {'IV/RV':12s} {'n':>7s} " +
          " ".join(f"{'+' + str(h) + 's':>8s}" for h in HORIZONS) +
          f" {'share up':>9s}")
    out = {}
    for position, name in enumerate(("lowest", "2nd", "3rd", "highest")):
        part = chunk[position * quarter:(position + 1) * quarter]
        if len(part) < 30:
            continue
        means = [statistics.fmean(r[f"open_{h}"] for r in part) for h in HORIZONS]
        share = sum(1 for r in part if r["open_5"] > 0) / len(part)
        band = f"{part[0]['iv_premium']:.2f}-{part[-1]['iv_premium']:.2f}"
        print(f"    {name:5s} {band:>10s} {len(part):>5,d} " +
              " ".join(f"{m:>+8.3f}" for m in means) + f" {share:>9.1%}")
        out[name] = {"n": len(part), "band": band,
                     "forward": dict(zip(map(str, HORIZONS), means)),
                     "share_up": share}
    return out


# ---------------------------------------------------------------------------
# the tradeable arms


def session_index(bars):
    spans, first = [], 0
    while first < len(bars):
        day = bars[first].timestamp[:10]
        last = first
        while last + 1 < len(bars) and bars[last + 1].timestamp[:10] == day:
            last += 1
        spans.append((day, first, last))
        first = last + 1
    return spans


def follow_trade(five, spans, day_at, day, side, implied, args):
    """Enter at the next open and take it, which is a taker on the way in."""
    position = day_at.get(day)
    if position is None or position + 1 >= len(spans):
        return None
    start = spans[position + 1][1]
    fill = five[start].open
    risk = args.stop_mult * implied
    if fill <= 0 or risk <= 0:
        return None
    stop = fill - side * risk
    target = fill + side * args.target_r * risk
    deadline = spans[min(position + args.hold_sessions, len(spans) - 1)][2]
    for index in range(start, min(deadline, len(five) - 1) + 1):
        bar = five[index]
        if (bar.low <= stop) if side > 0 else (bar.high >= stop):
            return fill, stop, "stop", bar.timestamp, risk
        if (bar.high >= target) if side > 0 else (bar.low <= target):
            return fill, target, "target", bar.timestamp, risk
    end = min(deadline, len(five) - 1)
    return fill, five[end].close, "time", five[end].timestamp, risk


def fade_trade(five, spans, day_at, day, side, close, implied, args):
    """Rest a limit against the move, which is a maker on the way in."""
    position = day_at.get(day)
    if position is None:
        return None
    limit = close - side * 2.0 * implied
    watch = spans[min(position + args.wait_sessions, len(spans) - 1)][2]
    fill_index = None
    for index in range(spans[position][2] + 1, watch + 1):
        bar = five[index]
        if (bar.low <= limit) if side > 0 else (bar.high >= limit):
            fill_index = index
            break
    if fill_index is None:
        return None
    bar = five[fill_index]
    fill = min(limit, bar.open) if side > 0 else max(limit, bar.open)
    risk = args.stop_mult * implied
    if fill <= 0 or risk <= 0:
        return None
    stop = fill - side * risk
    target = fill + side * args.target_r * risk
    hold_at = day_at.get(five[fill_index].timestamp[:10], position)
    deadline = spans[min(hold_at + args.hold_sessions - 1, len(spans) - 1)][2]
    for index in range(fill_index + 1, min(deadline, len(five) - 1) + 1):
        bar = five[index]
        if (bar.low <= stop) if side > 0 else (bar.high >= stop):
            return fill, stop, "stop", bar.timestamp, risk
        if (bar.high >= target) if side > 0 else (bar.low <= target):
            return fill, target, "target", bar.timestamp, risk
    end = min(deadline, len(five) - 1)
    return fill, five[end].close, "time", five[end].timestamp, risk


def arm(book, rows, args, action, move_direction, null_seed=None):
    """One of the four combinations, priced by the order type it really uses.

    ``null_seed`` replaces the trigger with an ordinary session -- same name,
    same count, same side, same exits.  Equities drift up, so any long arm earns
    something for free and the dip study measured random entries capturing
    63-75% of a real signal's result.  A long arm that does not clear its own
    drift null has not been shown to contain a signal at all.
    """
    by_ticker = defaultdict(list)
    if null_seed is None:
        for row in rows:
            if (row["direction"] == move_direction
                    and row["z_implied"] >= args.stretch):
                by_ticker[row["ticker"]].append(row)
    else:
        wanted = defaultdict(int)
        for row in rows:
            if (row["direction"] == move_direction
                    and row["z_implied"] >= args.stretch):
                wanted[row["ticker"]] += 1
        quiet = defaultdict(list)
        for row in rows:
            if row["z_implied"] < 1.0:
                quiet[row["ticker"]].append(row)
        for ticker, count in wanted.items():
            pool = quiet.get(ticker, [])
            if not pool:
                continue
            rng = random.Random(null_seed + ticker_seed(ticker))
            picked = rng.sample(pool, min(count, len(pool)))
            by_ticker[ticker] = sorted(picked, key=lambda r: r["day"])
    side = move_direction if action == "follow" else -move_direction
    trades = []
    for ticker, events_here in by_ticker.items():
        five = book[ticker][0]
        spans = session_index(five)
        day_at = {day: i for i, (day, _, _) in enumerate(spans)}
        open_until = ""
        for row in events_here:
            if row["day"] < open_until:
                continue
            outcome = (follow_trade(five, spans, day_at, row["day"], side,
                                    row["implied"], args)
                       if action == "follow" else
                       fade_trade(five, spans, day_at, row["day"], side,
                                  row["close"], row["implied"], args))
            if outcome is None:
                continue
            fill, price, reason, stamp, risk = outcome
            open_until = stamp[:10]
            entry_leg = args.taker_bp if action == "follow" else args.maker_bp
            exit_leg = args.maker_bp if reason == "target" else args.taker_bp
            trades.append({
                "ticker": ticker, "entry": row["day"], "exit": stamp[:10],
                "r": side * (price - fill) / risk
                     - (entry_leg + exit_leg) / 10_000 * fill / risk,
                "reason": reason, "stop_pct": risk / fill,
                "iv_premium": row["iv_premium"],
            })
    trades.sort(key=lambda t: t["entry"])
    return trades


def describe(trades):
    outcomes = [t["r"] for t in trades]
    reasons = defaultdict(int)
    for trade in trades:
        reasons[trade["reason"]] += 1
    total = len(trades)
    return {"trades": total, "mean_r": statistics.fmean(outcomes),
            "target_rate": reasons["target"] / total,
            "stop_rate": reasons["stop"] / total,
            "time_rate": reasons["time"] / total,
            "median_stop_pct": statistics.median(t["stop_pct"] for t in trades)}


def main(argv=None) -> int:
    args = parse_args(argv)
    book = data.load(args)
    implied = data.implied_moves(args.options, sorted(book), args.start)
    realized = data.realized_moves(book)
    print(f"{len(book)} instruments, implied for {len(implied)}, "
          f"realised for {len(realized)}")

    rows = events(book, implied, realized, args)
    print(f"{len(rows):,d} sessions with both vol measures\n")
    report = {"event_study": {}, "premium": {}, "arms": {}}

    print("########## event study: forward move after a surprise ##########")
    print("  Raw forward move, NOT signed by the trigger, so up and down are on")
    print("  the same axis and the drift is visible rather than absorbed.")
    print("  Measured from the NEXT open, so the untradeable gap is excluded.")
    print("  Units are implied daily moves.\n")
    baseline = raw_means(rows)
    quiet = [r for r in rows if r["z_implied"] < 1.0]
    print(f"  {'baseline':12s} {'n':>7s} " +
          " ".join(f"{'+' + str(h) + 's':>8s}" for h in HORIZONS))
    print(f"  {'every session':12s} {len(rows):>7,d} " +
          " ".join(f"{m:>+8.3f}" for m in baseline))
    if len(quiet) >= 100:
        print(f"  {'z below 1':12s} {len(quiet):>7,d} " +
              " ".join(f"{m:>+8.3f}" for m in raw_means(quiet)))
    report["baseline"] = dict(zip(map(str, HORIZONS), baseline))
    for label, subset in (("up moves", [r for r in rows if r["direction"] > 0]),
                          ("down moves", [r for r in rows if r["direction"] < 0])):
        bucket_table(subset, label, baseline)
        report["event_study"][label] = {
            f"{low:g}-{high:g}": {
                "n": len([r for r in subset if low <= r["z_implied"] < high]),
                "fwd_5": statistics.fmean(
                    [r["open_5"] for r in subset if low <= r["z_implied"] < high])
                if len([r for r in subset if low <= r["z_implied"] < high]) >= 40
                else None}
            for low, high in Z_BUCKETS}

    print("\n########## the same, cut by the vol risk premium ##########")
    for label, subset in (("up moves", [r for r in rows if r["direction"] > 0]),
                          ("down moves", [r for r in rows if r["direction"] < 0])):
        report["premium"][label] = premium_table(subset, label, args.stretch)

    print(f"\n########## the four tradeable arms at {args.stretch:g}x implied "
          f"##########")
    print(f"  Stop {args.stop_mult:g} implied moves, target {args.target_r:g}R, "
          f"held {args.hold_sessions} sessions.")
    print(f"  {'arm':28s} {'trades':>7s} {'target':>7s} {'stop':>7s} {'time':>6s} "
          f"{'meanR':>8s} {'Sharpe':>7s} {'[5-95%]':>15s}")
    print("  " + "-" * 92)
    for action in ("follow", "fade"):
        for move_direction, name in ((1, "up move"), (-1, "down move")):
            trades = arm(book, rows, args, action, move_direction)
            side = "long" if (move_direction if action == "follow"
                              else -move_direction) > 0 else "short"
            label = f"{action} {name} ({side})"
            if len(trades) < 100:
                print(f"  {label:28s} {len(trades):>7,d}   too few")
                continue
            stats = describe(trades)
            result = data.assess(trades, args)
            report["arms"][label] = {**stats, **(result or {})}
            band = ("[%.2f-%.2f]" % (result["sharpe_p05"], result["sharpe_p95"])
                    if result else "")
            print(f"  {label:28s} {stats['trades']:>7,d} "
                  f"{stats['target_rate']:>7.1%} {stats['stop_rate']:>7.1%} "
                  f"{stats['time_rate']:>6.1%} {stats['mean_r']:>+8.3f} "
                  f"{(result['sharpe'] if result else float('nan')):>7.2f} "
                  f"{band:>15s}", flush=True)

            spread = []
            for draw in range(5):
                drawn = arm(book, rows, args, action, move_direction,
                            null_seed=5000 + 137 * draw)
                if len(drawn) < 100:
                    continue
                outcome = data.assess(drawn, args)
                if outcome:
                    spread.append((outcome["sharpe"], describe(drawn)["mean_r"]))
            if spread:
                spread.sort()
                middle = spread[len(spread) // 2]
                report["arms"][label]["null_sharpe"] = middle[0]
                report["arms"][label]["null_mean_r"] = middle[1]
                print(f"  {'  ordinary sessions (null)':28s} {'':>7s} {'':>7s} "
                      f"{'':>7s} {'':>6s} {middle[1]:>+8.3f} {middle[0]:>7.2f} "
                      f"{('[%.2f-%.2f]' % (spread[0][0], spread[-1][0])):>15s}",
                      flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
