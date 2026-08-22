"""Fading a move the options did not price, into a zone the volume left behind.

The setup has three parts and each is separately falsifiable.

*The stretch.*  Options price an expected daily move; when a session closes two
or three times that far, something has happened the chain did not anticipate.
This is the one use of the option data the feature-power study actually
supports: implied vol forecasts *magnitude* (0.47 on SPY against next-session
absolute return, 0.43 on QQQ, five sub-periods out of five) and forecasts
direction not at all.  Here it is used only to normalise how large a move is,
which is a magnitude question.

*The zone.*  Support and resistance are taken from where volume actually
traded, not from drawn lines: a composite volume profile over the trailing
sessions gives a point of control and a value area, and the nearest of those
above the fill becomes the target.  Pivot levels run as an independent second
definition, because a result that holds under one construction of "zone" and
not the other is a result about the construction.

*The entry.*  A resting buy limit below the market.  This matters more than it
looks: the bracket study found the whole difference between a dead strategy and
a live one was the order type -- a breakout stop is a taker at 5bp a leg and
cannot be anything else, while a resting limit is a maker at 2.5bp.  A winner
here leaves on a limit at the zone and pays maker twice.

Long only.  The short side of every fade tested in this programme has gone
negative at tradeable cost with borrow removed entirely, so a short arm would
measure the borrow assumption rather than the setup.

Three controls, and the third decides whether "support and resistance" means
anything at all.

*A random-day null.*  The same limit, the same distance below the close, on days
that did not trigger.  If the stretch adds nothing, the fade was just buying
dips and the option data is decoration.

*The offset sweep.*  An entry that never fills cannot lose, so a fill rate is
reported beside every result; a rule that fills 3% of the time is a different
strategy from one that fills 60% of the time regardless of what it earns.

*A fixed target at the same distance.*  The zone target is replaced by a plain
multiple of risk, chosen per cell to match the zone's median distance.  If the
fixed target does as well, the zone contributed nothing and the result is about
holding a fade for a while, not about support and resistance.
"""

from __future__ import annotations

import argparse
import bisect
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

from savi_uz.multitimeframe_retest import confirmed_pivots  # noqa: E402
from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import resample_regular_session  # noqa: E402
from savi_uz.volume_profile import Bar, build_profile  # noqa: E402

TRADING_DAYS = 252.0
OFFSETS_VOL = (0.0, 0.5, 1.0, 2.0)
OFFSETS_PCT = (0.02, 0.05, 0.10)
STRETCHES = (2.0, 3.0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--options", type=Path,
                        default=Path("data/options/alphavantage.db"))
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--vol-source", choices=("implied", "realized"),
                        default="implied",
                        help="what normalises the move; realized needs no chain")
    parser.add_argument("--zone-minutes", type=int, default=30)
    parser.add_argument("--zone-sessions", type=int, default=20,
                        help="trailing sessions the profile is built from")
    parser.add_argument("--wait-sessions", type=int, default=3,
                        help="how long the resting limit stays live")
    parser.add_argument("--hold-sessions", type=int, default=5)
    parser.add_argument("--stop-mult", type=float, default=2.0,
                        help="stop distance below the fill, in implied daily moves")
    parser.add_argument("--maker-bp", type=float, default=2.5)
    parser.add_argument("--taker-bp", type=float, default=5.0)
    parser.add_argument("--leverage", type=float, default=20.0)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--null-draws", type=int, default=20)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/vol_stretch_zones.json"))
    return parser.parse_args(argv)


def ratio(numerator, denominator):
    if denominator is None or math.isnan(denominator) or denominator <= 0:
        return math.nan
    return numerator / denominator


# ---------------------------------------------------------------------------
# data


def daily_from(bars):
    """One bar per session, aggregated from the intraday series already loaded."""
    groups = defaultdict(list)
    for bar in bars:
        groups[bar.timestamp[:10]].append(bar)
    out = []
    for day in sorted(groups):
        rows = groups[day]
        volumes = [r.volume for r in rows if r.volume is not None]
        out.append(Bar(day, rows[0].open, max(r.high for r in rows),
                       min(r.low for r in rows), rows[-1].close,
                       sum(volumes) if volumes else None))
    return out


def load(args):
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    book = {}
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker"):
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='5min' AND ts>=? ORDER BY ts", (ticker, args.start)).fetchall()
        raw = adjust_bars([Bar(*r) for r in rows], splits.get(ticker, []))
        if len({b.timestamp[:10] for b in raw}) < 400:
            continue
        five = resample_regular_session(raw, minutes=5)
        zone = resample_regular_session(raw, minutes=args.zone_minutes)
        book[ticker] = (five, zone, daily_from(five))
    connection.close()
    return book


def realized_moves(book, window=20):
    """The same normaliser built from price alone, for the control.

    If a trailing realised deviation does the job the chain does, the setup does
    not need option data -- and can then be tested on the wider universe, which
    is the one place this result is most exposed.
    """
    panel = {}
    for ticker, (_, _, daily) in book.items():
        table, returns = {}, []
        for index in range(1, len(daily)):
            previous, current = daily[index - 1].close, daily[index].close
            if previous > 0 and current > 0:
                returns.append(math.log(current / previous))
            if len(returns) > window:
                returns.pop(0)
            if len(returns) == window:
                # Read on the following session, so the day's own return is
                # already excluded from the deviation that scales it.
                table[daily[index].timestamp[:10]] = statistics.pstdev(returns)
        if len(table) >= 100:
            panel[ticker] = table
    return panel


def implied_moves(path, tickers, start):
    """Expected one-session move in price terms, from the prior close's chain."""
    if not path.exists():
        return {}
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    panel = {}
    for ticker in tickers:
        table = {}
        for day, iv, spot in connection.execute(
                "SELECT observation_date, atm_iv, spot FROM av_daily WHERE symbol=? "
                "AND atm_iv IS NOT NULL AND observation_date>=? "
                "ORDER BY observation_date", (ticker, start[:10])):
            if iv and spot and iv > 0 and spot > 0:
                table[day[:10]] = float(iv) / math.sqrt(TRADING_DAYS)
        if len(table) >= 100:
            panel[ticker] = table
    connection.close()
    return panel


# ---------------------------------------------------------------------------
# zones


def session_index(bars):
    """First and last bar index of each session, oldest first."""
    spans, first = [], 0
    while first < len(bars):
        day = bars[first].timestamp[:10]
        last = first
        while last + 1 < len(bars) and bars[last + 1].timestamp[:10] == day:
            last += 1
        spans.append((day, first, last))
        first = last + 1
    return spans


def zone_levels(zone_bars, spans, position, sessions):
    """Resistance candidates above, from volume and from pivots, as of a session.

    Both are read from bars that closed before the trigger session ended, so
    neither can see the move it is supposed to be a level for.
    """
    first = max(0, position - sessions + 1)
    window = zone_bars[spans[first][1]:spans[position][2] + 1]
    if len(window) < 20:
        return None, None
    profile = build_profile(window)
    volume_levels = []
    if profile:
        volume_levels = sorted({profile.poc, profile.value_high, profile.high})
    pivot_levels = sorted({window[i].high for i in confirmed_pivots(
        window, end_index=len(window) - 1, span=2, lookback=len(window),
        direction=-1)})
    return volume_levels, pivot_levels


def nearest_above(levels, price):
    if not levels:
        return math.nan
    position = bisect.bisect_right(levels, price)
    return levels[position] if position < len(levels) else math.nan


# ---------------------------------------------------------------------------
# the trade


def triggers(daily, moves, stretch):
    """Sessions whose close fell at least ``stretch`` implied moves."""
    out = []
    for index in range(1, len(daily)):
        day = daily[index].timestamp[:10]
        previous = daily[index - 1].timestamp[:10]
        step = moves.get(previous)
        if step is None or step <= 0:
            continue
        implied = step * daily[index - 1].close
        change = daily[index].close - daily[index - 1].close
        if implied > 0 and change <= -stretch * implied:
            out.append((index, day, implied, daily[index].close))
    return out


def run_trade(five, five_spans, five_day_at, day, limit, stop, target, args):
    """Rest the limit, then manage the fill.  Returns None when it never fills."""
    position = five_day_at.get(day)
    if position is None:
        return None
    watch_last = five_spans[min(position + args.wait_sessions, len(five_spans) - 1)][2]
    fill_index = None
    for index in range(five_spans[position][2] + 1, watch_last + 1):
        if five[index].low <= limit:
            fill_index = index
            break
    if fill_index is None:
        return None
    fill = min(limit, five[fill_index].open)
    risk = fill - stop
    if risk <= 0:
        return None
    hold_day = five[fill_index].timestamp[:10]
    hold_at = five_day_at.get(hold_day, position)
    deadline = five_spans[min(hold_at + args.hold_sessions - 1,
                              len(five_spans) - 1)][2]
    for index in range(fill_index + 1, min(deadline, len(five) - 1) + 1):
        bar = five[index]
        if bar.low <= stop:                       # stop first when both touch
            return fill, stop, "stop", bar.timestamp, risk
        if bar.high >= target:
            return fill, target, "target", bar.timestamp, risk
    end = min(deadline, len(five) - 1)
    return fill, five[end].close, "time", five[end].timestamp, risk


def cost_r(reason, fill, risk, args):
    """Entry always rests as a limit; only a stop or a timeout pays taker out."""
    exit_leg = args.maker_bp if reason == "target" else args.taker_bp
    return (args.maker_bp + exit_leg) / 10_000 * fill / risk


def build(book, panel, args, stretch, offset, kind, zone_kind, fixed_rr=None,
          random_days=None):
    """Every trade at one cell.  ``fixed_rr`` replaces the zone with a multiple."""
    trades = []
    for ticker, (five, zone_bars, daily) in book.items():
        moves = panel.get(ticker)
        if not moves:
            continue
        five_spans = session_index(five)
        five_day_at = {day: i for i, (day, _, _) in enumerate(five_spans)}
        zone_spans = session_index(zone_bars)
        zone_day_at = {day: i for i, (day, _, _) in enumerate(zone_spans)}

        if random_days is None:
            events = triggers(daily, moves, stretch)
        else:
            rng = random.Random(random_days + hash(ticker) % 10_000)
            pool = [(i, daily[i].timestamp[:10],
                     moves.get(daily[i - 1].timestamp[:10], 0.0) * daily[i - 1].close,
                     daily[i].close)
                    for i in range(1, len(daily))
                    if moves.get(daily[i - 1].timestamp[:10])]
            triggered = {d for _, d, _, _ in triggers(daily, moves, stretch)}
            quiet = [row for row in pool if row[1] not in triggered and row[2] > 0]
            want = min(len(triggered), len(quiet))
            events = sorted(rng.sample(quiet, want), key=lambda r: r[0]) if want else []

        for _, day, implied, close in events:
            if implied <= 0:
                continue
            drop = offset * implied if kind == "vol" else offset * close
            limit = close - drop
            stop = limit - args.stop_mult * implied
            position = zone_day_at.get(day)
            if position is None:
                continue
            if fixed_rr is not None:
                target = limit + fixed_rr * (limit - stop)
            else:
                volume_levels, pivot_levels = zone_levels(
                    zone_bars, zone_spans, position, args.zone_sessions)
                levels = volume_levels if zone_kind == "volume" else pivot_levels
                target = nearest_above(levels or [], limit)
                if math.isnan(target) or target <= limit:
                    continue
            outcome = run_trade(five, five_spans, five_day_at, day,
                                limit, stop, target, args)
            if outcome is None:
                continue
            fill, price, reason, exit_stamp, risk = outcome
            trades.append({
                "ticker": ticker, "entry": day, "exit": exit_stamp[:10],
                "r": (price - fill) / risk - cost_r(reason, fill, risk, args),
                "dir": 1, "reason": reason, "stop_pct": risk / fill,
                "reward": (target - fill) / risk,
            })
    trades.sort(key=lambda t: t["entry"])
    return trades


# ---------------------------------------------------------------------------
# assessment


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


def daily_r(taken, weight=None):
    by_day = defaultdict(float)
    for trade in taken:
        by_day[trade["exit"]] += trade["r"] * (weight(trade) if weight else 1.0)
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
    def drawdown(risk):
        return statistics.median(abs(path(v, risk)[1]) for _, v in series)
    if drawdown(hi) < target:
        return hi
    for _ in range(28):
        mid = math.sqrt(lo * hi)
        if drawdown(mid) < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def sharpe(stream):
    sd = statistics.pstdev(stream)
    return statistics.fmean(stream) / sd * math.sqrt(252) if sd > 0 else float("nan")


def assess(pooled, args):
    if len(pooled) < 100:
        return None
    caps = [cap(pooled, args.max_positions, random.Random(s))
            for s in range(args.trials)]
    series = []
    for taken in caps:
        marks = daily_r(taken)
        days = sorted(marks)
        series.append((days, [marks[d] for d in days]))
    risk = solve_risk(series, args.target_dd)
    cagrs, sharpes, counts = [], [], []
    for (days, values), taken in zip(series, caps):
        nav, _ = path(values, risk)
        years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25
        cagrs.append((nav / 1000.0) ** (1 / years) - 1 if nav > 0 else -1.0)
        sharpes.append(sharpe([v * risk for v in values]))
        counts.append(len(taken))
    spread = sorted(sharpes)
    return {"offered": len(pooled), "taken": int(statistics.median(counts)),
            "risk_fraction": risk, "sharpe": statistics.median(spread),
            "sharpe_p05": spread[int(.05 * len(spread))],
            "sharpe_p95": spread[min(int(.95 * len(spread)), len(spread) - 1)],
            "cagr": statistics.median(cagrs)}


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
            "median_reward": statistics.median(t["reward"] for t in trades),
            "median_stop_pct": statistics.median(t["stop_pct"] for t in trades)}


def line(label, stats, result):
    band = ("[%.2f-%.2f]" % (result["sharpe_p05"], result["sharpe_p95"])
            if result else "")
    print(f"  {label:26s} {stats['trades']:>7,d} {stats['median_reward']:>7.2f} "
          f"{stats['target_rate']:>7.1%} {stats['stop_rate']:>7.1%} "
          f"{stats['time_rate']:>6.1%} {stats['mean_r']:>+7.3f} "
          f"{(result['sharpe'] if result else float('nan')):>7.2f} {band:>15s}",
          flush=True)


def header():
    print(f"  {'cell':26s} {'trades':>7s} {'reward':>7s} {'target':>7s} "
          f"{'stop':>7s} {'time':>6s} {'meanR':>7s} {'Sharpe':>7s} {'[5-95%]':>15s}")
    print("  " + "-" * 100)


def leverage_report(trades, args):
    taken = cap(trades, args.max_positions, random.Random(0))
    if not taken:
        return None
    per_position = args.leverage / args.max_positions
    marks = daily_r(taken, weight=lambda t: per_position * t["stop_pct"])
    days = sorted(marks)
    nav, worst = path([marks[d] for d in days], 1.0)
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25
    risks = sorted(per_position * t["stop_pct"] for t in taken)
    return {"per_position_leverage": per_position,
            "median_risk_per_trade": statistics.median(risks),
            "worst_risk_per_trade": risks[-1],
            "max_drawdown": worst, "ruined": nav <= 0.0,
            "cagr": (nav / 1000.0) ** (1 / years) - 1 if nav > 0 else -1.0}


def main(argv=None) -> int:
    args = parse_args(argv)
    book = load(args)
    panel = (realized_moves(book) if args.vol_source == "realized"
             else implied_moves(args.options, sorted(book), args.start))
    print(f"{len(book)} instruments, {args.vol_source} moves for {len(panel)}, "
          f"limit rests {args.wait_sessions} sessions, held {args.hold_sessions}, "
          f"stop {args.stop_mult:g} implied moves below the fill")
    census = {}
    for stretch in STRETCHES:
        total = sum(len(triggers(daily, panel.get(t, {}), stretch))
                    for t, (_, _, daily) in book.items())
        census[f"{stretch:g}x"] = total
        print(f"  sessions closing {stretch:g}x implied or worse: {total:,d}")

    report = {"census": census, "vol_offsets": {}, "pct_offsets": {},
              "pivot_zones": {}, "fixed_target": {}, "null": {}, "leverage": {}}

    print("\n########## entry offset in implied daily moves, volume zones ##########")
    header()
    cells = {}
    for stretch in STRETCHES:
        for offset in OFFSETS_VOL:
            trades = build(book, panel, args, stretch, offset, "vol", "volume")
            if len(trades) < 100:
                print(f"  {f'{stretch:g}x, {offset:g} moves below':26s} "
                      f"{len(trades):>7,d}   too few")
                continue
            stats, result = describe(trades), assess(trades, args)
            cells[(stretch, offset)] = trades
            report["vol_offsets"][f"{stretch:g}x|{offset:g}m"] = {
                **stats, **(result or {})}
            line(f"{stretch:g}x, {offset:g} moves below", stats, result)

    print("\n########## entry offset as a percent of price ##########")
    header()
    for stretch in STRETCHES:
        for offset in OFFSETS_PCT:
            trades = build(book, panel, args, stretch, offset, "pct", "volume")
            if len(trades) < 100:
                print(f"  {f'{stretch:g}x, {offset:.0%} below':26s} "
                      f"{len(trades):>7,d}   too few")
                continue
            stats, result = describe(trades), assess(trades, args)
            report["pct_offsets"][f"{stretch:g}x|{offset:.0%}"] = {
                **stats, **(result or {})}
            line(f"{stretch:g}x, {offset:.0%} below", stats, result)

    if not cells:
        print("\nno cell produced enough trades")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        return 1

    best = max(cells, key=lambda k: report["vol_offsets"][
        f"{k[0]:g}x|{k[1]:g}m"]["sharpe"])
    stretch, offset = best
    print(f"\n########## controls at {stretch:g}x, {offset:g} moves below ##########")
    header()
    line("volume zones (incumbent)", describe(cells[best]), assess(cells[best], args))

    pivots = build(book, panel, args, stretch, offset, "vol", "pivot")
    if len(pivots) >= 100:
        stats, result = describe(pivots), assess(pivots, args)
        report["pivot_zones"] = {**stats, **(result or {})}
        line("pivot zones (replication)", stats, result)

    reward = describe(cells[best])["median_reward"]
    fixed = build(book, panel, args, stretch, offset, "vol", "volume",
                  fixed_rr=reward)
    if len(fixed) >= 100:
        stats, result = describe(fixed), assess(fixed, args)
        report["fixed_target"] = {**stats, **(result or {})}
        line(f"fixed {reward:.2f}R target (no zone)", stats, result)

    quiet = build(book, panel, args, stretch, offset, "vol", "volume",
                  random_days=4242)
    if len(quiet) >= 100:
        stats, result = describe(quiet), assess(quiet, args)
        report["null"] = {**stats, **(result or {})}
        line("quiet days, same limit (null)", stats, result)

    print(f"\n########## what {args.leverage:g}x does to the equity path ##########")
    lev = leverage_report(cells[best], args)
    if lev:
        report["leverage"] = lev
        print(f"    position size            {lev['per_position_leverage']:.2f}x equity"
              f"  ({args.leverage:g}x gross across {args.max_positions} slots)")
        print(f"    equity risked per trade  {lev['median_risk_per_trade']:.2%} median,"
              f"  {lev['worst_risk_per_trade']:.2%} worst")
        print(f"    max drawdown             {lev['max_drawdown']:.1%}"
              f"{'    RUINED' if lev['ruined'] else ''}")
        print(f"    CAGR                     {lev['cagr']:.1%}")
    incumbent = report["vol_offsets"][f"{stretch:g}x|{offset:g}m"]
    if incumbent.get("risk_fraction") and incumbent.get("median_stop_pct"):
        supported = incumbent["risk_fraction"] / incumbent["median_stop_pct"]
        report["leverage"]["supported_leverage_at_matched_dd"] = supported
        print(f"    leverage the matched {args.target_dd:.0%} drawdown supports: "
              f"{supported:.2f}x")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
