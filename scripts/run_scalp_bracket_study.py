"""A 3:1 bracket held one to three sessions, and whether a small edge carries it.

Everything else in this programme lets winners run and takes the fat tail as
payment for a 25% win rate.  This does the opposite: a fixed target at three
times the stop, a hard time limit, and no pyramid.  The thesis is that a small
edge which is worthless to a fat-tail book can still carry a bracket, because a
bracket banks its result on every trade rather than waiting for one in twenty
to pay for the rest.

The conditions tested are not invented here.

*Volume and price* -- relative share volume, cross-sectional strength, and how
far the fill chased past the channel edge.  The last three replicated across two
universes and three bar sizes in the trade-level study.

*Option activity* -- and the reason to expect anything is specific.  The feature
power study found these series forecast **magnitude and not direction**: against
next-session absolute return, ATM implied vol scores 0.47 on SPY and 0.43 on
QQQ, gamma balance -0.35 and -0.31, total option volume 0.20 and 0.15, every one
holding its sign in all five sub-periods.  Against *signed* return nothing
clears 0.08.  A directional book cannot use a magnitude forecast and eleven
overlays died proving it.  A bracket can: its worst outcome is not the stop, it
is the time limit expiring with the trade unresolved, having paid the round trip
for a coin flip.  Whether the target is reachable inside three sessions is
exactly a magnitude question.

*Macro regime* -- the VIX term structure, credit spreads, and the curve, each
lagged a session.  These are regime labels rather than signals, and the
programme has rejected regime overlays before, so they are carried mainly as a
falsifier: if the option features work only when the macro label agrees, that is
a smaller claim than it looks.

Every condition is read from the session *before* the entry, and the option
snapshot is an end-of-day observation, so a trade on session D reads D-1.

On leverage: it is not a return knob.  In R space the round trip costs
``cost / stop_distance`` whatever the size, so leverage cancels out of Sharpe
entirely and multiplies mean and drawdown together.  What 20x decides is whether
the equity path survives, so the leverage section reports the per-trade equity
risk it implies and the drawdown it produces rather than a flattering CAGR.

Bracket ambiguity is resolved the way ``sweep_engulf`` resolves it: when one bar
touches both stop and target the order is unknowable and the stop is charged.
Signals are found on the resampled interval and the bracket is then resolved on
the underlying five-minute bars, so a 3:1 target is never decided by the high
and low of a bar half a session long.
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

from savi_uz.split_adjust import adjust_bars, load_splits  # noqa: E402
from savi_uz.sweep_engulf import _fill_on_bar, resample_regular_session  # noqa: E402
from savi_uz.turtle import relative_volume, rolling_extremes, wilder_atr  # noqa: E402
from savi_uz.volume_profile import Bar  # noqa: E402

ENTRY_WINDOW = 55
ATR_WINDOW = 20
STOPS = (0.5, 1.0, 1.5, 2.0)
WINDOWS = (1, 2, 3)

PRICE_KEYS = ("rs_rank", "extension", "rel_volume", "mkt_mom")
OPTION_KEYS = ("iv_rel", "optvol_rel", "pcv_rel", "gamma_balance", "net_gex_rel")
MACRO_KEYS = ("vix_term", "vix_rel", "credit_rel", "curve")
ALL_KEYS = PRICE_KEYS + OPTION_KEYS + MACRO_KEYS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--options", type=Path,
                        default=Path("data/options/alphavantage.db"))
    parser.add_argument("--macro", type=Path, default=Path("data/macro/macro.db"))
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--rr", type=float, default=3.0,
                        help="target as a multiple of the stop")
    parser.add_argument("--cost-bp", type=float, default=10.0,
                        help="flat round trip, used unless --realistic-cost")
    parser.add_argument("--realistic-cost", action="store_true",
                        help="price each leg by the order that fills it")
    parser.add_argument("--taker-bp", type=float, default=5.0,
                        help="one taker leg; a stop order is always one")
    parser.add_argument("--maker-bp", type=float, default=2.5,
                        help="one maker leg; only a resting limit earns it")
    parser.add_argument("--leverage", type=float, default=20.0)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--target-dd", type=float, default=0.18)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--null-draws", type=int, default=20)
    parser.add_argument("--market-ticker", default="SPY")
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/scalp_bracket.json"))
    return parser.parse_args(argv)


def ratio(numerator, denominator):
    if denominator is None or math.isnan(denominator) or denominator <= 0:
        return math.nan
    return numerator / denominator


# ---------------------------------------------------------------------------
# data


def load(args):
    """Five-minute regular-session bars per ticker, plus the signal interval."""
    splits = load_splits(args.bars)
    connection = sqlite3.connect(f"file:{args.bars}?mode=ro", uri=True)
    five, signal = {}, {}
    for (ticker,) in connection.execute(
            "SELECT DISTINCT ticker FROM bars WHERE frequency='5min' ORDER BY ticker"):
        rows = connection.execute(
            "SELECT ts,open,high,low,close,volume FROM bars WHERE ticker=? AND "
            "frequency='5min' AND ts>=? ORDER BY ts", (ticker, args.start)).fetchall()
        raw = adjust_bars([Bar(*r) for r in rows], splits.get(ticker, []))
        if len({b.timestamp[:10] for b in raw}) < 400:
            continue
        five[ticker] = resample_regular_session(raw, minutes=5)
        signal[ticker] = resample_regular_session(raw, minutes=args.minutes)
    connection.close()
    return five, signal


def trailing(values, window):
    """Mean of the ``window`` values ending at each index, NaN until filled."""
    out = [math.nan] * len(values)
    running, live = 0.0, 0
    for index, value in enumerate(values):
        running += value
        live += 1
        if index >= window:
            running -= values[index - window]
            live -= 1
        if live == window:
            out[index] = running / window
    return out


def option_panel(path, tickers, start):
    """Per name and session, option activity read against its own recent norm.

    Levels are not comparable across names -- MSTR trades a different option
    book from WMT -- so everything that has a scale is divided by its trailing
    twenty-session mean.  Gamma balance is already a ratio and is left alone.
    """
    if not path.exists():
        return {}
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    panel = {}
    for ticker in tickers:
        rows = connection.execute(
            "SELECT observation_date, atm_iv, total_volume, put_call_volume, "
            "gamma_balance, absolute_gex, net_gex FROM av_daily WHERE symbol=? "
            "AND observation_date>=? ORDER BY observation_date",
            (ticker, start[:10])).fetchall()
        if len(rows) < 60:
            continue
        days = [r[0][:10] for r in rows]
        def column(position):
            return [float(r[position]) if r[position] is not None else math.nan
                    for r in rows]
        iv, volume, pcv = column(1), column(2), column(3)
        balance, absolute, net = column(4), column(5), column(6)

        def clean(series):
            return [0.0 if math.isnan(v) else v for v in series]
        iv_mean = trailing(clean(iv), 20)
        volume_mean = trailing(clean(volume), 20)
        pcv_mean = trailing(clean(pcv), 20)
        table = {}
        for index, day in enumerate(days):
            table[day] = {
                "iv_rel": ratio(iv[index], iv_mean[index]),
                "optvol_rel": ratio(volume[index], volume_mean[index]),
                "pcv_rel": ratio(pcv[index], pcv_mean[index]),
                "gamma_balance": balance[index] if balance[index] == balance[index]
                else math.nan,
                "net_gex_rel": ratio(net[index], absolute[index])
                if absolute[index] and absolute[index] > 0 else math.nan,
            }
        panel[ticker] = (days, table)
    connection.close()
    return panel


def macro_panel(path, start):
    """VIX term structure, credit and the curve, one row per session."""
    if not path.exists():
        return ([], {})
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    wanted = ("VIXCLS", "VXVCLS", "BAA10Y", "T10Y2Y")
    series = {name: {} for name in wanted}
    for name in wanted:
        for obs_date, value in connection.execute(
                "SELECT obs_date, value FROM observations WHERE series_id=? "
                "AND value IS NOT NULL AND obs_date>=? ORDER BY obs_date",
                (name, start[:10])):
            series[name][obs_date[:10]] = float(value)
    connection.close()
    days = sorted(series["VIXCLS"])
    if not days:
        return ([], {})
    vix = [series["VIXCLS"][d] for d in days]
    vix_mean = trailing(vix, 60)
    credit = [series["BAA10Y"].get(d, math.nan) for d in days]
    credit_mean = trailing([0.0 if math.isnan(v) else v for v in credit], 60)
    table = {}
    for index, day in enumerate(days):
        three_month = series["VXVCLS"].get(day)
        table[day] = {
            "vix_term": ratio(vix[index], three_month) if three_month else math.nan,
            "vix_rel": ratio(vix[index], vix_mean[index]),
            "credit_rel": ratio(credit[index], credit_mean[index]),
            "curve": series["T10Y2Y"].get(day, math.nan),
        }
    return (days, table)


def lookup(days, table, day, keys):
    """The most recent row strictly before ``day``, so nothing is read early."""
    position = bisect.bisect_left(days, day) - 1
    if position < 0:
        return {k: math.nan for k in keys}
    return table[days[position]]


def strength_ranks(signal):
    """Cross-sectional percentile of the trailing 60-bar return, per stamp."""
    pool = defaultdict(list)
    for ticker, bars in signal.items():
        for index in range(61, len(bars)):
            base = bars[index - 61].close
            if base > 0:
                pool[bars[index].timestamp].append(
                    ((bars[index - 1].close - base) / base, ticker))
    ranks = {}
    for stamp, rows in pool.items():
        rows.sort()
        size = len(rows)
        for position, (_, ticker) in enumerate(rows):
            ranks[(ticker, stamp)] = position / (size - 1) if size > 1 else 0.5
    return ranks


def market_series(signal, ticker):
    """Trailing 20-bar momentum of the index proxy, in its own N."""
    bars = signal.get(ticker)
    if not bars:
        return {}
    atr = wilder_atr(bars, ATR_WINDOW)
    return {bars[i].timestamp: ratio(bars[i - 1].close - bars[i - 21].close, atr[i - 1])
            for i in range(22, len(bars))}


# ---------------------------------------------------------------------------
# signals and the bracket


def sessions_of(bars):
    """Ordered session dates, and the index of each session's last bar."""
    last = {}
    for index, bar in enumerate(bars):
        last[bar.timestamp[:10]] = index
    return sorted(last), last


def fresh_breakouts(signal_bars, five_bars, five_index):
    """One signal per fresh channel break, resolved to the bar that traded through.

    Firing on every bar that sits above the channel would count a single trend
    as hundreds of entries.  The signal arms only when price has fallen back
    inside, which is what makes it a breakout rather than a state.
    """
    highs = [b.high for b in signal_bars]
    levels = rolling_extremes(highs, ENTRY_WINDOW, True)
    atr = wilder_atr(signal_bars, ATR_WINDOW)
    volume = relative_volume(signal_bars, 20)
    found, armed = [], True
    for index in range(max(ENTRY_WINDOW, ATR_WINDOW) + 62, len(signal_bars)):
        level, n = levels[index], atr[index - 1]
        if math.isnan(level):
            continue
        if signal_bars[index].high < level:
            armed = True
            continue
        if not armed:
            continue
        armed = False
        if not n or math.isnan(n) or n <= 0:
            continue
        start = five_index.get(signal_bars[index].timestamp)
        if start is None:
            continue
        stop_at = (five_index.get(signal_bars[index + 1].timestamp, len(five_bars))
                   if index + 1 < len(signal_bars) else len(five_bars))
        for j in range(start, min(stop_at, len(five_bars))):
            if five_bars[j].high >= level:
                found.append({
                    "five_index": j, "stamp": five_bars[j].timestamp,
                    "signal_stamp": signal_bars[index].timestamp,
                    "fill": max(level, five_bars[j].open), "n": n,
                    "extension": ratio(signal_bars[index - 1].close - level, n),
                    "rel_volume": volume[index - 1],
                })
                break
    return found


def resolve(five_bars, days, last_bar, start, stop, target, window):
    """Walk the bracket forward until it resolves or the time limit expires.

    The entry bar itself is checked for the stop only.  Entry happened inside it
    at a known price but on an unknown path, so a target touch in the same five
    minutes cannot be credited while a stop touch has to be charged.
    """
    position = bisect.bisect_left(days, five_bars[start].timestamp[:10])
    deadline = last_bar[days[min(position + window - 1, len(days) - 1)]]
    if five_bars[start].low <= stop:
        return stop, "stop", five_bars[start].timestamp
    for index in range(start + 1, min(deadline, len(five_bars) - 1) + 1):
        outcome = _fill_on_bar(five_bars[index], 1, stop, target)
        if outcome:
            return outcome[0], outcome[1], five_bars[index].timestamp
    end = min(deadline, len(five_bars) - 1)
    return five_bars[end].close, "time", five_bars[end].timestamp


def build(prepared, args, stop_atr, window, conditions):
    """Every bracketed trade at one (stop, window) cell, one position per name."""
    trades = []
    for ticker, (found, five_bars, days, last_bar) in prepared.items():
        open_until = ""
        for signal in found:
            if signal["stamp"] < open_until:
                continue                      # a position is already live here
            risk = stop_atr * signal["n"]
            fill = signal["fill"]
            if risk <= 0 or fill <= 0:
                continue
            price, reason, exit_stamp = resolve(
                five_bars, days, last_bar, signal["five_index"],
                fill - risk, fill + args.rr * risk, window)
            open_until = exit_stamp
            # A breakout entry is a buy stop and therefore always a taker.  Only
            # the target exit rests as a limit and earns the maker rebate; a
            # stopped or timed-out trade leaves by market order and pays taker
            # again.  Charging a flat maker round trip here would be pricing an
            # execution this strategy cannot obtain.
            if args.realistic_cost:
                exit_leg = (args.maker_bp if reason in ("target", "gap_target")
                            else args.taker_bp)
                round_trip = args.taker_bp + exit_leg
            else:
                round_trip = args.cost_bp
            trade = {
                "ticker": ticker, "entry": signal["stamp"], "exit": exit_stamp,
                "r": (price - fill) / risk - round_trip / 10_000 * fill / risk,
                "dir": 1, "reason": reason, "stop_pct": risk / fill,
            }
            trade.update(conditions(ticker, signal))
            trades.append(trade)
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
    """Net R banked per session.  Holds last at most three sessions, so a trade
    is marked at its exit rather than carried mark to market."""
    by_day = defaultdict(float)
    for trade in taken:
        by_day[trade["exit"][:10]] += trade["r"] * (weight(trade) if weight else 1.0)
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
    return {
        "trades": total, "mean_r": statistics.fmean(outcomes),
        "hit_rate": (reasons["target"] + reasons["gap_target"]) / total,
        "stop_rate": (reasons["stop"] + reasons["gap_stop"]) / total,
        "time_rate": reasons["time"] / total,
        "median_stop_pct": statistics.median(t["stop_pct"] for t in trades),
    }


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    position = 0
    while position < len(order):
        stop = position
        while (stop + 1 < len(order)
               and values[order[stop + 1]] == values[order[position]]):
            stop += 1
        mean = (position + stop) / 2
        for k in range(position, stop + 1):
            out[order[k]] = mean
        position = stop + 1
    return out


def spearman(pairs):
    rows = [(x, y) for x, y in pairs if not math.isnan(x) and not math.isnan(y)]
    if len(rows) < 100:
        return math.nan, len(rows)
    xs, ys = _ranks([r[0] for r in rows]), _ranks([r[1] for r in rows])
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in xs)
                            * sum((b - my) ** 2 for b in ys))
    return (numerator / denominator if denominator else math.nan), len(rows)


def condition_table(trades, keys):
    """Each condition against net R, the hit rate, and the time-out rate.

    The time-out column is the one to read for the option features: a magnitude
    forecast should move whether the bracket resolves at all, and only then show
    up in expectancy.
    """
    report, ranked = {}, []
    for key in keys:
        rho, count = spearman([(t.get(key, math.nan), t["r"]) for t in trades])
        if math.isnan(rho):
            continue
        rows = [t for t in trades if not math.isnan(t.get(key, math.nan))]
        rows.sort(key=lambda t: t[key])
        fifth = len(rows) // 5
        if fifth < 40:
            continue
        low, high = rows[:fifth], rows[-fifth:]
        entry = {
            "rho": rho, "n": count,
            "q1": describe(low), "q5": describe(high),
        }
        report[key] = entry
        ranked.append((abs(rho), key, entry))
    ranked.sort(reverse=True)
    print(f"  {'condition':16s} {'rho':>7s} {'n':>7s} {'Q1 hit':>7s} {'Q5 hit':>7s} "
          f"{'Q1 time':>8s} {'Q5 time':>8s} {'Q1 meanR':>9s} {'Q5 meanR':>9s}")
    print("  " + "-" * 86)
    for _, key, entry in ranked:
        low, high = entry["q1"], entry["q5"]
        print(f"  {key:16s} {entry['rho']:>+7.3f} {entry['n']:>7,d} "
              f"{low['hit_rate']:>7.1%} {high['hit_rate']:>7.1%} "
              f"{low['time_rate']:>8.1%} {high['time_rate']:>8.1%} "
              f"{low['mean_r']:>+9.3f} {high['mean_r']:>+9.3f}")
    return report


def leverage_report(trades, args):
    """What the requested leverage does to the equity path, R sequence fixed.

    Position size is the leverage divided across the slot cap: holding
    ``max_positions`` names at 20x each is 120x gross and no venue offers it.
    Per-trade equity risk is then that size times the stop distance.
    """
    taken = cap(trades, args.max_positions, random.Random(0))
    if not taken:
        return None
    per_position = args.leverage / args.max_positions
    marks = daily_r(taken, weight=lambda t: per_position * t["stop_pct"])
    days = sorted(marks)
    nav, worst = path([marks[d] for d in days], 1.0)
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25
    risks = sorted(per_position * t["stop_pct"] for t in taken)
    return {
        "leverage": args.leverage, "per_position_leverage": per_position,
        "median_risk_per_trade": statistics.median(risks),
        "worst_risk_per_trade": risks[-1],
        "risk_at_full_leverage_per_position":
            args.leverage * statistics.median(t["stop_pct"] for t in taken),
        "max_drawdown": worst, "ruined": nav <= 0.0,
        "cagr": (nav / 1000.0) ** (1 / years) - 1 if nav > 0 else -1.0,
    }


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    args = parse_args(argv)
    five, signal = load(args)
    print(f"{len(five)} instruments, {args.minutes}-minute signals resolved on "
          f"5-minute bars, {args.rr:g}:1 bracket, {args.cost_bp:g}bp round trip",
          flush=True)

    ranks = strength_ranks(signal)
    market = market_series(signal, args.market_ticker)
    options = option_panel(args.options, sorted(signal), args.start)
    macro_days, macro_table = macro_panel(args.macro, args.start)
    print(f"option panel covers {len(options)} of {len(signal)} names; "
          f"macro rows {len(macro_days):,d}", flush=True)

    def conditions(ticker, entry):
        stamp = entry["signal_stamp"]
        day = stamp[:10]
        row = {
            "rs_rank": ranks.get((ticker, stamp), math.nan),
            "extension": entry["extension"],
            "rel_volume": entry["rel_volume"],
            "mkt_mom": market.get(stamp, math.nan),
        }
        if ticker in options:
            days, table = options[ticker]
            row.update(lookup(days, table, day, OPTION_KEYS))
        else:
            row.update({k: math.nan for k in OPTION_KEYS})
        row.update(lookup(macro_days, macro_table, day, MACRO_KEYS)
                   if macro_days else {k: math.nan for k in MACRO_KEYS})
        return row

    prepared = {}
    for ticker, signal_bars in signal.items():
        five_bars = five[ticker]
        five_index = {bar.timestamp: i for i, bar in enumerate(five_bars)}
        days, last_bar = sessions_of(five_bars)
        prepared[ticker] = (fresh_breakouts(signal_bars, five_bars, five_index),
                            five_bars, days, last_bar)
    print(f"{sum(len(v[0]) for v in prepared.values()):,d} fresh breakouts\n",
          flush=True)

    report = {"grid": {}, "conditions": {}, "arms": {}, "leverage": {}}
    print("########## the grid: stop distance against holding window ##########")
    print(f"  {'stop':>5s} {'window':>7s} {'trades':>7s} {'hit':>6s} {'stop':>6s} "
          f"{'time':>6s} {'meanR':>7s} {'Sharpe':>7s} {'[5-95%]':>15s}")
    cells = {}
    for stop_atr in STOPS:
        for window in WINDOWS:
            trades = build(prepared, args, stop_atr, window, conditions)
            if len(trades) < 200:
                continue
            stats = describe(trades)
            result = assess(trades, args)
            cells[(stop_atr, window)] = trades
            report["grid"][f"{stop_atr:g}N|{window}s"] = {**stats, **(result or {})}
            band = ("[%.2f-%.2f]" % (result["sharpe_p05"], result["sharpe_p95"])
                    if result else "")
            print(f"  {stop_atr:>4.1f}N {window:>6d}s {stats['trades']:>7,d} "
                  f"{stats['hit_rate']:>6.1%} {stats['stop_rate']:>6.1%} "
                  f"{stats['time_rate']:>6.1%} {stats['mean_r']:>+7.3f} "
                  f"{(result['sharpe'] if result else float('nan')):>7.2f} "
                  f"{band:>15s}", flush=True)

    if not cells:
        print("no cell produced enough trades")
        return 1

    best = max(cells, key=lambda k: report["grid"][f"{k[0]:g}N|{k[1]}s"]["sharpe"])
    trades = cells[best]
    print(f"\n########## conditions at {best[0]:g}N stop, {best[1]}-session window "
          f"##########")
    print("  Price and volume first, then option activity, then macro regime.\n"
          "  Read the time-out columns: a magnitude forecast should move whether\n"
          "  the bracket resolves before it moves what it pays.\n")
    report["conditions"] = condition_table(trades, ALL_KEYS)

    # A combined filter from whatever separated the time-out rate, plus its exact
    # reversal and a random subset of the same size.
    survivors = [(k, v["rho"]) for k, v in report["conditions"].items()
                 if abs(v["rho"]) >= 0.02
                 and abs(v["q5"]["time_rate"] - v["q1"]["time_rate"]) >= 0.03]
    print(f"\n  conditions that moved both expectancy and the time-out rate: "
          f"{[k for k, _ in survivors] or 'none'}")

    if survivors:
        medians = {k: statistics.median([t[k] for t in trades
                                         if not math.isnan(t.get(k, math.nan))])
                   for k, _ in survivors}
        def score(trade):
            total = 0
            for key, rho in survivors:
                value = trade.get(key, math.nan)
                if math.isnan(value):
                    continue
                if (value >= medians[key]) == (rho > 0):
                    total += 1
            return total
        need = max(1, (len(survivors) + 1) // 2)
        good = [t for t in trades if score(t) >= need]
        bad = [t for t in trades if score(t) < need]

        print(f"\n  {'arm':32s} {'offered':>8s} {'taken':>7s} {'hit':>6s} "
              f"{'time':>6s} {'meanR':>7s} {'Sharpe':>7s} {'[5-95%]':>15s} {'CAGR':>8s}")
        for label, pooled in (("all breakouts (incumbent)", trades),
                              ("favourable conditions", good),
                              ("unfavourable (reversal)", bad)):
            result = assess(pooled, args)
            if not result:
                continue
            stats = describe(pooled)
            report["arms"][label] = {**stats, **result}
            print(f"  {label:32s} {result['offered']:>8,d} {result['taken']:>7,d} "
                  f"{stats['hit_rate']:>6.1%} {stats['time_rate']:>6.1%} "
                  f"{stats['mean_r']:>+7.3f} {result['sharpe']:>7.2f} "
                  f"{('[%.2f-%.2f]' % (result['sharpe_p05'], result['sharpe_p95'])):>15s} "
                  f"{result['cagr']:>8.1%}", flush=True)

        nulls = []
        for draw in range(args.null_draws):
            rng = random.Random(7700 + draw)
            outcome = assess(rng.sample(trades, min(len(good), len(trades))), args)
            if outcome:
                nulls.append(outcome["sharpe"])
        if nulls:
            nulls.sort()
            edge = report["arms"].get("favourable conditions", {}).get(
                "sharpe", math.nan)
            beat = sum(1 for x in nulls if x >= edge) / len(nulls)
            print(f"  {'random subset, same size (null)':32s} {'':>8s} {'':>7s} "
                  f"{'':>6s} {'':>6s} {'':>7s} {statistics.median(nulls):>7.2f} "
                  f"{('[%.2f-%.2f]' % (nulls[0], nulls[-1])):>15s}")
            print(f"  -> favourable beats the coin in {1 - beat:.0%} of draws")
            report["arms"]["random null"] = {
                "sharpe": statistics.median(nulls), "low": nulls[0],
                "high": nulls[-1], "share_null_at_or_above": beat}
        pools = (("all breakouts", trades), ("favourable", good))
    else:
        pools = (("all breakouts", trades),)

    print(f"\n########## what {args.leverage:g}x does to the equity path ##########")
    for label, pooled in pools:
        lev = leverage_report(pooled, args)
        if not lev:
            continue
        report["leverage"][label] = lev
        print(f"  {label}")
        print(f"    position size            {lev['per_position_leverage']:.2f}x equity"
              f"  ({args.leverage:g}x gross across {args.max_positions} slots)")
        print(f"    equity risked per trade  {lev['median_risk_per_trade']:.2%} median,"
              f"  {lev['worst_risk_per_trade']:.2%} worst")
        print(f"    at {args.leverage:g}x per position instead: "
              f"{lev['risk_at_full_leverage_per_position']:.1%} per trade")
        print(f"    max drawdown             {lev['max_drawdown']:.1%}"
              f"{'    RUINED' if lev['ruined'] else ''}")
        print(f"    CAGR                     {lev['cagr']:.1%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
