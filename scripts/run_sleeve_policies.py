"""What should the defensive sleeve actually do?

A cash sleeve that is never touched is not a strategy, it is a smaller position
wearing a different name: holding 30% in cash and risking r on the rest is
arithmetically the same as holding nothing back and risking 0.7r.  So the only
interesting question is whether any *rule* for moving money between the sleeve
and the strategy beats simply running smaller.

Policies compared, all on the same trade list and the same tie-break orderings:

* **all in**            no sleeve, risk scaled down to match average exposure
* **static sleeve**     70/30 fixed, sleeve never moves
* **rebalanced**        restore 70/30 whenever the split drifts past a band,
                        which mechanically adds after losses and trims after gains
* **drawdown deploy**   move sleeve cash into the strategy once it is a set
                        distance off its peak, and refill on recovery
* **vol targeted**      scale risk by target / trailing realised volatility

The control that matters is `static sleeve` versus `rebalanced`: if the rule
adds nothing over holding the same fixed split, then the sleeve's only job is to
be smaller, and no amount of policy will change that.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from datetime import date
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", type=Path,
                        default=Path("out/strategy/turtle_trades.csv"))
    parser.add_argument("--interval", default="30-minute")
    parser.add_argument("--system", default="System 2 (55/20)")
    parser.add_argument("--long-only", action="store_true", default=True)
    parser.add_argument("--risk", type=float, default=0.004,
                        help="fraction of trading capital risked per R")
    parser.add_argument("--sleeve", type=float, default=0.30)
    parser.add_argument("--band", type=float, default=0.05,
                        help="rebalance when the sleeve weight drifts this far")
    parser.add_argument("--deploy-drawdown", type=float, default=0.20)
    parser.add_argument("--vol-target", type=float, default=0.20)
    parser.add_argument("--vol-window", type=int, default=60)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/sleeve_policies.json"))
    return parser.parse_args(argv)


def load(path: Path, interval: str, system: str, long_only: bool):
    with path.open(encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle)
                if r["interval"] == interval and r["system"] == system]
    for row in rows:
        row["net_r"] = float(row["net_r"])
        row["direction"] = int(row["direction"])
    if long_only:
        rows = [r for r in rows if r["direction"] > 0]
    return rows


def cap(rows, limit, rng):
    shuffled = list(rows)
    rng.shuffle(shuffled)
    live, taken = [], []
    for row in sorted(shuffled, key=lambda r: r["entry_timestamp"]):
        live = [x for x in live if x["exit_timestamp"] > row["entry_timestamp"]]
        if len(live) >= limit:
            continue
        live.append(row)
        taken.append(row)
    return taken


def daily_r(rows):
    out = defaultdict(float)
    for row in rows:
        out[row["exit_timestamp"][:10]] += row["net_r"]
    return out


def simulate(by_day, days, policy, args):
    """Walk the calendar; returns the total-NAV path."""
    trading = 1000.0 * (1.0 - args.sleeve)
    sleeve = 1000.0 * args.sleeve
    if policy == "all in":
        trading, sleeve = 1000.0, 0.0
    peak_trading = trading
    path = []
    returns: list[float] = []
    for day in days:
        total_before = trading + sleeve
        risk = args.risk
        if policy == "vol targeted" and len(returns) >= args.vol_window:
            window = returns[-args.vol_window:]
            realised = statistics.stdev(window) * math.sqrt(252.0)
            if realised > 0:
                # Cap the scale so a quiet stretch cannot invent leverage.
                risk *= min(args.vol_target / realised, 2.0)
        gain = by_day.get(day, 0.0) * risk * trading
        trading = max(0.0, trading + gain)
        peak_trading = max(peak_trading, trading)

        if policy == "rebalanced":
            total = trading + sleeve
            want = total * (1.0 - args.sleeve)
            if total > 0 and abs(trading - want) / total > args.band:
                trading, sleeve = want, total - want
        elif policy == "drawdown deploy":
            if peak_trading > 0 and trading < peak_trading * (1 - args.deploy_drawdown):
                move = sleeve * 0.5
                trading += move
                sleeve -= move
            elif trading >= peak_trading and sleeve < 1000.0 * args.sleeve:
                give = min(trading * 0.10, 1000.0 * args.sleeve - sleeve)
                trading -= give
                sleeve += give

        total_after = trading + sleeve
        path.append(total_after)
        returns.append(total_after / total_before - 1.0 if total_before else 0.0)
    return path


def summarise(path, days):
    peak, maxdd = path[0], 0.0
    for value in path:
        peak = max(peak, value)
        maxdd = min(maxdd, value / peak - 1.0)
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.25
    cagr = (path[-1] / path[0]) ** (1 / years) - 1 if years > 0 and path[-1] > 0 else -1.0
    rets = [path[i] / path[i - 1] - 1.0 for i in range(1, len(path)) if path[i - 1] > 0]
    vol = statistics.stdev(rets) * math.sqrt(252.0) if len(rets) > 1 else 0.0
    return {
        "final": path[-1], "cagr": cagr, "maxdd": maxdd, "vol": vol,
        "sharpe": (statistics.mean(rets) / statistics.stdev(rets) * math.sqrt(252.0))
                  if len(rets) > 1 and statistics.stdev(rets) else math.nan,
        "calmar": cagr / abs(maxdd) if maxdd else math.nan,
    }


def main(argv=None):
    args = parse_args(argv)
    rows = load(args.trades, args.interval, args.system, args.long_only)
    if not rows:
        raise SystemExit("error: no trades matched")
    stamps = sorted({r["exit_timestamp"][:10] for r in rows})
    start, end = date.fromisoformat(stamps[0]), date.fromisoformat(stamps[-1])
    days, day = [], start
    while day <= end:
        if day.weekday() < 5:
            days.append(day.isoformat())
        day = date.fromordinal(day.toordinal() + 1)

    policies = ["all in", "static sleeve", "rebalanced", "drawdown deploy",
                "vol targeted"]
    report = {"settings": vars(args) | {"trades": len(rows), "days": len(days)}}
    report["settings"] = {k: str(v) for k, v in report["settings"].items()}

    collected = {p: defaultdict(list) for p in policies}
    for seed in range(args.trials):
        taken = cap(rows, args.max_positions, random.Random(seed))
        by_day = daily_r(taken)
        for policy in policies:
            stats = summarise(simulate(by_day, days, policy, args), days)
            for key, value in stats.items():
                collected[policy][key].append(value)

    pick = lambda xs, f: sorted(xs)[int(f * len(xs))]
    print(f"{args.interval} / {args.system}"
          f"{' / long only' if args.long_only else ''}, "
          f"{len(rows):,} signals, risk {args.risk:.2%} per R, "
          f"sleeve {args.sleeve:.0%}\n")
    print(f"  {'policy':18s} {'median $':>10s} {'5-95%':>21s} {'CAGR':>7s} "
          f"{'maxDD':>8s} {'Calmar':>7s} {'Sharpe':>7s}")
    for policy in policies:
        c = collected[policy]
        print(f"  {policy:18s} ${pick(c['final'], .5):>9,.0f} "
              f"${pick(c['final'], .05):>9,.0f}-${pick(c['final'], .95):>9,.0f} "
              f"{pick(c['cagr'], .5):>7.1%} {pick(c['maxdd'], .5):>8.1%} "
              f"{pick(c['calmar'], .5):>7.2f} {pick(c['sharpe'], .5):>7.2f}")
        report[policy] = {k: {"p05": pick(v, .05), "median": pick(v, .5),
                              "p95": pick(v, .95)} for k, v in c.items()}

    # The control: paired, same ordering, rebalanced minus static.
    wins = sum(1 for a, b in zip(collected["rebalanced"]["final"],
                                 collected["static sleeve"]["final"]) if a > b)
    dd_wins = sum(1 for a, b in zip(collected["rebalanced"]["maxdd"],
                                    collected["static sleeve"]["maxdd"]) if a > b)
    print(f"\n  PAIRED CONTROL, rebalanced vs static sleeve on identical orderings:")
    print(f"    higher final NAV in {wins}/{args.trials} runs")
    print(f"    shallower drawdown in {dd_wins}/{args.trials} runs")
    report["control"] = {"rebalanced_nav_wins": wins, "rebalanced_dd_wins": dd_wins,
                         "trials": args.trials}

    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
