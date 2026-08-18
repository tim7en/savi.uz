"""Portfolio-level analytics for the Turtle trade list.

The engine reports each trade in R, which is risk normalised but says nothing
about what an account would have done.  This turns the trade list into an equity
curve under an explicit sizing convention, and reports the exposure that
convention actually implies.

Sizing convention, stated because it drives every number below:

* One unit risks ``risk_fraction`` of equity for a 1N move, which is the
  original rule at 1%.  "2x leverage" here means 2% per unit, so both the return
  and the drawdown scale with it.
* Each trade's profit and loss is realised on its exit date.  Same-day exits are
  summed before compounding, so two winners closing together do not compound
  against each other.
* Risk is a fixed fraction of *current* equity, so the curve compounds.
* A portfolio cap limits how many units may be open at once across the whole
  book.  This is not optional dressing: the original rules cap units per market,
  per correlated group and per direction, and without a cap a 27 instrument book
  puts so much simultaneous risk on that a single correlated cluster ruins the
  account.  Signals arriving when the book is full are skipped, which is
  conservative -- the real rules would take a reduced size instead.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
from datetime import date
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", type=Path,
                        default=Path("out/strategy/turtle_trades.csv"))
    parser.add_argument("--interval", default="daily")
    parser.add_argument("--system", default="System 2 (55/20)")
    parser.add_argument("--max-positions", type=int, default=6,
                        help="cap on simultaneously open positions across the book")
    parser.add_argument("--out", type=Path,
                        default=Path("out/strategy/turtle_portfolio.json"))
    return parser.parse_args(argv)


def load(path: Path, interval: str, system: str) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["interval"] == interval and row["system"] == system
        ]
    for row in rows:
        for key in ("net_r", "gross_r", "cost_r", "cost_basis_r", "entry", "exit"):
            row[key] = float(row[key])
        for key in ("direction", "units", "bars_held", "sessions_held"):
            row[key] = int(row[key])
    rows.sort(key=lambda row: row["exit_timestamp"])
    return rows


def apply_portfolio_cap(rows: list[dict], max_positions: int) -> tuple[list[dict], int]:
    """Take signals in time order only while the book has a free slot.

    The cap counts *positions*, not units.  A position's final unit count is the
    result of pyramiding that happens after entry, so admitting on unit count
    would decide today using tomorrow's information -- and would bias against
    exactly the trades that ran far enough to add units.
    """
    if max_positions <= 0:
        return list(rows), 0
    ordered = sorted(rows, key=lambda row: row["entry_timestamp"])
    open_positions: list[dict] = []
    taken: list[dict] = []
    rejected = 0
    for row in ordered:
        open_positions = [
            position for position in open_positions
            if position["exit_timestamp"] > row["entry_timestamp"]
        ]
        if len(open_positions) >= max_positions:
            rejected += 1
            continue
        open_positions.append(row)
        taken.append(row)
    taken.sort(key=lambda row: row["exit_timestamp"])
    return taken, rejected


def equity_curve(rows: list[dict], risk_fraction: float):
    """Compound realised R by exit date; returns dates, equity, drawdown."""
    by_day: dict[str, float] = collections.defaultdict(float)
    for row in rows:
        by_day[row["exit_timestamp"][:10]] += row["net_r"]
    days = sorted(by_day)
    equity = 1.0
    peak = 1.0
    curve, drawdowns = [], []
    for day in days:
        equity *= max(0.0, 1.0 + risk_fraction * by_day[day])
        peak = max(peak, equity)
        curve.append(equity)
        drawdowns.append(equity / peak - 1.0)
    return days, curve, drawdowns


def concurrency(rows: list[dict]) -> tuple[int, float, list[int]]:
    """How many units are open at once, which is the real leverage taken."""
    events: list[tuple[str, int]] = []
    for row in rows:
        events.append((row["entry_timestamp"], row["units"]))
        events.append((row["exit_timestamp"], -row["units"]))
    events.sort()
    open_units = 0
    peak = 0
    samples: list[int] = []
    for _, delta in events:
        open_units += delta
        peak = max(peak, open_units)
        samples.append(open_units)
    return peak, statistics.mean(samples) if samples else 0.0, samples


def streaks(rows: list[dict]) -> tuple[int, int]:
    longest_loss = current_loss = 0
    longest_win = current_win = 0
    for row in rows:
        if row["net_r"] > 0:
            current_win += 1
            current_loss = 0
        else:
            current_loss += 1
            current_win = 0
        longest_win = max(longest_win, current_win)
        longest_loss = max(longest_loss, current_loss)
    return longest_win, longest_loss


def slice_stats(rows: list[dict]) -> dict:
    if not rows:
        return {"trades": 0}
    rs = [row["net_r"] for row in rows]
    wins = [value for value in rs if value > 0]
    losses = [value for value in rs if value <= 0]
    return {
        "trades": len(rows),
        "units": sum(row["units"] for row in rows),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(rs),
        "profit_factor": (sum(wins) / -sum(losses)) if losses and sum(losses) else None,
        "total_r": sum(rs),
        "mean_r": statistics.mean(rs),
        "median_r": statistics.median(rs),
        "mean_win_r": statistics.mean(wins) if wins else None,
        "mean_loss_r": statistics.mean(losses) if losses else None,
        "largest_win_r": max(rs),
        "largest_loss_r": min(rs),
        "mean_bars_held": statistics.mean([row["bars_held"] for row in rows]),
        "mean_sessions_held": statistics.mean([row["sessions_held"] for row in rows]),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    every = load(args.trades, args.interval, args.system)
    if not every:
        raise SystemExit(f"error: no trades for {args.interval} / {args.system}")
    rows, rejected = apply_portfolio_cap(every, args.max_positions)

    first = date.fromisoformat(rows[0]["entry_timestamp"][:10])
    last = date.fromisoformat(rows[-1]["exit_timestamp"][:10])
    weeks = (last - first).days / 7.0
    years = (last - first).days / 365.25

    result = {
        "interval": args.interval,
        "max_positions": args.max_positions,
        "signals": len(every),
        "rejected_book_full": rejected,
        "system": args.system,
        "first_entry": first.isoformat(),
        "last_exit": last.isoformat(),
        "years": years,
        "instruments": len({row["ticker"] for row in rows}) if "ticker" in rows[0] else None,
        "overall": slice_stats(rows),
        "long": slice_stats([r for r in rows if r["direction"] > 0]),
        "short": slice_stats([r for r in rows if r["direction"] < 0]),
        "trades_per_week": len(rows) / weeks,
        "exit_reasons": dict(collections.Counter(r["exit_reason"] for r in rows)),
        "units_per_trade": dict(collections.Counter(r["units"] for r in rows)),
    }
    longest_win, longest_loss = streaks(rows)
    result["longest_win_streak"] = longest_win
    result["longest_loss_streak"] = longest_loss

    peak_units, mean_units, _ = concurrency(rows)
    result["peak_concurrent_units"] = peak_units
    result["mean_concurrent_units"] = mean_units

    curves = {}
    for label, fraction in (("1x", 0.01), ("2x", 0.02)):
        days, curve, drawdowns = equity_curve(rows, fraction)
        final = curve[-1] if curve else 1.0
        ruined = final <= 0.0 or min(drawdowns, default=0.0) <= -0.999
        curves[label] = {
            "risk_fraction": fraction,
            "final_multiple": final,
            "from_1000": 1000.0 * final,
            "ruined": ruined,
            "cagr": (final ** (1 / years) - 1) if years > 0 and final > 0 else None,
            "max_drawdown": min(drawdowns) if drawdowns else 0.0,
            "days": days,
            "equity": curve,
            "drawdown": drawdowns,
        }
    result["curves"] = curves

    yearly = collections.defaultdict(float)
    yearly_n = collections.Counter()
    for row in rows:
        yearly[row["exit_timestamp"][:4]] += row["net_r"]
        yearly_n[row["exit_timestamp"][:4]] += 1
    result["yearly_r"] = {k: yearly[k] for k in sorted(yearly)}
    result["yearly_trades"] = {k: yearly_n[k] for k in sorted(yearly_n)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    overall = result["overall"]
    print(f"{args.system} @ {args.interval}: {overall['trades']} trades over {years:.1f}y")
    print(f"  win rate {overall['win_rate']:.1%}  PF {overall['profit_factor']:.2f}  "
          f"total {overall['total_r']:+.0f}R")
    print(f"  trades/week {result['trades_per_week']:.2f}   "
          f"peak concurrent units {peak_units}  "
          f"(skipped {rejected} of {len(every)} signals, book full)")
    for label, data in curves.items():
        cagr = f"{data['cagr']:.1%}" if data["cagr"] is not None else "n/a"
        flag = "  RUINED" if data["ruined"] else ""
        print(f"  {label}: $1000 -> ${data['from_1000']:,.0f}  "
              f"CAGR {cagr}  maxDD {data['max_drawdown']:.1%}{flag}")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
