"""Map every Binance trad-FI contract to a tradable US-listed instrument.

133 of the 163 contracts are US equities already and map to themselves. The rest
are Hong Kong and Korea listings, commodity futures and two pre-IPO names, and
this scores candidate US proxies for each against the real underlying rather
than assuming an ADR or a country ETF is good enough.

Every pair is measured three ways, because a single correlation hides the two
things that matter most:

- **Daily, over lags -1/0/+1.** Asian sessions close hours before the US, so a
  genuine relationship shows up as the US proxy leading by a day. Reading only
  the same-day number understates every HK and KR pair.
- **Weekly.** Friday-to-Friday returns mostly remove the session offset, so this
  is the headline number and what the verdict is based on.
- **SPY-neutral weekly.** Correlation of residuals after market beta is removed.
  This is what separates a proxy that tracks the name from two things that both
  follow the US market -- the distinction that decides whether a country ETF is
  really standing in for a single stock.

Usage:
    PYTHONPATH=src python scripts/build_us_proxy_map.py --outdir out/tradfi
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from savi_uz.proxy_tracking import Tracking, measure, rank_candidates  # noqa: E402
from savi_uz.us_proxy_map import (  # noqa: E402
    MARKET_FACTOR,
    candidates_for,
    proxy_tickers,
)

DEFAULT_LOOKBACK_DAYS = 730


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--universe", type=Path, default=Path("out/tradfi/universe.csv"),
                        help="universe.csv produced by build_tradfi_risk_map.py")
    parser.add_argument("--closes", type=Path, default=Path(".cache/tradfi/yahoo_closes.csv"),
                        help="cached underlying closes from the risk-map run")
    parser.add_argument("--outdir", type=Path, default=Path("out/tradfi"), help="where to write the map")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/tradfi"), help="proxy download cache")
    parser.add_argument("--refresh", action="store_true", help="refetch proxy prices even if cached")
    parser.add_argument("--days", type=int, default=DEFAULT_LOOKBACK_DAYS, help="lookback window in days")
    return parser.parse_args(argv)


def load_universe(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"error: {path} not found; run build_tradfi_risk_map.py first")
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_underlying_closes(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"error: {path} not found; run build_tradfi_risk_map.py first")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def download_proxies(tickers: tuple[str, ...], start: date, cache: Path, refresh: bool) -> pd.DataFrame:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "us_proxy_closes.csv"
    if path.is_file() and not refresh:
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
        if set(tickers).issubset(cached.columns):
            print(f"[proxy] using cached closes for {len(tickers)} tickers")
            return cached

    import yfinance as yf

    print(f"[proxy] downloading {len(tickers)} US proxy tickers from {start} ...")
    frame = yf.download(list(tickers), start=start.isoformat(), interval="1d",
                        auto_adjust=False, progress=False, threads=True)
    closes = frame["Close"] if "Close" in frame.columns.get_level_values(0) else frame
    closes = closes.dropna(how="all")
    closes.to_csv(path)
    return closes


def score_instrument(
    row: dict[str, str], underlying: pd.DataFrame, proxies: pd.DataFrame, market: pd.Series
) -> list[Tracking]:
    base = row["base_asset"]
    ticker = row["yahoo_ticker"]
    if ticker not in underlying.columns:
        return []
    series = underlying[ticker].dropna()

    results = []
    for candidate in candidates_for(base):
        if candidate.ticker not in proxies.columns:
            continue
        results.append(
            measure(
                base_asset=base,
                underlying_name=ticker,
                underlying_prices=series,
                proxy_name=candidate.ticker,
                proxy_prices=proxies[candidate.ticker].dropna(),
                market_prices=market,
                kind=candidate.kind,
                rationale=candidate.rationale,
            )
        )
    return rank_candidates(results)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: object, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_report(
    direct: list[dict[str, str]], scored: dict[str, list[Tracking]], no_proxy: list[str], days: int
) -> str:
    lines = [
        "# US proxy map for Binance trad-FI contracts",
        "",
        f"Measured over the last {days} calendar days of daily closes.",
        "",
        f"- Contracts already listed in the US, mapping to themselves: **{len(direct)}**",
        f"- Non-US contracts with at least one scored proxy: **{len(scored)}**",
        f"- Contracts with no tradable proxy: **{len(no_proxy)}**",
        "",
        "`weekly` is the headline correlation of Friday-to-Friday log returns.",
        "`resid` is the same correlation after SPY beta is removed from both sides:",
        "it is what distinguishes tracking the name from tracking the market.",
        "`lag` is the daily lead/lag that maximises correlation; **+1 means the US",
        "proxy moved first**, which is what the session offset predicts for Asia.",
        "",
        "## Best proxy per contract",
        "",
        "| Contract | Underlying | Proxy | Kind | weekly | resid | beta | lag | Verdict |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for base, results in sorted(scored.items()):
        best = results[0]
        lines.append(
            f"| {base} | {best.underlying} | {best.proxy} | {best.kind} | "
            f"{_fmt(best.weekly_corr)} | {_fmt(best.residual_corr)} | {_fmt(best.beta)} | "
            f"{best.daily_lag:+d} | {best.verdict} |"
        )

    lines += ["", "## Every candidate scored", "",
              "| Contract | Proxy | Kind | weekly | daily(best) | same-day | lag | resid | beta | R2 | Verdict | Note |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for base, results in sorted(scored.items()):
        for result in results:
            lines.append(
                f"| {base} | {result.proxy} | {result.kind} | {_fmt(result.weekly_corr)} | "
                f"{_fmt(result.daily_corr)} | {_fmt(result.daily_corr_same_day)} | "
                f"{result.daily_lag:+d} | {_fmt(result.residual_corr)} | {_fmt(result.beta)} | "
                f"{_fmt(result.r_squared)} | {result.verdict} | {result.note} |"
            )

    if no_proxy:
        lines += ["", "## No tradable US proxy", "",
                  "These have no listing to track, so a US-only strategy cannot express them:",
                  ""]
        lines += [f"- `{base}`" for base in sorted(no_proxy)]

    lines += [
        "",
        "## Reading the verdicts",
        "",
        "- **direct** -- the contract is a US equity; trade the same symbol.",
        "- **strong** -- weekly correlation at or above 0.80 with specific risk left",
        "  after SPY is removed. Usable as a stand-in for the name.",
        "- **usable** -- 0.55 to 0.80. Carries most of the move but expect tracking error.",
        "- **market-beta** -- the headline correlation is real but nearly all of it is",
        "  SPY. The proxy follows the market, not the name; a strategy trading the",
        "  name's own risk is not hedged by it.",
        "- **weak** / **poor** -- below 0.55 / 0.30. Not a substitute.",
        "- **insufficient** -- too little overlapping history to judge.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    universe = load_universe(args.universe)
    underlying = load_underlying_closes(args.closes)
    start = date.today() - timedelta(days=args.days)
    proxies = download_proxies(proxy_tickers(), start, args.cache_dir, args.refresh)

    if MARKET_FACTOR not in proxies.columns:
        raise SystemExit(f"error: market factor {MARKET_FACTOR} missing from proxy download")
    market = proxies[MARKET_FACTOR].dropna()

    direct: list[dict[str, str]] = []
    scored: dict[str, list[Tracking]] = {}
    no_proxy: list[str] = []
    rows: list[dict[str, object]] = []

    for row in universe:
        base, region, ticker = row["base_asset"], row["region"], row["yahoo_ticker"]
        if region == "US" and ticker:
            direct.append(row)
            rows.append(
                {
                    "base_asset": base, "binance_symbol": row["binance_symbol"], "region": region,
                    "underlying": ticker, "us_proxy": ticker, "kind": "direct",
                    "weekly_corr": 1.0, "residual_corr": 1.0, "beta": 1.0, "r_squared": 1.0,
                    "daily_corr": 1.0, "daily_lag": 0, "overlap_weeks": "",
                    "verdict": "direct", "note": "already a US listing", "rationale": "same symbol",
                }
            )
            continue

        results = score_instrument(row, underlying, proxies, market)
        if not results:
            no_proxy.append(base)
            # Distinguish "nothing is listed anywhere" from "this is a US name
            # whose Yahoo ticker the universe run could not resolve" -- the
            # second is a mapping gap, not an untradable contract.
            if region == "US" and not ticker:
                note = (f"US contract but no Yahoo ticker resolved "
                        f"(universe mapping_status={row.get('mapping_status', '?')})")
            elif not candidates_for(base):
                note = "no public listing to track"
            else:
                note = "candidates defined but no overlapping price history"
            rows.append(
                {
                    "base_asset": base, "binance_symbol": row["binance_symbol"], "region": region,
                    "underlying": ticker, "us_proxy": "", "kind": "", "weekly_corr": "",
                    "residual_corr": "", "beta": "", "r_squared": "", "daily_corr": "",
                    "daily_lag": "", "overlap_weeks": "", "verdict": "none",
                    "note": note, "rationale": "",
                }
            )
            continue

        scored[base] = results
        for result in results:
            rows.append(
                {
                    "base_asset": base, "binance_symbol": row["binance_symbol"], "region": region,
                    "underlying": result.underlying, "us_proxy": result.proxy, "kind": result.kind,
                    "weekly_corr": round(result.weekly_corr, 4) if pd.notna(result.weekly_corr) else "",
                    "residual_corr": round(result.residual_corr, 4) if pd.notna(result.residual_corr) else "",
                    "beta": round(result.beta, 4) if pd.notna(result.beta) else "",
                    "r_squared": round(result.r_squared, 4) if pd.notna(result.r_squared) else "",
                    "daily_corr": round(result.daily_corr, 4) if pd.notna(result.daily_corr) else "",
                    "daily_lag": result.daily_lag, "overlap_weeks": result.overlap_weeks,
                    "verdict": result.verdict, "note": result.note, "rationale": result.rationale,
                }
            )

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "us_proxy_map.csv", rows)
    report = build_report(direct, scored, no_proxy, args.days)
    (args.outdir / "us_proxy_map.md").write_text(report, encoding="utf-8")

    print(f"\n{len(direct)} direct US listings, {len(scored)} proxied, {len(no_proxy)} with no proxy")
    print("\nbest proxy per non-US contract")
    for base, results in sorted(scored.items()):
        best = results[0]
        print(
            f"  {base:<16} -> {best.proxy:<6} {best.kind:<10} "
            f"weekly {_fmt(best.weekly_corr)}  resid {_fmt(best.residual_corr)}  "
            f"lag {best.daily_lag:+d}  {best.verdict}"
        )
    if no_proxy:
        print(f"\nno tradable proxy: {', '.join(sorted(no_proxy))}")
    print(f"\nwrote {args.outdir / 'us_proxy_map.csv'} and {args.outdir / 'us_proxy_map.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        traceback.print_exc(limit=0)
        raise SystemExit(130)
