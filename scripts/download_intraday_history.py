"""Download hourly bars from Tiingo for the theme leaders, without tripping the quota.

Symbols default to the US-tradable picks in `out/tradfi/theme_leaders.csv` -- the
two or three names that carry each theme -- which is 46 tickers rather than the
whole 163-contract universe.

Quota discipline, because a block would cost more than the data:

- Pacing defaults to 45 requests/hour, under the free tier's documented 50.
  Raise it with `--requests-per-hour` on a paid plan.
- `--max-requests` caps a single run so one invocation cannot eat the daily
  allowance. The default leaves room to run twice more the same day.
- Every response is cached on disk and every completed (ticker, year) window is
  recorded, so a resumed run asks only for what is missing and costs nothing for
  what it already has.
- A 429 stops the run immediately and reports where it got to. It never retries
  into a block.

Two source limits are handled rather than papered over:

- Tiingo caps a response at 10,000 rows *silently*, returning the recent end of
  the range. Asking for 2017-2026 in one call yields 2020 onward and looks like
  history simply starts later. Requests are chunked by year (~1,550 hourly bars)
  and any response that comes back at the cap is flagged.
- IEX intraday covers exchange-listed tickers only. The OTC ADRs here (`TCEHY`,
  `XIACY`, `MPNGY`, `PMRTY`, all `PINK`) have no intraday at any depth and fall
  back to daily bars, which do go back years.

Usage:
    PYTHONPATH=src python scripts/download_intraday_history.py --db data/intraday/bars.db
    PYTHONPATH=src python scripts/download_intraday_history.py --plan
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
import uuid
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.config import get_tiingo_api_key  # noqa: E402
from savi_uz.intraday_store import IntradayStore  # noqa: E402
from savi_uz.tiingo_sources import (  # noqa: E402
    DEFAULT_REQUESTS_PER_HOUR,
    DEFAULT_YEARS_PER_REQUEST,
    FREE_TIER_REQUESTS_PER_DAY,
    SUPPORTED_FREQUENCIES,
    TiingoClient,
    TiingoError,
    TiingoRateLimitError,
    max_safe_years,
    utc_now_iso,
    year_windows,
)

DEFAULT_START_YEAR = 2017
#: Leaves room for two more runs inside the free tier's daily allowance.
DEFAULT_MAX_REQUESTS = FREE_TIER_REQUESTS_PER_DAY // 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--leaders", type=Path, default=Path("out/tradfi/theme_leaders.csv"))
    parser.add_argument("--tickers", nargs="*", default=None, help="override the symbol list")
    parser.add_argument("--frequency", default="1hour", choices=SUPPORTED_FREQUENCIES)
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=date.today().year)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/tiingo"))
    parser.add_argument("--csv-dir", type=Path, default=None, help="also export tables to CSV")
    parser.add_argument(
        "--requests-per-hour", type=int, default=DEFAULT_REQUESTS_PER_HOUR,
        help=f"pacing (default {DEFAULT_REQUESTS_PER_HOUR}; free tier allows 50)",
    )
    parser.add_argument(
        "--max-requests", type=int, default=DEFAULT_MAX_REQUESTS,
        help=f"stop after this many live requests in one run (default {DEFAULT_MAX_REQUESTS})",
    )
    parser.add_argument(
        "--years-per-request", type=int, default=DEFAULT_YEARS_PER_REQUEST,
        help=f"calendar years per request (default {DEFAULT_YEARS_PER_REQUEST}); larger means "
             "fewer requests but risks the 10,000-row cap",
    )
    parser.add_argument("--refresh", action="store_true", help="ignore the cache and refetch")
    parser.add_argument("--plan", action="store_true",
                        help="print the request plan and quota estimate, then exit")
    parser.add_argument("--no-daily-fallback", action="store_true",
                        help="skip OTC tickers instead of fetching daily bars for them")
    return parser.parse_args(argv)


def load_leader_symbols(path: Path) -> dict[str, list[str]]:
    """US-tradable symbol -> the themes it represents."""
    if not path.is_file():
        raise SystemExit(f"error: {path} not found; run select_theme_leaders.py first")
    themes: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            symbol = (row.get("us_tradable") or "").strip()
            if symbol:
                themes[symbol].append(row["theme"])
    return dict(themes)


def print_plan(
    symbols: dict[str, list[str]], args: argparse.Namespace, known: dict[str, tuple[str, int]],
    done: set[tuple[str, int]],
) -> None:
    years = list(range(args.start_year, args.end_year + 1))
    chunks = year_windows(
        date(args.start_year, 1, 1), date(args.end_year, 12, 31), args.years_per_request
    )
    outstanding = 0
    metadata_needed = 0
    for symbol in symbols:
        if symbol not in known:
            metadata_needed += 1
        for first, last in chunks:
            span = range(first.year, last.year + 1)
            if any((symbol, year) not in done for year in span):
                outstanding += 1

    total = metadata_needed + outstanding
    hours = total / max(args.requests_per_hour, 1)
    print(f"symbols                : {len(symbols)}")
    print(f"years                  : {years[0]}-{years[-1]} ({len(years)})")
    safe = max_safe_years(args.frequency)
    print(f"years per request      : {args.years_per_request} "
          f"({len(chunks)} chunks; up to {safe} is safe for {args.frequency})")
    print(f"metadata requests due  : {metadata_needed}")
    print(f"bar-window requests due: {outstanding} (upper bound; pre-listing years are skipped)")
    print(f"total requests due     : {total}")
    print(f"pacing                 : {args.requests_per_hour}/hour -> ~{hours:.1f} hours")
    print(f"per-run cap            : {args.max_requests}")
    if total > args.max_requests:
        runs = -(-total // args.max_requests)
        print(f"                         needs ~{runs} runs at this cap; each resumes where it stopped")


def download_symbol(
    client: TiingoClient, store: IntradayStore, symbol: str, themes: list[str],
    args: argparse.Namespace, done: set[tuple[str, int]], run_id: str,
) -> tuple[int, int]:
    """Fetch every missing window for one symbol. Returns (bars, windows)."""
    meta = client.fetch_metadata(symbol)
    store.upsert_symbol(meta, ", ".join(sorted(set(themes))), utc_now_iso())

    frequency = args.frequency
    use_daily = not meta.has_intraday
    if use_daily:
        if args.no_daily_fallback:
            print(f"[{symbol:<6}] skipped: {meta.exchange} has no intraday feed")
            store.log(run_id, utc_now_iso(), "TIINGO", symbol, 0, "skipped",
                      f"{meta.exchange} has no IEX intraday")
            return 0, 0
        frequency = "daily"

    # Never ask for years before the ticker existed; that is most of the budget
    # that would otherwise be spent on empty responses.
    listing_start = meta.start_date or date(args.start_year, 1, 1)
    start = max(date(args.start_year, 1, 1), listing_start)
    end = min(date(args.end_year, 12, 31), date.today())
    if start > end:
        print(f"[{symbol:<6}] no history in range (listed {listing_start})")
        return 0, 0

    total_bars = total_windows = 0
    for first, last in year_windows(start, end, args.years_per_request):
        span = range(first.year, last.year + 1)
        if all((symbol, year) in done for year in span) and not args.refresh:
            continue
        if client.budget_exhausted():
            raise TiingoError("request budget reached")

        if use_daily:
            bars, truncated = client.fetch_daily(symbol, first, last)
        else:
            bars, truncated = client.fetch_intraday(symbol, first, last, frequency)

        written = store.write_bars(bars)
        store.mark_window(
            symbol, frequency, first.year, bars, truncated, utc_now_iso(), last_year=last.year
        )
        total_bars += written
        total_windows += 1
        if truncated:
            print(f"[{symbol:<6}] {first.year}-{last.year} hit the {len(bars)}-row cap; "
                  "narrow --years-per-request and rerun with --refresh")
            store.log(run_id, utc_now_iso(), "TIINGO", f"{symbol}/{first.year}-{last.year}",
                      written, "truncated", "response hit the row cap")

    # One request buys the whole split/dividend history, without which the raw
    # intraday bars cannot be turned into a tradable return series.
    adjustments = client.fetch_adjustments(symbol, start, end)
    store.write_adjustments(adjustments)
    split_count = sum(1 for row in adjustments if row.is_split)

    label = f"{frequency}{' (daily fallback)' if use_daily else ''}"
    if split_count:
        label += f", {split_count} split(s)"
    print(f"[{symbol:<6}] {total_bars:>6,} bars over {total_windows:>2} window(s)  {label}"
          f"  listed {listing_start}")
    store.log(run_id, utc_now_iso(), "TIINGO", symbol, total_bars, "ok", label)
    return total_bars, total_windows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.start_year > args.end_year:
        print(f"error: --start-year {args.start_year} is after --end-year {args.end_year}")
        return 2

    if args.tickers:
        symbols = {ticker.upper(): ["(manual)"] for ticker in args.tickers}
    else:
        symbols = load_leader_symbols(args.leaders)

    with IntradayStore(args.db) as store:
        done = store.completed_windows(args.frequency) | store.completed_windows("daily")
        known = store.known_symbols()

        if args.plan:
            print_plan(symbols, args, known, done)
            return 0

        try:
            api_key = get_tiingo_api_key()
        except ValueError as exc:
            print(f"error: {exc}")
            return 2

        if args.requests_per_hour > 50:
            print(f"note: pacing at {args.requests_per_hour}/hour, above the free tier's 50 -- "
                  "this assumes a paid plan")

        run_id = uuid.uuid4().hex[:12]
        client = TiingoClient(
            api_key, cache_dir=args.cache_dir, requests_per_hour=args.requests_per_hour,
            max_requests=args.max_requests, refresh=args.refresh,
        )
        print(f"run {run_id}: {len(symbols)} symbols, {args.frequency}, "
              f"{args.start_year}-{args.end_year}, <={args.max_requests} requests "
              f"at {args.requests_per_hour}/hour\n")

        stopped = ""
        for symbol, themes in sorted(symbols.items()):
            try:
                download_symbol(client, store, symbol, themes, args, done, run_id)
            except TiingoRateLimitError as exc:
                stopped = f"quota exhausted at {symbol}: {exc}"
                store.log(run_id, utc_now_iso(), "TIINGO", symbol, 0, "rate-limited", str(exc))
                break
            except TiingoError as exc:
                if "budget" in str(exc):
                    stopped = f"per-run request cap reached at {symbol}"
                    break
                print(f"[{symbol:<6}] FAILED {exc}")
                store.log(run_id, utc_now_iso(), "TIINGO", symbol, 0, "error", str(exc))

        counts = store.table_counts()
        print(f"\nrequests made {client.requests_made}, served from cache {client.cache_hits}")
        for table, count in counts.items():
            print(f"  {table:<12}{count:>10,} rows")

        truncated = store.truncated_windows()
        if truncated:
            print(f"\n{len(truncated)} window(s) hit the row cap and lost their early part:")
            for ticker, frequency, year, rows in truncated[:10]:
                print(f"  {ticker} {frequency} {year}: {rows} rows")

        # Data quality is reported, never silently repaired: a backtest should
        # decide for itself what to do with a close outside its own bar range.
        violations = store.ohlc_violations()
        if violations:
            print("\nbars where open/close falls outside the bar's own high/low "
                  "(IEX resampling artefact, left as published):")
            for ticker, frequency, count in violations:
                print(f"  {ticker:<6} {frequency:<6} {count}")

        splits = store.splits()
        if splits:
            print(f"\n{len(splits)} split event(s) -- intraday bars are RAW, apply these "
                  "before computing returns:")
            for ticker, day, factor in splits[:12]:
                print(f"  {ticker:<6} {day}  {factor:g}:1")

        gaps = [row for row in store.missing_volume() if row[2]]
        if gaps:
            print("\nbars with no volume:")
            for ticker, frequency, missing, total in gaps:
                print(f"  {ticker:<6} {frequency:<6} {missing:,}/{total:,}")

        if args.csv_dir:
            exported = store.export_csv(args.csv_dir)
            print(f"\ncsv export -> {args.csv_dir} ({sum(exported.values()):,} rows)")

        if stopped:
            print(f"\nstopped: {stopped}")
            print("rerun the same command to continue; completed windows are skipped")
            return 1

    print(f"\ndatabase: {args.db}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        traceback.print_exc(limit=0)
        raise SystemExit(130)
