"""Download earnings and valuation history into SQLite.

Four sources, none of which covers the whole picture on its own:

- **Shiller** (`shiller_monthly`) -- S&P 500 price, earnings, dividends, CPI and
  CAPE, monthly, back to 1871. The long valuation history, updated by hand and
  usually a year or two behind.
- **SEC XBRL frames** (`sec_facts`) -- as-filed company fundamentals, every
  filer at once. XBRL was phased in over 2009-2011, so there is no company
  fundamental history here before 2009 at any price.
- **Yahoo** (`index_prices`) -- daily index closes back to 1927 and current,
  which is what bridges Shiller's stale tail. FRED's `SP500` cannot do this: it
  is under a rolling ten-year licence window.
- **Alpha Vantage** (`analyst_earnings`) -- reported-versus-expected EPS per
  company. Needs `ALPHAVANTAGE_API_KEY`; skipped with a note when absent.

Corporate profits (BEA) are FRED series and live in the macro database instead:
    PYTHONPATH=src python scripts/download_macro_history.py --groups corporate_profits

Usage:
    PYTHONPATH=src python scripts/download_earnings_history.py --db data/equity/equity.db
"""

from __future__ import annotations

import argparse
import sys
import traceback
import uuid
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.config import get_alphavantage_api_key, get_sec_user_agent  # noqa: E402
from savi_uz.equity_catalog import (  # noqa: E402
    DEFAULT_ESTIMATE_TICKERS,
    INDEX_TICKERS,
    SEC_CONCEPTS,
    SEC_FIRST_YEAR,
    SEC_SPARSE_FRAME_FACTS,
    quarters,
)
from savi_uz.equity_sources import (  # noqa: E402
    AlphaVantageEarningsClient,
    SecFramesClient,
    ShillerClient,
    SourceError,
    YahooIndexClient,
    utc_now_iso,
)
from savi_uz.equity_store import EquityStore  # noqa: E402

DEFAULT_START_YEAR = 2000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=Path("data/equity/equity.db"), help="SQLite database path")
    parser.add_argument("--csv-dir", type=Path, default=None, help="also export every table to CSV here")
    parser.add_argument(
        "--start-year", type=int, default=DEFAULT_START_YEAR,
        help=f"earliest year for SEC and index data (default: {DEFAULT_START_YEAR}); "
             "Shiller is always stored in full, its history is the point of it",
    )
    parser.add_argument("--end-year", type=int, default=date.today().year, help="latest year (default: this year)")
    parser.add_argument(
        "--tickers", nargs="*", default=list(DEFAULT_ESTIMATE_TICKERS),
        help="symbols for Alpha Vantage estimates (its free tier is ~25 calls a day)",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/equity"), help="download cache")
    parser.add_argument("--skip-shiller", action="store_true", help="skip the Shiller workbook")
    parser.add_argument("--skip-sec", action="store_true", help="skip SEC XBRL frames")
    parser.add_argument("--skip-index", action="store_true", help="skip Yahoo index history")
    parser.add_argument("--skip-estimates", action="store_true", help="skip Alpha Vantage estimates")
    parser.add_argument(
        "--estimates-per-minute", type=int, default=5,
        help="Alpha Vantage request rate for estimates (default 5, the free-tier "
             "allowance); raise it to match a premium plan's limit",
    )
    return parser.parse_args(argv)


def download_shiller(store: EquityStore, run_id: str) -> int:
    client = ShillerClient()
    print("[shiller] downloading ie_data.xls from every mirror, keeping the freshest ...")
    try:
        url, rows = client.fetch()
    except SourceError as exc:
        store.log(run_id, utc_now_iso(), "SHILLER", "ie_data.xls", 0, "error", str(exc))
        print(f"[shiller] FAILED {exc}")
        return 0

    written = store.write_shiller(rows, url)
    store.log(run_id, utc_now_iso(), "SHILLER", url, written, "ok",
              f"{rows[0].obs_date} to {rows[-1].obs_date}")
    print(f"[shiller] {written:,} months, {rows[0].obs_date} to {rows[-1].obs_date}")

    last_price, last_earnings = store.shiller_earnings_gap()
    if last_price and last_earnings and last_price != last_earnings:
        print(f"[shiller] price runs to {last_price} but earnings only to {last_earnings}")
    return written


def download_sec(store: EquityStore, args: argparse.Namespace, run_id: str) -> tuple[int, int]:
    user_agent = get_sec_user_agent()
    client = SecFramesClient(user_agent)
    first_year = max(args.start_year, SEC_FIRST_YEAR)
    if args.start_year < SEC_FIRST_YEAR:
        print(f"[sec] XBRL begins {SEC_FIRST_YEAR}; {args.start_year}-{SEC_FIRST_YEAR - 1} has no filings to fetch")

    periods = quarters(first_year, args.end_year)
    total = failed = 0
    print(f"[sec] {len(SEC_CONCEPTS)} concepts over {len(periods)} quarters as {user_agent!r}")

    for concept in SEC_CONCEPTS:
        concept_rows = sparse = 0
        for period in periods:
            try:
                facts = client.fetch_frame(concept, period)
            except SourceError as exc:
                failed += 1
                store.log(run_id, utc_now_iso(), "SEC", f"{concept.tag}/{period}", 0, "error", str(exc))
                continue
            if not facts:
                continue
            written = store.write_sec_facts(facts)
            store.write_sec_frame(concept.tag, concept.frame(period), facts[0].unit, written, utc_now_iso())
            concept_rows += written
            if written < SEC_SPARSE_FRAME_FACTS:
                sparse += 1
        total += concept_rows
        note = f", {sparse} sparse frame(s)" if sparse else ""
        store.log(run_id, utc_now_iso(), "SEC", concept.tag, concept_rows, "ok", note.strip(", "))
        print(f"[sec] {concept.tag:<52} {concept_rows:>8,} facts{note}")
    return total, failed


def download_index(store: EquityStore, args: argparse.Namespace, run_id: str) -> int:
    client = YahooIndexClient(cache_dir=args.cache_dir)
    start = date(args.start_year, 1, 1)
    total = 0
    for ticker, label in INDEX_TICKERS:
        try:
            bars = client.fetch(ticker, start)
        except SourceError as exc:
            store.log(run_id, utc_now_iso(), "YAHOO", ticker, 0, "error", str(exc))
            print(f"[index] {ticker:<10} FAILED {exc}")
            continue
        written = store.write_index_prices(bars)
        total += written
        span = f"{bars[0].obs_date} to {bars[-1].obs_date}" if bars else "no data"
        store.log(run_id, utc_now_iso(), "YAHOO", ticker, written, "ok", span)
        print(f"[index] {ticker:<10} {written:>7,} bars  {span}  ({label})")
    return total


def download_estimates(store: EquityStore, args: argparse.Namespace, run_id: str) -> int:
    try:
        api_key = get_alphavantage_api_key()
    except ValueError:
        print("[estimates] skipped: no ALPHAVANTAGE_API_KEY in .env "
              "(free key at https://www.alphavantage.co/support/#api-key)")
        store.log(run_id, utc_now_iso(), "ALPHAVANTAGE", "estimates", 0, "skipped", "no api key")
        return 0

    client = AlphaVantageEarningsClient(
        api_key, max_per_minute=args.estimates_per_minute)
    total = 0
    for ticker in args.tickers:
        try:
            rows = client.fetch_earnings(ticker)
        except SourceError as exc:
            store.log(run_id, utc_now_iso(), "ALPHAVANTAGE", ticker, 0, "error", str(exc))
            print(f"[estimates] {ticker:<8} FAILED {exc}")
            # A quota message means every later ticker fails the same way.
            if "rate limit" in str(exc).lower() or "premium" in str(exc).lower():
                print("[estimates] stopping: daily quota reached")
                break
            continue
        written = store.write_analyst_earnings(rows)
        total += written
        store.log(run_id, utc_now_iso(), "ALPHAVANTAGE", ticker, written, "ok")
        print(f"[estimates] {ticker:<8} {written:>4} quarters")
    return total


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.start_year > args.end_year:
        print(f"error: --start-year {args.start_year} is after --end-year {args.end_year}")
        return 2

    run_id = uuid.uuid4().hex[:12]
    print(f"run {run_id}: {args.start_year}-{args.end_year}, database {args.db}")

    failures = 0
    with EquityStore(args.db) as store:
        if not args.skip_shiller:
            download_shiller(store, run_id)
        if not args.skip_index:
            download_index(store, args, run_id)
        if not args.skip_sec:
            _, failures = download_sec(store, args, run_id)
        if not args.skip_estimates:
            download_estimates(store, args, run_id)

        print("\ncoverage")
        for key, first, last, rows in store.coverage():
            print(f"  {key:<12} {str(first):<12} to {str(last):<12} {rows:>9,} rows")

        print("\ntables")
        for table, count in store.table_counts().items():
            print(f"  {table:<20}{count:>10,} rows")

        if args.csv_dir:
            exported = store.export_csv(args.csv_dir)
            print(f"\ncsv export -> {args.csv_dir} "
                  f"({sum(exported.values()):,} rows across {len(exported)} files)")

    print(f"\ndatabase: {args.db}")
    if failures:
        print(f"{failures} SEC frame(s) failed; see the fetch_log table")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        traceback.print_exc(limit=0)
        raise SystemExit(130)
