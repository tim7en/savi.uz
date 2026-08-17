"""Build a historic dataset from the FactSet Earnings Insight weekly PDF.

The report is the free market's reference for *forward* S&P 500 earnings
expectations -- forward 12-month P/E, blended growth, surprise rates -- and it
has no API. This walks every Friday in range, resolves the week's PDF, reads
page 1 and stores the numbers.

Two limits are properties of the source, not the script:

- **The archive begins 2017-02-03.** Nothing earlier is hosted, so a 2000 start
  silently becomes 2017.
- **Not every week has an edition.** FactSet skips holidays and quiet stretches
  between reporting seasons; roughly 40 of 52 weeks a year are published.

Page-1 text is stored with the parsed numbers, so `--reparse` replays improved
patterns over the whole history without downloading anything.

Usage:
    PYTHONPATH=src python scripts/download_factset_insight.py --db data/equity/equity.db
    PYTHONPATH=src python scripts/download_factset_insight.py --reparse
"""

from __future__ import annotations

import argparse
import sys
import traceback
import uuid
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.equity_sources import utc_now_iso  # noqa: E402
from savi_uz.equity_store import EquityStore  # noqa: E402
from savi_uz.factset_catalog import ARCHIVE_FIRST_DATE, CORE_FIELDS, fridays  # noqa: E402
from savi_uz.factset_sources import FactSetClient, FactSetError, parse_key_metrics  # noqa: E402

DEFAULT_START_YEAR = 2017


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=Path("data/equity/equity.db"), help="SQLite database path")
    parser.add_argument("--csv-dir", type=Path, default=None, help="also export every table to CSV here")
    parser.add_argument(
        "--start-year", type=int, default=DEFAULT_START_YEAR,
        help=f"earliest year (default: {DEFAULT_START_YEAR}; the archive itself starts {ARCHIVE_FIRST_DATE})",
    )
    parser.add_argument("--end-year", type=int, default=date.today().year, help="latest year (default: this year)")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/factset"), help="where PDFs are cached")
    parser.add_argument("--refresh", action="store_true", help="refetch PDFs even if cached")
    parser.add_argument(
        "--reparse", action="store_true",
        help="re-run the patterns over stored page text; no downloads",
    )
    return parser.parse_args(argv)


def reparse(store: EquityStore, run_id: str) -> int:
    """Replay the current patterns over every stored page-1 text."""
    stored = store.factset_page_texts()
    if not stored:
        print("[factset] nothing stored to re-parse")
        return 0

    improved = 0
    for report_date, source_url, page_text in stored:
        metrics = parse_key_metrics(page_text, date.fromisoformat(report_date), source_url)
        store.write_factset_report(metrics, utc_now_iso())
        if not metrics.missing_core:
            improved += 1
    store.log(run_id, utc_now_iso(), "FACTSET", "reparse", len(stored), "ok",
              f"{improved} editions with every core field")
    print(f"[factset] re-parsed {len(stored)} stored editions, "
          f"{improved} with every core field")
    return len(stored)


def download(store: EquityStore, args: argparse.Namespace, run_id: str) -> tuple[int, int, int]:
    client = FactSetClient(cache_dir=args.cache_dir, refresh=args.refresh)
    start = date(args.start_year, 1, 1)
    end = min(date(args.end_year, 12, 31), date.today())
    weeks = fridays(start, end)
    if args.start_year < ARCHIVE_FIRST_DATE.year:
        print(f"[factset] archive begins {ARCHIVE_FIRST_DATE}; "
              f"{args.start_year}-{ARCHIVE_FIRST_DATE.year - 1} has nothing to fetch")

    print(f"[factset] {len(weeks)} candidate weeks from {weeks[0]} to {weeks[-1]}" if weeks
          else "[factset] no weeks in range")

    stored = unpublished = failed = 0
    for index, friday in enumerate(weeks, start=1):
        try:
            metrics = client.fetch_week(friday)
        except FactSetError as exc:
            failed += 1
            store.log(run_id, utc_now_iso(), "FACTSET", friday.isoformat(), 0, "error", str(exc))
            print(f"[factset {index:>3}/{len(weeks)}] {friday} FAILED {exc}")
            continue

        if metrics is None:
            unpublished += 1
            continue

        store.write_factset_report(metrics, utc_now_iso())
        stored += 1
        note = f" missing core: {', '.join(metrics.missing_core)}" if metrics.missing_core else ""
        print(
            f"[factset {index:>3}/{len(weeks)}] {metrics.report_date} "
            f"{str(metrics.values.get('quarter', '?')):<8} "
            f"P/E {str(metrics.values.get('forward_12m_pe', '-')):<5} "
            f"growth {str(metrics.values.get('blended_earnings_growth', '-')):>6}%"
            f"{note}"
        )
    return stored, unpublished, failed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.start_year > args.end_year:
        print(f"error: --start-year {args.start_year} is after --end-year {args.end_year}")
        return 2

    run_id = uuid.uuid4().hex[:12]
    failed = 0
    with EquityStore(args.db) as store:
        if args.reparse:
            reparse(store, run_id)
        else:
            print(f"run {run_id}: FactSet Earnings Insight {args.start_year}-{args.end_year}, "
                  f"database {args.db}")
            stored, unpublished, failed = download(store, args, run_id)
            print(f"\n{stored} editions stored, {unpublished} weeks with no edition, {failed} failed")

        coverage = store.factset_field_coverage()
        if coverage:
            total = store.table_counts()["factset_reports"]
            print(f"\nfield coverage across {total} editions")
            for name, present in coverage:
                flag = "  core" if name in CORE_FIELDS else ""
                print(f"  {name:<36}{present:>5}/{total}{flag}")

        print("\ncoverage")
        for key, first, last, rows in store.coverage():
            print(f"  {key:<12} {str(first):<12} to {str(last):<12} {rows:>9,} rows")

        if args.csv_dir:
            exported = store.export_csv(args.csv_dir)
            print(f"\ncsv export -> {args.csv_dir} "
                  f"({sum(exported.values()):,} rows across {len(exported)} files)")

    print(f"\ndatabase: {args.db}")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        traceback.print_exc(limit=0)
        raise SystemExit(130)
