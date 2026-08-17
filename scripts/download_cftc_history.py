"""Download CFTC Commitments of Traders history into SQLite.

Pulls all six COT reports -- legacy, disaggregated and Traders in Financial
Futures, each futures-only and futures-plus-options -- from the CFTC's own
annual archives, back to 2000 by default.

Only the legacy report reaches 2000. The disaggregated and financial-trader
breakdowns were introduced in 2006 and no earlier history exists, so a run
starting in 2000 is legacy-only for its first six years. Pass ``--start-year
1986`` for the full legacy record.

No API key is needed; the archives are public files. The current year's ZIP is
rewritten weekly, so the incremental update is:

    PYTHONPATH=src python scripts/download_cftc_history.py --start-year 2026 --refresh

Usage:
    PYTHONPATH=src python scripts/download_cftc_history.py --db data/cftc/cot.db
"""

from __future__ import annotations

import argparse
import sys
import traceback
import uuid
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.cftc_catalog import (  # noqa: E402
    REPORT_KEYS,
    ReportSpec,
    archive_files,
    reports_for_keys,
)
from savi_uz.cftc_sources import (  # noqa: E402
    CftcArchiveClient,
    CftcDownloadError,
    utc_now_iso,
)
from savi_uz.cftc_store import CftcStore  # noqa: E402

DEFAULT_START_YEAR = 2000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=Path("data/cftc/cot.db"), help="SQLite database path")
    parser.add_argument("--csv-dir", type=Path, default=None, help="also export every table to CSV here")
    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_START_YEAR,
        help=f"earliest report year (default: {DEFAULT_START_YEAR}; 1986 for the full legacy record)",
    )
    parser.add_argument(
        "--end-year", type=int, default=date.today().year, help="latest report year (default: this year)"
    )
    parser.add_argument(
        "--reports", nargs="*", default=None, help=f"limit to these reports: {', '.join(REPORT_KEYS)}"
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/cftc"), help="where ZIPs are cached")
    parser.add_argument("--refresh", action="store_true", help="refetch archives even if cached")
    parser.add_argument("--timeout", type=float, default=180.0, help="per-request timeout in seconds")
    parser.add_argument("--list", action="store_true", help="print the archive plan and exit")
    return parser.parse_args(argv)


def download_report(
    store: CftcStore,
    client: CftcArchiveClient,
    spec: ReportSpec,
    start: date,
    end: date,
    run_id: str,
) -> tuple[int, int]:
    """Load every archive for one report. Returns (rows written, archives failed)."""
    archives = archive_files(spec, start.year, end.year)
    if not archives:
        print(f"[{spec.key}] nothing to fetch: history starts {spec.first_year}")
        return 0, 0

    written = failed = 0
    for index, archive in enumerate(archives, start=1):
        label = f"[{spec.key} {index:>2}/{len(archives)}] {archive.filename:<34}"
        try:
            chunk = client.load(spec, archive, start=start, end=end)
        except CftcDownloadError as exc:
            failed += 1
            store.log(run_id, utc_now_iso(), "CFTC", archive.filename, 0, "error", str(exc))
            print(f"{label} SKIPPED {exc}")
            continue

        rows = store.write_chunk(chunk)
        written += rows
        span = f"{chunk.first_date} to {chunk.last_date}" if rows else "no rows in range"
        if chunk.unparsed:
            span += f" ({chunk.unparsed} unreadable record(s))"
        store.log(run_id, utc_now_iso(), "CFTC", archive.filename, rows, "ok", span)
        print(f"{label} {rows:>7,} rows  {span}")

    store.upsert_report(spec, utc_now_iso())
    contracts = store.rebuild_contracts(spec)
    print(f"[{spec.key}] {written:,} rows, {contracts:,} contracts")
    return written, failed


def print_plan(specs: tuple[ReportSpec, ...], start_year: int, end_year: int) -> None:
    total = 0
    for spec in specs:
        archives = archive_files(spec, start_year, end_year)
        total += len(archives)
        covered = max(start_year, spec.first_year)
        print(f"{spec.key:<18} history from {spec.first_year}, fetching {covered}-{end_year}: "
              f"{len(archives)} archive(s)")
        if spec.notes:
            print(f"{'':<18} {spec.notes}")
    print(f"\n{total} archive(s) in total")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        specs = reports_for_keys(tuple(args.reports) if args.reports else None)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    if args.start_year > args.end_year:
        print(f"error: --start-year {args.start_year} is after --end-year {args.end_year}")
        return 2

    if args.list:
        print_plan(specs, args.start_year, args.end_year)
        return 0

    start = date(args.start_year, 1, 1)
    end = date(args.end_year, 12, 31)
    run_id = uuid.uuid4().hex[:12]
    print(f"run {run_id}: {len(specs)} COT report(s), {args.start_year}-{args.end_year}, database {args.db}")

    client = CftcArchiveClient(cache_dir=args.cache_dir, timeout=args.timeout, refresh=args.refresh)
    failures = 0
    with CftcStore(args.db) as store:
        for spec in specs:
            _, failed = download_report(store, client, spec, start, end, run_id)
            failures += failed

        print("\ncoverage")
        for key, first, last, rows, contracts in store.coverage():
            print(f"  {key:<18} {str(first):<12} to {str(last):<12} {rows:>9,} rows  {contracts:>5} contracts")

        print("\ntables")
        for table, count in store.table_counts().items():
            print(f"  {table:<22}{count:>10,} rows")

        if args.csv_dir:
            exported = store.export_csv(args.csv_dir)
            print(f"\ncsv export -> {args.csv_dir} "
                  f"({sum(exported.values()):,} rows across {len(exported)} files)")

    print(f"\ndatabase: {args.db}")
    if failures:
        print(f"{failures} archive(s) failed; see the fetch_log table")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        traceback.print_exc(limit=0)
        raise SystemExit(130)
