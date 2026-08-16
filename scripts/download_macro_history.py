"""Download the macro stack behind a rates view and store it with its vintages.

Covers the market-implied Fed path, the SEP dot plot, EFFR and other reference
rates, the Treasury curve, inflation, labour, credit spreads, and the Fed
balance sheet / RRP / TGA -- back to 2000 by default, or to each series' own
start with ``--start``.

Every value is stored three ways where the source supports it: the current
revision, the first print with the date it was released, and (for the SEP) the
full vintage matrix. That is what makes the history usable for backtests without
look-ahead.

Usage:
    PYTHONPATH=src python scripts/download_macro_history.py --db data/macro/macro.db
"""

from __future__ import annotations

import argparse
import sys
import traceback
import uuid
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.config import get_fred_api_key  # noqa: E402
from savi_uz.macro_catalog import (  # noqa: E402
    FED_PATH_HORIZON_MONTHS,
    GROUPS,
    NYFED_RATE_TYPES,
    SeriesSpec,
    VintagePolicy,
    catalog_for_groups,
)
from savi_uz.macro_sources import (  # noqa: E402
    FredClient,
    GswCurveClient,
    NyFedRatesClient,
    implied_policy_path,
    utc_now_iso,
)
from savi_uz.macro_store import MacroStore  # noqa: E402

DEFAULT_START = date(2000, 1, 1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=Path("data/macro/macro.db"), help="SQLite database path")
    parser.add_argument("--csv-dir", type=Path, default=None, help="also export every table to CSV here")
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        default=DEFAULT_START,
        help="earliest observation date (default: 2000-01-01; use 1900-01-01 for full history)",
    )
    parser.add_argument("--groups", nargs="*", default=None, help=f"limit to these groups: {', '.join(GROUPS)}")
    parser.add_argument(
        "--all-vintages",
        nargs="*",
        default=[],
        metavar="SERIES_ID",
        help="also pull the full revision matrix for these series (SEP series always get it)",
    )
    parser.add_argument("--skip-fred", action="store_true", help="skip FRED/ALFRED")
    parser.add_argument("--skip-nyfed", action="store_true", help="skip NY Fed reference rates")
    parser.add_argument("--skip-gsw", action="store_true", help="skip the fitted Treasury curve download")
    parser.add_argument("--skip-release-dates", action="store_true", help="skip the release calendars")
    parser.add_argument("--max-per-minute", type=int, default=100, help="FRED request cap (API allows 120)")
    return parser.parse_args(argv)


def download_fred(
    store: MacroStore,
    specs: tuple[SeriesSpec, ...],
    args: argparse.Namespace,
    run_id: str,
) -> tuple[int, int]:
    client = FredClient(get_fred_api_key(), max_per_minute=args.max_per_minute)
    forced_vintages = {series_id.upper() for series_id in args.all_vintages}
    release_ids: dict[int, str] = {}
    succeeded = failed = 0

    for index, spec in enumerate(specs, start=1):
        try:
            metadata = client.fetch_metadata(spec.series_id)
            release_id, release_name = client.fetch_release(spec.series_id)
            if release_id is not None:
                release_ids[release_id] = release_name

            store.upsert_series(
                {
                    "series_id": spec.series_id,
                    "source": "FRED",
                    "group_name": spec.group,
                    "label": spec.label,
                    "title": metadata.title,
                    "units": metadata.units,
                    "frequency": metadata.frequency,
                    "seasonal_adjustment": metadata.seasonal_adjustment,
                    "observation_start": metadata.observation_start,
                    "observation_end": metadata.observation_end,
                    "last_updated": metadata.last_updated,
                    "release_id": release_id,
                    "release_name": release_name,
                    "vintage_policy": spec.vintages.value,
                    "notes": metadata.notes,
                    "fetched_at": utc_now_iso(),
                }
            )

            observations = client.fetch_observations(spec.series_id, start=args.start)
            written = store.write_observations(spec.series_id, observations)

            wants_all = spec.vintages is VintagePolicy.ALL or spec.series_id.upper() in forced_vintages
            wants_first = spec.vintages is not VintagePolicy.LATEST or wants_all

            first_count = vintage_count = 0
            if wants_first:
                first_releases = client.fetch_first_releases(spec.series_id, start=args.start)
                first_count = store.write_first_releases(spec.series_id, first_releases)
            if wants_all:
                vintages = client.fetch_all_vintages(spec.series_id, start=args.start)
                vintage_count = store.write_vintages(spec.series_id, vintages)

            store.log(run_id, utc_now_iso(), "FRED", spec.series_id, written, "ok")
            succeeded += 1
            print(
                f"[fred {index:>3}/{len(specs)}] {spec.series_id:<16} {written:>6} obs"
                f"{f', {first_count} first prints' if first_count else ''}"
                f"{f', {vintage_count} vintages' if vintage_count else ''}"
            )
        except Exception as exc:
            failed += 1
            store.log(run_id, utc_now_iso(), "FRED", spec.series_id, 0, "error", f"{type(exc).__name__}: {exc}")
            print(f"[fred {index:>3}/{len(specs)}] {spec.series_id:<16} FAILED {type(exc).__name__}: {exc}")

    if release_ids and not args.skip_release_dates:
        for release_id, name in sorted(release_ids.items()):
            try:
                dates = client.fetch_release_dates(release_id, start=args.start)
                store.write_release_dates(release_id, dates)
                store.log(run_id, utc_now_iso(), "FRED", f"release:{release_id}", len(dates), "ok", name)
            except Exception as exc:
                store.log(run_id, utc_now_iso(), "FRED", f"release:{release_id}", 0, "error", str(exc))
        print(f"[fred] release calendars stored for {len(release_ids)} releases")

    return succeeded, failed


def download_nyfed(store: MacroStore, args: argparse.Namespace, run_id: str) -> int:
    client = NyFedRatesClient()
    total = 0
    for rate_type, secured in NYFED_RATE_TYPES:
        try:
            rates = client.fetch_rates(rate_type, secured, args.start, date.today())
            written = store.write_reference_rates(rates)
            revisions = sum(1 for rate in rates if rate.revision_indicator)
            total += written
            store.log(run_id, utc_now_iso(), "NYFED", rate_type.upper(), written, "ok",
                      f"{revisions} flagged revisions")
            print(f"[nyfed] {rate_type.upper():<6} {written:>6} rows, {revisions} flagged revisions")
        except Exception as exc:
            store.log(run_id, utc_now_iso(), "NYFED", rate_type.upper(), 0, "error", str(exc))
            print(f"[nyfed] {rate_type.upper():<6} FAILED {type(exc).__name__}: {exc}")
    return total


def download_gsw(store: MacroStore, args: argparse.Namespace, run_id: str) -> int:
    client = GswCurveClient()
    print("[gsw] downloading fitted Treasury curve (~17MB) ...")
    try:
        parameters, rates = client.fetch()
    except Exception as exc:
        store.log(run_id, utc_now_iso(), "GSW", "feds200628", 0, "error", str(exc))
        print(f"[gsw] FAILED {type(exc).__name__}: {exc}")
        return 0

    parameters = [fit for fit in parameters if fit.curve_date >= args.start]
    rates = [row for row in rates if row[0] >= args.start]

    store.write_gsw_params(parameters)
    store.write_gsw_rates(rates)
    path_rows = list(implied_policy_path(parameters, FED_PATH_HORIZON_MONTHS))
    store.write_fed_path(path_rows)
    store.log(run_id, utc_now_iso(), "GSW", "feds200628", len(parameters), "ok",
              f"{len(rates)} rate points, {len(path_rows)} path points")
    print(f"[gsw] {len(parameters)} curve fits, {len(rates)} rate points, {len(path_rows)} implied-path points")
    return len(parameters)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    specs = catalog_for_groups(tuple(args.groups) if args.groups else None)
    run_id = uuid.uuid4().hex[:12]

    print(f"run {run_id}: {len(specs)} FRED series from {args.start}, database {args.db}")
    if not args.skip_fred:
        try:
            get_fred_api_key()
        except ValueError as exc:
            print(f"error: {exc}")
            return 2

    with MacroStore(args.db) as store:
        failures = 0
        if not args.skip_fred:
            _, failures = download_fred(store, specs, args, run_id)
        if not args.skip_nyfed:
            download_nyfed(store, args, run_id)
        if not args.skip_gsw:
            download_gsw(store, args, run_id)

        counts = store.table_counts()
        print()
        for table, count in counts.items():
            print(f"  {table:<18}{count:>9,} rows")
        if args.csv_dir:
            exported = store.export_csv(args.csv_dir)
            print(f"\ncsv export -> {args.csv_dir} ({sum(exported.values()):,} rows across {len(exported)} files)")

    print(f"\ndatabase: {args.db}")
    if failures:
        print(f"{failures} series failed; see the fetch_log table")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        traceback.print_exc(limit=0)
        raise SystemExit(130)
