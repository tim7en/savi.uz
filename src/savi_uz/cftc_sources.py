"""Downloader and parser for the CFTC Commitments of Traders archives.

The archives are ZIPs holding a single quoted-CSV text file with a header row.
Header spelling differs by regime -- ``"Noncommercial Positions-Long (All)"`` in
the legacy files against ``"Prod_Merc_Positions_Long_All"`` in the newer ones --
so every name is normalised to snake_case before it reaches the store, which is
what lets one loader serve all six reports.

Files are cached on disk by name. The current year's ZIP is rewritten by the
CFTC every Friday, so a weekly update is ``--refresh --start-year <this year>``
rather than a full re-pull.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from savi_uz.cftc_catalog import TEXT_COLUMNS, ArchiveFile, ReportSpec

USER_AGENT = "savi-uz-cftc/1.0 (research)"

#: Values the CFTC uses for "not published"; both become NULL.
MISSING_VALUES = frozenset({"", ".", "n/a", "N/A"})

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_REPEATED_UNDERSCORE = re.compile(r"_+")


def normalize_column(name: str) -> str:
    """Fold a CFTC header cell to snake_case.

    Handles every spelling the archives use: spaces and parentheses in the
    legacy files, underscores in the newer ones, the stray double underscore in
    ``Swap__Positions_Short_All``, and ``%`` in the concentration-ratio columns.
    """
    folded = name.strip().lower().replace("%", "pct").replace("&", "and")
    folded = _NON_ALNUM.sub("_", folded)
    return _REPEATED_UNDERSCORE.sub("_", folded).strip("_")


def coerce_value(column: str, raw: str) -> Any:
    """Convert one cell, keeping code columns as text.

    Numbers arrive space-padded and occasionally comma-grouped. Code columns are
    excluded from numeric parsing because ``001602`` is a label whose leading
    zeros carry meaning.
    """
    text = raw.strip()
    if text in MISSING_VALUES:
        return None
    if column in TEXT_COLUMNS:
        return text
    candidate = text.replace(",", "")
    try:
        return int(candidate)
    except ValueError:
        pass
    try:
        return float(candidate)
    except ValueError:
        return text


def parse_report_date(text: str | None) -> date | None:
    """Parse a COT report date, whichever way the file spells it.

    Nearly every archive uses ISO, but the 2006-2016 Traders in Financial
    Futures bundles were exported from a spreadsheet and carry US-order dates
    with a midnight timestamp -- ``9/9/2014 12:00:00 AM`` -- under a column
    still named ``Report_Date_as_YYYY-MM-DD``. Parsing only ISO drops those two
    files entirely and silently.
    """
    if not text:
        return None
    stamp = text.strip()
    if not stamp:
        return None
    try:
        return date.fromisoformat(stamp[:10])
    except ValueError:
        pass
    try:
        month, day, year = (int(part) for part in stamp.split()[0].split("/"))
        return date(year, month, day)
    except (ValueError, TypeError):
        return None


class CftcDownloadError(RuntimeError):
    """A CFTC archive could not be fetched or did not look like a COT ZIP."""


@dataclass(frozen=True)
class ReportChunk:
    """Rows parsed out of one archive file, ready for the store."""

    spec: ReportSpec
    source: str
    columns: tuple[str, ...]
    rows: list[tuple[Any, ...]]
    first_date: date | None
    last_date: date | None
    #: Records the file held that never became rows: outside the requested
    #: window, malformed, or carrying a date this parser could not read. Split
    #: out because "filtered" and "unreadable" look identical in a row count.
    filtered: int = 0
    unparsed: int = 0


class CftcArchiveClient:
    """Fetches COT ZIPs, caching each one by filename."""

    def __init__(
        self,
        cache_dir: str | Path = ".cache/cftc",
        timeout: float = 180.0,
        refresh: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.refresh = refresh

    def download(self, archive: ArchiveFile) -> bytes:
        """Return the ZIP bytes, from cache unless ``refresh`` is set."""
        if not archive.url.startswith("https://"):
            raise CftcDownloadError(f"CFTC archive url must use https://: {archive.url}")

        cached = self.cache_dir / archive.filename
        if cached.is_file() and not self.refresh:
            return cached.read_bytes()

        request = Request(archive.url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(request, timeout=self.timeout) as response:  # nosec B310
                payload = response.read()
        except HTTPError as exc:
            raise CftcDownloadError(f"HTTP {exc.code} for {archive.url}") from exc
        except URLError as exc:
            raise CftcDownloadError(f"{archive.url} unreachable: {exc.reason}") from exc

        # The CFTC serves an HTML error page with a 200 for some retired names,
        # so trust the magic bytes rather than the status code.
        if not payload.startswith(b"PK"):
            raise CftcDownloadError(f"{archive.url} did not return a ZIP ({len(payload)} bytes)")

        cached.write_bytes(payload)
        return payload

    @staticmethod
    def _open_member(payload: bytes) -> tuple[str, io.TextIOWrapper]:
        try:
            archive = zipfile.ZipFile(io.BytesIO(payload))
        except zipfile.BadZipFile as exc:
            # Reached when a cached file was truncated mid-write; delete the
            # cache entry or pass --refresh.
            raise CftcDownloadError(f"not a readable ZIP ({len(payload)} bytes): {exc}") from exc
        members = [name for name in archive.namelist() if name.lower().endswith(".txt")]
        if not members:
            raise CftcDownloadError(f"no .txt member in archive; found {archive.namelist()}")
        member = members[0]
        # latin-1 never raises: a handful of pre-2000 contract names carry bytes
        # that are not valid UTF-8, and losing them is worse than mojibake.
        return member, io.TextIOWrapper(archive.open(member), encoding="latin-1", newline="")

    def load(
        self,
        spec: ReportSpec,
        archive: ArchiveFile,
        start: date | None = None,
        end: date | None = None,
    ) -> ReportChunk:
        """Download, parse and date-filter one archive file.

        Rows are materialised rather than streamed because the store writes them
        in one transaction; the largest single file is the 1995-2016 legacy
        bundle at ~133k rows, which is comfortable in memory.
        """
        payload = self.download(archive)
        member, handle = self._open_member(payload)
        with handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise CftcDownloadError(f"{archive.filename}:{member} is empty") from exc

            columns = tuple(normalize_column(cell) for cell in header)
            if len(columns) != spec.columns:
                raise CftcDownloadError(
                    f"{archive.filename}:{member} has {len(columns)} columns, "
                    f"expected {spec.columns} for {spec.key}"
                )
            for required in (spec.date_column, spec.contract_column):
                if required not in columns:
                    raise CftcDownloadError(
                        f"{archive.filename}:{member} has no {required!r} column"
                    )

            date_index = columns.index(spec.date_column)
            contract_index = columns.index(spec.contract_column)

            rows: list[tuple[Any, ...]] = []
            first_date = last_date = None
            filtered = unparsed = 0
            for record in reader:
                if len(record) != len(columns):
                    unparsed += 1
                    continue
                report_date = parse_report_date(record[date_index])
                if report_date is None:
                    unparsed += 1
                    continue
                if start is not None and report_date < start:
                    filtered += 1
                    continue
                if end is not None and report_date > end:
                    filtered += 1
                    continue
                values = tuple(
                    coerce_value(column, cell) for column, cell in zip(columns, record)
                )
                rows.append((record[contract_index].strip(), report_date.isoformat()) + values)
                first_date = report_date if first_date is None else min(first_date, report_date)
                last_date = report_date if last_date is None else max(last_date, report_date)

        # A file whose records were all unreadable is a parser failure wearing
        # the costume of an empty result; the 2006-2016 TFF bundles did exactly
        # this before their date format was handled.
        if unparsed and not rows and not filtered:
            raise CftcDownloadError(
                f"{archive.filename}:{member} yielded no usable rows out of {unparsed} records"
            )

        return ReportChunk(
            spec=spec,
            source=f"{archive.filename}:{member}",
            columns=("contract_code", "report_date") + columns,
            rows=rows,
            first_date=first_date,
            last_date=last_date,
            filtered=filtered,
            unparsed=unparsed,
        )


def iter_chunks(
    client: CftcArchiveClient,
    spec: ReportSpec,
    archives: tuple[ArchiveFile, ...],
    start: date | None = None,
    end: date | None = None,
) -> Iterator[ReportChunk]:
    for archive in archives:
        yield client.load(spec, archive, start=start, end=end)


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
