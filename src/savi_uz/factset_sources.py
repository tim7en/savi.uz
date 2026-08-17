"""Downloader and parser for the FactSet Earnings Insight PDF.

The parser is deliberately narrow: it reads page 1 only. That page is the "Key
Metrics" summary and is the one part of the report whose structure has survived
nine years; the body's section ordering and page numbers move between editions,
so anything keyed off them would rot.

Because the regexes are the fragile part, the extracted page-1 text is stored
alongside the parsed numbers. Re-parsing an improved pattern over the whole
history is then a local operation with no downloads and no risk of the archive
having moved on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from savi_uz.factset_catalog import (
    CORE_FIELDS,
    DOCUMENT_DATE_PATTERN,
    FIELDS,
    MONTHS,
    Candidate,
    candidate_urls,
)

USER_AGENT = "savi-uz-research/1.0"

#: A PDF that is not at least this big is an error page wearing a .pdf name.
MIN_PDF_BYTES = 10_000


class FactSetError(RuntimeError):
    """A report could not be fetched or read."""


@dataclass
class KeyMetrics:
    """Page 1 of one edition, parsed."""

    report_date: date
    source_url: str
    document_date: date | None
    values: dict[str, Any] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    page_text: str = ""

    @property
    def missing_core(self) -> tuple[str, ...]:
        return tuple(name for name in CORE_FIELDS if name not in self.values)


def normalise(text: str) -> str:
    """Collapse the PDF's line wrapping so patterns can span a line break.

    Also folds the typographic apostrophe, which appears in "Author's Note" and
    would otherwise have to be spelled two ways in every pattern.
    """
    folded = text.replace("’", "'").replace("‘", "'")
    folded = folded.replace("“", '"').replace("”", '"')
    folded = folded.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", folded).strip()


def parse_document_date(text: str) -> date | None:
    match = DOCUMENT_DATE_PATTERN.search(text)
    if not match:
        return None
    month, day, year = match.group(1), int(match.group(2)), int(match.group(3))
    try:
        return date(year, MONTHS[month], day)
    except (KeyError, ValueError):
        return None


def _cast(raw: str, kind: str) -> Any:
    if kind == "text":
        return raw.strip()
    cleaned = raw.replace(",", "").strip()
    try:
        return int(cleaned) if kind == "int" else float(cleaned)
    except ValueError:
        return None


def parse_key_metrics(
    text: str, report_date: date, source_url: str
) -> KeyMetrics:
    """Lift every catalogued field off the normalised page-1 text."""
    flat = normalise(text)
    values: dict[str, Any] = {}
    missing: list[str] = []
    for spec in FIELDS:
        raw = spec.search(flat)
        if raw is None:
            missing.append(spec.name)
            continue
        cast = _cast(raw, spec.kind)
        if cast is None:
            missing.append(spec.name)
            continue
        values[spec.name] = cast

    return KeyMetrics(
        report_date=report_date,
        source_url=source_url,
        document_date=parse_document_date(flat),
        values=values,
        missing=tuple(missing),
        page_text=flat,
    )


def extract_first_page(payload: bytes) -> str:
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise FactSetError(
            "reading the Earnings Insight PDF needs pdfplumber; pip install -r requirements.txt"
        ) from exc

    import io

    try:
        with pdfplumber.open(io.BytesIO(payload)) as document:
            if not document.pages:
                raise FactSetError("PDF has no pages")
            return document.pages[0].extract_text() or ""
    except FactSetError:
        raise
    except Exception as exc:
        raise FactSetError(f"could not read PDF: {exc}") from exc


class FactSetClient:
    """Resolves and caches weekly Earnings Insight PDFs.

    A week is tried across its candidate filenames; the first that returns a
    real PDF wins. A week where none do is a non-publication week -- FactSet
    skips holidays and quiet stretches between reporting seasons -- and is
    reported as such rather than as a failure.
    """

    def __init__(
        self,
        cache_dir: str | Path = ".cache/factset",
        timeout: float = 120.0,
        refresh: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.refresh = refresh

    def _cache_path(self, candidate: Candidate) -> Path:
        return self.cache_dir / candidate.url.rsplit("/", 1)[-1]

    def download(self, candidate: Candidate) -> bytes | None:
        """Bytes for one candidate URL, or None if it is not published."""
        cached = self._cache_path(candidate)
        if cached.is_file() and not self.refresh:
            return cached.read_bytes()

        request = Request(candidate.url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(request, timeout=self.timeout) as response:  # nosec B310
                payload = response.read()
        except HTTPError as exc:
            if exc.code == 404:
                return None
            raise FactSetError(f"HTTP {exc.code} for {candidate.url}") from exc
        except URLError as exc:
            raise FactSetError(f"{candidate.url} unreachable: {exc.reason}") from exc

        # FactSet answers a retired filename with an HTML stub and a 200 in some
        # cases, so the magic bytes decide rather than the status code.
        if not payload.startswith(b"%PDF") or len(payload) < MIN_PDF_BYTES:
            return None

        cached.write_bytes(payload)
        return payload

    def fetch_week(self, friday: date) -> KeyMetrics | None:
        """Parse the edition for one week, or None if none was published."""
        for candidate in candidate_urls(friday):
            payload = self.download(candidate)
            if payload is None:
                continue
            text = extract_first_page(payload)
            if not text.strip():
                raise FactSetError(f"{candidate.url} page 1 extracted to nothing")
            return parse_key_metrics(text, candidate.published, candidate.url)
        return None
