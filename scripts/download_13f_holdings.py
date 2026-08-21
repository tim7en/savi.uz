"""Download 13F-HR holdings from EDGAR for a curated list of concentrated managers.

Chapter six established the trap this script exists to avoid: a 13F is filed by
a legal entity, not by a person, and resolving managers by searching EDGAR for
the name each is known by produced seven wrong answers out of thirty-four, two
of them silent. Berkshire resolved to "Feltz Wealth PLAN Inc." Royce resolved to
its former parent. Baron files as BAMCO Inc and Pabrai as Dalal Street LLC, so
neither is findable by the name anyone would search.

The design follows from that. A manager is identified by an explicit CIK
wherever one has been verified. Where it has not, the script *searches and
reports the candidates* rather than picking the first hit, and refuses to
download until a CIK is pinned. Every resolved filer is printed with its EDGAR
name and 13F-HR count so the pairing can be eyeballed against what the manager
is known to be -- which is the only check available, there being no authoritative
name-to-CIK map for investment advisers.

Output is one row per disclosed holding with the filing's public date, which is
the date a follower could act on, distinct from the period the filing describes.
Conflating those two is worth several percentage points of imaginary return.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

USER_AGENT = "savi-uz-research sabitov.ty@gmail.com"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
BROWSE = "https://www.sec.gov/cgi-bin/browse-edgar"

#: CIKs verified by matching the EDGAR conformed name and 13F-HR filing count
#: against what the manager is publicly known to be. Extend deliberately.
#: Each was checked against a near-miss that the name search also returns, and
#: the filing count is what separates them. Baupost's decoy registration has 5
#: filings against the real entity's 105; Fundsmith's decoy fund has 1 against
#: the LLP's 55. Pabrai files as Dalal Street, and searching his own name
#: returns a different filer entirely.
MANAGERS: dict[str, str | None] = {
    "Baron (BAMCO)": "0001017918",          # BAMCO INC /NY/            106
    "Berkshire Hathaway": "0001067983",     # BERKSHIRE HATHAWAY INC    111
    "Baupost Group": "0001061768",          # BAUPOST GROUP LLC/MA      105
    "Pabrai (Dalal Street)": "0001549575",  # Dalal Street, LLC
    "Fundsmith": "0001569205",              # Fundsmith LLP              55
    "Royce": "0000906304",                  # ROYCE & ASSOCIATES LP      39
    "Wasatch": "0000814133",                # WASATCH ADVISORS LP        59
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS holdings (
    manager     TEXT NOT NULL,
    cik         TEXT NOT NULL,
    accession   TEXT NOT NULL,
    period      TEXT NOT NULL,
    filed       TEXT NOT NULL,
    issuer      TEXT,
    cusip       TEXT,
    value       REAL,
    shares      REAL,
    PRIMARY KEY (accession, cusip, issuer)
);
CREATE INDEX IF NOT EXISTS idx_holdings_mgr ON holdings(manager, period);
CREATE TABLE IF NOT EXISTS filers (
    manager   TEXT PRIMARY KEY,
    cik       TEXT,
    edgar_name TEXT,
    filings   INTEGER,
    checked_at TEXT
);
"""


class Pacer:
    """SEC asks for no more than ten requests a second; this stays well under."""

    def __init__(self, per_second: float = 5.0):
        self.interval = 1.0 / per_second
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_at - now)
            self.next_at = max(now, self.next_at) + self.interval
        if delay:
            time.sleep(delay)


@dataclass(frozen=True)
class Holding:
    manager: str
    cik: str
    accession: str
    period: str
    filed: str
    issuer: str
    cusip: str
    value: float
    shares: float


def fetch(url: str, pacer: Pacer, attempts: int = 4) -> str:
    for attempt in range(attempts):
        pacer.wait()
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read().decode("utf-8", "replace")
        except (OSError, TimeoutError) as exc:
            if attempt == attempts - 1:
                raise RuntimeError(f"{url}: {exc}") from exc
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def tag(block: str, name: str) -> str:
    match = re.search(rf"<(?:\w+:)?{name}>(.*?)</(?:\w+:)?{name}>", block, re.S)
    return match.group(1).strip() if match else ""


def search_candidates(name: str, pacer: Pacer) -> list[tuple[str, str]]:
    query = urllib.parse.urlencode({
        "company": name, "type": "13F-HR", "dateb": "", "owner": "include",
        "count": "10", "action": "getcompany", "output": "atom"})
    text = fetch(f"{BROWSE}?{query}", pacer)
    ciks = re.findall(r"CIK=(\d{10})", text)
    names = re.findall(r"<conformed-name>(.*?)</conformed-name>", text)
    seen, out = set(), []
    for cik, conformed in zip(ciks, names):
        if cik not in seen:
            seen.add(cik)
            out.append((conformed, cik))
    return out


def filings_for(cik: str, pacer: Pacer) -> tuple[str, list[dict]]:
    payload = json.loads(fetch(SUBMISSIONS.format(cik=cik), pacer))
    name = payload.get("name", "")
    out: list[dict] = []

    def collect(block: dict) -> None:
        forms = block.get("form", [])
        for index, form in enumerate(forms):
            if form != "13F-HR":
                continue
            out.append({
                "accession": block["accessionNumber"][index],
                "filed": block["filingDate"][index],
                "period": block["reportDate"][index],
            })

    collect(payload["filings"]["recent"])
    for extra in payload["filings"].get("files", []):
        collect(json.loads(fetch(
            f"https://data.sec.gov/submissions/{extra['name']}", pacer)))
    out.sort(key=lambda row: row["filed"])
    return name, out


def holdings_for(manager: str, cik: str, filing: dict, pacer: Pacer) -> list[Holding]:
    accession = filing["accession"].replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}"
    listing = fetch(base + "/", pacer)
    files = re.findall(r'href="([^"]+\.xml)"', listing)
    # The information table is not reliably named. Baron files it as
    # form13fInfoTable.xml, but Wasatch uses sc13f063026.xml, Berkshire 56757.xml
    # and Fundsmith FLLPq22026.xml. Selecting on the filename silently returned
    # nothing for every filer that does not follow the one convention, which cost
    # this download an entire manager. Identify it by content instead: the table
    # is whichever document actually contains infoTable elements.
    candidates = [f for f in files if not f.lower().endswith("primary_doc.xml")]
    candidates.sort(key=lambda f: "infotable" not in f.lower())
    rows: list[str] = []
    for path in candidates:
        url = ("https://www.sec.gov" + path if path.startswith("/")
               else base + "/" + path)
        document = fetch(url, pacer)
        rows = re.findall(r"<(?:\w+:)?infoTable>(.*?)</(?:\w+:)?infoTable>",
                          document, re.S)
        if rows:
            break
    if not rows:
        return []
    out = []
    for row in rows:
        try:
            value = float(tag(row, "value") or 0.0)
            shares = float(re.search(r"sshPrnamt>(.*?)<", row).group(1))
        except (AttributeError, ValueError):
            continue
        out.append(Holding(
            manager=manager, cik=cik, accession=filing["accession"],
            period=filing["period"], filed=filing["filed"],
            issuer=tag(row, "nameOfIssuer"), cusip=tag(row, "cusip"),
            value=value, shares=shares))
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/13f/holdings.db"))
    parser.add_argument("--json", type=Path, default=Path("data/13f/holdings_major.json"))
    parser.add_argument("--since", default="2013-01-01", help="earliest filing date")
    parser.add_argument("--requests-per-second", type=float, default=5.0)
    parser.add_argument("--resolve-only", action="store_true",
                        help="report filer candidates and stop")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    pacer = Pacer(args.requests_per_second)
    args.db.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(args.db)
    connection.executescript(SCHEMA)

    pinned: dict[str, str] = {}
    print("Filer resolution — check each name against what the manager is known to be:\n")
    for manager, cik in MANAGERS.items():
        if cik:
            name, filings = filings_for(cik, pacer)
            print(f"  {manager:24s} CIK {cik}  {name[:38]:40s} {len(filings):>4d} 13F-HR")
            pinned[manager] = cik
            connection.execute(
                "INSERT OR REPLACE INTO filers VALUES (?,?,?,?,datetime('now'))",
                (manager, cik, name, len(filings)))
        else:
            candidates = search_candidates(manager.split(" (")[0], pacer)
            print(f"  {manager:24s} UNPINNED — candidates:")
            for conformed, candidate in candidates[:5]:
                _, filings = filings_for(candidate, pacer)
                print(f"      CIK {candidate}  {conformed[:38]:40s} {len(filings):>4d} 13F-HR")
            print("      -> pin the right CIK in MANAGERS before downloading")
    connection.commit()

    if args.resolve_only or len(pinned) < len(MANAGERS):
        print(f"\n{len(pinned)} of {len(MANAGERS)} managers pinned; "
              f"{'resolve-only' if args.resolve_only else 'refusing to download'}")
        connection.close()
        return 0

    total = 0
    for manager, cik in pinned.items():
        _, filings = filings_for(cik, pacer)
        wanted = [f for f in filings if f["filed"] >= args.since]
        have = {r[0] for r in connection.execute(
            "SELECT DISTINCT accession FROM holdings WHERE manager=?", (manager,))}
        todo = [f for f in wanted if f["accession"] not in have]
        print(f"\n{manager}: {len(wanted)} filings since {args.since}, "
              f"{len(todo)} to fetch")
        for index, filing in enumerate(todo, 1):
            rows = holdings_for(manager, cik, filing, pacer)
            connection.executemany(
                "INSERT OR REPLACE INTO holdings VALUES (?,?,?,?,?,?,?,?,?)",
                [(h.manager, h.cik, h.accession, h.period, h.filed, h.issuer,
                  h.cusip, h.value, h.shares) for h in rows])
            connection.commit()
            total += len(rows)
            if index % 10 == 0 or index == len(todo):
                print(f"  {index}/{len(todo)} filings, {total:,} holdings", flush=True)

    rows = connection.execute(
        "SELECT manager,cik,accession,period,filed,issuer,cusip,value,shares "
        "FROM holdings ORDER BY filed, manager, issuer").fetchall()
    columns = ["mgr", "cik", "accession", "period", "filed", "issuer", "cusip",
               "value", "shares"]
    args.json.write_text(
        json.dumps([dict(zip(columns, row)) for row in rows]), encoding="utf-8")
    print(f"\n{len(rows):,} holdings across "
          f"{len({r[0] for r in rows})} managers -> {args.json}")
    connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
