"""Derive a release calendar for the COT tables, which store only the as-of date.

The Commitments of Traders report is compiled as of Tuesday's close and published
the following Friday at 15:30 ET.  Every COT table here carries the Tuesday and
nothing else, so a study that keys on ``report_date`` trades on positioning that
was not public for another three sessions.  That is look-ahead, and it is worth
roughly a week of hindsight in a series that only updates weekly.

This builds ``cot_release_calendar``: one row per distinct as-of date, carrying
the date the report actually became public and the first session on which it
could be acted on.  Studies join on it rather than on ``report_date``.

Two details the naive "as-of plus three days" rule gets wrong.

First, holidays.  When a market holiday falls between Tuesday and Friday the
release slips to the following Monday.  Counting *trading* days rather than
calendar days handles this without a hardcoded holiday list -- the calendar is
taken from ``index_prices``, which is an observed NYSE session series reaching
back to January 2000.

Second, the 2018-19 federal shutdown.  The CFTC kept compiling the report and
backfilled the whole backlog on resumption, so the as-of series in this database
is continuous and the delay is invisible from the data alone.  A plus-three-days
rule would therefore reintroduce weeks of look-ahead across exactly the window
where a positioning signal looks most attractive.  Those weeks are marked
``shutdown_unknown`` with a null release date so that any join drops them; the
alternative -- inventing a catch-up schedule -- would be a fabrication.
"""

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import sqlite3
import sys
from collections import Counter
from pathlib import Path

COT_TABLES = (
    "cot_legacy_futures", "cot_legacy_combined",
    "cot_disagg_futures", "cot_disagg_combined",
    "cot_tff_futures", "cot_tff_combined",
)

#: Publication was suspended for the duration of the lapse in appropriations.
#: Reports whose scheduled Friday falls inside this window have no knowable
#: release date from this database alone.
SHUTDOWN_START = dt.date(2018, 12, 22)
SHUTDOWN_END = dt.date(2019, 1, 25)

#: Tuesday close to Friday publication.
SESSIONS_TO_RELEASE = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS cot_release_calendar (
    report_date    TEXT PRIMARY KEY,
    release_date   TEXT,
    effective_date TEXT,
    lag_days       INTEGER,
    basis          TEXT NOT NULL
);
"""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/data/cftc/cot.db"))
    parser.add_argument("--calendar-db", type=Path,
                        default=Path("data/data/equity/equity.db"))
    parser.add_argument("--calendar-ticker", default="^GSPC")
    parser.add_argument("--check", action="store_true",
                        help="report what would be written, change nothing")
    return parser.parse_args(argv)


def trading_days(path: Path, ticker: str) -> list[dt.date]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT DISTINCT obs_date FROM index_prices WHERE ticker=? ORDER BY obs_date",
        (ticker,),
    ).fetchall()
    connection.close()
    return [dt.date.fromisoformat(r[0][:10]) for r in rows]


def as_of_dates(connection: sqlite3.Connection) -> list[dt.date]:
    present = {r[0] for r in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    seen: set[str] = set()
    for table in COT_TABLES:
        if table in present:
            seen.update(r[0] for r in connection.execute(
                f'SELECT DISTINCT report_date FROM "{table}"') if r[0])
    return sorted(dt.date.fromisoformat(d[:10]) for d in seen)


def nth_session_after(sessions: list[dt.date], day: dt.date, n: int) -> dt.date | None:
    """The nth trading day strictly after ``day``, or None past the calendar."""
    index = bisect.bisect_right(sessions, day) + n - 1
    return sessions[index] if index < len(sessions) else None


def first_session_from(sessions: list[dt.date], day: dt.date) -> dt.date | None:
    """``day`` itself when it trades, otherwise the next session."""
    index = bisect.bisect_left(sessions, day)
    return sessions[index] if index < len(sessions) else None


def scheduled_release(sessions: list[dt.date], day: dt.date) -> dt.date | None:
    """Publication is weekly on Friday, and slips when a holiday intervenes.

    Two constraints, and the release is whichever binds later.  The report goes
    out on the Friday of the as-of week, which is what fixes the odd Monday and
    Friday as-of dates that holiday weeks produce.  It also needs three sessions
    to compile, which is what pushes Thanksgiving and Good Friday weeks into the
    following Monday.  Taking the maximum satisfies both.
    """
    friday = day + dt.timedelta(days=(4 - day.weekday()) % 7 or 7)
    compiled = nth_session_after(sessions, day, SESSIONS_TO_RELEASE)
    if compiled is None:
        return None
    return first_session_from(sessions, max(friday, compiled))


def build(as_of: list[dt.date], sessions: list[dt.date]) -> list[tuple]:
    first, last = sessions[0], sessions[-1]
    rows = []
    for day in as_of:
        if day < first or day > last:
            rows.append((day.isoformat(), None, None, None, "no_calendar"))
            continue
        release = scheduled_release(sessions, day)
        if release is None:
            rows.append((day.isoformat(), None, None, None, "no_calendar"))
            continue
        if SHUTDOWN_START <= release <= SHUTDOWN_END:
            rows.append((day.isoformat(), None, None, None, "shutdown_unknown"))
            continue
        # The report lands at 15:30 ET, inside the session. Treat the next
        # session as the first one an entry rule may act on.
        effective = nth_session_after(sessions, release, 1)
        rows.append((
            day.isoformat(),
            release.isoformat(),
            effective.isoformat() if effective else None,
            (release - day).days,
            "scheduled",
        ))
    return rows


def report(rows: list[tuple]) -> None:
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    basis = Counter(r[4] for r in rows)
    scheduled = [r for r in rows if r[4] == "scheduled"]
    print(f"  as-of dates: {len(rows):,}")
    for name, count in basis.most_common():
        print(f"    {name:18s} {count:>6,d}")
    release_wd = Counter(
        weekdays[dt.date.fromisoformat(r[1]).weekday()] for r in scheduled)
    print(f"\n  release weekday: "
          + ", ".join(f"{k} {v:,}" for k, v in release_wd.most_common()))
    lags = Counter(r[3] for r in scheduled)
    print("  as-of to release, calendar days: "
          + ", ".join(f"{k}d {v:,}" for k, v in sorted(lags.items())))
    span = [r for r in scheduled if r[2]]
    print(f"  rows with an actionable session: {len(span):,}")


def main(argv=None) -> int:
    args = parse_args(argv)
    sessions = trading_days(args.calendar_db, args.calendar_ticker)
    if not sessions:
        raise SystemExit(f"error: no sessions for {args.calendar_ticker}")
    print(f"calendar: {len(sessions):,} sessions, "
          f"{sessions[0]} -> {sessions[-1]}\n")

    connection = sqlite3.connect(args.db)
    try:
        as_of = as_of_dates(connection)
        if not as_of:
            raise SystemExit(f"error: no COT report_date values in {args.db}")
        rows = build(as_of, sessions)
        report(rows)
        if args.check:
            print("\n  --check: nothing written")
            return 0
        connection.executescript(SCHEMA)
        connection.execute("DELETE FROM cot_release_calendar")
        connection.executemany(
            "INSERT INTO cot_release_calendar VALUES (?,?,?,?,?)", rows)
        connection.commit()
        print(f"\n  wrote cot_release_calendar into {args.db}")
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
