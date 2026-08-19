"""Live market-data dashboard helpers.

The research dashboard is deliberately local-first: SQLite remains the source
of truth and the browser only talks to a small localhost service.  This module
keeps the data-facing pieces independent from the HTTP server so summary and
refresh behaviour can be tested without opening a socket.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from .config import get_tiingo_api_key
from .intraday_store import IntradayStore
from .tiingo_sources import TiingoClient, utc_now_iso


DEFAULT_DB = Path("data/intraday/bars.db")
RECENT_OVERLAP_DAYS = 7


def _read_connection(path: str | Path) -> sqlite3.Connection:
    target = Path(path).resolve()
    connection = sqlite3.connect(f"file:{target.as_posix()}?mode=ro", uri=True, timeout=20)
    connection.row_factory = sqlite3.Row
    return connection


def _number(value: Any, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def tracked_snapshot(path: str | Path = DEFAULT_DB, today: date | None = None) -> dict[str, Any]:
    """Return compact latest-session price and volume statistics for every symbol.

    Five-minute volume is summed into UTC calendar sessions.  US regular-market
    bars never cross UTC midnight, so this is also the exchange session date.
    OTC symbols retain their Tiingo daily fallback frequency.
    """

    today = today or date.today()
    assets: list[dict[str, Any]] = []
    connection = _read_connection(path)
    try:
        symbols = connection.execute(
            "SELECT ticker, name, exchange, has_intraday, themes FROM symbols ORDER BY ticker"
        ).fetchall()

        for symbol in symbols:
            ticker = str(symbol["ticker"])
            frequency = "5min" if symbol["has_intraday"] else "daily"
            bounds = connection.execute(
                "SELECT MIN(ts) AS first_ts, MAX(ts) AS last_ts FROM bars "
                "WHERE ticker = ? AND frequency = ?",
                (ticker, frequency),
            ).fetchone()
            last_ts = bounds["last_ts"] if bounds else None
            first_ts = bounds["first_ts"] if bounds else None

            sessions: list[sqlite3.Row] = []
            if last_ts:
                cutoff = (date.fromisoformat(str(last_ts)[:10]) - timedelta(days=45)).isoformat()
                sessions = connection.execute(
                    """
                    WITH recent AS (
                        SELECT substr(ts, 1, 10) AS session,
                               MAX(ts) AS last_ts,
                               SUM(CASE WHEN volume > 0 THEN volume ELSE 0 END) AS session_volume,
                               SUM(CASE WHEN volume IS NULL THEN 1 ELSE 0 END) AS missing_volume_bars
                        FROM bars
                        WHERE ticker = ? AND frequency = ? AND ts >= ?
                        GROUP BY substr(ts, 1, 10)
                        ORDER BY session DESC
                        LIMIT 21
                    )
                    SELECT recent.session, recent.last_ts, bars.close,
                           recent.session_volume, recent.missing_volume_bars
                    FROM recent
                    JOIN bars ON bars.ticker = ? AND bars.frequency = ?
                             AND bars.ts = recent.last_ts
                    ORDER BY recent.session DESC
                    """,
                    (ticker, frequency, cutoff, ticker, frequency),
                ).fetchall()

            latest = sessions[0] if sessions else None
            previous = sessions[1] if len(sessions) > 1 else None
            prior_volumes = [float(row["session_volume"]) for row in sessions[1:21]
                             if row["session_volume"] is not None and row["session_volume"] > 0]
            average_volume = sum(prior_volumes) / len(prior_volumes) if prior_volumes else None
            close = float(latest["close"]) if latest and latest["close"] is not None else None
            previous_close = (
                float(previous["close"]) if previous and previous["close"] is not None else None
            )
            volume = (
                float(latest["session_volume"])
                if latest and latest["session_volume"] is not None else None
            )
            change = (
                (close / previous_close - 1.0) * 100.0
                if close is not None and previous_close not in (None, 0.0) else None
            )
            volume_ratio = (
                volume / average_volume
                if volume is not None and average_volume not in (None, 0.0) else None
            )
            session_date = str(latest["session"]) if latest else None
            age_days = (today - date.fromisoformat(session_date)).days if session_date else None

            assets.append({
                "ticker": ticker,
                "name": symbol["name"] or ticker,
                "exchange": symbol["exchange"] or "",
                "themes": symbol["themes"] or "",
                "frequency": frequency,
                "price": _number(close),
                "change_pct": _number(change, 3),
                "volume": _number(volume, 0),
                "average_volume_20d": _number(average_volume, 0),
                "volume_ratio": _number(volume_ratio, 3),
                "session_date": session_date,
                "last_timestamp": latest["last_ts"] if latest else None,
                "coverage_start": str(first_ts)[:10] if first_ts else None,
                "coverage_end": str(last_ts)[:10] if last_ts else None,
                "missing_volume_bars": int(latest["missing_volume_bars"] or 0) if latest else 0,
                "age_days": age_days,
                "status": "current" if age_days is not None and age_days <= 4 else "stale",
            })
    finally:
        # sqlite3.Connection's context manager commits/rolls back but does not
        # close.  An explicit close matters on Windows, where an open read
        # handle prevents temporary databases from being removed in tests.
        connection.close()

    latest_session = max((row["session_date"] for row in assets if row["session_date"]), default=None)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "today": today.isoformat(),
        "latest_session": latest_session,
        "count": len(assets),
        "current_count": sum(row["status"] == "current" for row in assets),
        "assets": assets,
        "volume_note": (
            "Five-minute volume is IEX-only and should be read as relative activity; "
            "OTC fallback rows use Tiingo daily volume."
        ),
    }


class RefreshManager:
    """Run one quota-safe recent-bar refresh and expose thread-safe progress."""

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB,
        requests_per_hour: int = 45,
        overlap_days: int = RECENT_OVERLAP_DAYS,
        client_factory: Callable[..., Any] = TiingoClient,
        api_key_factory: Callable[[], str] = get_tiingo_api_key,
        today_factory: Callable[[], date] = date.today,
    ) -> None:
        self.db_path = Path(db_path)
        self.requests_per_hour = requests_per_hour
        self.overlap_days = overlap_days
        self.client_factory = client_factory
        self.api_key_factory = api_key_factory
        self.today_factory = today_factory
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._state: dict[str, Any] = self._idle_state()

    @staticmethod
    def _idle_state() -> dict[str, Any]:
        return {
            "state": "idle", "running": False, "completed": 0, "total": 0,
            "current_symbol": None, "bars_received": 0, "errors": [],
            "started_at": None, "finished_at": None,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            state = dict(self._state)
            state["errors"] = [dict(item) for item in self._state["errors"]]
        total = state["total"]
        state["percent"] = round(state["completed"] / total * 100.0, 1) if total else 0.0
        return state

    def start(self) -> bool:
        """Start a refresh. Return ``False`` when one is already active."""

        api_key = self.api_key_factory()  # fail before claiming a job was started
        with self._lock:
            if self._state["running"]:
                return False
            self._state = self._idle_state() | {
                "state": "starting", "running": True,
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            self._thread = threading.Thread(
                target=self._run, args=(api_key,), name="dashboard-market-refresh", daemon=True
            )
            self._thread.start()
        return True

    def _set(self, **changes: Any) -> None:
        with self._lock:
            self._state.update(changes)

    def _append_error(self, ticker: str, message: str) -> None:
        with self._lock:
            self._state["errors"].append({"ticker": ticker, "message": message[:240]})

    def _run(self, api_key: str) -> None:
        run_id = "dash-" + uuid.uuid4().hex[:10]
        try:
            today = self.today_factory()
            start = today - timedelta(days=self.overlap_days)
            client = self.client_factory(
                api_key,
                requests_per_hour=self.requests_per_hour,
                max_requests=100,
                refresh=True,
            )
            with IntradayStore(self.db_path) as store:
                symbols = store.connection.execute(
                    "SELECT ticker, has_intraday FROM symbols ORDER BY ticker"
                ).fetchall()
                self._set(state="running", total=len(symbols))

                for index, (ticker, has_intraday) in enumerate(symbols, start=1):
                    self._set(current_symbol=ticker)
                    try:
                        if has_intraday:
                            bars, truncated = client.fetch_intraday(ticker, start, today, "5min")
                        else:
                            bars, truncated = client.fetch_daily(ticker, start, today)
                        if truncated:
                            raise RuntimeError("recent refresh unexpectedly hit Tiingo's row cap")
                        store.write_bars(bars)
                        store.log(
                            run_id, utc_now_iso(), "TIINGO_DASHBOARD", ticker, len(bars), "ok",
                            f"recent {start.isoformat()}..{today.isoformat()}",
                        )
                        with self._lock:
                            self._state["bars_received"] += len(bars)
                    except Exception as exc:  # one bad symbol must not hide the other 45
                        self._append_error(ticker, str(exc))
                        store.log(
                            run_id, utc_now_iso(), "TIINGO_DASHBOARD", ticker, 0, "error", str(exc),
                        )
                    self._set(completed=index)

            errors = len(self.status()["errors"])
            self._set(
                state="complete_with_errors" if errors else "complete",
                running=False,
                current_symbol=None,
                finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
        except Exception as exc:
            self._append_error("refresh", str(exc))
            self._set(
                state="failed", running=False, current_symbol=None,
                finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
