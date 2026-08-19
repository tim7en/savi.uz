"""Compact live payloads for the dashboard's research tabs."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import get_alphavantage_api_key
from .fundamentals import (
    BASE_URL,
    DEFAULT_FOLDER as FUNDAMENTALS_FOLDER,
    DEFAULT_REQUESTS_PER_MINUTE,
    PLAN_REQUESTS_PER_MINUTE,
    USER_AGENT,
    _float,
    _read_payload,
    load_universe,
)
from .macro_sources import RateLimiter


EQUITY_DB = Path("data/equity/equity.db")
MACRO_DB = Path("data/macro/macro.db")
CFTC_DB = Path("data/cftc/cot.db")
INTRADAY_DB = Path("data/intraday/bars.db")
MARKETDATA_OPTIONS_DB = Path("data/options/marketdata.db")
ALPHAVANTAGE_OPTIONS_DB = Path("data/options/alphavantage.db")
CROSS_ASSET_FOLDER = Path("data/cross_assets")


def _connect(path: str | Path) -> sqlite3.Connection:
    target = Path(path).resolve()
    connection = sqlite3.connect(f"file:{target.as_posix()}?mode=ro", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def _round(value: Any, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    rows = sorted(values)
    middle = len(rows) // 2
    return rows[middle] if len(rows) % 2 else (rows[middle - 1] + rows[middle]) / 2.0


def _percentile(values: list[float], current: float | None) -> float | None:
    if current is None or not values:
        return None
    return sum(value <= current for value in values) / len(values) * 100.0


def _series(connection: sqlite3.Connection, series_id: str, limit: int = 780) -> list[list[Any]]:
    rows = connection.execute(
        "SELECT obs_date, value FROM observations WHERE series_id = ? AND value IS NOT NULL "
        "ORDER BY obs_date DESC LIMIT ?", (series_id, limit),
    ).fetchall()
    return [[row[0], _round(row[1])] for row in reversed(rows)]


def _last(points: list[list[Any]]) -> list[Any] | None:
    return points[-1] if points else None


def earnings_analysis_snapshot(
    equity_db: str | Path = EQUITY_DB,
    fundamentals_folder: str | Path = FUNDAMENTALS_FOLDER,
    today: date | None = None,
) -> dict[str, Any]:
    """FactSet aggregate history plus company-level forward estimate revisions."""

    today = today or date.today()
    connection = _connect(equity_db)
    try:
        reports = connection.execute(
            "SELECT report_date, quarter, pct_reported, pct_positive_eps, pct_positive_revenue, "
            "blended_earnings_growth, estimated_earnings_growth, estimated_growth_at_quarter_start, "
            "forward_12m_pe, pe_5y_average, pe_10y_average, negative_guidance_count, "
            "positive_guidance_count, index_price, forward_12m_eps "
            "FROM factset_reports ORDER BY report_date"
        ).fetchall()
    finally:
        connection.close()

    history = [{key: row[key] for key in row.keys()} for row in reports]
    latest = history[-1] if history else {}
    folder = Path(fundamentals_folder)
    symbols, universe = load_universe(folder)
    estimates: list[dict[str, Any]] = []

    for ticker in symbols:
        overview, _ = _read_payload(folder, ticker, "overview")
        document, updated = _read_payload(folder, ticker, "earnings_estimates")
        rows = document.get("estimates", []) if isinstance(document, dict) else []
        quarter_rows = [row for row in rows if isinstance(row, dict) and row.get("horizon") == "fiscal quarter"]
        future = [row for row in quarter_rows if str(row.get("date") or "") >= today.isoformat()]
        selected = min(future, key=lambda row: str(row.get("date")), default={})
        if not selected and quarter_rows:
            selected = max(quarter_rows, key=lambda row: str(row.get("date") or ""))
        current = _float(selected.get("eps_estimate_average"))
        thirty = _float(selected.get("eps_estimate_average_30_days_ago"))
        revision = ((current / thirty - 1.0) * 100.0) if current is not None and thirty not in (None, 0.0) else None
        up = _float(selected.get("eps_estimate_revision_up_trailing_30_days"))
        down = _float(selected.get("eps_estimate_revision_down_trailing_30_days"))
        estimates.append({
            "ticker": ticker,
            "name": overview.get("Name") or ticker,
            "sector": overview.get("Sector") or "Unclassified",
            "period": selected.get("date"),
            "eps_estimate": current,
            "eps_30d_ago": thirty,
            "eps_revision_pct": _round(revision, 3),
            "analysts": _float(selected.get("eps_estimate_analyst_count")),
            "revision_up_30d": up,
            "revision_down_30d": down,
            "revision_breadth": (up - down) if up is not None and down is not None else None,
            "revenue_estimate": _float(selected.get("revenue_estimate_average")),
            "updated_at": updated,
            "status": "current" if updated and str(updated)[:10] == today.isoformat() else ("stale" if updated else "missing"),
        })

    covered = [row for row in estimates if row["eps_estimate"] is not None]
    revisions = [row["eps_revision_pct"] for row in covered if row["eps_revision_pct"] is not None]
    breadth = [row["revision_breadth"] for row in covered if row["revision_breadth"] is not None]
    sector_map: dict[str, list[dict[str, Any]]] = {}
    for row in covered:
        sector_map.setdefault(row["sector"], []).append(row)
    sectors = []
    for sector, rows in sorted(sector_map.items()):
        sector_revisions = [row["eps_revision_pct"] for row in rows if row["eps_revision_pct"] is not None]
        sector_breadth = [row["revision_breadth"] for row in rows if row["revision_breadth"] is not None]
        sectors.append({
            "sector": sector,
            "companies": len(rows),
            "median_eps_revision_pct": _round(_median(sector_revisions), 3),
            "positive_revision_pct": _round(sum(value > 0 for value in sector_revisions) / len(sector_revisions) * 100.0, 1) if sector_revisions else None,
            "net_revision_breadth": _round(sum(sector_breadth), 0) if sector_breadth else None,
        })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "factset_latest": latest,
        "factset_history": history[-260:],
        "factset_first": history[0]["report_date"] if history else None,
        "estimate_universe": universe,
        "estimate_count": len(estimates),
        "estimate_covered": len(covered),
        "estimate_current": sum(row["status"] == "current" for row in estimates),
        "median_eps_revision_pct": _round(_median(revisions), 3),
        "positive_revision_pct": _round(sum(value > 0 for value in revisions) / len(revisions) * 100.0, 1) if revisions else None,
        "net_revision_breadth": _round(sum(breadth), 0) if breadth else None,
        "sectors": sectors,
        "estimates": estimates,
    }


CROSS_ASSET_REQUESTS: tuple[tuple[str, dict[str, str], str], ...] = (
    ("WTI crude", {"function": "WTI", "interval": "daily"}, "wti"),
    ("Brent crude", {"function": "BRENT", "interval": "daily"}, "brent"),
    ("Natural gas", {"function": "NATURAL_GAS", "interval": "daily"}, "natural_gas"),
    ("Copper", {"function": "COPPER", "interval": "monthly"}, "copper"),
    ("Aluminium", {"function": "ALUMINUM", "interval": "monthly"}, "aluminum"),
    ("Wheat", {"function": "WHEAT", "interval": "monthly"}, "wheat"),
    ("Corn", {"function": "CORN", "interval": "monthly"}, "corn"),
    ("Treasury 2Y", {"function": "TREASURY_YIELD", "interval": "daily", "maturity": "2year"}, "treasury_2y"),
    ("Treasury 5Y", {"function": "TREASURY_YIELD", "interval": "daily", "maturity": "5year"}, "treasury_5y"),
    ("Treasury 10Y", {"function": "TREASURY_YIELD", "interval": "daily", "maturity": "10year"}, "treasury_10y"),
    ("Treasury 30Y", {"function": "TREASURY_YIELD", "interval": "daily", "maturity": "30year"}, "treasury_30y"),
)


def _cached_av_series(folder: Path, slug: str) -> tuple[list[list[Any]], str | None]:
    path = folder / f"{slug}.json"
    if not path.is_file():
        return [], None
    try:
        wrapper = json.loads(path.read_text(encoding="utf-8"))
        document = wrapper.get("data", {})
        points = []
        for row in document.get("data", []):
            value = _float(row.get("value"))
            if value is not None and row.get("date"):
                points.append([row["date"], value])
        return sorted(points), wrapper.get("timestamp")
    except (OSError, json.JSONDecodeError, AttributeError):
        return [], None


def _intraday_proxy(connection: sqlite3.Connection, ticker: str) -> dict[str, Any]:
    rows = connection.execute(
        """
        WITH sessions AS (
          SELECT substr(ts,1,10) day, MAX(ts) last_ts
          FROM bars WHERE ticker=? AND frequency='5min'
          GROUP BY substr(ts,1,10) ORDER BY day DESC LIMIT 260
        )
        SELECT sessions.day,bars.close FROM sessions JOIN bars
          ON bars.ticker=? AND bars.frequency='5min' AND bars.ts=sessions.last_ts
        ORDER BY sessions.day
        """, (ticker, ticker),
    ).fetchall()
    points = [[row[0], _round(row[1], 3)] for row in rows if row[1] is not None]
    change = (points[-1][1] / points[-2][1] - 1.0) * 100.0 if len(points) > 1 else None
    return {"ticker": ticker, "latest": _last(points), "change_pct": _round(change, 3), "history": points}


def cross_assets_snapshot(
    macro_db: str | Path = MACRO_DB,
    intraday_db: str | Path = INTRADAY_DB,
    cache_folder: str | Path = CROSS_ASSET_FOLDER,
) -> dict[str, Any]:
    macro = _connect(macro_db)
    intraday = _connect(intraday_db)
    try:
        yields = {sid: _series(macro, sid, 520) for sid in ("DGS2", "DGS5", "DGS10", "DGS30", "T10Y2Y")}
        vol = {sid: _series(macro, sid, 520) for sid in ("OVXCLS", "GVZCLS")}
        proxies = {ticker: _intraday_proxy(intraday, ticker) for ticker in ("GLD", "SLV", "XLE", "TBT", "TMF")}
    finally:
        macro.close()
        intraday.close()
    direct = {}
    root = Path(cache_folder)
    for label, _params, slug in CROSS_ASSET_REQUESTS:
        points, updated = _cached_av_series(root, slug)
        direct[slug] = {"label": label, "history": points[-520:], "latest": _last(points), "updated_at": updated}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "yields": yields, "volatility": vol, "proxies": proxies, "direct": direct,
        "direct_covered": sum(bool(row["history"]) for row in direct.values()),
        "direct_total": len(direct),
    }


def options_snapshot(
    marketdata_db: str | Path = MARKETDATA_OPTIONS_DB,
    alphavantage_db: str | Path = ALPHAVANTAGE_OPTIONS_DB,
) -> dict[str, Any]:
    av = _connect(alphavantage_db)
    market = _connect(marketdata_db)
    symbols: dict[str, Any] = {}
    try:
        for ticker in ("SPY", "QQQ"):
            rows = av.execute(
                "SELECT * FROM av_daily WHERE symbol=? ORDER BY observation_date DESC LIMIT 260",
                (ticker,),
            ).fetchall()
            history = [dict(row) for row in reversed(rows)]
            latest = dict(rows[0]) if rows else {}
            net_values = [float(row["net_gex"]) for row in rows if row["net_gex"] is not None]
            latest["gex_percentile"] = _round(_percentile(net_values, latest.get("net_gex")), 1)
            md = market.execute(
                "SELECT * FROM daily_gex WHERE symbol=? ORDER BY observation_date DESC LIMIT 1",
                (ticker,),
            ).fetchone()
            market_latest = dict(md) if md else {}
            put_wall = None
            if md:
                wall = market.execute(
                    "SELECT strike,SUM(ABS(gamma*open_interest*100*underlying_price*underlying_price*0.01)) exposure "
                    "FROM option_contracts WHERE symbol=? AND observation_date=? AND side='put' "
                    "AND gamma IS NOT NULL AND open_interest IS NOT NULL GROUP BY strike "
                    "ORDER BY exposure DESC LIMIT 1",
                    (ticker, md["observation_date"]),
                ).fetchone()
                put_wall = wall["strike"] if wall else None
            symbols[ticker] = {
                "latest": latest, "marketdata_latest": market_latest, "put_wall_strike": put_wall,
                "history": [{
                    "date": row["observation_date"], "net_gex": _round(row["net_gex"]),
                    "atm_iv": _round(row["atm_iv"]), "put_call_oi": _round(row["put_call_oi"]),
                    "zero_dte_share": _round(row["zero_dte_share"]),
                } for row in history],
            }
    finally:
        av.close()
        market.close()
    latest_dates = []
    for row in symbols.values():
        if row["latest"].get("observation_date"):
            latest_dates.append(row["latest"]["observation_date"])
        if row["marketdata_latest"].get("observation_date"):
            latest_dates.append(row["marketdata_latest"]["observation_date"])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "latest_date": max(latest_dates, default=None), "symbols": symbols,
        "realtime_available_on_plan": False,
        "plan_note": "Alpha Vantage realtime options require a 600/min or 1,200/min plan; this dashboard uses historical chains and locally computed GEX.",
    }


CFTC_CONTRACTS: tuple[tuple[str, str, str, str, str], ...] = (
    ("WTI crude", "cot_disagg_futures", "067651", "m_money_positions_long_all", "m_money_positions_short_all"),
    ("Natural gas", "cot_disagg_futures", "023651", "m_money_positions_long_all", "m_money_positions_short_all"),
    ("Gold", "cot_disagg_futures", "088691", "m_money_positions_long_all", "m_money_positions_short_all"),
    ("Corn", "cot_disagg_futures", "002602", "m_money_positions_long_all", "m_money_positions_short_all"),
    ("UST 10Y", "cot_tff_futures", "043602", "lev_money_positions_long_all", "lev_money_positions_short_all"),
    ("E-mini S&P", "cot_tff_futures", "13874A", "lev_money_positions_long_all", "lev_money_positions_short_all"),
    ("Euro FX", "cot_tff_futures", "099741", "lev_money_positions_long_all", "lev_money_positions_short_all"),
)


def cftc_snapshot(path: str | Path = CFTC_DB) -> dict[str, Any]:
    connection = _connect(path)
    markets = []
    try:
        for label, table, code, long_column, short_column in CFTC_CONTRACTS:
            rows = connection.execute(
                f"SELECT report_date,{long_column} long_side,{short_column} short_side,open_interest_all "
                f"FROM {table} WHERE contract_code=? ORDER BY report_date DESC LIMIT 156", (code,),
            ).fetchall()
            history = []
            for row in reversed(rows):
                if row["long_side"] is None or row["short_side"] is None:
                    continue
                net = float(row["long_side"]) - float(row["short_side"])
                share = net / float(row["open_interest_all"]) * 100.0 if row["open_interest_all"] else None
                history.append([row["report_date"], _round(share, 3), _round(net, 0)])
            latest = history[-1] if history else None
            previous = history[-2] if len(history) > 1 else None
            values = [row[1] for row in history if row[1] is not None]
            markets.append({
                "label": label, "report": "Managed money" if "disagg" in table else "Leveraged funds",
                "date": latest[0] if latest else None, "net_share_pct": latest[1] if latest else None,
                "net_contracts": latest[2] if latest else None,
                "weekly_change_pct_points": _round(latest[1] - previous[1], 3) if latest and previous and latest[1] is not None and previous[1] is not None else None,
                "percentile_3y": _round(_percentile(values, latest[1] if latest else None), 1),
                "history": history[-104:],
            })
    finally:
        connection.close()
    latest_date = max((row["date"] for row in markets if row["date"]), default=None)
    return {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "latest_date": latest_date, "markets": markets}


def fed_snapshot(path: str | Path = MACRO_DB) -> dict[str, Any]:
    connection = _connect(path)
    try:
        series_ids = ("DFF", "IORB", "SOFR", "DFEDTARU", "DFEDTARL", "WALCL", "WRESBAL", "WTREGEN", "RRPONTSYD")
        series = {sid: _series(connection, sid, 900 if sid in ("DFF", "IORB", "SOFR") else 180) for sid in series_ids}
    finally:
        connection.close()
    dff = series["DFF"]
    policy_change = dff[-1][1] - dff[-66][1] if len(dff) >= 66 else None
    direction = "steady"
    if policy_change is not None and policy_change <= -0.20:
        direction = "easing"
    elif policy_change is not None and policy_change >= 0.20:
        direction = "tightening"
    walcl = series["WALCL"]
    balance_change = (walcl[-1][1] / walcl[-14][1] - 1.0) * 100.0 if len(walcl) >= 14 and walcl[-14][1] else None
    latest = {sid: _last(points) for sid, points in series.items()}
    values = {sid: (row[1] if row else None) for sid, row in latest.items()}
    liquidity = None
    if all(values.get(sid) is not None for sid in ("WRESBAL", "WTREGEN", "RRPONTSYD")):
        # All three series are $m except RRP, which is $bn.
        liquidity = values["WRESBAL"] - values["WTREGEN"] - values["RRPONTSYD"] * 1000.0
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "latest": latest, "series": series, "policy_direction_3m": direction,
        "policy_change_3m_pct_points": _round(policy_change, 2),
        "balance_change_13w_pct": _round(balance_change, 2),
        "liquidity_proxy_millions": _round(liquidity, 0),
    }


class CrossAssetRefreshManager:
    """Fetch the direct commodity and Treasury series missing from local FRED."""

    def __init__(self, folder: str | Path = CROSS_ASSET_FOLDER,
                 requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
                 api_key_factory: Callable[[], str] = get_alphavantage_api_key) -> None:
        if not 1 <= requests_per_minute <= PLAN_REQUESTS_PER_MINUTE:
            raise ValueError(f"requests_per_minute must be 1..{PLAN_REQUESTS_PER_MINUTE}")
        self.folder = Path(folder)
        self.requests_per_minute = requests_per_minute
        self.api_key_factory = api_key_factory
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._state = self._idle()

    def _idle(self) -> dict[str, Any]:
        return {"state": "idle", "running": False, "completed": 0, "total": 0,
                "current_series": None, "files_updated": 0, "errors": [],
                "started_at": None, "finished_at": None}

    def status(self) -> dict[str, Any]:
        with self._lock:
            state = dict(self._state)
            state["errors"] = [dict(row) for row in self._state["errors"]]
        state["percent"] = round(state["completed"] / state["total"] * 100.0, 1) if state["total"] else 0.0
        return state

    def _set(self, **changes: Any) -> None:
        with self._lock:
            self._state.update(changes)

    def start(self) -> bool:
        key = self.api_key_factory()
        with self._lock:
            if self._state["running"]:
                return False
            self._state = self._idle() | {"state": "starting", "running": True, "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            self._thread = threading.Thread(target=self._run, args=(key,), daemon=True, name="dashboard-cross-assets-refresh")
            self._thread.start()
        return True

    def _run(self, api_key: str) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        limiter = RateLimiter(self.requests_per_minute)
        today = date.today().isoformat()
        queue = []
        for label, params, slug in CROSS_ASSET_REQUESTS:
            path = self.folder / f"{slug}.json"
            current = False
            if path.is_file():
                try:
                    current = str(json.loads(path.read_text(encoding="utf-8")).get("timestamp", ""))[:10] == today
                except (OSError, json.JSONDecodeError):
                    pass
            if not current:
                queue.append((label, params, slug))
        self._set(state="running", total=len(queue))
        try:
            for index, (label, params, slug) in enumerate(queue, start=1):
                self._set(current_series=label)
                try:
                    limiter.acquire()
                    query = urlencode(params | {"apikey": api_key})
                    request = Request(f"{BASE_URL}?{query}", headers={"User-Agent": USER_AGENT})
                    with urlopen(request, timeout=60) as response:  # nosec B310
                        document = json.loads(response.read().decode("utf-8"))
                    for key in ("Note", "Information", "Error Message"):
                        if key in document:
                            raise RuntimeError(str(document[key]))
                    wrapper = {"timestamp": datetime.now().astimezone().isoformat(), "label": label, "parameters": params, "data": document}
                    path = self.folder / f"{slug}.json"
                    temporary = path.with_suffix(path.suffix + ".tmp-" + uuid.uuid4().hex[:8])
                    temporary.write_text(json.dumps(wrapper, indent=2), encoding="utf-8")
                    os.replace(temporary, path)
                    with self._lock:
                        self._state["files_updated"] += 1
                except (HTTPError, URLError, json.JSONDecodeError, RuntimeError) as exc:
                    with self._lock:
                        self._state["errors"].append({"series": label, "message": str(exc)[:240]})
                self._set(completed=index)
            errors = len(self.status()["errors"])
            self._set(state="complete_with_errors" if errors else "complete", running=False, current_series=None,
                      finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        except Exception as exc:
            with self._lock:
                self._state["errors"].append({"series": "setup", "message": str(exc)[:240]})
            self._set(state="failed", running=False, current_series=None,
                      finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
