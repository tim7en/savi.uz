"""Extract a compact JSON payload for the six dashboard pages.

Reads the four databases the download scripts produce and emits one JSON file
holding every series a page plots. Long daily histories are thinned to weekly or
monthly on the way out -- a 26-year daily curve is 6,700 points that render as a
2px line either way, and the page has to stay under the artifact size ceiling.

Usage:
    PYTHONPATH=src python scripts/build_dashboard_data.py --outdir out/dashboard
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

NY = ZoneInfo("America/New_York")

MACRO_DB = Path("data/macro/macro.db")
CFTC_DB = Path("data/cftc/cot.db")
EQUITY_DB = Path("data/equity/equity.db")
INTRADAY_DB = Path("data/intraday/bars.db")


def connect(path: Path) -> sqlite3.Connection | None:
    if not path.is_file():
        print(f"  warning: {path} missing; its page will be empty")
        return None
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def fred_series(
    conn: sqlite3.Connection, series_id: str, start: str = "2000-01-01", every: int = 1
) -> list[list]:
    """[[date, value], ...] for one FRED series, thinned by ``every``."""
    rows = conn.execute(
        "SELECT obs_date, value FROM observations WHERE series_id = ? AND obs_date >= ? "
        "AND value IS NOT NULL ORDER BY obs_date",
        (series_id, start),
    ).fetchall()
    return [[row[0], _round(row[1])] for row in rows[::every]]


def series_title(conn: sqlite3.Connection, series_id: str) -> str:
    row = conn.execute("SELECT label, title FROM series WHERE series_id = ?", (series_id,)).fetchone()
    return (row[0] or row[1] or series_id) if row else series_id


def build_fed_page(conn: sqlite3.Connection | None) -> dict:
    """Policy rate, the corridor around it, and the balance sheet behind it."""
    if conn is None:
        return {}
    # Daily rates thinned to weekly; the policy path is a step function and
    # weekly sampling loses nothing a reader can see.
    policy = {
        sid: fred_series(conn, sid, every=5)
        for sid in ("DFF", "IORB", "SOFR", "DFEDTARU", "DFEDTARL")
    }
    # FRED does not publish these in one unit: WALCL, WRESBAL and WTREGEN are
    # millions of dollars while RRPONTSYD is billions. Plotting them together
    # without normalising puts a $2.9tn reserve balance and a $0bn RRP on the
    # same axis three orders of magnitude apart. Everything becomes $bn here.
    to_billions = {"WALCL": 1e-3, "WRESBAL": 1e-3, "WTREGEN": 1e-3, "RRPONTSYD": 1.0}
    balance = {}
    for sid, factor in to_billions.items():
        balance[sid] = [[day, _round(value * factor, 2)] for day, value in fred_series(conn, sid)]
    return {
        "policy": {k: v for k, v in policy.items() if v},
        "policy_titles": {sid: series_title(conn, sid) for sid in policy},
        "balance": {k: v for k, v in balance.items() if v},
        "balance_titles": {sid: series_title(conn, sid) for sid in balance},
    }


def build_market_page(conn: sqlite3.Connection | None) -> dict:
    """What the market prices, as opposed to what the Fed sets."""
    if conn is None:
        return {}
    curve = {sid: fred_series(conn, sid, every=5) for sid in ("DGS2", "DGS10", "DGS30")}
    spread = {sid: fred_series(conn, sid, every=5) for sid in ("T10Y2Y",)}
    breakeven = {sid: fred_series(conn, sid, every=5) for sid in ("T5YIE", "T10YIE", "T5YIFR")}
    stress = {sid: fred_series(conn, sid, every=5) for sid in ("BAA10Y", "VIXCLS")}
    nfci = {sid: fred_series(conn, sid) for sid in ("NFCI",)}

    # Group by horizon *before* thinning. The four horizons interleave in the
    # table, so a stride applied to the raw stream that shares a factor with the
    # horizon count samples the same horizon every time and silently returns one
    # series where there should be four.
    path_rows = conn.execute(
        "SELECT curve_date, horizon_months, forward_rate FROM fed_path "
        "WHERE horizon_months IN (3, 12, 24, 60) AND curve_date >= '2000-01-01' "
        "ORDER BY horizon_months, curve_date"
    ).fetchall()
    grouped: dict[str, list[list]] = defaultdict(list)
    for day, months, rate in path_rows:
        grouped[f"{months}m"].append([day, _round(rate)])
    fed_path = {key: points[::5] for key, points in grouped.items()}

    return {
        "curve": {k: v for k, v in curve.items() if v},
        "curve_titles": {sid: series_title(conn, sid) for sid in curve},
        "spread": {k: v for k, v in spread.items() if v},
        "breakeven": {k: v for k, v in breakeven.items() if v},
        "breakeven_titles": {sid: series_title(conn, sid) for sid in breakeven},
        "stress": {k: v for k, v in stress.items() if v},
        "nfci": {k: v for k, v in nfci.items() if v},
        "fed_path": dict(fed_path),
    }


def build_volatility_page(conn: sqlite3.Connection | None) -> dict:
    """The CBOE complex: level, term structure, and vol across asset classes."""
    if conn is None:
        return {}
    equity = {sid: fred_series(conn, sid, every=3) for sid in ("VIXCLS", "VXVCLS")}
    cross = {
        sid: fred_series(conn, sid, every=3)
        for sid in ("VXNCLS", "RVXCLS", "OVXCLS", "GVZCLS")
    }

    # 3-month minus 1-month. Positive is the normal upward-sloping term
    # structure; negative means near-dated vol is bid above far-dated, which is
    # what a market in stress looks like. Computed on the unthinned series so
    # the sign is not an artefact of sampling.
    near = dict(fred_series(conn, "VIXCLS"))
    far = dict(fred_series(conn, "VXVCLS"))
    spread = [
        [day, _round(far[day] - near[day], 3)]
        for day in sorted(set(near) & set(far))
        if near[day] is not None and far[day] is not None
    ]
    inverted = sum(1 for _, value in spread if value is not None and value < 0)

    # The full-history VIX, monthly maxima, so the 1990-2026 shape survives
    # thinning without losing the spikes that are the whole point.
    monthly: dict[str, float] = {}
    for day, value in fred_series(conn, "VIXCLS", start="1990-01-01"):
        if value is None:
            continue
        month = day[:7]
        monthly[month] = max(monthly.get(month, 0.0), value)
    history = [[month, _round(monthly[month], 2)] for month in sorted(monthly)]

    return {
        "equity": {k: v for k, v in equity.items() if v},
        "cross": {k: v for k, v in cross.items() if v},
        "spread": spread[::3],
        "inverted_days": inverted,
        "spread_days": len(spread),
        "history": history,
        "titles": {
            sid: series_title(conn, sid)
            for sid in ("VIXCLS", "VXVCLS", "VXNCLS", "RVXCLS", "OVXCLS", "GVZCLS")
        },
    }


#: Contracts worth plotting: the deepest in each report, by recent open interest.
DISAGG_CONTRACTS = (
    ("067651", "WTI crude"),
    ("023651", "Natural gas"),
    ("002602", "Corn"),
    ("088691", "Gold"),
)
TFF_CONTRACTS = (
    ("043602", "UST 10Y note"),
    ("13874A", "E-mini S&P 500"),
    ("099741", "Euro FX"),
)


def build_cftc_page(conn: sqlite3.Connection | None) -> dict:
    """Net positioning of the speculative cohort, as a share of open interest.

    Net contracts alone are not comparable across markets or across a decade of
    growth in open interest, so both are carried: the raw net and the share.
    """
    if conn is None:
        return {}

    def net_series(table: str, code: str, long_col: str, short_col: str) -> list[list]:
        rows = conn.execute(
            f"SELECT report_date, {long_col}, {short_col}, open_interest_all FROM {table} "
            f"WHERE contract_code = ? AND report_date >= '2010-01-01' ORDER BY report_date",
            (code,),
        ).fetchall()
        out = []
        for day, long_side, short_side, oi in rows:
            if long_side is None or short_side is None:
                continue
            net = long_side - short_side
            share = (net / oi * 100.0) if oi else None
            out.append([day, int(net), _round(share, 2)])
        return out

    managed = {
        label: net_series("cot_disagg_futures", code,
                          "m_money_positions_long_all", "m_money_positions_short_all")
        for code, label in DISAGG_CONTRACTS
    }
    levered = {
        label: net_series("cot_tff_futures", code,
                          "lev_money_positions_long_all", "lev_money_positions_short_all")
        for code, label in TFF_CONTRACTS
    }
    coverage = conn.execute(
        "SELECT report, first_date, last_date, rows FROM cot_reports ORDER BY report"
    ).fetchall()
    return {
        "managed_money": {k: v for k, v in managed.items() if v},
        "leveraged_funds": {k: v for k, v in levered.items() if v},
        "coverage": [list(row) for row in coverage],
    }


def build_earnings_page(conn: sqlite3.Connection | None) -> dict:
    """The long valuation record, and the forward expectations against it."""
    if conn is None:
        return {}
    shiller = conn.execute(
        "SELECT obs_date, cape, real_price, real_earnings FROM shiller_monthly "
        "WHERE cape IS NOT NULL ORDER BY obs_date"
    ).fetchall()
    factset = conn.execute(
        "SELECT report_date, forward_12m_pe, pe_5y_average, pe_10y_average, "
        "blended_earnings_growth, estimated_earnings_growth, pct_positive_eps "
        "FROM factset_reports ORDER BY report_date"
    ).fetchall()
    # One aggregate per quarter: how many filers reported, and the median EPS.
    sec = conn.execute(
        "SELECT frame, COUNT(*) FROM sec_facts WHERE concept = 'EarningsPerShareDiluted' "
        "GROUP BY frame ORDER BY frame"
    ).fetchall()
    return {
        "cape": [[r[0][:7], _round(r[1], 2)] for r in shiller],
        "real_price": [[r[0][:7], _round(r[2], 1)] for r in shiller if r[2] is not None],
        "real_earnings": [[r[0][:7], _round(r[3], 2)] for r in shiller if r[3] is not None],
        "forward_pe": [[r[0], _round(r[1], 2)] for r in factset if r[1] is not None],
        "pe_5y": [[r[0], _round(r[2], 2)] for r in factset if r[2] is not None],
        "pe_10y": [[r[0], _round(r[3], 2)] for r in factset if r[3] is not None],
        "blended_growth": [[r[0], _round(r[4], 2)] for r in factset if r[4] is not None],
        "estimated_growth": [[r[0], _round(r[5], 2)] for r in factset if r[5] is not None],
        "beat_rate": [[r[0], _round(r[6], 1)] for r in factset if r[6] is not None],
        "sec_filers": [[row[0], row[1]] for row in sec if row[0].startswith("CY")],
    }


def build_intraday_page(conn: sqlite3.Connection | None) -> dict:
    """SPY and QQQ hourly: the long shape, and the shape of the trading day."""
    if conn is None:
        return {}
    tickers = [row[0] for row in conn.execute(
        "SELECT DISTINCT ticker FROM bars WHERE frequency = '1hour' ORDER BY ticker"
    )]

    indexed: dict[str, list[list]] = {}
    hour_shape: dict[str, list[list]] = {}
    hour_volume: dict[str, list[list]] = {}
    stats: dict[str, dict] = {}

    for ticker in tickers:
        # Tiingo emits placeholder bars for closed markets: flat OHLC and zero
        # volume, about nine or ten a year. They are not trading and counting
        # them inflates the session count and flattens the return histogram.
        rows = conn.execute(
            "SELECT ts, close, volume FROM bars WHERE ticker = ? AND frequency = '1hour' "
            "AND NOT (volume = 0 AND open = high AND high = low AND low = close) "
            "ORDER BY ts", (ticker,)
        ).fetchall()
        if not rows:
            continue

        # Daily last-close, indexed to 100 at the first observation. Indexing is
        # what lets two instruments of different price share one axis instead of
        # inviting a second one.
        daily: dict[str, float] = {}
        for stamp, close, _ in rows:
            if close is not None:
                daily[stamp[:10]] = close
        days = sorted(daily)
        base = daily[days[0]]
        indexed[ticker] = [[day, _round(daily[day] / base * 100.0, 2)] for day in days]

        # Average return and volume by New York hour. The stored timestamps are
        # UTC, so the session slides an hour with US daylight saving; converting
        # first is what keeps 09:30 in one bucket year-round.
        by_hour_ret: dict[int, list[float]] = defaultdict(list)
        by_hour_vol: dict[int, list[float]] = defaultdict(list)
        previous = None
        for stamp, close, volume in rows:
            if close is None:
                continue
            moment = datetime.strptime(stamp[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
            hour = moment.astimezone(NY).hour
            if previous is not None and previous[0][:10] == stamp[:10]:
                by_hour_ret[hour].append((close / previous[1] - 1.0) * 10_000)  # basis points
            if volume:
                by_hour_vol[hour].append(volume)
            previous = (stamp, close)

        hour_shape[ticker] = [
            [hour, _round(sum(values) / len(values), 2)]
            for hour, values in sorted(by_hour_ret.items()) if values
        ]
        hour_volume[ticker] = [
            [hour, _round(sum(values) / len(values) / 1000.0, 1)]
            for hour, values in sorted(by_hour_vol.items()) if values
        ]
        stats[ticker] = {
            "bars": len(rows),
            "first": rows[0][0][:10],
            "last": rows[-1][0][:10],
            "days": len(days),
            "no_volume": sum(1 for r in rows if not r[2]),
        }

    # Recent detail: the last 45 sessions of hourly closes, where an hourly bar
    # is actually legible.
    recent: dict[str, list[list]] = {}
    for ticker in tickers:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=70)).strftime("%Y-%m-%d")
        rows = conn.execute(
            "SELECT ts, close FROM bars WHERE ticker = ? AND frequency = '1hour' AND ts >= ? "
            "AND NOT (volume = 0 AND open = high AND high = low AND low = close) "
            "ORDER BY ts", (ticker, cutoff),
        ).fetchall()
        recent[ticker] = [[r[0][:16].replace("T", " "), _round(r[1], 2)] for r in rows if r[1]]

    splits = conn.execute(
        "SELECT ticker, obs_date, split_factor FROM corporate_actions WHERE split_factor != 1.0"
    ).fetchall()
    return {
        "tickers": tickers,
        "indexed": indexed,
        "hour_shape": hour_shape,
        "hour_volume": hour_volume,
        "recent": recent,
        "stats": stats,
        "splits": [list(row) for row in splits],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("out/dashboard"))
    args = parser.parse_args(argv)

    macro, cftc, equity, intraday = (
        connect(MACRO_DB), connect(CFTC_DB), connect(EQUITY_DB), connect(INTRADAY_DB)
    )
    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "fed": build_fed_page(macro),
        "market": build_market_page(macro),
        "volatility": build_volatility_page(macro),
        "cftc": build_cftc_page(cftc),
        "earnings": build_earnings_page(equity),
        "intraday": build_intraday_page(intraday),
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    target = args.outdir / "data.json"
    target.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    size = target.stat().st_size

    print(f"wrote {target}  ({size/1024:.0f} KB)")
    for page in ("fed", "market", "volatility", "cftc", "earnings", "intraday"):
        block = payload[page]
        points = sum(
            len(v) for v in block.values() if isinstance(v, list)
        ) + sum(
            len(s) for v in block.values() if isinstance(v, dict)
            for s in v.values() if isinstance(s, list)
        )
        print(f"  {page:<10}{len(block):>3} blocks, {points:>7,} points")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        traceback.print_exc(limit=0)
        raise SystemExit(130)
