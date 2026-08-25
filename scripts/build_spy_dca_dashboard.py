"""Build a daily index-ETF core and hypothetical-DCA decision dashboard.

The account is deliberately simple: a persistent $10,000 ETF position at 1x.
Leverage is never retroactively applied to that core.  A separate, hypothetical
DCA ticket can use 1x/2x/3x only after the account has drawn down and only when
the CAPE and smoothed-VIX ceilings permit it.

Run with ``--refresh`` to update Yahoo daily ETF data, FRED volatility/rate data,
and the official Shiller workbook before rebuilding the self-contained page.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sqlite3
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_ACCOUNT = ROOT / "assets/spy_dca_account.json"
DEFAULT_CACHE = ROOT / ".cache/spy_dca_dashboard"
DEFAULT_YAHOO_CACHE = ROOT / ".cache/yahoo_daily/SPY.json"
DEFAULT_MACRO_DB = ROOT / "data/macro/macro.db"
DEFAULT_EQUITY_DB = ROOT / "data/equity/equity.db"
DEFAULT_OUTPUT = ROOT / "docs/spy-dca-dashboard.html"
DEFAULT_SNAPSHOT = ROOT / "out/strategy/spy_dca_dashboard/snapshot.json"


@dataclass(frozen=True)
class Signal:
    value: float
    ceiling: float
    label: str


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account", type=Path, default=DEFAULT_ACCOUNT)
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--volatility-series", default="VIXCLS")
    parser.add_argument("--volatility-label", default="VIX")
    parser.add_argument(
        "--fund-url",
        default="https://www.ssga.com/us/en/intermediary/etfs/state-street-spdr-sp-500-etf-trust-spy",
    )
    parser.add_argument("--updater-command", default="scripts/update_spy_dca_dashboard.ps1")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--yahoo-cache", type=Path, default=DEFAULT_YAHOO_CACHE)
    parser.add_argument("--macro-db", type=Path, default=DEFAULT_MACRO_DB)
    parser.add_argument("--equity-db", type=Path, default=DEFAULT_EQUITY_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args(argv)


def cape_signal(cape: float) -> Signal:
    if cape < 25.0:
        return Signal(cape, 3.0, "inexpensive")
    if cape <= 35.0:
        return Signal(cape, 2.0, "elevated")
    return Signal(cape, 1.0, "expensive")


def vix_signal(percentile: float) -> Signal:
    if percentile < 0.70:
        return Signal(percentile, 3.0, "normal")
    if percentile < 0.90:
        return Signal(percentile, 2.0, "stressed")
    return Signal(percentile, 1.0, "extreme")


def nav_dca_signal(drawdown: float) -> Signal:
    """Fresh-capital eligibility; the existing core always remains 1x.

    A drawdown is necessary, but not sufficient, for leveraged DCA.  This is
    intentionally the opposite of applying 3x to the whole account at a high.
    """
    if drawdown > -0.10:
        return Signal(drawdown, 1.0, "no drawdown gate")
    if drawdown > -0.20:
        return Signal(drawdown, 2.0, "drawdown gate open")
    return Signal(drawdown, 3.0, "deep drawdown gate")


def dca_leverage(drawdown: float, cape: float, vix_percentile: float) -> tuple[float, dict]:
    nav = nav_dca_signal(drawdown)
    valuation = cape_signal(cape)
    volatility = vix_signal(vix_percentile)
    applied = min(nav.ceiling, valuation.ceiling, volatility.ceiling)
    return applied, {"nav": nav, "cape": valuation, "vix": volatility}


def _unix(day: str) -> int:
    return int(pd.Timestamp(day, tz="UTC").timestamp())


def refresh_spy(path: Path, ticker: str = "SPY") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker, safe='')}"
        f"?period1={_unix('1993-01-01')}&period2={_unix('2035-01-01')}"
        "&interval=1d&events=div%2Csplits&includeAdjustedClose=true"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:  # nosec B310
        payload = response.read().decode("utf-8")
    parsed = json.loads(payload)
    if parsed.get("chart", {}).get("error") or not parsed.get("chart", {}).get("result"):
        raise RuntimeError(f"Yahoo returned no {ticker} data")
    path.write_text(payload, encoding="utf-8")


def load_spy(path: Path) -> pd.DataFrame:
    result = json.loads(path.read_text(encoding="utf-8"))["chart"]["result"][0]
    index = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_localize(None).normalize()
    raw = result["indicators"]["quote"][0]["close"]
    adjusted = result["indicators"].get("adjclose", [{}])[0].get("adjclose", raw)
    frame = pd.DataFrame({"close": raw, "adjusted": adjusted}, index=index)
    return frame.apply(pd.to_numeric, errors="coerce").dropna().sort_index()


def refresh_fred(series_id: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    request = urllib.request.Request(url, headers={"User-Agent": "savi-uz-research/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:  # nosec B310
        path.write_bytes(response.read())


def load_fred(series_id: str, cache: Path, macro_db: Path) -> pd.Series:
    path = cache / f"{series_id}.csv"
    if path.is_file():
        frame = pd.read_csv(path, parse_dates=["observation_date"])
        values = pd.to_numeric(frame[series_id], errors="coerce")
        series = pd.Series(values.to_numpy(), index=frame["observation_date"], dtype=float)
        series = series.dropna().sort_index()
        if not series.empty:
            return series
    connection = sqlite3.connect(f"file:{macro_db.resolve().as_posix()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT obs_date,value FROM observations WHERE series_id=? ORDER BY obs_date",
            (series_id,),
        ).fetchall()
    finally:
        connection.close()
    return pd.Series({pd.Timestamp(day): float(value) for day, value in rows if value is not None})


def refresh_shiller(path: Path) -> None:
    from savi_uz.equity_sources import ShillerClient

    source, rows = ShillerClient().fetch()
    usable = [row for row in rows if "cape" in row.values and "sp500_price" in row.values]
    if not usable:
        raise RuntimeError("Shiller workbook contains no usable CAPE row")
    row = usable[-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "obs_date": row.obs_date.isoformat(),
                "cape": row.values["cape"],
                "sp500_price": row.values["sp500_price"],
                "source": source,
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_shiller(cache: Path, equity_db: Path) -> dict:
    path = cache / "shiller_latest.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    connection = sqlite3.connect(f"file:{equity_db.resolve().as_posix()}?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT obs_date,cape,sp500_price,source_url FROM shiller_monthly "
            "WHERE cape IS NOT NULL AND sp500_price IS NOT NULL ORDER BY obs_date DESC LIMIT 1"
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise RuntimeError("no Shiller CAPE observation available")
    return {"obs_date": row[0], "cape": row[1], "sp500_price": row[2], "source": row[3]}


def _percentile_of_last(values: pd.Series, window: int = 252) -> float:
    clean = values.dropna()
    history = clean.iloc[-window:]
    if len(history) < 60:
        raise RuntimeError("not enough VIX history for a percentile")
    current = float(history.iloc[-1])
    return float((history <= current).mean())


def _asof_value(series: pd.Series, day: pd.Timestamp) -> tuple[pd.Timestamp, float]:
    eligible = series.loc[:day].dropna()
    if eligible.empty:
        raise RuntimeError(f"series contains no value on or before {day.date()}")
    return eligible.index[-1], float(eligible.iloc[-1])


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def _svg_chart(frame: pd.DataFrame, columns: list[tuple[str, str]], *, percent=False) -> str:
    width, height = 960, 270
    left, right, top, bottom = 58, 18, 18, 38
    plot_w, plot_h = width - left - right, height - top - bottom
    values = frame[[name for name, _ in columns]].astype(float)
    low, high = float(values.min().min()), float(values.max().max())
    if math.isclose(low, high):
        high = low + 1.0
    xs = np.linspace(left, left + plot_w, len(frame))
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="market history">']
    for tick in range(5):
        y = top + plot_h * tick / 4
        value = high - (high - low) * tick / 4
        label = _pct(value) if percent else f"{value:,.0f}"
        parts.append(f'<line class="grid" x1="{left}" x2="{left + plot_w}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="axis" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{label}</text>')
    for name, color in columns:
        ys = top + (high - values[name].to_numpy()) / (high - low) * plot_h
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.4"/>')
    for idx in (0, len(frame) // 2, len(frame) - 1):
        parts.append(f'<text class="axis" x="{xs[idx]:.1f}" y="{height - 12}" text-anchor="middle">{frame.index[idx].date()}</text>')
    parts.append("</svg>")
    return "".join(parts)


def build_snapshot(
    account: dict,
    spy: pd.DataFrame,
    vix: pd.Series,
    treasury: pd.Series,
    shiller: dict,
    *,
    ticker: str = "SPY",
    volatility_label: str = "VIX",
    valuation_proxy: pd.DataFrame | None = None,
    fund_url: str | None = None,
    updater_command: str = "scripts/update_spy_dca_dashboard.ps1",
) -> dict:
    entry_day = pd.Timestamp(account["entry_date"])
    entry_rows = spy.loc[:entry_day]
    if entry_rows.empty:
        raise RuntimeError(f"account entry date predates {ticker} data")
    entry_actual = entry_rows.index[-1]
    latest_day = spy.index[-1]
    latest = spy.iloc[-1]
    entry = spy.loc[entry_actual]
    initial = float(account["initial_investment"])
    account_value = initial * float(latest["adjusted"] / entry["adjusted"])
    effective_shares = account_value / float(latest["close"])
    since_entry = spy.loc[entry_actual:, "adjusted"]
    account_drawdown = float(since_entry.iloc[-1] / since_entry.cummax().iloc[-1] - 1.0)
    spy_drawdown = float(spy["adjusted"].iloc[-1] / spy["adjusted"].cummax().iloc[-1] - 1.0)
    ma200 = float(spy["close"].rolling(200).mean().iloc[-1])

    vix60 = vix.rolling(60, min_periods=40).mean().dropna()
    vix_day, vix_level = _asof_value(vix, latest_day)
    vix60_day, vix60_level = _asof_value(vix60, latest_day)
    vix_percentile = _percentile_of_last(vix60.loc[:latest_day])
    rate_day, rate = _asof_value(treasury, latest_day)

    shiller_day = pd.Timestamp(shiller["obs_date"])
    cape_proxy = valuation_proxy if valuation_proxy is not None else spy
    anchor_rows = cape_proxy.loc[: shiller_day + pd.offsets.MonthEnd(0)]
    anchor_raw = float(anchor_rows["close"].iloc[-1]) if not anchor_rows.empty else float(entry["close"])
    proxy_latest = float(cape_proxy.loc[:latest_day, "close"].iloc[-1])
    cape_estimate = float(shiller["cape"]) * proxy_latest / anchor_raw
    applied, signals = dca_leverage(account_drawdown, cape_estimate, vix_percentile)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "market_as_of": latest_day.date().isoformat(),
        "profile": {
            "ticker": ticker,
            "volatility_label": volatility_label,
            "fund_url": fund_url,
            "updater_command": updater_command,
            "cape_proxy": "S&P 500 / SPY",
        },
        "account": {
            "entry_date": entry_actual.date().isoformat(),
            "initial_investment": initial,
            "core_leverage": 1.0,
            "entry_price": float(entry["close"]),
            "current_price": float(latest["close"]),
            "effective_shares": effective_shares,
            "value": account_value,
            "profit_loss": account_value - initial,
            "return": account_value / initial - 1.0,
            "drawdown": account_drawdown,
        },
        "market": {
            "spy_price": float(latest["close"]),
            "spy_adjusted_high": float(spy["adjusted"].cummax().iloc[-1]),
            "spy_drawdown": spy_drawdown,
            "ma200": ma200,
            "above_ma200": float(latest["close"]) >= ma200,
            "vix": vix_level,
            "vix_as_of": vix_day.date().isoformat(),
            "vix_sma60": vix60_level,
            "vix_sma60_as_of": vix60_day.date().isoformat(),
            "vix_sma60_percentile": vix_percentile,
            "cape_estimate": cape_estimate,
            "cape_anchor": float(shiller["cape"]),
            "cape_anchor_date": shiller_day.date().isoformat(),
            "treasury_3m": rate / 100.0,
            "treasury_as_of": rate_day.date().isoformat(),
            "modeled_funding_hurdle": rate / 100.0 + 0.01,
        },
        "decision": {
            "dca_leverage": applied,
            "spy_cash_share": 0.80,
            "treasury_cash_share": 0.20,
            "nav_ceiling": signals["nav"].ceiling,
            "nav_label": signals["nav"].label,
            "cape_ceiling": signals["cape"].ceiling,
            "cape_label": signals["cape"].label,
            "vix_ceiling": signals["vix"].ceiling,
            "vix_label": signals["vix"].label,
        },
        "sources": {"shiller": shiller.get("source")},
    }


def build_page(snapshot: dict, spy: pd.DataFrame, vix: pd.Series) -> str:
    a, m, d = snapshot["account"], snapshot["market"], snapshot["decision"]
    profile = snapshot.get("profile", {})
    ticker = profile.get("ticker", "SPY")
    volatility_label = profile.get("volatility_label", "VIX")
    fund_url = profile.get(
        "fund_url",
        "https://www.ssga.com/us/en/intermediary/etfs/state-street-spdr-sp-500-etf-trust-spy",
    )
    updater_command = profile.get("updater_command", "scripts/update_spy_dca_dashboard.ps1")
    history = spy.iloc[-504:].copy()
    history["price100"] = history["adjusted"] / history["adjusted"].iloc[0] * 100.0
    history["ma200"] = spy["close"].rolling(200).mean().reindex(history.index)
    history["drawdown"] = history["adjusted"] / spy["adjusted"].cummax().reindex(history.index) - 1.0
    price_chart = _svg_chart(history.dropna(), [("close", "#182321"), ("ma200", "#118578")])
    dd_chart = _svg_chart(history, [("drawdown", "#b3483e")], percent=True)
    recommendation = "DCA at 1×" if d["dca_leverage"] == 1 else f'DCA tranche may use {d["dca_leverage"]:.0f}×'
    constraints = (
        (d["nav_ceiling"], "NAV drawdown"),
        (d["cape_ceiling"], "CAPE"),
        (d["vix_ceiling"], f"smoothed {volatility_label}"),
    )
    lowest = min(item[0] for item in constraints)
    reason = " and ".join(label for ceiling, label in constraints if ceiling == lowest)
    payload = html.escape(json.dumps({"leverage": d["dca_leverage"], "spyShare": d["spy_cash_share"]}))
    ladder = [
        (f"−10% {ticker} DD", "20% of episode Treasury", "Quality basket"),
        (f"−20% {ticker} DD", "30% of episode Treasury", f"{ticker} at 1×"),
        (f"−30% {ticker} DD", "30% of episode Treasury", "Quality basket"),
        (f"−50% {ticker} DD", "final 20%", f"{ticker} at 1×"),
    ]
    ladder_rows = "".join(
        f"<tr><td><strong>{level}</strong></td><td>{amount}</td><td>{asset}</td><td>{'Active' if m['spy_drawdown'] <= -float(level[1:3])/100 else 'Waiting'}</td></tr>"
        for level, amount, asset in ladder
    )
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{ticker} DCA decision dashboard</title>
<style>
:root{{--ink:#182321;--muted:#5f6f6b;--faint:#7f8d89;--paper:#fbfcf9;--bg:#edf1ed;--line:#cad4cf;--teal:#118578;--soft:#dceee9;--amber:#a46c00;--amberSoft:#f5ead1;--red:#b3483e;--redSoft:#f5dfdc;--mono:"Cascadia Mono",Consolas,monospace;--sans:Inter,system-ui,-apple-system,"Segoe UI",sans-serif;--serif:Georgia,"Times New Roman",serif}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 var(--sans)}}main{{width:min(1180px,calc(100% - 32px));margin:auto;padding:34px 0 70px}}header{{display:flex;justify-content:space-between;gap:24px;align-items:end;border-bottom:2px solid var(--ink);padding:16px 0 28px}}.eyebrow{{font:700 11px var(--mono);text-transform:uppercase;letter-spacing:.12em;color:var(--teal)}}h1{{font:600 clamp(37px,6vw,67px)/1 var(--serif);letter-spacing:-.045em;margin:9px 0 0;max-width:12ch}}h2{{font:600 28px/1.15 var(--serif);letter-spacing:-.025em;margin:0 0 9px}}p{{margin:0 0 14px}}.stamp{{text-align:right;color:var(--muted);font:12px/1.6 var(--mono)}}.hero{{display:grid;grid-template-columns:1.35fr .65fr;gap:16px;margin:22px 0}}.panel{{background:var(--paper);border:1px solid var(--line);padding:22px}}.decision{{border-top:6px solid var(--teal);padding:28px}}.decision h2{{font-size:43px;color:var(--teal)}}.decision .why{{color:var(--muted);max-width:65ch}}.answer{{background:var(--ink);color:white;border-radius:1px;padding:23px}}.answer b{{display:block;font:700 45px var(--mono)}}.answer span{{color:#c9d8d3}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}}.card{{background:var(--paper);border-top:4px solid var(--teal);padding:18px}}.card.warn{{border-color:var(--amber)}}.card.stop{{border-color:var(--red)}}.card b{{display:block;font:700 24px var(--mono)}}.card span{{display:block;color:var(--muted);font-size:12px;margin-top:6px}}section{{margin-top:42px}}.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.gate{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:18px}}.gate article{{background:var(--paper);border:1px solid var(--line);padding:16px}}.gate b{{display:block;font:700 24px var(--mono)}}.gate small{{color:var(--faint)}}label{{display:block;color:var(--muted);font-size:12px;margin:15px 0 5px}}input{{width:100%;font:700 25px var(--mono);padding:10px;border:1px solid var(--line);background:white;color:var(--ink)}}.ticket{{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);margin-top:15px}}.ticket div{{padding:15px;border-right:1px solid var(--line)}}.ticket div:last-child{{border:0}}.ticket b{{display:block;font:700 19px var(--mono)}}.ticket span{{font-size:11px;color:var(--faint)}}.note{{background:var(--soft);border-left:3px solid var(--teal);padding:16px 18px;color:var(--muted)}}.caution{{background:var(--amberSoft);border-left-color:var(--amber)}}.warning{{background:var(--redSoft);border-left-color:var(--red)}}figure{{margin:18px 0}}.figureHead{{display:flex;justify-content:space-between;color:var(--muted);font-size:12px;border-bottom:1px solid var(--line);padding-bottom:7px}}.plate{{background:var(--paper);border:1px solid var(--line);padding:12px;overflow:auto;margin-top:10px}}svg{{display:block;width:100%;min-width:690px}}.grid{{stroke:var(--line);stroke-width:1}}.axis{{font:10px var(--mono);fill:var(--faint)}}table{{width:100%;border-collapse:collapse;background:var(--paper);font-size:13px}}th,td{{padding:12px;text-align:left;border-bottom:1px solid var(--line)}}th{{font:700 10px var(--mono);letter-spacing:.07em;text-transform:uppercase;color:var(--faint)}}footer{{margin-top:45px;border-top:1px solid var(--line);padding-top:18px;color:var(--faint);font-size:11px}}a{{color:var(--teal)}}code{{font-family:var(--mono)}}@media(max-width:800px){{.hero,.grid2{{grid-template-columns:1fr}}.cards{{grid-template-columns:1fr 1fr}}.gate,.ticket{{grid-template-columns:1fr}}.ticket div{{border-right:0;border-bottom:1px solid var(--line)}}header{{display:block}}.stamp{{text-align:left;margin-top:16px}}}}@media(max-width:480px){{.cards{{grid-template-columns:1fr}}}}
</style></head><body><main data-ticket="{payload}">
<header><div><div class="eyebrow">Daily operating dashboard · hypothetical research policy</div><h1>Your {ticker} core, one decision at a time.</h1></div><div class="stamp">MARKET CLOSE {snapshot['market_as_of']}<br>BUILT {snapshot['generated_at'][:19].replace('T',' ')} UTC</div></header>
<div class="hero"><article class="panel decision"><div class="eyebrow">Today’s DCA decision</div><h2>{recommendation}</h2><p class="why">The binding ceiling is <strong>{reason}</strong>. The existing $10,000 core remains 1× regardless of this ticket. New leverage is considered only for fresh capital during a qualifying account drawdown.</p></article><aside class="answer"><span>Your original $10,000 is</span><b>1× {ticker}</b><span>$10,000 exposure · $0 borrowed at entry</span></aside></div>
<div class="cards"><article class="card"><b>{_money(a['value'])}</b><span>core value; P&amp;L {_money(a['profit_loss'])} ({_pct(a['return'])})</span></article><article class="card"><b>{a['effective_shares']:.3f}</b><span>effective {ticker} shares with distributions reinvested</span></article><article class="card warn"><b>{_pct(a['drawdown'])}</b><span>account drawdown from its own post-entry high</span></article><article class="card stop"><b>{d['dca_leverage']:.0f}×</b><span>maximum leverage for a hypothetical new DCA tranche today</span></article></div>

<section><div class="grid2"><div><div class="eyebrow">Account state</div><h2>The core is simple on purpose.</h2><p>You invested {_money(a['initial_investment'])} on {a['entry_date']} at approximately ${a['entry_price']:,.2f}. It is modeled as {ticker} total return, so distributions are reinvested. There is no financing cost because the core is 1×.</p><p class="note"><strong>Do not change the core to 3× at a high.</strong> The backtest’s large terminal values came with roughly 50–60% drawdowns and path-sensitive switching. This dashboard confines any leverage experiment to separately tracked future deposits.</p></div><div class="panel"><div class="eyebrow">Hypothetical contribution ticket</div><h2>What if I add money today?</h2><label for="dca">New cash contribution (USD)</label><input id="dca" type="number" min="0" step="100" value="10000"><div class="ticket"><div><b id="spyCash">—</b><span>cash assigned to {ticker}</span></div><div><b id="treasuryCash">—</b><span>cash retained in Treasury</span></div><div><b id="grossExposure">—</b><span>gross {ticker} exposure</span></div><div><b id="borrowed">—</b><span>financed amount</span></div></div><p class="caution" style="margin-top:14px"><strong>Execution interpretation:</strong> this shows the 80/20 research policy. If your instruction is instead “invest every dollar in {ticker},” use the full contribution at today’s displayed leverage; do not also count a Treasury sleeve.</p></div></div></section>

<section><div class="eyebrow">Market conditions</div><h2>Three independent ceilings; use the lowest.</h2><div class="gate"><article><small>Account NAV gate</small><b>{d['nav_ceiling']:.0f}×</b><p>{_pct(a['drawdown'])} · {d['nav_label']}</p></article><article><small>Broad-market CAPE ceiling</small><b>{d['cape_ceiling']:.0f}×</b><p>est. {m['cape_estimate']:.1f} · {d['cape_label']}</p></article><article><small>Smoothed-{volatility_label} ceiling</small><b>{d['vix_ceiling']:.0f}×</b><p>{_pct(m['vix_sma60_percentile'],0)} percentile · {d['vix_label']}</p></article></div><div class="cards"><article class="card"><b>${m['spy_price']:,.2f}</b><span>{ticker} close; {_pct(m['spy_drawdown'])} from total-return high</span></article><article class="card"><b>${m['ma200']:,.2f}</b><span>200-session average; {ticker} is {'above' if m['above_ma200'] else 'below'}</span></article><article class="card warn"><b>{m['vix']:.2f}</b><span>{volatility_label} as of {m['vix_as_of']}; 60-day average {m['vix_sma60']:.2f}</span></article><article class="card stop"><b>{_pct(m['modeled_funding_hurdle'],1)}</b><span>illustrative financing hurdle: 3m Treasury + 1%</span></article></div><p class="caution note"><strong>CAPE is slow, broad-market data.</strong> The displayed {m['cape_estimate']:.1f} is an S&amp;P 500 price-scaled estimate anchored to Shiller’s published {m['cape_anchor']:.1f} on {m['cape_anchor_date']}; it is not a {ticker}-specific valuation or an official real-time quote. {volatility_label} and Treasury observations can also lag the {ticker} close—always check the as-of dates.</p></section>

<section><div class="grid2"><figure><div class="figureHead"><strong>{ticker} close and 200-day average</strong><span>last 504 sessions</span></div><div class="plate">{price_chart}</div></figure><figure><div class="figureHead"><strong>{ticker} total-return drawdown</strong><span>last 504 sessions</span></div><div class="plate">{dd_chart}</div></figure></div></section>

<section><div class="eyebrow">Treasury deployment ladder</div><h2>Separate cash deployment from leverage.</h2><p>The 20% Treasury target belongs to new contributions and accumulated harvests; it is not created by borrowing. Each drawdown rung is measured from {ticker}’s total-return high and uses a share of the Treasury that existed at the start of that drawdown episode.</p><table><thead><tr><th>{ticker} drawdown</th><th>Deploy</th><th>Destination</th><th>Today</th></tr></thead><tbody>{ladder_rows}</tbody></table><p class="warning note" style="margin-top:15px"><strong>No automatic order placement.</strong> A dashboard signal is a review prompt. Before using 2× or 3×, define the instrument, financing and rebalance costs, maximum gross-account exposure, tax treatment, and a forced-liquidation rule. Never fund it with money required for living costs or secured against a home.</p></section>

<section><div class="eyebrow">Exact rulebook</div><h2>What changes the next DCA ticket?</h2><table><thead><tr><th>Gate</th><th>3× eligible</th><th>2× ceiling</th><th>1× ceiling</th></tr></thead><tbody><tr><td><strong>Account NAV drawdown</strong></td><td>−20% or deeper</td><td>−10% to −20%</td><td>less than −10%</td></tr><tr><td><strong>Broad-market CAPE estimate</strong></td><td>below 25</td><td>25–35</td><td>above 35</td></tr><tr><td><strong>60-day {volatility_label} rank vs prior year</strong></td><td>below 70th pct</td><td>70th–90th pct</td><td>90th pct or higher</td></tr></tbody></table><p class="note" style="margin-top:15px">Applied DCA leverage = the minimum of the three ceilings. At a new account high, the NAV gate is 1×—not 3×. A leveraged DCA lot returns to 1× once the account regains its previous high; the permanent core is already 1×.</p></section>

<footer><strong>Update:</strong> run <code>powershell -ExecutionPolicy Bypass -File {updater_command}</code>. It refreshes the inputs and rewrites this self-contained page; schedule that command once per trading day if desired. Sources: <a href="{fund_url}">{ticker} sponsor page</a>, <a href="https://fred.stlouisfed.org/series/{'VXNCLS' if volatility_label == 'VXN' else 'VIXCLS'}">FRED {volatility_label}</a>, <a href="https://fred.stlouisfed.org/series/DGS3MO">FRED DGS3MO</a>, and <a href="https://www.econ.yale.edu/~shiller/data.htm">Robert Shiller / Yale</a>. Yahoo daily prices are used for calculations. Historical/hypothetical research, not personalized investment advice.</footer>
</main><script>
const root=document.querySelector('main');const cfg=JSON.parse(root.dataset.ticket);const input=document.getElementById('dca');const usd=new Intl.NumberFormat('en-US',{{style:'currency',currency:'USD',maximumFractionDigits:0}});function update(){{const cash=Math.max(0,Number(input.value)||0);const spy=cash*cfg.spyShare;const treasury=cash-spy;const gross=spy*cfg.leverage;document.getElementById('spyCash').textContent=usd.format(spy);document.getElementById('treasuryCash').textContent=usd.format(treasury);document.getElementById('grossExposure').textContent=usd.format(gross);document.getElementById('borrowed').textContent=usd.format(Math.max(0,gross-spy));}}input.addEventListener('input',update);update();
</script></body></html>'''


def main(argv=None) -> int:
    args = parse_args(argv)
    errors = []
    if args.refresh:
        refresh_actions = [
            (args.ticker, lambda: refresh_spy(args.yahoo_cache, args.ticker)),
            (
                args.volatility_label,
                lambda: refresh_fred(
                    args.volatility_series, args.cache / f"{args.volatility_series}.csv"
                ),
            ),
            ("Treasury", lambda: refresh_fred("DGS3MO", args.cache / "DGS3MO.csv")),
            ("CAPE", lambda: refresh_shiller(args.cache / "shiller_latest.json")),
        ]
        if args.ticker.upper() != "SPY":
            refresh_actions.insert(
                1, ("SPY valuation proxy", lambda: refresh_spy(DEFAULT_YAHOO_CACHE, "SPY"))
            )
        for label, action in refresh_actions:
            try:
                action()
                print(f"refreshed {label}")
            except Exception as exc:
                errors.append(f"{label}: {exc}")
                print(f"warning: {label} refresh failed; using local fallback: {exc}", file=sys.stderr)

    account = json.loads(args.account.read_text(encoding="utf-8"))
    spy = load_spy(args.yahoo_cache)
    vix = load_fred(args.volatility_series, args.cache, args.macro_db)
    treasury = load_fred("DGS3MO", args.cache, args.macro_db)
    shiller = load_shiller(args.cache, args.equity_db)
    valuation_proxy = spy if args.ticker.upper() == "SPY" else load_spy(DEFAULT_YAHOO_CACHE)
    snapshot = build_snapshot(
        account,
        spy,
        vix,
        treasury,
        shiller,
        ticker=args.ticker.upper(),
        volatility_label=args.volatility_label,
        valuation_proxy=valuation_proxy,
        fund_url=args.fund_url,
        updater_command=args.updater_command,
    )
    snapshot["refresh_warnings"] = errors
    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    args.snapshot.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_page(snapshot, spy, vix), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
