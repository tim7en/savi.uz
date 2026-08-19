"""Download end-of-day option chains from Alpha Vantage.

The existing gamma series covers 250 sessions and no crisis, which is the single
biggest weakness in the overlay work.  Alpha Vantage serves full historical
chains back to 2008 with vendor-supplied greeks, full strike coverage and every
expiration, so the same measurements can be rebuilt across 2018, 2020 and 2022.

Design notes:

* **Aggregates come from the whole chain, contracts are stored filtered.**  The
  daily gamma and surface features are computed before any truncation, so the
  series carries no moneyness or maturity bias.  Only a bounded window of
  contracts is persisted, because the full chains would run to tens of gigabytes
  and gamma outside +/-15% of spot is negligible.
* **Resumable.**  Every (symbol, date) already stored is skipped, so the run can
  be interrupted and restarted without duplicating work or wasting requests.
* **Spot comes from the bar database**, not from the chain.  SPY and QQQ have
  never split, so their unadjusted closes are the correct reference for raw
  strike prices.
* **The key is read from .env and never logged.**

Terms: Alpha Vantage's Alpha X data is licensed for personal use and may not be
redistributed.  This stores it locally for private research only.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from savi_uz.option_features import Contract, snapshot_features  # noqa: E402

ENDPOINT = "https://www.alphavantage.co/query"
SCHEMA = """
CREATE TABLE IF NOT EXISTS av_daily (
  symbol TEXT, observation_date TEXT, spot REAL,
  contracts INTEGER, usable INTEGER,
  call_gex REAL, put_gex REAL, net_gex REAL, absolute_gex REAL,
  atm_iv REAL, iv_term_slope REAL, skew_moneyness REAL, skew_25delta REAL,
  put_call_oi REAL, put_call_volume REAL, gamma_balance REAL,
  gamma_flip_distance REAL, zero_dte_share REAL, vanna REAL,
  total_oi REAL, total_volume REAL,
  PRIMARY KEY (symbol, observation_date));
CREATE TABLE IF NOT EXISTS av_contracts (
  symbol TEXT, observation_date TEXT, expiration TEXT, dte INTEGER,
  side TEXT, strike REAL, iv REAL, delta REAL, gamma REAL,
  open_interest REAL, volume REAL, bid REAL, ask REAL);
CREATE INDEX IF NOT EXISTS av_contracts_key
  ON av_contracts(symbol, observation_date);
CREATE TABLE IF NOT EXISTS av_fetch_log (
  symbol TEXT, observation_date TEXT, status TEXT, rows INTEGER,
  message TEXT, fetched_at TEXT);
"""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=pathlib.Path,
                        default=pathlib.Path("data/options/alphavantage.db"))
    parser.add_argument("--bars", type=pathlib.Path,
                        default=pathlib.Path("data/intraday/bars.db"))
    parser.add_argument("--symbols", nargs="+", default=["SPY", "QQQ"])
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--workers", type=int, default=3,
                        help="concurrent requests; keep well under the plan's limit")
    parser.add_argument("--keep-moneyness", type=float, default=0.15,
                        help="store contracts within this fraction of spot")
    parser.add_argument("--keep-dte", type=int, default=120)
    parser.add_argument("--no-contracts", action="store_true",
                        help="store only the daily aggregates, skipping the "
                             "per-contract table -- the daily features are what "
                             "the strategy consumes, and the contract rows cost "
                             "roughly a gigabyte per symbol-pair")
    parser.add_argument("--limit", type=int, default=0, help="stop after N fetches")
    return parser.parse_args(argv)


def api_key() -> str:
    for line in pathlib.Path(".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            name, value = line.split("=", 1)
            if name.strip() == "ALPHAVANTAGE_API_KEY":
                return value.strip().strip('"').strip("'")
    raise SystemExit("error: ALPHAVANTAGE_API_KEY missing from .env")


def trading_days(bars_path: pathlib.Path, symbol: str, start: str, end: str):
    """Sessions and closes taken from the bar database."""
    connection = sqlite3.connect(f"file:{bars_path}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT DISTINCT substr(ts,1,10) FROM bars WHERE ticker=? AND "
        "frequency='5min' AND ts>=? AND ts<? ORDER BY 1", (symbol, start, end)
    ).fetchall()
    closes = {}
    for (day,) in rows:
        last = connection.execute(
            "SELECT close FROM bars WHERE ticker=? AND frequency='5min' AND "
            "substr(ts,1,10)=? ORDER BY ts DESC LIMIT 1", (symbol, day)).fetchone()
        if last:
            closes[day] = float(last[0])
    connection.close()
    return closes


def fetch(symbol: str, day: str, key: str, attempts: int = 4):
    params = {"function": "HISTORICAL_OPTIONS", "symbol": symbol,
              "date": day, "apikey": key}
    url = f"{ENDPOINT}?{urllib.parse.urlencode(params)}"
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = json.loads(response.read().decode())
        except Exception as error:                      # transport or decode
            time.sleep(2 * (attempt + 1))
            if attempt == attempts - 1:
                return None, f"transport: {error}"
            continue
        for gate in ("Note", "Information"):
            if gate in payload:                          # rate limited, back off
                time.sleep(15 * (attempt + 1))
                if attempt == attempts - 1:
                    return None, f"{gate}: {str(payload[gate])[:120]}"
                break
        else:
            if "Error Message" in payload:
                return None, f"error: {str(payload['Error Message'])[:120]}"
            return payload.get("data", []) or [], "ok"
    return None, "exhausted retries"


def to_number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build(rows, spot, day):
    """Contracts for the feature layer, plus the raw gamma aggregation."""
    contracts, call_gex, put_gex, absolute = [], 0.0, 0.0, 0.0
    usable = 0
    for row in rows:
        side = row.get("type")
        if side not in ("call", "put"):
            continue
        strike = to_number(row.get("strike"), None)
        if not strike:
            continue
        expiration = (row.get("expiration") or "")[:10]
        try:
            dte = (date.fromisoformat(expiration) - date.fromisoformat(day)).days
        except ValueError:
            continue
        iv = to_number(row.get("implied_volatility"), None)
        gamma = to_number(row.get("gamma"), None)
        delta = to_number(row.get("delta"), None)
        oi = to_number(row.get("open_interest"))
        volume = to_number(row.get("volume"))
        contracts.append({
            "expiration": expiration, "dte": max(dte, 0), "side": side,
            "strike": strike, "iv": iv, "delta": delta, "gamma": gamma,
            "open_interest": oi, "volume": volume,
            "bid": to_number(row.get("bid")), "ask": to_number(row.get("ask")),
        })
        if gamma and oi:
            usable += 1
            # Standard convention: dealers long calls, short puts. Dollar gamma
            # per 1% move, with the 100-share contract multiplier.
            dollars = gamma * oi * 100.0 * spot * spot * 0.01
            absolute += abs(dollars)
            if side == "call":
                call_gex += dollars
            else:
                put_gex += dollars
    return contracts, call_gex, put_gex, absolute, usable


def main(argv=None):
    args = parse_args(argv)
    key = api_key()
    args.db.parent.mkdir(parents=True, exist_ok=True)
    store = sqlite3.connect(args.db)
    store.executescript(SCHEMA)
    store.commit()

    todo = []
    for symbol in args.symbols:
        closes = trading_days(args.bars, symbol, args.start, args.end)
        have = {d for (d,) in store.execute(
            "SELECT observation_date FROM av_daily WHERE symbol=?", (symbol,))}
        for day, spot in sorted(closes.items()):
            if day not in have:
                todo.append((symbol, day, spot))
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo):,} (symbol, session) pairs to fetch "
          f"across {args.symbols}", flush=True)
    if not todo:
        return 0

    done = {"n": 0, "rows": 0, "fail": 0}
    started = time.time()

    def work(item):
        symbol, day, spot = item
        rows, status = fetch(symbol, day, key)
        if rows is None:
            return symbol, day, spot, None, status
        return symbol, day, spot, rows, status

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for symbol, day, spot, rows, status in pool.map(work, todo):
            stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            if rows is None:
                store.execute("INSERT INTO av_fetch_log VALUES (?,?,?,?,?,?)",
                              (symbol, day, "fail", 0, status, stamp))
                done["fail"] += 1
                store.commit()
                continue
            contracts, call_gex, put_gex, absolute, usable = build(rows, spot, day)
            feature_rows = [
                Contract(c["side"], c["strike"], c["dte"], c["iv"], c["gamma"],
                         c["open_interest"], c["volume"])
                for c in contracts
            ]
            feats = snapshot_features(feature_rows, spot)
            store.execute(
                "INSERT OR REPLACE INTO av_daily VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (symbol, day, spot, len(contracts), usable,
                 call_gex, put_gex, call_gex - put_gex, absolute,
                 feats.get("atm_iv"), feats.get("iv_term_slope"),
                 feats.get("skew_moneyness"), feats.get("skew_25delta"),
                 feats.get("put_call_oi"), feats.get("put_call_volume"),
                 feats.get("gamma_balance"), feats.get("gamma_flip_distance"),
                 feats.get("zero_dte_share"), feats.get("vanna"),
                 feats.get("total_oi"), feats.get("total_volume")))
            if not args.no_contracts:
                keep = [
                    (symbol, day, c["expiration"], c["dte"], c["side"], c["strike"],
                     c["iv"], c["delta"], c["gamma"], c["open_interest"],
                     c["volume"], c["bid"], c["ask"])
                    for c in contracts
                    if c["dte"] <= args.keep_dte
                    and abs(c["strike"] / spot - 1.0) <= args.keep_moneyness
                ]
                store.executemany(
                    "INSERT INTO av_contracts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", keep)
            store.execute("INSERT INTO av_fetch_log VALUES (?,?,?,?,?,?)",
                          (symbol, day, "ok", len(rows), "", stamp))
            store.commit()
            done["n"] += 1
            done["rows"] += len(rows)
            if done["n"] % 50 == 0:
                rate = done["n"] / max(time.time() - started, 1e-9)
                left = (len(todo) - done["n"] - done["fail"]) / max(rate, 1e-9)
                print(f"  {done['n']:5,d}/{len(todo):,}  {done['rows']:,} chain rows  "
                      f"{rate * 60:.0f}/min  ~{left / 60:.0f} min left  "
                      f"fails {done['fail']}", flush=True)

    print(f"\ndone: {done['n']:,} sessions stored, {done['fail']} failures, "
          f"{done['rows']:,} chain rows seen")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
