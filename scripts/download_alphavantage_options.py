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
  and gamma outside +/-20% of spot is negligible.
* **Resumable.**  Every (symbol, date) already stored is skipped, so the run can
  be interrupted and restarted without duplicating work or wasting requests.
* **Spot comes from unadjusted closes** (``raw_closes``), not from the chain
  and not from the bar database.  Bars are back-adjusted for splits and
  dividends while strikes are not, so an adjusted spot compares two different
  units and corrupts every moneyness-dependent feature.
* **The key is read from .env and never logged.**

Terms: Alpha Vantage's Alpha X data is licensed for personal use and may not be
redistributed.  This stores it locally for private research only.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from savi_uz.option_features import Contract, snapshot_features  # noqa: E402

ENDPOINT = "https://www.alphavantage.co/query"

# Alpha Vantage's option endpoint names share classes without a separator, while
# the bar database follows the hyphenated convention used everywhere else here.
# Only the outbound request is translated: rows stay keyed on the local symbol so
# they still join against bars and the rest of the project. Without this, every
# BRK-B request comes back "No data for symbol BRK-B" and silently stores an
# empty session.
API_SYMBOLS = {"BRK-B": "BRKB"}

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
    parser.add_argument("--keep-moneyness", type=float, default=0.20,
                        help="store contracts within this fraction of spot")
    parser.add_argument("--keep-dte", type=int, default=45)
    parser.add_argument("--no-contracts", action="store_true",
                        help="store only the daily aggregates, skipping the "
                             "per-contract table -- the daily features are what "
                             "the strategy consumes; the contract rows are what "
                             "makes a feature fix recomputable without re-fetching")
    parser.add_argument("--targets", type=pathlib.Path, default=None,
                        help="JSON list of [symbol, date] pairs to fetch, instead "
                             "of every session in a date range")
    parser.add_argument("--limit", type=int, default=0, help="stop after N fetches")
    return parser.parse_args(argv)


def acquire_lock(db_path: pathlib.Path):
    """Refuse to start when another copy of this script is already downloading.

    ``av_contracts`` carries no unique constraint, so two instances racing on the
    same (symbol, date) both insert a full chain and the duplicates are invisible
    to the resume logic, which only consults ``av_daily``. A run that overlapped
    another one this way produced 14% duplicate contract rows before it was
    noticed. They also split one rate-limit allowance between them, so each
    appears to be running at a fraction of its true speed.
    """
    lock = db_path.with_suffix(".lock")
    if lock.exists():
        try:
            pid = int(lock.read_text().strip())
        except (ValueError, OSError):
            pid = None
        if pid is not None:
            try:
                os.kill(pid, 0)                 # signal 0 only tests existence
            except OSError:
                pass                            # stale lock, previous run died
            else:
                raise SystemExit(
                    f"error: another download is already running (pid {pid}). "
                    f"Stop it first, or remove {lock} if that pid is dead.")
    lock.write_text(str(os.getpid()), encoding="utf-8")
    return lock


def api_key() -> str:
    for line in pathlib.Path(".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            name, value = line.split("=", 1)
            if name.strip() == "ALPHAVANTAGE_API_KEY":
                return value.strip().strip('"').strip("'")
    raise SystemExit("error: ALPHAVANTAGE_API_KEY missing from .env")


def session_closes(store, symbol: str, start: str, end: str):
    """Sessions and *unadjusted* closes -- the reference that matches strikes.

    Spot used to come from the bar database, whose prices are back-adjusted for
    splits and dividends.  Strikes are not adjusted, so that comparison ran in
    two different units: a pre-split AAPL session priced spot near $120 against
    a chain struck near $500, and every moneyness-dependent feature -- ATM
    volatility, skew, the gamma flip -- was measured against the wrong end of
    the curve.  Dividends did the same thing quietly, drifting a steady payer a
    few percent a year with no split anywhere.

    ``raw_closes`` is populated by download_raw_closes.py and holds the price as
    it actually printed, so the two sides now agree.
    """
    return {
        day: float(close)
        for day, close in store.execute(
            "SELECT observation_date, raw_close FROM raw_closes "
            "WHERE symbol=? AND observation_date>=? AND observation_date<? "
            "AND raw_close IS NOT NULL AND raw_close>0 "
            "ORDER BY observation_date", (symbol, start, end))
    }


class _Pacer:
    """Even spacing between request starts, shared across worker threads.

    Backing off only *after* the vendor complains is too late: a burst above the
    account's per-minute allowance gets the whole run throttled, and throughput
    then collapses to a fraction of the nominal rate for hours. Spacing the
    starts keeps the run inside the allowance instead of recovering from it.
    """

    def __init__(self, requests_per_minute: float):
        self.interval = 60.0 / max(requests_per_minute, 1e-9)
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_at - now)
            self.next_at = max(now, self.next_at) + self.interval
        if delay:
            time.sleep(delay)


PACER = _Pacer(70.0)


def fetch(symbol: str, day: str, key: str, attempts: int = 4):
    params = {"function": "HISTORICAL_OPTIONS",
              "symbol": API_SYMBOLS.get(symbol, symbol),
              "date": day, "apikey": key}
    url = f"{ENDPOINT}?{urllib.parse.urlencode(params)}"
    for attempt in range(attempts):
        PACER.wait()
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
    lock = acquire_lock(args.db)
    store = sqlite3.connect(args.db)
    # Write-ahead logging, because the contract table makes this write-heavy: the
    # default rollback journal creates, fsyncs and deletes a journal file on every
    # commit, and with ~330 contract rows per session that overhead grew until it,
    # not the vendor's rate limit, set the pace (70/min -> 56/min over the first
    # few hundred sessions, against a flat 70/min when only aggregates were
    # stored). synchronous=NORMAL skips the per-commit fsync; in WAL mode that
    # risks losing the last few transactions on a power cut but cannot corrupt the
    # database -- and the run is resumable, so a lost minute costs a minute.
    store.execute("PRAGMA journal_mode=WAL")
    store.execute("PRAGMA synchronous=NORMAL")
    store.executescript(SCHEMA)
    store.commit()

    todo = []
    if args.targets:
        # An explicit (symbol, date) list: fetch only the sessions an event study
        # actually needs, rather than every session in a range. Spot still comes
        # from the bar store, so a pair with no bar is skipped rather than guessed.
        wanted = defaultdict(set)
        for symbol, day in json.loads(args.targets.read_text(encoding="utf-8")):
            wanted[symbol].add(day)
        for symbol in sorted(wanted):
            closes = session_closes(store, symbol, "2000-01-01", "2100-01-01")
            have = {d for (d,) in store.execute(
                "SELECT observation_date FROM av_daily WHERE symbol=?", (symbol,))}
            for day in sorted(wanted[symbol]):
                if day not in have and day in closes:
                    todo.append((symbol, day, closes[day]))
    else:
        for symbol in args.symbols:
            closes = session_closes(store, symbol, args.start, args.end)
            if not closes:
                print(f"  skipping {symbol}: no raw closes -- run "
                      f"download_raw_closes.py first", flush=True)
                continue
            have = {d for (d,) in store.execute(
                "SELECT observation_date FROM av_daily WHERE symbol=?", (symbol,))}
            for day, spot in sorted(closes.items()):
                if day not in have:
                    todo.append((symbol, day, spot))
    if args.limit:
        todo = todo[:args.limit]
    scope = (f"from {args.targets}" if args.targets
             else f"across {len(args.symbols)} symbols")
    print(f"{len(todo):,} (symbol, session) pairs to fetch {scope}", flush=True)
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
