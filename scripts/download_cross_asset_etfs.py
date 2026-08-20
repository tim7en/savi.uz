"""Daily OHLC for non-equity ETFs, from Alpha Vantage.

The equity book already holds six non-equity instruments, and their daily R
stream correlates +0.03 with the equity sleeve -- +0.004 on the equity sleeve's
down days.  Six such names lifted Sharpe as much as thirty-two more equities did,
so the cheapest remaining improvement is more of them rather than more stocks.

Every symbol here is tagged with what, if anything, is tradeable on Binance.  The
honest picture is that Binance carries gold and silver and the major FX crosses
and nothing else in this list -- no oil, no bonds, no broad commodity, no
volatility -- so a Binance-only book cannot hold most of what makes this sleeve
work.  The tags are recorded rather than used as a filter: knowing that duration
is the strongest diversifier and also untradeable there is more useful than
quietly dropping it.

Written to its own database rather than into ``bars.db``, because the options
download is holding that file for hours at a time and a writer competing with it
buys nothing.  The schema matches, so ``--bars`` on any existing script points
here unchanged.
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

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    ticker     TEXT NOT NULL,
    frequency  TEXT NOT NULL,
    ts         TEXT NOT NULL,
    open       REAL,
    high       REAL,
    low        REAL,
    close      REAL,
    volume     REAL,
    PRIMARY KEY (ticker, frequency, ts)
);
CREATE INDEX IF NOT EXISTS idx_bars_ticker_ts ON bars (ticker, ts);
CREATE TABLE IF NOT EXISTS venue_map (
    ticker    TEXT PRIMARY KEY,
    sleeve    TEXT,
    binance   TEXT
);
"""

#: ticker -> (sleeve, what trades on Binance for the same exposure, or None)
UNIVERSE = {
    # metals -- the one place Binance has real coverage
    "GLD":  ("metals", "PAXGUSDT / XAUUSDT"),
    "SLV":  ("metals", "XAGUSDT"),
    "GDX":  ("metals", None),
    "PPLT": ("metals", None),
    # energy -- nothing on Binance
    "USO":  ("energy", None),
    "UNG":  ("energy", None),
    "BNO":  ("energy", None),
    # broad commodity and agriculture -- nothing on Binance
    "DBC":  ("commodity", None),
    "DBA":  ("commodity", None),
    "GSG":  ("commodity", None),
    # duration and credit -- nothing on Binance, and the strongest diversifier
    "TLT":  ("duration", None),
    "IEF":  ("duration", None),
    "SHY":  ("duration", None),
    "TIP":  ("duration", None),
    "LQD":  ("credit", None),
    "HYG":  ("credit", None),
    # dollar and FX -- Binance lists the major crosses
    "UUP":  ("fx", "USDT complex"),
    "FXE":  ("fx", "EURUSDT"),
    "FXY":  ("fx", "JPY pairs, thin"),
    "FXB":  ("fx", "GBPUSDT"),
    # volatility -- nothing on Binance
    "VXX":  ("volatility", None),
    # US equity sectors. Added after the macro-regime study found nothing: that
    # book held commodities, duration, FX and volatility, so the contrast with
    # the clearest rate mechanism -- utilities and real estate as bond proxies,
    # financials earning on a steeper curve, technology as long-duration growth
    # -- was absent from it entirely. The SPDR sectors carry history to 1998.
    "XLF":  ("sector: financials", None),
    "XLU":  ("sector: utilities", None),
    "XLK":  ("sector: technology", None),
    "XLV":  ("sector: health care", None),
    "XLP":  ("sector: staples", None),
    "XLI":  ("sector: industrials", None),
    "XLY":  ("sector: discretionary", None),
    "XLB":  ("sector: materials", None),
    "XLRE": ("sector: real estate", None),
    "XLC":  ("sector: communications", None),
    "XLE":  ("sector: energy", None),
}


#: Equity substitutes.  Everything above prices what stocks are valued *against*
#: -- the curve, the dollar, the commodity complex -- and nothing in it is a
#: stock.  That gap matters because the measured weakness of the equity book is
#: not its rules but its universe: twenty-nine single names picked in 2026,
#: which compounded at 34.2% a year simply held, so a long-only breakout system
#: on that list looks good whatever it does.
#:
#: A fund cannot be picked that way.  Its constituents rotate by published rule,
#: the sponsor never asks whether the holdings did well, and SPY in 2007 is SPY
#: in 2026.  The selection rule here is therefore stated in advance and contains
#: no reference to outcome: *every US-listed fund giving equity exposure -- broad
#: index, style box, industry, or country -- that was trading before 2007 and is
#: liquid enough to short.*  Post-2007 entries are tagged so they can be dropped.
#:
#: One residual bias survives and is worth naming rather than hiding: funds do
#: close, and a list drawn today cannot contain the ones that did.  It is far
#: smaller than the single-name version -- the rule selects on breadth and age,
#: not on return -- but it is not zero.
#:
#: All of these short freely and all are optionable, which is what "goes both
#: ways" has to mean once the crypto venue is off the table.
EQUITY_SUBSTITUTES = {
    # broad US market -- the direct substitute for holding stocks
    "SPY":  ("index: large cap", None),
    "DIA":  ("index: mega cap", None),
    "QQQ":  ("index: nasdaq 100", None),
    "MDY":  ("index: mid cap", None),
    "IWM":  ("index: small cap", None),
    "RSP":  ("index: equal weight", None),
    # the Russell style grid, listed as a set in 2000 and unchanged since.
    # Kept whole rather than sampled: taking three of six would be a choice, and
    # the point of the rule is that no choice is made.
    "IWF":  ("style: large growth", None),
    "IWD":  ("style: large value", None),
    "IWP":  ("style: mid growth", None),
    "IWS":  ("style: mid value", None),
    "IWO":  ("style: small growth", None),
    "IWN":  ("style: small value", None),
    # industry funds, finer than the eleven sectors.  A breakout is an industry
    # event more often than a sector one -- semis move without technology, banks
    # without financials -- and the sector fund averages that away.
    "SMH":  ("industry: semiconductors", None),
    "IBB":  ("industry: biotech", None),
    "XBI":  ("industry: biotech equal wt", None),
    "KRE":  ("industry: regional banks", None),
    "KBE":  ("industry: banks", None),
    "ITB":  ("industry: homebuilders", None),
    "XHB":  ("industry: home supply", None),
    "XOP":  ("industry: oil and gas E&P", None),
    "OIH":  ("industry: oil services", None),
    "XME":  ("industry: metals and mining", None),
    "IYT":  ("industry: transports", None),
    "XRT":  ("industry: retail", None),
    "ITA":  ("industry: aerospace defence", None),
    "IYR":  ("industry: real estate", None),
    "GDXJ": ("industry: junior miners", None),
    # country and region.  The single-name book holds three of these already
    # (EWJ, EWT, EWY); the rest of the developed and emerging list is here for
    # the same reason the style grid is kept whole.
    "EFA":  ("region: developed ex-US", None),
    "EEM":  ("region: emerging", None),
    "VGK":  ("region: europe", None),
    "ILF":  ("region: latin america", None),
    "FXI":  ("country: china", None),
    "EWZ":  ("country: brazil", None),
    "EWG":  ("country: germany", None),
    "EWU":  ("country: united kingdom", None),
    "EWC":  ("country: canada", None),
    "EWA":  ("country: australia", None),
    "EWH":  ("country: hong kong", None),
    "EWW":  ("country: mexico", None),
    "EWJ":  ("country: japan", None),
    "EWT":  ("country: taiwan", None),
    "EWY":  ("country: south korea", None),
    # post-2007, so excluded by the pre-2007 rule and tagged rather than
    # dropped.  The factor funds are the awkward case: they were launched
    # *because* the factors had backtested well, which is a selection story in
    # the sponsor rather than in this list, and no cutoff repairs it.
    "JETS": ("industry: airlines (2015)", None),
    "XAR":  ("industry: aerospace (2011)", None),
    "INDA": ("country: india (2012)", None),
    "MTUM": ("factor: momentum (2013)", None),
    "QUAL": ("factor: quality (2013)", None),
    "USMV": ("factor: low volatility (2011)", None),
    "VLUE": ("factor: value (2013)", None),
}

UNIVERSE.update(EQUITY_SUBSTITUTES)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=pathlib.Path,
                        default=pathlib.Path("data/cross_assets/etf_daily.db"))
    parser.add_argument("--symbols", nargs="+", default=sorted(UNIVERSE))
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="pause between calls; the options download shares this key")
    parser.add_argument("--force", action="store_true",
                        help="refetch symbols already stored")
    return parser.parse_args(argv)


def api_key() -> str:
    for line in pathlib.Path(".env").read_text(encoding="utf-8").splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "ALPHAVANTAGE_API_KEY":
            return value.strip().strip('"').strip("'")
    raise SystemExit("error: ALPHAVANTAGE_API_KEY missing from .env")


def fetch(symbol: str, key: str, attempts: int = 4):
    """Daily OHLC, split-adjusted, full history.

    The adjusted close is deliberately ignored: the engine needs a consistent
    OHLC bar, and mixing a dividend-adjusted close with unadjusted highs and lows
    produces true ranges that never happened.  Splits are handled by taking the
    adjusted series' ratio, which keeps the bar internally consistent.
    """
    query = urllib.parse.urlencode({
        "function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": symbol,
        "outputsize": "full", "apikey": key})
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                f"https://www.alphavantage.co/query?{query}",
                headers={"User-Agent": "research/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode())
        except Exception as error:
            if attempt == attempts - 1:
                return None, f"{type(error).__name__}: {error}"
            time.sleep(2 ** attempt)
            continue
        series = payload.get("Time Series (Daily)")
        if series:
            return series, "ok"
        note = (payload.get("Note") or payload.get("Information")
                or payload.get("Error Message") or str(payload)[:160])
        if "limit" in note.lower() or "frequency" in note.lower():
            time.sleep(15)
            continue
        return None, note
    return None, "exhausted"


def rows_for(symbol: str, series: dict):
    """OHLC bars restated in post-split terms, dividends left alone.

    Splits are taken from the vendor's own split coefficient rather than inferred
    from the ratio of adjusted to raw close.  The inference is what an earlier
    version did, treating any ratio inside 0.2 to 5.0 as dividends: a two-for-one
    split produces a ratio of almost exactly 0.5 and sailed through untouched,
    so every pre-split price was stored raw.  That silently doubled the older
    half of XLK, XLU, SLV and others, and turned a 27-year sector comparison into
    nonsense while looking entirely plausible.

    Dividends are deliberately not removed: the engine needs an internally
    consistent bar, and a dividend-adjusted close beside unadjusted highs and lows
    produces true ranges that never happened.
    """
    days = sorted(series)
    cumulative, factors = 1.0, {}
    # Walk backwards: a bar before a split must be divided by everything that
    # happened after it.
    for day in reversed(days):
        factors[day] = cumulative
        try:
            coefficient = float(series[day].get("8. split coefficient", 1.0))
        except (TypeError, ValueError):
            coefficient = 1.0
        if abs(coefficient - 1.0) > 1e-9:
            cumulative /= coefficient
    out = []
    for day in days:
        values = series[day]
        scale = factors[day]
        try:
            out.append((symbol, "daily", f"{day}T00:00:00.000Z",
                        float(values["1. open"]) * scale,
                        float(values["2. high"]) * scale,
                        float(values["3. low"]) * scale,
                        float(values["4. close"]) * scale,
                        float(values["6. volume"])))
        except (KeyError, ValueError):
            continue
    return out


def main(argv=None):
    args = parse_args(argv)
    key = api_key()
    args.db.parent.mkdir(parents=True, exist_ok=True)
    store = sqlite3.connect(args.db)
    store.executescript(SCHEMA)
    for ticker, (sleeve, binance) in UNIVERSE.items():
        store.execute("INSERT OR REPLACE INTO venue_map VALUES (?,?,?)",
                      (ticker, sleeve, binance))
    store.commit()

    have = {t for (t,) in store.execute(
        "SELECT DISTINCT ticker FROM bars WHERE frequency='daily'")}
    todo = [s for s in args.symbols if args.force or s not in have]
    print(f"{len(todo)} of {len(args.symbols)} symbols to fetch"
          f"{'' if not have else f'; {len(have)} already stored'}\n", flush=True)

    ok = fail = 0
    for symbol in todo:
        series, status = fetch(symbol, key)
        if series is None:
            print(f"  {symbol:5s} FAILED  {status[:90]}", flush=True)
            fail += 1
            time.sleep(args.sleep)
            continue
        rows = rows_for(symbol, series)
        store.executemany("INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?,?)", rows)
        store.commit()
        sleeve, binance = UNIVERSE.get(symbol, ("?", None))
        print(f"  {symbol:5s} {len(rows):>6,d} bars  {rows[0][2][:10]} -> "
              f"{rows[-1][2][:10]}  {sleeve:10s} "
              f"binance: {binance or '--'}", flush=True)
        ok += 1
        time.sleep(args.sleep)

    total = store.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
    names = store.execute(
        "SELECT COUNT(DISTINCT ticker) FROM bars WHERE frequency='daily'").fetchone()[0]
    store.close()
    print(f"\n  {ok} stored, {fail} failed; {names} symbols, {total:,} daily bars")
    print(f"  wrote {args.db}")
    return 1 if fail and not ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
