# savi.uz

Data ingestion scaffold for a daily strategy that combines:

- US equities 5-minute OHLC data (AlphaVantage)
- US options chains for major symbols like `SPY` and `QQQ` (AlphaVantage)
- Macro datasets including Fed policy rate and yield-based forward-rate proxy (AlphaVantage)
- CFTC Commitments of Traders positioning, all six reports, back to 2000 (or 1986 for legacy)
- Earnings and valuation: Shiller's 1871- CAPE history, SEC XBRL company fundamentals,
  index prices, and the FactSet Earnings Insight forward-estimate series
- Binance trad-FI perpetuals, mapped to their Yahoo Finance underlyings and clustered
  for uncorrelated position selection

## Quick start

Copy `.env.example` to `.env` and fill in your keys (the file is gitignored):

```bash
cp .env.example .env
```

`FRED_API_KEY` is free and required for macro history — without it FRED serves only
the latest revision, with no vintages and no release dates.

Run tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Binance trad-FI risk map

Binance lists ~160 `TRADIFI_PERPETUAL` contracts on USDⓈ-M futures: US equities and
ETFs, Hong Kong and Korea listings, commodities including gold (`XAUUSDT`) and silver
(`XAGUSDT`), and a couple of pre-IPO names. The perps themselves are only months old,
so their own price history is far too short for correlation work. The script pulls the
live contract list from Binance, maps each contract to the Yahoo ticker for its
underlying, and does the risk analysis on years of daily underlying data.

```bash
pip install -r requirements.txt
PYTHONPATH=src python scripts/build_tradfi_risk_map.py --outdir out/tradfi
```

### Mapping validation

About a third of the Binance tickers cannot be guessed (`HK0700`, `KODEX200`,
`CSOPSAMSUNG2L`, `SKHYNIX`), and some that look obvious are traps -- Yahoo's `PENG-USD`
is a memecoin, not Penguin Solutions. Every candidate mapping is therefore scored
against the contract's own Binance klines before it is trusted, on two tests:

- **Price-ratio stability.** Unrelated securities do not hold a constant price ratio for
  weeks. This is the test that survives the session-time mismatch on HK/KR names, where
  a UTC daily bar straddles two local sessions and wrecks return correlation. Measured
  over a recent window so a mid-history share split cannot look like a broken mapping.
- **Rank correlation of returns**, outlier-resistant, over lags -1/0/+1.

Contracts listed too recently to test fall back to `assumed` if the mapping is curated,
and are flagged as such in the report. Everything that fails is excluded from the panel
and listed in `universe.csv` for review.

### Analysis

- Daily (or `--freq weekly`) log returns, differenced per instrument so a name that does
  not trade every session keeps its true close-to-close returns
- Pairwise correlation with shrinkage toward the universe average, repaired to positive
  semi-definite
- Average-linkage clustering on `sqrt(0.5 * (1 - rho))` distance, cut at
  `--corr-threshold`, on both raw returns and SPY-neutral residuals
- Effective number of bets, PCA variance profile, and a threshold sensitivity curve
- A greedy low-correlation basket: most liquid name per cluster, subject to a pairwise
  correlation cap and a Binance 24h volume floor

### Output

`out/tradfi/` gets `report.md` (the readable summary), `universe.csv` (every contract
with its mapping and validation verdict), `metrics.csv` (per-contract volatility, beta,
liquidity, cluster ids), `correlation_raw.csv` and `correlation_residual.csv`,
`seed_group_coherence.csv`, and `clusters.json`.

Downloads are cached under `--cache-dir` (default `.cache/tradfi`); pass `--refresh` to
refetch. Useful flags: `--seed-only` restricts the run to the hand-labelled universe in
`savi_uz.seed_groups`, `--freq weekly` cross-checks cross-region correlations against
the non-synchronous-close bias, `--include-mirrors` admits Yahoo's mirror of the Binance
contract itself as a price source for names with no public underlying.

## Macro history with vintages

```bash
PYTHONPATH=src python scripts/download_macro_history.py --db data/macro/macro.db --csv-dir data/macro/csv
```

Pulls 80 FRED series back to 2000 (`--start 1900-01-01` for each series' full run),
plus NY Fed reference rates and the Board's fitted Treasury curve. Roughly 1M rows
in ~6 minutes.

| Group | Contents |
|---|---|
| `policy_rate` | DFF, EFFR, target range, IORB, SOFR, OBFR |
| `market_implied_path` | Full Treasury curve 1m–30y, TIPS reals, 2s10s, 3m10y, fitted forwards, Kim-Wright term premium |
| `fed_sep` | SEP dots: fed funds median/central tendency/range, GDP, PCE, core PCE, unemployment |
| `inflation` | CPI, core CPI, PCE, core PCE, 5y/10y breakevens, 5y5y forward, Michigan, Cleveland Fed |
| `labor` | Payrolls, U-3, U-6, claims, continuing claims, participation, JOLTS openings and quits, AHE, hours |
| `credit` | ICE BofA OAS (HY, CCC, IG, BBB), Moody's Baa/Aaa spreads, Chicago Fed NFCI and subindices, StL stress index, VIX |
| `balance_sheet` | WALCL, securities held outright, reserves, TGA, ON RRP volume and award rate, discount window |
| `corporate_profits` | BEA corporate profits (CP, CPATAX, CPROFIT, pre-tax), profits/GDP, GDP and real GDP — quarterly to 1947 |

### Release and vintage timestamps

Values are stored three ways, because "what was known when" is a different question
from "what is true now":

- `observations` — current revision of every value
- `first_release` — the initial print with `release_date`, i.e. what the market
  actually traded on, and `superseded_on` for when it was replaced
- `vintages` — the full revision matrix, collected for the SEP by default and for
  anything named in `--all-vintages`
- `release_dates` — each release's publication calendar
- `reference_rates` — NY Fed rates with their own `revision_indicator` and intraday
  percentiles, which FRED flattens away

The SEP is the case that makes this necessary: FRED encodes the dot plot as a single
series whose observation date is the *projection's target year* and whose vintage date
is the *FOMC meeting that published it*. Reading only current values collapses fourteen
years of dots into fourteen numbers. Stored properly, `SELECT ... WHERE
series_id='FEDTARMD'` reconstructs each meeting's full projected path.

ALFRED refuses any window holding more than 2000 vintage dates, so long daily series
are fetched over chunked real-time windows and reassembled.

### Market-implied Fed path

`fed_path` holds the forward rate at 3, 6, 9, 12, 18, 24, 36, 48 and 60 months, derived
daily from the Svensson parameters in the Board's GSW table (`gsw_params`, back to 1961).
Because the parameters are stored, any other horizon can be computed after the fact.

This is a **Treasury** forward curve and therefore includes a term premium — it is a
proxy for the expected policy path, not an OIS-implied path. True fed funds futures
probabilities (CME FedWatch) have no free API; `THREEFYTP10` is stored so the term
premium can be netted off, and the SEP dots give the Committee's own path for contrast.

### Known source limits

- FRED serves the **ICE BofA** OAS indices under a rolling ~3-year licence window, so
  `BAMLH0A0HYM2` and friends start in 2023 regardless of `--start`. `BAA10Y`, `AAA10Y`,
  `NFCICREDIT` and `STLFSI4` carry the pre-2023 credit-stress record instead — the
  Baa-10y spread peaks at 6.16% in December 2008 and NFCI at 3.10.
- ALFRED only archives vintages from whenever a series was added, so first-print
  coverage is shorter than observation coverage for some series (`T5YIE` has values from
  2003 but vintages only from 2014). Compare `observations` against `first_release`
  counts in the `coverage()` view before assuming a clean real-time history.
- SEP series begin in January 2012, when the FOMC first published fed funds projections.

## CFTC Commitments of Traders

```bash
PYTHONPATH=src python scripts/download_cftc_history.py --db data/cftc/cot.db
```

Pulls all six COT reports from the CFTC's own annual archives back to 2000 by
default. No API key: these are public files. 82 archives, ~975k rows, about
five minutes and a 712MB database.

| Report | Table | Breakdown | Coverage |
|---|---|---|---|
| `legacy_futures` | `cot_legacy_futures` | Non-commercial / commercial / non-reportable | 1986- |
| `legacy_combined` | `cot_legacy_combined` | Same, options delta-weighted in | 1995-03-21- |
| `disagg_futures` | `cot_disagg_futures` | Producer-merchant / swap dealer / managed money / other | 2006-06-13- |
| `disagg_combined` | `cot_disagg_combined` | Same, with options | 2006-06-13- |
| `tff_futures` | `cot_tff_futures` | Dealer / asset manager / leveraged funds / other | 2006-06-13- |
| `tff_combined` | `cot_tff_combined` | Same, with options | 2006-06-13- |

The three regimes do not share a start date, and that is a property of the
source rather than a gap in the download: the disaggregated and financial-trader
breakdowns were introduced in 2006 and no earlier history exists. A run starting
in 2000 is legacy-only for its first six years. `--start-year 1986` gets the
full legacy record, which is monthly and then biweekly before 1992 and only
becomes weekly around 2000.

Useful flags: `--reports tff_futures` limits the pull, `--list` prints the
archive plan without fetching, `--csv-dir` exports every table, `--start-year`
and `--end-year` bound the range.

### Storage

One wide table per report, mirroring the published file: COT already arrives as
one row per contract per Tuesday, and the columns are what analysts name
directly, so long-form storage would turn 975k rows into roughly 150M for no
gain. Tables are built from the header of the first file ingested and widened
with `ALTER TABLE` if the CFTC adds a column.

Header spelling differs by regime -- `"Noncommercial Positions-Long (All)"` in
the legacy files against `"Prod_Merc_Positions_Long_All"` in the newer ones --
so every name is normalised to snake_case. Each table also carries a canonical
`contract_code` / `report_date` pair as its primary key, which is the one thing
the three header conventions do not agree on. Re-running updates rows in place.

Contract market codes are stored as TEXT. `001602` is a label, not a quantity,
and integer parsing would silently eat the leading zero.

`cot_contracts` is the lookup table: one row per contract per report, with the
**most recent** market name, the exchange, and the date span it covers.

### Weekly update

The CFTC rewrites the current year's ZIP every Friday, so the incremental update
is one year with the cache bypassed:

```bash
PYTHONPATH=src python scripts/download_cftc_history.py --start-year 2026 --refresh
```

Archives are cached under `--cache-dir` (default `.cache/cftc`, 151MB for the
full history), so re-runs over past years cost nothing.

### Known source quirks

- **Contracts get renamed.** `043602` was `10-YEAR U.S. TREASURY NOTES - CHICAGO
  BOARD OF TRADE` until 2022-02-01 and `UST 10Y NOTE - CHICAGO BOARD OF TRADE`
  from 2022-02-08. The code is continuous across the rename and the name is not,
  so join on `contract_code`; `cot_contracts.market_name` gives the current name
  for searching.
- **The 2006-2016 TFF bundles are not ISO.** Both ship dates as
  `9/9/2014 12:00:00 AM` under a column named `Report_Date_as_YYYY-MM-DD`, while
  every other archive uses ISO. `parse_report_date` handles both and the store
  normalises to ISO; the raw column keeps whatever the file said. A file that
  yields no usable rows now raises rather than reporting an empty result, since
  a parser failure and an out-of-range archive otherwise look identical.
- **Report dates are almost always Tuesdays**, published the following Friday at
  15:30 ET -- but not invariably: 22 of the 1390 legacy dates fall on a Monday,
  Wednesday or Friday, shifted by holiday weeks. Derive the publication lag from
  the date rather than assuming a fixed weekday. The archives carry no
  publication timestamp, so unlike the FRED tables there is no vintage dimension
  here; for backtests, lag the report date by three days to avoid look-ahead.

## Earnings and valuation

```bash
PYTHONPATH=src python scripts/download_earnings_history.py --db data/equity/equity.db
```

Four sources, none of which covers the whole picture, into one database.

| Source | Table | Coverage | Rows |
|---|---|---|---|
| Shiller | `shiller_monthly` | 1871-01 to 2024-09, monthly | 1,845 |
| Yahoo index closes | `index_prices` | 2000-01-03 to today, daily | 26,776 |
| SEC XBRL frames | `sec_facts` | 2009-02-19 onward, quarterly | 2,731,125 |
| Alpha Vantage | `analyst_earnings` | needs a key | 0 |

### What each one can and cannot reach

**Shiller** is the only free source with S&P 500 earnings, dividends, CPI and CAPE
back to 1871. Two things to know. It is maintained by hand and runs behind --
and the mirrors disagree, the Yale copy ending a full year earlier than the one
on shillerdata.com. `ShillerClient` fetches every mirror and keeps whichever has
the later last observation, so a stale mirror costs end date rather than the
whole download. Within a copy, price is filled before earnings: at the time of
writing price runs to 2024-09 and earnings only to 2024-06. `shiller_earnings_gap()`
reports that lag, which is what a CAPE calculation has to bridge.

Shiller's date column is `YYYY.MM` stored as a float, so **October arrives as
`1871.1`, not `1871.10`**. Parsed naively it becomes January and silently
corrupts every October in a 154-year series. The parser formats to two decimals
first; the month histogram comes out 153-154 apiece rather than 307 Januaries.

**SEC XBRL cannot reach 2000.** XBRL was phased in over 2009-2011: the 2009Q1
frame holds 475 filers against 4,994 by 2023. There is no route to company
fundamentals before 2009 through this API at any price, and a run started
earlier reports the fact and begins at 2009.

The download uses the `frames` endpoint -- one request returns every filer's
value for a concept in a period -- rather than per-company `companyfacts`, which
is what makes market-wide history affordable: 720 requests instead of ~8,000.
Revenue is collected under both `Revenues` and the ASC 606 tag, because filers
migrated between them in 2018 and neither spans the period alone.

`data.sec.gov` rejects default library user agents with a 403. `SEC_USER_AGENT`
overrides the default; their fair-access policy asks that it identify you.

**S&P 500 price does not come from FRED.** FRED's `SP500` only reaches back to
2016 under a rolling ten-year licence window, the same restriction as the ICE
BofA spreads. Yahoo `^GSPC` covers 1927 to today and is what fills
`index_prices`, alongside `^SP500TR`, `^NDX` and `^RUT`.

**Corporate profits** are FRED series and live in the macro database, not here:

```bash
PYTHONPATH=src python scripts/download_macro_history.py --groups corporate_profits
```

That group holds BEA's `CP`, `CPATAX`, `CPROFIT`, profits as a share of GDP and
GDP itself, quarterly back to 1947 -- the macro counterpart to company earnings,
measured from the national accounts rather than from filings, and reaching six
decades further back than XBRL.

**Alpha Vantage** estimates need `ALPHAVANTAGE_API_KEY`; the step prints a note
and skips when it is absent. The free tier is roughly 25 calls a day, so the
universe is a short list rather than the market -- override with `--tickers`.
Quota exhaustion arrives as HTTP 200 with an explanatory body rather than an
error status, so it is detected from the JSON and stops the loop.

## US proxy map

```bash
PYTHONPATH=src python scripts/build_us_proxy_map.py --outdir out/tradfi
```

For testing strategies that can only trade US hours: maps every Binance trad-FI
contract to a US-listed instrument, and **measures** how well each proxy tracks
its real underlying rather than assuming an ADR or a country ETF is good enough.

132 contracts are US equities and map to themselves. 28 are scored against
candidate proxies; 3 have nothing to map to.

### How tracking is measured

Three numbers, because one correlation hides the two things that matter:

- **Daily over lags -1/0/+1.** Hong Kong closes at 08:00 UTC and Korea at 06:00,
  the US at 20:00 or later. A US instrument's day-T return therefore carries
  news the Asian market cannot reflect until T+1, so same-day correlation
  understates the link and a real relationship shows up as the US proxy leading.
- **Weekly.** Friday-to-Friday returns mostly remove the session offset. This is
  the headline number and what the verdict is based on.
- **SPY-neutral weekly.** Correlation of residuals after market beta is removed
  from both sides. This separates a proxy that tracks *the name* from two things
  that both follow the US market, and it demotes an otherwise good-looking
  correlation to `market-beta` when nothing specific is left.

Beta and R² are reported alongside, because a correlation with the wrong slope
still does not size a position.

### What the data says

The session offset is real and large. Every Korean contract's best lag is **+1**
-- the US instrument moves first -- and daily correlation badly understates the
relationship that weekly returns recover:

| Contract | Proxy | daily | weekly | best lag |
|---|---|---:|---:|---:|
| SAMSUNG | EWY | 0.36 | 0.63 | +1 |
| KODEX200 | EWY | 0.43 | 0.75 | +1 |
| SKHYNIX | MU | 0.46 | 0.71 | +1 |
| HYUNDAI | EWY | 0.25 | 0.53 | +1 |

Hong Kong ADRs and commodities are synchronous enough to peak at lag 0.

**ADRs are near-substitutes.** Tencent via `TCEHY` is 0.92 weekly, Xiaomi via
`XIACY` 0.94, Meituan via `MPNGY` 0.90, Pop Mart via `PMRTY` 0.81 -- and the
SPY-neutral figure is the same or higher in each case, so the tracking is the
company, not the market.

**Commodities are effectively exact**: gold/`GLD` 0.98, silver/`SLV` 0.97,
WTI/`USO` 0.99, copper/`CPER` 0.99, platinum/`PPLT` and palladium/`PALL` 0.98.

**Korea is where the assumption breaks.** There is no liquid US ADR for Samsung
or SK Hynix, so the routes are the memory peer (`MU`) or the country ETF
(`EWY`), and they carry only part of the move: Samsung 0.63, SK Hynix 0.71.
Below that, `NAVER` (0.24) and `LGELECTRONICS` (0.27) are not tradable through a
US proxy at all -- `EWY` is a 100-name index and these are single stocks inside
it. `KODEX200` is the exception at 0.75, because it is itself a KOSPI 200 index
fund and `EWY` is the US equivalent exposure.

`ZHONGJI` (0.33 against optical-module peers) and `GIGADEV` (0.38 against
semiconductor ETFs) are sector-level only.

### Output

`out/tradfi/us_proxy_map.csv` has every candidate scored; `us_proxy_map.md` is
the readable version with a best-proxy-per-contract table. Verdicts are
`direct`, `strong` (>=0.80), `usable` (0.55-0.80), `market-beta` (correlated but
only through SPY), `weak`, `poor`, `insufficient`.

No proxy exists for `ANTHROPIC` and `OPENAI` (pre-IPO, no listing anywhere), or
for `BBX`, which is a US contract whose Yahoo ticker the universe run could not
resolve -- a mapping gap rather than an untradable name.

## Themes and their leaders

```bash
PYTHONPATH=src python scripts/select_theme_leaders.py --outdir out/tradfi
```

Scores each hand-labelled theme in `savi_uz.seed_groups` against the measured
correlation panel and picks the two or three names that actually carry it.

### Scoring

Themes are scored by **factor strength**, the first principal component's
eigenvalue rescaled to `(lambda1 - 1) / (n - 1)`. Average pairwise correlation
is the obvious statistic and it fails here twice:

- **Signs.** "Vol / rates diversifiers" holds `TMF` (3x long Treasuries) and
  `TBT` (2x short). A signed average reports **-0.29** and buries a perfectly
  tight factor. Eigenvalues are unchanged by flipping a variable's sign, so the
  same theme scores **0.44** and `TMF` is simply flagged as the inverse leg.
- **Size.** A raw variance share cannot fall below `1/n`, so a two-name theme
  scores at least 0.50 however unrelated its members are, while an eight-name
  theme starts at 0.125. The rescaling puts every theme on one scale; under an
  equicorrelation model it returns exactly the common correlation.

Leaders are ranked by absolute PC1 loading, then deduplicated by underlying:
Binance lists `TENCENT` and `HK0700` as separate contracts on the same stock,
and two contracts on one company are one thing to track, not two.

### Result

15 of the 17 measurable themes hold together. Two do not:

- **Healthcare** (0.15) -- `LLY` and `NVO` are one obesity-drug trade; `HIMS` is
  telehealth. Different businesses wearing one label.
- **Autos / mobility** (0.16) -- `TSLA`/`RIVN` (EV), `HYUNDAI` (legacy auto) and
  `UBER` (rideshare) share a word, not a risk factor.

### The hand labels are coarse

The clustering found tighter groups than the seed table, and 81 of the 150
instruments in the panel carry no hand label at all. Notably:

| Data-driven cluster | Strength | vs. seed theme |
|---|---:|---|
| `HANMI, KODEX200, SAMSUNG, SKHYNIX, SAMSUNGEM` | 0.63 | "AI / semis Asia" scores 0.36 because it dilutes Korea with `TSM` and China names |
| `AAOI, CIEN, COHR, LITE, GLW` | 0.61 | optical networking -- not in the seed table at all |
| `SQQQ, TZA, UVXY` | 0.72 | inverse/vol, split across two seed groups |
| `ALAB, CRDO, AVGO, GEV, VRT, NVDA` | 0.54 | AI infrastructure, distinct from semicap equipment |

Output is `out/tradfi/theme_leaders.csv` and `theme_leaders.md`, with each pick
carrying its US-tradable symbol and proxy verdict from the proxy map, so a
US-only strategy can be built straight off the table.

## Intraday bars from Tiingo

```bash
PYTHONPATH=src python scripts/download_intraday_history.py --plan
PYTHONPATH=src python scripts/download_intraday_history.py --db data/intraday/bars.db
```

Hourly bars back to 2017 for the theme leaders -- the 46 US-tradable names that
carry the themes, rather than the whole 163-contract universe.

### Staying inside the quota

The free tier allows 50 requests/hour, 1000/day and 500 unique symbols/month, so
a full 2017- pull of 46 symbols is roughly 500 requests and does not fit in one
sitting. The script is built to be stopped and resumed:

- Pacing defaults to **45 requests/hour**, evenly spaced rather than bursted, so
  an interrupted run has never exceeded the rolling budget. `--requests-per-hour`
  raises it on a paid plan.
- `--max-requests` caps one invocation (default 333, leaving room to run twice
  more the same day).
- Every response is cached on disk **and** every completed `(ticker, year)`
  window is recorded in the `windows` table. A resumed run asks only for what is
  missing; cached windows cost nothing and do not count against the budget.
- A 429 stops the run immediately and reports where it got to. It never retries
  into a block.
- `--plan` prints the request count and estimated hours without spending
  anything.

One metadata request per symbol pays for itself: it returns the ticker's first
date, so no requests are spent on years before a company listed. `RKLB` starts
2021-08, `GEV` 2024-03.

### Source limits handled, not papered over

**The 10,000-row cap is silent.** A request spanning more than about six years of
hourly bars returns exactly 10,000 rows, no error, and it returns the *recent*
end of the range. Asking for 2017-2026 in one call hands back 2020 onward and
looks exactly like history starting in 2020 -- which is what it appeared to do
until the range was chunked. Requests are split into `--years-per-request`
chunks (default 3, ~4,700 hourly bars) and any response arriving at the cap is
flagged in the `windows` table and printed. `max_safe_years()` computes the
ceiling per frequency: 3 for hourly, 20 for daily.

Chunk size is the single biggest lever on cost. One year per request uses 16% of
each response and turns the 46-symbol 2020-2026 pull into ~500 requests and 11
hours; three years makes it 183 requests and about 4.

**Volume has to be asked for by name.** The IEX endpoint's default projection is
`date/open/high/low/close` and drops volume without a word. The client sends
`columns=open,high,low,close,volume` explicitly. Note that IEX is one venue with
a low single-digit share of consolidated volume, so these counts are a *relative*
activity measure, not tradable share volume.

**IEX intraday is exchange-listed only.** The OTC ADRs here -- `TCEHY`, `XIACY`,
`MPNGY`, `PMRTY`, all on `PINK` -- return zero intraday bars for any window,
recent or historic, while having years of daily history (`TCEHY` back to 2008).
They fall back to daily automatically; `--no-daily-fallback` skips them instead.

### Before you backtest on this

**Intraday bars are RAW.** NVDA's close goes 1,203 to 121 across its 10:1 split
on 2024-06-10, which a naive backtest reads as a 90% loss in one day. The
`corporate_actions` table carries `splitFactor`, `divCash`, `close` and
`adjClose` per day -- one request per symbol covers 20 years -- so a backtest can
rebuild an adjusted series. Split events are printed at the end of every run.

**Timestamps are UTC and the session shifts with US DST.** SPY has exactly six
bars a day, but they sit at 14:00-19:00 UTC in summer and 15:00-20:00 in winter.
Convert to `America/New_York` before assuming a fixed bar index.

**About 0.2% of bars have a close outside their own high/low.** IEX takes the
close from the last trade while high and low come from the interval aggregation,
and they occasionally disagree by a few cents -- 32 bars in SPY's 15,036, mostly
2018. The rows are stored exactly as published and the count is reported; a
backtest that assumes `low <= close <= high` should decide for itself what to do
rather than have the data quietly rewritten under it.

**Early volume is patchy.** 1,727 of SPY's 15,036 bars carry no volume, 73% of
them in 2017-2018 when IEX had less share. Post-2019 it is under 5% a year.

### A local TLS wrinkle

On Windows the system trust store carries an expired root that Python selects
for `api.tiingo.com`, so `urllib` fails verification while `curl` succeeds. The
client pins certifi's bundle rather than weakening verification.

### Output

`data/intraday/bars.db` holds `bars` (keyed by ticker, frequency and timestamp,
so hourly and daily coexist), `symbols` (exchange, listing date, which themes the
name represents), `corporate_actions` (split and dividend factors), `windows`
(the resume state and truncation flags) and `fetch_log`. `--csv-dir` exports all
of it.

Downloaded so far: `SPY` and `QQQ`, 15,036 hourly bars each covering
2017-01-03 to 2026-08-14 across 2,506 trading days, plus 2,417 days of
adjustment factors apiece.

## Volume-profile breakout study

```bash
PYTHONPATH=src python scripts/run_breakout_study.py --ticker SPY --frequency 5min
```

Does the shape of a session's volume profile, formed from the bars that have
already closed, say anything about how far price travels in the *next* bar?

### The rule the whole result rests on

Price and volume are both known only when a bar closes. So a feature row
observed at bar `t` may read **only** bars `1..t` of that session, and its target
may read **only** bar `t+1`. This is enforced structurally -- `build_samples`
walks each session forward and hands `build_profile` a prefix -- and asserted by
a test that rewrites the final bar of a session and requires every earlier row's
features to come back byte-identical.

Three more choices keep the question honest:

- **The last bar of each session is dropped.** Its next bar is the following
  morning, so keeping it would mix an overnight gap into an intraday study.
- **The split is chronological.** A random split leaks: adjacent bars in one
  session share almost the same profile and would land on both sides.
- **The target is the absolute move.** "Just before a breakout" is a claim about
  magnitude. Testing direction would be a far stronger claim.

### Controlling for the obvious

The raw table is dominated by two effects that have nothing to do with volume
profiles: sessions that have already travelled keep travelling (range-so-far
quintile 5 lifts the next move by **2.57x**), and the last hour is busier than
the middle (**1.32x**). Any profile feature correlated with either inherits that
lift without adding information.

So every bucket is also scored **within** its own range quintile *and* bar of the
session, pooled across strata. That is the number worth reading, and it is much
smaller than the raw one -- `Close vs POC` quintile 1 falls from 1.368 raw to
1.127 controlled, so roughly two thirds of its apparent edge was borrowed.

### What SPY hourly actually says

6,655 decision points across 2,228 sessions, baseline next-bar move 19.9bp.
Controlled lifts:

| Setup | Lift | t |
|---|---:|---:|
| Close in top fifth of the developing range | **0.765** | -10.7 |
| `B` shape (two separated distributions) | **0.763** | -9.8 |
| Volume concentration in top quintile | 0.875 | -5.1 |
| `P` shape (volume stacked high) | 0.881 | -3.7 |
| Close well below POC (quintile 1) | 1.127 | +4.9 |
| `b` shape (volume stacked low) | 1.103 | +3.9 |

The reliable finding is the **negative** one, and it is the opposite of the
usual framing: the profile identifies setups before a *quiet* bar far more
sharply than before a violent one. Price sitting at the top of the developing
range, or a session that has already built two separate distributions, precedes
a next bar roughly a quarter smaller than a same-range, same-time peer, at
t = -10. The breakout side is real but weaker -- price below the POC with volume
stacked low runs about 10-13% hot.

There is also a clear **asymmetry**: `b` (volume low, price working the bottom)
is loud while `P` (volume high) is quiet. Downside exploration is faster than
upside exploration, which is what the tails in the original question describe.

### Rerun at 5-minute resolution

78 bars a session instead of six, and the 5-minute feed also covers the opening
half hour that the hourly feed drops. 36,760 decision points across 564 sessions,
baseline next-bar move 4.4bp. Controlled for range quintile and bar of session:

| Close in range | n | Next-bar move | Lift | t |
|---|---:|---:|---:|---:|
| Q1 (bottom) | 7,327 | 6.3bp | **1.220** | +19.9 |
| Q2 | 7,352 | 4.4bp | 1.053 | +5.3 |
| Q3 | 7,339 | 3.9bp | 1.007 | +0.7 |
| Q4 | 7,337 | 3.5bp | 0.892 | -11.7 |
| Q5 (top) | 7,348 | 3.7bp | **0.826** | -19.7 |

**Position in the developing range is the signal, and it replicates.** The
gradient is monotonic, the magnitudes match the hourly run (1.088 / 0.765 there,
1.220 / 0.826 here), and it survives the chronological split with train-fitted
thresholds: `Close in range` Q1 is 1.498 train against **1.539** test. `Close vs
POC` behaves the same way, which is expected -- the two features are close
cousins.

**Profile shape does not survive the resolution change.** At hourly the shape
buckets spanned +-24% (`b` 1.103, `B` 0.763); at 5-minute they collapse to +-6%
(`D` 1.060, `P` 0.959). The reason is visible in the label mix:

| | B | D | P | b |
|---|---:|---:|---:|---:|
| hourly, 6 bars/session | 15% | 50% | 12% | 23% |
| 5-minute, 78 bars/session | 44% | 27% | 20% | 9% |

With 78 bars over 30 bins a histogram has far more genuine local minima, so the
peak-counting test flags multi-modality three times as often and `B` stops being
a discriminating label at all. **The shape vocabulary is not resolution-invariant
and its thresholds would need recalibrating per bar size.** The hourly shape
result should be read as an artefact of a six-observation profile, not a finding.

`--frequency 5min --min-prefix 12 --bins 30` runs it there; the module itself is
resolution-agnostic, the classifier constants are not.

### Audit of the look-ahead claim

The no-look-ahead claim was re-checked independently of the study code, by
recomputing every feature from the raw bars. Four checks, all clean:

| Check | Result |
|---|---|
| Samples drawn from synthetic zero-volume sessions | **0** of 6,655 |
| Forward returns that are not literally the next bar in the same session | **0** |
| Independently recomputed profile prefixes that disagree | **0** of 4,000 |
| Rows where `bars_elapsed` mismatches position in session | **0** |

**The bar timestamp convention was verified, not assumed.** Tiingo stamps a bar
with the *start* of its interval: the first 5-minute bar of a January session is
`14:30Z` with 43,875 shares, and the session opens at 14:30Z in EST — an
end-stamped bar there would cover 14:25-14:30, before the open. This is the
convention the study needs: the decision point at the close of bar `t` is wall
clock `t + interval`, which is exactly when bar `t+1` begins. The prediction is
implementable at the moment it is made.

Two real problems the audit did turn up, both now fixed:

- **Bucket thresholds were fitted on the whole sample.** The features were clean
  but the *evaluation* was not: the boundary between "wide value area" and
  "narrow" was chosen with knowledge of the period being scored. Quantile edges
  are now fitted on the training period and applied unchanged to the test
  period. `Close vs POC` Q1 holds up: 1.452 train, **1.304** test.
- **Tiingo emits placeholder bars for closed markets** — flat OHLC, zero volume,
  about nine or ten a year (2017-01-16 MLK, 02-20 Presidents', 04-14 Good
  Friday, 05-29 Memorial). The study already excluded them, but the dashboard
  was counting them as trading days: 2,506 sessions where the truth is **2,417**.
  A further 188 sessions in 2017-18 carry real prices with no volume at all.

One caveat that is *not* look-ahead but does affect realism: the forward return
is measured close-to-close, and `close[t]` does not always equal `open[t+1]` --
IEX does not always print at the boundary. The gap is small (median 0.21bp,
mean 0.55bp against a ~20bp baseline move, p99 5bp) but it is slippage the
measured return does not charge for.

Output is `out/strategy/breakout_<ticker>_<freq>.md` plus the full sample table
as CSV, so the raw decision points can be re-cut without rerunning the study.

### Seeing the profiles

```bash
PYTHONPATH=src python scripts/build_profile_gallery.py --count 20
PYTHONPATH=src python scripts/build_dashboard.py \
    --template assets/profile_gallery_template.html \
    --data out/gallery/profiles.json --out out/gallery/index.html
```

A page of the actual distributions, built by calling the same `build_profile`
the study calls, so the picture is the analysis rather than a second
reconstruction of it.

Each session is drawn as **two panels sharing one price axis**: price against
time on the left, volume against price on the right. They are not merged --
volume-at-price and price-over-time have different horizontal meanings, and one
combined axis would invent a relationship that is not there.

Volume bins use a validated three-step ordinal ramp of one hue: outside the
value area, inside it, and the point of control. The value area is also washed
across both panels, so the reader has a second, non-colour cue for the same
boundary.

The exemplar strip needed care. Picking the "clearest" double distribution by
*lowest* concentration finds the flattest session on record -- which looks like
no distribution at all rather than two. `bimodality()` scores it properly: the
weaker of the two modes against the stronger, times how far the trough between
them falls. Both terms matter, since a tall second peak over a shallow dip is
one broad distribution, and a deep dip beside a negligible bump is noise.

## Donchian breakouts with volume confirmation

```bash
PYTHONPATH=src python scripts/run_donchian_study.py --ticker SPY --frequency 5min
```

This compares rolling channels formed from the previous 5 and 10 completed
five-minute bars in the same session -- 25-minute versus 50-minute lookbacks.
The signal is the first qualifying close outside the channel and the simulated
entry is the next bar's open. Signal volume is divided by the median volume in
the same bar position over the preceding 20 clean sessions, so the opening
auction is not compared with lunchtime. The default sweep is 0x, 1x, 1.25x,
1.5x and 2x relative volume.

Only 2019-onward sessions with all 78 bars carrying positive volume enter the
study. This removes Tiingo's closed-market placeholders, early-close padding
and the sparse 2017-18 IEX volume archive. The chronological split is 2023-01-01.

`Sustainable` has an operational definition rather than a chart label: no
five-minute close returns inside the channel during the next 30 minutes and
price remains outside after 60 minutes. The report also measures re-entry,
30-minute stop-outs, full-session favorable/adverse excursion, and a 2 ATR stop
against a 2R target. If both stop and target appear inside one OHLC bar, the
stop is charged. Mean R and profit factor are gross of commissions, spread,
slippage and financing.

On SPY the 5-bar channel with a 1.0x floor is the clearest default. Mean gross
outcome is +0.039R before 2023 and +0.137R after; sustainable breaks rise from
33.4%/35.9% without confirmation to 34.4%/39.2%, while quick stop-outs are
13.0%/14.7%. Raising the floor further does not improve results monotonically.
The 10-bar 1.0x version is steadier but weaker at +0.056R/+0.072R, with fewer
quick stop-outs. Both long and short 5-bar breaks remain positive on both sides
of the split. Results from completed-session channels answer a different
question and are not mixed into this study.

Output is `out/strategy/donchian_<ticker>_<frequency>.md` and the complete event
table beside it as CSV.

## Multi-session profile breakouts and overnight holding

```bash
PYTHONPATH=src python scripts/run_composite_breakout_study.py --tickers SPY QQQ GLD
```

This study forms composite volume profiles from the immediately preceding 3 or
5 completed sessions, then observes the first close outside either the 70%
value area or the full composite high-low. Entry is at the next five-minute
open. Relative volume is matched to the same bar slot over the prior 20
sessions; an optional compression filter requires the prior range/daily-ATR
ratio to be in its trailing 25th percentile. Every threshold uses older data.

Holding horizons include the signal-session close, next regular open, and the
following 1/3/5 session closes. Non-overlapping strategy variants compare
intraday-ATR stops, daily-ATR stops, time exits, and delayed trailing stops.
Because the feed contains regular hours only, the path through after-hours and
pre-market is unknown. If the next regular open gaps through a stop, the model
fills at that open rather than granting the unavailable stop price.

The report also splits the portable 3-session value-area rule by direction.
Outputs are `out/strategy/composite_breakout_<ticker>_5min.md` and matching CSV
event tables.

## Overnight gaps: how much moves before the open, and does it hold

```bash
PYTHONPATH=src python scripts/run_gap_study.py --ticker SPY --frequency 5min
```

The intraday feed is **regular session only** -- 09:30 to 16:00 ET, exactly 78
five-minute bars, zero extended-hours bars. So the pre-market *path* is not
observable here. Its net result is: the gap from one close to the next open
contains every after-hours and pre-market tick.

(Tiingo can serve extended hours -- `afterHours=true` widens the window to
12:00-21:30 UTC, roughly 8:00am to 5:30pm ET. That is a separate download of
about 60 requests for SPY's full history.)

### How much

Across 1,485 gapped sessions the median absolute gap is **36bp** and the mean
**53bp**, against an average whole-session range of 122bp. **The overnight move
is about 43% the size of the entire regular session.** The largest was 1,097bp.

### Does it hold

Four measures, because they disagree and the disagreement is the point:

| Gap size | n | Median retained | IQR | Filled | Value overlap | Opening volume |
|---|---:|---:|---:|---:|---:|---:|
| 0-10bp | 136 | 0.40 | 10.4 | **91%** | 0.42 | 1.26x |
| 10-25bp | 400 | 0.94 | 3.4 | 74% | 0.39 | 1.29x |
| 25-50bp | 436 | 0.82 | 2.5 | 58% | 0.32 | 1.42x |
| 50-100bp | 340 | **1.13** | 1.5 | 32% | 0.20 | 1.50x |
| over 100bp | 173 | **1.07** | 1.0 | **22%** | **0.18** | **1.62x** |

Every column moves monotonically with size, and they all say the same thing:
**small gaps are noise that mean-reverts, large gaps are information the market
accepts.** A sub-10bp gap fills 91% of the time. A gap over 100bp fills 22% of
the time, closes having *extended* past itself, arrives on 1.6x normal opening
volume, and builds a value area that barely touches the previous session's.

The value-overlap column is the volume-profile answer to "is the price
sustained": it falls from 0.42 to 0.18 as gaps grow. Large gaps trade a genuinely
new distribution rather than returning to the old one.

`Retained` divides by the gap itself, so a near-zero gap makes it explode -- the
`IQR` column is how far to trust the median. It is around 1.0 for gaps above
50bp and 10.4 for gaps under 10bp, where the **fill rate is the measure to read**
and retention should be ignored.
