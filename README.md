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

### Two source limits worth knowing

**The 10,000-row cap is silent.** A request spanning more than about six years of
hourly bars returns exactly 10,000 rows, no error, and it returns the *recent*
end of the range. Asking for 2017-2026 in one call hands back 2020 onward and
looks exactly like history starting in 2020 -- which is what it appeared to do
until the range was chunked. Requests are split by calendar year (~1,550 hourly
bars each) and any response arriving at the cap is flagged in the `windows`
table and printed.

**IEX intraday is exchange-listed only.** The OTC ADRs in this universe --
`TCEHY`, `XIACY`, `MPNGY`, `PMRTY`, all on `PINK` -- return zero intraday bars
for any window, recent or historic, while having years of daily history
(`TCEHY` back to 2008). They fall back to daily bars automatically;
`--no-daily-fallback` skips them instead.

### A local TLS wrinkle

On Windows the system trust store carries an expired root that Python selects
for `api.tiingo.com`, so `urllib` fails verification while `curl` succeeds. The
client pins certifi's bundle rather than weakening verification.

### Output

`data/intraday/bars.db` holds `bars` (keyed by ticker, frequency and timestamp,
so hourly and daily coexist), `symbols` (exchange, listing date, which themes the
name represents), `windows` (the resume state and truncation flags) and
`fetch_log`. `--csv-dir` exports all of it.
