# savi.uz

Data ingestion scaffold for a daily strategy that combines:

- US equities 5-minute OHLC data (AlphaVantage)
- US options chains for major symbols like `SPY` and `QQQ` (AlphaVantage)
- Macro datasets including Fed policy rate and yield-based forward-rate proxy (AlphaVantage)
- CFTC Commitments of Traders positioning, all six reports, back to 2000 (or 1986 for legacy)
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
