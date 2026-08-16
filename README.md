# savi.uz

Data ingestion scaffold for a daily strategy that combines:

- US equities 5-minute OHLC data (AlphaVantage)
- US options chains for major symbols like `SPY` and `QQQ` (AlphaVantage)
- Macro datasets including Fed policy rate and yield-based forward-rate proxy (AlphaVantage)
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
