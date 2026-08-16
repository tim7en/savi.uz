# savi.uz

Data ingestion scaffold for a daily strategy that combines:

- US equities 5-minute OHLC data (AlphaVantage)
- US options chains for major symbols like `SPY` and `QQQ` (AlphaVantage)
- Macro datasets including Fed policy rate and yield-based forward-rate proxy (AlphaVantage)
- Binance trad-FI perpetuals, mapped to their Yahoo Finance underlyings and clustered
  for uncorrelated position selection

## Quick start

Set your AlphaVantage API key:

```bash
export ALPHAVANTAGE_API_KEY="your_key"
```

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
