# savi.uz

Data ingestion scaffold for a daily strategy that combines:

- US equities 5-minute OHLC data (AlphaVantage)
- US options chains for major symbols like `SPY` and `QQQ` (AlphaVantage)
- Macro datasets including Fed policy rate and yield-based forward-rate proxy (AlphaVantage)
- Binance trad-FI reference symbols for downstream uncorrelated portfolio clustering

## Quick start

Set your AlphaVantage API key:

```bash
export ALPHAVANTAGE_API_KEY="your_key"
```

Run tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```
