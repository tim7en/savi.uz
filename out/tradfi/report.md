# Binance trad-FI risk map

Generated 2026-08-16 22:05, daily returns over 730 calendar days.

## Universe

- Binance trad-FI perpetuals listed: **163**
- Mapped and validated against a Yahoo underlying: **154**
- In the correlation panel: **150**
- Mapping status: 149 verified, 6 assumed, 4 weak, 3 unlisted, 1 unverified

## How many independent bets are actually here?

- Effective number of bets (raw returns): **47.3** out of 150 contracts
- Effective number of bets (SPY-neutral residuals): **74.5**
- Principal components for 80% of variance: **51**
- Clusters at |rho| >= 0.50: **82** raw, **107** residual

How the picture changes with the merge threshold:

| Cut at rho | Clusters | Singletons | Largest block |
|-----------:|---------:|-----------:|--------------:|
| 0.30 | 34 | 19 | 61 |
| 0.40 | 59 | 42 | 44 |
| 0.50 | 82 | 62 | 29 |
| 0.60 | 104 | 84 | 13 |
| 0.70 | 123 | 105 | 6 |

## Data-driven clusters (raw returns)

Trade at most one name per block. The most liquid member is the natural expression of the block on Binance.

| # | Size | Avg intra-corr | Most liquid member | Members |
|---|------|----------------|--------------------|---------|
| 0 | 29 | 0.58 | SNDK ($295.4M) | AMAT, LRCX, KLAC, SOXL, SMH, ASML, EWT, TSM, IWM, QQQ, TQQQ, SPY, AMD, FLEX, CAT, TER, MRVL, MVLL, ARM, DRAM, EWY, KORU, MU, MUU, SNDK, SNXX, WDC, EWJ, PENG |
| 1 | 6 | 0.54 | NVDA ($6.0M) | ALAB, CRDO, AVGO, GEV, VRT, NVDA |
| 2 | 5 | 0.60 | AAOI ($5.0M) | AAOI, CIEN, COHR, LITE, GLW |
| 3 | 5 | 0.58 | MSTR ($5.6M) | COIN, HOOD, MSTR, BITO, SOFI |
| 4 | 5 | 0.62 | SKHYNIX ($84.5M) | HANMI, KODEX200, SAMSUNG, SKHYNIX, SAMSUNGEM |
| 5 | 4 | 0.65 | XAU ($103.8M) | XAG, XAU, XPD, XPT |
| 6 | 4 | 0.64 | TENCENT ($237.6K) | KUAISHOU, TENCENT, HK0700, MEITUAN |
| 7 | 3 | 0.58 | BX ($340.9K) | BX, GS, JPM |
| 8 | 3 | 0.63 | CRM ($272.0K) | ADBE, CRM, NOW |
| 9 | 3 | 0.65 | CL ($59.5M) | BZ, CL, XLE |
| 10 | 3 | 0.72 | SQQQ ($623.6K) | SQQQ, TZA, UVXY |
| 11 | 2 | 0.91 | INTC ($11.0M) | INTC, INTW |
| 12 | 2 | 0.62 | DELL ($1.2M) | DELL, HPE |
| 13 | 2 | 0.59 | NBIS ($8.1M) | NBIS, CRWV |
| 14 | 2 | 0.60 | RKLB ($1.9M) | RKLB, ASTS |
| 15 | 2 | 0.54 | META ($1.7M) | AMZN, META |
| 16 | 2 | 0.68 | CRWD ($204.0K) | PANW, CRWD |
| 17 | 2 | 0.52 | HYUNDAI ($426.2K) | HYUNDAI, LGELECTRONICS |
| 18 | 2 | 0.51 | GIGADEV ($232.4K) | KSTR, GIGADEV |
| 19 | 2 | 0.60 | COST ($91.2K) | WMT, COST |

**62 contracts cluster alone** at this threshold, in descending liquidity: TSLA, CRCL, GOOGL, SOXS, AAPL, PLTR, COPPER, MSFT, BABA, NOK, BMNR, BE, NATGAS, CBRS, AXTI, ONDS, IREN, SMCI, ORCL, NFLX, HK1810, NAVER, SHAZ, QCOM, APP, IBM, BRKB, WEN, USAR, STRC, PYPL, SONY, CSCO, LLY, FLNC, ZHONGJI, XBI, RIVN, KO, DIS, HIMS, DKNG, TTWO, POPMART, PAYP, GME, BOT, FWDI, TXN, RDDT, ZM, EBAY, V, BNC, SNOW, NVO, EWZ, URNM, TMF, TBT, HD, UBER.

## Hand-labelled groups vs measured correlation

| Seed group | Members in panel | Avg intra-corr |
|------------|------------------|----------------|
| Precious metals | 2 | 0.71 |
| Space | 2 | 0.60 |
| AI / semiconductors US | 8 | 0.57 |
| China internet | 5 | 0.55 |
| Country exposures | 4 | 0.54 |
| Software / cyber | 5 | 0.49 |
| Financials | 4 | 0.48 |
| Broad indices | 4 | 0.44 |
| Crypto-linked equities | 5 | 0.43 |
| US mega-cap tech | 5 | 0.39 |
| Consumer / retail | 4 | 0.35 |
| Consumer China | 2 | 0.35 |
| AI / semiconductors Asia | 6 | 0.33 |
| Industrials / energy | 3 | 0.30 |
| Healthcare | 3 | 0.14 |
| Autos / mobility | 4 | 0.14 |
| Vol / rates diversifiers | 3 | -0.29 |

## Recommended low-correlation basket

Greedy pick by Binance liquidity: one name per cluster, pairwise |rho| <= 0.35, 24h quote volume >= $1.0M, at least 30 Binance daily bars.

Selected **15** contracts; effective bets within the basket: **13.2**.

| Contract | Underlying | Region | Seed group | 24h volume | Ann. vol | Beta SPY | Max |rho| in basket |
|----------|------------|--------|------------|-----------:|---------:|---------:|-------------------:|
| SNDKUSDT | SNDK | US | - | $295.4M | 105.9% | 2.86 | 0.34 |
| XAUUSDT | GC=F | COMMODITY | Precious metals | $103.8M | 24.0% | 0.17 | 0.17 |
| SKHYNIXUSDT | 000660.KS | KR | AI / semiconductors Asia | $84.5M | 73.1% | 0.37 | 0.21 |
| CLUSDT | CL=F | COMMODITY | - | $59.5M | 45.6% | -0.15 | 0.10 |
| TSLAUSDT | TSLA | US | Autos / mobility | $8.4M | 59.8% | 2.27 | 0.34 |
| CRCLUSDT | CRCL | US | - | $7.9M | 108.2% | 2.80 | 0.26 |
| SOXSUSDT | SOXS | US | - | $3.4M | 245.8% | -6.75 | 0.30 |
| AAPLUSDT | AAPL | US | US mega-cap tech | $3.3M | 28.7% | 1.07 | 0.34 |
| MSFTUSDT | MSFT | US | US mega-cap tech | $2.2M | 28.5% | 0.92 | 0.31 |
| BABAUSDT | BABA | US | China internet | $1.6M | 45.8% | 0.92 | 0.25 |
| BMNRUSDT | BMNR | US | Crypto-linked equities | $1.4M | 237.8% | 4.61 | 0.24 |
| DELLUSDT | DELL | US | - | $1.2M | 60.8% | 1.90 | 0.34 |
| NATGASUSDT | NG=F | COMMODITY | - | $1.2M | 88.2% | 0.12 | 0.17 |
| CBRSUSDT | CBRS | US | - | $1.1M | 130.1% | 2.79 | 0.26 |
| AXTIUSDT | AXTI | US | - | $1.0M | 124.4% | 2.31 | 0.33 |

Read the beta column alongside the correlations. A pair can sit under the pairwise cap on daily moves and still carry the same directional exposure: the basket is decorrelated day to day, not market-neutral. Size against beta, or hedge the residual index exposure separately.

## Redundant pairs (same risk, two tickers)

| A | B | rho |
|---|---|-----|
| QQQUSDT | TQQQUSDT | 0.918 |
| HK0700USDT | TENCENTUSDT | 0.917 |
| MUUSDT | MUUUSDT | 0.915 |
| EWYUSDT | KORUUSDT | 0.914 |
| INTCUSDT | INTWUSDT | 0.907 |
| SMHUSDT | SOXLUSDT | 0.900 |
| MRVLUSDT | MVLLUSDT | 0.899 |
| IWMUSDT | TZAUSDT | -0.880 |
| QQQUSDT | SQQQUSDT | -0.879 |
| TBTUSDT | TMFUSDT | -0.876 |
| QQQUSDT | SPYUSDT | 0.874 |
| SQQQUSDT | TQQQUSDT | -0.874 |
| SPYUSDT | TQQQUSDT | 0.872 |
| BZUSDT | CLUSDT | 0.859 |
| KODEX200USDT | SAMSUNGUSDT | 0.840 |

## Caveats

- Correlations use Yahoo underlying closes. US, HK and KR sessions close at different times, so cross-region daily correlations are biased low; rerun with `--freq weekly` to check.
- Binance trad-FI perps are young (most under a year), so Binance-native history is too short for correlation work. The underlying's history is the proxy, and it ignores perp-specific basis, funding and the fact that these contracts halt when the cash market is closed.
- Liquidity figures are a single 24h snapshot and move a lot; re-check before sizing.
- Mappings marked `unverified` or `no-data` were excluded from the panel; review them in `universe.csv` before trading those contracts.
- Listed too recently on Binance to validate, so the mapping is taken on trust (`assumed`): HANMI, KODEX200, LGELECTRONICS, NAVER, SAMSUNGEM, ZHONGJI.
