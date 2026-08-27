# SPY–Treasury–Quality Rotation Strategy

## Comprehensive backtest report

**Report date:** August 24, 2026  
**Backtest period:** January 29, 1993 through August 24, 2026  
**Starting capital:** $100,000  
**Latest tested version:** portfolio-drawdown control with SPY-recovery quality exits and a five-year sunset

> This is an exploratory historical simulation, not an investment recommendation. The quality-stock universe is survivor-selected and therefore unsuitable for treating the reported result as an unbiased expected return.

## Executive summary

The strategy began with an **80% / 20% capital split**:

- **$80,000 in the SPY sleeve**;
- **$20,000 in the Treasury reserve**; and
- **$0 in individual quality stocks**.

However, the SPY sleeve began at **3× daily-reset leverage**. Therefore, the initial economic exposure was not simply 80% SPY and 20% Treasury:

- SPY equity allocation: 80% of capital;
- SPY gross market exposure: 80% × 3 = **240% of capital**;
- Treasury reserve: **20% of capital**;
- implied financed exposure above account equity: approximately **160% of capital**.

The phrase “80/20 portfolio” is consequently incomplete. It describes where the account equity was assigned, but not how much market risk the portfolio carried.

The latest strategy grew $100,000 to **$6.18 million**, compared with **$3.17 million** for 1× SPY buy-and-hold. The strategy's CAGR was 13.07%, versus 10.84% for SPY. That additional return came with substantially greater risk:

- maximum drawdown: **−66.48%**, versus −55.19% for SPY;
- annualized volatility: **31.32%**, versus 18.55% for SPY;
- Sharpe ratio versus Treasury: **0.469**, versus 0.511 for SPY;
- worst daily return: **−21.66%**, versus −10.94% for SPY; and
- time spent at least 20% below a prior portfolio high: **59.64%**, versus 16.90% for SPY.

The full-period terminal result favors the strategy, but SPY produced the better risk-adjusted return. The strategy also experienced a roughly **15.4-year recovery period** from its July 1998 portfolio high to its December 2013 recovery.

## Strategy specification

### 1. Starting allocation

| Component | Starting dollars | Capital weight | Initial gross exposure |
|---|---:|---:|---:|
| SPY sleeve | $80,000 | 80% | $240,000 at 3× |
| Treasury reserve | $20,000 | 20% | $20,000 |
| Quality stocks | $0 | 0% | $0 |
| Total account equity | $100,000 | 100% | — |

The Treasury reserve earns the prior-known daily 3-month constant-maturity Treasury rate, represented by FRED series DGS3MO. The assumed financing rate on leveraged SPY exposure is the same Treasury rate plus one percentage point.

### 2. Portfolio drawdown as the control signal

All downside actions use the **total portfolio NAV drawdown**, not the drawdown of SPY alone. The signal is observed at one close and executed at the following close.

This distinction matters because the initial 2.4× gross SPY exposure magnifies small SPY declines. A roughly 4%–5% SPY decline can produce a portfolio drawdown near 10%, causing the first action well before SPY itself is down 10%.

### 3. Reserve deployment ladder

| Portfolio drawdown | Reserve action | Leverage action |
|---|---|---|
| −10% | Deploy 25% of the reserve into the SPY sleeve | Remain at 3× |
| −20% | Deploy one-third of the remaining reserve into SPY | Reduce to 2× |
| −30% | Deploy one-half of the remaining reserve into SPY | Remain at 2× |
| −40% | Deploy all remaining reserve into quality stocks | Reduce to 1× |

If nothing else changes the reserve, the fractions divide it into four approximately equal tranches. Starting from $20,000, that is approximately $5,000 at each level.

The ladder resets after the total portfolio establishes a new high. Because leveraged portfolio NAV moves much more than SPY, the system generated **51 separate SPY reserve-deployment transactions** over the sample, compared with 19 when SPY drawdown was used as the signal.

### 4. Annual profit harvesting

At each calendar year-end, 10% of a positive annual SPY-sleeve trading profit is moved from the SPY sleeve into the Treasury reserve. A loss-making calendar year produces no harvest, and the annual profit counter then resets.

Across the sample:

- 23 annual harvests occurred; and
- cumulative gross transfers to Treasury were **$776,803**.

These transfers are internal portfolio flows. They are not external contributions and should not be added to terminal wealth.

### 5. Quality-stock deployment

At a −40% portfolio drawdown, the final reserve tranche is divided across the quality basket available at that time. The illustrative basket contains:

- mega-cap compounders: AAPL, MSFT, NVDA, AMZN, GOOGL, META and TSLA;
- durable compounders: BRK-B, COST and MCD; and
- dividend growers: WMT, KO, JNJ, PG, PEP, ADP, LOW and CL.

Each included company must have at least 20 usable quarterly earnings observations in the local dataset and sufficient price history at purchase.

The “1% risk per company” constraint is modeled as a tail-risk sizing limit: a hypothetical 79% loss in one stock must not cost more than 1% of account equity. This is not a stop-loss order. In practice, equal allocation of the available tranche was usually smaller than the tail-risk cap.

### 6. Quality-stock exit policy

For each −40% purchase episode:

1. Record SPY's pre-drawdown adjusted-close high.
2. Hold the quality-stock positions until SPY regains that high.
3. Sell 10% of each original position at SPY recovery.
4. Sell another 10% for every additional 10% SPY advance.
5. Complete the ten-tranche exit by SPY reaching 90% above the recorded high.
6. Liquidate any residual position five years after purchase if the ladder has not completed.
7. Move all sale proceeds back to the Treasury reserve.

The five-year sunset prevents residual individual positions from remaining on the balance indefinitely. In the backtest, it capped completed holding periods at approximately five years with almost no full-period return penalty versus an unlimited recovery ladder.

## How the Treasury/SPY split evolved

The portfolio was dynamic. It did not maintain an 80/20 allocation after inception.

### Allocation statistics across all 8,449 sessions

| Measure | Treasury reserve | SPY sleeve equity | Quality stocks | Gross SPY exposure |
|---|---:|---:|---:|---:|
| Average weight | 6.95% | 92.33% | 0.72% | 1.93× portfolio NAV |
| Median weight | 5.34% | 93.85% | 0.00% | 1.92× portfolio NAV |
| Minimum | 0.00% | 75.58% | 0.00% | 0.76× portfolio NAV |
| Maximum | 24.42% | 99.72% | 4.66% | 2.99× portfolio NAV |
| Ending weight | 0.87% | 97.87% | 1.26% | 2.94× portfolio NAV |

The Treasury reserve was effectively empty on **693 sessions**, or 8.20% of the sample. Quality stocks were present on 3,182 sessions, or 37.66% of the sample.

### Time spent at each leverage level

| Applied leverage | Sessions | Share of sample |
|---|---:|---:|
| 3× | 3,410 | 40.36% |
| 2× | 2,151 | 25.46% |
| 1× | 2,888 | 34.18% |

### Selected portfolio checkpoints

| Date | Portfolio value | SPY sleeve | Treasury | Quality | SPY equity weight | Treasury weight | Quality weight | Gross SPY exposure | Portfolio drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1993-01-29 | $100,000 | $80,000 | $20,000 | $0 | 80.00% | 20.00% | 0.00% | 2.40× | 0.00% |
| 2000-03-24 | $628,872 | $600,631 | $25,236 | $3,005 | 95.51% | 4.01% | 0.48% | 2.87× | −10.50% |
| 2002-10-09 | $255,965 | $225,926 | $27,755 | $2,285 | 88.26% | 10.84% | 0.89% | 0.88× | −63.57% |
| 2007-10-09 | $553,768 | $498,358 | $55,410 | $0 | 89.99% | 10.01% | 0.00% | 1.80× | −21.19% |
| 2009-03-09 | $235,504 | $177,984 | $57,520 | $0 | 75.58% | 24.42% | 0.00% | 0.76× | −66.48% |
| 2020-03-23 | $1,188,882 | $1,142,483 | $0 | $46,399 | 96.10% | 0.00% | 3.90% | 0.96× | −55.68% |
| 2022-10-12 | $2,151,783 | $2,064,165 | $0 | $87,618 | 95.93% | 0.00% | 4.07% | 0.96× | −49.54% |
| 2025-04-09 | $3,319,789 | $3,200,837 | $0 | $118,952 | 96.42% | 0.00% | 3.58% | 0.96× | −34.17% |
| 2026-08-24 | $6,178,113 | $6,046,744 | $53,815 | $77,554 | 97.87% | 0.87% | 1.26% | 2.94× | −5.77% |

The ending allocation illustrates a structural weakness: after recovery, leverage returned to 3× while the Treasury reserve represented less than 1% of the portfolio. The strategy therefore ended the sample near maximum gross exposure with very little immediate dry powder.

## Gross internal capital flows

| Flow | Transactions | Cumulative amount |
|---|---:|---:|
| Treasury deployed into SPY sleeve | 51 | $852,086 |
| Treasury deployed into quality stocks | 5 | $168,588 |
| Annual SPY profit harvested into Treasury | 23 | $776,803 |
| Quality recovery-ladder sales into Treasury | 570 | $229,034 |
| Five-year sunset sales into Treasury | 30 | $23,599 |
| Total quality sale proceeds | 600 | $252,633 |

These figures are cumulative gross flows and include repeated recycling of the same capital. They do not represent independent contributions or withdrawals.

## Quality deployment episodes

| Purchase date | Portfolio drawdown signal | SPY drawdown | Companies | Amount deployed | Approximate amount per company |
|---|---:|---:|---:|---:|---:|
| 1998-09-01 | −42.23% | −15.60% | 12 | $3,103 | $259 |
| 2018-12-24 | −41.36% | −19.35% | 18 | $27,288 | $1,516 |
| 2020-03-13 | −50.94% | −20.40% | 18 | $24,336 | $1,352 |
| 2022-05-12 | −41.52% | −17.62% | 18 | $58,820 | $3,268 |
| 2025-04-09 | −40.32% | −10.22% | 18 | $55,041 | $3,058 |

This table demonstrates the sensitivity of portfolio-based thresholds under leverage. The portfolio can reach a −40% drawdown while SPY is down only 10%–20%.

## Ending quality-stock balance

At the end of the sample, the quality sleeve was worth **$77,554**, or 1.26% of the portfolio. It contained two still-open purchase episodes, from May 2022 and April 2025. The remaining allocated cost was $50,292.

| Ticker | Ending market value | Remaining allocated cost | Value / remaining cost |
|---|---:|---:|---:|
| NVDA | $12,409 | $2,794 | 4.44× |
| GOOGL | $6,768 | $2,794 | 2.42× |
| JNJ | $5,135 | $2,794 | 1.84× |
| AAPL | $4,833 | $2,794 | 1.73× |
| AMZN | $4,533 | $2,794 | 1.62× |
| WMT | $4,022 | $2,794 | 1.44× |
| MSFT | $4,000 | $2,794 | 1.43× |
| META | $3,972 | $2,794 | 1.42× |
| KO | $3,966 | $2,794 | 1.42× |
| TSLA | $3,752 | $2,794 | 1.34× |
| COST | $3,536 | $2,794 | 1.27× |
| BRK-B | $3,122 | $2,794 | 1.12× |
| CL | $3,110 | $2,794 | 1.11× |
| ADP | $3,108 | $2,794 | 1.11× |
| LOW | $2,936 | $2,794 | 1.05× |
| PEP | $2,872 | $2,794 | 1.03× |
| MCD | $2,779 | $2,794 | 0.99× |
| PG | $2,702 | $2,794 | 0.97× |

## Performance comparison

| Metric | Latest strategy | SPY buy-and-hold |
|---|---:|---:|
| Starting value | $100,000 | $100,000 |
| Ending value | $6,178,113 | $3,165,186 |
| Wealth multiple | 61.78× | 31.65× |
| CAGR | 13.07% | 10.84% |
| Maximum drawdown | −66.48% | −55.19% |
| Annualized volatility | 31.32% | 18.55% |
| Sharpe versus Treasury | 0.469 | 0.511 |
| Worst day | −21.66% | −10.94% |
| Time at least 20% below high | 59.64% | 16.90% |

The strategy's maximum drawdown began from its July 17, 1998 portfolio high, reached its trough on March 9, 2009, and did not recover that high until December 23, 2013. An investor would have spent more than 15 years waiting to regain the 1998 peak.

The interactive growth and drawdown comparison is available in [strategy-vs-spy.html](./strategy-vs-spy.html).

## Rolling 20-year analysis

Fourteen annual-start 20-year cohorts were tested.

| Statistic | Latest strategy CAGR | SPY CAGR |
|---|---:|---:|
| Minimum | 5.13% | 5.58% |
| 10th percentile | 5.75% | 6.35% |
| Median | 8.82% | 8.70% |
| 90th percentile | 12.68% | 10.17% |
| Maximum | 13.22% | 10.83% |

The strategy beat SPY terminal wealth in 9 of 14 cohorts, or 64.29%. Its median rolling maximum drawdown was −66.80%, versus −55.19% for SPY.

The rolling evidence is materially less impressive than the full-period result. The median CAGR advantage was only 0.12 percentage points while the drawdown disadvantage remained substantial.

## Main findings

### The actual split was not continuously 80/20

The portfolio was 80% SPY sleeve and 20% Treasury only at inception. Across the full sample, Treasury averaged 6.95%, while the SPY sleeve averaged 92.33% of account equity. The strategy systematically migrated capital from reserve into SPY and quality stocks, then partially rebuilt reserve through harvesting and sales.

### The reserve was too small relative to the leverage

A 20% reserve appears meaningful next to an unlevered 80% SPY position. It is small relative to 240% initial SPY exposure. The initial reserve equaled only 8.33% of gross SPY exposure.

### Portfolio drawdown is the correct risk signal, but thresholds need calibration

Using portfolio NAV rather than SPY drawdown improved the tested maximum drawdown dramatically compared with the SPY-signal version. However, 3× leverage caused frequent threshold activations and reserve exhaustion during ordinary SPY corrections.

### The strategy earned more but did not use risk more efficiently

Full-period CAGR exceeded SPY by 2.23 percentage points, but the Sharpe ratio was lower and the maximum drawdown was more severe. The rolling median CAGR advantage was nearly negligible.

### The ending state remained aggressive

At the end of the simulation, gross SPY exposure was 2.94× portfolio NAV and Treasury was only 0.87%. A robust strategy should not depend on the sample ending before the next adverse move.

## Principal limitations

1. **Survivorship and look-ahead bias.** The quality basket uses current well-known companies and present-day earnings coverage. It is not a point-in-time investable universe.
2. **Daily-reset leverage model.** The simulation applies daily leveraged SPY returns and financing. It is not an exact history of a specific leveraged ETF, futures program or margin account.
3. **Unmodeled implementation risks.** Taxes, leveraged-fund expenses, tracking error, bid/ask spreads beyond the stock cost assumption, margin calls, liquidation, trading halts and market impact are omitted.
4. **Adjusted-close execution.** Adjusted closes incorporate distributions and splits, but are not directly executable historical prices.
5. **Small number of deep quality deployments.** Only five −40% portfolio episodes occurred in the full sample.
6. **Limited rolling cohorts.** Fourteen annual-start 20-year cohorts are informative but not independent observations.
7. **Tail-risk sizing is not a stop.** The 1% per-stock rule is a stress-based position cap, not a guarantee that losses will be limited to 1%.
8. **Path dependence.** Daily leverage, annual harvesting, drawdown resets and staged deployments make results highly dependent on the exact order of returns.

Leveraged ETFs generally target a daily multiple, and longer-period returns can diverge significantly from the stated multiple, particularly in volatile markets. See the [SEC/Investor.gov leveraged ETF bulletin](https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-alerts/sec) and [FINRA's geared ETP overview](https://www.finra.org/investors/insights/lowdown-leveraged-and-inverse-exchange-traded-products).

## Recommended next research version

The backtest does not support deploying the strategy unchanged. The next research version should test:

1. starting leverage of 1.5× or 2× rather than 3×;
2. a permanent Treasury floor of at least 10% of portfolio NAV;
3. leverage reduction before reserve deployment at the first drawdown threshold;
4. a maximum gross exposure cap after portfolio recovery;
5. point-in-time market capitalization, earnings quality and universe membership;
6. monthly or quarterly cohort starts for more complete start-date sensitivity; and
7. explicit taxes, fund expenses, slippage and liquidation constraints.

A safer example control schedule would be:

| Portfolio drawdown | Possible next-version action |
|---|---|
| −10% | Reduce leverage; do not deploy reserve yet |
| −20% | Reduce leverage further and deploy first reserve tranche |
| −30% | Reach 1× leverage and deploy second tranche |
| −40% | Deploy quality-stock tranche |
| −50% | Deploy remaining discretionary reserve, preserving the permanent Treasury floor |

## Data and reproducibility

- Strategy implementation: [`scripts/run_spy_quality_rotation.py`](../scripts/run_spy_quality_rotation.py)
- Full result record: [`out/strategy/spy_quality_rotation/results.json`](../out/strategy/spy_quality_rotation/results.json)
- Daily portfolio history: [`out/strategy/spy_quality_rotation/daily.csv`](../out/strategy/spy_quality_rotation/daily.csv)
- Allocation and transaction events: [`out/strategy/spy_quality_rotation/events.csv`](../out/strategy/spy_quality_rotation/events.csv)
- Rolling 20-year cohorts: [`out/strategy/spy_quality_rotation/rolling_20y.csv`](../out/strategy/spy_quality_rotation/rolling_20y.csv)
- Interactive comparison: [`docs/strategy-vs-spy.html`](./strategy-vs-spy.html)
- Treasury series: [FRED DGS3MO](https://fred.stlouisfed.org/series/DGS3MO)

The associated strategy tests cover leverage bands, next-close reserve deployment, portfolio-versus-SPY signal behavior, and SPY-recovery quality exits.
