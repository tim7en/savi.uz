# Dual-Guard Quality Compounder

## Operating guide

This strategy separates permanent capital from opportunistic capital. The
permanent portfolio remains invested without leverage. Treasury is reserved for
market drawdowns. Only new capital contributed while the account is underwater
may use leverage, and CAPE determines the size of that leverage.

The strategy therefore answers three different questions with three different
signals:

| Decision | Signal | Purpose |
|---|---|---|
| When to deploy Treasury | SPY total-return drawdown | Buy broad-market or quality assets during market declines |
| Whether a new contribution may use leverage | Flow-adjusted account NAV drawdown | Prevent leverage from being added near an account high |
| How much leverage a qualifying contribution may use | Prior-known Shiller CAPE | Use more exposure only when broad valuations are lower |

These signals are deliberately not interchangeable. SPY drawdown governs the
Treasury ladder. Account NAV drawdown governs new-capital eligibility. CAPE
governs only the leverage of an eligible new SPY tranche.

## 1. Starting portfolio

Treat the portfolio as two funded sleeves and one future funding source:

1. **Permanent SPY sleeve — 80% of account NAV.** Hold SPY at 1x. Reinvest
   dividends. Do not sell this sleeve merely because CAPE is high, and do not
   leverage it merely because CAPE is low.
2. **Treasury sleeve — 20% of account NAV.** Hold short-duration Treasury
   instruments or equivalent cash collateral. This sleeve earns the prior-known
   three-month Treasury rate in the backtest.
3. **Additional capital — outside the account until contributed.** The tested
   schedule is $10,000 at the beginning of every later year plus an additional
   $30,000 every third contribution year. Each contribution is divided 80% to
   the SPY decision and 20% to Treasury.

“Stay invested in SPY” means the permanent risk sleeve is always fully invested
at 1x. It does not mean the Treasury reserve is also exposed to SPY before a
drawdown signal.

## 2. Two drawdowns, two jobs

### SPY drawdown

SPY drawdown is calculated from SPY adjusted total return, including reinvested
dividends:

`SPY drawdown = current SPY total-return level / prior SPY high - 1`

It controls Treasury mobilization. A signal observed at one close is executed
on the following trading session in the backtest.

### Account NAV drawdown

Account drawdown is calculated from a flow-adjusted performance index. Deposits
are removed from the return clock, so adding cash cannot manufacture a recovery:

`NAV drawdown = flow-adjusted account index / prior account high - 1`

It controls whether the SPY portion of a new contribution is eligible for
leverage. The tested threshold is a 10% NAV drawdown.

## 3. Treasury mobilization

At each new SPY total-return high, record the Treasury balance. That amount is
the budget for the next drawdown episode. Deploy it once at each rung:

| SPY drawdown | Share of episode Treasury | Destination | Leverage |
|---:|---:|---|---:|
| -10% | 20% | Seven largest eligible companies at that date | 1x |
| -20% | 30% | SPY | 1x |
| -30% | 30% | Seven largest eligible companies at that date | 1x |
| -50% | 20% | SPY | 1x |

The two quality rungs are market-cap weighted. In live use, “quality seven”
should be selected using information available on the decision date. The
backtest uses a date-ranked historical-leader union, but it is not a complete
survivor-free universe and therefore carries survivorship risk.

If SPY crosses several unused rungs between observations, all newly crossed
rungs are eligible. A rung fires only once between one SPY high and its eventual
recovery.

### What happens at SPY recovery?

When SPY regains its old total-return high, the ladder is re-armed. Existing SPY
and quality purchases are not automatically sold. The actual Treasury remaining
at recovery becomes the next episode budget. This means Treasury must be rebuilt
through new contributions, Treasury interest and quality-stock harvesting; a
recovery does not magically recreate spent cash.

## 4. Quality-stock harvesting

Every quality lot is compared with SPY from its own purchase date:

`relative excess = quality-stock wealth / SPY wealth - 1`

At quarter boundaries and new ladder events, sell 5% of the lot's original
shares for every newly crossed 20-percentage-point relative-outperformance band.
For example, crossing +20% relative performance sells 5% of original shares;
crossing +40% sells another 5%.

Route proceeds in this order:

1. Rebuild the Treasury target that existed before the drawdown.
2. When prior-known CAPE is at least 35, rebuild Treasury to at least 20% of
   current NAV before buying more SPY.
3. After the applicable Treasury target is full, invest additional proceeds in
   unleveraged SPY.

Quality and Treasury-ladder positions are never leveraged in the tested model.

## 5. Mobilizing additional capital

Every scheduled contribution is split 80/20. The 20% Treasury portion always
goes to Treasury. The decision below applies only to the 80% SPY portion.

### Step 1 — Check account NAV

- If prior-close flow-adjusted NAV drawdown is less than 10%, buy SPY at 1x.
- If NAV drawdown is 10% or deeper, the contribution qualifies for valuation-
  conditioned leverage. Continue to the CAPE test.

This makes NAV a gate, not a leverage multiplier. A 30% NAV drawdown does not by
itself authorize more leverage than a 10% drawdown; CAPE makes that decision.

### Step 2 — Check prior-known CAPE

| Prior-known CAPE | New SPY tranche | Interpretation |
|---:|---:|---|
| Above 35 | 1x | Expensive market: invest, but do not borrow |
| 25 through 35 | 2x | Mild valuation: moderate leverage |
| Below 25 | 3x | Lower valuation: maximum tested tranche leverage |

CAPE is lagged by one month in the backtest. The rule never uses the final
revised value for the current month.

### Step 3 — Hold the tranche until account recovery

The entry leverage is fixed for that tranche. It does not change every time CAPE
moves between bands. When the combined flow-adjusted account NAV regains its
pre-drawdown high, convert all outstanding leveraged injection tranches to 1x
and merge them into the permanent SPY sleeve. The next drawdown begins with a
clean account-level high-water mark.

This recovery rule is crucial. Without it, new-capital leverage becomes a
permanent leveraged portfolio instead of a temporary drawdown deployment.

## 6. The role of CAPE

CAPE is a slow valuation measure, not a crash predictor. In this strategy it has
one narrow function: **size a new SPY contribution after NAV has already opened
the drawdown gate.**

CAPE does not:

- cause the permanent SPY sleeve to be sold;
- determine when Treasury rungs fire;
- select the seven quality stocks;
- predict the exact market bottom; or
- guarantee that 3x exposure is safe.

Cheap markets can become cheaper. In 2009, low CAPE authorized 3x injection
tranches while the market was still falling. NAV gating limits where leverage
can start, but it cannot eliminate path risk after entry.

### VIX overlay test — smoothed candidate

We also tested VIX as a dynamic ceiling on each active injection, using only the
prior close and its percentile versus the preceding 252 observations:

- below the 70th percentile: allow the CAPE leverage ceiling;
- 70th through 90th percentile: cap the tranche at 2x; and
- at or above the 90th percentile: cap the tranche at 1x.

Leverage was allowed to rise again if VIX subsided before NAV recovered. A
reversed control used 1x at low VIX, 2x in the middle and the CAPE ceiling at
high VIX.

The intuitive brake was not strong enough to adopt. Over 1993–2024 it ended at
$3,969,543 versus $4,019,327 without VIX, while maximum drawdown improved only
from -52.60% to -52.40%. Over 2007–2024 it improved maximum drawdown from
-55.58% to -53.51%, but ended 0.34% behind the no-VIX version. It also caused
276 lot-level leverage changes on 145 dates in the full sample, before charging
switching costs. The reversed control mostly behaved as a lower-leverage
strategy and does not establish that high VIX should invite more leverage.

We then ranked trailing simple averages of VIX rather than the raw close. The
5-, 20- and 60-session averages and a monthly-held raw signal were specified
before comparing their results. The 60-session version was the only one that
added full-sample return: terminal wealth reached $4,119,424, XIRR was 10.245%,
maximum drawdown was -52.33%, and leverage changed on only 11 dates. Over
2007–2024 it reached $1,240,949 with a -50.86% maximum drawdown, versus
$1,248,235 and -55.58% without VIX.

This makes the 60-session rule a candidate, not a validated rule. It was the
best of four smoothing choices selected on the same history, switching costs
remain omitted, and maximum account exposure remained above 2x. Freeze the
lookback before further testing, include switching costs, and combine it with
an account-level gross-exposure cap.

We separately tested withholding each scheduled $10,000 annual contribution in
a DGS3MO waiting pool, then buying unlevered SPY in four equal episode-base
rungs when the 60-session VIX percentile reached 70%, 80%, 90% and 95%. The
ladder re-armed below 50%; the $30,000 triennial contribution kept its ordinary
rule. This reduced 1993–2024 terminal wealth to $3,933,756 and XIRR to 10.020%,
versus $4,119,424 and 10.245% when annual cash was deployed normally. Maximum
drawdown worsened slightly, from -52.33% to -52.83%. The rule is rejected:
contribute on schedule and use VIX to constrain leverage, not to time the whole
annual deposit.

### Experimental standing-leverage alternative

A separate test starts the permanent 80% SPY core at 3x rather than limiting
leverage to new drawdown contributions. Prior-close flow-adjusted NAV reduces
the core to 2x at a 10% drawdown and 1x at a 20% drawdown. The slower reversal
restores 2x only after NAV rises above -10% and restores 3x only at a new NAV
high. Actual leverage is then capped by the lower of:

- NAV state: 3x / 2x / 1x;
- prior-known CAPE: 3x below 25, 2x from 25 through 35, 1x above 35; and
- 60-session VIX percentile: 3x below 70%, 2x through 90%, 1x above 90%.

The 20% Treasury and quality/Mag-7 drawdown ladder remain unchanged. Exposure
above 1x pays prior-known DGS3MO plus 1%.

| 1993–2024 path | Terminal | XIRR | Max DD | Volatility | Max gross |
|---|---:|---:|---:|---:|---:|
| Raw 3x with symmetric NAV reversals | $6,139,880 | 12.168% | -60.11% | 25.83% | 2.82x |
| Raw 3x with slow NAV reversals | $5,718,794 | 11.829% | -57.78% | 23.58% | 2.79x |
| Slow NAV + CAPE/VIX ceilings | $5,659,269 | 11.779% | -56.51% | 21.58% | 2.74x |
| Same, SPY core dividends to Treasury | $5,579,453 | 11.711% | -52.72% | 19.85% | 2.48x |
| Selective-injection dual guard | $4,119,424 | 10.245% | -52.33% | 17.60% | 1.17x |
| SPY 1x | $4,058,936 | 10.173% | -55.19% | 18.65% | 1.00x |

Routing actual SPY core cash dividends to Treasury moved $571,409 over the full
sample. It surrendered some terminal wealth but materially reduced volatility,
drawdown, financing cost and average exposure. Treasury interest performed
better when left to compound in Treasury; sweeping it annually into SPY was
worse and changed later NAV threshold crossings, demonstrating substantial path
dependence.

This standing alternative is economically stronger but operationally more
dangerous than selective drawdown injections. Maximum gross account exposure
still reached 2.48x, leverage-switching costs are omitted, and a prior-close NAV
rule cannot prevent gap or margin-liquidation risk. It is an experimental
candidate only until a hard account exposure cap and rolling-start validation
are tested.

## 7. Leverage and funding

Leverage applies to tranche equity, not to the entire account. An $8,000 SPY
tranche at 3x creates $24,000 of SPY exposure and $16,000 of financed exposure.
The backtest models daily-rebalanced exposure and charges the prior-known
three-month Treasury rate plus a 1% annual spread on exposure above 1x.

Permanent SPY, Treasury deployments, Mag-7 purchases and harvested-proceeds SPY
are all unleveraged.

### Recommended account exposure ceiling

The reported backtest does **not** yet impose a hard account exposure ceiling.
That omission matters most in a young account, when a $30,000 supplemental
contribution can be much larger than the original $10,000 balance. The 2007
sensitivity reached 2.16x total gross exposure even though only new capital was
eligible for leverage.

A proposed implementation constraint is 1.5x total account gross exposure. For
a new contribution:

1. Calculate leverage from CAPE.
2. Calculate projected gross exposure after the trade.
3. Reduce the tranche leverage until projected account exposure is no more than
   1.5x.
4. If the account is already at the cap, buy the contribution at 1x or retain
   the SPY portion as collateral until capacity becomes available.

This cap is a recommendation for the next test, not part of the performance
figures below.

## 8. Worked example

Assume the previous flow-adjusted account high is $100,000 and current
flow-adjusted NAV is $85,000, a 15% drawdown. A $10,000 contribution arrives:

- $2,000 enters Treasury.
- $8,000 is allocated to the SPY decision.
- Because NAV drawdown is deeper than 10%, check CAPE.
- If prior-known CAPE is 23, the desired tranche leverage is 3x.
- The $8,000 tranche creates $24,000 of SPY exposure, of which $16,000 is
  financed.
- Existing permanent SPY remains at 1x.
- When combined flow-adjusted account NAV recovers its old high, reduce that
  tranche to 1x and merge its remaining equity into permanent SPY.

If a 1.5x account cap would be breached, reduce the new tranche below 3x even
though CAPE is below 25. The account cap overrides CAPE.

## 9. Historical result

The full test covers 29 January 1993 through 30 September 2024. It begins with
$10,000 and adds $10,000 annually plus $30,000 every third contribution year.
Total capital contributed is $620,000.

| Strategy | Terminal wealth | XIRR | Maximum drawdown | Volatility |
|---|---:|---:|---:|---:|
| Dual guard + 60-session VIX brake | $4,119,424 | 10.245% | -52.33% | 17.60% |
| Dual guard + raw VIX brake | $3,969,543 | 10.064% | -52.40% | 17.57% |
| Dual-guard quality strategy | $4,019,327 | 10.125% | -52.60% | 17.70% |
| Same quality strategy, 1x core | $3,927,534 | 10.012% | -52.04% | 17.22% |
| SPY 1x | $4,058,936 | 10.173% | -55.19% | 18.65% |

The dual guard finished 0.98% behind SPY, with a 2.59-percentage-point smaller
maximum drawdown. Modeled financing cost was $16,105. Eight contributions used
2x or 3x leverage: 2002–2005, 2009–2010, 2019 and 2023.

In the 2007–2024 sensitivity, the strategy reached $1,248,235 versus $1,239,101
for SPY, but maximum drawdown was -55.58% versus -55.19% for SPY. This is not a
large or robust advantage.

## 10. Operating checklist

At every contribution date:

1. Record the prior-close flow-adjusted NAV and its high-water mark.
2. Calculate NAV drawdown without treating deposits as profit.
3. Send 20% of the contribution to Treasury.
4. If NAV drawdown is shallower than 10%, buy SPY at 1x with the remaining 80%.
5. If NAV drawdown is at least 10%, retrieve the prior-known monthly CAPE.
6. Assign 1x, 2x or 3x from the CAPE table.
7. Apply the proposed total-account exposure cap before execution.
8. Record tranche equity, leverage, entry date, CAPE and NAV drawdown.
9. Charge and record funding daily.
10. At account recovery, reduce all outstanding injection tranches to 1x.

At every market close:

1. Update SPY total-return drawdown for Treasury rungs.
2. Update flow-adjusted account NAV drawdown for the contribution gate and
   recovery reset.
3. Check whether SPY crossed an unused Treasury rung.
4. At quarter-end, check each quality lot's relative-performance harvest bands.
5. Keep the Treasury budget, spent rungs and leveraged injection ledger
   separately auditable.

## 11. Principal risks

- **Leverage path risk.** A 3x tranche can lose approximately 30% before funding
  on a subsequent 10% SPY fall.
- **Capital-scale risk.** A large contribution can dominate a young account.
- **CAPE timing risk.** Low CAPE does not identify the market bottom.
- **Margin and implementation risk.** Real margin requirements, futures rolls,
  leveraged-ETF decay, tracking error and forced liquidation are omitted.
- **Concentration risk.** The quality rungs can concentrate Treasury in seven
  mega-cap companies already represented heavily in SPY.
- **Historical-universe bias.** The backtest lacks a complete point-in-time
  listed and delisted universe for quality selection.
- **Treasury depletion.** Drawdown positions are not automatically sold at SPY
  recovery, so the next ladder may have less cash available.
- **Tax risk.** Harvests and deleveraging can realize taxable gains; taxes are
  omitted.
- **Threshold risk.** The 10%, 25 and 35 boundaries were tested in sample and
  have not been frozen for a true out-of-sample period.

## 12. Rule summary

> Hold the permanent SPY sleeve at 1x and maintain 20% Treasury. Deploy Treasury
> from the SPY drawdown ladder. Use account NAV drawdown only to decide whether
> new contributions may be levered. If NAV is down at least 10%, let prior-known
> CAPE choose 1x, 2x or 3x for the new SPY tranche. Return every leveraged
> injection to 1x when the combined account recovers its previous high. Never
> leverage Treasury or quality-stock purchases, and impose an account-level
> exposure cap before using the strategy with real money.

Historical simulation only, not investment advice. CAPE source: [Robert Shiller
/ Yale](https://www.econ.yale.edu/~shiller/data.htm). Treasury and funding
reference: [FRED DGS3MO](https://fred.stlouisfed.org/series/DGS3MO). See the
[backtest report](quality-ladder-cape-leverage.html) for equity curves, P&L,
drawdowns and comparison tables.
