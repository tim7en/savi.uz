# Documents

Rendered deliverables from the systematic-trading research programme. Each is a
self-contained HTML page — open directly in a browser, no build step and no
network access required.

These are kept here rather than under `out/`, which is gitignored because it
holds regenerable analysis output. A report someone reads is not regenerable
output; losing it to a clean checkout would be losing work.

## The programme in one paragraph

A long-only breakout system on US equities, tested to destruction. Twenty-six
proposals have been put through pre-registered controls. **Everything that tried
to predict failed; everything that survived was about execution, sizing or
timeframe.** The best configuration measured is a four-hour Donchian breakout
with two-unit pyramiding at Sharpe 0.95 against true taker cost — but its edge
scales with how much the underlying rose (1.33 on the best third of names, 0.48
on the worst), it has never been validated out-of-sample, and the universe
contains no delisted companies. The honest position is a candidate, not a
strategy.

## Reading order

| File | What it is | Published as |
|---|---|---|
| [`where-the-market-is-not-efficient.html`](where-the-market-is-not-efficient.html) | **Start here.** All 26 proposals sorted by what they tried to do, each with the control that decided it. | [artifact](https://claude.ai/code/artifact/128d4a7d-84dd-46a1-b03b-0031f36a52ea) |
| [`the-interval-was-the-parameter.html`](the-interval-was-the-parameter.html) | Seven bar sizes at real costs. The banked 30-minute book sits in a trough; four-hour is five times better. | [artifact](https://claude.ai/code/artifact/607cd415-38c6-4bf6-a0dc-edbdf751958d) |
| [`where-the-bar-closes.html`](where-the-bar-closes.html) | The largest conditional effect in the programme, and why it cannot be traded. | [artifact](https://claude.ai/code/artifact/be73089e-ed78-47b8-b920-6f4f9b83f4f9) |
| [`13f-the-filing-and-the-slot.html`](13f-the-filing-and-the-slot.html) | Chapters six and seven consolidated: what 13F filings can and cannot buy. | [artifact](https://claude.ai/code/artifact/3f6dccbe-cded-4b6c-8e32-dc6163ccce77) |
| [`six-proposals-one-survivor.html`](six-proposals-one-survivor.html) | Seven proposals in one sitting — ML filter, 3:1 scalp, S/R zones, 20x leverage, volatility surprise, option activity, earnings drift. | Local draft |
| [`research-agenda.html`](research-agenda.html) | Open experiments with kill criteria fixed before they run. The standing controls contract lives here. | [artifact](https://claude.ai/code/artifact/93859ddd-3e9d-40f5-aac9-56a14a675f61) |
| [`report-blueprint.html`](report-blueprint.html) | Chapter-by-chapter plan for the investor report. | [artifact](https://claude.ai/code/artifact/a8e1c677-b85c-4623-8a7c-c22bca8168e2) |
| [`chapter-06-the-research-department.html`](chapter-06-the-research-department.html) | The full six-chapter cross-asset book; chapter VI is the 13F work. | Local draft |
| [`chapter-07-when-six-slots-are-scarce.html`](chapter-07-when-six-slots-are-scarce.html) | 13F conviction as a six-slot allocation priority. | Local draft |

> **Superseded.** `research-agenda.html` still quotes Sharpe 2.64 for the banked
> 30-minute book at 2bp. Both legs of that strategy are stop orders and therefore
> taker fills, so **10bp is the applicable cost and the same book scores 0.20**,
> with a 5–95% band including zero. Every overlay in that ledger was tested
> against a configuration since shown to be near-worst in its own parameter range.
> The agenda needs a revision it has not had.

> **Missing.** `chapter-02-the-quarter-century.html` was deleted in commit
> `2c70b4e` and is superseded by chapter one of the six-chapter book inside
> `chapter-06-the-research-department.html`.

## What has been established

**Nothing predictive survived.** Dealer gamma, CFTC positioning, macro regime
(three separate constructions), earnings features, volume bursts, volume-profile
location, theme strength, relative-strength ranking, three moving-average
crossovers, option-surface machine learning, pre-breakout volume, range
compression, and single-name drawdown spillover all died to a reversal test, a
null, or a redundancy check.

**Six structural findings survived.** A slower bar (0.20 → 0.95 from 30 minutes
to four hours). Price momentum (IC 0.042, *p* < 0.001). Half size below the
moving average (beats its reversal at 50, 100 and 200 sessions). Two-unit rather
than four-unit pyramiding (Sharpe 1.13 peak, right tail also peaks there). A VWAP
limit entry at four hours (1.28 against a random-in-regime null of 0.79, 15 of 15
draws). Gold and miners as the only cross-asset sleeve that hedges (GDX +6.3%,
67% hit rate across 15 market drawdowns).

**Four facts that govern everything else.**

- **A stop order is a taker.** Entry and exit are both stops, so 10bp applies,
  not 5bp. Every historical figure in this repo quoted below that is optimistic.
- **Exposure is not a strategy.** Constant 70%, 85% and 100% produce identical
  Sharpe once matched to drawdown. Leverage re-parameterises; it does not improve.
- **Effective sample ≠ row count.** CFTC positioning: 991 filings, 15–45
  independent observations. The option ML panel: 31,783 rows, 246 independent
  dates.
- **Random capacity tie-breaks span a [0.57–0.96] Sharpe band.** Any comparison
  not paired on identical orderings cannot see an effect smaller than 0.4.

**The threat to all of it.** The universe is Binance's trad-FI list as it stands
today. Splitting by each name's own outcome gives Sharpe 1.33 on the third that
rose most and **0.48 on the worst third** — whose median member still returned
+54%, because nothing here was delisted. On money-weighted terms the book returns
18.1% IRR against SPY's 15.0% with monthly contributions, and that +3.1pp is the
same order as the survivorship bias, before tax on ~9,000 trades taxed annually
while an index defers.

## Data

Everything under `data/` is gitignored. Sizes are approximate.

| Path | What | Coverage |
|---|---|---|
| `data/intraday/bars_av.db` | Alpha Vantage consolidated 5-minute bars, 137 symbols | 2015-01 → 2026-07, 22.3M bars |
| `data/options/alphavantage.db` | EOD option chains reduced to daily aggregates (GEX, IV, skew, put/call) | 2015 → 2026 around events; a full year 2025-08 → 2026-07 |
| `data/13f/holdings.db` | 13F-HR holdings, 7 concentrated managers | 2013 → 2026, 96,171 holdings |
| `data/cross_assets/etf_30min.db` | 21 non-equity ETFs, 30-minute | 2017 → 2026 partial |
| `data/data/cross_assets/etf_daily.db` | Same 21 ETFs, daily | 2002 → 2026 |
| `data/data/macro/macro.db` | FRED/ALFRED, GSW curve, Fed path, VIX | 1990 → 2026 |
| `data/data/cftc/cot.db` | Commitments of Traders + derived release calendar | 2000 → 2026 |
| `data/data/equity/equity.db` | Index prices, SEC facts, earnings schema | 2000 → 2026 |
| `data/data/sp500_data/` | Per-symbol Alpha Vantage fundamentals and earnings JSON | 7,370 earnings files |

**The old 5-minute store (`data/data/intraday/bars.db`) is IEX-only and its
volume is unusable**: 1.89% of consolidated volume, no volume at all on 13.5% of
bars, and a rank correlation of 0.625 with the real tape on which bars were busy.
Two volume overlays were rejected on that series. Use `bars_av.db`.

## Studies

Each writes JSON to `out/strategy/`. All take `--help`.

| Script | Question |
|---|---|
| `run_parameter_sensitivity.py` | Entry-window and volume surface at maker/taker costs |
| `run_fast_confirm_study.py` | Interval ladder and confirmation cost, 5min → daily |
| `run_dip_buy_study.py` | VWAP limit entry against a random-in-regime null |
| `run_pyramid_breadth_study.py` | How far to pyramid; does the edge survive on names that fell |
| `run_account_simulation.py` | $10k with monthly contributions, against SPY, with exposure |
| `run_stress_episode_study.py` | Every SPY drawdown ≥10%, and VIX-quintile conditionals |
| `run_drawdown_event_study.py` | Option surface and cross-asset response around events |
| `run_option_ml_study.py` | Do option/earnings features beat price? (purged walk-forward) |
| `run_ma_regime_sizing.py`, `run_macro_gate_study.py`, `run_vix_sizing_study.py`, `run_ma_crossover_study.py` | Regime tilts, each against its own reversal |
| `run_capacity_priority_study.py` | Relative strength as a slot tie-break |
| `run_reserve_deployment_study.py` | Holding cash for a drawdown trigger |
| `compare_volume_sources.py` | IEX against consolidated volume |
| `run_cftc_positioning_study.py`, `repair_cftc_calendar.py` | Positioning, and the release-date repair it needed |
| `download_tradfi_bars.py`, `download_alphavantage_options.py`, `download_13f_holdings.py`, `update_tradfi_symbols.py` | Acquisition |

`download_alphavantage_options.py --targets` takes a JSON list of
`[symbol, date]` pairs, so an event study can fetch only the sessions it needs
instead of whole date ranges.

## The standing controls

Applies to every study. Each rule was written after a specific mistake.

1. **Matched drawdown, always.** Anything that changes exposure is scaled to the
   same median peak-to-trough loss before its return is read.
2. **Beat your own reversal.** A rule performing no better than its opposite
   carries no information, whatever its *p*-value.
3. **Persistence-matched nulls.** Autocorrelated state variables get a
   circular-shift control that keeps run lengths and destroys alignment.
4. **The kill criterion is written before the run.**
5. **Report effective sample size, not row count.**
6. **Quote costs at the level actually payable**, by execution type — a stop is a
   taker, a resting limit is a maker.
7. **Pair every comparison on identical capacity orderings.**
8. **Mark to market daily**, never at exit.
9. **Trailing percentiles and a one-session lag.** Never full-sample boundaries.
10. **Report the reversal.** Withdrawn results stay in the record.

## What has never been done

- **Out-of-sample validation.** No result in this repo has been scored on data
  held out with parameters frozen first. This is the largest single gap.
- **A point-in-time universe** containing delisted names, which would price the
  survivorship bias instead of estimating it.
- **Correlation-grouped capacity.** 72% of breakouts are refused at 30-minute
  bars, and the tie-break is decided at random.
- **Tax and funding** in any comparison against a passive index.
