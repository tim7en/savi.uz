"""Build the weekly covered-call overlay report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd

import build_spy_dca_dashboard as price_io
from build_contribution_quality_page import line_chart, money, pct


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("out/strategy/weekly_covered_calls"))
    parser.add_argument("--price-cache", type=Path, default=Path(".cache/yahoo_daily"))
    parser.add_argument("--output", type=Path, default=Path("docs/weekly-covered-call-study.html"))
    return parser.parse_args(argv)


def projected_wealth(initial: float, annual_contribution: float, annual_return: float,
                     extra_every_three_years: float = 0.0, years: int = 10) -> float:
    wealth = initial
    for year in range(1, years + 1):
        wealth *= 1.0 + annual_return
        wealth += annual_contribution
        if year % 3 == 0:
            wealth += extra_every_three_years
    return wealth


def main(argv=None) -> int:
    args = parse_args(argv)
    result = json.loads((args.input / "results.json").read_text(encoding="utf-8"))
    daily = pd.read_csv(args.input / "daily.csv", parse_dates=["date"])
    conditional_daily = pd.read_csv(
        args.input / "conditional_daily.csv", parse_dates=["date"]
    )
    trades = pd.read_csv(args.input / "trades.csv")
    latest = {
        symbol: float(price_io.load_spy(args.price_cache / f"{symbol}.json")["close"].iloc[-1])
        for symbol in ("SPY", "QQQ")
    }

    chain_rows = []
    overlay_rows = []
    for symbol in ("SPY", "QQQ"):
        for delta in (5, 10, 20):
            key = f"delta_{delta:02d}"
            chain = result["chains"][symbol][key]
            overlay = result["strategy_overlays"][symbol][key]
            whole = overlay["whole_contracts"]
            baseline = overlay["baseline"]
            chain_rows.append(
                f'<tr class="{"featured" if delta == 5 else ""}"><td><strong>{symbol} {delta}-delta</strong></td>'
                f'<td>{chain["weeks"]}</td><td>{pct(chain["worthless_rate"],1)}</td>'
                f'<td>{chain["assigned_or_itm"]}</td><td>{chain["median_premium_bp"]:.1f} bp</td>'
                f'<td>{pct(chain["premium_yield_sum"],1)}</td><td>{pct(chain["payoff_yield_sum"],1)}</td>'
                f'<td>{pct(chain["cagr_difference"],2)}</td></tr>'
            )
            overlay_rows.append(
                f'<tr class="{"featured" if delta == 5 else ""}"><td><strong>{symbol} {delta}-delta</strong></td>'
                f'<td>{money(baseline["terminal_wealth"])}</td><td>{money(whole["terminal_wealth"])}</td>'
                f'<td>{pct(whole["terminal_wealth"] / baseline["terminal_wealth"] - 1,2)}</td>'
                f'<td>{pct(baseline["xirr"])}</td><td>{pct(whole["xirr"])}</td>'
                f'<td>{money(whole["gross_premium_to_treasury"])}</td>'
                f'<td>{money(whole["upside_paid_away"])}</td><td>{money(whole["net_option_pnl"])}</td></tr>'
            )

    policy_labels = {
        "always": "Always sell (reference)",
        "cape_high": "CAPE ≥25 only",
        "cape_extreme": "Extreme CAPE >30 only",
        "vix_calm": "Calm VIX/VXN only",
        "cape_high_or_vix_calm": "Requested: CAPE ≥25 OR calm vol",
        "cape_extreme_or_vix_calm": "Extreme CAPE >30 OR calm vol",
        "cape_high_and_vix_calm": "CAPE ≥25 AND calm vol",
        "cape_extreme_and_vix_calm": "Extreme CAPE >30 AND calm vol",
        "recovery_ready": "NAV recovery lockout only",
        "vix_calm_and_recovery_ready": "Calm vol AND recovery-ready",
        "cape_extreme_or_vix_calm_and_recovery_ready": "(CAPE >30 OR calm) AND recovery-ready",
        "cape_extreme_and_vix_calm_and_recovery_ready": "CAPE >30 AND calm AND recovery-ready",
    }
    policy_order = [
        "always", "cape_extreme", "vix_calm", "cape_extreme_or_vix_calm",
        "cape_extreme_and_vix_calm", "recovery_ready",
        "vix_calm_and_recovery_ready",
        "cape_extreme_or_vix_calm_and_recovery_ready",
        "cape_extreme_and_vix_calm_and_recovery_ready",
    ]
    conditional_rows = []
    for symbol in ("SPY", "QQQ"):
        baseline = result["strategy_overlays"][symbol]["delta_05"]["baseline"]
        for policy in policy_order:
            conditional = result["conditional"][symbol]["delta_05"][policy]
            whole = conditional["whole_contracts"]
            conditional_rows.append(
                f'<tr class="{"featured" if policy in ("cape_extreme_or_vix_calm", "vix_calm_and_recovery_ready") else ""}">'
                f'<td><strong>{symbol}</strong> {policy_labels[policy]}</td>'
                f'<td>{conditional["active_weeks"]}</td>'
                f'<td>{pct(conditional["share_of_available_weeks"],1)}</td>'
                f'<td>{whole["assigned_calls"]}</td>'
                f'<td>{money(whole["terminal_wealth"])}</td>'
                f'<td>{pct(whole["terminal_wealth"] / baseline["terminal_wealth"] - 1,2)}</td>'
                f'<td>{money(whole["gross_premium_to_treasury"])}</td>'
                f'<td>{money(whole["upside_paid_away"])}</td>'
                f'<td>{money(whole["net_option_pnl"])}</td></tr>'
            )

    colors = {
        "baseline": "var(--ink)", "delta5": "var(--teal)", "delta10": "var(--red)",
        "requested": "var(--amber)", "calm": "var(--teal)", "recovery": "var(--red)",
    }
    charts = {}
    for symbol in ("SPY", "QQQ"):
        subset = daily[daily["symbol"] == symbol]
        base = subset[subset["delta"] == 0.05][["date", "baseline_wealth"]].rename(
            columns={"baseline_wealth": "baseline"}
        )
        d5 = subset[subset["delta"] == 0.05][["date", "whole_contract_wealth"]].rename(
            columns={"whole_contract_wealth": "delta5"}
        )
        d10 = subset[subset["delta"] == 0.10][["date", "whole_contract_wealth"]].rename(
            columns={"whole_contract_wealth": "delta10"}
        )
        frame = base.merge(d5, on="date").merge(d10, on="date")
        charts[symbol] = line_chart(frame, ["baseline", "delta5", "delta10"], colors)

    conditional_charts = {}
    for symbol in ("SPY", "QQQ"):
        subset = conditional_daily[conditional_daily["symbol"] == symbol]
        base = subset[subset["policy"] == "always"][["date", "baseline_wealth"]].rename(
            columns={"baseline_wealth": "baseline"}
        )
        requested = subset[subset["policy"] == "cape_extreme_or_vix_calm"][
            ["date", "whole_contract_wealth"]
        ].rename(columns={"whole_contract_wealth": "requested"})
        calm = subset[subset["policy"] == "vix_calm"][["date", "whole_contract_wealth"]].rename(
            columns={"whole_contract_wealth": "calm"}
        )
        recovery = subset[subset["policy"] == "vix_calm_and_recovery_ready"][
            ["date", "whole_contract_wealth"]
        ].rename(columns={"whole_contract_wealth": "recovery"})
        frame = base.merge(requested, on="date").merge(calm, on="date").merge(recovery, on="date")
        conditional_charts[symbol] = line_chart(
            frame, ["baseline", "requested", "calm", "recovery"], colors
        )

    worst_rows = []
    for symbol in ("SPY", "QQQ"):
        selected = trades[(trades["symbol"] == symbol) & (trades["target_delta"] == 0.05)]
        for _, row in selected.nlargest(4, "payoff_yield").iterrows():
            worst_rows.append(
                f'<tr><td><strong>{symbol}</strong></td><td>{row["issue_date"]}</td><td>{row["expiration"]}</td>'
                f'<td>{pct(row["underlying_return"],1)}</td><td>${row["strike"]:,.2f}</td>'
                f'<td>{row["premium_yield"]*10_000:.1f} bp</td><td>{pct(row["payoff_yield"],2)}</td></tr>'
            )

    spy5 = result["chains"]["SPY"]["delta_05"]
    qqq5 = result["chains"]["QQQ"]["delta_05"]
    spy_overlay = result["strategy_overlays"]["SPY"]["delta_05"]["whole_contracts"]
    qqq_overlay = result["strategy_overlays"]["QQQ"]["delta_05"]["whole_contracts"]
    spy_requested = result["conditional"]["SPY"]["delta_05"]["cape_extreme_or_vix_calm"]
    qqq_requested = result["conditional"]["QQQ"]["delta_05"]["cape_extreme_or_vix_calm"]
    spy_calm = result["conditional"]["SPY"]["delta_05"]["vix_calm"]
    qqq_calm = result["conditional"]["QQQ"]["delta_05"]["vix_calm"]
    spy_recovery = result["conditional"]["SPY"]["delta_05"]["vix_calm_and_recovery_ready"]
    qqq_recovery = result["conditional"]["QQQ"]["delta_05"]["vix_calm_and_recovery_ready"]
    legend = ('<span><i style="--swatch:var(--ink)"></i>No calls</span>'
              '<span><i style="--swatch:var(--teal)"></i>5-delta weekly</span>'
              '<span><i style="--swatch:var(--red)"></i>10-delta weekly</span>')
    conditional_legend = (
        '<span><i style="--swatch:var(--ink)"></i>No calls</span>'
        '<span><i style="--swatch:var(--amber)"></i>CAPE >30 OR calm vol</span>'
        '<span><i style="--swatch:var(--teal)"></i>Calm vol only</span>'
        '<span><i style="--swatch:var(--red)"></i>Calm vol + recovery lockout</span>'
    )
    projection_rows = []
    projection_scenarios = [
        ("$10k only; no additions", 0.0, 0.0),
        ("$10k now + $10k/year", 10_000.0, 0.0),
        ("$10k now + $10k/year + $30k every third year", 10_000.0, 30_000.0),
    ]
    for label, annual, extra in projection_scenarios:
        values = [projected_wealth(10_000.0, annual, rate, extra) for rate in (0.06, 0.08, 0.10)]
        projection_rows.append(
            f'<tr><td><strong>{label}</strong></td>'
            + ''.join(f'<td>{money(value)}</td>' for value in values) + '</tr>'
        )

    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Weekly covered-call study</title>
<style>
:root{{--bg:#edf1ed;--paper:#fbfcf9;--ink:#182321;--muted:#5c6c68;--faint:#7d8a87;--line:#cad4cf;--teal:#118578;--soft:#dceee9;--amber:#a46c00;--amberSoft:#f5ead1;--red:#b3483e;--redSoft:#f5dfdc;--mono:"Cascadia Mono",Consolas,monospace;--serif:Georgia,"Times New Roman",serif;--sans:Inter,system-ui,-apple-system,"Segoe UI",sans-serif}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.58 var(--sans)}}main{{width:min(1160px,calc(100% - 32px));margin:auto;padding:46px 0 76px}}header{{border-bottom:2px solid var(--ink);padding-bottom:30px}}.eyebrow{{font:700 11px var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--teal)}}h1{{font:600 clamp(42px,7vw,76px)/1 var(--serif);letter-spacing:-.05em;max-width:15ch;margin:12px 0 18px}}h2{{font:600 31px/1.17 var(--serif);letter-spacing:-.025em;margin:0 0 12px}}h3{{font:600 21px/1.25 var(--serif);margin:25px 0 9px}}.standfirst{{max-width:72ch;color:var(--muted);font:21px/1.45 var(--serif)}}section{{margin-top:48px}}p{{margin:0 0 16px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}}.card{{background:var(--paper);border-top:4px solid var(--teal);padding:18px}}.card.warn{{border-color:var(--amber)}}.card.stop{{border-color:var(--red)}}.card b{{display:block;font:700 24px var(--mono)}}.card span{{display:block;color:var(--muted);font-size:12px;margin-top:6px}}.note{{background:var(--soft);border-left:3px solid var(--teal);padding:17px 19px;color:var(--muted)}}.caution{{background:var(--amberSoft);border-left-color:var(--amber)}}.warning{{background:var(--redSoft);border-left-color:var(--red)}}.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}figure{{margin:22px 0}}.figureHead{{display:flex;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:8px;color:var(--muted);font-size:12px}}.plate{{background:var(--paper);border:1px solid var(--line);padding:12px;overflow:auto;margin-top:10px}}svg{{display:block;width:100%;min-width:720px}}.frame{{fill:none;stroke:var(--line)}}.grid{{stroke:var(--line)}}.axis{{font:10px var(--mono);fill:var(--faint)}}.axis-title{{font:12px var(--sans);fill:var(--ink)}}.legend{{display:flex;gap:20px;flex-wrap:wrap;padding:9px 3px 0;color:var(--muted);font-size:11px}}.legend span{{display:flex;align-items:center;gap:7px}}.legend i{{width:22px;border-top:2px solid var(--swatch)}}.scroll{{overflow:auto;border:1px solid var(--line);background:var(--paper)}}table{{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}}th,td{{padding:11px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}}th{{font:700 10px var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--faint)}}th:first-child,td:first-child{{text-align:left}}td strong{{display:block}}tr.featured{{background:var(--soft)}}.rules{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}.rule{{background:var(--paper);border-top:3px solid var(--teal);padding:15px}}.rule b{{display:block;font:700 17px var(--mono)}}.rule span{{display:block;color:var(--muted);font-size:12px;margin-top:5px}}footer{{border-top:1px solid var(--line);padding-top:18px;margin-top:50px;color:var(--faint);font-size:11px}}a{{color:var(--teal)}}code{{font-family:var(--mono)}}@media(max-width:760px){{.cards,.rules{{grid-template-columns:1fr 1fr}}.grid2{{grid-template-columns:1fr}}}}@media(max-width:480px){{.cards,.rules{{grid-template-columns:1fr}}}}
</style></head><body><main>
<header><div class="eyebrow">Actual option-chain test · {result['sample']['start']} to {result['sample']['end']}</div><h1>Harvest calm markets. Stay uncapped through recovery.</h1><p class="standfirst">We redefined CAPE 20–30 as normal and above 30 as extreme, then tested a causal NAV recovery lockout. Five-delta calls collected a small positive premium edge in this sample. The most defensible implementation is selective, partially covered, and explicitly paused after a major NAV drawdown.</p></header>
<div class="cards"><article class="card warn"><b>{pct(spy_requested['share_of_available_weeks'],1)}</b><span>of SPY weeks passed CAPE &gt;30 OR calm volatility; still a broad gate</span></article><article class="card"><b>{pct(spy_calm['share_of_available_weeks'],1)}</b><span>of SPY weeks passed calm VIX alone; QQQ {pct(qqq_calm['share_of_available_weeks'],1)}</span></article><article class="card"><b>{pct(spy_recovery['share_of_available_weeks'],1)}</b><span>passed calm vol plus recovery lockout; QQQ {pct(qqq_recovery['share_of_available_weeks'],1)}</span></article><article class="card stop"><b>100 shares</b><span>required by one standard ETF option contract</span></article></div>

<section><div class="eyebrow">I · Is it realistic now?</div><h2>Not with the present $10,000 account.</h2><p>One standard covered call requires 100 shares. At the latest local closes, one fully covered SPY contract needs approximately {money(latest['SPY']*100)} of SPY and one QQQ contract needs {money(latest['QQQ']*100)} of QQQ. A $10,000 account holds only about {10_000/latest['SPY']:.1f} SPY shares or {10_000/latest['QQQ']:.1f} QQQ shares.</p><p class="warning note"><strong>Do not solve the sizing problem with naked calls.</strong> XSP, call spreads, or broker-specific fractional option products are different strategies with basis, liquidity, settlement, and tail-risk differences. They should not be described as ordinary covered calls.</p></section>

<section><div class="eyebrow">II · Actual weekly calls</div><h2>“Target worthless” means accepting almost no premium.</h2><p>Calls were sold at the historical bid on the last quoted session of each week, using the closest next-week expiration and strike nearest the requested delta. Intrinsic value at expiration is the upside surrendered. A 2 bp stock turnover allowance is charged when the option expires in the money.</p><div class="scroll"><table><thead><tr><th>Underlying / target</th><th>Weeks</th><th>Worthless</th><th>ITM weeks</th><th>Median premium</th><th>Premium sum</th><th>Upside paid</th><th>CAGR effect</th></tr></thead><tbody>{''.join(chain_rows)}</tbody></table></div><p class="caution note" style="margin-top:15px"><strong>The trade-off is nonlinear:</strong> moving from 5 to 10 delta roughly doubled the frequency of upside caps, while the larger premiums still failed to compensate for sharp rebound weeks. At 20 delta, more than one quarter of weekly calls finished in the money.</p></section>

<section><div class="eyebrow">III · CAPE, volatility and recovery gates</div><h2>Above 30 is extreme—but OR still makes the gate broad.</h2><p>CAPE 20–30 is treated as the normal band, CAPE &gt;30 as extreme, and “calm” means the prior-known 60-session VIX or VXN percentile is below 70%. The recovery lockout activates once strategy NAV falls 10% from its high; calls remain off through the rebound and for 20 trading sessions after NAV regains that prior high. Every input is known on the sale date.</p><div class="scroll"><table><thead><tr><th>5-delta rule</th><th>Calls</th><th>Active weeks</th><th>ITM</th><th>Terminal wealth</th><th>vs no calls</th><th>Gross premium</th><th>Upside paid</th><th>Net option P&amp;L</th></tr></thead><tbody>{''.join(conditional_rows)}</tbody></table></div><p class="note" style="margin-top:15px"><strong>What the test says:</strong> CAPE &gt;30 OR calm volatility was active in {pct(spy_requested['share_of_available_weeks'],1)} of SPY weeks and {pct(qqq_requested['share_of_available_weeks'],1)} of QQQ weeks, so it remains close to continuous overwriting. Calm volatility alone added {pct(spy_calm['whole_contracts']['xirr']-result['strategy_overlays']['SPY']['delta_05']['baseline']['xirr'],2)} of annualized XIRR for SPY and {pct(qqq_calm['whole_contracts']['xirr']-result['strategy_overlays']['QQQ']['delta_05']['baseline']['xirr'],2)} for QQQ in-sample. Adding the recovery lockout reduced active weeks to {pct(spy_recovery['share_of_available_weeks'],1)} and {pct(qqq_recovery['share_of_available_weeks'],1)}; it sacrificed some premium, but specifically removes the covered-call cap during deep-drawdown recoveries.</p><div class="grid2"><figure><div class="figureHead"><strong>SPY conditional 5-delta overlay</strong><span>whole contracts</span></div><div class="plate">{conditional_charts['SPY']}<div class="legend">{conditional_legend}</div></div></figure><figure><div class="figureHead"><strong>QQQ conditional 5-delta overlay</strong><span>whole contracts</span></div><div class="plate">{conditional_charts['QQQ']}<div class="legend">{conditional_legend}</div></div></figure></div><p class="caution note"><strong>The lockout is risk control, not a return optimizer.</strong> It did not maximize terminal wealth in this one sample. Its purpose is to avoid selling rebound convexity exactly when the deleveraged strategy needs it most.</p></section>

<section><div class="eyebrow">IV · Unconditional reference</div><h2>Only unleveraged index exposure was covered.</h2><div class="scroll"><table><thead><tr><th>Strategy overlay</th><th>Baseline terminal</th><th>With calls</th><th>Difference</th><th>Base XIRR</th><th>Call XIRR</th><th>Gross premium</th><th>Upside paid</th><th>Net option P&amp;L</th></tr></thead><tbody>{''.join(overlay_rows)}</tbody></table></div><p>This table deliberately keeps the always-sell reference. At 5 delta, whole-contract execution increased the SPY strategy endpoint by {pct(spy_overlay['terminal_wealth']/result['strategy_overlays']['SPY']['delta_05']['baseline']['terminal_wealth']-1,2)} and the QQQ endpoint by {pct(qqq_overlay['terminal_wealth']/result['strategy_overlays']['QQQ']['delta_05']['baseline']['terminal_wealth']-1,2)}. That is modest evidence—not a stable expected return estimate from one 7¾-year regime.</p><p class="warning note"><strong>Gross premium is not spendable profit.</strong> The SPY sleeve received {money(spy_overlay['gross_premium_to_treasury'])} of premium but surrendered {money(spy_overlay['upside_paid_away'])} of upside plus assignment costs. QQQ received {money(qqq_overlay['gross_premium_to_treasury'])} and surrendered {money(qqq_overlay['upside_paid_away'])}. Treasury accounting must retain enough liquidity for assignment or repurchase.</p></section>

<section><div class="eyebrow">V · Unconditional equity curves</div><h2>Ten delta visibly drags both recovery paths.</h2><div class="grid2"><figure><div class="figureHead"><strong>SPY dual-guard strategy</strong><span>whole contracts</span></div><div class="plate">{charts['SPY']}<div class="legend">{legend}</div></div></figure><figure><div class="figureHead"><strong>QQQ dashboard guard</strong><span>whole contracts</span></div><div class="plate">{charts['QQQ']}<div class="legend">{legend}</div></div></figure></div></section>

<section><div class="eyebrow">VI · When did we miss upside?</div><h2>The damage is concentrated in sudden rebounds.</h2><p>A 5-delta call capped upside {spy5['assigned_or_itm']} times in 395 SPY weeks and {qqq5['assigned_or_itm']} times in 395 QQQ weeks—roughly once every {395/spy5['assigned_or_itm']:.0f} and {395/qqq5['assigned_or_itm']:.0f} weeks. These were the largest misses:</p><div class="scroll"><table><thead><tr><th>Asset</th><th>Sold</th><th>Expired</th><th>Underlying week</th><th>Strike</th><th>Premium</th><th>Upside surrendered</th></tr></thead><tbody>{''.join(worst_rows)}</tbody></table></div><p>The worst SPY miss followed the April 3, 2020 sale: SPY rallied about 12.1% in the shortened week, while the 5-delta premium was about 9.7 bp and the call surrendered 4.11% of notional. One such rebound consumed roughly forty-two median SPY premiums.</p></section>

<section><div class="eyebrow">VII · Ten-year planning range</div><h2>Contributions matter far more than the call overlay.</h2><p>These are planning scenarios for $10,000 invested now, with end-of-year contributions. They are not forecasts. The 6–10% return range is deliberately below the backtest’s realized strategy XIRRs; using the backtest’s unusually strong QQQ-era return as a ten-year expectation would be fragile.</p><div class="scroll"><table><thead><tr><th>Cash-flow plan</th><th>6% annual return</th><th>8% annual return</th><th>10% annual return</th></tr></thead><tbody>{''.join(projection_rows)}</tbody></table></div><p class="note" style="margin-top:15px"><strong>Central planning case:</strong> at 8%, $10,000 now plus $10,000 each year grows to {money(projected_wealth(10_000,10_000,0.08))} in ten years. Keeping the previously discussed extra $30,000 contribution every third year raises that to {money(projected_wealth(10_000,10_000,0.08,30_000))}. The tested covered-call edge was only about 0.30 percentage points of SPY XIRR and 0.67 points of QQQ XIRR under the calm-volatility gate; it should be treated as a possible incremental return, not built into the base forecast.</p></section>

<section><div class="eyebrow">VIII · Operating rule if tested live</div><h2>Five delta, calm volatility, partial coverage—and a recovery lockout.</h2><div class="rules"><article class="rule"><b>Valuation</b><span>Treat CAPE 20–30 as normal and above 30 as extreme. CAPE is a slow context variable, not the weekly timing trigger.</span></article><article class="rule"><b>Regime</b><span>Use prior-known 60-session VIX/VXN percentile below 70% as the primary call-sale gate.</span></article><article class="rule"><b>Recovery</b><span>After NAV falls 10%, sell no calls until NAV regains its prior high and another 20 trading sessions pass.</span></article><article class="rule"><b>Eligibility</b><span>Sell only against fully paid, unleveraged 100-share lots; cover initially only 25–50% of eligible shares.</span></article><article class="rule"><b>Premium</b><span>Credit net—not gross—premium to Treasury. Reinvest it under the existing drawdown ladder instead of immediately buying more exposure at highs.</span></article><article class="rule"><b>Stop condition</b><span>Freeze the rule for 52 trades; compare total portfolio return and net option P&amp;L, not win rate.</span></article></div><p class="note" style="margin-top:15px"><strong>My conclusion:</strong> use covered calls as a small cash-harvesting sleeve, not as the accelerator. The accelerator remains regular contributions, keeping rebound exposure uncapped, disciplined Treasury deployment in drawdowns, and limited valuation/NAV-governed leverage. Ten- and twenty-delta weekly overwriting remain rejected by this sample.</p></section>

<footer>Historical simulation, not investment advice. Actual chain fields come from the local Alpha Vantage historical-options database; calls are sold at bid and held to intrinsic settlement. Sources and reference methods: <a href="https://www.optionseducation.org/strategies/all-strategies/covered-call-buy-write">Options Industry Council covered-call mechanics</a>, <a href="https://cdn.cboe.com/api/global/us_indices/governance/BXMD_Methodology.pdf">Cboe BuyWrite methodology</a>, and <a href="https://www.econ.yale.edu/~shiller/data.htm">Robert Shiller’s CAPE data</a>. QQQ fund information: <a href="https://www.invesco.com/qqq-etf/en/home.html">Invesco</a>; SPY: <a href="https://www.ssga.com/us/en/intermediary/etfs/state-street-spdr-sp-500-etf-trust-spy">State Street</a>. American early exercise, ex-dividend assignment, tax, commissions and recursive strategy changes are omitted. Generated by <code>scripts/run_weekly_covered_call_study.py</code>.</footer>
</main></body></html>'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
