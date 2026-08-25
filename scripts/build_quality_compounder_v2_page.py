"""Build the self-contained revised Quality Compounder Harvest report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd

from build_contribution_quality_page import line_chart, money, pct


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("out/strategy/quality_compounder_v2"))
    parser.add_argument("--output", type=Path,
                        default=Path("docs/quality-compounder-v2.html"))
    parser.add_argument("--cape-input", type=Path,
                        default=Path("out/strategy/cape_leverage"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = json.loads((args.input / "results.json").read_text(encoding="utf-8"))
    cape_result = json.loads(
        (args.cape_input / "results.json").read_text(encoding="utf-8")
    )
    daily = pd.read_csv(args.input / "daily.csv", parse_dates=["date"])
    variants = result["variants"]
    shown = ["spy_1x", "immediate_20", "immediate_30", "treasury_first_20"]
    labels = {
        "spy_1x": "SPY 1x", "immediate_20": "20% Treasury · immediate",
        "immediate_30": "30% Treasury · immediate",
        "treasury_first_20": "20% Treasury · deposits parked",
    }
    colors = {
        "spy_1x_wealth": "var(--ink)", "immediate_20_wealth": "var(--teal)",
        "immediate_30_wealth": "var(--blue)",
        "treasury_first_20_wealth": "var(--brick)",
        "spy_1x_performance": "var(--ink)",
        "immediate_20_performance": "var(--teal)",
        "immediate_30_performance": "var(--blue)",
        "treasury_first_20_performance": "var(--brick)",
    }
    growth_cols = [f"{name}_wealth" for name in shown]
    performance_cols = [f"{name}_performance" for name in shown]
    growth = line_chart(daily, growth_cols, colors)
    drawdown = line_chart(daily, performance_cols, colors, performance=True)
    legend = "".join(
        f'<span><i style="--swatch:{colors[f"{name}_wealth"]}"></i>{html.escape(labels[name])}</span>'
        for name in shown
    )

    order = ["immediate_10", "immediate_20", "immediate_30",
             "treasury_first_10", "treasury_first_20", "treasury_first_30", "spy_1x"]
    table_labels = {
        "immediate_10": "10% reserve · immediate deposits",
        "immediate_20": "20% reserve · immediate deposits",
        "immediate_30": "30% reserve · immediate deposits",
        "treasury_first_10": "10% reserve · deposits parked",
        "treasury_first_20": "20% reserve · deposits parked",
        "treasury_first_30": "30% reserve · deposits parked",
        "spy_1x": "SPY 1x · matched deposits",
    }
    comparison_rows = "".join(
        f'<tr class="{"featured" if name == "immediate_20" else ""}"><td><strong>{html.escape(table_labels[name])}</strong></td>'
        f'<td>{money(variants[name]["terminal_wealth"])}</td>'
        f'<td>{pct(variants[name]["xirr"])}</td>'
        f'<td>{pct(variants[name]["time_weighted_cagr"])}</td>'
        f'<td>{pct(variants[name]["max_flow_adjusted_drawdown"])}</td>'
        f'<td>{pct(variants[name]["annual_volatility"])}</td>'
        f'<td>{variants[name]["longest_underwater_sessions"] / 252:.1f}y</td></tr>'
        for name in order
    )

    primary = variants["immediate_20"]
    parked = variants["treasury_first_20"]
    thirty = variants["immediate_30"]
    spy = variants["spy_1x"]
    roll = result["rolling_20y_treasury_first_20"]
    quality_events = [
        event for event in result["events"]["immediate_20"]
        if event["kind"] in {"deploy_mega_seven", "mega_seven_skipped"}
    ]
    quality_rows = "".join(
        f'<tr><td>{event["date"]}</td><td>{html.escape(event["kind"].replace("_", " "))}</td>'
        f'<td>{money(event["amount"])}</td><td>{html.escape(event["detail"])}</td></tr>'
        for event in quality_events
    )
    holdings = result["holdings"]["immediate_20"]
    holdings_rows = "".join(
        f'<tr><td><strong>{html.escape(row["ticker"])}</strong><small>{", ".join(row["entry_dates"])}</small></td>'
        f'<td>{money(row["market_value"])}</td><td>{pct(row["portfolio_weight"])}</td>'
        f'<td>{row["lots"]}</td><td>{row["harvests"]}</td></tr>'
        for row in holdings
    )
    event_summary = primary["events"]
    harvest_rows = "".join(
        f'<tr><td><strong>{html.escape(label)}</strong></td><td>{event_summary[key]["count"]}</td>'
        f'<td>{money(event_summary[key]["amount"])}</td></tr>'
        for key, label in [
            ("quarterly_spy_sweep", "Quarterly 20%-gate SPY sweeps"),
            ("annual_spy_sweep", "Annual 10%-gate SPY sweeps"),
            ("quality_harvest", "Quality 5% harvests"),
            ("quality_five_year_exit", "Five-year relative-CAGR exits"),
        ]
    )
    cape_daily = pd.read_csv(args.cape_input / "daily.csv", parse_dates=["date"])
    cape_names = ["spy_1x", "no_cape_cap", "cape_25_35",
                  "cape_percentile_80_95", "cape_15_20"]
    cape_labels = {
        "spy_1x": "SPY 1x", "no_cape_cap": "No CAPE cap",
        "cape_25_35": "CAPE 25/35 cap", "cape_percentile_80_95": "CAPE 80/95 percentile",
        "cape_15_20": "CAPE 15/20 cap",
    }
    cape_colors = {
        "spy_1x_wealth": "var(--ink)", "no_cape_cap_wealth": "var(--teal)",
        "cape_25_35_wealth": "var(--blue)",
        "cape_percentile_80_95_wealth": "var(--purple)",
        "cape_15_20_wealth": "var(--brick)",
    }
    cape_growth = line_chart(
        cape_daily, [f"{name}_wealth" for name in cape_names], cape_colors
    )
    cape_legend = "".join(
        f'<span><i style="--swatch:{cape_colors[f"{name}_wealth"]}"></i>{html.escape(cape_labels[name])}</span>'
        for name in cape_names
    )
    cape_order = ["no_cape_cap", "cape_30_40", "cape_25_35", "cape_20_25",
                  "cape_percentile_80_95", "cape_15_20", "spy_1x"]
    cape_table_labels = {
        "no_cape_cap": "No CAPE cap", "cape_30_40": "2x at 30; 1x at 40",
        "cape_25_35": "2x at 25; 1x at 35", "cape_20_25": "2x at 20; 1x at 25",
        "cape_percentile_80_95": "2x at 80th; 1x at 95th percentile",
        "cape_15_20": "2x at 15; 1x at 20", "spy_1x": "SPY 1x",
    }
    cape_rows = "".join(
        f'<tr><td><strong>{html.escape(cape_table_labels[name])}</strong></td>'
        f'<td>{money(cape_result["strategy_variants"][name]["terminal_wealth"])}</td>'
        f'<td>{pct(cape_result["strategy_variants"][name]["xirr"])}</td>'
        f'<td>{pct(cape_result["strategy_variants"][name]["max_flow_adjusted_drawdown"])}</td>'
        f'<td>{pct(cape_result["strategy_variants"][name]["annual_volatility"])}</td>'
        f'<td>{cape_result["strategy_variants"][name].get("mean_applied_leverage", 1.0):.2f}x</td></tr>'
        for name in cape_order
    )
    mega_rows = "".join(
        f'<tr><td>{row["entry"]}</td><td>{html.escape(row["horizon"].replace("_", " "))}</td>'
        f'<td>{pct(row["basket_return"])}</td><td>{pct(row["spy_return"])}</td>'
        f'<td>{pct(row["excess_return"])}</td></tr>'
        for row in cape_result["mega_seven_audit"] if row["horizon"] != "risk_audit"
    )
    risk_audits = [
        row for row in cape_result["mega_seven_audit"] if row["horizon"] == "risk_audit"
    ]
    decomposition_rows = "".join(
        f'<tr><td><strong>{html.escape(label)}</strong></td><td>{money(values["terminal_wealth"])}</td>'
        f'<td>{pct(values["xirr"])}</td><td>{pct(values["max_flow_adjusted_drawdown"])}</td>'
        f'<td>{pct(values["annual_volatility"])}</td></tr>'
        for name, label in [
            ("all_equity_constant_3x", "All capital, constant 3x"),
            ("eighty_twenty_constant_3x", "80/20, constant 3x"),
            ("all_equity_cape_25_35", "All capital, CAPE 25/35"),
        ] for values in [cape_result["decomposition"][name]]
    )
    html_text = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Quality Compounder Harvest — revised rules</title>
<style>
:root{{--bg:#eef2f0;--paper:#fbfcfb;--ink:#13201d;--muted:#53635f;--faint:#788783;--line:#ccd7d3;--teal:#087f72;--blue:#356da6;--purple:#7651a8;--brick:#b84138;--soft:#dcefea;--warn:#f4dedb;--mono:"Cascadia Mono",Consolas,monospace;--serif:Georgia,"Times New Roman",serif;--sans:system-ui,-apple-system,"Segoe UI",sans-serif}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:17px/1.62 var(--serif)}}main{{width:min(1100px,calc(100% - 34px));margin:auto;padding:54px 0 86px}}header{{border-bottom:2px solid var(--ink);padding-bottom:34px}}.eyebrow{{font:700 11px var(--mono);letter-spacing:.13em;text-transform:uppercase;color:var(--teal)}}h1{{font:650 clamp(40px,7vw,72px)/1.02 var(--serif);letter-spacing:-.045em;max-width:14ch;margin:13px 0 20px}}h1 em{{color:var(--teal)}}.standfirst{{font-size:22px;line-height:1.43;color:var(--muted);max-width:62ch}}section{{padding-top:50px;max-width:880px}}section.wide{{max-width:none}}h2{{font:650 31px/1.18 var(--serif);letter-spacing:-.025em;margin:0 0 14px}}p{{margin:0 0 18px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);margin-top:28px}}.card{{background:var(--paper);padding:17px;border-right:1px solid var(--line)}}.card:last-child{{border:0}}.card b{{display:block;font:700 22px var(--mono)}}.card span{{display:block;font:12px/1.4 var(--sans);color:var(--faint);margin-top:7px}}.rule{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:23px 0}}.step{{background:var(--paper);border-top:3px solid var(--teal);padding:15px}}.step:last-child{{border-color:var(--brick)}}.step b{{display:block;font:700 18px var(--mono)}}.step span{{font:12px/1.45 var(--sans);color:var(--faint)}}.note{{background:var(--soft);border-left:3px solid var(--teal);padding:18px 20px;color:var(--muted)}}.warning{{background:var(--warn);border-left-color:var(--brick)}}.scroll{{overflow:auto;border:1px solid var(--line);background:var(--paper);margin-top:22px}}table{{border-collapse:collapse;width:100%;font:13px/1.4 var(--sans);font-variant-numeric:tabular-nums}}th,td{{padding:12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}}th{{font:700 10px var(--mono);text-transform:uppercase;letter-spacing:.06em;color:var(--faint)}}th:first-child,td:first-child{{text-align:left}}td:first-child{{white-space:normal;min-width:180px}}td strong,td small{{display:block}}td small{{color:var(--faint);margin-top:3px}}tr.featured{{background:var(--soft)}}figure{{margin:30px 0}}.figure-head{{display:flex;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:9px;font:13px var(--sans);color:var(--muted)}}.plate{{background:var(--paper);border:1px solid var(--line);padding:14px;margin-top:13px;overflow:auto}}svg{{display:block;width:100%;min-width:720px}}.frame{{fill:none;stroke:var(--line)}}.grid{{stroke:var(--line)}}.axis{{font:10px var(--mono);fill:var(--faint)}}.axis-title{{font:12px var(--sans);fill:var(--ink)}}.legend{{display:flex;gap:18px;flex-wrap:wrap;padding:8px 4px 0;font:11px var(--sans);color:var(--muted)}}.legend span{{display:flex;gap:7px;align-items:center}}.legend i{{width:22px;border-top:2px solid var(--swatch)}}figcaption,footer{{font:12px/1.55 var(--sans);color:var(--faint)}}footer{{border-top:1px solid var(--line);margin-top:55px;padding-top:20px;max-width:920px}}a{{color:var(--teal)}}code{{font-family:var(--mono)}}@media(max-width:760px){{.cards,.rule{{grid-template-columns:1fr 1fr}}.card:nth-child(2){{border-right:0}}}}
</style></head><body><main>
<header><div class="eyebrow">Monthly $10,000 · {result["sample"]["start"]} to {result["sample"]["end"]}</div>
<h1>Save the harvest. <em>Do not park the future.</em></h1>
<p class="standfirst">The revised rules improve the quality-selection mechanics, but the contribution rule dominates the result: waiting for a leveraged NAV recovery leaves too much new money in Treasury for too long.</p>
<div class="cards"><div class="card"><b>{money(primary["terminal_wealth"])}</b><span>20% reserve, immediate deposits; SPY {money(spy["terminal_wealth"])}</span></div>
<div class="card"><b>{pct(primary["xirr"])}</b><span>money-weighted return; SPY {pct(spy["xirr"])}</span></div>
<div class="card"><b>{pct(primary["max_flow_adjusted_drawdown"])}</b><span>flow-adjusted max drawdown; SPY {pct(spy["max_flow_adjusted_drawdown"])}</span></div>
<div class="card"><b>{money(primary["total_contributed"])}</b><span>initial $10k plus 403 monthly deposits</span></div></div></header>

<section><div class="eyebrow">I · Revised rules</div><h2>Two harvest gates, one leverage clock.</h2>
<p>Core SPY leverage is 3x above a 15% NAV drawdown, 2x from −15% to −30%, and 1x below −30%. Recovery hysteresis permits 1x→2x only above −10%; 3x returns only at a new flow-adjusted NAV high.</p>
<div class="rule"><div class="step"><b>−10%</b><span>10% of episode-start Treasury to unlevered SPY.</span></div><div class="step"><b>−20%</b><span>Another 20% of episode-start Treasury to unlevered SPY.</span></div><div class="step"><b>−30%</b><span>Another 30% of episode-start Treasury to unlevered SPY.</span></div><div class="step"><b>−40%</b><span>Final 40% to exactly seven date-ranked mega caps.</span></div></div>
<p>The mega-cap sleeve is weighted by the market-cap estimate at the signal close. A profitable position sells 5% of current shares when SPY first regains the episode high or after a stock quarter of at least +20%. After five years it exits immediately whenever its trailing five-year CAGR falls below SPY.</p>
<p>Quarterly SPY harvesting now requires the financed sleeve itself to return at least +20%; only then is 10% of dollar P&amp;L swept. Separately, when SPY's calendar-year return is at least +10%, 1% of the core sleeve moves to Treasury.</p>
<p class="note">Signals use the prior close and execute at the next close. Financing uses prior-known DGS3MO +1%; Treasury earns DGS3MO; stock trades cost 5 bp. Dividends remain reinvested in this backtest.</p></section>

<section class="wide"><div class="eyebrow">II · Full-history comparison</div><h2>Immediate monthly investment nearly tied SPY; parking did not.</h2>
<figure><div class="figure-head"><strong>Account value with matched monthly deposits</strong><span>log scale</span></div><div class="plate">{growth}<div class="legend">{legend}</div></div></figure>
<figure><div class="figure-head"><strong>Flow-adjusted drawdown</strong><span>deposits removed</span></div><div class="plate">{drawdown}<div class="legend">{legend}</div></div></figure>
<div class="scroll"><table><thead><tr><th>Path</th><th>Terminal</th><th>XIRR</th><th>TWR CAGR</th><th>Max DD</th><th>Volatility</th><th>Longest underwater</th></tr></thead><tbody>{comparison_rows}</tbody></table></div>
<p class="warning note"><strong>Risk-adjusted result:</strong> the 20% immediate path ended only {pct(primary["terminal_wealth"] / spy["terminal_wealth"] - 1)} above SPY and its XIRR advantage was only {pct(primary["xirr"] - spy["xirr"], 2)}. It carried {pct(primary["annual_volatility"])} volatility versus SPY's {pct(spy["annual_volatility"])} and stayed underwater roughly {primary["longest_underwater_sessions"] / 252:.1f} years at worst.</p></section>

<section><div class="eyebrow">III · Is 20% Treasury too much?</div><h2>No. Ten percent was the weak setting.</h2>
<p>Holding contribution treatment constant, the 30% immediate path ended at {money(thirty["terminal_wealth"])} versus {money(primary["terminal_wealth"])} for 20%, with a {pct(thirty["max_flow_adjusted_drawdown"])} versus {pct(primary["max_flow_adjusted_drawdown"])} drawdown. Those are small in-sample differences. The 10% path ended at {money(variants["immediate_10"]["terminal_wealth"])} and suffered a {pct(variants["immediate_10"]["max_flow_adjusted_drawdown"])} loss.</p>
<p>The meaningful design range here is 20–30%, not because cash itself creates return, but because too little reserve leaves the leveraged sleeve with less dry powder and a worse path. This single history does not identify a precise optimum.</p></section>

<section><div class="eyebrow">IV · Monthly deposits</div><h2>Do not wait for the old NAV high.</h2>
<p>Parking every monthly $10,000 contribution in Treasury until NAV recovery reduced full-period terminal wealth to {money(parked["terminal_wealth"])}. Across {roll["cohorts"]} overlapping 20-year histories, median terminal wealth was {money(roll["strategy_terminal"]["median"])} versus {money(roll["spy_terminal"]["median"])} for SPY, and the parked strategy won only {pct(roll["beats_spy_share"], 0)} of cohorts.</p>
<p class="note"><strong>Better implementation:</strong> allocate normal monthly contributions immediately to the standing 80/20 policy. During an active drawdown, temporarily direct only the next scheduled rung amount—or at most a fixed number of months of contributions—to Treasury. A recovery trigger tied to an old leveraged NAV high is too slow for recurring savings.</p></section>

<section class="wide"><div class="eyebrow">V · Historical mega-cap trades</div><h2>The ranking is date-aware, with a hard data boundary.</h2>
<p>Quarterly share counts are treated as known 90 days after fiscal period end and multiplied by raw close; adjusted prices are used for investment returns. The 2001 trigger was skipped because the local share-count history begins in 2005.</p>
<div class="scroll"><table><thead><tr><th>Date</th><th>Action</th><th>Amount</th><th>Market-cap weights</th></tr></thead><tbody>{quality_rows}</tbody></table></div>
<div class="scroll"><table><thead><tr><th>Open holding / entry</th><th>Market value</th><th>Portfolio weight</th><th>Lots</th><th>Harvests</th></tr></thead><tbody>{holdings_rows}</tbody></table></div>
<p class="warning note"><strong>Selection warning:</strong> this is a fixed union of surviving historical mega-cap leaders, not a point-in-time database containing every listed and delisted company. The date-specific ranking is cleaner than retrospectively buying today's Magnificent Seven, but it is still not survivor-free.</p></section>

<section><div class="eyebrow">VI · Cash harvested</div><h2>The gates were selective.</h2>
<div class="scroll"><table><thead><tr><th>Mechanism</th><th>Events</th><th>Moved to Treasury</th></tr></thead><tbody>{harvest_rows}</tbody></table></div></section>

<section class="wide"><div class="eyebrow">VII · Did the selected seven beat SPY?</div><h2>Yes twice—but with more downside and only two observations.</h2>
<div class="scroll"><table><thead><tr><th>Entry</th><th>Horizon</th><th>Selected seven</th><th>SPY</th><th>Relative excess</th></tr></thead><tbody>{mega_rows}</tbody></table></div>
<p>The March 2020 basket beat SPY by 21.3% cumulatively through August 2026; the October 2022 basket beat by only 1.1%. The corresponding maximum drawdowns were {pct(risk_audits[0]["basket_max_drawdown"])} versus {pct(risk_audits[0]["spy_max_drawdown"])} and {pct(risk_audits[1]["basket_max_drawdown"])} versus {pct(risk_audits[1]["spy_max_drawdown"])}. This is concentrated recovery beta, not proof of a persistent stock-selection edge.</p></section>

<section class="wide"><div class="eyebrow">VIII · CAPE as a leverage ceiling</div><h2>Valuation helped returns; it did not solve drawdown.</h2>
<p>CAPE is lagged one month and only caps the leverage allowed by the NAV ladder. The study stops in September 2024, the end of the local Shiller series.</p>
<figure><div class="figure-head"><strong>Matched monthly deposits</strong><span>CAPE sample through September 2024</span></div><div class="plate">{cape_growth}<div class="legend">{cape_legend}</div></div></figure>
<div class="scroll"><table><thead><tr><th>CAPE policy</th><th>Terminal</th><th>XIRR</th><th>Max DD</th><th>Volatility</th><th>Mean leverage</th></tr></thead><tbody>{cape_rows}</tbody></table></div>
<p>The 25/35 rule improved terminal wealth and Sharpe versus no cap but suffered a {pct(cape_result["strategy_variants"]["cape_25_35"]["max_flow_adjusted_drawdown"])} drawdown. The historical-percentile rule had the best XIRR at {pct(cape_result["strategy_variants"]["cape_percentile_80_95"]["xirr"])}, but still lost {pct(cape_result["strategy_variants"]["cape_percentile_80_95"]["max_flow_adjusted_drawdown"])} peak-to-trough. Only the aggressive 15/20 rule beat SPY's drawdown—and it averaged {cape_result["strategy_variants"]["cape_15_20"]["mean_applied_leverage"]:.2f}x, effectively becoming an unlevered strategy, while ending below SPY.</p>
<p class="warning note"><strong>Decision:</strong> CAPE can be a slow ceiling—never a green light to increase leverage during a fall. It does not replace the NAV drawdown ladder, and this small threshold grid is in-sample.</p></section>

<section><div class="eyebrow">IX · Why “3x” did not produce 3x wealth</div><h2>The strategy was rarely 3x—and constant 3x was nearly wiped out.</h2>
<p>The uncapped strategy averaged only {cape_result["strategy_variants"]["no_cape_cap"]["mean_applied_leverage"]:.2f}x: it spent {pct(cape_result["strategy_variants"]["no_cape_cap"]["time_at_1x"], 0)} of sessions at 1x and only {pct(cape_result["strategy_variants"]["no_cape_cap"]["time_at_3x"], 0)} at 3x. Treasury, financing, deleveraging, quality holdings and profit sweeps all separate it from a constant 3x index.</p>
<div class="scroll"><table><thead><tr><th>Mechanical exposure</th><th>Terminal</th><th>XIRR</th><th>Max DD</th><th>Volatility</th></tr></thead><tbody>{decomposition_rows}</tbody></table></div>
<p>Constant 3x did create more terminal wealth from recurring contributions, but its {pct(cape_result["decomposition"]["all_equity_constant_3x"]["max_flow_adjusted_drawdown"])} drawdown is economically close to ruin. Matching SPY-like drawdown by valuation alone required almost permanent 1x exposure. The useful objective is therefore not “match 3x”; it is “earn a modest excess return while keeping survival probability acceptable.”</p></section>

<section><div class="eyebrow">X · Personal dividends</div><h2>Separate spending cash from rescue cash.</h2>
<p>The cleanest operational structure is a third sleeve: let cash dividends from unlevered SPY and quality stocks settle into an income account, then transfer a fixed quarterly amount to yourself. Do not label those withdrawals “Treasury”—the Treasury sleeve has a risk-management job and should remain available for drawdown rungs.</p>
<p>During accumulation, reinvest dividends unless current income is genuinely needed. During withdrawal, turn off dividend reinvestment only for the unlevered and quality holdings, withdraw no more than received cash after taxes, and reinvest any surplus. Review the policy using total return, because a distribution transfers value out of the fund rather than creating free return. See the <a href="https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins/fund-distributions-investor-bulletin">SEC's fund-distribution bulletin</a>. U.S. taxpayers should also note that reinvested dividends generally remain reportable income; see <a href="https://www.irs.gov/faqs/capital-gains-losses-and-sale-of-home/stocks-options-splits-traders/stocks-options-splits-traders-2">IRS guidance</a>.</p></section>

<footer>Historical simulation, not investment advice. Yahoo adjusted total returns; raw Yahoo closes for market-cap estimation; local Alpha Vantage balance-sheet shares; FRED DGS3MO. Taxes, margin calls, market impact, slippage in the SPY sleeve, ETF tracking error, and intraday gaps are omitted. Daily-reset geared products can diverge materially from their stated multiple over longer periods; see <a href="https://www.finra.org/investors/insights/lowdown-leveraged-and-inverse-exchange-traded-products">FINRA's leveraged-product explanation</a>. Generated by <code>scripts/run_quality_compounder_v2.py</code>.</footer>
</main></body></html>'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
