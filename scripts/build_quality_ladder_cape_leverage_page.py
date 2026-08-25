"""Build the CAPE-leveraged quality-ladder performance page."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import pandas as pd

from build_contribution_quality_page import line_chart, money, pct


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("out/strategy/quality_ladder_cape_leverage"),
    )
    parser.add_argument(
        "--sensitivity", type=Path,
        default=Path("out/strategy/quality_ladder_cape_leverage_2007"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("docs/quality-ladder-cape-leverage.html"),
    )
    return parser.parse_args(argv)


def linear_chart(frame: pd.DataFrame, columns: list[str], colors: dict[str, str],
                 aria: str) -> str:
    width, height = 980, 370
    left, right, top, bottom = 92, 24, 22, 46
    plot_w, plot_h = width - left - right, height - top - bottom
    sampled = frame.set_index("date").resample("ME").last()
    values = {column: sampled[column].dropna().astype(float) for column in columns}
    low = min(0.0, min(series.min() for series in values.values()))
    high = max(series.max() for series in values.values())
    pad = max((high - low) * 0.04, 1.0)
    low, high = low - (pad if low < 0 else 0), high + pad
    ticks = [low + (high - low) * i / 5 for i in range(6)]
    start, end = sampled.index[0], sampled.index[-1]
    span = max((end - start).days, 1)
    x = lambda stamp: left + (stamp - start).days / span * plot_w
    y = lambda value: top + (high - value) / (high - low) * plot_h
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(aria)}">',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" class="frame"/>',
    ]
    for tick in ticks:
        yy = y(tick)
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{yy:.1f}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-9}" y="{yy+4:.1f}" text-anchor="end" class="axis">{money(tick)}</text>')
    if low < 0 < high:
        yy = y(0.0)
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{yy:.1f}" y2="{yy:.1f}" class="zero"/>')
    for year in range(((start.year + 4) // 5) * 5, end.year + 1, 5):
        xx = x(pd.Timestamp(year, 1, 1))
        parts.append(f'<text x="{xx:.1f}" y="{height-17}" text-anchor="middle" class="axis">{year}</text>')
    for column, series in values.items():
        points = " ".join(
            f"{x(stamp):.1f},{y(value):.1f}" for stamp, value in series.items()
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[column]}" stroke-width="2.3" vector-effect="non-scaling-stroke"/>')
    parts.append('</svg>')
    return "".join(parts)


def cape_chart(frame: pd.DataFrame) -> str:
    width, height = 980, 320
    left, right, top, bottom = 68, 24, 20, 44
    plot_w, plot_h = width - left - right, height - top - bottom
    sampled = frame.set_index("date").resample("ME").last()
    values = sampled["cape_known"].dropna().astype(float)
    low, high = 15.0, max(45.0, math.ceil(values.max() / 5) * 5)
    start, end = sampled.index[0], sampled.index[-1]
    span = max((end - start).days, 1)
    x = lambda stamp: left + (stamp - start).days / span * plot_w
    y = lambda value: top + (high - value) / (high - low) * plot_h
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Lagged CAPE and leverage thresholds">',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="var(--red-soft)"/>',
        f'<rect x="{left}" y="{y(35):.1f}" width="{plot_w}" height="{y(25)-y(35):.1f}" fill="var(--gold)"/>',
        f'<rect x="{left}" y="{y(25):.1f}" width="{plot_w}" height="{height-bottom-y(25):.1f}" fill="var(--green-soft)"/>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" class="frame"/>',
    ]
    for tick in range(15, int(high) + 1, 5):
        yy = y(tick)
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{yy:.1f}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" class="axis">{tick}</text>')
    points = " ".join(
        f"{x(stamp):.1f},{y(value):.1f}" for stamp, value in values.items()
    )
    parts.append(f'<polyline points="{points}" fill="none" stroke="var(--ink)" stroke-width="2.2" vector-effect="non-scaling-stroke"/>')
    for year in range(((start.year + 4) // 5) * 5, end.year + 1, 5):
        xx = x(pd.Timestamp(year, 1, 1))
        parts.append(f'<text x="{xx:.1f}" y="{height-15}" text-anchor="middle" class="axis">{year}</text>')
    parts.append('</svg>')
    return "".join(parts)


def main(argv=None):
    args = parse_args(argv)
    result = json.loads((args.input / "results.json").read_text(encoding="utf-8"))
    sensitivity = json.loads(
        (args.sensitivity / "results.json").read_text(encoding="utf-8")
    )
    daily = pd.read_csv(args.input / "daily.csv", parse_dates=["date"])
    variants = result["variants"]
    primary = variants["quality_cape"]
    fresh = variants["quality_cape_fresh_cape"]
    dual = variants["quality_dual_guard_injections"]
    spy_dual = variants["spy_dual_guard_injections"]
    no_brake = variants["quality_cape_no_brake"]
    spy = variants["spy_1x"]
    quality_1x = variants["quality_1x"]
    spy_ladder = variants["spy_ladder_cape"]
    no_harvest = variants["quality_cape_no_harvest"]

    labels = {
        "quality_cape": "Quality ladder + CAPE + NAV brake",
        "quality_cape_fresh_cape": "Quality ladder + fresh-capital CAPE sleeve",
        "quality_dual_guard_injections": "Dual guard: 1x core + CAPE drawdown injections",
        "quality_cape_no_brake": "Quality ladder + CAPE, no NAV brake",
        "quality_cape_no_harvest": "CAPE quality ladder, no harvest",
        "spy_ladder_cape": "SPY-only ladder + CAPE leverage",
        "spy_dual_guard_injections": "Dual guard with SPY-only Treasury ladder",
        "cape_core_80_20": "CAPE core, no ladder",
        "cape_core_80_20_fresh_cape": "CAPE core + fresh-capital CAPE sleeve",
        "quality_1x": "Quality ladder, 1x core",
        "static_80_20": "Static contribution-split 80/20",
        "spy_1x": "SPY 1x",
    }
    colors = {
        "quality_dual_guard_injections_wealth": "var(--purple)",
        "quality_cape_fresh_cape_wealth": "var(--purple)",
        "quality_cape_wealth": "var(--teal)",
        "quality_1x_wealth": "var(--blue)",
        "spy_1x_wealth": "var(--ink)",
        "quality_dual_guard_injections_performance_index": "var(--purple)",
        "quality_cape_fresh_cape_performance_index": "var(--purple)",
        "quality_cape_performance_index": "var(--teal)",
        "quality_1x_performance_index": "var(--blue)",
        "spy_1x_performance": "var(--ink)",
        "quality_dual_guard_injections_pnl": "var(--purple)",
        "quality_cape_fresh_cape_pnl": "var(--purple)",
        "quality_cape_pnl": "var(--teal)",
        "quality_1x_pnl": "var(--blue)",
        "spy_1x_pnl": "var(--ink)",
    }
    growth = line_chart(
        daily,
        ["quality_dual_guard_injections_wealth", "quality_1x_wealth", "spy_1x_wealth"],
        colors,
    )
    drawdown = line_chart(
        daily,
        ["quality_dual_guard_injections_performance_index", "quality_1x_performance_index", "spy_1x_performance"],
        colors,
        performance=True,
    )
    pnl = linear_chart(
        daily,
        ["quality_dual_guard_injections_pnl", "quality_1x_pnl", "spy_1x_pnl"],
        colors,
        "Cumulative profit and loss net of contributions",
    )
    cape_plot = cape_chart(daily)
    legend = (
        '<span><i style="--swatch:var(--purple)"></i>Dual-guard injection strategy</span>'
        '<span><i style="--swatch:var(--blue)"></i>Quality ladder, 1x core</span>'
        '<span><i style="--swatch:var(--ink)"></i>SPY 1x</span>'
    )

    order = [
        "quality_dual_guard_injections", "spy_dual_guard_injections",
        "quality_1x", "quality_cape_fresh_cape", "quality_cape",
        "quality_cape_no_brake", "quality_cape_no_harvest", "spy_ladder_cape",
        "cape_core_80_20_fresh_cape", "cape_core_80_20", "static_80_20", "spy_1x",
    ]
    rows = "".join(
        f'<tr class="{"featured" if name == "quality_dual_guard_injections" else ""}">'
        f'<td><strong>{html.escape(labels[name])}</strong></td>'
        f'<td>{money(variants[name]["terminal_wealth"])}</td>'
        f'<td>{pct(variants[name]["xirr"])}</td>'
        f'<td>{pct(variants[name]["time_weighted_cagr"])}</td>'
        f'<td>{pct(variants[name]["max_flow_adjusted_drawdown"])}</td>'
        f'<td>{pct(variants[name]["annual_volatility"])}</td>'
        f'<td>{variants[name].get("mean_gross_exposure", 1.0):.2f}x</td></tr>'
        for name in order
    )
    post_rows = "".join(
        f'<tr class="{"featured" if name == "quality_dual_guard_injections" else ""}">'
        f'<td><strong>{html.escape(labels[name])}</strong></td>'
        f'<td>{money(sensitivity["variants"][name]["terminal_wealth"])}</td>'
        f'<td>{pct(sensitivity["variants"][name]["xirr"])}</td>'
        f'<td>{pct(sensitivity["variants"][name]["max_flow_adjusted_drawdown"])}</td>'
        f'<td>{pct(sensitivity["variants"][name]["annual_volatility"])}</td></tr>'
        for name in order
    )

    terminal_advantage = primary["terminal_wealth"] / spy["terminal_wealth"] - 1.0
    pnl_advantage = primary["net_gain"] - spy["net_gain"]
    recovery_required = 1.0 / (1.0 + primary["max_flow_adjusted_drawdown"]) - 1.0
    selection_effect = primary["terminal_wealth"] / spy_ladder["terminal_wealth"] - 1.0
    harvest_effect = primary["terminal_wealth"] / no_harvest["terminal_wealth"] - 1.0
    leverage_effect = primary["terminal_wealth"] / quality_1x["terminal_wealth"] - 1.0
    brake_wealth_effect = primary["terminal_wealth"] / no_brake["terminal_wealth"] - 1.0
    brake_drawdown_improvement = abs(
        primary["max_flow_adjusted_drawdown"] - no_brake["max_flow_adjusted_drawdown"]
    )
    post = sensitivity["variants"]
    post_advantage = post["quality_cape"]["terminal_wealth"] / post["spy_1x"]["terminal_wealth"] - 1.0
    fresh_terminal_advantage = fresh["terminal_wealth"] / spy["terminal_wealth"] - 1.0
    fresh_vs_braked = fresh["terminal_wealth"] / primary["terminal_wealth"] - 1.0
    fresh_drawdown_cost = abs(fresh["max_flow_adjusted_drawdown"]) - abs(primary["max_flow_adjusted_drawdown"])
    post_fresh = post["quality_cape_fresh_cape"]
    post_fresh_advantage = post_fresh["terminal_wealth"] / post["spy_1x"]["terminal_wealth"] - 1.0
    dual_vs_spy = dual["terminal_wealth"] / spy["terminal_wealth"] - 1.0
    dual_vs_quality_1x = dual["terminal_wealth"] / quality_1x["terminal_wealth"] - 1.0
    dual_selection_effect = dual["terminal_wealth"] / spy_dual["terminal_wealth"] - 1.0
    post_dual = post["quality_dual_guard_injections"]
    post_dual_vs_spy = post_dual["terminal_wealth"] / post["spy_1x"]["terminal_wealth"] - 1.0
    levered_injections = [
        event for event in result["events"]["quality_dual_guard_injections"]
        if event["kind"] == "contribution"
        and (" at 2x" in event["detail"] or " at 3x" in event["detail"])
    ]
    injection_rows = "".join(
        f'<tr><td><strong>{event["date"]}</strong></td>'
        f'<td>{money(event["amount"])}</td>'
        f'<td>{pct(event["drawdown"], 1)}</td>'
        f'<td>{html.escape(event["detail"])}</td></tr>'
        for event in levered_injections
    )
    pnl_gap_text = (
        f"{money(abs(pnl_advantage))} more"
        if pnl_advantage >= 0
        else f"{money(abs(pnl_advantage))} less"
    )

    html_text = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>CAPE-leveraged quality ladder</title>
<style>
:root{{--bg:#f1f4f2;--paper:#fcfdfc;--ink:#14201d;--muted:#53635f;--faint:#778681;--line:#ccd7d3;--teal:#087f72;--blue:#356da6;--purple:#7555a6;--brick:#b84138;--soft:#dcefea;--warn:#f5dfdc;--gold:#f5ecd0;--green-soft:#e1eee5;--red-soft:#f3dddd;--mono:"Cascadia Mono",Consolas,monospace;--serif:Georgia,"Times New Roman",serif;--sans:system-ui,-apple-system,"Segoe UI",sans-serif}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:17px/1.62 var(--serif)}}main{{width:min(1100px,calc(100% - 34px));margin:auto;padding:52px 0 80px}}header{{border-bottom:2px solid var(--ink);padding-bottom:34px}}.eyebrow{{font:700 11px var(--mono);letter-spacing:.13em;text-transform:uppercase;color:var(--teal)}}h1{{font:650 clamp(40px,7vw,72px)/1.02 var(--serif);letter-spacing:-.045em;max-width:14ch;margin:13px 0 20px}}h1 em{{color:var(--brick)}}.standfirst{{font-size:22px;line-height:1.43;color:var(--muted);max-width:66ch}}section{{padding-top:48px;max-width:880px}}section.wide{{max-width:none}}h2{{font:650 31px/1.18 var(--serif);letter-spacing:-.025em;margin:0 0 14px}}h3{{font:650 21px/1.3 var(--serif);margin:30px 0 10px}}p{{margin:0 0 18px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);margin-top:28px}}.card{{background:var(--paper);padding:17px;border-right:1px solid var(--line)}}.card:last-child{{border:0}}.card b{{display:block;font:700 22px var(--mono)}}.card span{{display:block;font:12px/1.4 var(--sans);color:var(--faint);margin-top:7px}}.rule{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:23px 0}}.step{{background:var(--paper);border-top:4px solid var(--teal);padding:17px}}.step:nth-child(2){{border-color:#b38108}}.step:last-child{{border-color:var(--brick)}}.step b{{display:block;font:700 22px var(--mono)}}.step span{{font:12px/1.45 var(--sans);color:var(--faint)}}.note{{background:var(--soft);border-left:3px solid var(--teal);padding:18px 20px;color:var(--muted)}}.warning{{background:var(--warn);border-left-color:var(--brick)}}.caution{{background:var(--gold);border-left-color:#9a6a00}}.scroll{{overflow:auto;border:1px solid var(--line);background:var(--paper);margin-top:22px}}table{{border-collapse:collapse;width:100%;font:13px/1.4 var(--sans);font-variant-numeric:tabular-nums}}th,td{{padding:12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}}th{{font:700 10px var(--mono);text-transform:uppercase;letter-spacing:.06em;color:var(--faint)}}th:first-child,td:first-child{{text-align:left}}td:first-child{{white-space:normal;min-width:180px}}td strong,td small{{display:block}}tr.featured{{background:var(--soft)}}figure{{margin:30px 0}}.figure-head{{display:flex;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:9px;font:13px var(--sans);color:var(--muted)}}.plate{{background:var(--paper);border:1px solid var(--line);padding:14px;margin-top:13px;overflow:auto}}svg{{display:block;width:100%;min-width:720px}}.frame{{fill:none;stroke:var(--line)}}.grid{{stroke:var(--line)}}.zero{{stroke:var(--brick);stroke-width:1.5}}.axis{{font:10px var(--mono);fill:var(--faint)}}.axis-title{{font:12px var(--sans);fill:var(--ink)}}.legend{{display:flex;gap:18px;flex-wrap:wrap;padding:8px 4px 0;font:11px var(--sans);color:var(--muted)}}.legend span{{display:flex;gap:7px;align-items:center}}.legend i{{width:22px;border-top:2px solid var(--swatch)}}footer{{font:12px/1.55 var(--sans);color:var(--faint);border-top:1px solid var(--line);margin-top:55px;padding-top:20px;max-width:960px}}a{{color:var(--teal)}}code{{font-family:var(--mono)}}@media(max-width:760px){{.cards{{grid-template-columns:1fr 1fr}}.rule{{grid-template-columns:1fr}}.card:nth-child(2){{border-right:0}}}}@media(max-width:520px){{.cards{{grid-template-columns:1fr}}}}
</style></head><body><main>
<header><div class="eyebrow">CAPE sample - {result["sample"]["start"]} to {result["sample"]["end"]}</div>
<h1>NAV chooses when. <em>CAPE chooses how much.</em></h1>
<p class="standfirst">The permanent SPY core never exceeds 1x. Only new capital arriving during an account drawdown can use leverage, with its entry exposure selected by CAPE. Once the combined account recovers its prior high, every injected tranche converts back to 1x.</p>
<div class="cards"><div class="card"><b>{money(dual["terminal_wealth"])}</b><span>dual-guard terminal wealth; SPY {money(spy["terminal_wealth"])}</span></div>
<div class="card"><b>{pct(dual["xirr"])}</b><span>dual-guard XIRR; SPY {pct(spy["xirr"])}</span></div>
<div class="card"><b>{pct(dual["max_flow_adjusted_drawdown"])}</b><span>maximum drawdown; SPY {pct(spy["max_flow_adjusted_drawdown"])}</span></div>
<div class="card"><b>{dual["mean_gross_exposure"]:.2f}x</b><span>mean gross exposure; maximum {dual["max_gross_exposure"]:.2f}x</span></div></div></header>

<section><div class="eyebrow">I - Two-account specification</div><h2>Permanent capital and opportunity capital have different jobs.</h2>
<p>Begin with {money(result["cash_flows"]["initial"])} at 80% SPY core and 20% Treasury. Add {money(result["cash_flows"]["annual"])} at the first session of every later year and an additional {money(result["cash_flows"]["additional_every_third_year"])} in contribution years 3, 6, 9 and so on. Total contributed was {money(result["cash_flows"]["total_contributed"])} across {result["cash_flows"]["contribution_events"]} later deposits.</p>
<div class="rule"><div class="step"><b>CAPE &lt;25</b><span>Drawdown-time SPY injection enters at 3x.</span></div><div class="step"><b>25-35</b><span>Drawdown-time SPY injection enters at 2x.</span></div><div class="step"><b>CAPE &gt;35</b><span>New capital buys SPY at 1x.</span></div></div>
<p><strong>NAV is the gate:</strong> the permanent 80% SPY core stays at 1x. If the prior-close flow-adjusted account NAV is less than 10% below its high, a new contribution also enters at 1x regardless of CAPE. At a drawdown of 10% or more, CAPE determines only the leverage of that new SPY tranche. The remaining 20% of every contribution enters Treasury.</p>
<p><strong>Recovery is the reset:</strong> each leveraged injection holds its entry exposure until the combined account regains the high that preceded the drawdown. On the following session it is converted to 1x and merged into the permanent core. Monthly CAPE is lagged one month, and exposure above 1x pays prior-known DGS3MO +1%.</p>
<p>The Treasury ladder remains unchanged: quality at -10%, SPY at -20%, quality at -30%, and SPY at -50% of SPY drawdown. Those rescue tranches remain unlevered.</p>
<p class="note">This is a daily-rebalanced leverage model, not a promise that a leveraged ETF, futures account or margin account would reproduce it. ETF tracking, margin liquidation and taxes are omitted.</p></section>

<section class="wide"><div class="eyebrow">II - P&amp;L versus SPY</div><h2>The dual guard nearly matched SPY with less full-sample drawdown.</h2>
<figure><div class="figure-head"><strong>Cumulative P&amp;L after subtracting all deposits</strong><span>not realized or tax-adjusted</span></div><div class="plate">{pnl}<div class="legend">{legend}</div></div></figure>
<p>The dual-guard strategy produced {money(dual["net_gain"])} of cumulative net gain versus {money(spy["net_gain"])} for SPY. Terminal wealth was {pct(dual_vs_spy, 2)} versus SPY with identical cash flows, while XIRR differed by only {(dual["xirr"] - spy["xirr"]) * 10_000:.0f} basis points.</p>
<figure><div class="figure-head"><strong>Account value with matched deposits</strong><span>log scale</span></div><div class="plate">{growth}<div class="legend">{legend}</div></div></figure>
<figure><div class="figure-head"><strong>Flow-adjusted drawdown</strong><span>deposits removed from return clock</span></div><div class="plate">{drawdown}<div class="legend">{legend}</div></div></figure>
<p class="note"><strong>Risk result:</strong> full-sample maximum drawdown was {pct(dual["max_flow_adjusted_drawdown"])} versus {pct(spy["max_flow_adjusted_drawdown"])} for SPY, and annualized volatility was {pct(dual["annual_volatility"])} versus {pct(spy["annual_volatility"])}.</p></section>

<section class="wide"><div class="eyebrow">III - Full comparison</div><h2>Selective leverage added return without levering the permanent core.</h2>
<div class="scroll"><table><thead><tr><th>Path</th><th>Terminal</th><th>XIRR</th><th>TWR CAGR</th><th>Max DD</th><th>Volatility</th><th>Mean exposure</th></tr></thead><tbody>{rows}</tbody></table></div>
<p>Selective injection leverage added {pct(dual_vs_quality_1x, 1)} of terminal wealth over the same quality/Treasury strategy with a permanently 1x core. The quality-stock rungs added {pct(dual_selection_effect, 2)} over using SPY for every Treasury rung. Total modeled financing cost was only {money(dual["financing_cost"])} because leverage was confined to qualifying contribution tranches.</p>
<p class="caution note">The average injected-capital weight was {pct(dual["mean_injection_weight"], 1)}, with a full-sample maximum of {pct(dual["max_injection_weight"], 1)}. These percentages depend strongly on contribution size relative to the account: a $40,000 triennial contribution is immaterial to a mature account but can dominate a recently started $10,000 account.</p></section>

<section class="wide"><div class="eyebrow">IV - Historical injections</div><h2>Eight deposits qualified for 2x or 3x exposure.</h2>
<figure><div class="figure-head"><strong>Prior-known monthly CAPE</strong><span>green 3x / amber 2x / red 1x</span></div><div class="plate">{cape_plot}</div></figure>
<div class="scroll"><table><thead><tr><th>Date</th><th>Total contribution</th><th>Account DD</th><th>Execution rule</th></tr></thead><tbody>{injection_rows}</tbody></table></div>
<p>There were {len(levered_injections)} levered contribution events and {dual["events"]["injection_leverage_reset"]["count"]} completed recovery resets. The injection sleeve was active for {pct(dual["injection_active_share"], 1)} of sessions; outside these episodes, all SPY capital remained at 1x.</p></section>

<section class="wide"><div class="eyebrow">V - 2007 clean-data sensitivity</div><h2>It edged past SPY, but starting-account size matters.</h2>
<div class="scroll"><table><thead><tr><th>2007-2024 path</th><th>Terminal</th><th>XIRR</th><th>Max DD</th><th>Volatility</th></tr></thead><tbody>{post_rows}</tbody></table></div>
<p>With {money(sensitivity["cash_flows"]["total_contributed"])} contributed, the dual guard reached {money(post_dual["terminal_wealth"])} versus {money(post["spy_1x"]["terminal_wealth"])} for SPY, an advantage of {pct(post_dual_vs_spy, 2)}. XIRR was {pct(post_dual["xirr"])} versus {pct(post["spy_1x"]["xirr"])}.</p>
<p class="warning note">Maximum drawdown was {pct(post_dual["max_flow_adjusted_drawdown"])} versus SPY's {pct(post["spy_1x"]["max_flow_adjusted_drawdown"])}. Maximum gross exposure reached {post_dual["max_gross_exposure"]:.2f}x because the 2009 and 2010 injections were large relative to the initial $10,000 account. A practical implementation needs an account-level gross-exposure cap.</p></section>

<section><div class="eyebrow">VI - Decision</div><h2>This is the cleanest version tested so far.</h2>
<p>The economic separation is coherent: old capital compounds at 1x, NAV prevents leverage near highs, CAPE controls the size of genuinely new drawdown exposure, and recovery automatically removes the borrowing. The quality/Treasury mechanism remains independent.</p>
<p class="warning note"><strong>Bottom line:</strong> the historical result is close to SPY with modestly better full-sample drawdown, and it slightly beats SPY after 2007. The remaining flaw is scale, not signal selection. Before considering implementation, cap total account gross exposure—1.5x is a reasonable next test—so a large contribution cannot make a young account effectively 2x or more.</p></section>

<footer><strong>Full operating manual:</strong> <a href="dual-guard-quality-compounder-guide.md">Dual-Guard Quality Compounder Guide</a>.<br>Historical simulation, not investment advice. Adjusted total returns reinvest dividends. Monthly CAPE is lagged one month and the test stops at the local September 2024 observation. CAPE source: <a href="https://www.econ.yale.edu/~shiller/data.htm">Robert Shiller / Yale</a>. Treasury and funding reference: <a href="https://fred.stlouisfed.org/series/DGS3MO">FRED DGS3MO</a>. Financing is prior-known DGS3MO +1%; quality and rescue trades cost 5 bp. Margin calls, leveraged-ETF tracking differences, taxes, market impact and a survivor-free historical top-seven universe are omitted. Generated by <code>scripts/run_quality_ladder_cape_leverage.py</code>.</footer>
</main></body></html>'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
