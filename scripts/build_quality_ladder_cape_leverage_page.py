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
    no_brake = variants["quality_cape_no_brake"]
    spy = variants["spy_1x"]
    quality_1x = variants["quality_1x"]
    spy_ladder = variants["spy_ladder_cape"]
    no_harvest = variants["quality_cape_no_harvest"]

    labels = {
        "quality_cape": "Quality ladder + CAPE + NAV brake",
        "quality_cape_fresh_cape": "Quality ladder + fresh-capital CAPE sleeve",
        "quality_cape_no_brake": "Quality ladder + CAPE, no NAV brake",
        "quality_cape_no_harvest": "CAPE quality ladder, no harvest",
        "spy_ladder_cape": "SPY-only ladder + CAPE leverage",
        "cape_core_80_20": "CAPE core, no ladder",
        "cape_core_80_20_fresh_cape": "CAPE core + fresh-capital CAPE sleeve",
        "quality_1x": "Quality ladder, 1x core",
        "static_80_20": "Static contribution-split 80/20",
        "spy_1x": "SPY 1x",
    }
    colors = {
        "quality_cape_fresh_cape_wealth": "var(--purple)",
        "quality_cape_wealth": "var(--teal)",
        "quality_1x_wealth": "var(--blue)",
        "spy_1x_wealth": "var(--ink)",
        "quality_cape_fresh_cape_performance_index": "var(--purple)",
        "quality_cape_performance_index": "var(--teal)",
        "quality_1x_performance_index": "var(--blue)",
        "spy_1x_performance": "var(--ink)",
        "quality_cape_fresh_cape_pnl": "var(--purple)",
        "quality_cape_pnl": "var(--teal)",
        "quality_1x_pnl": "var(--blue)",
        "spy_1x_pnl": "var(--ink)",
    }
    growth = line_chart(
        daily,
        ["quality_cape_fresh_cape_wealth", "quality_cape_wealth", "spy_1x_wealth"],
        colors,
    )
    drawdown = line_chart(
        daily,
        ["quality_cape_fresh_cape_performance_index", "quality_cape_performance_index", "spy_1x_performance"],
        colors,
        performance=True,
    )
    pnl = linear_chart(
        daily,
        ["quality_cape_fresh_cape_pnl", "quality_cape_pnl", "spy_1x_pnl"],
        colors,
        "Cumulative profit and loss net of contributions",
    )
    cape_plot = cape_chart(daily)
    legend = (
        '<span><i style="--swatch:var(--purple)"></i>Fresh contributions follow CAPE</span>'
        '<span><i style="--swatch:var(--teal)"></i>All core obeys NAV brake</span>'
        '<span><i style="--swatch:var(--ink)"></i>SPY 1x</span>'
    )

    order = [
        "quality_cape_fresh_cape", "quality_cape", "quality_cape_no_brake",
        "quality_cape_no_harvest", "spy_ladder_cape", "cape_core_80_20_fresh_cape",
        "cape_core_80_20", "quality_1x", "static_80_20", "spy_1x",
    ]
    rows = "".join(
        f'<tr class="{"featured" if name == "quality_cape_fresh_cape" else ""}">'
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
        f'<tr class="{"featured" if name == "quality_cape_fresh_cape" else ""}">'
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
<h1>Fresh capital helps. <em>It also bypasses the brake.</em></h1>
<p class="standfirst">Existing core capital still falls to 1x at a 10% NAV drawdown. New contributions made while underwater now enter a separate sleeve at the leverage permitted by CAPE. Return improves, but the fresh sleeve can rebuild dangerous exposure before the account recovers.</p>
<div class="cards"><div class="card"><b>{money(fresh["terminal_wealth"])}</b><span>fresh-sleeve terminal wealth; SPY {money(spy["terminal_wealth"])}</span></div>
<div class="card"><b>{pct(fresh["xirr"])}</b><span>fresh-sleeve XIRR; SPY {pct(spy["xirr"])}</span></div>
<div class="card"><b>{pct(fresh["max_flow_adjusted_drawdown"])}</b><span>maximum drawdown; old brake path {pct(primary["max_flow_adjusted_drawdown"])}</span></div>
<div class="card"><b>{fresh["mean_gross_exposure"]:.2f}x</b><span>mean exposure; fresh sleeve active {pct(fresh["fresh_capital_active_share"], 1)} of sessions</span></div></div></header>

<section><div class="eyebrow">I - Corrected specification</div><h2>CAPE sets the ceiling; NAV can shut it down.</h2>
<p>Begin with {money(result["cash_flows"]["initial"])} at 80% SPY core and 20% Treasury. Add {money(result["cash_flows"]["annual"])} at the first session of every later year and an additional {money(result["cash_flows"]["additional_every_third_year"])} in contribution years 3, 6, 9 and so on. Total contributed was {money(result["cash_flows"]["total_contributed"])} across {result["cash_flows"]["contribution_events"]} later deposits.</p>
<div class="rule"><div class="step"><b>CAPE &lt;25</b><span>Standing SPY core at 3x.</span></div><div class="step"><b>25-35</b><span>Standing SPY core at 2x.</span></div><div class="step"><b>CAPE &gt;35</b><span>Standing SPY core at 1x.</span></div></div>
<p>Monthly CAPE is usable only from the next month. <strong>At a 10% flow-adjusted NAV drawdown, the core drops to 1x on the following session and remains at 1x until NAV makes a new high.</strong> It then restores only the leverage currently permitted by CAPE. Exposure above 1x pays prior-known DGS3MO +1%.</p>
<p><strong>New rule tested:</strong> while that brake is active, the 80% SPY portion of every annual contribution enters a separate sleeve at 3x below CAPE 25, 2x from 25 through 35, or 1x above 35. The remaining 20% still enters Treasury. At NAV recovery, the fresh sleeve merges into the legacy core and becomes subject to the next account-level brake.</p>
<p>The Treasury ladder remains unchanged: quality at -10%, SPY at -20%, quality at -30%, and SPY at -50% of SPY drawdown. Those rescue tranches remain unlevered.</p>
<p class="note">This is a daily-rebalanced leverage model, not a promise that a leveraged ETF, futures account or margin account would reproduce it. ETF tracking, margin liquidation and taxes are omitted.</p></section>

<section class="wide"><div class="eyebrow">II - P&amp;L versus SPY</div><h2>More return than the strict brake, but still just behind SPY.</h2>
<figure><div class="figure-head"><strong>Cumulative P&amp;L after subtracting all deposits</strong><span>not realized or tax-adjusted</span></div><div class="plate">{pnl}<div class="legend">{legend}</div></div></figure>
<p>The fresh-sleeve strategy produced {money(fresh["net_gain"])} of cumulative net gain versus {money(spy["net_gain"])} for SPY. It improved terminal wealth by {pct(fresh_vs_braked, 1)} over forcing every contribution to obey the active NAV brake, but still finished {pct(fresh_terminal_advantage, 1)} versus SPY with identical cash flows.</p>
<figure><div class="figure-head"><strong>Account value with matched deposits</strong><span>log scale</span></div><div class="plate">{growth}<div class="legend">{legend}</div></div></figure>
<figure><div class="figure-head"><strong>Flow-adjusted drawdown</strong><span>deposits removed from return clock</span></div><div class="plate">{drawdown}<div class="legend">{legend}</div></div></figure>
<p class="warning note"><strong>This is not free incremental return:</strong> maximum drawdown worsened by {fresh_drawdown_cost:.1%} versus the strict NAV brake and was also worse than SPY's {pct(spy["max_flow_adjusted_drawdown"])}. Fresh deposits followed CAPE even while the older account remained deleveraged.</p></section>

<section class="wide"><div class="eyebrow">III - Full comparison</div><h2>Separating old and new capital changes the middle ground.</h2>
<div class="scroll"><table><thead><tr><th>Path</th><th>Terminal</th><th>XIRR</th><th>TWR CAGR</th><th>Max DD</th><th>Volatility</th><th>Mean exposure</th></tr></thead><tbody>{rows}</tbody></table></div>
<p>The NAV brake improved the no-brake drawdown by {brake_drawdown_improvement:.1%}, but reduced terminal wealth by {pct(abs(brake_wealth_effect), 0)}. The corrected CAPE strategy ended {pct(leverage_effect, 1)} versus the same quality ladder held continuously at 1x. Quality selection added {pct(selection_effect, 1)} over the corrected SPY-only ladder, while harvesting reduced terminal wealth by {pct(abs(harvest_effect), 2)} versus leaving the selected stocks untouched.</p>
<p>The fresh contribution sleeve raised full-sample XIRR from {pct(primary["xirr"])} to {pct(fresh["xirr"])} and terminal wealth from {money(primary["terminal_wealth"])} to {money(fresh["terminal_wealth"])}. Financing cost increased from {money(primary["financing_cost"])} to {money(fresh["financing_cost"])}.</p>
<p class="caution note">The strongest tested combination was the simpler CAPE core plus NAV brake with no drawdown ladder: {money(variants["cape_core_80_20"]["terminal_wealth"])}, {pct(variants["cape_core_80_20"]["xirr"])} XIRR and {pct(variants["cape_core_80_20"]["max_flow_adjusted_drawdown"])} maximum drawdown. This is an in-sample comparison, not yet a validated replacement rule.</p>
<p>Corrected-strategy financing charges accumulated to {money(primary["financing_cost"])}. Actual core leverage was 3x for {pct(primary["time_at_3x"], 1)} of sessions, 2x for {pct(primary["time_at_2x"], 1)}, and 1x for {pct(primary["time_at_1x"], 1)}.</p></section>

<section class="wide"><div class="eyebrow">IV - CAPE regime</div><h2>CAPE now governs fresh money while NAV governs old money.</h2>
<figure><div class="figure-head"><strong>Prior-known monthly CAPE</strong><span>green 3x / amber 2x / red 1x</span></div><div class="plate">{cape_plot}</div></figure>
<p>The background shows the CAPE ceiling, not necessarily applied leverage. The NAV brake was active {pct(primary["nav_brake_share"], 1)} of sessions, reducing average core leverage to {primary["mean_core_leverage"]:.2f}x and average total gross exposure to {primary["mean_gross_exposure"]:.2f}x. There were {primary["events"]["nav_deleverage"]["count"]} deleveraging events and {primary["events"]["nav_leverage_restore"]["count"]} completed restorations.</p>
<p>Under the new rule, the fresh sleeve was present for {pct(fresh["fresh_capital_active_share"], 1)} of sessions but averaged only {pct(fresh["mean_fresh_capital_weight"], 1)} of total NAV over the full history. Even that modest average raised mean gross exposure to {fresh["mean_gross_exposure"]:.2f}x.</p></section>

<section class="wide"><div class="eyebrow">V - 2007 clean-data sensitivity</div><h2>The extra return came with a 71% drawdown.</h2>
<div class="scroll"><table><thead><tr><th>2007-2024 path</th><th>Terminal</th><th>XIRR</th><th>Max DD</th><th>Volatility</th></tr></thead><tbody>{post_rows}</tbody></table></div>
<p>With {money(sensitivity["cash_flows"]["total_contributed"])} contributed, the strict-brake strategy reached {money(post["quality_cape"]["terminal_wealth"])} versus {money(post["spy_1x"]["terminal_wealth"])} for SPY, a relative result of {pct(post_advantage, 1)}.</p>
<p>The fresh-sleeve version reached {money(post_fresh["terminal_wealth"])} and beat SPY terminal wealth by {pct(post_fresh_advantage, 1)}, but maximum drawdown deteriorated to {pct(post_fresh["max_flow_adjusted_drawdown"])}. Contributions made during the 2008 decline accumulated leveraged exposure before the legacy NAV recovered.</p></section>

<section><div class="eyebrow">VI - Decision</div><h2>The idea accelerates return, not safety.</h2>
<p>Letting new money ignore the account drawdown prevents the NAV brake from trapping every future deposit at 1x. But during a long recovery, repeated deposits can make the exempt sleeve large enough to dominate account risk.</p>
<p class="warning note"><strong>Bottom line:</strong> CAPE-only leverage for fresh contributions is too permissive in this form. It improves the strict-brake return, but the full sample still trails SPY and the post-2007 drawdown reaches 71%. The next sensible test is to cap fresh capital at 2x and give the fresh sleeve its own 10% drawdown brake, rather than allowing 3x until the whole account recovers.</p></section>

<footer>Historical simulation, not investment advice. Adjusted total returns reinvest dividends. Monthly CAPE is lagged one month and the test stops at the local September 2024 observation. CAPE source: <a href="https://www.econ.yale.edu/~shiller/data.htm">Robert Shiller / Yale</a>. Treasury and funding reference: <a href="https://fred.stlouisfed.org/series/DGS3MO">FRED DGS3MO</a>. Financing is prior-known DGS3MO +1%; quality and rescue trades cost 5 bp. Margin calls, leveraged-ETF tracking differences, taxes, market impact and a survivor-free historical top-seven universe are omitted. Generated by <code>scripts/run_quality_ladder_cape_leverage.py</code>.</footer>
</main></body></html>'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
