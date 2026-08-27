"""Build the self-contained quality-ladder harvest backtest report."""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

from build_contribution_quality_page import line_chart, money, pct


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("out/strategy/quality_ladder_harvest"),
    )
    parser.add_argument(
        "--sensitivity", type=Path,
        default=Path("out/strategy/quality_ladder_harvest_2007"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("docs/quality-ladder-harvest.html"),
    )
    return parser.parse_args(argv)


def weight_chart(frame: pd.DataFrame) -> str:
    width, height = 980, 330
    left, right, top, bottom = 72, 24, 20, 45
    plot_w, plot_h = width - left - right, height - top - bottom
    sampled = frame.set_index("date").resample("ME").last()
    series = {
        "Treasury": sampled["quality_ladder_treasury"] / sampled["quality_ladder_wealth"],
        "Quality": sampled["quality_ladder_quality_weight"],
    }
    high = max(0.25, math.ceil(max(value.max() for value in series.values()) * 20) / 20)
    ticks = [high * i / 5 for i in range(6)]
    start, end = sampled.index[0], sampled.index[-1]
    span = max((end - start).days, 1)
    x = lambda stamp: left + (stamp - start).days / span * plot_w
    y = lambda value: top + (high - value) / high * plot_h
    colors = {"Treasury": "var(--blue)", "Quality": "var(--teal)"}
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Portfolio sleeve weights">',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" class="frame"/>',
    ]
    for tick in ticks:
        yy = y(tick)
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{yy:.1f}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" class="axis">{tick:.0%}</text>')
    target_y = y(0.20)
    parts.append(f'<line x1="{left}" x2="{width-right}" y1="{target_y:.1f}" y2="{target_y:.1f}" class="target"/>')
    for year in range(((start.year + 4) // 5) * 5, end.year + 1, 5):
        xx = x(pd.Timestamp(year, 1, 1))
        parts.append(f'<text x="{xx:.1f}" y="{height-16}" text-anchor="middle" class="axis">{year}</text>')
    for label, values in series.items():
        points = " ".join(
            f"{x(stamp):.1f},{y(float(value)):.1f}"
            for stamp, value in values.dropna().items()
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[label]}" stroke-width="2.2" vector-effect="non-scaling-stroke"/>')
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
    primary = variants["quality_ladder"]
    spy = variants["spy_1x"]
    spy_ladder = variants["spy_ladder"]
    static = variants["static_80_20"]
    no_harvest = variants["quality_ladder_no_harvest"]

    shown = ["quality_ladder", "spy_ladder", "static_80_20", "spy_1x"]
    labels = {
        "quality_ladder": "Quality ladder + harvest",
        "spy_ladder": "SPY-only ladder",
        "static_80_20": "Static contribution-split 80/20",
        "spy_1x": "SPY 1x",
    }
    colors = {
        "quality_ladder_wealth": "var(--teal)",
        "spy_ladder_wealth": "var(--blue)",
        "static_80_20_wealth": "var(--amber)",
        "spy_1x_wealth": "var(--ink)",
        "quality_ladder_performance_index": "var(--teal)",
        "spy_ladder_performance_index": "var(--blue)",
        "static_80_20_performance_index": "var(--amber)",
        "spy_1x_performance": "var(--ink)",
    }
    growth = line_chart(
        daily,
        [f"{name}_wealth" for name in shown],
        colors,
    )
    performance_cols = [
        "quality_ladder_performance_index",
        "spy_ladder_performance_index",
        "static_80_20_performance_index",
        "spy_1x_performance",
    ]
    drawdown = line_chart(daily, performance_cols, colors, performance=True)
    allocation = weight_chart(daily)
    legend = "".join(
        f'<span><i style="--swatch:{colors[f"{name}_wealth"]}"></i>{html.escape(labels[name])}</span>'
        for name in shown
    )

    order = [
        "quality_ladder", "quality_ladder_no_harvest", "spy_ladder",
        "static_80_20", "spy_1x",
    ]
    table_labels = {
        **labels,
        "quality_ladder_no_harvest": "Quality ladder, no harvesting",
    }
    comparison_rows = "".join(
        f'<tr class="{"featured" if name == "quality_ladder" else ""}">'
        f'<td><strong>{html.escape(table_labels[name])}</strong></td>'
        f'<td>{money(variants[name]["terminal_wealth"])}</td>'
        f'<td>{pct(variants[name]["xirr"])}</td>'
        f'<td>{pct(variants[name]["time_weighted_cagr"])}</td>'
        f'<td>{pct(variants[name]["max_flow_adjusted_drawdown"])}</td>'
        f'<td>{pct(variants[name]["annual_volatility"])}</td>'
        f'<td>{variants[name]["longest_underwater_sessions"] / 252:.1f}y</td></tr>'
        for name in order
    )

    sensitivity_rows = "".join(
        f'<tr class="{"featured" if name == "quality_ladder" else ""}">'
        f'<td><strong>{html.escape(table_labels[name])}</strong></td>'
        f'<td>{money(sensitivity["variants"][name]["terminal_wealth"])}</td>'
        f'<td>{pct(sensitivity["variants"][name]["xirr"])}</td>'
        f'<td>{pct(sensitivity["variants"][name]["max_flow_adjusted_drawdown"])}</td>'
        f'<td>{pct(sensitivity["variants"][name]["annual_volatility"])}</td></tr>'
        for name in order
    )

    deployments = [
        event for event in result["events"]["quality_ladder"]
        if event["kind"] in {"deploy_quality_seven", "quality_rung_skipped"}
    ]
    deployment_rows = "".join(
        f'<tr><td>{event["date"]}</td><td>{html.escape(event["kind"].replace("_", " "))}</td>'
        f'<td>{money(event["amount"])}</td><td>{html.escape(event["detail"])}</td></tr>'
        for event in deployments
    )

    holdings = result["holdings"]["quality_ladder"]
    holdings_rows = "".join(
        f'<tr><td><strong>{html.escape(row["ticker"])}</strong><small>{", ".join(row["entry_dates"])}</small></td>'
        f'<td>{money(row["market_value"])}</td><td>{pct(row["portfolio_weight"])}</td>'
        f'<td>{row["lots"]}</td><td>{row["harvest_bands"]}</td>'
        f'<td>{pct(row["harvested_original_share"], 1)}</td></tr>'
        for row in holdings
    )

    harvested_by_ticker = defaultdict(lambda: {"events": 0, "amount": 0.0})
    for event in result["events"]["quality_ladder"]:
        if event["kind"] != "quality_relative_harvest":
            continue
        ticker = event["detail"].split(";", 1)[0]
        harvested_by_ticker[ticker]["events"] += 1
        harvested_by_ticker[ticker]["amount"] += event["amount"]
    harvest_rows = "".join(
        f'<tr><td><strong>{html.escape(ticker)}</strong></td><td>{row["events"]}</td><td>{money(row["amount"])}</td></tr>'
        for ticker, row in sorted(
            harvested_by_ticker.items(),
            key=lambda item: item[1]["amount"], reverse=True,
        )
    )

    terminal_gap = primary["terminal_wealth"] / spy["terminal_wealth"] - 1.0
    selection_gap = primary["terminal_wealth"] / spy_ladder["terminal_wealth"] - 1.0
    harvest_gap = primary["terminal_wealth"] / no_harvest["terminal_wealth"] - 1.0
    static_gap = primary["terminal_wealth"] / static["terminal_wealth"] - 1.0
    post = sensitivity["variants"]
    post_spy_gap = post["quality_ladder"]["terminal_wealth"] / post["spy_1x"]["terminal_wealth"] - 1.0
    post_ladder_gap = post["quality_ladder"]["terminal_wealth"] / post["spy_ladder"]["terminal_wealth"] - 1.0

    html_text = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Quality Drawdown Ladder - backtest</title>
<style>
:root{{--bg:#f1f4f2;--paper:#fcfdfc;--ink:#14201d;--muted:#53635f;--faint:#778681;--line:#ccd7d3;--teal:#087f72;--blue:#356da6;--amber:#9a6a00;--soft:#dcefea;--brick:#b84138;--warn:#f5dfdc;--gold:#f5ecd0;--mono:"Cascadia Mono",Consolas,monospace;--serif:Georgia,"Times New Roman",serif;--sans:system-ui,-apple-system,"Segoe UI",sans-serif}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:17px/1.62 var(--serif)}}main{{width:min(1100px,calc(100% - 34px));margin:auto;padding:52px 0 80px}}header{{border-bottom:2px solid var(--ink);padding-bottom:34px}}.eyebrow{{font:700 11px var(--mono);letter-spacing:.13em;text-transform:uppercase;color:var(--teal)}}h1{{font:650 clamp(40px,7vw,72px)/1.02 var(--serif);letter-spacing:-.045em;max-width:14ch;margin:13px 0 20px}}h1 em{{color:var(--teal)}}.standfirst{{font-size:22px;line-height:1.43;color:var(--muted);max-width:66ch}}section{{padding-top:48px;max-width:880px}}section.wide{{max-width:none}}h2{{font:650 31px/1.18 var(--serif);letter-spacing:-.025em;margin:0 0 14px}}h3{{font:650 21px/1.3 var(--serif);margin:30px 0 10px}}p{{margin:0 0 18px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);margin-top:28px}}.card{{background:var(--paper);padding:17px;border-right:1px solid var(--line)}}.card:last-child{{border:0}}.card b{{display:block;font:700 22px var(--mono)}}.card span{{display:block;font:12px/1.4 var(--sans);color:var(--faint);margin-top:7px}}.rule{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:23px 0}}.step{{background:var(--paper);border-top:3px solid var(--teal);padding:15px}}.step:nth-child(even){{border-color:var(--blue)}}.step b{{display:block;font:700 18px var(--mono)}}.step span{{font:12px/1.45 var(--sans);color:var(--faint)}}.note{{background:var(--soft);border-left:3px solid var(--teal);padding:18px 20px;color:var(--muted)}}.warning{{background:var(--warn);border-left-color:var(--brick)}}.caution{{background:var(--gold);border-left-color:var(--amber)}}.scroll{{overflow:auto;border:1px solid var(--line);background:var(--paper);margin-top:22px}}table{{border-collapse:collapse;width:100%;font:13px/1.4 var(--sans);font-variant-numeric:tabular-nums}}th,td{{padding:12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}}th{{font:700 10px var(--mono);text-transform:uppercase;letter-spacing:.06em;color:var(--faint)}}th:first-child,td:first-child{{text-align:left}}td:first-child{{white-space:normal;min-width:165px}}td strong,td small{{display:block}}td small{{color:var(--faint);margin-top:3px}}tr.featured{{background:var(--soft)}}figure{{margin:30px 0}}.figure-head{{display:flex;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:9px;font:13px var(--sans);color:var(--muted)}}.plate{{background:var(--paper);border:1px solid var(--line);padding:14px;margin-top:13px;overflow:auto}}svg{{display:block;width:100%;min-width:720px}}.frame{{fill:none;stroke:var(--line)}}.grid{{stroke:var(--line)}}.target{{stroke:var(--brick);stroke-dasharray:5 5}}.axis{{font:10px var(--mono);fill:var(--faint)}}.axis-title{{font:12px var(--sans);fill:var(--ink)}}.legend{{display:flex;gap:18px;flex-wrap:wrap;padding:8px 4px 0;font:11px var(--sans);color:var(--muted)}}.legend span{{display:flex;gap:7px;align-items:center}}.legend i{{width:22px;border-top:2px solid var(--swatch)}}figcaption,footer{{font:12px/1.55 var(--sans);color:var(--faint)}}footer{{border-top:1px solid var(--line);margin-top:55px;padding-top:20px;max-width:960px}}a{{color:var(--teal)}}code{{font-family:var(--mono)}}@media(max-width:760px){{.cards,.rule{{grid-template-columns:1fr 1fr}}.card:nth-child(2){{border-right:0}}}}@media(max-width:520px){{.cards,.rule{{grid-template-columns:1fr}}}}
</style></head><body><main>
<header><div class="eyebrow">Unlevered test - {result["sample"]["start"]} to {result["sample"]["end"]}</div>
<h1>The ladder worked. <em>The harvest did not add return.</em></h1>
<p class="standfirst">Starting 80% SPY and 20% Treasury reduced risk and the quality selections modestly beat an SPY-only deployment ladder. But repeated 5% harvesting lowered terminal wealth, never refilled the reserve, and left the strategy behind SPY.</p>
<div class="cards"><div class="card"><b>{money(primary["terminal_wealth"])}</b><span>strategy terminal wealth; SPY {money(spy["terminal_wealth"])}</span></div>
<div class="card"><b>{pct(primary["xirr"])}</b><span>strategy XIRR; SPY {pct(spy["xirr"])}</span></div>
<div class="card"><b>{pct(primary["max_flow_adjusted_drawdown"])}</b><span>strategy max drawdown; SPY {pct(spy["max_flow_adjusted_drawdown"])}</span></div>
<div class="card"><b>{money(primary["events"]["quality_relative_harvest"]["amount"])}</b><span>118 quality-stock harvest sales</span></div></div></header>

<section><div class="eyebrow">I - Exact tested rule</div><h2>Four rungs consume one Treasury snapshot.</h2>
<p>Begin with {money(result["cash_flows"]["initial"])} and add {money(result["cash_flows"]["monthly_contribution"])} monthly. Every deposit is split 80% SPY / 20% Treasury. At the latest SPY adjusted-total-return high, freeze the available Treasury as the episode budget. Signals use the prior close and trades occur at the next close.</p>
<div class="rule"><div class="step"><b>-10%</b><span>20% of episode Treasury to the point-in-time top seven.</span></div><div class="step"><b>-20%</b><span>30% of episode Treasury to unlevered SPY.</span></div><div class="step"><b>-30%</b><span>30% of episode Treasury to the point-in-time top seven.</span></div><div class="step"><b>-50%</b><span>Final 20% of episode Treasury to unlevered SPY.</span></div></div>
<p>Each quality lot is compared with SPY from its own entry. At quarter-end and at a new rung, every additional 20 percentage points of relative wealth outperformance sells 5% of the lot's original shares. Proceeds rebuild the pre-fall Treasury dollar target; only proceeds above that target buy SPY. When lagged CAPE is at least 35, the reserve target is raised to at least 20% of current NAV.</p>
<p class="note">A stock at +12% while SPY is +10% has only 1.8% relative wealth excess and does not trigger. At SPY +10%, a stock needs +32% to reach the 20% relative hurdle.</p></section>

<section class="wide"><div class="eyebrow">II - Full result</div><h2>Lower risk than SPY, but lower wealth too.</h2>
<figure><div class="figure-head"><strong>Matched monthly contributions</strong><span>log account value</span></div><div class="plate">{growth}<div class="legend">{legend}</div></div></figure>
<figure><div class="figure-head"><strong>Flow-adjusted drawdown</strong><span>cash flows removed</span></div><div class="plate">{drawdown}<div class="legend">{legend}</div></div></figure>
<div class="scroll"><table><thead><tr><th>Path</th><th>Terminal</th><th>XIRR</th><th>TWR CAGR</th><th>Max DD</th><th>Volatility</th><th>Longest underwater</th></tr></thead><tbody>{comparison_rows}</tbody></table></div>
<p class="warning note">The strategy ended {pct(terminal_gap, 1)} behind SPY. It reduced maximum drawdown by {abs(primary["max_flow_adjusted_drawdown"] - spy["max_flow_adjusted_drawdown"]):.1%} and volatility by {abs(primary["annual_volatility"] - spy["annual_volatility"]):.1%}, but its XIRR was {(primary["xirr"] - spy["xirr"])*100:.2f} percentage points lower. This is a smoother path, not an outperforming SPY substitute.</p></section>

<section><div class="eyebrow">III - What produced value?</div><h2>Selection helped slightly; harvesting gave some back.</h2>
<p>The quality strategy finished {pct(selection_gap, 2)} above the otherwise identical SPY-only ladder and {pct(static_gap, 1)} above static contribution-split 80/20. But leaving the quality positions untouched finished at {money(no_harvest["terminal_wealth"])}: repeated harvesting reduced terminal wealth by {pct(harvest_gap, 2)}.</p>
<p class="caution note"><strong>CAPE was inert.</strong> The CAPE-aware and no-CAPE paths were identical. All {money(primary["harvest_to_reserve"])} of quality-sale proceeds were still needed for the ordinary refill target; zero reached SPY and CAPE redirected an additional {money(primary["cape_incremental_reserve"])}.</p></section>

<section class="wide"><div class="eyebrow">IV - Treasury refill</div><h2>The reserve never returned to 20/80.</h2>
<figure><div class="figure-head"><strong>Sleeve weights through time</strong><span>dashed line = 20% Treasury</span></div><div class="plate">{allocation}<div class="legend"><span><i style="--swatch:var(--blue)"></i>Treasury</span><span><i style="--swatch:var(--teal)"></i>Quality</span><span><i style="--swatch:var(--brick)"></i>20% target</span></div></div></figure>
<p>Ending Treasury was {money(primary["ending_treasury"])} ({pct(primary["ending_treasury_weight"])} of NAV), against a nominal refill target of {money(primary["ending_reserve_target"])}. The shortfall was {money(primary["ending_reserve_shortfall"])}. No harvested proceeds crossed the waterfall into SPY.</p>
<p class="warning note">This reveals a rule conflict: “rebuild the old Treasury dollar amount” does not guarantee restoration to 20% of a portfolio that has compounded. If 80/20 is the intended standing policy, it requires an explicit rebalance or a 20%-of-current-NAV target in every valuation regime—not only when CAPE is high.</p></section>

<section class="wide"><div class="eyebrow">V - Clean-data sensitivity</div><h2>The post-2007 sample tells the same story.</h2>
<p>The early share-count history is unavailable, so six quality rungs from 1997-2001 were skipped rather than filled using hindsight. Starting in 2007 removes those skips and gives every quality signal a date-ranked selection.</p>
<div class="scroll"><table><thead><tr><th>2007-2024 path</th><th>Terminal</th><th>XIRR</th><th>Max DD</th><th>Volatility</th></tr></thead><tbody>{sensitivity_rows}</tbody></table></div>
<p>The quality ladder beat its SPY-only ladder by {pct(post_ladder_gap, 1)}, but remained {pct(post_spy_gap, 1)} behind SPY. Harvesting again reduced terminal wealth versus holding the selected companies.</p></section>

<section class="wide"><div class="eyebrow">VI - Historical selections</div><h2>Eight quality deployments were testable.</h2>
<div class="scroll"><table><thead><tr><th>Date</th><th>Action</th><th>Amount</th><th>Point-in-time weights / data issue</th></tr></thead><tbody>{deployment_rows}</tbody></table></div>
<p class="warning note">The ranking uses share counts known with a 90-day reporting lag and raw prices on the signal date, but the candidate union is a fixed set of surviving historical leaders. It omits delisted former giants. The result is date-aware, not survivor-free.</p></section>

<section class="wide"><div class="eyebrow">VII - Harvest and remaining holdings</div><h2>The rule did not fully exit the quality sleeve.</h2>
<div class="scroll"><table><thead><tr><th>Ticker</th><th>Harvest events</th><th>Gross proceeds after sale cost</th></tr></thead><tbody>{harvest_rows}</tbody></table></div>
<div class="scroll"><table><thead><tr><th>Open ticker / lot entries</th><th>Market value</th><th>NAV weight</th><th>Lots</th><th>20% bands</th><th>Original shares sold</th></tr></thead><tbody>{holdings_rows}</tbody></table></div>
<p>Quality stocks still represented {pct(primary["ending_quality_weight"])} of ending NAV and reached {pct(primary["max_quality_weight"])} at maximum. Underperforming lots never trigger a relative-outperformance harvest, while 5%-of-original-share sales require twenty qualifying bands for a complete exit.</p></section>

<section><div class="eyebrow">VIII - Decision</div><h2>Keep the ladder; revise the refill and exit rules.</h2>
<p>The strongest result is the staged deployment itself: it moved the path materially above static 80/20 while keeping drawdown below SPY. The point-in-time quality selection added only a small in-sample increment, and harvesting winners into an underfilled reserve reduced compounding.</p>
<p class="note"><strong>Next defensible test:</strong> preserve the four rungs, compare full 20%-of-current-NAV reserve restoration against the nominal-dollar target, and add a predefined exit for quality lots that never outperform SPY. Those are corrections to incomplete state rules; they should be tested separately rather than folded into this result.</p></section>

<footer>Historical simulation, not investment advice. {money(primary["total_contributed"])} total contributions; adjusted total returns reinvest dividends; Treasury earns prior-known DGS3MO; drawdown and harvest trades cost 5 bp. Taxes, market impact and complete delisting history are omitted. CAPE is lagged one month; its source series ends September 2024. Historical CAPE source: <a href="https://www.econ.yale.edu/~shiller/data.htm">Robert Shiller / Yale</a>. Diversification guidance: <a href="https://www.investor.gov/introduction-investing/getting-started/asset-allocation">Investor.gov</a>. Generated by <code>scripts/run_quality_ladder_harvest.py</code>.</footer>
</main></body></html>'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
