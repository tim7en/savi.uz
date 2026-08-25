"""Build the self-contained QQQ DCA guard backtest report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd

from build_contribution_quality_page import line_chart, money, pct


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("out/strategy/qqq_dca_backtest"))
    parser.add_argument("--output", type=Path, default=Path("docs/qqq-dca-backtest.html"))
    return parser.parse_args(argv)


def exposure_chart(frame: pd.DataFrame, column: str) -> str:
    width, height = 980, 260
    left, right, top, bottom = 60, 24, 20, 42
    sampled = frame.set_index("date")[column].resample("ME").last().dropna()
    high = max(2.25, float(sampled.max()) * 1.04)
    xspan = max((sampled.index[-1] - sampled.index[0]).days, 1)
    x = lambda stamp: left + (stamp - sampled.index[0]).days / xspan * (width-left-right)
    y = lambda value: top + (high-value) / high * (height-top-bottom)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Gross exposure history">']
    for tick in (0, 1, 2):
        yy = y(tick)
        parts.append(f'<line class="grid" x1="{left}" x2="{width-right}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        parts.append(f'<text class="axis" x="{left-8}" y="{yy+4:.1f}" text-anchor="end">{tick}×</text>')
    for year in range(2000, 2025, 5):
        stamp = pd.Timestamp(year, 1, 1)
        if sampled.index[0] <= stamp <= sampled.index[-1]:
            xx = x(stamp)
            parts.append(f'<text class="axis" x="{xx:.1f}" y="{height-14}" text-anchor="middle">{year}</text>')
    points = " ".join(f"{x(stamp):.1f},{y(value):.1f}" for stamp, value in sampled.items())
    parts.append(f'<polyline points="{points}" fill="none" stroke="var(--teal)" stroke-width="2.3"/>')
    parts.append('</svg>')
    return "".join(parts)


def days_label(value):
    return "not recovered" if value is None else f"{value / 365.2425:.1f}y"


def main(argv=None) -> int:
    args = parse_args(argv)
    result = json.loads((args.input / "results.json").read_text(encoding="utf-8"))
    daily = pd.read_csv(args.input / "daily.csv", parse_dates=["date"])
    variants = result["variants"]
    primary = variants["dashboard_quality_annual"]
    qqq_only = variants["dashboard_qqq_only_annual"]
    no_leverage = variants["quality_ladder_no_leverage"]
    qqq = variants["qqq_1x"]
    three = variants["qqq_3x_theoretical"]
    monthly = variants["dashboard_quality_monthly"]

    frame = daily[[
        "date",
        "dashboard_quality_annual_wealth", "dashboard_quality_annual_performance_index",
        "quality_ladder_no_leverage_wealth", "quality_ladder_no_leverage_performance_index",
        "qqq_1x_wealth", "qqq_1x_performance_index",
        "dashboard_quality_annual_gross_exposure",
    ]].copy()
    colors = {
        "dashboard_quality_annual_wealth": "var(--teal)",
        "dashboard_quality_annual_performance_index": "var(--teal)",
        "quality_ladder_no_leverage_wealth": "var(--amber)",
        "quality_ladder_no_leverage_performance_index": "var(--amber)",
        "qqq_1x_wealth": "var(--ink)",
        "qqq_1x_performance_index": "var(--ink)",
    }
    growth = line_chart(
        frame,
        ["qqq_1x_wealth", "dashboard_quality_annual_wealth", "quality_ladder_no_leverage_wealth"],
        colors,
    )
    drawdown = line_chart(
        frame,
        ["qqq_1x_performance_index", "dashboard_quality_annual_performance_index", "quality_ladder_no_leverage_performance_index"],
        colors,
        performance=True,
    )
    exposure = exposure_chart(frame, "dashboard_quality_annual_gross_exposure")
    legend = ('<span><i style="--swatch:var(--ink)"></i>QQQ 1×</span>'
              '<span><i style="--swatch:var(--teal)"></i>Dashboard guard</span>'
              '<span><i style="--swatch:var(--amber)"></i>Same ladder, no leverage</span>')

    comparison = [
        ("dashboard_quality_annual", "Dashboard guard", "annual contribution; QQQ + quality Treasury ladder"),
        ("dashboard_qqq_only_annual", "QQQ-only guard", "all Treasury rungs buy QQQ"),
        ("quality_ladder_no_leverage", "No-leverage quality ladder", "same Treasury ladder; every equity sleeve at 1×"),
        ("qqq_ladder_no_leverage", "No-leverage QQQ ladder", "all Treasury rungs buy QQQ at 1×"),
        ("static_qqq_treasury", "Static contribution split", "initial QQQ; later 80/20, Treasury never deployed"),
        ("qqq_1x", "QQQ 1×", "all capital invested immediately; matched annual cash flows"),
    ]
    comparison_rows = "".join(
        f'<tr class="{"featured" if key == "dashboard_quality_annual" else ""}"><td><strong>{html.escape(label)}</strong><small>{html.escape(note)}</small></td>'
        f'<td>{money(variants[key]["terminal_wealth"])}</td><td>{pct(variants[key]["xirr"])}</td>'
        f'<td>{pct(variants[key]["max_flow_adjusted_drawdown"])}</td><td>{pct(variants[key]["annual_volatility"])}</td>'
        f'<td>{variants[key]["longest_underwater_sessions"] / 252:.1f}y</td>'
        f'<td>{variants[key]["max_gross_exposure"]:.2f}×</td></tr>'
        for key, label, note in comparison
    )

    dotcom_rows = "".join(
        f'<tr><td><strong>{label}</strong></td><td>{row["dotcom"]["peak_date"]}</td>'
        f'<td>{row["dotcom"]["trough_date"]}</td><td>{pct(row["dotcom"]["max_drawdown"], 1)}</td>'
        f'<td>{row["dotcom"]["recovery_date"] or "not recovered"}</td>'
        f'<td>{days_label(row["dotcom"]["calendar_days_to_recovery"])}</td></tr>'
        for label, row in [
            ("Dashboard guard", primary), ("No-leverage quality ladder", no_leverage),
            ("QQQ 1×", qqq), ("QQQ theoretical 3×", three),
        ]
    )

    cohorts = result["rolling_10y"]
    cohort_wins = sum(row["terminal_ratio"] > 1.0 for row in cohorts)
    sorted_ratios = sorted(row["terminal_ratio"] for row in cohorts)
    median_ratio = sorted_ratios[len(sorted_ratios) // 2]
    cohort_rows = "".join(
        f'<tr class="{"win" if row["terminal_ratio"] > 1 else "loss"}"><td>{row["start"][:4]}</td>'
        f'<td>{money(row["strategy_terminal"])}</td><td>{money(row["qqq_terminal"])}</td>'
        f'<td>{row["terminal_ratio"]:.3f}×</td>'
        f'<td>{(row["strategy_xirr"] - row["qqq_xirr"]) * 100:+.2f} pp</td>'
        f'<td>{pct(row["strategy_max_drawdown"], 1)}</td><td>{pct(row["qqq_max_drawdown"], 1)}</td></tr>'
        for row in cohorts
    )

    holding_rows = "".join(
        f'<tr><td><strong>{row["ticker"]}</strong><small>{row["lots"]} open lots</small></td>'
        f'<td>{money(row["market_value"])}</td><td>{pct(row["portfolio_weight"])}</td>'
        f'<td>{row["harvest_bands"]}</td><td>{pct(row["harvested_original_share"], 1)}</td></tr>'
        for row in result["holdings"]["dashboard_quality_annual"]
    )

    terminal_edge = primary["terminal_wealth"] / qqq["terminal_wealth"] - 1.0
    no_leverage_edge = primary["terminal_wealth"] / no_leverage["terminal_wealth"] - 1.0
    qqq_only_edge = qqq_only["terminal_wealth"] / primary["terminal_wealth"] - 1.0

    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>QQQ DCA guard backtest</title>
<style>
:root{{--bg:#edf1ed;--paper:#fbfcf9;--ink:#182321;--muted:#5c6c68;--faint:#7d8a87;--line:#cad4cf;--teal:#118578;--soft:#dceee9;--amber:#a46c00;--amberSoft:#f5ead1;--red:#b3483e;--redSoft:#f5dfdc;--mono:"Cascadia Mono",Consolas,monospace;--serif:Georgia,"Times New Roman",serif;--sans:Inter,system-ui,-apple-system,"Segoe UI",sans-serif}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.58 var(--sans)}}main{{width:min(1160px,calc(100% - 32px));margin:auto;padding:46px 0 76px}}header{{border-bottom:2px solid var(--ink);padding-bottom:30px}}.eyebrow{{font:700 11px var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--teal)}}h1{{font:600 clamp(42px,7vw,76px)/1 var(--serif);letter-spacing:-.05em;max-width:14ch;margin:12px 0 18px}}h2{{font:600 31px/1.17 var(--serif);letter-spacing:-.025em;margin:0 0 12px}}h3{{font:600 21px/1.25 var(--serif);margin:25px 0 9px}}.standfirst{{max-width:72ch;color:var(--muted);font:21px/1.45 var(--serif)}}section{{margin-top:48px}}p{{margin:0 0 16px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}}.card{{background:var(--paper);border-top:4px solid var(--teal);padding:18px}}.card.warn{{border-color:var(--amber)}}.card.stop{{border-color:var(--red)}}.card b{{display:block;font:700 24px var(--mono)}}.card span{{display:block;color:var(--muted);font-size:12px;margin-top:6px}}.note{{background:var(--soft);border-left:3px solid var(--teal);padding:17px 19px;color:var(--muted)}}.caution{{background:var(--amberSoft);border-left-color:var(--amber)}}.warning{{background:var(--redSoft);border-left-color:var(--red)}}.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}figure{{margin:22px 0}}.figureHead{{display:flex;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:8px;color:var(--muted);font-size:12px}}.plate{{background:var(--paper);border:1px solid var(--line);padding:12px;overflow:auto;margin-top:10px}}svg{{display:block;width:100%;min-width:720px}}.frame{{fill:none;stroke:var(--line)}}.grid{{stroke:var(--line)}}.axis{{font:10px var(--mono);fill:var(--faint)}}.axis-title{{font:12px var(--sans);fill:var(--ink)}}.legend{{display:flex;gap:20px;flex-wrap:wrap;padding:9px 3px 0;color:var(--muted);font-size:11px}}.legend span{{display:flex;align-items:center;gap:7px}}.legend i{{width:22px;border-top:2px solid var(--swatch)}}.scroll{{overflow:auto;border:1px solid var(--line);background:var(--paper)}}table{{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}}th,td{{padding:11px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}}th{{font:700 10px var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--faint)}}th:first-child,td:first-child{{text-align:left}}td:first-child{{white-space:normal;min-width:150px}}td strong,td small{{display:block}}td small{{color:var(--faint)}}tr.featured{{background:var(--soft)}}tr.loss{{background:#fff8f4}}.rules{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}.rule{{background:var(--paper);border-top:3px solid var(--teal);padding:15px}}.rule b{{display:block;font:700 17px var(--mono)}}.rule span{{display:block;color:var(--muted);font-size:12px;margin-top:5px}}footer{{border-top:1px solid var(--line);padding-top:18px;margin-top:50px;color:var(--faint);font-size:11px}}a{{color:var(--teal)}}code{{font-family:var(--mono)}}@media(max-width:760px){{.cards,.rules{{grid-template-columns:1fr 1fr}}.grid2{{grid-template-columns:1fr}}}}@media(max-width:480px){{.cards,.rules{{grid-template-columns:1fr}}}}
</style></head><body><main>
<header><div class="eyebrow">QQQ inception test · {result['sample']['start']} to {result['sample']['end']}</div><h1>Better ending wealth. Nearly the same crash.</h1><p class="standfirst">The guarded contribution rule beat QQQ buy-and-hold from QQQ’s 1999 launch, but the dot-com collapse still erased more than four fifths of flow-adjusted NAV. The result rewards staying solvent and contributing—not avoiding the bear market.</p></header>
<div class="cards"><article class="card"><b>{money(primary['terminal_wealth'])}</b><span>dashboard guard; QQQ 1× {money(qqq['terminal_wealth'])}</span></article><article class="card"><b>{pct(primary['xirr'])}</b><span>XIRR; QQQ 1× {pct(qqq['xirr'])}</span></article><article class="card stop"><b>{pct(primary['max_flow_adjusted_drawdown'],1)}</b><span>maximum drawdown; QQQ 1× {pct(qqq['max_flow_adjusted_drawdown'],1)}</span></article><article class="card warn"><b>{primary['max_gross_exposure']:.2f}×</b><span>maximum account gross exposure; mean {primary['mean_gross_exposure']:.2f}×</span></article></div>

<section><div class="eyebrow">I · Result</div><h2>The edge came from drawdown contributions, not protection.</h2><p>With {money(result['cash_flows']['annual_total_contributed'])} total contributed, the dashboard guard finished {pct(terminal_edge,1)} above QQQ 1× and added {(primary['xirr']-qqq['xirr'])*100:.2f} percentage points of XIRR. Maximum drawdown improved by only {(primary['max_flow_adjusted_drawdown']-qqq['max_flow_adjusted_drawdown'])*100:.2f} points, while annual volatility rose from {pct(qqq['annual_volatility'],1)} to {pct(primary['annual_volatility'],1)}.</p><p>The contribution-leverage overlay accounted for a {pct(no_leverage_edge,1)} terminal advantage over the otherwise identical no-leverage quality ladder. It used leverage on {primary['leveraged_contributions']} annual contribution dates, changed VXN-controlled lot leverage {primary['vxn_leverage_changes']} times, and paid modeled financing of {money(primary['financing_cost'])}. Switching costs were not charged.</p><p class="warning note"><strong>Interpretation:</strong> an 81% drawdown requires a 437% gain merely to recover. This strategy may improve the use of new cash, but it does not make concentrated technology exposure defensive.</p></section>

<section><div class="eyebrow">II · Equity and drawdown</div><h2>Contributions after the crash dominate the ending value.</h2><figure><div class="figureHead"><strong>Account value with matched annual cash flows</strong><span>log scale</span></div><div class="plate">{growth}<div class="legend">{legend}</div></div></figure><figure><div class="figureHead"><strong>Flow-adjusted drawdown</strong><span>cash contributions removed from the performance clock</span></div><div class="plate">{drawdown}<div class="legend">{legend}</div></div></figure><figure><div class="figureHead"><strong>Dashboard guard gross exposure</strong><span>core remains 1×; temporary contribution lots lift account exposure</span></div><div class="plate">{exposure}</div></figure></section>

<section><div class="eyebrow">III · Comparison</div><h2>The quality selector did not improve the full-sample result.</h2><div class="scroll"><table><thead><tr><th>Path</th><th>Terminal</th><th>XIRR</th><th>Max DD</th><th>Volatility</th><th>Longest underwater</th><th>Max gross</th></tr></thead><tbody>{comparison_rows}</tbody></table></div><p class="caution note" style="margin-top:15px">The QQQ-only guard finished {pct(qqq_only_edge,1)} above the mixed quality version. Early dot-com quality rungs were skipped because point-in-time share-count data could not support seven names; no historical leaders were fabricated. The quality result is survivor-biased after coverage begins, so the cleaner evidence is the QQQ-only ladder and the leverage/no-leverage comparison.</p></section>

<section><div class="eyebrow">IV · Dot-com survival test</div><h2>Twelve years underwater—even after the guard.</h2><div class="scroll"><table><thead><tr><th>Path</th><th>Peak</th><th>Trough</th><th>Drawdown</th><th>Recovered</th><th>Peak-to-recovery</th></tr></thead><tbody>{dotcom_rows}</tbody></table></div><p>The dashboard guard recovered its March 2000 flow-adjusted high in March 2012, about three years earlier than QQQ 1×. The theoretical constant-3× path lost {pct(abs(three['max_flow_adjusted_drawdown']),2)} at its worst, never recovered its old performance high by September 2024, remained {pct(abs(three['current_drawdown']),1)} below that high, and accumulated {money(three['financing_cost'])} of modeled financing.</p><p class="warning note"><strong>Why the 3× terminal value is deceptive:</strong> it still ended at {money(three['terminal_wealth'])} because repeated outside contributions arrived after the near-wipeout and then participated in the long technology bull market. Its time-weighted CAGR from the original performance path was only {pct(three['time_weighted_cagr'],2)}. A large dollar endpoint does not repair a 99.98% historical loss for the original capital.</p></section>

<section><div class="eyebrow">V · Start-date robustness</div><h2>The full-sample advantage is concentrated in early cohorts.</h2><p>Across {len(cohorts)} independent ten-year starts, the strategy beat QQQ in {cohort_wins}; median terminal wealth was only {pct(median_ratio-1,1)} above QQQ. The best result began into the dot-com collapse. Most 2009–2013 starts lagged QQQ because Treasury and risk caps diluted a strong bull market.</p><div class="scroll"><table><thead><tr><th>Start</th><th>Strategy terminal</th><th>QQQ terminal</th><th>Ratio</th><th>XIRR edge</th><th>Strategy DD</th><th>QQQ DD</th></tr></thead><tbody>{cohort_rows}</tbody></table></div></section>

<section><div class="eyebrow">VI · Exact rules tested</div><h2>Core, contribution, and Treasury are separate sleeves.</h2><div class="rules"><article class="rule"><b>Initial core</b><span>$10,000 fully in QQQ at 1×. It is never changed to 3× at a high.</span></article><article class="rule"><b>New cash</b><span>$10,000 annually plus $30,000 every third year; 80% QQQ and 20% Treasury.</span></article><article class="rule"><b>NAV gate</b><span>Fresh QQQ cash: 1× above −10% NAV DD, max 2× after −10%, max 3× after −20%.</span></article><article class="rule"><b>CAPE ceiling</b><span>Lagged broad-market CAPE: 3× below 25, 2× at 25–35, 1× above 35.</span></article><article class="rule"><b>VXN ceiling</b><span>60-day VXN rank: 3× below 70%, 2× at 70–90%, 1× above 90%; missing history means 1×.</span></article><article class="rule"><b>Recovery</b><span>Leveraged contribution lots return to 1× when flow-adjusted NAV regains its high.</span></article></div><h3>Treasury ladder</h3><p>At QQQ drawdowns, deploy 20% of episode Treasury to the quality basket at −10%, 30% to unlevered QQQ at −20%, another 30% to quality at −30%, and the final 20% to unlevered QQQ at −50%. Treasury earns prior-known DGS3MO. Borrowed exposure pays DGS3MO +1%.</p></section>

<section><div class="eyebrow">VII · Ending quality holdings</div><h2>Small satellite positions—not the source of the main edge.</h2><div class="scroll"><table><thead><tr><th>Ticker</th><th>Market value</th><th>Weight</th><th>Harvest bands</th><th>Original shares sold</th></tr></thead><tbody>{holding_rows}</tbody></table></div><p>The entire ending quality sleeve was {money(primary['ending_quality'])}, or {pct(primary['ending_quality']/primary['terminal_wealth'],1)} of wealth. Treasury ended at only {pct(primary['ending_treasury_weight'],1)} because the policy deploys reserve during drawdowns and does not constantly rebalance back to 20%.</p></section>

<section><div class="eyebrow">VIII · Decision</div><h2>Promising as a contribution rule; unacceptable as unrestricted leverage.</h2><p>The backtest supports a narrow claim: when a long-lived investor keeps contributing through severe QQQ drawdowns, carefully capped leverage on fresh capital can improve terminal wealth. It does not support putting the original core at 3×, borrowing against a home, or expecting shallow drawdowns.</p><p class="note"><strong>Next prudent test:</strong> freeze VXN leverage at entry instead of changing it daily, charge 5–10 bp for every leverage change, cap total account gross exposure at 1.5× and 1.75×, and compare against a simple QQQ/short-Treasury rebalance. The 15-cohort result is too mixed to call the current edge robust.</p></section>

<footer>Historical simulation, not investment advice. QQQ adjusted total returns and raw prices are from Yahoo; QQQ’s official fund description is at <a href="https://www.invesco.com/qqq-etf/en/home.html">Invesco</a>. Volatility data: <a href="https://fred.stlouisfed.org/series/VXNCLS">FRED VXNCLS</a>. Treasury/funding reference: <a href="https://fred.stlouisfed.org/series/DGS3MO">FRED DGS3MO</a>. CAPE: <a href="https://www.econ.yale.edu/~shiller/data.htm">Robert Shiller / Yale</a>. Leveraged products can compound differently from their stated daily multiple: <a href="https://www.finra.org/investors/insights/lowdown-leveraged-and-inverse-exchange-traded-products">FINRA</a>. Quality selection is not survivor-free; taxes, margin calls, switching costs and implementation slippage are omitted. Generated by <code>scripts/run_qqq_dca_backtest.py</code>.</footer>
</main></body></html>'''
    # Keep the page builder independent of private helpers in older reports.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
