"""Build the self-contained contribution-quality backtest report."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import pandas as pd


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", type=Path,
                        default=Path("out/strategy/contribution_quality"))
    parser.add_argument("--fallback", type=Path,
                        default=Path("out/strategy/contribution_quality_min2"))
    parser.add_argument("--output", type=Path,
                        default=Path("docs/contribution-quality-strategy.html"))
    return parser.parse_args(argv)


def money(value):
    return f"${value:,.0f}"


def pct(value, digits=2):
    return f"{value:.{digits}%}"


def line_chart(frame, columns, colors, *, performance=False):
    width, height = 980, 380
    left, right, top, bottom = 84, 24, 22, 46
    plot_w, plot_h = width - left - right, height - top - bottom
    sampled = frame.set_index("date").resample("ME").last()
    data = {}
    for column in columns:
        values = sampled[column].astype(float).dropna()
        if performance:
            values = values / values.cummax() - 1.0
        data[column] = values
    if performance:
        low, high = math.floor(min(v.min() for v in data.values()) * 10) / 10, 0.0
        ticks = [low + (high - low) * i / 4 for i in range(5)]
        y = lambda value: top + (high - value) / (high - low) * plot_h
        label = lambda value: pct(value, 0)
    else:
        low = min(v[v > 0].min() for v in data.values())
        high = max(v.max() for v in data.values())
        log_low, log_high = math.log10(low), math.log10(high)
        y = lambda value: top + (log_high - math.log10(value)) / (log_high - log_low) * plot_h
        ticks = [10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000,
                 10_000_000, 30_000_000, 100_000_000]
        ticks = [value for value in ticks if low <= value <= high]
        label = money
    start, end = sampled.index[0], sampled.index[-1]
    span = max((end - start).days, 1)
    x = lambda stamp: left + (stamp - start).days / span * plot_w
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Backtest chart">',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" class="frame"/>',
    ]
    for tick in ticks:
        yy = y(tick)
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{yy:.1f}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-9}" y="{yy+4:.1f}" text-anchor="end" class="axis">{label(tick)}</text>')
    for year in range(((start.year + 4) // 5) * 5, end.year + 1, 5):
        xx = x(pd.Timestamp(year, 1, 1))
        parts.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{top}" y2="{height-bottom}" class="grid"/>')
        parts.append(f'<text x="{xx:.1f}" y="{height-17}" text-anchor="middle" class="axis">{year}</text>')
    for column in columns:
        points = " ".join(
            f"{x(stamp):.1f},{y(value):.1f}" for stamp, value in data[column].items()
            if performance or value > 0
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[column]}" stroke-width="2.2" vector-effect="non-scaling-stroke"/>')
    title = "Flow-adjusted drawdown" if performance else "Account value (log scale)"
    parts.append(f'<text x="18" y="195" transform="rotate(-90 18 195)" class="axis-title">{title}</text></svg>')
    return "".join(parts)


def main(argv=None):
    args = parse_args(argv)
    strict = json.loads((args.strict / "results.json").read_text(encoding="utf-8"))
    fallback = json.loads((args.fallback / "results.json").read_text(encoding="utf-8"))
    strict_daily = pd.read_csv(args.strict / "daily.csv", parse_dates=["date"])
    fallback_daily = pd.read_csv(args.fallback / "daily.csv", parse_dates=["date"])
    frame = strict_daily[["date", "strategy_wealth", "strategy_performance_index",
                          "spy_wealth", "spy_performance_index"]].copy()
    frame = frame.rename(columns={
        "strategy_wealth": "strict_wealth",
        "strategy_performance_index": "strict_performance",
    })
    frame["fallback_wealth"] = fallback_daily["strategy_wealth"]
    frame["fallback_performance"] = fallback_daily["strategy_performance_index"]
    colors = {
        "spy_wealth": "var(--ink)", "strict_wealth": "var(--teal)",
        "fallback_wealth": "var(--brick)", "spy_performance_index": "var(--ink)",
        "strict_performance": "var(--teal)", "fallback_performance": "var(--brick)",
    }
    growth = line_chart(frame, ["spy_wealth", "strict_wealth", "fallback_wealth"], colors)
    drawdown = line_chart(frame, ["spy_performance_index", "strict_performance", "fallback_performance"], colors, performance=True)
    legend = ('<span><i style="--swatch:var(--ink)"></i>SPY 1x</span>'
              '<span><i style="--swatch:var(--teal)"></i>Strict: require 5 names</span>'
              '<span><i style="--swatch:var(--brick)"></i>Sensitivity: allow 2 names</span>')

    rows = []
    for label, result, key, note in [
        ("Strict rule", strict, "strategy", "At least five quality names required"),
        ("Two-name sensitivity", fallback, "strategy", "Forces the two available survivors"),
        ("SPY 1x", strict, "spy_1x", "Same deposits; dividends reinvested"),
    ]:
        stats = result[key]
        rows.append(
            f'<tr><td><strong>{html.escape(label)}</strong><small>{html.escape(note)}</small></td>'
            f'<td>{money(stats["terminal_wealth"])}</td><td>{pct(stats["xirr"])}</td>'
            f'<td>{pct(stats["time_weighted_cagr"])}</td>'
            f'<td>{pct(stats["max_flow_adjusted_drawdown"])}</td>'
            f'<td>{pct(stats["annual_volatility"])}</td>'
            f'<td>{stats["longest_underwater_sessions"] / 252:.1f}y</td></tr>'
        )

    rolling_rows = []
    for label, result, prefix in [
        ("Strict rule", strict, "strategy"),
        ("Two-name sensitivity", fallback, "strategy"),
        ("SPY 1x", strict, "spy"),
    ]:
        roll = result["rolling_20y"]
        terminal = roll[f"{prefix}_terminal"]
        xirr_values = roll[f"{prefix}_xirr"]
        drawdown_values = roll[f"{prefix}_max_drawdown"]
        rolling_rows.append(
            f'<tr><td><strong>{html.escape(label)}</strong></td>'
            f'<td>{money(terminal["p10"])}</td><td>{money(terminal["median"])}</td>'
            f'<td>{money(terminal["p90"])}</td><td>{pct(xirr_values["median"])}</td>'
            f'<td>{pct(drawdown_values["median"])}</td></tr>'
        )

    holdings = fallback["open_quality_positions"]
    holding_rows = "".join(
        f'<tr><td><strong>{html.escape(row["ticker"])}</strong><small>{", ".join(row["entry_dates"])}</small></td>'
        f'<td>{money(row["market_value"])}</td><td>{money(row["remaining_cost"])}</td>'
        f'<td>{row["unrealized_multiple"]:.2f}x</td><td>{pct(row["portfolio_weight"])}</td>'
        f'<td>{row["harvests"]}</td></tr>' for row in holdings
    ) or '<tr><td colspan="6">No open quality positions.</td></tr>'
    quality_events = [
        event for event in fallback["events"]
        if event["kind"] in {"deploy_reserve_quality", "quality_guardrail_exit"}
    ]
    event_rows = "".join(
        f'<tr><td>{event["date"]}</td><td>{html.escape(event["kind"].replace("_", " "))}</td>'
        f'<td>{money(event["amount"])}</td><td>{html.escape(event["detail"])}</td></tr>'
        for event in quality_events
    )

    strict_stats, spy_stats, fallback_stats = strict["strategy"], strict["spy_1x"], fallback["strategy"]
    terminal_gap = strict_stats["terminal_wealth"] / spy_stats["terminal_wealth"] - 1
    fallback_gap = fallback_stats["terminal_wealth"] / spy_stats["terminal_wealth"] - 1
    html_text = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Contribution-funded quality rescue strategy</title>
<style>
:root{{--bg:#eef2f0;--paper:#fbfcfb;--ink:#13201d;--muted:#53635f;--faint:#778681;--line:#ccd7d3;--teal:#087f72;--soft:#dcefea;--brick:#b84138;--warn:#f4dedb;--amber:#a97500;--mono:"Cascadia Mono",Consolas,monospace;--serif:Georgia,"Times New Roman",serif;--sans:system-ui,-apple-system,"Segoe UI",sans-serif}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:17px/1.62 var(--serif)}}main{{width:min(1100px,calc(100% - 34px));margin:auto;padding:55px 0 85px}}header{{border-bottom:2px solid var(--ink);padding-bottom:34px}}.eyebrow{{font:700 11px var(--mono);letter-spacing:.13em;text-transform:uppercase;color:var(--teal)}}h1{{font:650 clamp(40px,7vw,72px)/1.02 var(--serif);letter-spacing:-.045em;max-width:14ch;margin:13px 0 20px}}h1 em{{color:var(--teal)}}.standfirst{{font-size:22px;line-height:1.43;color:var(--muted);max-width:62ch}}section{{padding-top:50px;max-width:870px}}section.wide{{max-width:none}}h2{{font:650 31px/1.18 var(--serif);letter-spacing:-.025em;margin:0 0 14px}}p{{margin:0 0 18px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);margin-top:28px}}.card{{background:var(--paper);padding:17px;border-right:1px solid var(--line)}}.card:last-child{{border:0}}.card b{{display:block;font:700 22px var(--mono)}}.card span{{display:block;font:12px/1.4 var(--sans);color:var(--faint);margin-top:7px}}.rule{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:23px 0}}.step{{background:var(--paper);border-top:3px solid var(--teal);padding:15px}}.step b{{display:block;font:700 18px var(--mono)}}.step span{{font:12px/1.45 var(--sans);color:var(--faint)}}.note{{background:var(--soft);border-left:3px solid var(--teal);padding:18px 20px;color:var(--muted)}}.warning{{background:var(--warn);border-left-color:var(--brick)}}.scroll{{overflow:auto;border:1px solid var(--line);background:var(--paper);margin-top:22px}}table{{border-collapse:collapse;width:100%;font:13px/1.4 var(--sans);font-variant-numeric:tabular-nums}}th,td{{padding:12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}}th{{font:700 10px var(--mono);text-transform:uppercase;letter-spacing:.06em;color:var(--faint)}}th:first-child,td:first-child{{text-align:left}}td:first-child{{white-space:normal;min-width:170px}}td strong,td small{{display:block}}td small{{color:var(--faint);margin-top:3px}}figure{{margin:30px 0}}.figure-head{{display:flex;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:9px;font:13px var(--sans);color:var(--muted)}}.plate{{background:var(--paper);border:1px solid var(--line);padding:14px;margin-top:13px;overflow:auto}}svg{{display:block;width:100%;min-width:720px}}.frame{{fill:none;stroke:var(--line)}}.grid{{stroke:var(--line)}}.axis{{font:10px var(--mono);fill:var(--faint)}}.axis-title{{font:12px var(--sans);fill:var(--ink)}}.legend{{display:flex;gap:18px;flex-wrap:wrap;padding:8px 4px 0;font:11px var(--sans);color:var(--muted)}}.legend span{{display:flex;gap:7px;align-items:center}}.legend i{{width:22px;border-top:2px solid var(--swatch)}}figcaption,footer{{font:12px/1.55 var(--sans);color:var(--faint)}}footer{{border-top:1px solid var(--line);margin-top:55px;padding-top:20px;max-width:900px}}code{{font-family:var(--mono)}}
@media(max-width:760px){{.cards,.rule{{grid-template-columns:1fr 1fr}}.card:nth-child(2){{border-right:0}}}}
</style></head><body><main>
<header><div class="eyebrow">Daily total returns · {strict["sample"]["start"]} to {strict["sample"]["end"]}</div>
<h1>Fund the fall. <em>Count the risk.</em></h1>
<p class="standfirst">A cash-flow-matched test of 3x-to-2x-to-1x SPY exposure, an 80/20 SPY–Treasury starting split, staged reserve deployment, and quality-stock purchases at deep portfolio drawdowns.</p>
<div class="cards"><div class="card"><b>{money(strict_stats["terminal_wealth"])}</b><span>strict ending wealth; SPY was {money(spy_stats["terminal_wealth"])}</span></div>
<div class="card"><b>{pct(strict_stats["xirr"])}</b><span>strict money-weighted return; SPY was {pct(spy_stats["xirr"])}</span></div>
<div class="card"><b>{pct(strict_stats["max_flow_adjusted_drawdown"])}</b><span>strict maximum drawdown; SPY was {pct(spy_stats["max_flow_adjusted_drawdown"])}</span></div>
<div class="card"><b>{money(strict_stats["total_contributed"])}</b><span>initial capital plus all matched deposits</span></div></div></header>

<section><div class="eyebrow">I · Exact rulebook</div><h2>Deposits do not reset the drawdown clock.</h2>
<p>Start with {money(strict["cash_flows"]["initial"])} and allocate 80% to the financed SPY sleeve and 20% to Treasury. Add {money(strict["cash_flows"]["annual"])} at the first close of every later calendar year and another {money(strict["cash_flows"]["additional_every_third_year"])} in contribution years 3, 6, 9, and so on. Each deposit is split 80/20.</p>
<div class="rule"><div class="step"><b>−10%</b><span>Deploy 10% of episode-start Treasury into unlevered SPY.</span></div><div class="step"><b>−20%</b><span>Deploy another 20% into unlevered SPY.</span></div><div class="step"><b>−30%</b><span>Deploy another 30% into unlevered SPY.</span></div><div class="step"><b>−40%</b><span>Deploy remaining Treasury into 5–7 qualifying quality stocks.</span></div></div>
<p>Core leverage begins at 3x, falls to 2x at a 15% flow-adjusted NAV drawdown and to 1x at 30%. From 1x it returns to 2x only above −10%; 3x returns only after NAV fully recovers. Ten percent of positive financed SPY-sleeve P&amp;L is swept to Treasury quarterly.</p>
<p>At −40%, candidates need five years of history, positive point-in-time earnings, five-year total-return CAGR above SPY, and a drawdown at least five percentage points deeper than SPY. After SPY regains its pre-crisis high, a profitable stock can harvest 1% of current shares at quarter boundaries when its price is above the prior harvest price.</p>
<p class="note">Financing is prior-known 3-month Treasury yield plus 1%; reserve earns the prior-known Treasury yield; stock trades cost 5 bp. Adjusted-close returns reinvest dividends. Taxes, margin calls, ETF tracking error, market impact, and intraday gaps are omitted.</p></section>

<section class="wide"><div class="eyebrow">II · Full history</div><h2>The strict strategy did not beat SPY in ending wealth.</h2>
<p>The strict result ended {pct(terminal_gap)} behind matched SPY despite a {pct(strict_stats["time_weighted_cagr"] - spy_stats["time_weighted_cagr"])} higher time-weighted CAGR. Deposit timing matters: XIRR and terminal wealth are the relevant investor outcomes, and both favored SPY over the full path.</p>
<figure><div class="figure-head"><strong>Account growth with identical deposits</strong><span>log scale</span></div><div class="plate">{growth}<div class="legend">{legend}</div></div><figcaption>The sensitivity line is not the stated five-name strategy. It allows the two survivor-selected names available at each quality trigger.</figcaption></figure>
<figure><div class="figure-head"><strong>Unitised drawdown</strong><span>cash flows removed</span></div><div class="plate">{drawdown}<div class="legend">{legend}</div></div><figcaption>External deposits cannot erase a loss in this chart. All paths reached their worst drawdown on 9 March 2009.</figcaption></figure>
<div class="scroll"><table><thead><tr><th>Path</th><th>Ending wealth</th><th>XIRR</th><th>TWR CAGR</th><th>Max DD</th><th>Volatility</th><th>Longest underwater</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>

<section class="wide"><div class="eyebrow">III · Rolling 20-year histories</div><h2>The median was better; the tail risk was not.</h2>
<p>Across {strict["rolling_20y"]["cohorts"]} overlapping annual-start 20-year windows, the strict strategy beat matched SPY in {pct(strict["rolling_20y"]["strategy_beats_spy_terminal_share"], 0)} of cohorts. These are historical scenarios, not independent observations or a forecast.</p>
<div class="scroll"><table><thead><tr><th>Path</th><th>P10 terminal</th><th>Median terminal</th><th>P90 terminal</th><th>Median XIRR</th><th>Median max DD</th></tr></thead><tbody>{''.join(rolling_rows)}</tbody></table></div></section>

<section><div class="eyebrow">IV · The quality gate</div><h2>The five-name rule never fired.</h2>
<p>The portfolio crossed the −40% rung on 23 March 2001 and 17 March 2020. Exactly two names passed on each date, below the required minimum of five, so strict execution left the remaining Treasury parked.</p>
<p class="warning note"><strong>Sensitivity only:</strong> relaxing the minimum to two raised ending wealth to {money(fallback_stats["terminal_wealth"])} ({pct(fallback_gap)} versus SPY), but it also deepened maximum drawdown to {pct(fallback_stats["max_flow_adjusted_drawdown"])}. The 2020 picks were NVDA and META. Because the universe is today's surviving compounders, this uplift is highly exposed to survivorship and selection bias.</p>
<div class="scroll"><table><thead><tr><th>Date</th><th>Action</th><th>Amount</th><th>Detail</th></tr></thead><tbody>{event_rows}</tbody></table></div></section>

<section class="wide"><div class="eyebrow">V · Sensitivity holdings at the end</div><h2>One survivor explains much of the spread.</h2>
<div class="scroll"><table><thead><tr><th>Ticker / entry</th><th>Market value</th><th>Remaining cost</th><th>Multiple</th><th>Weight</th><th>Harvests</th></tr></thead><tbody>{holding_rows}</tbody></table></div>
<p>NVDA alone was {pct(holdings[0]["portfolio_weight"] if holdings else 0)} of final sensitivity wealth. That concentration is why the two-name line must not be treated as an expected-return estimate.</p></section>

<footer>Historical simulation, not investment advice. Data: Yahoo adjusted-close total returns and FRED DGS3MO. Quality candidates are a fixed present-day basket; delisted names and historical index membership are absent. Generated by <code>scripts/run_contribution_quality_strategy.py</code> and <code>scripts/build_contribution_quality_page.py</code>.</footer>
</main></body></html>'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
