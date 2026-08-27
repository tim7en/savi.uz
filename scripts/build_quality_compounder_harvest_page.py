"""Build the self-contained quality-harvest backtest report."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("out/strategy/quality_compounder_harvest"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("docs/quality-compounder-harvest.html"),
    )
    return parser.parse_args(argv)


def pct(value, digits=2):
    return f"{value:.{digits}%}"


def money(value):
    return f"${value:,.0f}"


def line_chart(frame, columns, colors, *, drawdown=False):
    width, height = 960, 370
    left, right, top, bottom = 82, 22, 22, 45
    plot_w, plot_h = width - left - right, height - top - bottom
    sampled = frame.set_index("date").resample("ME").last().dropna(how="all")
    data = {}
    for column in columns:
        values = sampled[column].astype(float)
        if drawdown:
            values = values / values.cummax() - 1.0
        data[column] = values
    if drawdown:
        low, high = min(series.min() for series in data.values()), 0.0
        low = math.floor(low * 10.0) / 10.0
        ticks = [low + (high - low) * i / 4 for i in range(5)]
        y = lambda value: top + (high - value) / (high - low) * plot_h
        tick_label = lambda value: pct(value, 0)
    else:
        positives = [value for series in data.values() for value in series if value > 0]
        low, high = min(positives), max(positives)
        log_low, log_high = math.log10(low), math.log10(high)
        y = lambda value: top + (log_high - math.log10(value)) / (log_high - log_low) * plot_h
        candidates = [100_000, 250_000, 500_000, 1_000_000, 2_500_000, 5_000_000]
        ticks = [value for value in candidates if low <= value <= high]
        tick_label = lambda value: money(value)
    start, end = sampled.index[0], sampled.index[-1]
    span = max((end - start).days, 1)
    x = lambda stamp: left + (stamp - start).days / span * plot_w
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape("Drawdown comparison" if drawdown else "Growth comparison")}">',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" class="frame"/>',
    ]
    for tick in ticks:
        yy = y(tick)
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{yy:.1f}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" class="axis">{tick_label(tick)}</text>')
    for year in range(((start.year + 4) // 5) * 5, end.year + 1, 5):
        stamp = pd.Timestamp(year=year, month=1, day=1)
        xx = x(stamp)
        parts.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{top}" y2="{height-bottom}" class="grid"/>')
        parts.append(f'<text x="{xx:.1f}" y="{height-17}" text-anchor="middle" class="axis">{year}</text>')
    for column in columns:
        series = data[column].dropna()
        points = " ".join(
            f"{x(stamp):.1f},{y(value):.1f}" for stamp, value in series.items()
            if drawdown or value > 0
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{colors[column]}" '
            'stroke-width="2" vector-effect="non-scaling-stroke"/>'
        )
    parts.append('<text x="18" y="190" transform="rotate(-90 18 190)" class="axis-title">' +
                 ('Drawdown from prior high' if drawdown else 'Portfolio value (log scale)') + '</text>')
    parts.append('<text x="500" y="366" text-anchor="middle" class="axis-title">Date</text></svg>')
    return "".join(parts)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = json.loads((args.input / "results.json").read_text(encoding="utf-8"))
    daily = pd.read_csv(args.input / "daily.csv", parse_dates=["date"])
    baseline_name = "harvest_5_step_20"
    active_best_name = result["ranking"][0]["variant"]
    full_best_name = max(
        result["full_period"],
        key=lambda name: result["full_period"][name]["cagr"],
    )
    baseline = result["full_period"][baseline_name]
    spy = result["benchmark"]["spy_1x"]
    active_best = result["full_period"][active_best_name]
    full_best = result["full_period"][full_best_name]

    labels = {
        "spy_1x": "SPY 1x",
        baseline_name: "5% / +20% baseline",
        active_best_name: active_best_name.replace("harvest_", "").replace("_step_", "% / +") + "%",
    }
    columns = list(dict.fromkeys(["spy_1x", baseline_name, active_best_name]))
    colors = {
        "spy_1x": "var(--ink)",
        baseline_name: "var(--teal)",
        active_best_name: "var(--brick)",
    }
    if active_best_name == baseline_name:
        colors[baseline_name] = "var(--teal)"
    legend = "".join(
        f'<span><i style="--swatch:{colors[column]}"></i>{html.escape(labels[column])}</span>'
        for column in columns
    )
    growth_svg = line_chart(daily, columns, colors)
    drawdown_svg = line_chart(daily, columns, colors, drawdown=True)

    ranking_rows = []
    for row in result["ranking"]:
        stats = result["full_period"][row["variant"]]
        event = stats["events"].get("quality_profit_harvest", {"events": 0, "amount": 0})
        ranking_rows.append(
            "<tr>"
            f'<td><strong>{html.escape(row["variant"].replace("harvest_", "").replace("_step_", "% at +"))}%</strong></td>'
            f'<td>{pct(stats["cagr"])}</td><td>{pct(stats["max_drawdown"])}</td>'
            f'<td>{pct(row["quality_active_median_excess_cagr"])}</td>'
            f'<td>{pct(row["quality_active_beats_spy_share"], 0)}</td>'
            f'<td>{event["events"]:,}</td><td>{money(event["amount"])}</td></tr>'
        )

    positions = baseline["quality_lots"]["open_positions"]
    position_rows = "".join(
        "<tr>"
        f'<td><strong>{html.escape(row["ticker"])}</strong><small>{", ".join(row["entry_dates"])}</small></td>'
        f'<td>{money(row["market_value"])}</td>'
        f'<td>{money(row["remaining_allocated_cost"])}</td>'
        f'<td>{row["unrealized_multiple"]:.2f}x</td>'
        f'<td>{pct(row["portfolio_weight"])}</td></tr>'
        for row in positions
    )

    html_text = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Point-in-time implementation and backtest of a quality-stock profit-harvest and compounder guardrail policy.">
<title>Quality Harvest and Compounder Guardrail</title>
<style>
:root{{--bg:#eef2f0;--paper:#fbfcfb;--ink:#12201d;--muted:#52625e;--faint:#768681;--line:#ccd7d3;--teal:#087f72;--teal2:#dcefea;--brick:#b84138;--brick2:#f4dedb;--amber:#a97500;--mono:"Cascadia Mono",Consolas,monospace;--serif:Georgia,"Times New Roman",serif;--sans:system-ui,-apple-system,"Segoe UI",sans-serif}}
@media(prefers-color-scheme:dark){{:root{{--bg:#101715;--paper:#17201e;--ink:#e7eeeb;--muted:#bcc9c5;--faint:#8b9995;--line:#2c3935;--teal:#4ab7a8;--teal2:#15302b;--brick:#e17468;--brick2:#3a211f;--amber:#efc64b}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:17px/1.65 var(--serif)}}main{{width:min(1080px,calc(100% - 34px));margin:auto;padding:56px 0 90px}}header{{max-width:930px;border-bottom:2px solid var(--ink);padding-bottom:34px}}.eyebrow{{font:700 11px/1.3 var(--mono);letter-spacing:.13em;text-transform:uppercase;color:var(--teal)}}h1{{font:650 clamp(42px,7vw,76px)/1 var(--serif);letter-spacing:-.04em;margin:13px 0 20px;max-width:13ch}}h1 em{{color:var(--teal)}}.standfirst{{font-size:22px;line-height:1.45;color:var(--muted);max-width:54ch;margin:0}}section{{padding-top:52px;max-width:850px}}section.wide{{max-width:none}}h2{{font:650 31px/1.15 var(--serif);letter-spacing:-.02em;margin:0 0 13px}}p{{margin:0 0 18px}}.lead{{font-size:19px;color:var(--muted);max-width:56ch}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);margin-top:30px}}.card{{background:var(--paper);padding:17px;border-right:1px solid var(--line)}}.card:last-child{{border:0}}.card b{{display:block;font:700 23px/1.1 var(--mono)}}.card span{{display:block;font:12px/1.4 var(--sans);color:var(--faint);margin-top:8px}}.rule{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:24px 0}}.step{{background:var(--paper);border-top:3px solid var(--teal);padding:16px}}.step:nth-child(3){{border-color:var(--amber)}}.step:nth-child(4){{border-color:var(--brick)}}.step b{{display:block;font:700 18px var(--mono)}}.step span{{font:12px/1.45 var(--sans);color:var(--faint)}}.note{{background:var(--teal2);border-left:3px solid var(--teal);padding:18px 20px;color:var(--muted)}}.warning{{background:var(--brick2);border-left-color:var(--brick)}}.scroll{{overflow:auto;border:1px solid var(--line);background:var(--paper);margin-top:24px}}table{{border-collapse:collapse;width:100%;font:13px/1.4 var(--sans);font-variant-numeric:tabular-nums}}th,td{{padding:12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}}th{{font:700 10px var(--mono);letter-spacing:.07em;text-transform:uppercase;color:var(--faint)}}th:first-child,td:first-child{{text-align:left}}td:first-child{{white-space:normal;min-width:180px}}td strong,td small{{display:block}}td small{{color:var(--faint);margin-top:3px}}tr:last-child td{{border-bottom:0}}figure{{margin:32px 0}}.figure-head{{display:flex;justify-content:space-between;gap:18px;padding-bottom:9px;border-bottom:1px solid var(--line);font:13px var(--sans);color:var(--muted)}}.figure-head span{{font:10px var(--mono);text-transform:uppercase;letter-spacing:.08em;color:var(--faint)}}.plate{{background:var(--paper);border:1px solid var(--line);padding:14px;margin-top:14px;overflow:auto}}svg{{display:block;width:100%;min-width:720px;height:auto}}.frame{{fill:none;stroke:var(--line)}}.grid{{stroke:var(--line);stroke-width:1}}.axis{{font:10px var(--mono);fill:var(--faint)}}.axis-title{{font:12px var(--sans);fill:var(--ink)}}.legend{{display:flex;gap:18px;flex-wrap:wrap;padding:8px 4px 0;font:11px var(--sans);color:var(--muted)}}.legend span{{display:flex;align-items:center;gap:7px}}.legend i{{display:inline-block;width:22px;border-top:2px solid var(--swatch)}}figcaption{{font:13px/1.5 var(--sans);color:var(--faint);margin-top:9px;max-width:88ch}}code{{font-family:var(--mono)}}footer{{max-width:850px;border-top:1px solid var(--line);margin-top:58px;padding-top:20px;font:12px/1.55 var(--sans);color:var(--faint)}}
@media(max-width:760px){{.cards,.rule{{grid-template-columns:1fr 1fr}}.card:nth-child(2){{border-right:0}}.card{{border-bottom:1px solid var(--line)}}}}
</style></head><body><main>
<header><div class="eyebrow">Daily adjusted returns · {result["sample"]["start"]} to {result["sample"]["end"]}</div><h1>Harvest winners. <em>Audit the escape hatch.</em></h1><p class="standfirst">The requested 5% / +20% quality-stock rule is implemented without future-price signals. It raised historical terminal wealth in this fixed basket, but it did not reduce the portfolio's deepest loss—and it did not close a single full-period lot under the two-condition compounder test.</p>
<div class="cards"><div class="card"><b>{pct(baseline["cagr"])}</b><span>baseline CAGR; SPY was {pct(spy["cagr"])}</span></div><div class="card"><b>{pct(baseline["max_drawdown"])}</b><span>baseline max drawdown; SPY was {pct(spy["max_drawdown"])}</span></div><div class="card"><b>{money(baseline["terminal"])}</b><span>ending baseline wealth from {money(result["policy"]["initial"])}</span></div><div class="card"><b>0</b><span>full-period compounder exits; 51 lots remain open</span></div></div></header>

<section><div class="eyebrow">I · Mechanical policy</div><h2>Signals today, trades at the next close.</h2><div class="rule"><div class="step"><b>Quality entry</b><span>At the portfolio −40% rung, only names with 20 already-reported quarters and an intact earnings guardrail qualify.</span></div><div class="step"><b>5% at +20%</b><span>Sell 5% of current shares whenever the prior close crosses the next multiplicative rung; send proceeds to reserve.</span></div><div class="step"><b>Five-year review</b><span>Compute trailing five-year adjusted-total-return CAGR from data available at the signal close.</span></div><div class="step"><b>Two breaks</b><span>Close only when CAGR is below 5% and TTM earnings are non-positive or persistently declining.</span></div></div>
<p class="note"><strong>Threshold convention:</strong> “from the last harvest level” is literal. The +20% ladder is 1.20×, 1.44×, 1.728× and so on—not additive +20%, +40%, +60% from the original entry. The implementation uses adjusted close, so the ladder measures total return rather than price appreciation alone.</p></section>

<section class="wide"><div class="eyebrow">II · Strategy versus SPY</div><h2>Higher return came with a deeper drawdown.</h2><figure><div class="figure-head"><strong>Growth of {money(100000)}</strong><span>adjusted total return · log scale</span></div><div class="plate">{growth_svg}<div class="legend">{legend}</div></div><figcaption>The strategy starts 80% in a financed, daily-reset 3x SPY sleeve and 20% in Treasury reserve, then steps leverage down with portfolio drawdown. It is not a direct unlevered substitute for SPY.</figcaption></figure>
<figure><div class="figure-head"><strong>Drawdown from each path's prior high</strong><span>daily portfolio NAV</span></div><div class="plate">{drawdown_svg}<div class="legend">{legend}</div></div><figcaption>The harvest variants share the same roughly {pct(baseline["max_drawdown"])} historical maximum drawdown. Profit harvesting did not address the dominant leverage risk.</figcaption></figure></section>

<section class="wide"><div class="eyebrow">III · Parameter grid</div><h2>No harvest setting is a robust winner.</h2><p class="lead">All-cohort median CAGRs tie because many 20-year windows never activate a quality purchase. The table therefore reports median excess CAGR only among the eight quality-active cohorts; those overlapping cohorts are scenarios, not independent trials.</p><div class="scroll"><table><thead><tr><th>Harvest / rung</th><th>Full CAGR</th><th>Full drawdown</th><th>Active median excess</th><th>Active beat SPY</th><th>Harvest events</th><th>Moved to reserve</th></tr></thead><tbody>{''.join(ranking_rows)}</tbody></table></div>
<p>The best full-period CAGR was <strong>{pct(full_best["cagr"])}</strong> for <code>{html.escape(full_best_name)}</code>. The best active-cohort median excess CAGR belonged to <code>{html.escape(active_best_name)}</code> at <strong>{pct(result["ranking"][0]["quality_active_median_excess_cagr"])}</strong>. Because those winners differ and the gaps are small, the grid does not justify optimizing away from the simpler 5% / +20% baseline.</p></section>

<section><div class="eyebrow">IV · The forever-hold question</div><h2>The guardrail was too permissive to guarantee cleanup.</h2><p>The baseline bought 51 lots across three crisis entries. By the final date, all 51 remained open; the oldest had been held {baseline["quality_lots"]["oldest_open_holding_years"]:.1f} years. The full-period rule generated {baseline["events"]["quality_profit_harvest"]["events"]} partial harvests but no compounder exits.</p><p class="note warning"><strong>Decision:</strong> the “CAGR below 5% AND earnings broken” rule is a quality-break exit, not a true sunset. If the operational goal is to ensure capital eventually rotates back to SPY, add an unconditional outer limit—such as exit at year 10—or change AND to OR. Both are materially different policies and should be tested before adoption.</p></section>

<section class="wide"><div class="eyebrow">V · Baseline positions still held</div><h2>What remained on balance.</h2><p class="lead">Market value and remaining allocated cost after all 5% harvests, for the requested 5% / +20% rule.</p><div class="scroll"><table><thead><tr><th>Name and entry dates</th><th>Market value</th><th>Remaining cost</th><th>Unrealized multiple</th><th>Portfolio weight</th></tr></thead><tbody>{position_rows}</tbody></table></div></section>

<section><div class="eyebrow">VI · Bias audit</div><h2>Trade timing is clean; stock selection is not.</h2><p>Price thresholds use the prior close and execute at the next close. Five-year CAGR stops at the signal close. Earnings use report availability, not fiscal-period end: before-open reports may be used that day, while after-close or unknown-time reports are delayed to the next calendar day. Missing fundamentals never force an exit.</p><p class="note warning"><strong>Unresolved bias:</strong> the 18 candidates are a present-day, survivor-selected basket; historical market-cap membership, delisted-company returns, and vintage fundamental snapshots are unavailable. The EPS vendor may also revise old observations. This is a temporally clean exit-policy comparison on fixed candidates—not a survivor-free estimate of stock-selection alpha.</p></section>

<footer>Historical simulation, not investment advice. Generated by <code>scripts/run_quality_compounder_harvest_study.py</code> and <code>scripts/build_quality_compounder_harvest_page.py</code>. Financing uses prior-known 3-month Treasury yield plus 1%; stock trades cost 5 bp per side; taxes, market impact, margin calls, ETF tracking error, and execution gaps are omitted.</footer>
</main></body></html>'''
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_text, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
