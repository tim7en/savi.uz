"""Build a compact investor guide from the Quality Compounder V2 backtest."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_contribution_quality_page import line_chart, money, pct


THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("out/strategy/quality_compounder_v2")
    )
    parser.add_argument(
        "--cape-input", type=Path, default=Path("out/strategy/cape_leverage")
    )
    parser.add_argument(
        "--analysis-output",
        type=Path,
        default=Path("out/strategy/simple_investor_guide/results.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/simple-compounder-guide.html")
    )
    return parser.parse_args(argv)


def _date_or_none(value):
    return None if value is None else pd.Timestamp(value).strftime("%Y-%m-%d")


def drawdown_events(dates, values, thresholds=THRESHOLDS):
    """Return the first threshold crossing in every peak-to-recovery episode."""
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    values = pd.Series(values, dtype=float).reset_index(drop=True)
    if values.empty or values.isna().any() or (values <= 0).any():
        raise ValueError("dates and positive, finite values are required")

    peak_level = float(values.iloc[0])
    peak_date = dates.iloc[0]
    crossed: set[float] = set()
    raw_events = []

    for i in range(1, len(values)):
        value = float(values.iloc[i])
        if value >= peak_level:
            peak_level = value
            peak_date = dates.iloc[i]
            crossed.clear()
            continue

        drawdown = value / peak_level - 1.0
        for threshold in thresholds:
            threshold = float(threshold)
            if threshold not in crossed and drawdown <= -threshold + 1e-12:
                raw_events.append(
                    {
                        "threshold": threshold,
                        "peak_date": peak_date,
                        "peak_level": peak_level,
                        "entry_date": dates.iloc[i],
                        "entry_level": value,
                        "entry_index": i,
                    }
                )
                crossed.add(threshold)

    events = []
    value_array = values.to_numpy()
    for event in raw_events:
        i = event["entry_index"]
        future = value_array[i:]
        old_high_matches = np.flatnonzero(future >= event["peak_level"])
        plus_ten_matches = np.flatnonzero(future >= event["entry_level"] * 1.10)
        recovery_i = i + int(old_high_matches[0]) if len(old_high_matches) else None
        plus_ten_i = i + int(plus_ten_matches[0]) if len(plus_ten_matches) else None
        trough_end = recovery_i if recovery_i is not None else len(values) - 1
        trough = float(values.iloc[i : trough_end + 1].min())
        one_year_i = i + 252
        events.append(
            {
                "threshold": event["threshold"],
                "peak_date": _date_or_none(event["peak_date"]),
                "entry_date": _date_or_none(event["entry_date"]),
                "entry_drawdown": event["entry_level"] / event["peak_level"] - 1.0,
                "further_loss_after_entry": trough / event["entry_level"] - 1.0,
                "old_high_recovery_date": _date_or_none(
                    dates.iloc[recovery_i] if recovery_i is not None else None
                ),
                "days_to_old_high": (
                    int((dates.iloc[recovery_i] - event["entry_date"]).days)
                    if recovery_i is not None
                    else None
                ),
                "plus_10_date": _date_or_none(
                    dates.iloc[plus_ten_i] if plus_ten_i is not None else None
                ),
                "days_to_plus_10": (
                    int((dates.iloc[plus_ten_i] - event["entry_date"]).days)
                    if plus_ten_i is not None
                    else None
                ),
                "one_year_return": (
                    float(values.iloc[one_year_i] / event["entry_level"] - 1.0)
                    if one_year_i < len(values)
                    else None
                ),
            }
        )
    return events


def summarize_events(events, thresholds=THRESHOLDS):
    summary = []
    for threshold in thresholds:
        selected = [row for row in events if row["threshold"] == float(threshold)]
        plus_ten = [row["days_to_plus_10"] for row in selected if row["days_to_plus_10"] is not None]
        old_high = [row["days_to_old_high"] for row in selected if row["days_to_old_high"] is not None]
        one_year = [row["one_year_return"] for row in selected if row["one_year_return"] is not None]
        further = [row["further_loss_after_entry"] for row in selected]
        summary.append(
            {
                "threshold": float(threshold),
                "episodes": len(selected),
                "median_days_to_plus_10": int(np.median(plus_ten)) if plus_ten else None,
                "worst_days_to_plus_10": max(plus_ten) if plus_ten else None,
                "median_days_to_old_high": int(np.median(old_high)) if old_high else None,
                "worst_days_to_old_high": max(old_high) if old_high else None,
                "median_one_year_return": float(np.median(one_year)) if one_year else None,
                "worst_further_loss": min(further) if further else None,
            }
        )
    return summary


def _days(value):
    if value is None:
        return "not reached"
    if value < 365:
        return f"{value}d"
    return f"{value / 365.25:.1f}y"


def _optional_pct(value):
    return "n/a" if value is None else pct(value, 1)


def main(argv=None):
    args = parse_args(argv)
    results = json.loads((args.input / "results.json").read_text(encoding="utf-8"))
    cape_results = json.loads(
        (args.cape_input / "results.json").read_text(encoding="utf-8")
    )
    daily = pd.read_csv(args.input / "daily.csv", parse_dates=["date"])

    spy_events = drawdown_events(daily["date"], daily["spy_1x_performance"])
    nav_events = drawdown_events(daily["date"], daily["immediate_20_performance"])
    spy_summary = summarize_events(spy_events)
    nav_summary = summarize_events(nav_events)

    spy_perf = daily["spy_1x_performance"].astype(float)
    strategy_perf = daily["immediate_20_performance"].astype(float)
    latest = daily.iloc[-1]
    spy_drawdown = float(spy_perf.iloc[-1] / spy_perf.cummax().iloc[-1] - 1.0)
    strategy_drawdown = float(
        strategy_perf.iloc[-1] / strategy_perf.cummax().iloc[-1] - 1.0
    )
    current_cape = 41.18
    current_cape_as_of = "2026-08-01"
    treasury_rate = 0.0388
    treasury_rate_as_of = "2026-08-24"
    modeled_financing = treasury_rate + 0.01
    nav_leverage = float(latest["immediate_20_leverage"])
    cape_cap = 1.0 if current_cape >= 35 else (2.0 if current_cape >= 25 else 3.0)
    guided_leverage = min(nav_leverage, cape_cap)
    current = {
        "as_of": latest["date"].strftime("%Y-%m-%d"),
        "spy_drawdown": spy_drawdown,
        "strategy_nav_drawdown": strategy_drawdown,
        "nav_only_leverage": nav_leverage,
        "cape": current_cape,
        "cape_as_of": current_cape_as_of,
        "cape_cap": cape_cap,
        "guided_leverage": guided_leverage,
        "treasury_3m": treasury_rate,
        "treasury_as_of": treasury_rate_as_of,
        "modeled_financing_hurdle": modeled_financing,
        "spy_trailing_12m_return": float(spy_perf.iloc[-1] / spy_perf.iloc[-253] - 1.0),
        "spy_63d_volatility": float(
            spy_perf.pct_change().iloc[-63:].std(ddof=1) * np.sqrt(252)
        ),
        "treasury_share": float(
            latest["immediate_20_treasury"] / latest["immediate_20_wealth"]
        ),
    }

    analysis = {
        "sample": results["sample"],
        "current_conditions": current,
        "spy_drawdown_summary": spy_summary,
        "strategy_nav_drawdown_summary": nav_summary,
        "spy_events": spy_events,
        "strategy_nav_events": nav_events,
    }
    args.analysis_output.parent.mkdir(parents=True, exist_ok=True)
    args.analysis_output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    primary = results["variants"]["immediate_20"]
    spy = results["variants"]["spy_1x"]
    chart_frame = daily[
        ["date", "spy_1x_wealth", "immediate_20_wealth", "spy_1x_performance", "immediate_20_performance"]
    ].copy()
    colors = {
        "spy_1x_wealth": "var(--ink)",
        "immediate_20_wealth": "var(--teal)",
        "spy_1x_performance": "var(--ink)",
        "immediate_20_performance": "var(--teal)",
    }
    growth_chart = line_chart(
        chart_frame, ["spy_1x_wealth", "immediate_20_wealth"], colors
    )
    drawdown_chart = line_chart(
        chart_frame,
        ["spy_1x_performance", "immediate_20_performance"],
        colors,
        performance=True,
    )
    legend = (
        '<span><i style="--swatch:var(--ink)"></i>SPY 1x</span>'
        '<span><i style="--swatch:var(--teal)"></i>Quality Compounder</span>'
    )

    threshold_rows = "".join(
        f'<tr class="{"thin" if row["episodes"] < 3 else ""}">'
        f'<td><strong>-{row["threshold"]:.0%}</strong></td>'
        f'<td>{row["episodes"]}</td>'
        f'<td>{_days(row["median_days_to_plus_10"])}</td>'
        f'<td>{_days(row["worst_days_to_plus_10"])}</td>'
        f'<td>{_days(row["median_days_to_old_high"])}</td>'
        f'<td>{_days(row["worst_days_to_old_high"])}</td>'
        f'<td>{_optional_pct(row["worst_further_loss"])}</td>'
        f'<td>{_optional_pct(row["median_one_year_return"])}</td></tr>'
        for row in spy_summary
    )
    nav_rows = "".join(
        f'<tr class="{"thin" if row["episodes"] < 3 else ""}">'
        f'<td><strong>-{row["threshold"]:.0%}</strong></td><td>{row["episodes"]}</td>'
        f'<td>{_days(row["median_days_to_old_high"])}</td>'
        f'<td>{_days(row["worst_days_to_old_high"])}</td>'
        f'<td>{_optional_pct(row["worst_further_loss"])}</td></tr>'
        for row in nav_summary
    )
    severe_rows = "".join(
        f'<tr><td><strong>{row["peak_date"][:4]} episode</strong><small>20% entry {row["entry_date"]}</small></td>'
        f'<td>{_optional_pct(row["further_loss_after_entry"])}</td>'
        f'<td>{row["plus_10_date"] or "not reached"}<small>{_days(row["days_to_plus_10"])}</small></td>'
        f'<td>{row["old_high_recovery_date"] or "not reached"}<small>{_days(row["days_to_old_high"])}</small></td></tr>'
        for row in spy_events
        if row["threshold"] == 0.20
    )

    comparison_rows = "".join(
        f'<tr class="{"featured" if key == "immediate_20" else ""}">'
        f'<td><strong>{html.escape(label)}</strong><small>{html.escape(note)}</small></td>'
        f'<td>{money(stats["terminal_wealth"])}</td><td>{pct(stats["xirr"])}</td>'
        f'<td>{pct(stats["time_weighted_cagr"])}</td>'
        f'<td>{pct(stats["max_flow_adjusted_drawdown"])}</td>'
        f'<td>{pct(stats["annual_volatility"])}</td>'
        f'<td>{stats["longest_underwater_sessions"] / 252:.1f}y</td></tr>'
        for key, label, note, stats in [
            ("immediate_20", "Quality Compounder", "20% Treasury; monthly deposits invested immediately", primary),
            ("spy_1x", "SPY 1x", "same cash flows; adjusted total return", spy),
        ]
    )

    cape_base = cape_results["strategy_variants"]["no_cape_cap"]
    cape_25_35 = cape_results["strategy_variants"]["cape_25_35"]
    cape_spy = cape_results["strategy_variants"]["spy_1x"]
    cape_rows = "".join(
        f'<tr><td><strong>{label}</strong></td><td>{money(row["terminal_wealth"])}</td>'
        f'<td>{pct(row["xirr"])}</td><td>{pct(row["max_flow_adjusted_drawdown"])}</td>'
        f'<td>{row.get("mean_applied_leverage", 1.0):.2f}x</td></tr>'
        for label, row in [
            ("NAV rule, no CAPE cap", cape_base),
            ("CAPE 25/35 ceiling", cape_25_35),
            ("SPY 1x", cape_spy),
        ]
    )

    html_text = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Quality Compounder - investor guide</title>
<style>
:root{{--bg:#f1f4f2;--paper:#fcfdfc;--ink:#14201d;--muted:#53635f;--faint:#778681;--line:#ccd7d3;--teal:#087f72;--soft:#dcefea;--brick:#b84138;--warn:#f5dfdc;--amber:#996c00;--amber-soft:#f5ecd0;--mono:"Cascadia Mono",Consolas,monospace;--serif:Georgia,"Times New Roman",serif;--sans:system-ui,-apple-system,"Segoe UI",sans-serif}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:17px/1.62 var(--serif)}}main{{width:min(1100px,calc(100% - 34px));margin:auto;padding:52px 0 80px}}header{{border-bottom:2px solid var(--ink);padding-bottom:34px}}.eyebrow{{font:700 11px var(--mono);letter-spacing:.13em;text-transform:uppercase;color:var(--teal)}}h1{{font:650 clamp(40px,7vw,72px)/1.02 var(--serif);letter-spacing:-.045em;max-width:13ch;margin:13px 0 20px}}h1 em{{color:var(--teal)}}.standfirst{{font-size:22px;line-height:1.43;color:var(--muted);max-width:65ch}}section{{padding-top:48px;max-width:880px}}section.wide{{max-width:none}}h2{{font:650 31px/1.18 var(--serif);letter-spacing:-.025em;margin:0 0 14px}}h3{{font:650 21px/1.3 var(--serif);margin:30px 0 10px}}p{{margin:0 0 18px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);margin-top:28px}}.card{{background:var(--paper);padding:17px;border-right:1px solid var(--line)}}.card:last-child{{border:0}}.card b{{display:block;font:700 22px var(--mono)}}.card span{{display:block;font:12px/1.4 var(--sans);color:var(--faint);margin-top:7px}}.state{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:24px 0}}.state article{{background:var(--paper);border-top:4px solid var(--teal);padding:18px}}.state article.warn{{border-color:var(--amber)}}.state article.stop{{border-color:var(--brick)}}.state b{{display:block;font:700 24px var(--mono)}}.state small{{display:block;color:var(--faint);font:12px/1.45 var(--sans);margin-top:6px}}.rule{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:23px 0}}.step{{background:var(--paper);border-top:3px solid var(--teal);padding:15px}}.step:last-child{{border-color:var(--brick)}}.step b{{display:block;font:700 18px var(--mono)}}.step span{{font:12px/1.45 var(--sans);color:var(--faint)}}.note{{background:var(--soft);border-left:3px solid var(--teal);padding:18px 20px;color:var(--muted)}}.warning{{background:var(--warn);border-left-color:var(--brick)}}.caution{{background:var(--amber-soft);border-left-color:var(--amber)}}.scroll{{overflow:auto;border:1px solid var(--line);background:var(--paper);margin-top:22px}}table{{border-collapse:collapse;width:100%;font:13px/1.4 var(--sans);font-variant-numeric:tabular-nums}}th,td{{padding:12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}}th{{font:700 10px var(--mono);text-transform:uppercase;letter-spacing:.06em;color:var(--faint)}}th:first-child,td:first-child{{text-align:left}}td:first-child{{white-space:normal;min-width:165px}}td strong,td small{{display:block}}td small{{color:var(--faint);margin-top:3px}}tr.featured{{background:var(--soft)}}tr.thin{{background:#faf4e4}}figure{{margin:30px 0}}.figure-head{{display:flex;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:9px;font:13px var(--sans);color:var(--muted)}}.plate{{background:var(--paper);border:1px solid var(--line);padding:14px;margin-top:13px;overflow:auto}}svg{{display:block;width:100%;min-width:720px}}.frame{{fill:none;stroke:var(--line)}}.grid{{stroke:var(--line)}}.axis{{font:10px var(--mono);fill:var(--faint)}}.axis-title{{font:12px var(--sans);fill:var(--ink)}}.legend{{display:flex;gap:18px;flex-wrap:wrap;padding:8px 4px 0;font:11px var(--sans);color:var(--muted)}}.legend span{{display:flex;gap:7px;align-items:center}}.legend i{{width:22px;border-top:2px solid var(--swatch)}}.verdict{{font-size:20px;color:var(--brick)}}figcaption,footer{{font:12px/1.55 var(--sans);color:var(--faint)}}footer{{border-top:1px solid var(--line);margin-top:55px;padding-top:20px;max-width:960px}}a{{color:var(--teal)}}code{{font-family:var(--mono)}}@media(max-width:760px){{.cards,.rule,.state{{grid-template-columns:1fr 1fr}}.card:nth-child(2){{border-right:0}}}}@media(max-width:520px){{.cards,.rule,.state{{grid-template-columns:1fr}}}}
</style></head><body><main>
<header><div class="eyebrow">Decision guide - data through {current["as_of"]}</div>
<h1>Buy the drawdown. <em>Do not mortgage the recovery.</em></h1>
<p class="standfirst">A compact rulebook for the Quality Compounder, its tested return against SPY, the historical wait after drawdown entries, and the line between deployable Treasury and household leverage.</p>
<div class="cards"><div class="card"><b>{money(primary["terminal_wealth"])}</b><span>strategy terminal wealth; SPY {money(spy["terminal_wealth"])}</span></div>
<div class="card"><b>{pct(primary["xirr"])}</b><span>strategy XIRR; SPY {pct(spy["xirr"])}</span></div>
<div class="card"><b>{pct(primary["max_flow_adjusted_drawdown"])}</b><span>strategy max drawdown; SPY {pct(spy["max_flow_adjusted_drawdown"])}</span></div>
<div class="card"><b>{primary["longest_underwater_sessions"] / 252:.1f}y</b><span>longest strategy wait for a new NAV high</span></div></div></header>

<section class="wide"><div class="eyebrow">I - Market state</div><h2>Today is not a drawdown deployment signal.</h2>
<div class="state"><article><b>{pct(current["spy_drawdown"], 1)}</b><span>SPY below total-return high</span><small>12-month return {pct(current["spy_trailing_12m_return"], 1)}; 63-day annualized volatility {pct(current["spy_63d_volatility"], 1)}.</small></article>
<article class="warn"><b>{pct(current["strategy_nav_drawdown"], 1)}</b><span>strategy NAV below high</span><small>The tested NAV-only rule is currently {current["nav_only_leverage"]:.0f}x. Treasury is {pct(current["treasury_share"], 1)} of NAV.</small></article>
<article class="stop"><b>{current["guided_leverage"]:.0f}x ceiling</b><span>if the experimental CAPE overlay is used</span><small>CAPE estimate {current["cape"]:.2f}; 3-month Treasury {pct(current["treasury_3m"], 2)}; modeled strategy financing {pct(current["modeled_financing_hurdle"], 2)}.</small></article></div>
<p class="caution note"><strong>Two different answers:</strong> the tested NAV-only system says {current["nav_only_leverage"]:.0f}x. The proposed CAPE 25/35 risk ceiling says no more than {current["cape_cap"]:.0f}x. The return chart below is the NAV-only backtest; it must not be presented as evidence for today's CAPE-capped exposure.</p></section>

<section><div class="eyebrow">II - The rulebook</div><h2>One clock for risk, one budget for opportunity.</h2>
<p><strong>Standing allocation:</strong> invest the initial account and each monthly $10,000 contribution immediately at 80% SPY sleeve / 20% Treasury. Dividends are reinvested. Do not park every contribution until the old NAV high.</p>
<p><strong>Leverage:</strong> the financed SPY sleeve begins at 3x, falls to 2x at a 15% flow-adjusted NAV drawdown and to 1x at 30%. From 1x, restore 2x only above -10%; restore 3x only at a new NAV high. If the valuation overlay is adopted, CAPE 25-35 caps exposure at 2x and CAPE at least 35 caps it at 1x.</p>
<div class="rule"><div class="step"><b>-10%</b><span>Deploy 10% of episode-start Treasury into unlevered SPY.</span></div><div class="step"><b>-20%</b><span>Deploy another 20% into unlevered SPY.</span></div><div class="step"><b>-30%</b><span>Deploy another 30% into unlevered SPY.</span></div><div class="step"><b>-40%</b><span>Deploy final 40% into seven date-ranked mega caps.</span></div></div>
<p><strong>Harvest:</strong> after a financed-sleeve quarter of at least +20%, sweep 10% of that dollar profit to Treasury. After a SPY calendar year of at least +10%, sweep 1% of the core. For the quality sleeve, sell 5% of a profitable position after its own +20% quarter or when SPY regains the episode high; after five years, exit if its trailing five-year CAGR is below SPY.</p>
<p class="note"><strong>Operational guide:</strong> no rung is active now. Normal contributions follow 80/20. Treasury deployment is based on flow-adjusted strategy NAV, not the raw dollar balance and not the percentage decline in SPY.</p></section>

<section class="wide"><div class="eyebrow">III - Strategy versus SPY</div><h2>The endpoint tied SPY; the path was materially worse.</h2>
<figure><div class="figure-head"><strong>Account value with matched $10,000 monthly deposits</strong><span>log scale</span></div><div class="plate">{growth_chart}<div class="legend">{legend}</div></div></figure>
<figure><div class="figure-head"><strong>Flow-adjusted drawdown</strong><span>deposits removed from the return clock</span></div><div class="plate">{drawdown_chart}<div class="legend">{legend}</div></div></figure>
<div class="scroll"><table><thead><tr><th>Path</th><th>Terminal</th><th>XIRR</th><th>TWR CAGR</th><th>Max DD</th><th>Volatility</th><th>Longest underwater</th></tr></thead><tbody>{comparison_rows}</tbody></table></div>
<p class="warning note">The strategy finished only {pct(primary["terminal_wealth"] / spy["terminal_wealth"] - 1, 2)} above SPY and added about {(primary["xirr"] - spy["xirr"]) * 10_000:.1f} basis points of XIRR, while volatility rose from {pct(spy["annual_volatility"], 1)} to {pct(primary["annual_volatility"], 1)} and the longest underwater period almost doubled. This is not a demonstrated risk-adjusted improvement.</p></section>

<section class="wide"><div class="eyebrow">IV - When drawdown capital worked</div><h2>A new purchase can win before the old portfolio recovers.</h2>
<p>These rows use SPY adjusted total return and take only the first crossing of each threshold in a peak-to-recovery episode. “+10%” is the date a fresh purchase first gained 10% gross. “Old high” is the date the pre-drawdown portfolio recovered. Financing, tax and execution costs are not deducted from the +10% clock.</p>
<div class="scroll"><table><thead><tr><th>SPY DD entry</th><th>Episodes</th><th>Median to +10%</th><th>Worst to +10%</th><th>Median to old high</th><th>Worst to old high</th><th>Worst fall after entry</th><th>Median 1y return</th></tr></thead><tbody>{threshold_rows}</tbody></table></div>
<p class="caution note"><strong>There was no single best threshold:</strong> 10% drawdowns were common and had the shortest median wait to the old high. Fresh capital placed at 20-30% drawdowns reached +10% sooner at the median, but those entries occurred inside the longest bear markets. Treat 20-30% as an acceleration zone, not a reason to keep all savings idle. At 40% there were only two episodes, at 50% one, and at 60% none.</p>
<h3>The four SPY bear markets that crossed 20%</h3>
<div class="scroll"><table><thead><tr><th>Episode / first 20% entry</th><th>Further loss after entry</th><th>Fresh capital +10%</th><th>Old high recovered</th></tr></thead><tbody>{severe_rows}</tbody></table></div>
<p>The dot-com 20% purchase earned its first +10% in 67 days, yet the old SPY high was not recovered for 5.6 years from that entry. In 2008 the same clocks were about 2.5 and 4.1 years. Recovery-dependent borrowing therefore creates a maturity mismatch even when the purchase eventually succeeds.</p>

<h3>The strategy NAV is a faster and harsher clock</h3>
<div class="scroll"><table><thead><tr><th>Strategy NAV DD</th><th>Episodes</th><th>Median to old NAV high</th><th>Worst to old NAV high</th><th>Worst fall after entry</th></tr></thead><tbody>{nav_rows}</tbody></table></div>
<p>Because NAV includes a leveraged sleeve, a 20% strategy drawdown is not the same as a 20% SPY drawdown. The drawdown ladder triggered far more often, and the worst strategy recovery lasted {primary["longest_underwater_sessions"] / 252:.1f} trading years.</p></section>

<section class="wide"><div class="eyebrow">V - CAPE</div><h2>Use valuation as a ceiling, not a sell signal.</h2>
<p>The CAPE study lagged the monthly value by one month. The 25/35 ceiling means at most 2x when CAPE is 25-35 and 1x when CAPE is at least 35. On the common 1993-2024 sample it improved terminal wealth, but it did not reduce the worst drawdown versus either the uncapped strategy or SPY.</p>
<div class="scroll"><table><thead><tr><th>Policy</th><th>Terminal</th><th>XIRR</th><th>Max DD</th><th>Mean leverage</th></tr></thead><tbody>{cape_rows}</tbody></table></div>
<p class="note">At the current third-party CAPE estimate of {current["cape"]:.2f}, the overlay caps exposure at 1x. That is a risk-budget decision. This backtest does not show that CAPE times the next decline.</p></section>

<section><div class="eyebrow">VI - Borrowing against a flat</div><h2 class="verdict">My answer: do not make the home loan part of this strategy.</h2>
<p>At a 50% account drawdown, recovery to the old high requires +100%. At a 60% drawdown it requires +150%. Those are exactly the moments when employment, credit availability and asset prices may be stressed together. A HELOC may have a variable rate, can be frozen if the home value or the borrower's finances deteriorate, and nonpayment can put the home at risk.</p>
<p>Borrowing an amount equal to the remaining account does not reset household wealth. If a $100 peak falls to $50, then $50 is borrowed and invested, the account shows $100 but owes $50. A further 20% market fall leaves $80 of assets and $50 of debt: only $30 of net financial equity before interest, a 70% loss from the original $100 peak.</p>
<p class="warning note"><strong>Safer funding hierarchy:</strong> (1) pre-funded Treasury reserve, (2) scheduled contributions supported by stable outside income, (3) optional excess cash not needed for several years. Home equity should remain outside the strategy. If borrowing is still considered, it needs individualized legal, tax and regulated financial advice; the loan must be serviceable without market recovery, dividends, or selling the flat.</p>
<p>Broker margin is not a solution either: maintenance requirements can rise, and a broker may liquidate positions without advance notice. The strategy backtest omits forced-liquidation mechanics, taxes, home-loan fees and the possibility that future drawdowns exceed the historical sample.</p></section>

<footer><strong>Method and sources.</strong> Historical simulation, not investment advice. Backtest: Yahoo adjusted total returns with dividends reinvested, prior-known DGS3MO + 1% financing, and matched monthly deposits; data and assumptions are in <code>out/strategy/quality_compounder_v2</code>. Current SPY state is calculated from the local series through {current["as_of"]}. Current CAPE is a third-party estimate as of {current["cape_as_of"]}: <a href="https://www.macroradar.io/shiller-pe-ratio">MacroRadar</a>; definition and historical source: <a href="https://www.econ.yale.edu/~shiller/data.htm">Robert Shiller / Yale</a>. Treasury rate is from the <a href="https://www.federalreserve.gov/releases/h15/">Federal Reserve H.15</a> as of {current["treasury_as_of"]}. Home-equity risks: <a href="https://www.consumerfinance.gov/ask-cfpb/what-is-a-home-equity-line-of-credit-heloc-en-107/">CFPB HELOC guide</a> and <a href="https://www.consumerfinance.gov/ask-cfpb/what-is-a-home-equity-loan-en-106/">CFPB home-equity loan guide</a>. Margin risks: <a href="https://www.finra.org/investors/insights/margin-calls">FINRA</a>. Generated by <code>scripts/build_simple_investor_guide.py</code>.</footer>
</main></body></html>'''

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
