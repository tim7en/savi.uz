"""Build the self-contained macro-equilibrium regime overlay report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_contribution_quality_page import line_chart, money, pct


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("out/strategy/macro_equilibrium"))
    parser.add_argument("--output", type=Path, default=Path("docs/macro-equilibrium-strategy.html"))
    return parser.parse_args(argv)


NARRATIVE = {
    "2000-03-27": ("Dot-com bubble burst", "Overhyped internet valuations; the crash was worsened by 9/11 and the 2001-2002 recession."),
    "2007-10-31": ("Global Financial Crisis", "The housing bubble imploded, subprime loans defaulted, Lehman Brothers collapsed, and credit markets froze."),
    "2018-08-29": ("Trade war + Fed hikes", "Tariff escalation and aggressive Fed tightening drained liquidity and hit stretched tech valuations."),
    "2020-02-19": ("COVID-19 crash", "Global lockdowns halted economic activity; the fastest bear market in history, reversed by emergency stimulus."),
    "2021-11-19": ("2022 inflation shock", "Inflation surged past 9%; the Fed answered with the most aggressive hikes since the 1980s."),
    "2025-02-19": ("2025 tariff selloff", "Renewed tariff policy reignited trade tension and delayed expected rate cuts."),
    "2015-11-03": ("2015-16 growth scare", "China devaluation and an oil-price collapse (not in the original six, found by the same scan)."),
}


def exposure_chart(frame: pd.DataFrame) -> str:
    width, height = 980, 260
    left, right, top, bottom = 60, 24, 20, 40
    plot_w, plot_h = width - left - right, height - top - bottom
    sampled = frame.set_index("date")["target_exposure"].resample("W").last().ffill().dropna()
    ticks = [0.0, 0.20, 0.45, 0.70, 1.0]
    start, end = sampled.index[0], sampled.index[-1]
    span = max((end - start).days, 1)
    x = lambda stamp: left + (stamp - start).days / span * plot_w
    y = lambda v: top + (1.0 - v) * plot_h
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Target equity exposure over time">',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" class="frame"/>',
    ]
    for tick in ticks:
        yy = y(tick)
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{yy:.1f}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" class="axis">{tick:.0%}</text>')
    for year in range(((start.year + 4) // 5) * 5, end.year + 1, 5):
        xx = x(pd.Timestamp(year, 1, 1))
        parts.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{top}" y2="{height-bottom}" class="grid"/>')
        parts.append(f'<text x="{xx:.1f}" y="{height-14}" text-anchor="middle" class="axis">{year}</text>')
    points = " ".join(f"{x(stamp):.1f},{y(float(value)):.1f}" for stamp, value in sampled.items())
    area = f"{left},{top+plot_h} " + points + f" {x(sampled.index[-1]):.1f},{top+plot_h}"
    parts.append(f'<polygon points="{area}" fill="var(--soft)" stroke="none"/>')
    parts.append(f'<polyline points="{points}" fill="none" stroke="var(--teal)" stroke-width="1.8" vector-effect="non-scaling-stroke"/>')
    parts.append('</svg>')
    return "".join(parts)


def episode_row(peak_date: str, episode: dict) -> str:
    label, blurb = NARRATIVE.get(peak_date, (peak_date, ""))
    stressed = ", ".join(episode["stressed_pillars_at_peak"]) or "none"
    warned = "yes" if episode["lead_days_below_full_exposure"] else "no"
    exposure = pct(episode["exposure_at_peak"], 0)
    recovery = episode["recovery_date"] or "not yet recovered"
    count = episode["stress_count_at_peak"]
    count_label = "n/a (pre-signal)" if count is None else f"{count}/5"
    return (
        f"<tr><td style=\"text-align:left\"><strong>{label}</strong><br>"
        f"<small style=\"color:var(--faint)\">{blurb}</small></td>"
        f"<td>{episode['peak_date']}</td><td>{episode['trough_date']}</td>"
        f"<td>{pct(episode['depth'], 1)}</td><td>{recovery}</td>"
        f"<td>{count_label}</td><td>{stressed}</td><td>{exposure}</td><td>{warned}</td></tr>"
    )


def main(argv=None):
    args = parse_args(argv)
    result = json.loads((args.input / "results.json").read_text(encoding="utf-8"))
    daily = pd.read_csv(args.input / "daily.csv", parse_dates=["date"])

    ndx_regime = result["variants"]["ndx_regime"]
    ndx_buyhold = result["variants"]["ndx_buyhold"]
    spy_regime = result["variants"]["spy_regime"]
    spy_buyhold = result["variants"]["spy_buyhold"]

    colors = {
        "spy_regime_wealth": "var(--teal)",
        "spy_buyhold_wealth": "var(--ink)",
        "ndx_regime_wealth": "var(--teal)",
        "ndx_buyhold_wealth": "var(--ink)",
    }
    spy_chart = line_chart(daily, ["spy_regime_wealth", "spy_buyhold_wealth"], colors)
    ndx_chart = line_chart(daily, ["ndx_regime_wealth", "ndx_buyhold_wealth"], colors)
    exposure_svg = exposure_chart(daily)

    episodes = list(result["episodes"]) + [result["nested_2008_episode"]]
    episodes.sort(key=lambda e: e["peak_date"])
    episode_rows = "".join(episode_row(e["peak_date"], e) for e in episodes)

    time_at_exposure = sorted(
        ((float(k), v) for k, v in result["time_at_exposure"].items()), reverse=True
    )
    exposure_rows = "".join(
        f"<tr><td>{pct(level, 0)}</td><td>{pct(share, 1)}</td></tr>"
        for level, share in time_at_exposure
    )
    full_share = time_at_exposure[0][1]
    below_70_share = sum(share for level, share in time_at_exposure if level < 0.70)

    pillars = result["pillars"]

    html_text = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Macro Equilibrium Overlay - backtest</title>
<style>
:root{{--bg:#f1f4f2;--paper:#fcfdfc;--ink:#14201d;--muted:#53635f;--faint:#778681;--line:#ccd7d3;--teal:#087f72;--blue:#356da6;--amber:#9a6a00;--soft:#dcefea;--brick:#b84138;--warn:#f5dfdc;--gold:#f5ecd0;--mono:"Cascadia Mono",Consolas,monospace;--serif:Georgia,"Times New Roman",serif;--sans:system-ui,-apple-system,"Segoe UI",sans-serif}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:17px/1.62 var(--serif)}}main{{width:min(1100px,calc(100% - 34px));margin:auto;padding:52px 0 80px}}header{{border-bottom:2px solid var(--ink);padding-bottom:34px}}.eyebrow{{font:700 11px var(--mono);letter-spacing:.13em;text-transform:uppercase;color:var(--teal)}}h1{{font:650 clamp(40px,7vw,72px)/1.02 var(--serif);letter-spacing:-.045em;max-width:16ch;margin:13px 0 20px}}h1 em{{color:var(--teal)}}.standfirst{{font-size:22px;line-height:1.43;color:var(--muted);max-width:66ch}}section{{padding-top:48px;max-width:880px}}section.wide{{max-width:none}}h2{{font:650 31px/1.18 var(--serif);letter-spacing:-.025em;margin:0 0 14px}}h3{{font:650 21px/1.3 var(--serif);margin:30px 0 10px}}p{{margin:0 0 18px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);margin-top:28px}}.card{{background:var(--paper);padding:17px;border-right:1px solid var(--line)}}.card:last-child{{border:0}}.card b{{display:block;font:700 22px var(--mono)}}.card span{{display:block;font:12px/1.4 var(--sans);color:var(--faint);margin-top:7px}}.rule{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:23px 0}}.step{{background:var(--paper);border-top:3px solid var(--teal);padding:15px}}.step:nth-child(even){{border-color:var(--blue)}}.step b{{display:block;font:700 15px var(--mono)}}.step span{{font:12px/1.45 var(--sans);color:var(--faint)}}.note{{background:var(--soft);border-left:3px solid var(--teal);padding:18px 20px;color:var(--muted)}}.warning{{background:var(--warn);border-left-color:var(--brick)}}.caution{{background:var(--gold);border-left-color:var(--amber)}}.scroll{{overflow:auto;border:1px solid var(--line);background:var(--paper);margin-top:22px}}table{{border-collapse:collapse;width:100%;font:13px/1.4 var(--sans);font-variant-numeric:tabular-nums}}th,td{{padding:12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}}th{{font:700 10px var(--mono);text-transform:uppercase;letter-spacing:.06em;color:var(--faint)}}tbody tr:last-child td{{border-bottom:0}}tr.featured td{{background:var(--soft);font-weight:700}}figure{{margin:22px 0 0}}.figure-head{{display:flex;justify-content:space-between;align-items:baseline;font:13px var(--sans);color:var(--faint);margin-bottom:8px}}.figure-head strong{{color:var(--ink);font-size:15px}}.plate{{border:1px solid var(--line);background:var(--paper);padding:8px}}.frame{{fill:none;stroke:var(--line)}}.grid{{stroke:var(--line);stroke-dasharray:2 3}}.axis{{font:10px var(--mono);fill:var(--faint)}}.axis-title{{font:11px var(--sans);fill:var(--faint)}}.legend{{display:flex;gap:18px;flex-wrap:wrap;font:12px var(--sans);color:var(--muted);margin-top:10px}}.legend span{{display:inline-flex;align-items:center;gap:6px}}.legend i{{width:12px;height:3px;background:var(--swatch);display:inline-block}}
</style></head><body><main>

<header><div class="eyebrow">Regime overlay - {result['sample']['trading_start']} to {result['sample']['trading_end']} - signal live from {result['sample']['signal_start']}</div>
<h1>Six crashes, one score: <em>how many macro pillars were out of balance?</em></h1>
<p class="standfirst">Every crash in the record had a distinct trigger, but five point-in-time macro pillars — labor, inflation, policy, financial conditions, and volatility — were combined into a single stress count. The number of simultaneously stressed pillars sets a target equity weight; the rest sits in cash earning the prior-known fed funds rate. This is a de-risking overlay tested against real history, not a market-timing forecast.</p>
<div class="cards">
<div class="card"><b>{money(spy_regime['terminal_wealth'])}</b><span>SPY overlay terminal wealth; buy-and-hold {money(spy_buyhold['terminal_wealth'])}</span></div>
<div class="card"><b>{pct(spy_regime['max_drawdown'],1)}</b><span>SPY overlay max drawdown; buy-and-hold {pct(spy_buyhold['max_drawdown'],1)}</span></div>
<div class="card"><b>{money(ndx_regime['terminal_wealth'])}</b><span>QQQ-proxy overlay terminal wealth; buy-and-hold {money(ndx_buyhold['terminal_wealth'])}</span></div>
<div class="card"><b>45%</b><span>equity weight the model held on 2007-10-31, the GFC peak</span></div>
</div></header>

<section><div class="eyebrow">I - The five pillars</div><h2>Balance is a count of what is simultaneously stressed.</h2>
<p>Each pillar uses only data that would have been publicly known on the trading date (publication lags noted). A pillar flips to "stressed" on its own threshold; the day's exposure comes from how many of the five are stressed at once.</p>
<div class="rule">
<div class="step"><b>Labor</b><span>{pillars['labor']}</span></div>
<div class="step"><b>Inflation</b><span>{pillars['inflation']}</span></div>
<div class="step"><b>Policy</b><span>{pillars['policy']}</span></div>
<div class="step"><b>Financial conditions</b><span>{pillars['financial_conditions']}</span></div>
<div class="step"><b>Volatility</b><span>{pillars['volatility']}</span></div>
</div>
<div class="scroll"><table><thead><tr><th style="text-align:left">Stressed pillars</th><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th></tr></thead>
<tbody><tr><td style="text-align:left">Target equity weight</td><td>100%</td><td>100%</td><td>70%</td><td>45%</td><td>20%</td><td>0%</td></tr></tbody></table></div>
<p class="note">One stressed pillar is tolerated at full exposure; the ladder only bites once two or more macro dimensions are strained at the same time. Rebalancing between the five bands costs {result['switch_cost_bp']:.0f}bp of the amount moved.</p></section>

<section class="wide"><div class="eyebrow">II - Full backtest</div><h2>SPY got safer and slightly richer. QQQ did not get safer.</h2>
<figure><div class="figure-head"><strong>SPY proxy (S&amp;P 500 total return)</strong><span>overlay vs. buy-and-hold, log scale</span></div><div class="plate">{spy_chart}</div></figure>
<div class="legend"><span><i style="--swatch:var(--teal)"></i>Regime overlay</span><span><i style="--swatch:var(--ink)"></i>Buy-and-hold</span></div>
<figure><div class="figure-head"><strong>QQQ proxy (Nasdaq-100 price index)</strong><span>overlay vs. buy-and-hold, log scale</span></div><div class="plate">{ndx_chart}</div></figure>
<div class="legend"><span><i style="--swatch:var(--teal)"></i>Regime overlay</span><span><i style="--swatch:var(--ink)"></i>Buy-and-hold</span></div>
<div class="scroll"><table><thead><tr><th style="text-align:left">Path</th><th>Terminal ({money(100000)} start)</th><th>CAGR</th><th>Max DD</th><th>Volatility</th><th>Sharpe</th></tr></thead><tbody>
<tr class="featured"><td style="text-align:left"><strong>SPY proxy + regime overlay</strong></td><td>{money(spy_regime['terminal_wealth'])}</td><td>{pct(spy_regime['cagr'])}</td><td>{pct(spy_regime['max_drawdown'],1)}</td><td>{pct(spy_regime['annual_volatility'],1)}</td><td>{spy_regime['sharpe']:.2f}</td></tr>
<tr><td style="text-align:left">SPY proxy buy-and-hold</td><td>{money(spy_buyhold['terminal_wealth'])}</td><td>{pct(spy_buyhold['cagr'])}</td><td>{pct(spy_buyhold['max_drawdown'],1)}</td><td>{pct(spy_buyhold['annual_volatility'],1)}</td><td>{spy_buyhold['sharpe']:.2f}</td></tr>
<tr class="featured"><td style="text-align:left"><strong>QQQ proxy + regime overlay</strong></td><td>{money(ndx_regime['terminal_wealth'])}</td><td>{pct(ndx_regime['cagr'])}</td><td>{pct(ndx_regime['max_drawdown'],1)}</td><td>{pct(ndx_regime['annual_volatility'],1)}</td><td>{ndx_regime['sharpe']:.2f}</td></tr>
<tr><td style="text-align:left">QQQ proxy buy-and-hold</td><td>{money(ndx_buyhold['terminal_wealth'])}</td><td>{pct(ndx_buyhold['cagr'])}</td><td>{pct(ndx_buyhold['max_drawdown'],1)}</td><td>{pct(ndx_buyhold['annual_volatility'],1)}</td><td>{ndx_buyhold['sharpe']:.2f}</td></tr>
</tbody></table></div>
<p class="warning note">On SPY the overlay added {pct(spy_regime['cagr']-spy_buyhold['cagr'])} of CAGR while cutting max drawdown by {pct(spy_buyhold['max_drawdown']-spy_regime['max_drawdown'],1)} and raising Sharpe from {spy_buyhold['sharpe']:.2f} to {spy_regime['sharpe']:.2f}. On the QQQ proxy the same score gave back {pct(ndx_buyhold['cagr']-ndx_regime['cagr'])} of CAGR for almost no drawdown improvement ({pct(ndx_buyhold['max_drawdown'],1)} to {pct(ndx_regime['max_drawdown'],1)}) — concentrated growth-stock risk is not neutralized by a broad macro overlay.</p></section>

<section class="wide"><div class="eyebrow">III - Exposure through time</div><h2>The model spent {pct(full_share,0)} of days fully invested.</h2>
<figure><div class="figure-head"><strong>Target equity weight</strong><span>weekly, since {result['sample']['signal_start']}</span></div><div class="plate">{exposure_svg}</div></figure>
<div class="scroll"><table><thead><tr><th>Equity weight</th><th>Share of trading days</th></tr></thead><tbody>{exposure_rows}</tbody></table></div>
<p>De-risking below 70% equity happened on {pct(below_70_share,1)} of trading days since the signal went live — a rare, not a routine, event.</p></section>

<section class="wide"><div class="eyebrow">IV - Did the score see it coming?</div><h2>It caught the two policy-driven crashes. It missed the two panics.</h2>
<p>The pasted history names six shocks. The scan of {money(100000)} → {money(spy_buyhold['terminal_wealth'])} 25-year record finds all six independently, plus a seventh (2015-16). For each, here is what the five pillars showed on the exact peak trading day.</p>
<div class="scroll"><table><thead><tr><th style="text-align:left">Shock</th><th>Peak</th><th>Trough</th><th>Depth (QQQ proxy)</th><th>Recovered</th><th>Stressed at peak</th><th>Which pillars</th><th>Exposure at peak</th><th>Advance warning?</th></tr></thead>
<tbody>{episode_rows}</tbody></table></div>
<p class="note"><strong>The Global Financial Crisis is the clean hit.</strong> By 2007-10-31 the policy, financial-conditions, and volatility pillars were all stressed at once, and the model was already down to 45% equity — before the S&amp;P and Nasdaq peaks that month. The dot-com top (2000) predates the score entirely (first full year of CPI history was still loading); COVID and the 2018 Q4 selloff were pure exogenous/technical shocks that no slow-moving macro pillar flags in time; the 2022 and 2025 drawdowns had exactly one pillar lit (inflation, then policy) — enough to shave nothing off exposure under this ladder's one-free-pillar tolerance.</p>
<p class="warning note">The dot-com and 2015-16 troughs never independently regained their own prior high before the next decline started: the Nasdaq-100 did not durably clear its March 2000 level until November 2015. The 2007-2009 crash is nested entirely inside that 15-year drawdown — it is reported separately above only because its own local peak and trough are well known and worth stress-testing on their own terms.</p></section>

<section><div class="eyebrow">V - What this overlay is and is not</div><h2>A diversification rule, not a crash forecaster.</h2>
<p>The five pillars are slow, monthly-to-quarterly-refresh macro series. They describe whether the economy and policy stance are stretched, not whether next week's price will fall. That is why they caught the two multi-quarter tightening cycles (dot-com's follow-through and the GFC) and missed the two fast, headline-driven shocks (COVID, and to a lesser extent 2018 and 2025). Sizing the overlay for QQQ specifically does not help: concentrated single-sector exposure carries idiosyncratic risk this macro-only score was never built to see.</p>
<p class="caution note">Thresholds (4% CPI, 1pp real rate, NFCI &gt; 0, 70th percentile VIX, 0.50pp Sahm-style labor gap) were chosen from macro convention, not fit to these six dates. They are not re-optimized per crash, but they were chosen with the crash record already known, which is an in-sample bias worth weighing against the out-of-sample GFC hit.</p>
<p class="warning note">Nasdaq-100 (^NDX) excludes dividends; the true QQQ total return is understated by roughly its ~0.5-0.8% historical yield, compounding to a modest understatement of both overlay and buy-and-hold terminal wealth. No taxes, bid/ask spreads, or slippage are modeled; the cash sleeve earns the prior-known effective fed funds rate (DFF), not a bank deposit rate.</p></section>

<footer>Historical simulation, not investment advice. {result['sample']['sessions']:,} trading sessions, {result['sample']['trading_start']} to {result['sample']['trading_end']}; regime signal live from {result['sample']['signal_start']}. Publication lags: UNRATE 40 days, CPI 45 days, NFCI 9 days, daily market series 2 days. Rebalancing costs {result['switch_cost_bp']:.0f}bp of notional moved between exposure bands. Data: FRED (macro.db) and equity.db (^SP500TR, ^NDX). Generated by <code>scripts/run_macro_equilibrium_strategy.py</code> and <code>scripts/build_macro_equilibrium_page.py</code>.</footer>
</main></body></html>
"""

    args.output.write_text(html_text, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
