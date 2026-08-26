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
    parser.add_argument("--valuation-input", type=Path, default=Path("out/strategy/valuation_regression"))
    parser.add_argument("--pca-input", type=Path, default=Path("out/strategy/macro_pca"))
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


PC_LABELS = {
    "PC1": "Policy cycle &amp; curve shape",
    "PC2": "Inflation &amp; liquidity regime",
    "PC3": "Real rates &amp; financial conditions",
    "PC4": "Labor-market shocks",
    "PC5": "Residual",
}


def scree_chart(ratios: list[float]) -> str:
    width, height = 560, 220
    left, right, top, bottom = 50, 20, 16, 36
    plot_w, plot_h = width - left - right, height - top - bottom
    n = len(ratios)
    bar_w = plot_w / n * 0.6
    high = 0.5
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Explained variance by component">',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" class="frame"/>',
    ]
    for tick in (0, 0.25, 0.5):
        yy = top + (high - tick) / high * plot_h
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{yy:.1f}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" class="axis">{tick:.0%}</text>')
    for i, ratio in enumerate(ratios):
        cx = left + plot_w * (i + 0.5) / n
        bar_h = min(ratio, high) / high * plot_h
        parts.append(f'<rect x="{cx-bar_w/2:.1f}" y="{top+plot_h-bar_h:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="var(--teal)"/>')
        parts.append(f'<text x="{cx:.1f}" y="{height-14}" text-anchor="middle" class="axis">PC{i+1}</text>')
    parts.append('</svg>')
    return "".join(parts)


def exposure_chart(frame: pd.DataFrame) -> str:
    width, height = 980, 260
    left, right, top, bottom = 60, 24, 20, 40
    plot_w, plot_h = width - left - right, height - top - bottom
    sampled = frame.set_index("date")["target_exposure"].resample("W").last().ffill().dropna()
    ticks = [0.0, 0.30, 0.65, 1.0, 1.5, 2.0]
    high = 2.0
    start, end = sampled.index[0], sampled.index[-1]
    span = max((end - start).days, 1)
    x = lambda stamp: left + (stamp - start).days / span * plot_w
    y = lambda v: top + (high - v) / high * plot_h
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Target equity exposure over time">',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" class="frame"/>',
    ]
    for tick in ticks:
        yy = y(tick)
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{yy:.1f}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" class="axis">{tick:.0%}</text>')
    leverage_y = y(1.0)
    parts.append(f'<line x1="{left}" x2="{width-right}" y1="{leverage_y:.1f}" y2="{leverage_y:.1f}" class="target"/>')
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
    valuation = json.loads((args.valuation_input / "results.json").read_text(encoding="utf-8"))
    pca_result = json.loads((args.pca_input / "results.json").read_text(encoding="utf-8"))

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
    leveraged_share = sum(share for level, share in time_at_exposure if level > 1.0)
    unlevered_share = sum(share for level, share in time_at_exposure if level == 1.0)
    derisked_share = sum(share for level, share in time_at_exposure if level < 1.0)
    max_tier_share = time_at_exposure[0][1]
    nested_gfc = result["nested_2008_episode"]

    pillars = result["pillars"]

    uw = valuation["user_window"]
    fh = valuation["full_shiller_history"]
    ref = valuation["reference_result_pasted_by_user"]

    def reg_row(label, fit, extra_label=None):
        coefs = fit["coefficients"]
        coef_text = ", ".join(f"{c:+.3f}" for c in coefs)
        return (
            f"<tr><td style=\"text-align:left\">{label}</td><td>{fit['n']}</td>"
            f"<td>{fit['intercept']:.2f}</td><td>{coef_text}</td>"
            f"<td>{pct(fit['r2'],1)}</td><td>{pct(fit['adjusted_r2'],1)}</td><td>{fit['rmse']:.2f}pp</td></tr>"
        )

    valuation_rows = "".join([
        reg_row("CAPE only (our replication, 1946-2013)", uw["cape_only"]),
        reg_row("Excess CAPE Yield (our replication, 1946-2013)", uw["excess_cape_yield"]),
        reg_row("ECY + dividend retention rate (our adaptation)", uw["ecy_plus_retention_rate"]),
        reg_row("ECY + earnings-cycle z-score (our adaptation)", uw["ecy_plus_earnings_cycle"]),
        reg_row("CAPE only (full Shiller history, 1881-2014)", fh["cape_only"]),
        reg_row("Excess CAPE Yield (full Shiller history, 1881-2014)", fh["excess_cape_yield"]),
    ])

    oos = uw["ecy_expanding_window_oos"]

    pca_sample = pca_result["sample"]
    pc_ratios = pca_result["explained_variance_ratio_top"]
    scree_svg = scree_chart(pc_ratios)
    pc_rows = "".join(
        f"<tr><td style=\"text-align:left\">{label['component']} — {PC_LABELS.get(label['component'],'')}</td>"
        f"<td>{pct(label['explained_variance'],1)}</td>"
        f"<td style=\"text-align:left;white-space:normal\">"
        + ", ".join(
            f"{name.replace('_',' ')} ({value:+.2f})"
            for name, value in sorted(
                ((k, v) for k, v in label.items() if k not in ("component", "explained_variance")),
                key=lambda kv: -abs(kv[1]),
            )[:4]
        )
        + "</td></tr>"
        for label in pca_result["loadings"]
    )

    def pca_reg_row(label, key):
        row = pca_result["regressions"][key]
        return (
            f"<tr><td style=\"text-align:left\">{label}</td>"
            f"<td>{pct(row['pc1_only']['r2'],1)}</td>"
            f"<td>{pct(row['all_5_pcs']['r2'],1)}</td></tr>"
        )

    pca_reg_rows = "".join([
        pca_reg_row("SPY drawdown (contemporaneous)", "spy_drawdown_contemporaneous"),
        pca_reg_row("QQQ proxy drawdown (contemporaneous)", "ndx_drawdown_contemporaneous"),
        pca_reg_row("SPY return, next 1 month", "spy_return_fwd_1m"),
        pca_reg_row("QQQ proxy return, next 1 month", "ndx_return_fwd_1m"),
        pca_reg_row("SPY return, next 12 months", "spy_return_fwd_12m"),
        pca_reg_row("QQQ proxy return, next 12 months", "ndx_return_fwd_12m"),
    ])

    extra_sections = f"""
<section class="wide"><div class="eyebrow">VI - Valuation regime</div><h2>CAPE's rate adjustment replicates almost exactly. Profitability adds little, and for a reason.</h2>
<p>A user-supplied first pass regressed subsequent 10-year real S&amp;P 500 returns on CAPE, then on Excess CAPE Yield (ECY = 1/CAPE minus the real 10-year rate), then added aggregate corporate ROE. Reported: CAPE alone R²=39.0%, ECY R²=52.3%, ECY+ROE R²=55.1% — with a <em>negative</em> ROE coefficient, because current aggregate profitability tends to mean-revert rather than persist. Re-running the same January-observation regressions on our own copy of Shiller's dataset reproduces those first two numbers almost exactly.</p>
<div class="scroll"><table><thead><tr><th style="text-align:left">Specification</th><th>N</th><th>Intercept</th><th>Coefficient(s)</th><th>R²</th><th>Adj. R²</th><th>RMSE</th></tr></thead>
<tbody>{valuation_rows}</tbody></table></div>
<p class="note">We don't hold the long-run (1946+) FRED aggregate-ROE series locally, so the third regression isn't replicated as-is. Using only Shiller's own dividend/earnings columns, the same story shows up anyway: adding an earnings-cycle z-score (today's real earnings versus their own trailing 10-year average) lifts R² only marginally (52.3% → {pct(uw['ecy_plus_earnings_cycle']['r2'],1)}) and its coefficient is <strong>negative</strong> ({uw['ecy_plus_earnings_cycle']['coefficients'][1]:+.2f}) — elevated aggregate profitability, like elevated aggregate ROE, predicts <em>lower</em> subsequent 10-year real returns, not higher. Both proxies point at the same mean-reversion mechanism the user's ROE result surfaced.</p>
<p class="warning note">The full 1881-2014 Shiller history tells a weaker version of the same story: CAPE alone only reaches R²={pct(fh['cape_only']['r2'],1)} and ECY R²={pct(fh['excess_cape_yield']['r2'],1)} — well below the 1946-2013 window's 39.0%/52.3%. The rate-adjustment edge is concentrated in the modern (post-1946, and especially post-1980s rate-cycle) sample, not a constant of market history.</p>
<p class="caution note"><strong>The stated statistical caveat holds up.</strong> An expanding-window out-of-sample refit of the ECY regression drops R² from 52.3% in-sample to {pct(oos['oos_r2'],1)} out-of-sample — but the Spearman rank correlation between predicted and realized 10-year returns stays at {oos['rank_correlation']:.2f}. ECY quintile 1 (least attractive) averaged {pct(uw['ecy_quintiles'][0]['mean_forward_10y_real_return'],1)} annualized real return over the next decade; quintile 5 (most attractive) averaged {pct(uw['ecy_quintiles'][4]['mean_forward_10y_real_return'],1)}. This is a ranking tool, not a precise forecaster — as the original analysis said, and 68 mostly-overlapping 10-year windows overstate the effective sample size for any of these R² figures.</p></section>

<section class="wide"><div class="eyebrow">VII - How much of SPY/QQQ risk is macro?</div><h2>A principal-components view: most of drawdown, little of forward return.</h2>
<p>Eleven monthly macro series ({', '.join(name.replace('_',' ') for name in pca_sample['macro_variables'])}) were standardized and reduced to five principal components, covering {pca_sample['start']} to {pca_sample['end']} ({pca_sample['months']} months — the real-yield series TIPS/DFII10 sets the start date).</p>
<figure><div class="figure-head"><strong>Explained variance by component</strong><span>share of the 11-variable macro panel</span></div><div class="plate">{scree_svg}</div></figure>
<div class="scroll"><table><thead><tr><th style="text-align:left">Component</th><th>Variance explained</th><th style="text-align:left">Dominant loadings</th></tr></thead><tbody>{pc_rows}</tbody></table></div>
<p>The top three components (policy cycle, inflation/liquidity, real rates &amp; financial conditions) already cover {pct(sum(pc_ratios[:3]),1)} of macro-panel variance; the labor-shock and residual components add little.</p>
<div class="scroll"><table><thead><tr><th style="text-align:left">Target</th><th>R² (PC1 only)</th><th>R² (all 5 PCs)</th></tr></thead><tbody>{pca_reg_rows}</tbody></table></div>
<p class="warning note"><strong>The drawdown R² is partly circular.</strong> VIX and the Baa-10Y credit spread are themselves market prices of the same equity stress being measured, so a {pct(pca_result['regressions']['spy_drawdown_contemporaneous']['all_5_pcs']['r2'],0)} contemporaneous R² for SPY drawdown mostly confirms that risk gauges move together, not that slow macro data caused or forecast the drawdown. The more informative numbers are the <em>forward</em>-return regressions: five macro components explain only {pct(pca_result['regressions']['spy_return_fwd_1m']['all_5_pcs']['r2'],1)} of next-month SPY returns and {pct(pca_result['regressions']['spy_return_fwd_12m']['all_5_pcs']['r2'],1)} of next-12-month SPY returns (QQQ proxy: {pct(pca_result['regressions']['ndx_return_fwd_1m']['all_5_pcs']['r2'],1)} and {pct(pca_result['regressions']['ndx_return_fwd_12m']['all_5_pcs']['r2'],1)}). PC1 alone — the single biggest macro factor — explains almost none of either horizon on its own; the explanatory power only shows up once all five components are combined.</p>
<p class="note">Read together with Section VI: valuation (ECY) explains roughly half of 10-year forward variance on its own, while the broader macro panel here explains under a fifth of 12-month forward variance. Slow-moving macro conditions say much more about whether a drawdown is severe once it is underway than about when the next one starts or how large next year's return will be.</p></section>
"""


    html_text = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Macro Equilibrium Overlay - backtest</title>
<style>
:root{{--bg:#f1f4f2;--paper:#fcfdfc;--ink:#14201d;--muted:#53635f;--faint:#778681;--line:#ccd7d3;--teal:#087f72;--blue:#356da6;--amber:#9a6a00;--soft:#dcefea;--brick:#b84138;--warn:#f5dfdc;--gold:#f5ecd0;--mono:"Cascadia Mono",Consolas,monospace;--serif:Georgia,"Times New Roman",serif;--sans:system-ui,-apple-system,"Segoe UI",sans-serif}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:17px/1.62 var(--serif)}}main{{width:min(1100px,calc(100% - 34px));margin:auto;padding:52px 0 80px}}header{{border-bottom:2px solid var(--ink);padding-bottom:34px}}.eyebrow{{font:700 11px var(--mono);letter-spacing:.13em;text-transform:uppercase;color:var(--teal)}}h1{{font:650 clamp(40px,7vw,72px)/1.02 var(--serif);letter-spacing:-.045em;max-width:16ch;margin:13px 0 20px}}h1 em{{color:var(--teal)}}.standfirst{{font-size:22px;line-height:1.43;color:var(--muted);max-width:66ch}}section{{padding-top:48px;max-width:880px}}section.wide{{max-width:none}}h2{{font:650 31px/1.18 var(--serif);letter-spacing:-.025em;margin:0 0 14px}}h3{{font:650 21px/1.3 var(--serif);margin:30px 0 10px}}p{{margin:0 0 18px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);margin-top:28px}}.card{{background:var(--paper);padding:17px;border-right:1px solid var(--line)}}.card:last-child{{border:0}}.card b{{display:block;font:700 22px var(--mono)}}.card span{{display:block;font:12px/1.4 var(--sans);color:var(--faint);margin-top:7px}}.rule{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:23px 0}}.step{{background:var(--paper);border-top:3px solid var(--teal);padding:15px}}.step:nth-child(even){{border-color:var(--blue)}}.step b{{display:block;font:700 15px var(--mono)}}.step span{{font:12px/1.45 var(--sans);color:var(--faint)}}.note{{background:var(--soft);border-left:3px solid var(--teal);padding:18px 20px;color:var(--muted)}}.warning{{background:var(--warn);border-left-color:var(--brick)}}.caution{{background:var(--gold);border-left-color:var(--amber)}}.scroll{{overflow:auto;border:1px solid var(--line);background:var(--paper);margin-top:22px}}table{{border-collapse:collapse;width:100%;font:13px/1.4 var(--sans);font-variant-numeric:tabular-nums}}th,td{{padding:12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}}th{{font:700 10px var(--mono);text-transform:uppercase;letter-spacing:.06em;color:var(--faint)}}tbody tr:last-child td{{border-bottom:0}}tr.featured td{{background:var(--soft);font-weight:700}}figure{{margin:22px 0 0}}.figure-head{{display:flex;justify-content:space-between;align-items:baseline;font:13px var(--sans);color:var(--faint);margin-bottom:8px}}.figure-head strong{{color:var(--ink);font-size:15px}}.plate{{border:1px solid var(--line);background:var(--paper);padding:8px}}.frame{{fill:none;stroke:var(--line)}}.grid{{stroke:var(--line);stroke-dasharray:2 3}}.target{{stroke:var(--brick);stroke-dasharray:5 4}}.axis{{font:10px var(--mono);fill:var(--faint)}}.axis-title{{font:11px var(--sans);fill:var(--faint)}}.legend{{display:flex;gap:18px;flex-wrap:wrap;font:12px var(--sans);color:var(--muted);margin-top:10px}}.legend span{{display:inline-flex;align-items:center;gap:6px}}.legend i{{width:12px;height:3px;background:var(--swatch);display:inline-block}}
</style></head><body><main>

<header><div class="eyebrow">Regime overlay - {result['sample']['trading_start']} to {result['sample']['trading_end']} - signal live from {result['sample']['signal_start']}</div>
<h1>Six crashes, one score: <em>how much leverage does balance earn?</em></h1>
<p class="standfirst">Five point-in-time macro pillars — labor, inflation, policy, financial conditions, and volatility — are combined into a single stress count. A fully balanced regime (0-1 stressed pillars) now borrows to 1.5x-2x; each additional stressed pillar steps exposure down, to 0% at 5/5. Borrowed notional costs DFF + {result['borrow_spread_bp']:.0f}bp; de-risked cash earns DFF flat. Leverage raises the return and Sharpe edge substantially — and raises max drawdown right along with it.</p>
<div class="cards">
<div class="card"><b>{money(spy_regime['terminal_wealth'])}</b><span>SPY overlay terminal wealth; buy-and-hold {money(spy_buyhold['terminal_wealth'])}</span></div>
<div class="card"><b>{pct(spy_regime['max_drawdown'],1)}</b><span>SPY overlay max drawdown; buy-and-hold {pct(spy_buyhold['max_drawdown'],1)}</span></div>
<div class="card"><b>{money(ndx_regime['terminal_wealth'])}</b><span>QQQ-proxy overlay terminal wealth; buy-and-hold {money(ndx_buyhold['terminal_wealth'])}</span></div>
<div class="card"><b>{pct(nested_gfc['exposure_at_peak'],0)}</b><span>equity weight the model held on 2007-10-31, the GFC peak</span></div>
</div></header>

<section><div class="eyebrow">I - The five pillars</div><h2>Balance is a count of what is simultaneously stressed.</h2>
<p>Each pillar uses only data that would have been publicly known on the trading date (publication lags noted). A pillar flips to "stressed" on its own threshold; the day's exposure — which can now exceed 100% — comes from how many of the five are stressed at once.</p>
<div class="rule">
<div class="step"><b>Labor</b><span>{pillars['labor']}</span></div>
<div class="step"><b>Inflation</b><span>{pillars['inflation']}</span></div>
<div class="step"><b>Policy</b><span>{pillars['policy']}</span></div>
<div class="step"><b>Financial conditions</b><span>{pillars['financial_conditions']}</span></div>
<div class="step"><b>Volatility</b><span>{pillars['volatility']}</span></div>
</div>
<div class="scroll"><table><thead><tr><th style="text-align:left">Stressed pillars</th><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th></tr></thead>
<tbody><tr><td style="text-align:left">Target equity weight</td><td>200%</td><td>150%</td><td>100%</td><td>65%</td><td>30%</td><td>0%</td></tr></tbody></table></div>
<p class="note">A fully balanced regime borrows to 2x; one stressed pillar still borrows, at 1.5x. The ladder only drops to plain, unlevered 100% once two pillars are stressed at once, and only reaches cash at all five. Rebalancing between bands costs {result['switch_cost_bp']:.0f}bp of the amount moved; borrowed notional is charged DFF + {result['borrow_spread_bp']:.0f}bp.</p></section>

<section class="wide"><div class="eyebrow">II - Full backtest</div><h2>Leverage bought return and Sharpe. It also bought drawdown.</h2>
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
<p class="warning note">Leverage adds {pct(spy_regime['cagr']-spy_buyhold['cagr'])} of CAGR to SPY and {pct(ndx_regime['cagr']-ndx_buyhold['cagr'])} to the QQQ proxy, raising Sharpe on both (SPY {spy_buyhold['sharpe']:.2f}→{spy_regime['sharpe']:.2f}, QQQ proxy {ndx_buyhold['sharpe']:.2f}→{ndx_regime['sharpe']:.2f}). The cost: max drawdown gets <em>worse</em> than buy-and-hold on both — SPY {pct(spy_buyhold['max_drawdown'],1)}→{pct(spy_regime['max_drawdown'],1)}, QQQ proxy {pct(ndx_buyhold['max_drawdown'],1)}→{pct(ndx_regime['max_drawdown'],1)} — because the ladder is leveraged up precisely when pillars look calm, and calm does not mean safe.</p></section>

<section class="wide"><div class="eyebrow">III - Exposure through time</div><h2>The model was leveraged {pct(leveraged_share,0)} of the time.</h2>
<figure><div class="figure-head"><strong>Target equity weight</strong><span>weekly, since {result['sample']['signal_start']}; dashed line = 1x (unlevered)</span></div><div class="plate">{exposure_svg}</div></figure>
<div class="scroll"><table><thead><tr><th>Equity weight</th><th>Share of trading days</th></tr></thead><tbody>{exposure_rows}</tbody></table></div>
<p>Above 1x (borrowing) accounts for {pct(leveraged_share,1)} of trading days since the signal went live; plain unlevered 100% for {pct(unlevered_share,1)}; below 1x (de-risked) for only {pct(derisked_share,1)}. This ladder is a leverage strategy that occasionally de-risks, not a de-risking strategy that occasionally leans in.</p></section>

<section class="wide"><div class="eyebrow">IV - Did the score see it coming?</div><h2>It de-risked into the GFC. It was maximally leveraged into COVID and 2018.</h2>
<p>The pasted history names six shocks. The scan of the full record finds all six independently, plus a seventh (2015-16). For each, here is what the five pillars showed, and how much the model had borrowed, on the exact peak trading day.</p>
<div class="scroll"><table><thead><tr><th style="text-align:left">Shock</th><th>Peak</th><th>Trough</th><th>Depth (QQQ proxy)</th><th>Recovered</th><th>Stressed at peak</th><th>Which pillars</th><th>Exposure at peak</th><th>Advance warning?</th></tr></thead>
<tbody>{episode_rows}</tbody></table></div>
<p class="note"><strong>The Global Financial Crisis is the only genuine advance warning in the sample.</strong> By 2007-10-31 the policy, financial-conditions, and volatility pillars were all stressed at once, and the model had already been de-risked to {pct(nested_gfc['exposure_at_peak'],0)} for 139 consecutive sessions — before the S&amp;P and Nasdaq peaks that month. Every other shock shows the model at or above 100% at the peak, and two of them — COVID and the 2018 Q4 selloff — show it at the maximum <strong>200%</strong>: zero pillars were stressed on the day each crash began, so the ladder had borrowed all the way up right before the fastest declines in the sample. 2015-16, 2022, and 2025 each had exactly one pillar lit (financial conditions, then inflation, then policy) and were still leveraged 150% at their peaks.</p>
<p class="warning note">The dot-com and 2015-16 troughs never independently regained their own prior high before the next decline started: the Nasdaq-100 did not durably clear its March 2000 level until November 2015. The 2007-2009 crash is nested entirely inside that 15-year drawdown — it is reported separately above only because its own local peak and trough are well known and worth stress-testing on their own terms.</p></section>

<section><div class="eyebrow">V - What this overlay is and is not</div><h2>A leverage rule that trusts calm macro data more than it should.</h2>
<p>The five pillars are slow, monthly-to-quarterly-refresh macro series. They describe whether the economy and policy stance are stretched, not whether next week's price will fall — and "not stretched" is not the same claim as "safe to lever to 2x." That distinction is exactly what COVID and the 2018 Q4 selloff expose: both were fast, headline-driven shocks with zero pillars stressed at the top, and the ladder's response to "nothing looks wrong" was maximum leverage, not caution. Sizing the overlay for QQQ specifically does not help either: concentrated single-sector exposure carries idiosyncratic risk this macro-only score was never built to see.</p>
<p class="caution note">Thresholds (4% CPI, 1pp real rate, NFCI &gt; 0, 70th percentile VIX, 0.50pp Sahm-style labor gap) and the leverage ladder itself (2x/1.5x/1x/0.65x/0.3x/0x) were chosen from macro convention and the repo's other drawdown-ladder strategies, not fit to these six dates. They are not re-optimized per crash, but they were chosen with the crash record already known, which is an in-sample bias worth weighing against the out-of-sample GFC hit.</p>
<p class="warning note">Nasdaq-100 (^NDX) excludes dividends; the true QQQ total return is understated by roughly its ~0.5-0.8% historical yield, compounding to a modest understatement of both overlay and buy-and-hold terminal wealth. No taxes, bid/ask spreads, margin-call liquidation, or changing broker leverage limits are modeled; the cash sleeve earns the prior-known effective fed funds rate (DFF) flat, and borrowed notional pays DFF + {result['borrow_spread_bp']:.0f}bp.</p></section>
{extra_sections}
<footer>Historical simulation, not investment advice. {result['sample']['sessions']:,} trading sessions, {result['sample']['trading_start']} to {result['sample']['trading_end']}; regime signal live from {result['sample']['signal_start']}. Publication lags: UNRATE 40 days, CPI 45 days, NFCI 9 days, daily market series 2 days. Rebalancing costs {result['switch_cost_bp']:.0f}bp of notional moved between exposure bands; borrowed notional (exposure above 1x) is charged DFF + {result['borrow_spread_bp']:.0f}bp. Data: FRED (macro.db) and equity.db (^SP500TR, ^NDX, Shiller monthly). Generated by <code>scripts/run_macro_equilibrium_strategy.py</code>, <code>scripts/run_valuation_regime_regression.py</code>, <code>scripts/run_macro_pca_explained_variance.py</code> and <code>scripts/build_macro_equilibrium_page.py</code>.</footer>
</main></body></html>
"""

    args.output.write_text(html_text, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
