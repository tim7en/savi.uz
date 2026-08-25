"""Build the weekly covered-call overlay report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd

import build_spy_dca_dashboard as price_io
from build_contribution_quality_page import line_chart, money, pct


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("out/strategy/weekly_covered_calls"))
    parser.add_argument("--price-cache", type=Path, default=Path(".cache/yahoo_daily"))
    parser.add_argument("--output", type=Path, default=Path("docs/weekly-covered-call-study.html"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = json.loads((args.input / "results.json").read_text(encoding="utf-8"))
    daily = pd.read_csv(args.input / "daily.csv", parse_dates=["date"])
    trades = pd.read_csv(args.input / "trades.csv")
    latest = {
        symbol: float(price_io.load_spy(args.price_cache / f"{symbol}.json")["close"].iloc[-1])
        for symbol in ("SPY", "QQQ")
    }

    chain_rows = []
    overlay_rows = []
    for symbol in ("SPY", "QQQ"):
        for delta in (5, 10, 20):
            key = f"delta_{delta:02d}"
            chain = result["chains"][symbol][key]
            overlay = result["strategy_overlays"][symbol][key]
            whole = overlay["whole_contracts"]
            baseline = overlay["baseline"]
            chain_rows.append(
                f'<tr class="{"featured" if delta == 5 else ""}"><td><strong>{symbol} {delta}-delta</strong></td>'
                f'<td>{chain["weeks"]}</td><td>{pct(chain["worthless_rate"],1)}</td>'
                f'<td>{chain["assigned_or_itm"]}</td><td>{chain["median_premium_bp"]:.1f} bp</td>'
                f'<td>{pct(chain["premium_yield_sum"],1)}</td><td>{pct(chain["payoff_yield_sum"],1)}</td>'
                f'<td>{pct(chain["cagr_difference"],2)}</td></tr>'
            )
            overlay_rows.append(
                f'<tr class="{"featured" if delta == 5 else ""}"><td><strong>{symbol} {delta}-delta</strong></td>'
                f'<td>{money(baseline["terminal_wealth"])}</td><td>{money(whole["terminal_wealth"])}</td>'
                f'<td>{pct(whole["terminal_wealth"] / baseline["terminal_wealth"] - 1,2)}</td>'
                f'<td>{pct(baseline["xirr"])}</td><td>{pct(whole["xirr"])}</td>'
                f'<td>{money(whole["gross_premium_to_treasury"])}</td>'
                f'<td>{money(whole["upside_paid_away"])}</td><td>{money(whole["net_option_pnl"])}</td></tr>'
            )

    colors = {
        "baseline": "var(--ink)", "delta5": "var(--teal)", "delta10": "var(--red)"
    }
    charts = {}
    for symbol in ("SPY", "QQQ"):
        subset = daily[daily["symbol"] == symbol]
        base = subset[subset["delta"] == 0.05][["date", "baseline_wealth"]].rename(
            columns={"baseline_wealth": "baseline"}
        )
        d5 = subset[subset["delta"] == 0.05][["date", "whole_contract_wealth"]].rename(
            columns={"whole_contract_wealth": "delta5"}
        )
        d10 = subset[subset["delta"] == 0.10][["date", "whole_contract_wealth"]].rename(
            columns={"whole_contract_wealth": "delta10"}
        )
        frame = base.merge(d5, on="date").merge(d10, on="date")
        charts[symbol] = line_chart(frame, ["baseline", "delta5", "delta10"], colors)

    worst_rows = []
    for symbol in ("SPY", "QQQ"):
        selected = trades[(trades["symbol"] == symbol) & (trades["target_delta"] == 0.05)]
        for _, row in selected.nlargest(4, "payoff_yield").iterrows():
            worst_rows.append(
                f'<tr><td><strong>{symbol}</strong></td><td>{row["issue_date"]}</td><td>{row["expiration"]}</td>'
                f'<td>{pct(row["underlying_return"],1)}</td><td>${row["strike"]:,.2f}</td>'
                f'<td>{row["premium_yield"]*10_000:.1f} bp</td><td>{pct(row["payoff_yield"],2)}</td></tr>'
            )

    spy5 = result["chains"]["SPY"]["delta_05"]
    qqq5 = result["chains"]["QQQ"]["delta_05"]
    spy_overlay = result["strategy_overlays"]["SPY"]["delta_05"]["whole_contracts"]
    qqq_overlay = result["strategy_overlays"]["QQQ"]["delta_05"]["whole_contracts"]
    legend = ('<span><i style="--swatch:var(--ink)"></i>No calls</span>'
              '<span><i style="--swatch:var(--teal)"></i>5-delta weekly</span>'
              '<span><i style="--swatch:var(--red)"></i>10-delta weekly</span>')

    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Weekly covered-call study</title>
<style>
:root{{--bg:#edf1ed;--paper:#fbfcf9;--ink:#182321;--muted:#5c6c68;--faint:#7d8a87;--line:#cad4cf;--teal:#118578;--soft:#dceee9;--amber:#a46c00;--amberSoft:#f5ead1;--red:#b3483e;--redSoft:#f5dfdc;--mono:"Cascadia Mono",Consolas,monospace;--serif:Georgia,"Times New Roman",serif;--sans:Inter,system-ui,-apple-system,"Segoe UI",sans-serif}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.58 var(--sans)}}main{{width:min(1160px,calc(100% - 32px));margin:auto;padding:46px 0 76px}}header{{border-bottom:2px solid var(--ink);padding-bottom:30px}}.eyebrow{{font:700 11px var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--teal)}}h1{{font:600 clamp(42px,7vw,76px)/1 var(--serif);letter-spacing:-.05em;max-width:15ch;margin:12px 0 18px}}h2{{font:600 31px/1.17 var(--serif);letter-spacing:-.025em;margin:0 0 12px}}h3{{font:600 21px/1.25 var(--serif);margin:25px 0 9px}}.standfirst{{max-width:72ch;color:var(--muted);font:21px/1.45 var(--serif)}}section{{margin-top:48px}}p{{margin:0 0 16px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}}.card{{background:var(--paper);border-top:4px solid var(--teal);padding:18px}}.card.warn{{border-color:var(--amber)}}.card.stop{{border-color:var(--red)}}.card b{{display:block;font:700 24px var(--mono)}}.card span{{display:block;color:var(--muted);font-size:12px;margin-top:6px}}.note{{background:var(--soft);border-left:3px solid var(--teal);padding:17px 19px;color:var(--muted)}}.caution{{background:var(--amberSoft);border-left-color:var(--amber)}}.warning{{background:var(--redSoft);border-left-color:var(--red)}}.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}figure{{margin:22px 0}}.figureHead{{display:flex;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:8px;color:var(--muted);font-size:12px}}.plate{{background:var(--paper);border:1px solid var(--line);padding:12px;overflow:auto;margin-top:10px}}svg{{display:block;width:100%;min-width:720px}}.frame{{fill:none;stroke:var(--line)}}.grid{{stroke:var(--line)}}.axis{{font:10px var(--mono);fill:var(--faint)}}.axis-title{{font:12px var(--sans);fill:var(--ink)}}.legend{{display:flex;gap:20px;flex-wrap:wrap;padding:9px 3px 0;color:var(--muted);font-size:11px}}.legend span{{display:flex;align-items:center;gap:7px}}.legend i{{width:22px;border-top:2px solid var(--swatch)}}.scroll{{overflow:auto;border:1px solid var(--line);background:var(--paper)}}table{{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}}th,td{{padding:11px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}}th{{font:700 10px var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--faint)}}th:first-child,td:first-child{{text-align:left}}td strong{{display:block}}tr.featured{{background:var(--soft)}}.rules{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}.rule{{background:var(--paper);border-top:3px solid var(--teal);padding:15px}}.rule b{{display:block;font:700 17px var(--mono)}}.rule span{{display:block;color:var(--muted);font-size:12px;margin-top:5px}}footer{{border-top:1px solid var(--line);padding-top:18px;margin-top:50px;color:var(--faint);font-size:11px}}a{{color:var(--teal)}}code{{font-family:var(--mono)}}@media(max-width:760px){{.cards,.rules{{grid-template-columns:1fr 1fr}}.grid2{{grid-template-columns:1fr}}}}@media(max-width:480px){{.cards,.rules{{grid-template-columns:1fr}}}}
</style></head><body><main>
<header><div class="eyebrow">Actual option-chain test · {result['sample']['start']} to {result['sample']['end']}</div><h1>Worthless 95% of the time. Not free money.</h1><p class="standfirst">Far-out-of-the-money weekly calls usually expired harmlessly, but their premiums were tiny. Moving premium to Treasury worked only at approximately 5 delta in this sample; selling closer strikes repeatedly surrendered too much rebound upside.</p></header>
<div class="cards"><article class="card"><b>{pct(spy5['worthless_rate'],1)}</b><span>SPY 5-delta calls expired worthless; {spy5['assigned_or_itm']} upside-cap weeks</span></article><article class="card"><b>{pct(qqq5['worthless_rate'],1)}</b><span>QQQ 5-delta calls expired worthless; {qqq5['assigned_or_itm']} upside-cap weeks</span></article><article class="card warn"><b>{spy5['median_premium_bp']:.1f} bp</b><span>median weekly SPY premium; QQQ {qqq5['median_premium_bp']:.1f} bp</span></article><article class="card stop"><b>100 shares</b><span>required by one standard ETF option contract</span></article></div>

<section><div class="eyebrow">I · Is it realistic now?</div><h2>Not with the present $10,000 account.</h2><p>One standard covered call requires 100 shares. At the latest local closes, one fully covered SPY contract needs approximately {money(latest['SPY']*100)} of SPY and one QQQ contract needs {money(latest['QQQ']*100)} of QQQ. A $10,000 account holds only about {10_000/latest['SPY']:.1f} SPY shares or {10_000/latest['QQQ']:.1f} QQQ shares.</p><p class="warning note"><strong>Do not solve the sizing problem with naked calls.</strong> XSP, call spreads, or broker-specific fractional option products are different strategies with basis, liquidity, settlement, and tail-risk differences. They should not be described as ordinary covered calls.</p></section>

<section><div class="eyebrow">II · Actual weekly calls</div><h2>“Target worthless” means accepting almost no premium.</h2><p>Calls were sold at the historical bid on the last quoted session of each week, using the closest next-week expiration and strike nearest the requested delta. Intrinsic value at expiration is the upside surrendered. A 2 bp stock turnover allowance is charged when the option expires in the money.</p><div class="scroll"><table><thead><tr><th>Underlying / target</th><th>Weeks</th><th>Worthless</th><th>ITM weeks</th><th>Median premium</th><th>Premium sum</th><th>Upside paid</th><th>CAGR effect</th></tr></thead><tbody>{''.join(chain_rows)}</tbody></table></div><p class="caution note" style="margin-top:15px"><strong>The trade-off is nonlinear:</strong> moving from 5 to 10 delta roughly doubled the frequency of upside caps, while the larger premiums still failed to compensate for sharp rebound weeks. At 20 delta, more than one quarter of weekly calls finished in the money.</p></section>

<section><div class="eyebrow">III · Applied to the two strategies</div><h2>Only unleveraged index exposure was covered.</h2><div class="scroll"><table><thead><tr><th>Strategy overlay</th><th>Baseline terminal</th><th>With calls</th><th>Difference</th><th>Base XIRR</th><th>Call XIRR</th><th>Gross premium</th><th>Upside paid</th><th>Net option P&amp;L</th></tr></thead><tbody>{''.join(overlay_rows)}</tbody></table></div><p>At 5 delta, whole-contract execution increased the SPY strategy endpoint by {pct(spy_overlay['terminal_wealth']/result['strategy_overlays']['SPY']['delta_05']['baseline']['terminal_wealth']-1,2)} and the QQQ endpoint by {pct(qqq_overlay['terminal_wealth']/result['strategy_overlays']['QQQ']['delta_05']['baseline']['terminal_wealth']-1,2)}. That is modest evidence—not a stable expected return estimate from one 7¾-year regime.</p><p class="warning note"><strong>Gross premium is not spendable profit.</strong> The SPY sleeve received {money(spy_overlay['gross_premium_to_treasury'])} of premium but surrendered {money(spy_overlay['upside_paid_away'])} of upside plus assignment costs. QQQ received {money(qqq_overlay['gross_premium_to_treasury'])} and surrendered {money(qqq_overlay['upside_paid_away'])}. Treasury accounting must retain enough liquidity for assignment or repurchase.</p></section>

<section><div class="eyebrow">IV · Equity curves</div><h2>Ten delta visibly drags both recovery paths.</h2><div class="grid2"><figure><div class="figureHead"><strong>SPY dual-guard strategy</strong><span>whole contracts</span></div><div class="plate">{charts['SPY']}<div class="legend">{legend}</div></div></figure><figure><div class="figureHead"><strong>QQQ dashboard guard</strong><span>whole contracts</span></div><div class="plate">{charts['QQQ']}<div class="legend">{legend}</div></div></figure></div></section>

<section><div class="eyebrow">V · When did we miss upside?</div><h2>The damage is concentrated in sudden rebounds.</h2><p>A 5-delta call capped upside {spy5['assigned_or_itm']} times in 395 SPY weeks and {qqq5['assigned_or_itm']} times in 395 QQQ weeks—roughly once every {395/spy5['assigned_or_itm']:.0f} and {395/qqq5['assigned_or_itm']:.0f} weeks. These were the largest misses:</p><div class="scroll"><table><thead><tr><th>Asset</th><th>Sold</th><th>Expired</th><th>Underlying week</th><th>Strike</th><th>Premium</th><th>Upside surrendered</th></tr></thead><tbody>{''.join(worst_rows)}</tbody></table></div><p>The worst SPY miss followed the April 3, 2020 sale: SPY rallied about 12.1% in the shortened week, while the 5-delta premium was about 9.7 bp and the call surrendered 4.11% of notional. One such rebound consumed roughly forty-two median SPY premiums.</p></section>

<section><div class="eyebrow">VI · Operating rule if tested live</div><h2>Five delta only—and treat it as experimental.</h2><div class="rules"><article class="rule"><b>Eligibility</b><span>Sell only against a fully paid, unleveraged 100-share index lot. Never count leveraged shares twice.</span></article><article class="rule"><b>Strike</b><span>Closest listed weekly call near 5 delta; skip the week if bid, spread, or strike quality is poor.</span></article><article class="rule"><b>Size</b><span>Cover no more than 25–50% of eligible shares initially; keep the rest uncapped for rebounds.</span></article><article class="rule"><b>Premium</b><span>Credit gross premium to Treasury, but separately reserve expected assignment/repurchase liquidity.</span></article><article class="rule"><b>No rolling rescue</b><span>Do not buy back a losing call merely to avoid assignment unless that rule is independently tested.</span></article><article class="rule"><b>Stop condition</b><span>Review after 52 trades; compare net option P&amp;L and total portfolio return, not win rate or premium alone.</span></article></div><p class="note" style="margin-top:15px"><strong>My conclusion:</strong> a small 5-delta overlay can be realistic once the account owns full 100-share lots, but it is a minor Treasury-harvesting sleeve—not the engine of the strategy. Ten- and twenty-delta weekly overwriting are rejected by this sample.</p></section>

<footer>Historical simulation, not investment advice. Actual chain fields come from the local Alpha Vantage historical-options database; calls are sold at bid and held to intrinsic settlement. Standard contract deliverables and covered-call mechanics: <a href="https://www.optionseducation.org/strategies/all-strategies/covered-call-buy-write">Options Industry Council</a>. QQQ fund information: <a href="https://www.invesco.com/qqq-etf/en/home.html">Invesco</a>; SPY: <a href="https://www.ssga.com/us/en/intermediary/etfs/state-street-spdr-sp-500-etf-trust-spy">State Street</a>. American early exercise, ex-dividend assignment, tax, commissions and recursive strategy changes are omitted. Generated by <code>scripts/run_weekly_covered_call_study.py</code>.</footer>
</main></body></html>'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
