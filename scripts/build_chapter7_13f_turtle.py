"""Render Chapter Seven from the frozen full-panel 13F Turtle result."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path,
                        default=Path("out/strategy/13f_turtle_full_panel.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("docs/chapter-07-when-six-slots-are-scarce.html"))
    return parser.parse_args(argv)


def median(row: dict, name: str) -> float:
    return float(row[name]["median"])


def calmar(row: dict) -> float:
    return median(row, "cagr") / abs(median(row, "max_drawdown_exit_marked"))


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def number(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def row(label: str, item: dict, note: str = "") -> str:
    return f"""<tr>
  <td><strong>{html.escape(label)}</strong><small>{html.escape(note)}</small></td>
  <td>{int(median(item, 'trades')):,}</td>
  <td>{number(median(item, 'total_r'))}</td>
  <td>{number(median(item, 'profit_factor'), 3)}</td>
  <td>{pct(median(item, 'cagr'))}</td>
  <td>{pct(median(item, 'max_drawdown_exit_marked'))}</td>
  <td>{number(calmar(item), 3)}</td>
</tr>"""


def bars(rows: list[tuple[str, float, float]], title: str) -> str:
    width, height, left, right, top, bottom = 880, 292, 218, 44, 44, 48
    inner_width = width - left - right
    maximum = max(value for _, value, _ in rows) * 1.12
    row_height = 48
    labels = []
    marks = []
    grid = []
    for index in range(5):
        value = maximum * index / 4
        x = left + inner_width * index / 4
        grid.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{height - bottom}" class="grid"/>'
            f'<text x="{x:.1f}" y="{height - 19}" text-anchor="middle" class="axis">{value:.2f}</text>'
        )
    for index, (label, value, accent) in enumerate(rows):
        y = top + index * row_height + 10
        bar_width = inner_width * value / maximum
        labels.append(f'<text x="{left - 14}" y="{y + 17}" text-anchor="end" class="label">{html.escape(label)}</text>')
        marks.append(
            f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="25" rx="3" fill="{accent}"/>'
            f'<text x="{left + bar_width + 9:.1f}" y="{y + 17}" class="value">{value:.3f}</text>'
        )
    return f"""<figure>
  <div class="fig-head"><span>{html.escape(title)}</span><span>higher is better</span></div>
  <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
    <style>
      .grid{{stroke:var(--rule);stroke-width:1}} .axis{{font:11px var(--mono);fill:var(--ink-3)}}
      .label{{font:600 12px var(--sans);fill:var(--ink-2)}} .value{{font:600 12px var(--mono);fill:var(--ink)}}
    </style>{''.join(grid)}{''.join(labels)}{''.join(marks)}
  </svg>
</figure>"""


def render(result: dict) -> str:
    full = result["full"]
    validation = result["validation_2023_plus"]
    priority = result["conviction_priority"]
    tilts = result["conviction_tilt"]
    tilt_2 = tilts["max_2.00x"]
    scope = result["scope"]

    full_chart = bars([
        ("Baseline", calmar(full["baseline"]), "#506276"),
        ("Hard 13F gate", calmar(full["13f_watchlist"]), "#A6463B"),
        ("Conviction priority", calmar(priority["full"]), "#007F73"),
        ("2x risk cap", calmar(tilt_2["full"]), "#B76718"),
    ], "Full-period Calmar ratio")
    validation_chart = bars([
        ("Baseline", calmar(validation["baseline"]), "#506276"),
        ("Hard 13F gate", calmar(validation["13f_watchlist"]), "#A6463B"),
        ("Conviction priority", calmar(priority["validation_2023_plus"]), "#007F73"),
        ("2x risk cap", calmar(tilt_2["validation_2023_plus"]), "#B76718"),
    ], "2023+ validation Calmar ratio")

    full_rows = "\n".join([
        row("Baseline", full["baseline"], "Random same-day tie breaks"),
        row("Hard 13F gate", full["13f_watchlist"], "Only active high-conviction windows"),
        row("Conviction priority", priority["full"], "All breakouts; conviction gets contested slots"),
        row("2x risk cap", tilt_2["full"], "All breakouts; max tilt at 10% conviction"),
    ])
    validation_rows = "\n".join([
        row("Baseline", validation["baseline"], "Random same-day tie breaks"),
        row("Hard 13F gate", validation["13f_watchlist"], "Only active high-conviction windows"),
        row("Conviction priority", priority["validation_2023_plus"], "All breakouts; conviction gets contested slots"),
        row("2x risk cap", tilt_2["validation_2023_plus"], "All breakouts; max tilt at 10% conviction"),
    ])

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chapter Seven: When Six Slots Are Scarce</title>
<style>
  :root {{
    --ground:#F2F4F3; --surface:#FBFCFB; --ink:#141C1B; --ink-2:#3C4A47;
    --ink-3:#71807C; --rule:#D3DAD8; --teal:#007F73; --amber:#B76718;
    --brick:#A6463B; --slate:#506276; --teal-soft:#DDEEEA; --amber-soft:#F5E7D6;
    --serif:"Iowan Old Style","Palatino Linotype",Georgia,serif;
    --sans:"Segoe UI",Candara,sans-serif;
    --mono:"Cascadia Mono","SFMono-Regular",Consolas,monospace;
  }}
  @media (prefers-color-scheme:dark) {{ :root:not([data-theme="light"]) {{
    --ground:#12191B; --surface:#18211F; --ink:#E3EAE7; --ink-2:#C2CDC9;
    --ink-3:#93A29E; --rule:#2C3835; --teal:#48B7AA; --amber:#D4924D;
    --brick:#DC7C72; --slate:#94A4C3; --teal-soft:#10302B; --amber-soft:#382717;
  }} }}
  html {{ overflow-x:clip; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; overflow-x:clip; background:var(--ground); color:var(--ink); font:17px/1.66 var(--serif); }}
  .wrap {{ width:min(1040px,calc(100% - 40px)); margin:0 auto; padding:72px 0 100px; }}
  header {{ max-width:760px; padding-bottom:42px; border-bottom:2px solid var(--ink); }}
  .eyebrow, .label {{ font:600 11px/1.3 var(--mono); letter-spacing:.12em; text-transform:uppercase; color:var(--ink-3); }}
  h1 {{ font:600 clamp(40px,6vw,72px)/1.02 var(--serif); letter-spacing:0; margin:14px 0 20px; max-width:13ch; }}
  h1 em {{ color:var(--teal); font-style:italic; }}
  .standfirst {{ font-size:22px; line-height:1.48; color:var(--ink-2); margin:0; max-width:42ch; }}
  .tallies {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); margin:36px 0 0; border:1px solid var(--rule); }}
  .tally {{ background:var(--surface); min-height:112px; padding:18px; border-right:1px solid var(--rule); }}
  .tally:last-child {{ border:0; }}
  .tally b {{ display:block; font:600 27px/1 var(--mono); letter-spacing:0; }}
  .tally span {{ display:block; color:var(--ink-3); font:13px/1.35 var(--sans); margin-top:9px; }}
  section {{ max-width:760px; padding-top:58px; }}
  h2 {{ font:600 31px/1.15 var(--serif); letter-spacing:0; margin:0 0 12px; }}
  h3 {{ font:600 13px/1.3 var(--sans); letter-spacing:.1em; text-transform:uppercase; color:var(--teal); margin:36px 0 12px; }}
  p {{ margin:0 0 18px; }}
  strong {{ font-weight:700; }}
  .lead {{ color:var(--ink-2); font-size:19px; max-width:43ch; }}
  .note {{ border-left:3px solid var(--teal); background:var(--teal-soft); padding:17px 20px; color:var(--ink-2); font-size:16px; }}
  .rule {{ border-left-color:var(--amber); background:var(--amber-soft); }}
  .grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; margin:30px 0 0; max-width:920px; }}
  .metric {{ border-top:2px solid var(--ink); padding:14px 0 0; }}
  .metric b {{ display:block; font:600 31px/1 var(--mono); letter-spacing:0; }}
  .metric span {{ color:var(--ink-3); font:13px/1.35 var(--sans); display:block; margin-top:8px; }}
  figure {{ width:100%; margin:34px 0; }}
  .fig-head {{ display:flex; justify-content:space-between; gap:16px; padding-bottom:9px; border-bottom:1px solid var(--rule); color:var(--ink-2); font:600 13px/1.4 var(--sans); }}
  .fig-head span:last-child {{ color:var(--ink-3); font:11px/1.4 var(--mono); text-transform:uppercase; letter-spacing:.08em; }}
  svg {{ display:block; width:100%; height:auto; margin-top:16px; }}
  .scroll {{ width:calc(100vw - 40px); max-width:1000px; overflow-x:auto; margin:28px 0; }}
  table {{ border-collapse:collapse; min-width:780px; width:100%; font:14px/1.45 var(--sans); }}
  th {{ text-align:right; color:var(--ink-3); font:600 10px/1.25 var(--mono); letter-spacing:.08em; text-transform:uppercase; padding:0 10px 9px; border-bottom:1px solid var(--rule); }}
  th:first-child {{ text-align:left; padding-left:0; }}
  td {{ text-align:right; padding:12px 10px; border-bottom:1px solid var(--rule); font-variant-numeric:tabular-nums; }}
  td:first-child {{ text-align:left; padding-left:0; min-width:220px; }}
  td strong {{ display:block; color:var(--ink); }}
  td small {{ display:block; color:var(--ink-3); font:11px/1.25 var(--sans); margin-top:3px; }}
  .protocol {{ margin:26px 0 0; padding:0; list-style:none; counter-reset:item; }}
  .protocol li {{ counter-increment:item; position:relative; padding:14px 0 14px 42px; border-bottom:1px solid var(--rule); }}
  .protocol li::before {{ content:counter(item,decimal-leading-zero); position:absolute; left:0; top:15px; color:var(--teal); font:600 12px var(--mono); }}
  footer {{ max-width:760px; border-top:1px solid var(--rule); margin-top:64px; padding-top:20px; color:var(--ink-3); font:13px/1.5 var(--sans); }}
  code {{ font:12px var(--mono); background:var(--surface); border:1px solid var(--rule); padding:2px 5px; }}
  @media (max-width:700px) {{ .wrap {{ width:min(100% - 28px,1040px); padding-top:46px; }} .tallies {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .tally:nth-child(2) {{ border-right:0; }} .tally:nth-child(-n+2) {{ border-bottom:1px solid var(--rule); }} h1 {{ font-size:44px; }} .grid {{ grid-template-columns:1fr; }} .scroll {{ width:calc(100vw - 28px); }} }}
</style>
</head>
<body>
<main class="wrap">
  <header>
    <div class="eyebrow">Chapter Seven &middot; 13F conviction and capacity</div>
    <h1>When Six Slots Are <em>Scarce</em></h1>
    <p class="standfirst">A public filing cannot say when to buy. It can say which breakout deserves a scarce place in a full book, and how much risk the book can justify taking.</p>
    <div class="tallies">
      <div class="tally"><b>{scope['ticker_count']}</b><span>high-conviction 13F names with daily history</span></div>
      <div class="tally"><b>661,577</b><span>Alpha Vantage daily bars, locally cached</span></div>
      <div class="tally"><b>{full['baseline']['signals_before_cap']:,}</b><span>independent Turtle positions offered to six slots</span></div>
      <div class="tally"><b>{full['baseline']['rejected_book_full']['median']:,}</b><span>positions refused because the book was already full</span></div>
    </div>
  </header>

  <section>
    <div class="eyebrow">I &middot; The question changes with capacity</div>
    <h2>The filing is not an entry.</h2>
    <p class="lead">The original 13F result was clear: copying a manager after a forty-five-day disclosure lag is not a trade. The unresolved question was narrower: when a six-position Turtle book has more valid breakouts than room, can disclosed conviction decide which one gets the slot?</p>
    <p>The test takes every long-only 55-day channel breakout across the {scope['ticker_count']}-name panel. Entry remains a stop at the channel edge; the system retains its 20-day exit, 2N protective stop, half-N pyramiding to four units and 2bp round-trip cost. A name becomes conviction-active only after its actual public 13F filing, where a concentrated manager reported a new position of at least 5% of the disclosed book. It stays active for one year.</p>
    <div class="note"><strong>There are two distinct uses of that information.</strong> A hard gate takes only active names. Priority leaves every valid breakout eligible, but gives an active high-conviction name first claim on a same-day slot. The second is an allocation rule; the first is a different strategy.</div>
    <div class="grid">
      <div class="metric"><b>{result['breakout_candidates_before_trade_lifecycle']['baseline']:,}</b><span>raw daily channel breaches before the Turtle lifecycle resolves overlapping signals</span></div>
      <div class="metric"><b>{result['breakout_candidates_before_trade_lifecycle']['13f_watchlist']:,}</b><span>of those breaches while an eligible high-conviction 13F window was active</span></div>
    </div>
  </section>

  <section>
    <div class="eyebrow">II &middot; What the allocator sees</div>
    <h2>A hard gate is a trade-off. Priority is the cleaner result.</h2>
    <p>The hard gate greatly reduced drawdown in the 2023+ validation slice, but it also cut the full-period CAGR from {pct(median(full['baseline'], 'cagr'))} to {pct(median(full['13f_watchlist'], 'cagr'))}. It is therefore not a free quality filter. Capacity priority uses the same full opportunity set, changes no stop or exit, and only breaks same-day contention using information public before the entry.</p>
    {full_chart}
    {validation_chart}
    <div class="note rule"><strong>The result that survives both periods:</strong> priority raises the full-period Calmar ratio from {number(calmar(full['baseline']), 3)} to {number(calmar(priority['full']), 3)}, while validation rises from {number(calmar(validation['baseline']), 3)} to {number(calmar(priority['validation_2023_plus']), 3)}. It does this with fewer selected trades and lower exit-marked drawdown, rather than by simply increasing aggregate leverage.</div>
  </section>

  <section>
    <div class="eyebrow">III &middot; Conviction and risk</div>
    <h2>More size helps here. That does not make it a rule yet.</h2>
    <p>The sizing test leaves every breakout in the book. It applies a multiplier that rises linearly from 1.0 at zero disclosed conviction to the cap at 10% of a manager's book. The largest tested cap, 2x, did not double portfolio exposure: qualifying positions were uncommon, so the mean risk multiplier was only {number(median(tilt_2['full'], 'mean_risk_multiplier'), 3)}x over the full period and {number(median(tilt_2['validation_2023_plus'], 'mean_risk_multiplier'), 3)}x in validation.</p>
    <p>That cap improved return and Calmar in both samples, but the drawdown rose from {pct(median(full['baseline'], 'max_drawdown_exit_marked'))} to {pct(median(tilt_2['full'], 'max_drawdown_exit_marked'))} over the full period. It was tested alongside two smaller caps after the original 13F result was known. The right conclusion is not &ldquo;double risk&rdquo;; it is that the multiplier is a candidate for a frozen forward test.</p>
    <div class="scroll">
      <table>
        <thead><tr><th>Variant</th><th>Trades</th><th>Total R</th><th>PF</th><th>CAGR</th><th>Exit-marked DD</th><th>Calmar</th></tr></thead>
        <tbody>{full_rows}</tbody>
      </table>
    </div>
    <h3>2023+ validation</h3>
    <div class="scroll">
      <table>
        <thead><tr><th>Variant</th><th>Trades</th><th>Total R</th><th>PF</th><th>CAGR</th><th>Exit-marked DD</th><th>Calmar</th></tr></thead>
        <tbody>{validation_rows}</tbody>
      </table>
    </div>
  </section>

  <section>
    <div class="eyebrow">IV &middot; What is licensed</div>
    <h2>A forward rule, not a retrospective promise.</h2>
    <ol class="protocol">
      <li>Track actual public 13F filings. Admit only newly disclosed positions at or above 5% of a concentrated manager's reported equity book.</li>
      <li>Keep each name on the watchlist for one year after the filing. Do not buy it because of the filing.</li>
      <li>Use the existing long-only daily 55/20 Turtle entry, stop, pyramid and exit rules unchanged.</li>
      <li>When same-day entries compete for the sixth slot, rank active names by their largest public manager weight before using a random tie break.</li>
      <li>Freeze any risk multiplier before forward deployment. The 2x cap is evidence to test, not permission to change live risk.</li>
    </ol>
    <div class="note"><strong>Three limits travel with this result.</strong> The {scope['ticker_count']}-name panel is the union of historically observed high-conviction positions, so its absolute return inherits hindsight-universe bias. Alpha Vantage daily history is a current vendor history, not an archived vintage. And the drawdowns here are marked only when a trade exits; intratrade adverse excursion would be worse. The priority comparison is cleaner because it holds the panel, entries, exits and capacity fixed, but it is still a research result rather than a live mandate.</div>
  </section>

  <footer>
    Inputs: <code>data/13f/holdings_major.json</code>, <code>data/13f/h13f.pkl</code>, <code>data/13f/book13f.pkl</code>, and <code>data/13f/alphavantage_daily.db</code>. Generated from <code>out/strategy/13f_turtle_full_panel.json</code> by <code>scripts/build_chapter7_13f_turtle.py</code>. The data and analysis output are locally cached and intentionally gitignored; this rendered chapter and its generator are the durable research record.
  </footer>
</main>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = json.loads(args.result.read_text(encoding="utf-8"))
    output = render(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(output, encoding="utf-8")
    print(f"wrote {args.out} ({len(output):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())