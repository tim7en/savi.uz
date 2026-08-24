"""Build the self-contained SPY reverse-leverage and reserve research page."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import pandas as pd


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path,
                        default=Path("out/strategy/spy_staged_funding/results.json"))
    parser.add_argument("--daily", type=Path,
                        default=Path("out/strategy/spy_staged_funding/comparison.csv"))
    parser.add_argument("--all-at-50", type=Path,
                        default=Path("out/strategy/spy_reverse_vault/results.json"))
    parser.add_argument("--twenty-year", type=Path,
                        default=Path("out/strategy/spy_20y_contributions/results.json"))
    parser.add_argument("--five-x", type=Path,
                        default=Path("out/strategy/spy_account_drawdown_5x/results.json"))
    parser.add_argument("--grid", type=Path,
                        default=Path("out/strategy/spy_grid_margin/results.json"))
    parser.add_argument("--rescue", type=Path,
                        default=Path("out/strategy/spy_rescue_capital/results.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("docs/spy-drawdown-funding.html"))
    return parser.parse_args(argv)


def pct(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def money(value: float) -> str:
    return f"${value:,.0f}"


def performance(series: pd.Series) -> dict:
    years = (series.index[-1] - series.index[0]).days / 365.2425
    drawdown = series / series.cummax() - 1.0
    return {"terminal": float(series.iloc[-1]),
            "cagr": float((series.iloc[-1] / series.iloc[0]) ** (1.0 / years) - 1.0),
            "max_drawdown": float(drawdown.min())}


def sampled(frame: pd.DataFrame, every: int = 8) -> pd.DataFrame:
    keep = frame.iloc[::every].copy()
    if keep.index[-1] != frame.index[-1]:
        keep = pd.concat([keep, frame.iloc[[-1]]])
    return keep


def chart_axes(index: pd.DatetimeIndex, width: int, height: int,
               left: int, right: int, top: int, bottom: int) -> tuple:
    first, last = index[0].value, index[-1].value
    inner_width, inner_height = width - left - right, height - top - bottom

    def x(stamp) -> float:
        return left + (pd.Timestamp(stamp).value - first) / (last - first) * inner_width

    years = [year for year in range(index[0].year + 2, index[-1].year + 1, 5)]
    ticks = []
    for year in years:
        stamp = pd.Timestamp(year=year, month=1, day=1)
        position = x(stamp)
        ticks.append(
            f'<line x1="{position:.1f}" x2="{position:.1f}" y1="{top}" '
            f'y2="{height-bottom}" class="gridline"/>'
            f'<text x="{position:.1f}" y="{height-13}" text-anchor="middle" '
            f'class="chart-axis">{year}</text>')
    return x, inner_height, "".join(ticks)


def growth_svg(frame: pd.DataFrame) -> str:
    data = sampled(frame, 8)
    width, height, left, right, top, bottom = 920, 430, 74, 24, 22, 42
    x, inner_height, x_ticks = chart_axes(data.index, width, height,
                                         left, right, top, bottom)
    columns = [
        ("reverse_no_reserve", "Reverse, no reserve", "var(--slate)", ""),
        ("staged_no_external", "Staged, harvested profits", "var(--teal)", ""),
        ("monthly", "+$100 monthly", "var(--blue)", ""),
        ("bimonthly", "+$200 every two months", "var(--violet)", "5 4"),
        ("spy_hold", "SPY 1x", "var(--ink)", ""),
    ]
    positive = data[[item[0] for item in columns]].where(lambda item: item > 0)
    low = 10 ** math.floor(math.log10(float(positive.min().min())))
    high = 10 ** math.ceil(math.log10(float(positive.max().max())))
    log_low, log_high = math.log10(low), math.log10(high)

    def y(value: float) -> float:
        share = (math.log10(value) - log_low) / (log_high - log_low)
        return top + (1.0 - share) * inner_height

    grid = []
    tick = low
    while tick <= high:
        position = y(tick)
        label = (f"${tick / 1_000_000:g}m" if tick >= 1_000_000
                 else f"${tick / 1_000:g}k")
        grid.append(
            f'<line x1="{left}" x2="{width-right}" y1="{position:.1f}" '
            f'y2="{position:.1f}" class="gridline"/>'
            f'<text x="{left-10}" y="{position+4:.1f}" text-anchor="end" '
            f'class="chart-axis">{label}</text>')
        tick *= 10
    paths, legend = [], []
    for column, label, color, dash in columns:
        points = [f"{x(stamp):.1f},{y(float(value)):.1f}"
                  for stamp, value in data[column].items() if value > 0]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        paths.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{color}" stroke-width="2.2"{dash_attr} '
            f'vector-effect="non-scaling-stroke"/>')
        legend.append(
            f'<span><i style="--swatch:{color};--dash:{"dashed" if dash else "solid"}"></i>'
            f'{html.escape(label)}</span>')
    return f"""<figure>
      <div class="figure-head"><strong>Growth of the account</strong><span>log scale · $10,000 start</span></div>
      <div class="plate"><svg viewBox="0 0 {width} {height}" role="img" aria-label="Growth of reverse leverage, reserve, monthly contribution, bimonthly contribution, and SPY accounts">
        {x_ticks}{''.join(grid)}{''.join(paths)}
      </svg><div class="legend">{''.join(legend)}</div></div>
      <figcaption>External deposits lift the blue and violet ending balances. Their return is therefore reported as XIRR, not as CAGR on the original $10,000.</figcaption>
    </figure>"""


def drawdown_svg(frame: pd.DataFrame, frequency: list[dict]) -> str:
    data = sampled(frame[["strategy_drawdown"]], 5)
    width, height, left, right, top, bottom = 920, 360, 66, 24, 22, 42
    x, inner_height, x_ticks = chart_axes(data.index, width, height,
                                         left, right, top, bottom)
    low, high = -0.92, 0.0

    def y(value: float) -> float:
        return top + (high - value) / (high - low) * inner_height

    horizontal = []
    for value in (0.0, -0.2, -0.3, -0.5, -0.8):
        position = y(value)
        klass = "zero" if value == 0 else "threshold"
        horizontal.append(
            f'<line x1="{left}" x2="{width-right}" y1="{position:.1f}" '
            f'y2="{position:.1f}" class="{klass}"/>'
            f'<text x="{left-9}" y="{position+4:.1f}" text-anchor="end" '
            f'class="chart-axis">{value:.0%}</text>')
    points = [f"{x(stamp):.1f},{y(float(value)):.1f}"
              for stamp, value in data["strategy_drawdown"].items()]
    area = (f"M {x(data.index[0]):.1f} {y(0):.1f} L "
            + " L ".join(points)
            + f" L {x(data.index[-1]):.1f} {y(0):.1f} Z")
    dots = []
    colors = {0.2: "var(--amber)", 0.3: "var(--orange)",
              0.5: "var(--brick)", 0.8: "var(--deep)"}
    for item in frequency:
        threshold = float(item["threshold"])
        for day in item["dates"]:
            stamp = pd.Timestamp(day)
            if stamp < data.index[0] or stamp > data.index[-1]:
                continue
            actual = float(frame["strategy_drawdown"].asof(stamp))
            dots.append(f'<circle cx="{x(stamp):.1f}" cy="{y(actual):.1f}" r="4" '
                        f'fill="{colors[threshold]}"/>')
    return f"""<figure>
      <div class="figure-head"><strong>Where the reserve is allowed to move</strong><span>flow-adjusted strategy drawdown</span></div>
      <div class="plate"><svg viewBox="0 0 {width} {height}" role="img" aria-label="Reverse strategy drawdown with twenty, thirty, fifty, and eighty percent deployment marks">
        {x_ticks}{''.join(horizontal)}
        <path d="{area}" fill="var(--brick-soft)"/>
        <polyline points="{' '.join(points)}" fill="none" stroke="var(--brick)" stroke-width="1.8" vector-effect="non-scaling-stroke"/>
        {''.join(dots)}
      </svg><div class="legend"><span><i style="--swatch:var(--amber)"></i>−20%</span><span><i style="--swatch:var(--orange)"></i>−30%</span><span><i style="--swatch:var(--brick)"></i>−50%</span><span><i style="--swatch:var(--deep)"></i>−80%</span></div></div>
      <figcaption>Each rung fires once within an episode. It becomes available again only after the strategy's performance NAV regains its previous high.</figcaption>
    </figure>"""


def outcome_row(name: str, note: str, ending: float, supplied: float,
                return_value: float, return_label: str, drawdown: float,
                featured: bool = False) -> str:
    klass = ' class="featured"' if featured else ""
    return f"""<tr{klass}><td><strong>{html.escape(name)}</strong><small>{html.escape(note)}</small></td>
      <td>{money(supplied)}</td><td>{money(ending)}</td>
      <td>{pct(return_value)} <small>{html.escape(return_label)}</small></td>
      <td>{pct(drawdown)}</td></tr>"""


def render(result: dict, daily: pd.DataFrame, all_at_50: dict | None,
           twenty_year: dict | None, five_x: dict | None,
           grid: dict | None, rescue: dict | None) -> str:
    sample = result["sample"]
    stats = result["results"]
    frequency = result["threshold_history"]
    reverse = performance(daily["reverse_no_reserve"])
    spy = performance(daily["spy_hold"])
    staged = stats["none"]
    monthly = stats["monthly"]
    bimonthly = stats["bimonthly"]
    all50 = all_at_50["vault_policy_combined"] if all_at_50 else None
    years = ((daily.index[-1] - daily.index[0]).days / 365.2425)

    outcomes = [
        outcome_row("SPY 1x", "Dividends reinvested; no leverage",
                    spy["terminal"], 10_000, spy["cagr"], "CAGR", spy["max_drawdown"]),
        outcome_row("Reverse 3→2→1", "No reserve; Treasury + 1% funding",
                    reverse["terminal"], 10_000, reverse["cagr"], "CAGR",
                    reverse["max_drawdown"]),
    ]
    if all50:
        outcomes.append(outcome_row(
            "Reserve deployed at −50%", "10% profit harvest; one deployment per episode",
            float(all50["terminal"]), 10_000, float(all50["cagr"]), "CAGR",
            float(all50["max_drawdown"])))
    outcomes += [
        outcome_row("Staged 20/30/50/80", "Profit harvesting only; no outside cash",
                    staged["terminal"], 10_000, staged["xirr"], "CAGR",
                    staged["max_combined_drawdown_flow_adjusted"], True),
        outcome_row("+$100 monthly", "$40,400 deposited into the reserve",
                    monthly["terminal"], monthly["total_cash_including_initial"],
                    monthly["xirr"], "XIRR",
                    monthly["max_combined_drawdown_flow_adjusted"]),
        outcome_row("+$200 every two months", "Same $40,400 external cash",
                    bimonthly["terminal"], bimonthly["total_cash_including_initial"],
                    bimonthly["xirr"], "XIRR",
                    bimonthly["max_combined_drawdown_flow_adjusted"]),
    ]

    frequency_rows = []
    for item in frequency:
        count = int(item["count"])
        rate = years / count
        frequency_rows.append(f"""<tr><td><strong>−{float(item['threshold']):.0%}</strong></td>
          <td>{count}</td><td>once / {rate:.1f} years</td>
          <td>{pct(item['forward_1y_median_annualized'])}</td>
          <td>{int(round(float(item['forward_1y_positive_share']) * item['forward_1y_n']))}/{item['forward_1y_n']}</td>
          <td>{pct(item['forward_3y_median_annualized'])}</td></tr>""")

    all50_sentence = ""
    if all50:
        all50_sentence = (
            f"Waiting for −50% finished at {money(float(all50['terminal']))} with "
            f"{pct(float(all50['cagr']))} CAGR and {pct(float(all50['max_drawdown']))} "
            "maximum drawdown. The staged rule used reserve cash earlier, finished "
            f"at {money(staged['terminal'])}, and improved the worst loss by only "
            f"{abs(staged['max_combined_drawdown_flow_adjusted'] - float(all50['max_drawdown'])):.2%}.")

    growth = growth_svg(daily)
    drawdown = drawdown_svg(daily, frequency)
    rescue_20 = rescue["rolling_20y"] if rescue else None
    rescue_30 = rescue["matched_30y"] if rescue else None
    rescue_new_20 = rescue["new_5_2_1_rolling_20y"] if rescue else None
    rescue_new_30 = rescue["new_5_2_1_matched_30y"] if rescue else None
    rescue_corrected_half_20 = rescue.get("corrected_3_2_1_half_balance_rolling_20y") if rescue else None
    rescue_corrected_half_30 = rescue.get("corrected_3_2_1_half_balance_matched_30y") if rescue else None
    rescue_corrected_equal_20 = rescue.get("corrected_3_2_1_equal_balance_rolling_20y") if rescue else None
    rescue_corrected_equal_30 = rescue.get("corrected_3_2_1_equal_balance_matched_30y") if rescue else None
    twenty_section = ""
    if twenty_year:
        roll = twenty_year["rolling_20y"]
        spy20 = twenty_year["spy_x1_rolling_20y"]
        terminal = roll["terminal_wealth"]
        recovery = roll["longest_underwater_years"]
        twenty_section = f"""
<section class="wide">
  <div class="eyebrow">V · A twenty-year savings plan</div><h2>The typical result was modest; the range was enormous.</h2>
  <p class="lead">With $833.33 contributed monthly for 20 years, total cash supplied was $200,000. Across 164 overlapping historical start months, the strategy's median ending wealth was {money(terminal['median'])}; the middle 80% ran from {money(terminal['p10'])} to {money(terminal['p90'])}.</p>
  <div class="split"><div class="metric"><b>{recovery['median']:.1f} years</b><span>median longest continuous wait to regain a prior performance high</span></div><div class="metric"><b>{pct(roll['max_drawdown']['median'])}</b><span>median worst flow-adjusted drawdown inside a 20-year window</span></div></div>
  <div class="scroll"><table><thead><tr><th>Plan</th><th>Worst</th><th>10th percentile</th><th>Median</th><th>90th percentile</th><th>Best</th></tr></thead><tbody>
    <tr class="featured"><td><strong>Reverse + staged reserve</strong><small>funding cost included</small></td><td>{money(terminal['min'])}</td><td>{money(terminal['p10'])}</td><td>{money(terminal['median'])}</td><td>{money(terminal['p90'])}</td><td>{money(terminal['max'])}</td></tr>
    <tr><td><strong>SPY 1x</strong><small>dividends reinvested</small></td><td>{money(spy20['terminal_wealth']['min'])}</td><td>{money(spy20['terminal_wealth']['p10'])}</td><td>{money(spy20['terminal_wealth']['median'])}</td><td>{money(spy20['terminal_wealth']['p90'])}</td><td>{money(spy20['terminal_wealth']['max'])}</td></tr>
  </tbody></table></div>
  <p class="note warning"><strong>Do not read the median as a forecast.</strong> These windows overlap heavily and every complete 20-year cohort includes the 2008 crash. The leveraged strategy beat SPY 1x in only {pct(roll['beats_spy_terminal_share'], 1)} of cohorts. Its large upside tail came with a median {pct(roll['time_underwater_share']['median'], 1)} of trading days below the prior high.</p>
</section>"""
    rescue_30_row = ""
    if rescue_30:
        rescue_30_row = f"""<tr><td><strong>Equal-capital rescue</strong><small>1x tranche at −60%; exit at +10%</small></td><td>{money(rescue_30['terminal_wealth'])}</td><td>{pct(rescue_30['xirr'])}</td><td>{pct(rescue_30['max_drawdown'])}</td><td>1.00x rescue</td><td>—</td></tr>"""
        rescue_30_row += f"""<tr><td><strong>5→2→1 plus rescue</strong><small>cut at −20/−50%; equal capital at −60%</small></td><td>{money(rescue_new_30['terminal_wealth'])}</td><td>{pct(rescue_new_30['xirr'])}</td><td>{pct(rescue_new_30['max_drawdown'])}</td><td>1.00x rescue</td><td>—</td></tr>"""
        if rescue_corrected_half_30:
            rescue_30_row += f"""<tr class="featured"><td><strong>3x to 2x to 1x + half-balance rescue</strong><small>cut at -30/-50%; quarterly profit sweep; rescue at -60%</small></td><td>{money(rescue_corrected_half_30['terminal_wealth'])}</td><td>{pct(rescue_corrected_half_30['xirr'])}</td><td>{pct(rescue_corrected_half_30['max_drawdown'])}</td><td>{rescue_corrected_half_30['mean_applied_leverage']:.2f}x</td><td>-</td></tr>"""
        if rescue_corrected_equal_30:
            rescue_30_row += f"""<tr><td><strong>3x to 2x to 1x + equal-balance rescue</strong><small>literal equal leftover at -60%; quarterly profit sweep</small></td><td>{money(rescue_corrected_equal_30['terminal_wealth'])}</td><td>{pct(rescue_corrected_equal_30['xirr'])}</td><td>{pct(rescue_corrected_equal_30['max_drawdown'])}</td><td>{rescue_corrected_equal_30['mean_applied_leverage']:.2f}x</td><td>-</td></tr>"""
    rescue_20_row = ""
    if rescue_20:
        rescue_20_row = f"""<tr><td><strong>Equal-capital rescue</strong><small>external top-up counted in XIRR</small></td><td>{money(rescue_20['terminal_wealth']['median'])}</td><td>{pct(rescue_20['xirr']['median'])}</td><td>{pct(rescue_20['max_drawdown']['median'])}</td><td>{rescue_20['longest_underwater_years']['median']:.1f} years</td><td>1.00x rescue</td><td>—</td></tr>"""
        rescue_20_row += f"""<tr><td><strong>5→2→1 plus rescue</strong><small>median external rescue {money(rescue_new_20['total_rescue_external']['median'])}</small></td><td>{money(rescue_new_20['terminal_wealth']['median'])}</td><td>{pct(rescue_new_20['xirr']['median'])}</td><td>{pct(rescue_new_20['max_drawdown']['median'])}</td><td>{rescue_new_20['longest_underwater_years']['median']:.1f} years</td><td>1.00x rescue</td><td>—</td></tr>"""
        if rescue_corrected_half_20:
            rescue_20_row += f"""<tr class="featured"><td><strong>3x to 2x to 1x + half-balance rescue</strong><small>quarterly sweep; median external rescue {money(rescue_corrected_half_20['total_rescue_external']['median'])}</small></td><td>{money(rescue_corrected_half_20['terminal_wealth']['median'])}</td><td>{pct(rescue_corrected_half_20['xirr']['median'])}</td><td>{pct(rescue_corrected_half_20['max_drawdown']['median'])}</td><td>{rescue_corrected_half_20['longest_underwater_years']['median']:.1f} years</td><td>{rescue_corrected_half_20['mean_applied_leverage']['median']:.2f}x</td><td>-</td></tr>"""
        if rescue_corrected_equal_20:
            rescue_20_row += f"""<tr><td><strong>3x to 2x to 1x + equal-balance rescue</strong><small>quarterly sweep; median external rescue {money(rescue_corrected_equal_20['total_rescue_external']['median'])}</small></td><td>{money(rescue_corrected_equal_20['terminal_wealth']['median'])}</td><td>{pct(rescue_corrected_equal_20['xirr']['median'])}</td><td>{pct(rescue_corrected_equal_20['max_drawdown']['median'])}</td><td>{rescue_corrected_equal_20['longest_underwater_years']['median']:.1f} years</td><td>{rescue_corrected_equal_20['mean_applied_leverage']['median']:.2f}x</td><td>-</td></tr>"""
    if grid:
        grid_base = grid["scenarios"]["base_OLHC"]
        grid_core = grid["scenarios"]["long_core_base_OLHC"]
        grid_neutral = grid["scenarios"]["neutral_base_OLHC"]
        grid_spy = grid["spy_x1"]
        grid_previous = grid["previous_5_3_3_1"]
        grid_fear = grid["fear_relever_5_3_3_1_3"]
        twenty_section += f"""
<section class="wide">
  <div class="eyebrow">V.B · The volatility grid</div><h2>The grid harvested movement but surrendered the trend.</h2>
  <p class="lead">This 30-year daily-OHLC proxy used 12 levels on each side at 0.4%, a prior-known 20-session EMA center, one basis point per fill, and a 5x cap reduced to 2x at a 20% account loss and 1x at 60%. Of each $10,000 annual contribution, 70% entered trading and 30% savings.</p>
  <div class="scroll"><table><thead><tr><th>Inventory model</th><th>Ending wealth</th><th>XIRR</th><th>Max drawdown</th><th>Median effective leverage</th><th>Time at 1x cap</th></tr></thead><tbody>
    <tr><td><strong>SPY 1x</strong><small>all $300,000 invested</small></td><td>{money(grid_spy['terminal_wealth'])}</td><td>{pct(grid_spy['xirr'])}</td><td>{pct(grid_spy['max_drawdown_flow_adjusted'])}</td><td>1.00x</td><td>—</td></tr>
    <tr class="featured"><td><strong>Previous 5→3→3→1 rule</strong><small>profit sweep and reserve; no grid</small></td><td>{money(grid_previous['terminal_wealth'])}</td><td>{pct(grid_previous['xirr'])}</td><td>{pct(grid_previous['max_drawdown'])}</td><td>{grid_previous['mean_applied_leverage']:.2f}x</td><td>—</td></tr>
    <tr><td><strong>Fear relever 5→3→3→1→3</strong><small>increase to 3x below 60%</small></td><td>{money(grid_fear['terminal_wealth'])}</td><td>{pct(grid_fear['xirr'])}</td><td>{pct(grid_fear['max_drawdown'])}</td><td>{grid_fear['mean_applied_leverage']:.2f}x</td><td>—</td></tr>
    {rescue_30_row}
    <tr><td><strong>Long-core grid</strong><small>50–100% of active cap</small></td><td>{money(grid_core['terminal_wealth'])}</td><td>{pct(grid_core['xirr'])}</td><td>{pct(grid_core['max_drawdown'])}</td><td>{grid_core['median_effective_leverage']:.2f}x</td><td>{pct(grid_core['time_at_1x_cap'], 1)}</td></tr>
    <tr><td><strong>Long-only grid</strong><small>0–100% of active cap</small></td><td>{money(grid_base['terminal_wealth'])}</td><td>{pct(grid_base['xirr'])}</td><td>{pct(grid_base['max_drawdown'])}</td><td>{grid_base['median_effective_leverage']:.2f}x</td><td>{pct(grid_base['time_at_1x_cap'], 1)}</td></tr>
    <tr><td><strong>Neutral grid</strong><small>short above / long below average</small></td><td>{money(grid_neutral['terminal_wealth'])}</td><td>{pct(grid_neutral['xirr'])}</td><td>{pct(grid_neutral['max_drawdown'])}</td><td>{grid_neutral['median_effective_leverage']:.2f}x</td><td>{pct(grid_neutral['time_at_1x_cap'], 1)}</td></tr>
  </tbody></table></div>
  <p class="note warning"><strong>Daily bars are not execution data.</strong> The high/low order, queue position, partial fills and spread are unknowable over 30 years. Open-low-high-close and open-high-low-close produced similar estimates here, but neither establishes an executable expected return. The base rule reached its 1x cap in 2002 and never recovered the old performance high.</p>
</section>"""
    if five_x:
        literal = five_x["variants"]["literal_profit_reserve_5_3_3_1"]
        sensitivity = five_x["variants"]["sensitivity_profit_reserve_5_3_2_1"]
        fear = five_x["variants"]["fear_relever_profit_reserve_5_3_3_1_3"]
        spy5 = five_x["spy_x1"]
        twenty_section += f"""
<section class="wide">
  <div class="eyebrow">V.C · The 5x account rule</div><h2>Five-times leverage did not produce five-times wealth.</h2>
  <p class="lead">The literal rule starts at 5x, latches at 3x after a 10% investment-sleeve loss, remains 3x at 30%, and falls to 1x after 50%. New contributions enter the trading sleeve; only 10% of positive annual trading profit moves to savings.</p>
  <div class="scroll"><table><thead><tr><th>Rule and signal</th><th>Median wealth</th><th>Median XIRR</th><th>Median drawdown</th><th>Longest recovery</th><th>Average leverage</th><th>Beat SPY</th></tr></thead><tbody>
    <tr><td><strong>SPY 1x</strong><small>same $200,000 contributions</small></td><td>{money(spy5['terminal_wealth']['median'])}</td><td>{pct(spy5['xirr']['median'])}</td><td>{pct(spy5['max_drawdown']['median'])}</td><td>{spy5['longest_underwater_years']['median']:.1f} years</td><td>1.00x</td><td>benchmark</td></tr>
    <tr class="featured"><td><strong>5→3→3→1</strong><small>leveraged sleeve drawdown</small></td><td>{money(literal['terminal_wealth']['median'])}</td><td>{pct(literal['xirr']['median'])}</td><td>{pct(literal['max_drawdown']['median'])}</td><td>{literal['longest_underwater_years']['median']:.1f} years</td><td>{literal['mean_applied_leverage']['median']:.2f}x</td><td>{pct(literal['beats_spy_terminal_share'], 1)}</td></tr>
    <tr><td><strong>5→3→2→1</strong><small>30% rung sensitivity</small></td><td>{money(sensitivity['terminal_wealth']['median'])}</td><td>{pct(sensitivity['xirr']['median'])}</td><td>{pct(sensitivity['max_drawdown']['median'])}</td><td>{sensitivity['longest_underwater_years']['median']:.1f} years</td><td>{sensitivity['mean_applied_leverage']['median']:.2f}x</td><td>{pct(sensitivity['beats_spy_terminal_share'], 1)}</td></tr>
    <tr><td><strong>5→3→3→1→3</strong><small>relever at 60% fear rung</small></td><td>{money(fear['terminal_wealth']['median'])}</td><td>{pct(fear['xirr']['median'])}</td><td>{pct(fear['max_drawdown']['median'])}</td><td>{fear['longest_underwater_years']['median']:.1f} years</td><td>{fear['mean_applied_leverage']['median']:.2f}x</td><td>{pct(fear['beats_spy_terminal_share'], 1)}</td></tr>
    {rescue_20_row}
  </tbody></table></div>
  <p class="note warning"><strong>The higher wealth came with a much harder path.</strong> The literal rule beat SPY in {pct(literal['beats_spy_terminal_share'], 1)} of rolling cohorts, but its median maximum loss was {pct(literal['max_drawdown']['median'])} and its longest recovery lasted a median {literal['longest_underwater_years']['median']:.1f} years.</p>
</section>"""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="A full-history SPY study of reverse leverage, profit harvesting, a Treasury reserve, and staged deployment during account drawdowns.">
<title>Funding the Fall — SPY leverage and reserve study</title>
<style>
  :root{{--ground:#eef1f0;--surface:#fbfcfb;--sunken:#e4e9e7;--ink:#111918;--ink2:#40504c;--ink3:#74827f;--rule:#ccd5d2;--teal:#087f72;--teal-soft:#dceeea;--blue:#2563a6;--violet:#7450a7;--slate:#607187;--amber:#d6a009;--orange:#d96a1f;--brick:#b84036;--brick-soft:#f4d9d5;--deep:#6f1717;--serif:"Iowan Old Style","Palatino Linotype",Georgia,serif;--sans:system-ui,-apple-system,"Segoe UI",sans-serif;--mono:"Cascadia Mono","SFMono-Regular",Consolas,monospace}}
  @media(prefers-color-scheme:dark){{:root{{--ground:#101716;--surface:#17201e;--sunken:#202a27;--ink:#e5ece9;--ink2:#bdc9c5;--ink3:#8b9a95;--rule:#2c3935;--teal:#4ab7a8;--teal-soft:#15302b;--blue:#65a5ea;--violet:#b496e3;--slate:#a3b0c5;--amber:#efc64b;--orange:#ef9457;--brick:#e17468;--brick-soft:#3a211f;--deep:#b94c4c}}}}
  *{{box-sizing:border-box}} html{{overflow-x:hidden}} body{{margin:0;background:var(--ground);color:var(--ink);font:17px/1.66 var(--serif);-webkit-font-smoothing:antialiased}} .wrap{{width:min(1040px,calc(100% - 40px));margin:auto;padding:68px 0 96px}} header{{max-width:820px;padding-bottom:38px;border-bottom:2px solid var(--ink)}} .eyebrow{{font:600 11px/1.3 var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--ink3)}} h1{{font:650 clamp(42px,7vw,76px)/.98 var(--serif);letter-spacing:-.035em;margin:14px 0 20px;max-width:12ch}} h1 em{{color:var(--teal);font-style:italic}} .standfirst{{font-size:22px;line-height:1.47;color:var(--ink2);max-width:43ch;margin:0}} .tallies{{display:grid;grid-template-columns:repeat(4,1fr);margin-top:34px;border:1px solid var(--rule)}} .tally{{background:var(--surface);padding:17px;border-right:1px solid var(--rule)}} .tally:last-child{{border:0}} .tally b{{display:block;font:650 25px/1 var(--mono)}} .tally span{{display:block;color:var(--ink3);font:12px/1.4 var(--sans);margin-top:8px}} section{{padding-top:56px;max-width:780px}} section.wide{{max-width:100%}} h2{{font:650 32px/1.14 var(--serif);letter-spacing:-.02em;margin:0 0 12px}} h3{{font:650 12px/1.3 var(--sans);letter-spacing:.1em;text-transform:uppercase;color:var(--teal);margin:34px 0 10px}} p{{margin:0 0 18px}} .lead{{font-size:19px;color:var(--ink2);max-width:44ch}} .note{{background:var(--teal-soft);border-left:3px solid var(--teal);padding:18px 20px;color:var(--ink2);font-size:16px}} .warning{{background:var(--brick-soft);border-left-color:var(--brick)}} .flow{{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;align-items:center;gap:10px;margin:28px 0}} .flow-card{{height:100%;background:var(--surface);border:1px solid var(--rule);padding:17px}} .flow-card b{{display:block;font:650 13px var(--sans);margin-bottom:6px}} .flow-card span{{color:var(--ink3);font:13px/1.45 var(--sans)}} .arrow{{color:var(--teal);font:700 20px var(--sans)}} .rungs{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:22px}} .rung{{background:var(--surface);border-top:3px solid var(--amber);padding:15px}} .rung:nth-child(2){{border-color:var(--orange)}} .rung:nth-child(3){{border-color:var(--brick)}} .rung:nth-child(4){{border-color:var(--deep)}} .rung b{{font:700 22px var(--mono);display:block}} .rung span{{font:12px/1.4 var(--sans);color:var(--ink3)}} figure{{margin:30px 0}} .figure-head{{display:flex;justify-content:space-between;gap:16px;padding-bottom:9px;border-bottom:1px solid var(--rule);font:13px/1.4 var(--sans);color:var(--ink2)}} .figure-head span{{font:10px var(--mono);text-transform:uppercase;letter-spacing:.08em;color:var(--ink3)}} .plate{{background:var(--surface);border:1px solid var(--rule);margin-top:14px;padding:14px;overflow-x:auto}} svg{{display:block;width:100%;min-width:680px;height:auto}} .chart-axis{{font:10px var(--mono);fill:var(--ink3)}} .gridline{{stroke:var(--rule);stroke-width:1}} .zero{{stroke:var(--ink3);stroke-width:1.2}} .threshold{{stroke:var(--rule);stroke-width:1;stroke-dasharray:5 4}} figcaption{{font:13px/1.5 var(--sans);color:var(--ink3);margin-top:10px;max-width:75ch}} .legend{{display:flex;flex-wrap:wrap;gap:10px 18px;padding:6px 4px 0;font:11px var(--sans);color:var(--ink2)}} .legend span{{display:flex;align-items:center;gap:7px}} .legend i{{width:22px;border-top:2px var(--dash,solid) var(--swatch);display:inline-block}} .scroll{{overflow-x:auto;border:1px solid var(--rule);background:var(--surface);margin-top:22px}} table{{border-collapse:collapse;width:100%;font:13px/1.4 var(--sans);font-variant-numeric:tabular-nums}} th{{font:600 10px var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--ink3);text-align:right;padding:11px 12px;border-bottom:1px solid var(--rule);white-space:nowrap}} th:first-child,td:first-child{{text-align:left}} td{{text-align:right;padding:12px;border-bottom:1px solid var(--rule);white-space:nowrap}} td:first-child{{white-space:normal;min-width:220px}} td strong{{display:block}} td small{{display:block;color:var(--ink3);font-size:11px;margin-top:3px}} tr:last-child td{{border-bottom:0}} tr.featured td{{background:var(--teal-soft)}} .split{{display:grid;grid-template-columns:1fr 1fr;gap:15px;margin-top:24px}} .metric{{background:var(--surface);border-top:2px solid var(--ink);padding:15px}} .metric b{{display:block;font:650 27px var(--mono)}} .metric span{{font:12px/1.4 var(--sans);color:var(--ink3)}} ol{{padding-left:22px;max-width:70ch}} li{{padding:4px 0}} footer{{max-width:780px;border-top:1px solid var(--rule);padding-top:20px;margin-top:62px;color:var(--ink3);font:12px/1.55 var(--sans)}} a{{color:var(--teal)}} @media(max-width:760px){{.wrap{{width:min(100% - 26px,1040px);padding-top:42px}}.tallies{{grid-template-columns:1fr 1fr}}.tally:nth-child(2){{border-right:0}}.tally:nth-child(-n+2){{border-bottom:1px solid var(--rule)}}.flow{{grid-template-columns:1fr}}.arrow{{transform:rotate(90deg);text-align:center}}.rungs{{grid-template-columns:1fr 1fr}}.split{{grid-template-columns:1fr}}}}
</style>
</head>
<body><main class="wrap">
<header>
  <div class="eyebrow">SPY · daily total return · {sample['start']} to {sample['end']}</div>
  <h1>Funding <em>the fall</em></h1>
  <p class="standfirst">Hold more exposure near market highs, harvest a fraction of good years into cash, then return that cash in stages when the leveraged account is already wounded.</p>
  <div class="tallies">
    <div class="tally"><b>{pct(reverse['cagr'])}</b><span>reverse-leverage CAGR after modeled funding</span></div>
    <div class="tally"><b>{pct(staged['max_combined_drawdown_flow_adjusted'])}</b><span>worst staged-reserve drawdown, no outside cash</span></div>
    <div class="tally"><b>{pct(monthly['xirr'])}</b><span>money-weighted return with $100 monthly</span></div>
    <div class="tally"><b>{sum(int(item['count']) for item in frequency)}</b><span>nested funding-rung crossings in {years:.1f} years</span></div>
  </div>
</header>

<section>
  <div class="eyebrow">I · The mechanism</div><h2>Two decisions, not one.</h2>
  <p class="lead">The leverage rule decides how much market to own. The reserve rule decides where uncommitted cash waits and when it may return.</p>
  <div class="flow"><div class="flow-card"><b>Trading sleeve</b><span>3× at SPY total-return highs, 2× after −10%, 1× after −50%. Signals apply next session.</span></div><div class="arrow">→</div><div class="flow-card"><b>Profit harvest</b><span>At each completed year-end, 10% of positive trading P&amp;L moves to the reserve.</span></div><div class="arrow">→</div><div class="flow-card"><b>Treasury reserve</b><span>Cash earns DGS3MO until the strategy's own performance drawdown unlocks a rung.</span></div></div>
  <div class="rungs"><div class="rung"><b>−20%</b><span>Deploy 25% of available reserve.</span></div><div class="rung"><b>−30%</b><span>Deploy one third of what remains.</span></div><div class="rung"><b>−50%</b><span>Deploy half of what remains.</span></div><div class="rung"><b>−80%</b><span>Deploy the balance.</span></div></div>
  <p class="note"><strong>Drawdown is flow-adjusted.</strong> A deposit does not create a return, and a profit transfer does not create a loss. “Recent high” means the latest recovered performance high since inception, not a rolling 12-month high.</p>
</section>

<section class="wide"><div class="eyebrow">II · The account paths</div><h2>More ending money is not automatically more return.</h2>{growth}</section>

<section class="wide">
  <div class="eyebrow">III · The measured trade-off</div><h2>The reserve bought four and a half points of survival.</h2>
  <p class="lead">Without outside cash, staged deployment reduced the worst loss from {pct(reverse['max_drawdown'])} to {pct(staged['max_combined_drawdown_flow_adjusted'])}. It also reduced CAGR from {pct(reverse['cagr'])} to {pct(staged['xirr'])}.</p>
  <div class="scroll"><table><thead><tr><th>Policy</th><th>Cash supplied</th><th>Ending value</th><th>Annual return</th><th>Max drawdown</th></tr></thead><tbody>{''.join(outcomes)}</tbody></table></div>
  <p class="note">{html.escape(all50_sentence)}</p>
</section>

<section class="wide"><div class="eyebrow">IV · Mobilization</div><h2>Shallow rungs happen. Deep rungs are anecdotes.</h2>{drawdown}
  <div class="scroll"><table><thead><tr><th>Rung</th><th>Count</th><th>Frequency</th><th>Median next 1y SPY</th><th>Positive 1y</th><th>Median next 3y annualized</th></tr></thead><tbody>{''.join(frequency_rows)}</tbody></table></div>
  <p class="note warning"><strong>These are not independent samples.</strong> The −30%, −50% and −80% marks can belong to the same long drawdown. There are only three −50% observations and one −80% observation. Their forward returns describe history; they do not estimate a stable probability.</p>
</section>

{twenty_section}

<section>
  <div class="eyebrow">VI · Monthly or every two months?</div><h2>The cadence was nearly irrelevant.</h2>
  <p>Both contribution cases supplied exactly {money(monthly['total_external_contributions'])} after the initial $10,000. Monthly contributions finished at {money(monthly['terminal'])}; the equal-budget bimonthly schedule finished at {money(bimonthly['terminal'])}. Their XIRRs differed by only {abs(monthly['xirr']-bimonthly['xirr']):.2%}.</p>
  <div class="split"><div class="metric"><b>{money(monthly['ending_reserve'])}</b><span>ending reserve with $100 monthly contributions</span></div><div class="metric"><b>{pct(monthly['max_combined_drawdown_flow_adjusted'])}</b><span>flow-adjusted combined drawdown; deposits are removed from performance</span></div></div>
  <p>The useful discipline is therefore simple: contribute on a schedule into the reserve, let it earn yield, and make mobilization conditional on drawdown. Trying to optimize the difference between one and two months did not matter in this sample.</p>
</section>

<section>
  <div class="eyebrow">VII · What the reserve cannot do</div><h2>Cash deployment is not risk reduction.</h2>
  <p>Once reserve cash moves into the trading sleeve, it takes the same market risk as the rest of the account. The reserve improves the combined path while it remains safe and adds recovery participation after deployment, but it does not change the leverage engine that created the loss.</p>
  <ol><li>A 3× daily position can be wiped out by a one-day 33⅓% loss before costs.</li><li>The worst modeled combined drawdown remains above 84% without outside contributions.</li><li>Regular contributions improve the path partly because more capital enters later; XIRR, not the ending balance, is the fair return measure.</li><li>Materially lower volatility requires a lower leverage cap or leverage tied to account drawdown—not only more cash mobilization.</li></ol>
  <p class="note warning"><strong>The next clean test:</strong> cap the trading sleeve at 2× after a 20% account drawdown and 1× after 40–50%, then apply the same reserve ladder. That changes risk itself rather than only recapitalizing losses.</p>
</section>

<section>
  <div class="eyebrow">VIII · Audit trail</div><h2>What is in—and what is absent.</h2>
  <p>SPY returns use adjusted close, so distributions are reinvested. Leverage is reset daily. Borrowed exposure pays the prior-known 3-month Treasury yield plus 1% over calendar days; parked reserves earn the Treasury yield. Taxes, bid/ask spread, leveraged-product fees, tracking error and forced liquidation are absent.</p>
  <p>SPY's sponsor dates the fund to January 1993 and describes it as tracking the price and yield performance of the S&amp;P 500. The rate series is the Federal Reserve's daily DGS3MO series. Leveraged and inverse products may reset daily and compound differently over longer periods.</p>
  <p><a href="https://www.ssga.com/us/en/individual/etfs/state-street-spdr-sp-500-etf-trust-spy">State Street: SPY</a> · <a href="https://fred.stlouisfed.org/series/DGS3MO">FRED: DGS3MO</a> · <a href="https://www.finra.org/investors/insights/lowdown-leveraged-and-inverse-exchange-traded-products">FINRA: leveraged and inverse products</a></p>
</section>

<footer>Historical simulation, not investment advice. Generated from <code>out/strategy/spy_staged_funding/results.json</code> and the associated daily paths by <code>scripts/build_spy_staged_funding_page.py</code>. The rendered page is self-contained; source analysis files are locally cached.</footer>
</main></body></html>"""


def main(argv=None) -> int:
    args = parse_args(argv)
    result = json.loads(args.result.read_text(encoding="utf-8"))
    daily = pd.read_csv(args.daily, parse_dates=["date"]).set_index("date")
    all_at_50 = (json.loads(args.all_at_50.read_text(encoding="utf-8"))
                 if args.all_at_50.exists() else None)
    twenty_year = (json.loads(args.twenty_year.read_text(encoding="utf-8"))
                   if args.twenty_year.exists() else None)
    five_x = (json.loads(args.five_x.read_text(encoding="utf-8"))
              if args.five_x.exists() else None)
    grid = (json.loads(args.grid.read_text(encoding="utf-8"))
            if args.grid.exists() else None)
    rescue = (json.loads(args.rescue.read_text(encoding="utf-8"))
              if args.rescue.exists() else None)
    page = render(result, daily, all_at_50, twenty_year, five_x, grid, rescue)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    print(f"Wrote {args.out} ({len(page):,} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
