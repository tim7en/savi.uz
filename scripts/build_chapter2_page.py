"""Render chapter two from the measured data, charts included.

The charts are generated here rather than hand-written so every coordinate comes
from the same JSON the prose quotes.  Colours are CSS custom properties defined
once on the page and referenced from inside the inline SVG, which is what lets a
single chart render correctly in both themes without duplicating it.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

SERIES = ("var(--s1)", "var(--s2)", "var(--s3)")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("out/report/chapter2.json"))
    parser.add_argument("--history", type=Path,
                        default=Path("out/report/chapter2_history.json"))
    parser.add_argument("--out", type=Path, default=Path("out/report/chapter2.html"))
    return parser.parse_args(argv)


def money(value):
    return f"{value:,.0f}"


def bars_chart(rows, width=760, row_height=30, pad_left=124, pad_right=64):
    """Grouped horizontal bars: two measures per category."""
    height = pad_top = 34
    height = pad_top + len(rows) * row_height + 26
    top = max(max(r[1], r[2]) for r in rows)
    scale = (width - pad_left - pad_right) / top
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="Growth of one hundred dollars by sector" '
             f'preserveAspectRatio="xMidYMid meet">']
    for gridline in (0, 250, 500, 750, 1000):
        if gridline > top:
            continue
        x = pad_left + gridline * scale
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{pad_top - 8}" '
                     f'x2="{x:.1f}" y2="{height - 22}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{height - 8}" '
                     f'text-anchor="middle">${gridline:,}</text>')
    for index, (name, price, total) in enumerate(rows):
        y = pad_top + index * row_height
        parts.append(f'<text class="cat" x="{pad_left - 10}" y="{y + 15}" '
                     f'text-anchor="end">{name}</text>')
        parts.append(f'<rect class="bar-a" x="{pad_left}" y="{y + 2}" '
                     f'width="{max(price * scale, 1):.1f}" height="9" rx="2"/>')
        parts.append(f'<rect class="bar-b" x="{pad_left}" y="{y + 13}" '
                     f'width="{max(total * scale, 1):.1f}" height="9" rx="2"/>')
        parts.append(f'<text class="val" x="{pad_left + total * scale + 6}" '
                     f'y="{y + 21}">${money(total)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def line_chart(series, bands=None, width=760, height=260, pad=(30, 16, 26, 48),
               label="", zero=False, log=False):
    """One or more monthly series; bands shade date ranges behind them."""
    top_pad, right_pad, bottom_pad, left_pad = pad
    months = sorted({m for s in series for m, _ in s["points"]})
    if not months:
        return ""
    index = {m: i for i, m in enumerate(months)}
    values = [v for s in series for _, v in s["points"]]
    low, high = min(values), max(values)
    if zero:
        span = max(abs(low), abs(high))
        low, high = -span, span
    pad_v = (high - low) * 0.08 or 1
    low, high = low - pad_v, high + pad_v
    plot_w = width - left_pad - right_pad
    plot_h = height - top_pad - bottom_pad

    def px(month):
        return left_pad + index[month] / max(len(months) - 1, 1) * plot_w

    def py(value):
        return top_pad + (high - value) / (high - low) * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="{label}" preserveAspectRatio="xMidYMid meet">']
    for band in bands or []:
        start, end = band
        if start[:7] in index and end[:7] in index:
            x1, x2 = px(start[:7]), px(end[:7])
            parts.append(f'<rect class="band" x="{x1:.1f}" y="{top_pad}" '
                         f'width="{max(x2 - x1, 2):.1f}" height="{plot_h}"/>')
    steps = 4
    for i in range(steps + 1):
        value = low + (high - low) * i / steps
        y = py(value)
        parts.append(f'<line class="grid" x1="{left_pad}" y1="{y:.1f}" '
                     f'x2="{width - right_pad}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{left_pad - 8}" y="{y + 4:.1f}" '
                     f'text-anchor="end">{value:,.0f}</text>')
    if zero:
        parts.append(f'<line class="zero" x1="{left_pad}" y1="{py(0):.1f}" '
                     f'x2="{width - right_pad}" y2="{py(0):.1f}"/>')
    for years in range(2000, 2027, 4):
        month = f"{years}-01"
        if month in index:
            parts.append(f'<text class="tick" x="{px(month):.1f}" '
                         f'y="{height - 6}" text-anchor="middle">{years}</text>')
    for slot, s in enumerate(series):
        points = " ".join(f"{px(m):.1f},{py(v):.1f}" for m, v in s["points"]
                          if m in index)
        parts.append(f'<polyline class="line s{slot + 1}" points="{points}"/>')
        last_month, last_value = s["points"][-1]
        parts.append(f'<circle class="dot s{slot + 1}" cx="{px(last_month):.1f}" '
                     f'cy="{py(last_value):.1f}" r="4"/>')
        parts.append(f'<text class="series-label s{slot + 1}" '
                     f'x="{px(last_month) - 8:.1f}" y="{py(last_value) - 10:.1f}" '
                     f'text-anchor="end">{s["name"]}</text>')
    parts.append("</svg>")
    return "".join(parts)


def diverging_bars(rows, width=760, row_height=28, pad_left=124):
    """Rate betas: negative to the left of a centre line, positive to the right."""
    height = 30 + len(rows) * row_height + 26
    span = max(abs(v) for _, v, _ in rows) * 1.18
    centre = pad_left + (width - pad_left - 60) / 2
    scale = (width - pad_left - 60) / 2 / span
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="Sector sensitivity to a one-point move in the ten-year '
             f'yield" preserveAspectRatio="xMidYMid meet">']
    for step in (-4, -2, 0, 2, 4):
        if abs(step) > span:
            continue
        x = centre + step * scale
        parts.append(f'<line class="{"zero" if step == 0 else "grid"}" x1="{x:.1f}" '
                     f'y1="22" x2="{x:.1f}" y2="{height - 22}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{height - 8}" '
                     f'text-anchor="middle">{step:+d}%</text>')
    for index, (name, beta, tstat) in enumerate(rows):
        y = 30 + index * row_height
        length = abs(beta) * scale
        x = centre - length if beta < 0 else centre
        css = "bar-neg" if beta < 0 else "bar-pos"
        parts.append(f'<text class="cat" x="{pad_left - 10}" y="{y + 13}" '
                     f'text-anchor="end">{name}</text>')
        parts.append(f'<rect class="{css}" x="{x:.1f}" y="{y + 3}" '
                     f'width="{max(length, 1):.1f}" height="13" rx="2"/>')
        anchor = "end" if beta < 0 else "start"
        offset = -6 if beta < 0 else 6
        strong = "" if abs(tstat) > 3 else " (not significant)"
        parts.append(f'<text class="val" x="{(x if beta < 0 else x + length) + offset:.1f}" '
                     f'y="{y + 14}" text-anchor="{anchor}">{beta:+.2f}%{strong}</text>')
    parts.append("</svg>")
    return "".join(parts)


def decade_bars(rows, width=760, bar=34):
    """Real total return by decade: a diverging column chart around zero."""
    height = 210
    span = max(abs(v) for _, v in rows) * 1.15
    step = (width - 60) / len(rows)
    zero = 40 + (height - 80) / 2
    scale = (height - 80) / 2 / span
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Real total '
             f'return by decade" preserveAspectRatio="xMidYMid meet">']
    parts.append(f'<line class="zero" x1="46" y1="{zero:.1f}" '
                 f'x2="{width - 14}" y2="{zero:.1f}"/>')
    for level in (-10, 0, 10):
        y = zero - level * scale
        if level:
            parts.append(f'<line class="grid" x1="46" y1="{y:.1f}" '
                         f'x2="{width - 14}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="40" y="{y + 4:.1f}" '
                     f'text-anchor="end">{level:+d}%</text>')
    for index, (name, value) in enumerate(rows):
        x = 52 + index * step
        h = abs(value) * scale
        y = zero - h if value > 0 else zero
        css = "bar-pos" if value > 0 else "bar-neg"
        parts.append(f'<rect class="{css}" x="{x:.1f}" y="{y:.1f}" '
                     f'width="{step - 8:.1f}" height="{max(h, 1):.1f}" rx="2"/>')
        parts.append(f'<text class="tick" x="{x + (step - 8) / 2:.1f}" '
                     f'y="{height - 22}" text-anchor="middle">{name[:4]}</text>')
        label_y = y - 5 if value > 0 else y + h + 12
        parts.append(f'<text class="val" x="{x + (step - 8) / 2:.1f}" '
                     f'y="{label_y:.1f}" text-anchor="middle">{value:+.0f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def main(argv=None):
    args = parse_args(argv)
    data = json.loads(args.data.read_text(encoding="utf-8"))
    sectors = data["sectors"]
    bench = data["benchmarks"]

    history = json.loads(args.history.read_text(encoding="utf-8"))
    cent = history["century"]
    decade_chart = decade_bars(list(cent["by_decade"].items()))
    drawdown_rows = "".join(
        f'<tr><td>{d["from"]}</td><td>{d["trough"]}</td>'
        f'<td class="n">{d["depth"]:.0f}%</td></tr>' for d in cent["drawdowns"][:6])
    names = {k: v["name"] for k, v in sectors.items()} if sectors else {}
    regime_rows = ""
    for s in history["regimes"]:
        best = (f'{names.get(s["best"], s["best"])} '
                f'{s["sectors"][s["best"]]:+.0f}%') if s["best"] else "&mdash;"
        worst = (f'{names.get(s["worst"], s["worst"])} '
                 f'{s["sectors"][s["worst"]]:+.0f}%') if s["worst"] else "&mdash;"
        regime_rows += (
            f'<tr><td>{s["label"]}</td><td class="n dim">{s["from"][:7]}</td>'
            f'<td class="n dim">{s["months"]}m</td>'
            f'<td class="n">{s["rate_from"]:.2f}&rarr;{s["rate_to"]:.2f}</td>'
            f'<td>{best}</td><td>{worst}</td></tr>')

    ordered = sorted(sectors.values(), key=lambda s: -s["total"]["hundred"])
    sector_bars = bars_chart([(s["name"], s["price"]["hundred"],
                               s["total"]["hundred"]) for s in ordered])

    sp_chart = line_chart([
        {"name": "total return", "points": bench["S&P 500 total"]["path"]},
        {"name": "price only", "points": bench["S&P 500 price"]["path"]},
    ], label="S&P 500 price index beside its total-return twin")

    inversions = [(a, b) for a, b in data["inversions"]]
    cape_chart = line_chart([{"name": "CAPE", "points": data["cape"]}],
                            bands=inversions,
                            label="Cyclically adjusted price-earnings ratio")
    curve_chart = line_chart(
        [{"name": "10y minus 2y", "points": data["curve_spread"]}],
        bands=inversions, zero=True,
        label="Ten-year minus two-year Treasury yield")

    others = sorted(data["others"].values(), key=lambda o: -o["total"]["hundred"])
    other_rows = "".join(
        f'<tr><td>{o["name"]}</td><td class="n">{o["total"]["from"][:7]}</td>'
        f'<td class="n">${money(o["total"]["hundred"])}</td>'
        f'<td class="n">{o["total"]["cagr"]:.1%}</td>'
        f'<td class="n">{o["total"]["max_drawdown"]:.0%}</td></tr>'
        for o in others)

    sector_rows = "".join(
        f'<tr><td>{s["name"]}</td>'
        f'<td class="n">${money(s["price"]["hundred"])}</td>'
        f'<td class="n">${money(s["total"]["hundred"])}</td>'
        f'<td class="n">{s["dividend_gap"]:+.0%}</td>'
        f'<td class="n">{s["total"]["cagr"]:.1%}</td>'
        f'<td class="n">{s["total"]["max_drawdown"]:.0%}</td>'
        f'<td class="n">{s["total"]["positive_years"]}/{s["total"]["total_years"]}</td>'
        f'</tr>' for s in ordered)

    betas = sorted(data["rate_betas"].values(), key=lambda b: b["beta"])
    beta_chart = diverging_bars([(b["name"], b["beta"], b["t"]) for b in betas])
    beta_rows = "".join(
        f'<tr><td>{b["name"]}</td><td class="n">{b["beta"]:+.2f}%</td>'
        f'<td class="n">{b["t"]:+.1f}</td>'
        + (f'<td class="n dim">{b["lagged_beta"]:+.2f}%</td>'
           f'<td class="n dim">{b["lagged_t"]:+.1f}</td>'
           if b["lagged_beta"] is not None
           else '<td class="n dim">&mdash;</td><td class="n dim">&mdash;</td>')
        + f'<td>{"bond proxy" if b["beta"] < -1 else "cyclical" if b["beta"] > 1 else "neither"}</td>'
        f'</tr>' for b in betas)

    names = {k: v["name"] for k, v in sectors.items()}
    order = [k for k, _ in sorted(sectors.items(), key=lambda z: z[1]["name"])]
    head = "".join(f'<th class="n">{names[k][:4]}</th>' for k in order)
    episode_rows = ""
    for episode in data["episodes"]:
        cells = ""
        values = episode["sectors"]
        if len(values) < 5:
            continue
        best = max(values, key=values.get)
        worst = min(values, key=values.get)
        for k in order:
            if k not in values:
                cells += '<td class="n dim">&mdash;</td>'
                continue
            css = "best" if k == best else "worst" if k == worst else ""
            cells += f'<td class="n {css}">{values[k]:+.0%}</td>'
        episode_rows += (f'<tr><td>{episode["name"]}</td>'
                         f'<td class="n dim">{episode["from"][:7]}</td>{cells}</tr>')

    template = Path(__file__).with_name("chapter2_template.html")
    html = template.read_text(encoding="utf-8")
    for key, value in (("__SECTOR_BARS__", sector_bars),
                       ("__SECTOR_ROWS__", sector_rows),
                       ("__SP_CHART__", sp_chart),
                       ("__CAPE_CHART__", cape_chart),
                       ("__CURVE_CHART__", curve_chart),
                       ("__OTHER_ROWS__", other_rows),
                       ("__BETA_CHART__", beta_chart),
                       ("__BETA_ROWS__", beta_rows),
                       ("__EPISODE_HEAD__", head),
                       ("__EPISODE_ROWS__", episode_rows),
                       ("__DECADE_CHART__", decade_chart),
                       ("__DRAWDOWN_ROWS__", drawdown_rows),
                       ("__REGIME_ROWS__", regime_rows)):
        html = html.replace(key, value)
    # The prose quotes figures by hand, so it can drift from the data silently.
    # These are the load-bearing ones; a mismatch stops the build rather than
    # shipping a chapter that disagrees with its own tables.
    cape = dict(data["cape"])
    checks = [
        ("S&P price", f"${bench['S&P 500 price']['hundred']:,.0f}"),
        ("S&P total", f"${bench['S&P 500 total']['hundred']:,.0f}"),
        ("CAPE 2000-03", f"{cape.get('2000-03')}"),
        ("CAPE 2009-03", f"{cape.get('2009-03')}"),
        ("CAPE 2021-12", f"{cape.get('2021-12')}"),
        ("utilities beta", f"{data['rate_betas']['XLU']['beta']:.2f}"),
        ("financials beta", f"{data['rate_betas']['XLF']['beta']:.2f}"),
    ]
    missing = [name for name, needle in checks
               if needle.lstrip("$").rstrip("%") not in html]
    if missing:
        raise SystemExit("prose disagrees with the data for: " + ", ".join(missing))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"  wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
