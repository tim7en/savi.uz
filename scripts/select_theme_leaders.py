"""Score every theme and pick the two or three names that carry it.

Reads the risk-map outputs and answers two questions: which of the hand-labelled
themes are real risk factors, and which members best represent each one.

Themes are scored by factor strength -- the first principal component's
eigenvalue rescaled by theme size -- rather than by average pairwise
correlation. Two things make the obvious statistic misleading: some themes hold
both directions of one factor (long and short Treasury ETFs), which a signed
average reports as incoherent when they are tightly coupled; and a raw variance
share cannot fall below 1/n, which flatters small themes.

The data-driven clusters from the same run are summarised alongside, because
they are what the correlations actually found, and there are far more of them
than there are hand-labelled themes.

Usage:
    PYTHONPATH=src python scripts/select_theme_leaders.py --outdir out/tradfi
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from savi_uz.seed_groups import SEED_RISK_GROUPS  # noqa: E402
from savi_uz.theme_leaders import (  # noqa: E402
    DEFAULT_LEADER_COUNT,
    ThemeSummary,
    factor_strength,
    principal_component,
    summarise_theme,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--outdir", type=Path, default=Path("out/tradfi"))
    parser.add_argument("--metrics", type=Path, default=Path("out/tradfi/metrics.csv"))
    parser.add_argument("--correlation", type=Path, default=Path("out/tradfi/correlation_raw.csv"))
    parser.add_argument("--clusters", type=Path, default=Path("out/tradfi/clusters.json"))
    parser.add_argument("--proxy-map", type=Path, default=Path("out/tradfi/us_proxy_map.csv"))
    parser.add_argument("--count", type=int, default=DEFAULT_LEADER_COUNT,
                        help="leaders to pick per theme (default: 3)")
    parser.add_argument("--min-liquidity", type=float, default=None,
                        help="liquidity_score floor, relaxed if it would empty a theme")
    parser.add_argument("--min-cluster-size", type=int, default=3,
                        help="smallest data-driven cluster to report as a theme")
    return parser.parse_args(argv)


def load_proxy_map(path: Path) -> dict[str, tuple[str, str]]:
    """base_asset -> (best US proxy, verdict); first row per base is the best."""
    if not path.is_file():
        return {}
    best: dict[str, tuple[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            base = row["base_asset"]
            if base not in best:
                best[base] = (row["us_proxy"], row["verdict"])
    return best


def _fmt(value: float, digits: int = 2) -> str:
    return "-" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def summarise_clusters(
    clusters: list[list[str]], corr: pd.DataFrame, metrics: pd.DataFrame, min_size: int
) -> list[tuple[int, list[str], float]]:
    """Size and internal coherence of each data-driven cluster, largest first."""
    rows = []
    for members in clusters:
        available = [symbol for symbol in members if symbol in corr.index]
        if len(available) < min_size:
            continue
        _, explained = principal_component(corr.loc[available, available])
        rows.append((len(available), available, factor_strength(explained, len(available))))
    return sorted(rows, key=lambda row: -row[0])


def build_report(
    themes: list[ThemeSummary],
    clusters: list[tuple[int, list[str], float]],
    metrics: pd.DataFrame,
    unthemed: int,
    count: int,
) -> str:
    real = [t for t in themes if t.is_real]
    fake = [t for t in themes if t.verdict == "not-a-theme"]

    lines = [
        "# Themes and their leaders",
        "",
        f"- Hand-labelled themes declared: **{len(SEED_RISK_GROUPS)}**",
        f"- Measurable against the correlation panel: **{len(themes)}**",
        f"- Holding together as one risk factor: **{len(real)}**",
        f"- Not a single factor: **{len(fake)}**",
        f"- Data-driven clusters of {clusters[0][0] if clusters else 0}+ members: "
        f"**{len(clusters)}**",
        f"- Instruments in the panel with no hand-labelled theme: **{unthemed}**",
        "",
        "Themes are scored by **factor strength**: the first principal component's",
        "eigenvalue rescaled to `(lambda1 - 1) / (n - 1)`. Two corrections are baked in.",
        "",
        "Eigenvalues do not change when a variable's sign is flipped, so a theme holding",
        "both directions of one factor -- `TMF` is 3x long Treasuries, `TBT` is 2x short --",
        "scores as the tight factor it is. A signed average correlation reports that same",
        "theme at -0.29 and buries it.",
        "",
        "The rescaling matters just as much: a raw variance share cannot fall below 1/n,",
        "so a two-name theme scores at least 0.50 no matter how unrelated its members are,",
        "while an eight-name theme starts at 0.125. Both raw numbers are shown so the",
        "adjustment is visible.",
        "",
        "A **negative loading** marks a name that expresses the theme inversely. It is",
        "still a representative of the factor, traded the other way round.",
        "",
        "## Theme scorecard",
        "",
        "| Theme | Members | Factor strength | PC1 var | avg abs rho | Verdict |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for theme in sorted(themes, key=lambda t: -(t.factor_strength if pd.notna(t.factor_strength) else -1)):
        lines.append(
            f"| {theme.label} | {len(theme.members)} | {_fmt(theme.factor_strength)} | "
            f"{_fmt(theme.explained_variance)} | {_fmt(theme.avg_abs_correlation)} | {theme.verdict} |"
        )

    lines += ["", f"## Top {count} to track per theme", "",
              "| Theme | Pick | Base | Loading | Liquidity | Region | US proxy | Proxy verdict |",
              "|---|---|---|---:|---:|---|---|---|"]
    for theme in sorted(themes, key=lambda t: -(t.factor_strength if pd.notna(t.factor_strength) else -1)):
        for index, leader in enumerate(theme.leaders):
            label = theme.label if index == 0 else ""
            direction = " (inverse)" if leader.is_inverse else ""
            proxy = leader.us_proxy if leader.region != "US" else leader.base_asset
            verdict = leader.proxy_verdict if leader.region != "US" else "direct"
            lines.append(
                f"| {label} | {leader.base_asset}{direction} | {leader.symbol} | "
                f"{_fmt(leader.loading)} | {_fmt(leader.liquidity)} | {leader.region} | "
                f"{proxy} | {verdict} |"
            )

    lines += ["", "## What the correlations found on their own", "",
              "The hand-labelled themes are coarse next to the measured structure.",
              "Largest data-driven clusters:", "",
              "| Size | Factor strength | Members |", "|---:|---:|---|"]
    for size, members, strength in clusters[:12]:
        bases = [
            str(metrics.at[symbol, "base_asset"]) if symbol in metrics.index else symbol
            for symbol in members
        ]
        shown = ", ".join(bases[:14]) + (" ..." if len(bases) > 14 else "")
        lines.append(f"| {size} | {_fmt(strength)} | {shown} |")

    lines += ["", "## Reading the verdicts", "",
              "Factor strength reads on a correlation scale.",
              "",
              "- **coherent** -- 0.45 or above. One factor; the leaders are genuine proxies.",
              "- **loose** -- 0.30 to 0.45. A real tilt with substantial idiosyncratic noise;",
              "  two or three names will not capture the whole theme.",
              "- **not-a-theme** -- under 0.30. The members were grouped by story, not by",
              "  shared risk. Track them individually or regroup them.",
              ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for path in (args.metrics, args.correlation):
        if not path.is_file():
            raise SystemExit(f"error: {path} not found; run build_tradfi_risk_map.py first")

    metrics = pd.read_csv(args.metrics, index_col=0)
    corr = pd.read_csv(args.correlation, index_col=0)
    proxy_by_base = load_proxy_map(args.proxy_map)
    liquidity = metrics["liquidity_score"]

    symbol_by_base = {str(row.base_asset): symbol for symbol, row in metrics.iterrows()}

    themes: list[ThemeSummary] = []
    for label, bases in SEED_RISK_GROUPS.items():
        members = [symbol_by_base[base] for base in bases if base in symbol_by_base]
        summary = summarise_theme(
            label, members, corr, liquidity, metrics, proxy_by_base,
            count=args.count, min_liquidity=args.min_liquidity,
        )
        if summary.verdict != "no-members":
            themes.append(summary)

    clusters_payload = json.loads(args.clusters.read_text(encoding="utf-8")) if args.clusters.is_file() else {}
    clusters = summarise_clusters(
        clusters_payload.get("clusters_raw", []), corr, metrics, args.min_cluster_size
    )
    unthemed = int(metrics["seed_group"].isna().sum())

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for theme in themes:
        for rank, leader in enumerate(theme.leaders, start=1):
            rows.append(
                {
                    "theme": theme.label,
                    "theme_factor_strength": round(theme.factor_strength, 4)
                    if pd.notna(theme.factor_strength) else "",
                    "theme_pc1_variance": round(theme.explained_variance, 4),
                    "theme_avg_abs_corr": round(theme.avg_abs_correlation, 4)
                    if pd.notna(theme.avg_abs_correlation) else "",
                    "theme_verdict": theme.verdict,
                    "rank": rank,
                    "base_asset": leader.base_asset,
                    "binance_symbol": leader.symbol,
                    "loading": round(leader.loading, 4),
                    "inverse": leader.is_inverse,
                    "liquidity_score": round(leader.liquidity, 3),
                    "region": leader.region,
                    "us_tradable": leader.base_asset if leader.region == "US" else leader.us_proxy,
                    "us_proxy_verdict": "direct" if leader.region == "US" else leader.proxy_verdict,
                }
            )
    if rows:
        with (args.outdir / "theme_leaders.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    report = build_report(themes, clusters, metrics, unthemed, args.count)
    (args.outdir / "theme_leaders.md").write_text(report, encoding="utf-8")

    print(f"{len(SEED_RISK_GROUPS)} themes declared, {len(themes)} measurable, "
          f"{sum(1 for t in themes if t.is_real)} hold together")
    print(f"{len(clusters)} data-driven clusters of {args.min_cluster_size}+ members, "
          f"{unthemed} instruments unthemed\n")
    for theme in sorted(themes, key=lambda t: -(t.factor_strength if pd.notna(t.factor_strength) else -1)):
        picks = ", ".join(
            f"{leader.base_asset}{'(-)' if leader.is_inverse else ''}" for leader in theme.leaders
        )
        print(f"  {theme.label:<28} strength {_fmt(theme.factor_strength)}  "
              f"PC1 {_fmt(theme.explained_variance)}  |rho| {_fmt(theme.avg_abs_correlation)}  "
              f"{theme.verdict:<13} {picks}")
    print(f"\nwrote {args.outdir / 'theme_leaders.csv'} and {args.outdir / 'theme_leaders.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        traceback.print_exc(limit=0)
        raise SystemExit(130)
