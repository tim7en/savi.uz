"""Build a risk map of Binance trad-FI perpetuals from Yahoo Finance daily data.

Pulls the live trad-FI contract list off Binance futures, resolves each contract
to its Yahoo underlying (validating the mapping against Binance's own price
history), then clusters the universe on daily returns to show which contracts
are genuinely independent bets and which are the same risk wearing a different
ticker.

Usage:
    PYTHONPATH=src python scripts/build_tradfi_risk_map.py --outdir out/tradfi
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.mapping_check import (  # noqa: E402
    ASSUMED,
    NO_DATA,
    UNVERIFIED,
    USABLE_STATUSES,
    MappingCheck,
    check_mapping,
    pick_best_mapping,
    unlisted_check,
)
from savi_uz.risk_clustering import (  # noqa: E402
    TRADING_DAYS_PER_YEAR,
    TRADING_WEEKS_PER_YEAR,
    annualized_volatility,
    average_intra_cluster_correlation,
    average_linkage,
    cluster_assignments,
    components_for_variance,
    correlation_matrix,
    distance_for_correlation,
    effective_number_of_bets,
    factor_betas,
    log_returns,
    max_correlation_to_others,
    resample_weekly,
    residual_returns,
    select_diversified_basket,
)
from savi_uz.seed_groups import SEED_RISK_GROUPS, seed_base_assets, seed_group_by_base  # noqa: E402
from savi_uz.tradfi_universe import (  # noqa: E402
    UNLISTED_BASES,
    YAHOO_SEARCH_HINTS,
    BinanceTradFiClient,
    TradFiInstrument,
    candidate_yahoo_tickers,
)

MARKET_FACTOR_TICKER = "SPY"
YAHOO_BATCH_SIZE = 50


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", type=Path, default=Path("out/tradfi"), help="directory for CSV/JSON/report output")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/tradfi"), help="raw download cache")
    parser.add_argument("--refresh", action="store_true", help="ignore cached downloads and refetch everything")
    parser.add_argument("--lookback-days", type=int, default=730, help="Yahoo history window (default: 730)")
    parser.add_argument("--freq", choices=("daily", "weekly"), default="daily", help="return frequency for correlations")
    parser.add_argument("--min-periods", type=int, default=60, help="minimum overlapping returns per correlation pair")
    parser.add_argument("--shrinkage", type=float, default=0.10, help="shrink correlations toward the universe average")
    parser.add_argument("--corr-threshold", type=float, default=0.50, help="correlation level at which to cut clusters")
    parser.add_argument("--basket-max-corr", type=float, default=0.35, help="max pairwise |rho| inside the picked basket")
    parser.add_argument(
        "--basket-min-volume",
        type=float,
        default=1_000_000.0,
        help="Binance 24h quote-volume floor for basket candidates (default: 1e6 USDT)",
    )
    parser.add_argument("--min-binance-bars", type=int, default=30, help="minimum Binance daily bars to be tradable")
    parser.add_argument("--seed-only", action="store_true", help="restrict the analysis to the seed table's base assets")
    parser.add_argument("--no-search", action="store_true", help="skip Yahoo symbol search for unmapped contracts")
    parser.add_argument("--include-mirrors", action="store_true", help="allow Yahoo's own Binance-derived series as data")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- cache


def _cache_path(cache_dir: Path, name: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / name


def _load_json_cache(path: Path, refresh: bool):
    if refresh or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json_cache(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, default=str), encoding="utf-8")


# --------------------------------------------------------------------------- fetch


def fetch_binance_side(args: argparse.Namespace) -> tuple[list[TradFiInstrument], dict, dict[str, dict[str, float]]]:
    client = BinanceTradFiClient()

    info_path = _cache_path(args.cache_dir, "binance_universe.json")
    cached = _load_json_cache(info_path, args.refresh)
    if cached is None:
        print("[binance] fetching trad-FI contract list ...")
        instruments = client.fetch_tradfi_instruments()
        _save_json_cache(info_path, [_instrument_to_dict(i) for i in instruments])
    else:
        instruments = [_instrument_from_dict(row) for row in cached]
    print(f"[binance] {len(instruments)} trad-FI perpetuals listed")

    ticker_path = _cache_path(args.cache_dir, "binance_ticker24h.json")
    cached_tickers = _load_json_cache(ticker_path, args.refresh)
    if cached_tickers is None:
        print("[binance] fetching 24h liquidity ...")
        liquidity = client.fetch_24h_liquidity()
        cached_tickers = {
            symbol: {
                "last_price": item.last_price,
                "quote_volume_24h": item.quote_volume_24h,
                "trade_count_24h": item.trade_count_24h,
                "price_change_pct_24h": item.price_change_pct_24h,
            }
            for symbol, item in liquidity.items()
        }
        _save_json_cache(ticker_path, cached_tickers)

    klines_path = _cache_path(args.cache_dir, "binance_daily_closes.json")
    cached_bars = _load_json_cache(klines_path, args.refresh)
    if cached_bars is None:
        symbols = [instrument.binance_symbol for instrument in instruments]
        print(f"[binance] fetching daily klines for {len(symbols)} contracts ...")
        bars = client.fetch_daily_bars_bulk(symbols, limit=500)
        cached_bars = {
            symbol: {day.isoformat(): close for day, close in bar.closes.items()} for symbol, bar in bars.items()
        }
        _save_json_cache(klines_path, cached_bars)

    binance_closes = {
        symbol: {date.fromisoformat(day): float(close) for day, close in series.items()}
        for symbol, series in cached_bars.items()
    }
    return instruments, cached_tickers, binance_closes


def _instrument_to_dict(instrument: TradFiInstrument) -> dict:
    return {
        "binance_symbol": instrument.binance_symbol,
        "base_asset": instrument.base_asset,
        "quote_asset": instrument.quote_asset,
        "underlying_type": instrument.underlying_type,
        "sub_types": list(instrument.sub_types),
        "status": instrument.status,
        "onboard_date": instrument.onboard_date.isoformat() if instrument.onboard_date else None,
    }


def _instrument_from_dict(row: dict) -> TradFiInstrument:
    return TradFiInstrument(
        binance_symbol=row["binance_symbol"],
        base_asset=row["base_asset"],
        quote_asset=row["quote_asset"],
        underlying_type=row["underlying_type"],
        sub_types=tuple(row["sub_types"]),
        status=row["status"],
        onboard_date=date.fromisoformat(row["onboard_date"]) if row["onboard_date"] else None,
    )


def resolve_search_hints(instruments: list[TradFiInstrument], args: argparse.Namespace) -> dict[str, list[str]]:
    """Ask Yahoo to name the tickers we cannot derive, for non-US listings only."""
    if args.no_search:
        return {}

    path = _cache_path(args.cache_dir, "yahoo_search.json")
    cached = _load_json_cache(path, args.refresh)
    if cached is not None:
        return cached

    import yfinance as yf

    from savi_uz.tradfi_universe import CURATED_YAHOO_TICKERS

    targets = {
        instrument.base_asset: YAHOO_SEARCH_HINTS.get(instrument.base_asset, instrument.base_asset)
        for instrument in instruments
        if instrument.base_asset not in CURATED_YAHOO_TICKERS
        and instrument.base_asset not in UNLISTED_BASES
        and instrument.region not in ("US",)
    }
    resolved: dict[str, list[str]] = {}
    for base, query in sorted(targets.items()):
        try:
            quotes = yf.Search(query, max_results=8).quotes
        except Exception as exc:  # network/parse failures must not sink the run
            print(f"[yahoo] search failed for {base!r}: {type(exc).__name__}")
            continue
        symbols = [
            quote["symbol"]
            for quote in quotes
            if quote.get("symbol") and quote.get("exchange") != "CCC" and quote.get("quoteType") in ("EQUITY", "ETF")
        ]
        if symbols:
            resolved[base] = symbols[:4]
            print(f"[yahoo] search {base:<16} -> {', '.join(resolved[base])}")

    _save_json_cache(path, resolved)
    return resolved


def _download_batches(tickers: list[str], start: date, batch_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download in batches, tolerating a batch that Yahoo fails outright."""
    import yfinance as yf

    close_frames, volume_frames = [], []
    for offset in range(0, len(tickers), batch_size):
        batch = tickers[offset : offset + batch_size]
        print(f"[yahoo] downloading {offset + 1}-{offset + len(batch)} of {len(tickers)} tickers ...")
        try:
            frame = yf.download(batch, start=start, interval="1d", auto_adjust=True, progress=False, threads=True)
        except Exception as exc:
            print(f"[yahoo] batch failed: {type(exc).__name__}: {exc}")
            continue
        if frame is None or frame.empty:
            continue
        if not isinstance(frame.columns, pd.MultiIndex):
            frame.columns = pd.MultiIndex.from_product([frame.columns, batch[:1]])
        close_frames.append(frame["Close"])
        volume_frames.append(frame["Volume"])

    closes = pd.concat(close_frames, axis=1, sort=True) if close_frames else pd.DataFrame()
    volumes = pd.concat(volume_frames, axis=1, sort=True) if volume_frames else pd.DataFrame()
    return closes.dropna(axis=1, how="all"), volumes


def _merge_panels(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Union of columns and dates, preferring freshly downloaded values."""
    if existing.empty:
        return incoming.loc[:, ~incoming.columns.duplicated()].sort_index()
    if incoming.empty:
        return existing
    incoming = incoming.loc[:, ~incoming.columns.duplicated()]
    merged = existing.reindex(existing.index.union(incoming.index))
    for column in incoming.columns:
        fresh = incoming[column].reindex(merged.index)
        merged[column] = fresh.combine_first(merged[column]) if column in merged.columns else fresh
    return merged.sort_index()


def download_yahoo_history(tickers: list[str], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adjusted closes and share volumes for every candidate ticker.

    Yahoo drops whole batches at random, so the cache is merged rather than
    overwritten: a flaky run can only ever add columns, never silently delete a
    working series that an earlier run captured.
    """
    close_path = _cache_path(args.cache_dir, "yahoo_closes.csv")
    volume_path = _cache_path(args.cache_dir, "yahoo_volumes.csv")
    attempted_path = _cache_path(args.cache_dir, "yahoo_attempted.json")

    closes, volumes = pd.DataFrame(), pd.DataFrame()
    attempted: set[str] = set()
    if not args.refresh and close_path.exists() and volume_path.exists():
        closes = pd.read_csv(close_path, index_col=0, parse_dates=True)
        volumes = pd.read_csv(volume_path, index_col=0, parse_dates=True)
        attempted = set(_load_json_cache(attempted_path, refresh=False) or [])

    start = (datetime.now() - timedelta(days=args.lookback_days)).date()
    pending = [ticker for ticker in tickers if ticker not in closes.columns and ticker not in attempted]
    if pending:
        new_closes, new_volumes = _download_batches(pending, start, YAHOO_BATCH_SIZE)
        closes = _merge_panels(closes, new_closes)
        volumes = _merge_panels(volumes, new_volumes)

        # One narrower retry for anything the batch pass lost to a transient failure.
        retry = [ticker for ticker in pending if ticker not in closes.columns]
        if retry:
            print(f"[yahoo] retrying {len(retry)} tickers that returned nothing ...")
            retry_closes, retry_volumes = _download_batches(retry, start, max(YAHOO_BATCH_SIZE // 5, 1))
            closes = _merge_panels(closes, retry_closes)
            volumes = _merge_panels(volumes, retry_volumes)

        attempted |= set(pending)
        closes.to_csv(close_path)
        volumes.to_csv(volume_path)
        _save_json_cache(attempted_path, sorted(attempted))

    dead = [ticker for ticker in tickers if ticker not in closes.columns]
    if dead:
        print(f"[yahoo] {len(dead)} tickers have no data at all (e.g. {', '.join(dead[:6])})")
    return closes, volumes


# --------------------------------------------------------------------------- mapping


def resolve_mappings(
    instruments: list[TradFiInstrument],
    binance_closes: dict[str, dict[date, float]],
    yahoo_closes: pd.DataFrame,
    search_hints: dict[str, list[str]],
) -> dict[str, MappingCheck]:
    """Score every candidate ticker per contract and keep the best-validating one."""
    yahoo_series = {
        ticker: {stamp.date(): float(value) for stamp, value in yahoo_closes[ticker].dropna().items()}
        for ticker in yahoo_closes.columns
    }

    mappings: dict[str, MappingCheck] = {}
    for instrument in instruments:
        symbol = instrument.binance_symbol
        if instrument.base_asset in UNLISTED_BASES:
            mappings[symbol] = unlisted_check(symbol)
            continue

        candidates = candidate_yahoo_tickers(instrument, search_hints.get(instrument.base_asset, ()))
        checks = [
            check_mapping(
                symbol,
                ticker,
                source,
                binance_closes.get(symbol, {}),
                yahoo_series[ticker],
                # HK/KR perps are USD-quoted on a local-currency underlying, so their
                # price ratio is an FX rate rather than 1:1.
                expect_unit_scale=source == "venue-mirror" or instrument.region not in ("HK", "KR"),
            )
            for ticker, source in candidates
            if ticker in yahoo_series
        ]
        best = pick_best_mapping(checks)
        mappings[symbol] = best if best is not None else unlisted_check(symbol)
    return mappings


def all_candidate_tickers(instruments: list[TradFiInstrument], search_hints: dict[str, list[str]]) -> list[str]:
    tickers = {MARKET_FACTOR_TICKER}
    for instrument in instruments:
        for ticker, _ in candidate_yahoo_tickers(instrument, search_hints.get(instrument.base_asset, ())):
            tickers.add(ticker)
    return sorted(tickers)


# --------------------------------------------------------------------------- analysis


def build_universe_frame(
    instruments: list[TradFiInstrument],
    liquidity: dict[str, dict[str, float]],
    binance_closes: dict[str, dict[date, float]],
    mappings: dict[str, MappingCheck],
) -> pd.DataFrame:
    seed_lookup = seed_group_by_base()
    rows = []
    for instrument in instruments:
        symbol = instrument.binance_symbol
        mapping = mappings[symbol]
        ticker_stats = liquidity.get(symbol, {})
        rows.append(
            {
                "binance_symbol": symbol,
                "base_asset": instrument.base_asset,
                "region": instrument.region,
                "underlying_type": instrument.underlying_type,
                "sub_types": ",".join(instrument.sub_types),
                "pre_ipo": instrument.is_pre_ipo,
                "onboard_date": instrument.onboard_date,
                "binance_bars": len(binance_closes.get(symbol, {})),
                "quote_volume_24h": ticker_stats.get("quote_volume_24h", np.nan),
                "trade_count_24h": ticker_stats.get("trade_count_24h", np.nan),
                "last_price": ticker_stats.get("last_price", np.nan),
                "yahoo_ticker": mapping.yahoo_ticker,
                "mapping_source": mapping.source,
                "mapping_status": mapping.status,
                "mapping_rank_corr": round(mapping.rank_corr, 4),
                "mapping_lag": mapping.best_lag,
                "mapping_scale_dispersion": round(mapping.scale_dispersion, 4),
                "mapping_scale_median": round(mapping.scale_median, 6),
                "mapping_overlap_days": mapping.overlap_days,
                "seed_group": seed_lookup.get(instrument.base_asset, ""),
            }
        )
    return pd.DataFrame(rows).set_index("binance_symbol").sort_index()


def build_price_panel(universe: pd.DataFrame, yahoo_closes: pd.DataFrame) -> pd.DataFrame:
    """Close-price panel keyed by Binance symbol for every usable mapping."""
    columns = {}
    for symbol, row in universe.iterrows():
        ticker = row["yahoo_ticker"]
        if not row["usable"] or ticker not in yahoo_closes.columns:
            continue
        columns[symbol] = yahoo_closes[ticker]
    return pd.DataFrame(columns).sort_index()


def analyse(prices: pd.DataFrame, market: pd.Series, args: argparse.Namespace) -> dict:
    if args.freq == "weekly":
        prices = resample_weekly(prices)
        market = resample_weekly(market.to_frame()).iloc[:, 0]
        periods_per_year = TRADING_WEEKS_PER_YEAR
    else:
        periods_per_year = TRADING_DAYS_PER_YEAR

    returns = log_returns(prices)
    market_returns = log_returns(market.to_frame()).iloc[:, 0].rename(MARKET_FACTOR_TICKER)
    factors = market_returns.to_frame()

    raw_corr, pair_counts = correlation_matrix(returns, min_periods=args.min_periods, shrinkage=args.shrinkage)
    residuals = residual_returns(returns, factors)
    residual_corr, _ = correlation_matrix(residuals, min_periods=args.min_periods, shrinkage=args.shrinkage)

    cut_distance = distance_for_correlation(args.corr_threshold)
    raw_tree = average_linkage(raw_corr)
    residual_tree = average_linkage(residual_corr)
    raw_clusters = raw_tree.cut(cut_distance)
    residual_clusters = residual_tree.cut(cut_distance)

    betas = factor_betas(returns, factors)
    return {
        "returns": returns,
        "periods_per_year": periods_per_year,
        "raw_corr": raw_corr,
        "residual_corr": residual_corr,
        "pair_counts": pair_counts,
        "raw_tree": raw_tree,
        "residual_tree": residual_tree,
        "raw_clusters": raw_clusters,
        "residual_clusters": residual_clusters,
        "betas": betas,
        "volatility": annualized_volatility(returns, periods_per_year),
        "observations": returns.notna().sum(),
    }


def build_metrics_frame(universe: pd.DataFrame, result: dict, yahoo_volumes: pd.DataFrame) -> pd.DataFrame:
    raw_corr = result["raw_corr"]
    symbols = list(raw_corr.index)
    raw_assignment = cluster_assignments(result["raw_clusters"])
    residual_assignment = cluster_assignments(result["residual_clusters"])

    adv_usd = {}
    for symbol in symbols:
        ticker = universe.at[symbol, "yahoo_ticker"]
        if ticker in yahoo_volumes.columns:
            shares = yahoo_volumes[ticker].tail(60)
            adv_usd[symbol] = float((shares.dropna()).mean())
        else:
            adv_usd[symbol] = np.nan

    metrics = pd.DataFrame(
        {
            "base_asset": universe.loc[symbols, "base_asset"],
            "region": universe.loc[symbols, "region"],
            "seed_group": universe.loc[symbols, "seed_group"],
            "yahoo_ticker": universe.loc[symbols, "yahoo_ticker"],
            "mapping_status": universe.loc[symbols, "mapping_status"],
            "quote_volume_24h": universe.loc[symbols, "quote_volume_24h"],
            "trade_count_24h": universe.loc[symbols, "trade_count_24h"],
            "binance_bars": universe.loc[symbols, "binance_bars"],
            "underlying_adv_shares_60d": pd.Series(adv_usd),
            "ann_volatility": result["volatility"].reindex(symbols),
            "beta_spy": result["betas"][MARKET_FACTOR_TICKER].reindex(symbols),
            "observations": result["observations"].reindex(symbols),
            "cluster_raw": raw_assignment.reindex(symbols),
            "cluster_residual": residual_assignment.reindex(symbols),
            "max_abs_corr": max_correlation_to_others(raw_corr).reindex(symbols),
            "avg_corr_universe": raw_corr.where(~np.eye(len(raw_corr), dtype=bool)).mean().reindex(symbols),
            "max_abs_resid_corr": max_correlation_to_others(result["residual_corr"]).reindex(symbols),
        }
    )
    metrics["liquidity_score"] = np.log10(metrics["quote_volume_24h"].clip(lower=1.0))
    return metrics.sort_values(["cluster_raw", "quote_volume_24h"], ascending=[True, False])


def seed_group_coherence(raw_corr: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Measured cohesion of each hand-labelled group that survived into the panel."""
    rows = []
    for label in SEED_RISK_GROUPS:
        members = [
            symbol
            for symbol in raw_corr.index
            if universe.at[symbol, "seed_group"] == label
        ]
        if not members:
            continue
        rows.append(
            {
                "seed_group": label,
                "members_available": len(members),
                "avg_intra_corr": average_intra_cluster_correlation(raw_corr, members),
                "members": ", ".join(universe.loc[members, "base_asset"]),
            }
        )
    return pd.DataFrame(rows).sort_values("avg_intra_corr", ascending=False)


def pick_basket(metrics: pd.DataFrame, raw_corr: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    eligible = metrics[
        (metrics["quote_volume_24h"] >= args.basket_min_volume)
        & (metrics["binance_bars"] >= args.min_binance_bars)
        & (metrics["mapping_status"].isin(USABLE_STATUSES))
    ]
    if eligible.empty:
        return []
    corr = raw_corr.loc[eligible.index, eligible.index]
    return select_diversified_basket(corr, eligible["liquidity_score"], args.basket_max_corr)


# --------------------------------------------------------------------------- output


def _fmt_money(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    for unit, cutoff in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(value) >= cutoff:
            return f"${value / cutoff:,.1f}{unit}"
    return f"${value:,.0f}"


def write_report(
    path: Path,
    args: argparse.Namespace,
    universe: pd.DataFrame,
    metrics: pd.DataFrame,
    result: dict,
    basket: list[str],
) -> None:
    raw_corr = result["raw_corr"]
    residual_corr = result["residual_corr"]
    lines: list[str] = []
    add = lines.append

    add("# Binance trad-FI risk map")
    add("")
    add(f"Generated {datetime.now():%Y-%m-%d %H:%M}, {args.freq} returns over {args.lookback_days} calendar days.")
    add("")

    add("## Universe")
    add("")
    add(f"- Binance trad-FI perpetuals listed: **{len(universe)}**")
    add(f"- Mapped and validated against a Yahoo underlying: **{int(universe['usable'].sum())}**")
    add(f"- In the correlation panel: **{len(raw_corr)}**")
    status_counts = universe["mapping_status"].value_counts()
    add(f"- Mapping status: {', '.join(f'{count} {status}' for status, count in status_counts.items())}")
    add("")

    enb_raw = effective_number_of_bets(raw_corr)
    enb_resid = effective_number_of_bets(residual_corr)
    add("## How many independent bets are actually here?")
    add("")
    add(f"- Effective number of bets (raw returns): **{enb_raw:.1f}** out of {len(raw_corr)} contracts")
    add(f"- Effective number of bets (SPY-neutral residuals): **{enb_resid:.1f}**")
    add(f"- Principal components for 80% of variance: **{components_for_variance(raw_corr, 0.80)}**")
    add(f"- Clusters at |rho| >= {args.corr_threshold:.2f}: **{len(result['raw_clusters'])}** raw, "
        f"**{len(result['residual_clusters'])}** residual")
    add("")

    add("## Data-driven clusters (raw returns)")
    add("")
    add("| # | Size | Avg intra-corr | Most liquid member | Members |")
    add("|---|------|----------------|--------------------|---------|")
    for index, cluster in enumerate(result["raw_clusters"]):
        members = metrics.loc[cluster]
        leader = members["quote_volume_24h"].idxmax()
        intra = average_intra_cluster_correlation(raw_corr, cluster)
        intra_text = "-" if not np.isfinite(intra) else f"{intra:.2f}"
        names = ", ".join(members["base_asset"])
        add(f"| {index} | {len(cluster)} | {intra_text} | {metrics.at[leader, 'base_asset']} "
            f"({_fmt_money(metrics.at[leader, 'quote_volume_24h'])}) | {names} |")
    add("")

    add("## Hand-labelled groups vs measured correlation")
    add("")
    coherence = seed_group_coherence(raw_corr, universe)
    add("| Seed group | Members in panel | Avg intra-corr |")
    add("|------------|------------------|----------------|")
    for _, row in coherence.iterrows():
        value = row["avg_intra_corr"]
        add(f"| {row['seed_group']} | {row['members_available']} | "
            f"{'-' if not np.isfinite(value) else f'{value:.2f}'} |")
    add("")

    add("## Recommended low-correlation basket")
    add("")
    add(f"Greedy pick by Binance liquidity, capped at pairwise |rho| <= {args.basket_max_corr:.2f}, "
        f"24h quote volume >= {_fmt_money(args.basket_min_volume)}, "
        f"at least {args.min_binance_bars} Binance daily bars.")
    add("")
    if basket:
        basket_corr = raw_corr.loc[basket, basket]
        add(f"Selected **{len(basket)}** contracts; effective bets within the basket: "
            f"**{effective_number_of_bets(basket_corr):.1f}**.")
        add("")
        add("| Contract | Underlying | Region | Seed group | 24h volume | Ann. vol | Beta SPY | Max |rho| in basket |")
        add("|----------|------------|--------|------------|-----------:|---------:|---------:|-------------------:|")
        for symbol in basket:
            row = metrics.loc[symbol]
            others = [other for other in basket if other != symbol]
            worst = max((abs(raw_corr.at[symbol, other]) for other in others), default=0.0)
            add(f"| {symbol} | {row['yahoo_ticker']} | {row['region']} | {row['seed_group'] or '-'} | "
                f"{_fmt_money(row['quote_volume_24h'])} | {row['ann_volatility']:.1%} | "
                f"{row['beta_spy']:.2f} | {worst:.2f} |")
    else:
        add("_No contracts cleared the liquidity and mapping filters._")
    add("")

    add("## Redundant pairs (same risk, two tickers)")
    add("")
    add("| A | B | rho |")
    add("|---|---|-----|")
    pairs = _top_pairs(raw_corr, 15)
    for left, right, value in pairs:
        add(f"| {left} | {right} | {value:.3f} |")
    add("")

    add("## Caveats")
    add("")
    add("- Correlations use Yahoo underlying closes. US, HK and KR sessions close at different times, "
        "so cross-region daily correlations are biased low; rerun with `--freq weekly` to check.")
    add("- Binance trad-FI perps are young (most under a year), so Binance-native history is too short "
        "for correlation work. The underlying's history is the proxy, and it ignores perp-specific "
        "basis, funding and the fact that these contracts halt when the cash market is closed.")
    add("- Liquidity figures are a single 24h snapshot and move a lot; re-check before sizing.")
    add(f"- Mappings marked `{UNVERIFIED}` or `{NO_DATA}` were excluded from the panel; "
        "review them in `universe.csv` before trading those contracts.")
    assumed = universe.index[universe["mapping_status"] == ASSUMED].tolist()
    if assumed:
        add(f"- Listed too recently on Binance to validate, so the mapping is taken on trust "
            f"(`{ASSUMED}`): {', '.join(universe.loc[assumed, 'base_asset'])}.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _top_pairs(corr: pd.DataFrame, count: int) -> list[tuple[str, str, float]]:
    values = corr.to_numpy(dtype=float).copy()
    np.fill_diagonal(values, np.nan)
    upper = np.triu(np.ones_like(values, dtype=bool), k=1)
    candidates = [
        (corr.index[i], corr.columns[j], values[i, j])
        for i, j in zip(*np.where(upper))
        if np.isfinite(values[i, j])
    ]
    candidates.sort(key=lambda item: -abs(item[2]))
    return candidates[:count]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    instruments, liquidity, binance_closes = fetch_binance_side(args)
    if args.seed_only:
        wanted = set(seed_base_assets())
        instruments = [i for i in instruments if i.base_asset in wanted]
        print(f"[filter] seed-only: {len(instruments)} contracts")

    search_hints = resolve_search_hints(instruments, args)
    tickers = all_candidate_tickers(instruments, search_hints)
    yahoo_closes, yahoo_volumes = download_yahoo_history(tickers, args)
    print(f"[yahoo] {yahoo_closes.shape[1]} tickers returned data over {yahoo_closes.shape[0]} sessions")

    mappings = resolve_mappings(instruments, binance_closes, yahoo_closes, search_hints)
    universe = build_universe_frame(instruments, liquidity, binance_closes, mappings)
    usable_sources = {"curated", "derived", "search"} | ({"venue-mirror"} if args.include_mirrors else set())
    universe["usable"] = universe["mapping_status"].isin(USABLE_STATUSES) & universe["mapping_source"].isin(
        usable_sources
    )

    prices = build_price_panel(universe, yahoo_closes)
    prices = prices.loc[:, prices.notna().sum() >= args.min_periods + 1]
    if MARKET_FACTOR_TICKER not in yahoo_closes.columns:
        raise SystemExit(f"market factor {MARKET_FACTOR_TICKER} missing from Yahoo download")
    print(f"[panel] {prices.shape[1]} contracts x {prices.shape[0]} sessions")

    result = analyse(prices, yahoo_closes[MARKET_FACTOR_TICKER], args)
    metrics = build_metrics_frame(universe, result, yahoo_volumes)
    basket = pick_basket(metrics, result["raw_corr"], args)

    order = result["raw_tree"].ordered_labels()
    universe.to_csv(args.outdir / "universe.csv")
    metrics.to_csv(args.outdir / "metrics.csv")
    result["raw_corr"].loc[order, order].round(4).to_csv(args.outdir / "correlation_raw.csv")
    result["residual_corr"].round(4).to_csv(args.outdir / "correlation_residual.csv")
    seed_group_coherence(result["raw_corr"], universe).to_csv(args.outdir / "seed_group_coherence.csv", index=False)
    (args.outdir / "clusters.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "settings": {
                    "freq": args.freq,
                    "lookback_days": args.lookback_days,
                    "corr_threshold": args.corr_threshold,
                    "basket_max_corr": args.basket_max_corr,
                    "basket_min_volume": args.basket_min_volume,
                },
                "effective_bets_raw": effective_number_of_bets(result["raw_corr"]),
                "effective_bets_residual": effective_number_of_bets(result["residual_corr"]),
                "clusters_raw": result["raw_clusters"],
                "clusters_residual": result["residual_clusters"],
                "basket": basket,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(args.outdir / "report.md", args, universe, metrics, result, basket)

    print()
    print(f"effective bets: {effective_number_of_bets(result['raw_corr']):.1f} raw / "
          f"{effective_number_of_bets(result['residual_corr']):.1f} SPY-neutral "
          f"across {len(result['raw_corr'])} contracts")
    print(f"clusters at |rho|>={args.corr_threshold}: {len(result['raw_clusters'])}")
    print(f"basket: {len(basket)} contracts -> {', '.join(basket) if basket else 'none'}")
    print(f"written to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
