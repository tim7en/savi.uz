"""Correlation clustering and diversification metrics for a tradable universe."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
TRADING_WEEKS_PER_YEAR = 52


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns, with non-positive prices treated as missing."""
    clean = prices.where(prices > 0)
    return np.log(clean).diff()


def resample_weekly(prices: pd.DataFrame) -> pd.DataFrame:
    """Last observation per ISO week; damps the stale-close bias across time zones."""
    return prices.resample("W-FRI").last()


def annualized_volatility(returns: pd.DataFrame, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> pd.Series:
    return returns.std() * np.sqrt(periods_per_year)


def correlation_matrix(
    returns: pd.DataFrame,
    min_periods: int = 60,
    shrinkage: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise correlation, shrunk toward the universe average and repaired to PSD.

    Returns the usable matrix plus the count of overlapping observations behind
    each pair, so thin estimates can be reported rather than silently trusted.
    """
    if not 0.0 <= shrinkage < 1.0:
        raise ValueError("shrinkage must be in [0, 1)")

    raw = returns.corr(min_periods=min_periods)
    overlap = returns.notna().astype(float)
    pair_counts = overlap.T @ overlap

    off_diagonal = raw.to_numpy()[~np.eye(len(raw), dtype=bool)]
    average_corr = float(np.nanmean(off_diagonal)) if np.isfinite(off_diagonal).any() else 0.0

    values = raw.to_numpy(dtype=float).copy()
    values[~np.isfinite(values)] = average_corr
    values = (1.0 - shrinkage) * values + shrinkage * average_corr
    np.fill_diagonal(values, 1.0)
    values = 0.5 * (values + values.T)
    values = _nearest_positive_definite(values)

    return pd.DataFrame(values, index=raw.index, columns=raw.columns), pair_counts


def _nearest_positive_definite(matrix: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Clip negative eigenvalues and renormalise back to unit diagonal."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if eigenvalues.min() >= floor:
        return matrix
    repaired = eigenvectors @ np.diag(np.clip(eigenvalues, floor, None)) @ eigenvectors.T
    scale = np.sqrt(np.diag(repaired))
    repaired = repaired / np.outer(scale, scale)
    np.fill_diagonal(repaired, 1.0)
    return repaired


def residual_returns(returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Returns with common-factor exposure regressed out (market-neutral view)."""
    residuals = {}
    for column in returns.columns:
        aligned = pd.concat([returns[column], factors], axis=1).dropna()
        if len(aligned) < 30:
            residuals[column] = pd.Series(np.nan, index=returns.index)
            continue
        y = aligned.iloc[:, 0].to_numpy()
        design = np.column_stack([np.ones(len(aligned)), aligned.iloc[:, 1:].to_numpy()])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        residuals[column] = pd.Series(y - design @ beta, index=aligned.index).reindex(returns.index)
    return pd.DataFrame(residuals, index=returns.index)


def factor_betas(returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """OLS betas of each asset on each factor (intercept fitted, not reported)."""
    rows = {}
    for column in returns.columns:
        aligned = pd.concat([returns[column], factors], axis=1).dropna()
        if len(aligned) < 30:
            rows[column] = pd.Series(np.nan, index=factors.columns)
            continue
        y = aligned.iloc[:, 0].to_numpy()
        design = np.column_stack([np.ones(len(aligned)), aligned.iloc[:, 1:].to_numpy()])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        rows[column] = pd.Series(beta[1:], index=factors.columns)
    return pd.DataFrame(rows).T


def correlation_distance(corr: pd.DataFrame) -> np.ndarray:
    """Metric distance ``sqrt(0.5 * (1 - rho))``: 0 at rho=1, 1 at rho=-1."""
    distance = np.sqrt(np.clip(0.5 * (1.0 - corr.to_numpy(dtype=float)), 0.0, None))
    np.fill_diagonal(distance, 0.0)
    return distance


def distance_for_correlation(threshold: float) -> float:
    return float(np.sqrt(max(0.5 * (1.0 - threshold), 0.0)))


@dataclass
class Dendrogram:
    """Average-linkage (UPGMA) hierarchy over a correlation distance matrix."""

    labels: list[str]
    merges: list[tuple[int, int, float]] = field(default_factory=list)
    leaf_order: list[int] = field(default_factory=list)

    def cut(self, max_distance: float) -> list[list[str]]:
        """Flat clusters formed by applying every merge below ``max_distance``."""
        parent = list(range(len(self.labels) + len(self.merges)))

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for index, (left, right, distance) in enumerate(self.merges):
            if distance > max_distance:
                continue
            new_node = len(self.labels) + index
            parent[find(left)] = new_node
            parent[find(right)] = new_node

        grouped: dict[int, list[str]] = {}
        for leaf, label in enumerate(self.labels):
            grouped.setdefault(find(leaf), []).append(label)

        order = {label: position for position, label in enumerate(self.ordered_labels())}
        clusters = [sorted(members, key=lambda label: order[label]) for members in grouped.values()]
        return sorted(clusters, key=lambda members: (-len(members), order[members[0]]))

    def ordered_labels(self) -> list[str]:
        return [self.labels[leaf] for leaf in self.leaf_order]


def average_linkage(corr: pd.DataFrame) -> Dendrogram:
    """Agglomerative average-linkage clustering with Lance-Williams updates."""
    labels = list(corr.columns)
    size = len(labels)
    if size == 0:
        return Dendrogram(labels=[])
    if size == 1:
        return Dendrogram(labels=labels, leaf_order=[0])

    distance = correlation_distance(corr)
    active = list(range(size))
    node_ids = {index: index for index in range(size)}
    counts = {index: 1 for index in range(size)}
    children: dict[int, tuple[int, int]] = {}
    merges: list[tuple[int, int, float]] = []

    working = distance.astype(float).copy()
    np.fill_diagonal(working, np.inf)

    for step in range(size - 1):
        submatrix = working[np.ix_(active, active)]
        flat_index = int(np.argmin(submatrix))
        i_local, j_local = divmod(flat_index, len(active))
        i, j = active[i_local], active[j_local]
        merge_distance = float(submatrix[i_local, j_local])

        new_node = size + step
        merges.append((node_ids[i], node_ids[j], merge_distance))
        children[new_node] = (node_ids[i], node_ids[j])

        merged_count = counts[i] + counts[j]
        for k in active:
            if k in (i, j):
                continue
            working[i, k] = working[k, i] = (counts[i] * working[i, k] + counts[j] * working[j, k]) / merged_count
        counts[i] = merged_count
        node_ids[i] = new_node
        active.remove(j)
        working[j, :] = np.inf
        working[:, j] = np.inf

    return Dendrogram(labels=labels, merges=merges, leaf_order=_leaf_order(size, children))


def _leaf_order(leaf_count: int, children: dict[int, tuple[int, int]]) -> list[int]:
    """Depth-first leaf sequence, used to quasi-diagonalise the correlation matrix."""
    root = max(children) if children else 0
    order: list[int] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node < leaf_count:
            order.append(node)
            continue
        left, right = children[node]
        stack.extend((right, left))
    return order


def effective_number_of_bets(corr: pd.DataFrame) -> float:
    """Entropy-based count of independent risk factors in the correlation matrix.

    Equals N for a perfectly diagonal matrix and 1 when everything moves as one.
    """
    eigenvalues = np.linalg.eigvalsh(corr.to_numpy(dtype=float))
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    weights = eigenvalues / eigenvalues.sum()
    return float(np.exp(-np.sum(weights * np.log(weights))))


def pca_variance_profile(corr: pd.DataFrame) -> pd.Series:
    """Cumulative share of variance explained by the leading principal components."""
    eigenvalues = np.sort(np.linalg.eigvalsh(corr.to_numpy(dtype=float)))[::-1]
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    cumulative = np.cumsum(eigenvalues) / eigenvalues.sum()
    return pd.Series(cumulative, index=pd.RangeIndex(1, len(cumulative) + 1, name="components"))


def components_for_variance(corr: pd.DataFrame, target: float = 0.80) -> int:
    profile = pca_variance_profile(corr)
    reached = profile[profile >= target]
    return int(reached.index[0]) if len(reached) else int(profile.index[-1])


def average_intra_cluster_correlation(corr: pd.DataFrame, members: list[str]) -> float:
    if len(members) < 2:
        return float("nan")
    block = corr.loc[members, members].to_numpy(dtype=float)
    off_diagonal = block[~np.eye(len(members), dtype=bool)]
    return float(np.mean(off_diagonal))


def max_correlation_to_others(corr: pd.DataFrame) -> pd.Series:
    """Largest absolute correlation each asset has with any other asset."""
    values = corr.abs().to_numpy(dtype=float).copy()
    np.fill_diagonal(values, np.nan)
    return pd.Series(np.nanmax(values, axis=1), index=corr.index)


def select_diversified_basket(
    corr: pd.DataFrame,
    scores: pd.Series,
    max_abs_correlation: float = 0.35,
    limit: int | None = None,
) -> list[str]:
    """Greedy pick of the highest-scoring assets that stay mutually uncorrelated."""
    ranked = scores.reindex(corr.index).dropna().sort_values(ascending=False)
    selected: list[str] = []
    for candidate in ranked.index:
        if limit is not None and len(selected) >= limit:
            break
        if all(abs(corr.at[candidate, chosen]) <= max_abs_correlation for chosen in selected):
            selected.append(candidate)
    return selected


def cluster_assignments(clusters: list[list[str]]) -> pd.Series:
    """Map each label to its cluster index, ordered as produced by ``Dendrogram.cut``."""
    mapping = {label: index for index, cluster in enumerate(clusters) for label in cluster}
    return pd.Series(mapping, name="cluster")
