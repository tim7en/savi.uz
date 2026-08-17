"""Pick the two or three names that best represent each theme.

Average pairwise correlation is the obvious way to score a theme and it is
wrong for this universe, because some themes deliberately contain both
directions of one factor. "Vol / rates diversifiers" holds TMF (3x long
Treasuries) and TBT (2x short Treasuries): they are the same rates bet with
opposite signs, so averaging their correlation reports -0.29 and makes a
perfectly coherent factor look like noise.

The leading principal component fixes this. Its explained-variance share
measures how much of the theme moves together *regardless of sign*, and each
member's loading says how strongly that member carries the common factor, with
the sign saying which way round it does so. A member with a large negative
loading is an inverse expression of the theme, not a failure of it.

Representativeness alone is not enough to trade, so the selection also takes
liquidity and refuses to return a name that cannot be traded at size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Thresholds apply to the size-adjusted factor strength, not to the raw
#: variance share. The share alone cannot go below 1/n, so a two-name theme
#: scores at least 0.50 however unrelated its members are, while an eight-name
#: theme starts at 0.125 -- comparing the two directly rewards small themes for
#: being small. The adjusted statistic is on a correlation scale.
COHERENT_FACTOR_STRENGTH = 0.45
LOOSE_FACTOR_STRENGTH = 0.30

#: A theme needs at least this many members before a principal component means
#: anything; with two members PC1 is just their correlation.
MIN_MEMBERS_FOR_PCA = 2

DEFAULT_LEADER_COUNT = 3


@dataclass(frozen=True)
class Leader:
    """One selected representative of a theme."""

    symbol: str
    base_asset: str
    loading: float
    liquidity: float
    region: str
    us_proxy: str
    proxy_verdict: str

    @property
    def is_inverse(self) -> bool:
        """A negative loading means the name expresses the theme the other way."""
        return self.loading < 0


@dataclass(frozen=True)
class ThemeSummary:
    label: str
    members: tuple[str, ...]
    explained_variance: float
    #: Size-adjusted factor strength; comparable across themes of any size.
    factor_strength: float
    avg_abs_correlation: float
    leaders: tuple[Leader, ...]
    verdict: str

    @property
    def is_real(self) -> bool:
        return self.verdict in ("coherent", "loose")


def principal_component(corr: pd.DataFrame) -> tuple[pd.Series, float]:
    """Leading eigenvector of a correlation submatrix and its variance share.

    The eigenvector is sign-normalised so the largest-magnitude loading is
    positive; that fixes an arbitrary flip in the decomposition and makes
    "negative loading" mean "inverse to the theme's main expression" rather
    than an artefact of which way numpy happened to point.
    """
    if corr.empty:
        return pd.Series(dtype=float), float("nan")
    matrix = corr.to_numpy(dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0)
    # Symmetrise: the stored matrix is shrunk and PSD-repaired, and tiny
    # asymmetries there would make the eigenvalues complex.
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    leading = vectors[:, -1]
    if leading[np.argmax(np.abs(leading))] < 0:
        leading = -leading
    explained = float(values[-1] / len(matrix)) if len(matrix) else float("nan")
    return pd.Series(leading, index=corr.index), explained


def factor_strength(explained_variance: float, member_count: int) -> float:
    """Variance share rescaled so themes of different sizes compare fairly.

    ``(lambda1 - 1) / (n - 1)``, which maps the attainable range ``[1/n, 1]``
    onto ``[0, 1]``. Under an equicorrelation model this is exactly the common
    correlation, so the number reads on a correlation scale -- and because
    eigenvalues are unchanged by flipping the sign of any variable, it measures
    the factor whether or not the theme mixes long and short expressions of it.
    """
    if member_count < 2 or np.isnan(explained_variance):
        return float("nan")
    eigenvalue = explained_variance * member_count
    return float((eigenvalue - 1.0) / (member_count - 1))


def average_absolute_correlation(corr: pd.DataFrame) -> float:
    """Mean |rho| over distinct pairs; sign-blind, so inverse pairs still count."""
    if len(corr) < 2:
        return float("nan")
    matrix = corr.to_numpy(dtype=float)
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    upper = upper[~np.isnan(upper)]
    return float(np.mean(np.abs(upper))) if upper.size else float("nan")


def classify_theme(strength: float, member_count: int) -> str:
    """Verdict from the size-adjusted factor strength."""
    if member_count < MIN_MEMBERS_FOR_PCA:
        return "too-few-members"
    if np.isnan(strength):
        return "unmeasured"
    if strength >= COHERENT_FACTOR_STRENGTH:
        return "coherent"
    if strength >= LOOSE_FACTOR_STRENGTH:
        return "loose"
    return "not-a-theme"


def select_leaders(
    loadings: pd.Series,
    liquidity: pd.Series,
    count: int = DEFAULT_LEADER_COUNT,
    min_liquidity: float | None = None,
    underlying: pd.Series | None = None,
) -> tuple[str, ...]:
    """The ``count`` names carrying most of the theme, subject to liquidity.

    Ranked by absolute loading so an inverse expression can still represent the
    theme. The liquidity floor is applied first and relaxed if it would empty
    the theme, because returning nothing is less useful than returning the
    tradable-but-thin name with its liquidity attached.

    When ``underlying`` is supplied, only one contract per underlying security
    is returned. Binance lists the same stock more than once -- ``TENCENT`` and
    ``HK0700`` are both 0700.HK -- and two contracts on one company are one
    thing to track, not two.
    """
    if loadings.empty:
        return ()
    ranked = loadings.reindex(loadings.abs().sort_values(ascending=False).index)

    if min_liquidity is not None:
        eligible = [s for s in ranked.index if float(liquidity.get(s, float("-inf"))) >= min_liquidity]
        if eligible:
            ranked = ranked.loc[eligible]

    if underlying is None:
        return tuple(ranked.index[:count])

    picked: list[str] = []
    seen: set[str] = set()
    for symbol in ranked.index:
        key = str(underlying.get(symbol, "") or symbol)
        if key in seen:
            continue
        seen.add(key)
        picked.append(symbol)
        if len(picked) == count:
            break
    return tuple(picked)


def summarise_theme(
    label: str,
    members: list[str],
    corr: pd.DataFrame,
    liquidity: pd.Series,
    metadata: pd.DataFrame,
    proxy_by_base: dict[str, tuple[str, str]],
    count: int = DEFAULT_LEADER_COUNT,
    min_liquidity: float | None = None,
) -> ThemeSummary:
    available = [symbol for symbol in members if symbol in corr.index]
    if not available:
        return ThemeSummary(label, (), float("nan"), float("nan"), float("nan"), (), "no-members")

    submatrix = corr.loc[available, available]
    loadings, explained = principal_component(submatrix)
    strength = factor_strength(explained, len(available))
    verdict = classify_theme(strength, len(available))
    underlying = (
        metadata["yahoo_ticker"] if "yahoo_ticker" in metadata.columns else None
    )
    chosen = select_leaders(
        loadings, liquidity, count=count, min_liquidity=min_liquidity, underlying=underlying
    )

    leaders = []
    for symbol in chosen:
        base = str(metadata.at[symbol, "base_asset"]) if symbol in metadata.index else symbol
        proxy, proxy_verdict = proxy_by_base.get(base, ("", ""))
        leaders.append(
            Leader(
                symbol=symbol,
                base_asset=base,
                loading=float(loadings.get(symbol, float("nan"))),
                liquidity=float(liquidity.get(symbol, float("nan"))),
                region=str(metadata.at[symbol, "region"]) if symbol in metadata.index else "",
                us_proxy=proxy,
                proxy_verdict=proxy_verdict,
            )
        )

    return ThemeSummary(
        label=label,
        members=tuple(available),
        explained_variance=explained,
        factor_strength=strength,
        avg_abs_correlation=average_absolute_correlation(submatrix),
        leaders=tuple(leaders),
        verdict=verdict,
    )
