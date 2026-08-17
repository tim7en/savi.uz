"""Does the shape of a session's volume profile say anything about the next bar?

The whole result hinges on one rule, so it is enforced structurally rather than
by care: **a feature row observed at bar t may only read bars 1..t of that
session, and its target may only read bar t+1.** Both price and volume are known
only when a bar closes, so a profile that includes bar t+1 -- or a session-wide
profile evaluated at bar 3 -- would be reading the answer. `build_samples` walks
each session forward and hands `build_profile` a prefix; it never sees the
future, and `test_breakout_study` asserts that a mutated future cannot change a
past feature.

Two more choices keep the question honest:

- The **last bar of a session is dropped**. Its "next bar" is the next morning,
  so including it would mix an overnight gap -- a different phenomenon, and a
  much larger one -- into an intraday study.
- The target is **absolute** return. "Just before the breakout" is a claim about
  magnitude, not direction; testing it as a directional signal would be a
  different and much stronger claim.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from savi_uz.volume_profile import (
    MIN_BARS_FOR_PROFILE,
    Bar,
    Profile,
    build_profile,
)

#: Bars must be at least this far into the session before a profile is formed.
MIN_PREFIX_BARS = MIN_BARS_FOR_PROFILE

#: Share of the prefix that must actually carry volume. IEX reported no volume
#: at all from August 2017 to April 2018, and partial volume either side of it.
#: Without this floor a session with a handful of volumed bars out of seventy
#: builds a profile that looks like a full session's and is nothing of the kind.
MIN_VOLUME_COVERAGE = 0.60


@dataclass(frozen=True)
class Sample:
    """One decision point: everything known at the close of ``timestamp``."""

    session: str
    timestamp: str
    bars_elapsed: int
    close: float

    # -- features, all from bars 1..t -----------------------------------
    shape: str
    poc_position: float
    close_vs_poc: float
    value_width: float
    concentration: float
    close_position: float
    range_pct: float
    volume_ratio: float

    # -- target, strictly from bar t+1 ----------------------------------
    forward_return: float
    forward_abs: float
    forward_range: float


def _session_of(timestamp: str) -> str:
    return timestamp[:10]


def group_sessions(bars: list[Bar]) -> dict[str, list[Bar]]:
    """Bars grouped by calendar session, each in time order."""
    sessions: dict[str, list[Bar]] = defaultdict(list)
    for bar in sorted(bars, key=lambda b: b.timestamp):
        sessions[_session_of(bar.timestamp)].append(bar)
    return dict(sessions)


def build_samples(
    bars: list[Bar],
    bins: int = 24,
    min_prefix: int = MIN_PREFIX_BARS,
    min_volume_coverage: float = MIN_VOLUME_COVERAGE,
) -> list[Sample]:
    """Walk every session forward, one decision point per closed bar.

    At bar ``t`` the profile is built from bars ``0..t`` inclusive -- all closed
    -- and the target reads bar ``t+1``, which at decision time is unknown. The
    loop stops before the final bar of each session so no target is an overnight
    move.
    """
    samples: list[Sample] = []
    for session, session_bars in sorted(group_sessions(bars).items()):
        if len(session_bars) < min_prefix + 1:
            continue
        volumes: list[float] = []
        for index in range(min_prefix - 1, len(session_bars) - 1):
            prefix = session_bars[: index + 1]
            profile = build_profile(prefix, bins=bins)
            if profile is None or profile.price_range <= 0:
                continue

            current = session_bars[index]
            nxt = session_bars[index + 1]
            if not current.close or not nxt.close:
                continue

            volumes = [b.volume for b in prefix if b.volume]
            if len(volumes) < 2 or not current.volume:
                continue
            if len(volumes) / len(prefix) < min_volume_coverage:
                continue
            earlier = volumes[:-1]
            volume_ratio = current.volume / (sum(earlier) / len(earlier))

            span = profile.price_range
            samples.append(
                Sample(
                    session=session,
                    timestamp=current.timestamp,
                    bars_elapsed=index + 1,
                    close=current.close,
                    shape=profile.shape,
                    poc_position=profile.poc_position,
                    close_vs_poc=(current.close - profile.poc) / span,
                    value_width=profile.value_width,
                    concentration=profile.concentration(),
                    close_position=(current.close - profile.low) / span,
                    range_pct=span / current.close,
                    volume_ratio=volume_ratio,
                    forward_return=nxt.close / current.close - 1.0,
                    forward_abs=abs(nxt.close / current.close - 1.0),
                    forward_range=(nxt.high - nxt.low) / current.close,
                )
            )
    return samples


# ------------------------------------------------------------------ stats --


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def welch_t(a: list[float], b: list[float]) -> float:
    """Welch t-statistic. Unequal variances, which these buckets always have."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = stdev(a) ** 2 / len(a), stdev(b) ** 2 / len(b)
    if va + vb <= 0:
        return float("nan")
    return (mean(a) - mean(b)) / math.sqrt(va + vb)


@dataclass(frozen=True)
class Bucket:
    label: str
    count: int
    mean_abs: float
    mean_range: float
    mean_signed: float
    lift: float
    t_stat: float


def bucket_by(
    samples: list[Sample], key, labels: list[str], baseline: list[Sample] | None = None
) -> list[Bucket]:
    """Group samples by a categorical key and score each group against the rest.

    ``lift`` is the group's mean absolute next-bar move divided by the mean over
    every other sample, so 1.0 is no edge whatever the volatility regime.
    """
    pool = baseline if baseline is not None else samples
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[key(sample)].append(sample)

    out = []
    for label in labels:
        rows = groups.get(label, [])
        if not rows:
            continue
        others = [s for s in pool if key(s) != label]
        mine = [s.forward_abs for s in rows]
        theirs = [s.forward_abs for s in others]
        base = mean(theirs)
        out.append(
            Bucket(
                label=label,
                count=len(rows),
                mean_abs=mean(mine),
                mean_range=mean([s.forward_range for s in rows]),
                mean_signed=mean([s.forward_return for s in rows]),
                lift=mean(mine) / base if base else float("nan"),
                t_stat=welch_t(mine, theirs),
            )
        )
    return out


def quantile_edges(values: list[float], parts: int) -> list[float]:
    ordered = sorted(values)
    if not ordered:
        return []
    return [ordered[min(len(ordered) - 1, int(len(ordered) * i / parts))] for i in range(1, parts)]


def bucket_numeric(
    samples: list[Sample], key, name: str, parts: int = 5, edges: list[float] | None = None
) -> list[Bucket]:
    """Split a continuous feature at quantiles and score each slice.

    ``edges`` may be supplied so a test period is cut at thresholds fitted on
    the training period. Letting the test period pick its own thresholds is a
    quiet look-ahead: the boundary between "wide value area" and "narrow" would
    then be chosen with knowledge of the very data being scored.
    """
    if edges is None:
        edges = quantile_edges([key(s) for s in samples], parts)
    if not edges:
        return []
    # Bucket count follows the edges, not `parts`. When train-fitted edges are
    # supplied the two can disagree, and trusting `parts` would then label the
    # top bucket with an index no edge produces.
    buckets = len(edges) + 1

    def label_of(sample: Sample) -> str:
        value = key(sample)
        for i, edge in enumerate(edges):
            if value < edge:
                return f"{name} Q{i + 1}"
        return f"{name} Q{buckets}"

    return bucket_by(samples, label_of, [f"{name} Q{i + 1}" for i in range(buckets)])


def quantile_labeller(samples: list[Sample], key, parts: int, edges: list[float] | None = None):
    """A function mapping a sample to its quantile bucket of ``key``.

    Pass ``edges`` to reuse thresholds fitted elsewhere -- on the training
    period -- rather than refitting them on the data being scored.
    """
    if edges is None:
        edges = quantile_edges([key(s) for s in samples], parts)

    def label(sample: Sample) -> int:
        value = key(sample)
        for index, edge in enumerate(edges):
            if value < edge:
                return index
        return len(edges)

    return label


def stratified_buckets(samples: list[Sample], key, labels: list[str], controls: list) -> list[Bucket]:
    """Lift measured *within* control strata, then pooled by weight.

    This is the test that matters. Two of the strongest raw signals here --
    session range so far, and how late in the session it is -- are well-known
    volatility effects, and any feature correlated with them inherits their lift
    without adding information. Comparing a bucket only against samples that
    share its stratum removes that borrowed edge; what survives is the feature's
    own contribution.
    """
    strata: dict[tuple, list[Sample]] = defaultdict(list)
    for sample in samples:
        strata[tuple(control(sample) for control in controls)].append(sample)

    # Each sample is normalised by its own stratum's mean, so the ratios are
    # comparable across strata. The t-statistic is then computed on those same
    # normalised values -- testing the quantity the lift actually reports,
    # rather than the raw move, which would answer a different question.
    normalised: dict[str, list[float]] = defaultdict(list)
    raw: dict[str, list[float]] = defaultdict(list)
    ranges: dict[str, list[float]] = defaultdict(list)
    signed: dict[str, list[float]] = defaultdict(list)

    for rows in strata.values():
        if len(rows) < 20:            # too thin to compare inside
            continue
        stratum_mean = mean([r.forward_abs for r in rows])
        if not stratum_mean:
            continue
        grouped: dict[str, list[Sample]] = defaultdict(list)
        for row in rows:
            grouped[key(row)].append(row)
        for label, members in grouped.items():
            if len(members) < 5:
                continue
            normalised[label] += [m.forward_abs / stratum_mean for m in members]
            raw[label] += [m.forward_abs for m in members]
            ranges[label] += [m.forward_range for m in members]
            signed[label] += [m.forward_return for m in members]

    out = []
    for label in labels:
        if not normalised.get(label):
            continue
        others = [v for other, values in normalised.items() if other != label for v in values]
        out.append(
            Bucket(
                label=label,
                count=len(normalised[label]),
                mean_abs=mean(raw[label]),
                mean_range=mean(ranges[label]),
                mean_signed=mean(signed[label]),
                lift=mean(normalised[label]),
                t_stat=welch_t(normalised[label], others),
            )
        )
    return out


def split_by_date(samples: list[Sample], cutoff: str) -> tuple[list[Sample], list[Sample]]:
    """Chronological split. A random split would leak: adjacent bars in one
    session share a profile, so shuffling puts near-duplicates on both sides."""
    train = [s for s in samples if s.session < cutoff]
    test = [s for s in samples if s.session >= cutoff]
    return train, test
