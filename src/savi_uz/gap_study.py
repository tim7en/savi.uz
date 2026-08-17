"""Overnight gaps: how far price travels before the open, and whether it holds.

The intraday feed here covers the regular session only -- 09:30 to 16:00 ET, no
extended hours -- so the pre-market *path* is not observable. What is observable
is its net result: the gap from one session's close to the next session's open
contains every tick of after-hours and pre-market trading, exactly.

"Sustained" is then a forward question about the session that follows, and it is
measured four ways because they disagree and the disagreement is the finding:

- **Retention.** Where the close sits relative to the gap, as a fraction of it.
  1.0 means the whole gap was held, 0.0 means it was given back exactly, above 1
  means the move extended, below 0 means price crossed back through the prior
  close. This is the single number that answers the question.
- **Fill.** Whether price touched the prior close at any point in the session.
  A gap can be filled intraday and still close having retained most of itself,
  so fill and retention are not the same claim.
- **Acceptance.** Whether the new session builds its value area away from the
  old one. This is the volume-profile answer: a move the market accepts trades
  a new distribution, a move it rejects returns to the prior one.
- **Volume confirmation.** Whether the opening bars carry more volume than the
  session usually does.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from savi_uz.volume_profile import Bar, Profile, build_profile

#: A gap smaller than this is noise around the prior close, not an event.
MIN_GAP_BP = 5.0

#: Bars treated as "the open" when checking whether volume confirmed the move.
OPENING_BARS = 6


@dataclass(frozen=True)
class Gap:
    """One overnight gap and what the following session did with it."""

    session: str
    prior_close: float
    open: float
    close: float
    high: float
    low: float

    gap_bp: float
    direction: int                 # +1 up, -1 down

    # -- what the session did with it --------------------------------
    retained: float                # close relative to the gap; 1.0 = fully held
    filled: bool
    bars_to_fill: int | None
    session_return_bp: float       # open to close
    range_bp: float

    # -- did the market accept the new level? ------------------------
    value_overlap: float           # 0 = new distribution, 1 = same as yesterday
    poc_shift_bp: float
    opening_volume_ratio: float

    @property
    def extended(self) -> bool:
        return self.retained > 1.0

    @property
    def reversed_through(self) -> bool:
        """Price crossed back past the prior close, not merely filled the gap."""
        return self.retained < 0.0


def _overlap(low_a: float, high_a: float, low_b: float, high_b: float) -> float:
    """Share of the smaller band that the two bands share."""
    span = min(high_a - low_a, high_b - low_b)
    if span <= 0:
        return 0.0
    return max(0.0, min(high_a, high_b) - max(low_a, low_b)) / span


def build_gaps(
    sessions: dict[str, list[Bar]],
    bins: int = 30,
    min_gap_bp: float = MIN_GAP_BP,
    min_volume_coverage: float = 0.90,
) -> list[Gap]:
    """One record per session that opened away from the prior close.

    Sessions are paired with the immediately preceding *available* session, and
    a pair spanning a hole in the data is dropped rather than treated as an
    overnight move -- a weekend is one thing, a missing month is another.
    """
    days = sorted(sessions)
    gaps: list[Gap] = []

    for index in range(1, len(days)):
        today, yesterday = days[index], days[index - 1]
        bars, prior_bars = sessions[today], sessions[yesterday]
        if len(bars) < 10 or len(prior_bars) < 10:
            continue

        covered = sum(1 for b in bars if b.volume) / len(bars)
        prior_covered = sum(1 for b in prior_bars if b.volume) / len(prior_bars)
        if covered < min_volume_coverage or prior_covered < min_volume_coverage:
            continue

        prior_close = prior_bars[-1].close
        opening = bars[0].open
        if not prior_close or not opening:
            continue

        gap_bp = (opening / prior_close - 1.0) * 10_000
        if abs(gap_bp) < min_gap_bp:
            continue
        direction = 1 if gap_bp > 0 else -1

        closing = bars[-1].close
        high = max(b.high for b in bars)
        low = min(b.low for b in bars)

        # Retention: where the close landed along the gap. Dividing by the gap
        # itself makes a 200bp gap and a 20bp gap directly comparable.
        span = opening - prior_close
        retained = (closing - prior_close) / span if span else 0.0

        # Fill: did price trade back to the prior close at any point?
        filled = False
        bars_to_fill = None
        for position, bar in enumerate(bars):
            touched = bar.low <= prior_close if direction > 0 else bar.high >= prior_close
            if touched:
                filled = True
                bars_to_fill = position + 1
                break

        today_profile = build_profile(bars, bins=bins)
        prior_profile = build_profile(prior_bars, bins=bins)
        overlap = poc_shift = 0.0
        if today_profile and prior_profile and today_profile.price_range > 0:
            overlap = _overlap(
                today_profile.value_low, today_profile.value_high,
                prior_profile.value_low, prior_profile.value_high,
            )
            poc_shift = (today_profile.poc / prior_profile.poc - 1.0) * 10_000

        volumes = [b.volume for b in bars if b.volume]
        opening_volume = [b.volume for b in bars[:OPENING_BARS] if b.volume]
        ratio = (
            (sum(opening_volume) / len(opening_volume)) / (sum(volumes) / len(volumes))
            if opening_volume and volumes else float("nan")
        )

        gaps.append(
            Gap(
                session=today,
                prior_close=prior_close,
                open=opening,
                close=closing,
                high=high,
                low=low,
                gap_bp=gap_bp,
                direction=direction,
                retained=retained,
                filled=filled,
                bars_to_fill=bars_to_fill,
                session_return_bp=(closing / opening - 1.0) * 10_000,
                range_bp=(high - low) / opening * 10_000,
                value_overlap=overlap,
                poc_shift_bp=poc_shift,
                opening_volume_ratio=ratio,
            )
        )
    return gaps


def group_sessions(bars: list[Bar]) -> dict[str, list[Bar]]:
    sessions: dict[str, list[Bar]] = defaultdict(list)
    for bar in sorted(bars, key=lambda b: b.timestamp):
        sessions[bar.timestamp[:10]].append(bar)
    return dict(sessions)


# ---------------------------------------------------------------- buckets --


@dataclass(frozen=True)
class GapBucket:
    label: str
    count: int
    mean_gap_bp: float
    median_retained: float
    #: Interquartile range of retention. Retention divides by the gap itself, so
    #: a near-zero gap makes it explode; this is how far to trust the median.
    retained_iqr: float
    fill_rate: float
    extend_rate: float
    reverse_rate: float
    mean_overlap: float
    mean_volume_ratio: float
    mean_range_bp: float


def median(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def mean(values: list[float]) -> float:
    clean = [v for v in values if v == v]
    return sum(clean) / len(clean) if clean else float("nan")


def iqr(values: list[float]) -> float:
    ordered = sorted(v for v in values if v == v)
    if len(ordered) < 4:
        return float("nan")
    return ordered[int(len(ordered) * 0.75)] - ordered[int(len(ordered) * 0.25)]


def summarise(gaps: list[Gap], label: str) -> GapBucket:
    """Retention is summarised by the median, not the mean: a handful of
    sessions that extend several times the gap would otherwise drag the average
    above 1 and suggest gaps typically extend, which is not what happens."""
    return GapBucket(
        label=label,
        count=len(gaps),
        mean_gap_bp=mean([abs(g.gap_bp) for g in gaps]),
        median_retained=median([g.retained for g in gaps]),
        retained_iqr=iqr([g.retained for g in gaps]),
        fill_rate=mean([1.0 if g.filled else 0.0 for g in gaps]),
        extend_rate=mean([1.0 if g.extended else 0.0 for g in gaps]),
        reverse_rate=mean([1.0 if g.reversed_through else 0.0 for g in gaps]),
        mean_overlap=mean([g.value_overlap for g in gaps]),
        mean_volume_ratio=mean([g.opening_volume_ratio for g in gaps]),
        mean_range_bp=mean([g.range_bp for g in gaps]),
    )


def bucket_by_size(gaps: list[Gap], edges: tuple[float, ...] = (10, 25, 50, 100)) -> list[GapBucket]:
    """Group by absolute gap size in basis points."""
    groups: dict[str, list[Gap]] = defaultdict(list)
    for gap in gaps:
        size = abs(gap.gap_bp)
        label = f"over {edges[-1]:.0f}bp"
        for index, edge in enumerate(edges):
            if size < edge:
                low = 0 if index == 0 else edges[index - 1]
                label = f"{low:.0f}-{edge:.0f}bp"
                break
        groups[label].append(gap)

    order = [f"{0 if i == 0 else edges[i-1]:.0f}-{e:.0f}bp" for i, e in enumerate(edges)]
    order.append(f"over {edges[-1]:.0f}bp")
    return [summarise(groups[label], label) for label in order if groups.get(label)]
