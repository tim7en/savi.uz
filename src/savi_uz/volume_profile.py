"""Volume profiles built from OHLCV bars, and the shape vocabulary over them.

A volume profile is volume-at-price, not volume-at-time. Bars give the wrong
projection of that, so each bar's volume is spread across its own high-low range
before binning -- the standard reconstruction when tick data is not on hand. It
is an approximation: a bar that traded most of its size at the close is modelled
as if it traded evenly through the range. The finer the bars, the smaller the
error, which is why resolution matters more here than history length.

Everything in this module is computed from a *prefix* of a session. Nothing
reads a bar that has not closed, because the study that consumes it is trying to
predict the next bar and any leak would flatter the result.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Share of volume that defines the value area, the market-profile convention.
VALUE_AREA_SHARE = 0.70

#: Enough bins to resolve a shape, few enough that hourly bars do not shatter
#: into noise.
DEFAULT_BINS = 24

#: A session needs at least this many closed bars before a profile means
#: anything; below it the "shape" is an artefact of one or two bars.
MIN_BARS_FOR_PROFILE = 3

#: POC inside the middle band is balanced; outside it the profile is one-sided.
POC_UPPER_BAND = 0.65
POC_LOWER_BAND = 0.35

#: A second peak counts as a separate distribution only if it is this fraction
#: of the main peak and separated from it by a genuine trough.
SECOND_PEAK_SHARE = 0.60
TROUGH_SHARE = 0.55


@dataclass(frozen=True)
class Bar:
    """One closed bar. Volume may be absent; such bars are excluded."""

    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float | None


@dataclass(frozen=True)
class Profile:
    """Volume-at-price over a set of closed bars."""

    edges: tuple[float, ...]
    volume: tuple[float, ...]
    poc: float
    value_low: float
    value_high: float
    total_volume: float
    low: float
    high: float
    shape: str
    peaks: int

    @property
    def price_range(self) -> float:
        return self.high - self.low

    @property
    def poc_position(self) -> float:
        """Where the point of control sits in the range: 0 at low, 1 at high."""
        span = self.price_range
        return 0.5 if span <= 0 else (self.poc - self.low) / span

    @property
    def value_width(self) -> float:
        """Value-area width as a share of the whole range.

        Narrow means volume is stacked in a small band -- the coiled case the
        breakout study is looking for.
        """
        span = self.price_range
        return 1.0 if span <= 0 else (self.value_high - self.value_low) / span

    def concentration(self, top: int = 3) -> float:
        """Share of volume in the busiest ``top`` bins. Peakedness, scale-free."""
        if self.total_volume <= 0:
            return 0.0
        return sum(sorted(self.volume, reverse=True)[:top]) / self.total_volume


def _bin_index(price: float, low: float, width: float, bins: int) -> int:
    if width <= 0:
        return 0
    return max(0, min(bins - 1, int((price - low) / width)))


def build_profile(bars: list[Bar], bins: int = DEFAULT_BINS) -> Profile | None:
    """Volume-at-price across ``bars``, or None if there is nothing to profile.

    Each bar's volume is spread uniformly over the bins its high-low range
    touches, weighted by how much of the bin the bar covers, so a bar that
    straddles two bins does not dump its whole size into one.
    """
    usable = [b for b in bars if b.volume and b.high >= b.low]
    if len(usable) < 1:
        return None

    low = min(b.low for b in usable)
    high = max(b.high for b in usable)
    if high <= low:
        # A session that never moved: one bin, and the POC is the price.
        total = sum(b.volume for b in usable)
        return Profile(
            edges=(low, high), volume=(total,), poc=low, value_low=low, value_high=high,
            total_volume=total, low=low, high=high, shape="flat", peaks=1,
        )

    width = (high - low) / bins
    buckets = [0.0] * bins
    for bar in usable:
        first = _bin_index(bar.low, low, width, bins)
        last = _bin_index(bar.high, low, width, bins)
        touched = last - first + 1
        share = bar.volume / touched
        for index in range(first, last + 1):
            buckets[index] += share

    total = sum(buckets)
    if total <= 0:
        return None

    poc_index = max(range(bins), key=lambda i: buckets[i])
    poc = low + (poc_index + 0.5) * width

    # Value area: grow out from the POC, always taking the richer neighbour,
    # until the target share of volume is enclosed.
    lower = upper = poc_index
    captured = buckets[poc_index]
    target = total * VALUE_AREA_SHARE
    while captured < target and (lower > 0 or upper < bins - 1):
        below = buckets[lower - 1] if lower > 0 else -1.0
        above = buckets[upper + 1] if upper < bins - 1 else -1.0
        if above >= below:
            upper += 1
            captured += buckets[upper]
        else:
            lower -= 1
            captured += buckets[lower]

    edges = tuple(low + i * width for i in range(bins + 1))
    shape, peaks = classify_shape(buckets, (poc - low) / (high - low))
    return Profile(
        edges=edges,
        volume=tuple(buckets),
        poc=poc,
        value_low=low + lower * width,
        value_high=low + (upper + 1) * width,
        total_volume=total,
        low=low,
        high=high,
        shape=shape,
        peaks=peaks,
    )


def count_peaks(buckets: list[float]) -> int:
    """Separated volume modes: a second peak needs a real trough before it.

    Without the trough test every jagged histogram reads as multi-modal, which
    would put most sessions in the double-distribution bucket and make the shape
    feature meaningless.
    """
    if not buckets:
        return 0
    peak = max(buckets)
    if peak <= 0:
        return 0

    modes = 0
    index = 0
    size = len(buckets)
    while index < size:
        if buckets[index] < peak * SECOND_PEAK_SHARE:
            index += 1
            continue
        # Walk to the end of this above-threshold run.
        run_end = index
        while run_end + 1 < size and buckets[run_end + 1] >= peak * SECOND_PEAK_SHARE:
            run_end += 1
        modes += 1
        # Require a genuine dip before another run can count.
        trough = run_end + 1
        while trough < size and buckets[trough] >= peak * TROUGH_SHARE:
            trough += 1
        index = max(trough, run_end + 1)
    return modes


def bimodality(volume: list[float]) -> float:
    """How convincingly two-humped a histogram is, in [0, 1].

    Scores the best split into two modes: the height of the weaker mode against
    the stronger, times how far the trough between them falls. Both terms are
    needed -- a tall second peak with a shallow dip is one broad distribution,
    and a deep dip beside a negligible second peak is noise.
    """
    if len(volume) < 5:
        return 0.0
    peak = max(volume)
    if peak <= 0:
        return 0.0
    best = 0.0
    for split in range(2, len(volume) - 2):
        left = max(volume[:split])
        right = max(volume[split:])
        weaker = min(left, right)
        if weaker <= 0:
            continue
        left_at = volume.index(left)
        right_at = split + volume[split:].index(right)
        trough = min(volume[left_at:right_at + 1]) if right_at > left_at else weaker
        separation = 1.0 - trough / weaker
        best = max(best, (weaker / peak) * separation)
    return best


def classify_shape(buckets: list[float], poc_position: float) -> tuple[str, int]:
    """Market-profile shape vocabulary, from the volume distribution.

    - ``P``  volume stacked at the top with a thin tail below: buying tail,
      typically short covering that stalls.
    - ``b``  volume stacked at the bottom with a thin tail above: long
      liquidation that stalls.
    - ``B``  two separated distributions -- the classic w, an unresolved
      session with two acceptance areas.
    - ``D``  balanced, single peak near the middle.
    """
    peaks = count_peaks(buckets)
    if peaks >= 2:
        return "B", peaks
    if poc_position >= POC_UPPER_BAND:
        return "P", peaks
    if poc_position <= POC_LOWER_BAND:
        return "b", peaks
    return "D", peaks


SHAPE_NAMES = {
    "P": "P - volume stacked high, thin below",
    "b": "b - volume stacked low, thin above",
    "B": "B/w - two distributions",
    "D": "D - balanced, single peak",
    "flat": "flat - no range",
}
