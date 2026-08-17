"""Extract session volume profiles for the gallery page.

Uses the same `build_profile` the breakout study uses, so what the page shows is
exactly what the analysis measured -- not a second, prettier reconstruction.

Each session yields the binned volume-at-price, the point of control, the value
area, and the intraday price path, plus the shape label.

Usage:
    PYTHONPATH=src python scripts/build_profile_gallery.py --count 20
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.volume_profile import (  # noqa: E402
    SHAPE_NAMES,
    Bar,
    bimodality,
    build_profile,
)

#: A session must be this well covered before its profile means anything --
#: IEX reported no volume at all between August 2017 and April 2018.
MIN_COVERAGE = 0.90


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/intraday/bars.db"))
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--frequency", default="5min")
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument("--count", type=int, default=20, help="consecutive sessions to show")
    parser.add_argument("--end", default=None, help="last session to include (default: latest)")
    parser.add_argument("--outdir", type=Path, default=Path("out/gallery"))
    return parser.parse_args(argv)


def load_sessions(db: Path, ticker: str, frequency: str) -> dict[str, list[Bar]]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT ts, open, high, low, close, volume FROM bars "
        "WHERE ticker = ? AND frequency = ? ORDER BY ts",
        (ticker, frequency),
    ).fetchall()
    sessions: dict[str, list[Bar]] = collections.defaultdict(list)
    for row in rows:
        sessions[row[0][:10]].append(Bar(*row))
    return dict(sessions)


def session_payload(day: str, bars: list[Bar], bins: int) -> dict | None:
    profile = build_profile(bars, bins=bins)
    if profile is None or profile.price_range <= 0:
        return None

    closes = [b.close for b in bars if b.close]
    opening, closing = bars[0].open, bars[-1].close
    total = sum(b.volume for b in bars if b.volume)

    # Where in the range each bin sits, so the page can draw without recomputing.
    width = (profile.high - profile.low) / bins
    return {
        "date": day,
        "low": round(profile.low, 3),
        "high": round(profile.high, 3),
        "open": round(opening, 3),
        "close": round(closing, 3),
        "poc": round(profile.poc, 3),
        "value_low": round(profile.value_low, 3),
        "value_high": round(profile.value_high, 3),
        "shape": profile.shape,
        "peaks": profile.peaks,
        "bin_width": round(width, 4),
        "volume": [round(v, 1) for v in profile.volume],
        "total_volume": round(total, 0),
        "range_pct": round((profile.high - profile.low) / closing * 100, 3),
        "return_pct": round((closing / opening - 1) * 100, 3),
        "value_width": round(profile.value_width, 4),
        "concentration": round(profile.concentration(), 4),
        "poc_position": round(profile.poc_position, 4),
        # Price path, thinned so twenty of them stay light on the page.
        "path": [round(c, 2) for c in closes[::2]],
        "bars": len(bars),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.db.is_file():
        raise SystemExit(f"error: {args.db} not found")

    sessions = load_sessions(args.db, args.ticker, args.frequency)
    usable = []
    for day, bars in sorted(sessions.items()):
        volumed = sum(1 for b in bars if b.volume)
        if len(bars) < 20 or volumed / len(bars) < MIN_COVERAGE:
            continue
        if args.end and day > args.end:
            continue
        usable.append(day)
    if not usable:
        raise SystemExit("error: no sessions with enough volume coverage")

    chosen = usable[-args.count:]
    payload = [session_payload(day, sessions[day], args.bins) for day in chosen]
    payload = [p for p in payload if p]

    # One exemplar per shape, picked as the clearest case so the vocabulary has
    # a picture beside it. For the single-peak shapes that is the most
    # concentrated session; for the double distribution it is genuine
    # bimodality -- two tall modes with a deep trough between them. Scoring `B`
    # by low concentration instead just finds the flattest session, which looks
    # like no distribution at all rather than two.
    exemplars: dict[str, dict] = {}
    for day in usable:
        row = session_payload(day, sessions[day], args.bins)
        if not row:
            continue
        shape = row["shape"]
        score = bimodality(row["volume"]) if shape == "B" else row["concentration"]
        best = exemplars.get(shape)
        if best is None or score > best["_score"]:
            row["_score"] = score
            exemplars[shape] = row
    for row in exemplars.values():
        row.pop("_score", None)

    shapes = collections.Counter(
        p["shape"] for p in (session_payload(d, sessions[d], args.bins) for d in usable) if p
    )

    out = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "ticker": args.ticker,
        "frequency": args.frequency,
        "bins": args.bins,
        "sessions": payload,
        "exemplars": exemplars,
        "shape_names": SHAPE_NAMES,
        "shape_counts": dict(shapes),
        "covered_sessions": len(usable),
        "span": [usable[0], usable[-1]],
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    target = args.outdir / "profiles.json"
    target.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")

    print(f"{args.ticker} {args.frequency}: {len(usable):,} covered sessions "
          f"({usable[0]} -> {usable[-1]})")
    print(f"gallery: {len(payload)} sessions, {chosen[0]} -> {chosen[-1]}")
    print(f"shape mix across all covered sessions: {dict(shapes)}")
    print(f"exemplars: {sorted(exemplars)}")
    print(f"wrote {target} ({target.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
