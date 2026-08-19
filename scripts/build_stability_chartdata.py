"""Compact chart data for the stability report."""

from __future__ import annotations

import json
import pathlib


def histogram(values, bins=24, lo=None, hi=None):
    lo = min(values) if lo is None else lo
    hi = max(values) if hi is None else hi
    if hi <= lo:
        hi = lo + 1.0
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = int((v - lo) / width)
        counts[min(max(idx, 0), bins - 1)] += 1
    return {"lo": lo, "hi": hi, "width": width, "counts": counts}


def pack(path, label):
    d = json.load(open(path))
    c = d["curves"]
    every = c["real_total_r"] + c["uniform_total_r"] + c["shuffled_total_r"]
    lo, hi = min(every), max(every)
    pad = (hi - lo) * 0.05
    return {
        "label": label,
        "instruments": d["instruments"],
        "trades": d["real_trades"],
        "real": d["real"], "uniform": d["uniform_null"],
        "shuffled": d["shuffled_null"], "bootstrap": d["trade_bootstrap"],
        "subsets": d["subsets"], "beats": d["null_beats_real"],
        "hist": {
            "real": histogram(c["real_total_r"], 24, lo - pad, hi + pad),
            "uniform": histogram(c["uniform_total_r"], 24, lo - pad, hi + pad),
            "shuffled": histogram(c["shuffled_total_r"], 24, lo - pad, hi + pad),
        },
        "boot": histogram(sorted(c["bootstrap_final"])[:-5], 26),
    }


out = {
    "daily": pack("out/strategy/stability_daily.json", "Daily"),
    "m30": pack("out/strategy/stability_30m.json", "30-minute"),
}
pathlib.Path("out/strategy/stability_chartdata.json").write_text(
    json.dumps(out, separators=(",", ":")), encoding="utf-8")
print("wrote out/strategy/stability_chartdata.json",
      len(json.dumps(out, separators=(",", ":"))), "bytes")
for k, v in out.items():
    r, u, s = v["real"], v["uniform"], v["shuffled"]
    print(f"  {v['label']:10s} real {r['total_r_median']:+8.0f}R  "
          f"uniform {u['total_r_median']:+8.0f}R  shuffled {s['total_r_median']:+8.0f}R  "
          f"edge {r['total_r_median']/u['total_r_median']-1:+.0%} / "
          f"{r['total_r_median']/s['total_r_median']-1:+.0%}")
