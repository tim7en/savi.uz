"""Inline the dashboard payload into the page template.

The artifact has to be self-contained -- no external fetches survive the CSP --
so the JSON is embedded in a `<script type="application/json">` block rather
than loaded at runtime.

Usage:
    PYTHONPATH=src python scripts/build_dashboard.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TEMPLATE = Path("assets/dashboard_template.html")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--data", type=Path, default=Path("out/dashboard/data.json"))
    parser.add_argument("--out", type=Path, default=Path("out/dashboard/index.html"))
    args = parser.parse_args(argv)

    for path in (args.template, args.data):
        if not path.is_file():
            print(f"error: {path} not found")
            return 2

    payload = args.data.read_text(encoding="utf-8")
    # A literal "</script>" inside the block would close it early; escaping the
    # angle bracket keeps the JSON valid and the tag intact.
    payload = payload.replace("<", "\\u003c")

    html = args.template.read_text(encoding="utf-8")
    if "__DATA__" not in html:
        print("error: template has no __DATA__ placeholder")
        return 2
    html = html.replace("__DATA__", payload)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
