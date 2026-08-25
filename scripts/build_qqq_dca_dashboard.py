"""Build the QQQ/VXN version of the daily DCA dashboard."""

from __future__ import annotations

from build_spy_dca_dashboard import main as build_main


def main(argv=None) -> int:
    args = list(argv or [])
    return build_main(
        [
            "--ticker", "QQQ",
            "--volatility-series", "VXNCLS",
            "--volatility-label", "VXN",
            "--account", "assets/qqq_dca_account.json",
            "--cache", ".cache/qqq_dca_dashboard",
            "--yahoo-cache", ".cache/yahoo_daily/QQQ.json",
            "--output", "docs/qqq-dca-dashboard.html",
            "--snapshot", "out/strategy/qqq_dca_dashboard/snapshot.json",
            "--fund-url", "https://www.invesco.com/qqq-etf/en/home.html",
            "--updater-command", "scripts/update_qqq_dca_dashboard.ps1",
            *args,
        ]
    )


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
