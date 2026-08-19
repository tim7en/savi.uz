"""Serve the live tracked-assets dashboard on localhost.

Usage:
    PYTHONPATH=src python scripts/run_dashboard.py --open

The service has no external web dependencies.  Its refresh endpoint updates a
seven-day overlap for every tracked symbol at Tiingo's quota-safe 45 requests
per hour and exposes progress for the browser to poll.
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from savi_uz.dashboard import DEFAULT_DB, RefreshManager, tracked_snapshot  # noqa: E402
from savi_uz.fundamentals import (  # noqa: E402
    DEFAULT_FOLDER as DEFAULT_FUNDAMENTALS_FOLDER,
    DEFAULT_REQUESTS_PER_MINUTE as DEFAULT_AV_REQUESTS_PER_MINUTE,
    PLAN_REQUESTS_PER_MINUTE,
    FundamentalsRefreshManager,
    EarningsEstimatesRefreshManager,
    fundamentals_snapshot,
)
from savi_uz.dashboard_sections import (  # noqa: E402
    ALPHAVANTAGE_OPTIONS_DB,
    CFTC_DB,
    CROSS_ASSET_FOLDER,
    EQUITY_DB,
    INTRADAY_DB,
    MACRO_DB,
    MARKETDATA_OPTIONS_DB,
    CrossAssetRefreshManager,
    cftc_snapshot,
    cross_assets_snapshot,
    earnings_analysis_snapshot,
    fed_snapshot,
    options_snapshot,
)


DEFAULT_PAGE = Path("assets/tracked_dashboard.html")


class DashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], db_path: Path, page_path: Path,
                 fundamentals_folder: Path, requests_per_hour: int,
                 alphavantage_requests_per_minute: int) -> None:
        super().__init__(address, DashboardHandler)
        self.db_path = db_path
        self.page_path = page_path
        self.fundamentals_folder = fundamentals_folder
        self.refresh_manager = RefreshManager(db_path, requests_per_hour=requests_per_hour)
        self.fundamentals_refresh_manager = FundamentalsRefreshManager(
            fundamentals_folder, requests_per_minute=alphavantage_requests_per_minute
        )
        self.earnings_estimates_refresh_manager = EarningsEstimatesRefreshManager(
            fundamentals_folder, requests_per_minute=alphavantage_requests_per_minute
        )
        self.cross_asset_refresh_manager = CrossAssetRefreshManager(
            CROSS_ASSET_FOLDER, requests_per_minute=alphavantage_requests_per_minute
        )


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardHTTPServer

    def _send(self, body: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: object, status: int = HTTPStatus.OK) -> None:
        self._send(
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            try:
                self._send(self.server.page_path.read_bytes(), "text/html; charset=utf-8")
            except OSError as exc:
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
        elif path == "/api/stocks":
            try:
                self._json(tracked_snapshot(self.server.db_path))
            except Exception as exc:
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
        elif path == "/api/refresh":
            self._json(self.server.refresh_manager.status())
        elif path == "/api/fundamentals":
            try:
                self._json(fundamentals_snapshot(self.server.fundamentals_folder))
            except Exception as exc:
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
        elif path == "/api/fundamentals/refresh":
            self._json(self.server.fundamentals_refresh_manager.status())
        elif path == "/api/earnings-analysis":
            try:
                self._json(earnings_analysis_snapshot(EQUITY_DB, self.server.fundamentals_folder))
            except Exception as exc:
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
        elif path == "/api/earnings-analysis/refresh":
            self._json(self.server.earnings_estimates_refresh_manager.status())
        elif path == "/api/cross-assets":
            try:
                self._json(cross_assets_snapshot(MACRO_DB, INTRADAY_DB, CROSS_ASSET_FOLDER))
            except Exception as exc:
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
        elif path == "/api/cross-assets/refresh":
            self._json(self.server.cross_asset_refresh_manager.status())
        elif path == "/api/options":
            try:
                self._json(options_snapshot(MARKETDATA_OPTIONS_DB, ALPHAVANTAGE_OPTIONS_DB))
            except Exception as exc:
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
        elif path == "/api/cftc":
            try:
                self._json(cftc_snapshot(CFTC_DB))
            except Exception as exc:
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
        elif path == "/api/fed":
            try:
                self._json(fed_snapshot(MACRO_DB))
            except Exception as exc:
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
        elif path == "/api/health":
            self._json({"ok": True})
        elif path == "/favicon.ico":
            self._send(b"", "image/x-icon", HTTPStatus.NO_CONTENT)
        else:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if path not in (
            "/api/refresh", "/api/fundamentals/refresh",
            "/api/earnings-analysis/refresh", "/api/cross-assets/refresh",
        ):
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            managers = {
                "/api/refresh": self.server.refresh_manager,
                "/api/fundamentals/refresh": self.server.fundamentals_refresh_manager,
                "/api/earnings-analysis/refresh": self.server.earnings_estimates_refresh_manager,
                "/api/cross-assets/refresh": self.server.cross_asset_refresh_manager,
            }
            manager = managers[path]
            started = manager.start()
        except ValueError as exc:
            self._json({"error": str(exc)}, HTTPStatus.SERVICE_UNAVAILABLE)
            return
        status = manager.status()
        self._json(status, HTTPStatus.ACCEPTED if started else HTTPStatus.CONFLICT)

    def log_message(self, fmt: str, *args: object) -> None:
        # Keep one useful line per request without BaseHTTPRequestHandler's
        # reverse-DNS-looking noise.
        if sys.stdout is not None:  # pythonw has no console stream on Windows
            sys.stdout.write(f"{self.address_string()} {self.command} {self.path} {fmt % args}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--fundamentals-folder", type=Path, default=DEFAULT_FUNDAMENTALS_FOLDER)
    parser.add_argument("--page", type=Path, default=DEFAULT_PAGE)
    parser.add_argument("--requests-per-hour", type=int, default=45)
    parser.add_argument(
        "--alphavantage-requests-per-minute", type=int,
        default=DEFAULT_AV_REQUESTS_PER_MINUTE,
        help="Alpha Vantage pacing (default 72; premium plan ceiling 75)",
    )
    parser.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    args = parser.parse_args(argv)

    if not args.db.is_file():
        print(f"error: database not found: {args.db}")
        return 2
    if not args.page.is_file():
        print(f"error: dashboard page not found: {args.page}")
        return 2
    if not (args.fundamentals_folder / "sp500_symbols.json").is_file():
        print(f"error: S&P fundamentals folder not found: {args.fundamentals_folder}")
        return 2
    if not 1 <= args.requests_per_hour <= 50:
        print("error: --requests-per-hour must be between 1 and Tiingo's free-tier ceiling of 50")
        return 2
    if not 1 <= args.alphavantage_requests_per_minute <= PLAN_REQUESTS_PER_MINUTE:
        print(f"error: --alphavantage-requests-per-minute must be between 1 and {PLAN_REQUESTS_PER_MINUTE}")
        return 2

    server = DashboardHTTPServer(
        (args.host, args.port), args.db, args.page, args.fundamentals_folder,
        args.requests_per_hour, args.alphavantage_requests_per_minute,
    )
    url = f"http://{args.host}:{server.server_port}/"
    print(f"Savi dashboard: {url}")
    print(f"Database:       {args.db}")
    print(f"Fundamentals:   {args.fundamentals_folder}")
    print("Press Ctrl+C to stop.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
