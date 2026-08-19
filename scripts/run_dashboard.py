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


DEFAULT_PAGE = Path("assets/tracked_dashboard.html")


class DashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], db_path: Path, page_path: Path,
                 requests_per_hour: int) -> None:
        super().__init__(address, DashboardHandler)
        self.db_path = db_path
        self.page_path = page_path
        self.refresh_manager = RefreshManager(db_path, requests_per_hour=requests_per_hour)


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
        elif path == "/api/health":
            self._json({"ok": True})
        elif path == "/favicon.ico":
            self._send(b"", "image/x-icon", HTTPStatus.NO_CONTENT)
        else:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if urlparse(self.path).path != "/api/refresh":
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            started = self.server.refresh_manager.start()
        except ValueError as exc:
            self._json({"error": str(exc)}, HTTPStatus.SERVICE_UNAVAILABLE)
            return
        status = self.server.refresh_manager.status()
        self._json(status, HTTPStatus.ACCEPTED if started else HTTPStatus.CONFLICT)

    def log_message(self, fmt: str, *args: object) -> None:
        # Keep one useful line per request without BaseHTTPRequestHandler's
        # reverse-DNS-looking noise.
        sys.stdout.write(f"{self.address_string()} {self.command} {self.path} {fmt % args}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--page", type=Path, default=DEFAULT_PAGE)
    parser.add_argument("--requests-per-hour", type=int, default=45)
    parser.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    args = parser.parse_args(argv)

    if not args.db.is_file():
        print(f"error: database not found: {args.db}")
        return 2
    if not args.page.is_file():
        print(f"error: dashboard page not found: {args.page}")
        return 2
    if not 1 <= args.requests_per_hour <= 50:
        print("error: --requests-per-hour must be between 1 and Tiingo's free-tier ceiling of 50")
        return 2

    server = DashboardHTTPServer((args.host, args.port), args.db, args.page, args.requests_per_hour)
    url = f"http://{args.host}:{server.server_port}/"
    print(f"Savi dashboard: {url}")
    print(f"Database:       {args.db}")
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
