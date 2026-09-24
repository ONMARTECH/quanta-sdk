"""Quanta Cognitive Cockpit — Lightweight Native Python SSE Live Server.

Serves the Cognitive Cockpit UI and streams real-time state changes via
Server-Sent Events (SSE). Completely eliminates browser polling lag,
background tab throttling, and file:// CORS restrictions.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

QUANTA_ROOT = Path(__file__).resolve().parent.parent.parent
DASHBOARD_FILE = QUANTA_ROOT / "quanta" / "cognitive" / "dashboard" / "index.html"
ROOT_STATE_FILE = QUANTA_ROOT / "quanta_cognitive_state.json"


def find_latest_state_file() -> Path:
    """Finds the freshest quanta_cognitive_state.json across workspace and brain artifact dirs."""
    candidates = []
    if ROOT_STATE_FILE.exists():
        candidates.append(ROOT_STATE_FILE)

    brain_base = Path.home() / ".gemini" / "antigravity" / "brain"
    if brain_base.exists():
        for state_path in brain_base.glob("*/quanta_cognitive_state.json"):
            candidates.append(state_path)

    if not candidates:
        return ROOT_STATE_FILE

    return max(candidates, key=lambda p: p.stat().st_mtime if p.exists() else 0.0)


def load_freshest_state() -> tuple[dict[str, Any], float]:
    """Loads freshest state dictionary along with its modification timestamp."""
    state_file = find_latest_state_file()
    if not state_file.exists():
        return {"turn_count": 0, "engrams": [], "mean_zeno_pinning": 0.885}, 0.0

    try:
        mtime = state_file.stat().st_mtime
        with open(state_file, encoding="utf-8", errors="replace") as f:
            data = json.load(f)
            return data, mtime
    except Exception as e:
        logger.debug("Failed reading state file %s: %s", state_file, e)
        return {"turn_count": 0, "engrams": [], "mean_zeno_pinning": 0.885}, 0.0


class CognitiveCockpitRequestHandler(SimpleHTTPRequestHandler):
    """HTTP and SSE request handler for the Quanta Cognitive Cockpit."""

    server_version = "QuantaCognitiveServer/1.0"

    def do_OPTIONS(self) -> None:
        """Handle CORS pre-flight requests."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Last-Event-ID")
        self.end_headers()

    def do_GET(self) -> None:
        """Routes GET requests to HTML dashboard, JSON API, or SSE live stream."""
        path = self.path.split("?")[0]

        if path in ("/", "/index.html", "/dashboard.html"):
            self._serve_dashboard()
        elif path == "/api/state":
            self._serve_state_api()
        elif path == "/events":
            self._serve_sse_stream()
        elif path == "/healthz":
            self._serve_health()
        else:
            self.send_error(404, f"Path '{path}' not found")

    def _serve_health(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b'{"status":"healthy","server":"quanta-cognitive-sse"}\n')

    def _serve_dashboard(self) -> None:
        if not DASHBOARD_FILE.exists():
            self.send_error(404, "Dashboard HTML not found")
            return

        try:
            content = DASHBOARD_FILE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading dashboard: {e}")

    def _serve_state_api(self) -> None:
        state_data, _ = load_freshest_state()
        encoded = json.dumps(state_data, ensure_ascii=False).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(encoded)

    def _serve_sse_stream(self) -> None:
        """Streams state changes to the browser in real time via Server-Sent Events."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        # Send initial connection event
        last_mtime = 0.0
        state_data, mtime = load_freshest_state()
        last_mtime = mtime
        self._send_sse_message(state_data)

        heartbeat_counter = 0
        try:
            while True:
                time.sleep(0.5)
                heartbeat_counter += 1

                # Check if state modified
                curr_state, curr_mtime = load_freshest_state()
                if curr_mtime > last_mtime or curr_state.get("turn_count") != state_data.get("turn_count"):
                    last_mtime = curr_mtime
                    state_data = curr_state
                    self._send_sse_message(curr_state)
                    heartbeat_counter = 0

                # Send comment ping every 15s to keep connection open
                if heartbeat_counter >= 30:
                    heartbeat_counter = 0
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()

        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            logger.debug("SSE stream disconnected: %s", e)

    def _send_sse_message(self, data: dict[str, Any]) -> None:
        payload = json.dumps(data, ensure_ascii=False)
        msg = f"event: state\ndata: {payload}\n\n".encode()
        self.wfile.write(msg)
        self.wfile.flush()

    def log_message(self, format: str, *args: Any) -> None:
        if "/events" in args or "/healthz" in args:
            return
        logger.info("%s - - [%s] %s\n", self.client_address[0], self.log_date_time_string(), format % args)


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def find_free_port(start_port: int = 8765, host: str = "127.0.0.1") -> int:
    port = start_port
    while is_port_in_use(port, host):
        port += 1
        if port > start_port + 50:
            raise RuntimeError("No free port found in range")
    return port


class QuantaCognitiveServer:
    """Manager for the background Cognitive Cockpit Server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = find_free_port(port, host)
        self.httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self, blocking: bool = False) -> str:
        """Starts the SSE HTTP server."""
        self.httpd = ThreadingHTTPServer((self.host, self.port), CognitiveCockpitRequestHandler)
        url = f"http://{self.host}:{self.port}"
        logger.info("Quanta Cognitive Cockpit SSE Server started at %s", url)

        if blocking:
            try:
                self.httpd.serve_forever()
            except KeyboardInterrupt:
                self.stop()
        else:
            self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
            self._thread.start()

        return url

    def stop(self) -> None:
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Quanta Cognitive Cockpit SSE Live Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    server = QuantaCognitiveServer(host=args.host, port=args.port)
    url = server.start(blocking=True)
    print(f"🚀 Quanta Cognitive Cockpit Server running at: {url}")


if __name__ == "__main__":
    main()
