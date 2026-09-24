import json
import time
import urllib.request

from quanta.cognitive.server import (
    QuantaCognitiveServer,
    find_latest_state_file,
    load_freshest_state,
)


def test_find_and_load_freshest_state():
    state_file = find_latest_state_file()
    assert state_file.exists()
    data, mtime = load_freshest_state()
    assert isinstance(data, dict)
    assert "turn_count" in data
    assert mtime > 0

def test_cognitive_server_lifecycle():
    server = QuantaCognitiveServer(port=8765)
    url = server.start(blocking=False)
    assert url.startswith("http://")
    time.sleep(0.5)

    try:
        # 1. Health check
        with urllib.request.urlopen(f"{url}/healthz", timeout=3) as resp:
            assert resp.status == 200
            body = json.loads(resp.read().decode())
            assert body["status"] == "healthy"

        # 2. State API
        with urllib.request.urlopen(f"{url}/api/state", timeout=3) as resp:
            assert resp.status == 200
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"
            state = json.loads(resp.read().decode())
            assert "turn_count" in state

        # 3. HTML Dashboard
        with urllib.request.urlopen(f"{url}/", timeout=3) as resp:
            assert resp.status == 200
            html = resp.read().decode()
            assert "Quanta Bilişsel Hakem & Telemetri Kokpiti" in html
            assert "setupSSE" in html
    finally:
        server.stop()
