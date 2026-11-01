"""Access gate: other devices need the per-run token for the API, the WebSocket and the API docs;
this laptop (a loopback client) needs none, and the built UI stays public so the phone page loads."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.responses import HTMLResponse
from starlette.websockets import WebSocketDisconnect

from spectrum.access import TOKEN, AccessGate, is_loopback
from spectrum.main import app

DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"  # what main.py serves, when built
PHONE = "192.168.1.20"
AUTH = {"X-Spectrum-Token": TOKEN}


def client(host: str = PHONE) -> TestClient:
    return TestClient(app, client=(host, 50000))


@pytest.mark.parametrize("host,loopback", [
    ("127.0.0.1", True), ("127.9.8.7", True), ("::1", True), ("::ffff:127.0.0.1", True),
    (PHONE, False), ("::ffff:192.168.1.20", False), ("fe80::1", False), ("testclient", False), ("", False),
])
def test_loopback_means_this_machine_only(host, loopback):
    assert is_loopback((host, 50000)) is loopback
    assert is_loopback(None) is False


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "::ffff:127.0.0.1"])
def test_this_laptop_needs_no_token_and_is_given_it(host):
    res = client(host).get("/api/config")
    assert res.status_code == 200
    assert res.json()["access_token"] == TOKEN


@pytest.mark.parametrize("method,path", [
    ("GET", "/api/config"), ("GET", "/api/snapshot"), ("POST", "/api/ledger/tamper"),
    ("POST", "/api/director/reset"), ("GET", "/docs"), ("GET", "/redoc"), ("GET", "/openapi.json"),
])
def test_other_devices_without_the_token_get_401(method, path):
    res = client().request(method, path)
    assert res.status_code == 401
    assert "http://localhost:8787" in res.json()["detail"]


def test_a_wrong_token_is_refused():
    assert client().get("/api/config", headers={"X-Spectrum-Token": TOKEN + "x"}).status_code == 401
    assert client().get("/api/config?t=nope").status_code == 401


def test_other_devices_with_the_token_get_in_but_are_never_handed_it():
    res = client().get("/api/config", headers=AUTH)
    assert res.status_code == 200
    assert "access_token" not in res.json() and TOKEN not in res.text
    assert client().get(f"/api/config/flags?t={TOKEN}").status_code == 200
    assert client().post("/api/sim", json={}, headers=AUTH).status_code == 200


def test_websocket_without_the_token_is_closed_with_1008():
    with client().websocket_connect("/ws") as ws, pytest.raises(WebSocketDisconnect) as closed:
        ws.receive_text()
    assert closed.value.code == 1008


def test_websocket_with_the_token_streams():
    with client().websocket_connect(f"/ws?t={TOKEN}") as ws:
        assert ws.receive_json()["kind"] == "hello"


def test_the_ui_files_stay_public():
    # frontend/dist may not be built, so a stub stands in for the static mount and the SPA fallback
    async def ui(scope, receive, send):
        await HTMLResponse('<div id="root"></div>')(scope, receive, send)

    gate = TestClient(AccessGate(ui, token=TOKEN), client=(PHONE, 50000))
    for path in ("/", "/index.html", "/assets/index-3f2a.js", "/favicon.svg", "/m/approve", "/dossier/INC-1"):
        assert gate.get(path).status_code == 200, path
    assert gate.get("/api/snapshot").status_code == 401
    assert gate.post("/m/approve").status_code == 401  # writes are gated wherever they are routed


@pytest.mark.skipif(not DIST.exists(), reason="frontend not built (make build)")
def test_the_built_ui_stays_public():
    for path in ("/", "/m/approve"):
        res = client().get(path)
        assert res.status_code == 200 and '<div id="root">' in res.text, path
