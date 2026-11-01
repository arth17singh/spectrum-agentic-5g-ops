"""Access gate for devices other than this laptop.

The console on this laptop is a loopback client and needs nothing. Every other device (the
presenter's phone on the venue Wi-Fi under `make demo`) must present the per-run access token for
every /api request (reads too), every WebSocket and the API docs: in the X-Spectrum-Token header, or
as ?t= (browsers cannot set headers on a WebSocket). The built UI (static files, index.html, SPA
routes) holds no data and stays public, so the phone page can load and pick ?t= out of its URL.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import secrets
from urllib.parse import parse_qs

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

# New on every server start unless SPECTRUM_TOKEN pins it (so a paired phone survives a restart).
TOKEN = os.environ.get("SPECTRUM_TOKEN") or secrets.token_urlsafe(16)

HELP = "Open the console at http://localhost:8787, or use the phone link from the Demo Director."

# Data and controls; a route added under one of these later is gated without further work.
GATED_PREFIXES = ("/api", "/ws", "/docs", "/redoc", "/openapi.json")


def is_loopback(client: tuple[str, int] | None) -> bool:
    """True for 127.0.0.0/8, ::1 and ::ffff:127.x.x.x. `client` is the ASGI peer address: uvicorn
    rewrites it from X-Forwarded-For only for proxies in --forwarded-allow-ips, so never parse that."""
    if not client:
        return False
    try:
        ip = ipaddress.ip_address(client[0])
    except ValueError:  # not an address, e.g. Starlette's TestClient reports "testclient"
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:  # dual-stack socket
        ip = ip.ipv4_mapped
    return ip.is_loopback


def _gated(scope: Scope) -> bool:
    if scope["type"] == "websocket" or scope["method"] not in ("GET", "HEAD"):
        return True  # every socket, and every write wherever it is routed: the UI files are read-only
    return scope["path"].startswith(GATED_PREFIXES)


class AccessGate:
    """Pure ASGI middleware over both http and websocket scopes, so routes added later are covered."""

    def __init__(self, app: ASGIApp, token: str = TOKEN) -> None:
        self.app = app
        self.token = token.encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (scope["type"] in ("http", "websocket") and _gated(scope)
                and not is_loopback(scope.get("client")) and not self._has_token(scope)):
            await _refuse(scope, receive, send)
            return
        await self.app(scope, receive, send)

    def _has_token(self, scope: Scope) -> bool:
        supplied = dict(scope["headers"]).get(b"x-spectrum-token")
        if supplied is None:
            supplied = parse_qs(scope["query_string"].decode("latin-1")).get("t", [""])[0].encode()
        return secrets.compare_digest(supplied, self.token)


async def _refuse(scope: Scope, receive: Receive, send: Send) -> None:
    if scope["type"] == "http":
        await JSONResponse({"detail": f"Access token required. {HELP}"}, status_code=401)(scope, receive, send)
        return
    # Accept, then close with 1008. A close before accept becomes an HTTP 403 handshake failure,
    # which browsers report as 1006, the same as a dropped network; 1008 lets the page say why.
    await receive()  # websocket.connect
    await send({"type": "websocket.accept"})
    await send({"type": "websocket.close", "code": 1008, "reason": "access token required"})


def log_phone_link(lan_ip: str) -> None:
    logging.getLogger("uvicorn.error").info(
        "Other devices need the access token. Phone link under make demo: http://%s:8787/m/approve?t=%s "
        "(new on every restart; set SPECTRUM_TOKEN to pin it)", lan_ip, TOKEN)
