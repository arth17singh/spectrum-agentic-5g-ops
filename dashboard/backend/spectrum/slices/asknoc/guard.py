"""Request guard for every /api/asknoc route. It runs after AccessGate (access.py), which already lets only
loopback clients or token holders reach /api at all.

- Loopback clients only. A phone (token holder) may read /health, which then answers `loopback_only`, so the
  page can say so without a console error; every other route answers 403.
- Host must name this laptop (localhost, 127.0.0.1, ::1; any port): a DNS-rebinding page has its own host name.
- GET: a cross-site fetch is refused.
- POST: Origin must be present and be http on one of those host names (or listed in ASKNOC_ALLOWED_ORIGINS;
  under `make dev` it is http://localhost:5173, vite.config.ts:17). Any port is accepted: what stops a page on
  another local port is the custom header below, which forces a CORS preflight the server never answers
  (main.py:68 has no CORS). Sec-Fetch-Site same-origin or none; Content-Type application/json; a
  Content-Length of at most 8 KiB.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

from fastapi import HTTPException, Request

from spectrum.access import is_loopback

LOCAL = {"localhost", "127.0.0.1", "::1"}
MAX_BODY = 8192
CLIENT_HEADER = "asknoc"


def refuse(status: int, code: str, message: str, headers: dict | None = None) -> HTTPException:
    return HTTPException(status, detail={"error": code, "message": message}, headers=headers)


def hostname(hostport: str) -> str:
    try:
        return (urlsplit("//" + hostport).hostname or "").lower()
    except ValueError:
        return ""


def origin_ok(origin: str) -> bool:
    extra = {o.strip().rstrip("/") for o in (os.environ.get("ASKNOC_ALLOWED_ORIGINS") or "").split(",") if o.strip()}
    if origin.rstrip("/") in extra:
        return True
    try:
        u = urlsplit(origin)
        return u.scheme == "http" and (u.hostname or "").lower() in LOCAL and not u.path.strip("/")
    except ValueError:
        return False


async def guard(request: Request) -> None:
    loopback = is_loopback(request.client)
    request.state.loopback = loopback
    if not loopback:
        if request.method == "GET" and request.url.path.rstrip("/") == "/api/asknoc/health":
            return
        raise refuse(403, "loopback_only", "Ask the NOC works only on the presenter laptop (http://localhost:8787).")
    if hostname(request.headers.get("host", "")) not in LOCAL:
        raise refuse(403, "bad_host", "Open the console as http://localhost:8787.")
    site = (request.headers.get("sec-fetch-site") or "").lower()
    if request.method in ("GET", "HEAD"):
        if site == "cross-site":
            raise refuse(403, "cross_site", "Cross-site requests are refused.")
        return
    origin = request.headers.get("origin")
    if not origin or not origin_ok(origin):
        raise refuse(403, "bad_origin", "Requests must come from this console's own page.")
    if site and site not in ("same-origin", "none"):
        raise refuse(403, "cross_site", "Cross-site requests are refused.")
    if request.headers.get("x-spectrum-client") != CLIENT_HEADER:
        raise refuse(403, "missing_client_header", "Requests must come from the Ask the NOC page.")
    ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    if ctype != "application/json":
        raise refuse(415, "bad_content_type", "Send JSON.")
    length = request.headers.get("content-length")
    if length is None:
        raise refuse(411, "length_required", "Content-Length is required.")
    try:
        n = int(length)
    except ValueError:
        raise refuse(400, "bad_length", "Bad Content-Length.") from None
    if n > MAX_BODY:
        raise refuse(413, "too_large", f"The request is larger than {MAX_BODY} bytes.")
