"""The only door to the kit: MCP JSON-RPC over HTTPS to the AgentCore Gateway, as the app identity team_app.

Exactly three tools may be called (ALLOWED), plus `tools/list` for the health check. There is no generic
call: anything else raises before a byte is sent. Request shape as checkpoint_kit.py L874-878 and
t3/suite/common.py:159-165; answers are JSON or SSE (last `data:` line), as common.py:200-213.

Failures are classified by WHEN they happened, because the kit has no idempotency:
- `before_send`: DNS, refused connection, TLS, connect timeout. urllib wraps errors raised while the request
  is being written in URLError (urllib.request.AbstractHTTPHandler.do_open); nothing reached AWS.
- `after_send`: anything raised while waiting for or reading the answer. The kit may have acted.
- HTTP 5xx and 429 come from AWS after the request arrived: the kit may have acted.
"""

from __future__ import annotations

import http.client
import json
import re
import socket
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime

from .config import KitConfig, NotConfigured

START, CONTINUE, STATUS = "front___start_case", "front___continue_case", "front___case_status"
ALLOWED = (START, CONTINUE, STATUS)
FRONT_TOOLS = set(ALLOWED)
DENIED_TEXT = "Tool Execution Denied"
RULE = re.compile(r"denied due to ([A-Za-z][A-Za-z0-9_]*?)(?:-[a-z0-9_]{6,})?(?:\]|\s|$|,|\.)")


class TransportError(Exception):
    def __init__(self, phase: str, category: str):
        super().__init__(f"{phase}:{category}")
        self.phase, self.category = phase, category


def _category(reason) -> str:
    if isinstance(reason, socket.gaierror):
        return "dns"
    if isinstance(reason, ConnectionRefusedError):
        return "refused"
    if isinstance(reason, ssl.SSLError):
        return "tls"
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return "connect_timeout"
    return "network"


def http_post(url: str, body: bytes, headers: dict, timeout: float) -> tuple[int, dict, str]:
    """(status, headers, text). HTTP errors are answers. Raises TransportError with no URL in it."""
    req = urllib.request.Request(url, data=body, method="POST", headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)  # noqa: S310 - https Gateway URL from the kit
    except urllib.error.HTTPError as e:
        try:
            text = e.read().decode("utf-8", "replace")
        except OSError:
            text = ""
        return e.code, dict(e.headers.items()) if e.headers else {}, text
    except urllib.error.URLError as e:
        raise TransportError("before_send", _category(e.reason)) from None
    except (TimeoutError, socket.timeout, http.client.HTTPException, OSError) as e:
        raise TransportError("after_send", type(e).__name__) from None
    try:
        with resp:
            return resp.status, dict(resp.headers.items()), resp.read().decode("utf-8", "replace")
    except (TimeoutError, socket.timeout, http.client.HTTPException, OSError) as e:
        raise TransportError("after_send", type(e).__name__) from None


def parse_mcp(raw: str):
    raw = raw or ""
    if raw.lstrip().startswith("{"):
        try:
            return json.loads(raw)
        except ValueError:
            return None
    data = [line[5:].strip() for line in raw.splitlines() if line.startswith("data:")]
    if not data:
        return None
    try:
        return json.loads(data[-1])
    except ValueError:
        return None


def deny_rule(text: str) -> str | None:
    """The Cedar rule named in a deny, without its policy-id suffix; None for a default deny."""
    m = RULE.search(text or "")
    return m.group(1) if m else None


@dataclass
class Outcome:
    """What one Gateway call came back as. `text` is RAW (may hold secrets): redact before showing it.

    kind: ok | tool_error | denied | auth | throttled | server | client_error | rpc_error | is_error |
          unparseable | before_send | after_send
    """
    kind: str
    http_status: int | None = None
    data: dict = field(default_factory=dict)
    text: str = ""
    gateway_date: str | None = None
    category: str | None = None
    latency_ms: float = 0.0

    @property
    def maybe_delivered(self) -> bool:
        """True when the kit may have acted although we did not get a usable answer."""
        return self.kind in ("after_send", "server", "throttled")


def _date(headers: dict) -> str | None:
    v = next((v for k, v in headers.items() if k.lower() == "date"), None)
    if not v:
        return None
    try:
        return parsedate_to_datetime(v).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    except (TypeError, ValueError):
        return None


def classify(status: int, headers: dict, raw: str, listing: bool = False) -> Outcome:
    out = Outcome("ok", http_status=status, gateway_date=_date(headers or {}))
    if status in (401, 403) and DENIED_TEXT not in (raw or ""):
        out.kind, out.text = "auth", raw[:2000]
        return out
    if status == 429:
        out.kind, out.text = "throttled", raw[:2000]
        return out
    if status >= 500:
        out.kind, out.text = "server", raw[:2000]
        return out
    if status >= 400:
        out.kind, out.text = ("denied" if DENIED_TEXT in (raw or "") else "client_error"), raw[:2000]
        return out
    answer = parse_mcp(raw)
    if not isinstance(answer, dict):
        out.kind, out.text = "unparseable", (raw or "")[:500]
        return out
    if "error" in answer:
        err = answer.get("error") or {}
        msg = str(err.get("message", "")) if isinstance(err, dict) else str(err)
        code = err.get("code") if isinstance(err, dict) else None
        out.kind = "denied" if (code == -32002 or DENIED_TEXT in msg) else "rpc_error"
        out.text = msg[:2000]
        return out
    result = answer.get("result") or {}
    if listing:
        out.data = {"tools": sorted(t.get("name", "") for t in result.get("tools", []) if isinstance(t, dict))}
        return out
    text = "".join(c.get("text", "") for c in result.get("content", []) if isinstance(c, dict))
    out.text = text[:20000]
    if result.get("isError"):
        out.kind = "is_error"
        return out
    try:
        data = json.loads(text)
    except ValueError:
        out.kind = "unparseable"
        return out
    if not isinstance(data, dict):
        out.kind = "unparseable"
        return out
    out.data = data
    if "error" in data:
        out.kind, out.text = "tool_error", str(data["error"])[:2000]
    return out


class KitClient:
    def __init__(self, config: KitConfig, post=http_post, clock=time.time):
        self.config, self.post, self.clock = config, post, clock
        self._id = 0

    def _send(self, method: str, params: dict, timeout: float, listing: bool = False) -> Outcome:
        view = self.config.load()
        if not view.configured:
            raise NotConfigured(view.problem or "not configured")
        self._id += 1
        body = json.dumps({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}).encode()
        headers = {"Authorization": f"Bearer {view.token.reveal()}", "Content-Type": "application/json",
                   "Accept": "application/json, text/event-stream"}
        t0 = self.clock()
        try:
            status, rheaders, raw = self.post(view.gateway_url.reveal(), body, headers, timeout)
        except TransportError as e:
            return Outcome(e.phase, category=e.category, latency_ms=round((self.clock() - t0) * 1000, 1))
        out = classify(status, rheaders, raw, listing=listing)
        out.latency_ms = round((self.clock() - t0) * 1000, 1)
        return out

    def call(self, tool: str, arguments: dict, timeout: float) -> Outcome:
        if tool not in FRONT_TOOLS:
            raise ValueError(f"tool {tool!r} is not one of the app's three front-door tools")
        return self._send("tools/call", {"name": tool, "arguments": arguments}, timeout)

    def list_tools(self, timeout: float = 10) -> Outcome:
        return self._send("tools/list", {}, timeout, listing=True)
