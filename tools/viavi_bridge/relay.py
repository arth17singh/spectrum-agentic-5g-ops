#!/usr/bin/env python3
"""MCP relay: local streamable-HTTP /mcp -> same-origin fetch("/mcp") in the signed-in Viavi tab.

Usage: python -I tools/viavi_bridge/relay.py --port 8790 --profile-dir <chrome profile dir>

The chrome-devtools-axi session (`viavi`, headed) must already be signed in to the Viavi app, with the
app tab (path /sba/ui/) selected. Gateway errors are retried per the rules in README.md.

Every POST/DELETE to http://127.0.0.1:<port>/mcp is replayed, one at a time, as a fetch inside
the browser tab whose path starts with /sba/ui/ (driven with the chrome-devtools-axi CLI, session
`viavi`). Status, Content-Type, Mcp-Session-Id and the full body come back. GET returns 405.
Never logs the RSG host, cookies, session ids or bodies: only method, status, sizes, timings.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

AXI = os.environ.get("CHROME_DEVTOOLS_AXI_BIN", "/opt/homebrew/bin/chrome-devtools-axi")
# The profile directory is set by --profile-dir (or CHROME_DEVTOOLS_AXI_USER_DATA_DIR) in main().
AXI_ENV = {
    "CHROME_DEVTOOLS_AXI_SESSION": "viavi",
    "CHROME_DEVTOOLS_AXI_HEADED": "1",
}
FORWARD = ("Content-Type", "Accept", "MCP-Protocol-Version", "Mcp-Session-Id", "Last-Event-ID")
CHUNK = 200_000  # base64 chars per eval read (eval --full handles 300k)
LOCK = threading.Lock()


class RelayError(Exception):
    pass


def axi_eval(js: str, timeout: float = 900.0) -> str:
    """Run JS in the selected tab; return its result unwrapped to a plain string."""
    env = {**os.environ, **AXI_ENV}
    p = subprocess.run(
        [AXI, "eval", "--full", js], capture_output=True, text=True, timeout=timeout, env=env
    )
    out = p.stdout
    line = next((l for l in out.splitlines() if l.startswith("result:")), None)
    if p.returncode != 0 or line is None:
        err = next((l for l in out.splitlines() if l.startswith("error:")), out[:200])
        raise RelayError(f"browser eval failed: {err[:300]}")
    v = line[len("result:"):].strip()
    while isinstance(v, str) and len(v) >= 2 and v[0] == '"' and v[-1] == '"':
        v = json.loads(v)
    return v


def check_tab() -> str:
    return axi_eval("JSON.stringify(location.pathname)")


def browser_fetch(method: str, headers: dict[str, str], body: bytes) -> tuple[int, dict, bytes]:
    req = base64.b64encode(
        json.dumps(
            {"method": method, "headers": headers, "body": base64.b64encode(body).decode()}
        ).encode()
    ).decode()
    js = (
        "(async () => {"
        f"const q = JSON.parse(atob('{req}'));"
        "if (!location.pathname.startsWith('/sba/ui')) return btoa(JSON.stringify({err: 'tab not on /sba/ui'}));"
        "const bin = atob(q.body); const u8 = new Uint8Array(bin.length);"
        "for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);"
        "const init = {method: q.method, headers: q.headers, redirect: 'manual', credentials: 'same-origin'};"
        "if (q.method === 'POST') init.body = u8;"
        "let r; try { r = await fetch('/mcp', init); } catch (e) { return btoa(JSON.stringify({err: 'fetch: ' + e})); }"
        "if (r.type === 'opaqueredirect' || (r.status >= 300 && r.status < 400)) return btoa(JSON.stringify({err: 'redirect (sign-in expired?)', status: r.status}));"
        "const b = new Uint8Array(await r.arrayBuffer()); let s = '';"
        "for (let i = 0; i < b.length; i += 32768) s += String.fromCharCode.apply(null, b.subarray(i, i + 32768));"
        "const b64 = btoa(s); const small = b64.length <= 150000;"
        "window.__relayBody = small ? null : b64;"
        "return btoa(JSON.stringify({status: r.status, ct: r.headers.get('content-type'),"
        " sid: r.headers.get('mcp-session-id'), n: b64.length, inline: small ? b64 : null}));"
        "})()"
    )
    meta = json.loads(base64.b64decode(axi_eval(js)))
    if "err" in meta:
        raise RelayError(meta["err"])
    n = int(meta["n"])
    if meta.get("inline") is not None:
        return meta["status"], {"ct": meta.get("ct"), "sid": meta.get("sid")}, base64.b64decode(meta["inline"])
    parts = []
    for i in range(0, n, CHUNK):
        parts.append(axi_eval(f"window.__relayBody.slice({i}, {i + CHUNK})"))
    b64 = "".join(parts)
    if len(b64) != n:
        raise RelayError(f"short read {len(b64)} of {n}")
    axi_eval("(window.__relayBody = null, 'ok')")
    return meta["status"], {"ct": meta.get("ct"), "sid": meta.get("sid")}, base64.b64decode(b64)


SAFE_RPC = {"initialize", "notifications/initialized", "tools/list", "ping"}
SAFE_TOOLS = {f"tools/call:{t}" for t in (
    "reference", "get_schema", "get_available_actions", "list_use_cases", "list_kpis", "scenario_list",
    "scenario_get", "simulation_list", "simulation_status", "simulation_query")}
GATEWAY = (502, 503, 504)
IDEMPOTENT_TOOLS = {"tools/call:simulation_command", "tools/call:simulation_stop"}


def parse_tool_text(out: bytes, ct: str | None):
    """JSON in the text content of a tools/call reply (JSON or SSE); None if absent."""
    t = out.decode("utf-8", "replace")
    if ct and "event-stream" in ct:
        ds = [l[5:].strip() for l in t.splitlines() if l.startswith("data:")]
        t = ds[-1] if ds else "{}"
    try:
        m = json.loads(t)
        res = m.get("result") or {}
        if res.get("isError"):
            return None
        text = "\n".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
        return json.loads(text)
    except (ValueError, AttributeError):
        return None


def synth_result(rid, payload: dict) -> bytes:
    return json.dumps({"jsonrpc": "2.0", "id": rid, "result": {
        "content": [{"type": "text", "text": json.dumps(payload)}], "isError": False}}).encode()


_HOSTISH = re.compile(r"(https?://\S+|\b(?:\d{1,3}\.){3}\d{1,3}\b|\b[\w-]+(?:\.[\w-]+)+\.(?:com|net|org|io|aws|internal)\b)")


def rpc_label(body: bytes) -> str:
    """'method' or 'tools/call:<tool>' of a JSON-RPC body (no arguments: they may carry ids)."""
    try:
        m = json.loads(body)
    except ValueError:
        return "?"
    if not isinstance(m, dict):
        return "batch"
    meth = str(m.get("method", "response"))
    if meth == "tools/call":
        return f"tools/call:{(m.get('params') or {}).get('name')}"
    return meth


def html_summary(out: bytes) -> str:
    """Masked <title>/<h1>/first text of an error page, for the log."""
    t = out[:4000].decode("utf-8", "replace")
    parts = []
    for tag in ("title", "h1", "center"):
        mm = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", t, re.S | re.I)
        if mm:
            inner = re.sub(r"<[^>]+>|\s+", " ", mm.group(1)).strip()[:120]
            parts.append(f"{tag}={inner}")
    if not parts:
        parts.append("text=" + re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", t)).strip()[:200])
    return " | " + _HOSTISH.sub("<masked>", "; ".join(parts))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # no URLs/headers/bodies in logs
        pass

    def _reply(self, status: int, body: bytes, ct: str | None = None, sid: str | None = None):
        self.send_response(status)
        if ct:
            self.send_header("Content-Type", ct)
        if sid:
            self.send_header("Mcp-Session-Id", sid)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _fetch_logged(self, method, rpc, headers, body, t0, attempt):
        with LOCK:
            status, h, out = browser_fetch(method, headers, body)
        sys.stderr.write(
            f"relay {method} {rpc} -> {status} {h['ct']} in={len(body)}B out={len(out)}B "
            f"{time.monotonic() - t0:.2f}s{'' if attempt == 1 else f' (attempt {attempt})'}"
            f"{html_summary(out) if status >= 400 else ''}\n"
        )
        sys.stderr.flush()
        return status, h, out

    def _side_tool(self, headers, name, args):
        """A read-only tools/call the relay makes itself (same session headers); parsed JSON or None."""
        body = json.dumps({"jsonrpc": "2.0", "id": f"relay-{time.monotonic_ns()}", "method": "tools/call",
                           "params": {"name": name, "arguments": args}}).encode()
        for attempt in range(1, 5):
            try:
                status, h, out = self._fetch_logged("POST", f"side:{name}", headers, body, time.monotonic(), attempt)
            except Exception:  # noqa: BLE001
                return None
            if status == 200:
                return parse_tool_text(out, h["ct"])
            if status not in GATEWAY:
                return None
            time.sleep(1.5 * attempt)
        return None

    def _relay(self, method: str):
        if self.path.split("?")[0] != "/mcp":
            return self._reply(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        headers = {h: self.headers[h] for h in FORWARD if self.headers.get(h)}
        t0 = time.monotonic()
        rpc = rpc_label(body) if method == "POST" else "-"
        msg = json.loads(body) if rpc not in ("?", "-", "batch") else {}
        args = ((msg.get("params") or {}).get("arguments") or {}) if isinstance(msg, dict) else {}
        # Gateway errors (CloudFront 502 "The request could not be satisfied", seen on ~3% of calls):
        # - requests that change nothing, and idempotent ones (simulation_command sets ABSOLUTE power/tilt;
        #   simulation_stop), are retried;
        # - simulation_advance / simulation_start are NOT blindly retried: the relay first checks with
        #   simulation_status / simulation_list whether the call reached the RSG; if it did, it answers with
        #   a synthesized success (marked relay_note); only if it did not is the call sent again.
        idem = rpc.split(":")[0] in SAFE_RPC or rpc in SAFE_TOOLS or rpc in IDEMPOTENT_TOOLS
        before = None
        if rpc == "tools/call:simulation_advance":
            before = self._side_tool(headers, "simulation_status", {"test_id": args.get("test_id")})
        elif rpc == "tools/call:simulation_start":
            before = self._side_tool(headers, "simulation_list", {"limit": 50})
        attempt = 0
        while True:
            attempt += 1
            try:
                status, h, out = self._fetch_logged(method, rpc, headers, body, t0, attempt)
            except Exception as exc:  # noqa: BLE001
                m = str(exc)[:300]
                sys.stderr.write(f"relay {method} {rpc} error after {time.monotonic() - t0:.1f}s: {m}\n")
                sys.stderr.flush()
                return self._reply(502, json.dumps({"relay_error": m}).encode(), "application/json")
            if status not in GATEWAY or attempt >= 4:
                break
            if idem:
                time.sleep(1.5 * attempt)
                continue
            if before is None or rpc not in ("tools/call:simulation_advance", "tools/call:simulation_start"):
                break  # unknown state-changing call: return the error as is
            time.sleep(3.0)
            landed = self._landed(rpc, headers, args, before)
            if landed is None:
                break  # could not tell: do not guess
            if landed is not False:
                note = f"relay: gateway {status} on {rpc.split(':')[1]}, but the RSG shows it took effect"
                sys.stderr.write(f"relay {rpc} confirmed after gateway {status}: synthesized success\n")
                sys.stderr.flush()
                return self._reply(200, synth_result(msg.get("id"), {**landed, "relay_note": note}),
                                   "application/json", h["sid"])
            # did not land: send it again
        self._reply(status, out, h["ct"], h["sid"])

    def _landed(self, rpc, headers, args, before):
        """dict (took effect, with the reply fields the adapter reads), False (did not), None (unknown)."""
        if rpc == "tools/call:simulation_advance":
            now = self._side_tool(headers, "simulation_status", {"test_id": args.get("test_id")})
            if not isinstance(now, dict) or not isinstance(before, dict):
                return None
            moved = float(now.get("sim_time") or 0) > float(before.get("sim_time") or 0)
            running = "PAUSED" not in str(now.get("status")) and "STOPPED" not in str(now.get("status"))
            return {"test_id": args.get("test_id"), "accepted": True} if (moved or running) else False
        now = self._side_tool(headers, "simulation_list", {"limit": 50})
        if not isinstance(now, dict) or not isinstance(before, dict):
            return None
        old = {x.get("test_id") for x in before.get("simulations", [])}
        new = [x for x in now.get("simulations", [])
               if x.get("test_id") not in old and "STOPPED" not in str(x.get("status"))]
        return {"test_id": new[0]["test_id"]} if len(new) == 1 else (False if not new else None)

    def do_POST(self):
        self._relay("POST")

    def do_DELETE(self):
        self._relay("DELETE")

    def do_GET(self):
        if self.path == "/health":
            return self._reply(200, b'{"ok":true}', "application/json")
        self._reply(405, b"", None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--profile-dir", default=os.environ.get("CHROME_DEVTOOLS_AXI_USER_DATA_DIR"),
                    help="the signed-in Chrome profile of the chrome-devtools-axi `viavi` session")
    a = ap.parse_args()
    if not a.profile_dir:
        sys.exit("--profile-dir (or CHROME_DEVTOOLS_AXI_USER_DATA_DIR) is required")
    AXI_ENV["CHROME_DEVTOOLS_AXI_USER_DATA_DIR"] = a.profile_dir
    if a.port in (8787, 8798) or not 8790 <= a.port <= 8797:
        sys.exit("port must be 8790-8797 (not 8798)")
    path = check_tab()
    if not path.startswith("/sba/ui"):
        sys.exit("selected tab is not the Viavi app (/sba/ui/); selectpage it first")
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    sys.stderr.write(f"relay listening on 127.0.0.1:{a.port}/mcp\n")
    sys.stderr.flush()
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
