"""Fakes for the Ask the NOC tests: a fake kit folder with MARKER secrets, a fake Gateway transport, fake AWS reads.

Answer shapes are the kit's own (checkpoint_kit.py L241-262 for the tool results, wrapped as MCP
tools/call results as the Gateway returns them; deny text from team13-kit/check-full-2.log). The completed
status and reply come from the one approved live smoke run (case-d59bd7e0, 2 Nov 2026).
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

from spectrum.slices.asknoc.aws import AwsUnavailable
from spectrum.slices.asknoc.config import KitConfig
from spectrum.slices.asknoc.kit import KitClient, TransportError
from spectrum.slices.asknoc.service import AskNoc, Timing
from spectrum.slices.asknoc.store import Store

MARK_URL = "https://gwmarker-abc.gateway.bedrock-agentcore.us-east-2.amazonaws.com/mcp"
MARK_ISSUER = "https://issuermarker.lambda-url.us-east-2.on.aws"
MARK_ACCOUNT = "".join(["1111", "2222", "3333"])  # assembled: no 12-digit literal in git
MARK_BUILDER = "bldr-marker-" + "-".join(["203", "0", "113", "5"]) + ".example.test"  # documentation range
MARK_A2A = "A2A-MARKER-SECRET-0000"
DATE = "Tue, 03 Nov 2026 05:01:23 GMT"
SMOKE_CASE = "case-d59bd7e0"
SMOKE_REPLY = ('ran_monitor reports cell 3 as the worst served, with low cell-edge SINR (verdict "degraded"; '
               "coverage 0.585, SLA 0.935, SINR p10 0.42 dB, mean 7.93 dB). Nothing was proposed or changed.")
CHG = "chg-bde360f4d4b4"
CHANGE = "cell=2;d_power_db=-1.0;d_tilt_deg=+1.0"


def jwt(claims: dict) -> str:
    enc = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")  # noqa: E731
    return f"{enc({'alg': 'RS256', 'kid': 'team13-1'})}.{enc(claims)}.c2lnbmF0dXJlLW1hcmtlcg"


TOKEN_CLAIMS = {"iss": MARK_ISSUER, "aud": "team13-gateway", "sub": "app-team_app", "agent": "team_app",
                "swarm": "team13", "role": "app"}


def kit_dir(tmp: Path, exp_offset: float = 86400 * 10, url: str = MARK_URL) -> Path:
    d = tmp / "kit"
    (d / "tokens").mkdir(parents=True, exist_ok=True)
    (d / "kit-state.json").write_text(json.dumps({
        "gateway_url": url, "issuer": MARK_ISSUER, "gateway_id": "team13-checkpoint-x",
        "gateway_arn": f"arn:aws:bedrock-agentcore:us-east-2:{MARK_ACCOUNT}:gateway/team13-checkpoint-x"}))
    (d / "kit.json").write_text(json.dumps({"name": "team13", "builder_host": MARK_BUILDER, "a2a_token": MARK_A2A}))
    (d / "tokens" / "team_app.txt").write_text(jwt({**TOKEN_CLAIMS, "iat": int(time.time()),
                                                    "exp": int(time.time() + exp_offset)}))
    return d


def mcp_ok(data: dict, sse: bool = False) -> tuple[int, dict, str]:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"content": [
        {"type": "text", "text": json.dumps(data, separators=(",", ":"))}]}})
    return 200, {"Date": DATE}, (f"event: message\ndata: {body}\n\n" if sse else body)


def mcp_deny(message: str = "Tool Execution Denied: Tool call not allowed due to policy enforcement "
             "[No policy applies to the request (denied by default).]") -> tuple[int, dict, str]:
    return 200, {"Date": DATE}, json.dumps({"jsonrpc": "2.0", "id": 1, "error": {"code": -32002, "message": message}})


def started(case_id: str = SMOKE_CASE, state: str = "TASK_STATE_SUBMITTED"):
    return mcp_ok({"case_id": case_id, "state": state})


def status_running(case_id: str = SMOKE_CASE, waiting=None):
    return mcp_ok({"case_id": case_id, "state": "TASK_STATE_WORKING", "waiting_approvals": waiting or []})


def status_done(case_id: str = SMOKE_CASE, reply: str = SMOKE_REPLY, state: str = "TASK_STATE_COMPLETED", waiting=None):
    return mcp_ok({"case_id": case_id, "state": state, "waiting_approvals": waiting or [], "reply": reply})


def waiting_row(chg: str = CHG, change: str = CHANGE):
    return [{"approval_id": chg, "change": json.dumps({"item_id": "step-0", "change": change,
                                                       "reason": "edge SINR low"})}]


class FakePost:
    """Answers by tool name (or 'tools/list'); each entry is a (status, headers, text) or an exception."""

    def __init__(self, script: dict | None = None):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.calls: list[tuple[str, dict, dict, float]] = []

    def __call__(self, url, body, headers, timeout):
        req = json.loads(body)
        name = req["params"].get("name") if req["method"] == "tools/call" else req["method"]
        self.calls.append((name, req, headers, timeout))
        queue = self.script.get(name) or []
        item = queue.pop(0) if len(queue) > 1 else (queue[0] if queue else (500, {}, "no script"))
        if isinstance(item, Exception):
            raise item
        return item

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


class FakeAws:
    def __init__(self, events=None, rows=None, down: str | None = None):
        self.events = events or []
        self.rows = rows or {}
        self.down = down
        self.filters: list[str] = []

    def _check(self):
        if self.down:
            raise AwsUnavailable(self.down)

    def filter_events(self, term, start_ms, limit=1000):
        self._check()
        self.filters.append(term)
        return [e for e in self.events if term in (e.get("message") or "")]

    def get_row(self, pk):
        self._check()
        return self.rows.get(pk)

    def probe(self):
        self._check()


class Clock:
    def __init__(self, t: float | None = None):
        self.t = time.time() if t is None else t

    def __call__(self) -> float:
        return self.t


FAST = dict(poll_delays=(0.005,), error_delays=(0.005,), reconcile_every_s=0.005)


def service(tmp: Path, post: FakePost | None = None, aws: FakeAws | None = None, *, clock=None, offline=False,
            timing: Timing | None = None, store: Store | None = None, env: dict | None = None,
            exp_offset: float = 86400 * 10, url: str = MARK_URL) -> AskNoc:
    cfg = KitConfig(str(kit_dir(tmp, exp_offset, url)), env={})
    clk = clock or time.time
    return AskNoc(cfg, KitClient(cfg, post or FakePost(), clock=clk), store or Store(memory=True), aws or FakeAws(),
                  offline=offline, timing=timing or Timing(), clock=clk, env=env if env is not None else {})


def log_event(msg: dict, ts_ms: int = 1_800_000_000_000) -> dict:
    return {"timestamp": ts_ms, "eventId": str(ts_ms), "message": json.dumps({"ts": ts_ms / 1000, **msg})}


def transport(phase: str, category: str = "x") -> TransportError:
    return TransportError(phase, category)
