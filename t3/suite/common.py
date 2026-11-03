"""Shared pieces of the Track 3 suite: the Gateway client, tokens, evidence writing, time and hashing.

No AWS client is created at import time. Everything that touches AWS or the network is passed in, so the
tests run offline with fakes.
"""
from __future__ import annotations

import datetime as dt
import email.utils
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from t3.suite.suite_hash import canonical

TEAM = "team13"
SCHEMA_VERSION = "1"
APPROVAL_RE = re.compile(r"chg-[0-9a-f]{12}")
CASE_RE = re.compile(r"case-[0-9a-f]{8}")
AGENTS = ("noc_lead", "ran_monitor", "ran_optimizer", "ran_explainer", "team_app")
RUNNING = ("TASK_STATE_SUBMITTED", "TASK_STATE_WORKING")


# ---------- hashing and time ----------
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj) -> str:
    return sha256_text(canonical(obj))


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def to_iso(value) -> str | None:
    """AWS times in any of their shapes (epoch s/ms, Logs Insights '2026-11-02 20:19:04.123',
    ISO, HTTP Date) to ISO-8601 UTC with a Z."""
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        d = value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
    elif isinstance(value, (int, float)) or (isinstance(value, str) and re.fullmatch(r"\d+(\.\d+)?", value)):
        v = float(value)
        d = dt.datetime.fromtimestamp(v / 1000 if v > 1e11 else v, dt.timezone.utc)
    else:
        s = str(value).strip()
        try:
            d = dt.datetime.fromisoformat(s.replace("Z", "+00:00").replace(" ", "T", 1))
        except ValueError:
            parsed = email.utils.parsedate_to_datetime(s)  # HTTP Date header
            d = parsed
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def epoch(value) -> float:
    iso = to_iso(value)
    return dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def code_sha(repo: str | Path | None = None) -> str:
    """The commit the code runs from: T3_CODE_SHA (set in Lambda), else git HEAD (bare hex)."""
    if os.environ.get("T3_CODE_SHA"):
        return os.environ["T3_CODE_SHA"]
    root = Path(repo or Path(__file__).resolve().parents[2])
    try:
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "t3"],
                               capture_output=True, text=True).stdout.strip()
        if dirty:  # the contracts want a bare hex sha: warn instead of a suffix; commit before a live run
            import sys
            print(f"WARNING: uncommitted changes under t3/; code_sha {sha[:12]} does not describe them",
                  file=sys.stderr)
        return sha
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ---------- suite placeholders ----------
def fill(value, case_id: str, run_id: str):
    """Replace {case} and {run_id} in every string of an args object."""
    if isinstance(value, str):
        return value.replace("{case}", case_id).replace("{run_id}", run_id)
    if isinstance(value, list):
        return [fill(v, case_id, run_id) for v in value]
    if isinstance(value, dict):
        return {k: fill(v, case_id, run_id) for k, v in value.items()}
    return value


def change_key(args: dict) -> str:
    """The kit's approval id: the same proposed change always gets the same id."""
    fields = [args.get(k) for k in ("case_id", "item_id", "change")]
    return "chg-" + hashlib.sha256(json.dumps(fields).encode()).hexdigest()[:12]


# ---------- tokens ----------
class Tokens:
    """Agent tokens, never stored in files of the repo. Sources, first match wins:
    a JSON secret in Secrets Manager (name in T3_TOKENS_SECRET, read through an injected client),
    environment variables T3_TOKEN_<AGENT> (e.g. T3_TOKEN_NOC_LEAD), or a token folder like the kit's
    tokens/ (<agent>.txt), given as tokens_dir or T3_TOKENS_DIR."""

    def __init__(self, secret_name=None, secrets_client=None, tokens_dir=None, env=None):
        self.env = os.environ if env is None else env
        self.secret_name = secret_name or self.env.get("T3_TOKENS_SECRET")
        self.secrets_client = secrets_client
        self.tokens_dir = tokens_dir or self.env.get("T3_TOKENS_DIR")
        self._secret = None

    def _from_secret(self):
        if self._secret is None and self.secret_name and self.secrets_client is not None:
            raw = self.secrets_client.get_secret_value(SecretId=self.secret_name)["SecretString"]
            self._secret = json.loads(raw)
        return self._secret or {}

    def get(self, agent: str) -> str:
        secret = self._from_secret()
        if agent in secret:
            return secret[agent]
        env_name = "T3_TOKEN_" + agent.upper()
        if self.env.get(env_name):
            return self.env[env_name]
        if self.tokens_dir:
            p = Path(self.tokens_dir) / f"{agent}.txt"
            if p.exists():
                return p.read_text().strip()
        raise KeyError(f"no token for {agent}: set {env_name}, T3_TOKENS_SECRET or T3_TOKENS_DIR")


# ---------- the Gateway (MCP JSON-RPC over HTTPS, standard library only) ----------
def http_post(url, body: bytes, headers: dict, timeout: float):
    """Returns (status, response headers, text). HTTP errors are answers too: Cedar refusals can
    arrive as HTTP errors."""
    req = urllib.request.Request(url, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers.items()), resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers.items()) if exc.headers else {}, exc.read().decode("utf-8", "replace")


class Gateway:
    def __init__(self, url: str, tokens: Tokens, post=http_post, clock=time.time):
        if not url:
            raise ValueError("no Gateway URL: set T3_GATEWAY_URL")
        self.url, self.tokens, self.post, self.clock = url, tokens, post, clock

    def call(self, agent: str, tool: str, arguments: dict, timeout: float = 300) -> dict:
        """One tools/call as `agent`. Returns the outcome the caller saw:
        decision ALLOW unless the Gateway answered with an HTTP error, a JSON-RPC error or isError."""
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                           "params": {"name": tool, "arguments": arguments}}).encode()
        headers = {"Authorization": f"Bearer {self.tokens.get(agent)}", "Content-Type": "application/json",
                   "Accept": "application/json, text/event-stream"}
        t0 = self.clock()
        try:
            status, rheaders, raw = self.post(self.url, body, headers, timeout)
        except Exception as exc:  # noqa: BLE001 - a network failure is an outcome, recorded as such
            return {"decision": "ERROR", "text": str(exc)[:2000], "http_status": None, "gateway_date": None,
                    "sent_at": t0, "latency_ms": round((self.clock() - t0) * 1000, 1), "transport_error": True}
        latency = round((self.clock() - t0) * 1000, 1)
        date = next((v for k, v in rheaders.items() if k.lower() == "date"), None)
        out = {"http_status": status, "gateway_date": to_iso(date) if date else None, "sent_at": t0,
               "latency_ms": latency}
        answer = parse_mcp(raw)
        if status >= 400 or answer is None or "error" in answer:
            err = (answer or {}).get("error") if answer else None
            out.update(decision="DENY", text=json.dumps(err)[:2000] if err else raw[:2000])
            return out
        result = answer.get("result") or {}
        text = "".join(c.get("text", "") for c in result.get("content", []) if isinstance(c, dict))
        out.update(decision="DENY" if result.get("isError") else "ALLOW", text=text[:20000])
        if out["decision"] == "ALLOW":
            try:
                data = json.loads(text)
                if isinstance(data, dict) and "error" in data:
                    out["tool_error"] = str(data["error"])[:500]
            except ValueError:
                pass
        return out

    def call_json(self, agent, tool, arguments, timeout=300) -> dict:
        out = self.call(agent, tool, arguments, timeout)
        if out["decision"] != "ALLOW":
            raise RuntimeError(f"{tool} refused for {agent}: {out['text'][:400]}")
        return json.loads(out["text"])


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


# ---------- evidence (W6's tools/evlib.py, injected so tests can use a fake) ----------
def load_evlib():
    import importlib
    return importlib.import_module("tools.evlib")


def envelope(kind: str, run_id: str, aws_time=None, aws_time_source=None, synthetic=False, **fields) -> dict:
    return {"schema_version": SCHEMA_VERSION, "kind": kind, "run_id": run_id, "team": TEAM,
            "created_at": utc_now_iso(), "aws_time": to_iso(aws_time) if aws_time else None,
            "aws_time_source": aws_time_source if aws_time else None, "synthetic": bool(synthetic), **fields}


# ---------- a small DynamoDB helper (low-level client shape) ----------
def ddb_get(ddb, table: str, key: dict) -> dict | None:
    item = ddb.get_item(TableName=table, Key={k: {"S": v} for k, v in key.items()}, ConsistentRead=True).get("Item")
    if not item:
        return None
    return {k: next(iter(v.values())) for k, v in item.items()}


def ddb_put(ddb, table: str, item: dict, condition: str | None = None) -> None:
    kw = {"TableName": table, "Item": {k: ({"N": str(v)} if isinstance(v, (int, float)) and not isinstance(v, bool)
                                           else {"S": str(v)}) for k, v in item.items()}}
    if condition:
        kw["ConditionExpression"] = condition
    ddb.put_item(**kw)
