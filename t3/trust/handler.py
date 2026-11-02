"""Lambda entry for ``team13-trust-gate``. Called SYNCHRONOUSLY by ``ran___apply_change``
(InvocationType RequestResponse) with the change record it is about to write, so apply can write
that record once with ``trust_before`` and ``trust_after`` filled. Never called by agents.

Request:  ``{"record": <change record, trust fields empty>}`` (or the record itself)
Response: ``{"trust_before", "trust_after", "e", "score", "action", "control_record_key"?, ...}``
          (see t3/trust/README.md). ``action`` is "pause" only on the call that paused the version.

The gate keeps the policy-write rights: on a pause it creates the Cedar forbid itself (without
waiting for ACTIVE, to answer in about 2 s) and writes its own ``control`` record.

Configuration (environment only; nothing secret lives in code):
    T3_TRUST_STORE      "memory" | "file:<path>" | "ddb:<table>"     (default memory: tests only)
    T3_TRUST_LIVE       "1" creates/deletes real Cedar policies; anything else is a dry run
    T3_TRUST_THRESHOLDS "v1" (default; thresholds.json) or "v2" (thresholds_v2.json), only after the lead approves
    T3_GATEWAY_ARN      the kit Gateway's ARN                         (required when live)
    T3_POLICY_ENGINE_ID the kit policy engine's id                    (required when live)
    EVIDENCE_S3 / EVIDENCE_DIR   where control records go (tools.evlib's own variables)
"""

from __future__ import annotations

import os
import time
from pathlib import PurePosixPath

_GATE = None


class _EvlibWriter:
    """Control records through tools.evlib's validation and store, PutObject only.

    evlib.append_records re-reads an existing object (s3:GetObject) when the key already exists;
    the gate's role has no GetObject. The key is sha256 of the record's canonical line, so an
    existing key means the identical record is already there: treat it as written."""

    def append_records(self, run_id, kind, records):
        from tools import evlib  # W6

        store = evlib.default_store()
        if not isinstance(store, evlib.S3Store):
            return evlib.append_records(run_id, kind, records, store=store)
        out = []
        for line in evlib._prepare(run_id, kind, records):
            rel = f"records/{run_id}/{kind}/{evlib.record_key(line)}.json"
            try:
                out.append(store.put_new(rel, line))
            except FileExistsError:
                out.append(store.uri(rel))
        return out


def _build_gate():
    from .gate import TrustGate
    from .policy import PolicyClient
    from .store import DynamoTrustStore, FileTrustStore, MemoryTrustStore

    spec = os.environ.get("T3_TRUST_STORE", "memory")
    if spec.startswith("file:"):
        store = FileTrustStore(spec[5:])
    elif spec.startswith("ddb:"):
        import boto3
        store = DynamoTrustStore(boto3.resource("dynamodb").Table(spec[4:]))
    else:
        store = MemoryTrustStore()
    live = os.environ.get("T3_TRUST_LIVE") == "1"
    if live:
        import boto3
        pc = PolicyClient(gateway_arn=os.environ["T3_GATEWAY_ARN"], engine_id=os.environ["T3_POLICY_ENGINE_ID"],
                          ac=boto3.client("bedrock-agentcore-control"), dry_run=False, wait_active=False)
    else:
        pc = PolicyClient()
    if os.environ.get("EVIDENCE_S3") or os.environ.get("EVIDENCE_DIR"):
        writer = _EvlibWriter()
    else:
        from .evio import EvidenceWriter
        writer = EvidenceWriter()  # memory: tests and local runs without an evidence target
    from .gate import thresholds_path

    version = os.environ.get("T3_TRUST_THRESHOLDS", "v1")  # v1 until the lead approves a switch
    gate = TrustGate(store, thresholds_path(version), pc, writer, synthetic=not live)
    gate.thresholds_version = version
    return gate


def handle(gate, event) -> dict:
    t0 = time.perf_counter()
    rec = event.get("record", event) if isinstance(event, dict) else None
    if not isinstance(rec, dict) or rec.get("kind") != "change":
        return {"trust_before": None, "trust_after": None, "e": None, "score": None, "action": None,
                "skipped": "event is not a change record"}
    try:
        out = gate.update(rec)
    except Exception as exc:  # e.g. create_policy failed: the trust update is stored, the pause will retry
        out = _from_history(gate, rec)
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    resp = {"trust_before": out.get("trust_before"), "trust_after": out.get("trust_after"),
            "e": out.get("e"), "score": out.get("score"), "action": out.get("action"),
            "direction_agree": out.get("direction_agree"), "status": out.get("status"),
            "band": out.get("band"), "n": out.get("n"),
            "thresholds_sha256": gate.thresholds_sha256,
            "thresholds_version": getattr(gate, "thresholds_version", None)}
    for k in ("skipped", "error"):
        if out.get(k):
            resp[k] = out[k]
    mv = event.get("model_version") if isinstance(event, dict) else None
    pv = (rec.get("predictor") or {}).get("version")
    if mv is not None and pv is not None and str(mv) != str(pv):
        resp["model_version_mismatch"] = {"payload": mv, "record": pv}  # the record's version is scored
    if out.get("action") == "pause":
        ctrl = out["control"]
        paths = gate.last_control_paths
        resp["control_record_key"] = PurePosixPath(paths[0]).stem if paths else None
        resp["policy_name"] = ctrl["policy_name"]
        if gate.last_control_error:
            resp["control_write_error"] = gate.last_control_error
    resp["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return resp


def _from_history(gate, rec) -> dict:
    """Trust values of `rec` from the store after a failure later in update (the pause step)."""
    p = rec.get("predictor") or {}
    try:
        st = gate.state(str(p["model_id"]), str(p["version"]))
        h = [x for x in st["history"] if x.get("change_id") == rec.get("change_id")]
    except Exception:  # noqa: BLE001 - the store itself failed: nothing to report
        return {"action": None}
    if not h:
        return {"action": None}
    return {"trust_before": h[-1].get("trust_before"), "trust_after": h[-1]["trust"], "e": h[-1]["e"],
            "score": h[-1]["score"], "action": None, "status": st["status"], "n": st["n"]}


def handler(event, context=None):
    global _GATE
    if _GATE is None:
        _GATE = _build_gate()
    return handle(_GATE, event)
