"""Lambda entry for ``team13-trust-gate`` (amendment 1, A7). Invoked asynchronously by
``ran___apply_change`` with the `change` record it just wrote; never called by agents.

Event: ``{"record": <change record>}`` or the change record itself.
Returns the gate's decision (``trust_before``, ``trust_after``, ``action``, ...).

Configuration (environment only; nothing secret lives in code):
    T3_TRUST_STORE      "memory" | "file:<path>" | "ddb:<table>"   (default memory: tests only)
    T3_TRUST_LIVE       "1" to create/delete real Cedar policies; anything else is a dry run
    T3_GATEWAY_ARN      the kit Gateway's ARN            (required when live)
    T3_POLICY_ENGINE_ID the kit policy engine's id        (required when live)
    T3_RECORDS          "local:<dir>" for a local writer; otherwise ``tools.evlib.append_records``
    AWS_REGION          set by Lambda

The role this runs under may only create/delete/list/get policies on our policy engine and
read/write trust rows (see ``iam_trust_gate_policy.json``). Tool Lambdas never get those rights.
"""

from __future__ import annotations

import os

_GATE = None


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
                          ac=boto3.client("bedrock-agentcore-control"), dry_run=False)
    else:
        pc = PolicyClient()

    rec_spec = os.environ.get("T3_RECORDS", "")
    if rec_spec.startswith("local:"):
        from .local_writer import LocalRecordWriter
        writer = LocalRecordWriter(rec_spec[6:])
    else:
        from tools import evlib  # W6
        writer = evlib
    return TrustGate(store, None, pc, writer, synthetic=not live)


def handler(event, context=None):
    global _GATE
    if _GATE is None:
        _GATE = _build_gate()
    rec = event.get("record", event) if isinstance(event, dict) else None
    if not isinstance(rec, dict) or rec.get("kind") != "change":
        return {"action": None, "skipped": "event is not a change record"}
    out = _GATE.update(rec)
    out.pop("control", None)
    return out
