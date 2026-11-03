"""Trust state per (model_id, version). In-memory and local-file stores for tests and demos,
and a DynamoDB store for production (the table is injected; nothing here builds a client).

State fields: ``m`` (raw EWMA accumulator), ``n`` (changes scored this epoch), ``trust``
(bias-corrected value, None before the first change), ``status`` (active|paused), ``epoch``
(bumped on every person-made resume), ``policy_name``, ``history`` (trust after each change,
capped).

``claim_pause`` is the one atomic step: it flips active -> paused only if nobody else did, so two
concurrent tool invocations can never both create a forbid or write two control records.
"""

from __future__ import annotations

import copy
import json
import os
import threading
from decimal import Decimal
from pathlib import Path
from typing import Any

HISTORY_CAP = 200


def new_state() -> dict[str, Any]:
    return {"m": 0.0, "n": 0, "trust": None, "recent_e": [], "status": "active", "epoch": 0,
            "policy_name": None, "history": []}


def key(model_id: str, version: str) -> str:
    return f"{model_id}@{version}"


class MemoryTrustStore:
    def __init__(self):
        self._d: dict[str, dict] = {}
        self._lock = threading.Lock()

    def get(self, model_id: str, version: str) -> dict:
        with self._lock:
            return copy.deepcopy(self._d.get(key(model_id, version)) or new_state())

    def put(self, model_id: str, version: str, state: dict) -> None:
        with self._lock:
            st = copy.deepcopy(state)
            st["history"] = st["history"][-HISTORY_CAP:]
            # status only changes through claim_pause / release_pause
            st["status"] = (self._d.get(key(model_id, version)) or new_state())["status"]
            self._d[key(model_id, version)] = st

    def claim_pause(self, model_id: str, version: str) -> bool:
        with self._lock:
            st = self._d.setdefault(key(model_id, version), new_state())
            if st["status"] == "paused":
                return False
            st["status"] = "paused"
            return True

    def release_pause(self, model_id: str, version: str) -> bool:
        with self._lock:
            st = self._d.setdefault(key(model_id, version), new_state())
            if st["status"] != "paused":
                return False
            st["status"] = "active"
            return True

    def all(self) -> dict[str, dict]:
        with self._lock:
            return copy.deepcopy(self._d)


class FileTrustStore(MemoryTrustStore):
    """MemoryTrustStore persisted to one JSON file after every write (single process only)."""

    def __init__(self, path: str | os.PathLike):
        super().__init__()
        self.path = Path(path)
        if self.path.exists():
            self._d = json.loads(self.path.read_text())

    def _save(self):
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._d, indent=1, sort_keys=True))
        tmp.replace(self.path)

    def put(self, model_id, version, state):
        super().put(model_id, version, state)
        with self._lock:
            self._save()

    def claim_pause(self, model_id, version):
        ok = super().claim_pause(model_id, version)
        with self._lock:
            self._save()
        return ok

    def release_pause(self, model_id, version):
        ok = super().release_pause(model_id, version)
        with self._lock:
            self._save()
        return ok


def _to_ddb(x):
    if isinstance(x, float):
        return Decimal(repr(x))
    if isinstance(x, dict):
        return {k: _to_ddb(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_ddb(v) for v in x]
    return x


def _from_ddb(x):
    if isinstance(x, Decimal):
        return int(x) if x == x.to_integral_value() and "." not in str(x) else float(x)
    if isinstance(x, dict):
        return {k: _from_ddb(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_from_ddb(v) for v in x]
    return x


class DynamoTrustStore:
    """Production store: one item per model version, ``pk`` = "t3-trust#<model>@<version>".

    Built to live in the kit's approvals table next to the case rows (A10). The pause flag is
    stored as ``gate_status``, never ``status``: the kit's ``approve`` lists rows that have a
    ``status`` attribute, and trust rows must never show up there. ``table`` is a boto3
    ``Table`` resource injected by the caller; ``claim_pause`` is a conditional update so only
    one invocation wins. (Exercised with a fake table in tests; never against AWS by W5.)
    """

    PK_PREFIX = "t3-trust#"

    def __init__(self, table):
        self.table = table

    def _pk(self, model_id, version):
        return {"pk": f"{self.PK_PREFIX}{key(model_id, version)}"}

    def get(self, model_id, version):
        item = self.table.get_item(Key=self._pk(model_id, version), ConsistentRead=True).get("Item")
        st = new_state()
        if not item:
            return st
        item = _from_ddb(item)
        st.update({k: v for k, v in item.items() if k not in ("pk", "gate_status")})
        st["status"] = item.get("gate_status", "active")
        return st

    def put(self, model_id, version, state):
        st = dict(state)
        st["history"] = st["history"][-HISTORY_CAP:]
        st.pop("status", None)  # the pause flag only changes through claim/release
        names = {f"#{k}": k for k in st}
        values = {f":{k}": _to_ddb(v) for k, v in st.items()}
        self.table.update_item(Key=self._pk(model_id, version),
                               UpdateExpression="SET " + ", ".join(f"#{k} = :{k}" for k in st),
                               ExpressionAttributeNames=names, ExpressionAttributeValues=values)

    def _flip(self, model_id, version, cond, new, vals):
        from botocore.exceptions import ClientError
        try:
            self.table.update_item(Key=self._pk(model_id, version), UpdateExpression="SET #s = :new",
                                   ConditionExpression=cond, ExpressionAttributeNames={"#s": "gate_status"},
                                   ExpressionAttributeValues={":new": new, **vals})
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise

    def claim_pause(self, model_id, version):
        return self._flip(model_id, version, "attribute_not_exists(#s) OR #s <> :new", "paused", {})

    def release_pause(self, model_id, version):
        return self._flip(model_id, version, "#s = :p", "active", {":p": "paused"})
