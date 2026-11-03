"""Case state and approvals in one store (contract amendment 1, A10).

Selection (`from_env`):
- `T3_STORE=local:<dir>`: JSON files under <dir> (local and Docker runs); the tool also emulates
  the kit's interceptor there (it creates PENDING approvals), since no Gateway is in front of it.
- otherwise the kit's approvals table `APPROVALS_TABLE` (e.g. team13-approvals, key `pk`):
  * approvals are the kit's rows: `pk` = approval id, `status` PENDING | APPROVED | REJECTED | APPLIED;
  * a case row is `pk = "t3-" + case_id` with `case_id, run_id, doc, version` and NO `status`
    attribute, so the kit's `approve` (which lists rows by status) never shows it. `doc` is the case
    dict as JSON text: {case_id, run_id, scenario, seed, config, action_log, twin_kind, predictions,
    created_at};
  * the run row `pk = "t3-active"` holds the current `run_id`, set by the runner.
  Applying a change is ONE TransactWriteItems: the kit's conditional APPROVED -> APPLIED update and
  the case-row write (conditional on its version). Either both happen or neither does.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Protocol

SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
CASE_FIELDS = ("case_id", "run_id", "scenario", "seed", "config", "action_log", "twin_kind",
               "predictions", "created_at")
CASE_PREFIX = "t3-"
ACTIVE_PK = "t3-active"
TRUST_PREFIX = "t3-trust#"   # W5's trust rows in the same table (t3/trust/store.py)


class ConflictError(RuntimeError):
    """The case row changed between read and write; recompute and retry."""


class Store(Protocol):
    emulate_interceptor: bool

    def get_case(self, case_id: str) -> dict | None: ...
    def put_case(self, case: dict, expected_version: int | None) -> None: ...
    def active_run_id(self) -> str | None: ...
    def mapped_run_id(self, case_id: str) -> str | None: ...
    def approval_status(self, key: str) -> str | None: ...
    def approval_item(self, key: str) -> dict | None: ...
    def model_paused(self, model_id: str, version: str) -> bool: ...
    def ensure_pending(self, key: str, item: dict) -> str: ...
    def apply_and_save(self, key: str, case: dict, expected_version: int | None) -> tuple[bool, str | None]:
        """Atomically APPROVED -> APPLIED and write the case. (False, status seen) if not approved;
        raises ConflictError if the case row changed."""


def _safe(name: str) -> str:
    if not isinstance(name, str) or not SAFE_ID.fullmatch(name):
        raise ValueError(f"unsafe id {name!r}")
    return name


def _body(case: dict) -> dict:
    return {k: case.get(k) for k in CASE_FIELDS}


# ---------------------------------------------------------------- local files
class LocalStore:
    emulate_interceptor = True

    def __init__(self, root: str | Path):
        self.root = Path(root)
        for d in ("cases", "approvals"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    def _case_path(self, case_id: str) -> Path:
        return self.root / "cases" / f"{_safe(case_id)}.json"

    def _appr_path(self, key: str) -> Path:
        return self.root / "approvals" / f"{_safe(key)}.json"

    @staticmethod
    def _read(p: Path) -> dict | None:
        return json.loads(p.read_text()) if p.exists() else None

    @staticmethod
    def _write(p: Path, data: dict) -> None:
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(p)

    def get_case(self, case_id: str) -> dict | None:
        row = self._read(self._case_path(case_id))
        return row if row and "action_log" in row else None   # a mapping-only file is not a case yet

    def mapped_run_id(self, case_id: str) -> str | None:
        row = self._read(self._case_path(case_id)) or {}
        return row.get("run_id") if "action_log" not in row else None

    def map_case(self, case_id: str, run_id: str) -> None:
        """Local stand-in for the suite runner's map_case: a mapping row without a case doc."""
        self._write(self._case_path(case_id), {"case_id": case_id, "run_id": _safe(run_id)})

    def put_case(self, case: dict, expected_version: int | None) -> None:
        p = self._case_path(case["case_id"])
        current = (self._read(p) or {}).get("version")
        if current != expected_version:
            raise ConflictError(f"case {case['case_id']} changed (version {current}, expected {expected_version})")
        self._write(p, {**_body(case), "version": (expected_version or 0) + 1})

    def active_run_id(self) -> str | None:
        return (self._read(self.root / "active.json") or {}).get("run_id")

    def set_active_run_id(self, run_id: str) -> None:
        self._write(self.root / "active.json", {"pk": ACTIVE_PK, "run_id": _safe(run_id)})

    def approval_status(self, key: str) -> str | None:
        return (self._read(self._appr_path(key)) or {}).get("status")

    def approval_item(self, key: str) -> dict | None:
        return self._read(self._appr_path(key))

    def model_paused(self, model_id: str, version: str) -> bool:
        """The trust gate's pause flag for one model version (local: trust/<model>@<version>.json)."""
        row = self._read(self.root / "trust" / f"{_safe(f'{model_id}@{version}')}.json") or {}
        return row.get("gate_status") == "paused"

    def ensure_pending(self, key: str, item: dict) -> str:
        current = self._read(self._appr_path(key))
        if current:
            return current["status"]
        self._write(self._appr_path(key), {"pk": key, "status": "PENDING", "created_at": int(time.time()), **item})
        return "PENDING"

    def apply_and_save(self, key: str, case: dict, expected_version: int | None) -> tuple[bool, str | None]:
        appr = self._read(self._appr_path(key))
        if not appr or appr.get("status") != "APPROVED":
            return False, (appr or {}).get("status")
        self.put_case(case, expected_version)  # may raise ConflictError before the approval is used
        self._write(self._appr_path(key), {**appr, "status": "APPLIED", "applied_at": int(time.time())})
        return True, None

    # stands in for the kit's `approve` command
    def decide(self, key: str, status: str, by: str, why: str = "") -> dict:
        item = self._read(self._appr_path(key))
        if not item or item.get("status") != "PENDING":
            raise ValueError(f"{key} is not waiting for a decision")
        item.update(status=status, decided_by=by, decided_at=int(time.time()), decision_reason=why)
        self._write(self._appr_path(key), item)
        return item

    def pending(self) -> list[dict]:
        rows = [json.loads(p.read_text()) for p in sorted((self.root / "approvals").glob("*.json"))]
        return [r for r in rows if r.get("status") == "PENDING"]


# ---------------------------------------------------------------- DynamoDB (the kit's table)
class DynamoStore:
    emulate_interceptor = False  # in AWS the kit's interceptor owns PENDING rows

    def __init__(self, table: str, client: Any = None):
        self.table = table
        self._client = client
        from boto3.dynamodb.types import TypeDeserializer, TypeSerializer

        self._ser, self._de = TypeSerializer(), TypeDeserializer()

    @property
    def client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("dynamodb")
        return self._client

    def _get(self, pk: str) -> dict | None:
        item = self.client.get_item(TableName=self.table, Key={"pk": {"S": pk}}, ConsistentRead=True).get("Item")
        return {k: self._de.deserialize(v) for k, v in item.items()} if item else None

    def _case_item(self, case: dict, expected_version: int | None) -> dict:
        item = {"pk": CASE_PREFIX + case["case_id"], "case_id": case["case_id"], "run_id": case["run_id"],
                "doc": json.dumps(_body(case), separators=(",", ":")), "version": (expected_version or 0) + 1}
        if expected_version is None:  # keep what the runner put on its mapping row (unit, source, ...)
            mapped = self._get(item["pk"]) or {}
            if not mapped.get("doc"):
                item = {**{k: v for k, v in mapped.items() if k not in ("doc", "version", "status")}, **item}
        put = {"TableName": self.table, "Item": {k: self._ser.serialize(v) for k, v in item.items()}}
        if expected_version is None:
            # the first write creates the case, or fills in the runner's doc-less mapping row
            put["ConditionExpression"] = "attribute_not_exists(pk) OR attribute_not_exists(doc)"
        else:
            put["ConditionExpression"] = "version = :v"
            put["ExpressionAttributeValues"] = {":v": {"N": str(expected_version)}}
        return put

    def get_case(self, case_id: str) -> dict | None:
        """The case doc, or None if there is none yet. The suite runner maps a case to its run with a
        row that has no `doc` (t3/suite/runner.py map_case): that is not a case yet."""
        row = self._get(CASE_PREFIX + _safe(case_id))
        if not row or not row.get("doc"):
            return None
        doc = json.loads(row["doc"])
        doc["version"] = int(row.get("version") or 0)
        return doc

    def mapped_run_id(self, case_id: str) -> str | None:
        """The run a runner-mapped case belongs to (a t3-<case_id> row without a doc)."""
        row = self._get(CASE_PREFIX + _safe(case_id)) or {}
        return row.get("run_id") if not row.get("doc") else None

    def put_case(self, case: dict, expected_version: int | None) -> None:
        from botocore.exceptions import ClientError

        try:
            self.client.put_item(**self._case_item(case, expected_version))
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise ConflictError(f"case {case['case_id']} changed concurrently") from exc
            raise

    def active_run_id(self) -> str | None:
        row = self._get(ACTIVE_PK)
        return row.get("run_id") if row else None

    def approval_status(self, key: str) -> str | None:
        row = self._get(key)
        return row.get("status") if row else None

    def approval_item(self, key: str) -> dict | None:
        """The kit's approval row: status, and after `approve`, decided_by (the approver's AWS
        identity, kit 935-946), decided_at and decision_reason."""
        return self._get(key)

    def model_paused(self, model_id: str, version: str) -> bool:
        """The trust gate's pause flag (W5 DynamoTrustStore: pk "t3-trust#<model>@<version>",
        attribute gate_status). Read-only for the tools."""
        row = self._get(f"{TRUST_PREFIX}{model_id}@{version}") or {}
        return row.get("gate_status") == "paused"

    def ensure_pending(self, key: str, item: dict) -> str:
        return self.approval_status(key) or "MISSING"

    def apply_and_save(self, key: str, case: dict, expected_version: int | None) -> tuple[bool, str | None]:
        from botocore.exceptions import ClientError

        approval = {"Update": {
            "TableName": self.table, "Key": {"pk": {"S": key}},
            # the kit's own update (checkpoint_kit.py 161-167)
            "UpdateExpression": "SET #s = :applied, applied_at = :t",
            "ConditionExpression": "#s = :approved",
            "ExpressionAttributeNames": {"#s": "status"},
            "ExpressionAttributeValues": {":applied": {"S": "APPLIED"}, ":approved": {"S": "APPROVED"},
                                          ":t": {"N": str(int(time.time()))}},
            "ReturnValuesOnConditionCheckFailure": "ALL_OLD"}}
        try:
            self.client.transact_write_items(TransactItems=[approval, {"Put": self._case_item(case, expected_version)}])
            return True, None
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "TransactionCanceledException":
                raise
            reasons = exc.response.get("CancellationReasons") or []
            first = reasons[0] if reasons else {}
            if first.get("Code") == "ConditionalCheckFailed":
                old = first.get("Item") or {}
                return False, (self._de.deserialize(old["status"]) if "status" in old else None)
            if len(reasons) > 1 and reasons[1].get("Code") == "ConditionalCheckFailed":
                raise ConflictError(f"case {case['case_id']} changed concurrently") from exc
            raise


# ---------------------------------------------------------------- selection
def from_env(env: dict[str, str] | None = None) -> Store:
    env = os.environ if env is None else env
    spec = env.get("T3_STORE", "")
    if spec.startswith("local:"):
        return LocalStore(spec[len("local:"):] or ".")
    if spec:
        raise ValueError(f"T3_STORE={spec!r}: only local:<dir> is supported; leave it unset for DynamoDB")
    if not env.get("APPROVALS_TABLE"):
        raise ValueError("set T3_STORE=local:<dir>, or APPROVALS_TABLE (the kit's approvals table)")
    return DynamoStore(env["APPROVALS_TABLE"])


def main(argv: list[str] | None = None) -> int:
    """Local approvals: python -m t3.tools.store <dir> [approve|reject chg-... "why"] | <dir> run <run_id>"""
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(main.__doc__)
        return 2
    store = LocalStore(args[0])
    if len(args) == 1:
        for row in store.pending():
            print(row["pk"], row.get("case_id"), row.get("change"))
        return 0
    if args[1] == "run":
        store.set_active_run_id(args[2])
        return 0
    verb, key, why = args[1], args[2], " ".join(args[3:])
    item = store.decide(key, {"approve": "APPROVED", "reject": "REJECTED"}[verb], "person:local", why)
    print(json.dumps(item))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
