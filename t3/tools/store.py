"""Case state and approvals, behind two small interfaces.

Selection (see `from_env`):
- `T3_STORE=local:<dir>`: both live in JSON files under <dir> (local and Docker runs).
- otherwise `CASES_TABLE` (partition key `case_id`, string) and `APPROVALS_TABLE` (the kit's
  `<team>-approvals` table, partition key `pk`) in DynamoDB.

Case rows hold `{case_id, run_id, scenario, seed, config, action_log, twin_kind, predictions,
created_at}`. In DynamoDB a row is `{case_id, run_id, doc, version}` where `doc` is that dict as
JSON text (floats and nested lists stay exact; no Decimal conversion). A row the runner wrote
with only `case_id` and `run_id` (and optionally scenario/seed/config/twin_kind) is accepted
and completed on first use.

Approvals follow the kit's table layout: `pk` = approval id, `status` PENDING | APPROVED |
REJECTED | APPLIED. The interceptor creates PENDING rows; a person sets APPROVED; the tool
moves APPROVED to APPLIED with a conditional update, so one approval applies one change.
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


class ConflictError(RuntimeError):
    """The case row changed between read and write; the caller may retry."""


class CaseStore(Protocol):
    def get(self, case_id: str) -> dict | None: ...
    def put(self, case: dict, expected_version: int | None) -> None: ...


class ApprovalStore(Protocol):
    def status(self, key: str) -> str | None: ...
    def mark_applied(self, key: str) -> tuple[bool, str | None]:
        """APPROVED -> APPLIED atomically. Returns (applied, status seen if not applied)."""
    def ensure_pending(self, key: str, item: dict) -> str: ...


def _safe(name: str) -> str:
    if not isinstance(name, str) or not SAFE_ID.fullmatch(name):
        raise ValueError(f"unsafe id {name!r}")
    return name


# ---------------------------------------------------------------- local files
class LocalCaseStore:
    def __init__(self, root: str | Path):
        self.dir = Path(root) / "cases"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, case_id: str) -> Path:
        return self.dir / f"{_safe(case_id)}.json"

    def get(self, case_id: str) -> dict | None:
        p = self._path(case_id)
        return json.loads(p.read_text()) if p.exists() else None

    def put(self, case: dict, expected_version: int | None) -> None:
        p = self._path(case["case_id"])
        current = json.loads(p.read_text()).get("version") if p.exists() else None
        if current != expected_version:
            raise ConflictError(f"case {case['case_id']} changed (version {current}, expected {expected_version})")
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({**case, "version": (expected_version or 0) + 1}, indent=1))
        tmp.replace(p)


class LocalApprovalStore:
    """Same states as the kit's table. `approve()` stands in for the kit's `approve` command."""

    def __init__(self, root: str | Path):
        self.dir = Path(root) / "approvals"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{_safe(key)}.json"

    def _read(self, key: str) -> dict | None:
        p = self._path(key)
        return json.loads(p.read_text()) if p.exists() else None

    def _write(self, item: dict) -> None:
        self._path(item["pk"]).write_text(json.dumps(item, indent=1))

    def status(self, key: str) -> str | None:
        item = self._read(key)
        return item and item.get("status")

    def mark_applied(self, key: str) -> tuple[bool, str | None]:
        item = self._read(key)
        if not item or item.get("status") != "APPROVED":
            return False, (item or {}).get("status")
        self._write({**item, "status": "APPLIED", "applied_at": int(time.time())})
        return True, None

    def ensure_pending(self, key: str, item: dict) -> str:
        current = self._read(key)
        if current:
            return current["status"]
        self._write({"pk": key, "status": "PENDING", "created_at": int(time.time()), **item})
        return "PENDING"

    def decide(self, key: str, status: str, by: str, why: str = "") -> dict:
        item = self._read(key)
        if not item or item.get("status") != "PENDING":
            raise ValueError(f"{key} is not waiting for a decision")
        item.update(status=status, decided_by=by, decided_at=int(time.time()), decision_reason=why)
        self._write(item)
        return item

    def pending(self) -> list[dict]:
        rows = [json.loads(p.read_text()) for p in sorted(self.dir.glob("*.json"))]
        return [r for r in rows if r.get("status") == "PENDING"]


# ---------------------------------------------------------------- DynamoDB
class DynamoCaseStore:
    def __init__(self, table):
        self.table = table  # a boto3 Table resource (or a fake with get_item/put_item)

    def get(self, case_id: str) -> dict | None:
        item = self.table.get_item(Key={"case_id": case_id}, ConsistentRead=True).get("Item")
        if not item:
            return None
        doc = json.loads(item["doc"]) if item.get("doc") else {}
        for k, v in item.items():  # top-level attributes a runner wrote
            if k not in ("doc", "version") and k not in doc:
                doc[k] = int(v) if k == "seed" and v is not None else v
        doc["version"] = int(item["version"]) if item.get("version") is not None else 0
        return doc

    def put(self, case: dict, expected_version: int | None) -> None:
        from botocore.exceptions import ClientError

        body = {k: case.get(k) for k in CASE_FIELDS}
        item = {"case_id": case["case_id"], "run_id": case["run_id"], "doc": json.dumps(body),
                "version": (expected_version or 0) + 1}
        if expected_version is None:
            cond = {"ConditionExpression": "attribute_not_exists(case_id)"}
        elif expected_version == 0:  # a runner-written row without a version
            cond = {"ConditionExpression": "attribute_not_exists(version) OR version = :v",
                    "ExpressionAttributeValues": {":v": 0}}
        else:
            cond = {"ConditionExpression": "version = :v", "ExpressionAttributeValues": {":v": expected_version}}
        try:
            self.table.put_item(Item=item, **cond)
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise ConflictError(f"case {case['case_id']} changed concurrently") from exc
            raise


class DynamoApprovalStore:
    """The kit's approvals table. `mark_applied` is the kit's APPROVED -> APPLIED update."""

    def __init__(self, table):
        self.table = table

    def status(self, key: str) -> str | None:
        item = self.table.get_item(Key={"pk": key}, ConsistentRead=True).get("Item")
        return item and item.get("status")

    def mark_applied(self, key: str) -> tuple[bool, str | None]:
        from botocore.exceptions import ClientError

        try:
            self.table.update_item(
                Key={"pk": key},
                UpdateExpression="SET #s = :applied, applied_at = :t",
                ConditionExpression="#s = :approved",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={":applied": "APPLIED", ":approved": "APPROVED",
                                           ":t": int(time.time())})
            return True, None
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "ConditionalCheckFailedException":
                raise
            return False, self.status(key)

    def ensure_pending(self, key: str, item: dict) -> str:
        # In AWS the kit's interceptor owns PENDING rows; the tool never creates them.
        return self.status(key) or "MISSING"


# ---------------------------------------------------------------- selection
def from_env(env: dict[str, str] | None = None) -> tuple[Any, Any, bool]:
    """(case store, approval store, emulate_interceptor)."""
    env = os.environ if env is None else env
    spec = env.get("T3_STORE", "")
    if spec.startswith("local:"):
        root = spec[len("local:"):] or "."
        return LocalCaseStore(root), LocalApprovalStore(root), env.get("T3_EMULATE_INTERCEPTOR", "1") == "1"
    if spec:
        raise ValueError(f"T3_STORE={spec!r}: only local:<dir> is supported; leave it unset for DynamoDB")
    if not env.get("CASES_TABLE") or not env.get("APPROVALS_TABLE"):
        raise ValueError("set T3_STORE=local:<dir>, or both CASES_TABLE and APPROVALS_TABLE")
    import boto3

    ddb = boto3.resource("dynamodb")
    return (DynamoCaseStore(ddb.Table(env["CASES_TABLE"])),
            DynamoApprovalStore(ddb.Table(env["APPROVALS_TABLE"])), False)


def main(argv: list[str] | None = None) -> int:
    """Local approvals: python -m t3.tools.store <dir> [approve|reject chg-... "why"]"""
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(main.__doc__)
        return 2
    store = LocalApprovalStore(args[0])
    if len(args) == 1:
        for row in store.pending():
            print(row["pk"], row.get("case_id"), row.get("change"))
        return 0
    verb, key, why = args[1], args[2], " ".join(args[3:])
    item = store.decide(key, {"approve": "APPROVED", "reject": "REJECTED"}[verb], "person:local", why)
    print(json.dumps(item))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
