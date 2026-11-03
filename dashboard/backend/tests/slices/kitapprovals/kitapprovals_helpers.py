"""Shared fakes and clients for the kitapprovals tests. Nothing here touches AWS."""

from __future__ import annotations

import hashlib
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from spectrum.slices.kitapprovals import router
from spectrum.slices.kitapprovals.aws import Aws
from spectrum.slices.kitapprovals.fake import FakeDdb, FakeLogs, FakeSts
from spectrum.slices.kitapprovals.pin import PinCheck, write_pin
from spectrum.slices.kitapprovals.routes import Service, get_service

PIN = "246810"
MARK_ACCOUNT = "111122223333"
MARK_EMAIL = "marker@example.com"
MARK_ARN = f"arn:aws:sts::{MARK_ACCOUNT}:assumed-role/AWSReservedSSO_bb26-Participant_abc/{MARK_EMAIL}"
NOW = 1_800_000_000.0


def change_attr(change: str, item_id: str = "step-0", reason: str = "raise coverage on cell 2") -> str:
    """Exactly what the kit's interceptor stores (KIT L295)."""
    return json.dumps({"item_id": item_id, "change": change, "reason": reason})


def kit_key(case_id: str, item_id: str, change: str) -> str:
    return "chg-" + hashlib.sha256(json.dumps([case_id, item_id, change]).encode()).hexdigest()[:12]


def pending_row(change="cell=2;d_power_db=-1.0;d_tilt_deg=+1.0", case_id="case-1a2b3c4d", item_id="step-0",
                age=400, **extra) -> dict:
    return {"pk": kit_key(case_id, item_id, change), "status": "PENDING", "agent": "noc_lead", "case_id": case_id,
            "change": change_attr(change, item_id), "created_at": int(NOW) - age, **extra}


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


class Clock:
    def __init__(self, t: float = NOW):
        self.t = t

    def __call__(self) -> float:
        return self.t


def make(tmp_path, monkeypatch, items=None, streams=("control",), pin=True, page_size=2, offline=False):
    monkeypatch.setenv("APPROVALS_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("APPROVERS", raising=False)
    monkeypatch.delenv("APPROVALS_PSEUDONYM", raising=False)
    monkeypatch.delenv("KITAPPROVALS_FIXTURE", raising=False)
    if pin:
        write_pin(PIN, tmp_path / "state" / "pin")
    ddb, logs, sts = FakeDdb(items or [], page_size=page_size), FakeLogs(set(streams)), FakeSts(MARK_ARN)
    clock = Clock()
    svc = Service(aws=Aws(ddb, logs, sts, "FIXTURE"), offline=offline, pin=PinCheck(clock=clock), clock=clock)
    return svc, ddb, logs, sts, clock


def client(svc, host="localhost:8787", peer=("127.0.0.1", 50000)) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_service] = lambda: svc
    return TestClient(app, base_url=f"http://{host}", client=peer)


def post_headers(host="localhost:8787") -> dict:
    return {"Origin": f"http://{host}", "X-Spectrum-Client": "kitapprovals", "Content-Type": "application/json",
            "Sec-Fetch-Site": "same-origin"}


def decide(c: TestClient, row: dict, decision="approve", reason="checked: coverage gain predicted", pin=PIN,
           change_sha=None, headers=None):
    body = {"decision": decision, "reason": reason, "change_sha256": change_sha or sha(row["change"]), "pin": pin}
    return c.post(f"/api/kitapprovals/{row['pk']}/decide", content=json.dumps(body),
                  headers=headers if headers is not None else post_headers())
