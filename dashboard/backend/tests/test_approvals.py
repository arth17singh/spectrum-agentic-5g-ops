"""The approval route refuses what it cannot attribute. An unknown decision, an unknown approver or a missing
approver is turned away before anything is signed or ledgered, and four-eyes still needs two different people."""

import copy

import pytest
from fastapi.testclient import TestClient

from spectrum import main
from spectrum.engines.oversight import HUMANS

DECISIONS = ["approve", "approve_with_changes", "reject", "escalate"]


@pytest.fixture(scope="module")
def client():
    # A loopback client, like the local console. Not entered as a context manager, so the lifespan's
    # simulation loop never starts and the engine only changes through these requests.
    return TestClient(main.app, base_url="http://127.0.0.1:8787", client=("127.0.0.1", 50000))


def new_request(tier: str = "high") -> dict:
    """File a fresh request. Its plan id is not `restore-...`, so approving it releases no agent."""
    return main.engine.oversight.create(
        title="Test change", tier=tier, requested_by="coordinator", incident=None,
        plan={"id": "plan-test", "steps": []}, preflight={"blast_radius_pct": 6.0},
        reasons=[{"policy": "P-003", "text": "blast radius > 5%"}], dissent=[], model_opinions=[],
        summary="Test change")


def ledger_height() -> int:
    return len(main.engine.ledger.records)


def test_unknown_decision_is_400_and_lists_the_allowed_ones(client):
    item = new_request()
    before, height = copy.deepcopy(item), ledger_height()
    res = client.post(f"/api/approvals/{item['id']}/aprove", json={"by": "a.rivera"})
    assert res.status_code == 400
    assert "approve, approve_with_changes, reject, escalate" in res.json()["detail"]
    assert item == before and ledger_height() == height


@pytest.mark.parametrize("decision", DECISIONS)
def test_unknown_approver_is_400_before_anything_is_signed(client, decision):
    item = new_request()
    before, height = copy.deepcopy(item), ledger_height()
    res = client.post(f"/api/approvals/{item['id']}/{decision}", json={"by": "mallory", "reason": "test"})
    assert res.status_code == 400
    detail = res.json()["detail"]
    assert "mallory" in detail and all(known in detail for known in HUMANS)
    # no approval slot, history entry, escalation or status change, and no ledger record
    assert item == before and ledger_height() == height


def test_engine_refuses_an_unknown_approver_instead_of_signing_as_a_rivera():
    item = new_request()
    before, height = copy.deepcopy(item), ledger_height()
    with pytest.raises(ValueError, match="unknown approver"):
        main.engine.oversight.decide(item["id"], "approved", by="mallory")
    assert item == before and ledger_height() == height


def test_missing_approver_is_422(client):
    item = new_request()
    before, height = copy.deepcopy(item), ledger_height()
    url = f"/api/approvals/{item['id']}/approve"
    assert client.post(url, json={"device": "console"}).status_code == 422
    assert client.post(url).status_code == 422
    assert item == before and ledger_height() == height


def test_known_approver_signs_a_single_approval_as_themselves(client):
    item = new_request("high")
    res = client.post(f"/api/approvals/{item['id']}/approve", json={"by": "j.okafor", "device": "mobile"})
    assert res.status_code == 200
    out = res.json()["item"]
    assert out["status"] == "approved"
    assert [(a["by"], a["name"]) for a in out["approvals"]] == [("j.okafor", "J. Okafor")]
    rec = main.engine.ledger.records[out["approvals"][0]["ledger_seq"] - 1]
    assert rec["type"] == "approval.decision" and rec["actor"] == "human:j.okafor"
    assert rec["payload"]["approval"] == item["id"] and rec["payload"]["by"] == "j.okafor"


def test_four_eyes_needs_two_different_known_approvers(client):
    item = new_request("critical")
    url = f"/api/approvals/{item['id']}/approve"
    # an unknown id used to be signed as A. Rivera, so it could fill a slot; now it fills none
    assert client.post(url, json={"by": "mallory"}).status_code == 400
    first = client.post(url, json={"by": "a.rivera"}).json()["item"]
    assert first["status"] == "pending" and [a["by"] for a in first["approvals"]] == ["a.rivera"]
    height = ledger_height()
    again = client.post(url, json={"by": "a.rivera"})  # the same person again is ignored, as before
    assert again.status_code == 200
    assert again.json()["item"]["status"] == "pending" and len(again.json()["item"]["approvals"]) == 1
    assert ledger_height() == height
    second = client.post(url, json={"by": "j.okafor"}).json()["item"]
    assert second["status"] == "approved"
    assert [a["by"] for a in second["approvals"]] == ["a.rivera", "j.okafor"]
