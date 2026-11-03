"""Proposal-only rows of the swarm evaluation (t3/evals/live_swarm.py, W4 e55eb1e) are flagged, never decidable."""

import pytest

from spectrum.slices.kitapprovals import logic
from slices.kitapprovals.kitapprovals_helpers import client, decide, make, pending_row

RUN = "t3.swarm-live-20261103T0500Z"


def _mapping(case_id, **kw):
    # exactly what live_swarm.py writes before seeding (common.ddb_put, attribute_not_exists(pk))
    return {"pk": f"t3-{case_id}", "case_id": case_id, "run_id": RUN, "unit": "t3.evals.swarm-proposals",
            "created_at": 1, "source": "t3/evals/live_swarm.py", **kw}


@pytest.mark.parametrize("row,want", [
    ({"run_id": RUN}, True), ({"unit": "t3.evals.swarm-proposals"}, True), ({"source": "t3/evals/live_swarm.py"}, True),
    ({"run_id": "adhoc-case-1a2b3c4d"}, False), ({"run_id": "t3.allow-x"}, False), ({}, False)])
def test_is_eval_mapping(row, want):
    assert logic.is_eval_mapping(row) is want


def test_eval_rows_are_flagged_and_refused(tmp_path, monkeypatch):
    real = pending_row()
    ev_case = "case-1f2e3d4c-i00s0"
    ev = pending_row(case_id=ev_case)
    # after ran___seed_case the tools rewrite the row with doc + run_id (store.py _case_item): still detected
    seeded = "case-1f2e3d4c-i01s0"
    ev2 = pending_row(case_id=seeded, change="cell=1;d_power_db=+1.0;d_tilt_deg=+0.0")
    rows = [real, ev, ev2, _mapping(ev_case), {"pk": f"t3-{seeded}", "case_id": seeded, "run_id": RUN, "doc": "{}",
                                                "version": 2}]
    svc, ddb, *_ = make(tmp_path, monkeypatch, rows)
    c = client(svc)
    got = {r["approval_id"]: r["proposal_only"] for r in c.get("/api/kitapprovals/pending").json()["rows"]}
    assert got == {real["pk"]: False, ev["pk"]: True, ev2["pk"]: True}
    assert c.get(f"/api/kitapprovals/{ev['pk']}").json()["proposal_only"] is True
    for row in (ev, ev2):
        r = decide(c, row)
        assert r.status_code == 409 and r.json()["error"] == "proposal_only"
    assert not [op for op, _ in ddb.calls if op == "UpdateItem"]
    assert decide(c, real).status_code == 200
    # no mapping row, or one rewritten with an adhoc run id: the eval case-id shape still flags it
    lone = pending_row(case_id="case-0a0b0c0d-i17s2")
    svc2, *_ = make(tmp_path, monkeypatch, [lone, {"pk": "t3-case-0a0b0c0d-i17s2", "run_id": "adhoc-x", "doc": "{}"}])
    assert client(svc2).get("/api/kitapprovals/pending").json()["rows"][0]["proposal_only"] is True
    # the GetItem for the mapping aliases its attribute names
    gets = [kw for op, kw in ddb.calls if op == "GetItem" and kw.get("ProjectionExpression")]
    assert gets and all("#u" in kw["ProjectionExpression"] for kw in gets)
