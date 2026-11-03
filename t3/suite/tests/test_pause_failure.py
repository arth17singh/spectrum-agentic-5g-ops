"""Run 1 (t3.deny-20261103T051458Z) stopped at item 23 of 27: the pause forbid came back CREATE_FAILED and the
runner raised. A failed pause must cost only the pause item; every other item still runs, and check.py then
grades all 27 (the pause item as not passed), so check_unit part e sees n_items == the declared count."""
from __future__ import annotations

from t3.suite import check, pause_floor
from t3.suite.tests.fakes import Clock, FakeDDB, FakeEvlib, FakeGateway, FakePolicyClient
from t3.suite.tests.test_suite import N_DENY, make_runner


class FailingPolicyClient(FakePolicyClient):
    def get_policy(self, policyEngineId, policyId):
        return {"policyId": policyId, "status": "CREATE_FAILED"}


def test_a_failed_pause_skips_only_its_item_and_the_run_completes():
    clock, ddb, ev = Clock(), FakeDDB(), FakeEvlib()
    policy = FailingPolicyClient(clock)
    gw = FakeGateway(clock, ddb, policy)
    r = make_runner(clock, ddb, gw, ev, policy, unit="deny")
    summary = r.run()
    calls = ev.runner_calls["t3-test-run"]
    graded = [c for c in calls if not c["extra"].get("setup")]
    assert len(graded) == N_DENY == 27
    assert summary["skipped"] == ["t3-036"] and "CREATE_FAILED" in summary["pause"]["error"]
    assert calls[-1]["item_id"] == "setup-close"           # the run still closed with its run id
    ev2 = FakeEvlib()
    code = check.check("t3-test-run", calls, gw.cp_rows, gw.span_rows, declared=ev.declared[0], evlib=ev2,
                       log=lambda *a: None)
    res = ev2.results[0]
    assert code == 1 and res["n_items"] == 27 and res["value"] == 26 / 27     # honest: one item not run
    assert [m["item_id"] for m in res["details"]["mismatches"]] == ["t3-036"]


def test_pause_statement_uses_the_form_the_validator_accepts():
    p = pause_floor.CedarPause("x", "engine", "arn:aws:bedrock-agentcore:us-east-2:0:gateway/g", version="v1")
    s = p.statement()
    assert "unless { context.input has model_version && context.input.model_version != \"v1\" }" in s
    assert "!(context.input has" not in s
