"""Pure rules of the Twin view: rung, provenance, approver pseudonyms, missing fields."""

from spectrum.slices.twinview import logic


def change(**kw):
    base = {"kind": "change", "run_id": "r1", "case_id": "c1", "change_id": "chg-000000000001", "item_id": "step-0",
            "change": [{"cell": 1, "d_power_db": 2.0, "d_tilt_deg": 0.0}], "change_text": "cell=1;d_power_db=+2.0;d_tilt_deg=+0.0",
            "twin": {"kind": "maveric-inprocess", "version": "v", "scenario": "s", "seed": 1, "step": 0},
            "measured_delta": {"coverage_ratio": 0.01, "sla_ratio": 0.0, "sinr_p10_db": 0.1, "sinr_mean_db": 0.1},
            "created_at": "2026-11-02T20:00:00Z", "aws_time": None, "synthetic": False}
    base.update(kw)
    return base


def test_rung_needs_the_preset_to_tell_r1_from_r2():
    p, src = logic.preset_of(change(), None)
    assert p is None and src == "not recorded"
    r = logic.rung("maveric-inprocess", p, src)
    assert r["code"] is None and r["known"] is False
    assert logic.rung("maveric-hosted", "official", "x")["code"] == "R1"
    assert logic.rung("maveric-inprocess", "viavi-like", "x")["code"] == "R2"
    assert logic.rung("viavi", None, "x")["code"] == "R3"
    assert logic.rung("none", None, "x")["code"] is None


def test_preset_sources_in_order():
    assert logic.preset_of(change(), {"twin": {"config": None}}) == ("official", "declared.twin.config")
    assert logic.preset_of(change(), {"twin": {"config": {"preset": "viavi-like"}}})[0] == "viavi-like"
    # twin.preset (contract) wins over a string twin.config, which wins over the declaration
    c = change(twin={"kind": "maveric-inprocess", "step": 0, "preset": "viavi-like", "config": "official"})
    assert logic.preset_of(c, {"twin": {"config": "official"}}) == ("viavi-like", "change.twin.preset")
    c = change(twin={"kind": "maveric-inprocess", "step": 0, "config": "viavi-like"})
    assert logic.preset_of(c, {"twin": {"config": "official"}}) == ("viavi-like", "change.twin.config")
    c = change(twin={"kind": "maveric-inprocess", "step": 0, "config": {"twin": {}}})  # a dict config on a change is not a preset
    assert logic.preset_of(c, None) == (None, "not recorded")


def test_approved_by_on_the_change_is_the_first_source():
    d = {"kind": "decision", "run_id": "r1", "case_id": "c1", "tool": "ran___apply_change", "decision": "ALLOW",
         "approval_id": "chg-000000000001", "time": "2026-11-02T20:00:05Z", "approved_by": "person:other"}
    c = change(approved_by="arn:aws:sts::123456789012:assumed-role/Ops/kim", approved_at="2026-11-02T20:00:04Z")
    a = logic.build_case("c1", [c], [d], [], {})["windows"][0]["approval"]
    assert a == {"status": "APPROVED", "by": "kim", "at": "2026-11-02T20:00:04Z", "reason": None,
                 "source": "change.approved_by", "allowed_at": None}


def test_provenance_real_only_when_every_record_is_aws_timed_and_not_synthetic():
    real = change(aws_time="2026-11-02T20:00:01Z")
    assert logic.provenance([real])["state"] == "real"
    assert logic.provenance([change()])["state"] == "unverified"  # tools write aws_time null today
    assert logic.provenance([real, change(synthetic=True)])["state"] == "mixed"
    assert logic.provenance([change(synthetic=True, aws_time="2026-11-02T20:00:01Z")])["state"] == "synthetic"
    assert logic.provenance([])["state"] == "empty"


def test_pseudonym_never_shows_an_account_id():
    assert logic.pseudonym("arn:aws:sts::123456789012:assumed-role/Admin/jane") == "jane"
    assert logic.pseudonym("person:navigator-1") == "person:navigator-1"
    assert "123456789012" not in logic.pseudonym("someone 123456789012")
    assert logic.pseudonym("") is None and logic.pseudonym(None) is None


def test_a_record_without_twin_state_is_drawn_from_headline_kpis_only():
    out = logic.build_case("c1", [change()], [], [], {"r1": None})
    w = out["windows"][0]
    assert w["has_state"] is False and w["before"] is None and w["cells"][0]["applied"] is None
    fields = [m["field"] for m in out["missing"]]
    assert "twin_state" in fields and "twin.preset" in fields and "approved_by" in fields
    assert out["provenance"]["state"] == "unverified"


def test_decision_with_approved_by_gives_the_person():
    d = {"kind": "decision", "run_id": "r1", "case_id": "c1", "tool": "ran___apply_change", "decision": "ALLOW",
         "approval_id": "chg-000000000001", "time": "2026-11-02T20:00:05Z",
         "approved_by": "arn:aws:sts::123456789012:assumed-role/Ops/sam"}
    out = logic.build_case("c1", [change()], [d], [], {})
    a = out["windows"][0]["approval"]
    assert a["by"] == "sam" and a["allowed_at"] == "2026-11-02T20:00:05Z" and a["source"] == "decision"
