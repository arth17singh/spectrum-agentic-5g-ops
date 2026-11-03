"""Offline tests of the Viavi operate declaration and records: built from a fake staging tree and a
fake model (all marked synthetic), validated by contracts/ through tools/evlib.py."""

from __future__ import annotations

import hashlib
import json

import pytest

import declare_viavi_operate as dv
import evlib
import record_viavi_operate as rv

RUN_ID = "t1.rl.viavi_operate-20990101T000000Z-test"
SHA40 = "0" * 40


@pytest.fixture
def setup(tmp_path):
    staging = tmp_path / "staging"
    for rel, text in ((dv.SCENARIO_REL, '{"cells": 4}'), (dv.RSG_CONFIG_REL, "rsg_scenario: x\n"),
                      (dv.STARTER_REL / "config" / "env_viavi.yaml", "use_case: cco\n"),
                      (dv.STARTER_REL / "config" / "reward.yaml", "w: 1\n"),
                      (dv.STARTER_REL / "ran_rl-1.2.9-py3-none-any.whl", "wheel")):
        p = staging / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    model = tmp_path / "model.zip"
    model.write_bytes(b"fake model")
    model_sha = hashlib.sha256(b"fake model").hexdigest()
    prov = tmp_path / "prov.json"
    prov.write_text(json.dumps({"artifact": "t1/rl/artifacts/model.zip", "artifact_sha256": model_sha,
                                "model_id": "t1-ppo-cco-sm", "version": "50k-test",
                                "training": {"run_id": "t1.rl.train-x"}, "evaluation": {"run_id": "t1.rl.eval-x"}}))
    store = evlib.LocalStore(tmp_path / "evidence")
    return staging, model, model_sha, prov, store


def declare(setup, synthetic=True):
    staging, model, _, prov, store = setup
    decl = dv.build(RUN_ID, model, staging, provenance=prov, code_sha=SHA40)
    decl["synthetic"] = synthetic
    evlib.write_declared(decl, store)
    return evlib.read_declared(RUN_ID, store)


def report(model_sha, n_steps):
    steps = [{"step": i, "raw_action": [0.5, -1.0, 0.0, 1.0, 2.0, 0.0, -3.0, 1.0],
              "filtered_action": [0.5, -1.0, 0.0, 1.0, 2.0, 0.0, -3.0, 1.0], "rule_ids": [],
              "objective": 1.5 * i,
              "kpis": {"coverage_ratio": 0.9, "sla_ratio": 0.95, "sinr_mean_db": float("nan") if i == 3 else 7.0}}
             for i in range(1, n_steps + 1)]
    return {"model": "m", "model_sha256": model_sha, "use_case": "cco", "twin": "HttpDigitalTwin", "seed": 0,
            "episode": {"objective": 3.0}, "steps": steps}


def test_declaration_is_valid_and_names_the_run(setup):
    d = declare(setup)
    assert d["unit"] == "t1.rl.viavi_operate"
    assert d["target"] == {"metric": "n_items", "op": "==", "value": 5, "unit": "count"}
    assert d["twin"]["kind"] == "viavi" and d["twin"]["scenario"] == "ran-rl-maveric-4cell"
    assert d["twin"]["config"]["model"]["artifact_sha256"] == setup[2]
    assert d["test_set_sha256"] == hashlib.sha256(b'{"cells": 4}').hexdigest()


def test_declaration_refuses_a_model_that_is_not_the_provenance(setup):
    staging, model, _, prov, _ = setup
    model.write_bytes(b"another model")
    with pytest.raises(SystemExit, match="provenance"):
        dv.build(RUN_ID, model, staging, provenance=prov, code_sha=SHA40)


@pytest.mark.parametrize("mutation", [
    {"target": {"metric": "n_items", "op": "=>", "value": 5, "unit": "count"}},
    {"test_set_sha256": "not-hex"},
    {"twin": {"kind": "viavi-x", "version": "v", "scenario": "s", "seed": 0}},
])
def test_contracts_reject_a_bad_declaration(setup, mutation):
    staging, model, _, prov, _ = setup
    rec = evlib._envelope({**dv.build(RUN_ID, model, staging, provenance=prov, code_sha=SHA40), **mutation},
                          "declared", RUN_ID)
    rec["declared_at"] = evlib.utc_now()
    with pytest.raises(evlib.EvidenceError):
        evlib.validate(rec)


def test_five_steps_pass_and_are_written_as_objects(setup, tmp_path):
    d = declare(setup)
    items, details = rv.build(d, report(setup[2], 5), "a" * 64)
    assert [it["item_id"] for it in items] == [f"step-{i}" for i in range(1, 6)]
    assert items[2]["extra"]["kpis"]["sinr_mean_db"] is None  # NaN from the twin goes as null
    store = setup[4]
    evlib.append_records(RUN_ID, "item", items, store=store, layout="objects")
    result = evlib.make_result(d, items, script=rv.SCRIPT, sha=d["code_sha"], details=details)
    evlib.write_result(result, store=store)
    assert len(store.list(f"records/{RUN_ID}/item/")) == 5
    res = evlib.read_result(RUN_ID, store)
    assert (res["value"], res["passed"], res["n_items"]) == (5, True, 5)


def test_three_steps_fail_honestly(setup):
    d = declare(setup)
    items, details = rv.build(d, report(setup[2], 3), "a" * 64)
    result = evlib.make_result(d, items, script=rv.SCRIPT, sha=d["code_sha"], details=details)
    evlib.validate(evlib._envelope(result, "result", RUN_ID))
    assert (result["value"], result["passed"]) == (3, False)


def test_record_refuses_a_report_for_another_model(setup):
    d = declare(setup)
    with pytest.raises(SystemExit, match="model_sha256"):
        rv.build(d, report("0" * 64, 5), "a" * 64)


def test_record_refuses_an_empty_report(setup):
    d = declare(setup)
    with pytest.raises(SystemExit, match="no steps"):
        rv.build(d, report(setup[2], 0), "a" * 64)


def test_record_cli_end_to_end(setup, tmp_path, capsys):
    declare(setup)
    rep = tmp_path / "operate.json"
    rep.write_text(json.dumps(report(setup[2], 5)))
    assert rv.main(["--run-id", RUN_ID, "--report", str(rep), "--evidence-dir", str(setup[4].root)]) == 0
    assert "value=5 passed=True" in capsys.readouterr().out
