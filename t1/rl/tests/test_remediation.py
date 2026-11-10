"""remediation-v1: policy wrapper compatibility, twin labels, candidates, BC artifact, evaluation records."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from t1.rl.policy import Policy, load, parse_change
from t1.rl.remediation import bc
from t1.rl.remediation import data as D
from t3.evals import incidents as inc
from t3.evals.graders import load_targets
from t3.evals.twin_backend import ReplayTwin, single_thread

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "t1" / "rl" / "artifacts"
single_thread()

# ppo_sm_50k / ppo_v0 proposals recorded BEFORE t1/rl/policy.py gained observation_for and the deadzone
# (Darwin-arm64, venv torch 2.15.0: the twin is a float32 GP, so other platforms may differ at 0.1 steps).
GOLDEN = {
    0: ("cell=0;d_power_db=-1.4;d_tilt_deg=-0.3|cell=1;d_power_db=+4.0;d_tilt_deg=-0.2|cell=2;d_power_db=+8.0;"
        "d_tilt_deg=-2.0|cell=3;d_power_db=+1.9;d_tilt_deg=+2.2",
        "cell=0;d_power_db=-0.3;d_tilt_deg=-1.8|cell=1;d_power_db=+1.0;d_tilt_deg=+5.0|cell=2;d_power_db=+2.9;"
        "d_tilt_deg=+4.5|cell=3;d_power_db=-1.7;d_tilt_deg=+3.9"),
    2: ("cell=0;d_power_db=-1.2;d_tilt_deg=-0.3|cell=1;d_power_db=+3.8;d_tilt_deg=-0.1|cell=2;d_power_db=+8.0;"
        "d_tilt_deg=-2.6|cell=3;d_power_db=+2.0;d_tilt_deg=+2.0", None),
    30: ("cell=0;d_power_db=-1.0;d_tilt_deg=+0.0|cell=1;d_power_db=+3.0;d_tilt_deg=+0.1|cell=2;d_power_db=+8.0;"
         "d_tilt_deg=-1.0|cell=3;d_power_db=+2.2;d_tilt_deg=+1.8", None),
}


def faulted_state(spec: dict) -> dict:
    targets = load_targets()
    tw = ReplayTwin(spec["scenario"], spec["seed"], spec["preset"])
    tw.reset()
    state = None
    for ch in inc.script(spec, targets, with_fault=True):
        state = tw.apply(ch)
    return state


# --------------------------------------------------------------------------- the policy wrapper

@pytest.mark.skipif(f"{platform.system()}-{platform.machine()}" != "Darwin-arm64",
                    reason="golden proposals were recorded on Darwin-arm64")
def test_existing_policies_propose_exactly_as_before():
    sm, v0 = load(ARTIFACTS / "ppo_sm_50k.zip"), load(ARTIFACTS / "ppo_v0.zip")
    assert sm.deadzone == (0.0, 0.0) and v0.deadzone == (0.0, 0.0)
    for k, (want_sm, want_v0) in GOLDEN.items():
        state = faulted_state(inc.candidate(k))
        assert sm.propose_text(state) == want_sm
        if want_v0:
            assert v0.propose_text(state) == want_v0


class FixedModel:
    """Stands in for an SB3 model: always the same normalised action."""

    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)

    def predict(self, obs, deterministic=True):
        return self.action, None


def test_deadzone_zeroes_only_small_deltas_and_is_off_by_default():
    state = faulted_state(inc.candidate(1))
    # power +0.05*8 = 0.4 dB on cell 0, +0.5*8 = 4 dB on cell 1; tilt -0.1*5 = -0.5 deg on cell 2, +0.8*5 = +4 deg on 3
    action = [0.05, 0.5, 0.0, 0.0, 0.0, 0.0, -0.1, 0.8]
    plain = Policy(FixedModel(action), {"obs": "masked"})
    dz = Policy(FixedModel(action), {"obs": "masked", "postprocess": {"deadzone_power_db": 1.0, "deadzone_tilt_deg": 1.0}})
    assert {c["cell"] for c in plain.propose(state)} == {0, 1, 2, 3}
    got = {c["cell"]: (c["d_power_db"], c["d_tilt_deg"]) for c in dz.propose(state)}
    assert set(got) == {1, 3} and got[1][0] == pytest.approx(4.0) and got[3][1] == pytest.approx(4.0)
    with pytest.raises(ValueError):
        Policy(FixedModel(action), {"postprocess": {"deadband": 1.0}})


# --------------------------------------------------------------------------- data

def test_candidates_file_is_the_generator_output_on_clean_dev_seeds():
    specs = D.load_candidates()  # refuses a file the generator does not reproduce
    meta = json.loads(D.META_PATH.read_text())
    assert len(specs) == D.N_CANDIDATES == meta["n_candidates"]
    assert meta["exclusion"]["clean"] is True and not any(meta["spec_hash_overlap"].values())
    seeds = [s["seed"] for s in specs]
    assert min(seeds) >= D.SEED_BASE and len(set(seeds)) == len(seeds)
    assert all(inc.split_of(s) == "dev" for s in seeds[:200])
    assert all(s["role"] == D.role_of(s["seed"]) for s in specs[:200])
    test_seeds = {json.loads(x)["seed"] for x in inc.TEST_PATH.read_text().splitlines() if x.strip()}
    assert not test_seeds & set(seeds)
    assert {s["fault"][0]["cell"] for s in specs} == {0, 1, 2, 3}


def test_unit_target_scales_back_to_the_label():
    from ran_rl.env.config import EnvConfig
    from ran_rl.env.observation import scale_action

    from t1.rl.policy import quantise_toward_zero

    label = np.array([6.0, 0.0, -3.0, 4.0, -5.0, 0.0, 3.0, -2.5])
    got = quantise_toward_zero(scale_action(D.unit_target(label), EnvConfig().safety))
    assert np.allclose(got, label)


@pytest.fixture(scope="module")
def kept_rows():
    specs = D.load_candidates()[:12]
    rows = D.generate(specs, load_targets())
    assert any(r["kept"] for r in rows)
    return specs, rows


def test_labels_are_the_way_back_to_the_commissioned_configuration(kept_rows):
    _, rows = kept_rows
    for r in [r for r in rows if r["kept"]][:3]:
        kinds = {s["kind"]: np.asarray(s["label"]) for s in r["samples"]}
        assert set(kinds) == set(D.SAMPLE_KINDS)
        inverse = np.asarray(r["inverse"])
        for k in ("fault@2", "fault@3", "fault@4"):
            assert np.allclose(kinds[k], inverse)
        for k in ("fixed@3", "fixed@4", "healthy@2", "healthy@3", "healthy@4"):
            assert not np.any(kinds[k])
        assert np.allclose(kinds["half@3"] + np.trunc(np.round(inverse / 2 / 0.1, 6)) * 0.1, inverse)
        n = len(inverse) // 2
        assert set(np.flatnonzero((inverse[:n] != 0) | (inverse[n:] != 0)).tolist()) == set(r["fault_cells"])
        assert r["recovery_of_inverse"] is not None
    dropped = [r for r in rows if not r["kept"]]
    assert all(r["samples"] == [] for r in dropped)


# --------------------------------------------------------------------------- BC artifact

@pytest.fixture(scope="module")
def tiny_artifact(kept_rows, tmp_path_factory):
    specs, rows = kept_rows
    for r in rows:  # the fixture has too few incidents for every role: train on all kept, validate on all kept
        r["role"] = "train" if r["kept"] else r["role"]
    val = [dict(r, role="val") for r in rows if r["kept"]]
    out = bc.train(rows + val, {s["seed"]: s for s in specs}, load_targets(), version="v1-test",
                   workdir=tmp_path_factory.mktemp("rounds"),
                   hyper={"epochs": 15, "patience": 15, "dagger_rounds": 1, "dagger_epochs": 5, "dagger_patience": 5})
    d = tmp_path_factory.mktemp("model")
    art = bc.save(out["model"], d, "v1-test", {"run_id": "test"})
    return Path(art["zip"]), out["record"]


def test_bc_artifact_is_a_drop_in_for_the_policy_loader(tiny_artifact):
    path, record = tiny_artifact
    assert record["rounds"][0]["fit"]["epochs_run"] >= 1 and len(record["rounds"]) == 2
    pol = load(path)
    assert (pol.model_id, pol.version, pol.obs_mode) == ("t1-remediation-bc", "v1-test", "masked")
    assert pol.deadzone == (1.0, 1.0)
    from stable_baselines3 import PPO

    base = PPO.load(str(ARTIFACTS / "ppo_sm_50k.zip"), device="cpu")
    assert pol._model.observation_space == base.observation_space
    assert pol._model.action_space == base.action_space
    assert str(pol._model.policy) == str(base.policy)  # same actor-critic architecture
    text = pol.propose_text(faulted_state(inc.candidate(3)))
    assert text == "" or parse_change(text)


# --------------------------------------------------------------------------- evaluation records

@pytest.fixture(scope="module")
def tiny_heldout(tmp_path_factory):
    """A real (twin-built) two-incident held-out file with its sidecar and meta, like the frozen one."""
    from t1.rl.remediation import heldout as H

    gen = H.generate(strata=[2, 3], quota=1, max_candidates=8, log=lambda m: None)
    d = tmp_path_factory.mktemp("heldout")
    rc = H.self_replay([r["incident"] for r in gen["chosen"]], load_targets())
    H.write_outputs(gen, d, "0" * 40, None, None, smoke=True, replay_check=rc)
    return d / H.SUITE_NAME


def test_evaluate_smoke_items_and_part_e_replay(tiny_artifact, tiny_heldout, tmp_path):
    sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools" / "tests")]
    check_unit = pytest.importorskip("check_unit")
    evfake = pytest.importorskip("evfake")
    from t1.rl.remediation import evaluate as E

    path, _ = tiny_artifact
    ev = tmp_path / "ev"
    rid = "t3.evals.remediation-policy-smoke-test"
    assert E.main(["--run-id", rid, "--evidence-dir", str(ev), "--output-dir", str(tmp_path / "out"),
                   "--policy", str(path), "--smoke", "--limit-heldout", "2", "--heldout-file", str(tiny_heldout),
                   "--limit-test", "1", "--limit-confirm", "1", "--holdout", "1", "--declare"]) == 0
    items = [json.loads(x) for x in (ev / f"records/{rid}/item.jsonl").read_text().splitlines()]
    res = json.loads((ev / f"results/{rid}.json").read_text())
    assert len(items) == res["n_items"] == 6
    assert {it["extra"]["suite"] for it in items} == {"heldout", "test-masked", "test-visible", "confirm",
                                                      "dev-holdout"}
    for it in items:  # the exact inverse knows the fault: right cell, nothing on the healthy network
        o = it["extra"]["operators"]["oracle_inverse"]
        assert o["model_id"] == "oracle-exact-inverse-v1" and o["diagnosis_hit"] is True
        assert o["healthy_false_change"] is False and o["cells_touched"] == len(it["extra"]["incident"]["fault_cells"])
    for it in items:
        ops = it["extra"]["operators"]
        assert set(ops) == set(E.OPERATORS)
        d = it["extra"]["diff"]
        assert it["score"] == d["percell_recovered"] == (float(ops["v1"]["percell_recovered"])
                                                          - float(ops["ppo_sm_50k"]["percell_recovered"]))
        assert it["passed"] is ops["v1"]["percell_recovered"]
        assert d["network_recovery"] == pytest.approx(ops["v1"]["recovery"] - ops["ppo_sm_50k"]["recovery"])
        assert d["diagnosis_hit"] == float(ops["v1"]["diagnosis_hit"]) - float(ops["ppo_sm_50k"]["diagnosis_hit"])
        assert all(o["replay_consistent"] for o in ops.values())
        assert all(o["percell_version"] == "percell-v1" for o in ops.values())
        for m in E.MEASURES:
            assert all(m in o for o in ops.values())
    decl = json.loads((ev / f"declared/{rid}.json").read_text())
    assert decl["target"] == {"metric": "joint_ci_low_margin", "op": ">", "value": 0.0, "unit": "margin"}
    names = [c["name"] for c in decl["twin"]["config"]["joint_target"]["components"]]
    assert names == ["percell_recovered", "network_recovery_noninferior", "diagnosis_hit"]
    joint = res["details"]["joint"]
    assert [c["name"] for c in joint["components"]] == names
    assert res["value"] == pytest.approx(min(c["ci_low"] - c["threshold"] for c in joint["components"]))
    assert res["passed"] is (res["value"] > 0) is all(c["passed"] for c in joint["components"])
    assert set(res["details"]["replication"]) == {"test-visible", "confirm", "dev-holdout"}
    assert set(res["details"]["continuity"]) == {"test-masked"}
    assert res["details"]["gain"]["subset"] == {"field": "extra.suite", "equals": "heldout"}
    assert res["details"]["gain"]["paired_n"] == 2
    assert set(res["details"]["oracle_ceiling"]) == {"heldout", "test-masked", "test-visible", "confirm", "dev-holdout"}
    for f in res["details"]["flags"]:
        assert f["flag"] == "ORACLE_CEILING_LOW" and f["oracle_percell_recovered"] < 0.9
    # the independent checker replays the joint target, component by component, from the items (fake S3)

    def part_e(result_bytes: bytes):
        aws = evfake.FakeAws(now="2026-11-04T08:00:00Z")
        t = datetime(2026, 11, 4, 7, 0, 0, tzinfo=timezone.utc)
        for rel in (f"declared/{rid}.json", f"records/{rid}/item.jsonl", f"results/{rid}.json"):
            t += timedelta(seconds=1)
            body = result_bytes if rel.startswith("results/") else (ev / rel).read_bytes()
            aws.s3.put("fixture-bucket", "evidence/" + rel, body, t.strftime("%Y-%m-%dT%H:%M:%SZ"))
        ck = check_unit.Checker(rid, check_unit.S3Evidence(aws.client("s3"), "s3://fixture-bucket/evidence/"), aws,
                                None, poll_s=0.0, now=aws.now())
        ck.load()
        ck.decl_time = ck.ev.time(ck.decl_rel)
        ck.part_e()
        return ck.parts["e"]

    good = part_e((ev / f"results/{rid}.json").read_bytes())
    assert good.state == check_unit.PASS, "\n".join(good.lines)
    assert sum("ci_low" in line for line in good.lines) >= 3
    # a component changed after the declaration (a looser threshold) is caught
    bad = json.loads((ev / f"results/{rid}.json").read_text())
    bad["details"]["joint"]["components"][1]["threshold"] = -0.5
    assert part_e(json.dumps(bad).encode()).state == check_unit.FAIL
    # a recorded interval that does not replay is caught
    bad = json.loads((ev / f"results/{rid}.json").read_text())
    bad["details"]["joint"]["components"][2]["spec"]["ci_low"] += 0.01
    assert part_e(json.dumps(bad).encode()).state == check_unit.FAIL


def test_evaluate_refuses_limits_outside_smoke_and_big_smokes():
    from t1.rl.remediation import evaluate as E

    with pytest.raises(SystemExit):
        E.main(["--print-declaration", "--limit-test", "1"])
    with pytest.raises(SystemExit):
        E.main(["--print-declaration", "--smoke", "--limit-test", "18", "--limit-confirm", "18", "--holdout", "1"])
    with pytest.raises(SystemExit):  # the held-out set (1,008) is never a smoke unless limited
        E.main(["--print-declaration", "--smoke", "--limit-test", "1", "--limit-confirm", "1", "--holdout", "1"])


# --------------------------------------------------------------------------- learning curve

def test_learning_curve_subsets_are_nested_and_keep_v1_order():
    from t1.rl.remediation import learning_curve as LC

    seeds = np.repeat(np.arange(1000, 1040), 3)
    a = {"seed": seeds, "role": np.where(seeds < 1030, "train", "val"),
         "kind": np.tile(np.array(["fault@2", "dagger-fault@4", "healthy@2"]), 40),
         "target": np.arange(120, dtype=np.float32)[:, None] * np.ones((1, 8), dtype=np.float32),
         "obs": {"power": np.arange(120, dtype=np.float32)[:, None] * np.ones((1, 4), dtype=np.float32)}}
    m = LC.fraction_masks(a, LC.FRACTIONS)
    sets = [set(seeds[m[f]["mask"]].tolist()) for f in LC.FRACTIONS]
    assert [len(s) for s in sets] == [3, 8, 15, 30] and all(x <= y for x, y in zip(sets, sets[1:]))
    full = LC.subset(a, m[1.0]["mask"])
    assert full["target"][:, 0].tolist() == sorted(full["target"][:, 0].tolist())  # original order kept
    assert not np.isin(a["kind"][m[1.0]["mask"]], ["dagger-fault@4"]).any()  # twin-labelled states only
    assert set(seeds[LC.twin_labelled(a, "val")].tolist()) == set(range(1030, 1040))


def test_learning_curve_smoke_and_part_e_replay(kept_rows, tiny_heldout, tmp_path):
    sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools" / "tests")]
    check_unit = pytest.importorskip("check_unit")
    evfake = pytest.importorskip("evfake")
    from t1.rl.remediation import learning_curve as LC

    _, rows = kept_rows
    kept = [dict(r) for r in rows if r["kept"]]
    for i, r in enumerate(kept):
        r["role"] = "val" if i % 3 == 2 else "train"
    arr = D.to_arrays(kept, bc.OBS_MODE)
    ds = tmp_path / "train-out" / "dataset.npz"
    ds.parent.mkdir()
    np.savez_compressed(ds, target=arr["target"], seed=np.array([i[0] for i in arr["info"]]),
                        role=np.array([i[1] for i in arr["info"]]), kind=np.array([i[2] for i in arr["info"]]),
                        **{f"obs_{k}": v for k, v in arr["obs"].items()})
    ev = tmp_path / "ev"
    rid = "t1.rl.remediation-learning-curve-smoke-test"
    assert LC.main(["--run-id", rid, "--evidence-dir", str(ev), "--output-dir", str(tmp_path / "out"),
                    "--train-output", str(ds.parent), "--smoke", "--heldout-file", str(tiny_heldout),
                    "--limit-heldout", "2", "--epochs", "2", "--no-dataset-check", "--declare"]) == 0
    items = [json.loads(x) for x in (ev / f"records/{rid}/item.jsonl").read_text().splitlines()]
    res = json.loads((ev / f"results/{rid}.json").read_text())
    decl = json.loads((ev / f"declared/{rid}.json").read_text())
    assert decl["target"]["metric"] == LC.METRIC and decl["twin"]["config"]["decision_rule"] == LC.DECISION
    assert len(items) == res["n_items"] == 2 and set(res["details"]["curve"]) == {"lc010", "lc025", "lc050", "lc100"}
    for it in items:
        mods = it["extra"]["models"]
        assert it["score"] == float(mods["lc100"]["percell_recovered"]) - float(mods["lc050"]["percell_recovered"])
    assert res["details"]["decision"] == (LC.DECISION["if_pass"] if res["passed"] else LC.DECISION["if_miss"])

    aws = evfake.FakeAws(now="2026-11-04T09:00:00Z")
    t = datetime(2026, 11, 4, 8, 0, 0, tzinfo=timezone.utc)
    for rel in (f"declared/{rid}.json", f"records/{rid}/item.jsonl", f"results/{rid}.json"):
        t += timedelta(seconds=1)
        aws.s3.put("fixture-bucket", "evidence/" + rel, (ev / rel).read_bytes(), t.strftime("%Y-%m-%dT%H:%M:%SZ"))
    ck = check_unit.Checker(rid, check_unit.S3Evidence(aws.client("s3"), "s3://fixture-bucket/evidence/"), aws, None,
                            poll_s=0.0, now=aws.now())
    ck.load()
    ck.decl_time = ck.ev.time(ck.decl_rel)
    ck.part_e()
    assert ck.parts["e"].state == check_unit.PASS, "\n".join(ck.parts["e"].lines)
