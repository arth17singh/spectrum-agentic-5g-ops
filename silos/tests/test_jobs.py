"""Tests for the service-neutral core, the job entry points (dry runs and a real run into a local evidence dir),
the launch request builder and the declaration templates. Nothing here touches AWS.

    <python> -m pytest silos/tests -q        (from the repo root)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from silos import declarations, evio, features as F, generate as G, job_baselines, job_compare, job_data, job_evaluate, launch, nn_core as nn, scorerun as sr
from silos.tests.test_silos import tiny_cfg

CODE_SHA = "a" * 40


# ---------------------------------------------------------------------------------------------
# nn_core
# ---------------------------------------------------------------------------------------------
def test_nn_core_layout_equals_features():
    assert nn.INPUT_COLUMNS == F.INPUT_COLUMNS and nn.LABELS == F.LABELS and nn.N_INPUTS == F.N_INPUTS == 38
    for k, v in F.RANGES.items():
        assert nn.RANGES[k] == v
    x = np.random.default_rng(0).uniform(F._LO, F._HI, size=(5, 38))
    assert np.allclose(nn.encode(x), 2 * (x - F._LO) / (F._HI - F._LO) - 1)


def test_initial_weights_have_exact_names_and_shapes_and_survive_safetensors():
    p = nn.init_params(3)
    nn.check_params(p)
    assert set(p) == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "out.weight", "out.bias"}
    assert p["fc1.weight"].shape == (128, 38) and p["out.weight"].shape == (4, 128)
    blob = nn.safetensors_dumps(p, {"k": "v"})
    q = nn.safetensors_loads(blob)
    assert all(np.array_equal(p[k], q[k]) for k in p) and all(q[k].dtype == np.float32 for k in q)
    assert nn.safetensors_dumps(p) == nn.safetensors_dumps(q)  # deterministic bytes
    bad = dict(p); bad["fc1.weight"] = np.zeros((128, 37), dtype=np.float32)
    with pytest.raises(ValueError):
        nn.check_params(bad)
    with pytest.raises(ValueError):
        nn.check_params({k: v for k, v in p.items() if k != "out.bias"})


def test_safetensors_header_follows_the_format():
    import struct

    blob = nn.safetensors_dumps({"a": np.arange(6, dtype=np.float32).reshape(2, 3)})
    (n,) = struct.unpack("<Q", blob[:8])
    header = json.loads(blob[8 : 8 + n])
    assert n % 8 == 0 and header["a"] == {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]} and len(blob) == 8 + n + 24


def test_torch_and_numpy_forward_agree_and_state_dict_keys_are_the_exchanged_names():
    torch = pytest.importorskip("torch")
    m = nn.torch_model()
    assert set(m.state_dict()) == set(nn.SHAPES)
    nn.load_params_into(m, nn.init_params(1))
    x = np.random.default_rng(1).uniform(F._LO, F._HI, size=(7, 38))
    with torch.no_grad():
        a = m(torch.tensor(nn.encode(x), dtype=torch.float32)).numpy() / nn.LABEL_SCALE
    assert np.allclose(a, nn.forward_np(nn.params_from_torch(m), x), atol=1e-5)


def test_scores_and_directions():
    y = np.array([[0.02, 0.0, 1.0, -0.5], [0.0, 0.03, 0.0, 0.0]])
    assert np.allclose(nn.row_scores(y, y), 0)
    assert np.allclose(nn.row_scores(np.zeros_like(y), y)[0], np.mean([1.0, 0.0, 2.0, 0.5 / 0.35]))
    d = nn.direction_accuracy(np.array([[1, 0, 1, 1], [0, 1, 0, 0]], dtype=float), y)
    assert d["d_coverage_ratio"] == 1.0 and d["d_sinr_mean_db"] == 0.0


def test_bootstrap_resamples_clusters_and_can_fail():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(16), 50)
    clear = rng.normal(0.5, 0.2, size=800)
    b = nn.paired_bootstrap(clear, groups, n_boot=500, seed=1)
    assert b["ci_lo"] > 0 and b["n_groups"] == 16
    # a difference that is positive on average but driven by two clusters cannot clear zero
    lumpy = rng.normal(0.0, 0.2, size=800) + np.where(groups < 2, 2.0, -0.2)
    b2 = nn.paired_bootstrap(lumpy, groups, n_boot=500, seed=1)
    assert b2["mean"] > 0 and b2["ci_lo"] < 0
    assert nn.paired_bootstrap(clear, groups, n_boot=500, seed=1) == b  # seeded


# ---------------------------------------------------------------------------------------------
# a tiny data tree with its manifest inside, shared by the job tests
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("tree")
    G.generate(tiny_cfg(), root, workers=4, manifest_path=root / "manifest.json")
    return root


@pytest.fixture()
def job_env(tree, tmp_path, monkeypatch):
    ev = tmp_path / "evidence"
    for k in ("RUN_ID", "EVIDENCE_RUN_ID", "EVIDENCE_S3", "DATA_COPY_S3"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("DATA_DIR", str(tree))
    monkeypatch.setenv("EVIDENCE_DIR", str(ev))
    monkeypatch.setenv("CODE_SHA", CODE_SHA)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "model"))
    return ev


def test_job_data_dry_run_generates_checks_and_validates_without_writing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("EVIDENCE_DIR", str(tmp_path / "ev"))
    rc = job_data.main(["--dry-run", "--out-dir", str(tmp_path / "out"), "--workers", "4"])
    out = capsys.readouterr().out
    assert rc == 0 and '"passed": true' in out
    assert not (tmp_path / "ev").exists()  # nothing written
    assert (tmp_path / "out" / "manifest.json").exists()


def test_job_data_print_declaration(capsys):
    assert job_data.main(["--print-declaration"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["unit"] == "t2.silo_data" and d["target"] == {"metric": "pass_count", "op": ">=", "value": 14, "unit": "files (checked, uploaded, SHA-256 verified)"}
    assert "simulated silos" in d["population"] and d["track"] == "track-2"


@pytest.mark.parametrize("args", [["--method", "local", "--silo", "B"], ["--method", "central"], ["--method", "fedavg"]])
def test_job_baselines_dry_run(job_env, args, capsys):
    rc = job_baselines.main(args + ["--split", "disjoint", "--dry-run"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["passed"] and out["n_items"] == 50 and not job_env.exists()
    assert set(out["mae"]) == set(nn.LABELS)


def test_job_baselines_real_run_writes_a_recomputable_unit(job_env, monkeypatch):
    monkeypatch.setenv("RUN_ID", "t2-test-local-c")
    rc = job_baselines.main(["--method", "local", "--silo", "C", "--split", "disjoint", "--epochs", "1", "--declare-first"])
    assert rc == 0
    ev = evio.get_evlib()
    store = ev.default_store()
    decl, res = ev.read_declared("t2-test-local-c", store), ev.read_result("t2-test-local-c", store)
    items = ev.read_records("t2-test-local-c", "item", store)
    assert decl["unit"] == "t2.local_only.C.disjoint" and decl["target"]["metric"] == "n_items" and decl["code_sha"] == CODE_SHA
    assert res["passed"] and res["value"] == len(items) == decl["n_items"] == 50
    assert abs(ev.summarize_items(items, "n_items") - res["value"]) == 0
    assert all(np.isfinite(it["score"]) and len(it["pred"]) == 4 for it in items)
    mdl = ev.read_model("t2-response-local-c-disjoint", "v1", store)
    assert mdl["model_kind"] == "t2-response" and mdl["artifact_uri"].endswith("model.safetensors")
    blob = (Path(os.environ["MODEL_DIR"]) / "model.safetensors").read_bytes()
    assert nn.sha256_bytes(blob) == mdl["artifact_sha256"]
    nn.check_params(nn.safetensors_loads(blob))
    assert res["details"]["mean_score"] == pytest.approx(np.mean([it["score"] for it in items]))
    # the declaration is earlier than the records
    assert decl["declared_at"] <= res["checked_at"]


def test_evaluate_then_compare_beats_and_measure(job_env, monkeypatch, tmp_path, capsys):
    ev = evio.get_evlib()
    # two score runs on the same shared test: a trained model (local C) and the untrained initial weights
    monkeypatch.setenv("RUN_ID", "t2-test-trained")
    assert job_baselines.main(["--method", "local", "--silo", "B", "--split", "disjoint", "--epochs", "30", "--declare-first"]) == 0
    w = tmp_path / "fl.safetensors"
    w.write_bytes(nn.safetensors_dumps(nn.init_params(7)))
    monkeypatch.setenv("RUN_ID", "t2-test-init")
    assert job_evaluate.main(["--label", "federated", "--split", "disjoint", "--weights", str(w), "--session-id", "S", "--declare-first"]) == 0
    capsys.readouterr()
    # a different-shaped file is refused
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(nn.safetensors_dumps({"fc1.weight": np.zeros((2, 2), dtype=np.float32)}))
    with pytest.raises(ValueError):
        job_evaluate.main(["--label", "dp", "--split", "disjoint", "--weights", str(bad), "--dry-run"])
    # pair the two runs both ways: the differences are exact negatives of each other
    monkeypatch.setenv("RUN_ID", "t2-test-cmp")
    rc1 = job_compare.main(["--kind", "beats", "--x", "t2-test-trained", "--y", "t2-test-init", "--name", "a_vs_b", "--split", "disjoint", "--declare-first"])
    out1 = json.loads(capsys.readouterr().out)
    res = ev.read_result("t2-test-cmp", ev.default_store())
    items = ev.read_records("t2-test-cmp", "item", ev.default_store())
    assert res["details"]["bootstrap"]["method"].startswith("paired bootstrap resampling clusters")
    assert res["target"]["metric"] == "mean_score" and res["value"] == pytest.approx(out1["mean_diff"]) == pytest.approx(np.mean([it["score"] for it in items]))
    assert all(it["score"] == pytest.approx(it["y_score"] - it["x_score"]) for it in items)
    monkeypatch.setenv("RUN_ID", "t2-test-cmp2")
    rc2 = job_compare.main(["--kind", "beats", "--x", "t2-test-init", "--y", "t2-test-trained", "--name", "b_vs_a", "--split", "disjoint", "--declare-first"])
    out2 = json.loads(capsys.readouterr().out)
    assert out2["mean_diff"] == pytest.approx(-out1["mean_diff"])
    # the declaration can fail: it needs the mean AND the interval lower bound above zero, and at most one direction can pass
    assert rc1 == (0 if out1["mean_diff"] > 0 and out1["ci_lower_gt0"] else 1)
    assert rc2 == (0 if out2["mean_diff"] > 0 and out2["ci_lower_gt0"] else 1)
    assert not (rc1 == 0 and rc2 == 0)
    # a measurement never fails on the sign
    monkeypatch.setenv("RUN_ID", "t2-test-cmp3")
    assert job_compare.main(["--kind", "measure", "--x", "t2-test-init", "--y", "t2-test-trained", "--name", "m", "--split", "disjoint", "--declare-first"]) == 0


def test_compare_refuses_runs_on_different_test_sets(job_env, monkeypatch, tmp_path):
    ev = evio.get_evlib()
    monkeypatch.setenv("RUN_ID", "t2-test-one")
    job_baselines.main(["--method", "local", "--silo", "A", "--split", "disjoint", "--epochs", "1", "--declare-first"])
    monkeypatch.setenv("RUN_ID", "t2-test-two")
    job_baselines.main(["--method", "local", "--silo", "A", "--split", "overlap", "--epochs", "1", "--declare-first"])
    with pytest.raises(SystemExit):
        job_compare.main(["--kind", "measure", "--x", "t2-test-one", "--y", "t2-test-two", "--name", "x", "--dry-run"])


# ---------------------------------------------------------------------------------------------
# launch request, packaging, declaration templates
# ---------------------------------------------------------------------------------------------
def test_launch_request_is_complete_and_never_calls_aws(monkeypatch, capsys):
    for k, v in {"ROLE_ARN": "arn:aws:iam::111122223333:role/x", "IMAGE_URI": "img", "SOURCE_S3_URI": "s3://b/src.tgz", "OUTPUT_S3": "s3://b/out/",
                 "EVIDENCE_S3": "s3://b/evidence/", "CODE_SHA": CODE_SHA, "DATA_COPY_S3": "s3://b/t2-silos/r/", "SDK_S3_URI": "s3://b/sdk.zip",
                 "SILO_BUCKET_A": "a", "SILO_BUCKET_B": "b", "SILO_BUCKET_C": "c"}.items():
        monkeypatch.setenv(k, v)
    assert launch.main(["data", "--dry-run"]) == 0
    cap = capsys.readouterr()
    req = json.loads(cap.out)
    assert req["Environment"]["JOB_MODULE"] == "silos.job_data" and req["Environment"]["JOB_NEEDS_SDK"] == "1"
    assert req["ResourceConfig"]["InstanceType"] == "ml.m5.4xlarge" and req["StoppingCondition"]["MaxRuntimeInSeconds"] == 3600
    assert json.loads(req["HyperParameters"]["sagemaker_program"]) == "run_job.py"
    assert "estimate" in cap.err and "unset" not in cap.err
    assert launch.main(["baseline", "--method", "local", "--silo", "B", "--split", "overlap"]) == 0
    req2 = json.loads(capsys.readouterr().out)
    assert req2["Environment"]["JOB_ARGS"].startswith("--method local --silo B --split overlap")
    assert len(req2["TrainingJobName"]) <= 63 and "_" not in req2["TrainingJobName"]


def test_launch_prints_placeholders_when_unset(monkeypatch, capsys):
    for k in ("ROLE_ARN", "IMAGE_URI", "SOURCE_S3_URI", "OUTPUT_S3", "EVIDENCE_S3", "CODE_SHA", "DATA_COPY_S3", "SDK_S3_URI", "SILO_BUCKET_A", "SILO_BUCKET_B", "SILO_BUCKET_C", "RUN_ID"):
        monkeypatch.delenv(k, raising=False)
    launch.main(["compare", "--kind", "beats", "--x", "r1", "--y", "r2", "--name", "n"])
    cap = capsys.readouterr()
    assert "<ROLE_ARN>" in cap.out and "ROLE_ARN" in cap.err


def test_package_contains_code_evlib_and_shim_but_no_tests_or_data(tmp_path):
    import tarfile

    info = launch.package(tmp_path / "src.tgz")
    names = set(tarfile.open(tmp_path / "src.tgz").getnames())
    assert {"run_job.py", "requirements.txt", "tools/evlib.py", "silos/job_data.py", "silos/config.yaml", "silos/manifest.json"} <= names
    assert not any("tests" in n or n.endswith(".csv") or "__pycache__" in n for n in names)
    assert len(info["sha256"]) == 64


def test_committed_templates_match_what_the_code_generates():
    cur = declarations.build_all()
    on_disk = {p.name.replace(".template.json", ""): json.loads(p.read_text()) for p in declarations.DIR.glob("*.template.json")}
    assert set(cur) == set(on_disk) and len(cur) == 17
    for k in cur:
        assert json.loads(json.dumps(cur[k], sort_keys=True)) == on_disk[k], k


def test_declarations_say_what_they_must():
    t = declarations.build_all()
    assert t["compare_federated_vs_B_disjoint"]["target"]["op"] == ">" and "bootstrap" in t["compare_federated_vs_B_disjoint"]
    assert t["compare_federated_vs_C_disjoint"]["declared_also"].count("lower bound") == 1
    assert t["fl_dp_disjoint"]["privacy"]["epsilon"] == 3.0 and "1e-5" in t["fl_dp_disjoint"]["privacy"]["planning_value_not_used"]
    assert t["fl_dp_disjoint"]["privacy"]["delta_by_silo"]["A"] == pytest.approx(1 / 24000**2)
    for k, d in t.items():
        assert d["track"] == "track-2" and "simulated silos" in d["population"] and d["run_id"] == "<RUN_ID>", k
    text = json.dumps(t).lower()
    assert ("cross" + "trainer") not in text and ("4835" + "98626742") not in text
    for k in ("local_only_A_disjoint", "central_disjoint", "fl_federated_disjoint", "compare_dp_cost_disjoint", "compare_distance_to_central_disjoint"):
        assert t[k]["target"]["op"] in (">=", ">") and t[k]["target"]["metric"] in ("n_items", "mean_score")


def test_dp_accounting_service_vs_standard_counts():
    pytest.importorskip("opacus")
    from silos import dp_accounting as dp

    e_service = dp.service_epsilon(24000, 256, 1.0, 1, 20)
    e_standard = dp.standard_epsilon(24000, 256, 1.0, 1, 20)
    assert 1.5 < e_service < 3.0 < e_standard and dp.delta_for(3000) == pytest.approx(1.11e-7, rel=0.01)


def test_declare_only_writes_the_declaration_and_nothing_else(job_env, monkeypatch, capsys):
    monkeypatch.setenv("RUN_ID", "t2-test-declare-only")
    assert job_baselines.main(["--method", "central", "--split", "disjoint", "--declare-only"]) == 0
    ev = evio.get_evlib()
    d = ev.read_declared("t2-test-declare-only", ev.default_store())
    assert d["unit"] == "t2.central.disjoint" and d["code_sha"] == CODE_SHA and d["declared_at"]
    assert ev.read_result("t2-test-declare-only", ev.default_store()) is None and ev.record_kinds("t2-test-declare-only", ev.default_store()) == []
