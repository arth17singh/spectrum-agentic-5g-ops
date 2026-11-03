"""Offline tests of silos/job_convergence.py: the per-round curve of a federated session, on a tiny data tree and a fake session
whose per-round aggregates are written with silos.nn_core (nothing touches AWS).

    <python> -m pytest silos/tests/test_convergence.py -q        (from the repo root)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from silos import evio, generate as G, job_baselines, job_convergence as jc, job_evaluate, nn_core as nn, scorerun as sr
from silos.tests.test_silos import tiny_cfg

CODE_SHA = "a" * 40
SID = "s1d2ab3c4d5e6f7a8b9c0"  # 21 characters like the real one
FIRST, LAST = 2, 5
PUB = "t2-test-published"
REFS = {"A": "t2-test-lo-a", "B": "t2-test-lo-b", "C": "t2-test-lo-c", "central": "t2-test-central", "fedavg": "t2-test-fedavg"}
EXAMPLE = Path(__file__).resolve().parents[2] / "tools" / "sm" / "examples" / "t2-fl-convergence.json"


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("conv-tree")
    G.generate(tiny_cfg(), root, workers=4, manifest_path=root / "manifest.json")
    return root


@pytest.fixture()
def env(tree, tmp_path, monkeypatch):
    ev = tmp_path / "evidence"
    for k in ("RUN_ID", "EVIDENCE_RUN_ID", "EVIDENCE_S3", "DATA_COPY_S3"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("DATA_DIR", str(tree))
    monkeypatch.setenv("EVIDENCE_DIR", str(ev))
    monkeypatch.setenv("CODE_SHA", CODE_SHA)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "model"))
    return ev


def write_session(root: Path, rounds, clients=("c1d1", "c2e2", "c3f3"), session=SID, ext="safetensor"):
    """Aggregates laid out like the service's: a nested directory per (client, round) and the file `<prefix>-<session>-<member>-<client>-<k>-global`."""
    blobs = {}
    for k in rounds:
        params = nn.init_params(100 + k)
        for c in clients:
            # like the real files: the same tensors, a header whose metadata names the client (so the bytes differ between clients)
            blob = nn.safetensors_dumps(params, {"svc_session_id": session, "svc_client_id": c, "svc_iteration": str(k), "svc_is_local_model": "False"})
            blobs.setdefault(k, blob)  # the first client's copy: the one that gets published
            d = root / f"svc_{session}_{c}_{k}_aggregation_2026-11-03T04:41:34+00:00_x.json" / f"svc-{session}-m0-{c}-{k}-global"
            d.mkdir(parents=True)
            (d / f"svc-{session}-m0-{c}-{k}-global.{ext}").write_bytes(blob)
    return blobs


def setup_references(monkeypatch, tmp_path, final_blob: bytes) -> Path:
    """The published run (job_evaluate on the final aggregate) and the five reference runs, in the local evidence dir."""
    pub = tmp_path / "published.safetensors"
    pub.write_bytes(final_blob)
    monkeypatch.setenv("RUN_ID", PUB)
    assert job_evaluate.main(["--label", "federated", "--split", "disjoint", "--weights", str(pub), "--session-id", SID, "--declare-first"]) == 0
    for s in "ABC":
        monkeypatch.setenv("RUN_ID", REFS[s])
        assert job_baselines.main(["--method", "local", "--silo", s, "--split", "disjoint", "--epochs", "1", "--declare-first"]) == 0
    monkeypatch.setenv("RUN_ID", REFS["central"])
    assert job_baselines.main(["--method", "central", "--split", "disjoint", "--epochs", "1", "--declare-first"]) == 0
    monkeypatch.setenv("RUN_ID", REFS["fedavg"])
    assert job_baselines.main(["--method", "fedavg", "--split", "disjoint", "--declare-first"]) == 0
    return pub


def args(rounds_dir: Path, pub: Path, *extra: str, rounds=f"{FIRST}:{LAST}") -> list[str]:
    return ["--split", "disjoint", "--rounds", rounds, "--rounds-dir", str(rounds_dir), "--published", str(pub), "--published-run", PUB,
            "--local-only", f"A={REFS['A']}", "--local-only", f"B={REFS['B']}", "--local-only", f"C={REFS['C']}",
            "--central", REFS["central"], "--fedavg", REFS["fedavg"], *extra]


@pytest.fixture()
def fx(env, tmp_path, monkeypatch):
    rounds = tmp_path / "rounds"
    blobs = write_session(rounds, range(FIRST, LAST + 1))
    pub = setup_references(monkeypatch, tmp_path, blobs[LAST])
    return rounds, pub, blobs


def run(monkeypatch, argv, run_id="t2-test-conv"):
    monkeypatch.setenv("RUN_ID", run_id)
    rc = jc.main(argv + ["--declare-first"])
    ev = evio.get_evlib()
    store = ev.default_store()
    return rc, ev.read_result(run_id, store), ev.read_records(run_id, "item", store), ev.read_declared(run_id, store)


def test_curve_equals_the_scorer_of_the_published_model_round_by_round(fx, monkeypatch):
    rounds, pub, blobs = fx
    rc, res, items, decl = run(monkeypatch, args(rounds, pub))
    d = res["details"]
    assert rc == 0 and res["passed"] and res["metric"] == "pass_count" and res["value"] == len(items) == decl["n_items"] == 12
    assert d["metric"] == "scaled_mae" and d["lower_is_better"] is True and d["rounds"] == [2, 3, 4, 5]
    assert set(d["curve"]) == {"A", "B", "C", "mean"} and all(len(v) == 4 for v in d["curve"].values())
    assert d["curve"]["mean"] == pytest.approx([np.mean([d["curve"][s][i] for s in "ABC"]) for i in range(4)])
    # every point equals what job_evaluate's scorer gives that round's bytes
    manifest = sr.load_manifest()
    test = nn.read_xy(Path(sr.fetch(["disjoint/shared_test/test.csv"], manifest)) / "disjoint" / "shared_test" / "test.csv")
    for i, k in enumerate(d["rounds"]):
        summ = sr.summarize(test, nn.forward_np(nn.safetensors_loads(blobs[k]), test["x"]))
        for s in "ABC":
            assert d["curve"][s][i] == summ["per_segment"][f"region_{s}"]["mean_score"]
    # one item per (round, silo), with mae and n; passed on every item
    assert {it["item_id"] for it in items} == {f"round{k:02d}:{s}" for k in range(2, 6) for s in "ABC"}
    assert all(it["passed"] and it["extra"]["n"] > 0 and it["extra"]["mae"] == it["score"] and it["extra"]["clients_identical"] for it in items)
    # the check_unit-style recomputation of the declared target
    assert ev_pass_count(items) == 12


def ev_pass_count(items) -> int:
    return evio.get_evlib().summarize_items(items, "pass_count")


def test_final_round_check_and_published_run_are_one_and_the_same(fx, monkeypatch):
    rounds, pub, _ = fx
    _, res, _, _ = run(monkeypatch, args(rounds, pub))
    ev = evio.get_evlib()
    published = ev.read_result(PUB, ev.default_store())
    f = res["details"]["final_round_check"]
    assert f["round"] == 5 and f["matches_published"] is True and f["max_abs_diff"] == 0.0 and f["published_run_id"] == PUB and f["tensors_equal"] and f["file_sha256_equal"]
    assert f["score_max_abs_diff"] <= jc.SCORE_TOL and f["published_file_sha256"] == published["details"]["artifact_sha256"]
    for s in "ABC":
        assert res["details"]["curve"][s][-1] == pytest.approx(published["details"]["per_segment"][f"region_{s}"]["mean_score"], abs=jc.SCORE_TOL)


def test_reference_is_quoted_from_the_evidence_results_not_recomputed(fx, monkeypatch):
    rounds, pub, _ = fx
    _, res, _, _ = run(monkeypatch, args(rounds, pub))
    ev = evio.get_evlib()
    ref = res["details"]["reference"]
    assert set(ref) == {"local_only", "central", "fedavg_local", "no_op"}
    for s in "ABC":
        lo = ev.read_result(REFS[s], ev.default_store())["details"]["per_segment"]
        assert ref["local_only"][s] == lo[f"region_{s}"]["mean_score"]  # the silo's own model on the silo's own segment
        assert ref["local_only"]["cross_segment"][s]["A"] == lo["region_A"]["mean_score"]
    assert ref["local_only"]["run_ids"] == {"A": REFS["A"], "B": REFS["B"], "C": REFS["C"]}
    ce = ev.read_result(REFS["central"], ev.default_store())["details"]["per_segment"]
    assert [ref["central"][s] for s in "ABC"] == [ce[f"region_{s}"]["mean_score"] for s in "ABC"] and ref["central"]["run_ids"] == {"all": REFS["central"]}
    assert ref["central"]["mean"] == pytest.approx(np.mean([ref["central"][s] for s in "ABC"]))
    assert ref["fedavg_local"]["run_ids"]["all"] == REFS["fedavg"] and ref["no_op"]["run_ids"]["all"] == PUB


def test_session_comes_from_files_and_records_and_says_what_it_does_not_know(fx, monkeypatch):
    rounds, pub, _ = fx
    _, res, _, _ = run(monkeypatch, args(rounds, pub))
    s = res["details"]["session"]
    assert s["session_id_short"] == SID[:8] and s["rounds_completed"] == 4 and s["clients"] == 3
    assert s["secure_aggregation"] is None and "not asserted" in s["sources"]["secure_aggregation"]
    assert s["dp"] == "none"


def test_a_last_round_that_is_not_the_published_model_fails_the_declaration(env, tmp_path, monkeypatch):
    rounds = tmp_path / "rounds"
    blobs = write_session(rounds, range(FIRST, LAST + 1))
    pub = setup_references(monkeypatch, tmp_path, nn.safetensors_dumps(nn.init_params(7)))  # a different model was published
    rc, res, items, _ = run(monkeypatch, args(rounds, pub))
    f = res["details"]["final_round_check"]
    assert rc == 1 and not res["passed"] and f["matches_published"] is False and f["max_abs_diff"] > 0 and not f["tensors_equal"]
    bad = {it["item_id"] for it in items if not it["passed"]}
    assert bad == {f"round05:{s}" for s in "ABC"} and res["value"] == 9


def test_clients_whose_copies_differ_fail_that_round(fx, monkeypatch):
    rounds, pub, _ = fx
    victim = sorted(rounds.rglob("*-3-global.safetensor"))[1]
    victim.write_bytes(nn.safetensors_dumps(nn.init_params(55), {"svc_iteration": "3"}))
    rc, res, items, _ = run(monkeypatch, args(rounds, pub))
    assert rc == 1 and {it["item_id"] for it in items if not it["passed"]} == {f"round03:{s}" for s in "ABC"}
    assert [r["clients_identical"] for r in res["details"]["per_round"]] == [True, False, True, True]
    # the untouched rounds' copies differ in bytes (client metadata) yet hold identical tensors
    assert [r["files_byte_identical"] for r in res["details"]["per_round"]] == [False] * 4


def test_missing_rounds_foreign_sessions_and_wrong_files_are_refused(fx, monkeypatch, tmp_path):
    rounds, pub, _ = fx
    monkeypatch.setenv("RUN_ID", "t2-test-refuse")
    with pytest.raises(SystemExit, match="missing rounds"):
        jc.main(args(rounds, pub, rounds="2:6", ) + ["--dry-run"])
    other = tmp_path / "rounds2"
    write_session(other, [2], session="zz9y8x7w6v5u4t3s2r1q0")
    write_session(other, [3])
    with pytest.raises(SystemExit, match="2 sessions"):
        jc.main(args(other, pub, rounds="2:3") + ["--dry-run"])
    wrong = tmp_path / "wrong.safetensors"
    wrong.write_bytes(nn.safetensors_dumps(nn.init_params(9)))
    with pytest.raises(SystemExit, match="wrong file"):
        jc.main(args(rounds, wrong) + ["--dry-run"])
    mislabeled = tmp_path / "rounds3"
    write_session(mislabeled, [2])
    f = sorted(mislabeled.rglob("*-2-global.safetensor"))[0]
    f.write_bytes(nn.safetensors_dumps(nn.init_params(1), {"svc_iteration": "9"}))
    with pytest.raises(SystemExit, match="metadata says iteration 9"):
        jc.main(args(mislabeled, pub, rounds="2:2") + ["--dry-run"])
    loc = tmp_path / "rounds4"
    write_session(loc, [2])
    g = sorted(loc.rglob("*-2-global.safetensor"))[0]
    g.write_bytes(nn.safetensors_dumps(nn.init_params(1), {"svc_is_local_model": "True"}))
    with pytest.raises(SystemExit, match="LOCAL model"):
        jc.main(args(loc, pub, rounds="2:2") + ["--dry-run"])
    with pytest.raises(SystemExit, match="--rounds"):
        jc.main(args(rounds, pub, rounds="5:2") + ["--dry-run"])


def test_discover_matches_names_by_pattern_in_any_nesting_and_extension(tmp_path):
    write_session(tmp_path / "x", [2, 3], clients=("aa1", "bb2"), ext="safetensors")
    (tmp_path / "x" / "FINAL_PATH").write_text("/somewhere")
    (tmp_path / "x" / "initial-weights.safetensors").write_bytes(b"x")
    found = jc.discover(tmp_path / "x")
    assert sorted(found["by_round"]) == [2, 3] and found["clients"] == ["aa1", "bb2"] and found["session_id"] == SID and len(found["by_round"][2]) == 2
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no '"):
        jc.discover(tmp_path / "empty")


def test_a_dry_run_writes_nothing(fx, monkeypatch, tmp_path):
    rounds, pub, _ = fx
    monkeypatch.setenv("RUN_ID", "t2-test-conv-dry")
    assert jc.main(args(rounds, pub) + ["--dry-run"]) == 0
    ev = evio.get_evlib()
    assert ev.read_result("t2-test-conv-dry", ev.default_store()) is None and ev.record_kinds("t2-test-conv-dry", ev.default_store()) == []


def test_committed_example_is_what_the_job_declares_and_names_no_service_or_account(env, monkeypatch, capsys):
    ex = json.loads(EXAMPLE.read_text())
    det = ex["twin"]["config"]["details"]
    argv = ["--split", "disjoint", "--rounds", f"{det['rounds'][0]}:{det['rounds'][1]}", "--published-run", det["published_run"],
            "--central", det["reference_runs"]["central"], "--fedavg", det["reference_runs"]["fedavg_local"], "--print-runner-declaration"]
    for s, rid in det["reference_runs"]["local_only"].items():
        argv += ["--local-only", f"{s}={rid}"]
    assert jc.main(argv) == 0
    got = json.loads(capsys.readouterr().out)
    # the example carries the test-set hash of the AWS data copy; here the tiny tree has its own
    assert {**got, "test_set_sha256": ex["test_set_sha256"]} == ex
    assert ex["unit"] == "t2.fl_convergence.disjoint" and ex["n_items"] == ex["target"]["value"] == 60 and ex["target"]["op"] == "=="
    assert ex["target"]["metric"] == "pass_count" and ex["track"] == "track-2" and det["rounds"] == [2, 21]
    assert not {"run_id", "code_sha", "region"} & set(ex)
    text = (EXAMPLE.read_text() + Path(jc.__file__).read_text()).lower()
    assert ("cross" + "trainer") not in text and ("4835" + "98626742") not in text
