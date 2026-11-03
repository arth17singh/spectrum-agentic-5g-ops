import hashlib
import json

import pytest

from t3.evals import incidents as I
from t3.evals.graders import load_targets
from t3.evals.twin_backend import ReplayTwin, graded

T = load_targets()
FROZEN_SHA = "485b3bde1dc974a07e5950115e1fc1a8fa0dd2cb6cfc88e377a4ac7a01588a63"


def test_real_twin_is_used_and_the_standin_refuses_to_load_beside_it():
    import importlib.util

    from t3.evals import twin_backend

    if importlib.util.find_spec("twin.replay") is None:
        pytest.skip("W2's twin is not on the path")
    assert twin_backend.BACKEND == "maveric-inprocess"
    assert twin_backend.ReplayTwin.__module__ == "twin.replay"
    with pytest.raises(ImportError, match="real twin"):
        import t3.evals.twin_standin  # noqa: F401
    meta = json.loads(I.META_PATH.read_text())
    assert meta["twin_backend"] == "maveric-inprocess" and meta["test_sha256"] == FROZEN_SHA
    assert "20f9013399da360b26214e11de15c9f9edbb8dcb9bcc4cf1d89fb7e194d3dd12" in {s["test_sha256"] for s in meta["supersedes"]}


def test_frozen_hash_is_stable_and_recorded():
    data = I.TEST_PATH.read_bytes()
    assert hashlib.sha256(data).hexdigest() == FROZEN_SHA
    assert I.HASH_PATH.read_text().split()[0] == FROZEN_SHA
    incs, digest = I.load_test()
    assert digest == FROZEN_SHA
    assert 20 <= len(incs) <= 40
    assert len({i["incident_id"] for i in incs}) == len(incs)


def test_candidates_and_split_are_pure_functions_of_the_index():
    assert [I.candidate(i) for i in range(20)] == [I.candidate(i) for i in range(20)]
    assert [I.split_of(s) for s in range(4000, 4100)] == [I.split_of(s) for s in range(4000, 4100)]


def test_no_seed_in_both_splits_and_every_test_incident_is_test_split():
    test, _ = I.load_test()
    dev = [json.loads(x) for x in I.DEV_PATH.read_text().splitlines() if x.strip()]
    assert not ({i["seed"] for i in test} & {i["seed"] for i in dev})
    assert all(I.split_of(i["seed"]) == "test" for i in test)
    assert all(I.split_of(i["seed"]) == "dev" for i in dev)


def test_faults_stay_inside_the_sdk_safety_limits_and_break_the_network():
    from ran_rl.env.config import SafetyBounds

    b = SafetyBounds()
    test, _ = I.load_test()
    for inc in test:
        for f in inc["fault"]:
            assert b.power_delta_min_db <= f["d_power_db"] <= b.power_delta_max_db
            assert abs(f["d_tilt_deg"]) <= b.tilt_delta_max_deg
        # the filter let the fault through unchanged
        n = 4
        for f in inc["fault"]:
            assert inc["fault_filtered"]["power_delta_db"][f["cell"] % n] == f["d_power_db"]
            assert inc["fault_filtered"]["tilt_delta_deg"][f["cell"] % n] == f["d_tilt_deg"]
        h, u = inc["healthy_kpis_by_step"][-1], inc["unfixed_kpis_by_step"][-1]
        assert any(h[k] - u[k] > T["tolerance"][k] for k in T["graded_kpis"]), inc["incident_id"]
        assert inc["preset"] in ("official", "viavi-like")


def test_both_presets_and_several_fault_kinds_are_present():
    test, _ = I.load_test()
    assert {i["preset"] for i in test} == {"official", "viavi-like"}
    assert len({i["template"] for i in test}) >= 5
    assert any(len(i["fault_cells"]) == 2 for i in test)


def test_viavi_like_preset_is_stationary():
    test, _ = I.load_test()
    vv = next(i for i in test if i["preset"] == "viavi-like")
    hs = vv["healthy_kpis_by_step"][1:]  # after commissioning nothing moves
    assert all(h == hs[0] for h in hs)


def test_masked_tier_hides_the_fault_in_a_heterogeneous_configuration():
    test, _ = I.load_test()
    assert {i["tier"] for i in test} == {"visible", "masked"}
    for inc in test:
        if inc["tier"] == "visible":
            assert inc["commissioning"] == []
        else:
            assert inc["commissioning"] == I.commissioning(inc["seed"], "masked")
    assert any(len(i["commissioning"]) >= 3 for i in test if i["tier"] == "masked")


def _frozen_platform_is_here() -> bool:
    import platform

    meta = json.loads(I.META_PATH.read_text())
    from t3.evals.twin_backend import BACKEND

    return (meta.get("platform") == f"{platform.system()}-{platform.machine()}" and meta.get("twin_backend") == BACKEND
            and meta.get("suite") == I.SUITE_VERSION)


def test_replay_is_deterministic_on_one_incident():
    from t3.evals.tests._local import local_suite

    inc = local_suite()[1][0]
    runs = []
    for _ in range(2):
        t = ReplayTwin(inc["scenario"], inc["seed"], inc["preset"])
        t.reset()
        sc = I.script(inc, T, with_fault=True)
        runs.append([graded(t.apply(sc[s - 1] if s <= len(sc) else [])) for s in range(1, inc["grade_step"] + 1)])
    assert runs[0] == runs[1]
    for k in T["graded_kpis"]:
        assert runs[0][-1][k] == pytest.approx(inc["unfixed_kpis_by_step"][-1][k], rel=1e-6, abs=1e-8)


def test_regenerating_the_suite_reproduces_the_frozen_bytes():
    if not _frozen_platform_is_here():
        pytest.skip("the suite of record was frozen on another platform/backend/version (see suite/incidents-meta.json); "
                    "the float32 GP twin replays it to 1e-6 only there: run this check in a runner job")
    out = I.generate()
    assert I.to_bytes(out["test"]) == I.TEST_PATH.read_bytes()


def test_candidates_file_is_the_generators_input():
    assert I.CANDIDATES_PATH.read_bytes() == I.candidates_bytes()
    lines = I.CANDIDATES_PATH.read_text().splitlines()
    assert len(lines) == I.N_CANDIDATES
    assert json.loads(lines[0]) == I.candidate(0)


def test_freeze_writes_only_into_out_and_records_one_item_per_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(I, "N_CANDIDATES", 6)
    monkeypatch.setattr(I, "CANDIDATES_PATH", tmp_path / "absent.jsonl")
    out_dir, ev_dir = tmp_path / "out" / "suite", tmp_path / "ev"
    before = sorted(p.name for p in I.SUITE_DIR.iterdir())
    stamps = {p.name: p.stat().st_mtime_ns for p in I.SUITE_DIR.iterdir()}
    rc = I.main(["--out", str(out_dir), "--evidence-dir", str(ev_dir), "--run-id", "t-freeze"])
    assert rc == 0
    assert sorted(p.name for p in I.SUITE_DIR.iterdir()) == before  # the code tree is untouched
    assert {p.name: p.stat().st_mtime_ns for p in I.SUITE_DIR.iterdir()} == stamps
    assert sorted(p.name for p in out_dir.iterdir()) == sorted(
        [I.TEST_PATH.name, I.DEV_PATH.name, I.HASH_PATH.name, I.META_PATH.name])
    test_bytes = (out_dir / I.TEST_PATH.name).read_bytes()
    assert (out_dir / I.HASH_PATH.name).read_text().split()[0] == hashlib.sha256(test_bytes).hexdigest()
    meta = json.loads((out_dir / I.META_PATH.name).read_text())
    assert meta["suite"] == I.SUITE_VERSION and meta["frozen_at"].endswith("Z") and meta["candidates"] == 6
    items = [json.loads(x) for x in (ev_dir / "records" / "t-freeze" / "item.jsonl").read_text().splitlines()]
    assert len(items) == 6 and len({i["item_id"] for i in items}) == 6
    for it, i in zip(items, range(6)):
        spec = I.candidate(i)
        assert it["input_sha256"] == hashlib.sha256(I.canonical_line(spec).encode()).hexdigest()
        assert len(it["output_sha256"]) == 64
    kept_ids = {i["extra"]["incident_id"] for i in items if i["extra"]["kept"]}
    frozen_ids = {json.loads(x)["incident_id"] for x in test_bytes.decode().splitlines()}
    frozen_ids |= {json.loads(x)["incident_id"] for x in (out_dir / I.DEV_PATH.name).read_text().splitlines()}
    assert kept_ids == frozen_ids
    res = json.loads((ev_dir / "results" / "t-freeze.json").read_text())
    assert res["metric"] == "n_items" and res["value"] == 6 and res["n_items"] == 6 and res["passed"] is True
    assert res["target"] == {"metric": "n_items", "op": "==", "value": 6, "unit": "count"}
    assert res["details"]["test_sha256"] == hashlib.sha256(test_bytes).hexdigest()


def test_replay_mismatch_names_the_kpi_and_the_difference():
    from t3.evals import run as R
    from t3.evals.operators import NoAction
    from t3.evals.tests._local import local_suite

    inc = json.loads(json.dumps(local_suite()[1][0]))
    inc["post_fault_kpis"]["coverage_ratio"] += 0.005
    with pytest.raises(R.ReplayMismatch) as e:
        R.run_incident(inc, NoAction(), T)
    msg = str(e.value)
    assert inc["incident_id"] in msg and "coverage_ratio: replayed" in msg and "diff -5.000e-03" in msg
    assert "sla_ratio" not in msg


def test_evshim_takes_the_pinned_sha_and_never_asks_git(monkeypatch):
    from t3.evals import evshim

    sha = "0123456789abcdef0123456789abcdef01234567"
    monkeypatch.setenv("CODE_SHA", sha)

    def boom(*a, **k):
        raise AssertionError("git was called")

    monkeypatch.setattr(evshim.subprocess, "run", boom)
    if evshim._ev is not None:
        monkeypatch.setattr(evshim._ev, "code_sha", boom)
        monkeypatch.setattr(evshim._ev, "code_dirty", boom)
    assert evshim.code_sha() == sha and evshim.code_dirty() is False
    monkeypatch.setenv("CODE_SHA", "not-a-sha")
    with pytest.raises(ValueError):
        evshim.code_sha()
