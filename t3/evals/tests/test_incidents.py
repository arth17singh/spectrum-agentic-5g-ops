import hashlib
import json

import pytest

from t3.evals import incidents as I
from t3.evals.graders import load_targets
from t3.evals.twin_backend import ReplayTwin, graded

T = load_targets()
FROZEN_SHA = "c0846996a08357836efcc7af1b7ab285d63b8c3622a846a580661d7bef388f0a"


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
    hs = vv["healthy_kpis_by_step"]
    assert all(h == hs[0] for h in hs)


def test_replay_is_deterministic_on_one_incident():
    inc = I.load_test()[0][0]
    runs = []
    for _ in range(2):
        t = ReplayTwin(inc["scenario"], inc["seed"], inc["preset"])
        t.reset()
        runs.append([graded(t.apply(inc["fault"] if s == 1 else [])) for s in range(1, 5)])
    assert runs[0] == runs[1]
    for k in T["graded_kpis"]:
        assert runs[0][-1][k] == pytest.approx(inc["unfixed_kpis_by_step"][-1][k], rel=1e-6, abs=1e-8)


def test_regenerating_the_suite_reproduces_the_frozen_bytes():
    out = I.generate()
    assert I.to_bytes(out["test"]) == I.TEST_PATH.read_bytes()
