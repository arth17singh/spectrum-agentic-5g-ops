"""remediation-v2: the config-history rule, the baseline contract, the held-out guard, the frozen v2 candidates,
the deadzone rule, and the v2 gate's evaluation profile."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from t1.rl.policy import (CELL_SHARE_DIM, CONFIG_KEYS, DELTA_KEYS, KPI_DELTA_DIM, SERVES_UES_DIM, SHARE_KEYS,
                          Policy, observation_for)
from t1.rl.remediation import guard
from t1.rl.remediation import v2 as V
from t3.evals import incidents as inc
from t3.evals.graders import load_targets
from t3.evals.twin_backend import single_thread

ROOT = Path(__file__).resolve().parents[3]
single_thread()


@pytest.fixture(scope="module")
def targets():
    return load_targets()


@pytest.fixture(scope="module")
def kept(targets):
    """The first kept v2 train-role candidate, with its samples and baseline (twin replay)."""
    for s in V.load_candidates()[:40]:
        if s["role"] != "train":
            continue
        r = V.incident_samples(s, targets)
        if r["kept"]:
            return s, r
    pytest.skip("no kept train candidate among the first 40")


# --------------------------------------------------------------------------- the config-history rule

def test_baseline_configuration_never_reaches_the_observation(kept):
    _, r = kept
    state, base = r["samples"][0]["state"], r["baseline"]
    o1 = observation_for(state, "masked_delta", baseline=base)
    rng = np.random.default_rng(7)
    other = dict(base, power_dbm=list(rng.uniform(10, 40, 4)), tilt_deg=list(rng.uniform(-5, 15, 4)))
    o2 = observation_for(state, "masked_delta", baseline=other)
    assert o1.keys() == o2.keys()
    for k in o1:
        assert np.array_equal(o1[k], o2[k]), f"{k} depends on the baseline's configuration"


def test_no_observation_key_is_a_config_difference(kept):
    _, r = kept
    o = observation_for(r["samples"][0]["state"], "masked_delta", baseline=r["baseline"])
    assert not set(CONFIG_KEYS) & set(DELTA_KEYS)
    assert not {"power_dbm", "tilt_deg"} & set(SHARE_KEYS)
    assert o["kpi_delta"].shape == (KPI_DELTA_DIM,)
    assert o["cell_share"].shape == (CELL_SHARE_DIM,)
    assert o["serves_ues"].shape == (SERVES_UES_DIM,)
    assert set(o) == set(observation_for(r["samples"][0]["state"], "masked")) | {"kpi_delta", "cell_share",
                                                                                   "serves_ues"}


def test_baseline_kpis_do_reach_the_observation(kept):
    _, r = kept
    state, base = r["samples"][0]["state"], r["baseline"]
    o1 = observation_for(state, "masked_delta", baseline=base)
    moved = dict(base, sinr_p10_db=[x + 3.0 for x in base["sinr_p10_db"]],
                 coverage_share=[max(0.0, x - 0.2) for x in base["coverage_share"]])
    o2 = observation_for(state, "masked_delta", baseline=moved)
    assert not np.array_equal(o1["kpi_delta"], o2["kpi_delta"])


def test_masked_delta_needs_a_baseline_and_the_shares(kept):
    _, r = kept
    with pytest.raises(ValueError, match="baseline"):
        observation_for(r["samples"][0]["state"], "masked_delta")
    no_shares = {k: v for k, v in r["samples"][0]["state"].items() if k not in SHARE_KEYS}
    with pytest.raises(ValueError, match="per-cell counters"):
        observation_for(no_shares, "masked_delta", baseline=r["baseline"])
    model = V.new_model()
    pol = Policy(model, V.meta_for("vtest"))
    with pytest.raises(ValueError, match="baseline"):
        pol.propose(r["samples"][0]["state"])
    assert isinstance(pol.propose(r["samples"][0]["state"], baseline=r["baseline"]), list)


# --------------------------------------------------------------------------- the no-UE rule

# Frozen v2 candidates whose replays reach a cell with zero UEs (found by scanning candidates-v2, dev data only):
# index 215 (seed 3000444): cell 3 serves no UEs at the baseline (step 1) and after the fault (step 2);
# index 131 (seed 3000264): the fault leaves cell 0 with no UEs (step 2) while it served UEs at the baseline.
ZERO_UE_CASES = ((215, 3), (131, 0))


def _no_nan(o: dict) -> bool:
    return all(np.all(np.isfinite(np.asarray(v, dtype=np.float64))) for v in o.values())


def _shapes_ok(o: dict) -> bool:
    return (o["kpi_delta"].shape == (KPI_DELTA_DIM,) and o["cell_share"].shape == (CELL_SHARE_DIM,)
            and o["serves_ues"].shape == (SERVES_UES_DIM,))


@pytest.mark.parametrize("index,cell", ZERO_UE_CASES)
def test_a_cell_without_ues_gets_a_defined_fixed_length_observation(targets, index, cell):
    spec = V.load_candidates()[index]
    tw, now, base = V.run_script(spec, targets, with_fault=True)
    assert now["ue_count"][cell] == 0 and now["coverage_share"][cell] is None and now["sla_share"][cell] is None
    o = observation_for(now, "masked_delta", baseline=base)
    assert _shapes_ok(o) and _no_nan(o)
    n = 4
    serve_now, serve_base = o["serves_ues"][:n], o["serves_ues"][n:]
    assert serve_now[cell] == 0.0
    assert serve_base[cell] == (1.0 if base["ue_count"][cell] > 0 else 0.0)
    # every UE-defined delta of that cell is 0 (rsrp/sinr mean and p10: kpi_delta[4..19]; shares: [22..29])
    d = o["kpi_delta"]
    for k in range(1, 5):
        assert d[k * n + cell] == 0.0
    assert d[22 + cell] == 0.0 and d[26 + cell] == 0.0
    # its current shares are 0.0 before the 2x-1 mapping
    assert o["cell_share"][cell] == -1.0 and o["cell_share"][n + cell] == -1.0
    # the ue-share delta stays defined: a cell that lost its users shows it
    if base["ue_count"][cell] > 0:
        assert d[cell] < 0.0


def test_non_finite_values_are_undefined_not_poison(kept):
    _, r = kept
    state, base = dict(r["samples"][0]["state"]), dict(r["baseline"])
    clean = observation_for(state, "masked_delta", baseline=base)
    state["rsrp_p10_dbm"] = [float("nan")] + list(state["rsrp_p10_dbm"][1:])
    state["coverage_share"] = [float("nan")] + list(state["coverage_share"][1:])
    base["sinr_mean_db"] = list(base["sinr_mean_db"][:2]) + [float("inf"), base["sinr_mean_db"][3]]
    base["sla_share"] = list(base["sla_share"][:3]) + [None]
    state["sla_ratio"] = float("nan")
    o = observation_for(state, "masked_delta", baseline=base)
    assert _shapes_ok(o) and _no_nan(o)
    n = 4
    d = o["kpi_delta"]
    assert d[2 * n + 0] == 0.0          # rsrp_p10 delta of cell 0 (NaN now)
    assert d[22 + 0] == 0.0             # coverage-share delta of cell 0 (NaN now)
    assert d[3 * n + 2] == 0.0          # sinr_mean delta of cell 2 (inf at the baseline)
    assert d[26 + 3] == 0.0             # sla-share delta of cell 3 (None at the baseline)
    assert d[21] == 0.0                 # the network sla_ratio delta (NaN now)
    assert o["cell_share"][0] == -1.0   # an undefined current share is 0.0 before 2x-1
    untouched = [i for i in range(KPI_DELTA_DIM) if i not in (2 * n, 22, 3 * n + 2, 29, 21)]
    assert np.array_equal(d[untouched], clean["kpi_delta"][untouched])
    nan_ue = dict(r["samples"][0]["state"], ue_count=[float("nan")] + list(r["samples"][0]["state"]["ue_count"][1:]))
    o2 = observation_for(nan_ue, "masked_delta", baseline=r["baseline"])
    assert _shapes_ok(o2) and _no_nan(o2) and o2["serves_ues"][0] == 0.0


@pytest.mark.parametrize("index,cell", ZERO_UE_CASES)
def test_training_and_inference_build_the_identical_vector(targets, index, cell):
    """The training path (jsonable rows -> observation_for) and the inference path (the live twin's state with
    shares, the baseline rebuilt from the action log, Policy.observation) agree exactly, zero-UE cell included."""
    spec = V.load_candidates()[index]
    tw, now, base = V.run_script(spec, targets, with_fault=True)
    train_obs = observation_for(V.jsonable(now), "masked_delta", baseline=V.jsonable(base))
    built, _ = inc.build(spec, targets)   # None when the keep rule drops it; begin() needs scenario, seed, preset
    op = V.BaselineAware(Policy(V.new_model(), V.meta_for("vtest")))
    op.begin(built or spec, {"twin": tw, "targets": targets, "run_id": "test"})
    infer_obs = op.policy.observation(V.with_shares(tw.state, tw), baseline=op._baseline)
    assert set(train_obs) == set(infer_obs)
    for k in train_obs:
        assert np.array_equal(np.asarray(train_obs[k]), np.asarray(infer_obs[k])), k
    assert isinstance(op.act(tw.state, []), (list, type(None)))


def test_v2_sample_generation_survives_a_zero_ue_baseline(targets):
    spec = V.load_candidates()[215]
    r = V.incident_samples(spec, targets)
    if r["kept"]:
        assert r["baseline"]["coverage_share"][3] is None
        a = V.to_arrays([r])
        assert np.all(np.isfinite(a["target"]))
        assert all(np.all(np.isfinite(v.astype(np.float64))) for v in a["obs"].values())
    else:
        assert r["reason"] and "TypeError" not in r["reason"]


# --------------------------------------------------------------------------- the baseline contract

def test_baseline_is_the_commissioned_state_and_the_log_replay_agrees(kept, targets):
    from t3.evals.run import run_incident

    spec, r = kept
    assert V.baseline_step(targets) == targets["episode"]["fault_step"] - 1
    tw, _, base = V.run_script(spec, targets, with_fault=True)
    assert V.jsonable(base) == r["baseline"]
    built, _ = inc.build(spec, targets)
    from_log = V.baseline_from_log(built, tw.action_log, targets)
    assert V.jsonable(from_log) == r["baseline"]
    op = V.BaselineAware(Policy(V.new_model(), V.meta_for("vtest")))
    it = run_incident(built, op, targets)
    assert V.jsonable(op._baseline) == r["baseline"]
    assert len(it["extra"]["change_texts"]) == targets["episode"]["operator_steps"]


def test_samples_carry_residual_labels_and_the_healthy_run_shares_the_baseline(kept):
    _, r = kept
    kinds = [s["kind"] for s in r["samples"]]
    assert kinds == ["fault@2", "fault@3", "fault@4", "fixed@3", "fixed@4", "half@3", "healthy@2", "healthy@3",
                     "healthy@4"]
    lab = {s["kind"]: np.asarray(s["label"]) for s in r["samples"]}
    assert np.allclose(lab["fault@2"], r["inverse"])
    assert not np.any(lab["fixed@3"]) and not np.any(lab["healthy@2"])
    inverse = np.asarray(r["inverse"])
    half = np.trunc(np.round(inverse / 2 / 0.1, 6)) * 0.1
    assert np.allclose(lab["half@3"], inverse - half, atol=1e-9)   # the residual after half the fix
    for s in r["samples"]:
        assert set(SHARE_KEYS) <= set(s["state"])


# --------------------------------------------------------------------------- the held-out guard

def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=ROOT, capture_output=True, text=True,
                          env={"PYTHONPATH": f"{ROOT}:{ROOT / 'contracts'}", "PATH": "/usr/bin:/bin"}, timeout=300)


def test_guard_paths():
    assert guard.is_heldout(ROOT / "t1/rl/remediation/suite/heldout-v2.jsonl")
    assert guard.is_heldout("/x/t1/rl/remediation/suite/heldout-v9-meta.json")
    assert not guard.is_heldout(ROOT / "t1/rl/remediation/candidates-v2.jsonl")
    assert not guard.is_heldout("/x/t1/rl/remediation/heldout.py")
    assert not guard.is_heldout("/x/other/suite/heldout-v2.jsonl")


def test_guard_refuses_the_gate_and_allows_the_rest(tmp_path):
    suite = tmp_path / "t1" / "rl" / "remediation" / "suite"
    suite.mkdir(parents=True)
    (suite / "heldout-fake.jsonl").write_text("{}\n")
    (suite / "other.jsonl").write_text("{}\n")
    p = _run(f"""
        from pathlib import Path
        from t1.rl.remediation import guard
        guard.install("test")
        print(Path({str(suite / 'other.jsonl')!r}).read_text().strip())
        try:
            open({str(suite / 'heldout-fake.jsonl')!r})
        except guard.HeldoutReadRefused as e:
            print("REFUSED", type(e).__name__)
        try:
            import numpy as np
            np.loadtxt({str(suite / 'heldout-fake.jsonl')!r})
        except PermissionError:
            print("REFUSED numpy")
    """)
    assert p.returncode == 0, p.stderr[-2000:]
    assert p.stdout.split() == ["{}", "REFUSED", "HeldoutReadRefused", "REFUSED", "numpy"]


def test_training_entry_point_installs_the_guard(tmp_path):
    suite = tmp_path / "t1" / "rl" / "remediation" / "suite"
    suite.mkdir(parents=True)
    (suite / "heldout-fake.jsonl").write_text("{}\n")
    p = _run(f"""
        import contextlib, io
        from t1.rl.remediation import guard, job_train_v2
        with contextlib.redirect_stdout(io.StringIO()):
            job_train_v2.main(["--print-declaration"])
        assert guard.installed()
        try:
            open({str(suite / 'heldout-fake.jsonl')!r})
            print("OPENED")
        except guard.HeldoutReadRefused:
            print("REFUSED")
    """)
    assert p.returncode == 0, p.stderr[-2000:]
    assert p.stdout.strip() == "REFUSED"


# --------------------------------------------------------------------------- the frozen v2 candidates

def test_v2_candidates_are_the_generator_output_on_fresh_clean_dev_seeds():
    from t3.evals.lessons import HELDOUT_SEED_BLOCK

    specs = V.load_candidates()   # check=True: regenerated byte for byte
    meta = json.loads(V.META_PATH.read_text())
    seeds = [int(s["seed"]) for s in specs]
    assert len(specs) == V.N_CANDIDATES == meta["n_candidates"]
    assert min(seeds) >= V.SEED_BASE
    assert not [s for s in seeds if HELDOUT_SEED_BLOCK[0] <= s <= HELDOUT_SEED_BLOCK[1]]
    assert all(inc.split_of(s) == "dev" for s in seeds)
    assert meta["exclusion"]["clean"] is True
    assert all(v == 0 for k, v in meta["spec_hash_overlap"].items() if k != "heldout_gate")
    assert meta["spec_hash_overlap"]["heldout_gate"]["seeds_inside"] == 0
    from t1.rl.remediation import data as D

    v1 = {int(s["seed"]) for s in D.load_candidates(check=False)}
    assert not v1 & set(seeds)


# --------------------------------------------------------------------------- the deadzone rule

def _rows(spec: dict[str, list[tuple[bool, bool, bool]]]) -> list[dict]:
    keys = list(spec)
    n = len(spec[keys[0]])
    return [{"by_dz": {k: {"percell_recovered": spec[k][i][0], "success": spec[k][i][1],
                           "healthy_false_change": spec[k][i][2]} for k in keys}} for i in range(n)]


def test_deadzone_rule_adopts_one_only_if_it_cuts_false_changes_at_no_cost():
    t, f = True, False
    rows = _rows({"0.0/0.0": [(t, t, t), (t, t, t), (f, t, f), (f, f, f)],
                  "0.5/0.5": [(t, t, f), (t, t, t), (f, t, f), (f, f, f)],
                  "1.0/1.0": [(t, t, f), (t, t, f), (f, t, f), (f, f, f)],
                  "1.5/1.5": [(f, t, f), (t, t, f), (f, t, f), (f, f, f)]})
    assert V.choose_deadzone(rows)["chosen"] == [1.0, 1.0]   # 1.5 costs per-cell recovery
    same = _rows({k: [(t, t, f), (f, f, f)] for k in ("0.0/0.0", "0.5/0.5", "1.0/1.0", "1.5/1.5")})
    assert V.choose_deadzone(same)["chosen"] == [0.0, 0.0]   # nothing to cut: no deadzone


# --------------------------------------------------------------------------- the v2 gate's evaluation profile

def test_v2_profile_operators_items_and_declaration(tmp_path):
    from t1.rl.remediation import evaluate as E

    prof = E.PROFILES["v2"]
    tg = E.declared(prof)
    v3 = E.declared(E.PROFILES["v1"])
    assert tg["unit"] == "t3.evals.remediation-policy-v2"
    assert [(c["name"], c["score_field"], c["op"], c["threshold"]) for c in tg["primary"]["components"]] == \
        [(c["name"], c["score_field"], c["op"], c["threshold"]) for c in v3["primary"]["components"]]
    assert tg["heldout_use"]["use"] == 2
    specs = E.operator_specs(E.V1_PATH, E.PPO_PATH, Path("/x/remediation_v2.zip"), prof)
    assert list(specs) == ["v2", "v1", "ppo_sm_50k", "oracle_inverse"]
    assert specs["v2"]["kind"] == "ppo_delta"
    assert list(E.operator_specs(E.V1_PATH, E.PPO_PATH)) == list(E.OPERATORS)   # v1's gate unchanged

    def op(rec, pc, diag):
        return {"recovery": rec, "percell_recovered": pc, "diagnosis_hit": diag, "change_texts": [""] * 3,
                "latency_ms": 1.0}

    row = {"suite": "heldout", "incident": {"incident_id": "inc-x", "seed": 1},
           "operators": {"v2": op(0.9, True, True), "v1": op(0.5, False, True), "ppo_sm_50k": op(1.0, False, False),
                         "oracle_inverse": op(1.0, True, True)}}
    it = E.item_for(row, prof)
    assert it["score"] == 1.0 and it["passed"] is True
    assert it["extra"]["diff"] == {"percell_recovered": 1.0, "network_recovery": pytest.approx(-0.1),
                                   "diagnosis_hit": 1.0}
    assert it["extra"]["diff_vs_v1"] == {"percell_recovered": 1.0, "network_recovery": pytest.approx(0.4),
                                         "diagnosis_hit": 0.0}
    assert "diff_vs_v1" not in E.item_for({**row, "operators": {**row["operators"], "config_revert": op(0, 0, 0),
                                                                "rule_baseline": op(0, 0, 0)}})["extra"]
