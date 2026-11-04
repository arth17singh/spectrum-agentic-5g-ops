"""The confirmation unit and its frozen set (fakes everywhere except two real twin rebuilds)."""
from __future__ import annotations

import json
import platform

import pytest

from t3.evals import confirm as C
from t3.evals import incidents as I
from t3.evals import lessons as L
from t3.evals import self_improve as S
from t3.evals.evshim import Evidence
from t3.evals.graders import load_targets

STATE = {"power_dbm": [40.0, 40.0, 40.0, 40.0], "tilt_deg": [4.0, 4.0, 9.0, 4.0]}


def inc(seed=5001, iid=None, split="confirm", pre=(0.95, 0.90, 1.0), post=(0.80, 0.70, -1.0)):
    return {"incident_id": iid or f"inc-x-{seed}", "seed": seed, "split": split, "preset": "official",
            "scenario": "nominal", "tier": "masked", "template": "tilt+5", "commissioning": [],
            "fault": [{"cell": 2, "d_power_db": 0.0, "d_tilt_deg": 5.0}],
            "pre_fault_kpis": {"coverage_ratio": pre[0], "sla_ratio": pre[1], "sinr_p10_db": pre[2]},
            "post_fault_kpis": {"coverage_ratio": post[0], "sla_ratio": post[1], "sinr_p10_db": post[2]}}


def ok_verification(change="cell=2;d_power_db=+0.0;d_tilt_deg=-5.0"):
    return {"verifier": L.VERIFIER, "backend": "fake", "recovered": True, "recovery": 1.0, "harmful": False,
            "steps_to_fix": 1, "clamped_rules": [], "change_texts": [change, "", ""], "state_before_fix": STATE}


def fake_verify(incident, proposals, targets=None):
    fixed = (list(proposals) + [""])[0] == "FIX"
    return {"verifier": "fake", "backend": "fake", "recovered": fixed, "recovery": 1.0 if fixed else 0.2,
            "harmful": False, "steps_to_fix": 1 if fixed else None, "clamped_rules": [],
            "change_texts": ["cell=2;d_power_db=+0.0;d_tilt_deg=-5.0" if fixed else "", "", ""],
            "state_before_fix": STATE}


def dev_seed():
    s = 5000
    while I.split_of(s) != "dev":
        s += 1
    return s


def lesson_file(tmp_path, records=None):
    """A practice store with one verified own-attempt lesson (or the given raw records)."""
    p = tmp_path / "t3.evals.self-improve-20990101T000000Z.lessons.jsonl"
    if records is not None:
        p.write_text("".join(json.dumps(r) + "\n" for r in records))
        return p
    st = L.LessonStore(p, guard=L.TestSplitGuard([], []))
    st.learn(inc(dev_seed(), iid="inc-d0", split="dev"), ["cell=2;d_power_db=+0.0;d_tilt_deg=-5.0"],
             ok_verification(), source="practice:own-attempt")
    return p


# --------------------------------------------------------------------------- the frozen set

def test_frozen_confirmation_set_matches_the_test_grid():
    incs, digest = C.load_confirm()  # checks the sha256 sidecar
    meta = json.loads(C.CONFIRM_META_PATH.read_text())
    assert digest == meta["sha256"] and len(incs) == 18
    assert all(i["tier"] == "masked" and i["split"] == "confirm" and i["suite"] == C.CONFIRM_VERSION for i in incs)
    assert all(int(i["seed"]) >= C.CONFIRM_SEED_BASE for i in incs)
    test = {t["incident_id"]: t for t in C.masked_test_incidents()}
    matched = meta["generator"]["matched_to"]
    assert sorted(matched.values()) == sorted(test)                       # one fresh twin per test incident
    for i in incs:
        assert C.grid_cell(i) == C.grid_cell(test[matched[i["incident_id"]]])
    assert C.mix(incs) == C.mix(list(test.values()))
    assert meta["replaced_draw"]["committed"] is False and meta["replaced_draw"]["shown_to_any_agent"] is False


def test_confirmation_set_is_disjoint_from_every_seed_and_spec():
    incs, _ = C.load_confirm()
    assert C.overlap_checks(incs)["total"] == 0
    seeds = {int(i["seed"]) for i in incs}
    assert all(not (seeds & v) for v in C.explicit_seed_sets().values())
    assert min(seeds) > max(r["hi"] for r in C.KNOWN_SEED_RANGES)
    meta = json.loads(C.CONFIRM_META_PATH.read_text())
    assert meta["exclusion"]["clean"] and meta["exclusion"]["literal_scan"]["hits"] == []


@pytest.mark.skipif(f"{platform.system()}-{platform.machine()}" != json.loads(C.CONFIRM_META_PATH.read_text())["platform"],
                    reason="the float32 GP twin replays frozen KPIs only on the platform that froze them")
def test_two_confirmation_incidents_rebuild_byte_for_byte():
    targets = load_targets()
    frozen = C.CONFIRM_PATH.read_text().splitlines()
    for line in (frozen[0], frozen[-1]):
        x = json.loads(line)
        built, why = I.build({k: x[k] for k in L.SPEC_KEYS}, targets)
        assert built is not None, why
        assert I.canonical_line({**built, "suite": C.CONFIRM_VERSION, "split": "confirm"}) == line


# --------------------------------------------------------------------------- nothing learns from it

def test_guard_refuses_the_confirmation_set():
    g = L.TestSplitGuard()
    incs, _ = C.load_confirm()
    for i in incs:
        with pytest.raises(L.LeakageError, match="confirmation"):
            g.check(i)
    for d in [json.loads(x) for x in I.DEV_PATH.read_text().splitlines() if x.strip()]:
        g.check(d)  # dev incidents are still learnable
    st = L.LessonStore(None)
    with pytest.raises(L.LeakageError):
        st.learn(dict(incs[0], split="dev"), ["cell=2;d_power_db=+0.0;d_tilt_deg=-5.0"], ok_verification())
    assert st.records == []
    assert g.overlap([{"source_seed": int(incs[0]["seed"])}]) == 1


def test_frozen_lessons_are_read_only(tmp_path):
    p = lesson_file(tmp_path)
    fl = C.FrozenLessons(p)
    before = fl.digest()
    hits = fl.retrieve(L.symptoms_of(inc()), k=3)
    assert hits and hits[0][0]["lesson_id"] == "L-01"
    with pytest.raises(RuntimeError, match="read-only"):
        fl.learn(inc(dev_seed(), split="dev"), ["FIX"], ok_verification())
    assert fl.digest() == before and p.read_bytes() == fl.raw
    info = fl.info()
    assert info["verified"] == 1 and info["source_run_id"] == "t3.evals.self-improve-20990101T000000Z"
    assert info["sources"] == {"practice:own-attempt": 1} and info["read_only"]


# --------------------------------------------------------------------------- the unit

def run_unit(tmp_path, monkeypatch, incidents, lessons_path, propose=None):
    monkeypatch.setattr(L, "verify", fake_verify)
    calls = []

    def default_propose(cond, incs, store):
        calls.append(cond)
        on = store is not None and bool(store.lessons)
        return {i["incident_id"]: (["FIX" if on else "", "", ""], 5.0, ["L-01"] if on else []) for i in incs}

    copies = {}

    def copy(key, data):
        copies[key] = data
        return "local://" + key

    unit = C.Confirm(run_id="t3.evals.confirm-test-0001", ev=Evidence(tmp_path / "ev"),
                     propose=propose or default_propose, model_id="fake", synthetic=True,
                     lessons=C.FrozenLessons(lessons_path), targets=load_targets(), log=lambda s: None,
                     copy_lessons=copy, checkpoint_dir=tmp_path / "ck")
    return unit, calls, copies


def test_confirm_unit_with_fakes(tmp_path, monkeypatch):
    incs = [inc(990_901 + k, iid=f"inc-c{k}") for k in range(4)]
    unit, calls, copies = run_unit(tmp_path, monkeypatch, incs, lesson_file(tmp_path))
    out = unit.run(incs, "c" * 64)
    r = out["result"]
    assert calls == list(C.CONDITIONS) == ["lessons_on", "lessons_off"]
    assert r["passed"] and r["value"] > 0 and r["details"]["gain_on_vs_off"]["paired_n"] == 4
    assert r["details"]["conditions"]["lessons_on"]["recovered"] == 4
    assert r["details"]["conditions"]["lessons_off"]["recovered"] == 0
    assert r["details"]["lessons"]["unchanged_during_run"] and r["details"]["lessons"]["heldout_overlap"] == 0
    assert r["details"]["lessons_retrieved_for"] == {"incidents": 4, "of": 4}
    ev = Evidence(tmp_path / "ev")
    d = ev.read_declared("t3.evals.confirm-test-0001")
    assert d["target"] == C.TARGET and d["subset"] == "split=confirm tier=masked" and d["n_items"] == 8
    assert d["twin"]["config"]["lessons"]["sha256"] == unit.lessons.sha256
    assert d["twin"]["config"]["conditions"] == ["lessons_on", "lessons_off"]
    assert copies == {"outputs/t3.evals.confirm-test-0001/lessons.jsonl": unit.lessons.raw}
    items = ev.read_records("t3.evals.confirm-test-0001", "item")
    assert len(items) == 8 and all(i["synthetic"] for i in items)
    assert json.loads((tmp_path / "ck" / "t3.evals.confirm-test-0001.phases-written.json").read_text()) == \
        list(C.CONDITIONS)


def test_confirm_refuses_lessons_from_a_heldout_incident(tmp_path, monkeypatch):
    confirm_seed = int(C.load_confirm()[0][0]["seed"])
    bad = [{"status": "verified", "lesson_id": "L-01", "source_seed": confirm_seed, "source_incident": "x",
            "source": "practice:own-attempt", "symptoms": L.symptoms_of(inc()), "pattern": []}]
    incs = [inc(990_901, iid="inc-c0")]
    unit, calls, _ = run_unit(tmp_path, monkeypatch, incs, lesson_file(tmp_path, bad))
    with pytest.raises(L.LeakageError):
        unit.run(incs, "c" * 64)
    assert calls == [] and Evidence(tmp_path / "ev").read_declared("t3.evals.confirm-test-0001") is None


def test_confirm_runs_only_the_confirmation_split(tmp_path, monkeypatch):
    incs = [inc(990_901, iid="inc-t0", split="test")]
    unit, calls, _ = run_unit(tmp_path, monkeypatch, incs, lesson_file(tmp_path))
    with pytest.raises(ValueError, match="confirmation set"):
        unit.run(incs, "c" * 64)
    assert calls == []


def test_live_backend_declares_the_confirmation_suite(tmp_path, monkeypatch):
    from t3.evals import live_swarm
    seen = {"texts": []}

    class FakeLive:
        def __init__(self, **kw):
            self.kw, self.rows, self.items = kw, [], []

        def declaration(self, incidents, digest, steps):
            return {"run_id": self.kw["run_id"], "subset": "tier=masked", "test_set_sha256": digest,
                    "twin": {"config": {"suite": "t3/evals/suite/incidents-test.jsonl"}}}

        def run_incident(self, incident, idx, steps):
            seen["texts"].append(self.kw["ticket"](incident, self.kw["run_id"]))
            for s in range(steps):
                self.rows.append({"incident_id": incident["incident_id"], "step": s, "ok": True, "change_text": ""})
            self.items.append({"extra": {"incident_id": incident["incident_id"]}, "latency_ms": 1.0})

    class FakeEv:
        def __init__(self):
            self.declared = []

        def write_declared(self, d):
            self.declared.append(d)

    monkeypatch.setattr(live_swarm, "LiveSwarm", FakeLive)
    fl = C.FrozenLessons(lesson_file(tmp_path))
    info = {**fl.info(), "copied_to": "outputs/r/lessons.jsonl"}
    ev = FakeEv()
    propose = S.live_backend(base_run_id="t3.evals.confirm-x", gateway=None, ddb=None, evlib=ev, cfg={},
                             targets=load_targets(), model_id="m", digests={"test": "d" * 64, "dev": "d" * 64},
                             unit=C.UNIT, patch_declaration=C.patch_declaration_for(info), log=lambda s: None)
    i = inc(990_901, iid="inc-vvl-mas-990901-tilt+5")
    on = propose("lessons_on", [i], fl)
    off = propose("lessons_off", [i], None)
    assert [d["unit"] for d in ev.declared] == ["t3.evals.confirm.lessons_on", "t3.evals.confirm.lessons_off"]
    assert all(d["subset"] == "split=confirm tier=masked" and d["test_set_sha256"] == "d" * 64 for d in ev.declared)
    assert all(d["twin"]["config"]["suite"] == C.SUITE_REL and d["twin"]["config"]["ticket"] == S.TICKET
               for d in ev.declared)
    assert ev.declared[0]["twin"]["config"]["lessons"]["sha256"] == fl.sha256
    assert ev.declared[1]["twin"]["config"]["lessons"] is None
    assert "Verified past lessons" in seen["texts"][0] and "Verified past lessons" not in seen["texts"][1]
    assert all("990901" not in t and "tilt+5" not in t for t in seen["texts"])
    assert on[i["incident_id"]][2] == ["L-01"] and off[i["incident_id"]][2] == []
    assert fl.digest() == C.FrozenLessons(fl.source).digest()  # retrieval changed nothing


def test_dry_run_counts(capsys):
    assert C.main(["--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["incidents"] == 18 and plan["conditions"] == ["lessons_on", "lessons_off"]
    assert plan["live_gateway_cases"] == {"per_condition": 54, "total": 108, "incident_runs": 36}
    assert plan["wall_time_estimate"]["workers=3"] == "42 min" and plan["wall_time_estimate"]["workers=6"] == "21 min"


def test_live_refuses_lessons_that_are_not_own_attempts(tmp_path):
    p = lesson_file(tmp_path, [{"status": "verified", "lesson_id": "L-01", "source_seed": 5001,
                                "source_incident": "inc-d", "source": "practice:dev-label",
                                "symptoms": L.symptoms_of(inc()), "pattern": []}])
    with pytest.raises(SystemExit, match="own"):
        C.main(["--live", "--lessons", str(p), "--run-id", "t3.evals.confirm-test-0002"])
    empty = tmp_path / "empty.lessons.jsonl"
    empty.write_text("")
    with pytest.raises(SystemExit, match="no verified lesson"):
        C.main(["--live", "--lessons", str(empty), "--run-id", "t3.evals.confirm-test-0003"])
