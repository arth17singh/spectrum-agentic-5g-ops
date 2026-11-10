"""Verified lesson memory and the self-improvement unit (fakes everywhere except one real twin replay)."""
from __future__ import annotations

import json

import pytest

from t3.evals import incidents as I
from t3.evals import lessons as L
from t3.evals import self_improve as S
from t3.evals.evshim import Evidence
from t3.evals.graders import load_targets


def inc(seed=5001, iid=None, split="dev", pre=(0.95, 0.90, 1.0), post=(0.80, 0.70, -1.0), fault=None):
    return {"incident_id": iid or f"inc-x-{seed}", "seed": seed, "split": split, "preset": "official",
            "scenario": "nominal", "tier": "masked", "template": "tilt+5", "commissioning": [],
            "fault": fault or [{"cell": 2, "d_power_db": 0.0, "d_tilt_deg": 5.0}],
            "pre_fault_kpis": {"coverage_ratio": pre[0], "sla_ratio": pre[1], "sinr_p10_db": pre[2]},
            "post_fault_kpis": {"coverage_ratio": post[0], "sla_ratio": post[1], "sinr_p10_db": post[2]}}


STATE = {"power_dbm": [40.0, 40.0, 40.0, 40.0], "tilt_deg": [4.0, 4.0, 9.0, 4.0]}


def ok_verification(change="cell=2;d_power_db=+0.0;d_tilt_deg=-5.0", recovered=True, recovery=1.0):
    return {"verifier": L.VERIFIER, "backend": "fake", "recovered": recovered, "recovery": recovery,
            "harmful": recovery is not None and recovery < 0, "steps_to_fix": 1 if recovered else None,
            "clamped_rules": [], "change_texts": [change, "", ""], "state_before_fix": STATE}


def dev_seed():
    s = 5000
    while I.split_of(s) != "dev":
        s += 1
    return s


def test_seed():
    s = 5000
    while I.split_of(s) != "test":
        s += 1
    return s


def store(tmp_path=None, test_incidents=()):
    return L.LessonStore(tmp_path / "l.jsonl" if tmp_path else None, guard=L.TestSplitGuard(list(test_incidents)))


def test_similarity_bounds():
    a = L.symptoms_of(inc())
    assert L.similarity(a, a) == 1.0
    b = L.symptoms_of(inc(post=(0.50, 0.40, -6.0)))
    assert 0.0 <= L.similarity(a, b) < 0.1


def test_pattern_is_relative_and_resolves_on_a_new_network():
    pat = L.describe("cell=2;d_power_db=+0.0;d_tilt_deg=-5.0", STATE)
    assert pat == [{"by": "tilt", "rank": 0, "d_power_db": 0.0, "d_tilt_deg": -5.0}]
    assert "cell" not in json.dumps(pat).replace("the cell", "")
    other = {"power_dbm": [40.0] * 4, "tilt_deg": [4.0, 10.0, 4.0, 4.0]}  # now cell 1 stands out
    assert L.resolve(pat, other) == [{"cell": 1, "d_power_db": 0.0, "d_tilt_deg": -5.0}]
    assert L.pattern_text(pat) == "lower tilt by 5 deg on the cell with the highest tilt"


def test_learn_verified_and_rejected(tmp_path):
    st = store(tmp_path)
    good = st.learn(inc(dev_seed()), ["cell=2;d_power_db=+0.0;d_tilt_deg=-5.0", "", ""], ok_verification())
    assert good["status"] == "verified" and good["lesson_id"] == "L-01"
    bad = st.learn(inc(dev_seed() + 0, iid="inc-bad"), ["cell=0;d_power_db=+0.0;d_tilt_deg=-5.0", "", ""],
                   ok_verification("cell=0;d_power_db=+0.0;d_tilt_deg=-5.0", recovered=False, recovery=-0.3))
    assert bad["status"] == "rejected" and "not learned" in bad["reason"] and "worse" in bad["reason"]
    none = st.learn(inc(dev_seed(), iid="inc-none"), ["", "", ""], ok_verification("", False, 0.0))
    assert none["status"] == "rejected"
    assert [r["status"] for r in L.LessonStore(tmp_path / "l.jsonl", guard=st.guard).records] == \
        ["verified", "rejected", "rejected"]  # persisted, append-only
    assert len(st.lessons) == 1 and len(st.rejected) == 2


def test_same_fix_same_symptoms_adds_support_not_a_lesson():
    st = store()
    for k in range(3):
        st.learn(inc(dev_seed(), iid=f"inc-{k}"), ["cell=2;d_power_db=+0.0;d_tilt_deg=-5.0"], ok_verification())
    assert len(st.lessons) == 1 and st.lessons[0]["support"] == 3


def test_leakage_guard_refuses_test_split():
    t = inc(test_seed(), iid="inc-test-1", split="test")
    st = store(test_incidents=[t])
    with pytest.raises(L.LeakageError):
        st.learn(t, ["cell=2;d_power_db=+0.0;d_tilt_deg=-5.0"], ok_verification())
    clone = dict(t, incident_id="renamed", split="dev")  # same spec bytes under another name
    with pytest.raises(L.LeakageError, match="spec hash|test split|seed"):
        st.learn(clone, ["cell=2;d_power_db=+0.0;d_tilt_deg=-5.0"], ok_verification())
    assert st.records == [] and st.guard.overlap(st.records) == 0


def test_frozen_test_file_guard_loads_and_blocks_its_seeds():
    g = L.TestSplitGuard()
    test, _ = I.load_test(I.TEST_PATH)
    with pytest.raises(L.LeakageError):
        g.check(test[0])
    dev = [json.loads(x) for x in I.DEV_PATH.read_text().splitlines() if x.strip()]
    for d in dev:
        g.check(d)  # no dev incident overlaps the test split


def test_retrieve_and_block():
    st = store()
    st.learn(inc(dev_seed(), iid="a"), ["cell=2;d_power_db=+0.0;d_tilt_deg=-5.0"], ok_verification())
    st.learn(inc(dev_seed(), iid="b", post=(0.40, 0.30, -8.0)), ["cell=0;d_power_db=+6.0;d_tilt_deg=+0.0"],
             ok_verification("cell=0;d_power_db=+6.0;d_tilt_deg=+0.0"))
    hits = st.retrieve(L.symptoms_of(inc()), k=2)
    assert hits[0][0]["lesson_id"] == "L-01" and hits[0][1] == 1.0 and hits[1][1] < hits[0][1]
    assert st.best_similarity(L.symptoms_of(inc())) == 1.0
    block = L.lesson_block(hits)
    assert block.startswith("Verified past lessons") and "L-01" in block
    assert "cell=" not in block and "inc-" not in block and "a)" not in block
    assert L.lesson_block([(hits[0][0], 0.1)]) == ""


def test_verify_on_the_real_twin_accepts_the_inverse_and_rejects_the_wrong_cell():
    targets = load_targets()
    built = None
    for i in range(I.N_CANDIDATES):
        spec = I.candidate(i)
        if spec["template"] == "power-6" and spec["tier"] == "visible":
            built, _ = I.build(spec, targets)
            if built:
                break
    assert built is not None
    good = L.verify(built, [L.inverse_fix(built), "", ""], targets)
    wrong = L.verify(built, [L.wrong_fix(built), "", ""], targets)
    nothing = L.verify(built, ["", "", ""], targets)
    assert good["recovered"] and good["steps_to_fix"] == 1
    assert not wrong["recovered"] and not nothing["recovered"] and nothing["recovery"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- the unit (fake verify)

def fake_verify(incident, proposals, targets=None):
    props = [p or "" for p in proposals]
    fixed = props[0] == "FIX"
    return {"verifier": "fake", "backend": "fake", "recovered": fixed, "recovery": 1.0 if fixed else 0.2,
            "harmful": False, "steps_to_fix": 1 if fixed else None, "clamped_rules": [],
            "change_texts": ["cell=2;d_power_db=+0.0;d_tilt_deg=-5.0" if fixed else "", "", ""],
            "state_before_fix": STATE}


def test_self_improve_unit_with_fakes(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "verify", fake_verify)
    test = [inc(test_seed() + 2 * k, iid=f"inc-t{k}", split="test") for k in range(6)]
    dev = [inc(dev_seed(), iid=f"inc-d{k}") for k in range(3)]
    calls = []

    def propose(phase, incidents, st):
        calls.append(phase)
        return {i["incident_id"]: (["FIX" if st is not None and st.lessons else "", "", ""] if phase != "practice"
                                   else ["FIX", "", ""], 5.0, []) for i in incidents}

    st = L.LessonStore(tmp_path / "l.jsonl", guard=L.TestSplitGuard(test))
    unit = S.SelfImprove(run_id="t3.self-improve-test-0001", ev=Evidence(tmp_path / "ev"), propose=propose,
                         model_id="fake", synthetic=True, store=st, targets=load_targets(), log=lambda s: None)
    out = unit.run(test, "0" * 64, dev, "1" * 64)
    r = out["result"]
    assert calls == ["before", "practice", "after", "control"]
    assert r["details"]["phases"]["before"]["recovered"] == 0 and r["details"]["phases"]["after"]["recovered"] == 6
    assert r["passed"] and r["value"] > 0 and r["details"]["learning_attributable"]
    assert r["details"]["lessons"] == {"verified": 1, "rejected": 0, "test_overlap": 0}
    ev = Evidence(tmp_path / "ev")
    assert ev.read_declared("t3.self-improve-test-0001")["target"] == S.TARGET
    items = ev.read_records("t3.self-improve-test-0001", "item")
    assert len(items) == 6 + 3 + 6 + 6 and all(i["synthetic"] for i in items)


def test_self_improve_refuses_practice_on_test(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "verify", fake_verify)
    test = [inc(test_seed(), iid="inc-t0", split="test")]
    st = L.LessonStore(None, guard=L.TestSplitGuard(test))
    unit = S.SelfImprove(run_id="t3.self-improve-test-0002", ev=Evidence(tmp_path / "ev"),
                         propose=lambda p, incs, s: {i["incident_id"]: (["FIX"], 1.0, []) for i in incs},
                         model_id="fake", synthetic=True, store=st, targets=load_targets(), log=lambda s: None)
    with pytest.raises(L.LeakageError):
        unit.run(test, "0" * 64, test, "0" * 64)  # practising on the test split is refused


def test_live_backend_only_changes_the_ticket(monkeypatch):
    from t3.evals import live_swarm

    seen = {}

    class FakeLive:
        def __init__(self, **kw):
            seen.update(kw)
            self.rows, self.items = [], []

        def declaration(self, incidents, digest, steps):
            return {"run_id": seen["run_id"], "subset": "tier=masked"}

        def run_incident(self, incident, idx, steps):
            text = seen["ticket"](incident, seen["run_id"])
            seen.setdefault("texts", []).append(text)
            for s in range(steps):
                self.rows.append({"incident_id": incident["incident_id"], "step": s, "ok": True,
                                  "change_text": "cell=2;d_power_db=+0.0;d_tilt_deg=-5.0" if s == 0 else ""})
                self.items.append({"extra": {"incident_id": incident["incident_id"]}, "latency_ms": 10.0})

    class FakeEv:
        def __init__(self):
            self.declared = []

        def write_declared(self, d):
            self.declared.append(d)

    monkeypatch.setattr(live_swarm, "LiveSwarm", FakeLive)
    ev = FakeEv()
    propose = S.live_backend(base_run_id="t3.si-x", gateway=None, ddb=None, evlib=ev, cfg={}, targets=load_targets(),
                             model_id="m", digests={"test": "a" * 64, "dev": "b" * 64})
    st = store()
    st.learn(inc(dev_seed(), iid="d"), ["cell=2;d_power_db=+0.0;d_tilt_deg=-5.0"], ok_verification())
    off = propose("before", [inc(iid="inc-t")], None)
    on = propose("after", [inc(iid="inc-t")], st)
    assert "Verified past lessons" not in seen["texts"][0]
    from t3.evals.swarm import task_text
    base, block = seen["texts"][1].split("\n\n", 1)
    assert base == S.opaque_ticket(inc(iid="inc-t"), "t3.si-x.after") and block.startswith("Verified past lessons")
    assert base == task_text({**inc(iid="inc-t"), "incident_id": S.opaque_label(inc(iid="inc-t"))}, "t3.si-x.after")
    assert on["inc-t"][2] == ["L-01"] and off["inc-t"][2] == []
    assert on["inc-t"][0][0] == "cell=2;d_power_db=+0.0;d_tilt_deg=-5.0" and on["inc-t"][1] == 30.0
    assert [d["unit"] for d in ev.declared] == ["t3.evals.self-improve.before", "t3.evals.self-improve.after"]


def test_opaque_ticket_hides_the_fault_template():
    i = inc(iid="inc-off-mas-4102-tilt+5")
    t = S.opaque_ticket(i, "r")
    assert "tilt+5" not in t and "4102" not in t and S.opaque_label(i) in t
    assert S.opaque_label(i) == "inc-" + __import__("hashlib").sha256(b"inc-off-mas-4102-tilt+5").hexdigest()[:8]


def test_live_mode_refuses_dev_labels(tmp_path):
    with pytest.raises(ValueError):
        S.SelfImprove(run_id="t3.self-improve-test-0003", ev=Evidence(tmp_path / "ev"), propose=lambda *a: {},
                      model_id="m", synthetic=False, store=L.LessonStore(None), mode="live", offline_labels=True)


def test_practice_learns_after_each_incident(tmp_path, monkeypatch):
    """With on_done, the second practice incident already sees the first one's lesson."""
    monkeypatch.setattr(L, "verify", fake_verify)
    test = [inc(test_seed() + 2 * k, iid=f"inc-t{k}", split="test") for k in range(2)]
    dev = [inc(dev_seed(), iid=f"inc-d{k}") for k in range(2)]
    seen_lessons = []

    def propose(phase, incidents, st, on_done=None):
        out = {}
        for i in incidents:
            if phase == "practice":
                seen_lessons.append(len(st.lessons))
            out[i["incident_id"]] = (["FIX", "", ""], 1.0, [])
            if on_done:
                on_done(i, out[i["incident_id"]][0])
        return out

    st = L.LessonStore(tmp_path / "l.jsonl", guard=L.TestSplitGuard(test))
    unit = S.SelfImprove(run_id="t3.self-improve-test-0004", ev=Evidence(tmp_path / "ev"), propose=propose,
                         model_id="fake", synthetic=True, store=st, targets=load_targets(), log=lambda s: None,
                         checkpoint_dir=tmp_path / "ck")
    out = unit.run(test, "0" * 64, dev, "1" * 64)
    assert seen_lessons == [0, 1]
    assert all(it["extra"]["learned"] for it in out["phases"]["practice"])
    assert json.loads((tmp_path / "ck" / "t3.self-improve-test-0004.phases-written.json").read_text()) == list(S.PHASES)


def test_live_backend_workers_checkpoint_and_resume(tmp_path, monkeypatch):
    from t3.evals import live_swarm
    made = []

    class FakeLive:
        def __init__(self, **kw):
            self.kw, self.rows, self.items = kw, [], []
            self._lock = __import__("threading").Lock()
            made.append(self)

        def declaration(self, incidents, digest, steps):
            return {"run_id": self.kw["run_id"], "twin": {"config": {}}, "n": len(incidents)}

        def run_incident(self, incident, idx, steps):
            if incident["incident_id"] == "boom" and len(made) == 1:
                raise RuntimeError("crash mid-phase")
            self.kw["ticket"](incident, self.kw["run_id"])
            with self._lock:
                for s in range(steps):
                    self.rows.append({"incident_id": incident["incident_id"], "step": s, "ok": True,
                                      "change_text": "C" if s == 0 else ""})
                self.items.append({"extra": {"incident_id": incident["incident_id"]}, "latency_ms": 1.0})

    class FakeEv:
        def __init__(self):
            self.declared = []

        def write_declared(self, d):
            self.declared.append(d)

    monkeypatch.setattr(live_swarm, "LiveSwarm", FakeLive)
    ev = FakeEv()
    propose = S.live_backend(base_run_id="t3.si-y", gateway=None, ddb=None, evlib=ev, cfg={}, targets=load_targets(),
                             model_id="m", digests={"test": "a" * 64, "dev": "b" * 64}, workers=1,
                             checkpoint_dir=tmp_path, log=lambda s: None)
    incs = [inc(iid="a1"), inc(iid="boom"), inc(iid="a3")]
    with pytest.raises(RuntimeError):
        propose("before", incs, None)
    assert ev.declared[0]["run_id"] == "t3.si-y.before"           # declared before the first call
    got = propose("before", incs, None)                          # resume: a1 skipped, the rest re-run
    assert set(got) == {"a1", "boom", "a3"}
    assert ev.declared[1]["run_id"] == "t3.si-y.before.r1" and ev.declared[1]["n"] == 2
    assert ev.declared[1]["twin"]["config"]["resumes"] == "t3.si-y.before"
    assert all(d["twin"]["config"]["ticket"] == S.TICKET for d in ev.declared)
    # workers > 1 runs every incident once
    made.clear()
    p2 = S.live_backend(base_run_id="t3.si-z", gateway=None, ddb=None, evlib=FakeEv(), cfg={}, targets=load_targets(),
                        model_id="m", digests={"test": "a" * 64, "dev": "b" * 64}, workers=3, log=lambda s: None)
    made.append(None)  # no crash path
    got = p2("after", [inc(iid=f"w{k}") for k in range(5)], None)
    assert sorted(got) == [f"w{k}" for k in range(5)] and all(v[0][0] == "C" for v in got.values())
