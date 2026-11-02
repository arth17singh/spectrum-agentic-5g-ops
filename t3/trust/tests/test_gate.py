import json

import pytest
from botocore.exceptions import ClientError

from t3.trust.gate import THRESHOLDS_PATH, TrustGate, load_thresholds
from t3.trust.evio import EvidenceWriter
from t3.trust.policy import PolicyClient
from t3.trust.records import CONTROL_FIELDS
from t3.trust.store import DynamoTrustStore, FileTrustStore, MemoryTrustStore

GOOD = {"coverage_ratio": 0.05, "sla_ratio": 0.02, "sinr_p10_db": 0.5}
BAD = {k: -v for k, v in GOOD.items()}  # every direction wrong, e = 2 * |delta| / scale


def rec(i, pred, meas=GOOD, model=("m", "v1"), **extra):
    r = {"kind": "change", "run_id": "run-1", "change_id": f"chg-{i}", "predicted_delta": dict(pred),
         "measured_delta": dict(meas), "predictor": {"model_id": model[0], "version": model[1]},
         "scored": True, "probe": False}
    r.update(extra)
    return r


def make(thresholds, **kw):
    w = EvidenceWriter()
    pc = PolicyClient(clock=lambda: 1000)
    g = TrustGate(kw.pop("store", MemoryTrustStore()), thresholds, pc, w, clock=lambda: "2026-11-02T00:00:00Z", **kw)
    return g, pc, w


def controls(w):
    return [r for (run, kind), rs in w.memory.items() if kind == "control" for r in rs]


def test_declared_thresholds_file_loads_with_hash():
    th, sha = load_thresholds()
    raw = THRESHOLDS_PATH.read_bytes()
    import hashlib
    assert sha == hashlib.sha256(raw).hexdigest() and len(sha) == 64
    assert th["warmup"] == 3 and th["recent_window"] == 3 and th["recent_needed"] == 2
    assert th["keys"] == ["coverage_ratio", "sla_ratio", "sinr_p10_db"]
    TrustGate(MemoryTrustStore())  # the default path validates


def test_ewma_is_bias_corrected(thresholds):
    a = thresholds["alpha"]
    m, n, t1 = TrustGate.ewma_step(0.0, 0, 0.8, a)
    assert t1 == pytest.approx(0.8)  # first trust equals the first score: no prior
    m, n, t2 = TrustGate.ewma_step(m, n, 0.2, a)
    assert t2 == pytest.approx(((1 - a) * a * 0.8 + a * 0.2) / (1 - (1 - a) ** 2))
    m, n, t = 0.0, 0, None
    for _ in range(50):
        m, n, t = TrustGate.ewma_step(m, n, 0.6, a)
    assert t == pytest.approx(0.6)


def test_score_parts(thresholds):
    g, _, _ = make(thresholds)
    s = g.score(GOOD, GOOD)
    assert s["e"] == 0 and s["direction_agree"] is True and s["score"] == pytest.approx(1.0)
    b = g.score(BAD, GOOD)
    assert b["e"] == pytest.approx((2 * 0.05 / 0.05 + 2 * 0.02 / 0.03 + 2 * 0.5 / 0.5) / 3)
    assert b["direction_agree"] is False and b["score"] == pytest.approx(0.0) and b["e_over_tau"]


def test_no_decision_during_warmup(thresholds):
    g, pc, w = make(thresholds)
    for i in range(thresholds["warmup"] - 1):
        out = g.update(rec(i, BAD))
        assert out["action"] is None and out["band"] == "warm-up"
    assert not pc.calls and not controls(w)
    out = g.update(rec(9, BAD))
    assert out["action"] == "pause" and out["n"] == thresholds["warmup"]


def test_two_of_last_three_rule_fires_without_ewma(thresholds):
    th = dict(thresholds, pause_below=0.01, watch_below=0.01)  # EWMA line out of the way
    g, pc, w = make(th)
    seq = [GOOD, GOOD, GOOD, GOOD, BAD, GOOD, GOOD, BAD, GOOD, BAD]  # 1 of 3, never 2 of 3 ... until the end
    actions = [g.update(rec(i, p))["action"] for i, p in enumerate(seq)]
    assert actions == [None] * 9 + ["pause"]
    assert "of the last 3" in controls(w)[0]["reason"]


def test_ewma_line_fires_without_tau(thresholds):
    th = dict(thresholds, tau=1e6)
    g, _, w = make(th)
    half = {k: 0.0 for k in GOOD}  # predicts no move: e = 1, direction False -> s = 0
    outs = [g.update(rec(i, half)) for i in range(3)]
    assert outs[-1]["action"] == "pause" and "pause line" in controls(w)[0]["reason"]


def test_pause_fires_once_and_is_idempotent(thresholds):
    g, pc, w = make(thresholds)
    outs = [g.update(rec(i, BAD)) for i in range(8)]
    assert [o["action"] for o in outs].count("pause") == 1
    assert all(o["status"] == "paused" for o in outs[2:])
    assert len(controls(w)) == 1 and [c["op"] for c in pc.calls] == ["create_policy"]
    assert g.pause("m", "v1", "again", run_id="run-1") is None
    assert len(controls(w)) == 1


def test_healthy_never_paused(thresholds):
    g, pc, w = make(thresholds)
    near = {k: v * 1.1 for k, v in GOOD.items()}
    outs = [g.update(rec(i, near)) for i in range(30)]
    assert all(o["action"] is None for o in outs) and not pc.calls and not controls(w)
    assert outs[-1]["band"] == "trusted"


def test_control_record_has_all_contract_fields(thresholds):
    g, _, w = make(thresholds)
    for i in range(3):
        g.update(rec(i, BAD))
    (c,) = controls(w)
    for f in ("schema_version", "kind", "run_id", "team", "created_at", "aws_time", "aws_time_source",
              "synthetic") + CONTROL_FIELDS:
        assert f in c, f
    assert c["by"] == "trust-gate" and c["action"] == "pause" and c["run_id"] == "run-1"
    assert c["policy_name"].startswith("team13_ranpause_") and c["synthetic"] is True  # dry-run policy
    json.dumps(c)


def test_versions_are_independent(thresholds):
    g, pc, w = make(thresholds)
    for i in range(5):
        g.update(rec(i, BAD, model=("m", "v2")))
        g.update(rec(i, GOOD, model=("m", "v1")))
    assert pc.is_paused("m", "v2") and not pc.is_paused("m", "v1")


def test_skips_probes_clipped_and_policy_only(thresholds):
    g, pc, _ = make(thresholds)
    assert "probe" in g.update(rec(0, BAD, probe=True))["skipped"]
    assert "clipped" in g.update(rec(1, BAD, scored=False, unscored_reason="clipped"))["skipped"]
    r = rec(2, BAD)
    r.pop("predictor")
    r["policy"] = {"model_id": "ppo", "version": "1"}
    assert "predictor" in g.update(r)["skipped"]
    assert g.state("m", "v1")["n"] == 0 and not pc.calls


def test_resume_is_person_only_and_restarts_probation(thresholds):
    g, pc, w = make(thresholds)
    for i in range(3):
        g.update(rec(i, BAD))
    with pytest.raises(ValueError):
        g.resume("m", "v1", person="", reason="ok", run_id="run-1")
    with pytest.raises(ValueError):
        g.resume("m", "v1", person="a.rivera", reason=" ", run_id="run-1")
    r = g.resume("m", "v1", person="a.rivera", reason="retrained", run_id="run-1")
    assert r["by"] == "person:a.rivera" and r["action"] == "resume"
    assert not pc.is_paused("m", "v1")
    st = g.state("m", "v1")
    assert st["n"] == 0 and st["trust"] is None and st["epoch"] == 1 and st["status"] == "active"
    assert g.resume("m", "v1", person="a.rivera", reason="again", run_id="run-1") is None
    assert g.update(rec(10, BAD))["action"] is None  # warm-up again
    assert [c["action"] for c in controls(w)] == ["pause", "resume"]


def test_policy_failure_releases_the_claim(thresholds):
    class Boom(PolicyClient):
        def pause(self, *a, **k):
            raise RuntimeError("throttled")

    store = MemoryTrustStore()
    g = TrustGate(store, thresholds, Boom(), EvidenceWriter())
    for i in range(2):
        g.update(rec(i, BAD))
    with pytest.raises(RuntimeError):
        g.update(rec(3, BAD))
    assert store.get("m", "v1")["status"] == "active"  # a later change can retry


def test_file_store_persists(tmp_path, thresholds):
    p = tmp_path / "trust.json"
    g, _, _ = make(thresholds, store=FileTrustStore(p))
    for i in range(3):
        g.update(rec(i, BAD))
    again = FileTrustStore(p)
    assert again.get("m", "v1")["status"] == "paused" and again.get("m", "v1")["n"] == 3


class FakeTable:
    """Just enough of a DynamoDB Table for DynamoTrustStore's expressions."""

    def __init__(self):
        self.items = {}

    def get_item(self, Key, ConsistentRead=True):
        it = self.items.get(Key["pk"])
        return {"Item": dict(it)} if it else {}

    def update_item(self, Key, UpdateExpression, ExpressionAttributeNames, ExpressionAttributeValues,
                    ConditionExpression=None):
        it = self.items.setdefault(Key["pk"], {"pk": Key["pk"]})
        cur = it.get("gate_status")
        vals = ExpressionAttributeValues
        if ConditionExpression == "attribute_not_exists(#s) OR #s <> :new" and cur == vals[":new"]:
            raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "UpdateItem")
        if ConditionExpression == "#s = :p" and cur != vals[":p"]:
            raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "UpdateItem")
        for part in UpdateExpression[len("SET "):].split(", "):
            n, v = part.split(" = ")
            it[ExpressionAttributeNames[n]] = vals[v]


def test_dynamo_store_never_writes_a_status_attribute(thresholds):
    t = FakeTable()
    g, pc, w = make(thresholds, store=DynamoTrustStore(t))
    outs = [g.update(rec(i, BAD)) for i in range(5)]
    assert [o["action"] for o in outs].count("pause") == 1
    (item,) = t.items.values()
    assert item["pk"] == "t3-trust#m@v1" and "status" not in item and item["gate_status"] == "paused"
    assert len(controls(w)) == 1
