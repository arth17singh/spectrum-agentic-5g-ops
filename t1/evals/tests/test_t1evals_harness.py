"""Harness dry run with the fake client: valid declared/item/result records, gain with a CI."""

from __future__ import annotations

import json

import pytest

from t1.evals import run as R
from t1.evals.clients import BedrockClient, Completion, Pricing, SageMakerClient
from team13_contracts import validate_record
from tools import evlib


def check(rec):
    """Both validators the evidence chain uses: W1's pydantic contract and W6's evlib checks."""
    validate_record(rec)
    evlib.validate(rec)


def _args(tmp_path, *extra):
    return R.parser().parse_args(["--evidence-dir", str(tmp_path), *extra])


def _load(tmp_path, run_id):
    declared = json.loads((tmp_path / "declared" / f"{run_id}.json").read_text())
    items = [json.loads(x) for x in (tmp_path / "records" / run_id / "item.jsonl").read_text().splitlines()]
    result = json.loads((tmp_path / "results" / f"{run_id}.json").read_text())
    outputs = [json.loads(x) for f in sorted((tmp_path / "outputs" / run_id).glob("outputs-*.jsonl"))
               for x in f.read_text().splitlines()]
    return declared, items, result, outputs


@pytest.mark.parametrize("unit", ["interface", "safety", "knowledge", "price_latency"])
def test_dry_run_records_valid(tmp_path, unit):
    r = R.run(_args(tmp_path, "--unit", unit, "--client", "fake", "--run-id", f"t1.{unit}-dry-base"))
    declared, items, result, outputs = _load(tmp_path, r["run_id"])
    for rec in [declared, result, *items]:
        check(rec)
        assert rec["synthetic"] is True and rec["run_id"] == r["run_id"]
    assert declared["n_items"] == len(items) == result["n_items"]
    assert evlib.read_records(r["run_id"], "item", store=evlib.LocalStore(tmp_path)) == items
    assert declared["declared_at"] <= items[0]["created_at"]  # declared before the first call
    assert declared["prompt_sha256"] and declared["parser"] and declared["decoding"]["temperature"] == 0.0
    # every output is stored and its hash matches the item record
    by_id = {o["item_id"]: o for o in outputs}
    import hashlib

    for it in items:
        assert hashlib.sha256(by_id[it["item_id"]]["output"].encode()).hexdigest() == it["output_sha256"]
        assert it["latency_ms"] > 0 and it["cost_usd"] is not None
    if unit != "price_latency":
        assert 0 < result["details"]["parse_fail_count"] < len(items)  # malformed outputs are counted
    else:
        assert result["metric"] == "p95_latency_ms" and "cost_per_1k_usd" in result["details"]["breakdown"]


def test_dry_run_execution_with_gain(tmp_path):
    pytest.importorskip("ran_rl")
    base = R.run(_args(tmp_path, "--unit", "execution", "--client", "fake", "--fake-profile", "base",
                       "--run-id", "t1.execution-dry-base", "--limit", "12"))
    tuned = R.run(_args(tmp_path, "--unit", "execution", "--client", "fake", "--fake-profile", "tuned",
                        "--run-id", "t1.execution-dry-tuned", "--limit", "12",
                        "--baseline-run-id", base["run_id"], "--bootstrap-resamples", "300"))
    g = tuned["details"]["gain"]
    assert set(g) >= {"value", "ci_low", "ci_high", "baseline_run_id", "paired_n"}
    assert g["paired_n"] == 12 and g["baseline_run_id"] == base["run_id"]
    assert g["ci_low"] <= g["value"] <= g["ci_high"]
    assert tuned["details"]["breakdown"]["twin_drift_items"] == 0
    declared, items, _, _ = _load(tmp_path, tuned["run_id"])
    assert declared["twin"]["kind"] == "maveric-inprocess" and declared["baseline_run_id"] == base["run_id"]
    assert all("gain" in it["extra"]["detail"] or not it["extra"]["parse_ok"] for it in items)


def test_gain_interface_and_same_set_enforced(tmp_path):
    base = R.run(_args(tmp_path, "--unit", "interface", "--client", "fake", "--run-id", "t1.if-b"))
    tuned = R.run(_args(tmp_path, "--unit", "interface", "--client", "fake", "--fake-profile", "tuned",
                        "--run-id", "t1.if-t", "--baseline-run-id", base["run_id"]))
    g = tuned["details"]["gain"]
    assert g["paired_n"] == base["n_items"]
    assert g["value"] == pytest.approx(tuned["value"] - base["value"])
    # a baseline on a different item selection is refused
    with pytest.raises(SystemExit):
        R.run(_args(tmp_path, "--unit", "interface", "--client", "fake", "--run-id", "t1.if-x",
                    "--limit", "5", "--baseline-run-id", base["run_id"]))


def test_evidence_never_overwritten(tmp_path):
    R.run(_args(tmp_path, "--unit", "safety", "--client", "fake", "--run-id", "t1.sf-once", "--limit", "3"))
    with pytest.raises(FileExistsError):
        R.run(_args(tmp_path, "--unit", "safety", "--client", "fake", "--run-id", "t1.sf-once", "--limit", "3"))


def test_bootstrap_known_values():
    this = [{"item_id": str(i), "passed": True, "score": 1.0} for i in range(10)]
    base = [{"item_id": str(i), "passed": i < 5, "score": 1.0 if i < 5 else 0.0} for i in range(10)]
    g = R.paired_bootstrap(this, base, "pass_rate", 500)
    assert g["value"] == pytest.approx(0.5) and 0.1 < g["ci_low"] <= 0.5 <= g["ci_high"] <= 0.9
    same = R.paired_bootstrap(this, this, "mean_score", 200)
    assert same["value"] == same["ci_low"] == same["ci_high"] == 0.0


class _FakeBedrock:
    def converse(self, **kw):
        assert kw["inferenceConfig"]["temperature"] == 0.0 and kw["system"][0]["text"] == "S"
        return {"output": {"message": {"content": [{"reasoningContent": {}}, {"text": '{"answer": "A"}'}]}},
                "usage": {"inputTokens": 100, "outputTokens": 7}, "metrics": {"latencyMs": 321},
                "ResponseMetadata": {"RequestId": "r1", "HTTPHeaders": {"date": "Mon, 02 Nov 2026 20:00:00 GMT"}}}


class _Body:
    def __init__(self, b):
        self.b = b

    def read(self):
        return self.b


class _FakeSM:
    def __init__(self, schema):
        self.schema = schema

    def invoke_endpoint(self, **kw):
        body = json.loads(kw["Body"])
        if self.schema == "messages":
            assert body["messages"][0] == {"role": "system", "content": "S"} and body["max_tokens"] == 50
            out = {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 9, "completion_tokens": 2}}
        else:
            assert body["inputs"].startswith("<|im_start|>system\nS") and body["parameters"]["do_sample"] is False
            out = [{"generated_text": "ok"}]
        return {"Body": _Body(json.dumps(out).encode()), "ResponseMetadata": {"HTTPHeaders": {}}}


def test_real_clients_with_stubbed_transport():
    c = BedrockClient("m", "us-east-2", client=_FakeBedrock()).generate("S", "U", 50)
    assert c.text == '{"answer": "A"}' and c.tokens_in == 100 and c.aws_time == "2026-11-02T20:00:00Z"
    assert Pricing(1.0, 2.0).cost(c) == pytest.approx((100 * 1.0 + 7 * 2.0) / 1e6)
    m = SageMakerClient("ep", "us-east-2", "m", "messages", client=_FakeSM("messages")).generate("S", "U", 50)
    assert m.text == "ok" and m.tokens_out == 2 and not m.tokens_estimated
    t = SageMakerClient("ep", "us-east-2", "m", "tgi", client=_FakeSM("tgi")).generate("S", "U", 50)
    assert t.text == "ok" and t.tokens_estimated
    assert Pricing(instance_per_hour=3.6).cost(Completion("x", latency_ms=1000)) == pytest.approx(0.001)
