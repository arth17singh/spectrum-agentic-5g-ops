"""Bedrock baselines: Price List lookup (fail closed), what Converse is sent per model, and the
harness guards (prices required, fail fast when every early call errors). No AWS call is made."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from t1.evals import bedrock_prices as B
from t1.evals import run as R
from t1.evals.clients import BedrockClient

FIX = Path(__file__).with_name("fixtures") / "bedrock_prices_use2.json"


def _items():
    return json.loads(FIX.read_text())["PriceList"]


def test_prices_from_the_price_list():
    g = B.lookup(_items(), "openai.gpt-oss-20b-1:0")
    assert (g["in_per_1m"], g["out_per_1m"]) == (0.07, 0.3)
    assert g["usagetypes"] == ["USE2-gpt-oss-20b-input-tokens", "USE2-gpt-oss-20b-output-tokens"]
    q = B.lookup(_items(), "qwen.qwen3-32b-v1:0")
    assert (q["in_per_1m"], q["out_per_1m"]) == (0.15, 0.6) and q["effective"].startswith("2026-10-01")


def test_prices_fail_closed():
    with pytest.raises(B.PriceNotFound):
        B.lookup(_items(), "meta.unknown-model")
    with pytest.raises(B.PriceNotFound):  # no standard output price -> refuse, never guess
        B.lookup([p for p in _items() if p["product"]["attributes"]["usagetype"] != "USE2-Qwen3-32B-output-tokens"],
                 "qwen.qwen3-32b-v1:0")
    assert B.main(["--model-id", "meta.unknown-model", "--from-file", str(FIX)]) == 3


class _Stub:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def converse(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise RuntimeError("ValidationException: unknown field")
        return {"output": {"message": {"content": [{"reasoningContent": {"reasoningText": {"text": "think" * 10}}},
                                                   {"text": '{"answer": "A"}'}]}},
                "usage": {"inputTokens": 120, "outputTokens": 60}, "metrics": {"latencyMs": 400},
                "stopReason": "end_turn", "ResponseMetadata": {"HTTPHeaders": {"date": "Mon, 02 Nov 2026 20:00:00 GMT"}}}


def test_what_converse_is_sent():
    oss = BedrockClient("openai.gpt-oss-20b-1:0", "us-east-2", client=_Stub())
    kw = oss.request("S", "U", 2048)
    assert kw["additionalModelRequestFields"] == {"reasoning_effort": "low"}
    assert kw["inferenceConfig"] == {"maxTokens": 2048, "temperature": 0.0}
    qwen = BedrockClient("qwen.qwen3-32b-v1:0", "us-east-2", client=_Stub())
    assert "additionalModelRequestFields" not in qwen.request("S", "U", 2048)
    d = qwen.decoding(2048)
    assert d["additionalModelRequestFields"] == {} and "measured as served" in d["thinking"]
    c = oss.generate("S", "U", 2048)
    assert c.text == '{"answer": "A"}' and c.meta["reasoning_chars"] == 50 and c.tokens_out == 60


def _args(tmp_path, *extra):
    return R.parser().parse_args(["--evidence-dir", str(tmp_path), "--client", "bedrock", "--unit", "interface",
                                  "--model-id", "openai.gpt-oss-20b-1:0", "--limit", "5", *extra])


def test_bedrock_needs_prices(tmp_path):
    with pytest.raises(SystemExit, match="price"):
        R.run(_args(tmp_path), client=BedrockClient("openai.gpt-oss-20b-1:0", "us-east-2", client=_Stub()))


PRICED = ["--price-in-per-1m", "0.07", "--price-out-per-1m", "0.3", "--price-source", "test"]


def test_bedrock_run_records_cost_and_decoding(tmp_path):
    from tools import evlib

    r = R.run(_args(tmp_path, *PRICED, "--run-id", "t1.if-oss", "--synthetic"),
              client=BedrockClient("openai.gpt-oss-20b-1:0", "us-east-2", client=_Stub()))
    store = evlib.LocalStore(tmp_path)
    decl = evlib.read_declared("t1.if-oss", store=store)
    assert decl["decoding"]["additionalModelRequestFields"] == {"reasoning_effort": "low"}
    items = evlib.read_records("t1.if-oss", "item", store=store)
    assert all(it["cost_usd"] == pytest.approx((120 * 0.07 + 60 * 0.3) / 1e6) for it in items)
    assert all(it["aws_time"] == "2026-11-02T20:00:00Z" for it in items)
    assert r["details"]["latency_and_price"]["cost_per_1k_usd"] == pytest.approx((120 * 0.07 + 60 * 0.3) / 1e3)


def test_fail_fast_when_every_early_call_errors(tmp_path):
    with pytest.raises(SystemExit, match="first 3 calls all failed"):
        R.run(_args(tmp_path, *PRICED, "--run-id", "t1.if-fail", "--synthetic"),
              client=BedrockClient("openai.gpt-oss-20b-1:0", "us-east-2", client=_Stub(fail=True)))
