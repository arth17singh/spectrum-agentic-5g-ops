"""The endpoint handler (t1/sft/serve/inference.py) with a fake tokenizer and model: W3's nlm___ask body in,
a body W3's parser and the eval client read out, thinking off exactly as the eval renders it, one log line."""

from __future__ import annotations

import hashlib
import json

import pytest

from t1.sft import chat
from t1.sft.serve import deploy, inference
from t1.sft.testing import FakeTokenizer

torch = pytest.importorskip("torch")


class Enc(dict):
    def to(self, device):
        return self


class Tok(FakeTokenizer):
    eos_token_id = 200_001  # <|im_end|> in FakeTokenizer
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = super().__call__(text, add_special_tokens)["input_ids"]
        return Enc(input_ids=torch.tensor([ids])) if return_tensors == "pt" else {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        inv = {self.convert_tokens_to_ids(t): t for t in self.SPECIAL}
        return "".join("" if i in inv and skip_special_tokens else inv.get(i, chr(i)) for i in ids)


class Model:
    def __init__(self, reply: str):
        self.reply, self.calls = reply, []

    def generate(self, input_ids, **kw):
        self.calls.append(kw)
        ids = [ord(c) for c in self.reply]
        cap = kw["max_new_tokens"]
        new = ids[:cap] if len(ids) >= cap else ids + [Tok.eos_token_id]  # cut at the cap, else stop on EOS
        return torch.cat([input_ids, torch.tensor([new])], dim=1)


def _m(reply="Lowering the tilt pushes coverage further out."):
    tok = Tok()
    return {"tok": tok, "model": Model(reply), "device": "cpu", "eos": [Tok.eos_token_id],
            "model_label": "test", "train_run_id": "t1.sft.train-x"}


def test_w3_body_in_openai_shape_out_and_w3_parses_it(capsys):
    from t3.tools.nlm import SYSTEM, parse_generation

    body = deploy.smoke_body()
    assert body["messages"][0]["content"] == SYSTEM  # the smoke sends exactly what nlm___ask sends
    raw = json.dumps(body).encode()
    m = _m()
    data = inference.input_fn(raw, "application/json")
    out, ctype = inference.output_fn(inference.predict_fn(data, m), "application/json")
    assert ctype == "application/json"
    resp = json.loads(out)
    assert parse_generation(resp) == "Lowering the tilt pushes coverage further out."
    assert resp["choices"][0]["finish_reason"] == "stop" and resp["usage"]["completion_tokens"] > 0
    assert resp["team13"]["request_sha256"] == hashlib.sha256(raw).hexdigest() and resp["team13"]["thinking"] == "off"
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith(inference.LOG_MARK)]
    assert len(line) == 1
    log = json.loads(line[0][len(inference.LOG_MARK) + 1:])
    assert log["response_sha256"] == hashlib.sha256(out.encode()).hexdigest()
    assert log["request_sha256"] == resp["team13"]["request_sha256"] and log["latency_ms"] >= 0
    assert m["model"].calls[0]["do_sample"] is True and m["model"].calls[0]["temperature"] == 0.2


def test_prompt_is_the_eval_harness_thinking_off_rendering():
    m = _m()
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
    prompt, cfg = inference.build_prompt(m["tok"], {"messages": msgs, "chat_template_kwargs": {"enable_thinking": True}})
    harness = chat.tokenizer_renderer(m["tok"])(msgs, add_generation_prompt=True, enable_thinking=False)
    assert prompt == harness and prompt.endswith(inference.THINK_OFF_SUFFIX)  # thinking stays off whatever is asked
    assert cfg["gen"]["do_sample"] is False and cfg["gen"]["max_new_tokens"] == inference.DEFAULT_MAX_NEW_TOKENS


def test_raw_inputs_and_caps():
    m = _m("x" * 50)
    _, cfg = inference.build_prompt(m["tok"], {"inputs": "hello", "parameters": {"max_new_tokens": 99999}})
    assert cfg["gen"]["max_new_tokens"] == inference.MAX_NEW_TOKENS_CAP and cfg["mode"] == "inputs"
    data = inference.input_fn(json.dumps({"messages": [{"role": "user", "content": "q"}], "max_tokens": 10}),
                              "application/json")
    resp = json.loads(inference.output_fn(inference.predict_fn(data, m))[0])
    assert resp["choices"][0]["finish_reason"] == "length" and resp["usage"]["completion_tokens"] == 10
    with pytest.raises(ValueError):
        inference.input_fn(b'{"prompt": "x"}', "application/json")
    with pytest.raises(ValueError):
        inference.input_fn(b"x", "text/plain")


def test_deploy_requests_name_capture_and_serve_the_merged_model():
    r = deploy.requests(account="000000000000", bucket="bb26-p-team13-data-000000000000-use2",
                        train_run_id="t1.sft.train-x", sums_sha256="a" * 64, instance="ml.g6.xlarge")
    assert r["endpoint"]["EndpointName"] == "team13-otel-nlm"  # the name the tools role may invoke
    src = r["model"]["PrimaryContainer"]["ModelDataSource"]["S3DataSource"]
    assert src == {"S3Uri": "s3://bb26-p-team13-data-000000000000-use2/serving/t1-otel-1.7b-lora/t1.sft.train-x/",
                   "S3DataType": "S3Prefix", "CompressionType": "None"}
    env = r["model"]["PrimaryContainer"]["Environment"]
    assert env["SAGEMAKER_PROGRAM"] == "inference.py" and env["NLM_TRAIN_RUN_ID"] == "t1.sft.train-x"
    assert "pytorch-inference:2.6.0-gpu-py312" in r["model"]["PrimaryContainer"]["Image"]
    v = r["config"]["ProductionVariants"][0]
    assert v["InstanceType"] == "ml.g6.xlarge" and v["InitialInstanceCount"] == 1
    cap = r["config"]["DataCaptureConfig"]
    assert cap["InitialSamplingPercentage"] == 100 and {c["CaptureMode"] for c in cap["CaptureOptions"]} == {"Input", "Output"}
    assert len(r["names"]["model"]) <= 63 and r["names"]["config"].endswith("-cfg")
