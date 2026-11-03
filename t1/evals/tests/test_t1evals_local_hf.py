"""The in-process client (`--client local-hf`): the code path a SageMaker job runs.

Two levels:
  * a fake tokenizer and model injected into LocalHFClient (needs torch only): checks the chat
    template call (enable_thinking=False), greedy decoding, token counts, latency, price and records;
  * a tiny random Qwen3 model with the real Qwen3 tokenizer and a LoRA adapter, loaded by
    transformers/peft from disk through `python -m t1.evals.run --client local-hf` (skipped unless
    transformers and peft are installed and T1_QWEN3_TOKENIZER_DIR points at Qwen3 tokenizer files).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from team13_contracts import validate_record  # noqa: E402
from tools import evlib  # noqa: E402

from t1.evals import run as R  # noqa: E402
from t1.evals.local_hf import LocalHFClient  # noqa: E402

ANSWER = '{"intent": "no_action", "cells": [], "parameter": null, "direction": null, "amount": null, "kpi": null}'


class FakeTok:
    pad_token_id = None
    eos_token_id = 0

    def __init__(self, template="TEMPLATE-A"):
        self.chat_template = template
        self.calls = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking=True):
        self.calls.append({"tokenize": tokenize, "add_generation_prompt": add_generation_prompt,
                           "enable_thinking": enable_thinking, "roles": [m["role"] for m in messages]})
        body = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return body + ("<assistant>" if enable_thinking else "<assistant><think></think>")

    def __call__(self, text, return_tensors, add_special_tokens, truncation=True):
        assert truncation is False  # prompts are never cut to model_max_length
        n = max(1, len(text) // 50)
        return {"input_ids": torch.ones((1, n), dtype=torch.long), "attention_mask": torch.ones((1, n), dtype=torch.long)}

    def decode(self, ids, skip_special_tokens):
        return ANSWER


class FakeModel:
    def __init__(self):
        self.kwargs = []

    def generate(self, input_ids, attention_mask, **kw):
        self.kwargs.append(kw)
        return torch.cat([input_ids, torch.full((1, 7), 5, dtype=torch.long)], dim=1)


def _client(tmp_path, template="TEMPLATE-A"):
    mdir = tmp_path / "model"
    mdir.mkdir(exist_ok=True)
    (mdir / "config.json").write_text('{"fake": true}')
    tok, model = FakeTok(template), FakeModel()
    c = LocalHFClient(str(mdir), "fake-local-hf", components=(tok, model, "cpu", "float32"), warmup=2,
                      instance_type="ml.g5.2xlarge")
    return c, tok, model


def _args(tmp_path, *extra):
    return R.parser().parse_args(["--evidence-dir", str(tmp_path / "ev"), "--client", "local-hf",
                                  "--model-dir", str(tmp_path / "model"), "--model-id", "fake-local-hf",
                                  "--instance-price-per-hour", "1.212", "--price-source", "test price, not real",
                                  "--synthetic", *extra])


def test_prompt_format_is_fixed(tmp_path):
    c, tok, _ = _client(tmp_path)
    text = c.render("SYS", "USER")
    assert text.endswith("<assistant><think></think>") and "<system>SYS<user>USER" in text
    last = tok.calls[-1]
    assert last == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False,
                    "roles": ["system", "user"]}
    assert c.thinking_flag_honoured is True
    d = c.decoding(256)
    assert d["chat_template_kwargs"] == {"enable_thinking": False} and d["do_sample"] is False
    assert d["temperature"] == 0.0 and d["batch_size"] == 1 and d["instance_type"] == "ml.g5.2xlarge"
    assert len(d["chat_template_sha256"]) == 64 and len(d["model_dir_sha256"]) == 64


def test_generate_is_greedy_and_counts_tokens(tmp_path):
    c, _, model = _client(tmp_path)
    out = c.generate("SYS", "USER", 64)
    assert out.text == ANSWER and out.tokens_out == 7 and out.tokens_in >= 1 and not out.tokens_estimated
    assert out.latency_ms > 0 and out.error is None
    kw = model.kwargs[-1]
    assert kw["do_sample"] is False and kw["num_beams"] == 1 and kw["max_new_tokens"] == 64
    assert kw["pad_token_id"] == 0  # falls back to eos when the tokenizer has no pad token


def test_local_hf_run_writes_valid_records(tmp_path):
    c, _, _ = _client(tmp_path)
    r = R.run(_args(tmp_path, "--unit", "interface", "--run-id", "t1.if-localhf", "--limit", "8"), client=c)
    store = evlib.LocalStore(tmp_path / "ev")
    decl = evlib.read_declared(r["run_id"], store=store)
    items = evlib.read_records(r["run_id"], "item", store=store)
    res = evlib.read_result(r["run_id"], store=store)
    for rec in [decl, res, *items]:
        validate_record(rec)
        evlib.validate(rec)
    assert decl["decoding"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert decl["decoding"]["transport"].startswith("in-process")
    assert len(items) == 8 and all(it["cost_usd"] == pytest.approx(1.212 * it["latency_ms"] / 1000 / 3600, rel=1e-3)
                                   for it in items)
    lp = res["details"]["latency_and_price"]
    assert lp["n_calls"] == 8 and lp["p95_ms"] >= lp["p50_ms"] > 0 and "nearest-rank" in lp["percentile_method"]
    assert lp["cost_per_1k_usd"] > 0 and lp["instance_type"] == "ml.g5.2xlarge"
    assert len(lp["warmup_ms_excluded"]) == 2
    files = json.loads((tmp_path / "ev" / "outputs" / r["run_id"] / "model_files.json").read_text())
    assert "config.json" in files["model_files"]


def test_local_hf_needs_a_price(tmp_path):
    c, _, _ = _client(tmp_path)
    a = R.parser().parse_args(["--evidence-dir", str(tmp_path / "ev"), "--client", "local-hf", "--unit", "interface",
                               "--model-dir", str(tmp_path / "model"), "--model-id", "m", "--limit", "2"])
    with pytest.raises(SystemExit):
        R.run(a, client=c)


def test_baseline_with_another_chat_template_is_refused(tmp_path):
    a_client, _, _ = _client(tmp_path, "TEMPLATE-A")
    R.run(_args(tmp_path, "--unit", "interface", "--run-id", "t1.if-a", "--limit", "4"), client=a_client)
    b_client, _, _ = _client(tmp_path, "TEMPLATE-B")
    with pytest.raises(SystemExit, match="chat_template_sha256"):
        R.run(_args(tmp_path, "--unit", "interface", "--run-id", "t1.if-b", "--limit", "4",
                    "--baseline-run-id", "t1.if-a"), client=b_client)


def test_code_sha_override(tmp_path, monkeypatch):
    assert R.resolve_code_sha("a" * 40) == "a" * 40
    monkeypatch.setenv("CODE_SHA", "b" * 40)
    assert R.resolve_code_sha(None) == "b" * 40
    with pytest.raises(SystemExit):
        R.resolve_code_sha("not-a-sha")


# --------------------------------------------------------------------------- real transformers path

def _tiny_qwen3(tmp_path: Path) -> tuple[Path, Path]:
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    tok_dir = os.environ.get("T1_QWEN3_TOKENIZER_DIR")
    if not tok_dir or not Path(tok_dir, "tokenizer.json").is_file():
        pytest.skip("set T1_QWEN3_TOKENIZER_DIR to a directory with the Qwen3 tokenizer files")
    tok = transformers.AutoTokenizer.from_pretrained(tok_dir)
    tok.model_max_length = 1100  # as in the OTel base's tokenizer_config; must not cut prompts
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                                   max_position_embeddings=4096, tie_word_embeddings=True)
    model = transformers.Qwen3ForCausalLM(cfg)
    mdir = tmp_path / "model"
    model.save_pretrained(mdir)
    tok.save_pretrained(mdir)
    lora = peft.get_peft_model(model, peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                                      task_type="CAUSAL_LM"))
    adir = tmp_path / "adapter"
    lora.save_pretrained(adir)
    return mdir, adir


@pytest.mark.parametrize("with_adapter", [False, True])
def test_tiny_qwen3_through_the_cli(tmp_path, with_adapter, capsys):
    mdir, adir = _tiny_qwen3(tmp_path)
    argv = ["--unit", "knowledge", "--client", "local-hf", "--model-dir", str(mdir), "--model-id",
            "tiny-random-qwen3" + ("+lora" if with_adapter else ""), "--limit", "3", "--max-tokens", "8",
            "--warmup", "1", "--instance-price-per-hour", "1.212", "--price-source", "test price, not real",
            "--synthetic", "--evidence-dir", str(tmp_path / "ev"), "--run-id", f"t1.kn-tiny-{int(with_adapter)}"]
    if with_adapter:
        argv += ["--adapter-dir", str(adir)]
    assert R.main(argv) == 0
    store = evlib.LocalStore(tmp_path / "ev")
    rid = f"t1.kn-tiny-{int(with_adapter)}"
    decl = evlib.read_declared(rid, store=store)
    for rec in [decl, evlib.read_result(rid, store=store), *evlib.read_records(rid, "item", store=store)]:
        validate_record(rec)
        evlib.validate(rec)
    d = decl["decoding"]
    assert d["thinking_flag_honoured"] is True  # the Qwen3 template really changes with enable_thinking
    assert d["adapter"] is with_adapter and d["adapter_merged"] is with_adapter
    assert ("adapter_dir_sha256" in d) is with_adapter


def test_long_prompts_are_not_truncated(tmp_path):
    """Knowledge prompts reach ~2,000 tokens; the OTel tokenizer says model_max_length 1100."""
    mdir, _ = _tiny_qwen3(tmp_path)
    from t1.evals import freeze
    from t1.evals.prompting import render

    longest = max(freeze.load_items("knowledge"), key=lambda it: len(render(it)[1]))
    c = LocalHFClient(str(mdir), "tiny", warmup=0, hash_files=False)
    assert c.tok.model_max_length == 1100
    out = c.generate(*render(longest), 4)
    assert out.error is None and out.tokens_in > 1100
    assert c.decoding(4)["truncation"] is False and c.decoding(4)["model_max_length_ignored"] == 1100
