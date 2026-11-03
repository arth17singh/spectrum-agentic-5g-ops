"""Chat rendering (thinking off), the trainer's and stager's pure helpers, the runner declarations and the
exact runner commands (put through tools/sm/run.py's own plan, offline)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from t1.sft import chat, common, declare, launch, stage_base, train_lora
from t1.sft.testing import MINI_TEMPLATE, FakeTokenizer
from tools import check_unit, evlib

FAKE_BUCKET = "bb26-p-team13-data-000000000000-use2"  # the runner tests' fake account; never a real id here
MSGS = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"},
        {"role": "assistant", "content": '{"decision": "refuse", "change": "", "reason": "r"}'}]


def test_split_text_puts_the_empty_think_block_in_the_target():
    render = chat.jinja_renderer(MINI_TEMPLATE)
    prompt, target = chat.split_text(render, MSGS)
    assert prompt.endswith("<|im_start|>assistant\n")
    assert target == chat.THINK_EMPTY + MSGS[2]["content"] + chat.END
    assert (prompt + chat.THINK_EMPTY) == render(MSGS[:2], add_generation_prompt=True, enable_thinking=False)


def test_a_template_without_the_thinking_off_block_is_refused():
    broken = MINI_TEMPLATE.replace("{{- '<think>\\n\\n</think>\\n\\n' -}}", "")
    with pytest.raises(chat.ChatFormatError):
        chat.split_text(chat.jinja_renderer(broken), MSGS)


def test_encode_masks_the_prompt_and_ends_on_im_end():
    tok = FakeTokenizer()
    e = chat.encode(tok, MSGS)
    assert e["labels"][: e["n_prompt"]] == [-100] * e["n_prompt"]
    assert e["labels"][e["n_prompt"]:] == e["input_ids"][e["n_prompt"]:]
    assert e["input_ids"][-1] == tok.convert_tokens_to_ids("<|im_end|>")
    assert e["input_ids"][e["n_prompt"]] == tok.convert_tokens_to_ids("<think>")


@pytest.mark.skipif(not os.environ.get("T1_SFT_CHAT_TEMPLATE"), reason="set T1_SFT_CHAT_TEMPLATE to the base "
                    "model's chat_template.jinja (an organiser file, never in git) to check it too")
def test_real_template_invariants():
    from t1.sft import templates
    from t1.sft.build import to_row

    render = chat.jinja_renderer(Path(os.environ["T1_SFT_CHAT_TEMPLATE"]).read_text())
    for ex in templates.interface_examples(20) + templates.safety_examples(14):
        chat.split_text(render, to_row(ex, "train")["messages"])


# --------------------------------------------------------------------------- trainer helpers

def test_batches_cover_every_example_once_and_are_seeded():
    import random

    ex = [{"input_ids": [0] * (i % 37 + 1), "labels": [1], "unit": "u", "id": i} for i in range(203)]
    b1 = train_lora.batches(ex, 8, random.Random(13))
    b2 = train_lora.batches(ex, 8, random.Random(13))
    assert [[e["id"] for e in b] for b in b1] == [[e["id"] for e in b] for b in b2]
    assert sorted(e["id"] for b in b1 for e in b) == list(range(203))


def test_collate_pads_right_and_masks_labels():
    torch = pytest.importorskip("torch")
    b = train_lora.collate([{"input_ids": [5, 6, 7], "labels": [-100, 6, 7]}, {"input_ids": [9], "labels": [9]}],
                           pad_id=0, device=torch.device("cpu"))
    assert b["input_ids"].tolist() == [[5, 6, 7], [9, 0, 0]]
    assert b["labels"].tolist() == [[-100, 6, 7], [9, -100, -100]]
    assert b["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def test_verify_sums_catches_a_changed_file(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    common.write_sha256sums(tmp_path)
    train_lora.verify_sums(tmp_path)
    (tmp_path / "b.txt").write_text("B")
    with pytest.raises(common.JobError, match="sha256"):
        train_lora.verify_sums(tmp_path)


def test_dev_items_score_final_over_base_and_recompute(tmp_path):
    from t1.sft import templates
    from t1.sft.build import to_row

    rows = [to_row(ex, "dev") for ex in templates.interface_examples(4)]
    dev_ex = [{"id": r["id"], "n_target": 30} for r in rows]
    base = {r["id"]: 2.0 for r in rows}
    final = {r["id"]: v for r, v in zip(rows, (0.2, 0.4, 1.0, 2.2))}
    items = train_lora.dev_item_records(rows, dev_ex, base, final, "test")
    assert [round(i["score"], 6) for i in items] == [0.1, 0.2, 0.5, 1.1]
    assert [i["passed"] for i in items] == [True, True, True, False]
    decl = {"run_id": "t1.sft.train-x", "unit": "t1.sft.train", "target": {"metric": "mean_score", "op": "<=",
                                                                          "value": 0.5, "unit": "ratio"}}
    res = common.write_evidence(tmp_path, decl, items, "t1/sft/train_lora.py", "a" * 40, {})
    assert res["value"] == pytest.approx(check_unit.recompute("mean_score", items)) == pytest.approx(0.475)
    assert res["passed"] is True and res["n_items"] == 4


def test_train_dry_run_checks_the_chat_rendering(tmp_path):
    from t1.sft import build

    a = build.parse_args(["--run-id", "t1.sft.data-dry", "--dry-run", "--workdir", str(tmp_path / "w")])
    s = build.run(a)
    tpl = tmp_path / "chat_template.jinja"
    tpl.write_text(MINI_TEMPLATE)
    plan = train_lora.dry_run(train_lora.parse_args(["--run-id", "t1.sft.train-dry", "--dry-run", "--data",
                                                     s["local_dir"], "--out", str(tmp_path / "o"),
                                                     "--chat-template", str(tpl)]))
    assert plan["chat_rendering_checked"] == s["counts"]["by_split"]["train"]
    assert plan["target"] == {"metric": "mean_score", "op": "<=", "value": 0.5, "unit": "ratio"}


# --------------------------------------------------------------------------- stage helpers

def test_stage_read_index_and_dry_run(tmp_path):
    idx = {"metadata": {"total_size": 8}, "weight_map": {"a.weight": "pytorch_model-00001-of-00001.bin"}}
    (tmp_path / stage_base.INDEX).write_text(json.dumps(idx))
    with pytest.raises(common.JobError, match="shards"):
        stage_base.read_index(tmp_path)
    (tmp_path / "pytorch_model-00001-of-00001.bin").write_bytes(b"x")
    assert stage_base.read_index(tmp_path) == idx
    plan = stage_base.dry_run(stage_base.parse_args(["--run-id", "t1.sft.stage-dry", "--dry-run", "--src",
                                                     str(tmp_path), "--dst", str(tmp_path / "out"),
                                                     "--evidence-dir", str(tmp_path / "ev")]))
    assert plan["n_tensors"] == 1


def test_stage_verify_items_per_tensor_with_tied_lm_head(tmp_path):
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(0)
    emb = torch.randn(6, 4, generator=g)
    src = {"model.embed_tokens.weight": emb, "lm_head.weight": emb.clone(), "w1": torch.randn(3, 4, generator=g),
           "w2": torch.randn(4, generator=g)}
    index = {"weight_map": {k: "s.bin" for k in src}}
    out = {"model.embed_tokens.weight": emb.to(torch.bfloat16), "w1": src["w1"].to(torch.bfloat16),
           "w2": (src["w2"] + 1).to(torch.bfloat16)}
    ver = stage_base.verify(src, out, index, tie=True)
    by = {r["tensor"]: r for r in ver["rows"]}
    assert by["lm_head.weight"]["passed"] and by["lm_head.weight"]["stored_as"].endswith("(tied)")
    assert by["w1"]["passed"] and not by["w2"]["passed"] and by["w2"]["problem"].startswith("values differ")
    assert by["w1"]["source_sha256"] != by["w1"]["output_sha256"]  # fp32 bytes vs bf16 bytes
    items = stage_base.item_records(ver, "test")
    assert len(items) == 4 and check_unit.recompute("fail_count", items) == 1
    out["extra.weight"] = torch.zeros(1, dtype=torch.bfloat16)
    ver2 = stage_base.verify(src, out, index, tie=True)
    items2 = stage_base.item_records(ver2, "test")
    assert len(items2) == 5 and check_unit.recompute("fail_count", items2) == 2


def test_stage_model_record_validates(tmp_path):
    summary = {"verify": {"failed": 0, "n_tensors_source": 311},
               "output_files": {"model.safetensors": {"size": 1, "sha256": "0" * 64}},
               "source_files": [{"name": "config.json", "size": 1, "sha256": "1" * 64}],
               "versions": {"python": "3.12"}, "code_sha": "a" * 40}
    rec = stage_base.model_record("t1.sft.stage-x", "s3://example-bucket/x/", "f" * 64, summary, None)
    evlib.write_model(rec, store=evlib.LocalStore(tmp_path))
    assert (tmp_path / "models" / "otel-llm-1.7b-it" / "t1.sft.stage-x.json").is_file()


def test_upload_tree_refuses_to_overwrite_locally(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "f.txt").write_text("x")
    common.upload_tree(src, str(tmp_path / "dst"))
    with pytest.raises(common.JobError, match="refusing to overwrite"):
        common.upload_tree(src, str(tmp_path / "dst"))


def test_the_input_must_be_the_declared_test_set():
    common.check_test_set({"test_set_sha256": "a" * 64}, "a" * 64, "x")
    with pytest.raises(common.JobError, match="test_set_sha256"):
        common.check_test_set({"test_set_sha256": "a" * 64}, "b" * 64, "x")


def test_build_refuses_eval_items_other_than_the_declared_ones(tmp_path):
    from t1.sft import build

    d = declare.through_runner(declare.load("build"), run_id="t1.sft.data-ts")
    d["test_set_sha256"] = "0" * 64
    p = tmp_path / "d.json"
    p.write_text(json.dumps(d))
    a = build.parse_args(["--run-id", "t1.sft.data-ts", "--out", str(tmp_path / "o"), "--evidence-dir",
                          str(tmp_path / "e"), "--declared", str(p), "--n-execution", "0"])
    with pytest.raises(common.JobError, match="test_set_sha256"):
        build.run(a)
    assert not (tmp_path / "o").exists()


def test_stored_declaration_must_name_the_run(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"run_id": "a"}))
    assert common.stored_declaration("a", str(p))["run_id"] == "a"
    with pytest.raises(common.JobError):
        common.stored_declaration("b", str(p))


# --------------------------------------------------------------------------- runner declarations and commands

@pytest.fixture
def clean_tree(monkeypatch):
    from tools.sm import package, spec

    monkeypatch.setattr(package, "tree_dirty", lambda repo: [])
    monkeypatch.delenv(spec.BUCKET_ENV, raising=False)  # the runner derives the bucket from --account


def test_static_declarations_pass_the_runners_own_builder():
    st = declare.through_runner(declare.load("stage_base"))
    assert st["target"] == {"metric": "fail_count", "op": "==", "value": 0, "unit": "count"} and st["n_items"] == 311
    bd = declare.through_runner(declare.load("build"))
    assert bd["test_set_sha256"] == "4af94c300ff312eb69122b2a9644bc6cd4ecb7ccf3cc298bedee93932d2250c5"  # 482fe5f
    assert bd["prompt_sha256"] == "bca9ecf54c3b9ac64c4f3a3d036e427f95605cf40c344b3357379c5a20a0629f"
    assert bd["n_items"] == 1220 and bd["twin"]["kind"] == "maveric-inprocess"


def _manifest(run_id="t1.sft.data-x"):
    return {"run_id": run_id, "dry_run": False, "files": {"dev.jsonl": {"sha256": "d" * 64}},
            "counts": {"written": {"by_split": {"dev": 421, "train": 3779}}},
            "overlap": {"final_check": {"eval_overlap_count": 0}}}


def test_train_declaration_is_filled_from_the_dataset_manifest():
    d = declare.train_declaration("t1.sft.stage-x", "t1.sft.data-x", _manifest(), "e" * 64)
    assert d["n_items"] == 421 and d["test_set_sha256"] == "d" * 64 and "baseline_run_id" not in d
    assert d["decoding"]["inputs"]["base_run_id"] == "t1.sft.stage-x"  # the base weights' run, not an eval baseline
    assert "subset" not in d and d["decoding"]["inputs"]["data_manifest_sha256"] == "e" * 64
    r = declare.through_runner(d)
    assert r["target"]["metric"] == "mean_score"
    s = declare.train_declaration("t1.sft.stage-x", "t1.sft.data-x", _manifest(), "e" * 64, smoke=True)
    assert s["n_items"] == 48 and s["target"]["op"] == "<" and "--limit-dev" in s["subset"]
    declare.through_runner(s)
    with pytest.raises(common.JobError):
        declare.train_declaration("t1.sft.stage-x", "t1.sft.data-y", _manifest(), "e" * 64)
    bad = _manifest()
    bad["overlap"]["final_check"]["eval_overlap_count"] = 1
    with pytest.raises(common.JobError):
        declare.train_declaration("t1.sft.stage-x", "t1.sft.data-x", bad, "e" * 64)


@pytest.mark.parametrize("job", ["stage_base", "build", "train_lora", "recheck"])
def test_runner_plans_each_job_offline(job, tmp_path, clean_tree, monkeypatch):
    from tools.sm import run as R

    monkeypatch.chdir(launch.REPO)  # the printed declaration paths are repo-relative

    decl = None
    if job == "train_lora":
        decl = tmp_path / "train.json"
        decl.write_text(json.dumps(declare.train_declaration("t1.sft.stage-x", "t1.sft.data-x", _manifest(), "e" * 64)))
    if job == "recheck":
        decl = tmp_path / "recheck.json"
        m = json.dumps({"run_id": "t1.sft.data-x"}).encode()
        import hashlib

        sums = f"{hashlib.sha256(m).hexdigest()}  manifest.json\n".encode()
        decl.write_text(json.dumps(declare.recheck_declaration("t1.sft.data-x", sums, m)))
    args = launch.runner_args(job, f"t1.sft.{job}-test", FAKE_BUCKET, stage_run_id="t1.sft.stage-x",
                              data_run_id="t1.sft.data-x", declaration=str(decl) if decl else None, profile=None)
    a = R.parser().parse_args([*args, "--account", "000000000000"])
    p = R.plan(a, None)
    entry = " ".join(p["argv"])
    for ph in ("{run_id}", "{evidence_dir}", "{output_dir}"):
        assert ph in entry  # filled in the job, not here
    chans = dict(p["channels"])
    if job == "stage_base":
        assert p["device"] == "cpu" and a.instance_type == "ml.m5.2xlarge" and not p["sdk"]
        assert chans == {"model": f"s3://{FAKE_BUCKET}/models/base/otel-llm-1.7b-it/"}
        assert "{channel:model}" in entry and "transformers==4.57.6" in p["requirements"]
    elif job == "build":
        assert p["device"] == "cpu" and a.instance_type == "ml.c5.4xlarge" and p["sdk"]
        assert "sdk" in chans and "ran-rl[maveric]" in p["requirements"]
        assert p["declaration"]["twin"]["config"]["runner"]["instance_type"] == "ml.c5.4xlarge"
    elif job == "recheck":
        assert p["device"] == "cpu" and a.instance_type == "ml.m5.xlarge" and not p["sdk"] and a.max_run == 1200
        assert chans == {"data": f"s3://{FAKE_BUCKET}/datasets/t1-sft/t1.sft.data-x/"}
        assert "--data-run-id t1.sft.data-x" in entry and "pydantic" in p["requirements"]
        assert p["declaration"]["test_set_sha256"] == "4af94c300ff312eb69122b2a9644bc6cd4ecb7ccf3cc298bedee93932d2250c5"
    else:
        assert p["device"] == "gpu" and a.instance_type == "ml.g6.2xlarge" and not p["sdk"]
        assert chans == {"model": f"s3://{FAKE_BUCKET}/models/base/otel-llm-1.7b-it-bf16/t1.sft.stage-x/",
                         "data": f"s3://{FAKE_BUCKET}/datasets/t1-sft/t1.sft.data-x/"}
        assert "peft" in p["requirements"] and a.max_run == 7200
    assert f"t1/sft/{job}.py" in p["files"]


def test_launch_prints_dry_run_and_no_wait_commands(capsys, monkeypatch):
    from tools.sm import spec

    monkeypatch.setenv(spec.BUCKET_ENV, FAKE_BUCKET)  # no STS call in tests
    assert launch.main(["stage_base", "--run-id", "t1.sft.stage-test"]) == 0
    out = capsys.readouterr().out
    assert "--dry-run" in out and "--no-wait" in out and "--finish t1.sft.stage-test" in out
    assert f"s3://{FAKE_BUCKET}/models/base/otel-llm-1.7b-it/" in out
    assert launch.main(["train_lora", "--run-id", "t1.sft.train-test"]) == 2  # needs stage/data/declaration
    assert launch.main(["recheck", "--run-id", "t1.sft.data-recheck-test"]) == 2  # needs data run + declaration


def test_bucket_comes_from_the_flag_the_environment_or_sts(monkeypatch):
    from tools.sm import spec

    assert launch.bucket("explicit-bucket", None) == "explicit-bucket"
    monkeypatch.setenv(spec.BUCKET_ENV, "env-bucket")
    assert launch.bucket(None, None) == "env-bucket"


def test_committed_files_hold_no_organiser_bucket():
    for p in list(declare.JOBS_DIR.glob("*.json")) + [Path(declare.__file__).with_name("staging") / "base_copy.json"]:
        text = p.read_text()
        assert not re.search(r"\b\d{12}\b", text), p
        assert "fyuz-otel" not in text and "bb26-p-team13" not in text, p
