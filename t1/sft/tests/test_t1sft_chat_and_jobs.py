"""Chat rendering invariants (thinking off), the trainer's pure helpers, stage helpers and the job templates."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from t1.sft import chat, common, declare, stage_base, train_lora
from t1.sft.testing import MINI_TEMPLATE, FakeTokenizer

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

    render = chat.jinja_renderer(Path(os.environ["T1_SFT_CHAT_TEMPLATE"]).read_text())
    from t1.sft.build import to_row

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
    assert plan["target"] == {"metric": "dev_loss_ratio", "op": "<=", "value": 0.5, "unit": "ratio"}


# --------------------------------------------------------------------------- stage helpers

def test_stage_read_index_and_dry_run(tmp_path):
    idx = {"metadata": {"total_size": 8}, "weight_map": {"a.weight": "pytorch_model-00001-of-00001.bin"}}
    (tmp_path / stage_base.INDEX).write_text(json.dumps(idx))
    with pytest.raises(common.JobError, match="shards"):
        stage_base.read_index(tmp_path)
    (tmp_path / "pytorch_model-00001-of-00001.bin").write_bytes(b"x")
    assert stage_base.read_index(tmp_path) == idx
    plan = stage_base.dry_run(stage_base.parse_args(["--run-id", "t1.sft.stage-dry", "--dry-run", "--src",
                                                     str(tmp_path), "--dst", str(tmp_path / "out")]))
    assert plan["n_tensors"] == 1 and plan["target"]["metric"] == "tensor_mismatch_count"


def test_stage_records_validate(tmp_path):
    from tools import evlib

    summary = {"verify": {"tensor_mismatch_count": 0, "n_tensors_source": 311, "problems": []},
               "output_files": {"model.safetensors": {"size": 1, "sha256": "0" * 64}},
               "source_files": [{"name": "config.json", "size": 1, "sha256": "1" * 64}],
               "versions": {"python": "3.12"}, "dst": "s3://example-bucket/x/", "code_sha": "a" * 40}
    rec = stage_base.model_record("t1.sft.stage-x", "s3://example-bucket/x/", "f" * 64, summary, None)
    evlib.validate(dict(rec, kind="model", schema_version="1", team="team13", created_at=evlib.utc_now(),
                        aws_time=None, aws_time_source=None, synthetic=False))
    res = stage_base.result_record("t1.sft.stage-x", "a" * 40, summary)
    evlib.validate(dict(res, kind="result", schema_version="1", team="team13", created_at=evlib.utc_now(),
                        checked_at=evlib.utc_now(), aws_time=None, aws_time_source=None, synthetic=False))
    store = evlib.LocalStore(tmp_path)
    evlib.write_model(rec, store=store)
    assert (tmp_path / "models" / "otel-llm-1.7b-it" / "t1.sft.stage-x.json").is_file()


def test_upload_tree_refuses_to_overwrite_locally(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "f.txt").write_text("x")
    common.upload_tree(src, str(tmp_path / "dst"))
    with pytest.raises(common.JobError, match="refusing to overwrite"):
        common.upload_tree(src, str(tmp_path / "dst"))


# --------------------------------------------------------------------------- job templates

VALUES = {"run_id": "t1.sft.x-20261103T000000Z-abcd", "code_sha": "a" * 40, "bucket": "example-bucket",
          "stage_run_id": "t1.sft.stage-20261103T000000Z-0001", "data_run_id": "t1.sft.data-20261103T000000Z-0002",
          "n_dev": 380, "dev_jsonl_sha256": "b" * 64, "data_manifest_sha256": "c" * 64}


@pytest.mark.parametrize("job", declare.JOBS)
def test_declaration_templates_validate_when_filled(job):
    rec = declare.declaration(job, VALUES)
    assert rec["run_id"] == VALUES["run_id"] and rec["code_sha"] == VALUES["code_sha"]
    assert rec["track"] == "track-1" and rec["region"] == "us-east-2"
    spec = declare.job_spec(job, VALUES)
    assert not declare.placeholders(spec)
    assert spec["args"][:2] == ["--run-id", VALUES["run_id"]]
    assert spec["entry"][:3] == ["python", "-m", f"t1.sft.{job}"]


def test_unfilled_placeholders_are_an_error():
    with pytest.raises(common.JobError, match="unfilled"):
        declare.declaration("train_lora", {"run_id": "r", "code_sha": "a" * 40})


def test_templates_hold_no_account_ids_or_organiser_bucket():
    for p in list(declare.JOBS_DIR.glob("*.json")) + [Path(declare.__file__).with_name("staging") / "base_copy.json"]:
        text = p.read_text()
        assert not re.search(r"\b\d{12}\b(?!\.dkr)", text.replace("763104351884.dkr", "")), p
        assert "fyuz-otel" not in text and "bb26-p-team13" not in text, p
