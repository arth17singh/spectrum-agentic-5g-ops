"""Stage the event base model: fp32 pickle shards -> bf16 safetensors, every file hashed.

CPU SageMaker job (ml.m5.2xlarge: 8 vCPU, 32 GiB; peak about 15 GiB). The input is the
server-side copy of the CHEDDAR prefix recorded in t1/sft/staging/base_copy.json:

    python -m t1.sft.stage_base --run-id R \\
        --src s3://<team-bucket>/models/base/otel-llm-1.7b-it/ \\
        --dst s3://<team-bucket>/models/base/otel-llm-1.7b-it-bf16/R/ \\
        --evidence s3://<team-bucket>/evidence/ --code-sha <sha> [--write-result] [--dry-run]

In order (a failed check stops the job before anything is uploaded):
  1. sha256 of every source file, next to its S3 ETag (the copy record only has ETags);
  2. the .bin index must name exactly the shards present; the shards are pickles, so they are read
     with torch.load(weights_only=True) (tensors only, no arbitrary code);
  3. transformers loads the model in bf16 and save_pretrained writes safetensors. This handles the
     tied lm_head/embed_tokens pair (tie_word_embeddings: true), which safetensors cannot store twice;
  4. every non-weight file (config, generation config, tokenizer, chat template, README) is copied
     byte for byte, so the staged directory differs from the source only in the weight format;
  5. verify every tensor the source index names: the output holds exactly its bf16 cast (lm_head:
     must equal embed_tokens in the source, then it is tied); no extra tensors; every output tensor is
     bf16. tensor_mismatch_count is the declared target (== 0);
  6. smoke: the staged copy loads with transformers, the chat template renders thinking off as the
     eval harness will, and greedy decoding returns tokens;
  7. SHA256SUMS, upload to --dst, stage_summary.json, then a model record through tools/evlib.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from t1.sft import common as K

UNIT = "t1.sft.stage"
INDEX = "pytorch_model.bin.index.json"
METRIC = "tensor_mismatch_count"
THINK_OFF_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _is_weight(name: str) -> bool:
    return name.endswith(".bin") or name == INDEX


# --------------------------------------------------------------------------- steps

def hash_sources(src_dir: Path, listing: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    etags = {o["name"]: o for o in (listing or [])}
    out = []
    for p in K.tree_files(src_dir):
        name = p.relative_to(src_dir).as_posix()
        t0 = time.time()
        sha = K.sha256_file(p)
        row = {"name": name, "size": p.stat().st_size, "sha256": sha}
        if name in etags:
            row["s3_etag"] = etags[name]["etag"]
            row["s3_last_modified"] = etags[name]["last_modified"]
        out.append(row)
        K.log(f"sha256 {name} {sha} ({row['size']} bytes, {time.time() - t0:.1f}s)")
    return out


def read_index(src_dir: Path) -> dict[str, Any]:
    p = src_dir / INDEX
    if not p.is_file():
        raise K.JobError(f"{INDEX} missing in {src_dir}")
    index = json.loads(p.read_text())
    shards = sorted(set(index["weight_map"].values()))
    present = sorted(q.name for q in src_dir.glob("*.bin"))
    if shards != present:
        raise K.JobError(f"index names shards {shards}, directory has {present}")
    return index


def load_source_tensors(src_dir: Path, index: dict[str, Any]) -> dict[str, Any]:
    import torch

    tensors: dict[str, Any] = {}
    for shard in sorted(set(index["weight_map"].values())):
        try:
            sd = torch.load(src_dir / shard, map_location="cpu", weights_only=True, mmap=True)
        except RuntimeError:  # an old-format (non-zip) checkpoint cannot be memory-mapped
            sd = torch.load(src_dir / shard, map_location="cpu", weights_only=True)
        for k, v in sd.items():
            if k in tensors:
                raise K.JobError(f"tensor {k} appears in more than one shard")
            tensors[k] = v
    want = set(index["weight_map"])
    if set(tensors) != want:
        raise K.JobError(f"shards hold {len(tensors)} tensors, index names {len(want)}: "
                         f"missing {sorted(want - set(tensors))[:5]}, extra {sorted(set(tensors) - want)[:5]}")
    return tensors


def convert(src_dir: Path, out_dir: Path) -> dict[str, Any]:
    """transformers bf16 load + save_pretrained(safetensors); returns the config it wrote."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(src_dir), dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if next(model.parameters()).dtype != torch.bfloat16:  # an older transformers ignores dtype=
        model = model.to(torch.bfloat16)
    model.save_pretrained(str(out_dir), safe_serialization=True, max_shard_size="5GB")
    written = json.loads((out_dir / "config.json").read_text())
    del model
    gc.collect()
    return written


def copy_non_weights(src_dir: Path, out_dir: Path) -> list[str]:
    """Byte copies of every non-weight source file (replacing the config save_pretrained wrote)."""
    copied = []
    for p in K.tree_files(src_dir):
        name = p.relative_to(src_dir).as_posix()
        if _is_weight(name):
            continue
        (out_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (out_dir / name).write_bytes(p.read_bytes())
        copied.append(name)
    return copied


def verify(src: dict[str, Any], out_dir: Path, index: dict[str, Any], tie: bool) -> dict[str, Any]:
    """Compare every tensor the source index names with the staged safetensors."""
    import torch
    from safetensors.torch import load_file

    out: dict[str, Any] = {}
    files = sorted(out_dir.glob("*.safetensors"))
    if not files:
        raise K.JobError("save_pretrained wrote no .safetensors file")
    for f in files:
        part = load_file(str(f))
        dup = set(part) & set(out)
        if dup:
            raise K.JobError(f"tensors in more than one safetensors file: {sorted(dup)[:5]}")
        out.update(part)
    problems: list[dict[str, str]] = []
    tied_in_source = None
    if "lm_head.weight" in src and "model.embed_tokens.weight" in src:
        tied_in_source = bool(torch.equal(src["lm_head.weight"], src["model.embed_tokens.weight"]))
    for name in sorted(index["weight_map"]):
        s = src[name]
        if name not in out:
            if name == "lm_head.weight" and tie and tied_in_source and "model.embed_tokens.weight" in out:
                continue  # stored once as embed_tokens; equal in the source, so nothing is lost
            problems.append({"tensor": name, "problem": "missing"})
            continue
        o = out[name]
        if o.dtype != torch.bfloat16:
            problems.append({"tensor": name, "problem": f"dtype {o.dtype}"})
        elif tuple(o.shape) != tuple(s.shape):
            problems.append({"tensor": name, "problem": f"shape {tuple(o.shape)} != {tuple(s.shape)}"})
        elif not torch.equal(s.to(torch.bfloat16), o):
            problems.append({"tensor": name, "problem": "values differ from the bf16 cast"})
    for name in sorted(set(out) - set(index["weight_map"])):
        problems.append({"tensor": name, "problem": "extra"})
    return {
        "n_tensors_source": len(index["weight_map"]),
        "n_tensors_output": len(out),
        "source_dtypes": dict(Counter(str(t.dtype) for t in src.values())),
        "output_dtypes": dict(Counter(str(t.dtype) for t in out.values())),
        "lm_head_tied_in_source": tied_in_source,
        METRIC: len(problems),
        "problems": problems[:50],
        "safetensors_files": [f.name for f in files],
    }


def smoke(out_dir: Path, max_new_tokens: int = 8) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(out_dir))
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Reply with one word: ready"}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if not text.endswith(THINK_OFF_SUFFIX):
        raise K.JobError(f"chat template with enable_thinking=False does not end with the empty think block: "
                         f"{text[-80:]!r}")
    model = AutoModelForCausalLM.from_pretrained(str(out_dir), dtype=torch.bfloat16)
    model.eval()
    enc = tok(text, return_tensors="pt")
    t0 = time.time()
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    new = gen[0, enc["input_ids"].shape[1]:]
    return {"prompt_tokens": int(enc["input_ids"].shape[1]), "new_tokens": int(new.shape[0]),
            "text": tok.decode(new, skip_special_tokens=False), "seconds": round(time.time() - t0, 2),
            "model_dtype": str(next(model.parameters()).dtype), "think_off_suffix_ok": True}


# --------------------------------------------------------------------------- records

def model_record(run_id: str, dst: str, artifact_sha: str, summary: dict[str, Any], aws_time: str | None) -> dict:
    return {
        "run_id": run_id,
        "model_id": K.BASE_MODEL_ID,
        "version": run_id,
        "model_kind": "llm-lora",
        "artifact_uri": dst,
        "artifact_sha256": artifact_sha,
        "status": "candidate",
        "trained_on": f"{K.BASE_HF_REPO}@{K.BASE_REVISION} (event base model, CHEDDAR copy); staged to bf16, "
                      "not trained",
        "aws_time": aws_time,
        "aws_time_source": "s3:last_modified (SHA256SUMS)" if aws_time else None,
        "metrics": {
            "note": "BASE MODEL, NO ADAPTER. model_kind is llm-lora only because the contract's enum has no "
                    "llm-base value.",
            "artifact_sha256_is": "sha256 of the SHA256SUMS file at artifact_uri (one line per file)",
            "hf_repo": K.BASE_HF_REPO,
            "revision": K.BASE_REVISION,
            "dtype": "bfloat16",
            METRIC: summary["verify"][METRIC],
            "n_tensors": summary["verify"]["n_tensors_source"],
            "files": summary["output_files"],
            "source_files": summary["source_files"],
            "smoke": summary.get("smoke"),
            "versions": summary["versions"],
            "code_sha": summary.get("code_sha"),
        },
    }


def result_record(run_id: str, code_sha: str, summary: dict[str, Any]) -> dict:
    v = summary["verify"][METRIC]
    return {
        "run_id": run_id,
        "unit": UNIT,
        "metric": METRIC,
        "value": v,
        "target": {"metric": METRIC, "op": "==", "value": 0, "unit": "count"},
        "passed": v == 0,
        "n_items": summary["verify"]["n_tensors_source"],
        "checker": {"script": "t1/sft/stage_base.py", "code_sha": code_sha},
        "details": {"problems": summary["verify"]["problems"], "artifact_uri": summary["dst"]},
    }


# --------------------------------------------------------------------------- main

def run(a: argparse.Namespace) -> dict[str, Any]:
    run_id = K.check_run_id(a.run_id)
    code_sha = K.code_sha_from_env(a.code_sha)
    t_start = time.time()
    work = Path(a.workdir or tempfile.mkdtemp(prefix="stage-"))
    src_dir, listing = K.materialise(a.src, work, "src", channel_env="SM_CHANNEL_BASE", region=a.region)
    K.log(f"source at {src_dir} ({'downloaded' if listing else 'local'})")
    timings: dict[str, float] = {"download_s": round(time.time() - t_start, 1)}

    t = time.time()
    source_files = hash_sources(src_dir, listing)
    timings["hash_source_s"] = round(time.time() - t, 1)
    index = read_index(src_dir)
    src_config = json.loads((src_dir / "config.json").read_text())
    tie = bool(src_config.get("tie_word_embeddings", False))

    out_dir = work / "out"
    out_dir.mkdir(parents=True, exist_ok=False)
    t = time.time()
    written_config = convert(src_dir, out_dir)
    timings["convert_s"] = round(time.time() - t, 1)
    config_diff = sorted(k for k in set(written_config) | set(src_config)
                         if written_config.get(k) != src_config.get(k))
    copied = copy_non_weights(src_dir, out_dir)

    t = time.time()
    src_tensors = load_source_tensors(src_dir, index)
    ver = verify(src_tensors, out_dir, index, tie)
    del src_tensors
    gc.collect()
    timings["verify_s"] = round(time.time() - t, 1)
    K.log(f"verify: {ver[METRIC]} problem(s) over {ver['n_tensors_source']} tensors")
    if ver[METRIC] != 0 and not a.allow_mismatch:
        raise K.JobError(f"{ver[METRIC]} tensor problem(s): {ver['problems'][:5]}")

    smoke_out = None
    if not a.skip_smoke:
        t = time.time()
        smoke_out = smoke(out_dir)
        timings["smoke_s"] = round(time.time() - t, 1)
        K.log(f"smoke: {smoke_out}")

    output_files, artifact_sha = K.write_sha256sums(out_dir)
    summary: dict[str, Any] = {
        "kind": "t1-sft-stage-summary",
        "run_id": run_id,
        "code_sha": code_sha,
        "src": a.src or os.environ.get("SM_CHANNEL_BASE"),
        "dst": a.dst,
        "hf_repo": K.BASE_HF_REPO,
        "revision": K.BASE_REVISION,
        "source_files": source_files,
        "output_files": output_files,
        "artifact_sha256": artifact_sha,
        "copied_non_weight_files": copied,
        "config_keys_save_pretrained_changed": config_diff,
        "verify": ver,
        "smoke": smoke_out,
        "versions": K.versions("torch", "transformers", "safetensors", "accelerate", "boto3"),
    }

    t = time.time()
    K.upload_tree(out_dir, a.dst, region=a.region)
    timings["upload_s"] = round(time.time() - t, 1)
    aws_time = K.s3_last_modified(a.dst.rstrip("/") + "/SHA256SUMS", region=a.region) if K.is_s3(a.dst) else None
    summary["aws_time"] = aws_time
    timings["total_s"] = round(time.time() - t_start, 1)
    summary["timings"] = timings
    summary_dir = work / "summary"
    summary_sha = K.write_json(summary_dir / "stage_summary.json", summary)
    K.upload_tree(summary_dir, a.dst, region=a.region)  # beside the model, outside SHA256SUMS
    summary["summary_sha256"] = summary_sha
    K.log(f"uploaded to {a.dst}; SHA256SUMS sha256 {artifact_sha}; summary sha256 {summary_sha}")

    store = K.evidence_store(a.evidence)
    if store is None:
        K.log("no evidence store (pass --evidence or set EVIDENCE_S3): model record not written")
    else:
        from tools import evlib

        rec = model_record(run_id, a.dst, artifact_sha, summary, aws_time)
        rec["metrics"]["summary_sha256"] = summary_sha
        K.log(f"model record: {evlib.write_model(rec, store=store)}")
        if a.write_result:
            if not code_sha:
                raise K.JobError("--write-result needs --code-sha (or CODE_SHA)")
            K.log(f"result record: {evlib.write_result(result_record(run_id, code_sha, summary), store=store)}")
    return summary


def dry_run(a: argparse.Namespace) -> dict[str, Any]:
    """No AWS call, no torch: check the arguments and print what the job would do."""
    run_id = K.check_run_id(a.run_id)
    plan: dict[str, Any] = {
        "job": "t1.sft.stage_base",
        "run_id": run_id,
        "code_sha": K.code_sha_from_env(a.code_sha),
        "src": a.src or "(SageMaker channel SM_CHANNEL_BASE)",
        "dst": a.dst,
        "evidence": a.evidence or os.environ.get("EVIDENCE_S3") or os.environ.get("EVIDENCE_DIR"),
        "steps": ["sha256 every source file (+ S3 ETag)", "read the .bin index and shards (weights_only)",
                  "transformers bf16 load + save_pretrained(safetensors)", "byte-copy non-weight files",
                  f"verify every tensor; declared target {METRIC} == 0", "smoke: chat template thinking off + "
                  "greedy decode", "SHA256SUMS, upload, stage_summary.json", "model record (evlib)"
                  + (" + result record" if a.write_result else "")],
        "target": {"metric": METRIC, "op": "==", "value": 0, "unit": "count"},
    }
    if a.src and not K.is_s3(a.src) and Path(a.src, INDEX).is_file():
        index = json.loads(Path(a.src, INDEX).read_text())
        plan["n_tensors"] = len(index["weight_map"])
        plan["shards"] = sorted(set(index["weight_map"].values()))
    if not a.dst:
        raise K.JobError("--dst is required")
    return plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m t1.sft.stage_base", description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--src", help="s3:// prefix or local dir of the copied base model (else SM_CHANNEL_BASE)")
    ap.add_argument("--dst", required=True, help="s3:// prefix or local dir for the bf16 safetensors copy")
    ap.add_argument("--evidence", help="evidence store (s3:// prefix or dir); else EVIDENCE_S3 / EVIDENCE_DIR")
    ap.add_argument("--code-sha", help="commit the job runs (else CODE_SHA)")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", K.DEFAULT_REGION))
    ap.add_argument("--workdir", help="scratch directory (default: a new temp dir)")
    ap.add_argument("--write-result", action="store_true", help="also write results/<run_id>.json")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--allow-mismatch", action="store_true", help="upload even if verification finds problems")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    try:
        out = dry_run(a) if a.dry_run else run(a)
    except K.JobError as e:
        K.log(f"FAILED: {e}")
        return 2
    if a.dry_run:
        print(K.dumps_doc(out), end="")
    else:
        K.log(f"done: {METRIC}={out['verify'][METRIC]} artifact_sha256={out['artifact_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
