"""Stage the event base model: fp32 pickle shards -> bf16 safetensors, every file and tensor hashed.

Runs as a CPU SageMaker job through the team runner (tools/sm/run.py; ml.m5.2xlarge, 8 vCPU, 32 GiB,
peak about 15 GiB). The copied CHEDDAR prefix (t1/sft/staging/base_copy.json) arrives as the input
channel `model`:

    --entry "-m t1.sft.stage_base --run-id {run_id} --src {channel:model}
             --src-uri s3://<team-bucket>/models/base/otel-llm-1.7b-it/
             --dst s3://<team-bucket>/models/base/otel-llm-1.7b-it-bf16/{run_id}/
             --evidence-dir {evidence_dir} --summary-dir {output_dir}"

In order (nothing is uploaded if a check fails, but the evidence of the failure is still written):
  1. sha256 of every source file, next to its S3 ETag (the copy record only has ETags);
  2. the .bin index must name exactly the shards present; the shards are pickles, so they are read
     with torch.load(weights_only=True) (tensors only, no arbitrary code);
  3. transformers loads the model in bf16 and save_pretrained writes safetensors. This handles the
     tied lm_head/embed_tokens pair (tie_word_embeddings: true), which safetensors cannot store twice;
  4. every non-weight file (config, generation config, tokenizer, chat template, README) is copied
     byte for byte, so the staged directory differs from the source only in the weight format;
  5. one item record per tensor the source index names (311): passed when the output holds exactly
     its bf16 cast (lm_head: equal to embed_tokens in the source, stored once, tied), with the sha256
     of the source tensor's bytes and of the output tensor's bytes. Declared target: fail_count == 0;
  6. smoke: the staged copy loads with transformers, the chat template renders thinking off as the
     eval harness will, and greedy decoding returns tokens;
  7. SHA256SUMS, upload to --dst as plain files (W12's eval jobs and the LoRA job mount the prefix as
     their `model` channel), stage_summary.json beside it, a model record (EVIDENCE_S3), and the items
     and result into --evidence-dir for the runner to upload and attest.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
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
METRIC = "fail_count"
GRADER = "t1/sft/stage_base.py:verify (output tensor == bf16 cast of the source tensor; lm_head tied to embed_tokens)"
THINK_OFF_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _is_weight(name: str) -> bool:
    return name.endswith(".bin") or name == INDEX


def tensor_sha256(t: Any) -> str:
    """sha256 of a tensor's raw bytes in memory order (dtype-specific: fp32 and bf16 differ)."""
    import torch

    t = t.detach().contiguous()
    b = t if t.dtype == torch.uint8 else t.view(torch.uint8)
    return hashlib.sha256(memoryview(b.numpy())).hexdigest()


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
            if etags[name]["size"] != row["size"]:
                raise K.JobError(f"{name}: {row['size']} bytes on disk, {etags[name]['size']} in S3")
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


def load_output_tensors(out_dir: Path) -> tuple[dict[str, Any], list[str]]:
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
    return out, [f.name for f in files]


def verify(src: dict[str, Any], out: dict[str, Any], index: dict[str, Any], tie: bool) -> dict[str, Any]:
    """One row per tensor the source index names: passed, the problem if any, and both tensors' sha256."""
    import torch

    tied_in_source = None
    if "lm_head.weight" in src and "model.embed_tokens.weight" in src:
        tied_in_source = bool(torch.equal(src["lm_head.weight"], src["model.embed_tokens.weight"]))
    rows = []
    for name in sorted(index["weight_map"]):
        s = src[name]
        stored_as, problem, o = name, None, out.get(name)
        if o is None and name == "lm_head.weight" and tie and "model.embed_tokens.weight" in out:
            stored_as, o = "model.embed_tokens.weight (tied)", out["model.embed_tokens.weight"]
            if not tied_in_source:
                problem = "lm_head differs from embed_tokens in the source, so tying it loses weights"
        if o is None:
            problem = "missing from the output"
        elif problem is None:
            if o.dtype != torch.bfloat16:
                problem = f"output dtype {o.dtype}"
            elif tuple(o.shape) != tuple(s.shape):
                problem = f"shape {tuple(o.shape)} != {tuple(s.shape)}"
            elif not torch.equal(s.to(torch.bfloat16), o):
                problem = "values differ from the bf16 cast"
        rows.append({"tensor": name, "passed": problem is None, "problem": problem, "stored_as": stored_as,
                     "shape": list(s.shape), "source_dtype": str(s.dtype),
                     "output_dtype": None if o is None else str(o.dtype),
                     "source_sha256": tensor_sha256(s),
                     "output_sha256": tensor_sha256(o) if o is not None else hashlib.sha256(b"").hexdigest()})
    extra = sorted(set(out) - set(index["weight_map"]))
    return {"rows": rows, "extra_output_tensors": extra, "lm_head_tied_in_source": tied_in_source,
            "n_tensors_source": len(index["weight_map"]), "n_tensors_output": len(out),
            "source_dtypes": dict(Counter(str(t.dtype) for t in src.values())),
            "output_dtypes": dict(Counter(str(t.dtype) for t in out.values())),
            "failed": sum(not r["passed"] for r in rows) + len(extra)}


def item_records(ver: dict[str, Any], backend: str) -> list[dict[str, Any]]:
    """One item per indexed tensor (check_unit part e recomputes fail_count from them). A tensor the
    output has but the source index does not name is an extra failing item, so the count no longer
    matches the declared 311 and the check fails loudly too."""
    empty = hashlib.sha256(b"").hexdigest()
    items = [{"item_id": r["tensor"], "input_sha256": r["source_sha256"], "output_sha256": r["output_sha256"],
              "passed": r["passed"], "score": 1.0 if r["passed"] else 0.0, "grader": GRADER, "backend": backend,
              "extra": {k: r[k] for k in ("problem", "stored_as", "shape", "source_dtype", "output_dtype")}}
             for r in ver["rows"]]
    items += [{"item_id": f"extra:{name}", "input_sha256": empty, "output_sha256": empty, "passed": False,
               "score": 0.0, "grader": GRADER, "backend": backend,
               "extra": {"problem": "in the output but not in the source index"}}
              for name in ver["extra_output_tensors"]]
    return items


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
    out = {"prompt_tokens": int(enc["input_ids"].shape[1]), "new_tokens": int(new.shape[0]),
           "text": tok.decode(new, skip_special_tokens=False), "seconds": round(time.time() - t0, 2),
           "model_dtype": str(next(model.parameters()).dtype), "think_off_suffix_ok": True}
    del model
    gc.collect()
    return out


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
            "tensors_failed": summary["verify"]["failed"],
            "n_tensors": summary["verify"]["n_tensors_source"],
            "files": summary["output_files"],
            "source_files": summary["source_files"],
            "smoke": summary.get("smoke"),
            "versions": summary["versions"],
            "code_sha": summary.get("code_sha"),
        },
    }


# --------------------------------------------------------------------------- main

def run(a: argparse.Namespace) -> dict[str, Any]:
    run_id = K.check_run_id(a.run_id)
    code_sha = K.code_sha_from_env(a.code_sha)
    decl = K.stored_declaration(run_id, a.declared)  # before any work: the target comes from here
    t_start = time.time()
    work = Path(a.workdir or tempfile.mkdtemp(prefix="stage-"))
    src_dir, listing = K.materialise(a.src, work, "src", channel_env="SM_CHANNEL_MODEL", region=a.region)
    if listing is None and a.src_uri:
        listing = K.list_s3(a.src_uri, region=a.region)
    K.log(f"source at {src_dir} ({'downloaded' if K.is_s3(a.src) else 'local/channel'})")
    K.check_test_set(decl, K.sha256_file(src_dir / INDEX), INDEX)
    timings: dict[str, float] = {"inputs_s": round(time.time() - t_start, 1)}

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
    out_tensors, st_files = load_output_tensors(out_dir)
    ver = verify(src_tensors, out_tensors, index, tie)
    del src_tensors, out_tensors
    gc.collect()
    timings["verify_s"] = round(time.time() - t, 1)
    K.log(f"verify: {ver['failed']} failed over {ver['n_tensors_source']} tensors "
          f"({len(ver['extra_output_tensors'])} extra)")
    ok = ver["failed"] == 0

    smoke_out = None
    if ok and not a.skip_smoke:
        t = time.time()
        smoke_out = smoke(out_dir)
        timings["smoke_s"] = round(time.time() - t, 1)
        K.log(f"smoke: {smoke_out}")

    output_files, artifact_sha = K.write_sha256sums(out_dir)
    summary: dict[str, Any] = {
        "kind": "t1-sft-stage-summary", "run_id": run_id, "code_sha": code_sha,
        "src": a.src or os.environ.get("SM_CHANNEL_MODEL"), "src_uri": a.src_uri, "dst": a.dst,
        "hf_repo": K.BASE_HF_REPO, "revision": K.BASE_REVISION,
        "source_files": source_files, "output_files": output_files, "artifact_sha256": artifact_sha,
        "safetensors_files": st_files, "copied_non_weight_files": copied,
        "config_keys_save_pretrained_changed": config_diff,
        "verify": {k: v for k, v in ver.items() if k != "rows"},
        "problems": [r for r in ver["rows"] if not r["passed"]][:50],
        "smoke": smoke_out,
        "versions": K.versions("torch", "transformers", "safetensors", "accelerate", "boto3"),
        "uploaded": False,
    }
    aws_time = None
    if ok or a.allow_mismatch:
        t = time.time()
        K.upload_tree(out_dir, a.dst, region=a.region)
        timings["upload_s"] = round(time.time() - t, 1)
        summary["uploaded"] = True
        aws_time = K.s3_last_modified(a.dst.rstrip("/") + "/SHA256SUMS", region=a.region) if K.is_s3(a.dst) else None
        K.log(f"uploaded to {a.dst}; SHA256SUMS sha256 {artifact_sha}")
    else:
        K.log(f"NOT uploaded: {ver['failed']} tensor(s) failed (the evidence below records which)")
    summary["aws_time"] = aws_time
    timings["total_s"] = round(time.time() - t_start, 1)
    summary["timings"] = timings
    summary_dir = work / "summary"
    summary_sha = K.write_json(summary_dir / "stage_summary.json", summary)
    if summary["uploaded"]:
        K.upload_tree(summary_dir, a.dst, region=a.region)  # beside the model, outside SHA256SUMS
    K.copy_to(a.summary_dir, summary_dir / "stage_summary.json")

    # items and result first: if anything after this fails, the runner still has the evidence to collect
    backend = f"{os.environ.get('SM_CURRENT_INSTANCE_TYPE', 'local')}; transformers save_pretrained safetensors bf16"
    items = item_records(ver, backend)
    res = K.write_evidence(a.evidence_dir, decl, items, "t1/sft/stage_base.py", code_sha,
                           {"artifact_uri": a.dst if summary["uploaded"] else None, "artifact_sha256": artifact_sha,
                            "extra_output_tensors": ver["extra_output_tensors"], "summary_sha256": summary_sha,
                            "uploaded": summary["uploaded"]})
    K.log(f"result: {res['metric']} = {res['value']} (target {res['target']['op']} {res['target']['value']}) "
          f"-> {'PASSED' if res['passed'] else 'MISSED'}; {len(items)} item records in {a.evidence_dir}")
    summary["result"] = {k: res[k] for k in ("metric", "value", "passed", "n_items")}

    if summary["uploaded"]:
        store = K.model_store(a.model_evidence)
        if store is None:
            K.log("no model evidence store (EVIDENCE_S3 unset, no --model-evidence): model record not written")
        else:
            from tools import evlib

            rec = model_record(run_id, a.dst, artifact_sha, summary, aws_time)
            rec["metrics"]["summary_sha256"] = summary_sha
            try:
                K.log(f"model record: {evlib.write_model(rec, store=store)}")
            except Exception as e:  # noqa: BLE001 - the result is already written; say so, do not lose it
                K.log(f"model record NOT written ({type(e).__name__}: {e}); the items and result stand")
    return summary


def dry_run(a: argparse.Namespace) -> dict[str, Any]:
    """No AWS call, no torch: check the arguments and print what the job would do."""
    run_id = K.check_run_id(a.run_id)
    plan: dict[str, Any] = {
        "job": "t1.sft.stage_base", "run_id": run_id, "code_sha": K.code_sha_from_env(a.code_sha),
        "src": a.src or "(channel SM_CHANNEL_MODEL)", "src_uri": a.src_uri, "dst": a.dst,
        "evidence_dir": a.evidence_dir, "summary_dir": a.summary_dir,
        "steps": ["read the stored declaration", "sha256 every source file (+ S3 ETag)",
                  "read the .bin index and shards (weights_only)", "transformers bf16 load + save_pretrained",
                  "byte-copy non-weight files", "one item per indexed tensor (bf16 cast equality, both sha256)",
                  "smoke: chat template thinking off + greedy decode", "SHA256SUMS, upload, stage_summary.json",
                  "model record (EVIDENCE_S3)", "items + result into --evidence-dir (fail_count == 0)"],
    }
    if a.src and not K.is_s3(a.src) and Path(a.src, INDEX).is_file():
        index = json.loads(Path(a.src, INDEX).read_text())
        plan["n_tensors"] = len(index["weight_map"])
        plan["shards"] = sorted(set(index["weight_map"].values()))
    return plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m t1.sft.stage_base", description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--src", help="local dir ({channel:model}) or s3:// prefix of the copied base model")
    ap.add_argument("--src-uri", help="s3:// prefix the channel came from, listed for the ETags")
    ap.add_argument("--dst", required=True, help="s3:// prefix (or dir) for the bf16 safetensors copy")
    ap.add_argument("--evidence-dir", required=True, help="local evlib store for items + result ({evidence_dir})")
    ap.add_argument("--model-evidence", help="where the model record goes (default EVIDENCE_S3)")
    ap.add_argument("--declared", help="the run's declaration as a JSON file (default: read from EVIDENCE_S3)")
    ap.add_argument("--summary-dir", help="also copy stage_summary.json here ({output_dir})")
    ap.add_argument("--code-sha", help="commit the job runs (default CODE_SHA)")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", K.DEFAULT_REGION))
    ap.add_argument("--workdir", help="scratch directory (default: a new temp dir)")
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
        K.log(f"done: {out['result']} artifact_sha256={out['artifact_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
