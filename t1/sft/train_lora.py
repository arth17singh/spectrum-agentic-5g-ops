"""LoRA SFT of the staged base model on the t1/sft dataset: adapter, merged model, model record.

Runs as a GPU SageMaker job through the team runner (tools/sm/run.py --gpu: ml.g6.2xlarge, 1x L4
24 GB, max_run 2 h, on-demand; --no-sdk so the image's CUDA torch stays; transformers 4.57.6 and peft
via --requirement). The staged base arrives as channel `model` (--model-s3), the dataset as `data`:

    --entry "-m t1.sft.train_lora --run-id {run_id} --base {channel:model} --data {channel:data}
             --out s3://<team-bucket>/models/t1-otel-1.7b-lora/{run_id}/
             --base-run-id <stage run id> --data-run-id <data run id>
             --evidence-dir {evidence_dir} --summary-dir {output_dir}"
    smoke (no code change; 20 optimizer steps): add --max-steps 20 --limit-train 160 --limit-dev 48
        --grad-accum 1 --epochs 1 --gen-check 2 --require-loss-drop (and declare n_items 48)

Hyperparameters (declared): LoRA r 16, alpha 32, dropout 0.05 on q/k/v/o/gate/up/down; AdamW
lr 2e-4, cosine with 3% warmup, 2 epochs, micro-batch 8 x accumulation 2, bf16, gradient
checkpointing, grad-norm clip 1.0, seed 13. Loss only on the supervised span (t1/sft/chat.py:
the empty think block, the JSON answer, <|im_end|>; thinking off as the eval harness calls it).

Checks: the base and data directories match their SHA256SUMS; every example's chat rendering
passes chat.py's invariants; no example is truncated (the job stops instead); dev loss is
measured on the base (step 0, adapters are zero at init) and after training. One item record per
dev example (score = its final / base loss); declared target mean_score <= 0.5, which check_unit
recomputes from the items; the token-weighted dev_loss_ratio is reported too. After merging (fp32
merge, saved bf16), the merged model is reloaded and greedy-decodes dev prompts; the eval's own
parser and graders read the outputs. Adapter, merged model and train_metrics.json go to --out as
plain files; the model record goes to EVIDENCE_S3; items and result go to --evidence-dir.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from t1.sft import common as K

UNIT = "t1.sft.train"
METRIC = "dev_loss_ratio"  # token-weighted final/base over the whole dev split (reported)
DECLARED_METRIC = "mean_score"  # per-example final/base, recomputable by check_unit from the items
TARGET_RATIO = 0.5
NONFINITE_SCORE = 1e6
GRADER = "t1/sft/train_lora.py:dev_loss (per dev example: token-mean cross-entropy over the supervised span, final / base)"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SKIP_FROM_BASE = {"SHA256SUMS", "stage_summary.json", "README.md", ".gitattributes", "config.json"}


# --------------------------------------------------------------------------- inputs

verify_sums = K.verify_sums  # kept here for callers of the earlier name


def load_rows(data_dir: Path, limit_train: int, limit_dev: int) -> tuple[list[dict], list[dict], dict[str, Any]]:
    def read(p: Path) -> list[dict]:
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

    train, dev = read(data_dir / "train.jsonl"), read(data_dir / "dev.jsonl")
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if manifest["overlap"]["final_check"]["eval_overlap_count"] != 0:
        raise K.JobError("the dataset's manifest reports eval overlap; refusing to train on it")
    if limit_train:  # a seeded sample across units (the files are ordered by unit)
        train = random.Random(13).sample(train, min(limit_train, len(train)))
    if limit_dev:
        dev = random.Random(14).sample(dev, min(limit_dev, len(dev)))
    return train, dev, manifest


def tokenize_all(tok: Any, rows: list[dict], max_len: int) -> tuple[list[dict], dict[str, Any]]:
    from t1.sft.chat import encode

    out, lens = [], []
    for r in rows:
        e = encode(tok, r["messages"])
        e["unit"] = r["unit"]
        e["id"] = r["id"]
        lens.append(len(e["input_ids"]))
        out.append(e)
    too_long = [r["id"] for r, n in zip(rows, lens) if n > max_len]
    if too_long:
        raise K.JobError(f"{len(too_long)} example(s) longer than --max-len {max_len} (max {max(lens)}): "
                         f"{too_long[:5]}; nothing is truncated, raise --max-len")
    stats = {"n": len(out), "max_tokens": max(lens) if lens else 0, "mean_tokens": round(sum(lens) / max(1, len(lens)), 1),
             "target_tokens": sum(e["n_target"] for e in out)}
    return out, stats


def batches(examples: list[dict], batch_size: int, rng: random.Random | None) -> list[list[dict]]:
    """Length-grouped batches: shuffle, sort within windows of 16 batches, shuffle the batches."""
    order = list(examples)
    if rng is not None:
        rng.shuffle(order)
    window = batch_size * 16
    out = []
    for i in range(0, len(order), window):
        chunk = sorted(order[i:i + window], key=lambda e: len(e["input_ids"]))
        out += [chunk[j:j + batch_size] for j in range(0, len(chunk), batch_size)]
    if rng is not None:
        rng.shuffle(out)
    return out


def collate(batch: list[dict], pad_id: int, device: Any) -> dict[str, Any]:
    import torch

    n = max(len(e["input_ids"]) for e in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, e in enumerate(batch):
        L = len(e["input_ids"])
        ids[i, :L] = torch.tensor(e["input_ids"])
        labels[i, :L] = torch.tensor(e["labels"])
        mask[i, :L] = 1
    return {"input_ids": ids.to(device), "labels": labels.to(device), "attention_mask": mask.to(device)}


def token_losses(model: Any, b: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Next-token cross-entropy on the supervised positions only: (per-token losses, their row
    index, count). The 152k-word lm_head runs on those positions alone; full logits for a batch
    (8 x 1,100 x 151,936 floats) would not fit beside the model on a 24 GB GPU."""
    import torch.nn.functional as F

    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    hs = inner.model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False).last_hidden_state
    tgt = b["labels"][:, 1:]
    keep = tgt != -100
    logits = inner.lm_head(hs[:, :-1, :][keep]).float()
    losses = F.cross_entropy(logits, tgt[keep], reduction="none")
    rows = keep.nonzero()[:, 0]
    return losses, rows, int(keep.sum().item())


def dev_loss(model: Any, dev: list[dict], pad_id: int, device: Any, batch_size: int) -> dict[str, Any]:
    import torch

    model.eval()
    tot, n = 0.0, 0
    per_unit: dict[str, list[float]] = {}
    per_example: dict[str, float] = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for batch in batches(dev, batch_size, None):
            b = collate(batch, pad_id, device)
            losses, rows, c = token_losses(model, b)
            sums = torch.zeros(len(batch), device=losses.device).index_add_(0, rows, losses)
            counts = torch.zeros(len(batch), device=losses.device).index_add_(0, rows, torch.ones_like(losses))
            tot += float(losses.sum())
            n += c
            for e, s, k in zip(batch, sums.tolist(), counts.tolist()):
                u = per_unit.setdefault(e["unit"], [0.0, 0.0])
                u[0] += s
                u[1] += k
                per_example[e["id"]] = s / max(1.0, k)
    model.train()
    return {"loss": tot / max(1, n), "tokens": n, "by_unit": {u: v[0] / max(1, v[1]) for u, v in sorted(per_unit.items())},
            "per_example": per_example}


def _curve_point(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k != "per_example"}


def dev_item_records(dev_rows: list[dict], dev_ex: list[dict], base: dict[str, float], final: dict[str, float],
                     backend: str) -> list[dict[str, Any]]:
    """One item per dev example: score = its final / base token-mean loss over the supervised span
    (declared target: mean_score <= 0.5, which check_unit recomputes from these)."""
    import hashlib

    from t1.evals import prompting

    n_target = {e["id"]: e["n_target"] for e in dev_ex}
    out = []
    for r in dev_rows:
        rid = r["id"]
        b, f = base[rid], final[rid]
        finite = math.isfinite(b) and math.isfinite(f) and b > 0
        sys_text, user = r["messages"][0]["content"], r["messages"][1]["content"]
        out.append({"item_id": rid, "input_sha256": prompting.input_sha256(sys_text, user),
                    "output_sha256": hashlib.sha256(r["messages"][2]["content"].encode("utf-8")).hexdigest(),
                    # evidence is JSON without NaN: a diverged loss is a large failing score, flagged
                    "score": f / b if finite else NONFINITE_SCORE, "passed": bool(finite and f < b),
                    "grader": GRADER, "backend": backend,
                    "extra": {"unit": r["unit"], "base_loss": b if math.isfinite(b) else None,
                              "final_loss": f if math.isfinite(f) else None, "nonfinite": not finite,
                              "target_tokens": n_target[rid]}})
    return out


# --------------------------------------------------------------------------- generation check

def gen_check(model_dir: Path, rows: list[dict], n_per_unit: int, device: Any) -> dict[str, Any]:
    """Reload the merged model and greedy-decode dev prompts (thinking off); the eval's parser reads
    every output. Interface outputs are graded by its grader; execution and safety outputs are
    compared with the label (the twin-graded score is the eval run's job, not this one's)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from t1.evals import canon as C
    from t1.evals.graders import grade_interface, parse_ops
    from t1.evals.parse import first_json_object

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16).to(device)
    model.eval()
    picked: list[dict] = []
    for unit in ("execution", "interface", "safety"):
        picked += [r for r in rows if r["unit"] == unit][:n_per_unit]
    out = []
    for r in picked:
        prompt = tok.apply_chat_template(r["messages"][:2], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        t0 = time.time()
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=192, do_sample=False)
        text = tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        obj = first_json_object(text)
        row = {"id": r["id"], "unit": r["unit"], "output": text, "parse_ok": obj is not None,
               "seconds": round(time.time() - t0, 2)}
        if r["unit"] == "interface":
            row["matches_label"] = bool(grade_interface({"expected": r["answer"]}, text)["passed"])
        else:
            parsed, err = parse_ops(text)
            row["parse_ok"] = parsed is not None
            want = C.canon(r["answer"]["change"]) if r["answer"]["decision"] == "apply" else ""
            row["matches_label"] = parsed is not None and parsed["decision"] == r["answer"]["decision"] \
                and parsed["change_text"] == want
            if err:
                row["parse_error"] = err
        out.append(row)
        K.log(f"gen {r['id']}: parse_ok={row['parse_ok']} matches_label={row.get('matches_label')} {text[:120]!r}")
    del model
    gc.collect()
    return {"n": len(out), "parse_ok": sum(o["parse_ok"] for o in out),
            "matches_label": sum(bool(o.get("matches_label")) for o in out), "outputs": out}


# --------------------------------------------------------------------------- main

def train(a: argparse.Namespace) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    run_id = K.check_run_id(a.run_id)
    code_sha = K.code_sha_from_env(a.code_sha)
    decl = K.stored_declaration(run_id, a.declared)  # before any work: the target comes from here
    t_start = time.time()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    work = Path(a.workdir or tempfile.mkdtemp(prefix="lora-"))
    base_dir, _ = K.materialise(a.base, work, "base", channel_env="SM_CHANNEL_MODEL", region=a.region)
    data_dir, _ = K.materialise(a.data, work, "data", channel_env="SM_CHANNEL_DATA", region=a.region)
    base_sums_sha, base_files = verify_sums(base_dir)
    data_sums_sha, _ = verify_sums(data_dir)
    K.check_test_set(decl, K.sha256_file(data_dir / "dev.jsonl"), "dev.jsonl")
    timings = {"inputs_s": round(time.time() - t_start, 1)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not a.allow_cpu:
        raise K.JobError("no GPU; pass --allow-cpu only for a tiny local test")

    tok = AutoTokenizer.from_pretrained(str(base_dir))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.convert_tokens_to_ids("<|endoftext|>")
    train_rows, dev_rows, manifest = load_rows(data_dir, a.limit_train, a.limit_dev)
    t = time.time()
    train_ex, train_stats = tokenize_all(tok, train_rows, a.max_len)
    dev_ex, dev_stats = tokenize_all(tok, dev_rows, a.max_len)
    timings["tokenize_s"] = round(time.time() - t, 1)
    K.log(f"train {train_stats}, dev {dev_stats}")

    model = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.bfloat16, attn_implementation="sdpa")
    model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout, target_modules=LORA_TARGETS,
                      bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    K.log(f"trainable params {trainable} of {total}")

    steps_per_epoch = math.ceil(math.ceil(len(train_ex) / a.batch_size) / a.grad_accum)
    total_steps = steps_per_epoch * a.epochs if not a.max_steps else min(a.max_steps, steps_per_epoch * a.epochs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.weight_decay,
                            betas=(0.9, 0.999))
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(a.warmup_ratio * total_steps)), total_steps)

    t = time.time()
    dev0 = dev_loss(model, dev_ex, pad_id, device, a.batch_size)
    timings["dev_eval_s"] = round(time.time() - t, 1)
    K.log(f"dev loss at step 0 (base, adapters zero): {dev0['loss']:.4f} {dev0['by_unit']}")
    curve: list[dict[str, Any]] = []
    dev_curve = [{"step": 0, **_curve_point(dev0)}]
    rng = random.Random(a.seed)
    step, micro, done, stopped_by = 0, 0, False, "steps"
    model.train()
    t_train = time.time()
    opt.zero_grad(set_to_none=True)
    run_loss, run_tok = 0.0, 0
    for epoch in range(a.epochs):
        for batch in batches(train_ex, a.batch_size, rng):
            b = collate(batch, pad_id, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                losses, _, n = token_losses(model, b)
            s = losses.sum()
            (s / max(1, n) / a.grad_accum).backward()
            run_loss += float(s.detach())
            run_tok += n
            micro += 1
            if micro % a.grad_accum:
                continue
            gnorm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            curve.append({"step": step, "epoch": epoch, "loss": run_loss / max(1, run_tok), "tokens": run_tok,
                          "lr": sched.get_last_lr()[0], "grad_norm": float(gnorm), "elapsed_s": round(time.time() - t_train, 1)})
            if step % a.log_every == 0 or step == 1:
                K.log(f"step {step}/{total_steps} epoch {epoch} train_loss={curve[-1]['loss']:.4f} "
                      f"lr={curve[-1]['lr']:.2e} grad_norm={curve[-1]['grad_norm']:.3f}")
            run_loss, run_tok = 0.0, 0
            if a.eval_every and step % a.eval_every == 0 and step < total_steps:
                d = dev_loss(model, dev_ex, pad_id, device, a.batch_size)
                dev_curve.append({"step": step, **_curve_point(d)})
                K.log(f"step {step} dev_loss={d['loss']:.4f}")
            if step >= total_steps:
                done = True
                break
            if a.train_budget_s and time.time() - t_train > a.train_budget_s:
                # leave time to save, merge and upload before SageMaker's max_run stops the job
                done, stopped_by = True, "train_budget"
                K.log(f"training budget {a.train_budget_s}s reached at step {step}/{total_steps}")
                break
        if done:
            break
    timings["train_s"] = round(time.time() - t_train, 1)
    dev_final = dev_loss(model, dev_ex, pad_id, device, a.batch_size)
    dev_curve.append({"step": step, **_curve_point(dev_final)})
    ratio = dev_final["loss"] / dev0["loss"] if dev0["loss"] > 0 else float("inf")
    K.log(f"dev loss final {dev_final['loss']:.4f} (base {dev0['loss']:.4f}, ratio {ratio:.4f})")
    first = [c["loss"] for c in curve[:5]]
    last = [c["loss"] for c in curve[-5:]]
    loss_dropped = bool(first and last and sum(last) / len(last) < sum(first) / len(first))
    if a.require_loss_drop and not (loss_dropped and dev_final["loss"] < dev0["loss"]):
        raise K.JobError(f"loss did not drop: first {first}, last {last}, dev {dev0['loss']} -> {dev_final['loss']}")

    out_dir = work / "out"
    adapter_dir, merged_dir = out_dir / "adapter", out_dir / "merged"
    model.save_pretrained(str(adapter_dir), safe_serialization=True)
    del model, opt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    t = time.time()
    base32 = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.float32).to(device)
    merged = PeftModel.from_pretrained(base32, str(adapter_dir)).merge_and_unload()
    merged = merged.to(torch.bfloat16)
    merged.config.use_cache = True
    merged.save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size="5GB")
    for name in sorted(set(base_files) - SKIP_FROM_BASE):  # tokenizer, chat template, generation config
        if name.endswith(".safetensors") or name.endswith(".safetensors.index.json"):
            continue
        (merged_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (merged_dir / name).write_bytes((base_dir / name).read_bytes())
    del merged, base32
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timings["merge_s"] = round(time.time() - t, 1)
    cfg_path = adapter_dir / "adapter_config.json"  # name the base by its public id, not the job's path
    acfg = json.loads(cfg_path.read_text())
    acfg["base_model_name_or_path"], acfg["revision"] = K.BASE_HF_REPO, K.BASE_REVISION
    cfg_path.write_text(json.dumps(acfg, indent=2, sort_keys=True) + "\n")

    gen = None
    if a.gen_check:
        t = time.time()
        gen = gen_check(merged_dir, dev_rows, a.gen_check, device)
        timings["gen_check_s"] = round(time.time() - t, 1)
    timings["total_s"] = round(time.time() - t_start, 1)

    hyper = {"lora_r": a.lora_r, "lora_alpha": a.lora_alpha, "lora_dropout": a.lora_dropout,
             "target_modules": LORA_TARGETS, "lr": a.lr, "scheduler": "cosine", "warmup_ratio": a.warmup_ratio,
             "epochs": a.epochs, "batch_size": a.batch_size, "grad_accum": a.grad_accum, "weight_decay": a.weight_decay,
             "max_len": a.max_len, "max_steps": a.max_steps, "train_budget_s": a.train_budget_s, "seed": a.seed,
             "precision": "bf16 autocast",
             "gradient_checkpointing": True, "grad_clip": 1.0, "optimizer": "AdamW",
             "loss": "token-mean cross-entropy over the supervised span (empty think block + JSON + <|im_end|>)",
             "limit_train": a.limit_train, "limit_dev": a.limit_dev}
    metrics = {
        "kind": "t1-sft-train-metrics", "run_id": run_id, "code_sha": code_sha, "hyperparameters": hyper,
        "base": {"uri": a.base, "run_id": a.base_run_id, "sha256sums_sha256": base_sums_sha},
        "data": {"uri": a.data, "run_id": a.data_run_id, "manifest_run_id": manifest.get("run_id"),
                 "sha256sums_sha256": data_sums_sha,
                 "manifest_sha256": K.sha256_file(data_dir / "manifest.json"), "train": train_stats, "dev": dev_stats},
        "steps": step, "steps_planned": total_steps, "stopped_by": stopped_by, "steps_per_epoch": steps_per_epoch,
        "trainable_params": trainable, "total_params": total,
        "train_loss_curve": curve, "dev_loss_curve": dev_curve, "dev_loss_base": dev0["loss"],
        "dev_loss_final": dev_final["loss"], METRIC: ratio, "target": {"metric": METRIC, "op": "<=",
                                                                      "value": TARGET_RATIO, "unit": "ratio"},
        "loss_dropped_first5_vs_last5": loss_dropped, "gen_check": gen, "timings": timings,
        "wall_s": timings["total_s"],
        "est_cost_usd_from_wall": round(a.price_per_hour * timings["total_s"] / 3600, 4) if a.price_per_hour else None,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "versions": K.versions("torch", "transformers", "peft", "accelerate", "safetensors"),
    }
    backend = (f"{os.environ.get('SM_CURRENT_INSTANCE_TYPE', 'local')} {metrics['device']}; "
               f"transformers {metrics['versions'].get('transformers')}, peft {metrics['versions'].get('peft')}")
    items = dev_item_records(dev_rows, dev_ex, dev0["per_example"], dev_final["per_example"], backend)
    metrics["dev_items"] = {"n": len(items), DECLARED_METRIC: sum(i["score"] for i in items) / max(1, len(items)),
                            "improved": sum(i["passed"] for i in items)}
    metrics = K.finite(metrics)
    K.write_json(out_dir / "train_metrics.json", metrics)
    files, artifact_sha = K.write_sha256sums(out_dir)
    t = time.time()
    K.upload_tree(out_dir, a.out, region=a.region)
    timings["upload_s"] = round(time.time() - t, 1)
    aws_time = K.s3_last_modified(a.out.rstrip("/") + "/SHA256SUMS", region=a.region) if K.is_s3(a.out) else None
    K.log(f"uploaded {len(files)} files to {a.out}; SHA256SUMS sha256 {artifact_sha}")
    K.copy_to(a.summary_dir, out_dir / "train_metrics.json")

    res = K.write_evidence(a.evidence_dir, decl, items, "t1/sft/train_lora.py", code_sha, K.finite(
                           {"artifact_uri": a.out, "artifact_sha256": artifact_sha, "dev_loss_base": dev0["loss"],
                            "dev_loss_final": dev_final["loss"], METRIC: ratio, "steps": step,
                            "steps_planned": total_steps, "stopped_by": stopped_by,
                            "gen_check": {k: v for k, v in (gen or {}).items() if k != "outputs"}}))
    K.log(f"result: {res['metric']} = {res['value']:.4f} over {res['n_items']} dev items (target "
          f"{res['target']['op']} {res['target']['value']}) -> {'PASSED' if res['passed'] else 'MISSED'}")
    metrics["result"] = {k: res[k] for k in ("metric", "value", "passed", "n_items")}

    store = K.model_store(a.model_evidence)
    if store is None:
        K.log("no model evidence store (EVIDENCE_S3 unset, no --model-evidence): model record not written")
    else:
        from tools import evlib

        rec = {"run_id": run_id, "model_id": K.LORA_MODEL_ID, "version": run_id, "model_kind": "llm-lora",
               "artifact_uri": a.out.rstrip("/") + "/merged/", "artifact_sha256": artifact_sha, "status": "candidate",
               "trained_on": a.data_run_id or a.data, "aws_time": aws_time,
               "aws_time_source": "s3:last_modified (SHA256SUMS)" if aws_time else None,
               "metrics": {"artifact_sha256_is": "sha256 of SHA256SUMS at the run prefix (adapter/, merged/, "
                                                 "train_metrics.json)",
                           "adapter_uri": a.out.rstrip("/") + "/adapter/", "base": metrics["base"],
                           "data": {k: v for k, v in metrics["data"].items()}, "hyperparameters": hyper,
                           "dev_loss_base": dev0["loss"], "dev_loss_final": dev_final["loss"], METRIC: ratio,
                           "dev_loss_by_unit_base": dev0["by_unit"], "dev_loss_by_unit_final": dev_final["by_unit"],
                           "steps": step, "steps_planned": total_steps, "stopped_by": stopped_by,
                           "train_loss_first": curve[0]["loss"] if curve else None,
                           "train_loss_last": curve[-1]["loss"] if curve else None,
                           "gen_check": {k: v for k, v in (gen or {}).items() if k != "outputs"},
                           "wall_s": timings["total_s"], "files": files, "code_sha": code_sha,
                           "result": metrics["result"]}}
        K.log(f"model record: {evlib.write_model(K.finite(rec), store=store)}")
    return metrics


def dry_run(a: argparse.Namespace) -> dict[str, Any]:
    """No AWS, no torch: the plan, and (given a local --data) the dataset's chat rendering checked
    against a chat template given with --chat-template (the base model's chat_template.jinja)."""
    K.check_run_id(a.run_id)
    plan: dict[str, Any] = {"job": "t1.sft.train_lora", "run_id": a.run_id, "code_sha": K.code_sha_from_env(a.code_sha),
                            "base": a.base or "(channel SM_CHANNEL_MODEL)", "data": a.data or "(channel SM_CHANNEL_DATA)",
                            "out": a.out, "evidence_dir": a.evidence_dir,
                            "target": {"metric": DECLARED_METRIC, "op": "<=", "value": TARGET_RATIO, "unit": "ratio"},
                            "hyperparameters": {k: getattr(a, k) for k in ("lora_r", "lora_alpha", "lora_dropout", "lr",
                                                                            "epochs", "batch_size", "grad_accum",
                                                                            "warmup_ratio", "max_len", "seed")}}
    if a.data and not K.is_s3(a.data) and a.chat_template:
        from t1.sft.chat import jinja_renderer, split_text

        render = jinja_renderer(Path(a.chat_template).read_text())
        rows = [json.loads(x) for x in Path(a.data, "train.jsonl").read_text().splitlines() if x.strip()]
        for r in rows:
            split_text(render, r["messages"])
        plan["chat_rendering_checked"] = len(rows)
    if not a.out:
        raise K.JobError("--out is required")
    return plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m t1.sft.train_lora", description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--base", help="staged bf16 base: dir ({channel:model}) or s3:// prefix (else SM_CHANNEL_MODEL)")
    ap.add_argument("--data", help="SFT dataset: s3:// prefix or dir (else SM_CHANNEL_DATA)")
    ap.add_argument("--out", help="s3:// prefix or dir for adapter/, merged/, train_metrics.json, SHA256SUMS")
    ap.add_argument("--base-run-id")
    ap.add_argument("--data-run-id")
    ap.add_argument("--evidence-dir", help="local evlib store for items + result ({evidence_dir})")
    ap.add_argument("--model-evidence", help="where the model record goes (default EVIDENCE_S3)")
    ap.add_argument("--declared", help="the run's declaration as a JSON file (default: read from EVIDENCE_S3)")
    ap.add_argument("--summary-dir", help="also copy train_metrics.json here ({output_dir})")
    ap.add_argument("--code-sha")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", K.DEFAULT_REGION))
    ap.add_argument("--workdir")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (smoke)")
    ap.add_argument("--train-budget-s", type=float, default=5400.0,
                    help="stop the training loop after this many seconds, keeping time to save/merge/upload (0: off)")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-dev", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--gen-check", type=int, default=3, help="dev prompts per unit decoded by the merged model")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--price-per-hour", type=float, help="on-demand $/h of the instance, for a wall-time estimate")
    ap.add_argument("--require-loss-drop", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--chat-template", help="dry run: a chat_template.jinja to check the data's rendering with")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if not a.dry_run and not (a.out and a.evidence_dir):
        ap.error("--out and --evidence-dir are required")
    return a


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    try:
        out = dry_run(a) if a.dry_run else train(a)
    except K.JobError as e:
        K.log(f"FAILED: {e}")
        return 2
    if a.dry_run:
        print(K.dumps_doc(out), end="")
    else:
        K.log(f"done: {out['result']}; {METRIC}={out[METRIC]} dev {out['dev_loss_base']} -> {out['dev_loss_final']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
