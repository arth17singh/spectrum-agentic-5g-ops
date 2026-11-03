"""Build the Track 1 SFT dataset: execution (twin-labelled), interface and safety examples.

Runs as a CPU SageMaker job through the team runner (tools/sm/run.py; ml.c5.4xlarge, 16 vCPU, one twin
per worker process; the runner's sdk channel installs ran-rl 1.2.9). Nothing heavy runs on a laptop:
--dry-run builds a handful of template examples and the overlap index, no twin, no AWS.

    --entry "-m t1.sft.build --run-id {run_id} --out s3://<team-bucket>/datasets/t1-sft/{run_id}/
             --evidence-dir {evidence_dir} --summary-dir {output_dir} --n-execution 2000 ..."

Rules it enforces (t1/evals/README.md "Rules for anyone building training data"):
  - execution states use twin seeds below 900000 (eval items use 900000 and up);
  - every prompt is prompting.render(...) of prompt.txt's SYSTEM + TASK sections, and every target is
    one JSON object in the shape the eval parser reads, with the canonical change string;
  - zero overlap with the frozen eval items (all five files, dev and test), by content hashes
    (t1/sft/overlap.py): an example is dropped if its item hash, answer-free input hash, exact prompt
    hash or (scenario, seed, step) matches any eval item; duplicates inside the set are dropped too;
    the written files are then re-hashed and re-checked;
  - no GSMA ot-lite (a test set), no TeleLogs (left out of v1), no knowledge examples.

Outputs at --out: train.jsonl, dev.jsonl (one example per line: messages + the template input,
answer, meta and hashes), manifest.json (counts, seed ranges, overlap report, hashes, timings) and
SHA256SUMS. The chat template is applied by train_lora.py, so this job needs no tokenizer.
Evidence (--evidence-dir): one item record per frozen eval item (1,220), passed when no written row
matches it by any hash; declared target fail_count == 0, which check_unit recomputes from the items.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from t1.sft import common as K

UNIT = "t1.sft.data"
METRIC = "fail_count"  # declared: eval items that any written row matches
OVERLAP = "eval_overlap_count"  # written rows that match any eval item (must also be 0)
REQUIRED_SDK = "1.2.9"
SPLIT_SALT = "t1-sft-split-v1"
ANSWER_KEYS = {  # the reply object's keys, in the order prompt.txt shows them
    "interface": ["intent", "cells", "parameter", "direction", "amount", "kpi"],
    "execution": ["decision", "change", "reason"],
    "safety": ["decision", "change", "reason"],
}
# The frozen eval item files at commit f1d3deb (merged into int/v4 at 3bf157d); the build checks the
# files it indexes against these, so the overlap check is against exactly those item sets.
F1D3DEB = {
    "MANIFEST.json": "b12ef1e21c452329bc6812de1568a48d0ecb2b2aa08f4b95326833263399d3c6",
    "execution.jsonl": "63d04053892d118721eed5c40460c9ae0c878043d464db7b9edc10e2223efd74",
    "interface.jsonl": "b8347f58171979b1c86a9c651259a8ce1793c2095600e152d55d2fbfee09210c",
    "knowledge.jsonl": "8a687445efd4fc5738615c96ab9da5e8907ee50003233527f24b1bbed48a2df9",
    "price_latency.jsonl": "4d55adddb52171878f30000e7d7fa969d77bb41cd6e0f33c041927cde185b408",
    "safety.jsonl": "1ce6f4494cffa54fc385d54dad0a2047a761692c3e3c1e2611928595eae84b75",
}


def items_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "evals" / "items"


# --------------------------------------------------------------------------- the SDK

def ensure_sdk(wheel: str | None, workdir: Path, region: str) -> dict[str, Any]:
    """ran-rl 1.2.9 with the Maveric extra (the eval items were frozen on it). Installed from the
    wheel given (s3:// object or prefix, a .whl path or a directory, else channel SM_CHANNEL_SDK)
    when it is not importable. The wheel is the organisers' and never enters git."""
    try:
        have = importlib.metadata.version("ran-rl")
    except importlib.metadata.PackageNotFoundError:
        have = None
    if have == REQUIRED_SDK:
        return {"ran_rl": have, "installed_by_job": False}
    src = wheel or os.environ.get("SM_CHANNEL_SDK")
    if not src:
        raise K.JobError(f"ran-rl {REQUIRED_SDK} is required (found {have}): pass --sdk-wheel or mount channel sdk")
    if K.is_s3(src):
        if src.endswith(".whl"):
            bucket, _, key = src[len("s3://"):].partition("/")
            dest = workdir / "sdk" / Path(key).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            K.s3_client(region).download_file(bucket, key, str(dest))
            src = str(dest)
        else:
            K.download_s3(src, workdir / "sdk", region)
            src = str(workdir / "sdk")
    p = Path(src)
    whl = p if p.is_file() else next(iter(sorted(p.glob(f"ran_rl-{REQUIRED_SDK}-*.whl"))), None)
    if whl is None or not whl.is_file():
        raise K.JobError(f"no ran_rl-{REQUIRED_SDK} wheel at {src}")
    K.log(f"installing {whl.name}[maveric]")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-input", "--disable-pip-version-check",
                           f"{whl}[maveric]"])
    importlib.invalidate_caches()
    have = importlib.metadata.version("ran-rl")
    if have != REQUIRED_SDK:
        raise K.JobError(f"installed ran-rl {have}, need {REQUIRED_SDK}")
    return {"ran_rl": have, "installed_by_job": True, "wheel": whl.name, "wheel_sha256": K.sha256_file(whl)}


def twin_info() -> dict[str, Any]:
    from twin.presets import config_hash, resolve_config

    return {"kind": "maveric-inprocess", "preset": "official", "replay": "twin.replay.ReplayTwin (W2)",
            "sdk_version": importlib.metadata.version("ran-rl"), "config_sha256": config_hash(resolve_config(None))}


# --------------------------------------------------------------------------- rows

def split_of(ex: dict[str, Any], dev_fraction: float) -> str:
    """dev or train from sha256(salt, unit, key); execution keys by twin seed, so a seed's states
    never straddle the split."""
    from t1.evals.common import h

    key = f"seed:{ex['meta']['seed']}" if ex["unit"] == "execution" else ex["key"]
    return "dev" if h(SPLIT_SALT, ex["unit"], key) % 10_000 < dev_fraction * 10_000 else "train"


def to_row(ex: dict[str, Any], split: str) -> dict[str, Any]:
    from t1.evals import prompting
    from t1.sft.templates import answer_text

    system, user = prompting.render({"template": ex["template"], "input": ex["input"]})
    return {"id": ex["key"], "unit": ex["unit"], "family": ex["family"], "split": split,
            "template": ex["template"], "input": ex["input"], "answer": ex["answer"],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user},
                         {"role": "assistant", "content": answer_text(ex["answer"])}],
            "meta": ex.get("meta", {}), "hashes": ex.get("hashes", {})}


def check_row(row: dict[str, Any]) -> None:
    """The target is what the eval's parser and graders read back, and the prompt is prompt.txt's."""
    from t1.evals import canon as C
    from t1.evals import prompting
    from t1.evals.graders import grade_interface, parse_ops
    from t1.evals.parse import first_json_object

    system, user = prompting.render({"template": row["template"], "input": row["input"]})
    msgs = row["messages"]
    if [m["role"] for m in msgs] != ["system", "user", "assistant"] or msgs[0]["content"] != system \
            or msgs[1]["content"] != user:
        raise AssertionError(f"{row['id']}: messages do not match prompt.txt")
    got = first_json_object(msgs[2]["content"])
    if got != row["answer"] or json.loads(msgs[2]["content"]) != got:
        raise AssertionError(f"{row['id']}: the parser reads {got!r}, not the answer (or there is text around it)")
    if list(got) != ANSWER_KEYS[row["template"]]:
        raise AssertionError(f"{row['id']}: answer keys {list(got)} are not in prompt.txt's order")
    if row["template"] == "interface":
        if not grade_interface({"expected": row["answer"]}, msgs[2]["content"])["passed"]:
            raise AssertionError(f"{row['id']}: interface answer does not grade against itself")
    elif row["template"] in ("execution", "safety"):
        parsed, err = parse_ops(msgs[2]["content"])
        if parsed is None:
            raise AssertionError(f"{row['id']}: parse_ops rejects the answer: {err}")
        want = C.canon(row["answer"]["change"]) if row["answer"]["decision"] == "apply" else ""
        if parsed["decision"] != row["answer"]["decision"] or parsed["change_text"] != want:
            raise AssertionError(f"{row['id']}: parse_ops reads {parsed}, label is {row['answer']}")
        if row["answer"]["decision"] == "refuse" and row["answer"]["change"] != "":
            raise AssertionError(f"{row['id']}: a refusal carries a change")
        if row["template"] == "execution" and (row["meta"].get("seed", 10**9) >= 900_000):
            raise AssertionError(f"{row['id']}: execution seed {row['meta'].get('seed')} is reserved for eval")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    data = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for r in rows)
    path.write_text(data, encoding="utf-8")
    return {"rows": len(rows), "bytes": len(data.encode("utf-8")), "sha256": K.sha256_bytes(data.encode("utf-8"))}


def counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"by_unit": dict(Counter(r["unit"] for r in rows)),
            "by_split": dict(Counter(r["split"] for r in rows)),
            "by_unit_split": {f"{u}/{s}": n for (u, s), n in sorted(Counter((r["unit"], r["split"]) for r in rows).items())},
            "by_unit_family": {f"{u}/{f}": n for (u, f), n in sorted(Counter((r["unit"], r["family"]) for r in rows).items())}}


# --------------------------------------------------------------------------- main

def plan_counts(a: argparse.Namespace) -> dict[str, Any]:
    from t1.sft.execgen import FAMILIES
    from t1.sft.templates import safety_counts

    base, extra = divmod(a.n_execution, len(FAMILIES))
    per_family = {f: base + (1 if i < extra else 0) for i, f in enumerate(FAMILIES)}
    return {"execution_per_family": per_family, "interface": a.n_interface,
            "safety_per_family": safety_counts(a.n_safety)}


def build(a: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    """Generate, filter, split, check and write; returns the manifest."""
    from t1.evals import graders, parse, prompting
    from t1.evals.gen_execution import CONSTANT_CHANGE, CONSTANT_MAX_SHARE, MIN_ORACLE_GAIN, candidates
    from t1.sft import execgen, overlap, templates

    t_start = time.time()
    plan = plan_counts(a)
    idx = overlap.build_eval_index(a.items_dir or items_dir())
    actual = {**idx.files, "MANIFEST.json": idx.manifest_sha256}
    identical = {name: actual.get(name) == sha for name, sha in F1D3DEB.items()}
    timings: dict[str, float] = {"eval_index_s": round(time.time() - t_start, 2)}

    t = time.time()
    examples = templates.interface_examples(a.n_interface)
    if a.dry_run:
        safety = templates.safety_examples(a.n_safety, filter_check=_filter_check_or_none())
    else:
        from t1.evals.twin_access import filter_check

        safety = templates.safety_examples(a.n_safety, filter_check=filter_check)
    examples += safety
    timings["templates_s"] = round(time.time() - t, 2)

    exec_report: dict[str, Any] = {"skipped": "dry run"} if a.dry_run else {}
    if not a.dry_run and a.n_execution > 0:
        t = time.time()
        gen = execgen.generate(plan["execution_per_family"], a.workers, a.time_budget_s, labeler=a.labeler,
                               log=K.log)
        for fam in execgen.FAMILIES:
            examples += [execgen.to_example(r) for r in gen["accepted"].get(fam, [])]
        exec_report = {k: v for k, v in gen.items() if k != "accepted"}
        timings["execution_s"] = round(time.time() - t, 1)
    elif a.dry_run:
        exec_report["sample_specs"] = [execgen.exec_spec(f, 0) for f in execgen.FAMILIES]

    t = time.time()
    kept, filt = overlap.filter_examples(examples, idx)
    rows = [to_row(ex, split_of(ex, a.dev_fraction)) for ex in kept]
    for row in rows:
        check_row(row)
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "dev"]
    files = {"train.jsonl": write_jsonl(out_dir / "train.jsonl", train),
             "dev.jsonl": write_jsonl(out_dir / "dev.jsonl", dev)}
    reread = overlap.read_jsonl(out_dir / "train.jsonl") + overlap.read_jsonl(out_dir / "dev.jsonl")
    final = overlap.check_disjoint(reread, idx)
    backend = f"{os.environ.get('SM_CURRENT_INSTANCE_TYPE', 'local')}; python hash sets over the written files"
    eval_items = overlap.eval_item_records(reread, idx, backend)
    final["eval_items_checked"] = len(eval_items)
    final["eval_items_overlapped"] = sum(not it["passed"] for it in eval_items)
    timings["filter_write_check_s"] = round(time.time() - t, 1)

    seeds = [r["meta"]["seed"] for r in rows if r["unit"] == "execution"]
    sdk = {"ran_rl": _sdk_version()}
    manifest = {
        "kind": "t1-sft-data-manifest", "manifest_version": 1, "run_id": a.run_id, "code_sha": a.code_sha,
        "created_at": K.utc_now(), "dry_run": bool(a.dry_run),
        "eval_contract": {
            "items_dir": "t1/evals/items", "commit": "f1d3deb (merged into int/v4 at 3bf157d)",
            "byte_identical_to_f1d3deb": identical, **idx.summary(),
            "parser": parse.PARSER_VERSION, "graders": graders.GRADER_VERSION,
            "prompt_sha256_used": prompting.prompt_sha256(),
        },
        "format": {
            "prompt": "t1.evals.prompting.render: prompt.txt ### SYSTEM + ### TASK <template> sections",
            "target": "one JSON object in prompt.txt's key order (interface: intent, cells, parameter, direction, "
                      "amount, kpi; execution and safety: decision, change, reason)",
            "change_string": "team13_contracts.change.canon (contract A2), via t1.evals.canon",
            "chat_template": "applied in train_lora.py: the base model's Qwen3 template with enable_thinking=False; "
                             "the assistant turn renders as <think>\\n\\n</think>\\n\\n{json}<|im_end|>",
        },
        "counts": {"planned": plan, "written": counts(rows)},
        "overlap": {**filt, "final_check": final, "eval_overlap_examples": filt["eval_overlap_examples"]},
        "seeds": {"rule": "execution twin seeds in [0, 900000); eval items use >= 900000",
                  "execution_min": min(seeds) if seeds else None, "execution_max": max(seeds) if seeds else None,
                  "execution_distinct": len(set(seeds)), "eval_min": idx.summary()["exec_seed_min"]},
        "execution": {
            "labeler": a.labeler, "k_candidates": len(candidates()), "min_oracle_gain": MIN_ORACLE_GAIN,
            "constant_change": CONSTANT_CHANGE, "constant_max_share": CONSTANT_MAX_SHARE,
            "families": list(execgen.FAMILIES), "operating_point_steps": list(execgen.OP_STEPS),
            "seed_tag": execgen.SEED_TAG, "twin": None if a.dry_run else twin_info(), "generation": exec_report,
        },
        "interface": {"gen": templates.IF_SEED},
        "safety": {"gen": templates.SF_SEED, "shares": templates.SAFETY_SHARES,
                   "policy": "over-limit and alarmed-cell requests refused (not reduced); out-of-authority refused; "
                             "pasted instructions never followed (the operator's own explicit in-limit change is "
                             "applied exactly; a request delegated to pasted text is refused); benign requests "
                             "applied exactly",
                   "label_check": "every applied change passes the SDK ActionSafetyFilter untouched; every refused "
                                  "over-limit or alarmed-cell request is one the filter would clip or freeze"},
        "split": {"salt": SPLIT_SALT, "dev_fraction": a.dev_fraction,
                  "rule": "dev if sha256(salt, unit, key)[:8] % 10000 < dev_fraction * 10000; execution key = twin seed"},
        "not_used": {"gsma_ot_lite": "a test set: never trained on", "telelogs": "left out of v1",
                     "knowledge": "no knowledge examples (the eval's knowledge unit is the forgetting check)"},
        "files": files, "timings": {**timings, "total_s": round(time.time() - t_start, 1)},
        "versions": {**K.versions("numpy", "torch", "gpytorch", "pydantic"), **sdk},
    }
    return manifest, eval_items


def _sdk_version() -> str | None:
    try:
        return importlib.metadata.version("ran-rl")
    except importlib.metadata.PackageNotFoundError:
        return None


def _filter_check_or_none():
    try:
        from t1.evals.twin_access import filter_check

        return filter_check
    except Exception:  # noqa: BLE001 - dry run without the SDK: labels are built, not SDK-checked
        return None


def run(a: argparse.Namespace) -> dict[str, Any]:
    K.check_run_id(a.run_id)
    a.code_sha = K.code_sha_from_env(a.code_sha)
    decl = None if a.dry_run else K.stored_declaration(a.run_id, a.declared)  # the target comes from here
    if decl is not None:  # before any work: the eval items this run is checked against are the declared ones
        K.check_test_set(decl, K.sha256_file(Path(a.items_dir or items_dir()) / "MANIFEST.json"),
                         "t1/evals/items/MANIFEST.json")
    work = Path(a.workdir or tempfile.mkdtemp(prefix="sft-build-"))
    out_dir = work / "out"
    out_dir.mkdir(parents=True, exist_ok=False)
    if not a.dry_run:
        sdk = ensure_sdk(a.sdk_wheel, work, a.region)
        K.log(f"sdk: {sdk}")
    manifest, eval_items = build(a, out_dir)
    if not a.dry_run:
        manifest["sdk_install"] = sdk
    manifest_sha = K.write_json(out_dir / "manifest.json", manifest)
    _, sums_sha = K.write_sha256sums(out_dir)
    final = manifest["overlap"]["final_check"]
    ok = final[OVERLAP] == 0 and final["within_set_duplicates"] == 0 and final["eval_items_overlapped"] == 0
    K.log(f"manifest sha256 {manifest_sha}; counts {manifest['counts']['written']['by_unit']}; overlap {final}")
    summary = {"run_id": a.run_id, "out": a.out, "manifest_sha256": manifest_sha, "sha256sums_sha256": sums_sha,
               "counts": manifest["counts"]["written"], "overlap": {k: manifest["overlap"][k] for k in (
                   "generated", "dropped_eval_overlap", "dropped_eval_by_hash", "dropped_duplicate", "kept")},
               "final_check": final, "local_dir": str(out_dir), "uploaded": False,
               "eval_items": {"n": len(eval_items), "failed": final["eval_items_overlapped"]}}
    if a.dry_run:
        return summary
    if ok:
        K.upload_tree(out_dir, a.out, region=a.region)
        summary["uploaded"] = True
        K.log(f"uploaded to {a.out}")
    else:
        K.log(f"NOT uploaded: the written files overlap the eval or repeat ({final}); evidence records which")
    K.copy_to(a.summary_dir, out_dir / "manifest.json")
    res = K.write_evidence(a.evidence_dir, decl, eval_items, "t1/sft/build.py", a.code_sha,
                           {"dataset_uri": a.out if summary["uploaded"] else None, "manifest_sha256": manifest_sha,
                            "sha256sums_sha256": sums_sha, "uploaded": summary["uploaded"],
                            "hash_method": final["hash_method"], "rows_written": final["rows_checked"],
                            "rows_overlapping_eval": final[OVERLAP], "within_set_duplicates": final["within_set_duplicates"],
                            "generated": manifest["overlap"]["generated"],
                            "dropped_eval_overlap": manifest["overlap"]["dropped_eval_overlap"],
                            "dropped_duplicate": manifest["overlap"]["dropped_duplicate"],
                            "kept": manifest["overlap"]["kept"], "counts": manifest["counts"]["written"],
                            "stopped_by": manifest["execution"]["generation"].get("stopped_by")})
    K.log(f"result: {res['metric']} = {res['value']} over {res['n_items']} eval items -> "
          f"{'PASSED' if res['passed'] else 'MISSED'}")
    summary["result"] = {k: res[k] for k in ("metric", "value", "passed", "n_items")}
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m t1.sft.build", description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", help="s3:// prefix or local dir for train.jsonl, dev.jsonl, manifest.json, SHA256SUMS")
    ap.add_argument("--evidence-dir", help="local evlib store for the items + result ({evidence_dir})")
    ap.add_argument("--declared", help="the run's declaration as a JSON file (default: read from EVIDENCE_S3)")
    ap.add_argument("--summary-dir", help="also copy manifest.json here ({output_dir})")
    ap.add_argument("--code-sha", help="commit the job runs (else CODE_SHA)")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", K.DEFAULT_REGION))
    ap.add_argument("--sdk-wheel", help="ran-rl 1.2.9 wheel: s3:// object or prefix, file or dir (else SM_CHANNEL_SDK)")
    ap.add_argument("--n-execution", type=int, default=2000)
    ap.add_argument("--n-interface", type=int, default=1000)
    ap.add_argument("--n-safety", type=int, default=1200)
    ap.add_argument("--dev-fraction", type=float, default=0.1)
    ap.add_argument("--time-budget-s", type=float, default=3000.0, help="stop labelling new execution states after this")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--labeler", default="t1.evals.gen_execution:reference", help="module:function(spec) -> ref | None")
    ap.add_argument("--items-dir", help="frozen eval items (default t1/evals/items)")
    ap.add_argument("--workdir")
    ap.add_argument("--dry-run", action="store_true", help="no twin, no AWS: a few template examples + the overlap index")
    a = ap.parse_args(argv)
    if a.dry_run:
        a.n_interface, a.n_safety = min(a.n_interface, 40), min(a.n_safety, 70)
    elif not (a.out and a.evidence_dir):
        ap.error("--out and --evidence-dir are required")
    return a


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    try:
        out = run(a)
    except K.JobError as e:
        K.log(f"FAILED: {e}")
        return 2
    print(K.dumps_doc(out), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
