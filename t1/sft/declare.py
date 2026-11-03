"""Declarations for the t1/sft jobs, in the format `tools/sm/run.py --declaration` takes.

    t1/sft/jobs/stage_base.declaration.json   static: pass it to the runner as is
    t1/sft/jobs/build.declaration.json        static: pass it to the runner as is
    t1/sft/jobs/train_lora.declaration.json   a template: the dataset run fixes n_items (its dev split)
                                              and the test set (dev.jsonl's sha256), so it is filled
                                              after the build job:

    python -m t1.sft.declare check
    python -m t1.sft.declare train_lora --stage-run-id S --data-run-id D \\
        --manifest s3://<team-bucket>/datasets/t1-sft/D/manifest.json --out FILE [--smoke] [--profile fyuz]

`check` puts every declaration through the runner's own builder (tools.sm.run.build_declaration, which
hashes test_set/prompt paths at the commit and validates with team13_contracts), so what passes here
is what the runner will store. Reading the manifest from S3 is one GetObject (read-only).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from t1.sft import common as K

JOBS_DIR = Path(__file__).with_name("jobs")
JOBS = ("stage_base", "build", "train_lora")
_ONE = re.compile(r"^\{\{([a-z_0-9]+)\}\}$")
_ANY = re.compile(r"\{\{([a-z_0-9]+)\}\}")
SMOKE = {"limit_dev": 48, "limit_train": 160, "max_steps": 20, "epochs": 1, "grad_accum": 1,
         "target": {"metric": "mean_score", "op": "<", "value": 1.0, "unit": "ratio"}}


def path(job: str) -> Path:
    if job not in JOBS:
        raise K.JobError(f"unknown job {job!r}; one of {JOBS}")
    return JOBS_DIR / f"{job}.declaration.json"


def load(job: str) -> dict[str, Any]:
    return json.loads(path(job).read_text())


def placeholders(obj: Any) -> set[str]:
    if isinstance(obj, str):
        return set(_ANY.findall(obj))
    if isinstance(obj, dict):
        return set().union(*(placeholders(v) for v in obj.values())) if obj else set()
    if isinstance(obj, list):
        return set().union(*(placeholders(v) for v in obj)) if obj else set()
    return set()


def fill(obj: Any, values: dict[str, Any]) -> Any:
    """Replace {{name}}; a string that is exactly one placeholder takes the value's type."""
    if isinstance(obj, str):
        m = _ONE.match(obj)
        if m and m.group(1) in values:
            return values[m.group(1)]
        return _ANY.sub(lambda mm: str(values[mm.group(1)]) if mm.group(1) in values else mm.group(0), obj)
    if isinstance(obj, dict):
        return {k: fill(v, values) for k, v in obj.items()}
    if isinstance(obj, list):
        return [fill(v, values) for v in obj]
    return obj


def through_runner(decl: dict[str, Any], run_id: str = "t1.sft.declare-check", sha: str | None = None) -> dict:
    """The declaration exactly as tools/sm/run.py would store it for a launch at commit `sha`
    (default HEAD). Raises K.JobError with the runner's refusal."""
    import tempfile

    from tools.sm import package
    from tools.sm import run as R

    sha = sha or package.resolve_commit(R.REPO, "HEAD")
    # a file, not JSON text: the runner tests Path(arg).is_file() first, which fails on long text (ENAMETOOLONG)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(decl, f)
    try:
        a = R.parser().parse_args(["--run-id", run_id, "--entry", "-m t1.sft.declare", "--declaration", f.name])
        return R.build_declaration(a, sha, {"launcher": "t1/sft/declare.py check"})
    except R.Refusal as e:
        raise K.JobError(f"the runner refuses this declaration: {e}") from None
    finally:
        Path(f.name).unlink(missing_ok=True)


def read_manifest(uri: str, profile: str | None = None, region: str = K.DEFAULT_REGION) -> tuple[dict, str]:
    if K.is_s3(uri):
        import boto3

        bucket, _, key = uri[len("s3://"):].partition("/")
        body = boto3.Session(profile_name=profile, region_name=region).client("s3").get_object(
            Bucket=bucket, Key=key)["Body"].read()
    else:
        body = Path(uri).read_bytes()
    return json.loads(body), hashlib.sha256(body).hexdigest()


def train_declaration(stage_run_id: str, data_run_id: str, manifest: dict[str, Any], manifest_sha: str,
                      smoke: bool = False) -> dict[str, Any]:
    if manifest.get("run_id") != data_run_id:
        raise K.JobError(f"the manifest is for run {manifest.get('run_id')!r}, not {data_run_id!r}")
    final = manifest["overlap"]["final_check"]
    if final.get("eval_overlap_count") != 0 or manifest.get("dry_run"):
        raise K.JobError("refusing to declare training on a dataset with eval overlap or from a dry run")
    n_dev = manifest["counts"]["written"]["by_split"].get("dev", 0)
    values = {"stage_run_id": K.check_run_id(stage_run_id), "data_run_id": K.check_run_id(data_run_id),
              "n_dev": min(n_dev, SMOKE["limit_dev"]) if smoke else n_dev,
              "subset": (f"random.Random(14).sample(dev.jsonl rows, {SMOKE['limit_dev']}) (--limit-dev, smoke)"
                         if smoke else None),
              "dev_jsonl_sha256": manifest["files"]["dev.jsonl"]["sha256"], "data_manifest_sha256": manifest_sha,
              "epochs": SMOKE["epochs"] if smoke else 2, "grad_accum": SMOKE["grad_accum"] if smoke else 2,
              "max_steps": SMOKE["max_steps"] if smoke else None}
    decl = fill(copy.deepcopy(load("train_lora")), values)
    if decl.get("subset") is None:
        decl.pop("subset", None)
    if smoke:
        decl["target"] = dict(SMOKE["target"])
        decl["decoding"]["smoke"] = {k: SMOKE[k] for k in ("limit_dev", "limit_train", "max_steps")}
    left = placeholders(decl)
    if left:
        raise K.JobError(f"unfilled placeholders: {sorted(left)}")
    return decl


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m t1.sft.declare", description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["check", "train_lora"])
    ap.add_argument("--stage-run-id")
    ap.add_argument("--data-run-id")
    ap.add_argument("--manifest", help="the dataset run's manifest.json (s3:// or a file)")
    ap.add_argument("--out", help="write the filled train_lora declaration here (outside the repo is fine)")
    ap.add_argument("--smoke", action="store_true", help="the 20-step smoke run's declaration")
    ap.add_argument("--profile")
    a = ap.parse_args(argv)
    try:
        if a.what == "check":
            for job in ("stage_base", "build"):
                d = through_runner(load(job))
                print(f"{job}: OK  unit={d['unit']} target={d['target']} n_items={d['n_items']} "
                      f"test_set_sha256={d['test_set_sha256']}")
            demo = {"run_id": "t1.sft.data-check", "dry_run": False, "files": {"dev.jsonl": {"sha256": "0" * 64}},
                    "counts": {"written": {"by_split": {"dev": 400}}},
                    "overlap": {"final_check": {"eval_overlap_count": 0}}}
            d = through_runner(train_declaration("t1.sft.stage-check", "t1.sft.data-check", demo, "1" * 64))
            print(f"train_lora (template, example values): OK  unit={d['unit']} target={d['target']}")
            return 0
        if not (a.stage_run_id and a.data_run_id and a.manifest and a.out):
            ap.error("train_lora needs --stage-run-id, --data-run-id, --manifest and --out")
        manifest, msha = read_manifest(a.manifest, a.profile)
        decl = train_declaration(a.stage_run_id, a.data_run_id, manifest, msha, a.smoke)
        through_runner(decl)
        Path(a.out).write_text(json.dumps(decl, indent=2, sort_keys=True) + "\n")
        print(f"wrote {a.out}: n_items={decl['n_items']} test_set_sha256={decl['test_set_sha256']} "
              f"target={decl['target']}")
        return 0
    except K.JobError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
