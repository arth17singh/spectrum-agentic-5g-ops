"""Fill the declaration of the Track 1 RL scoring run (unit t1.rl.eval) and print its launch commands.

    python -m t1.rl.eval_declaration --profile fyuz --out decl-eval.json \\
        [--run-id t1.rl.eval-...] [--url-secret-arn ARN --token-secret-arn ARN]

Read-only AWS (S3 GetObject, SageMaker DescribeTrainingJob; Environment is never read out or printed).
Writes ONE local file (--out), the `--declaration` for tools/sm/run.py, then prints the run.py command
lines for a dry run and for the launch. It never launches and never writes to S3.

Why its own run id: the training run's declaration names code_sha 9b6a102, which has no evaluation code,
and the runner refuses a job whose commit differs from its declaration. So the scoring run is declared
anew, just before its launch, with the SAME target, the SAME six items and the SAME test set (the sha256
of config/eval_scenarios.yaml) as the training declaration, and names the training run as lineage, the
training job and its model artifact's sha256 as the model under test, and the packed project's sha256.

Refuses (exit 2) unless: the training job is Completed with a model artifact; the packed project's
config/eval_scenarios.yaml, env.yaml, reward.yaml and team/rewards.py have the sha256 the training run
declared; the training declaration has the six items.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
import shlex
import sys
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "contracts"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from t1.rl.job_evaluate import CONFIG_SHAS, ITEM_IDS  # noqa: E402
from tools import evlib  # noqa: E402
from tools.sm import spec as smspec  # noqa: E402

SPEC_PATH = ROOT / "tools" / "sm" / "examples" / "t1-rl-eval.json"
MAX_BYTES = 512 * 1024 * 1024


class Refusal(RuntimeError):
    pass


def load_spec(path: Path = SPEC_PATH) -> dict:
    return json.loads(path.read_text())


def split_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise Refusal(f"not an s3:// URI: {uri!r}")
    b, _, k = uri[len("s3://"):].partition("/")
    return b, k


def get_bytes(s3, uri: str) -> bytes:
    b, k = split_uri(uri)
    r = s3.get_object(Bucket=b, Key=k)
    if int(r.get("ContentLength") or 0) > MAX_BYTES:
        raise Refusal(f"{uri} is larger than {MAX_BYTES} bytes")
    return r["Body"].read()


def tar_file(data: bytes, name: str) -> bytes | None:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as t:
        for m in t.getmembers():
            if m.isfile() and m.name.removeprefix("./") == name:
                f = t.extractfile(m)
                return f.read() if f else None
    return None


def unquote(v: Any) -> Any:
    return smspec.unquote_hp(v)


def build(aws, sp: dict, bucket: str = smspec.BUCKET) -> tuple[dict, dict]:
    """(declaration template for run.py --declaration, facts for the command lines)."""
    s3, sm = aws.client("s3"), aws.client("sagemaker")
    tr_rid, tr_job = sp["training"]["run_id"], sp["training"]["job_name"]
    ev_store = evlib.S3Store(f"s3://{bucket}/{smspec.EVIDENCE_PREFIX}", client=s3)
    train = evlib.read_declared(tr_rid, ev_store)
    if train is None:
        raise Refusal(f"no training declaration declared/{tr_rid}.json")
    tcfg = (train.get("twin") or {}).get("config") or {}
    if sorted(tcfg.get("items") or {}) != sorted(ITEM_IDS):
        raise Refusal(f"the training declaration's items are {sorted(tcfg.get('items') or {})}, not {sorted(ITEM_IDS)}")
    if train.get("n_items") != len(ITEM_IDS):
        raise Refusal(f"the training declaration's n_items is {train.get('n_items')}")
    d = sm.describe_training_job(TrainingJobName=tr_job)
    if d.get("TrainingJobStatus") != "Completed":
        raise Refusal(f"training job {tr_job} is {d.get('TrainingJobStatus')}, not Completed: wait for it")
    model_uri = (d.get("ModelArtifacts") or {}).get("S3ModelArtifacts")
    if not model_uri:
        raise Refusal(f"training job {tr_job} has no ModelArtifacts.S3ModelArtifacts")
    project_uri = unquote((d.get("HyperParameters") or {}).get("sagemaker_submit_directory"))
    if not isinstance(project_uri, str) or not project_uri.startswith("s3://"):
        raise Refusal(f"training job {tr_job} has no sagemaker_submit_directory")
    model = get_bytes(s3, model_uri)
    project = get_bytes(s3, project_uri)
    model_sha, project_sha = evlib.sha256_bytes(model), evlib.sha256_bytes(project)
    zip_bytes = tar_file(model, "model.zip")
    if zip_bytes is None:
        raise Refusal(f"{model_uri} has no model.zip")
    checks = {"config/eval_scenarios.yaml": train["test_set_sha256"]}
    checks.update({p: tcfg.get(k) for p, k in CONFIG_SHAS.items()})
    for rel, want in checks.items():
        b = tar_file(project, rel)
        if b is None:
            raise Refusal(f"the packed project {project_uri} has no {rel}")
        if evlib.sha256_bytes(b) != want:
            raise Refusal(f"{rel} in the packed project has sha256 {evlib.sha256_bytes(b)}, the training run "
                          f"declared {want}")
    cfg = {k: copy.deepcopy(v) for k, v in tcfg.items() if k not in ("runner", "training", "evaluation", "purpose",
                                                                       "logging", "dev_evidence")}
    tev = copy.deepcopy(tcfg.get("evaluation") or {})
    tev["where"] = (f"SageMaker CPU job ({sp['instance_type']}) launched by tools/sm/run.py under THIS run id, "
                    "t1/rl/job_evaluate.py; inputs: the training job's model.tar.gz and its own packed project "
                    "(identical configs, sha256-checked in the job)")
    cfg.update({
        "purpose": "Track 1 RL unit, judging run 16: score the PPO policy trained in the lineage run against the "
                   "SDK rule baseline on the fixed evaluation set",
        "evaluation": tev,
        "logging": "httpx/httpcore loggers at WARNING and the job's stdout/stderr masked, so the twin host is not "
                   "logged; the token is never logged",
        "eval": {
            "target_first_declared": (
                f"The target, the six items and the test set were first declared BEFORE training, in "
                f"declared/{tr_rid}.json (declared_at {train.get('declared_at')}); this declaration copies them "
                f"unchanged. It is a new run id because the training declaration's code_sha "
                f"({train.get('code_sha', '')[:7]}) has no evaluation code and the runner refuses a job whose "
                f"commit differs from its declaration; the training declaration's evaluation.where ('same run "
                f"id') is superseded by this one."),
            "training_run_id": tr_rid,
            "training_declared_at": train.get("declared_at"),
            "training_code_sha": train.get("code_sha"),
            "training_job_name": tr_job,
            "training_job_arn": d.get("TrainingJobArn"),
            "training_end_time": smspec.iso(d.get("TrainingEndTime")),
            "model_artifact_uri": model_uri,
            "model_artifact_sha256": model_sha,
            "model_artifact_bytes": len(model),
            "model_zip_sha256": evlib.sha256_bytes(zip_bytes),
            "project_uri": project_uri,
            "project_sha256": project_sha,
            "project_bytes": len(project),
            "model_form": "model.zip from model.tar.gz, passed to evaluate --model as the bare .zip: the SDK "
                          "checksum is then sha256(model.zip), which the SDK registry requires to equal the "
                          "release package's manifest model_sha256",
            "items_rule": "SAFETY, SERVICE, UPLIFT, REPEAT, TRACE from evaluation.json gates (pass is true); "
                          "COVERAGE: aggregates.overall.coverage_mean of the model >= the baseline's, no "
                          "tolerance; a gate with pass: null counts as NOT passed",
            "channels": sp["channels"],
        },
    })
    decl = {
        "unit": sp["unit"], "track": train["track"], "target": train["target"], "n_items": train["n_items"],
        "test_set_sha256": train["test_set_sha256"], "model_id": train["model_id"],
        "grader": ("ran_rl.evaluation.evaluate --with-baseline (RAN RL SDK 1.2.9) on the team-hosted Maveric twin "
                   "(http) in a SageMaker CPU job (t1/rl/job_evaluate.py); one item per SDK gate (SAFETY, SERVICE, "
                   "UPLIFT, REPEAT, TRACE) plus COVERAGE; the run passes only if all six pass"),
        "synthetic": False,
        "twin": {k: train["twin"][k] for k in ("kind", "version", "scenario", "seed")} | {"config": cfg},
    }
    facts = {"model_uri": model_uri, "project_uri": project_uri, "model_sha256": model_sha,
             "project_sha256": project_sha, "training_status": d.get("TrainingJobStatus")}
    return decl, facts


def commands(sp: dict, run_id: str, decl_path: str, facts: dict, url_arn: str | None, token_arn: str | None,
             profile: str | None) -> dict[str, str]:
    arns = {"RAN_RL_TWIN_URL": url_arn or f"<ARN of {sp['secrets']['RAN_RL_TWIN_URL']}>",
            "RAN_RL_ADAPTER_TOKEN": token_arn or f"<ARN of {sp['secrets']['RAN_RL_ADAPTER_TOKEN']}>"}
    argv = ["python", "-m", "tools.sm.run", "--run-id", run_id, "--declaration", decl_path,
            "--instance-type", sp["instance_type"], "--max-run", str(sp["max_run_s"])]
    for p in sp["paths"]:
        argv += ["--paths", p]
    argv += [f"--entry={sp['entry']}", "--inputs", f"model={facts['model_uri']}",
             "--inputs", f"project={facts['project_uri']}"]
    for k, v in arns.items():
        argv += ["--secret", f"{k}={v}"]
    if profile:
        argv += ["--profile", profile]
    base = shlex.join(argv)
    return {"dry_run": base + " --dry-run", "launch": base + " --no-wait",
            "finish": f"python -m tools.sm.run --finish {run_id}" + (f" --profile {profile}" if profile else "")}


def main(argv: list[str] | None = None, aws=None, out=print) -> int:
    ap = argparse.ArgumentParser(prog="python -m t1.rl.eval_declaration", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="where to write the declaration JSON (local file)")
    ap.add_argument("--run-id", help="default: a new t1.rl.eval-<UTC>-<hex> run id")
    ap.add_argument("--url-secret-arn")
    ap.add_argument("--token-secret-arn")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    ap.add_argument("--region", default=smspec.REGION)
    ap.add_argument("--bucket", default=smspec.BUCKET)
    ap.add_argument("--spec", default=str(SPEC_PATH))
    a = ap.parse_args(argv)
    sp = load_spec(Path(a.spec))
    rid = a.run_id or evlib.new_run_id(sp["unit"])
    if not evlib.RUN_ID_RE.match(rid):
        out(f"refused: run id {rid!r} is not valid")
        return 2
    for name, arn in (("--url-secret-arn", a.url_secret_arn), ("--token-secret-arn", a.token_secret_arn)):
        if arn and not smspec.SECRET_ARN_RE.match(arn):
            out(f"refused: {name} is not a Secrets Manager secret ARN")
            return 2
    if aws is None:
        from tools.sm.run import Aws  # noqa: PLC0415

        aws = Aws(a.profile, a.region)
    try:
        decl, facts = build(aws, sp, a.bucket)
    except Refusal as e:
        out(f"refused: {e}")
        return 2
    Path(a.out).write_text(json.dumps(decl, indent=2, sort_keys=True) + "\n")
    out(f"declaration template -> {a.out} (run.py adds run_id, code_sha, region and twin.config.runner)")
    out(f"model artifact  {facts['model_uri']}  sha256 {facts['model_sha256']}")
    out(f"packed project  {facts['project_uri']}  sha256 {facts['project_sha256']}")
    price = smspec.PRICES[sp["instance_type"]]
    out(f"instance        {sp['instance_type']} ${price}/h; at most ${price * sp['max_run_s'] / 3600:.3f} for "
        f"--max-run {sp['max_run_s']}s; expected about ${price * sp['expected_runtime_s'] / 3600:.2f} "
        f"(~{sp['expected_runtime_s'] // 60} min)")
    for k, v in commands(sp, rid, a.out, facts, a.url_secret_arn, a.token_secret_arn, a.profile).items():
        out(f"{k}:\n  {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
