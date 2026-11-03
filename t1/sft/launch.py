"""The exact tools/sm/run.py command for each t1/sft job. Prints commands; --dry-run runs the runner's
own dry run now (read-only: STS for the account; it writes nothing). It never launches: the printed
launch command does, from a clean tree at the commit to ship.

    python -m t1.sft.launch stage_base [--run-id R]
    python -m t1.sft.launch build [--run-id R]
    python -m t1.sft.launch train_lora --stage-run-id S --data-run-id D --declaration FILE [--smoke] [--run-id R]
    ... add --dry-run to run the runner's dry run

Order: stage_base and build are independent (CPU, different instance types); train_lora needs both
(its declaration is filled from the dataset's manifest by t1.sft.declare). Runner placeholders
({run_id} {evidence_dir} {output_dir} {channel:<name>}) stay literal here; the job fills them.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from t1.sft import common as K

REPO = Path(__file__).resolve().parents[2]
DECL = {"stage_base": REPO / "t1/sft/jobs/stage_base.declaration.json",
        "build": REPO / "t1/sft/jobs/build.declaration.json"}
UNITS = {"stage_base": "t1.sft.stage", "build": "t1.sft.data", "train_lora": "t1.sft.train"}
BASE_SRC = "models/base/otel-llm-1.7b-it/"
BASE_BF16 = "models/base/otel-llm-1.7b-it-bf16/"
DATASETS = "datasets/t1-sft/"
LORA_OUT = "models/t1-otel-1.7b-lora/"
REQ = {
    "stage_base": ["transformers==4.57.6", "accelerate>=1.0,<2", "safetensors>=0.4.3,<1"],
    "build": ["pydantic>=2.7,<3"],
    "train_lora": ["transformers==4.57.6", "peft>=0.17,<0.18", "accelerate>=1.0,<2", "safetensors>=0.4.3,<1",
                   "pydantic>=2.7,<3"],
}
SMOKE_ARGS = "--max-steps 20 --limit-train 160 --limit-dev 48 --grad-accum 1 --epochs 1 --gen-check 2 --train-budget-s 900"


def runner_args(job: str, run_id: str, bucket: str, *, stage_run_id: str | None = None,
                data_run_id: str | None = None, declaration: str | None = None, smoke: bool = False,
                profile: str | None = "fyuz") -> list[str]:
    """Arguments for `python -m tools.sm.run` (without --dry-run / --no-wait)."""
    s3 = f"s3://{bucket}/"
    if job == "stage_base":
        entry = ("-m t1.sft.stage_base --run-id {run_id} --src {channel:model} "
                 f"--src-uri {s3}{BASE_SRC} --dst {s3}{BASE_BF16}{{run_id}}/ "
                 "--evidence-dir {evidence_dir} --summary-dir {output_dir}")
        args = ["--declaration", str(DECL[job]), "--instance-type", "ml.m5.2xlarge", "--volume-gb", "60",
                "--max-run", "5400", "--no-sdk", "--inputs", f"model={s3}{BASE_SRC}"]
    elif job == "build":
        entry = ("-m t1.sft.build --run-id {run_id} "
                 f"--out {s3}{DATASETS}{{run_id}}/ "
                 "--evidence-dir {evidence_dir} --summary-dir {output_dir} --n-execution 2000 --n-interface 1000 "
                 "--n-safety 1200 --dev-fraction 0.1 --time-budget-s 3000 --workers 16")
        args = ["--declaration", str(DECL[job]), "--instance-type", "ml.c5.4xlarge", "--max-run", "4500"]
    elif job == "train_lora":
        if not (stage_run_id and data_run_id and declaration):
            raise K.JobError("train_lora needs --stage-run-id, --data-run-id and --declaration (t1.sft.declare)")
        entry = ("-m t1.sft.train_lora --run-id {run_id} --base {channel:model} --data {channel:data} "
                 f"--out {s3}{LORA_OUT}{{run_id}}/ --base-run-id {stage_run_id} --data-run-id {data_run_id} "
                 "--evidence-dir {evidence_dir} --summary-dir {output_dir} "
                 + (SMOKE_ARGS if smoke else "--train-budget-s 5400"))
        args = ["--declaration", declaration, "--gpu", "--instance-type", "ml.g6.2xlarge",
                "--model-s3", f"{s3}{BASE_BF16}{stage_run_id}/", "--inputs", f"data={s3}{DATASETS}{data_run_id}/",
                "--max-run", "1800" if smoke else "7200", "--no-sdk"]
    else:
        raise K.JobError(f"unknown job {job!r}")
    for r in REQ[job]:
        args += ["--requirement", r]
    out = ["--run-id", K.check_run_id(run_id), "--entry", entry, *args]
    if profile:
        out += ["--profile", profile]
    return out


def shell(args: list[str], *tail: str) -> str:
    return shlex.join(["python", "-m", "tools.sm.run", *args, *tail])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m t1.sft.launch", description=__doc__.split("\n")[0])
    ap.add_argument("job", choices=list(UNITS))
    ap.add_argument("--run-id", help="default: a new one (unit-UTC-hex)")
    ap.add_argument("--stage-run-id")
    ap.add_argument("--data-run-id")
    ap.add_argument("--declaration", help="train_lora: the filled declaration (python -m t1.sft.declare train_lora)")
    ap.add_argument("--smoke", action="store_true", help="train_lora: the 20-step smoke run")
    ap.add_argument("--profile", default="fyuz")
    ap.add_argument("--bucket", help="default: tools/sm/spec.py's team bucket")
    ap.add_argument("--dry-run", action="store_true", help="run the runner's dry run now (writes nothing)")
    a = ap.parse_args(argv)
    from tools import evlib
    from tools.sm import spec

    run_id = a.run_id or evlib.new_run_id(UNITS[a.job] + ("-smoke" if a.smoke else ""))
    try:
        args = runner_args(a.job, run_id, a.bucket or spec.BUCKET, stage_run_id=a.stage_run_id,
                           data_run_id=a.data_run_id, declaration=a.declaration, smoke=a.smoke, profile=a.profile)
    except K.JobError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 2
    print(f"# dry run (writes nothing):\n{shell(args, '--dry-run')}\n")
    print(f"# launch (declares to S3 first, then CreateTrainingJob):\n{shell(args, '--no-wait')}\n")
    print(f"# later:\npython -m tools.sm.run --finish {run_id} --profile {a.profile}")
    if a.dry_run:
        from tools.sm import run as R

        print("\n# ---- runner dry run ----", flush=True)
        return R.main([*args, "--dry-run"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
