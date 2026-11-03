"""Fill a t1/sft job's template (t1/sft/jobs/<job>.json), validate the declaration, optionally write it.

    python -m t1.sft.declare stage_base --run-id R --code-sha SHA --set bucket=<team-bucket>
    python -m t1.sft.declare train_lora --run-id R --code-sha SHA --set bucket=... --set stage_run_id=... \\
        --set data_run_id=... --set n_dev=380 --set dev_jsonl_sha256=... --set data_manifest_sha256=... \\
        [--write --evidence s3://<team-bucket>/evidence/]

Placeholders are "{{name}}"; a string that is exactly one placeholder takes the value's type (so
n_items can be an int). Any placeholder left unfilled is an error. The declaration is checked with
tools/evlib.validate (which runs team13_contracts.validate_record); --write stores it through
evlib.write_declared (declared_at stamped then, never overwritten). The job runner may do the
writing instead; this is the same template either way.
"""

from __future__ import annotations

import argparse
import copy
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


def load(job: str) -> dict[str, Any]:
    if job not in JOBS:
        raise K.JobError(f"unknown job {job!r}; one of {JOBS}")
    return json.loads((JOBS_DIR / f"{job}.json").read_text())


def placeholders(obj: Any) -> set[str]:
    if isinstance(obj, str):
        return set(_ANY.findall(obj))
    if isinstance(obj, dict):
        return set().union(*(placeholders(v) for v in obj.values())) if obj else set()
    if isinstance(obj, list):
        return set().union(*(placeholders(v) for v in obj)) if obj else set()
    return set()


def fill(obj: Any, values: dict[str, Any]) -> Any:
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


def _typed(v: str) -> Any:
    return int(v) if re.fullmatch(r"-?\d+", v) else v


def declaration(job: str, values: dict[str, Any]) -> dict[str, Any]:
    """The filled declaration, validated as it will be stored (created_at/declared_at stamped now)."""
    from tools import evlib

    rec = fill(copy.deepcopy(load(job)["declared"]), values)
    left = placeholders(rec)
    if left:
        raise K.JobError(f"unfilled placeholders in the {job} declaration: {sorted(left)}")
    now = evlib.utc_now()
    check = dict(rec, created_at=now, declared_at=now, aws_time=None, aws_time_source=None)
    evlib.validate(check)
    return rec


def job_spec(job: str, values: dict[str, Any]) -> dict[str, Any]:
    spec = fill(copy.deepcopy(load(job)["job"]), values)
    return spec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m t1.sft.declare", description=__doc__.split("\n")[0])
    ap.add_argument("job", choices=JOBS)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--code-sha", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--write", action="store_true", help="write declared/<run_id>.json through evlib")
    ap.add_argument("--evidence", help="s3:// prefix or dir (else EVIDENCE_S3 / EVIDENCE_DIR)")
    a = ap.parse_args(argv)
    values: dict[str, Any] = {"run_id": K.check_run_id(a.run_id), "code_sha": K.code_sha_from_env(a.code_sha)}
    for kv in a.set:
        k, _, v = kv.partition("=")
        values[k] = _typed(v)
    try:
        rec = declaration(a.job, values)
    except K.JobError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 2
    if a.write:
        from tools import evlib

        store = K.evidence_store(a.evidence)
        if store is None:
            print("FAILED: no evidence store", file=sys.stderr)
            return 2
        print(evlib.write_declared(rec, store=store))
    else:
        print(K.dumps_doc(rec), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
