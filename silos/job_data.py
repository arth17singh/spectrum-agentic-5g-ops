"""Job entry: regenerate the simulated-silo datasets on AWS and publish them with a SHA-256 manifest.

    python -m silos.job_data                      # inside the CPU job
    python -m silos.job_data --dry-run            # tiny config, nothing uploaded, nothing written; prints a summary
    python -m silos.job_data --print-declaration  # the declaration the runner writes BEFORE the run

Why: the datasets of record must have an AWS job as their provenance. The job runs `silos.generate` on the
organisers' in-process Maveric twin, runs `silos.checks` on the result, uploads each simulated silo's files to
that silo's bucket (the FL service's jobs read only same-Region buckets) and the whole tree plus the manifest to
the evidence copy, verifying every upload by SHA-256, and records one `item` per file and one `result`.

Environment (set by the job runner)
  RUN_ID / EVIDENCE_RUN_ID   evidence run id             CODE_SHA   40-hex commit the code was packaged from
  EVIDENCE_S3                evidence prefix             AWS_REGION region of the job (default us-east-2)
  SILO_BUCKET_A/_B/_C        the three silo buckets (us-west-2; names are given at launch, never committed)
  SILO_BUCKET_REGION         region of those buckets (default us-west-2)
  DATA_COPY_S3               s3://bucket/t2-silos/<data run>/  the evidence copy of the whole tree (same Region as the job)
  LAPTOP_COPY_S3             optional s3:// prefix holding the earlier laptop-generated files: the job reports how
                             many are byte-identical and the largest numeric difference (a measurement, no bar)

Byte identity with the laptop copy is NOT expected across CPU architectures (floating-point differences in the
GP models); the AWS copy is the dataset of record and its manifest replaces the laptop one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from . import checks as checks_mod
from . import evio
from .generate import DEFAULT_CONFIG, DEFAULT_MANIFEST, canonical, config_sha256, file_sha256, generate, load_config

UNIT = "t2.silo_data"
GRADER = "silos.checks (schema, ranges, seeds, proportional rows, operating points) + SHA-256 read-back from S3"
BACKEND = "silos.generate on the in-process Maveric twin, CPU job"
MODEL_ID = "t2-simulated-silos@v1"
SILOS = ("A", "B", "C")
SPLITS = ("disjoint", "overlap")


def tiny_config(cfg: dict) -> dict:
    cfg = json.loads(json.dumps(cfg))
    cfg["steps_per_cluster"] = 20
    for s in cfg["silos"].values():
        s["clusters"] = 1
    cfg["silos"]["A"]["clusters"] = 2
    cfg["test"]["clusters_per_slice"] = 1
    cfg["test"]["steps_per_cluster"] = 10
    return cfg


def expected_paths(cfg: dict) -> list[str]:
    out = []
    for sp in SPLITS:
        for s in cfg["silos"]:
            out += [f"{sp}/{s}/train.csv", f"{sp}/{s}/test.csv"]
        out.append(f"{sp}/shared_test/test.csv")
    return out


def declaration_for(cfg: dict, run_id: str, code_sha: str) -> dict:
    n = len(expected_paths(cfg))
    return evio.declaration(
        run_id=run_id, unit=UNIT, target={"metric": "pass_count", "op": ">=", "value": n, "unit": "files (checked, uploaded, SHA-256 verified)"},
        n_items=n, test_set_sha256=config_sha256(cfg), model_id=MODEL_ID, grader=GRADER, code_sha=code_sha,
        extra={"note": "Provenance of the simulated-silo datasets. test_set_sha256 is the sha256 of the generator config "
                       "(silos/config.yaml, canonical JSON). The comparison with the earlier laptop copy is a measurement, "
                       "not a target.",
               "entry": "python -m silos.job_data"})


# --------------------------------------------------------------------------------------------- S3
def _s3(region: str):
    import boto3

    return boto3.client("s3", region_name=region)


def _split_uri(uri: str) -> tuple[str, str]:
    assert uri.startswith("s3://"), uri
    b, _, k = uri[5:].partition("/")
    return b, k.rstrip("/")


def put_verified(client, bucket: str, key: str, path: Path, sha256_hex: str) -> dict:
    """PutObject with server-side SHA-256 integrity (ChecksumSHA256), then read the checksum back with HeadObject."""
    import base64

    b64 = base64.b64encode(bytes.fromhex(sha256_hex)).decode()
    client.put_object(Bucket=bucket, Key=key, Body=path.read_bytes(), ChecksumSHA256=b64, ContentType="text/csv")
    head = client.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
    got = head.get("ChecksumSHA256")
    return {"bucket": bucket, "key": key, "bytes": head["ContentLength"], "verified": got == b64}


def numeric_diff(a: Path, b: Path) -> dict:
    import pandas as pd

    da, db = pd.read_csv(a), pd.read_csv(b)
    if list(da.columns) != list(db.columns) or len(da) != len(db):
        return {"comparable": False, "rows": [len(da), len(db)]}
    num = [c for c in da.columns if da[c].dtype.kind in "fi"]
    mx = float((da[num] - db[num]).abs().to_numpy().max()) if num else 0.0
    skip = [c for c in da.columns if c == "generator_commit"]  # differs by design: the commit that generated the file
    other_equal = bool((da.drop(columns=num + skip) == db.drop(columns=num + skip)).all().all())
    return {"comparable": True, "max_abs_diff": mx, "non_numeric_equal": other_equal}


# ------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="tiny config; no S3, no evidence writes")
    ap.add_argument("--print-declaration", action="store_true")
    ap.add_argument("--declare-only", action="store_true", help="write the declaration via evlib and exit (the runner does this BEFORE launching the job)")
    ap.add_argument("--declare-first", action="store_true", help="write the declaration via evlib before running")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)

    cfg = load_config(DEFAULT_CONFIG)
    if a.dry_run:
        cfg = tiny_config(cfg)
    run_id = evio.env_run_id() or ("dryrun-t2-silo-data" if a.dry_run or a.print_declaration else None)
    if not run_id:
        sys.exit("set RUN_ID (or EVIDENCE_RUN_ID)")
    code_sha = evio.env_code_sha(required=not (a.dry_run or a.print_declaration))
    declared = declaration_for(cfg, run_id, code_sha)
    if a.print_declaration:
        evio.print_json(declared)
        return 0
    if a.declare_only:
        evio.get_evlib().write_declared(declared)
        print(f"declared {declared['run_id']} ({declared['unit']})")
        return 0
    if a.declare_first and not a.dry_run:
        evio.declare_first(declared)

    out = Path(a.out_dir or tempfile.mkdtemp(prefix="silos-"))
    manifest_path = out / "manifest.json"
    manifest = generate(cfg, out, workers=a.workers, manifest_path=manifest_path, progress=True)
    rep = checks_mod.run_checks(out, manifest_path, config_path=None if a.dry_run else DEFAULT_CONFIG)
    for name, ok, detail in rep.results:
        print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""), flush=True)

    committed = {}
    if DEFAULT_MANIFEST.exists() and not a.dry_run:
        committed = {f["path"]: f["sha256"] for f in json.loads(DEFAULT_MANIFEST.read_text())["files"]}
    laptop = os.environ.get("LAPTOP_COPY_S3", "").strip()
    bucket_of = {s: os.environ.get(f"SILO_BUCKET_{s}", "") for s in SILOS}
    silo_region = os.environ.get("SILO_BUCKET_REGION", "us-west-2")
    copy_uri = os.environ.get("DATA_COPY_S3", "").strip()
    uploads: dict[str, list[dict]] = {}
    if not a.dry_run:
        missing = [k for k, v in bucket_of.items() if not v] + ([] if copy_uri else ["DATA_COPY_S3"])
        if missing:
            sys.exit(f"missing environment: {missing}")
        s3_silo, s3_copy = _s3(silo_region), _s3(evio.region())
        cb, ck = _split_uri(copy_uri)
        for f in manifest["files"]:
            p, sha = out / f["path"], f["sha256"]
            ups = [put_verified(s3_copy, cb, f"{ck}/{f['path']}", p, sha)]
            for s in SILOS:  # a silo's bucket holds its own files and the shared test
                if f["silo"] in (s, "shared"):
                    ups.append(put_verified(s3_silo, bucket_of[s], f["path"], p, sha))
            uploads[f["path"]] = ups
        s3_copy.put_object(Bucket=cb, Key=f"{ck}/manifest.json", Body=manifest_path.read_bytes())
        for i, s in enumerate(SILOS, 1):
            doc = {"silo": s, "client_index": i, "run_id": run_id, "config_sha256": manifest["config_sha256"],
                   "files": [{"path": f["path"], "rows": f["rows"], "sha256": f["sha256"]} for f in manifest["files"] if f["silo"] in (s, "shared")]}
            s3_silo.put_object(Bucket=bucket_of[s], Key="silo.json", Body=json.dumps(doc, indent=1).encode(), ContentType="application/json")

    items, exact, diffs = [], 0, {}
    for f in manifest["files"]:
        path, sha = f["path"], f["sha256"]
        up_ok = a.dry_run or all(u["verified"] for u in uploads.get(path, []))
        lap = {}
        if laptop and not a.dry_run:
            lb, lk = _split_uri(laptop)
            tmp = out / "_laptop" / path
            tmp.parent.mkdir(parents=True, exist_ok=True)
            try:
                _s3(evio.region()).download_file(lb, f"{lk}/{path}", str(tmp))
                lap = numeric_diff(out / path, tmp)
                lap["bytes_identical"] = file_sha256(tmp) == sha
            except Exception as e:  # noqa: BLE001
                lap = {"error": repr(e)[:200]}
        same_as_committed = committed.get(path) == sha if committed else None
        exact += int(bool(same_as_committed) or bool(lap.get("bytes_identical")))
        diffs[path] = lap
        items.append(evio.item(
            item_id=path,
            input_sha256=hashlib.sha256(canonical({"config_sha256": manifest["config_sha256"], "path": path, "seeds": [s["seed"] for s in f["seeds"]]})).hexdigest(),
            output_sha256=sha, grader=GRADER, backend=BACKEND, passed=bool(rep.ok and up_ok),
            rows=f["rows"], clusters=f["clusters"], uploaded_to=[f"s3://{u['bucket']}/{u['key']}" for u in uploads.get(path, [])] if not a.dry_run else [],
            committed_laptop_sha256=committed.get(path), equals_committed_laptop_sha256=same_as_committed, laptop_comparison=lap or None))
    details = {"checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in rep.results], "config_sha256": manifest["config_sha256"],
               "generator_commit_in_job": "n/a (no git in the job); CODE_SHA is the code_sha of the declaration",
               "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "seconds": manifest["seconds"],
               "files_equal_to_laptop_copy": exact if (committed or laptop) else None,
               "laptop_note": "Byte identity across CPU architectures is not expected; max_abs_diff per file is in the item records.",
               "dry_run": a.dry_run}
    summary = evio.finish(declared=declared, items=items, script="silos/job_data.py", code_sha=code_sha, details=details,
                          dry_run=a.dry_run, sink=evio.Sink(run_id))
    r = summary["result"]
    print(json.dumps({"run_id": run_id, "passed": r["passed"], "value": r["value"], "target": r["target"], "n_items": r["n_items"],
                      "written": summary["written"]}, indent=1))
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
