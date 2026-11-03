"""Record the two Track 1 training curves as evidence (unit t1.curves), from finished AWS jobs.

A CPU job through the team runner (tools/sm/run.py, ml.m5.xlarge, no SDK). It reads, never trains:

  SFT (LoRA)  s3 .../models/t1-otel-1.7b-lora/<sft run id>/train_metrics.json, written by the SFT training job
              itself (train_loss_curve: one point per optimiser step; dev_loss_curve: step 0 and 3 later
              evaluations). It arrives as channel `sft_metrics`; the file's sha256 must equal its line in the
              run's SHA256SUMS (channel `sft_sums`), and that file's sha256 must be the one the model record
              carries (sources.json: sha256sums_sha256, the `artifact_sha256` of model t1-otel-1.7b-lora).
  RL (PPO)    CloudWatch Logs /aws/sagemaker/TrainingJobs, stream prefix = the training job name. The SDK
              prints `METRIC timesteps=N` then `METRIC episode_reward_mean=V` (and others) once per PPO
              rollout. There is no progress.csv or monitor file in the artifacts of that job (it crashed; its
              model.tar.gz is 1.7 KB), so the log is the only source. The job role reads it with
              logs:DescribeLogStreams + logs:GetLogEvents (AmazonSageMakerFullAccess); FilterLogEvents and
              Logs Insights are NOT allowed to it, so the log is paged with GetLogEvents.

Confidentiality: nearly every line of that log carries the twin's address. Only lines that match
`METRIC <name>=<number>` exactly are kept, the rest is dropped as it is read; the log is never written to
disk, to the output directory, to evidence or to stdout.

Evidence (a local evlib store, --evidence-dir {evidence_dir}; the runner uploads and attests it):
  one `item` record per curve point (passed when the value is finite), and one failing `<series>:incomplete`
  item per series that is missing or whose point count is not the declared one. The result is
  pass_rate >= 1.0 over the items, which check_unit recomputes; its `details` carry the curves for the
  dashboard:
    {"sft": {"run_id", "x": "step", "series": {"train_loss": [[step, v], ...], "dev_loss": [...]}, "source"},
     "rl":  {"train_run_id", "x": "timesteps", "series": {"mean_episode_reward": [[t, v], ...]},
             "crashed_at_timesteps", "served_checkpoint", "eval_run_id", "source"}}
  A series that cannot be read is null, with the reason in `series_reasons`; the other is still recorded.

Downsampling: see sources.json (at most 500 real points per series, uniform stride, first and last kept).
Nothing here is computed on a laptop except the tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

UNIT = "t1.curves"
GRADER_ID = "t1/curves/job_curves.py:v1"
METRIC_RE = re.compile(r"^METRIC ([A-Za-z_][A-Za-z0-9_]*)=(-?[0-9][0-9.eE+\-]*|nan|inf|-inf)$")
SOURCES_PATH = Path(__file__).resolve().parent / "sources.json"
DEFAULT_REGION = "us-east-2"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class JobError(RuntimeError):
    """A precondition failed; the job stops before writing anything misleading."""


def log(msg: str) -> None:
    print(f"[curves] {msg}", flush=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canon(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def num(v: float) -> int | float:
    """Whole numbers as ints (steps, timesteps), everything else as the float it is."""
    return int(v) if isinstance(v, float) and v.is_integer() and abs(v) < 2**53 else v


# --------------------------------------------------------------------------- downsampling

def downsample(points: list[tuple[Any, float]], max_points: int) -> list[tuple[Any, float]]:
    """At most max_points real points: uniform stride over the index, first and last kept."""
    n = len(points)
    if n <= max_points:
        return list(points)
    if max_points < 2:
        raise JobError("max_points must be at least 2")
    idx = [round(i * (n - 1) / (max_points - 1)) for i in range(max_points)]
    assert idx[0] == 0 and idx[-1] == n - 1 and len(set(idx)) == max_points
    return [points[i] for i in idx]


# --------------------------------------------------------------------------- SFT

def find_file(path: str | None, name: str) -> Path:
    """A channel directory (the file sits in it) or the file itself."""
    if not path:
        raise JobError(f"no path given for {name}")
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        hits = sorted(p.rglob(name))
        if hits:
            return hits[0]
    raise JobError(f"{name} not found at {path}")


def parse_sums(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            out[parts[1].lstrip("*")] = parts[0]
    return out


def read_sft(cfg: dict[str, Any], metrics_path: str | None, sums_path: str | None) -> dict[str, Any]:
    """Verified train_metrics.json -> {raw: {series: [(x, y)]}, extras: {...}} or {reason} per series.
    Chain: metrics file sha256 == its line in SHA256SUMS; SHA256SUMS sha256 == the model record's."""
    out: dict[str, Any] = {"raw": {}, "reasons": {}, "extra": {}, "input_sha256": None, "sums_sha256": None}
    names = list(cfg["series"])

    def fail_all(reason: str) -> dict[str, Any]:
        for n in names:
            out["reasons"][n] = reason
        return out

    try:
        sums_file = find_file(sums_path, "SHA256SUMS")
        sums_bytes = sums_file.read_bytes()
        out["sums_sha256"] = sha256_bytes(sums_bytes)
        if out["sums_sha256"] != cfg["sha256sums_sha256"]:
            return fail_all(f"SHA256SUMS sha256 {out['sums_sha256']} is not the model record's "
                            f"{cfg['sha256sums_sha256']}")
        listed = parse_sums(sums_bytes.decode("utf-8")).get(cfg["sums_member"])
        mfile = find_file(metrics_path, "train_metrics.json")
        mbytes = mfile.read_bytes()
        out["input_sha256"] = sha256_bytes(mbytes)
        if listed is None:
            return fail_all(f"SHA256SUMS does not list {cfg['sums_member']}")
        if listed != out["input_sha256"]:
            return fail_all(f"train_metrics.json sha256 {out['input_sha256']} differs from SHA256SUMS ({listed})")
        doc = json.loads(mbytes.decode("utf-8"))
    except (JobError, OSError, ValueError) as e:
        return fail_all(f"{e.__class__.__name__}: {e}")
    if doc.get("run_id") != cfg["run_id"]:
        return fail_all(f"train_metrics.json is for run {doc.get('run_id')!r}, not {cfg['run_id']!r}")
    for name, spec in cfg["series"].items():
        rows = doc.get(spec["field"])
        if not isinstance(rows, list) or not rows:
            out["reasons"][name] = f"{spec['field']} is missing or empty in train_metrics.json"
            continue
        pts, dropped, by_unit = [], 0, {}
        for r in rows:
            x, y = (r or {}).get(spec["x"]), (r or {}).get(spec["y"])
            if not (is_num(x) and is_num(y)):
                dropped += 1
                continue
            pts.append((num(x), y))
            if isinstance(r.get("by_unit"), dict):
                by_unit[str(num(x))] = {k: v for k, v in r["by_unit"].items() if is_num(v)}
        out["raw"][name] = pts
        if dropped:
            out["extra"].setdefault(name, {})["dropped_nonfinite"] = dropped
        if by_unit:
            out["extra"].setdefault(name, {})["by_unit"] = by_unit
    out["hyperparameters"] = {k: doc["hyperparameters"].get(k) for k in
                              ("epochs", "batch_size", "grad_accum", "lr", "lora_r", "lora_alpha")
                              if isinstance(doc.get("hyperparameters"), dict)}
    out["steps"] = doc.get("steps")
    return out


# --------------------------------------------------------------------------- RL (CloudWatch Logs)

def iter_log_messages(logs, group: str, stream_prefix: str):
    """Every message of every stream whose name starts with stream_prefix, oldest first, by paging
    GetLogEvents (allowed to the job role) until the forward token stops moving. Yields strings only; the
    caller must not keep or print them."""
    streams: list[str] = []
    token = None
    while True:
        kw: dict[str, Any] = {"logGroupName": group, "logStreamNamePrefix": stream_prefix}
        if token:
            kw["nextToken"] = token
        r = logs.describe_log_streams(**kw)
        streams += [s["logStreamName"] for s in r.get("logStreams", [])]
        token = r.get("nextToken")
        if not token:
            break
    if not streams:
        raise JobError(f"no log stream with prefix {stream_prefix!r} in {group}")
    for name in sorted(streams):
        tok = None
        while True:
            kw = {"logGroupName": group, "logStreamName": name, "startFromHead": True}
            if tok:
                kw["nextToken"] = tok
            r = logs.get_log_events(**kw)
            for ev in r.get("events", []):
                yield ev.get("message", "")
            nxt = r.get("nextForwardToken")
            if not nxt or nxt == tok:
                break
            tok = nxt


def read_metric_blocks(messages) -> tuple[list[dict[str, Any]], int]:
    """METRIC lines -> one block per `METRIC timesteps=` line {name: value}. Returns (blocks, lines scanned)."""
    blocks: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    scanned = 0
    for msg in messages:
        for line in str(msg).splitlines():
            scanned += 1
            m = METRIC_RE.match(line.strip())
            if not m:
                continue  # dropped on sight: everything else in this log may hold the twin address
            k, raw = m.groups()
            v = float(raw)
            if k == "timesteps":
                cur = {"timesteps": v}
                blocks.append(cur)
            elif cur is not None:
                cur[k] = v
    return blocks, scanned


def read_rl(cfg: dict[str, Any], logs) -> dict[str, Any]:
    out: dict[str, Any] = {"raw": {}, "reasons": {}, "scanned": 0, "blocks": 0}
    try:
        blocks, out["scanned"] = read_metric_blocks(iter_log_messages(logs, cfg["log_group"], cfg["train_job_name"]))
    except Exception as e:  # noqa: BLE001 - AccessDenied, throttling, no stream: the reason is the evidence
        code = ((getattr(e, "response", None) or {}).get("Error") or {}).get("Code") or e.__class__.__name__
        reason = f"CloudWatch Logs read failed ({code}): {str(e)[:200]}" if code != e.__class__.__name__ else \
            f"{e.__class__.__name__}: {str(e)[:200]}"
        for n in cfg["series"]:
            out["reasons"][n] = reason
        return out
    out["blocks"] = len(blocks)
    for name, spec in cfg["series"].items():
        pts, dropped = [], 0
        for b in blocks:
            y = b.get(spec["metric"])
            if y is None:
                continue
            if not math.isfinite(y):
                dropped += 1
                continue
            pts.append((num(b["timesteps"]), y))
        if not pts:
            out["reasons"][name] = f"no `METRIC {spec['metric']}=` lines in {cfg['log_group']} stream " \
                                   f"{cfg['train_job_name']}/* ({out['scanned']} lines scanned)"
            continue
        out["raw"][name] = pts
        if dropped:
            out.setdefault("dropped", {})[name] = dropped
    out["last_timesteps"] = num(blocks[-1]["timesteps"]) if blocks else None
    return out


def job_status(store, run_id: str) -> tuple[str | None, str | None]:
    """(status, job_name) from the training run's `job` record in evidence (written by tools/sm/run.py or the
    SDK launcher from DescribeTrainingJob), or (None, None)."""
    from tools import evlib

    try:
        recs = [r for r in evlib.read_records(run_id, "job", store) if r.get("job_kind") == "training"]
    except Exception:  # noqa: BLE001
        return None, None
    if not recs:
        return None, None
    r = recs[-1]
    return r.get("status"), r.get("job_name")


def eval_summary(store, eval_run_id: str) -> dict[str, Any] | None:
    from tools import evlib

    try:
        res = evlib.read_result(eval_run_id, store)
    except Exception:  # noqa: BLE001
        return None
    if not res:
        return None
    d = res.get("details") or {}
    out = {"run_id": eval_run_id, "metric": res.get("metric"), "value": res.get("value"), "passed": res.get("passed"),
           "checkpoint_steps": d.get("checkpoint_steps"), "items": d.get("items")}
    return {k: v for k, v in out.items() if v is not None}


# --------------------------------------------------------------------------- items and details

def point_item(series: str, xname: str, x, y: float, input_sha: str, backend: str,
               extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": f"{series}:{xname}={x}", "input_sha256": input_sha, "output_sha256": sha256_bytes(canon([x, y])),
        "grader": GRADER_ID, "backend": backend, "score": float(y), "passed": True,
        "extra": {"series": series, "x_name": xname, "x": x, "y": y, **extra},
    }


def incomplete_item(series: str, have: int, want: int, reason: str, backend: str) -> dict[str, Any]:
    return {
        "item_id": f"{series}:incomplete", "input_sha256": sha256_bytes(series.encode()),
        "output_sha256": sha256_bytes(canon({"have": have, "want": want})), "grader": GRADER_ID, "backend": backend,
        "score": None, "passed": False, "extra": {"series": series, "points": have, "declared_points": want,
                                                  "reason": reason[:500]},
    }


def build(a: argparse.Namespace, src: dict[str, Any], decl: dict[str, Any], logs, store) -> tuple[list[dict], dict]:
    maxp = int(src["downsample"]["max_points"])
    backend = f"{os.environ.get('SM_CURRENT_INSTANCE_TYPE', 'local')}; reads finished-job artifacts and logs only"
    items: list[dict[str, Any]] = []
    applied: dict[str, Any] = {}

    def emit(prefix: str, xname: str, raw: dict, reasons: dict, expect: dict, input_for, extra_for) -> dict:
        series: dict[str, Any] = {}
        for name, want in expect.items():
            full = f"{prefix}.{name}"
            pts = raw.get(name)
            if pts is None:
                series[name] = None
                items.append(incomplete_item(full, 0, want, reasons.get(name, "series not found"), backend))
                continue
            kept = downsample(pts, maxp)
            applied[full] = {"raw_points": len(pts), "kept": len(kept), "applied": len(kept) < len(pts)}
            series[name] = [[x, y] for x, y in kept]
            for x, y in kept:
                items.append(point_item(full, xname, x, y, input_for(name, x, y), backend, extra_for(name, x)))
            if len(pts) != want:
                reasons.setdefault(name, f"{len(pts)} points found, {want} declared")
                items.append(incomplete_item(full, len(pts), want, reasons[name], backend))
        return series

    # SFT
    s = src["sft"]
    sft = read_sft(s, a.sft_metrics, a.sft_sums)
    def sft_extra(name: str, x) -> dict[str, Any]:
        by_unit = sft["extra"].get(name, {}).get("by_unit", {}).get(str(x))
        return {"by_unit": by_unit} if by_unit else {}

    sft_series = emit("sft", "step", sft["raw"], sft["reasons"], s["expect_points"],
                      lambda n, x, y: sft["input_sha256"] or sha256_bytes(b""), sft_extra)
    # RL
    r = src["rl"]
    rl = read_rl(r, logs)
    status, jname = job_status(store, r["train_run_id"]) if store is not None else (None, None)
    rl_series = emit(
        "rl", "t", rl["raw"], rl["reasons"], r["expect_points"],
        lambda n, x, y: sha256_bytes(f"METRIC timesteps={x}\nMETRIC {r['series'][n]['metric']}={y!r}".encode()),
        lambda n, x: {})
    ev = eval_summary(store, r["eval_run_id"]) if store is not None else None
    steps = (ev or {}).get("checkpoint_steps")
    crashed = None
    if status == "Failed" and rl.get("last_timesteps") is not None:
        crashed = rl["last_timesteps"]
    sft_source = (f"{s['metrics_key']} in the team data bucket (train_loss_curve[].loss by step, dev_loss_curve[].loss "
                  f"by step); sha256 checked against {s['sums_key'].rsplit('/', 1)[-1]} (sha256 {s['sha256sums_sha256']}, "
                  f"the model record's artifact_sha256)")
    rl_source = (f"CloudWatch Logs {r['log_group']} stream prefix {r['train_job_name']}: `METRIC timesteps=` and "
                 f"`METRIC episode_reward_mean=` lines, one pair per PPO rollout of 1,024 steps; "
                 f"{rl['blocks']} blocks read")
    details = {
        "sft": {"run_id": s["run_id"], "x": "step", "series": sft_series, "source": sft_source,
                "series_reasons": sft["reasons"], "metrics_sha256": sft["input_sha256"],
                "sha256sums_sha256": sft["sums_sha256"], "steps": sft.get("steps"),
                "hyperparameters": sft.get("hyperparameters")},
        "rl": {"train_run_id": r["train_run_id"], "x": "timesteps", "series": rl_series,
               "crashed_at_timesteps": crashed,
               "crash_note": ("the training job ended Failed (twin connection reset); crashed_at_timesteps is the last "
                              "timesteps value it logged, the reset came during the next rollout") if crashed else None,
               "served_checkpoint": f"{steps // 1000}k" if isinstance(steps, int) else None,
               "eval_run_id": r["eval_run_id"], "source": rl_source, "series_reasons": rl["reasons"],
               "train_job_name": r["train_job_name"], "train_job_status": status, "eval": ev},
        "downsample": {"max_points": maxp, "rule": src["downsample"]["rule"], "by_series": applied},
        "sources_json_sha256": sha256_bytes(Path(a.sources).read_bytes()),
    }
    return items, details


# --------------------------------------------------------------------------- job

def stored_declaration(run_id: str, path: str | None) -> dict[str, Any]:
    from tools import evlib

    if path:
        d = json.loads(Path(path).read_text())
    else:
        s3 = os.environ.get("EVIDENCE_S3", "").strip()
        if not s3:
            raise JobError("no declaration: the runner sets EVIDENCE_S3 (a local run passes --declared FILE)")
        d = evlib.read_declared(run_id, evlib.S3Store(s3))
        if d is None:
            raise JobError(f"no declaration for {run_id} in {s3}: the launcher writes it before the job")
    if d.get("run_id") != run_id:
        raise JobError(f"the declaration is for run {d.get('run_id')!r}, not {run_id!r}")
    return d


def expected_records(src: dict[str, Any]) -> int:
    """Item records a complete run writes: every declared point, a series over max_points thinned to max_points."""
    cap = int(src["downsample"]["max_points"])
    return sum(min(n, cap) for part in ("sft", "rl") for n in src[part]["expect_points"].values())


def run(a: argparse.Namespace, logs=None, store=None) -> dict[str, Any]:
    from tools import evlib

    t0 = time.time()
    if not RUN_ID_RE.match(a.run_id or ""):
        raise JobError(f"run id {a.run_id!r} must match {RUN_ID_RE.pattern}")
    sources_bytes = Path(a.sources).read_bytes()
    src = json.loads(sources_bytes)
    decl = stored_declaration(a.run_id, a.declared)
    if decl.get("test_set_sha256") != sha256_bytes(sources_bytes):
        raise JobError(f"{a.sources} has sha256 {sha256_bytes(sources_bytes)}, the declaration's test_set_sha256 is "
                       f"{decl.get('test_set_sha256')}")
    want = expected_records(src)
    if decl.get("n_items") != want:
        raise JobError(f"the declaration says n_items {decl.get('n_items')}, sources.json expects {want} records")
    if logs is None:
        import boto3

        logs = boto3.client("logs", region_name=a.region)
    if store is None:
        if a.evidence_read:
            store = evlib.LocalStore(a.evidence_read)
        elif os.environ.get("EVIDENCE_S3", "").strip():
            store = evlib.S3Store(os.environ["EVIDENCE_S3"], region=a.region)
    items, details = build(a, src, decl, logs, store)
    bad = [i["item_id"] for i in items if not i["passed"]]
    log(f"{len(items)} item records, {len(bad)} incomplete series: {bad}")
    lstore = evlib.LocalStore(a.evidence_dir)
    evlib.append_records(a.run_id, "item", items, store=lstore)
    res = evlib.make_result(decl, items, "t1/curves/job_curves.py", decl.get("code_sha"), details=details)
    evlib.write_result(res, store=lstore)
    log(f"result {res['metric']} = {res['value']} over {res['n_items']} items -> "
        f"{'PASSED' if res['passed'] else 'MISSED'} ({time.time() - t0:.1f} s)")
    if a.summary_dir:
        Path(a.summary_dir).mkdir(parents=True, exist_ok=True)
        (Path(a.summary_dir) / "curves_summary.json").write_text(
            json.dumps({"kind": "t1-curves-summary", "run_id": a.run_id, "result": {
                k: res[k] for k in ("metric", "value", "passed", "n_items")},
                "series_points": {k: v.get("kept") for k, v in details["downsample"]["by_series"].items()}},
                       sort_keys=True, indent=2) + "\n")
    return res


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m t1.curves.job_curves", description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--sources", default=str(SOURCES_PATH), help="sources.json (the declared test set)")
    ap.add_argument("--sft-metrics", help="train_metrics.json, or its channel directory ({channel:sft_metrics})")
    ap.add_argument("--sft-sums", help="SHA256SUMS, or its channel directory ({channel:sft_sums})")
    ap.add_argument("--evidence-dir", required=True, help="local evlib store for items + result ({evidence_dir})")
    ap.add_argument("--evidence-read", help="read job/result records from this evlib dir (tests); default EVIDENCE_S3")
    ap.add_argument("--declared", help="the declaration as a JSON file (default: read from EVIDENCE_S3)")
    ap.add_argument("--summary-dir", help="also write curves_summary.json here ({output_dir})")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION") or DEFAULT_REGION)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None, logs=None, store=None) -> int:
    a = parse_args(argv)
    try:
        run(a, logs=logs, store=store)
    except JobError as e:
        log(f"FAILED: {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
