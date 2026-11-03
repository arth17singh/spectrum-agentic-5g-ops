"""Real-run views over the mirrored evidence store (make evidence-sync). Read-only, no AWS calls.

Inputs, all under the evidence root:
  declared/, results/, records/, outputs/<run>/job-map.json, models/      (S3 objects, byte-exact copies;
                                                                           file mtime = S3 LastModified)
  .sync/manifest.json      per object: sha256, size, S3 LastModified (written by tools/evidence_sync.py)
  .sync/jobs/<job>.json    DescribeTrainingJob / DescribeProcessingJob subset, read when the sync ran
  .sync/attest/<run>.json  each file's sha256 vs the `team13-evidence` line SageMaker logged for the job
  .sync/checks/<run>.json  cached tools/check_unit.py --json output (parts a-e)

Nothing here invents a number: a value that no file carries is None and the UI says so.
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

T1_UNITS = ("t1.price_latency", "t1.execution", "t1.safety", "t1.interface", "t1.knowledge")
NOT_HEADLINE = re.compile(r"(replay-check|refreeze|selftest|link-test|suite-freeze)")
OPS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, "<": lambda a, b: a < b,
       "==": lambda a, b: a == b or math.isclose(a, b, rel_tol=1e-9)}


def num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def _iso_mtime(p: Path) -> str | None:
    try:
        return datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    except OSError:
        return None


def _t(s: Any) -> datetime | None:
    if not isinstance(s, str) or not s:
        return None
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _load(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class Store:
    """A local evidence root (the mirror). Every read is fresh."""

    def __init__(self, root: Path):
        self.root = root
        self.sync = root / ".sync"

    # ---------------------------------------------------------------- raw
    def run_ids(self) -> list[str]:
        ids = set()
        for d in ("declared", "results"):
            for p in (self.root / d).glob("*.json"):
                ids.add(p.name[:-5])
        rec = self.root / "records"
        if rec.is_dir():
            ids.update(p.name for p in rec.iterdir() if p.is_dir() and not p.name.startswith("."))
        return sorted(ids)

    def declared(self, run: str) -> dict | None:
        return _load(self.root / "declared" / f"{run}.json")

    def result(self, run: str) -> dict | None:
        return _load(self.root / "results" / f"{run}.json")

    def job_map(self, run: str) -> dict | None:
        return _load(self.root / "outputs" / run / "job-map.json")

    def job_records(self, run: str) -> list[dict]:
        d = self.root / "records" / run / "job"
        out = [_load(p) for p in sorted(d.glob("*.json"))] if d.is_dir() else []
        j = self.root / "records" / run / "job.jsonl"
        if j.is_file():
            out += [json.loads(line) for line in j.read_text().splitlines() if line.strip()]
        return [x for x in out if isinstance(x, dict)]

    def described(self, job: str) -> dict | None:
        return _load(self.sync / "jobs" / f"{job}.json")

    def attest(self, run: str) -> dict | None:
        return _load(self.sync / "attest" / f"{run}.json")

    def check(self, run: str) -> dict | None:
        return _load(self.sync / "checks" / f"{run}.json")

    def manifest(self) -> dict:
        m = _load(self.sync / "manifest.json")
        return m if isinstance(m, dict) else {}

    def models(self) -> list[tuple[str, dict]]:
        out = []
        for p in sorted((self.root / "models").glob("*/*.json")):
            d = _load(p)
            if isinstance(d, dict):
                out.append((p.relative_to(self.root).as_posix(), d))
        return out

    def s3_time(self, rel: str) -> str | None:
        return _iso_mtime(self.root / rel)

    def source(self) -> dict:
        mf = self.sync / "manifest.json"
        return {"kind": "mirror", "label": "local byte-exact mirror of the S3 evidence prefix (make evidence-sync)",
                "synced_at": _iso_mtime(mf) if mf.exists() else None, "time_basis": "S3 LastModified (file mtime set by the sync)",
                "has_mirror": mf.exists()}


# --------------------------------------------------------------------- per run

def jobs_for(store: Store, run: str) -> list[dict]:
    """Every SageMaker job the run names, merged from the job-map, the job records and DescribeTrainingJob."""
    jm = store.job_map(run) or {}
    recs = store.job_records(run)
    names: list[str] = []
    for src in [jm, *recs]:
        n = src.get("job_name") or src.get("TrainingJobName")
        if isinstance(n, str) and n not in names:
            names.append(n)
    out = []
    for n in names:
        rec = next((r for r in recs if (r.get("job_name") or r.get("TrainingJobName")) == n), {})
        desc = store.described(n) or {}
        price = num(jm.get("price_usd_per_hour")) if jm.get("job_name") == n else None
        billable = num(desc.get("billable_s")) or num(rec.get("BillableTimeInSeconds")) or num(rec.get("billable_s"))
        cost, cost_basis = None, None
        if billable is not None and price is not None:
            cost, cost_basis = billable * price / 3600.0, f"billable s × ${price}/h (job-map price, {jm.get('price_source') or 'source not recorded'})"
        elif num(rec.get("est_cost_usd")) is not None:
            cost, cost_basis = num(rec.get("est_cost_usd")), "est_cost_usd on the job record (billable s × on-demand price)"
        out.append({
            "name": n,
            "kind": desc.get("kind") or rec.get("job_kind") or jm.get("job_kind") or "training",
            "status": desc.get("status") or rec.get("TrainingJobStatus") or rec.get("status"),
            "failure_reason": desc.get("failure_reason") or rec.get("FailureReason"),
            "instance_type": desc.get("instance_type") or rec.get("instance_type") or jm.get("instance_type")
            or ((rec.get("ResourceConfig") or {}).get("InstanceType")),
            "creation_time": desc.get("creation_time") or rec.get("CreationTime") or jm.get("creation_time") or rec.get("aws_start"),
            "start_time": desc.get("start_time") or rec.get("TrainingStartTime") or rec.get("aws_start"),
            "end_time": desc.get("end_time") or rec.get("TrainingEndTime") or rec.get("aws_end"),
            "billable_s": billable,
            "price_usd_per_hour": price,
            "cost_usd": cost,
            "cost_basis": cost_basis,
            "described_at": desc.get("described_at"),
            "time_source": "DescribeTrainingJob (cached by the sync)" if desc else "job record written from DescribeTrainingJob",
        })
    return out


def recompute(result: dict | None, declared: dict | None) -> str | None:
    if not isinstance(result, dict) or not isinstance(declared, dict):
        return None
    t = declared.get("target") or {}
    v, tv, op = num(result.get("value")), num(t.get("value")), t.get("op")
    if v is None or tv is None or op not in OPS:
        return None
    return "pass" if OPS[op](v, tv) else "fail"


def check_parts(store: Store, run: str) -> dict | None:
    c = store.check(run)
    if not isinstance(c, dict):
        return None
    s = c.get("summary") if isinstance(c.get("summary"), dict) else None
    parts = []
    raw = (s or {}).get("parts") or {}
    for k, p in (raw.items() if isinstance(raw, dict) else []):
        if isinstance(p, dict):
            lines = p.get("lines") if isinstance(p.get("lines"), list) else []
            parts.append({"id": k, "state": p.get("state"), "title": p.get("title"), "lines": [str(x) for x in lines[:14]],
                          "more_lines": max(0, len(lines) - 14)})
    return {"exit_code": c.get("exit_code"), "checked_at": c.get("checked_at"), "elapsed_s": c.get("elapsed_s"),
            "verdict": {0: "all parts passed", 1: "a part failed", 2: "nothing failed, a part could not be checked"}.get(c.get("exit_code"), "unknown"),
            "parts": parts, "window": (s or {}).get("window"), "first_aws_time": (s or {}).get("first_aws_time"),
            "window_sources": (s or {}).get("window_sources"), "text": None if s else (c.get("stdout") or "")[-4000:]}


def run_summary(store: Store, run: str) -> dict:
    d = store.declared(run) or {}
    r = store.result(run)
    jobs = jobs_for(store, run)
    first_job = min((j["creation_time"] for j in jobs if j.get("creation_time")), default=None)
    decl_s3 = store.s3_time(f"declared/{run}.json") if d else None
    a = store.attest(run) or {}
    chk = store.check(run)
    before = None
    if decl_s3 and first_job:
        before = _t(decl_s3) < _t(first_job)
    return {
        "run_id": run,
        "unit": d.get("unit") or (r or {}).get("unit"),
        "track": d.get("track"),
        "model_id": d.get("model_id"),
        "synthetic": bool(d.get("synthetic")) or bool((r or {}).get("synthetic")),
        "headline": not NOT_HEADLINE.search(run),
        "declared": bool(d),
        "declared_at": d.get("declared_at"),
        "declared_s3_time": decl_s3,
        "declared_before_job": before,
        "target": d.get("target"),
        "n_items_declared": d.get("n_items"),
        "result": None if r is None else {
            "metric": r.get("metric"), "value": num(r.get("value")), "n_items": r.get("n_items"),
            "checked_at": r.get("checked_at"), "stored_passed": r.get("passed"),
            "recomputed": recompute(r, d) if d else None, "s3_time": store.s3_time(f"results/{run}.json")},
        "jobs": jobs,
        "cost_usd": sum(j["cost_usd"] for j in jobs if j.get("cost_usd") is not None) if any(j.get("cost_usd") is not None for j in jobs) else None,
        "attest": a.get("counts"),
        "check": None if not isinstance(chk, dict) else {
            "exit_code": chk.get("exit_code"), "checked_at": chk.get("checked_at"),
            "parts": {k: v.get("state") for k, v in (((chk.get("summary") or {}).get("parts")) or {}).items() if isinstance(v, dict)}},
    }


# --------------------------------------------------------------------- Track 1

def variant(declared: dict) -> tuple[str, str]:
    """(variant, label): base | tuned | bedrock:<model> | sagemaker:<model>, from what the declaration records."""
    runner = (((declared.get("twin") or {}).get("config") or {}).get("runner") or {})
    entry = runner.get("entry") or declared.get("entry") or []
    entry = [str(x) for x in entry] if isinstance(entry, list) else []
    client = entry[entry.index("--client") + 1] if "--client" in entry and entry.index("--client") + 1 < len(entry) else None
    mid = str(declared.get("model_id") or "")
    blob = " ".join([declared.get("run_id", ""), mid]).lower()
    if client == "bedrock" or "bedrock" in blob:
        return f"bedrock:{mid}", f"Bedrock {mid}"
    if "--adapter-dir" in entry or declared.get("baseline_run_id") or re.search(r"(tuned|lora|sft|-ft\b)", blob):
        return "tuned", f"Tuned ({mid})"
    if client == "sagemaker":
        return f"sagemaker:{mid}", f"SageMaker {mid}"
    return "base", f"Base ({mid})"


def _ci(result: dict) -> dict | None:
    det = result.get("details") or {}
    g = det.get("gain") if isinstance(det.get("gain"), dict) else None
    if g and num(g.get("ci_low")) is not None:
        return {k: g.get(k) for k in ("value", "ci_low", "ci_high", "level", "method", "baseline_run_id", "paired_n")}
    for k in ("ci", "ci95", "interval"):
        v = det.get(k)
        if isinstance(v, dict) and num(v.get("low", v.get("ci_low"))) is not None:
            return {"ci_low": v.get("low", v.get("ci_low")), "ci_high": v.get("high", v.get("ci_high")), "level": v.get("level"),
                    "method": v.get("method")}
    return None


def t1_cell(store: Store, run: str, declared: dict, result: dict | None) -> dict:
    det = (result or {}).get("details") or {}
    lp = det.get("latency_and_price") or {}
    return {
        "run_id": run,
        "model_id": declared.get("model_id"),
        "target": declared.get("target"),
        "state": (recompute(result, declared) or "inconsistent") if result else "not checked",
        "value": num((result or {}).get("value")),
        "metric": (result or {}).get("metric"),
        "pass_rate": num(det.get("pass_rate")),
        "mean_score": num(det.get("mean_score")),
        "n_items": (result or {}).get("n_items"),
        "p50_ms": num(lp.get("p50_ms")), "p95_ms": num(lp.get("p95_ms")),
        "cost_per_1k_usd": num(lp.get("cost_per_1k_usd")),
        "instance_type": lp.get("instance_type"),
        "price_source": (lp.get("pricing") or {}).get("source"),
        "cost_method": lp.get("cost_method"),
        "parse_fail_rate": num(det.get("parse_fail_rate")),
        "secondary": det.get("secondary_checks") or [],
        "ci": _ci(result) if result else None,
        "checked_at": (result or {}).get("checked_at"),
        "declared_at": declared.get("declared_at"),
        "declared_s3_time": store.s3_time(f"declared/{run}.json"),
        "result_s3_time": store.s3_time(f"results/{run}.json") if result else None,
        "jobs": jobs_for(store, run),
    }


def track1(store: Store, runs: list[str], sup: dict | None = None) -> dict:
    sup = sup or {}
    by: dict[tuple[str, str], list[tuple[str, dict, dict | None]]] = {}
    labels: dict[str, str] = {}
    for run in runs:
        d = store.declared(run)
        if not isinstance(d, dict) or d.get("unit") not in T1_UNITS or NOT_HEADLINE.search(run):
            continue
        v, label = variant(d)
        labels.setdefault(v, label)
        by.setdefault((d["unit"], v), []).append((run, d, store.result(run)))
    order = ["base", "tuned"] + sorted(k for k in labels if k.startswith("bedrock:")) + sorted(
        k for k in labels if k not in ("base", "tuned") and not k.startswith("bedrock:"))
    cols = [{"variant": v, "label": labels.get(v, {"base": "Base model", "tuned": "Fine-tuned"}.get(v, v)), "seen": v in labels}
            for v in order if v in labels or v in ("base", "tuned")]
    if not any(c["variant"].startswith("bedrock:") for c in cols):
        cols.append({"variant": "bedrock", "label": "Bedrock (general model)", "seen": False})
    units = []
    for u in T1_UNITS:
        cells = {}
        for c in cols:
            cand = by.get((u, c["variant"]), [])
            live = [x for x in cand if x[0] not in sup] or cand
            with_result = [x for x in live if isinstance(x[2], dict)]
            pick = max(with_result or live, key=lambda x: (x[2] or {}).get("checked_at") or x[1].get("declared_at") or "", default=None)
            if pick is None:
                cells[c["variant"]] = None
            else:
                cell = t1_cell(store, *pick)
                cell["other_runs"] = [x[0] for x in cand if x[0] != pick[0] and x[0] not in sup]
                cell["superseded_runs"] = [{"run_id": x[0], **sup[x[0]], "value": num((x[2] or {}).get("value"))}
                                           for x in cand if x[0] in sup and x[0] != pick[0]]
                cell["superseded"] = sup.get(pick[0])
                cells[c["variant"]] = cell
        units.append({"unit": u, "cells": cells})
    # the bar: interface vs execution, per model column
    bar = []
    for c in cols:
        row = {"variant": c["variant"], "label": c["label"]}
        for u in ("t1.interface", "t1.execution"):
            cell = next(x["cells"].get(c["variant"]) for x in units if x["unit"] == u)
            row[u] = None if not cell or cell["state"] == "not checked" else {
                "run_id": cell["run_id"], "pass_rate": cell["pass_rate"], "value": cell["value"], "metric": cell["metric"],
                "mean_score": cell["mean_score"], "n_items": cell["n_items"]}
        bar.append(row)
    return {"columns": cols, "units": units, "bar": bar}


# --------------------------------------------------------------------- RL

def rl(store: Store, runs: list[str]) -> dict | None:
    evals = [r for r in runs if r.startswith("t1.rl.eval-") and store.result(r)]
    if not evals:
        return None
    run = max(evals, key=lambda r: store.result(r).get("checked_at") or "")
    res, decl = store.result(run), store.declared(run) or {}
    det = res.get("details") or {}
    mo, bo, th = det.get("model_overall") or {}, det.get("baseline_overall") or {}, det.get("gate_thresholds") or {}
    train_run = det.get("training_run_id")
    train_decl = store.declared(train_run) if train_run else None
    items_text = ((((train_decl or {}).get("twin") or {}).get("config") or {}).get("items") or {})
    up = num(th.get("objective_uplift"))
    sla_tol = num(th.get("sla_tolerance"))

    def measured(k):
        m_obj, b_obj = num(mo.get("objective_mean")), num(bo.get("objective_mean"))
        if k == "UPLIFT" and m_obj is not None and b_obj is not None and up is not None:
            return {"model": m_obj, "baseline": b_obj, "needed": b_obj * up,
                    "text": f"model objective {m_obj:.2f} vs needed {b_obj:.2f} × {up} = {b_obj * up:.2f}"}
        if k == "COVERAGE" and num(mo.get("coverage_mean")) is not None:
            return {"model": mo["coverage_mean"], "baseline": bo.get("coverage_mean"),
                    "text": f"coverage {mo['coverage_mean']:.3f} vs baseline {bo.get('coverage_mean', float('nan')):.3f}"}
        if k == "SERVICE" and num(mo.get("sla_ratio_mean")) is not None:
            needed = (bo.get("sla_ratio_mean") or 0) - (sla_tol or 0)
            return {"model": mo["sla_ratio_mean"], "baseline": bo.get("sla_ratio_mean"), "needed": needed,
                    "text": f"SLA {mo['sla_ratio_mean']:.3f} vs baseline {bo.get('sla_ratio_mean', float('nan')):.3f} − {sla_tol} = {needed:.3f}"}
        if k == "SAFETY":
            return {"model": mo.get("safety_violations"), "baseline": bo.get("safety_violations"),
                    "text": f"{mo.get('safety_violations')} hard safety violations (baseline {bo.get('safety_violations')})"}
        return None

    items = [{"id": k, "passed": v, "rule": items_text.get(k), "measured": measured(k)}
             for k, v in (det.get("items") or {}).items()]
    # lineage: training run -> its job -> the checkpoint the eval used
    train_jobs = jobs_for(store, train_run) if train_run else []
    model_entry = next(({"path": p, **{k: m.get(k) for k in ("model_id", "version", "artifact_sha256", "status", "aws_time", "aws_time_source")},
                         "steps": ((m.get("metrics") or {}).get("checkpoint") or {}).get("steps"),
                         "declared_timesteps": (m.get("metrics") or {}).get("declared_timesteps"),
                         "last_logged": (m.get("metrics") or {}).get("last_metric_timesteps")}
                        for p, m in store.models() if m.get("run_id") == train_run or train_run and train_run.replace(".", "-") in p), None)
    if model_entry and isinstance(model_entry.get("last_logged"), dict):
        ll = model_entry["last_logged"]
        model_entry["last_logged"] = {k: ll.get(k) for k in ("metric", "value", "timestamp", "log_group", "regex_source")}
    reward = [r for r in runs if r.startswith("t1.rl.reward-check-")]
    return {
        "run_id": run,
        "result": {"value": num(res.get("value")), "metric": res.get("metric"), "n_items": res.get("n_items"),
                   "target": decl.get("target"), "recomputed": recompute(res, decl), "checked_at": res.get("checked_at")},
        "items": items, "model_overall": mo, "baseline_overall": bo, "thresholds": th,
        "checkpoint_steps": det.get("checkpoint_steps"), "checkpoint_sha256": det.get("checkpoint_sha256"),
        "release": det.get("release"), "model_form": det.get("model_form"),
        "eval_jobs": jobs_for(store, run),
        "training": None if not train_run else {"run_id": train_run, "model_id": (train_decl or {}).get("model_id"),
                                                "declared_at": (train_decl or {}).get("declared_at"), "jobs": train_jobs,
                                                "model": model_entry},
        "reward_check": [run_summary(store, r) for r in reward],
    }


# --------------------------------------------------------------------- Track 3 / Track 2

def track3(store: Store, runs: list[str], sup: dict | None = None) -> list[dict]:
    sup = sup or {}
    out = []
    for run in runs:
        if not run.startswith("t3.evals.") or NOT_HEADLINE.search(run):
            continue
        d, r = store.declared(run) or {}, store.result(run)
        if not r:
            continue
        det = r.get("details") or {}
        out.append({"run_id": run, "model_id": det.get("model_id") or d.get("model_id"), "mode": det.get("mode"),
                    "subset": det.get("subset"), "n_items": r.get("n_items"), "metric": r.get("metric"),
                    "value": num(r.get("value")), "target": d.get("target"), "recomputed": recompute(r, d),
                    "mean_recovery": num(det.get("mean_recovery")), "success_rate": num(det.get("success_rate")),
                    "ci": _ci(r), "checked_at": r.get("checked_at"), "jobs": jobs_for(store, run)})
    # newest per model id is current; older runs of the same model are kept and marked
    latest: dict[str, str] = {}
    for x in sorted(out, key=lambda x: x["checked_at"] or ""):
        if x["run_id"] not in sup:
            latest[x["model_id"]] = x["run_id"]
    for x in out:
        x["current"] = latest.get(x["model_id"]) == x["run_id"]
        x["superseded"] = sup.get(x["run_id"])
    return sorted(out, key=lambda x: (not x["current"], -(x["mean_recovery"] or -9), x["run_id"]))


def track2(store: Store, runs: list[str]) -> list[dict]:
    out = []
    for run in runs:
        if not run.startswith("t2."):
            continue
        d, r = store.declared(run) or {}, store.result(run)
        det = (r or {}).get("details") or {}
        out.append({"run_id": run, "unit": d.get("unit"), "method": det.get("method"), "silo": det.get("silo"),
                    "mean_score": num(det.get("mean_score")), "mean_score_noop": num(det.get("mean_score_noop")),
                    "mae": num(det.get("mae")), "direction_accuracy": num(det.get("direction_accuracy")),
                    "n_items": (r or {}).get("n_items"), "state": recompute(r, d) if r else "not checked",
                    "target": d.get("target"), "checked_at": (r or {}).get("checked_at"), "jobs": jobs_for(store, run)})
    return out


# --------------------------------------------------------------------- explorer

def run_detail(store: Store, run: str, max_files: int = 60) -> dict:
    s = run_summary(store, run)
    man = store.manifest()
    att = (store.attest(run) or {}).get("files") or {}
    files = []
    for rel in sorted(k for k in man if k == f"declared/{run}.json" or k == f"results/{run}.json"
                      or k.startswith((f"records/{run}/", f"outputs/{run}/"))):
        m = man[rel]
        files.append({"path": rel, "sha256": m.get("sha256"), "size": m.get("size"), "s3_time": m.get("last_modified"),
                      "attest": (att.get(rel) or {}).get("state") if (rel.startswith("records/") or rel.startswith("results/")) else None,
                      "gone_from_s3": bool(m.get("gone_from_s3"))})
    kinds: dict[str, int] = {}
    for f in files:
        p = f["path"].split("/")
        if p[0] == "records" and len(p) >= 3:
            k = p[2][:-6] if p[2].endswith(".jsonl") else p[2]
            kinds[k] = kinds.get(k, 0) + 1
    samples = {}
    for k in kinds:
        d = store.root / "records" / run / k
        if d.is_dir():
            samples[k] = [_load(p) for p in sorted(d.glob("*.json"))[:2]]
    att_counts: dict[str, int] = {}
    for f in files:
        if f["attest"]:
            att_counts[f["attest"]] = att_counts.get(f["attest"], 0) + 1
    return {**s, "declaration": store.declared(run), "result_doc": store.result(run), "job_map": store.job_map(run),
            "files_total": len(files), "files": files[:max_files] + [f for f in files[max_files:] if f["attest"] == "MISMATCH"],
            "record_kinds": kinds, "record_samples": samples, "attest_counts": att_counts,
            "attest_source": (store.attest(run) or {}).get("source"), "attest_checked_at": (store.attest(run) or {}).get("checked_at"),
            "check_unit": check_parts(store, run)}


def root_from_env() -> Path | None:
    raw = os.environ.get("SPECTRUM_RECORDS_ROOT") or os.environ.get("SPECTRUM_EVIDENCE_DIR")
    if raw and raw.startswith("s3://"):
        return None
    if raw:
        return Path(raw[len("file://"):] if raw.startswith("file://") else raw)
    from spectrum.slices._shared.records import DEFAULT_ROOT

    return DEFAULT_ROOT
