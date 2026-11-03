"""AWS-recorded re-measurements of the trust-gate headline numbers (CPU SageMaker job entry points).

Two entry modules call into this file:

    python -m t3.trust.job_gate_eval --version v1|v2 --model healthy|degraded   [--dry-run]
    python -m t3.trust.job_ladder    --model healthy|degraded                   [--dry-run]

Environment, as the job runner sets it: ``EVIDENCE_S3`` (or ``EVIDENCE_DIR`` for local runs) and
``RUN_ID`` (``EVIDENCE_RUN_ID`` is accepted too). Each invocation is ONE evidence run, so every
headline number is one result that ``tools/check_unit.py`` can recompute from its item records:

  gate eval (4 runs)  items = one per twin seed 2000-2019 (version, model fixed by the run)
      healthy:  passed = never paused in 24 changes;    metric fail_count  (target <= 2, v2's declared bar)
      degraded: passed = paused by change 6;             metric pass_count  (target >= 19, v2's declared bar)
  ladder (2 runs)     items = one per seed 3000-3019 (5 steps); score = the R1 (MAVERIC) case gain
      healthy:  metric mean_score, target > 0 (and, declared alongside, bootstrap 95 % CI lower bound > 0)
      degraded: metric mean_score, target < 0
      the R2 (REHEARSAL) case of the same seed is in the item's extra; R2 and the settings shift are in the
      result details: measurements, not targets.

Every run also writes the underlying ``change`` records (and ``control`` records for pauses; the
policy client is a dry run, so those are ``synthetic: true``). Records are validated, then written at
the end: on S3 one object per record (evlib "objects" layout, so each record gets an AWS time from its
own LastModified); locally one JSONL file per kind. Readers merge both layouts (A11).

The declaration is written BEFORE the run, by the job runner (or by ``--declare-first`` here, which
waits 2 s so its S3 LastModified is strictly earlier). ``--print-declaration`` prints it;
``declarations/*.template.json`` are the same documents with placeholders.

These re-measure, on AWS, what was already measured on a laptop:
  gate eval: the one-shot evaluation declared in ed7d3c4 and run once on the laptop in dd15c54
             (t3/trust/runs/v2_eval.json) on the SAME seeds 2000-2019; nothing is re-tuned.
  ladder:    the laptop ladder run of f4a75a2 (t3/trust/runs/ladder_report.json), seeds 3000-3019.

``--dry-run`` writes nothing: it runs the first two seeds in memory, validates every record with
evlib, and prints the declaration and a summary.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .retention import bootstrap_mean_ci, bootstrap_retention, wilson

HERE = Path(__file__).parent
TESTSETS = HERE / "testsets"
DECLARATIONS = HERE / "declarations"
GATE_TESTSET = TESTSETS / "gate_eval_2000_2019.json"
LADDER_TESTSET = TESTSETS / "ladder_3000_3019.json"
GRADER_GATE = "t3.trust gate rule (A6) replayed on twin.replay; per-seed pause index"
GRADER_LADDER = "t3.trust ladder: objective gain vs exact no-change counterfactual (twin.replay)"
BACKEND = "twin.replay.ReplayTwin + t2.predict (numpy) on CPU"
REMEASURE_NOTE_GATE = ("Re-measurement on AWS of the one-shot evaluation declared in commit ed7d3c4 (thresholds_v2.json "
                       "sha256 395ed282...). These seeds (2000-2019) were used once before, on the laptop, in dd15c54 "
                       "(t3/trust/runs/v2_eval.json); nothing was re-tuned since. The target is v2's declared bar.")
REMEASURE_NOTE_LADDER = ("Re-measurement on AWS of the laptop ladder run in f4a75a2 (t3/trust/runs/ladder_report.json), "
                         "same seeds 3000-3019 and script; R2 and the settings shift are measurements, not targets.")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_json(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def env_run_id() -> str | None:
    return os.environ.get("RUN_ID") or os.environ.get("EVIDENCE_RUN_ID")


def code_sha_or_unknown() -> str:
    """The commit the job runs: $CODE_SHA or $TEAM13_CODE_SHA (set by tools/sm/run.py; containers often lack
    the repo history), else the repo HEAD, else "unknown" (which evlib/contracts reject, so a real run stops)."""
    env = (os.environ.get("CODE_SHA") or os.environ.get("TEAM13_CODE_SHA") or "").strip()
    if env:
        return env
    from .demo import code_sha

    return code_sha() or "unknown"


def gate_command(version: str, model: str) -> str:
    return f"python -m t3.trust.job_gate_eval --version {version} --model {model}"


def ladder_command(model: str) -> str:
    return f"python -m t3.trust.job_ladder --model {model}"


# ---------------------------------------------------------------- declarations
def gate_target(model: str) -> dict:
    if model == "healthy":
        return {"metric": "fail_count", "op": "<=", "value": 2, "unit": "runs (healthy model paused)"}
    return {"metric": "pass_count", "op": ">=", "value": 19, "unit": "runs (degraded model paused by change 6)"}


def ladder_target(model: str) -> dict:
    """healthy: the paired-bootstrap-v1 95 % lower bound of the mean R1 gain > 0 (check_unit replays it from the
    items with details.gain); degraded: the mean R1 gain < 0."""
    if model == "healthy":
        return {"metric": "gain_ci_low", "op": ">", "value": 0, "unit": "objective gain on MAVERIC (R1), 95% CI low"}
    return {"metric": "mean_score", "op": "<", "value": 0, "unit": "objective gain on MAVERIC (R1)"}


def gate_declaration(*, run_id: str, version: str, model: str, code_sha: str, region: str, testset: dict,
                     testset_sha: str, twin_version: str | None) -> dict:
    """Contract-shaped (no top-level extras): the run context lives in twin.config."""
    return {
        "run_id": run_id, "unit": f"t3.trust_gate.{version}.{model}", "track": "track-3",
        "target": gate_target(model), "n_items": len(testset["seeds"]), "test_set_sha256": testset_sha,
        "model_id": testset["models"][model]["model_id"], "grader": GRADER_GATE, "code_sha": code_sha,
        "region": region, "synthetic": False,
        "twin": {"kind": "maveric-inprocess", "version": twin_version or "ran-rl (version not resolved)",
                 "scenario": testset["scenario"], "seed": testset["seeds"][0],
                 "config": {"preset": testset["config"], "seeds": [testset["seeds"][0], testset["seeds"][-1]],
                            "thresholds_version": version, "thresholds_sha256": testset["thresholds"][version],
                            "command": gate_command(version, model),
                            "references": testset.get("references", {}),
                            "note": testset.get("provenance", REMEASURE_NOTE_GATE)}},
    }


def ladder_declaration(*, run_id: str, model: str, code_sha: str, region: str, testset: dict, testset_sha: str,
                       twin_version: str | None) -> dict:
    """Contract-shaped (no top-level extras): the run context lives in twin.config."""
    also = ("the target is the paired-bootstrap-v1 95 % CI lower bound of the mean R1 gain (spec in details.gain, "
            "replayed by check_unit from the items); the mean itself is details.gain.value") if model == "healthy" else None
    r1, r2 = testset["rungs"]["R1"], testset["rungs"]["R2"]
    return {
        "run_id": run_id, "unit": "t3.ladder", "track": "track-3", "target": ladder_target(model),
        "n_items": len(testset["seeds"]), "test_set_sha256": testset_sha,
        "model_id": testset["models"][model]["model_id"], "grader": GRADER_LADDER, "code_sha": code_sha,
        "region": region, "synthetic": False,
        "twin": {"kind": "maveric-inprocess", "version": twin_version or "ran-rl (version not resolved)",
                 "scenario": r1["scenario"], "seed": testset["seeds"][0],
                 "config": {"preset": r1["config"], "seeds": [testset["seeds"][0], testset["seeds"][-1]],
                            "rungs": {"R1": r1, "R2": r2}, "steps": testset["steps"], "declared_also": also,
                            "measured_not_targeted": ["R2 (REHEARSAL) gain and CI",
                                                      "settings shift REHEARSAL / MAVERIC"],
                            "command": ladder_command(model),
                            "references": testset.get("references", {}),
                            "note": testset.get("provenance", REMEASURE_NOTE_LADDER)}},
    }


GATE_TESTSET_FRESH = TESTSETS / "gate_eval_3000_3019.json"
LADDER_TESTSET_FRESH = TESTSETS / "ladder_4000_4019.json"
FRESH_NOTE_GATE = ("Fresh for the trust gate: twin seeds 3000-3019 have never been through the gate (calibration used "
                   "20000-20039, the demos 0 and 1000-1009, the v2 evaluation 2000-2019). They were used by the laptop "
                   "LADDER runs (f4a75a2), which choose changes with the predictors but never run the gate rule; no "
                   "threshold was chosen on them. Thresholds v1 as declared (sha256 d051c1e0...), untouched since 2c8a57c.")
FRESH_NOTE_LADDER = ("Fresh ladder seeds 4000-4019: never used by any run (the t2 dataset used 10000-10699, calibration "
                     "20000-20039, demos 0 and 1000-1009, gate evaluation 2000-2019, the laptop ladder 3000-3019). "
                     "Earlier laptop ladder: f4a75a2 / t3/trust/runs/ladder_report.json.")


def build_testsets() -> dict[Path, dict]:
    """The committed test-set specs (their sha256 is each declaration's test_set_sha256)."""
    from .calibrate import ARTIFACT
    from .gate import load_thresholds, thresholds_path
    from .scripted import load_script

    _, script_sha = load_script()
    art = HERE.parent.parent / ARTIFACT
    art_sha = sha256_bytes(art.read_bytes())
    models = {"healthy": {"model_id": "t2-response-v0@v0", "load": "t2.predict.load", "fault": None},
              "degraded": {"model_id": "t2-response-v0-degraded-tilt-reversed@v0", "load": "t2.predict.load_degraded",
                           "fault": "tilt_reversed"}}
    common = {"artifact": ARTIFACT, "artifact_sha256": art_sha, "models": models,
              "script": "t3/trust/scripted_changes.json", "script_sha256": script_sha}
    thresholds = {v: load_thresholds(thresholds_path(v))[1] for v in ("v1", "v2")}
    bar = {"healthy_false_pauses_max": 2, "degraded_paused_by_change_6_min": 19}
    rungs = {"R1": {"name": "maveric", "label": "MAVERIC", "config": "official", "scenario": "nominal"},
             "R2": {"name": "rehearsal", "label": "REHEARSAL (Viavi-like Maveric)", "config": "viavi-like",
                    "scenario": "viavi_stationary"}}
    gate = dict(common, what="trust-gate evaluation seeds (fresh for v2; used once before on the laptop, dd15c54)",
                seeds=list(range(2000, 2020)), changes=24, scenario="nominal", config="official",
                thresholds=thresholds, success_bar=bar, provenance=REMEASURE_NOTE_GATE,
                references={"declaration_commit": "ed7d3c4", "laptop_run_commit": "dd15c54",
                            "laptop_summary": "t3/trust/runs/v2_eval.json"})
    gate_fresh = dict(common, what="trust-gate evaluation seeds 3000-3019, fresh for the gate",
                      seeds=list(range(3000, 3020)), changes=24, scenario="nominal", config="official",
                      thresholds=thresholds, success_bar=bar, provenance=FRESH_NOTE_GATE,
                      references={"thresholds_v1_commit": "2c8a57c", "v2_declaration_commit": "ed7d3c4",
                                  "laptop_v2_eval": "t3/trust/runs/v2_eval.json"})
    lad = dict(common, what="ladder seeds (used once before on the laptop, f4a75a2)", seeds=list(range(3000, 3020)),
               steps=5, rungs=rungs, provenance=REMEASURE_NOTE_LADDER,
               references={"laptop_run_commit": "f4a75a2", "laptop_report": "t3/trust/runs/ladder_report.json"})
    lad_fresh = dict(common, what="ladder seeds 4000-4019, fresh", seeds=list(range(4000, 4020)), steps=5, rungs=rungs,
                     provenance=FRESH_NOTE_LADDER,
                     references={"laptop_run_commit": "f4a75a2", "laptop_report": "t3/trust/runs/ladder_report.json"})
    return {GATE_TESTSET: gate, LADDER_TESTSET: lad, GATE_TESTSET_FRESH: gate_fresh, LADDER_TESTSET_FRESH: lad_fresh}


def write_testsets() -> None:
    TESTSETS.mkdir(exist_ok=True)
    for path, doc in build_testsets().items():
        path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def testset_path(arg: str | None, default: Path) -> Path:
    p = Path(arg) if arg else default
    return p if p.is_absolute() else HERE.parent.parent / p


# ---------------------------------------------------------------- runner declarations (tools/sm/run.py --declaration)
RUNNER_RUNS = {
    # judged runs 17-19, on fresh seeds
    "run17_gate_v1_healthy": ("gate", "v1", "healthy", GATE_TESTSET_FRESH),
    "run18_gate_v1_degraded": ("gate", "v1", "degraded", GATE_TESTSET_FRESH),
    "run19_ladder_healthy": ("ladder", None, "healthy", LADDER_TESTSET_FRESH),
}


def rel(p: Path) -> str:
    return p.relative_to(HERE.parent.parent).as_posix()


def runner_declaration(kind: str, version: str | None, model: str, ts_path: Path, twin_version: str) -> tuple[dict, str]:
    """(declaration JSON for tools/sm/run.py --declaration, the --entry string). The runner adds run_id,
    code_sha, region and declared_at and hashes test_set at the pinned commit."""
    ts, ts_sha = load_testset(ts_path)
    args = dict(run_id="x", code_sha="x", region="us-east-2", testset=ts, testset_sha=ts_sha, twin_version=twin_version)
    if kind == "gate":
        d = gate_declaration(version=version, model=model, **args)
        entry = f"-m t3.trust.job_gate_eval --version {version} --model {model} --testset {rel(ts_path)}"
    else:
        d = ladder_declaration(model=model, **args)
        entry = f"-m t3.trust.job_ladder --model {model} --testset {rel(ts_path)}"
    d["twin"]["config"]["command"] = "python " + entry
    for k in ("run_id", "code_sha", "region", "test_set_sha256"):
        d.pop(k)
    d["test_set"] = rel(ts_path)
    return d, entry


def write_runner_declarations(twin_version: str) -> dict[str, str]:
    DECLARATIONS.mkdir(exist_ok=True)
    out = {}
    for name, (kind, v, m, ts) in RUNNER_RUNS.items():
        d, entry = runner_declaration(kind, v, m, ts, twin_version)
        (DECLARATIONS / f"{name}.json").write_text(json.dumps(d, indent=1) + "\n")
        out[name] = entry
    return out


def load_testset(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), sha256_bytes(raw)


# ---------------------------------------------------------------- items
def gate_item(*, version: str, model: str, seed: int, case: dict, testset_sha: str, max_n: int) -> dict:
    at = case["paused_at_change"]
    passed = at is None if model == "healthy" else (at is not None and at <= max_n)
    inp = {"version": version, "model": model, "model_id": case["model_id"], "seed": seed, "testset": testset_sha}
    out = {"paused_at_change": at, "trust": case["trust_trajectory"], "e": case["e_trajectory"]}
    return {"item_id": f"{version}/{model}/seed-{seed}", "input_sha256": sha256_json(inp),
            "output_sha256": sha256_json(out), "score": float(at) if at is not None else None, "passed": passed,
            "grader": GRADER_GATE, "backend": BACKEND,
            "extra": {"version": version, "role": model, "model_id": case["model_id"], "model_version": case["version"],
                      "twin_seed": seed, "paused_at_change": at, "min_trust_after_warmup": case["min_trust_after_warmup"],
                      "refused_after_pause": case["refused_after_pause"], "case_id": case["case_id"]}}


# ---------------------------------------------------------------- writing
class Sink:
    """Collect records in memory, validate them with evlib, write once per kind at the end."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.records: dict[str, list[dict]] = {}

    def append_records(self, run_id: str, kind: str, records: list[dict]) -> list[str]:
        if run_id != self.run_id:
            raise ValueError(f"record for run {run_id!r} in sink of {self.run_id!r}")
        self.records.setdefault(kind, []).extend(records)
        return [f"pending:records/{run_id}/{kind}.jsonl"]

    def validate(self) -> None:
        from ._evlib import get_evlib
        evlib = get_evlib()

        for kind, recs in self.records.items():
            evlib._prepare(self.run_id, kind, recs)  # raises on any invalid record

    def flush(self, store=None) -> list[str]:
        """On S3: one object per record (evlib's "objects" layout), so check_unit can give every record
        an AWS time from its own LastModified. Locally: one JSONL file per kind."""
        from ._evlib import get_evlib
        evlib = get_evlib()

        store = store or evlib.default_store()
        out = []
        for kind, recs in sorted(self.records.items()):
            if not recs:
                continue
            if isinstance(store, evlib.S3Store):
                out += evlib.append_records(self.run_id, kind, recs, store=store, layout="objects")
            else:
                out.append(evlib.write_records_jsonl(self.run_id, kind, recs, store=store))
        return out


# ---------------------------------------------------------------- gate evaluation
def run_gate_eval(*, version: str, model: str, run_id: str, seeds: list[int], twin_factory, kpis, predictor,
                  changes, thresholds_path, testset_sha: str, code_sha: str, synthetic: bool = False,
                  provenance: str | None = None) -> dict:
    from .demo import run_case
    from .gate import TrustGate
    from .policy import PolicyClient
    from .records import make_result_record
    from .store import MemoryTrustStore
    from .demo import thresholds_summary

    sink = Sink(run_id)
    items, cases = [], []
    gate = None
    for s in seeds:
        gate = TrustGate(MemoryTrustStore(), thresholds_path, PolicyClient(), sink, synthetic=synthetic)
        case = run_case(gate=gate, twin_factory=twin_factory, kpis=kpis, predictor=predictor, changes=changes,
                        scenario="nominal", seed=s, config="official", run_id=run_id,
                        case_id=f"case-{version}-{model}-{s}", writer=sink, synthetic=synthetic)
        cases.append(case)
        items.append(gate_item(version=version, model=model, seed=s, case=case, testset_sha=testset_sha,
                               max_n=int(gate.thresholds["max_changes_to_pause"])))
    sink.append_records(run_id, "item", [_env(it, run_id, "item", synthetic) for it in items])
    target = gate_target(model)
    n = len(items)
    k_pass = sum(1 for it in items if it["passed"])
    value = float(n - k_pass) if target["metric"] == "fail_count" else float(k_pass)
    passed = value <= target["value"] if target["op"] == "<=" else value >= target["value"]
    paused = [c["paused_at_change"] for c in cases]
    n_paused = sum(x is not None for x in paused)
    details = {"model_id": predictor.model_id, "version": predictor.version, "role": model,
               "thresholds_version": version, "thresholds": thresholds_summary(gate),
               "runs": n, "paused": n_paused, "paused_ci95": wilson(n_paused, n),
               "paused_by_6": sum(x is not None and x <= 6 for x in paused),
               "paused_by_6_ci95": wilson(sum(x is not None and x <= 6 for x in paused), n),
               "paused_at": paused, "seeds": seeds, "interval": "Wilson 95 %",
               "note": provenance or REMEASURE_NOTE_GATE, "policy_client": "dry run (pause forbids simulated; control records synthetic)"}
    result = make_result_record(run_id=run_id, unit=f"t3.trust_gate.{version}.{model}", metric=target["metric"],
                                value=value, target=target, passed=passed, n_items=n,
                                checker={"script": "t3/trust/jobs.py", "code_sha": code_sha}, details=details,
                                synthetic=synthetic)
    return {"sink": sink, "result": result, "items": items, "cases": cases}


def _env(rec: dict, run_id: str, kind: str, synthetic: bool) -> dict:
    from .records import envelope

    out = envelope(kind, run_id, synthetic=synthetic)
    out.update({k: v for k, v in rec.items() if k not in ("kind", "run_id")})
    return out


# ---------------------------------------------------------------- ladder
def run_ladder_job(*, model: str, run_id: str, seeds: list[int], rungs: list[dict], kpis, predictor, changes,
                   testset_sha: str, code_sha: str, synthetic: bool = False, provenance: str | None = None) -> dict:
    from .ladder import OBJECTIVE, OBJECTIVE_SHA, ladder
    from .records import make_result_record

    lad = ladder(predictor, rungs, kpis, seeds=seeds, changes=changes)
    sink = Sink(run_id)
    # One item per paired seed: score = the R1 (MAVERIC) case gain (what the target is on); every rung's
    # case for that seed, R2 included, is kept in the item's extra (no extra record kind, by contract).
    by_seed: dict[int, dict] = {}
    for rung_name, rung in lad["full"].items():
        for c in rung["cases"]:
            by_seed.setdefault(c["seed"], {})[rung_name] = {
                "label": rung["label"], "twin_kind": rung["twin_kind"], "config": rung["config"],
                "scenario": rung["scenario"], "gain": c["gain"], "step_gains": c["step_gains"],
                "picks": c["picks"], "pairing": c["pairing"]}
    items = []
    for s in seeds:
        cases = by_seed[s]
        inp = {"model_id": predictor.model_id, "version": predictor.version, "seed": s, "testset": testset_sha,
               "rungs": sorted(cases)}
        items.append(_env({"item_id": f"{model}/seed-{s}", "input_sha256": sha256_json(inp),
                           "output_sha256": sha256_json(cases), "score": float(cases["maveric"]["gain"]),
                           "passed": None, "grader": GRADER_LADDER, "backend": BACKEND,
                           "extra": {"twin_seed": s, "score_is": "R1 (MAVERIC) case gain", "rungs": cases}},
                          run_id, "item", synthetic))
    sink.append_records(run_id, "item", items)
    target = ladder_target(model)
    mean = float(np.mean([it["score"] for it in items]))
    gain = paired_bootstrap_spec(items)
    if target["metric"] == "gain_ci_low":
        value = gain["ci_low"]
    else:
        value = mean
    passed = value > 0 if target["op"] == ">" else value < 0
    details = dict(lad["details"])
    details.update({"role": model, "objective": OBJECTIVE, "objective_sha256": OBJECTIVE_SHA, "gain": gain,
                    "r1_mean_gain": mean, "r1_gain_ci95": [gain["ci_low"], gain["ci_high"]],
                    "note": REMEASURE_NOTE_LADDER if not provenance else provenance})
    result = make_result_record(run_id=run_id, unit="t3.ladder", metric=target["metric"], value=value,
                                target=target, passed=passed, n_items=len(items),
                                checker={"script": "t3/trust/jobs.py", "code_sha": code_sha}, details=details,
                                synthetic=synthetic)
    return {"sink": sink, "result": result, "items": items, "ladder": lad}


# ---------------------------------------------------------------- paired-bootstrap-v1 (tools/check_unit.py CI_FIELDS)
CI_SEED, CI_RESAMPLES, CI_LEVEL = 0, 10000, 0.95


def paired_bootstrap_spec(items: list[dict], *, seed: int = CI_SEED, resamples: int = CI_RESAMPLES,
                          level: float = CI_LEVEL) -> dict:
    """The interval check_unit part e replays: no baseline run (each item's score already is the gain over
    the exact no-change counterfactual), one pair per item, groups = item ids in sorted order (np.unique),
    numpy PCG64 stream, linear quantiles. Same arithmetic as check_unit.paired_bootstrap_v1."""
    ids = sorted(it["item_id"] for it in items)
    by = {it["item_id"]: float(it["score"]) for it in items}
    d = np.asarray([by[i] for i in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(np.asarray(ids), return_inverse=True)
    g_sum = np.bincount(inv, weights=d, minlength=len(uniq))
    g_n = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    idx = rng.integers(0, len(uniq), size=(resamples, len(uniq)))
    stats = g_sum[idx].sum(axis=1) / g_n[idx].sum(axis=1)
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(stats, [a, 1.0 - a])
    return {"method": "paired-bootstrap-v1", "seed": seed, "resamples": resamples, "level": level,
            "pair_key": "item_id", "score_field": "score", "baseline_run_id": None, "cluster_field": None,
            "subset": None, "value": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "paired_n": int(len(ids)), "numpy": np.__version__}


# ---------------------------------------------------------------- shared CLI flow
def finish(*, run_id: str, declaration: dict, out: dict, dry_run: bool, declare_first_done: bool) -> int:
    """Validate everything; in a dry run print and stop; otherwise write records then the result."""
    from ._evlib import get_evlib
    evlib = get_evlib()

    out["sink"].validate()
    evlib.validate(evlib._envelope(dict(out["result"]), "result", run_id))
    if dry_run:
        print(json.dumps({"dry_run": True, "run_id": run_id, "declaration": declaration,
                          "records": {k: len(v) for k, v in out["sink"].records.items()},
                          "result": {k: out["result"][k] for k in ("unit", "metric", "value", "target", "passed", "n_items")}},
                         indent=1, default=str))
        return 0
    store = evlib.default_store()
    if not declare_first_done and evlib.read_declared(run_id, store=store) is None:
        print(f"warning: no declared/{run_id}.json yet; the job runner should write it before the run")
    paths = out["sink"].flush(store=store)
    paths.append(evlib.write_result(out["result"], store=store))
    print(json.dumps({"run_id": run_id, "objects_written": len(paths), "result": paths[-1], "value": out["result"]["value"],
                      "passed": out["result"]["passed"]}, indent=1))
    return 0


def declare_first(declaration: dict) -> None:
    from ._evlib import get_evlib
    evlib = get_evlib()

    evlib.write_declared(declaration)
    time.sleep(2.0)  # S3 LastModified has one-second resolution: the declaration must be strictly earlier


# ---------------------------------------------------------------- declaration templates
PLACEHOLDERS = {"run_id": "<RUN_ID>", "code_sha": "<CODE_SHA: 40-hex commit the job image runs>",
                "region": "<AWS_REGION, e.g. us-east-2>", "twin_version": "<filled at run time from twin.replay>"}
RUNS = ([("gate", v, m) for v in ("v1", "v2") for m in ("healthy", "degraded")]
        + [("ladder", None, m) for m in ("healthy", "degraded")])


def template_docs() -> dict[str, dict]:
    gate_ts, gate_sha = load_testset(GATE_TESTSET)
    lad_ts, lad_sha = load_testset(LADDER_TESTSET)
    out = {}
    for kind, v, m in RUNS:
        common = dict(run_id=PLACEHOLDERS["run_id"], code_sha=PLACEHOLDERS["code_sha"], region=PLACEHOLDERS["region"],
                      twin_version=PLACEHOLDERS["twin_version"])
        if kind == "gate":
            doc = gate_declaration(version=v, model=m, testset=gate_ts, testset_sha=gate_sha, **common)
            name = f"gate_eval_{v}_{m}"
        else:
            doc = ladder_declaration(model=m, testset=lad_ts, testset_sha=lad_sha, **common)
            name = f"ladder_{m}"
        out[name] = doc
    return out


def write_templates() -> None:
    DECLARATIONS.mkdir(exist_ok=True)
    for name, doc in template_docs().items():
        (DECLARATIONS / f"{name}.template.json").write_text(json.dumps(doc, indent=1) + "\n")


if __name__ == "__main__":
    from .job_gate_eval import twin_version

    write_testsets()
    write_templates()
    tv = twin_version()
    if not tv:
        raise SystemExit("twin.replay is needed to name the twin version in the runner declarations")
    for name, entry in write_runner_declarations(tv).items():
        print(f"{name}: --entry \"{entry}\"")
    print("wrote", sorted(p.name for p in TESTSETS.iterdir()), sorted(p.name for p in DECLARATIONS.iterdir()))
