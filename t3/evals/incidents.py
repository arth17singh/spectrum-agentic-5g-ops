"""The frozen incident suite: scripted misconfigurations injected into the in-process Maveric twin.

Each incident starts from a healthy reset (scenario, seed, preset), applies the healthy per-cell
commissioning as step 1 (none on the "visible" tier; seeded per-cell offsets on the "masked" tier, so
the configuration alone does not reveal the faulted cell), then one fault change as step 2 (through
the SDK safety filter, which must clip neither), and records the twin's KPIs on two
trajectories through the grade step: ``healthy`` (no fault, no-op steps) and ``unfixed`` (fault, then
no-op steps). Operators are later graded against those, at the same step.

Candidates come from a fixed template list x preset; each candidate has its own seed. Seeds are
split into dev / test by sha256 of the seed alone (no seed is in both). A candidate is dropped if
the fault leaves no_action within tolerance (the fault must actually break the network) or if the
safety filter clipped the fault. The test split is frozen to ``suite/incidents-test.jsonl`` (one
canonical JSON line per incident) and its sha256 to ``suite/incidents-test.sha256``.

    python -m t3.evals.incidents --check                 # regenerate, compare with the frozen bytes
    python -m t3.evals.incidents --out DIR               # regenerate and freeze into DIR (never the code tree)
    python -m t3.evals.incidents --out DIR --evidence-dir E --run-id R   # ... plus evidence (a runner job)
    python -m t3.evals.incidents --write                 # freeze into t3/evals/suite (local, same platform only)
    python -m t3.evals.incidents --write-candidates      # refresh suite/candidates.jsonl (the generator's input)

The twin is a float32 GP: frozen KPIs replay to 1e-6 only on the platform that froze them. The suite
of record is frozen inside a SageMaker job (x86_64, the runner's CPU image) with ``--out {output_dir}``,
and every scored run that replays it runs there too. The job writes one ``item`` per candidate (its
input and output sha256) and a ``result`` (``n_items`` == the candidate count) through evlib, so
``tools/check_unit.py`` applies to the freeze itself. Its declared ``test_set_sha256`` is the sha256 of
``suite/candidates.jsonl``, the committed input: the candidate specs, one canonical line each.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from t3.evals.graders import grade_kpis, load_targets
from t3.evals.twin_backend import (
    BACKEND,
    PRESET_SCENARIO,
    ReplayTwin,
    graded,
    twin_version,
)

SUITE_DIR = Path(__file__).with_name("suite")
TEST_PATH = SUITE_DIR / "incidents-test.jsonl"
DEV_PATH = SUITE_DIR / "incidents-dev.jsonl"
HASH_PATH = SUITE_DIR / "incidents-test.sha256"
META_PATH = SUITE_DIR / "incidents-meta.json"

SUITE_VERSION = "t3-incidents-v4"
CANDIDATES_PATH = SUITE_DIR / "candidates.jsonl"
FREEZE_UNIT = "t3.evals.suite-freeze"
FREEZE_GRADER = ("t3/evals/incidents.py:v4 (twin replay; kept iff no_action fails the tolerance at the grade "
                 "step and the safety filter clipped nothing)")
FREEZE_MODEL_ID = "none (data generation: twin replay only, no model)"
# Earlier frozen test sets this one replaces (never reuse them for scoring).
SUPERSEDES = [
    {"suite": "t3-incidents-v1", "test_sha256": "c0846996a08357836efcc7af1b7ab285d63b8c3622a846a580661d7bef388f0a",
     "twin_backend": "maveric-inprocess-standin",
     "reason": "every cell started identical, so a config-diff heuristic solved it; replaced by the masked tier"},
    {"suite": "t3-incidents-v2", "test_sha256": "20f9013399da360b26214e11de15c9f9edbb8dcb9bcc4cf1d89fb7e194d3dd12",
     "twin_backend": "maveric-inprocess-standin",
     "reason": "frozen on t3/evals' stand-in twin; the twin backend changed to W2's twin.replay over ran-rl 1.2.9's "
               "Maveric twin (user-share-weighted pooled p10 KPIs), so the frozen KPIs no longer replay. Re-frozen "
               "once, before any swarm or model run used the suite."},
    {"suite": "t3-incidents-v3", "test_sha256": "485b3bde1dc974a07e5950115e1fc1a8fa0dd2cb6cfc88e377a4ac7a01588a63",
     "twin_backend": "maveric-inprocess", "platform": "Darwin-arm64",
     "reason": "frozen on the real twin but on Apple arm64; a SageMaker x86_64 job replays the float32 GP twin "
               "beyond the 1e-6 replay tolerance (inc-off-mas-4102-tilt+5). Re-frozen on x86_64 in a "
               "SageMaker job (the platform of record) before any swarm or model run."},
]
SEED_BASE = 4100
N_CANDIDATES = 84
SPLIT_SALT = "t3-evals-split-v1"
KPI_DECIMALS = 9  # rounding for the frozen file only; grading recomputes from the twin

# (name, [(cell offset, d_power_db, d_tilt_deg), ...]); the cell offset is added to the base cell.
# Every fault's exact inverse is inside the per-step safety limits (tilt +-5, power -6..+8) and in
# the oracle's candidate grid.
TEMPLATES: list[tuple[str, list[tuple[int, float, float]]]] = [
    ("tilt+5", [(0, 0.0, 5.0)]),
    ("power-6", [(0, -6.0, 0.0)]),
    ("tilt+4", [(0, 0.0, 4.0)]),
    ("power-4", [(0, -4.0, 0.0)]),
    ("power-3+tilt+3", [(0, -3.0, 3.0)]),
    ("power-6+tilt+5", [(0, -6.0, 5.0)]),
    ("two-cell:power-6|tilt+5", [(0, -6.0, 0.0), (1, 0.0, 5.0)]),
]
PRESETS = ("official", "viavi-like")


def canonical_line(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def split_of(seed: int) -> str:
    """'test' or 'dev', from sha256 of the seed only."""
    return "test" if hashlib.sha256(f"{SPLIT_SALT}:{seed}".encode()).digest()[0] % 2 == 0 else "dev"


TIERS = ("visible", "masked")
COMMISSION_POWER_DB = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
COMMISSION_TILT_DEG = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)


def commissioning(seed: int, tier: str, n_cells: int = 4) -> list[dict[str, Any]]:
    """The healthy per-cell configuration applied at step 1 (before the fault).

    "visible": none, every cell keeps the preset's identical power and tilt, so a faulted cell
    stands out in the configuration. "masked": seeded per-cell offsets, so the configuration alone
    no longer says which cell is wrong.
    """
    if tier == "visible":
        return []
    import numpy as np

    rng = np.random.default_rng([seed, 7])
    out = []
    for c in range(n_cells):
        dp = float(rng.choice(COMMISSION_POWER_DB))
        dt = float(rng.choice(COMMISSION_TILT_DEG))
        if dp or dt:
            out.append({"cell": c, "d_power_db": dp, "d_tilt_deg": dt})
    return out


def candidate(i: int) -> dict[str, Any]:
    """The i-th candidate spec (deterministic, no twin needed)."""
    seed = SEED_BASE + i
    preset = PRESETS[i % 2]
    tier = TIERS[(i // 2) % 2]
    name, parts = TEMPLATES[(i // 4) % len(TEMPLATES)]
    base_cell = (i // (4 * len(TEMPLATES))) % 4
    fault = [
        {"cell": (base_cell + off) % 4, "d_power_db": float(dp), "d_tilt_deg": float(dt)}
        for off, dp, dt in parts
    ]
    return {"seed": seed, "preset": preset, "scenario": PRESET_SCENARIO[preset], "tier": tier, "template": name,
            "commissioning": commissioning(seed, tier), "fault": fault}


def _r(d: dict[str, float]) -> dict[str, float]:
    return {k: round(v, KPI_DECIMALS) for k, v in d.items()}


def script(spec: dict[str, Any], targets: dict[str, Any], with_fault: bool) -> list[list[dict[str, Any]]]:
    """The changes applied at steps 1..fault_step: commissioning at step 1, the fault at fault_step."""
    ep = targets["episode"]
    steps: list[list[dict[str, Any]]] = [[] for _ in range(ep["fault_step"])]
    steps[0] = list(spec["commissioning"])
    if with_fault:
        steps[ep["fault_step"] - 1] = list(spec["fault"])
    return steps


def build(spec: dict[str, Any], targets: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Run the healthy and faulted trajectories. Returns (incident or None, reason)."""
    ep = targets["episode"]
    n_steps, fs = ep["grade_step"], ep["fault_step"]
    if fs < 2:
        raise ValueError("fault_step must leave step 1 for commissioning")
    healthy = ReplayTwin(spec["scenario"], spec["seed"], spec["preset"])
    faulted = ReplayTwin(spec["scenario"], spec["seed"], spec["preset"])
    s0 = healthy.reset()
    faulted.reset()
    h_script, u_script = script(spec, targets, False), script(spec, targets, True)
    h_traj, u_traj = [graded(s0)], [graded(s0)]
    for step in range(1, n_steps + 1):
        h_traj.append(graded(healthy.apply(h_script[step - 1] if step <= fs else [])))
        u_traj.append(graded(faulted.apply(u_script[step - 1] if step <= fs else [])))
    for entry in faulted.action_log[:fs]:
        if entry["clamped"]:
            return None, f"commissioning or fault clipped by the safety filter: {entry['rules']}"
    g = grade_kpis(u_traj[-1], h_traj[-1], u_traj[-1], targets)
    if g["success"]:
        return None, "no_action already within tolerance at the grade step (fault not material)"
    iid = (f"inc-{'off' if spec['preset'] == 'official' else 'vvl'}-{spec['tier'][:3]}-{spec['seed']}-"
           f"{spec['template']}")
    s_fault = faulted.action_log[fs - 1]
    return {
        "suite": SUITE_VERSION,
        "incident_id": iid,
        "split": split_of(spec["seed"]),
        "scenario": spec["scenario"],
        "seed": spec["seed"],
        "preset": spec["preset"],
        "tier": spec["tier"],
        "template": spec["template"],
        "commissioning": spec["commissioning"],
        "fault": spec["fault"],
        "fault_cells": sorted({f["cell"] for f in spec["fault"]}),
        "fault_step": fs,
        "grade_step": n_steps,
        "fault_filtered": s_fault["filtered"],
        "pre_fault_kpis": _r(h_traj[fs - 1]),
        "post_fault_kpis": _r(u_traj[fs]),
        "healthy_kpis_by_step": [_r(k) for k in h_traj],
        "unfixed_kpis_by_step": [_r(k) for k in u_traj],
        "material_kpis": [k for k, p in g["per_kpi"].items() if p["material"]],
    }, "kept"


def candidates_bytes() -> bytes:
    """The generator's input: every candidate spec, one canonical JSON line each (committed)."""
    return "".join(canonical_line(candidate(i)) + "\n" for i in range(N_CANDIDATES)).encode("utf-8")


def generate(targets: dict[str, Any] | None = None, timings: bool = False) -> dict[str, Any]:
    import time

    targets = targets or load_targets()
    kept: dict[str, list[dict[str, Any]]] = {"dev": [], "test": []}
    dropped = []
    outcomes = []
    for i in range(N_CANDIDATES):
        spec = candidate(i)
        t0 = time.perf_counter()
        inc, why = build(spec, targets)
        ms = (time.perf_counter() - t0) * 1000.0
        if inc is None:
            drop = {"seed": spec["seed"], "preset": spec["preset"], "tier": spec["tier"],
                    "template": spec["template"],
                    "fault": spec["fault"], "split": split_of(spec["seed"]), "reason": why}
            dropped.append(drop)
            outcomes.append({"spec": spec, "kept": False, "output": drop, "latency_ms": ms})
        else:
            kept[inc["split"]].append(inc)
            outcomes.append({"spec": spec, "kept": True, "output": inc, "latency_ms": ms})
    out = {"dev": kept["dev"], "test": kept["test"], "dropped": dropped}
    if timings:
        out["outcomes"] = outcomes
    return out


def to_bytes(incidents: list[dict[str, Any]]) -> bytes:
    return "".join(canonical_line(x) + "\n" for x in incidents).encode("utf-8")


def load_test(path: Path = TEST_PATH, verify_hash: bool = True) -> tuple[list[dict[str, Any]], str]:
    """The frozen test incidents and the sha256 of the file's bytes (checked against the sidecar)."""
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if verify_hash and Path(path).resolve() == TEST_PATH.resolve():
        want = HASH_PATH.read_text().split()[0]
        if digest != want:
            raise RuntimeError(f"{path} sha256 {digest} != frozen {want}: the test set changed")
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()], digest


def meta_for(out: dict[str, Any], digest: str) -> dict[str, Any]:
    return {
        "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "supersedes": SUPERSEDES,
        "suite": SUITE_VERSION, "candidates": N_CANDIDATES, "seed_base": SEED_BASE, "split_salt": SPLIT_SALT,
        "candidates_sha256": hashlib.sha256(candidates_bytes()).hexdigest(),
        "templates": [t[0] for t in TEMPLATES], "presets": list(PRESETS), "tiers": list(TIERS),
        "counts": {"test": len(out["test"]), "dev": len(out["dev"]), "dropped": len(out["dropped"])},
        "test_sha256": digest, "dev_sha256": hashlib.sha256(to_bytes(out["dev"])).hexdigest(),
        "twin_backend": BACKEND, "twin_version": twin_version(),
        "python": sys.version.split()[0], "platform": f"{platform.system()}-{platform.machine()}",
        "note": "the twin is a float32 GP: frozen KPIs replay to 1e-6 only on the platform that froze them; "
                "score this suite on that platform (the runner's x86_64 CPU image)",
        "dropped": out["dropped"],
    }


def write_suite(out: dict[str, Any], dest: Path) -> dict[str, Any]:
    """Write the four suite files into ``dest``. Refuses unless the real twin is the backend."""
    if BACKEND != "maveric-inprocess":
        raise SystemExit(f"refusing to freeze on twin backend {BACKEND!r}: the real twin (twin.replay) is required")
    test_bytes = to_bytes(out["test"])
    digest = hashlib.sha256(test_bytes).hexdigest()
    meta = meta_for(out, digest)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / TEST_PATH.name).write_bytes(test_bytes)
    (dest / DEV_PATH.name).write_bytes(to_bytes(out["dev"]))
    (dest / HASH_PATH.name).write_text(f"{digest}  {TEST_PATH.name}\n")
    (dest / META_PATH.name).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return meta


def write_evidence(out: dict[str, Any], meta: dict[str, Any], evidence_dir: Path, run_id: str,
                   dest: Path | None) -> dict[str, Any]:
    """One item per candidate and a result (n_items == candidate count), through evlib (a local store)."""
    from t3.evals.evshim import Evidence, code_sha

    items = []
    for o in out["outcomes"]:
        spec = o["spec"]
        items.append({
            "item_id": f"cand-{spec['seed']}",
            "input_sha256": hashlib.sha256(canonical_line(spec).encode("utf-8")).hexdigest(),
            "output_sha256": hashlib.sha256(canonical_line(o["output"]).encode("utf-8")).hexdigest(),
            "score": None,
            "passed": None,
            "grader": FREEZE_GRADER,
            "backend": BACKEND,
            "latency_ms": o["latency_ms"],
            "extra": {
                "kept": o["kept"], "split": split_of(spec["seed"]), "preset": spec["preset"], "tier": spec["tier"],
                "template": spec["template"],
                "incident_id": o["output"]["incident_id"] if o["kept"] else None,
                "reason": None if o["kept"] else o["output"]["reason"],
            },
        })
    ev = Evidence(evidence_dir)
    ev.append_records(run_id, "item", items)
    n = len(items)
    result = {
        "run_id": run_id,
        "unit": FREEZE_UNIT,
        "metric": "n_items",
        "value": n,
        "target": {"metric": "n_items", "op": "==", "value": N_CANDIDATES, "unit": "count"},
        "passed": n == N_CANDIDATES,
        "n_items": n,
        "checker": {"script": "t3/evals/incidents.py", "code_sha": code_sha()},
        "details": {
            "suite": SUITE_VERSION, "counts": meta["counts"], "test_sha256": meta["test_sha256"],
            "dev_sha256": meta["dev_sha256"], "candidates_sha256": meta["candidates_sha256"],
            "twin_backend": meta["twin_backend"], "twin_version": meta["twin_version"],
            "platform": meta["platform"], "python": meta["python"], "frozen_at": meta["frozen_at"],
            "suite_files": None if dest is None else str(dest),
        },
    }
    ev.write_result(result)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="freeze into t3/evals/suite (the code tree; local only)")
    ap.add_argument("--out", type=Path, help="freeze into this directory instead (a job: {output_dir}/suite)")
    ap.add_argument("--check", action="store_true", help="regenerate and compare with the frozen test file")
    ap.add_argument("--evidence-dir", type=Path, help="also write item + result records here (needs --run-id)")
    ap.add_argument("--run-id", help="run id for the evidence records")
    ap.add_argument("--write-candidates", action="store_true", help="refresh suite/candidates.jsonl and exit")
    a = ap.parse_args(argv)
    if a.write_candidates:
        CANDIDATES_PATH.write_bytes(candidates_bytes())
        print(json.dumps({"candidates": N_CANDIDATES,
                          "sha256": hashlib.sha256(candidates_bytes()).hexdigest()}))
        return 0
    if a.write and a.out:
        ap.error("--write and --out are exclusive")
    if bool(a.evidence_dir) != bool(a.run_id):
        ap.error("--evidence-dir and --run-id go together")
    if a.evidence_dir and not (a.write or a.out):
        ap.error("--evidence-dir records a freeze: pass --out (or --write)")
    if CANDIDATES_PATH.exists() and CANDIDATES_PATH.read_bytes() != candidates_bytes():
        raise SystemExit(f"{CANDIDATES_PATH} does not match the generator's candidates: refresh it with "
                         "--write-candidates and commit before freezing")
    out = generate(timings=bool(a.evidence_dir))
    test_bytes = to_bytes(out["test"])
    digest = hashlib.sha256(test_bytes).hexdigest()
    summary: dict[str, Any] = {
        "test": len(out["test"]), "dev": len(out["dev"]), "dropped": len(out["dropped"]),
        "test_sha256": digest, "backend": BACKEND, "platform": f"{platform.system()}-{platform.machine()}",
        "tiers_test": {t: sum(1 for i in out["test"] if i["tier"] == t) for t in TIERS},
    }
    dest = SUITE_DIR if a.write else a.out
    if dest is not None:
        meta = write_suite(out, dest)
        summary.update(frozen_at=meta["frozen_at"], written_to=str(dest))
        if a.evidence_dir:
            res = write_evidence(out, meta, a.evidence_dir, a.run_id, dest)
            summary.update(evidence_run_id=a.run_id, result_passed=res["passed"])
    if a.check:
        frozen = TEST_PATH.read_bytes()
        summary["matches_frozen"] = frozen == test_bytes
        if frozen != test_bytes:
            summary["first_difference"] = first_difference(out["test"])
        print(json.dumps(summary))
        return 0 if frozen == test_bytes else 1
    print(json.dumps(summary))
    return 0


def first_difference(fresh: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Where a regenerated test split first departs from the frozen file, KPI by KPI."""
    frozen = {x["incident_id"]: x for x in load_test(TEST_PATH, verify_hash=False)[0]}
    now = {x["incident_id"]: x for x in fresh}
    if set(frozen) != set(now):
        return {"only_frozen": sorted(set(frozen) - set(now))[:5], "only_regenerated": sorted(set(now) - set(frozen))[:5]}
    for iid in sorted(now):
        f, n = frozen[iid], now[iid]
        for field in ("post_fault_kpis", "pre_fault_kpis"):
            for k, v in n[field].items():
                if f[field].get(k) != v:
                    return {"incident_id": iid, "field": field, "kpi": k, "frozen": f[field].get(k),
                            "regenerated": v, "diff": v - f[field].get(k, 0.0)}
        if canonical_line(f) != canonical_line(n):
            keys = sorted(k for k in n if f.get(k) != n.get(k))
            return {"incident_id": iid, "fields": keys[:6]}
    return None


if __name__ == "__main__":
    raise SystemExit(main())
