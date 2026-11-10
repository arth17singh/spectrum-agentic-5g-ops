"""The large frozen held-out test set for remediation policies: 1,008 masked faults, stratified (heldout-v2).

heldout-v1 (frozen 4 Nov 08:21Z) did not replay in the first evaluation's replay check (before any result) and
is kept as history; v2 is regenerated from the same reservation. Every process is pinned to one torch thread
with the evaluation's own function (t3/evals/run.py single_thread), the host is recorded in the meta, and
every chosen incident is replayed through run_incident's own 1e-6 check before anything is written
(``self_replay``; ``write_outputs`` refuses otherwise).

Same generator procedure as the t3 test split (``t3/evals/incidents.py``): the same templates, presets,
masked-tier commissioning (``incidents.commissioning``), episode (t3/evals/targets.json) and keep rule
(``incidents.build`` v4: kept iff no_action fails the tolerance at the grade step and the safety filter
clipped nothing). Only the sampling differs: STRATIFIED over 7 templates x 4 base cells x 2 presets = 56
strata, 18 incidents each (the two-cell template faults the base cell and its neighbour, as in
``incidents.candidate`` and ``t3/evals/faults.py``).

Candidates are fixed by their index alone, so the result does not depend on worker count or timing:

    candidate k: stratum s = k % 56, attempt j = k // 56, seed = SEED_BASE + k
    stratum s = (template * 4 + base_cell) * 2 + preset

Per stratum, the incident set is the first 18 KEPT candidates in attempt order. Candidates are evaluated
in waves (attempts 0-27 of every stratum, then 12 more attempts for each stratum still short), always as
a contiguous prefix per stratum, so wave sizes change only how many surplus candidates are evaluated,
never which incidents are chosen. A stratum still short after ATTEMPTS_MAX attempts is reported short
(the declared n_items target then fails).

Seeds come from a reserved block (SEED_BASE .. SEED_BASE + 56 * ATTEMPTS_MAX - 1) that no other part of the
repo uses. ``--write-reservation`` records that check (known ranges, every suite's explicit seeds, the
remediation training candidates, spec-hash overlap, a literal scan of the repo) in
``heldout-reservation.json``, committed before the job; the job refuses to run unless the generator still
reproduces the reserved candidate specs byte for byte.

The twin is a float32 GP whose KPIs replay to 1e-6 only on the platform that froze them, so the set of
record is generated inside a SageMaker job (x86_64, the runner's CPU image), where the evaluation runs too.

    python -m t1.rl.remediation.heldout --write-reservation          # local, once; commit the file
    python -m t1.rl.remediation.heldout --print-declaration > /tmp/rem-heldout-decl.json
    python -m tools.sm.run --run-id t1.rl.remediation-heldout-<UTC> --declaration /tmp/rem-heldout-decl.json \\
      --instance-type ml.m5.xlarge --max-run 3600 --requirement "stable-baselines3==2.9.2" \\
      --requirement "gymnasium==1.4.0" --requirement "pydantic>=2.7,<3" \\
      --entry "-m t1.rl.remediation.heldout --run-id {run_id} --evidence-dir {evidence_dir} \\
               --output-dir {output_dir} --workers 4"
    python -m t1.rl.remediation.heldout --check PATH --workers N      # regenerate (same platform), compare

Local smoke (at most 40 candidates, labelled synthetic, never a claim):
    python -m t1.rl.remediation.heldout --smoke --strata 4 --quota 2 --max-candidates 40 \\
      --run-id smoke-<x> --evidence-dir /tmp/e --output-dir /tmp/o --declare

Outputs ({output_dir}): heldout-v2.jsonl (one canonical incident per line, in candidate-index order),
heldout-v2.sha256, heldout-v2-meta.json (counts per stratum, drops and reasons, seed range, host, code sha,
the self-replay check), heldout-v2-replay-check.json and heldout-v2-candidates-log.jsonl (every evaluated
candidate). Records: one ``item`` per chosen incident and a ``result`` with n_items == 1008; if the
self-replay check fails, nothing is written and the job fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from t3.evals import incidents as inc

VERSION = "t1-remediation-heldout-v2"
# v1 (frozen 4 Nov 08:21Z by run t1.rl.remediation-heldout-20261104T081231Z) stays committed as history: its first
# incident did not replay in the first evaluation (ReplayMismatch), before any result; v2 is regenerated from
# the same reservation with every process pinned and a full self-replay guard before anything is written.
V1_SUITE_NAME = "heldout-v1.jsonl"
UNIT = "t1.rl.remediation-heldout"
SEED_BASE = 2_000_001
QUOTA = 18
ATTEMPTS_MAX = 400  # per stratum; a stratum that keeps fewer than 18 of 400 is reported short
TIER = "masked"
WAVES = (28, 12)  # attempts per stratum in the first wave, then in each later wave (stratum still short)
N_CELLS = 4
# (template index, base cell, preset index); stratum s = (t * 4 + c) * 2 + p
STRATA: list[tuple[int, int, int]] = [(t, c, p) for t in range(len(inc.TEMPLATES)) for c in range(N_CELLS)
                                      for p in range(len(inc.PRESETS))]
N_STRATA = len(STRATA)
N_TARGET = QUOTA * N_STRATA
SEED_LO, SEED_HI = SEED_BASE, SEED_BASE + N_STRATA * ATTEMPTS_MAX - 1
SMOKE_MAX = 40

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SUITE_DIR = HERE / "suite"
SUITE_NAME = "heldout-v2.jsonl"
SUITE_PATH = SUITE_DIR / SUITE_NAME
HASH_PATH = SUITE_DIR / "heldout-v2.sha256"
META_NAME = "heldout-v2-meta.json"
LOG_NAME = "heldout-v2-candidates-log.jsonl"
REPLAY_NAME = "heldout-v2-replay-check.json"
RESERVATION_PATH = HERE / "heldout-reservation.json"
RESERVATION_REL = "t1/rl/remediation/heldout-reservation.json"
GRADER = ("t1/rl/remediation/heldout.py:v1 = t3/evals/incidents.py build v4 on the masked tier (kept iff "
          "no_action fails the tolerance at the grade step and the safety filter clipped nothing); per stratum "
          "the first 18 kept candidates in attempt order")
MODEL_ID = "none (data generation: twin replay only, no model)"
# files that define or hold the reserved block (never counted as a literal-scan hit)
SCAN_OWN = ("t1/rl/remediation/heldout.py", "t1/rl/remediation/heldout-reservation.json",
            "t1/rl/tests/test_heldout.py", "t1/rl/remediation/suite/heldout-v1.jsonl",
            "t1/rl/remediation/suite/heldout-v1-meta.json", "t1/rl/remediation/suite/heldout-v2.jsonl",
            "t1/rl/remediation/suite/heldout-v2-meta.json")


# --------------------------------------------------------------------------- candidates

def stratum_info(s: int) -> dict[str, Any]:
    t, c, p = STRATA[s]
    return {"stratum": s, "template": inc.TEMPLATES[t][0], "base_cell": c, "preset": inc.PRESETS[p]}


def index_of(s: int, j: int) -> int:
    return j * N_STRATA + s


def spec_for(k: int) -> dict[str, Any]:
    """Candidate k (deterministic, no twin needed)."""
    from t3.evals.twin_backend import PRESET_SCENARIO

    s, j = k % N_STRATA, k // N_STRATA
    t, c, p = STRATA[s]
    name, parts = inc.TEMPLATES[t]
    preset = inc.PRESETS[p]
    seed = SEED_BASE + k
    fault = [{"cell": (c + off) % N_CELLS, "d_power_db": float(dp), "d_tilt_deg": float(dt)} for off, dp, dt in parts]
    return {"index": k, "stratum": s, "attempt": j, "base_cell": c, "seed": seed, "preset": preset,
            "scenario": PRESET_SCENARIO[preset], "tier": TIER, "template": name,
            "commissioning": inc.commissioning(seed, TIER), "fault": fault}


def all_specs(attempts_max: int = ATTEMPTS_MAX) -> list[dict[str, Any]]:
    return [spec_for(k) for k in range(N_STRATA * attempts_max)]


def candidates_bytes(specs: list[dict[str, Any]]) -> bytes:
    return "".join(inc.canonical_line(s) + "\n" for s in specs).encode("utf-8")


def generator_input_sha256() -> str:
    """sha256 of every reserved candidate spec, one canonical line each (the declared test_set_sha256)."""
    return hashlib.sha256(candidates_bytes(all_specs())).hexdigest()


# --------------------------------------------------------------------------- reservation (local, pre-job)

def _num_scan(lo: int, hi: int, root: Path = REPO, own: Iterable[str] = SCAN_OWN) -> dict[str, Any]:
    """Every whole number in [lo, hi] written in a repo text file (plain or with _ separators). Digit runs
    inside identifiers or hex strings (a letter or _ on either side) are not numbers and are not counted."""
    from t3.evals import confirm as C

    rx = re.compile(r"(?<![0-9A-Za-z_])([0-9]{1,3}(?:_[0-9]{3})+|[0-9]{6,9})(?![0-9A-Za-z_])")
    skip = {(root / p).resolve() for p in own}
    hits, scanned = [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in C.SCAN_SKIP_DIRS]
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix not in C.SCAN_SUFFIXES or p.resolve() in skip:
                continue
            try:
                text = p.read_text(errors="ignore")
            except OSError:
                continue
            scanned += 1
            for m in rx.finditer(text):
                v = int(m.group(1).replace("_", ""))
                if lo <= v <= hi:
                    hits.append({"file": str(p.relative_to(root)), "value": v})
    return {"files_scanned": scanned, "hits": hits[:20], "n_hits": len(hits), "own_files_skipped": list(own)}


def exclusion_checks(specs: list[dict[str, Any]], scan: bool = True) -> dict[str, Any]:
    """The reserved seeds against every seed range and seed set the repo uses (expected: clean)."""
    from t1.rl.remediation import data as D
    from t3.evals import confirm as C

    seeds = sorted(int(s["seed"]) for s in specs)
    lo, hi = seeds[0], seeds[-1]
    sets = C.explicit_seed_sets()
    conf = {int(i["seed"]) for i in C.load_confirm()[0]}
    rem = {int(s["seed"]) for s in D.load_candidates(check=False)}
    sets["confirm"] = conf
    sets["remediation_train_candidates"] = rem
    mine = set(seeds)
    collisions = {k: sorted(mine & v)[:10] for k, v in sets.items()}
    ranges = C.KNOWN_SEED_RANGES + [
        {"what": "confirmation suite", "lo": min(conf), "hi": max(conf),
         "source": "t3/evals/suite/incidents-confirm.jsonl"},
        {"what": "remediation training candidates", "lo": min(rem), "hi": max(rem),
         "source": "t1/rl/remediation/candidates.jsonl (data.SEED_BASE 1,000,000)"}]
    overlapping = [r["what"] for r in ranges if r["lo"] <= hi and lo <= r["hi"]]
    out: dict[str, Any] = {
        "seed_block": [lo, hi], "n_seeds": len(seeds), "known_ranges": ranges, "ranges_overlapping": overlapping,
        "explicit_sets": {k: len(v) for k, v in sets.items()},
        "explicit_collisions": {k: v for k, v in collisions.items() if v},
        "split_of": {"dev": sum(1 for s in seeds if inc.split_of(s) == "dev"),
                     "test": sum(1 for s in seeds if inc.split_of(s) == "test")},
        "split_of_note": "split_of labels every seed dev or test; this set is its own split ('heldout')"}
    if scan:
        out["literal_scan"] = _num_scan(lo, hi)
    out["clean"] = (not overlapping and not out["explicit_collisions"]
                    and (not scan or out["literal_scan"]["n_hits"] == 0))
    return out


def spec_overlap(specs: list[dict[str, Any]]) -> dict[str, int]:
    """Spec-hash overlap (seed, preset, scenario, tier, template, commissioning, fault) with every suite."""
    from t1.rl.remediation import data as D
    from t3.evals import confirm as C

    mine = {D.spec_hash(s) for s in specs}
    files = {"test": inc.TEST_PATH, "dev": inc.DEV_PATH, "candidates": inc.CANDIDATES_PATH, "confirm": C.CONFIRM_PATH,
             "remediation_train_candidates": D.CANDIDATES_PATH}
    out = {}
    for name, p in files.items():
        rows = [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
        out[name] = len(mine & {D.spec_hash(r) for r in rows})
    return out


def reservation() -> dict[str, Any]:
    specs = all_specs()
    return {"version": VERSION, "seed_base": SEED_BASE, "seed_block": [SEED_LO, SEED_HI], "quota": QUOTA,
            "n_strata": N_STRATA, "n_target": N_TARGET, "attempts_max": ATTEMPTS_MAX, "tier": TIER,
            "waves": list(WAVES), "templates": [t[0] for t in inc.TEMPLATES], "presets": list(inc.PRESETS),
            "index_rule": "candidate k: stratum k % 56, attempt k // 56, seed SEED_BASE + k; "
                          "stratum s = (template * 4 + base_cell) * 2 + preset",
            "selection": "per stratum the first QUOTA kept candidates in attempt order",
            "generator_input_sha256": hashlib.sha256(candidates_bytes(specs)).hexdigest(),
            "n_candidate_specs": len(specs),
            "exclusion": exclusion_checks(specs, scan=True), "spec_hash_overlap": spec_overlap(specs)}


def load_reservation(path: Path = RESERVATION_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def check_reservation(res: dict[str, Any]) -> None:
    """Refuse unless the reservation is clean and the generator still reproduces its candidate specs."""
    if not res["exclusion"]["clean"] or any(res["spec_hash_overlap"].values()):
        raise SystemExit(f"{RESERVATION_REL} is not clean: refusing to generate")
    for k, want in (("seed_base", SEED_BASE), ("quota", QUOTA), ("attempts_max", ATTEMPTS_MAX), ("tier", TIER),
                    ("n_strata", N_STRATA)):
        if res[k] != want:
            raise SystemExit(f"{RESERVATION_REL} {k}={res[k]!r} but the generator has {want!r}")
    got = generator_input_sha256()
    if got != res["generator_input_sha256"]:
        raise SystemExit(f"the generator's candidate specs (sha256 {got}) differ from the reservation "
                         f"({res['generator_input_sha256']}): re-run --write-reservation and commit")


# --------------------------------------------------------------------------- one candidate

def build_one(spec: dict[str, Any], targets: dict[str, Any]) -> dict[str, Any]:
    """incidents.build on one candidate; the kept incident carries the held-out suite's fields."""
    t0 = time.perf_counter()
    built, why = inc.build(spec, targets)
    out: dict[str, Any] = {"index": spec["index"], "stratum": spec["stratum"], "attempt": spec["attempt"],
                           "seed": spec["seed"], "kept": built is not None, "reason": why,
                           "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3)}
    if built is not None:
        out["incident"] = {**built, "suite": VERSION, "split": "heldout", "split_of_seed": built["split"],
                           "stratum": spec["stratum"], "attempt": spec["attempt"], "index": spec["index"],
                           "base_cell": spec["base_cell"]}
    return out


def _init() -> None:
    """Pin torch to one thread with the evaluation's own function (t3/evals/run.py single_thread)."""
    from t3.evals.run import single_thread

    single_thread()


def torch_threads() -> int | None:
    try:
        import torch

        return int(torch.get_num_threads())
    except ImportError:
        return None


def _threads_in_worker(_: Any) -> int | None:
    return torch_threads()


def host_info() -> dict[str, Any]:
    """Where the numbers were made: the twin is a float32 GP, so the CPU and the math library matter."""
    import os
    import subprocess

    cpu = None
    try:
        if Path("/proc/cpuinfo").exists():
            cpu = next((ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines()
                        if ln.startswith("model name")), None)
        else:
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                 timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001 - informational only
        cpu = None
    info: dict[str, Any] = {"platform": f"{platform.system()}-{platform.machine()}", "cpu_model": cpu,
                            "cpu_count": os.cpu_count(), "python": sys.version.split()[0],
                            "instance_type": os.environ.get("TEAM13_INSTANCE_TYPE") or os.environ.get(
                                "SM_CURRENT_INSTANCE_TYPE"), "torch_threads": torch_threads()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["mkl"] = bool(torch.backends.mkl.is_available())
        info["openmp"] = bool(torch.backends.openmp.is_available())
    except ImportError:
        pass
    return info


def _replay_one(args: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    """One incident through the evaluation's own replay check (t3/evals/run.py run_incident, no_action)."""
    from t3.evals.operators import make
    from t3.evals.run import ReplayMismatch, run_incident

    incident, targets = args
    _init()
    try:
        run_incident(incident, make("no_action"), targets)
        return {"incident_id": incident["incident_id"], "ok": True}
    except ReplayMismatch as exc:
        return {"incident_id": incident["incident_id"], "ok": False, "error": str(exc)[-400:]}


def self_replay(incidents: list[dict[str, Any]], targets: dict[str, Any], workers: int = 1) -> dict[str, Any]:
    """Replay every chosen incident exactly as the evaluation will (fresh twin, run_incident's 1e-6 check)."""
    t0 = time.time()
    work = [(i, targets) for i in incidents]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
            out = list(pool.map(_replay_one, work, chunksize=8))
            threads = sorted({t for t in pool.map(_threads_in_worker, range(workers))})
    else:
        _init()
        out = [_replay_one(w) for w in work]
        threads = [torch_threads()]
    bad = [o for o in out if not o["ok"]]
    return {"path": "t3/evals/run.py run_incident (operator no_action), the evaluation's replay check",
            "n": len(out), "mismatches": len(bad), "first_mismatches": bad[:5], "ok": not bad,
            "worker_torch_threads": threads, "wall_s": round(time.time() - t0, 1)}


def _work(args: tuple[list[int], dict[str, Any]]) -> list[dict[str, Any]]:
    ks, targets = args
    return [build_one(spec_for(k), targets) for k in ks]


# --------------------------------------------------------------------------- selection (pure)

def select(results: dict[int, dict[str, Any]], strata: Iterable[int], quota: int) -> dict[int, dict[str, Any]]:
    """Per stratum: its evaluated attempts (a contiguous prefix from attempt 0) and the first ``quota`` kept.

    Depends only on the outcomes of evaluated candidates (keyed by index), never on evaluation order."""
    per: dict[int, dict[str, Any]] = {}
    for s in strata:
        j, chosen, kept = 0, [], 0
        reasons: dict[str, int] = {}
        while index_of(s, j) in results:
            r = results[index_of(s, j)]
            if r["kept"]:
                kept += 1
                if len(chosen) < quota:
                    chosen.append(r)
            else:
                key = str(r["reason"]).split(":")[0].split(" (")[0]
                reasons[key] = reasons.get(key, 0) + 1
            j += 1
        per[s] = {"evaluated": j, "kept": kept, "chosen": chosen, "filled": len(chosen) >= quota,
                  "drops": reasons}
    return per


def generate(*, workers: int = 1, quota: int = QUOTA, strata: Iterable[int] | None = None,
             attempts_max: int = ATTEMPTS_MAX, waves: tuple[int, int] = WAVES, targets: dict[str, Any] | None = None,
             max_candidates: int | None = None,
             evaluate: Callable[[list[int], dict[str, Any]], list[dict[str, Any]]] | None = None,
             log: Callable[[str], None] = lambda m: print(m, flush=True)) -> dict[str, Any]:
    """Evaluate waves of candidates until every stratum has ``quota`` kept incidents (or is exhausted)."""
    from t3.evals.graders import load_targets

    targets = targets or load_targets()
    strata = list(range(N_STRATA)) if strata is None else list(strata)
    results: dict[int, dict[str, Any]] = {}
    if evaluate is None:
        _init()  # the main process too, before any twin work (it does none today; pinned anyway)
    pool = ProcessPoolExecutor(max_workers=workers, initializer=_init) if workers > 1 and evaluate is None else None

    def run(ks: list[int]) -> list[dict[str, Any]]:
        if evaluate is not None:
            return evaluate(ks, targets)
        if pool is None:
            return _work((ks, targets))
        chunks = [ks[i:i + 4] for i in range(0, len(ks), 4)]
        return [r for part in pool.map(_work, [(c, targets) for c in chunks]) for r in part]

    t0, wave = time.time(), 0
    try:
        while True:
            per = select(results, strata, quota)
            open_ = [s for s in strata if not per[s]["filled"] and per[s]["evaluated"] < attempts_max]
            if not open_:
                break
            w = waves[0] if wave == 0 else waves[1]
            ks = sorted(index_of(s, j) for s in open_
                        for j in range(per[s]["evaluated"], min(per[s]["evaluated"] + w, attempts_max)))
            if max_candidates is not None:
                ks = ks[: max(0, max_candidates - len(results))]
                if not ks:
                    break
            for r in run(ks):
                results[r["index"]] = r
            wave += 1
            per = select(results, strata, quota)
            log(f"[heldout] wave {wave}: {len(ks)} candidates, {len(results)} evaluated, "
                f"{sum(1 for s in strata if per[s]['filled'])}/{len(strata)} strata full, "
                f"{round(time.time() - t0, 1)} s")
    finally:
        if pool is not None:
            pool.shutdown()
    per = select(results, strata, quota)
    chosen = sorted((r for s in strata for r in per[s]["chosen"]), key=lambda r: r["index"])
    return {"results": results, "per_stratum": per, "chosen": chosen, "strata": strata, "quota": quota,
            "attempts_max": attempts_max, "waves": list(waves), "waves_run": wave,
            "wall_s": round(time.time() - t0, 1)}


# --------------------------------------------------------------------------- outputs

def to_bytes(incidents: list[dict[str, Any]]) -> bytes:
    return "".join(inc.canonical_line(x) + "\n" for x in incidents).encode("utf-8")


def log_bytes(results: dict[int, dict[str, Any]], chosen: list[dict[str, Any]]) -> bytes:
    picked = {r["index"] for r in chosen}
    rows = [{k: r[k] for k in ("index", "stratum", "attempt", "seed", "kept", "reason", "latency_ms")}
            | {"chosen": r["index"] in picked} for _, r in sorted(results.items())]
    return "".join(inc.canonical_line(x) + "\n" for x in rows).encode("utf-8")


def meta_for(gen: dict[str, Any], digest: str, log_sha: str, code_sha: str, res: dict[str, Any] | None,
             in_job_exclusion: dict[str, Any] | None, smoke: bool,
             replay_check: dict[str, Any] | None = None) -> dict[str, Any]:
    from t3.evals.twin_backend import BACKEND, twin_version

    per = gen["per_stratum"]
    drops: dict[str, int] = {}
    for s in gen["strata"]:
        for k, v in per[s]["drops"].items():
            drops[k] = drops.get(k, 0) + v
    evaluated = sorted(int(r["seed"]) for r in gen["results"].values())
    chosen_seeds = [int(r["seed"]) for r in gen["chosen"]]
    return {
        "version": VERSION, "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "smoke": bool(smoke), "sha256": digest, "file": SUITE_NAME, "n_incidents": len(gen["chosen"]),
        "n_target": gen["quota"] * len(gen["strata"]), "quota": gen["quota"], "n_strata": len(gen["strata"]),
        "attempts_max": gen["attempts_max"], "waves": gen["waves"], "waves_run": gen["waves_run"],
        "candidates_evaluated": len(gen["results"]), "candidates_kept": sum(1 for r in gen["results"].values()
                                                                           if r["kept"]),
        "drops": drops, "short_strata": [stratum_info(s) | {"chosen": len(per[s]["chosen"])}
                                         for s in gen["strata"] if not per[s]["filled"]],
        "per_stratum": [stratum_info(s) | {k: per[s][k] for k in ("evaluated", "kept", "filled", "drops")}
                        | {"chosen": len(per[s]["chosen"]),
                           "last_chosen_attempt": per[s]["chosen"][-1]["attempt"] if per[s]["chosen"] else None}
                        for s in gen["strata"]],
        "seed_block": [SEED_LO, SEED_HI], "seeds_evaluated": [evaluated[0], evaluated[-1]] if evaluated else None,
        "seeds_chosen": [min(chosen_seeds), max(chosen_seeds)] if chosen_seeds else None,
        "split_of_chosen": {"dev": sum(1 for s in chosen_seeds if inc.split_of(s) == "dev"),
                            "test": sum(1 for s in chosen_seeds if inc.split_of(s) == "test")},
        "candidate_log": {"file": LOG_NAME, "sha256": log_sha},
        "reservation": None if res is None else {"file": RESERVATION_REL,
                                                 "generator_input_sha256": res["generator_input_sha256"],
                                                 "exclusion_clean": res["exclusion"]["clean"],
                                                 "literal_scan": res["exclusion"].get("literal_scan"),
                                                 "spec_hash_overlap": res["spec_hash_overlap"]},
        "in_job_exclusion": in_job_exclusion, "keep_rule": GRADER, "tier": TIER,
        "twin_backend": BACKEND, "twin_version": twin_version(), "code_sha": code_sha,
        "python": sys.version.split()[0], "platform": f"{platform.system()}-{platform.machine()}",
        "wall_s": gen["wall_s"], "host": host_info(), "replay_check": replay_check,
        "supersedes": {"version": "t1-remediation-heldout-v1", "file": V1_SUITE_NAME,
                       "sha256": "8e5a56ce0e23878fe3ff6058d4a3a8f7df13f69e98115aae1b764db15b9b6c0e",
                       "reason": "its first incident did not replay in the first evaluation (ReplayMismatch, before "
                                 "any result); regenerated from the same reservation, every process pinned to one "
                                 "torch thread, every incident self-replayed before writing"},
        "note": "the twin is a float32 GP: frozen KPIs replay to 1e-6 only where they were frozen; evaluate this set "
                "on the same platform and instance type (meta.host), and the evaluation's fail-fast check replays "
                "it before it starts"}


class ReplayGuard(RuntimeError):
    """The chosen incidents do not replay through the evaluation's own check: nothing is written."""


def write_outputs(gen: dict[str, Any], dest: Path, code_sha: str, res: dict[str, Any] | None,
                  in_job_exclusion: dict[str, Any] | None, smoke: bool,
                  replay_check: dict[str, Any] | None) -> dict[str, Any]:
    """Freeze the suite, but only if every chosen incident replayed (``self_replay``) on this host."""
    incidents = [r["incident"] for r in gen["chosen"]]
    if replay_check is None or not replay_check.get("ok") or replay_check.get("n") != len(incidents):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / REPLAY_NAME).write_text(json.dumps(replay_check, indent=1, sort_keys=True, default=str) + "\n")
        raise ReplayGuard(f"refusing to write {SUITE_NAME}: the self-replay check did not pass for all "
                          f"{len(incidents)} incidents ({None if replay_check is None else replay_check.get('mismatches')} "
                          f"mismatches); see {REPLAY_NAME}")
    data = to_bytes(incidents)
    digest = hashlib.sha256(data).hexdigest()
    lb = log_bytes(gen["results"], gen["chosen"])
    log_sha = hashlib.sha256(lb).hexdigest()
    meta = meta_for(gen, digest, log_sha, code_sha, res, in_job_exclusion, smoke, replay_check)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / REPLAY_NAME).write_text(json.dumps(replay_check, indent=1, sort_keys=True) + "\n")
    (dest / SUITE_NAME).write_bytes(data)
    (dest / HASH_PATH.name).write_text(f"{digest}  {SUITE_NAME}\n")
    (dest / META_NAME).write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    (dest / LOG_NAME).write_bytes(lb)
    return meta


def load_heldout(path: Path = SUITE_PATH, verify_hash: bool = True) -> tuple[list[dict[str, Any]], str]:
    """The frozen held-out incidents and the sha256 of the file's bytes (checked against the sidecar)."""
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if verify_hash:
        side = Path(path).with_suffix(".sha256")
        want = side.read_text().split()[0]
        if digest != want:
            raise RuntimeError(f"{path} sha256 {digest} != frozen {want}: the held-out set changed")
    return [json.loads(x) for x in data.decode("utf-8").splitlines() if x.strip()], digest


# --------------------------------------------------------------------------- evidence

def declaration(code_sha: str, run_id: str | None = None, smoke: bool = False,
                n_target: int = N_TARGET, res: dict[str, Any] | None = None) -> dict[str, Any]:
    from t3.evals.twin_backend import BACKEND, twin_version

    res = res if res is not None else load_reservation()
    d: dict[str, Any] = {
        "unit": UNIT, "track": "track-1",
        "target": {"metric": "n_items", "op": "==", "value": n_target, "unit": "count"},
        "n_items": n_target,
        "test_set_sha256": res["generator_input_sha256"],
        "model_id": MODEL_ID, "grader": GRADER, "synthetic": bool(smoke),
        "twin": {"kind": "maveric-inprocess", "version": twin_version(), "scenario": "remediation-heldout",
                 "seed": SEED_BASE,
                 "config": {"backend": BACKEND, "version": VERSION, "reservation": RESERVATION_REL,
                            "seed_block": [SEED_LO, SEED_HI], "quota": QUOTA, "strata": N_STRATA,
                            "strata_rule": "7 templates x 4 base cells x 2 presets; stratum s = (template*4 + "
                                           "base_cell)*2 + preset; candidate k: stratum k % 56, attempt k // 56, "
                                           "seed SEED_BASE + k",
                            "attempts_max": ATTEMPTS_MAX, "waves": list(WAVES), "tier": TIER,
                            "selection": "per stratum the first 18 kept candidates in attempt order (independent "
                                         "of workers and wave sizes)",
                            "test_set_sha256_is": "sha256 of every reserved candidate spec, one canonical line each "
                                                  "(the generator input); the frozen set's own sha256 is in the "
                                                  "result",
                            "purpose": "primary held-out set for remediation policies (frozen before any policy "
                                       "is evaluated on it)",
                            "executed": "SageMaker training job started by tools/sm/run.py (x86_64 CPU image); "
                                        "in-process twin, no network twin, no model",
                            "freeze": "outputs heldout-v1.jsonl + .sha256 + -meta.json; the lead commits them to "
                                      "t1/rl/remediation/suite/",
                            "smoke": bool(smoke)}}}
    if run_id:
        d["run_id"] = run_id
    return d


def item_for(r: dict[str, Any]) -> dict[str, Any]:
    from t3.evals.twin_backend import BACKEND

    i = r["incident"]
    spec = spec_for(r["index"])
    return {"item_id": f"heldout-{r['seed']}",
            "input_sha256": hashlib.sha256(inc.canonical_line(spec).encode()).hexdigest(),
            "output_sha256": hashlib.sha256(inc.canonical_line(i).encode()).hexdigest(),
            "score": None, "passed": None, "grader": GRADER, "backend": BACKEND, "latency_ms": float(r["latency_ms"]),
            "extra": {"incident_id": i["incident_id"], "stratum": r["stratum"], "attempt": r["attempt"],
                      "index": r["index"], "template": i["template"], "preset": i["preset"],
                      "base_cell": i["base_cell"], "fault_cells": i["fault_cells"], "split_of_seed": i["split_of_seed"],
                      "material_kpis": i["material_kpis"]}}


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id")
    ap.add_argument("--evidence-dir", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--write-reservation", action="store_true", help="write heldout-reservation.json (local, pre-job)")
    ap.add_argument("--print-declaration", action="store_true")
    ap.add_argument("--declare", action="store_true", help="also write the declaration (local runs only)")
    ap.add_argument("--check", type=Path, help="regenerate (same platform) and compare with this frozen file")
    ap.add_argument("--smoke", action="store_true", help="label everything synthetic smoke (local tests)")
    ap.add_argument("--strata", type=int, help="first N strata only (smoke)")
    ap.add_argument("--quota", type=int, help="incidents per stratum (smoke)")
    ap.add_argument("--max-candidates", type=int, help=f"hard cap on evaluated candidates (smoke, <= {SMOKE_MAX})")
    a = ap.parse_args(argv)

    if a.write_reservation:
        res = reservation()
        RESERVATION_PATH.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
        print(json.dumps({"written": RESERVATION_REL, "clean": res["exclusion"]["clean"],
                          "spec_hash_overlap": res["spec_hash_overlap"], "seed_block": res["seed_block"],
                          "generator_input_sha256": res["generator_input_sha256"],
                          "literal_scan": {k: res["exclusion"]["literal_scan"][k] for k in ("files_scanned", "n_hits")}},
                         indent=1))
        return 0 if res["exclusion"]["clean"] and not any(res["spec_hash_overlap"].values()) else 1

    from t3.evals.evshim import Evidence, code_sha

    _init()  # the main process is pinned before any twin work, like every worker
    sha = code_sha()
    smoke_limits = a.strata is not None or a.quota is not None or a.max_candidates is not None
    if smoke_limits and not a.smoke:
        ap.error("--strata, --quota and --max-candidates are for local smoke runs (with --smoke)")
    if a.smoke and (a.max_candidates is None or a.max_candidates > SMOKE_MAX):
        ap.error(f"a smoke run needs --max-candidates <= {SMOKE_MAX}")
    strata = list(range(a.strata)) if a.strata else None
    quota = a.quota or QUOTA
    n_target = quota * (len(strata) if strata else N_STRATA)
    res = load_reservation()
    if a.print_declaration:
        print(json.dumps(declaration(sha, smoke=a.smoke, n_target=n_target, res=res), indent=2, sort_keys=True))
        return 0
    check_reservation(res)

    if a.check:
        from t3.evals.graders import load_targets

        frozen = a.check.read_bytes()
        incs = [json.loads(x) for x in frozen.decode("utf-8").splitlines() if x.strip()]
        rc = self_replay(incs, load_targets(), a.workers)
        gen = generate(workers=a.workers)
        fresh = to_bytes([r["incident"] for r in gen["chosen"]])
        same = fresh == frozen
        print(json.dumps({"matches_frozen": same, "fresh_sha256": hashlib.sha256(fresh).hexdigest(),
                          "frozen_sha256": hashlib.sha256(frozen).hexdigest(), "replay_check": rc,
                          "host": host_info()}))
        return 0 if same and rc["ok"] else 1

    if not (a.run_id and a.evidence_dir and a.output_dir):
        ap.error("--run-id, --evidence-dir and --output-dir are required to generate")
    ev = Evidence(a.evidence_dir)
    if a.declare:  # a local run; in a job tools/sm/run.py wrote it (with code_sha and region) before the job
        ev.write_declared({**declaration(sha, a.run_id, a.smoke, n_target, res), "code_sha": sha, "region": "us-east-2"})
    in_job = exclusion_checks(all_specs(), scan=False)  # the literal scan is in the committed reservation
    if not in_job["clean"]:
        raise SystemExit(f"seed exclusion failed in the job: {json.dumps(in_job['explicit_collisions'])}")
    gen = generate(workers=a.workers, quota=quota, strata=strata, max_candidates=a.max_candidates)
    from t3.evals.graders import load_targets

    rc = self_replay([r["incident"] for r in gen["chosen"]], load_targets(), a.workers)
    print(f"[heldout] self-replay: {rc['n']} incidents, {rc['mismatches']} mismatches, worker torch threads "
          f"{rc['worker_torch_threads']}, {rc['wall_s']} s", flush=True)
    meta = write_outputs(gen, a.output_dir, sha, res, {k: in_job[k] for k in ("ranges_overlapping",
                                                                               "explicit_collisions", "clean")},
                         a.smoke, rc)  # raises ReplayGuard (nothing written, no items, no result) unless all replay
    items = [item_for(r) for r in gen["chosen"]]
    for it in items:
        it["synthetic"] = bool(a.smoke)
    if items:
        ev.append_records(a.run_id, "item", items)
    n = len(items)
    result = {
        "run_id": a.run_id, "unit": UNIT, "metric": "n_items", "value": n,
        "target": {"metric": "n_items", "op": "==", "value": n_target, "unit": "count"},
        "passed": n == n_target, "n_items": n, "synthetic": bool(a.smoke),
        "checker": {"script": "t1/rl/remediation/heldout.py", "code_sha": sha},
        "details": {"heldout_sha256": meta["sha256"], "candidate_log_sha256": meta["candidate_log"]["sha256"],
                    "candidates_evaluated": meta["candidates_evaluated"], "candidates_kept": meta["candidates_kept"],
                    "drops": meta["drops"], "short_strata": meta["short_strata"], "platform": meta["platform"],
                    "python": meta["python"], "twin_version": meta["twin_version"], "wall_s": meta["wall_s"],
                    "host": meta["host"], "replay_check": {k: rc[k] for k in ("n", "mismatches", "ok",
                                                                             "worker_torch_threads", "path")},
                    "outputs": [SUITE_NAME, HASH_PATH.name, META_NAME, LOG_NAME, REPLAY_NAME]},
    }
    ev.write_result(result)
    print(json.dumps({"run_id": a.run_id, "n_items": n, "target": n_target, "passed": n == n_target,
                      "sha256": meta["sha256"], "evaluated": meta["candidates_evaluated"],
                      "kept": meta["candidates_kept"], "short_strata": len(meta["short_strata"]),
                      "wall_s": meta["wall_s"], "platform": meta["platform"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
