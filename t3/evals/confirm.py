"""The confirmation set: does the lesson gain hold on fresh faults nobody has seen? Declared before it runs.

The self-improvement unit (self_improve.py) measures the lesson gain on the 18 masked test incidents, which
earlier runs have already looked at (the judged swarm run, the offline calibration and the offline
self-improvement trial). This unit runs ONCE on 18 brand-new masked incidents (``suite/incidents-confirm.jsonl``),
frozen before any run, on seeds that no file in the repo uses, with the lessons that the main run's practice
phase learned:

1. LESSONS_ON  - the nearest verified lessons appended to the ticket (retrieval only), the 18 incidents.
2. LESSONS_OFF - the same 18 incidents, no lessons.

ON runs first, so any hidden carry-over between runs (agent-side memory, warm caches) can only favour OFF:
the test is conservative. Nothing is learned during this run: the lesson store is a read-only snapshot
(``FrozenLessons``; its sha256 and counts go in the declaration and a copy goes to
``outputs/<run_id>/lessons.jsonl``). Every incident is graded by the twin (``lessons.verify``), never by a
model. Primary: paired bootstrap (paired-bootstrap-v1, targets.json's bootstrap block) of recovery
ON - OFF by incident id. Declared target: ci_low > 0.

The suite, frozen once with ``--freeze`` and committed before any run (so the run's code_sha fixes it), is
MATCHED to the 18 masked test incidents, one fresh incident per test incident:

- the SAME candidate grid as the test set's generator (``incidents.candidate``): each test incident came
  from grid index i (seed 4100 + i); its twin here is grid index j = i + 112 k (112 = the grid's full
  period: preset, tier, template and base cell all repeat), so it has the same preset, masked tier, fault
  template and faulted cell(s), on a fresh seed ``CONFIRM_SEED_BASE + j`` (fresh UE placement and fresh
  commissioning offsets);
- the SAME filters (``incidents.build``: the fault must be material at the grade step, and the safety filter
  must clip neither the commissioning nor the fault); k is the first of 0, 1, 2, ... that passes;
- its own split ("confirm"): ``lessons.TestSplitGuard`` refuses to learn from it.

The selection uses design factors only (grid position and the build filters), never a KPI of a fix or any
agent's output. A first, unmatched hash-pick draw was generated on the laptop and replaced before it was
committed or shown to any agent: its preset mix (15 official / 3 viavi-like) departed from the test set's
(6 / 12). It is recorded in the meta file (``replaced_draw``).

    python -m t3.evals.confirm --freeze                    # once: write the suite files (never overwrites)
    python -m t3.evals.confirm --check                     # regenerate on this platform, compare the bytes
    python -m t3.evals.confirm --dry-run [--lessons PATH]  # plan, Gateway case count, wall time; no calls
    python -m t3.evals.confirm --evidence-dir DIR --lessons PATH       # offline stand-in (synthetic)
    python -m t3.evals.confirm --live --run-id t3.evals.confirm-<UTC> --lessons PATH --workers 3   # spends
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from t3.evals import incidents as inc_mod
from t3.evals import lessons as L
from t3.evals import self_improve as S
from t3.evals.graders import load_targets
from t3.evals.stats import METHOD, paired_bootstrap
from t3.evals.twin_backend import BACKEND, twin_version

UNIT = "t3.evals.confirm"
SCRIPT = "t3/evals/confirm.py"
TARGET = {"metric": "gain_ci_low_on_vs_off", "op": ">", "value": 0.0, "unit": "recovery"}
CONDITIONS = ("lessons_on", "lessons_off")  # in this order: hidden carry-over can only favour OFF

CONFIRM_VERSION = "t3-incidents-confirm-v1"
CONFIRM_SEED_BASE = 990_000
GRID_PERIOD = 112  # incidents.candidate repeats preset (2), tier (4), template (28) and base cell (112)
MAX_K = 12         # fresh grid periods tried per test incident before giving up
REPLACED_DRAW = {
    "rule": "hash pick v0: every masked grid candidate on seeds 990000 + i (i < 84), the incidents.build filters, "
            "then the first 18 kept in sha256('t3-evals-confirm-v1:<seed>') order",
    "sha256": "313548bc6568be9f818a1e8283ba61ec329f3d73713e92d6ad08f5cf53b8b4c8",
    "kept": 28, "built": 42,
    "presets": {"official": 15, "viavi-like": 3},
    "templates": {"power-3+tilt+3": 2, "power-4": 4, "power-6": 3, "power-6+tilt+5": 4, "tilt+4": 2, "tilt+5": 1,
                  "two-cell:power-6|tilt+5": 2},
    "why_replaced": "its preset mix departed from the 18 masked test incidents (6 official / 12 viavi-like): 11 of 21 "
                    "viavi-like candidates were immaterial on the fresh seeds against 3 of 21 official. Replaced by "
                    "the matched-grid rule so the confirmation set has the test set's composition.",
    "committed": False, "shown_to_any_agent": False, "run_on": "nothing (no proposal, no grading of any fix)",
}
CONFIRM_PATH = L.CONFIRM_PATH
CONFIRM_HASH_PATH = inc_mod.SUITE_DIR / "incidents-confirm.sha256"
CONFIRM_META_PATH = inc_mod.SUITE_DIR / "incidents-confirm-meta.json"
SUITE_REL = "t3/evals/suite/incidents-confirm.jsonl"
PER_CASE_S = 70.0
REPO = Path(__file__).resolve().parents[2]

# Every seed range used anywhere in the repo (grep of t1, t2, t3, silos, twin; 3 Nov). All lie below
# CONFIRM_SEED_BASE; the explicit seed files and a literal scan of the repo are checked at freeze time too.
KNOWN_SEED_RANGES: list[dict[str, Any]] = [
    {"what": "Track 1 SFT execution data", "lo": 0, "hi": 899_999,
     "source": "t1/sft/execgen.py: h(seed key) % SEED_LIMIT (900000)"},
    {"what": "Track 1 execution eval (reserved)", "lo": 900_000, "hi": 989_999,
     "source": "t1/evals/gen_execution.py: SEED_BASE 900000 + h % 90000"},
    {"what": "RL evaluation scenarios and the PPO seed", "lo": 42, "hi": 404,
     "source": "RL hand-in evaluation.json (101-105, 200-204, 300-304, 400-404); PPO seed 42"},
    {"what": "trust gate jobs, ladders and demo", "lo": 99, "hi": 4019,
     "source": "t3/trust: seed 99, 1000+s, 2000-2019, 3000-3019, 4000-4019"},
    {"what": "t3 incident suite candidates (test + dev)", "lo": 4100, "hi": 4183,
     "source": "t3/evals/incidents.py: SEED_BASE 4100, N_CANDIDATES 84"},
    {"what": "Live Fix fault cache and runs", "lo": 4161, "hi": 7099,
     "source": "t3/evals/faults.py DEV_SEED_BASE 7000 (cache 7002-7041); Live Fix runs on 4161 and 7002"},
    {"what": "Track 2 episodes", "lo": 10_000, "hi": 99_999, "source": "t2/gen.py: BASE_SEED 10000 + episode"},
    {"what": "silo seeds", "lo": 100_000, "hi": 499_999, "source": "silos/config.yaml: seed_base 100000-400000"},
]
LIVEFIX_RUN_SEEDS = {4161, 7002}
SCAN_SUFFIXES = {".py", ".json", ".jsonl", ".yaml", ".yml", ".md", ".txt", ".csv", ".toml", ".ts", ".tsx", ".html"}
SCAN_SKIP_DIRS = {".git", ".claude", "node_modules", "dist", ".venv", "venv", "__pycache__", ".pytest_cache"}


# --------------------------------------------------------------------------- the suite

def confirm_candidate(i: int) -> dict[str, Any]:
    """``incidents.candidate(i)`` on the fresh seed CONFIRM_SEED_BASE + i (its commissioning follows the seed)."""
    spec = inc_mod.candidate(i)
    seed = CONFIRM_SEED_BASE + i
    return {**spec, "seed": seed, "commissioning": inc_mod.commissioning(seed, spec["tier"])}


def grid_cell(spec: dict[str, Any]) -> tuple:
    """What a matched incident must share with its test incident: preset, tier, template and the fault."""
    return (spec["preset"], spec["tier"], spec["template"],
            tuple((f["cell"], f["d_power_db"], f["d_tilt_deg"]) for f in spec["fault"]))


def masked_test_incidents() -> list[dict[str, Any]]:
    return sorted((i for i in _jsonl(inc_mod.TEST_PATH) if i.get("tier") == "masked"), key=lambda i: int(i["seed"]))


def generate_confirm(targets: dict[str, Any] | None = None, max_k: int = MAX_K) -> dict[str, Any]:
    """One fresh incident per masked test incident: grid index j = i + GRID_PERIOD * k for the first k whose
    candidate passes ``incidents.build``. Deterministic on one platform."""
    targets = targets or load_targets()
    considered: list[dict[str, Any]] = []
    chosen: list[dict[str, Any]] = []
    matched: dict[str, str] = {}
    for t in masked_test_incidents():
        i = int(t["seed"]) - inc_mod.SEED_BASE
        if grid_cell(inc_mod.candidate(i)) != grid_cell(t):
            raise RuntimeError(f"{t['incident_id']}: not grid index {i} of incidents.candidate")
        for k in range(max_k):
            j = i + GRID_PERIOD * k
            spec = confirm_candidate(j)
            if grid_cell(spec) != grid_cell(t):
                raise RuntimeError(f"grid index {j} does not repeat index {i}")
            inc, why = inc_mod.build(spec, targets)
            considered.append({"matches": t["incident_id"], "index": j, "k": k, "seed": spec["seed"],
                               "preset": spec["preset"], "template": spec["template"],
                               "fault_cells": sorted({f["cell"] for f in spec["fault"]}),
                               "kept": inc is not None, "reason": why})
            if inc is not None:
                chosen.append({**inc, "suite": CONFIRM_VERSION, "split": "confirm"})
                matched[inc["incident_id"]] = t["incident_id"]
                break
        else:
            raise RuntimeError(f"{t['incident_id']}: no material fresh twin in {max_k} grid periods")
    chosen.sort(key=lambda x: int(x["seed"]))
    return {"incidents": chosen, "considered": considered, "matched_to": matched}


def load_confirm(path: Path = CONFIRM_PATH, verify_hash: bool = True) -> tuple[list[dict[str, Any]], str]:
    """The frozen confirmation incidents and the sha256 of the file's bytes (checked against the sidecar)."""
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if verify_hash and Path(path).resolve() == CONFIRM_PATH.resolve():
        want = CONFIRM_HASH_PATH.read_text().split()[0]
        if digest != want:
            raise RuntimeError(f"{path} sha256 {digest} != frozen {want}: the confirmation set changed")
    return [json.loads(x) for x in data.decode("utf-8").splitlines() if x.strip()], digest


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def _seeds_in(obj: Any) -> set[int]:
    """Every integer stored under a key named "seed" anywhere in a JSON value."""
    found: set[int] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "seed" and isinstance(v, int):
                found.add(v)
            found |= _seeds_in(v)
    elif isinstance(obj, list):
        for v in obj:
            found |= _seeds_in(v)
    return found


def explicit_seed_sets() -> dict[str, set[int]]:
    """The seeds the repo's suite files name."""
    out = {"test": {int(i["seed"]) for i in _jsonl(inc_mod.TEST_PATH)},
           "dev": {int(i["seed"]) for i in _jsonl(inc_mod.DEV_PATH)},
           "candidates": {int(c["seed"]) for c in _jsonl(inc_mod.CANDIDATES_PATH)},
           "livefix_runs": set(LIVEFIX_RUN_SEEDS)}
    lf = inc_mod.SUITE_DIR / "livefix-faults.json"
    out["livefix_faults"] = _seeds_in(json.loads(lf.read_text())) if lf.exists() else set()
    return out


def literal_scan(seeds: list[int], root: Path = REPO, exclude: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Look for each seed as a whole number in every text file of the repo (outside the confirmation files)."""
    rx = re.compile(r"(?<![0-9])(" + "|".join(str(s) for s in sorted(seeds)) + r")(?![0-9])")
    skip = {p.resolve() for p in exclude}
    hits, scanned = [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SCAN_SKIP_DIRS]
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix not in SCAN_SUFFIXES or p.resolve() in skip:
                continue
            try:
                text = p.read_text(errors="ignore")
            except OSError:
                continue
            scanned += 1
            for m in rx.finditer(text):
                hits.append({"file": str(p.relative_to(root)), "seed": int(m.group(1))})
    return {"files_scanned": scanned, "hits": hits}


def overlap_checks(chosen: list[dict[str, Any]]) -> dict[str, Any]:
    """Seeds, ids and spec hashes of the chosen incidents vs the frozen test, dev and candidate files."""
    sets = {"test": _jsonl(inc_mod.TEST_PATH), "dev": _jsonl(inc_mod.DEV_PATH),
            "candidates": _jsonl(inc_mod.CANDIDATES_PATH)}
    seeds = {int(i["seed"]) for i in chosen}
    ids = {i["incident_id"] for i in chosen}
    hashes = {L._spec_hash(i) for i in chosen}
    out = {}
    for name, rows in sets.items():
        out[name] = {"n": len(rows),
                     "seed_overlap": len(seeds & {int(r["seed"]) for r in rows}),
                     "id_overlap": len(ids & {r["incident_id"] for r in rows if "incident_id" in r}),
                     "spec_hash_overlap": len(hashes & {L._spec_hash(r) for r in rows})}
    out["total"] = sum(v["seed_overlap"] + v["id_overlap"] + v["spec_hash_overlap"] for v in out.values()
                       if isinstance(v, dict))
    return out


def mix(incidents: list[dict[str, Any]]) -> dict[str, Any]:
    return {"templates": dict(sorted(collections.Counter(i["template"] for i in incidents).items())),
            "presets": dict(sorted(collections.Counter(i["preset"] for i in incidents).items())),
            "fault_cells": dict(sorted(collections.Counter(
                ",".join(str(c) for c in i["fault_cells"]) for i in incidents).items()))}


def exclusion_checks(chosen: list[dict[str, Any]]) -> dict[str, Any]:
    seeds = sorted(int(i["seed"]) for i in chosen)
    explicit = explicit_seed_sets()
    collisions = {k: sorted(set(seeds) & v) for k, v in explicit.items()}
    in_ranges = [{"seed": s, "range": r["what"]} for s in seeds for r in KNOWN_SEED_RANGES if r["lo"] <= s <= r["hi"]]
    scan = literal_scan(seeds, exclude=(CONFIRM_PATH, CONFIRM_HASH_PATH, CONFIRM_META_PATH))
    return {"known_ranges": KNOWN_SEED_RANGES, "seeds_in_known_ranges": in_ranges,
            "explicit_sets": {k: len(v) for k, v in explicit.items()},
            "explicit_collisions": {k: v for k, v in collisions.items() if v},
            "literal_scan": {"files_scanned": scan["files_scanned"], "hits": scan["hits"]},
            "clean": not in_ranges and not any(collisions.values()) and not scan["hits"]}


def suite_bytes(incidents: list[dict[str, Any]]) -> bytes:
    return inc_mod.to_bytes(incidents)


def freeze(dest: Path = inc_mod.SUITE_DIR, targets: dict[str, Any] | None = None, write: bool = True
           ) -> dict[str, Any]:
    """Generate the set, run the exclusion and overlap checks, and write the three files (never overwrites)."""
    if BACKEND != "maveric-inprocess":
        raise SystemExit(f"refusing to freeze on twin backend {BACKEND!r}: the real twin (twin.replay) is required")
    targets = targets or load_targets()
    gen = generate_confirm(targets)
    chosen = gen["incidents"]
    excl, over = exclusion_checks(chosen), overlap_checks(chosen)
    if not excl["clean"] or over["total"]:
        raise SystemExit(f"refusing to freeze: exclusion {json.dumps(excl)[:400]} overlap {over}")
    data = suite_bytes(chosen)
    digest = hashlib.sha256(data).hexdigest()
    test_masked = masked_test_incidents()
    meta = {
        "suite": CONFIRM_VERSION, "split": "confirm", "unit": UNIT,
        "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": digest, "n": len(chosen), "tier": "masked",
        "generator": {"procedure": "matched grid: for each masked test incident (grid index i of "
                                   "incidents.candidate), grid index j = i + GRID_PERIOD * k on the fresh seed "
                                   "CONFIRM_SEED_BASE + j, for the first k whose candidate passes incidents.build "
                                   "(material at the grade step, nothing clipped); same preset, tier, template and "
                                   "faulted cell(s) as its test incident",
                      "seed_base": CONFIRM_SEED_BASE, "grid_period": GRID_PERIOD, "max_k": MAX_K,
                      "candidates_built": len(gen["considered"]),
                      "dropped": [c for c in gen["considered"] if not c["kept"]],
                      "k_used": dict(sorted(collections.Counter(c["k"] for c in gen["considered"] if c["kept"]).items())),
                      "matched_to": gen["matched_to"],
                      "test_suite_version": inc_mod.SUITE_VERSION, "targets_episode": targets["episode"]},
        "replaced_draw": REPLACED_DRAW,
        "seeds": [int(i["seed"]) for i in chosen],
        "seed_range": [min(int(i["seed"]) for i in chosen), max(int(i["seed"]) for i in chosen)],
        "split_of_labels": {str(i["seed"]): inc_mod.split_of(int(i["seed"])) for i in chosen},
        "split_of_note": "incidents.split_of labels every seed test or dev by its own salt; these incidents are "
                         "their own split ('confirm') whatever it says, and lessons.TestSplitGuard refuses them",
        "mix": mix(chosen), "test_masked_mix": mix(test_masked),
        "exclusion": excl, "overlap": over,
        "twin_backend": BACKEND, "twin_version": twin_version(), "python": sys.version.split()[0],
        "platform": f"{platform.system()}-{platform.machine()}",
        "note": "frozen on this platform; the twin is a float32 GP, so frozen KPIs replay to 1e-6 only here. "
                "Grading (lessons.verify) rebuilds each incident from its spec on the grading machine. The "
                "tickets state the frozen pre/post-fault KPIs. No agent or model has seen these incidents.",
    }
    if write:
        dest.mkdir(parents=True, exist_ok=True)
        for p in (dest / CONFIRM_PATH.name, dest / CONFIRM_HASH_PATH.name, dest / CONFIRM_META_PATH.name):
            if p.exists():
                raise SystemExit(f"{p} exists: the confirmation set is frozen once and never overwritten")
        with open(dest / CONFIRM_PATH.name, "xb") as f:
            f.write(data)
        with open(dest / CONFIRM_HASH_PATH.name, "x") as f:
            f.write(f"{digest}  {CONFIRM_PATH.name}\n")
        with open(dest / CONFIRM_META_PATH.name, "x") as f:
            f.write(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return {"meta": meta, "bytes": data, "incidents": chosen}


# --------------------------------------------------------------------------- the lessons (read-only)

class FrozenLessons(L.LessonStore):
    """A read-only snapshot of a practice run's lesson store: retrieve only. ``learn`` raises, and the
    store has no path, so nothing is ever written back."""

    def __init__(self, path: str | Path, guard: L.TestSplitGuard | None = None) -> None:
        super().__init__(None, guard=guard)
        self.source = Path(path)
        self.raw = self.source.read_bytes()
        self.sha256 = hashlib.sha256(self.raw).hexdigest()
        self.records = [json.loads(x) for x in self.raw.decode("utf-8").splitlines() if x.strip()]

    def learn(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("the confirmation run is read-only: it never learns")

    def _append(self, rec: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("the confirmation run is read-only: it never learns")

    def digest(self) -> str:
        """sha256 of the in-memory records (to show they did not change during the run)."""
        return hashlib.sha256("".join(inc_mod.canonical_line(r) + "\n" for r in self.records).encode()).hexdigest()

    def info(self, source_run_id: str | None = None) -> dict[str, Any]:
        return {"file": self.source.name, "sha256": self.sha256, "records": len(self.records),
                "verified": len(self.lessons), "rejected": len(self.rejected),
                "sources": dict(sorted(collections.Counter(str(r.get("source")) for r in self.records).items())),
                "source_run_id": source_run_id or source_run_of(self.source),
                "read_only": True, "min_similarity": S.LESSON_MIN_SIM, "k": 3, "version": L.VERSION}


def source_run_of(path: Path) -> str | None:
    """The practice run a lesson file came from, from its name (``<run>.lessons.jsonl`` locally, or
    ``outputs/<run>/lessons.jsonl`` as published)."""
    p = Path(path)
    if p.name.endswith(".lessons.jsonl"):
        return p.name[: -len(".lessons.jsonl")]
    if p.name == "lessons.jsonl" and p.parent.name:
        return p.parent.name
    return None


# --------------------------------------------------------------------------- the unit

class Confirm:
    """Runs LESSONS_ON then LESSONS_OFF with a ``propose(condition, incidents, store)`` backend (the same
    shape as self_improve's), verifies every incident on the twin, and writes the evidence."""

    def __init__(self, *, run_id: str, ev: Any, propose: Callable[..., dict[str, tuple[list[str], float, list[str]]]],
                 model_id: str, synthetic: bool, lessons: FrozenLessons, targets: dict[str, Any] | None = None,
                 log: Callable[[str], None] = print, mode: str = "offline", workers: int = 1, resume: bool = False,
                 checkpoint_dir: Path | None = None, copy_lessons: Callable[[str, bytes], str | None] | None = None,
                 lessons_run_id: str | None = None) -> None:
        self.run_id, self.ev, self.propose, self.model_id = run_id, ev, propose, model_id
        self.synthetic, self.lessons, self.targets, self.log = synthetic, lessons, targets or load_targets(), log
        self.mode, self.workers, self.resume = mode, max(1, int(workers)), resume
        self.copy_lessons = copy_lessons
        self.lessons_info = {**lessons.info(lessons_run_id), "copied_to": f"outputs/{run_id}/lessons.jsonl"}
        self._marker = Path(checkpoint_dir) / f"{run_id}.phases-written.json" if checkpoint_dir else None

    def declaration(self, incidents: list[dict[str, Any]], digest: str) -> dict[str, Any]:
        steps = self.targets["episode"]["operator_steps"]
        return {
            "run_id": self.run_id, "synthetic": self.synthetic, "unit": UNIT, "track": "track-3",
            "target": dict(TARGET), "n_items": len(incidents) * len(CONDITIONS),
            "n_cases": len(incidents) * len(CONDITIONS) * steps, "test_set_sha256": digest,
            "subset": "split=confirm tier=masked", "model_id": self.model_id,
            "grader": f"{L.VERIFIER}; primary = {METHOD} of recovery (lessons_on - lessons_off) paired by incident id",
            "code_sha": S._code_sha(), "region": "us-east-2",
            "twin": {"kind": "maveric-inprocess", "version": twin_version(), "scenario": "incident-suite",
                     "seed": min(int(i["seed"]) for i in incidents),
                     "config": {"backend": BACKEND, "suite": SUITE_REL, "suite_version": CONFIRM_VERSION,
                                "conditions": list(CONDITIONS),
                                "order": "lessons_on first, then lessons_off: any hidden carry-over between runs "
                                         "can only favour OFF (a conservative test)",
                                "lessons": self.lessons_info,
                                "learning": "none: the lesson store is a read-only snapshot (FrozenLessons)",
                                "fresh": "frozen before any run on seeds no repo file uses (incidents-confirm-meta."
                                         "json: exclusion and overlap checks); run once, never tuned on",
                                "graded_on": "this platform's rebuild of each incident from its spec; healthy and "
                                             "unfixed counterfactuals rebuilt with it",
                                "mode": self.mode,
                                "ticket": S.TICKET if self.mode == "live" else "n/a (offline stand-in has no ticket)",
                                "workers": self.workers}},
        }

    def _items(self, cond: str, incidents: list[dict[str, Any]], got: dict[str, Any]) -> list[dict[str, Any]]:
        items = []
        for inc in incidents:
            props, wall_ms, used = got.get(inc["incident_id"], (["", "", ""], 0.0, []))
            v = L.verify(inc, props, self.targets)
            items.append({
                "item_id": f"{cond}:{inc['incident_id']}",
                "input_sha256": hashlib.sha256(inc_mod.canonical_line(inc).encode()).hexdigest(),
                "output_sha256": hashlib.sha256("\n".join(v["change_texts"]).encode()).hexdigest(),
                "score": v["recovery"] if v["recovery"] is not None else 0.0,
                "passed": v["recovered"], "grader": L.VERIFIER, "backend": BACKEND, "latency_ms": wall_ms,
                "synthetic": self.synthetic,
                "extra": {"phase": cond, "incident_id": inc["incident_id"], "recovered": v["recovered"],
                          "recovery": v["recovery"], "harmful": v["harmful"], "steps_to_fix": v["steps_to_fix"],
                          "change_texts": v["change_texts"], "lessons_used": used,
                          "clamped_rules": v["clamped_rules"], "incomplete": inc["incident_id"] not in got},
            })
        return items

    def _write_phase(self, name: str, items: list[dict[str, Any]]) -> None:
        written: list[str] = []
        if self._marker and self._marker.exists():
            written = json.loads(self._marker.read_text())
        if name in written:
            self.log(f"{name}: items already written by an earlier attempt")
            return
        self.ev.append_records(self.run_id, "item", items)
        if self._marker:
            self._marker.parent.mkdir(parents=True, exist_ok=True)
            self._marker.write_text(json.dumps(written + [name]))

    def run(self, incidents: list[dict[str, Any]], digest: str) -> dict[str, Any]:
        not_confirm = [i["incident_id"] for i in incidents if i.get("split") != "confirm"]
        if not_confirm:
            raise ValueError(f"the confirmation unit runs only the frozen confirmation set: {not_confirm[:3]}")
        overlap = self.lessons.guard.overlap(self.lessons.records)
        if overlap:  # checked before anything is declared or called
            raise L.LeakageError(f"{overlap} stored lessons came from a held-out incident (test or confirmation)")
        snapshot = self.lessons.digest()
        if not self.resume:
            self.ev.write_declared(self.declaration(incidents, digest))  # before any call
            if self.copy_lessons is not None:
                self.lessons_info["copied_uri"] = self.copy_lessons(self.lessons_info["copied_to"], self.lessons.raw)
        else:
            self.log(f"resume {self.run_id}: the declaration and the lessons copy were written by the first attempt")
        phases: dict[str, list[dict[str, Any]]] = {}
        for cond in CONDITIONS:
            store = self.lessons if cond == "lessons_on" else None
            phases[cond] = self._items(cond, incidents, self.propose(cond, incidents, store))
            self._write_phase(cond, phases[cond])
            self.log(f"{cond}: {S.SelfImprove.summary(phases[cond])}")
        boot = self.targets["bootstrap"]

        def by_id(name: str) -> dict[str, float]:
            return {it["extra"]["incident_id"]: it["score"] for it in phases[name]}

        gain = paired_bootstrap(by_id("lessons_on"), by_id("lessons_off"), resamples=int(boot["resamples"]),
                                seed=int(boot["seed"]), ci=float(boot["ci"]))
        unchanged = self.lessons.digest() == snapshot
        value = gain["ci_low"] if gain else float("-inf")
        retrieved = sum(1 for it in phases["lessons_on"] if it["extra"]["lessons_used"])
        result = {
            "run_id": self.run_id, "synthetic": self.synthetic, "unit": UNIT, "metric": TARGET["metric"],
            "value": value, "target": dict(TARGET), "passed": bool(gain) and value > TARGET["value"] and unchanged,
            "n_items": sum(len(phases[c]) for c in CONDITIONS),
            "checked_at": S._utc(), "checker": {"script": SCRIPT, "code_sha": S._code_sha()},
            "details": {"gain_on_vs_off": gain, "method": METHOD, "bootstrap": boot,
                        "conditions": {c: S.SelfImprove.summary(phases[c]) for c in CONDITIONS},
                        "lessons": {**self.lessons_info, "unchanged_during_run": unchanged, "heldout_overlap": overlap},
                        "lessons_retrieved_for": {"incidents": retrieved, "of": len(phases["lessons_on"])},
                        "mode": self.mode, "ticket": S.TICKET if self.mode == "live" else None,
                        "workers": self.workers,
                        "incomplete": {c: sum(1 for it in phases[c] if it["extra"]["incomplete"]) for c in CONDITIONS}},
        }
        self.ev.write_result(result)
        return {"result": result, "phases": phases}


def patch_declaration_for(lessons_info: dict[str, Any]) -> Callable[[dict[str, Any], str], None]:
    """Each condition's sub-run declaration (written by self_improve.live_backend before its first call)
    names the confirmation suite and, for lessons_on, the read-only store it retrieves from."""
    def patch(decl: dict[str, Any], phase: str) -> None:
        decl["subset"] = "split=confirm tier=masked"
        conf = decl.setdefault("twin", {}).setdefault("config", {})
        conf["suite"] = SUITE_REL
        conf["suite_version"] = CONFIRM_VERSION
        conf["condition"] = phase
        conf["lessons"] = lessons_info if phase == "lessons_on" else None
        conf["learning"] = "none (read-only lessons)"
    return patch


def wall_estimate(n: int, steps: int, per_case_s: float = PER_CASE_S) -> dict[str, str]:
    return {f"workers={w}": f"{-(-n // w) * steps * len(CONDITIONS) * per_case_s / 60:.0f} min" for w in (1, 3, 6)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--freeze", action="store_true", help="write the suite files once (never overwrites)")
    ap.add_argument("--check", action="store_true", help="regenerate on this platform and compare the bytes")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--live", action="store_true", help="the governed swarm through the checkpoint (spends)")
    ap.add_argument("--run-id")
    ap.add_argument("--lessons", help="the main run's lesson store (JSONL); read-only")
    ap.add_argument("--lessons-run-id", help="the practice run the lessons came from (default: from the file name)")
    ap.add_argument("--evidence-dir", help="offline: local evidence directory")
    ap.add_argument("--tokens-dir")
    ap.add_argument("--resume", action="store_true", help="live: continue a crashed run with the same --run-id")
    ap.add_argument("--workers", type=int, default=1, help="live: incidents in parallel within a condition")
    ap.add_argument("--timeout", type=float, default=300.0, help="live: seconds to wait for the lead per case")
    a = ap.parse_args(argv)
    from t3.evals.twin_backend import single_thread
    single_thread()  # the torch reduction order the suite was frozen with
    targets = load_targets()
    steps = targets["episode"]["operator_steps"]
    if a.freeze:
        out = freeze(targets=targets)
        m = out["meta"]
        print(json.dumps({k: m[k] for k in ("suite", "sha256", "n", "seed_range", "mix", "test_masked_mix")}
                         | {"candidates_built": m["generator"]["candidates_built"],
                            "k_used": m["generator"]["k_used"],
                            "dropped": len(m["generator"]["dropped"]),
                            "exclusion_clean": m["exclusion"]["clean"],
                            "literal_scan_files": m["exclusion"]["literal_scan"]["files_scanned"],
                            "overlap_total": m["overlap"]["total"]},
                         indent=1))
        return 0
    if a.check:
        fresh = suite_bytes(generate_confirm(targets)["incidents"])
        same = CONFIRM_PATH.read_bytes() == fresh
        print(json.dumps({"matches_frozen": same, "platform": f"{platform.system()}-{platform.machine()}"}))
        return 0 if same else 1
    incidents, digest = load_confirm()
    lessons = FrozenLessons(a.lessons) if a.lessons else None
    if a.dry_run:
        n_cases = len(incidents) * steps
        plan = {"unit": UNIT, "target": TARGET, "suite": SUITE_REL, "sha256": digest, "incidents": len(incidents),
                "conditions": list(CONDITIONS), "ticket": S.TICKET, "learning": "none (read-only lessons)",
                "live_gateway_cases": {"per_condition": n_cases, "total": n_cases * len(CONDITIONS),
                                       "incident_runs": len(incidents) * len(CONDITIONS)},
                "wall_time_estimate": {"assumes_s_per_case": PER_CASE_S, **wall_estimate(len(incidents), steps)}}
        if lessons is not None:
            plan["lessons"] = {**lessons.info(a.lessons_run_id),
                               "heldout_overlap": lessons.guard.overlap(lessons.records)}
        print(json.dumps(plan, indent=1))
        return 0
    if lessons is None:
        raise SystemExit("--lessons PATH is required: the main run's lesson store (read-only)")
    if a.live:
        bad = [r for r in lessons.records if r.get("source") != "practice:own-attempt"]
        if bad:
            raise SystemExit(f"{len(bad)} lesson records are not the swarm's own verified attempts "
                             f"(e.g. {bad[0].get('source')}): a live confirmation uses only own-attempt lessons")
        if not lessons.lessons:
            raise SystemExit("the lesson store has no verified lesson: lessons_on would equal lessons_off")
    from t3.evals.evshim import new_run_id
    if a.resume and not a.run_id:
        raise SystemExit("--resume needs the --run-id of the run to continue")
    run_id = a.run_id or new_run_id(UNIT)
    if not a.live:
        from t3.evals.evshim import Evidence
        if not a.evidence_dir:
            raise SystemExit("offline needs --evidence-dir (local)")
        ev_root = Path(a.evidence_dir)

        def copy_local(key: str, data: bytes) -> str:
            p = ev_root / key
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "xb") as f:
                f.write(data)
            return str(p)

        unit = Confirm(run_id=run_id, ev=Evidence(ev_root), propose=S.offline_backend(targets),
                       model_id="standin:config-revert+lessons-v1", synthetic=True, lessons=lessons, targets=targets,
                       mode="offline", copy_lessons=copy_local, lessons_run_id=a.lessons_run_id,
                       checkpoint_dir=ev_root / "checkpoints")
    else:
        import boto3

        from t3.evals.live_swarm import DEFAULT_MODEL
        from t3.suite import common
        from t3.suite.runner import env_config
        cfg = env_config()
        session = boto3.session.Session(region_name=cfg["region"])
        secrets = session.client("secretsmanager") if os.environ.get("T3_TOKENS_SECRET") else None
        gw = common.Gateway(cfg["gateway_url"], common.Tokens(secrets_client=secrets, tokens_dir=a.tokens_dir))
        evlib = common.load_evlib()
        model_id = os.environ.get("T3_SWARM_MODEL_ID", DEFAULT_MODEL)
        print(f"LIVE {run_id}: {len(incidents) * steps * len(CONDITIONS)} Gateway cases")
        info = {**lessons.info(a.lessons_run_id), "copied_to": f"outputs/{run_id}/lessons.jsonl"}
        unit = Confirm(run_id=run_id, ev=evlib, model_id=model_id, synthetic=False, lessons=lessons, targets=targets,
                       mode="live", workers=a.workers, resume=a.resume, checkpoint_dir=common.CALLS_DIR,
                       copy_lessons=lambda key, data: evlib.default_store().put_new(key, data),
                       lessons_run_id=a.lessons_run_id,
                       propose=S.live_backend(base_run_id=run_id, gateway=gw, ddb=session.client("dynamodb"),
                                              evlib=evlib, cfg=cfg, targets=targets, model_id=model_id,
                                              digests={"test": digest, "dev": digest}, timeout_s=a.timeout,
                                              workers=a.workers, checkpoint_dir=common.CALLS_DIR, unit=UNIT,
                                              patch_declaration=patch_declaration_for(info)))
    out = unit.run(incidents, digest)
    r = out["result"]
    print(json.dumps({"run_id": run_id, "value": r["value"], "passed": r["passed"],
                      "conditions": r["details"]["conditions"], "gain": r["details"]["gain_on_vs_off"],
                      "lessons_retrieved_for": r["details"]["lessons_retrieved_for"],
                      "lessons": {k: r["details"]["lessons"][k] for k in ("sha256", "verified", "rejected",
                                                                            "unchanged_during_run")}},
                     indent=1, default=str))
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
