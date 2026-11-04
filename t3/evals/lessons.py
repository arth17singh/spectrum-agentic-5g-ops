"""Verified lesson memory: the swarm learns from practice incidents, and only from fixes the twin confirms.

A lesson is learned from a DEV incident only (never the frozen test split or the confirmation set;
``learn`` refuses both). It holds:

- ``symptoms``: what the symptoms-only ticket states (``swarm.task_text``): the KPI drop from the
  pre-fault to the post-fault state. Never the fault, the template or a cell id.
- ``pattern``: the fix expressed relative to the configuration the agent can observe, e.g. "lower tilt
  by 5 deg on the cell with the highest tilt". ``resolve`` turns it back into a concrete change on a
  new network.
- ``source_incident``: the dev incident it came from, and ``verification``: the twin replay of the fix
  (healthy and unfixed counterfactuals at the same grade step, graded by ``graders.grade_kpis``).

A fix that does not bring every graded KPI back within tolerance at the grade step is stored as
``rejected`` with the reason ("not learned"); ``retrieve`` never returns it. No model grades anything.

    store = LessonStore(path)                         # local JSONL (optionally mirrored with put_new)
    v = verify(incident, ["cell=0;d_power_db=+0.0;d_tilt_deg=-5.0", "", ""])
    store.learn(incident, proposals, v)               # -> lesson (verified) or rejected record
    hits = store.retrieve(symptoms_of(test_incident), k=3)   # [(lesson, similarity in [0,1]), ...]
    text = lesson_block(hits)                         # appended to the ticket for front___continue_case
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

from t3.evals import incidents as inc_mod
from t3.evals.changes import canon, nonzero
from t3.evals.graders import grade_kpis, load_targets
from t3.evals.twin_backend import BACKEND, ReplayTwin, graded

VERSION = "t3-lessons-v1"
VERIFIER = "t3/evals/lessons.py:verify v1 (twin replay at the grade step, graders.grade_kpis)"
SYMPTOM_KEYS = ("d_coverage", "d_sla", "d_sinr_p10_db", "post_coverage")
# How far apart two symptom vectors may be per feature before similarity halves (roughly): a ticket
# states coverage and SLA to 3 decimals and SINR to 0.1 dB.
SYMPTOM_SCALE = {"d_coverage": 0.05, "d_sla": 0.05, "d_sinr_p10_db": 1.5, "post_coverage": 0.10}
SPEC_KEYS = ("seed", "preset", "scenario", "tier", "template", "commissioning", "fault")


class LeakageError(ValueError):
    """Refused: the incident is (or overlaps) the frozen test split."""


# --------------------------------------------------------------------------- symptoms

def symptoms_of(incident: dict[str, Any]) -> dict[str, float]:
    """The numbers the symptoms-only ticket states (pre -> post fault), nothing else."""
    pre, post = incident["pre_fault_kpis"], incident["post_fault_kpis"]
    return {
        "d_coverage": round(post["coverage_ratio"] - pre["coverage_ratio"], 3),
        "d_sla": round(post["sla_ratio"] - pre["sla_ratio"], 3),
        "d_sinr_p10_db": round(post["sinr_p10_db"] - pre["sinr_p10_db"], 1),
        "post_coverage": round(post["coverage_ratio"], 3),
    }


def similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """exp(-d^2 / 2) over the scaled symptom features: 1.0 = the same ticket, -> 0 as they differ."""
    d2 = sum(((a[k] - b[k]) / SYMPTOM_SCALE[k]) ** 2 for k in SYMPTOM_KEYS)
    return round(math.exp(-0.5 * d2), 4)


# --------------------------------------------------------------------------- relative patterns

def _ranked(values: list[float]) -> list[int]:
    """Cell indices sorted by value, highest first (ties: lower cell index first)."""
    return sorted(range(len(values)), key=lambda c: (-values[c], c))


def describe(change: list[dict[str, Any]] | str, state: dict[str, Any]) -> list[dict[str, Any]]:
    """Express a concrete change relative to the observable configuration (no cell ids).

    Each item gets the dimension it mainly acts on (tilt if it changes tilt, else power) and the
    cell's rank on that dimension (0 = highest) in ``state`` (the configuration before the fix)."""
    items = nonzero(change)
    out = []
    for it in items:
        by = "tilt" if it["d_tilt_deg"] != 0.0 else "power"
        values = [float(x) for x in state["tilt_deg" if by == "tilt" else "power_dbm"]]
        out.append({"by": by, "rank": _ranked(values).index(int(it["cell"])),
                    "d_power_db": it["d_power_db"], "d_tilt_deg": it["d_tilt_deg"]})
    return out


def resolve(pattern: list[dict[str, Any]], state: dict[str, Any]) -> list[dict[str, Any]]:
    """The concrete change a relative pattern means on ``state`` (each cell used at most once)."""
    used: set[int] = set()
    out = []
    for p in pattern:
        values = [float(x) for x in state["tilt_deg" if p["by"] == "tilt" else "power_dbm"]]
        order = [c for c in _ranked(values) if c not in used]
        if not order:
            break
        cell = order[min(int(p["rank"]), len(order) - 1)]
        used.add(cell)
        out.append({"cell": cell, "d_power_db": float(p["d_power_db"]), "d_tilt_deg": float(p["d_tilt_deg"])})
    return out


_ORD = {0: "highest", 1: "2nd-highest", 2: "2nd-lowest", 3: "lowest"}


def pattern_text(pattern: list[dict[str, Any]]) -> str:
    parts = []
    for p in pattern:
        acts = []
        if p["d_tilt_deg"]:
            acts.append(f"{'lower' if p['d_tilt_deg'] < 0 else 'raise'} tilt by {abs(p['d_tilt_deg']):g} deg")
        if p["d_power_db"]:
            acts.append(f"{'raise' if p['d_power_db'] > 0 else 'lower'} power by {abs(p['d_power_db']):g} dB")
        unit = "tilt" if p["by"] == "tilt" else "power"
        parts.append(f"{' and '.join(acts)} on the cell with the {_ORD.get(int(p['rank']), 'rank ' + str(p['rank']))} "
                     f"{unit}")
    return "; ".join(parts) or "no change"


# --------------------------------------------------------------------------- verification

_LOCAL: dict[str, dict[str, Any]] = {}


def local_incident(incident: dict[str, Any], targets: dict[str, Any]) -> dict[str, Any]:
    """The incident rebuilt on THIS machine from its spec (the twin is a float32 GP: frozen KPIs replay
    to 1e-6 only on the platform that froze them), so healthy/unfixed are this platform's numbers."""
    key = hashlib.sha256(inc_mod.canonical_line({k: incident[k] for k in SPEC_KEYS}).encode()).hexdigest()
    if key not in _LOCAL:
        built, why = inc_mod.build({k: incident[k] for k in SPEC_KEYS}, targets)
        if built is None:
            raise ValueError(f"{incident.get('incident_id')}: does not rebuild as an incident here ({why})")
        _LOCAL[key] = built
    return _LOCAL[key]


def verify(incident: dict[str, Any], proposals: Iterable[Any], targets: dict[str, Any] | None = None,
           twin_cls: Callable[..., Any] = ReplayTwin) -> dict[str, Any]:
    """Replay the incident with ``proposals`` (one change per operator step; "", None or [] = no change)
    and grade each step against the same-step healthy and unfixed counterfactuals.

    Returns {recovered, recovery, harmful, steps_to_fix, per_step, change_texts, state_before_fix, ...}.
    ``recovered`` = every graded KPI within tolerance at the grade step (graders.grade_kpis success)."""
    targets = targets or load_targets()
    ep = targets["episode"]
    local = local_incident(incident, targets)
    props = [nonzero(p) for p in proposals][: ep["operator_steps"]]
    props += [[]] * (ep["operator_steps"] - len(props))
    twin = twin_cls(local["scenario"], local["seed"], local["preset"])
    twin.reset()
    state = None
    for change in inc_mod.script(local, targets, with_fault=True):
        state = twin.apply(change)
    before = {"power_dbm": [float(x) for x in state["power_dbm"]], "tilt_deg": [float(x) for x in state["tilt_deg"]]}
    per_step = []
    steps_to_fix = None
    for i, change in enumerate(props, start=1):
        state = twin.apply(change)
        step = twin.step
        g = grade_kpis(graded(state), local["healthy_kpis_by_step"][step], local["unfixed_kpis_by_step"][step],
                       targets)
        per_step.append({"step": step, "change": canon(change) if change else "", "success": g["success"],
                         "recovery": g["recovery"]})
        if g["success"] and steps_to_fix is None and any(nonzero(p) for p in props[:i]):
            steps_to_fix = i
    final = graded(state)
    g = grade_kpis(final, local["healthy_kpis_by_step"][twin.step], local["unfixed_kpis_by_step"][twin.step], targets)
    clamped = [r for e in twin.action_log[ep["fault_step"]:] for r in e.get("rules", [])]
    rec = g["recovery"]
    return {
        "verifier": VERIFIER,
        "backend": BACKEND,
        "recovered": bool(g["success"]),
        "recovery": rec,
        "harmful": rec is not None and rec < 0.0,
        "steps_to_fix": steps_to_fix if g["success"] else None,
        "per_step": per_step,
        "change_texts": [canon(p) if p else "" for p in props],
        "final_kpis": final,
        "clamped_rules": clamped,
        "state_before_fix": before,
    }


def inverse_fix(incident: dict[str, Any]) -> str:
    """The exact inverse of the fault (a dev incident's ground-truth label)."""
    return canon([{"cell": f["cell"], "d_power_db": -f["d_power_db"], "d_tilt_deg": -f["d_tilt_deg"]}
                  for f in incident["fault"]])


def wrong_fix(incident: dict[str, Any], n_cells: int = 4) -> str:
    """The inverse applied to the wrong cell(s) (shifted by 2): a fix the verifier must reject."""
    return canon([{"cell": (f["cell"] + 2) % n_cells, "d_power_db": -f["d_power_db"],
                   "d_tilt_deg": -f["d_tilt_deg"]} for f in incident["fault"]])


# --------------------------------------------------------------------------- leakage guard

def _spec_hash(incident: dict[str, Any]) -> str:
    return hashlib.sha256(inc_mod.canonical_line({k: incident[k] for k in SPEC_KEYS}).encode()).hexdigest()


# The confirmation set (t3/evals/confirm.py): fresh held-out incidents, frozen once, never learned from.
CONFIRM_PATH = inc_mod.SUITE_DIR / "incidents-confirm.jsonl"


def load_confirm_file(path: Path = CONFIRM_PATH) -> list[dict[str, Any]]:
    """The frozen confirmation incidents ([] before the set is frozen). The sha256 sidecar is checked by
    ``confirm.load_confirm``; the guard only needs the seeds, ids and spec bytes."""
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


# The remediation held-out set (t1/rl/remediation/heldout.py): its whole reserved seed block, and once frozen
# its incidents' ids and spec bytes. The block is repeated here so t3 needs no t1 import; a test checks it
# against heldout.SEED_LO / SEED_HI.
HELDOUT_SEED_BLOCK = (2_000_001, 2_022_400)
HELDOUT_PATH = inc_mod.SUITE_DIR.parents[2] / "t1" / "rl" / "remediation" / "suite" / "heldout-v1.jsonl"


def load_heldout_file(path: Path = HELDOUT_PATH) -> list[dict[str, Any]]:
    """The frozen remediation held-out incidents ([] before the set is frozen); the guard needs ids and specs."""
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def in_heldout_block(seed: int) -> bool:
    return HELDOUT_SEED_BLOCK[0] <= int(seed) <= HELDOUT_SEED_BLOCK[1]


class TestSplitGuard:
    """Refuses any incident from (or overlapping) a held-out set: the frozen test split and the
    confirmation set (``suite/incidents-confirm.jsonl``). Same seed, same id, or the same spec bytes (the
    Track 1 data check's hash-overlap idea, applied to incident specs)."""

    def __init__(self, test_incidents: list[dict[str, Any]] | None = None,
                 confirm_incidents: list[dict[str, Any]] | None = None,
                 heldout_incidents: list[dict[str, Any]] | None = None) -> None:
        if test_incidents is None:
            test_incidents, _ = inc_mod.load_test(inc_mod.TEST_PATH, verify_hash=False)
        if confirm_incidents is None:
            confirm_incidents = load_confirm_file()
        if heldout_incidents is None:
            heldout_incidents = load_heldout_file()
        self.seeds = {int(i["seed"]) for i in test_incidents}
        self.ids = {i["incident_id"] for i in test_incidents}
        self.hashes = {_spec_hash(i) for i in test_incidents}
        self.confirm_seeds = {int(i["seed"]) for i in confirm_incidents}
        self.confirm_ids = {i["incident_id"] for i in confirm_incidents}
        self.confirm_hashes = {_spec_hash(i) for i in confirm_incidents}
        self.heldout_ids = {i["incident_id"] for i in heldout_incidents}
        self.heldout_hashes = {_spec_hash(i) for i in heldout_incidents}

    def check(self, incident: dict[str, Any]) -> None:
        why = []
        if incident.get("split") == "test" or inc_mod.split_of(int(incident["seed"])) == "test":
            why.append("seed is in the test split")
        if incident.get("split") == "confirm":
            why.append("incident is in the confirmation split")
        if incident.get("split") == "heldout" or in_heldout_block(int(incident["seed"])):
            why.append("seed is in the remediation held-out block")
        if incident.get("incident_id") in self.heldout_ids:
            why.append("incident id appears in the remediation held-out set")
        if _spec_hash(incident) in self.heldout_hashes:
            why.append("spec hash overlaps a remediation held-out incident")
        if int(incident["seed"]) in self.seeds:
            why.append("seed appears in the frozen test file")
        if int(incident["seed"]) in self.confirm_seeds:
            why.append("seed appears in the confirmation set")
        if incident.get("incident_id") in self.ids:
            why.append("incident id appears in the frozen test file")
        if incident.get("incident_id") in self.confirm_ids:
            why.append("incident id appears in the confirmation set")
        h = _spec_hash(incident)
        if h in self.hashes:
            why.append("spec hash overlaps a test incident")
        if h in self.confirm_hashes:
            why.append("spec hash overlaps a confirmation incident")
        if why:
            raise LeakageError(f"{incident.get('incident_id')}: refused to learn ({'; '.join(why)})")

    def overlap(self, lessons: Iterable[dict[str, Any]]) -> int:
        """How many stored lessons came from a held-out incident: test, confirmation or the remediation
        held-out set and its seed block (must be 0)."""
        seeds, ids, hashes = (self.seeds | self.confirm_seeds, self.ids | self.confirm_ids | self.heldout_ids,
                              self.hashes | self.confirm_hashes | self.heldout_hashes)
        return sum(1 for l in lessons
                   if l.get("source_seed") in seeds or l.get("source_incident") in ids
                   or l.get("source_spec_sha256") in hashes
                   or (l.get("source_seed") is not None and in_heldout_block(int(l["source_seed"]))))


# --------------------------------------------------------------------------- the store

class LessonStore:
    """Verified lessons and rejected attempts, one canonical JSON line each (append-only file).

    ``mirror`` is optional: an object with ``put_new(key, bytes)`` (evlib's store) used by ``publish``."""

    def __init__(self, path: str | Path | None = None, guard: TestSplitGuard | None = None,
                 mirror: Any = None) -> None:
        self.path = Path(path) if path else None
        self.guard = guard if guard is not None else TestSplitGuard()
        self.mirror = mirror
        self.records: list[dict[str, Any]] = []
        if self.path and self.path.exists():
            self.records = [json.loads(x) for x in self.path.read_text().splitlines() if x.strip()]

    @property
    def lessons(self) -> list[dict[str, Any]]:
        return [r for r in self.records if r["status"] == "verified"]

    @property
    def rejected(self) -> list[dict[str, Any]]:
        return [r for r in self.records if r["status"] == "rejected"]

    def _append(self, rec: dict[str, Any]) -> dict[str, Any]:
        self.records.append(rec)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(inc_mod.canonical_line(rec) + "\n")
        return rec

    def learn(self, incident: dict[str, Any], proposals: list[Any], verification: dict[str, Any],
              source: str = "practice") -> dict[str, Any]:
        """Store a verified lesson if the fix recovered the incident, else a rejected record.

        Refuses (LeakageError) any incident from the test split, before anything is stored."""
        self.guard.check(incident)
        changes = [nonzero(p) for p in proposals]
        first = next((i for i, c in enumerate(changes) if c), None)
        base = {"version": VERSION, "source_incident": incident["incident_id"], "source_seed": int(incident["seed"]),
                "source_spec_sha256": _spec_hash(incident), "source": source,
                "symptoms": symptoms_of(incident), "change_texts": verification["change_texts"],
                "verification": {k: verification[k] for k in ("verifier", "backend", "recovered", "recovery",
                                                              "harmful", "steps_to_fix", "clamped_rules")}}
        if first is None:
            return self._append({**base, "status": "rejected", "reason": "no change proposed"})
        if not verification["recovered"]:
            r = verification["recovery"]
            why = "made the network worse than leaving it alone" if verification["harmful"] else \
                f"the cell did not recover (recovery {r:.2f})" if r is not None else "the cell did not recover"
            return self._append({**base, "status": "rejected", "reason": f"not learned: {why}"})
        # described relative to the post-fault configuration (what the agent observed before any fix)
        pattern = describe(changes[first], verification["state_before_fix"])
        key = json.dumps(pattern, sort_keys=True)
        for l in self.lessons:  # the same fix for near-identical symptoms: more support, not a new lesson
            if json.dumps(l["pattern"], sort_keys=True) == key and similarity(l["symptoms"], base["symptoms"]) >= 0.95:
                l["support"] = l.get("support", 1) + 1
                l.setdefault("also_from", []).append(incident["incident_id"])
                return l
        lid = f"L-{len(self.lessons) + 1:02d}"
        return self._append({**base, "status": "verified", "lesson_id": lid, "pattern": pattern,
                             "pattern_text": pattern_text(pattern), "support": 1})

    def retrieve(self, symptoms: dict[str, float], k: int = 3, min_similarity: float = 0.0
                 ) -> list[tuple[dict[str, Any], float]]:
        """The k verified lessons nearest to ``symptoms``, with similarity in [0, 1] (best first)."""
        scored = [(l, similarity(l["symptoms"], symptoms)) for l in self.lessons]
        scored = [s for s in scored if s[1] >= min_similarity]
        scored.sort(key=lambda s: (-s[1], -s[0].get("support", 1), s[0]["lesson_id"]))
        return scored[:k]

    def best_similarity(self, symptoms: dict[str, float]) -> float:
        """The ``lesson`` signal for the confidence score: similarity of the nearest verified lesson."""
        hits = self.retrieve(symptoms, k=1)
        return hits[0][1] if hits else 0.0

    def publish(self, key: str) -> str | None:
        """Mirror the whole file to the evidence store (put_new: never overwrites)."""
        if self.mirror is None:
            return None
        body = "".join(inc_mod.canonical_line(r) + "\n" for r in self.records).encode()
        return self.mirror.put_new(key, body)


def lesson_block(hits: list[tuple[dict[str, Any], float]], min_similarity: float = 0.3) -> str:
    """A short ticket appendix for front___continue_case: verified past lessons, no cell ids, no test data."""
    rows = [(l, s) for l, s in hits if s >= min_similarity]
    if not rows:
        return ""
    lines = ["Verified past lessons (from practice incidents; each fix was replayed on the twin and recovered "
             "service). Use them as hints, still observe and preflight first:"]
    for l, s in rows:
        sy = l["symptoms"]
        lines.append(f"- {l['lesson_id']} (similarity {s:.2f}): when coverage fell {abs(sy['d_coverage']):.3f}, "
                     f"SLA {abs(sy['d_sla']):.3f}, p10 SINR {sy['d_sinr_p10_db']:+.1f} dB -> "
                     f"{pattern_text(l['pattern'])} worked.")
    return "\n".join(lines)
