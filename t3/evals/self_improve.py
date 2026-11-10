"""Does the swarm get better from verified lessons? A measured unit, declared before it runs.

Phases, in order, all graded by the twin (``lessons.verify``), never by a model:

1. BEFORE   - lessons OFF, the 18 masked incidents of the frozen test split.
2. PRACTICE - the DEV split only: the operator's own attempts (and, offline, the dev ground-truth
              inverse fix) are replayed on the twin; verified fixes become lessons, failed ones are
              stored as rejected ("not learned"). ``LessonStore.learn`` refuses any test incident.
3. AFTER    - lessons ON (nearest verified lessons appended to the ticket), the same 18 incidents.
4. CONTROL  - lessons OFF again, the same 18: run-to-run noise. If CONTROL also "improves", the gain
              is noise, not learning.

Per incident: recovered (all graded KPIs within tolerance at the grade step), recovery, steps to fix,
harmful (recovery < 0), wall time. Primary: paired bootstrap (stats.paired_bootstrap, targets.json's
bootstrap block) of recovery AFTER - BEFORE by incident id. Declared target: ci_low > 0.

Two modes:
- offline (default): a STAND-IN proposer (``LessonGuided``: config_revert when it has no lesson, the
  nearest verified lesson resolved on the observed configuration when it has one). No AWS, no model.
  Every record is synthetic=True: it tests the pipeline and the verifier, it is not a swarm result.
- --live: the governed swarm through the checkpoint (``live_swarm.LiveSwarm.run_incident``, proposal
  mode, nobody approves); only the ticket changes (``lessons.lesson_block`` appended). Spends Bedrock.

    python -m t3.evals.self_improve --dry-run                       # plan + Gateway case count, no calls
    python -m t3.evals.self_improve --evidence-dir DIR              # offline, synthetic
    python -m t3.evals.self_improve --live --run-id t3.self-improve-<UTC>   # main session only, after OK
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from t3.evals import incidents as inc_mod
from t3.evals import lessons as L
from t3.evals.graders import load_targets
from t3.evals.operators import ConfigRevert
from t3.evals.stats import METHOD, paired_bootstrap
from t3.evals.swarm import task_text
from t3.evals.twin_backend import BACKEND, ReplayTwin, twin_version

UNIT = "t3.evals.self-improve"
SCRIPT = "t3/evals/self_improve.py"
TARGET = {"metric": "gain_ci_low_after_vs_before", "op": ">", "value": 0.0, "unit": "recovery"}
PHASES = ("before", "practice", "after", "control")
LESSON_MIN_SIM = 0.30
TICKET = "opaque-id-v1"


def opaque_label(incident: dict[str, Any]) -> str:
    """The label the agents see: "inc-" + the first 8 hex of sha256(incident_id). The same rule as Live Fix's
    ``opaque_label`` (dashboard livefix/suite.py). Suite ids name the fault template ("inc-...-tilt+5"), so the
    real id stays in the evidence records only (GAPS G-025)."""
    return "inc-" + hashlib.sha256(str(incident["incident_id"]).encode()).hexdigest()[:8]


def opaque_ticket(incident: dict[str, Any], run_id: str) -> str:
    """``swarm.task_text`` word for word, with the incident id replaced by its opaque label. The judged suite's
    ``task_text`` is unchanged."""
    return task_text({**incident, "incident_id": opaque_label(incident)}, run_id)


# --------------------------------------------------------------------------- incidents

def test_masked() -> tuple[list[dict[str, Any]], str]:
    incs, digest = inc_mod.load_test(inc_mod.TEST_PATH)
    return [i for i in incs if i.get("tier") == "masked"], digest


def dev_incidents() -> tuple[list[dict[str, Any]], str]:
    data = inc_mod.DEV_PATH.read_bytes()
    return [json.loads(x) for x in data.decode().splitlines() if x.strip()], hashlib.sha256(data).hexdigest()


def cost_estimate(n_test: int, n_dev: int, steps: int) -> dict[str, Any]:
    """Gateway cases for a live run (each case = seed + one lead turn with its student calls + polls)."""
    per_phase = {"before": n_test * steps, "practice": n_dev * steps, "after": n_test * steps,
                 "control": n_test * steps}
    return {"cases_per_phase": per_phase, "cases_total": sum(per_phase.values()),
            "note": "each case is one noc_lead turn (it delegates to the students) through the checkpoint; "
                    "the swarm unit's earlier live run used the same case shape (18 x 3 = 54 cases)"}


# --------------------------------------------------------------------------- offline stand-in

class LessonGuided:
    """STAND-IN proposer (synthetic): config_revert without a lesson; with lessons, the nearest verified
    lesson (similarity >= LESSON_MIN_SIM) resolved on the observed configuration at the first step, and
    config_revert afterwards only if coverage is still below the ticket's pre-fault level."""

    name = "lesson_guided_standin"
    synthetic = True

    def __init__(self, store: L.LessonStore | None) -> None:
        self.store = store
        self.revert = ConfigRevert()
        self.model_id = "standin:config-revert+lessons-v1" if store is not None else "standin:config-revert"

    def begin(self, incident: dict[str, Any]) -> None:
        self.inc = incident
        self.hits = self.store.retrieve(L.symptoms_of(incident), k=3, min_similarity=LESSON_MIN_SIM) \
            if self.store is not None else []
        self.used: list[str] = []

    def act(self, state: dict[str, Any], history: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        if self.hits and not history:
            lesson, _ = self.hits[0]
            self.used.append(lesson["lesson_id"])
            return L.resolve(lesson["pattern"], state)
        if self.hits and history:
            pre = self.inc["pre_fault_kpis"]["coverage_ratio"]
            if history[-1]["coverage"] >= pre - 0.02:
                return None
        return self.revert.act(state, history)


def play_offline(incident: dict[str, Any], op: LessonGuided, targets: dict[str, Any]) -> list[str]:
    """The operator's proposals, one per operator step, on this platform's rebuild of the incident."""
    from t3.evals.changes import canon, nonzero
    from t3.evals.twin_backend import graded

    local = L.local_incident(incident, targets)
    twin = ReplayTwin(local["scenario"], local["seed"], local["preset"])
    twin.reset()
    for ch in inc_mod.script(local, targets, with_fault=True):
        state = twin.apply(ch)
    op.begin(incident)
    history: list[dict[str, Any]] = []
    out = []
    for _ in range(targets["episode"]["operator_steps"]):
        change = nonzero(op.act(state, history))
        state = twin.apply(change)
        out.append(canon(change) if change else "")
        history.append({"change": change, "coverage": graded(state)["coverage_ratio"]})
    return out


# --------------------------------------------------------------------------- the unit

class SelfImprove:
    """Runs the four phases with a ``propose(phase, incidents, store) -> {incident_id: (proposals, wall_ms,
    lessons_used)}`` backend, verifies everything on the twin, and writes the evidence."""

    def __init__(self, *, run_id: str, ev: Any, propose: Callable[..., dict[str, tuple[list[str], float, list[str]]]],
                 model_id: str, synthetic: bool, store: L.LessonStore, targets: dict[str, Any] | None = None,
                 offline_labels: bool = False, log: Callable[[str], None] = print, mode: str = "offline",
                 workers: int = 1, resume: bool = False, checkpoint_dir: Path | None = None) -> None:
        if mode == "live" and offline_labels:
            raise ValueError("live practice learns from the swarm's own attempts only (no dev labels)")
        self.run_id, self.ev, self.propose, self.model_id = run_id, ev, propose, model_id
        self.synthetic, self.store, self.targets, self.log = synthetic, store, targets or load_targets(), log
        self.offline_labels, self.mode, self.workers = offline_labels, mode, max(1, int(workers))
        self._learn_lock = threading.Lock()
        self.resume = resume
        self._marker = Path(checkpoint_dir) / f"{run_id}.phases-written.json" if checkpoint_dir else None

    def declaration(self, test: list[dict[str, Any]], digest: str, dev_digest: str, n_dev: int) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "synthetic": self.synthetic, "unit": UNIT, "track": "track-3",
            "target": dict(TARGET), "n_items": len(test) * 3, "n_cases": len(test) * 3 + n_dev,
            "test_set_sha256": digest, "subset": "tier=masked", "model_id": self.model_id,
            "grader": f"{L.VERIFIER}; primary = {METHOD} of recovery (after - before) paired by incident id",
            "code_sha": _code_sha(), "region": "us-east-2",
            "twin": {"kind": "maveric-inprocess", "version": twin_version(), "scenario": "incident-suite",
                     "seed": min(i["seed"] for i in test),
                     "config": {"backend": BACKEND, "suite": "t3/evals/suite/incidents-test.jsonl",
                                "suite_version": inc_mod.SUITE_VERSION, "practice_suite":
                                "t3/evals/suite/incidents-dev.jsonl", "practice_sha256": dev_digest,
                                "phases": list(PHASES), "mode": self.mode,
                                "lessons": {"version": L.VERSION, "min_similarity": LESSON_MIN_SIM, "k": 3,
                                            "learn_from": "dev split only (LessonStore.learn refuses the test "
                                                          "split: seed, id and spec-hash checks)",
                                            "offline_dev_labels": self.offline_labels},
                                "graded_on": "this platform's rebuild of each incident from its spec (the frozen "
                                             "KPIs replay to 1e-6 only on x86_64); healthy and unfixed "
                                             "counterfactuals rebuilt with it",
                                "control": "lessons off again on the same incidents (run-to-run noise)",
                                "ticket": TICKET if self.mode == "live" else "n/a (offline stand-in has no ticket)",
                                "practice_learning": ("own attempts only: each practice incident's own proposals, "
                                                      "verified on the twin, learned as soon as that incident "
                                                      "completes; no dev ground-truth labels"
                                                      if not self.offline_labels else
                                                      "own attempts + dev ground-truth inverse (offline only)"),
                                "workers": self.workers,
                                "parallel_note": ("with workers > 1, practice incidents run concurrently, so a "
                                                  "lesson learned from one is not visible to incidents already "
                                                  "running" if self.workers > 1 else "sequential")}},
        }

    def _items(self, phase: str, incidents: list[dict[str, Any]], got: dict[str, Any]) -> list[dict[str, Any]]:
        items = []
        for inc in incidents:
            props, wall_ms, used = got.get(inc["incident_id"], (["", "", ""], 0.0, []))
            v = self._verified.pop((phase, inc["incident_id"]), None) or L.verify(inc, props, self.targets)
            items.append({
                "item_id": f"{phase}:{inc['incident_id']}",
                "input_sha256": hashlib.sha256(inc_mod.canonical_line(inc).encode()).hexdigest(),
                "output_sha256": hashlib.sha256("\n".join(v["change_texts"]).encode()).hexdigest(),
                "score": v["recovery"] if v["recovery"] is not None else 0.0,
                "passed": v["recovered"], "grader": L.VERIFIER, "backend": BACKEND, "latency_ms": wall_ms,
                "synthetic": self.synthetic,
                "extra": {"phase": phase, "incident_id": inc["incident_id"], "recovered": v["recovered"],
                          "recovery": v["recovery"], "harmful": v["harmful"], "steps_to_fix": v["steps_to_fix"],
                          "change_texts": v["change_texts"], "lessons_used": used,
                          "clamped_rules": v["clamped_rules"],
                          "incomplete": inc["incident_id"] not in got},
                "_v": v, "_inc": inc,
            })
        return items

    def learn_now(self, inc: dict[str, Any], proposals: list[str]) -> dict[str, Any]:
        """Practice: verify this incident's own proposals on the twin and learn (or reject) at once, so later
        practice incidents can retrieve it. Thread-safe."""
        v = L.verify(inc, proposals, self.targets)
        with self._learn_lock:
            self._verified[("practice", inc["incident_id"])] = v
            rec = self.store.learn(inc, v["change_texts"], v, source="practice:own-attempt")
            self._learned[inc["incident_id"]] = rec
        return rec

    def _write_phase(self, name: str, items: list[dict[str, Any]]) -> None:
        """Item records go out as soon as a phase ends, so a later crash keeps them; a resumed run does not
        write a phase's items twice (a marker file beside the checkpoints records which phases went out)."""
        written: list[str] = []
        if self._marker and self._marker.exists():
            written = json.loads(self._marker.read_text())
        if name in written:
            self.log(f"{name}: items already written by an earlier attempt")
            return
        self.ev.append_records(self.run_id, "item", [{k: v for k, v in it.items() if not k.startswith("_")}
                                                     for it in items])
        if self._marker:
            self._marker.parent.mkdir(parents=True, exist_ok=True)
            self._marker.write_text(json.dumps(written + [name]))

    @staticmethod
    def summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(items) or 1
        fixed = [i["extra"]["steps_to_fix"] for i in items if i["extra"]["steps_to_fix"]]
        return {"n": len(items),
                "recovered": sum(i["passed"] for i in items),
                "recovery_rate": sum(i["passed"] for i in items) / n,
                "mean_recovery": sum(i["score"] for i in items) / n,
                "harmful": sum(i["extra"]["harmful"] for i in items),
                "mean_steps_to_fix": (sum(fixed) / len(fixed)) if fixed else None,
                "mean_wall_ms": sum(i["latency_ms"] for i in items) / n}

    def run(self, test: list[dict[str, Any]], digest: str, dev: list[dict[str, Any]], dev_digest: str
            ) -> dict[str, Any]:
        self._verified: dict[tuple[str, str], dict[str, Any]] = {}
        self._learned: dict[str, dict[str, Any]] = {}
        for r in self.store.records:  # a resumed run: practice incidents already learned (or rejected) stay so
            if r.get("source") == "practice:own-attempt":
                self._learned[r["source_incident"]] = r
                for other in r.get("also_from") or []:
                    self._learned.setdefault(other, r)
        if not self.resume:
            self.ev.write_declared(self.declaration(test, digest, dev_digest, len(dev)))  # before any phase
        else:
            self.log(f"resume {self.run_id}: the unit declaration was written by the first attempt (not rewritten)")
        phases: dict[str, list[dict[str, Any]]] = {}
        # 1. before
        phases["before"] = self._items("before", test, self.propose("before", test, None))
        self._write_phase("before", phases["before"])
        self.log(f"before: {self.summary(phases['before'])}")
        # 2. practice (dev only): each incident is verified and learned as soon as it completes
        got = self.propose("practice", dev, self.store,
                           **({"on_done": self.learn_now} if _accepts(self.propose, "on_done") else {}))
        for inc in dev:  # a backend that did not call on_done (or a missed incident): learn now
            if inc["incident_id"] in got and inc["incident_id"] not in self._learned:
                self.learn_now(inc, got[inc["incident_id"]][0])
        practice = self._items("practice", dev, got)
        for it in practice:
            inc = it["_inc"]
            rec = self._learned.get(inc["incident_id"])
            it["extra"]["learned"] = rec.get("lesson_id") if rec and rec["status"] == "verified" else None
            it["extra"]["not_learned"] = rec.get("reason") if rec else "incomplete: no proposals"
            if self.offline_labels:  # offline only: the dev split's ground-truth inverse, verified like any fix
                lv = L.verify(inc, [L.inverse_fix(inc), "", ""], self.targets)
                self.store.learn(inc, lv["change_texts"], lv, source="practice:dev-label")
        phases["practice"] = practice
        self._write_phase("practice", practice)
        overlap = self.store.guard.overlap(self.store.records)
        if overlap:
            raise L.LeakageError(f"{overlap} stored lessons overlap the test split")
        self.log(f"practice: {self.summary(practice)}; lessons {len(self.store.lessons)}, "
                 f"rejected {len(self.store.rejected)}")
        # 3. after, 4. control
        phases["after"] = self._items("after", test, self.propose("after", test, self.store))
        self._write_phase("after", phases["after"])
        self.log(f"after: {self.summary(phases['after'])}")
        phases["control"] = self._items("control", test, self.propose("control", test, None))
        self._write_phase("control", phases["control"])
        self.log(f"control: {self.summary(phases['control'])}")
        boot = self.targets["bootstrap"]

        def by_id(name: str) -> dict[str, float]:
            return {it["extra"]["incident_id"]: it["score"] for it in phases[name]}

        def pb(a: str, b: str) -> dict[str, Any] | None:
            return paired_bootstrap(by_id(a), by_id(b), resamples=int(boot["resamples"]), seed=int(boot["seed"]),
                                    ci=float(boot["ci"]))

        gain, noise = pb("after", "before"), pb("control", "before")
        value = gain["ci_low"] if gain else float("-inf")
        result = {
            "run_id": self.run_id, "synthetic": self.synthetic, "unit": UNIT, "metric": TARGET["metric"],
            "value": value, "target": dict(TARGET), "passed": bool(gain) and value > TARGET["value"],
            "n_items": sum(len(phases[p]) for p in ("before", "after", "control")),
            "checked_at": _utc(), "checker": {"script": SCRIPT, "code_sha": _code_sha()},
            "details": {"gain_after_vs_before": gain, "gain_control_vs_before": noise,
                        "learning_attributable": bool(gain and noise and gain["ci_low"] > 0 and not noise["excludes_zero"]),
                        "method": METHOD, "bootstrap": boot,
                        "phases": {p: self.summary(phases[p]) for p in PHASES},
                        "lessons": {"verified": len(self.store.lessons), "rejected": len(self.store.rejected),
                                    "test_overlap": overlap},
                        "mode": self.mode, "ticket": TICKET if self.mode == "live" else None,
                        "workers": self.workers,
                        "incomplete": {p: sum(1 for it in phases[p] if it["extra"]["incomplete"]) for p in PHASES}},
        }
        self.ev.write_result(result)
        return {"result": result, "phases": phases}


def _accepts(fn: Callable[..., Any], name: str) -> bool:
    import inspect
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _utc() -> str:
    from t3.suite.common import utc_now_iso
    return utc_now_iso()


def _code_sha() -> str:
    from t3.evals.evshim import code_sha
    return code_sha()


# --------------------------------------------------------------------------- backends

def offline_backend(targets: dict[str, Any]) -> Callable[..., dict[str, tuple[list[str], float, list[str]]]]:
    def propose(phase: str, incidents: list[dict[str, Any]], store: L.LessonStore | None, on_done=None):
        op = LessonGuided(store)
        out = {}
        for inc in incidents:
            t0 = time.perf_counter()
            props = play_offline(inc, op, targets)
            out[inc["incident_id"]] = (props, (time.perf_counter() - t0) * 1000.0, list(op.used))
            if on_done is not None:
                on_done(inc, props)
        return out
    return propose


def live_backend(*, base_run_id: str, gateway: Any, ddb: Any, evlib: Any, cfg: dict[str, Any],
                 targets: dict[str, Any], model_id: str, digests: dict[str, str], timeout_s: float = 300.0,
                 workers: int = 1, checkpoint_dir: Path | None = None, log: Callable[[str], None] = print,
                 unit: str = UNIT, patch_declaration: Callable[[dict[str, Any], str], None] | None = None):
    """The governed swarm through the checkpoint (LiveSwarm.run_incident); only the ticket changes, and every
    ticket carries the opaque label (``opaque_ticket``), never the suite's incident id.

    Crash safety: each phase's sub-run declares before its first call and LiveSwarm writes an item record per
    case as it ends. Each incident's proposals are also checkpointed to ``checkpoint_dir`` as soon as it ends;
    a re-run with the same run id skips finished incidents and runs the rest under a fresh sub-run
    (``<run>.<phase>.r<n>``, its own declaration) so no case id is reused.

    ``unit`` names each phase's sub-run (``<unit>.<phase>``); ``patch_declaration(decl, phase)`` lets another
    unit (t3/evals/confirm.py) state its own suite in each sub-run's declaration before it is written."""
    from t3.evals.live_swarm import LiveSwarm

    ck_dir = Path(checkpoint_dir) if checkpoint_dir else None

    def propose(phase: str, incidents: list[dict[str, Any]], store: L.LessonStore | None, on_done=None):
        used: dict[str, list[str]] = {}
        lock = threading.Lock()
        ck_path = ck_dir / f"{base_run_id}.{phase}.checkpoint.jsonl" if ck_dir else None
        done: dict[str, tuple[list[str], float, list[str]]] = {}
        attempt = 0
        if ck_path and ck_path.exists():
            for line in ck_path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("kind") == "attempt":
                        attempt = max(attempt, int(r["attempt"]) + 1)
                    else:
                        done[r["incident_id"]] = (r["proposals"], float(r["wall_ms"]), r["lessons_used"])
        todo = [(idx, inc) for idx, inc in enumerate(incidents) if inc["incident_id"] not in done]
        if done:
            log(f"{phase}: resuming, {len(done)} incidents already done, {len(todo)} to run (attempt {attempt})")
        if not todo:
            return done

        def ticket(incident: dict[str, Any], run_id: str) -> str:
            base = opaque_ticket(incident, run_id)
            if store is None:
                return base
            with lock:
                hits = store.retrieve(L.symptoms_of(incident), k=3, min_similarity=LESSON_MIN_SIM)
            used[incident["incident_id"]] = [h[0]["lesson_id"] for h in hits]
            block = L.lesson_block(hits, LESSON_MIN_SIM)
            return f"{base}\n\n{block}" if block else base

        sub = f"{base_run_id}.{phase}" + (f".r{attempt}" if attempt else "")
        live = LiveSwarm(run_id=sub, gateway=gateway, ddb=ddb, evlib=evlib, cfg=cfg,
                         targets=targets, model_id=model_id, timeout_s=timeout_s, ticket=ticket)
        steps = targets["episode"]["operator_steps"]
        decl = live.declaration([inc for _, inc in todo], digests["dev" if phase == "practice" else "test"], steps)
        if phase == "practice":
            decl["subset"] = "split=dev (practice; t3/evals/suite/incidents-dev.jsonl)"
        decl["unit"] = f"{unit}.{phase}"
        conf = decl.setdefault("twin", {}).setdefault("config", {})
        conf["ticket"] = TICKET
        conf["workers"] = workers
        if attempt:
            conf["resumes"] = f"{base_run_id}.{phase}" + (f".r{attempt - 1}" if attempt > 1 else "")
        if patch_declaration is not None:
            patch_declaration(decl, phase)
        evlib.write_declared(decl)  # each phase's cases are their own declared run, declared before any call
        if ck_path:
            ck_path.parent.mkdir(parents=True, exist_ok=True)
            with ck_path.open("a") as f:
                f.write(json.dumps({"kind": "attempt", "attempt": attempt, "sub_run": sub}) + "\n")

        def one(arg: tuple[int, dict[str, Any]]) -> None:
            idx, inc = arg
            live.run_incident(inc, idx, steps)
            with getattr(live, "_lock", lock):
                rows = sorted((r for r in live.rows if r["incident_id"] == inc["incident_id"]),
                              key=lambda r: r["step"])
                items = [i for i in live.items if i["extra"]["incident_id"] == inc["incident_id"]]
            res = ([r["change_text"] if r["ok"] else "" for r in rows],
                   sum(float(i.get("latency_ms") or 0) for i in items), used.get(inc["incident_id"], []))
            with lock:
                done[inc["incident_id"]] = res
                if ck_path:
                    with ck_path.open("a") as f:
                        f.write(json.dumps({"incident_id": inc["incident_id"], "proposals": res[0],
                                            "wall_ms": res[1], "lessons_used": res[2], "sub_run": sub}) + "\n")
            if on_done is not None:
                on_done(inc, res[0])

        if workers <= 1:
            for arg in todo:
                one(arg)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(one, todo))
        return done
    return propose


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id")
    ap.add_argument("--evidence-dir", help="offline: local evidence directory")
    ap.add_argument("--lessons", help="where the lesson store (JSONL) is written (default: evidence dir)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--live", action="store_true", help="the governed swarm through the checkpoint (spends)")
    ap.add_argument("--practice-limit", type=int, help="only the first N dev incidents (a trial)")
    ap.add_argument("--no-dev-labels", action="store_true", help="offline: learn from own attempts only")
    ap.add_argument("--tokens-dir")
    ap.add_argument("--resume", action="store_true", help="live: continue a crashed run with the same --run-id")
    ap.add_argument("--workers", type=int, default=1, help="live: incidents in parallel within a phase")
    ap.add_argument("--timeout", type=float, default=300.0, help="live: seconds to wait for the lead per case")
    a = ap.parse_args(argv)
    targets = load_targets()
    steps = targets["episode"]["operator_steps"]
    test, digest = test_masked()
    dev, dev_digest = dev_incidents()
    if a.practice_limit:
        dev = dev[: a.practice_limit]
    est = cost_estimate(len(test), len(dev), steps)
    if a.dry_run:
        per_case_s = 70.0
        wall = {f"workers={w}": f"{sum(-(-n // w) * steps for n in (len(test), len(dev), len(test), len(test))) * per_case_s / 3600:.1f} h"
                for w in (1, 3, 6)}
        print(json.dumps({"unit": UNIT, "target": TARGET, "test_masked": len(test), "dev": len(dev),
                          "ticket": TICKET, "live_practice": "own attempts only, learned per incident",
                          "live_gateway_cases": est,
                          "wall_time_estimate": {"assumes_s_per_case": per_case_s, **wall}}, indent=1))
        return 0
    from t3.evals.evshim import new_run_id
    if a.resume and not a.run_id:
        raise SystemExit("--resume needs the --run-id of the run to continue")
    run_id = a.run_id or new_run_id(UNIT)
    if not a.live:
        from t3.evals.evshim import Evidence
        if not a.evidence_dir:
            raise SystemExit("offline needs --evidence-dir (local)")
        ev = Evidence(a.evidence_dir)
        store = L.LessonStore(a.lessons or Path(a.evidence_dir) / "lessons" / f"{run_id}.jsonl")
        unit = SelfImprove(run_id=run_id, ev=ev, propose=offline_backend(targets),
                           model_id="standin:config-revert+lessons-v1", synthetic=True, store=store,
                           targets=targets, offline_labels=not a.no_dev_labels, mode="offline")
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
        print(f"LIVE {run_id}: {est['cases_total']} Gateway cases")
        store = L.LessonStore(a.lessons or common.CALLS_DIR / f"{run_id}.lessons.jsonl",
                              mirror=evlib.default_store())
        unit = SelfImprove(run_id=run_id, ev=evlib, model_id=model_id, synthetic=False, store=store,
                           targets=targets, offline_labels=False, mode="live",
                           workers=a.workers, resume=a.resume, checkpoint_dir=common.CALLS_DIR,
                           propose=live_backend(base_run_id=run_id, gateway=gw, ddb=session.client("dynamodb"),
                                                evlib=evlib, cfg=cfg, targets=targets, model_id=model_id,
                                                digests={"test": digest, "dev": dev_digest}, timeout_s=a.timeout,
                                                workers=a.workers, checkpoint_dir=common.CALLS_DIR))
    out = unit.run(test, digest, dev, dev_digest)
    if a.live:
        store.publish(f"outputs/{run_id}/lessons.jsonl")
    r = out["result"]
    print(json.dumps({"run_id": run_id, "value": r["value"], "passed": r["passed"],
                      "phases": r["details"]["phases"], "control": r["details"]["gain_control_vs_before"],
                      "lessons": r["details"]["lessons"]}, indent=1, default=str))
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
