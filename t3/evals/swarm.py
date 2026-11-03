"""The ``swarm`` operator: drives the live governed system through the checkpoint's front door.

The harness keeps its own twin and grades on it; the swarm runs its own cases (replayed from
``(scenario, seed, config, action_log)``). Two modes:

``proposal`` (the scored mode; no person approves anything)
    Every harness step opens ONE front-door case at the harness's current state: its
    ``start_action_log`` is every change applied so far (commissioning, fault, earlier proposals).
    The swarm works the case up to the checkpoint's approval refusal on ``ran___apply_change`` and
    STOPS there. The client returns the PROPOSED canonical change string (and the refusal's approval
    id). The harness grades that proposal with its own twin replay; it never goes through the
    governed apply tool. ``proposal_steps`` (default 3, the episode length) caps the cases per
    incident; later steps are no-ops.

``full`` (2-3 demo cases with real approvals)
    One case per incident, opened at the post-fault state. Each step returns the change the swarm got
    APPLIED via ``ran___apply_change`` after Cedar, the students and the human approval; refused,
    unapproved or timed-out steps count as no change. The harness re-applies it on its own twin.

In both modes, ``measured_kpis`` from the client (if given) are compared with the harness twin and any
mismatch is recorded on the item (``extra.kpi_mismatch``). Every item carries ``extra.mode``.

Client protocol (the lead implements it over the runner; ``DryRunSwarmClient`` is a fake):

    client.open_case(case: dict) -> case_id
        case = {incident_id, run_id, mode, step, scenario, seed, preset,
                start_action_log: [[change], ...], task_text}
    client.step(case_id, state, history) -> {
        "change_text": "cell=1;d_power_db=+0.0;d_tilt_deg=-5.0" | "" | None,
                       # full: what was applied; proposal: what was proposed before the refusal
        "applied": bool,              # full mode
        "refused": bool,              # proposal mode: apply was refused pending approval
        "approval_id": str | None, "decisions": [...]?, "measured_kpis": {...}? }
    client.close_case(case_id)                       # optional

Load a real client with ``--swarm-client module:factory`` (``factory()`` returns the client; it reads
endpoints and tokens from environment variables, never from files).
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol

from t3.evals.changes import canon, nonzero, parse

AGENTS = ("noc_lead", "ran_monitor", "ran_optimizer", "ran_explainer")  # amendment A1
APPLY_TOOL = "ran___apply_change"


class SwarmClient(Protocol):
    def open_case(self, case: dict[str, Any]) -> str: ...

    def step(self, case_id: str, state: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]: ...


def task_text(incident: dict[str, Any], run_id: str) -> str:
    """The ticket the noc_lead receives. States symptoms only: never the fault or the cell."""
    pre, post = incident["pre_fault_kpis"], incident["post_fault_kpis"]
    return (
        f"run_id={run_id} incident={incident['incident_id']}. Network KPIs dropped after the last "
        f"configuration step: coverage {pre['coverage_ratio']:.3f} -> {post['coverage_ratio']:.3f}, "
        f"SLA {pre['sla_ratio']:.3f} -> {post['sla_ratio']:.3f}, p10 SINR {pre['sinr_p10_db']:.1f} -> "
        f"{post['sinr_p10_db']:.1f} dB. Restore service with at most 3 changes. Observe first, use "
        f"preflight, and send every change as the canonical string."
    )


MODES = ("proposal", "full")


class SwarmOperator:
    name = "swarm"

    def __init__(self, client: Any, run_id: str = "adhoc", mode: str = "proposal", proposal_steps: int = 3) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.client = client
        self.run_id = run_id
        self.mode = mode
        self.proposal_steps = proposal_steps
        self.synthetic = bool(getattr(client, "synthetic", False))
        self.model_id = getattr(client, "model_id", "team13-swarm")
        self.case_id: str | None = None
        self.case_ids: list[str] = []
        self.trace: list[dict[str, Any]] = []

    def _open(self, start_log: list[list[dict[str, Any]]], step: int) -> str:
        cid = self.client.open_case({
            "incident_id": self.incident["incident_id"],
            "run_id": self.run_id,
            "mode": self.mode,
            "step": step,
            "scenario": self.incident["scenario"],
            "seed": self.incident["seed"],
            "preset": self.incident["preset"],
            "start_action_log": start_log,
            "task_text": task_text(self.incident, self.run_id),
        })
        self.case_ids.append(cid)
        return cid

    def begin(self, incident: dict[str, Any], ctx: dict[str, Any]) -> None:
        self.run_id = ctx.get("run_id", self.run_id)
        self.incident = incident
        self.twin = ctx.get("twin")
        self.trace = []
        self.case_ids = []
        self.case_id = None
        if self.mode == "full":
            self.case_id = self._open([incident.get("commissioning", []), incident["fault"]], 0)

    def act(self, state: dict[str, Any], history: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        if self.mode == "proposal":
            if len(history) >= self.proposal_steps:
                return []
            start_log = [e["change"] for e in self.twin.action_log] if self.twin is not None else []
            self.case_id = self._open(start_log, len(history))
        out = self.client.step(self.case_id, state, history) or {}
        text = out.get("change_text") or ""
        if self.mode == "full":
            take = bool(out.get("applied")) and bool(text)
        else:
            take = bool(text)  # the proposal itself; nothing was applied by the governed tool
        change = nonzero(parse(text)) if take else []
        self.trace.append({
            "mode": self.mode,
            "case_id": self.case_id,
            "change_text": canon(change) if change else "",
            "applied": bool(out.get("applied")),
            "refused": bool(out.get("refused")),
            "approval_id": out.get("approval_id"),
            "measured_kpis": out.get("measured_kpis"),
            "n_decisions": len(out.get("decisions") or []),
        })
        if self.mode == "proposal":
            self._close(self.case_id)
        return change

    def _close(self, cid: str | None) -> None:
        close = getattr(self.client, "close_case", None)
        if callable(close) and cid:
            close(cid)

    def end(self) -> None:
        if self.mode == "full":
            self._close(self.case_id)

    def extra(self) -> dict[str, Any]:
        return {"mode": self.mode, "case_ids": list(self.case_ids), "swarm_steps": self.trace}


class DryRunSwarmClient:
    """FAKE swarm for dry runs (synthetic). No network, no model, no AWS.

    It "decides" with the ``config_revert`` heuristic and "applies" every change it proposes, as the
    canonical string. It never sees the incident's fault. Its numbers say nothing about the swarm.
    """

    synthetic = True
    model_id = "swarm-dryrun-fake"

    def __init__(self) -> None:
        self.cases: dict[str, dict[str, Any]] = {}

    def open_case(self, case: dict[str, Any]) -> str:
        cid = f"case-dryrun-{len(self.cases):02d}"
        self.cases[cid] = case
        return cid

    def step(self, case_id: str, state: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
        from t3.evals.operators import ConfigRevert

        change = ConfigRevert().act(state, history)
        mode = self.cases[case_id].get("mode", "full")
        if not change:
            return {"change_text": "", "applied": False, "refused": False, "approval_id": None}
        aid = f"chg-dryrun-{case_id}-{len(history)}"
        if mode == "proposal":  # the real checkpoint refuses apply pending approval; we stop there
            return {"change_text": canon(change), "applied": False, "refused": True, "approval_id": aid}
        return {"change_text": canon(change), "applied": True, "refused": False, "approval_id": aid}

    def close_case(self, case_id: str) -> None:
        self.cases.pop(case_id, None)


def load_client(spec: str | None) -> Any:
    """``module:factory`` -> factory(); None -> the dry-run fake."""
    if not spec:
        return DryRunSwarmClient()
    mod, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod), attr)()
