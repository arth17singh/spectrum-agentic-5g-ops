"""The ``swarm`` operator: drives the live governed system through the checkpoint's front door.

The harness keeps its own twin and grades on it. The swarm runs its own case (in the cases table,
replayed from ``(scenario, seed, config, action_log)``), so the adapter opens each case at the
post-fault state by handing over the fault step as the case's starting action log. Each harness step
asks the client for the swarm's next decision and gets back the change it actually got APPLIED via
``ran___apply_change`` (after Cedar, the students and the human approval), as the canonical A2
string. Refused, unapproved or timed-out steps come back as no change. The harness then applies the
same requested change on its own twin; both twins are deterministic replays, so they land on the same
state (``measured_kpis`` from the client, when given, are compared and any mismatch is recorded).

Client protocol (the lead implements it over the runner; ``DryRunSwarmClient`` is a fake):

    client.open_case(case: dict) -> case_id
        case = {incident_id, run_id, scenario, seed, preset, start_action_log: [commissioning, fault],
                task_text}
    client.step(case_id, state, history) -> {
        "change_text": "cell=1;d_power_db=+0.0;d_tilt_deg=-5.0" | "" | None,   # what was applied
        "applied": bool, "approval_id": str | None, "decisions": [...]?, "measured_kpis": {...}? }
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


class SwarmOperator:
    name = "swarm"

    def __init__(self, client: Any, run_id: str = "adhoc") -> None:
        self.client = client
        self.run_id = run_id
        self.synthetic = bool(getattr(client, "synthetic", False))
        self.model_id = getattr(client, "model_id", "team13-swarm")
        self.case_id: str | None = None
        self.trace: list[dict[str, Any]] = []

    def begin(self, incident: dict[str, Any], ctx: dict[str, Any]) -> None:
        self.run_id = ctx.get("run_id", self.run_id)
        self.trace = []
        self.case_id = self.client.open_case({
            "incident_id": incident["incident_id"],
            "run_id": self.run_id,
            "scenario": incident["scenario"],
            "seed": incident["seed"],
            "preset": incident["preset"],
            "start_action_log": [incident.get("commissioning", []), incident["fault"]],
            "task_text": task_text(incident, self.run_id),
        })

    def act(self, state: dict[str, Any], history: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        out = self.client.step(self.case_id, state, history) or {}
        text = out.get("change_text") or ""
        applied = bool(out.get("applied")) and bool(text)
        change = nonzero(parse(text)) if applied else []
        self.trace.append({
            "case_id": self.case_id,
            "change_text": canon(change) if change else "",
            "applied": applied,
            "approval_id": out.get("approval_id"),
            "measured_kpis": out.get("measured_kpis"),
            "n_decisions": len(out.get("decisions") or []),
        })
        return change

    def end(self) -> None:
        close = getattr(self.client, "close_case", None)
        if callable(close) and self.case_id:
            close(self.case_id)

    def extra(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "swarm_steps": self.trace}


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
        if not change:
            return {"change_text": "", "applied": False, "approval_id": None}
        return {"change_text": canon(change), "applied": True, "approval_id": f"chg-dryrun-{case_id}-{len(history)}"}

    def close_case(self, case_id: str) -> None:
        self.cases.pop(case_id, None)


def load_client(spec: str | None) -> Any:
    """``module:factory`` -> factory(); None -> the dry-run fake."""
    if not spec:
        return DryRunSwarmClient()
    mod, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod), attr)()
