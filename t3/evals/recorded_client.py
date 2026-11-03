"""A swarm client that REPLAYS the proposals recorded by the live phase (t3/evals/live_swarm.py).

The scored swarm run (t3/evals/run.py --operator swarm --mode proposal --subset masked) runs on x86 in a
SageMaker job, where the frozen suite replays to its tolerance. The swarm itself acted earlier, live, through
the checkpoint's front door; it cannot run in the job (the job role reaches neither DynamoDB nor AgentCore).
So the job loads the recording and, at every harness step, returns exactly what the live swarm proposed for
that incident and step. Before answering it checks that the harness asks for the same case the live swarm saw:
the same number of steps so far, and the same earlier proposals. A mismatch stops the run.

    --swarm-client t3.evals.recorded_client:factory
    T3_SWARM_RECORDING=<path to swarm-actions.jsonl>   (default: the job's `recording` input channel)
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any

from t3.evals.changes import canon, nonzero

DEFAULT_GLOB = "/opt/ml/input/data/recording/*.jsonl"


def step_text(change: Any) -> str:
    """One logged step (a change list, a canonical string, or a no-op) -> canonical text ("" for no-op)."""
    if change in (None, "", []):
        return ""
    cells = nonzero(change)
    return canon(cells) if cells else ""


class RecordedSwarmClient:
    synthetic = False

    def __init__(self, rows: list[dict[str, Any]], fault_step: int = 2):
        self.rows = {(r["incident_id"], int(r["step"])): r for r in rows}
        if len(self.rows) != len(rows):
            raise ValueError("the recording has two rows for one (incident, step)")
        models = {r.get("model_id") for r in rows}
        self.model_id = models.pop() if len(models) == 1 else "team13-swarm"
        self.live_run_ids = sorted({r.get("run_id") for r in rows if r.get("run_id")})
        self.fault_step = fault_step
        self.open: dict[str, dict[str, Any]] = {}

    def open_case(self, case: dict[str, Any]) -> str:
        key = (case["incident_id"], int(case["step"]))
        if key not in self.rows:
            raise KeyError(f"no recorded live case for incident {key[0]} step {key[1]}")
        row = self.rows[key]
        log = [step_text(c) for c in case["start_action_log"]]
        if len(log) != self.fault_step + key[1]:
            raise ValueError(f"{key}: the harness is at step {len(log)}, the recording at {self.fault_step + key[1]}")
        earlier = log[self.fault_step:]
        recorded = [self.rows[(key[0], k)]["change_text"] or "" for k in range(key[1])]
        if [step_text(x) for x in recorded] != earlier:
            raise ValueError(f"{key}: earlier proposals differ: harness {earlier}, recording {recorded}")
        self.open[row["case_id"]] = row
        return row["case_id"]

    def step(self, case_id: str, state: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
        row = self.open[case_id]
        return {"change_text": row.get("change_text") or "", "applied": False, "refused": bool(row.get("refused")),
                "approval_id": row.get("approval_id")}

    def close_case(self, case_id: str) -> None:
        self.open.pop(case_id, None)


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def factory() -> RecordedSwarmClient:
    path = os.environ.get("T3_SWARM_RECORDING")
    if not path:
        found = sorted(glob.glob(DEFAULT_GLOB))
        if len(found) != 1:
            raise FileNotFoundError(f"set T3_SWARM_RECORDING (found {found} under {DEFAULT_GLOB})")
        path = found[0]
    return RecordedSwarmClient(load_rows(path))
