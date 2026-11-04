"""The exact-inverse oracle: the per-cell ceiling of the remediation evaluation.

It reads the injected fault from the incident (which no real operator can) and undoes it exactly at the
first operator step, then does nothing. If even this fix does not restore every cell (percell-v1) on an
incident, no operator can, and the per-cell rule or the grade step is too strict there. It is reported on
every set and is never part of pass/fail.

On the same network without the fault (the evaluation's healthy check) it is told there is no fault, so it
proposes nothing: ``knows_fault`` lets the harness hand it the fault-free incident.
"""

from __future__ import annotations

from typing import Any


class ExactInverse:
    name = "oracle_inverse"
    model_id = "oracle-exact-inverse-v1"
    synthetic = False
    knows_fault = True

    def __init__(self) -> None:
        self._inverse: list[dict[str, Any]] = []
        self._done = False

    def begin(self, incident: dict[str, Any], ctx: dict[str, Any] | None = None) -> None:
        self._inverse = [{"cell": int(f["cell"]), "d_power_db": -float(f["d_power_db"]),
                          "d_tilt_deg": -float(f["d_tilt_deg"])} for f in incident.get("fault") or []]
        self._done = False

    def act(self, state: dict[str, Any], history: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        if self._done or history:
            return None
        self._done = True
        return [dict(c) for c in self._inverse] or None
