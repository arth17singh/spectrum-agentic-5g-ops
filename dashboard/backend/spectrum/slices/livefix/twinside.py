"""The twin-side helper process (t3/evals/livefix_worker.py) and the one place that maps names between parts.

The console's Python has no Maveric twin, so fault incidents, cell health, confidence and lessons are computed by
a long-lived worker started with LIVEFIX_T3_PYTHON (an interpreter with ran-rl). Without it those features say
"not available" and the dev-incident picker still works.

Mappings (UI <-> t3.evals), defined only here:
  fault  UI FaultKey            -> t3.evals.faults key
  health faults green/amber/red -> UI good/degraded/critical
  tier   confidence one-person/two-person/hold -> UI one_person/two_person/hold
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from typing import Any

from .suite import REPO

FAULT_KEYS = {"tilt_drift": "tilt_drift", "pa_fault": "pa_fault", "bad_config": "bad_push",
              "hardware": "hw_failure", "cascade": "cascade"}
TEMPLATE = {"tilt_drift": "tilt+5", "pa_fault": "power-6", "bad_config": "power-3+tilt+3",
            "hardware": "power-6+tilt+5", "cascade": "two-cell:power-6|tilt+5"}
HEALTH = {"green": "good", "amber": "degraded", "red": "critical"}
TIER = {"one-person": "one_person", "two-person": "two_person", "hold": "hold"}
# Offline calibration of the confidence score (t3/evals/calibration.py, recorded swarm proposals): MISSED.
CALIBRATION = {"calibrated": False, "ece": 0.167, "target_ece": 0.15, "brier": 0.179, "base_rate_brier": 0.113,
               "n": 23, "label": "calibration: ECE 0.167, target ≤0.15, MISSED (n=23, offline)"}


class Unavailable(RuntimeError):
    pass


class TwinSide:
    def __init__(self, python: str | None = None, timeout_s: float = 120.0):
        self.python = python if python is not None else os.environ.get("LIVEFIX_T3_PYTHON") or ""
        self.timeout_s = timeout_s
        self._p: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._id = 0

    @property
    def available(self) -> bool:
        return bool(self.python) and os.path.exists(self.python)

    def _proc(self) -> subprocess.Popen:
        if self._p is None or self._p.poll() is not None:
            if not self.available:
                raise Unavailable("the twin-side helper is not configured (set LIVEFIX_T3_PYTHON to a Python with "
                                  "ran-rl)")
            env = dict(os.environ, PYTHONPATH=f"{REPO}:{REPO / 'contracts'}")
            self._p = subprocess.Popen([self.python, "-m", "t3.evals.livefix_worker"], cwd=str(REPO), env=env,
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                       text=True, bufsize=1)
        return self._p

    def call(self, op: str, **args: Any) -> Any:
        with self._lock:
            p = self._proc()
            self._id += 1
            p.stdin.write(json.dumps({"id": self._id, "op": op, "args": args}, default=str) + "\n")
            p.stdin.flush()
            line = p.stdout.readline()
            if not line:
                self._p = None
                raise Unavailable("the twin-side helper stopped")
            res = json.loads(line)
        if not res.get("ok"):
            raise RuntimeError(res.get("error") or "twin-side helper error")
        return res["result"]

    def close(self) -> None:
        if self._p is not None and self._p.poll() is None:
            self._p.terminate()
