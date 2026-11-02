"""Operators: anything that looks at the network and proposes a change, behind one interface.

    op.begin(incident, ctx)            optional; called once per incident before the first act
    op.act(state, history) -> list[{cell, d_power_db, d_tilt_deg}] | None
    op.end()                           optional

``state`` is the twin state dict (``ran_rl.twin.base`` schema) after the previous step. ``history``
is one entry per earlier operator step: ``{step, change, kpis}``. Returning ``None`` or ``[]`` means
"no change this step"; the harness still advances the twin, so every operator is graded at the same
step. ``ctx`` gives ``twin`` (the harness's live twin, read only), ``targets`` and, for the oracle
only, the incident's healthy and unfixed KPI trajectories.

Operators: ``no_action``, ``rule_baseline`` (the SDK's ``RuleBasedController``), ``config_revert`` (revert the odd cell
out to the median configuration: the simple heuristic), ``ppo`` (W9's
``t1.rl.policy`` via ``.propose(state)``, injected; a labelled stub until merge), ``oracle``
(best-of-K single-cell changes, each replayed on a fresh copy of the twin: the upper bound) and
``swarm`` (``t3.evals.swarm.SwarmOperator``).
"""

from __future__ import annotations

import itertools
from typing import Any, Protocol, runtime_checkable


from t3.evals.changes import nonzero
from t3.evals.graders import grade_kpis
from t3.evals.twin_backend import ReplayTwin, graded, requested_changes


@runtime_checkable
class Operator(Protocol):
    name: str
    model_id: str
    synthetic: bool

    def act(self, state: dict[str, Any], history: list[dict[str, Any]]) -> list[dict[str, Any]] | None: ...


class NoAction:
    name = "no_action"
    model_id = "no_action"
    synthetic = False

    def act(self, state, history):
        return None


class RuleBaseline:
    """The SDK's rule-based baseline controller (``ran_rl.evaluation.baseline``), unmodified."""

    name = "rule_baseline"
    synthetic = False

    def __init__(self, safety: dict[str, Any] | None = None) -> None:
        import importlib.metadata

        self.model_id = f"ran_rl.evaluation.baseline:RuleBasedController@{importlib.metadata.version('ran-rl')}"
        self._safety = safety

    def begin(self, incident, ctx):
        self._safety = ctx["twin"].config["safety"]

    def act(self, state, history):
        from ran_rl.env.config import SafetyBounds
        from ran_rl.env.observation import scale_action
        from ran_rl.evaluation.baseline import RuleBasedController

        bounds = SafetyBounds(**(self._safety or {}))
        unit = RuleBasedController(bounds).act(state)
        raw = scale_action(unit, bounds)
        n = len(state["power_dbm"])
        out = []
        for c in range(n):
            dp, dt = float(raw[c]), float(raw[n + c])
            if dp or dt:
                out.append({"cell": c, "d_power_db": dp, "d_tilt_deg": dt})
        return out


class StubPolicy:
    """STUB for W9's PPO policy (``t1.rl.policy``) until merge: proposes nothing. Synthetic."""

    model_id = "ppo-stub"
    version = "stub"
    stub = True

    def propose(self, state):
        return []


def load_ppo(path: str | None):
    """W9's ``t1.rl.policy.load(path)`` when importable and a path is given, else the stub."""
    if path:
        from t1.rl.policy import load  # type: ignore[import-not-found]

        return load(path)
    try:
        import t1.rl.policy  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        pass
    return StubPolicy()


class PPO:
    """Wraps any object with ``.propose(state) -> change list | canonical string``."""

    name = "ppo"

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        mid = getattr(policy, "model_id", "ppo")
        ver = getattr(policy, "version", None)
        self.model_id = f"{mid}@{ver}" if ver else str(mid)
        self.synthetic = bool(getattr(policy, "stub", False))

    def begin(self, incident, ctx):
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

    def act(self, state, history):
        return nonzero(self.policy.propose(state))


class ConfigRevert:
    """A simple non-learned heuristic: move the cell whose power or tilt sits furthest from the other
    cells' median back towards that median (one cell per step, within the per-step limits).

    It works because every preset starts all cells at the same power and tilt, so a misconfigured
    cell stands out in the configuration itself. It is the "simpler option" any swarm must beat.
    """

    name = "config_revert"
    model_id = "config-revert-median-v1"
    synthetic = False

    def act(self, state, history):
        import statistics

        p, t = list(state["power_dbm"]), list(state["tilt_deg"])
        best = None
        for c in range(len(p)):
            dp = max(-6.0, min(8.0, statistics.median(p[:c] + p[c + 1:]) - p[c]))
            dt = max(-5.0, min(5.0, statistics.median(t[:c] + t[c + 1:]) - t[c]))
            score = abs(dp) / 6.0 + abs(dt) / 5.0
            if score > 0.05 and (best is None or score > best[0]):
                best = (score, c, round(dp, 1), round(dt, 1))
        if best is None:
            return None
        return [{"cell": best[1], "d_power_db": best[2], "d_tilt_deg": best[3]}]


# --- oracle ----------------------------------------------------------------------------------------

POWER_GRID = (-6.0, -4.0, -3.0, 0.0, 3.0, 4.0, 6.0)
TILT_GRID = (-5.0, -4.0, -3.0, -1.0, 0.0, 1.0, 3.0, 4.0, 5.0)


def candidate_grid(n_cells: int = 4) -> list[list[dict[str, Any]]]:
    """Every single-cell change on the grid (power and tilt together allowed), no empty change."""
    out = []
    for cell in range(n_cells):
        for dp, dt in itertools.product(POWER_GRID, TILT_GRID):
            if dp or dt:
                out.append([{"cell": cell, "d_power_db": dp, "d_tilt_deg": dt}])
    return out


class Oracle:
    """Greedy best-of-K: at each step replay every candidate (and no change) on a fresh copy of the
    twin, grade it against that step's healthy and unfixed KPIs with the real grader, keep the best.

    It sees the next step's UE positions through the replay, which no deployable operator can: it is
    the upper bound, not a competitor. Copies are made with ``ReplayTwin.from_log`` (public API).
    """

    name = "oracle"
    synthetic = False

    def __init__(self, grid: list[list[dict[str, Any]]] | None = None) -> None:
        self.grid = grid if grid is not None else candidate_grid()
        self.model_id = f"oracle-best-of-{len(self.grid)}-greedy"

    def begin(self, incident, ctx):
        self.inc = incident
        self.twin = ctx["twin"]
        self.targets = ctx["targets"]

    def _score(self, kp, step):
        g = grade_kpis(kp, self.inc["healthy_kpis_by_step"][step], self.inc["unfixed_kpis_by_step"][step],
                       self.targets)
        tol = self.targets["tolerance"]
        margin = sum(
            min(p["final"] - p["healthy"], 0.0) / tol[k] for k, p in g["per_kpi"].items()
        )
        return (g["success"], g["recovery"] if g["recovery"] is not None else 0.0, margin)

    def act(self, state, history):
        prefix = requested_changes(self.twin)
        step = len(prefix) + 1
        tw = self.twin
        best_change: list[dict[str, Any]] = []
        best = self._score(graded(_replay(tw, prefix + [[]])), step)
        for cand in self.grid:
            s = self._score(graded(_replay(tw, prefix + [cand])), step)
            if s > best:
                best, best_change = s, cand
        return best_change


def _replay(tw: Any, changes: list[list[dict[str, Any]]]) -> dict[str, Any]:
    t = ReplayTwin(tw.scenario, tw.seed, tw.config)
    state = t.reset()
    for ch in changes:
        state = t.apply(ch)
    return state


def make(name: str, **kw: Any) -> Any:
    if name == "no_action":
        return NoAction()
    if name == "rule_baseline":
        return RuleBaseline()
    if name == "ppo":
        return PPO(kw.get("policy") or load_ppo(kw.get("ppo_path")))
    if name == "oracle":
        return Oracle()
    if name == "config_revert":
        return ConfigRevert()
    if name == "swarm":
        from t3.evals.swarm import SwarmOperator, load_client

        return SwarmOperator(kw.get("client") or load_client(kw.get("swarm_client")))
    raise ValueError(f"unknown operator {name!r}")


OPERATORS = ("no_action", "rule_baseline", "config_revert", "ppo", "oracle", "swarm")
