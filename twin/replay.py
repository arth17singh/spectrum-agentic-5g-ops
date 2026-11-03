"""ReplayTwin: the in-process Maveric twin, rebuilt deterministically from a case record.

    t = ReplayTwin("nominal", seed=7)             # config None -> the "official" preset
    s0 = t.reset()                                # state dict (ran_rl.twin.base schema)
    s1 = t.apply([{"cell": 1, "d_power_db": 2.0, "d_tilt_deg": -1.0}])
    t.action_log                                  # JSON-safe list, one entry per apply()
    same = ReplayTwin.from_log("nominal", 7, None, t.action_log)   # lands on the identical state

Why it is exact: ``MavericDigitalTwin`` draws every random number from one ``numpy`` Generator
seeded in ``reset(scenario, seed)``; nothing else is random. ``apply`` does what
``RANOptimizationEnv.step`` does: the requested deltas go through the SDK's ``ActionSafetyFilter``
against the current state, ``twin.apply_action`` receives the filtered vectors, then
``twin.advance(step_seconds)`` runs one step. Replaying the same (scenario, seed, config,
requested changes) therefore reproduces every float bit for bit on the same platform and library
versions. Across machines (x86 vs arm64 Lambda) tiny float differences are possible: each log
entry stores the post-step headline KPIs, and ``from_log`` raises ``ReplayMismatch`` if the replay
differs from them by more than 1e-6 relative (``verify=False`` switches that off).

Config: see ``twin.presets``. ``ReplayTwin.config`` is the fully resolved dict; store that.
"""

from __future__ import annotations

import copy
import importlib.metadata
import math
import threading
from typing import Any

from twin._change import canon_change, dec, filter_change, parse_change, with_kpis  # noqa: F401
from twin.kpis import HEADLINE, headline
from twin.presets import config_hash, resolve_config

SCHEMA = 1
_RTOL = 1e-6
_ATOL = 1e-9


class ReplayMismatch(RuntimeError):
    """A replay did not reproduce what the log says (different filtering or different KPIs)."""


def sdk_version() -> str:
    return importlib.metadata.version("ran-rl")


# --- twin construction -------------------------------------------------------------------------

_PROTOTYPES: dict[str, Any] = {}
_LOCK = threading.Lock()


def _scenarios(cfg: dict[str, Any]) -> dict[str, Any]:
    from ran_rl.twin.scenarios import Scenario, load_scenarios

    scenarios = load_scenarios()
    scenarios.update({name: Scenario(**fields) for name, fields in cfg["scenarios"].items()})
    return scenarios


def _build_twin(cfg: dict[str, Any]) -> Any:
    from ran_rl.twin.factory import MAVERIC_MODELS_DIR, MAVERIC_TOPOLOGY
    from ran_rl.twin.maveric.twin import MavericDigitalTwin, MavericTwinConfig

    return MavericDigitalTwin(
        cfg["models_dir"] or MAVERIC_MODELS_DIR,
        cfg["topology"] or MAVERIC_TOPOLOGY,
        MavericTwinConfig(**cfg["twin"]),
        scenarios=_scenarios(cfg),
    )


def _clone(proto: Any) -> Any:
    """An independent twin sharing the prototype's read-only GP models and topology.

    ``reset()`` re-creates the UE arrays, the generator, the scenario and the step counter; the four
    per-cell arrays it fills in place are copied here. The prototype itself is never reset.
    """
    twin = copy.copy(proto)
    for name in ("_power", "_tilt", "_alarm", "_asleep"):
        setattr(twin, name, getattr(proto, name).copy())
    return twin


def _new_twin(cfg: dict[str, Any], reuse_models: bool) -> Any:
    if not reuse_models:
        return _build_twin(cfg)
    key = config_hash(cfg)
    with _LOCK:
        if key not in _PROTOTYPES:
            _PROTOTYPES[key] = _build_twin(cfg)
        return _clone(_PROTOTYPES[key])


def _snapshot(twin: Any) -> Any:
    """An independent copy of a twin IN ITS CURRENT STATE (mid-episode), sharing only read-only models.

    Copies every mutable attribute ``MavericDigitalTwin`` changes in place or per step: the per-cell
    arrays, the UE arrays and motion state, and the random generator (so the copy continues the
    same random stream). Relies on ran-rl 1.2.9 attribute names; a test proves the copy and the
    original then evolve identically.
    """
    new = _clone(twin)
    for name in ("_ue", "_speed", "_heading", "_heading_mean"):
        setattr(new, name, getattr(twin, name).copy())
    new._rng = copy.deepcopy(twin._rng)
    return new


# One twin object per config for the life of the process (Lambda container): reset() per call
# instead of constructing. Only one ReplayTwin may hold it at a time (a lease).
_SHARED: dict[str, list[Any]] = {}  # key -> [twin, current lease]


def _lease_shared(cfg: dict[str, Any]) -> tuple[Any, object, str]:
    key = config_hash(cfg)
    with _LOCK:
        if key not in _SHARED:
            _SHARED[key] = [_build_twin(cfg), None]
        lease = object()
        _SHARED[key][1] = lease
        return _SHARED[key][0], lease, key


# --- ReplayTwin ---------------------------------------------------------------------------------


class ReplayTwin:
    """A Maveric twin plus the log of everything applied to it.

    Args:
        scenario: a name from the SDK's scenarios.yaml, or one the config defines
            (``twin.presets.VIAVI_SCENARIO`` for "viavi-like").
        seed: the twin seed passed to ``reset``.
        config: None (official), a preset name, or a dict (see ``twin.presets``).
        reuse_models: share the loaded GP models between instances with the same config (saves
            about 0.1 s per instance; results are identical, which the tests check).
        shared: Lambda mode. Use ONE twin object per config for the whole process and only
            ``reset()`` it per call (W3b: flat memory, about 40% faster than rebuilding). The new
            instance takes over the shared twin; an older shared instance of the same config then
            raises on use. Use ``ReplayTwin.cached(...)`` / ``from_log(..., shared=True)``.
    """

    def __init__(
        self,
        scenario: str,
        seed: int,
        config: dict[str, Any] | str | None = None,
        *,
        reuse_models: bool = True,
        shared: bool = False,
    ) -> None:
        self.scenario = scenario
        self.seed = int(seed)
        self.config = resolve_config(config)
        self._lease: object | None = None
        self._lease_key: str | None = None
        if shared:
            self._twin, self._lease, self._lease_key = _lease_shared(self.config)
        else:
            self._twin = _new_twin(self.config, reuse_models)
        self._safety: Any = None
        self._state: dict[str, Any] | None = None
        self.action_log: list[dict[str, Any]] = []

    # -- episode --------------------------------------------------------------------------------

    @classmethod
    def cached(cls, scenario: str, seed: int, config: dict[str, Any] | str | None = None) -> ReplayTwin:
        """Cached-factory form for Lambda: ``ReplayTwin(..., shared=True)``."""
        return cls(scenario, seed, config, shared=True)

    def _check_lease(self) -> None:
        if self._lease_key is not None and _SHARED[self._lease_key][1] is not self._lease:
            raise RuntimeError("this shared ReplayTwin was superseded by a newer one of the same config")

    def reset(self) -> dict[str, Any]:
        """Start the episode (clears the action log) and return the initial state."""
        self._check_lease()
        from ran_rl.env.config import SafetyBounds
        from ran_rl.env.safety import ActionSafetyFilter

        self._safety = ActionSafetyFilter(SafetyBounds(**self.config["safety"]))
        kwargs = {}
        if self.config["noise_floor_dbm"] is not None:
            kwargs["noise_floor_dbm"] = self.config["noise_floor_dbm"]
        self._state = self._twin.reset(self.scenario, self.seed, **kwargs)
        self.action_log = []
        return copy.deepcopy(self._state)

    def apply(self, change: list[dict[str, Any]]) -> dict[str, Any]:
        """Apply ``[{cell, d_power_db, d_tilt_deg}]`` (or the canonical string
        ``cell=1;d_power_db=+2.0;d_tilt_deg=-1.0|cell=2;...``) through the safety filter, advance one step.

        The request and what the filter let through are appended to ``action_log``. An empty list
        is a no-op step. Returns the new state.
        """
        if self._state is None:
            raise RuntimeError("call reset() first")
        self._check_lease()
        action, entry = filter_change(change, self._state, self._safety, len(self.action_log))
        self._twin.apply_action(action)
        self._state = self._twin.advance(self.config["step_seconds"])
        self.action_log.append(with_kpis(entry, self._state))
        return copy.deepcopy(self._state)

    # -- counterfactuals (contract amendment A5) -----------------------------------------------

    def _trial(self, change: list[dict[str, Any]] | str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply ``change`` and advance one step on a SNAPSHOT; the live twin is untouched."""
        if self._state is None:
            raise RuntimeError("call reset() first")
        self._check_lease()
        action, entry = filter_change(change, self._state, self._safety, len(self.action_log))
        trial = _snapshot(self._twin)
        trial.apply_action(action)
        state = trial.advance(self.config["step_seconds"])
        return state, entry

    def noop_kpis(self) -> dict[str, float]:
        """Headline KPIs after advancing one step with NO change, from the current state.

        Does not mutate the live twin (it advances a snapshot).
        """
        state, _ = self._trial([])
        return headline(state)

    def counterfactual(self, change: list[dict[str, Any]] | str) -> dict[str, Any]:
        """Effect of ``change`` against the no-change counterfactual at the same step.

        Returns ``{baseline, after, delta, clamped, rules, requested, filtered}`` where baseline
        is the KPIs after a no-change step, after is the KPIs after the change and a step, and
        delta = after - baseline for coverage_ratio, sla_ratio, sinr_p10_db and sinr_mean_db.
        Both futures start from the same state and the same random stream, so UE motion is
        identical and the difference is caused by the change only (exact, not statistical).
        ``clamped`` is true if the safety filter altered the request (record it as unscored).
        The live twin (state, log, step) is NOT changed.
        """
        after_state, entry = self._trial(change)
        base_state, _ = self._trial([])
        after, baseline = headline(after_state), headline(base_state)
        return {
            "baseline": baseline,
            "after": after,
            "delta": {k: after[k] - baseline[k] for k in HEADLINE},
            "clamped": entry["clamped"],
            "rules": entry["rules"],
            "requested": entry["requested"],
            "filtered": entry["filtered"],
        }

    def counterfactual_delta(self, change: list[dict[str, Any]] | str) -> dict[str, float]:
        """``counterfactual(change)["delta"]``: {coverage_ratio, sla_ratio, sinr_p10_db, sinr_mean_db}."""
        return self.counterfactual(change)["delta"]

    # -- views ----------------------------------------------------------------------------------

    @property
    def state(self) -> dict[str, Any]:
        """A copy of the latest state."""
        self._check_lease()
        if self._state is None:
            raise RuntimeError("call reset() first")
        return copy.deepcopy(self._state)

    @property
    def step(self) -> int:
        """Steps applied since reset (len(action_log))."""
        return len(self.action_log)

    # -- persistence ----------------------------------------------------------------------------

    def to_record(self) -> dict[str, Any]:
        """JSON-safe case record: everything needed to rebuild this twin."""
        return {
            "schema": SCHEMA,
            "scenario": self.scenario,
            "seed": self.seed,
            "config": copy.deepcopy(self.config),
            "config_sha256": config_hash(self.config),
            "sdk_version": sdk_version(),
            "action_log": copy.deepcopy(self.action_log),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any], **kwargs: Any) -> ReplayTwin:
        if record.get("schema") != SCHEMA:
            raise ValueError(f"unsupported record schema {record.get('schema')!r}")
        return cls.from_log(
            record["scenario"], record["seed"], record["config"], record["action_log"], **kwargs
        )

    @classmethod
    def from_log(
        cls,
        scenario: str,
        seed: int,
        config: dict[str, Any] | str | None,
        action_log: list[dict[str, Any]],
        *,
        verify: bool = True,
        reuse_models: bool = True,
        shared: bool = False,
    ) -> ReplayTwin:
        """Rebuild a twin and replay ``action_log``; the result is in the identical state.

        The requested changes are replayed (not the filtered ones), so the filter runs again; with
        ``verify`` the filtered vectors and the post-step KPIs must match what was logged.
        """
        twin = cls(scenario, seed, config, reuse_models=reuse_models, shared=shared)
        twin.reset()
        for i, logged in enumerate(action_log):
            if logged.get("step", i) != i:
                raise ReplayMismatch(f"log entry {i} carries step {logged.get('step')}")
            twin.apply(logged["change"])
            if not verify:
                continue
            new = twin.action_log[-1]
            if new["filtered"] != logged["filtered"]:
                raise ReplayMismatch(
                    f"step {i}: filtered action {new['filtered']} != logged {logged['filtered']}"
                )
            for key, want in logged.get("kpis", {}).items():
                got, want = dec(new["kpis"][key]), dec(want)
                same = (math.isnan(got) and math.isnan(want)) or math.isclose(
                    got, want, rel_tol=_RTOL, abs_tol=_ATOL
                )
                if not same:
                    raise ReplayMismatch(f"step {i}: {key} replayed {got!r}, logged {want!r}")
        return twin

    def exact_kpis(self) -> dict[str, float]:
        """Exact pooled sinr_mean / sinr_p10 / rsrp_p10 over every UE of the current state.

        Recomputes the UE-level SINR and serving RSRP the way ``MavericDigitalTwin.observe`` does,
        using the twin's private attributes (so it only works for the pinned SDK, ran-rl 1.2.9 and
        raises otherwise). For reference and validation of ``twin.kpis``; the hosted adapters
        cannot offer it, which is why ``kpis(state)`` is the portable definition.
        """
        import numpy as np

        if sdk_version() != "1.2.9":
            raise RuntimeError("exact_kpis relies on ran-rl 1.2.9 internals")
        from ran_rl.twin.maveric.model import predict
        from ran_rl.twin.maveric.vendor.attachment import attach

        tw = self._twin
        rsrp = predict(tw._models, tw._cells, tw._ue, tw._tilt, tw._power).rsrp_dbm
        rsrp[:, tw._alarm | tw._asleep] = -np.inf
        _, served, sinr = attach(rsrp, tw._freq, tw._noise_floor_dbm)
        return {
            "sinr_mean_db": float(np.mean(sinr)),
            "sinr_p10_db": float(np.percentile(sinr, 10)),
            "rsrp_p10_dbm": float(np.percentile(served, 10)),
        }

    def headline(self) -> dict[str, float]:
        """predicted/measured-shaped KPIs of the current state."""
        return headline(self.state)
