"""AdapterClient: the organisers' twin adapters (hosted Maveric, Viavi) with ReplayTwin's semantics.

Adapter routes (ran_rl_host/adapter/app.py): POST /reset {scenario, seed[, noise_floor_dbm]},
GET /observe, POST /action {power_delta_db, tilt_delta_deg}, POST /advance {seconds}, POST /close,
GET|POST /scenario (Viavi), GET /health (no token). Bearer token on every route but /health.

The adapter does NOT run the safety filter; the SDK environment does, on the client side. So
``apply`` here runs the same ``ActionSafetyFilter`` locally against the last observed state, posts
the filtered deltas to /action, then posts /advance with ``step_seconds`` and returns the new state:
the same sequence as ReplayTwin.apply. The adapter keeps one session per process, so a client
must be the only user of its adapter.

URL and token come only from arguments or environment variables, never from files:
``AdapterClient.from_env("maveric")`` reads RAN_RL_TWIN_URL / RAN_RL_ADAPTER_TOKEN and
``from_env("viavi")`` reads RAN_RL_VIAVI_TWIN_URL / RAN_RL_VIAVI_ADAPTER_TOKEN (the names the SDK uses).
The token is never logged or put in an exception message.

Config: the safety bounds, step_seconds and noise floor of ``twin.presets`` (hosted Maveric uses the
"official" values). For Viavi pass ``config={"step_seconds": 60.0}`` (its window); the server-side
twin settings (UE count, power) are the organisers' and cannot be set from here.
"""

from __future__ import annotations

import copy
import os
from typing import Any

import httpx

from twin._change import filter_change, with_kpis
from twin.presets import resolve_config

ENV_NAMES = {
    "maveric": ("RAN_RL_TWIN_URL", "RAN_RL_ADAPTER_TOKEN"),
    "viavi": ("RAN_RL_VIAVI_TWIN_URL", "RAN_RL_VIAVI_ADAPTER_TOKEN"),
}


class AdapterClient:
    """Same reset / apply / observe semantics as ReplayTwin, over an organiser adapter."""

    def __init__(
        self,
        url: str,
        token: str,
        config: dict[str, Any] | str | None = None,
        *,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        client: httpx.Client | None = None,
    ) -> None:
        from ran_rl.twin.http import HttpDigitalTwin

        if not url or not token:
            raise ValueError("url and token are required")
        self.config = resolve_config(config)
        self._client = client or httpx.Client(base_url=url, timeout=timeout_s)
        self._http = HttpDigitalTwin(
            url, token, timeout_s=timeout_s, max_retries=max_retries, client=self._client
        )
        self._safety: Any = None
        self._state: dict[str, Any] | None = None
        self.scenario: str | None = None
        self.seed: int | None = None
        self.action_log: list[dict[str, Any]] = []

    @classmethod
    def from_env(cls, kind: str = "maveric", **kwargs: Any) -> AdapterClient:
        """Build from the environment variables for ``kind`` ("maveric" or "viavi")."""
        if kind not in ENV_NAMES:
            raise ValueError(f"kind must be one of {sorted(ENV_NAMES)}")
        url_name, token_name = ENV_NAMES[kind]
        url, token = os.environ.get(url_name), os.environ.get(token_name)
        if not url or not token:
            raise RuntimeError(f"set {url_name} and {token_name} in the environment")
        return cls(url, token, **kwargs)

    # -- ReplayTwin-compatible surface ------------------------------------------------------------

    def reset(self, scenario: str, seed: int) -> dict[str, Any]:
        from ran_rl.env.config import SafetyBounds
        from ran_rl.env.safety import ActionSafetyFilter

        self._safety = ActionSafetyFilter(SafetyBounds(**self.config["safety"]))
        kwargs = {}
        if self.config["noise_floor_dbm"] is not None:
            kwargs["noise_floor_dbm"] = self.config["noise_floor_dbm"]
        self.scenario, self.seed = scenario, int(seed)
        self._state = self._http.reset(scenario, self.seed, **kwargs)
        self.action_log = []
        return copy.deepcopy(self._state)

    def apply(self, change: list[dict[str, Any]]) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("call reset() first")
        action, entry = filter_change(change, self._state, self._safety, len(self.action_log))
        self._http.apply_action(action)
        self._state = self._http.advance(self.config["step_seconds"])
        self.action_log.append(with_kpis(entry, self._state))
        return copy.deepcopy(self._state)

    def observe(self) -> dict[str, Any]:
        """The adapter's current state (``GET /observe``), without advancing time."""
        self._state = self._http.observe()
        return copy.deepcopy(self._state)

    @property
    def state(self) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("call reset() first")
        return copy.deepcopy(self._state)

    @property
    def step(self) -> int:
        return len(self.action_log)

    # -- adapter extras ---------------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """``GET /health`` (no token needed)."""
        return self._client.get("/health").json()

    def scenario_status(self) -> dict[str, Any]:
        """Viavi: ``GET /scenario``."""
        return self._http.scenario_status()

    def prepare_scenario(self, force: bool = False) -> dict[str, Any]:
        """Viavi: ``POST /scenario``."""
        return self._http.prepare_scenario(force)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> AdapterClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
