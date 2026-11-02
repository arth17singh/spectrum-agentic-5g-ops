"""AdapterClient against a fake adapter that implements the documented routes
(ran_rl_host/adapter/app.py: bearer token, /reset /observe /action /advance /close /health) around the
SDK's Maveric twin. The real adapter is exercised by test_parity.py."""

import httpx
import pytest
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from twin.http import AdapterClient
from twin.replay import ReplayTwin, _build_twin
from twin.presets import resolve_config

TOKEN = "throwaway-test-token"


def fake_adapter(calls):
    app = FastAPI()
    state = {"twin": None}

    def authorized(request: Request):
        if request.headers.get("Authorization") != f"Bearer {TOKEN}":
            raise HTTPException(status_code=401, detail="missing or invalid bearer token")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/reset", dependencies=[Depends(authorized)])
    async def reset(request: Request):
        body = await request.json()
        calls.append(("reset", body))
        state["twin"] = _build_twin(resolve_config(None))
        return state["twin"].reset(**{k: v for k, v in body.items() if v is not None})

    @app.get("/observe", dependencies=[Depends(authorized)])
    def observe():
        return state["twin"].observe()

    @app.post("/action", dependencies=[Depends(authorized)])
    async def action(request: Request):
        body = await request.json()
        calls.append(("action", body))
        state["twin"].apply_action(body)
        return {"status": "ok"}

    @app.post("/advance", dependencies=[Depends(authorized)])
    async def advance(request: Request):
        body = await request.json()
        calls.append(("advance", body))
        return state["twin"].advance(body["seconds"])

    @app.post("/close", dependencies=[Depends(authorized)])
    def close():
        return {"status": "ok"}

    return app


def test_client_matches_replay_and_filters_before_posting():
    calls = []
    http = TestClient(fake_adapter(calls), base_url="http://adapter.test")
    client = AdapterClient("http://adapter.test", TOKEN, client=http)
    local = ReplayTwin("congestion", 5)
    assert client.reset("congestion", 5) == local.reset()
    assert client.health() == {"status": "ok"}
    for change in ([{"cell": 2, "d_power_db": 40.0, "d_tilt_deg": -9.0}], [], [{"cell": 0, "d_power_db": 1.5}]):
        assert client.apply(change) == local.apply(change)
    assert client.action_log == local.action_log
    # the adapter never saw the out-of-range request: the filter ran on our side
    first_action = [b for name, b in calls if name == "action"][0]
    assert first_action["power_delta_db"][2] == 8.0 and first_action["tilt_delta_deg"][2] == -2.0  # tilt 6 -> floor 4
    assert [b for name, b in calls if name == "advance"][0] == {"seconds": 3600.0}
    assert client.observe() == local.state
    assert client.step == 3
    client.close()


def test_wrong_token_is_refused_and_not_echoed():
    http = TestClient(fake_adapter([]), base_url="http://adapter.test")
    client = AdapterClient("http://adapter.test", "wrong-token-value", client=http)
    with pytest.raises(httpx.HTTPStatusError) as err:
        client.reset("nominal", 1)
    assert "wrong-token-value" not in str(err.value) and TOKEN not in str(err.value)


def test_env_and_arguments_only(monkeypatch):
    for name in ("RAN_RL_TWIN_URL", "RAN_RL_ADAPTER_TOKEN", "RAN_RL_VIAVI_TWIN_URL", "RAN_RL_VIAVI_ADAPTER_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError):
        AdapterClient.from_env("maveric")
    with pytest.raises(ValueError):
        AdapterClient("", "t")
    monkeypatch.setenv("RAN_RL_VIAVI_TWIN_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("RAN_RL_VIAVI_ADAPTER_TOKEN", "x")
    c = AdapterClient.from_env("viavi", config={"step_seconds": 60.0})
    assert c.config["step_seconds"] == 60.0
    with pytest.raises(RuntimeError):
        c.apply([])  # before reset
