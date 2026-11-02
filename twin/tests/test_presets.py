import json

import pytest

from twin.presets import OFFICIAL, VIAVI_LIKE, VIAVI_SCENARIO, config_hash, resolve_config
from twin.replay import ReplayTwin


def test_official_equals_sdk_defaults():
    from ran_rl.env.config import EnvConfig, SafetyBounds
    from ran_rl.twin.maveric.twin import MavericTwinConfig

    cfg = resolve_config(None)
    assert cfg["preset"] == OFFICIAL
    assert cfg["twin"] == MavericTwinConfig().model_dump()
    assert cfg["safety"] == SafetyBounds().model_dump()
    assert cfg["step_seconds"] == EnvConfig().step_seconds == 3600.0
    assert cfg["scenarios"] == {}


def test_resolve_is_idempotent_and_json_safe():
    for name in (None, OFFICIAL, VIAVI_LIKE, {"preset": VIAVI_LIKE, "twin": {"n_ues": 30}}):
        once = resolve_config(name)
        assert resolve_config(once) == once
        assert json.loads(json.dumps(once)) == once
    assert config_hash(resolve_config(None)) != config_hash(resolve_config(VIAVI_LIKE))


def test_overrides_merge_and_unknown_keys_fail():
    cfg = resolve_config({"twin": {"n_ues": 50}})
    assert cfg["twin"]["n_ues"] == 50 and cfg["twin"]["initial_power_dbm"] == 23.0
    with pytest.raises(Exception):
        resolve_config({"twin": {"n_uess": 50}})
    with pytest.raises(ValueError):
        resolve_config({"bogus": 1})
    with pytest.raises(ValueError):
        resolve_config("nope")


def test_viavi_like_values():
    cfg = resolve_config(VIAVI_LIKE)
    assert cfg["twin"]["n_ues"] == 60
    assert cfg["twin"]["initial_power_dbm"] == 40.0
    assert cfg["twin"]["initial_tilt_deg"] == 6.0
    assert cfg["step_seconds"] == 60.0
    assert VIAVI_SCENARIO in cfg["scenarios"]


def test_viavi_like_twin_is_60_stationary_ues_at_40dbm():
    t = ReplayTwin(VIAVI_SCENARIO, 5, VIAVI_LIKE)
    s0 = t.reset()
    assert s0["power_dbm"] == [40.0] * 4 and s0["tilt_deg"] == [6.0] * 4
    assert sum(s0["ue_count"]) == 60
    last = s0
    for _ in range(5):
        last = t.apply([])
        # stationary: nobody moves, so with no action the whole state stays identical
        assert last["extra"]["ue_displacement_m_mean"] == 0.0
        assert last["ue_count"] == s0["ue_count"]
        assert last["sinr_mean_db"] == s0["sinr_mean_db"]
    assert last["phase"] == 1.0  # 5-step episode


def test_official_nominal_ues_do_move():
    t = ReplayTwin("nominal", 5)
    t.reset()
    assert t.apply([])["extra"]["ue_displacement_m_mean"] > 0
    assert sum(t.state["ue_count"]) == 200
