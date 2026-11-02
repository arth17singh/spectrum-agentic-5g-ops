"""Policy returns in-limit, canonical, deterministic changes; observation masking holds."""

from __future__ import annotations

import copy
import re
from pathlib import Path

import numpy as np
import pytest

from t1.rl.obs import mask_obs
from t1.rl.policy import Policy, canon_change, load, parse_change, quantise_toward_zero

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "ppo_v0.zip"
CANON = re.compile(r"^cell=\d;d_power_db=[+-]\d+\.\d;d_tilt_deg=[+-]\d+\.\d(\|cell=\d;d_power_db=[+-]\d+\.\d;d_tilt_deg=[+-]\d+\.\d)*$")


def twin_state(seed: int, scenario: str = "nominal") -> dict:
    from ran_rl.env.config import EnvConfig, TwinConfig
    from ran_rl.twin.factory import build_twin

    twin = build_twin(EnvConfig(twin=TwinConfig(backend="maveric")))
    return twin.reset(scenario, seed)


@pytest.fixture(scope="module")
def policy(tmp_path_factory):
    """The committed artifact if present, else a few-step PPO trained here (the interface is what is tested)."""
    if ARTIFACT.exists():
        return load(ARTIFACT)
    import ran_rl
    from ran_rl.env.config import EnvConfig
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from t1.rl.obs import MaskedObs

    venv = DummyVecEnv([lambda: MaskedObs(ran_rl.make_env(twin="maveric", env_config=EnvConfig()))])
    model = PPO("MultiInputPolicy", venv, device="cpu", n_steps=64, batch_size=32, seed=0)
    model.learn(128)
    return Policy(model)


@pytest.fixture(scope="module")
def states():
    out = [twin_state(s, sc) for s, sc in [(1, "nominal"), (2, "congestion"), (3, "interference"), (4, "low_load")]]
    ext = copy.deepcopy(out[0])
    ext["power_dbm"], ext["tilt_deg"] = [46.0, 10.0, 12.0, 45.9], [15.0, 4.0, 14.9, 4.05]  # at / near absolute bounds
    low = copy.deepcopy(out[1])
    low["power_dbm"], low["tilt_deg"] = [10.0, 10.2, 11.0, 30.0], [4.0, 4.0, 5.0, 8.0]
    alarmed = twin_state(5, "outage")
    alarmed["alarm"] = [True, False, False, False]
    return out + [ext, low, alarmed]


def test_changes_are_within_sdk_limits(policy, states):
    for st in states:
        ch = policy.propose(st)
        dp, dt = np.zeros(4), np.zeros(4)
        for c in ch:
            dp[c["cell"]], dt[c["cell"]] = c["d_power_db"], c["d_tilt_deg"]
        assert (dp >= -6.0).all() and (dp <= 8.0).all() and (np.abs(dt) <= 5.0).all()
        power, tilt = np.array(st["power_dbm"]) + dp, np.array(st["tilt_deg"]) + dt
        assert (power >= 10.0 - 1e-9).all() and (power <= 46.0 + 1e-9).all()
        assert (tilt >= 4.0 - 1e-9).all() and (tilt <= 15.0 + 1e-9).all()
        for a in np.flatnonzero(st["alarm"]):  # alarm_freeze
            assert dp[a] == 0 and dt[a] == 0
        for v in list(dp) + list(dt):  # quantised to 0.1
            assert abs(v * 10 - round(v * 10)) < 1e-6
        assert [c["cell"] for c in ch] == sorted(c["cell"] for c in ch)


def test_canonical_string_and_round_trip(policy, states):
    for st in states:
        ch, txt = policy.propose(st), policy.propose_text(st)
        assert txt == canon_change(ch)
        assert (txt == "") == (ch == [])
        if txt:
            assert CANON.match(txt), txt
            assert parse_change(txt) == [
                {"cell": c["cell"], "d_power_db": round(c["d_power_db"], 1), "d_tilt_deg": round(c["d_tilt_deg"], 1)} for c in ch]
    assert canon_change([{"cell": 3, "d_power_db": -0.04, "d_tilt_deg": 0.0}, {"cell": 1, "d_power_db": 2.0, "d_tilt_deg": -1.25}]) \
        == "cell=1;d_power_db=+2.0;d_tilt_deg=-1.2|cell=3;d_power_db=+0.0;d_tilt_deg=+0.0"


def test_quantisation_never_exceeds_magnitude():
    v = np.array([7.96, -5.99, 0.049, -0.05, 3.0, 0.3 - 1e-12])
    q = quantise_toward_zero(v)
    assert (np.abs(q) <= np.abs(v) + 1e-9).all()
    assert q.tolist() == pytest.approx([7.9, -5.9, 0.0, 0.0, 3.0, 0.3], abs=1e-9)


def test_propose_is_deterministic_and_ignores_hour_phase_and_absolute_counts(policy, states):
    st = states[0]
    base = policy.propose_text(st)
    assert policy.propose_text(copy.deepcopy(st)) == base
    other = copy.deepcopy(st)
    other.update(hour=17.0, phase=0.9, extra={"z": 1})
    other["ue_count"] = [2 * c for c in st["ue_count"]]
    assert policy.propose_text(other) == base
    shares = {k: v for k, v in st.items() if k != "ue_count"}
    shares["user_share"] = [c / sum(st["ue_count"]) for c in st["ue_count"]]
    assert policy.propose_text(shares) == base


def test_policy_metadata_label(policy):
    assert policy.model_id and policy.version


def test_mask_obs_zeroes_hour_phase_prev_and_uses_share():
    obs = {"ue_count": np.array([-0.5, -0.5, -0.5, -0.5], dtype=np.float32),  # 50 users each
           "hour": np.array([0.7], dtype=np.float32), "phase": np.array([0.3], dtype=np.float32),
           "prev_action": np.ones(8, dtype=np.float32), "power": np.zeros(4, dtype=np.float32),
           "alarms": np.zeros(4, dtype=np.int8)}
    m = mask_obs(obs)
    assert m["hour"].item() == 0 and m["phase"].item() == 0 and not m["prev_action"].any()
    assert np.allclose(m["ue_count"], 2 * 0.25 - 1)  # equal shares of 4 cells
    assert obs["hour"].item() == pytest.approx(0.7)  # input not mutated
