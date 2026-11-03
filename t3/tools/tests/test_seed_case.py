"""ran___seed_case: the evaluation harness opens a NEW case at an incident's state (Cedar: app token only)."""
from __future__ import annotations

import json

from t3.tools.tests.test_ran_tools import World

CASE = "case-abc12345-i01s0"
LOG = ["cell=1;d_power_db=+1.0;d_tilt_deg=+0.0", [], "cell=2;d_power_db=+0.0;d_tilt_deg=-2.0"]


def seed(w, case=CASE, log=LOG, **kw):
    args = {"case_id": case, "scenario": "nominal", "seed": 7, "preset": "official",
            "start_action_log_json": json.dumps(log), **kw}
    return w.call("ran___seed_case", **args)


def test_seed_opens_the_case_at_the_given_state_under_the_runners_run():
    w = World()
    w.ddb.set_plain(f"t3-{CASE}", case_id=CASE, run_id="t3.swarm-live-1")   # the harness maps it first
    out = seed(w)
    assert out["seeded"] is True and out["step"] == 3 and out["run_id"] == "t3.swarm-live-1"
    obs = w.call("ran___observe", case_id=CASE)
    assert obs["twin"]["step"] == 3 and obs["run_id"] == "t3.swarm-live-1"
    doc = w.case_doc(CASE)
    assert len(doc["action_log"]) == 3 and doc["scenario"] == "nominal" and doc["seed"] == 7
    # the next change proposed by the swarm is step-3, after the seeded steps
    assert w.call("ran___propose", case_id=CASE)["item_id"] == "step-3"


def test_seed_refuses_an_existing_case_and_bad_input():
    w = World()
    assert seed(w)["seeded"] is True
    assert "already exists" in seed(w)["error"]
    assert "error" in seed(w, case="case-abc12345-i02s0", log=["not a change"])
    assert "error" in w.call("ran___seed_case", case_id="case-abc12345-i03s0", scenario="nominal")
