import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from spectrum.slices.swarmtower import logic, router

RUN = "t3.swarm-live-20990101T000000Z"


def test_parse_change_text():
    assert logic.parse_change_text("cell=0;d_power_db=-1.2;d_tilt_deg=-0.3|cell=3;d_power_db=+2.0;d_tilt_deg=+2.0") == [
        {"cell": 0, "d_power_db": -1.2, "d_tilt_deg": -0.3}, {"cell": 3, "d_power_db": 2.0, "d_tilt_deg": 2.0}]
    assert logic.parse_change_text("") == [] and logic.parse_change_text(None) == []


def test_parse_incident_id():
    m = logic.parse_incident_id("inc-off-mas-4138-tilt+4")
    assert m["seed"] == 4138 and m["template"] == "tilt+4" and m["preset"].startswith("Maveric official") and m["tier"].startswith("masked")
    assert logic.parse_incident_id("inc-vvl-mas-4103-pwr-3")["preset"].startswith("Maveric viavi-like")
    assert logic.parse_incident_id("nope")["seed"] is None


def item(iid, step, change="", refused=False, state="TASK_STATE_COMPLETED", problem=None):
    return {"item_id": f"{iid}#s{step}", "passed": problem is None, "score": 0.0 if problem else 1.0, "latency_ms": 10.0,
            "aws_time_resolved": "2099-01-01T00:00:00Z", "aws_time_source_resolved": "gateway_date",
            "extra": {"incident_id": iid, "step": step, "case_id": f"case-x-s{step}", "change_text": change,
                      "approval_id": "chg-1" if refused else None, "refused": refused, "state": state, "problem": problem}}


def test_group_steps_and_outcomes():
    items = [item("inc-off-mas-1-tilt+5", 2), item("inc-off-mas-1-tilt+5", 0, "cell=0;d_power_db=+1.0;d_tilt_deg=-2.0", refused=True),
             item("inc-off-mas-1-tilt+5", 1, problem="timed out waiting for the lead", state="TASK_STATE_WORKING")]
    actions = [{"incident_id": "inc-off-mas-1-tilt+5", "step": 0, "model_id": "noc_lead=a;students=b", "start_action_log": ["x"]}]
    scores = {"inc-off-mas-1-tilt+5": {"success": True, "recovery": 0.9, "incident": {"fault": [{"cell": 0, "d_power_db": 0.0, "d_tilt_deg": 5.0}]}}}
    (inc,) = logic.group_incidents(items, actions, scores)
    assert [s["step"] for s in inc["steps"]] == [0, 1, 2] and inc["recovered"] is True
    assert [s["outcome"]["code"] for s in inc["steps"]] == ["held", "problem", "none"]
    assert inc["steps"][1]["state_words"].startswith("the lead was still working")
    assert inc["models"] == [{"role": "noc lead", "model": "a"}, {"role": "students", "model": "b"}]


def w(root: Path, rel: str, obj) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(obj if isinstance(obj, str) else json.dumps(obj))


def test_route_reads_the_run(tmp_path, monkeypatch):
    rec = {"item_id": "inc-off-mas-1-tilt+5#s0", "passed": True, "score": 1.0, "latency_ms": 5.0, "run_id": RUN,
           "aws_time": "2099-01-01T00:00:01.000Z", "aws_time_source": "gateway:http-date",
           "extra": {"incident_id": "inc-off-mas-1-tilt+5", "step": 0, "case_id": "case-x-i00s0", "refused": True,
                     "approval_id": "chg-1", "change_text": "cell=1;d_power_db=+2.0;d_tilt_deg=+0.0", "state": "TASK_STATE_COMPLETED"}}
    w(tmp_path, f"declared/{RUN}.json", {"run_id": RUN, "unit": "t3.evals.swarm-proposals", "model_id": "noc_lead=a"})
    w(tmp_path, f"records/{RUN}/item/aaaaaaaaaaaaaaaaaaaaaaaa.json", rec)
    w(tmp_path, f"outputs/{RUN}/swarm-actions.jsonl", json.dumps({"incident_id": "inc-off-mas-1-tilt+5", "step": 0, "model_id": "noc_lead=a"}) + "\n")
    monkeypatch.setenv("SPECTRUM_RECORDS_ROOT", str(tmp_path))
    monkeypatch.setenv("SPECTRUM_SILOS_DIR", str(tmp_path / "no-silos"))
    app = FastAPI()
    app.include_router(router)
    d = TestClient(app).get("/api/swarmtower").json()
    s = d["incidents"][0]["steps"][0]
    assert d["run"]["run_id"] == RUN and s["cells"][0]["d_power_db"] == 2.0 and s["aws_time_source"] == "gateway_date"
    assert d["aws_time_summary"]["k"] == 1 and d["silos"] is None
