import copy

from spectrum.slices.trustgate import logic
from slices.trustgate.trustgate_fakes import FakeReader, all_of, all_results

HEALTHY = ("t2-response-mlp", "v1")
DEGRADED = ("t2-response-mlp", "v1-degraded")


def detail(key, changes=None, controls=None, results=None, model=True):
    r = FakeReader()
    rec = r.model(*key) if model else None
    return logic.model_detail(key, all_of("change") if changes is None else changes,
                              all_of("control") if controls is None else controls, rec,
                              all_results() if results is None else results)


def test_only_predictors_have_a_trust_line():
    keys = logic.model_keys(all_of("change"), all_of("control"))
    assert keys == [HEALTHY, DEGRADED]  # PPO (t1-ppo-maveric) and probes are never listed


def test_healthy_model_has_no_pause_and_unscored_change_is_excluded():
    d = detail(HEALTHY)
    assert d["paused"] is False and d["pauses"] == [] and d["paused_by"] is None
    assert d["n_changes"] == 11 and d["n_scored"] == 10 and d["n_unscored"] == 1
    assert len(d["points"]) == 10
    clipped = [r for r in d["rows"] if not r["scored"]]
    assert [r["unscored_reason"] for r in clipped] == ["clipped"]
    assert clipped[0]["trust_after"] is None
    assert clipped[0]["change_id"] not in {p["change_id"] for p in d["points"]}
    # exactly one change is over tau (e = 1.10), and it is not 2 of the last 3
    assert [p["over_tau"] for p in d["points"]].count(True) == 1
    assert d["trust_latest"] == d["points"][-1]["trust_after"]


def test_unscored_change_never_enters_trust_even_if_it_carries_trust_fields():
    ch = copy.deepcopy(all_of("change"))
    for c in ch:
        if c.get("scored") is False and c.get("predictor"):
            c["trust_after"] = 0.01
    d = detail(HEALTHY, changes=ch)
    assert all(p["trust_after"] != 0.01 for p in d["points"])
    assert d["trust_latest"] != 0.01


def test_degraded_model_pause_is_by_the_trust_gate_at_control_time():
    d = detail(DEGRADED, controls=[c for c in all_of("control") if c["action"] == "pause"])
    assert d["paused"] is True and d["paused_by"] == "trust-gate" and d["paused_by_kind"] == "trust-gate"
    ctl = next(c for c in all_of("control") if c["action"] == "pause")
    assert d["paused_at"] == ctl["time"] and d["pauses"][0]["time"] == ctl["time"]
    assert d["pauses"][0]["policy_name"] == ctl["policy_name"] and d["pauses"][0]["reason"] == ctl["reason"]
    assert [p["over_tau"] for p in d["points"]] == [False, False, True, True, True]


def test_person_resume_clears_the_pause_and_is_labelled_a_person():
    d = detail(DEGRADED)
    assert [(e["action"], e["by_kind"]) for e in d["pauses"]] == [("pause", "trust-gate"), ("resume", "person")]
    assert d["pauses"][1]["by"].startswith("person:")
    assert d["paused"] is False and d["status_basis"] == "control" and d["status_conflict"] is False


def test_person_pause_is_not_shown_as_a_trust_gate_pause():
    ctl = copy.deepcopy([c for c in all_of("control") if c["action"] == "pause"])
    ctl[0]["by"] = "person:approver-1"
    d = detail(DEGRADED, controls=ctl)
    assert d["paused"] and d["paused_by_kind"] == "person" and d["pauses"][0]["by_kind"] == "person"


def test_status_conflict_between_model_record_and_controls_is_reported():
    r = FakeReader()
    rec = r.model(*DEGRADED)
    rec["status"] = "paused"  # the model record says paused, the newest control says resume
    d = logic.model_detail(DEGRADED, all_of("change"), all_of("control"), rec, all_results())
    assert d["paused"] is False and d["status_conflict"] is True and d["model_status"] == "paused"


def test_no_control_uses_model_record_status():
    rec = FakeReader().model(*HEALTHY)
    rec["status"] = "paused"
    d = logic.model_detail(HEALTHY, all_of("change"), [], rec, all_results())
    assert d["paused"] is True and d["status_basis"] == "model" and d["paused_by"] is None


def test_thresholds_come_from_the_result_and_are_not_invented():
    d = detail(DEGRADED)
    th = d["thresholds"]
    assert th["tau"] == 1.0 and th["pause_below"] == 0.55 and d["pause_below"] == 0.55
    assert th["scales"] == {"coverage_ratio": 0.08, "sla_ratio": 0.06, "sinr_p10_db": 1.0}
    assert th["source"].startswith("result:t3-trust-pause")
    none = detail(DEGRADED, results=[])
    assert none["thresholds"] is None and none["pause_below"] is None
    assert all(p["over_tau"] is None for p in none["points"])  # without tau nothing is flagged


def test_thresholds_fall_back_to_model_metrics_and_ignore_other_models():
    rec = FakeReader().model(*HEALTHY)
    rec["metrics"] = {"thresholds": {"tau": 2.5, "pause_below": 0.4}}
    other = copy.deepcopy(all_results()[0])
    other["details"] = {"model_id": "someone-else", "version": "v9", "thresholds": {"tau": 9.0}}
    d = logic.model_detail(HEALTHY, all_of("change"), all_of("control"), rec, [other])
    assert d["thresholds"]["tau"] == 2.5 and d["thresholds"]["source"] == "model.metrics"


def test_rows_are_copied_not_recomputed_and_sorted_by_time():
    ch = list(reversed(all_of("change")))
    d = detail(DEGRADED, changes=ch)
    times = [r["t"] for r in d["rows"]]
    assert times == sorted(times)
    src = {c["change_id"]: c for c in all_of("change")}
    for r in d["rows"]:
        c = src[r["change_id"]]
        assert r["error_summary"] == c["error"]["summary"]
        assert r["direction_agree"] == c["direction_agree"]
        assert r["trust_after"] == c["trust_after"]
        by_key = {k["key"]: k for k in r["kpis"]}
        for k, v in c["predicted_delta"].items():
            assert by_key[k]["predicted_delta"] == v and by_key[k]["measured_delta"] == c["measured_delta"][k]
            assert by_key[k]["error"] == c["error"][k]
    assert d["n_direction_known"] == 5 and d["n_direction_agree"] == 2


def test_probes_are_kept_apart_and_have_no_model():
    rows = logic.probe_rows(all_of("change"))
    assert len(rows) == 2 and all(r["probe"] and not r["scored"] and r["error_summary"] is None for r in rows)
    for k in logic.model_keys(all_of("change"), all_of("control")):
        assert not any(r["probe"] for r in detail(k)["rows"])


def test_unusable_records_are_skipped_not_guessed():
    ch = copy.deepcopy(all_of("change"))
    next(c for c in ch if (c.get("predictor") or {}).get("version") == "v1")["created_at"] = "not a time"
    ch.append("junk")  # type: ignore[arg-type]
    d = detail(HEALTHY, changes=ch)
    assert d["n_changes"] == 10
    ctl = [{"model_id": "t2-response-mlp", "version": "v1", "action": "stop", "by": "trust-gate", "time": "2026-11-02T20:00:00Z"}]
    assert detail(HEALTHY, controls=ctl)["pauses"] == []


# ------------------------------------------------------------------ ladder

def test_latest_ladders_keep_rehearsal_and_viavi_apart():
    ls = logic.latest_ladders(all_results())
    got = {(l["model_id"], l["version"], l["top_label"]) for l in ls}
    assert got == {("t1-ppo-maveric", "v3", "REHEARSAL (Viavi-like Maveric)"), ("t1-ppo-maveric", "v4", "VIAVI")}
    reh = next(l for l in ls if l["version"] == "v3")
    assert reh["has_viavi"] is False and reh["label_ok"] is True
    assert [r["label"] for r in reh["rungs"]] == ["MAVERIC", "REHEARSAL (Viavi-like Maveric)"]
    v = next(l for l in ls if l["version"] == "v4")
    assert v["has_viavi"] is True and v["retention_ci"] is None


def test_ladder_numbers_are_shown_as_recorded_never_recomputed():
    res = copy.deepcopy([r for r in all_results() if r["unit"] == "t3.ladder"][0])
    res["details"]["retention"] = 0.123  # deliberately not gain2/gain1
    view = logic.ladder_view(res)
    assert view["retention"] == 0.123
    assert [r["gain"] for r in view["rungs"]] == [r["gain"] for r in res["details"]["rungs"]]


def test_a_newer_ladder_supersedes_an_older_one_of_the_same_kind():
    base = [r for r in all_results() if r["unit"] == "t3.ladder" and r["details"]["version"] == "v3"][0]
    older, newer = copy.deepcopy(base), copy.deepcopy(base)
    older["checked_at"], older["details"]["retention"] = "2026-11-02T10:00:00Z", 0.1
    newer["checked_at"], newer["details"]["retention"] = "2026-11-02T11:00:00Z", 0.9
    older["run_id"], newer["run_id"] = "old", "new"
    out = logic.latest_ladders([newer, older])
    assert len(out) == 1 and out[0]["retention"] == 0.9 and out[0]["run_id"] == "new"


def test_a_rehearsal_can_never_pass_as_viavi():
    res = copy.deepcopy([r for r in all_results() if r["unit"] == "t3.ladder" and r["details"]["version"] == "v3"][0])
    res["details"]["rungs"][1]["label"] = "VIAVI"  # claims Viavi but ran on Maveric
    view = logic.ladder_view(res)
    assert view["has_viavi"] is False and view["label_ok"] is False
    assert "maveric-inprocess" in view["rungs"][1]["label_problem"]
    res2 = copy.deepcopy(res)
    res2["details"]["rungs"][1].update(label="REHEARSAL (Viavi-like Maveric)", twin_kind="viavi")  # the other way round
    assert logic.ladder_view(res2)["label_ok"] is False


def test_unknown_rung_label_or_wrong_unit_is_not_a_ladder():
    res = copy.deepcopy([r for r in all_results() if r["unit"] == "t3.ladder"][0])
    res["details"]["rungs"][0]["label"] = "Viavi"
    assert logic.ladder_view(res) is None
    res = copy.deepcopy([r for r in all_results() if r["unit"] == "t3.ladder"][0])
    res["unit"] = "t3.trust-pause"
    assert logic.ladder_view(res) is None
    assert logic.latest_ladders([]) == []
