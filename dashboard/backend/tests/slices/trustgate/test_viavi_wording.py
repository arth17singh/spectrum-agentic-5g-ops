from spectrum.slices.trustgate.logic import MEASURE_ONLY_REASON, NO_VIAVI_REASON, viavi_wording


def L():
    return {"retention": None, "real_viavi": False, "has_viavi": False, "retention_heading": "x",
            "retention_null_reason": "no VIAVI rung (no Viavi access yet): stale"}


def test_measure_only_when_a_viavi_run_exists():
    l = viavi_wording([L()], "t1.rl.viavi_operate-1")[0]
    assert l["retention_heading"] == "Retention: not measured" and l["retention_null_reason"] == MEASURE_ONLY_REASON
    assert "access" not in l["retention_null_reason"]


def test_no_viavi_run_only_when_none_exists():
    l = viavi_wording([L()], None)[0]
    assert l["retention_heading"] == "Retention: not measured (no Viavi run)" and l["retention_null_reason"] == NO_VIAVI_REASON
    assert "access" not in NO_VIAVI_REASON


def test_a_measured_retention_is_untouched():
    l = {**L(), "retention": 0.9, "retention_heading": "Retention REHEARSAL -> VIAVI"}
    assert viavi_wording([l], "t1.rl.viavi_operate-1")[0]["retention_heading"] == "Retention REHEARSAL -> VIAVI"
