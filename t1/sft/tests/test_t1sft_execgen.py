"""Execution generation: training seeds only, deterministic rounds, reasons from the visible state (fake labeller)."""

from __future__ import annotations

import re

from t1.evals import canon as C
from t1.sft import execgen
from t1.sft.build import check_row, to_row

FAKE = "t1.sft.testing:fake_reference"


def test_specs_use_training_seeds_only():
    for fam in execgen.FAMILIES:
        for j in range(300):
            spec = execgen.exec_spec(fam, j)
            assert 0 <= spec["seed"] < 900_000
            for ch in spec["prefix"]:
                assert ch == "" or C.canon(ch) == ch
    assert execgen.exec_spec("outage", 7) == execgen.exec_spec("outage", 7)


def test_generate_is_deterministic_and_counts_rejections():
    per = {f: 3 for f in execgen.FAMILIES}
    a = execgen.generate(per, workers=1, time_budget_s=60, labeler=FAKE, log=lambda m: None)
    b = execgen.generate(per, workers=1, time_budget_s=60, labeler=FAKE, log=lambda m: None)
    assert a["stopped_by"] == "target_count"
    assert {f: [r["j"] for r in v] for f, v in a["accepted"].items()} == \
        {f: [r["j"] for r in v] for f, v in b["accepted"].items()}
    for f, s in a["stats"].items():
        assert s["accepted"] == 3 and s["attempts"] >= 3
    assert sum(s["rejected_by_keep_rules"] for s in a["stats"].values()) > 0


def test_generate_with_a_process_pool_matches_in_process():
    per = {"outage": 2, "congestion": 2}
    a = execgen.generate(per, workers=1, time_budget_s=60, labeler=FAKE, log=lambda m: None)
    b = execgen.generate(per, workers=2, time_budget_s=120, labeler=FAKE, log=lambda m: None)
    assert {f: [r["j"] for r in v] for f, v in a["accepted"].items()} == \
        {f: [r["j"] for r in v] for f, v in b["accepted"].items()}


def test_time_budget_and_errors_are_reported():
    g = execgen.generate({"outage": 2}, workers=1, time_budget_s=-1, labeler=FAKE, log=lambda m: None)
    assert g["stopped_by"] == "time_budget" and g["accepted"]["outage"] == []
    g = execgen.generate({"power_drop": 2}, workers=1, time_budget_s=60, labeler="t1.sft.testing:failing_reference",
                         log=lambda m: None)
    assert g["stats"]["power_drop"]["errors"] > 0 and g["errors"]


def test_examples_render_and_parse_as_the_eval_reads_them():
    g = execgen.generate({f: 2 for f in execgen.FAMILIES}, workers=1, time_budget_s=60, labeler=FAKE,
                         log=lambda m: None)
    for fam in execgen.FAMILIES:
        for r in g["accepted"][fam]:
            ex = execgen.to_example(r)
            row = to_row(ex, "train")
            check_row(row)
            assert ex["meta"]["seed"] < 900_000 and ex["meta"]["step"] == len(ex["meta"]["prefix"])
            assert ex["answer"]["change"] == r["ref"]["oracle_change"]
            assert "Network state:" in row["messages"][1]["content"]


def test_reason_states_only_visible_facts():
    facts = {"power_dbm": [23.0, 23.0, 29.0, 23.0], "tilt_deg": [6.0, 6.0, 6.0, 11.0],
             "ue_count": [40.0, 0.0, 77.0, 51.0], "rsrp_p10_dbm": [-101.234, float("-inf"), -99.5, -104.0],
             "sinr_p10_db": [1.0, float("nan"), 2.0, -1.5]}
    alarm = [False, True, False, False]
    r = execgen.reason("cell=3;d_power_db=+6.0;d_tilt_deg=+0.0", facts, alarm)
    assert r == "Raise cell 3 power by 6.0 dB: cell 1 is down (alarm raised) and cell 3 has the weakest edge RSRP (-104.00 dBm)."
    r = execgen.reason("cell=2;d_power_db=+0.0;d_tilt_deg=-3.0", facts, [False] * 4)
    assert r.startswith("Lower cell 2 tilt by 3.0 degrees: cell 2 carries the most users (77 UEs)")
    for word in ("outage", "power_drop", "misconfig", "interference", "congestion", "fault"):
        assert not re.search(word, r, re.I)
