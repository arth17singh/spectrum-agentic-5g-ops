"""The large held-out set: candidate mapping, selection independent of workers and waves, freeze files."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

from t1.rl.remediation import heldout as H
from t3.evals import incidents as inc

FAKE_TARGETS = {"episode": {"fault_step": 2, "operator_steps": 3, "grade_step": 5}}


def fake_kept(k: int) -> bool:
    """A deterministic keep rule with very different keep rates per stratum (stratum 0 never keeps)."""
    s = k % H.N_STRATA
    if s == 0:
        return False
    rate = 0.15 + 0.8 * ((s * 37) % 11) / 10.0
    return int(hashlib.sha256(f"keep:{k}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < rate


def fake_evaluate(ks: list[int], targets: dict) -> list[dict]:
    out = []
    for k in ks:
        spec = H.spec_for(k)
        kept = fake_kept(k)
        r = {"index": k, "stratum": spec["stratum"], "attempt": spec["attempt"], "seed": spec["seed"],
             "kept": kept, "reason": "kept" if kept else "no_action already within tolerance (fake)",
             "latency_ms": 1.0}
        if kept:
            r["incident"] = {"incident_id": f"fake-{spec['seed']}", "seed": spec["seed"], "template": spec["template"],
                             "preset": spec["preset"], "tier": spec["tier"], "fault": spec["fault"],
                             "fault_cells": sorted({f["cell"] for f in spec["fault"]}), "base_cell": spec["base_cell"],
                             "split_of_seed": inc.split_of(spec["seed"]), "material_kpis": ["coverage_ratio"],
                             "suite": H.VERSION, "split": "heldout"}
        out.append(r)
    return out


def chosen_indices(gen: dict) -> list[int]:
    return [r["index"] for r in gen["chosen"]]


def test_grid_and_candidate_mapping():
    assert H.N_STRATA == 56 and H.N_TARGET == 1008
    assert len(set(H.STRATA)) == 56
    assert H.SEED_LO == H.SEED_BASE and H.SEED_HI == H.SEED_BASE + 56 * H.ATTEMPTS_MAX - 1
    seen = set()
    for k in range(3 * H.N_STRATA):
        sp = H.spec_for(k)
        s, j = k % 56, k // 56
        t, c, p = H.STRATA[s]
        assert (sp["stratum"], sp["attempt"], sp["seed"]) == (s, j, H.SEED_BASE + k)
        assert sp["tier"] == "masked" and sp["template"] == inc.TEMPLATES[t][0] and sp["preset"] == inc.PRESETS[p]
        cells = sorted({f["cell"] for f in sp["fault"]})
        if sp["template"].startswith("two-cell"):
            assert cells == sorted({c, (c + 1) % 4})
        else:
            assert cells == [c]
        assert sp["commissioning"] == inc.commissioning(sp["seed"], "masked")
        if j == 0:
            seen.add((sp["template"], c, sp["preset"]))
    assert len(seen) == 56  # every (template, base cell, preset) once per attempt round


def test_selection_is_independent_of_waves_and_order():
    base = H.generate(evaluate=fake_evaluate, targets=FAKE_TARGETS, attempts_max=60, quota=6, log=lambda m: None)
    for waves in ((1, 1), (5, 3), (60, 60), (13, 7)):
        g = H.generate(evaluate=fake_evaluate, targets=FAKE_TARGETS, attempts_max=60, quota=6, waves=waves,
                       log=lambda m: None)
        assert chosen_indices(g) == chosen_indices(base)
    # select() sees only outcomes keyed by index: any insertion order gives the same choice
    items = list(base["results"].items())
    for seed in range(3):
        random.Random(seed).shuffle(items)
        per = H.select(dict(items), base["strata"], 6)
        assert sorted(r["index"] for s in base["strata"] for r in per[s]["chosen"]) == chosen_indices(base)


def test_chosen_are_first_kept_in_attempt_order_and_short_stratum_reported():
    g = H.generate(evaluate=fake_evaluate, targets=FAKE_TARGETS, attempts_max=40, quota=5, log=lambda m: None)
    per = g["per_stratum"]
    assert not per[0]["filled"] and per[0]["evaluated"] == 40 and per[0]["chosen"] == []  # never keeps
    for s in g["strata"][1:]:
        kept_js = [j for j in range(per[s]["evaluated"]) if fake_kept(H.index_of(s, j))]
        assert [r["attempt"] for r in per[s]["chosen"]] == kept_js[:5]
        if per[s]["filled"]:
            assert per[s]["evaluated"] > kept_js[4]  # the evaluated prefix reaches the 5th kept attempt
    assert len(g["chosen"]) < 5 * H.N_STRATA


def test_max_candidates_keeps_contiguous_prefixes():
    g = H.generate(evaluate=fake_evaluate, targets=FAKE_TARGETS, strata=range(4), quota=50, attempts_max=100,
                   max_candidates=37, log=lambda m: None)
    assert len(g["results"]) == 37
    for s in range(4):
        js = sorted(r["attempt"] for r in g["results"].values() if r["stratum"] == s)
        assert js == list(range(len(js)))


def test_outputs_freeze_and_verify(tmp_path: Path):
    g = H.generate(evaluate=fake_evaluate, targets=FAKE_TARGETS, strata=range(1, 5), quota=3, attempts_max=30,
                   log=lambda m: None)
    meta = H.write_outputs(g, tmp_path, "0" * 40, None, None, smoke=True)
    data = (tmp_path / H.SUITE_NAME).read_bytes()
    assert meta["sha256"] == hashlib.sha256(data).hexdigest() and meta["n_incidents"] == 12
    assert (tmp_path / H.HASH_PATH.name).read_text() == f"{meta['sha256']}  {H.SUITE_NAME}\n"
    rows, digest = H.load_heldout(tmp_path / H.SUITE_NAME)
    assert digest == meta["sha256"] and [r["seed"] for r in rows] == sorted(r["seed"] for r in rows)
    log = [json.loads(x) for x in (tmp_path / H.LOG_NAME).read_text().splitlines()]
    assert len(log) == meta["candidates_evaluated"] and sum(x["chosen"] for x in log) == 12
    assert hashlib.sha256((tmp_path / H.LOG_NAME).read_bytes()).hexdigest() == meta["candidate_log"]["sha256"]
    (tmp_path / H.SUITE_NAME).write_bytes(data.replace(b"fake-", b"fakE-", 1))
    with pytest.raises(RuntimeError):
        H.load_heldout(tmp_path / H.SUITE_NAME)


def test_reservation_is_clean_and_matches_the_generator():
    res = H.load_reservation()
    assert res["generator_input_sha256"] == H.generator_input_sha256()
    assert res["seed_block"] == [H.SEED_LO, H.SEED_HI] and res["n_candidate_specs"] == 56 * H.ATTEMPTS_MAX
    assert res["exclusion"]["clean"] and not res["exclusion"]["ranges_overlapping"]
    assert not res["exclusion"]["explicit_collisions"] and res["exclusion"]["literal_scan"]["n_hits"] == 0
    assert not any(res["spec_hash_overlap"].values())
    H.check_reservation(res)  # raises SystemExit if the generator drifted


def test_number_scan_ignores_identifiers_and_hex(tmp_path: Path):
    lo, hi = H.SEED_LO, H.SEED_LO + 999
    (tmp_path / "a.json").write_text(json.dumps({"sha": f"ab{lo + 5}cd", "seed": lo + 7, "id": f"x_{lo + 9}"}))
    (tmp_path / "b.py").write_text(f"SEED = {lo + 11:_}\nOTHER = {hi + 1}\n")
    scan = H._num_scan(lo, hi, root=tmp_path, own=())
    assert sorted(h["value"] for h in scan["hits"]) == [lo + 7, lo + 11]
