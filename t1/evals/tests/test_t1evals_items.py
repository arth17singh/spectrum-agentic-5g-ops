"""Item generation is deterministic and the frozen hashes are stable."""

from __future__ import annotations

import json

import pytest

from t1.evals import common, freeze, gen_interface, gen_price_latency, gen_safety
from t1.evals.common import dumps


def _frozen(unit):
    return freeze.read_lines(unit)


def test_manifest_matches_files():
    assert freeze.verify() == []


@pytest.mark.parametrize("unit", common.UNITS)
def test_unit_hash_and_splits(unit):
    m = json.loads(freeze.MANIFEST.read_text())["units"][unit]
    lines = _frozen(unit)
    assert common.lines_sha256(lines) == m["sha256"]
    assert m["splits"]["dev"]["n"] + m["splits"]["test"]["n"] == m["n_items"] == len(lines)
    items = [json.loads(x) for x in lines]
    assert len({it["item_id"] for it in items}) == len(items)
    for it in items:
        assert it["split"] == common.split_of(unit, it["item_id"])
        assert dumps(it) in lines  # every line is canonical JSON
    assert 0.1 < m["splits"]["dev"]["n"] / m["n_items"] < 0.32


def test_interface_regenerates_identically():
    assert [dumps(it) for it in gen_interface.generate()] == _frozen("interface")


def test_safety_regenerates_identically():
    assert [dumps(it) for it in gen_safety.generate()] == _frozen("safety")


def test_price_latency_regenerates_identically():
    src = {u: freeze.load_items(u) for u in gen_price_latency.MIX}
    assert [dumps(it) for it in gen_price_latency.generate(src)] == _frozen("price_latency")


def test_split_is_stable_and_order_free():
    assert common.split_of("interface", "if-0000") == common.split_of("interface", "if-0000")
    s = {common.split_of("x", f"id-{i}") for i in range(200)}
    assert s == {"dev", "test"}


def test_execution_sample_regenerates_identically():
    pytest.importorskip("ran_rl")
    from t1.evals import gen_execution as E

    frozen = [json.loads(x) for x in _frozen("execution")]
    for it in [frozen[0], frozen[len(frozen) // 2], frozen[-1]]:
        spec = E._spec(it["meta"]["family"], it["meta"]["gen_index"])
        ref = E.reference(spec)
        n = int(it["item_id"].rsplit("-", 1)[1])
        again = E.make_item(it["meta"]["family"], n, spec, ref, it["meta"]["gen_index"])
        assert dumps(again) == dumps(it)


def test_execution_fork_equals_fresh_replay():
    pytest.importorskip("ran_rl")
    from t1.evals import canon as C
    from t1.evals import gen_execution as E
    from t1.evals.twin_standin import ReplayTwin

    it = json.loads(_frozen("execution")[0])
    spec = E._spec(it["meta"]["family"], it["meta"]["gen_index"])
    base = E.build_state(spec)
    forked = E._fork(base).apply(C.parse(it["expected"]["oracle_change"]))
    fresh = ReplayTwin.from_log(spec["scenario"], spec["seed"], None,
                                [{"change": C.parse(p)} for p in spec["prefix"]])
    again = fresh.apply(C.parse(it["expected"]["oracle_change"]))
    assert forked["coverage_ratio"] == again["coverage_ratio"]
    assert forked["sla_ratio"] == again["sla_ratio"]
    assert forked["sinr_mean_db"] == again["sinr_mean_db"]


def test_knowledge_matches_pinned_source_when_cached():
    import os
    from pathlib import Path

    from t1.evals import gen_knowledge as K

    cache = Path(os.environ.get("T1_EVALS_SOURCE_CACHE", str(freeze.DEFAULT_CACHE)))
    if not all((cache / f).is_file() for f, _ in K.FILES.values()):
        pytest.skip("ot-lite source files not cached (set T1_EVALS_SOURCE_CACHE or run freeze --units knowledge)")
    assert [dumps(it) for it in K.generate(K.fetch(cache))] == _frozen("knowledge")


def test_knowledge_shape():
    items = [json.loads(x) for x in _frozen("knowledge")]
    groups = {}
    for it in items:
        groups[it["meta"]["group"]] = groups.get(it["meta"]["group"], 0) + 1
        if it["expected"]["type"] == "mcq":
            n = it["expected"]["n_choices"]
            assert it["expected"]["answer"] in "ABCDEFGH"[:n]
            assert it["input"]["choices"].count("\n") == n - 1
    assert groups == {"core": 400, "telelogs": 100, "telemath": 100}
