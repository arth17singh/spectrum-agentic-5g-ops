"""Behavioural tests for the parts of the mock engine that must be real: ledger crypto, secure
aggregation maths, policy enforcement, and the scripted demo path."""

import pytest

from spectrum.engine import Engine


@pytest.fixture(scope="module")
def engine():
    return Engine(seed=7)


def run(e: Engine, seconds: float) -> None:
    for _ in range(int(seconds * 10)):
        e.tick(0.1)


def test_warm_state_is_deterministic_and_verified():
    a, b = Engine(seed=7), Engine(seed=7)
    assert a.ledger.records[-1]["hash"] == b.ledger.records[-1]["hash"]
    assert a.ledger.verify(emit=False)["ok"]
    assert a.fl.round >= 12


def test_tamper_is_detected_at_the_exact_record_and_anchor_still_holds(engine):
    seq = engine.ledger.tamper()["seq"]
    res = engine.ledger.verify(emit=False)
    assert not res["ok"]
    assert res["broken"]["seq"] == seq
    proof = engine.ledger.proof(seq)
    assert proof["anchored"] and not proof["matches"]
    engine.ledger.restore()
    assert engine.ledger.verify(emit=False)["ok"]
    assert engine.ledger.proof(seq)["matches"]


def test_secure_aggregation_masks_cancel_exactly(engine):
    engine.fl.start()
    while engine.fl.secagg is None or engine.fl.secagg["round"] < engine.fl.round:
        run(engine, 1)
    assert engine.fl.secagg["max_abs_error"] == 0
    assert engine.fl.secagg["pairwise_masks"] == 10


def test_policy_blocks_mass_shutdown_and_protects_hospitals(engine):
    site = engine.swarm.agents["site-17"]
    d = engine.trust.authorize(engine.swarm._principal(site), "cell_sleep",
                               {"id": "region", "site": site["site"], "tags": ["hospital"]},
                               {"target_count": 120}, emit=False)
    assert d["decision"] == "deny"
    assert {"P-001", "P-007"} <= set(d["policies"])


def test_large_change_needs_a_human(engine):
    coord = engine.swarm.agents["coordinator"]
    d = engine.trust.authorize(engine.swarm._principal(coord), "set_tx_power", {"id": "cells", "tags": []},
                               {"blast_radius_pct": 12.5, "plan_id": "plan-x"}, emit=False)
    assert d["decision"] == "needs_human" and "P-003" in d["policies"]


def test_full_demo_script_runs_end_to_end():
    e = Engine(seed=7)
    e.reset()
    for n in range(1, 15):
        e.beat(n)
        if n in (2, 4):
            run(e, 26)
        elif n == 5:
            run(e, 20)
        elif n == 7:
            run(e, 14)
        elif n == 9:
            pending = [i for i in e.oversight.items.values() if i["status"] == "pending"]
            e.oversight.decide(pending[0]["id"], "approved", device="mobile")
            run(e, 6)
        else:
            run(e, 4)
    assert e.proving.runs and e.proving.runs[-1]["state"] == "passed"
    assert e.release.stage_index >= 1
    assert e.swarm.incidents[-1]["status"] == "resolved"
    assert e.swarm.agents["site-17"]["status"] == "quarantined"
    assert not e.ledger.verify(emit=False)["ok"]  # beat 12 tampered on purpose
    assert not [x for x in e.bus.history if x["type"] == "system.error"]
