"""No-op changes (lead decision): ran___propose answers "no change" cleanly; ran___apply_change
refuses an empty or all-zero change with a ToolError before any table write. W1's canon stays
strict on an empty change and keeps zero cells in a real change."""
from __future__ import annotations

import pytest

from fakes import FakeTwin, fake_kpis
from t3.tools import app, store
from test_ran_tools import APPLY, CASE, World


@pytest.fixture(autouse=True)
def _reset():
    yield
    app.set_deps(None)


class _Policy:
    model_id, version = "ppo-maveric", "v0-test"

    def __init__(self, out):
        self.out = out

    def propose(self, state):
        return self.out


@pytest.mark.parametrize("proposal", [
    [], "", None,
    [{"cell": 0, "d_power_db": 0.0, "d_tilt_deg": 0.0}, {"cell": 3, "d_power_db": -0.0, "d_tilt_deg": 0.0}],
    [{"cell": 1, "d_power_db": 0.04, "d_tilt_deg": -0.03}],          # zero after the 0.1 rounding
    "cell=1;d_power_db=+0.0;d_tilt_deg=+0.0",
])
def test_propose_no_change_is_a_clean_answer(proposal):
    w = World(policy=_Policy(proposal))
    out = w.call("ran___propose", case_id=CASE)
    assert not out.get("error"), out
    assert out["no_change"] is True and out["item_id"] is None and out["change"] is None
    assert out["reason"] == "policy proposes no change"
    assert w.case_doc()["predictions"] == []                            # no item remembered
    assert [k for k in w.ddb.items if not k.startswith("t3-")] == []    # no approval row
    assert w.records() == []                                            # no change record


def test_propose_real_change_keeps_zero_cells():
    w = World(policy=_Policy([{"cell": 0, "d_power_db": 0.0, "d_tilt_deg": 0.0},
                              {"cell": 2, "d_power_db": 0.0, "d_tilt_deg": 1.0}]))
    out = w.call("ran___propose", case_id=CASE)
    assert out["change"] == "cell=0;d_power_db=+0.0;d_tilt_deg=+0.0|cell=2;d_power_db=+0.0;d_tilt_deg=+1.0"
    assert out["item_id"] == "step-0" and "no_change" not in out


def test_propose_malformed_policy_output_is_a_tool_error_not_a_crash():
    w = World(policy=_Policy([{"cell": "x", "d_power_db": 1.0, "d_tilt_deg": 0.0}]))
    out = w.call("ran___propose", case_id=CASE)
    assert "unusable change" in out["error"] or "bad change" in out["error"]
    assert "ValueError" not in out["error"]


@pytest.mark.parametrize("change", ["", "cell=1;d_power_db=+0.0;d_tilt_deg=+0.0",
                                    "cell=0;d_power_db=+0.0;d_tilt_deg=+0.0|cell=2;d_power_db=-0.0;d_tilt_deg=+0.0"])
def test_apply_empty_or_all_zero_change_is_refused_before_any_write(change):
    w = World()
    args = dict(APPLY, change=change)
    key = w.approved(args)            # even an approved one is refused, and the approval stays unused
    before = {k: dict(v) for k, v in w.ddb.items.items()}
    out = w.call("ran___apply_change", **args)
    assert ("bad change" in out["error"]) or ("nothing to apply" in out["error"])
    assert "ValueError" not in out["error"]
    assert w.ddb.items == before and w.ddb.plain(key)["status"] == "APPROVED"
    assert not [c for c in w.ddb.calls if c[0] in ("put_item", "transact_write_items")]
    assert w.records() == []


def test_apply_all_zero_change_in_local_mode_creates_no_pending_row(tmp_path):
    s = store.from_env({"T3_STORE": f"local:{tmp_path}"})
    app.set_deps(app.Deps(store=s, twin_factory=FakeTwin, kpis=fake_kpis, tool_name_fallback=True))
    out = app.handler({"_tool": "ran___apply_change",
                       **dict(APPLY, change="cell=1;d_power_db=+0.0;d_tilt_deg=+0.0")}, None)
    assert "nothing to apply" in out["error"] and s.pending() == []
