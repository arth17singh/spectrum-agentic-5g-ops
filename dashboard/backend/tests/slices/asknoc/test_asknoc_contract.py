"""Contract with the organisers' checkpoint kit, pinned without any live call.

The fixture holds the front door's answer shapes as the kit code returns them (checkpoint_kit.py L241-262,
wrapped by the Gateway as MCP tools/call results) and the shapes seen in the one approved live run
(case-d59bd7e0, 2 Nov 2026: start_case returned at once, then SUBMITTED/WORKING, then COMPLETED with a reply).
The drift guard runs only where the private kit folder is present (ASKNOC_KIT_DIR); it parses the kit file
with `ast` (never imports it: the module calls STS at import) and compares the Lambda source hash with the
one that was proved equal to the deployed functions' code (plan section 1.1).
"""

import ast
import hashlib
import json
import os
from pathlib import Path

import pytest

from spectrum.slices.asknoc.kit import classify
from spectrum.slices.asknoc.service import RUNNING, parse_waiting

FIX = json.loads((Path(__file__).parent / "fixtures" / "kit_shapes.json").read_text())
LAMBDA_SRC_SHA256 = "04009b0a219ee7c0277f2b08a2f7c24a2409dcb53c04c5c1359374d5e9b3bc8f"


def wrap(data) -> tuple[int, dict, str]:
    return 200, {}, json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"content": [
        {"type": "text", "text": json.dumps(data, separators=(",", ":"))}]}})


@pytest.mark.parametrize("name", sorted(FIX["answers"]))
def test_fixture_answers_classify_as_documented(name):
    case = FIX["answers"][name]
    out = classify(*wrap(case["data"]))
    assert out.kind == case["kind"], name
    if case["kind"] == "ok":
        assert set(out.data) >= set(case["required_keys"])


def test_running_states_are_the_kits():
    assert set(RUNNING) == set(FIX["running_states"])


def test_waiting_approvals_change_is_a_json_string_in_the_kit():
    w = parse_waiting(FIX["answers"]["status_completed_waiting"]["data"]["waiting_approvals"])
    assert w[0]["approval_id"] == "chg-bde360f4d4b4" and w[0]["change"] == "cell=2;d_power_db=-1.0;d_tilt_deg=+1.0"


def test_deny_from_the_kit_check_log():
    status, headers, raw = 200, {}, json.dumps({"jsonrpc": "2.0", "id": 1, "error": FIX["deny_error"]})
    assert classify(status, headers, raw).kind == "denied"


@pytest.mark.skipif(not os.environ.get("ASKNOC_KIT_DIR"), reason="private kit folder not present (CI, gate)")
def test_drift_guard_lambda_src_unchanged():
    src = (Path(os.environ["ASKNOC_KIT_DIR"]) / "checkpoint_kit.py").read_text()
    lam = next(n.value.value for n in ast.parse(src).body
               if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "LAMBDA_SRC")
    assert hashlib.sha256(lam.encode()).hexdigest() == LAMBDA_SRC_SHA256, \
        "the kit's front-door code changed: re-check the shapes in fixtures/kit_shapes.json"
