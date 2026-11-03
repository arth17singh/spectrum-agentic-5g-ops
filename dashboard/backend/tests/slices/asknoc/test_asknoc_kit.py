"""The Gateway client: request shape, the three-tool allow-list, and the classification of every answer."""

import json

import pytest

from slices.asknoc.asknoc_fakes import (MARK_URL, FakePost, jwt, kit_dir, mcp_deny, mcp_ok, started, status_done,
                                        transport)
from spectrum.slices.asknoc.config import KitConfig, Secret
from spectrum.slices.asknoc.kit import ALLOWED, KitClient, classify, deny_rule


def client(tmp_path, post):
    cfg = KitConfig(str(kit_dir(tmp_path)), env={})
    return KitClient(cfg, post), cfg


def test_request_is_the_kits_jsonrpc_shape(tmp_path):
    post = FakePost({"front___start_case": [started()]})
    kit, cfg = client(tmp_path, post)
    out = kit.call("front___start_case", {"request": "hello"}, 45)
    name, req, headers, timeout = post.calls[0]
    # checkpoint_kit.py L874-878 / t3/suite/common.py:162-165
    assert req["jsonrpc"] == "2.0" and req["method"] == "tools/call"
    assert req["params"] == {"name": "front___start_case", "arguments": {"request": "hello"}}
    assert headers["Content-Type"] == "application/json"
    assert headers["Accept"] == "application/json, text/event-stream"
    assert headers["Authorization"] == "Bearer " + cfg.load().token.reveal()
    assert timeout == 45
    assert out.kind == "ok" and out.data == {"case_id": "case-d59bd7e0", "state": "TASK_STATE_SUBMITTED"}
    assert out.gateway_date == "2026-11-03T05:01:23.000Z"


@pytest.mark.parametrize("tool", ["ran___apply_change", "swarm___ask_agent", "work___lookup", "nlm___ask",
                                  "case___open_case", "front___approve", ""])
def test_only_the_three_front_door_tools(tmp_path, tool):
    post = FakePost()
    kit, _ = client(tmp_path, post)
    with pytest.raises(ValueError):
        kit.call(tool, {}, 1)
    assert post.calls == []
    assert set(ALLOWED) == {"front___start_case", "front___continue_case", "front___case_status"}


def test_sse_answer_is_parsed(tmp_path):
    post = FakePost({"front___case_status": [status_done()[:2] + (mcp_ok({"state": "TASK_STATE_WORKING"}, sse=True)[2],)]})
    kit, _ = client(tmp_path, post)
    assert kit.call("front___case_status", {"case_id": "case-d59bd7e0"}, 5).data["state"] == "TASK_STATE_WORKING"


def test_token_is_reread_when_the_file_changes(tmp_path):
    post = FakePost({"front___case_status": [status_done()]})
    kit, cfg = client(tmp_path, post)
    kit.call("front___case_status", {"case_id": "case-d59bd7e0"}, 5)
    (cfg.dir / "tokens" / "team_app.txt").write_text(jwt({"role": "app", "n": 2}))
    kit.call("front___case_status", {"case_id": "case-d59bd7e0"}, 5)
    assert post.calls[0][2]["Authorization"] != post.calls[1][2]["Authorization"]


def test_secrets_never_in_repr(tmp_path):
    _, cfg = client(tmp_path, FakePost())
    view = cfg.load()
    assert repr(view.token) == "<redacted>" and str(view.gateway_url) == "<redacted>"
    assert "eyJ" not in repr(view) and MARK_URL not in repr(view)
    assert repr(Secret("x")) == "<redacted>"


DENY_NAMED = ("Tool Execution Denied: Tool call not allowed due to policy enforcement [Policy evaluation denied "
              "due to team13_ran_require_human_approval-veivx1abcd]")


@pytest.mark.parametrize("answer,kind", [
    (started(), "ok"),
    (mcp_ok({"error": "unknown case case-unknown0"}), "tool_error"),
    (mcp_ok({"error": "<urlopen error timed out>"}), "tool_error"),
    (mcp_deny(), "denied"),
    (mcp_deny(DENY_NAMED), "denied"),
    ((403, {}, json.dumps({"jsonrpc": "2.0", "error": {"code": -32002, "message": "Tool Execution Denied: x"}})), "denied"),
    ((401, {}, '{"message":"Unauthorized"}'), "auth"),
    ((403, {}, '{"message":"Forbidden"}'), "auth"),
    ((429, {}, "slow down"), "throttled"),
    ((502, {}, "bad gateway"), "server"),
    ((504, {}, "timeout"), "server"),
    ((400, {}, "bad"), "client_error"),
    ((200, {}, "<html>captive portal</html>"), "unparseable"),
    ((200, {}, json.dumps({"jsonrpc": "2.0", "id": 1, "error": {"code": -32602, "message": "bad params"}})), "rpc_error"),
    ((200, {}, json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"isError": True, "content": [{"type": "text", "text": "no"}]}})), "is_error"),
])
def test_classify(answer, kind):
    assert classify(*answer).kind == kind


def test_deny_rule_names_the_rule_without_suffix():
    assert deny_rule(DENY_NAMED) == "team13_ran_require_human_approval"
    assert deny_rule(mcp_deny()[2]) is None


@pytest.mark.parametrize("phase", ["before_send", "after_send"])
def test_transport_errors_keep_their_phase(tmp_path, phase):
    kit, _ = client(tmp_path, FakePost({"front___start_case": [transport(phase, "dns")]}))
    out = kit.call("front___start_case", {"request": "x"}, 1)
    assert out.kind == phase and out.maybe_delivered == (phase == "after_send")


def test_5xx_and_429_may_have_been_delivered():
    assert classify(502, {}, "").maybe_delivered and classify(429, {}, "").maybe_delivered
    assert not classify(401, {}, "").maybe_delivered


def test_not_configured(tmp_path):
    cfg = KitConfig(None, env={})
    assert not cfg.load().configured and "ASKNOC_KIT_DIR" in cfg.load().problem


def test_half_written_kit_state_keeps_last_good(tmp_path):
    _, cfg = client(tmp_path, FakePost())
    assert cfg.load().configured
    (cfg.dir / "kit-state.json").write_text('{"gateway_url": "https://half')
    view = cfg.load()
    assert view.configured and view.stale
