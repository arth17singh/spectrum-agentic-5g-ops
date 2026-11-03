"""The Cedar rules generate deterministically, match the committed files and pass the offline checks."""
from __future__ import annotations

import json

import pytest

from t3.policies import generate as g
from t3.tools import catalog

GW = "arn:aws:bedrock-agentcore:us-east-2:000000000000:gateway/team13-checkpoint-abc123"


def test_generation_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    g.write(a)
    g.write(b)
    files_a = {p.relative_to(a): p.read_bytes() for p in a.rglob("*") if p.is_file()}
    files_b = {p.relative_to(b): p.read_bytes() for p in b.rglob("*") if p.is_file()}
    assert files_a == files_b and len(files_a) >= 10


def test_committed_files_are_current():
    assert g.diff() == [], "run: python -m t3.policies.generate"


def test_offline_checks_pass():
    assert g.check(g.rules()) == []
    assert g.main(["--check"]) == 0


def test_rule_set():
    rules = {r["name"]: r for r in g.rules()}
    assert set(rules) == {
        "team13_ran_allow_observe", "team13_ran_allow_propose", "team13_ran_allow_predict",
        "team13_ran_allow_preflight", "team13_ran_allow_apply_change", "team13_ran_nlm_safe_questions",
        "team13_ran_screen_nlm_answers", "team13_ran_screen_nlm_answers_kit_name", "team13_ran_require_case_id",
        "team13_ran_require_human_approval", "team13_ran_no_kit_apply_change", "team13_ran_nlm_kit_name_explainer_only",
        "team13_ran_seed_app_only", "team13_ran_seed_never_agents"}
    # M2: both NLM names reach our code, so both are screened
    assert rules["team13_ran_screen_nlm_answers"]["actions"] == ["nlm___ask"]
    assert rules["team13_ran_screen_nlm_answers_kit_name"]["actions"] == ["nlm___ask_nlm"]
    # M1: our agents can never call the kit's work___apply_change (it would consume the approval)
    kit_apply = rules["team13_ran_no_kit_apply_change"]
    assert kit_apply["actions"] == ["work___apply_change"] and kit_apply["mode"] == g.LOOSE
    assert kit_apply["statement"].startswith("forbid")
    assert '["noc_lead", "ran_explainer", "ran_monitor", "ran_optimizer"]' in kit_apply["statement"]
    assert "require_human_approval" not in kit_apply["name"]  # the interceptor must not rewrite this refusal
    assert '["noc_lead", "ran_monitor"]' in rules["team13_ran_allow_observe"]["statement"]
    assert '["noc_lead"]' in rules["team13_ran_allow_apply_change"]["statement"]
    for t in ("propose", "predict", "preflight"):
        assert '["ran_optimizer"]' in rules[f"team13_ran_allow_{t}"]["statement"]
    nlm = rules["team13_ran_nlm_safe_questions"]["statement"]
    assert '"nlm___ask"' in nlm and '["ran_explainer"]' in nlm and "context.input.question" in nlm
    assert nlm.count('.confidenceScore.lessThanOrEqual(decimal("0.4"))') == 2
    screen = rules["team13_ran_screen_nlm_answers"]
    assert screen["statement"].startswith("suppressOutput") and screen["mode"] == g.LOOSE
    assert screen["statement"].count('.confidenceScore.greaterThan(decimal("0.2"))') == 3
    approval = rules["team13_ran_require_human_approval"]
    assert approval["mode"] == g.LOOSE and approval["actions"] == ["ran___apply_change"]
    assert 'context.input.approval == "APPROVED"' in approval["statement"]
    assert sorted(rules["team13_ran_require_case_id"]["actions"]) == sorted(catalog.all_tools() + ["nlm___ask_nlm"])
    assert 'context.input.case_id like "case-*"' in rules["team13_ran_require_case_id"]["statement"]


def test_checks_catch_mistakes():
    rules = g.rules()
    broken = [dict(r) for r in rules]
    broken[0]["statement"] = broken[0]["statement"].replace("};", "};;")
    broken[1]["mode"] = g.LOOSE
    broken[2]["statement"] = broken[2]["statement"].replace("principal.getTag", "context.input.agent_tag || principal.getTag", 1)
    problems = g.check(broken)
    assert any("exactly one statement" in p for p in problems)
    assert any("should use FAIL_ON_ANY_FINDINGS" in p for p in problems)
    assert any("declares no input field 'agent_tag'" in p for p in problems)
    without_case = [r for r in rules if not r["name"].endswith("require_case_id")]
    assert any("the case-id forbid must cover every tool" in p for p in g.check(without_case))
    without_permit = [r for r in rules if r["name"] != "team13_ran_allow_preflight"]
    assert any("ran___preflight: no permit" in p for p in g.check(without_permit))


def test_pause_rules():
    r = g.pause_rule("t2-response", "v0-degraded")
    assert r["name"] == "team13_ranpause_predict_t2_response_v0_degraded" and r["mode"] == g.LOOSE
    assert r["statement"] == (
        'forbid (principal, action == AgentCore::Action::"ran___predict", resource == @GW@)\n'
        'unless { context.input has model_version && context.input.model_version != "v0-degraded" };')
    assert not r["name"].startswith("team13_pause_nlm")  # the kit's `resume nlm` deletes that prefix
    nlm_pause = g.pause_rule("otel-nlm", "v0")
    assert nlm_pause["actions"] == ["nlm___ask", "nlm___ask_nlm"]   # M2: both names reach our NLM
    assert 'action in [AgentCore::Action::"nlm___ask", AgentCore::Action::"nlm___ask_nlm"]' in nlm_pause["statement"]
    assert g.check([nlm_pause] + g.rules()) == []
    half = dict(nlm_pause, actions=["nlm___ask"],
                statement=nlm_pause["statement"].replace(', AgentCore::Action::"nlm___ask_nlm"', ""))
    assert any("must cover both" in p for p in g.check([half] + g.rules()))
    no_kit_screen = [r for r in g.rules() if not r["name"].endswith("_kit_name")]
    assert any("answer screen must cover both" in p for p in g.check(no_kit_screen))
    with pytest.raises(ValueError):
        g.pause_rule("t2-response", 'v0" || true')  # no Cedar injection through the version
    assert g.pause_rule("ppo-maveric", "v0")["actions"] == ["ran___propose"]
    long = g.pause_rule("t2-response", "2026-11-02T23:59:59-sha-0123456789abcdef", suffix="1793651100")
    assert len(long["name"]) <= g.NAME_MAX and g.NAME_RE.fullmatch(long["name"])
    assert long["name"] != g.pause_rule("t2-response", "2026-11-02T23:59:59-sha-0123456789abcdef",
                                        suffix="1793651101")["name"]
    assert g.check([long] + g.rules()) == []
    with pytest.raises(KeyError):
        g.pause_rule("unknown-model", "v0")


def test_fill_puts_in_the_gateway_arn_only():
    s = g.fill(g.rules()[0]["statement"], GW)
    assert f'resource == AgentCore::Gateway::"{GW}"' in s and "@GW@" not in s
    with pytest.raises(ValueError):
        g.fill("x", "not-an-arn")


def test_manifest_lists_every_file():
    m = json.loads(g.render()["manifest.json"])
    assert [r["name"] for r in m["rules"]] == [r["name"] for r in g.rules()]
    assert set(m["tools"]) == set(catalog.all_tools())
