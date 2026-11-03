"""Our generated rules, evaluated by the real Cedar engine (cedarpy), offline.

Skipped when cedarpy is not importable. Run with it, for example:
    uv pip install --target <dir> cedarpy  &&  PYTHONPATH=<dir> python -m pytest t3/policies/tests

AgentCore extensions that plain Cedar cannot parse are translated for the test only:
- `when guardrails { ... }` becomes `when { context.guardrails_ok }`, so a request states whether the
  prompt-attack guardrail would pass;
- `suppressOutput` rules (answer screening) are left out: they act on the answer, not the decision.
The kit's own rules that matter here are restated in plain Cedar below, written from the guide's rule
table (team agents may ask nlm___ask_nlm if the guardrail passes; may_change agents may call
work___apply_change), not copied from the kit.
"""
from __future__ import annotations

import re

import pytest

from t3.policies import generate as g

cedarpy = pytest.importorskip("cedarpy")

GW = "arn:aws:bedrock-agentcore:us-east-2:000000000000:gateway/team13-checkpoint-test"
GW_ENTITY = f'AgentCore::Gateway::"{GW}"'
TEAM = ('principal.hasTag("swarm") && principal.getTag("swarm") == "team13" && '
        'principal.hasTag("role") && principal.getTag("role") == "agent"')
KIT_EQUIVALENT = [
    # the kit's allow_nlm_safe_questions, as the guide describes it
    f'permit (principal is AgentCore::OAuthUser, action == AgentCore::Action::"nlm___ask_nlm", resource == {GW_ENTITY}) '
    f'when {{ context.guardrails_ok && {TEAM} }};',
    # the kit's may_apply_change (may_change = [noc_lead])
    f'permit (principal is AgentCore::OAuthUser, action == AgentCore::Action::"work___apply_change", resource == {GW_ENTITY}) '
    f'when {{ {TEAM} && principal.hasTag("agent") && ["noc_lead"].contains(principal.getTag("agent")) }};',
]


def plain(statement: str) -> str | None:
    if statement.startswith("suppressOutput"):
        return None
    s = g.fill(statement, GW)
    return re.sub(r"when guardrails \{.*?\n\}", "when { context.guardrails_ok }", s, flags=re.S)


def policy_text(extra_rules=()) -> str:
    rules = list(g.rules()) + list(extra_rules)
    stmts = [p for p in (plain(r["statement"]) for r in rules) if p] + KIT_EQUIVALENT
    return "\n".join(stmts)


def decide(policies: str, agent: str, action: str, inp: dict, guardrails_ok: bool = True) -> str:
    entities = [
        {"uid": {"type": "AgentCore::OAuthUser", "id": f"agent-{agent}"}, "attrs": {}, "parents": [],
         "tags": {"agent": agent, "swarm": "team13", "role": "agent"}},
        {"uid": {"type": "AgentCore::Gateway", "id": GW}, "attrs": {}, "parents": []},
    ]
    request = {"principal": f'AgentCore::OAuthUser::"agent-{agent}"', "action": f'AgentCore::Action::"{action}"',
               "resource": GW_ENTITY, "context": {"input": inp, "guardrails_ok": guardrails_ok}}
    result = cedarpy.is_authorized(request, policies, entities)
    assert not result.diagnostics.errors, result.diagnostics.errors
    return "ALLOW" if result.allowed else "DENY"


CASE = {"case_id": "case-0000abcd"}
Q = {**CASE, "question": "What does downtilt do?"}


def test_both_nlm_names_are_denied_while_the_nlm_is_paused():
    live = policy_text()
    assert decide(live, "ran_explainer", "nlm___ask", Q) == "ALLOW"
    assert decide(live, "ran_explainer", "nlm___ask_nlm", Q) == "ALLOW"
    paused = policy_text([g.pause_rule("otel-nlm", "v0")])
    for name in ("nlm___ask", "nlm___ask_nlm"):
        assert decide(paused, "ran_explainer", name, Q) == "DENY", name
        assert decide(paused, "ran_explainer", name, {**Q, "model_version": "v0"}) == "DENY", name
        # naming another version passes Cedar; the tool then refuses a version it does not serve
        assert decide(paused, "ran_explainer", name, {**Q, "model_version": "v1"}) == "ALLOW", name


def test_pausing_the_response_model_denies_predict_only():
    paused = policy_text([g.pause_rule("t2-response", "v0")])
    pred = {**CASE, "item_id": "step-0", "change": "cell=1;d_power_db=+0.0;d_tilt_deg=+1.0"}
    assert decide(policy_text(), "ran_optimizer", "ran___predict", pred) == "ALLOW"
    assert decide(paused, "ran_optimizer", "ran___predict", pred) == "DENY"
    assert decide(paused, "ran_optimizer", "ran___propose", CASE) == "ALLOW"


def test_kit_nlm_name_is_held_to_our_rules():
    live = policy_text()
    for agent in ("noc_lead", "ran_monitor", "ran_optimizer"):   # only ran_explainer, on either name
        assert decide(live, agent, "nlm___ask_nlm", Q) == "DENY", agent
        assert decide(live, agent, "nlm___ask", Q) == "DENY", agent
    assert decide(live, "ran_explainer", "nlm___ask_nlm", {"question": "q"}) == "DENY"            # no case id
    assert decide(live, "ran_explainer", "nlm___ask_nlm", {"case_id": "x-1", "question": "q"}) == "DENY"
    for name in ("nlm___ask", "nlm___ask_nlm"):                                                      # prompt attack
        assert decide(live, "ran_explainer", name, Q, guardrails_ok=False) == "DENY", name


@pytest.mark.parametrize("approval", [None, "PENDING", "MISSING", "NOT_ALLOWED", "REJECTED", "APPLIED",
                                      "approved", "Approved", "APPROVED ", " APPROVED", "", "APPROVED\n"])
def test_apply_is_denied_unless_the_injected_approval_is_exactly_approved(approval):
    live = policy_text()
    inp = {**CASE, "item_id": "step-0", "change": "cell=1;d_power_db=+0.0;d_tilt_deg=+1.0", "reason": "r"}
    if approval is not None:
        inp["approval"] = approval
    assert decide(live, "noc_lead", "ran___apply_change", inp) == "DENY"


def test_only_the_lead_with_approved_passes_cedar():
    live = policy_text()
    inp = {**CASE, "item_id": "step-0", "change": "cell=1;d_power_db=+0.0;d_tilt_deg=+1.0", "approval": "APPROVED"}
    assert decide(live, "noc_lead", "ran___apply_change", inp) == "ALLOW"
    for agent in ("ran_monitor", "ran_optimizer", "ran_explainer"):
        assert decide(live, agent, "ran___apply_change", inp) == "DENY", agent
    assert decide(live, "noc_lead", "ran___apply_change", {**inp, "case_id": "nope"}) == "DENY"


def test_kit_apply_tool_is_denied_to_our_agents_even_when_approved():
    live = policy_text()
    inp = {**CASE, "item_id": "step-0", "change": "cell=1;d_power_db=+0.0;d_tilt_deg=+1.0", "approval": "APPROVED"}
    for agent in ("noc_lead", "ran_monitor", "ran_optimizer", "ran_explainer"):
        assert decide(live, agent, "work___apply_change", inp) == "DENY", agent


def test_per_agent_permits():
    live = policy_text()
    pred = {**CASE, "item_id": "step-0", "change": "cell=1;d_power_db=+0.0;d_tilt_deg=+1.0"}
    table = {"ran___observe": (CASE, {"noc_lead", "ran_monitor"}),
             "ran___propose": (CASE, {"ran_optimizer"}),
             "ran___predict": (pred, {"ran_optimizer"}),
             "ran___preflight": (pred, {"ran_optimizer"}),
             "nlm___ask": (Q, {"ran_explainer"})}
    for action, (inp, allowed) in table.items():
        for agent in ("noc_lead", "ran_monitor", "ran_optimizer", "ran_explainer"):
            want = "ALLOW" if agent in allowed else "DENY"
            assert decide(live, agent, action, inp) == want, (action, agent)
