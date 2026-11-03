"""ran___seed_case sets twin state: only the app's token may call it; every agent is refused."""
from __future__ import annotations

import pytest

from t3.policies import generate as g
from t3.tools import catalog

AGENTS = ("noc_lead", "ran_monitor", "ran_optimizer", "ran_explainer")


def _seed_rules():
    return [r for r in g.rules() if catalog.SEED_FULL in r["actions"]]


def test_seed_has_an_app_only_permit_and_a_forbid_for_everyone_else():
    rules = {r["name"]: r for r in _seed_rules()}
    assert set(rules) == {"team13_ran_seed_app_only", "team13_ran_seed_never_agents"}
    permit, forbid = rules["team13_ran_seed_app_only"]["statement"], rules["team13_ran_seed_never_agents"]["statement"]
    assert permit.startswith("permit") and g.APP_EXPR in permit and 'like "case-*"' in permit
    assert forbid.startswith("forbid (principal,") and f"unless {{ {g.APP_EXPR} }}" in forbid
    # no agent permit names the seed tool, and no agent prompt does either
    for r in g.rules():
        if r["statement"].startswith("permit") and catalog.SEED_FULL in r["actions"]:
            assert 'getTag("role") == "agent"' not in r["statement"]
    assert catalog.SEED not in sum(g.AGENT_TOOLS.values(), [])
    assert g.check(g.rules()) == []


GW = "arn:aws:bedrock-agentcore:us-east-2:000000000000:gateway/team13-checkpoint-test"


def _decide(role: str, agent: str, inp: dict) -> str:
    import re
    cedarpy = pytest.importorskip("cedarpy")   # the real Cedar engine, when installed
    policies = "\n".join(re.sub(r"when guardrails \{.*?\n\}", "when { context.guardrails_ok }",
                                g.fill(r["statement"], GW), flags=re.S)
                         for r in g.rules() if not r["statement"].startswith("suppressOutput"))
    entities = [{"uid": {"type": "AgentCore::OAuthUser", "id": f"u-{agent}"}, "attrs": {}, "parents": [],
                 "tags": {"agent": agent, "swarm": "team13", "role": role}},
                {"uid": {"type": "AgentCore::Gateway", "id": GW}, "attrs": {}, "parents": []}]
    req = {"principal": f'AgentCore::OAuthUser::"u-{agent}"', "action": f'AgentCore::Action::"{catalog.SEED_FULL}"',
           "resource": f'AgentCore::Gateway::"{GW}"', "context": {"input": inp, "guardrails_ok": True}}
    return cedarpy.is_authorized(req, policies, entities).decision.name.upper()


@pytest.mark.parametrize("agent", AGENTS)
def test_no_agent_can_seed_a_case(agent):
    assert _decide("agent", agent, {"case_id": "case-abc12345-i01s0"}) == "DENY"


def test_the_app_can_seed_with_a_case_id_only():
    assert _decide("app", "team_app", {"case_id": "case-abc12345-i01s0"}) == "ALLOW"
    assert _decide("app", "team_app", {}) == "DENY"
