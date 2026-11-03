"""Team13's Cedar rules for the RAN tools, generated deterministically and checked offline.

    python -m t3.policies.generate            write t3/policies/generated/ (rules + manifest.json)
    python -m t3.policies.generate --check   exit 1 if the committed files differ or a check fails

The files keep the token @GW@ where the Gateway's ARN goes; `fill()` (used by
t3/deploy/register.py) puts the real ARN in at deploy time, so no account id is committed.

Style and validation modes follow the organisers' kit: permits are created with
FAIL_ON_ANY_FINDINGS; forbid and suppressOutput rules with IGNORE_ALL_FINDINGS (the validator calls
them "overly restrictive" by design). Guardrail scores are read per category
(`["CATEGORY"].confidenceScore`), never through an aggregate.

Rules (prefix `<team>_ran_`):
- allow_<tool>: one permit per tool, naming the agents that may call it (AGENT_TOOLS);
- nlm_safe_questions: the NLM permit, only if the question scores <= 0.4 for jailbreak and
  prompt injection (same semantics as the kit's allow_nlm_safe_questions);
- screen_nlm_answers: suppressOutput when the answer scores > 0.2 for violence, hate or
  misconduct (same semantics as the kit's screen_nlm_answers), on nlm___ask AND nlm___ask_nlm:
  after the takeover of target `nlm` both names reach our code;
- no_kit_apply_change: forbid the kit's work___apply_change for our four agents. It hashes its
  approval exactly like ran___apply_change, so it would consume an approval without changing the twin;
- require_case_id: forbid every one of our tools without a case id like "case-*";
- require_human_approval: forbid ran___apply_change unless the checkpoint injected
  approval == "APPROVED". The kit's interceptor rewrites the refusal only when the Cedar error
  mentions "require_human_approval", so that text must stay in the name;
- pause templates (`pause_rule`, prefix `<team>_ranpause_`, contract amendment 1 A8): withdraw one
  model version by forbidding the tool that serves it, unless the caller names another version.
  Never `<team>_pause_nlm...`: the kit's `resume nlm` deletes every rule with that prefix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from t3.tools import catalog

STRICT, LOOSE = "FAIL_ON_ANY_FINDINGS", "IGNORE_ALL_FINDINGS"
NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
NAME_MAX = 48
HERE = Path(__file__).resolve().parent
OUT = HERE / "generated"

# Who may call what. noc_lead is the only agent that changes the network (it must also be in the
# kit's may_change list, or the interceptor never asks for an approval).
AGENT_TOOLS = {
    "noc_lead": [catalog.OBSERVE, catalog.APPLY],
    "ran_monitor": [catalog.OBSERVE],
    "ran_optimizer": [catalog.PROPOSE, catalog.PREDICT, catalog.PREFLIGHT],
    "ran_explainer": ["@NLM@"],
}
APPLY_AGENTS = [a for a, tools in AGENT_TOOLS.items() if catalog.APPLY in tools]

# Which tool serves which model, for pause rules: pausing a model forbids the tool that serves it.
MODEL_TOOLS = {
    "t2-response": catalog.PREDICT,     # W9 Track 2 response model (scored by the trust gate)
    "ppo-maveric": catalog.PROPOSE,     # W9 RL policy
    "otel-nlm": "@NLM@",                # our tuned OTel model (or its labelled stand-in)
}
EXAMPLE_PAUSES = [("t2-response", "v0"), ("t2-response", "v0-degraded"), ("ppo-maveric", "v0"), ("otel-nlm", "v0")]

# Kit actions our rules may name: the kit's NLM tool (now served by our Lambda) and its example apply.
KNOWN_KIT_ACTIONS = {catalog.KIT_NLM_ASK, catalog.KIT_WORK_APPLY}
KIT_WORK_APPLY_FIELDS = {"case_id", "item_id", "change", "reason", "approval", "special_context"}  # kit 376-382

GUARD_IN = ("JAILBREAK", "PROMPT_INJECTION")
GUARD_OUT = ("VIOLENCE", "HATE", "MISCONDUCT")
OUTPUT_PATHS = ["$.content[*].text", "$.result.content[*].text"]


def _full(tool: str) -> str:
    return catalog.NLM_ASK if tool == "@NLM@" else catalog.ran(tool)


def _action(full_name: str) -> str:
    return f'AgentCore::Action::"{full_name}"'


def team_expr(team: str) -> str:
    # The kit's token claims: swarm = team name, role = agent (the app's token has role = app).
    return (f'principal.hasTag("swarm") && principal.getTag("swarm") == "{team}" && '
            'principal.hasTag("role") && principal.getTag("role") == "agent"')


def agent_in(agents: list[str]) -> str:
    return f'principal.hasTag("agent") && {json.dumps(sorted(agents))}.contains(principal.getTag("agent"))'


def input_guard(field: str, categories=GUARD_IN, limit: str = "0.4") -> str:
    cats = json.dumps(list(categories))
    return "\n  && ".join(f'BedrockGuardrails::PromptAttack({cats}, [{field}])["{c}"]'
                          f'.confidenceScore.lessThanOrEqual(decimal("{limit}"))' for c in categories)


def output_guard(categories=GUARD_OUT, limit: str = "0.2") -> str:
    cats, paths = json.dumps(list(categories)), json.dumps(OUTPUT_PATHS)
    return "\n  || ".join(f'BedrockGuardrails::ContentFilter({cats}, {paths})["{c}"]'
                          f'.confidenceScore.greaterThan(decimal("{limit}"))' for c in categories)


def safe_name(*parts: str) -> str:
    """A legal, unique policy name (<= 48 chars, letters/digits/_); long names keep a hash suffix."""
    raw = "_".join(re.sub(r"[^A-Za-z0-9]+", "_", p).strip("_").lower() for p in parts if p)
    if len(raw) <= NAME_MAX:
        return raw
    digest = hashlib.sha256(raw.encode()).hexdigest()[:6]
    return raw[:NAME_MAX - 7].rstrip("_") + "_" + digest


def rules(team: str = "team13") -> list[dict]:
    """[{name, mode, statement, actions}] in a fixed order."""
    p = f"{team}_ran"
    who_may: dict[str, list[str]] = {}
    for agent, tools in AGENT_TOOLS.items():
        for t in tools:
            who_may.setdefault(_full(t), []).append(agent)
    out = []
    nlm = catalog.NLM_ASK
    for full in catalog.all_tools():
        if full == nlm:
            continue
        agents = sorted(who_may.get(full, []))
        if not agents:
            continue
        out.append({"name": safe_name(p, "allow", full.split("___", 1)[1]), "mode": STRICT, "actions": [full],
                    "statement": f"""
permit (principal is AgentCore::OAuthUser, action == {_action(full)}, resource == @GW@)
when {{ {team_expr(team)} && {agent_in(agents)} }};"""})
    nlm_agents = sorted(who_may.get(nlm, []))
    out.append({"name": safe_name(p, "nlm_safe_questions"), "mode": STRICT, "actions": [nlm], "statement": f"""
permit (principal is AgentCore::OAuthUser, action == {_action(nlm)}, resource == @GW@)
when guardrails {{
  {input_guard("context.input.question")}
}}
unless {{ !({team_expr(team)} && {agent_in(nlm_agents)}) }};"""})
    # Both names reach our code after the takeover of `nlm`. One suppressOutput rule per action, in
    # the kit's exact `action ==` form (it never uses `action in` for suppressOutput).
    for action, suffix in ((catalog.NLM_ASK, ""), (catalog.KIT_NLM_ASK, "_kit_name")):
        out.append({"name": safe_name(p, "screen_nlm_answers" + suffix), "mode": LOOSE, "actions": [action],
                    "statement": f"""
suppressOutput (principal, action == {_action(action)}, resource == @GW@)
when guardrails {{
  {output_guard()}
}};"""})
    every = catalog.all_tools()
    out.append({"name": safe_name(p, "require_case_id"), "mode": LOOSE, "actions": every, "statement": f"""
forbid (principal, action in [{", ".join(_action(a) for a in every)}], resource == @GW@)
unless {{ context.input has case_id && context.input.case_id like "case-*" }};"""})
    apply_tool = catalog.ran(catalog.APPLY)
    out.append({"name": f"{p}_require_human_approval", "mode": LOOSE, "actions": [apply_tool], "statement": f"""
forbid (principal, action == {_action(apply_tool)}, resource == @GW@)
when {{ {agent_in(APPLY_AGENTS)} }}
unless {{ context.input has approval && context.input.approval == "APPROVED" }};"""})
    # The kit's work___apply_change keys its approval exactly like ran___apply_change (same
    # [case_id, item_id, change] hash), so calling it would consume a person's approval without
    # changing the twin. Our agents may never call it. The kit's `check` only expects that call to
    # be REFUSED, which this keeps.
    kit_apply = catalog.KIT_WORK_APPLY
    out.append({"name": f"{p}_no_kit_apply_change", "mode": LOOSE, "actions": [kit_apply], "statement": f"""
forbid (principal, action == {_action(kit_apply)}, resource == @GW@)
when {{ {agent_in(list(AGENT_TOOLS))} }};"""})
    for r in out:
        r["statement"] = r["statement"].strip()
    return out


VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}")


def pause_rule(model_id: str, version: str, team: str = "team13", suffix: str = "") -> dict:
    """The forbid that withdraws one model version (amendment 1, A8).

    It refuses the tool that serves the model when the call names no version or names this one.
    The tools refuse a named version they do not serve, so naming another version cannot reach the
    paused model. The `has` guard matters: a forbid that reads a missing attribute errors instead of
    denying. Policy names are unique in the account: pass a `suffix` (e.g. a timestamp) to pause
    the same version again after a resume.
    """
    if model_id not in MODEL_TOOLS:
        raise KeyError(f"unknown model {model_id!r}; known: {sorted(MODEL_TOOLS)}")
    if not VERSION_RE.fullmatch(version):
        raise ValueError(f"version {version!r}: letters, digits, _ . : - only")
    tool = _full(MODEL_TOOLS[model_id])
    tools = list(catalog.NLM_NAMES) if tool == catalog.NLM_ASK else [tool]  # the NLM answers on both names
    scope = (f"action == {_action(tools[0])}" if len(tools) == 1
             else f"action in [{', '.join(_action(t) for t in tools)}]")
    name = safe_name(f"{team}_ranpause", tool.split("___", 1)[1], model_id, version, suffix)
    return {"name": name, "mode": LOOSE, "actions": tools, "model_id": model_id, "version": version,
            "statement": f"forbid (principal, {scope}, resource == @GW@)\n"
                         f"when {{ !(context.input has model_version) || context.input.model_version == "
                         f"{json.dumps(version)} }};"}


def fill(statement: str, gateway_arn: str) -> str:
    if not re.fullmatch(r"arn:aws:bedrock-agentcore:[a-z0-9-]+:\d{12}:gateway/[A-Za-z0-9_-]+", gateway_arn):
        raise ValueError("not a Gateway ARN")
    return statement.replace("@GW@", f'AgentCore::Gateway::"{gateway_arn}"')


# ---------------------------------------------------------------- offline checks
def _balanced(text: str) -> bool:
    pairs, stack, in_str, esc = {")": "(", "]": "[", "}": "{"}, [], False, False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack and not in_str


def check(rule_list: list[dict], team: str = "team13") -> list[str]:
    """Problems Cedar's validator or the kit's interceptor would trip on. Empty list = fine."""
    problems, names = [], set()
    tools = catalog.all_tools()
    for r in rule_list:
        n, s = r["name"], r["statement"]
        if not NAME_RE.fullmatch(n) or len(n) > NAME_MAX:
            problems.append(f"{n}: illegal policy name")
        if not (n.startswith(f"{team}_ran_") or n.startswith(f"{team}_ranpause_")):
            problems.append(f"{n}: must start with {team}_ran_ or {team}_ranpause_")
        if n.startswith(f"{team}_pause_nlm"):
            problems.append(f"{n}: the kit's 'resume nlm' would delete it")
        if n in names:
            problems.append(f"{n}: duplicate name")
        names.add(n)
        effect = s.split("(", 1)[0].strip()
        if effect not in ("permit", "forbid", "suppressOutput"):
            problems.append(f"{n}: unknown effect {effect!r}")
        if r["mode"] != (STRICT if effect == "permit" else LOOSE):
            problems.append(f"{n}: {effect} should use {STRICT if effect == 'permit' else LOOSE}")
        if not _balanced(s) or not s.endswith(";") or s.count(";") != 1:
            problems.append(f"{n}: unbalanced or not exactly one statement")
        if s.count("resource == @GW@") != 1:
            problems.append(f"{n}: must name the Gateway exactly once")
        if "maxConfidenceScore" in s:
            problems.append(f"{n}: aggregate guardrail score fails every call")
        acted = re.findall(r'AgentCore::Action::"([^"]+)"', s)
        if sorted(set(acted)) != sorted(set(r["actions"])):
            problems.append(f"{n}: actions in text {acted} != declared {r['actions']}")
        for a in acted:
            if a not in tools and a not in KNOWN_KIT_ACTIONS:
                problems.append(f"{n}: unknown tool {a}")
                continue
            declared = catalog.input_fields(a) if a != catalog.KIT_WORK_APPLY else KIT_WORK_APPLY_FIELDS
            for field in set(re.findall(r"context\.input\.([A-Za-z_][A-Za-z0-9_]*)", s)) | \
                    set(re.findall(r"context\.input has ([A-Za-z_][A-Za-z0-9_]*)", s)):
                if field not in declared:
                    problems.append(f"{n}: {a} declares no input field {field!r}")
        if effect == "permit" and "principal is AgentCore::OAuthUser" not in s:
            problems.append(f"{n}: permits must name AgentCore::OAuthUser principals")
    permitted = {a for r in rule_list if r["statement"].startswith("permit") for a in r["actions"]}
    for t in tools:
        if t not in permitted:
            problems.append(f"{t}: no permit (default deny refuses every call)")
    case_rules = [r for r in rule_list if r["name"].endswith("_require_case_id")]
    if len(case_rules) != 1 or sorted(case_rules[0]["actions"]) != sorted(tools):
        problems.append("the case-id forbid must cover every tool")
    approval = [r for r in rule_list if "require_human_approval" in r["name"]]
    if len(approval) != 1 or approval[0]["actions"] != [catalog.ran(catalog.APPLY)] \
            or not approval[0]["actions"][0].endswith("___apply_change"):
        problems.append("need exactly one require_human_approval forbid on a ___apply_change tool")
    # whatever governs the NLM must govern both of its names
    screened = {a for r in rule_list if r["statement"].startswith("suppressOutput") for a in r["actions"]}
    if not set(catalog.NLM_NAMES) <= screened:
        problems.append(f"the answer screen must cover both {' and '.join(catalog.NLM_NAMES)}")
    for r in rule_list:
        if r["name"].startswith(f"{team}_ranpause_") and set(r["actions"]) & set(catalog.NLM_NAMES) \
                and sorted(r["actions"]) != sorted(catalog.NLM_NAMES):
            problems.append(f"{r['name']}: must cover both {' and '.join(catalog.NLM_NAMES)}")
    return problems


# ---------------------------------------------------------------- files
def render(team: str = "team13") -> dict[str, str]:
    """{relative path: file text} for the generated directory."""
    files, manifest = {}, {"team": team, "tools": catalog.all_tools(), "rules": [], "pause_examples": []}
    for r in rules(team):
        path = f"{r['name']}.cedar"
        files[path] = f"// {r['name']}  validationMode={r['mode']}\n// generated by t3/policies/generate.py\n" \
                      f"{r['statement']}\n"
        manifest["rules"].append({"name": r["name"], "mode": r["mode"], "actions": r["actions"], "file": path})
    for model_id, version in EXAMPLE_PAUSES:
        r = pause_rule(model_id, version, team)
        path = f"pause/{r['name']}.cedar"
        files[path] = f"// {r['name']}  validationMode={r['mode']}  (template: created only to pause)\n" \
                      f"{r['statement']}\n"
        manifest["pause_examples"].append({"name": r["name"], "mode": r["mode"], "actions": r["actions"],
                                           "model_id": model_id, "version": version, "file": path})
    files["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    return files


def write(out: Path = OUT, team: str = "team13") -> list[str]:
    files = render(team)
    for rel, text in files.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text(text)
    return sorted(files)


def diff(out: Path = OUT, team: str = "team13") -> list[str]:
    want = render(team)
    have = {str(p.relative_to(out)): p.read_text() for p in out.rglob("*") if p.is_file()} if out.exists() else {}
    return sorted(k for k in set(want) | set(have) if want.get(k) != have.get(k))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team", default="team13")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", action="store_true", help="verify only; exit 1 on a problem or a stale file")
    a = ap.parse_args(argv)
    problems = check(rules(a.team), a.team) + check([pause_rule(m, v, a.team) for m, v in EXAMPLE_PAUSES]
                                                    + rules(a.team), a.team)
    for pr in sorted(set(problems)):
        print("PROBLEM", pr)
    if a.check:
        stale = diff(a.out, a.team)
        for s in stale:
            print("STALE", s)
        print("ok" if not (problems or stale) else "failed")
        return 1 if problems or stale else 0
    if problems:
        return 1
    for rel in write(a.out, a.team):
        print("wrote", a.out / rel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
