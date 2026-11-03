"""Names and Gateway tool schemas of Team13's RAN tools (contract amendment 1, A1-A3, A8).

One source of truth for the Lambda (dispatch), the Cedar generator (actions and the
`context.input` fields the rules read) and the deploy script (Gateway targets).

The Gateway names a tool `<target>___<tool>` and Cedar sees that full name as the action.
Frozen names (A1): ran___observe, ran___propose, ran___predict, ran___preflight,
ran___apply_change, nlm___ask.

`nlm___ask` lives in the target named `nlm`, which the organisers' kit creates on its first deploy
(for its own `nlm___ask_nlm`) and skips on every later deploy. register.py therefore TAKES OVER
that target: it points it at our Lambda with two tools, the kit's `ask_nlm` (same schema, so the
kit's rules and `pause nlm` keep working) and our `ask`. Our handler serves both.
"""
from __future__ import annotations

RAN_TARGET = "ran"
NLM_TARGET = "nlm"

OBSERVE = "observe"
PROPOSE = "propose"
PREDICT = "predict"
PREFLIGHT = "preflight"
APPLY = "apply_change"
ASK = "ask"
KIT_ASK = "ask_nlm"   # the kit's NLM tool name, kept in the taken-over target

RAN_TOOLS = (OBSERVE, PROPOSE, PREDICT, PREFLIGHT, APPLY)
# The evaluation harness's tool (t3/evals/live_swarm.py): opens a case at an incident's state. It SETS twin
# state, so Cedar lets only the app's token call it (t3/policies/generate.py: seed_app_only); it is not one
# of the six frozen agent tools and no agent prompt names it.
SEED = "seed_case"


def full(target: str, tool: str) -> str:
    return f"{target}___{tool}"


def ran(tool: str) -> str:
    return full(RAN_TARGET, tool)


NLM_ASK = full(NLM_TARGET, ASK)
SEED_FULL = full(RAN_TARGET, SEED)
KIT_NLM_ASK = full(NLM_TARGET, KIT_ASK)
NLM_NAMES = (NLM_ASK, KIT_NLM_ASK)      # both reach our code after the takeover of target `nlm`
KIT_WORK_APPLY = "work___apply_change"  # the kit's example apply tool (its own Lambda, not our twin)


def all_tools() -> list[str]:
    """Our six frozen tool names (A1), in a fixed order. Each has its own permit."""
    return [ran(t) for t in RAN_TOOLS] + [NLM_ASK]


def governed_tools() -> list[str]:
    """Every name that reaches our Lambda: the six plus the kit's nlm___ask_nlm, which the takeover of
    target `nlm` routes to the same code. Every forbid and screen of ours must cover all of them."""
    return all_tools() + [KIT_NLM_ASK]


# ---------- Gateway tool schemas (the subset of JSON Schema the Gateway accepts) ----------
def _p(type_: str, description: str) -> dict:
    return {"type": type_, "description": description}


CASE_ID = _p("string", "The case_id returned by open_case (case-...). Every call needs it.")
SPECIAL = _p("object", "Added by the Agent Builder. Leave it out.")
CHANGE = _p("string", "The change as one canonical string, e.g. cell=2;d_power_db=+0.0;d_tilt_deg=-2.0 "
                      "(several cells joined with |, sorted by cell). Copy it verbatim from ran___propose.")
ITEM_ID = _p("string", "The item_id from ran___propose, e.g. step-0. Copy it verbatim.")
MODEL_VERSION = _p("string", "Optional. The model version you expect to answer; leave it out to use the current one.")


def _tool(name: str, description: str, props: dict, required: list[str]) -> dict:
    # case_id is deliberately not "required": a call without it must reach Cedar, which refuses
    # it with a recorded decision (the case-id forbid), instead of failing schema validation.
    schema = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return {"name": name, "description": description, "inputSchema": schema}


def ran_tool_schemas() -> list[dict]:
    base = {"case_id": CASE_ID, "special_context": SPECIAL}
    return [
        _tool(OBSERVE, "Read the RAN twin's current KPIs for this case (coverage, SLA, SINR, per cell).",
              dict(base), []),
        _tool(PROPOSE, "Ask the trained RL policy for a change for this case. Returns item_id and change: "
                       "pass both verbatim to ran___predict and ran___apply_change.",
              {**base, "model_version": MODEL_VERSION}, []),
        _tool(PREDICT, "Predict the KPI change (delta against doing nothing) with the learned response model. "
                       "The prediction is stored and later compared with what the twin measures.",
              {**base, "item_id": ITEM_ID, "change": CHANGE, "model_version": MODEL_VERSION}, ["change"]),
        _tool(PREFLIGHT, "Try a change on a copy of this case's twin and report the KPI change. Changes nothing.",
              {**base, "change": CHANGE}, ["change"]),
        # Mirrors the kit's apply_change schema (checkpoint_kit.py 376-382): approval and
        # special_context must be declared or the interceptor's injected value fails validation.
        _tool(SEED, "Evaluation harness only (the app's token; agents are refused): open a new case at an "
                    "incident's state.",
              {"case_id": CASE_ID, "scenario": _p("string", "Twin scenario."),
               "seed": _p("integer", "Twin seed."), "preset": _p("string", "Twin preset, e.g. official."),
               "start_action_log_json": _p("string", "JSON list of the steps applied so far, each a canonical "
                                                     "change string or [] for a no-op step.")},
              ["scenario", "seed", "start_action_log_json"]),
        _tool(APPLY, "Apply a change to this case's twin. Needs a person's approval; one approval applies "
                     "one change. If refused for approval, give the person the approval id, and after they "
                     "approve call again with exactly the same values.",
              {"case_id": CASE_ID, "item_id": ITEM_ID, "change": CHANGE,
               "reason": _p("string", "Why this change solves the problem."),
               "approval": _p("string", "Set by the checkpoint. Leave it out."),
               "special_context": SPECIAL},
              ["item_id", "change"]),
    ]


def nlm_tool_schemas() -> list[dict]:
    """Both tools of the taken-over `nlm` target: the kit's ask_nlm (its schema) and our ask."""
    return [
        # The kit's schema plus an optional model_version, so the NLM pause rule (which covers both
        # names, because both reach our code) can test `context.input has model_version` here too.
        _tool(KIT_ASK, "Ask the domain model (NLM) a question.",
              {"case_id": CASE_ID, "question": _p("string", "The question."), "model_version": MODEL_VERSION,
               "special_context": SPECIAL},
              ["question"]),
        _tool(ASK, "Ask the network language model (NLM) a question. The answer says which model answered.",
              {"case_id": CASE_ID, "question": _p("string", "The question."), "model_version": MODEL_VERSION,
               "special_context": SPECIAL}, ["question"]),
    ]


def targets() -> dict[str, list[dict]]:
    """{target name: [tool schemas]} for the Gateway."""
    return {RAN_TARGET: ran_tool_schemas(), NLM_TARGET: nlm_tool_schemas()}


def input_fields(full_name: str) -> set[str]:
    """The top-level argument names a tool declares (what Cedar can read as context.input.*)."""
    for target, tools in targets().items():
        for t in tools:
            if full(target, t["name"]) == full_name:
                return set(t["inputSchema"]["properties"])
    raise KeyError(full_name)
